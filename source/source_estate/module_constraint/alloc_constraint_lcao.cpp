#include "source_estate/module_constraint/alloc_constraint_lcao.h"

#include "source_base/parallel_reduce.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_lcao_report.h"
#include "source_estate/module_constraint/constraint_inject_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

#include <algorithm>
#include <stdexcept>

namespace alloc
{
namespace
{

void reduce_pool(std::vector<double>& values)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(values.data(),
                                 static_cast<int>(values.size()));
#else
    (void)values;
#endif
}

bool all_blocks_are_gamma(const hamilt::HContainer<double>& H)
{
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair
            = H.get_atom_pair(static_cast<int>(iap));
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
            // Branch: a non-zero cell vector invalidates the dense Gamma
            // contraction for the whole container.
            if (R.x != 0 || R.y != 0 || R.z != 0)
            {
                return false;
            }
        }
    }
    return true;
}

template <typename F>
void for_each_gamma_block(const hamilt::HContainer<double>& H,
                          const UnitCell& ucell,
                          F&& callback)
{
    const Parallel_Orbitals* pv = H.get_paraV();
    // Branch: without the AO distribution the local blocks cannot be mapped
    // to global indices.
    if (pv == nullptr)
    {
        return;
    }
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair
            = H.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const hamilt::BaseMatrix<double>* matrix
            = pair.find_matrix(0, 0, 0);
        // Branch: a pair without an R=0 block contributes nothing.
        if (matrix == nullptr)
        {
            continue;
        }
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        const int base_i = ucell.get_iat2iwt()[iat1];
        const int base_j = ucell.get_iat2iwt()[iat2];
        for (int i = 0; i < matrix->get_row_size(); ++i)
        {
            for (int j = 0; j < matrix->get_col_size(); ++j)
            {
                callback(base_i + rows[i], base_j + cols[j],
                         matrix->get_value(i, j));
            }
        }
    }
}

} // namespace

AllocConstraintLcao& AllocConstraintLcao::instance()
{
    static AllocConstraintLcao state;
    return state;
}

void AllocConstraintLcao::reset()
{
    active_ = false;
    nconstraint_ = 0;
    ntarget_ = 0;
    norb_ = 0;
    target_rank_ = 0;
    target_cond_ = 1.0;
    target_ = LcaoTargetSpace{};
    target_constraint_.clear();
    SB_.clear();
    kernels_.clear();
    mu_eff_.clear();
}

bool AllocConstraintLcao::build(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& S,
    const hamilt::HContainer<double>& h_layout,
    const std::vector<std::vector<double>>& cw,
    const std::vector<std::vector<int>>& constraint_atoms,
    ModuleGint::GintInfo* gint_info,
    const Parallel_Orbitals* paraV,
    const int target_l,
    const double rank_tol,
    std::string& error)
{
    reset();
    // Branch: an uninitialized Gint layout cannot build the weight matrices.
    if (gint_info == nullptr)
    {
        error = "allocation constraint needs an initialized GintInfo";
        return false;
    }
    if (cw.size() != constraint_atoms.size())
    {
        error = "allocation constraint weight/atom lists differ";
        return false;
    }
    if (constraint_atoms.empty())
    {
        error = "allocation constraint has no constrained components";
        return false;
    }
    if (!all_blocks_are_gamma(S))
    {
        error = "allocation constraint currently supports Gamma-only S(R)";
        return false;
    }

    try
    {
        target_ = build_lcao_target_from_container(ucell, S, target_l,
                                                   rank_tol);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }

    nconstraint_ = static_cast<int>(cw.size());
    ntarget_ = target_.ntarget;
    norb_ = target_.norb;
    target_rank_ = target_.rank;
    target_cond_ = target_.cond;
    // Branch: no selected target orbital means the allocation operator is the
    // plain spatial weight operator; S3 requires at least one target.
    if (ntarget_ == 0)
    {
        error = "allocation constraint selected zero target orbitals";
        return false;
    }

    target_constraint_.assign(ntarget_, -1);
    for (int a = 0; a < ntarget_; ++a)
    {
        const int iat = target_.atom[a];
        int owner = -1;
        for (int alpha = 0; alpha < nconstraint_; ++alpha)
        {
            // Branch: a target atom owned by this constraint is assigned to
            // it; a second owner is an ambiguous fragment definition.
            if (std::find(constraint_atoms[alpha].begin(),
                          constraint_atoms[alpha].end(), iat)
                != constraint_atoms[alpha].end())
            {
                if (owner >= 0)
                {
                    error = "target atom belongs to more than one constraint";
                    return false;
                }
                owner = alpha;
            }
        }
        target_constraint_[a] = owner;
    }

    // SB = S B, with B the orthonormal target coefficients.  This is the
    // only full-basis contraction required by the low-rank kernels.
    SB_.assign(static_cast<size_t>(norb_) * ntarget_, 0.0);
    for_each_gamma_block(S, ucell, [&](const int gi, const int gj,
                                       const double value) {
        for (int a = 0; a < ntarget_; ++a)
        {
            SB_[static_cast<size_t>(gi) * ntarget_ + a]
                += value * target_.Cprime[static_cast<size_t>(gj) * ntarget_
                                          + a];
        }
    });
    reduce_pool(SB_);

    std::vector<hamilt::HContainer<double>> weights;
    try
    {
        weights = constraint::ConstraintInjectLCAO::build(
            cw, gint_info, paraV, &h_layout);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
    if (weights.size() != cw.size())
    {
        error = "weight-matrix builder returned the wrong count";
        return false;
    }

    kernels_.clear();
    kernels_.reserve(nconstraint_);
    for (int alpha = 0; alpha < nconstraint_; ++alpha)
    {
        std::vector<double> BtW(static_cast<size_t>(ntarget_) * norb_, 0.0);
        std::vector<double> BtWB(static_cast<size_t>(ntarget_) * ntarget_, 0.0);
        for_each_gamma_block(weights[alpha], ucell,
            [&](const int gi, const int gj, const double value) {
                for (int a = 0; a < ntarget_; ++a)
                {
                    const double b_i = target_.Cprime[static_cast<size_t>(gi)
                                                          * ntarget_ + a];
                    BtW[static_cast<size_t>(a) * norb_ + gj] += b_i * value;
                    for (int b = 0; b < ntarget_; ++b)
                    {
                        BtWB[static_cast<size_t>(a) * ntarget_ + b]
                            += b_i * value
                               * target_.Cprime[static_cast<size_t>(gj)
                                                    * ntarget_ + b];
                    }
                }
            });
        reduce_pool(BtW);
        reduce_pool(BtWB);

        hamilt::HContainer<double> K(h_layout);
        K.set_zero();
        const Parallel_Orbitals* pv = K.get_paraV();
        // Branch: without the AO distribution the kernel cannot be mapped to
        // the Hamiltonian blocks; this is a wiring bug, not a silent skip.
        if (pv == nullptr)
        {
            error = "Hamiltonian layout has no Parallel_Orbitals";
            return false;
        }
        for (size_t iap = 0; iap < K.size_atom_pairs(); ++iap)
        {
            hamilt::AtomPair<double>& pair
                = K.get_atom_pair(static_cast<int>(iap));
            const int iat1 = pair.get_atom_i();
            const int iat2 = pair.get_atom_j();
            hamilt::BaseMatrix<double>* matrix = pair.find_matrix(0, 0, 0);
            const hamilt::BaseMatrix<double>* wmatrix
                = weights[alpha].find_matrix(iat1, iat2, 0, 0, 0);
            // Branch: only R=0 Hamiltonian blocks are present in this Gamma
            // path; a missing weight block contributes zero.
            if (matrix == nullptr || wmatrix == nullptr)
            {
                continue;
            }
            const std::vector<int> rows = pv->get_indexes_row(iat1);
            const std::vector<int> cols = pv->get_indexes_col(iat2);
            const int base_i = ucell.get_iat2iwt()[iat1];
            const int base_j = ucell.get_iat2iwt()[iat2];
            for (int i = 0; i < matrix->get_row_size(); ++i)
            {
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    const int mu = base_i + rows[i];
                    const int nu = base_j + cols[j];
                    double value = wmatrix->get_value(i, j);
                    for (int a = 0; a < ntarget_; ++a)
                    {
                        const double sb_mu
                            = SB_[static_cast<size_t>(mu) * ntarget_ + a];
                        const double sb_nu
                            = SB_[static_cast<size_t>(nu) * ntarget_ + a];
                        value -= sb_mu
                                 * BtW[static_cast<size_t>(a) * norb_ + nu];
                        value -= BtW[static_cast<size_t>(a) * norb_ + mu]
                                 * sb_nu;
                        for (int b = 0; b < ntarget_; ++b)
                        {
                            value += sb_mu
                                     * BtWB[static_cast<size_t>(a) * ntarget_
                                            + b]
                                     * SB_[static_cast<size_t>(nu) * ntarget_
                                           + b];
                        }
                        // Branch: Q_alpha owns only targets assigned to this
                        // constraint.
                        if (target_constraint_[a] == alpha)
                        {
                            value += sb_mu * sb_nu;
                        }
                    }
                    matrix->get_value(i, j) = value;
                }
            }
        }
        kernels_.push_back(std::move(K));
    }

    mu_eff_.assign(nconstraint_, 0.0);
    active_ = true;
    return true;
}

void AllocConstraintLcao::set_mu_eff(const std::vector<double>& mu_eff)
{
    // Branch: the operator is inactive until build() has produced its
    // kernels; keep the multiplier state empty in that case.
    if (!active_)
    {
        mu_eff_.clear();
        return;
    }
    // Branch: a mismatched multiplier vector would pair a multiplier with the
    // wrong kernel, so refuse it instead of truncating silently.
    if (static_cast<int>(mu_eff.size()) != nconstraint_)
    {
        throw std::invalid_argument("allocation mu count does not match kernels");
    }
    mu_eff_ = mu_eff;
}

bool AllocConstraintLcao::project_moments(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& S,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const int nspin,
    std::vector<double>& m_T_constraint,
    std::string& error) const
{
    // Branch: an inactive state has no target space to project onto.
    if (!active_)
    {
        error = "allocation constraint is not initialized";
        return false;
    }
    std::vector<double> n_T_target;
    std::vector<double> m_T_target;
    try
    {
        observe_lcao_target_from_container(ucell, S, target_, dmr, nspin,
                                           n_T_target, m_T_target);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
    m_T_constraint.assign(nconstraint_, 0.0);
    for (int a = 0; a < ntarget_; ++a)
    {
        const int alpha = target_constraint_[a];
        // Branch: unowned targets enter the global projector P but no ALM
        // constraint sum.
        if (alpha >= 0)
        {
            m_T_constraint[alpha] += m_T_target[a];
        }
    }
    return true;
}

} // namespace alloc
