#include "source_estate/module_constraint/alloc_constraint_lcao.h"

#include "source_base/parallel_reduce.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_lcao_report.h"
#include "source_estate/module_constraint/constraint_inject_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_lcao/module_gint/gint_interface.h"

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

void reduce_pool(double& value)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(value);
#else
    (void)value;
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
    injection_up_.clear();
    injection_down_.clear();
    gint_info_ = nullptr;
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
    injection_up_.assign(nconstraint_, 1.0);
    injection_down_.assign(nconstraint_, -1.0);
    gint_info_ = gint_info;
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

void AllocConstraintLcao::set_injection_signs(
    const std::vector<double>& up,
    const std::vector<double>& down)
{
    // Guard: the signs are paired with kernels by constraint index.
    if (static_cast<int>(up.size()) != nconstraint_
        || static_cast<int>(down.size()) != nconstraint_)
    {
        throw std::invalid_argument(
            "allocation injection-sign count does not match kernels");
    }
    injection_up_ = up;
    injection_down_ = down;
}

double AllocConstraintLcao::injection_sign(const int alpha,
                                           const int spin) const
{
    // Guard: only the two collinear channels are represented in S3.
    if (alpha < 0 || alpha >= nconstraint_ || (spin != 0 && spin != 1))
    {
        throw std::out_of_range("allocation injection sign index out of range");
    }
    return (spin == 0) ? injection_up_[alpha] : injection_down_[alpha];
}

bool AllocConstraintLcao::project_moments(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& S,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const int nspin,
    std::vector<double>& n_T_constraint,
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
    n_T_constraint.assign(nconstraint_, 0.0);
    m_T_constraint.assign(nconstraint_, 0.0);
    for (int a = 0; a < ntarget_; ++a)
    {
        const int alpha = target_constraint_[a];
        // Branch: unowned targets enter the global projector P but no ALM
        // constraint sum.
        if (alpha >= 0)
        {
            n_T_constraint[alpha] += n_T_target[a];
            m_T_constraint[alpha] += m_T_target[a];
        }
    }
    return true;
}

bool AllocConstraintLcao::project_allocations(
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const int nspin,
    std::vector<double>& n_alloc,
    std::vector<double>& m_alloc,
    std::string& error) const
{
    // Branch: the allocation observable exists only after build() has made
    // the geometry-fixed kernels and their layout.
    if (!active_ || static_cast<int>(kernels_.size()) != nconstraint_)
    {
        error = "allocation constraint is not initialized";
        return false;
    }
    // Branch: S3 supports only the collinear one/two-channel density matrix.
    if (nspin != 1 && nspin != 2)
    {
        error = "allocation observable supports nspin 1/2";
        return false;
    }
    // Guard: one DMR pointer per spin channel is required to separate n and
    // the collinear magnetization consistently.
    if (static_cast<int>(dmr.size()) != nspin)
    {
        error = "DMR spin-channel count does not match nspin";
        return false;
    }

    n_alloc.assign(nconstraint_, 0.0);
    m_alloc.assign(nconstraint_, 0.0);
    for (int spin = 0; spin < nspin; ++spin)
    {
        // Guard: every channel must be present before a trace is attempted.
        if (dmr[spin] == nullptr)
        {
            error = "allocation observable received a null DMR channel";
            return false;
        }
        for (int alpha = 0; alpha < nconstraint_; ++alpha)
        {
            double trace = 0.0;
            // Guard: trace() also checks the flat HContainer layouts, so a
            // stale kernel can never be paired with a different DMR layout.
            if (!constraint::ConstraintInjectLCAO::trace(
                    kernels_[alpha], *dmr[spin], trace))
            {
                error = "allocation kernel/DMR layout mismatch";
                return false;
            }
            reduce_pool(trace);
            n_alloc[alpha] += trace;
            // Branch: collinear magnetization is the up-minus-down trace.
            if (nspin == 2)
            {
                m_alloc[alpha] += (spin == 0 ? trace : -trace);
            }
        }
    }
    return true;
}

bool AllocConstraintLcao::complement_density(
    const UnitCell& ucell,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const int nspin,
    const int nrxx,
    std::vector<std::vector<double>>& rho,
    std::string& error) const
{
    // Branch: the complementary density is only meaningful after the target
    // space and kernels have been built on this geometry.
    if (!active_ || gint_info_ == nullptr)
    {
        error = "allocation constraint is not initialized for force";
        return false;
    }
    // Guard: the complementary density uses the same collinear channel
    // convention as the observable.
    if (nspin != 1 && nspin != 2)
    {
        error = "allocation force supports nspin 1/2";
        return false;
    }
    // Guard: force densities and projection observables must see the same
    // number of spin channels.
    if (static_cast<int>(dmr.size()) != nspin)
    {
        error = "DMR spin-channel count does not match nspin";
        return false;
    }
    // Guard: an unallocated grid cannot receive the Gint density.
    if (nrxx <= 0)
    {
        error = "allocation force needs a positive density-grid size";
        return false;
    }

    std::vector<hamilt::HContainer<double>> density_complement;
    density_complement.reserve(nspin);
    std::vector<hamilt::HContainer<double>*> density_pointers;
    density_pointers.reserve(nspin);
    const size_t dense_size = static_cast<size_t>(norb_) * norb_;
    for (int spin = 0; spin < nspin; ++spin)
    {
        // Branch: every DMR channel must share the Gamma-only contract used
        // by the target projection.
        if (dmr[spin] == nullptr || !all_blocks_are_gamma(*dmr[spin]))
        {
            error = "periodic multi-k DMR is not supported by allocation force";
            return false;
        }

        std::vector<double> D(dense_size, 0.0);
        for_each_gamma_block(*dmr[spin], ucell,
            [&](const int gi, const int gj, const double value) {
                D[static_cast<size_t>(gi) * norb_ + gj] += value;
            });
        reduce_pool(D);

        // P = (S B)(S B)^T.  Accumulate X = B^T S D and
        // Y = D S B once, then P D P through the R x R matrix Z.
        std::vector<double> X(static_cast<size_t>(ntarget_) * norb_, 0.0);
        std::vector<double> Y(static_cast<size_t>(norb_) * ntarget_, 0.0);
        for (int i = 0; i < norb_; ++i)
        {
            for (int j = 0; j < norb_; ++j)
            {
                const double d = D[static_cast<size_t>(i) * norb_ + j];
                for (int a = 0; a < ntarget_; ++a)
                {
                    const double sb_i
                        = SB_[static_cast<size_t>(i) * ntarget_ + a];
                    const double sb_j
                        = SB_[static_cast<size_t>(j) * ntarget_ + a];
                    X[static_cast<size_t>(a) * norb_ + j] += sb_i * d;
                    Y[static_cast<size_t>(i) * ntarget_ + a] += d * sb_j;
                }
            }
        }
        std::vector<double> Z(static_cast<size_t>(ntarget_) * ntarget_, 0.0);
        for (int a = 0; a < ntarget_; ++a)
        {
            for (int b = 0; b < ntarget_; ++b)
            {
                for (int j = 0; j < norb_; ++j)
                {
                    Z[static_cast<size_t>(a) * ntarget_ + b]
                        += X[static_cast<size_t>(a) * norb_ + j]
                           * SB_[static_cast<size_t>(j) * ntarget_ + b];
                }
            }
        }

        density_complement.emplace_back(*dmr[spin]);
        hamilt::HContainer<double>& Dc = density_complement.back();
        const Parallel_Orbitals* pv = Dc.get_paraV();
        // Branch: a layout-less DMR cannot be mapped to this rank's blocks.
        if (pv == nullptr)
        {
            error = "DMR has no Parallel_Orbitals layout";
            return false;
        }
        for (size_t iap = 0; iap < Dc.size_atom_pairs(); ++iap)
        {
            hamilt::AtomPair<double>& pair
                = Dc.get_atom_pair(static_cast<int>(iap));
            const int iat1 = pair.get_atom_i();
            const int iat2 = pair.get_atom_j();
            hamilt::BaseMatrix<double>* matrix = pair.find_matrix(0, 0, 0);
            // Branch: only the Gamma R=0 block exists in this path.
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
                const int gi = base_i + rows[i];
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    const int gj = base_j + cols[j];
                    double value = D[static_cast<size_t>(gi) * norb_ + gj];
                    for (int a = 0; a < ntarget_; ++a)
                    {
                        value -= SB_[static_cast<size_t>(gi) * ntarget_ + a]
                                 * X[static_cast<size_t>(a) * norb_ + gj];
                        value -= Y[static_cast<size_t>(gi) * ntarget_ + a]
                                 * SB_[static_cast<size_t>(gj) * ntarget_ + a];
                        for (int b = 0; b < ntarget_; ++b)
                        {
                            value += SB_[static_cast<size_t>(gi) * ntarget_ + a]
                                     * Z[static_cast<size_t>(a) * ntarget_ + b]
                                     * SB_[static_cast<size_t>(gj) * ntarget_ + b];
                        }
                    }
                    matrix->get_value(i, j) = value;
                }
            }
        }
        density_pointers.push_back(&Dc);
    }

    rho.assign(nspin, std::vector<double>(nrxx, 0.0));
    std::vector<double*> rho_pointers(nspin, nullptr);
    for (int spin = 0; spin < nspin; ++spin)
    {
        rho_pointers[spin] = rho[spin].data();
    }
    ModuleGint::cal_gint_rho(density_pointers, nspin, rho_pointers.data());
    return true;
}

} // namespace alloc
