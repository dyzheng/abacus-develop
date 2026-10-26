#include "source_estate/module_constraint/alloc_spin_lcao.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_constraint.h"
#include "source_estate/module_constraint/alloc_lcao.h"
#include "source_estate/module_constraint/alloc_spin.h"
#include "source_estate/module_constraint/constraint_inject_lcao.h"
#include "source_lcao/module_gint/gint_info.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

#include <algorithm>
#include <cmath>
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

int scalar_norb(const UnitCell& ucell)
{
    int norb = 0;
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        norb += ucell.atoms[ucell.iat2it[iat]].nw;
    }
    return norb;
}

std::vector<int> scalar_iat2iwt(const UnitCell& ucell, const int npol)
{
    std::vector<int> table(ucell.nat + 1, 0);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        // Guard: the spinor layout must be exactly npol copies of the scalar
        // orbitals before any index can be divided by npol.
        if (ucell.get_iat2iwt()[iat] % npol != 0)
        {
            throw std::invalid_argument(
                "spinor iat2iwt is not divisible by npol");
        }
        table[iat] = ucell.get_iat2iwt()[iat] / npol;
    }
    table[ucell.nat] = scalar_norb(ucell);
    return table;
}

std::vector<int> select_target_orbitals(const UnitCell& ucell,
                                        const std::vector<int>& base,
                                        const int target_l,
                                        std::vector<int>& target_atom)
{
    std::vector<int> orbital;
    target_atom.clear();
    // Branch: a negative shell index selects nothing.
    if (target_l < 0)
    {
        return orbital;
    }
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const Atom& atom = ucell.atoms[ucell.iat2it[iat]];
        // Branch: this atom type has no shell with the requested angular
        // momentum, so it contributes only to the complementary channel.
        if (target_l > atom.nwl
            || target_l >= static_cast<int>(atom.l_nchi.size())
            || atom.l_nchi[target_l] <= 0)
        {
            continue;
        }
        int offset = 0;
        for (int L = 0; L < target_l; ++L)
        {
            offset += atom.l_nchi[L] * (2 * L + 1);
        }
        for (int n = 0; n < atom.l_nchi[target_l]; ++n)
        {
            for (int m = 0; m < 2 * target_l + 1; ++m)
            {
                orbital.push_back(base[iat] + offset
                                  + n * (2 * target_l + 1) + m);
                target_atom.push_back(iat);
            }
        }
    }
    return orbital;
}

template <typename F>
void for_each_complex_spinor_block(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& H,
    F&& callback)
{
    const Parallel_Orbitals* pv = H.get_paraV();
    // Guard: without the AO distribution local block indices cannot be
    // mapped to global scalar orbitals.
    if (pv == nullptr)
    {
        throw std::invalid_argument("spinor HContainer has no paraV");
    }
    const int npol = ucell.get_npol();
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<std::complex<double>>& pair
            = H.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
            const hamilt::BaseMatrix<std::complex<double>>* matrix
                = pair.find_matrix(R);
            // Branch: a pair without this R block contributes nothing.
            if (matrix == nullptr)
            {
                continue;
            }
            for (int i = 0; i < matrix->get_row_size(); ++i)
            {
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    callback(iat1, iat2, rows[i], cols[j],
                             matrix->get_value(i, j));
                }
            }
        }
    }
}

template <typename F>
void for_each_real_scalar_block(
    const UnitCell& ucell,
    const std::vector<int>& scalar_base,
    const hamilt::HContainer<double>& H,
    const Parallel_Orbitals* pv,
    F&& callback)
{
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair
            = H.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
            const hamilt::BaseMatrix<double>* matrix = pair.find_matrix(R);
            // Branch: absent R blocks contribute zero to the k=Gamma sum.
            if (matrix == nullptr)
            {
                continue;
            }
            for (int i = 0; i < matrix->get_row_size(); ++i)
            {
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    callback(scalar_base[iat1] + rows[i],
                             scalar_base[iat2] + cols[j],
                             matrix->get_value(i, j));
                }
            }
        }
    }
}

std::complex<double> pauli_entry(const ModuleBase::Vector3<double>& lambda,
                                 const int row,
                                 const int col)
{
    // Branch: sigma_z is diagonal and changes sign between the two spins.
    if (row == col)
    {
        return std::complex<double>(row == 0 ? lambda.z : -lambda.z, 0.0);
    }
    // Branch: the upper off-diagonal is lambda_x - i lambda_y; the lower one
    // is its conjugate.
    if (row == 0)
    {
        return std::complex<double>(lambda.x, -lambda.y);
    }
    return std::complex<double>(lambda.x, lambda.y);
}

} // namespace

bool extract_spinor_scalar_overlap(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& S,
    const int norb,
    std::vector<double>& S_scalar,
    std::string& error)
{
    try
    {
        const int npol = ucell.get_npol();
        // Guard: this adapter is the spinor (nspin=4) counterpart, where each
        // scalar AO has exactly two spin components.
        if (npol != 2)
        {
            error = "spinor overlap extraction requires npol = 2";
            return false;
        }
        const std::vector<int> base = scalar_iat2iwt(ucell, npol);
        S_scalar.assign(static_cast<size_t>(norb) * norb, 0.0);
        for_each_complex_spinor_block(ucell, S,
            [&](const int iat1, const int iat2, const int r, const int c,
                const std::complex<double>& value) {
                const int s = r % npol;
                const int t = c % npol;
                // Branch: only the spin-diagonal (s == t) channel is the
                // scalar overlap; the off-diagonal blocks are zero for a
                // spin-independent overlap operator.
                if (s == t)
                {
                    const int mu = base[iat1] + r / npol;
                    const int nu = base[iat2] + c / npol;
                    S_scalar[static_cast<size_t>(mu) * norb + nu]
                        += value.real();
                }
            });
        reduce_pool(S_scalar);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
    return true;
}

bool extract_pauli_density(const UnitCell& ucell,
                           const hamilt::HContainer<double>& dmr,
                           const int norb,
                           std::vector<double>& rho0,
                           std::vector<double>& rhox,
                           std::vector<double>& rhoy,
                           std::vector<double>& rhoz,
                           std::string& error)
{
    const Parallel_Orbitals* pv = dmr.get_paraV();
    // Guard: a layout-less DMR cannot be mapped to global scalar orbitals.
    if (pv == nullptr)
    {
        error = "Pauli DMR has no Parallel_Orbitals layout";
        return false;
    }
    const int npol = ucell.get_npol();
    // Guard: the Pauli packing is defined only for the two-spinor case.
    if (npol != 2)
    {
        error = "Pauli density extraction requires npol = 2";
        return false;
    }
    const std::vector<int> base = scalar_iat2iwt(ucell, npol);
    const size_t size = static_cast<size_t>(norb) * norb;
    rho0.assign(size, 0.0);
    rhox.assign(size, 0.0);
    rhoy.assign(size, 0.0);
    rhoz.assign(size, 0.0);

    for (size_t iap = 0; iap < dmr.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair
            = dmr.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
            const hamilt::BaseMatrix<double>* matrix = pair.find_matrix(R);
            // Branch: a missing R block contributes nothing at k=Gamma.
            if (matrix == nullptr)
            {
                continue;
            }
            // Guard: the local row/column maps must cover exactly this
            // block.  In a 2D block-cyclic distribution an atom block need
            // not start at spinor component zero, so deriving the spin index
            // from the local matrix index alone would be wrong in MPI.
            if (rows.size() != static_cast<size_t>(matrix->get_row_size())
                || cols.size()
                       != static_cast<size_t>(matrix->get_col_size()))
            {
                error = "Pauli DMR block/local index mismatch";
                return false;
            }
            for (int i = 0; i < matrix->get_row_size(); ++i)
            {
                const int s = rows[i] % npol;
                const int mu = base[iat1] + rows[i] / npol;
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    const int t = cols[j] % npol;
                    const int nu = base[iat2] + cols[j] / npol;
                    const size_t index = static_cast<size_t>(mu) * norb + nu;
                    // The four interleaved slots written by
                    // func_xyz_to_updown are (rho0, rho_x, rho_y, rho_z).
                    // Branch (s,t): the spinor slot selects exactly one Pauli
                    // component, independently of the local block origin.
                    if (s == 0 && t == 0)
                    {
                        rho0[index] += matrix->get_value(i, j);
                    }
                    else if (s == 0 && t == 1)
                    {
                        rhox[index] += matrix->get_value(i, j);
                    }
                    else if (s == 1 && t == 0)
                    {
                        rhoy[index] += matrix->get_value(i, j);
                    }
                    else
                    {
                        rhoz[index] += matrix->get_value(i, j);
                    }
                }
            }
        }
    }
    reduce_pool(rho0);
    reduce_pool(rhox);
    reduce_pool(rhoy);
    reduce_pool(rhoz);
    return true;
}

bool build_spinor_kernel_container(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& layout,
    const std::vector<int>& scalar_iat2iwt,
    const std::vector<double>& K,
    const int norb,
    const constraint::ConstraintKind kind,
    const ModuleBase::Vector3<double>& lambda,
    hamilt::HContainer<std::complex<double>>& out,
    std::string& error)
{
    // Guard: K is one scalar norb x norb matrix; a wrong size would index
    // outside the dense allocation below.
    if (static_cast<int>(K.size()) != norb * norb)
    {
        error = "allocation spinor kernel has wrong K size";
        return false;
    }
    // Guard: the scalar base table must cover every atom.
    if (static_cast<int>(scalar_iat2iwt.size()) < ucell.nat)
    {
        error = "allocation spinor kernel has an incomplete atom table";
        return false;
    }
    const Parallel_Orbitals* pv = layout.get_paraV();
    // Guard: the complex HContainer layout must carry its distribution.
    if (pv == nullptr)
    {
        error = "allocation spinor kernel layout has no Parallel_Orbitals";
        return false;
    }
    const int npol = ucell.get_npol();
    // Guard: only the two-spinor expansion is implemented.
    if (npol != 2)
    {
        error = "allocation spinor kernel requires npol = 2";
        return false;
    }

    out = hamilt::HContainer<std::complex<double>>(layout);
    out.set_zero();
    for (size_t iap = 0; iap < out.size_atom_pairs(); ++iap)
    {
        hamilt::AtomPair<std::complex<double>>& pair
            = out.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        hamilt::BaseMatrix<std::complex<double>>* matrix
            = pair.find_matrix(0, 0, 0);
        // Branch: only the R=0 block exists in the single-Gamma closure; a
        // pair without one contributes nothing to H(k=0).
        if (matrix == nullptr)
        {
            continue;
        }
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        // Guard: the local row/column maps must cover exactly this block.
        // This keeps the spinor component and scalar orbital independent of
        // where a 2D block-cyclic distribution happens to start the block.
        if (rows.size() != static_cast<size_t>(matrix->get_row_size())
            || cols.size() != static_cast<size_t>(matrix->get_col_size()))
        {
            error = "allocation spinor kernel block/local index mismatch";
            return false;
        }
        for (int i = 0; i < matrix->get_row_size(); ++i)
        {
            const int s = rows[i] % npol;
            const int mu = scalar_iat2iwt[iat1] + rows[i] / npol;
            for (int j = 0; j < matrix->get_col_size(); ++j)
            {
                const int t = cols[j] % npol;
                const int nu = scalar_iat2iwt[iat2] + cols[j] / npol;
                const double k = K[static_cast<size_t>(mu) * norb + nu];
                std::complex<double> value(0.0, 0.0);
                // Branch: charge constraints couple through the identity in
                // spin space; spin constraints use the full lambda.sigma
                // block.  Off-diagonal charge slots stay zero.
                if (kind == constraint::ConstraintKind::Spin)
                {
                    value = k * pauli_entry(lambda, s, t);
                }
                else if (s == t)
                {
                    value = std::complex<double>(k, 0.0);
                }
                matrix->get_value(i, j) = value;
            }
        }
    }
    return true;
}

AllocSpinConstraintLcao& AllocSpinConstraintLcao::instance()
{
    static AllocSpinConstraintLcao state;
    return state;
}

void AllocSpinConstraintLcao::reset()
{
    active_ = false;
    nconstraint_ = 0;
    norb_ = 0;
    ntarget_ = 0;
    rank_tol_ = 1e-10;
    ucell_ = nullptr;
    space_ = AllocConstraintAlgebra{};
    kinds_.clear();
    W_dense_.clear();
    K_dense_.clear();
    lambda_.clear();
    scalar_iat2iwt_.clear();
    layout_template_.reset();
    kernels_.clear();
}

bool AllocSpinConstraintLcao::build(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& S,
    const hamilt::HContainer<std::complex<double>>& h_layout,
    const std::vector<std::vector<double>>& cw,
    const std::vector<std::vector<int>>& constraint_atoms,
    const std::vector<constraint::ConstraintKind>& kinds,
    const std::vector<ModuleBase::Vector3<double>>& lambda,
    ModuleGint::GintInfo* gint_info,
    const Parallel_Orbitals* paraV,
    const int target_l,
    const double rank_tol,
    std::string& error)
{
    reset();
    // Guard: the Gint layout is required to integrate the constraint weight
    // matrices, and the three per-constraint arrays must be parallel.
    if (gint_info == nullptr)
    {
        error = "spin allocation constraint needs an initialized GintInfo";
        return false;
    }
    if (cw.size() != constraint_atoms.size() || cw.size() != kinds.size()
        || cw.size() != lambda.size())
    {
        error = "spin allocation weight/atom/kind/lambda lists differ";
        return false;
    }
    if (cw.empty())
    {
        error = "spin allocation constraint has no constrained components";
        return false;
    }
    // Guard: the class is the nspin=4 spinor closure only.
    if (ucell.get_npol() != 2)
    {
        error = "spin allocation constraint requires npol = 2 (nspin=4)";
        return false;
    }
    // Guard: the weight matrices below are scalar AO matrices; the caller
    // must not hand in a spinor-doubled distribution.
    if (paraV == nullptr || paraV->get_global_row_size() % 2 != 0
        || paraV->get_global_col_size() % 2 != 0)
    {
        error = "spin allocation constraint needs an even spinor paraV";
        return false;
    }

    try
    {
        nconstraint_ = static_cast<int>(cw.size());
        ucell_ = &ucell;
        kinds_ = kinds;
        lambda_ = lambda;
        rank_tol_ = rank_tol;
        norb_ = scalar_norb(ucell);
        scalar_iat2iwt_ = scalar_iat2iwt(ucell, ucell.get_npol());

        std::vector<double> S_scalar;
        if (!extract_spinor_scalar_overlap(ucell, S, norb_, S_scalar, error))
        {
            return false;
        }

        std::vector<int> target_atom;
        const std::vector<int> target_orbital
            = select_target_orbitals(ucell, scalar_iat2iwt_, target_l,
                                     target_atom);
        ntarget_ = static_cast<int>(target_orbital.size());
        // Branch: without a target orbital the allocation operator degenerates
        // to the plain spatial weight and S3/S4 require at least one.
        if (ntarget_ == 0)
        {
            error = "spin allocation constraint selected zero target orbitals";
            return false;
        }

        std::vector<int> target_constraint(ntarget_, -1);
        for (int a = 0; a < ntarget_; ++a)
        {
            int owner = -1;
            for (int alpha = 0; alpha < nconstraint_; ++alpha)
            {
                const std::vector<int>& atoms = constraint_atoms[alpha];
                // Branch: a target atom owned by this constraint is assigned
                // to it; a second owner is an ambiguous fragment.
                if (std::find(atoms.begin(), atoms.end(),
                              target_atom[a]) != atoms.end())
                {
                    if (owner >= 0)
                    {
                        error = "target atom belongs to more than one constraint";
                        return false;
                    }
                    owner = alpha;
                }
            }
            target_constraint[a] = owner;
        }

        LcaoTargetSpace target;
        std::vector<double> C(static_cast<size_t>(norb_) * ntarget_, 0.0);
        for (int a = 0; a < ntarget_; ++a)
        {
            C[static_cast<size_t>(target_orbital[a]) * ntarget_ + a] = 1.0;
        }
        build_lcao_target(S_scalar, norb_, C, ntarget_, target_atom, target,
                          rank_tol);
        build_alloc_constraint_algebra(S_scalar, norb_, target_orbital,
                                       target_atom, target_constraint,
                                       target.ST, nconstraint_, space_,
                                       rank_tol);

        // Build the scalar W_i.  The nspin=4 production distribution is
        // spinor-doubled, so derive a scalar Parallel_Orbitals with the same
        // 2D block-cyclic structure for the scalar Gint integration.
        Parallel_Orbitals scalar_pv;
        const int mg = paraV->get_global_row_size() / 2;
        const int ng = paraV->get_global_col_size() / 2;
        const int nb = paraV->get_block_size() / 2;
#ifdef __MPI
        scalar_pv.set(mg, ng, nb, paraV->blacs_ctxt);
#else
        (void)nb;
        scalar_pv.set_serial(mg, ng);
#endif
        scalar_pv.set_atomic_trace(scalar_iat2iwt_.data(), ucell.nat, norb_);

        std::vector<hamilt::HContainer<double>> weights;
        try
        {
            weights = constraint::ConstraintInjectLCAO::build(
                cw, gint_info, &scalar_pv, nullptr);
        }
        catch (const std::exception& e)
        {
            error = e.what();
            return false;
        }
        if (weights.size() != cw.size())
        {
            error = "spin allocation weight builder returned the wrong count";
            return false;
        }

        W_dense_.assign(nconstraint_,
                        std::vector<double>(static_cast<size_t>(norb_) * norb_,
                                            0.0));
        for (int alpha = 0; alpha < nconstraint_; ++alpha)
        {
            for_each_real_scalar_block(ucell, scalar_iat2iwt_, weights[alpha],
                &scalar_pv,
                [&](const int gi, const int gj, const double value) {
                    W_dense_[alpha][static_cast<size_t>(gi) * norb_ + gj]
                        += value;
                });
            reduce_pool(W_dense_[alpha]);
        }

        K_dense_.clear();
        K_dense_.reserve(nconstraint_);
        for (int alpha = 0; alpha < nconstraint_; ++alpha)
        {
            std::vector<double> K;
            build_alloc_constraint_kernel(space_, W_dense_[alpha], alpha, K);
            K_dense_.push_back(std::move(K));
        }

        layout_template_
            = std::make_unique<hamilt::HContainer<std::complex<double>>>(
                h_layout);
        layout_template_->set_zero();
        active_ = true;
        if (!rebuild_kernels(error))
        {
            reset();
            return false;
        }
    }
    catch (const std::exception& e)
    {
        error = e.what();
        reset();
        return false;
    }
    return true;
}

bool AllocSpinConstraintLcao::set_lambda(
    const std::vector<ModuleBase::Vector3<double>>& lambda,
    std::string& error)
{
    // Guard: the multiplier list is one vector per constraint.
    if (lambda.size() != static_cast<size_t>(nconstraint_))
    {
        error = "spin allocation lambda count does not match constraints";
        return false;
    }
    // Branch: repeated fixed-vector calls with the same values are no-ops.
    bool changed = lambda.size() != lambda_.size();
    for (size_t a = 0; a < lambda.size() && !changed; ++a)
    {
        changed = lambda[a].x != lambda_[a].x || lambda[a].y != lambda_[a].y
                  || lambda[a].z != lambda_[a].z;
    }
    if (!changed)
    {
        return true;
    }
    lambda_ = lambda;
    return rebuild_kernels(error);
}

bool AllocSpinConstraintLcao::rebuild_kernels(std::string& error)
{
    // Guard: kernels can only be rebuilt after build() prepared the scalar
    // K_i and the layout template.
    if (!active_ || !layout_template_)
    {
        error = "spin allocation state is not initialized";
        return false;
    }
    if (K_dense_.size() != static_cast<size_t>(nconstraint_)
        || lambda_.size() != static_cast<size_t>(nconstraint_))
    {
        error = "spin allocation kernel inputs are inconsistent";
        return false;
    }
    kernels_.clear();
    kernels_.reserve(nconstraint_);
    for (int alpha = 0; alpha < nconstraint_; ++alpha)
    {
        hamilt::HContainer<std::complex<double>> kernel;
        if (!build_spinor_kernel_container(
                *ucell_, *layout_template_, scalar_iat2iwt_,
                K_dense_[alpha], norb_,
                kinds_[alpha], lambda_[alpha], kernel, error))
        {
            return false;
        }
        kernels_.push_back(std::move(kernel));
    }
    return true;
}

bool AllocSpinConstraintLcao::project(
    const UnitCell& ucell,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    std::vector<double>& n_alloc,
    std::vector<ModuleBase::Vector3<double>>& m_alloc,
    std::string& error) const
{
    // Guard: projection requires the geometry-fixed target/kernel state.
    if (!active_)
    {
        error = "spin allocation constraint is not initialized";
        return false;
    }
    // Guard: nspin=4 owns exactly one Pauli DMR container.
    if (dmr.size() != 1 || dmr[0] == nullptr)
    {
        error = "spin allocation projection needs exactly one Pauli DMR";
        return false;
    }
    std::vector<double> rho0;
    std::vector<double> rhox;
    std::vector<double> rhoy;
    std::vector<double> rhoz;
    if (!extract_pauli_density(ucell, *dmr[0], norb_, rho0, rhox, rhoy, rhoz,
                               error))
    {
        return false;
    }
    try
    {
        observe_alloc_spin_moments(space_, W_dense_, rho0, rhox, rhoy, rhoz,
                                   n_alloc, m_alloc);
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
    return true;
}

} // namespace alloc
