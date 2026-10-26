#include "source_lcao/module_operator_lcao/alloc_constraint_lcao_operator.h"

#include "source_basis/module_nao/two_center_bundle.h"
#include "source_base/tool_quit.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_constraint_lcao.h"
#include "source_estate/module_constraint/alloc_spin_lcao.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_operator_lcao/overlap.h"

#include <cmath>
#include <memory>
#include <type_traits>

namespace hamilt
{
namespace
{

// Fill an HContainer with the same layout as the DMR channels.
// Both the S4.1 force and the S4.2 stress feed the same adjoint E through the
// two-center overlap kernel, so this mapping must stay in one place.
bool fill_alloc_energy_dm(const UnitCell& ucell,
                          const int norb,
                          const HContainer<double>& layout,
                          const std::vector<double>& energy_density,
                          HContainer<double>& energy_dm,
                          std::string& error)
{
    const Parallel_Orbitals* pv = layout.get_paraV();
    // Branch: the DMR layout must carry its orbital distribution; otherwise
    // dense indices cannot be mapped into this rank's blocks.
    if (pv == nullptr)
    {
        error = "allocation dS term has no Parallel_Orbitals";
        return false;
    }
    for (size_t iap = 0; iap < energy_dm.size_atom_pairs(); ++iap)
    {
        hamilt::AtomPair<double>& pair
            = energy_dm.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        // The energy density is indexed by scalar orbitals, while the DMR
        // layout may be a spinor layout (npol=2 for nspin=4).  Keeping the
        // spinor layout is required because the two-center kernel iterates in
        // npol steps and reads only the (s=0,t=0) slot.
        const int npol = ucell.get_npol();
        const int base_i = ucell.get_iat2iwt()[iat1] / npol;
        const int base_j = ucell.get_iat2iwt()[iat2] / npol;
        // The allocation algebra is evaluated at k=Gamma, so S(k=0) is the
        // sum of every R block.  Its adjoint E therefore multiplies each
        // dS(R), not only dS(R=0).  The real Gamma DMR is already folded to
        // one block, while the nspin=4 DMR retains all periodic images.
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            hamilt::BaseMatrix<double>* matrix
                = pair.find_matrix(pair.get_R_index(iR));
            // Branch: a pair without this R block contributes nothing.
            if (matrix == nullptr)
            {
                continue;
            }
            for (int i = 0; i < matrix->get_row_size(); ++i)
            {
                // Branch: only the first spinor component belongs to the
                // scalar energy adjoint; off-diagonal spin slots stay zero.
                if (rows[i] % npol != 0)
                {
                    continue;
                }
                const int gi = base_i + rows[i] / npol;
                for (int j = 0; j < matrix->get_col_size(); ++j)
                {
                    // Branch: the scalar adjoint is spin diagonal, so only
                    // the (s=0,t=0) slot is populated.
                    if (cols[j] % npol != 0)
                    {
                        continue;
                    }
                    const int gj = base_j + cols[j] / npol;
                    matrix->get_value(i, j)
                        = energy_density[static_cast<size_t>(gi) * norb + gj];
                }
            }
        }
    }
    return true;
}

} // namespace

template <typename TK, typename TR>
void AllocConstraintLcaoOperator<TK, TR>::contributeHR()
{
    // Branch A: S3 collinear path -- real Gamma-only HContainer storage.
    // The kernels are scalar and shared by both spin channels with an
    // explicit injection sign.
    if constexpr (std::is_same<TR, double>::value)
    {
        alloc::AllocConstraintLcao& state
            = alloc::AllocConstraintLcao::instance();
        // Branch: before the first reference SCF the target space and kernels
        // do not exist yet, so the unconstrained Hamiltonian is used.
        if (!state.active() || state.kernels().empty())
        {
            return;
        }
        // Guard: a multiplier/kernel count mismatch would silently omit or
        // mis-pair a constraint.
        if (state.mu_eff().size() != state.kernels().size())
        {
            ModuleBase::WARNING_QUIT(
                "AllocConstraintLcaoOperator::contributeHR",
                "allocation multiplier/kernel count mismatch");
        }
        const int spin = (PARAM.inp.nspin == 2) ? this->current_spin : 0;
        double* h = this->hR->get_wrapper();
        for (size_t alpha = 0; alpha < state.kernels().size(); ++alpha)
        {
            const double coefficient
                = state.mu_eff()[alpha] * state.injection_sign(
                      static_cast<int>(alpha), spin);
            // Branch: a zero multiplier contributes nothing.
            if (coefficient == 0.0)
            {
                continue;
            }
            const HContainer<double>& kernel = state.kernels()[alpha];
            // Guard: a layout mismatch would pair unrelated AO blocks.
            if (kernel.get_nnr() != this->hR->get_nnr())
            {
                ModuleBase::WARNING_QUIT(
                    "AllocConstraintLcaoOperator::contributeHR",
                    "allocation kernel/H(R) layout mismatch");
            }
            const double* k = kernel.get_wrapper();
            for (size_t i = 0; i < this->hR->get_nnr(); ++i)
            {
                h[i] += coefficient * k[i];
            }
        }
    }
    // Branch B: S4.3 non-collinear path -- complex spinor HContainer.  Each
    // kernel already carries K_i tensor (lambda_i.sigma) (or K_i tensor I for
    // a charge constraint), so the contribution is a plain flat add at the
    // single Gamma point.
    else if constexpr (std::is_same<TR, std::complex<double>>::value)
    {
        alloc::AllocSpinConstraintLcao& state
            = alloc::AllocSpinConstraintLcao::instance();
        // Branch: before the first allocation build there is no kernel and
        // the unconstrained Hamiltonian is used.
        if (!state.active() || state.kernels().empty())
        {
            return;
        }
        std::complex<double>* h = this->hR->get_wrapper();
        for (const HContainer<std::complex<double>>& kernel : state.kernels())
        {
            // Guard: a layout mismatch would pair unrelated spinor blocks.
            if (kernel.get_nnr() != this->hR->get_nnr())
            {
                ModuleBase::WARNING_QUIT(
                    "AllocConstraintLcaoOperator::contributeHR",
                    "non-collinear allocation kernel/H(R) layout mismatch");
            }
            const std::complex<double>* k = kernel.get_wrapper();
            for (size_t i = 0; i < this->hR->get_nnr(); ++i)
            {
                h[i] += k[i];
            }
        }
    }
}

template class AllocConstraintLcaoOperator<double, double>;
template class AllocConstraintLcaoOperator<std::complex<double>, double>;
template class AllocConstraintLcaoOperator<std::complex<double>, std::complex<double>>;

bool cal_alloc_constraint_dS_force(
    const UnitCell& ucell,
    const Grid_Driver& gd,
    const LCAO_Orbitals& orb,
    const TwoCenterBundle& two_center_bundle,
    const std::vector<const HContainer<double>*>& dmr,
    const int nspin,
    const std::vector<double>& mu,
    ModuleBase::matrix& force,
    std::string& error)
{
    const bool real_active = alloc::AllocConstraintLcao::instance().active();
    const bool spin_active = alloc::AllocSpinConstraintLcao::instance().active();
    // Branch: no dense target data exists before the first allocation build;
    // the first reference SCF deliberately has no constraint force either.
    if (!real_active && !spin_active)
    {
        return true;
    }
    // Guard: the two-center path needs the usual Gamma DMR layout.
    if (dmr.empty() || dmr[0] == nullptr)
    {
        error = "allocation dS force has no DMR layout";
        return false;
    }

    std::vector<double> energy_density;
    int norb = 0;
    if (nspin == 4)
    {
        // Guard: the spinor closure must have built its own target state; a
        // real active state here would indicate a stale allocation geometry.
        if (!spin_active)
        {
            error = "non-collinear allocation dS force has no active spin state";
            return false;
        }
        if (!alloc::AllocSpinConstraintLcao::instance()
                 .s_derivative_force_density(ucell, dmr, energy_density,
                                             error))
        {
            return false;
        }
        norb = alloc::AllocSpinConstraintLcao::instance().norb();
    }
    else
    {
        // Guard: collinear dS force requires the real allocation state.
        if (!real_active)
        {
            error = "collinear allocation dS force has no active real state";
            return false;
        }
        if (!alloc::AllocConstraintLcao::instance().s_derivative_force_density(
                ucell, dmr, nspin, mu, energy_density, error))
        {
            return false;
        }
        norb = alloc::AllocConstraintLcao::instance().norb();
    }
    // Branch: an all-zero adjoint (all multipliers zero, or a reference SCF)
    // contributes nothing and need not touch the two-center tables.
    bool nonzero = false;
    for (const double value : energy_density)
    {
        // Branch: one non-round-off element is enough to run the kernel.
        if (std::abs(value) > 1e-16)
        {
            nonzero = true;
            break;
        }
    }
    if (!nonzero)
    {
        return true;
    }

    // The two-center overlap kernel advances the density pointer in npol
    // steps and reads only (s=0,t=0).  Copying the DMR container preserves
    // that spinor layout for nspin=4 and the scalar layout otherwise.
    std::unique_ptr<hamilt::HContainer<double>> energy_dm
        = std::make_unique<hamilt::HContainer<double>>(*dmr[0]);
    energy_dm->set_zero();
    if (!fill_alloc_energy_dm(ucell, norb, *energy_dm, energy_density,
                              *energy_dm, error))
    {
        return false;
    }
    const std::vector<ModuleBase::Vector3<double>> kvec_d(
        1, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    hamilt::Overlap<hamilt::OperatorLCAO<double, double>> overlap_force(
        nullptr, kvec_d, nullptr, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb.get());
    ModuleBase::matrix force_s(ucell.nat, 3);
    ModuleBase::matrix stress_s(3, 3);
    overlap_force.cal_force_stress(true, false, energy_dm.get(), force_s,
                                   stress_s);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            // The overlap kernel reports +Tr(E dS/dR) for its (energy
            // density, overlap) pair.  The constraint derivative enters the
            // total energy with the opposite sign, so its force is the
            // negative of that contraction.
            force(iat, dim) -= force_s(iat, dim);
        }
    }
    return true;
}

bool cal_alloc_constraint_dS_stress(
    const UnitCell& ucell,
    const Grid_Driver& gd,
    const LCAO_Orbitals& orb,
    const TwoCenterBundle& two_center_bundle,
    const std::vector<const HContainer<double>*>& dmr,
    const int nspin,
    const std::vector<double>& mu,
    ModuleBase::matrix& stress,
    std::string& error)
{
    const bool real_active = alloc::AllocConstraintLcao::instance().active();
    const bool spin_active = alloc::AllocSpinConstraintLcao::instance().active();
    // Branch: no dense target data exists before the first allocation build;
    // the first reference SCF deliberately has no constraint stress either.
    if (!real_active && !spin_active)
    {
        return true;
    }
    // Guard: the two-center path needs the usual Gamma DMR layout.
    if (dmr.empty() || dmr[0] == nullptr)
    {
        error = "allocation dS stress has no DMR layout";
        return false;
    }
    // Guard: the overlap kernel writes a full 3 x 3 tensor.
    if (stress.nr != 3 || stress.nc != 3)
    {
        error = "allocation dS stress buffer must be 3 x 3";
        return false;
    }

    std::vector<double> energy_density;
    int norb = 0;
    if (nspin == 4)
    {
        // Guard: the spinor closure must own the active target state.
        if (!spin_active)
        {
            error = "non-collinear allocation dS stress has no active spin state";
            return false;
        }
        if (!alloc::AllocSpinConstraintLcao::instance()
                 .s_derivative_force_density(ucell, dmr, energy_density,
                                             error))
        {
            return false;
        }
        norb = alloc::AllocSpinConstraintLcao::instance().norb();
    }
    else
    {
        // Guard: collinear dS stress requires the real allocation state.
        if (!real_active)
        {
            error = "collinear allocation dS stress has no active real state";
            return false;
        }
        if (!alloc::AllocConstraintLcao::instance().s_derivative_force_density(
                ucell, dmr, nspin, mu, energy_density, error))
        {
            return false;
        }
        norb = alloc::AllocConstraintLcao::instance().norb();
    }
    // Branch: an all-zero adjoint contributes nothing and need not touch the
    // two-center tables.
    bool nonzero = false;
    for (const double value : energy_density)
    {
        // Branch: one non-round-off element is enough to run the kernel.
        if (std::abs(value) > 1e-16)
        {
            nonzero = true;
            break;
        }
    }
    if (!nonzero)
    {
        return true;
    }

    // The two-center overlap kernel advances the density pointer in npol
    // steps and reads only (s=0,t=0).  Copying the DMR container preserves
    // that spinor layout for nspin=4 and the scalar layout otherwise.
    std::unique_ptr<hamilt::HContainer<double>> energy_dm
        = std::make_unique<hamilt::HContainer<double>>(*dmr[0]);
    energy_dm->set_zero();
    if (!fill_alloc_energy_dm(ucell, norb, *energy_dm, energy_density,
                              *energy_dm, error))
    {
        return false;
    }

    const std::vector<ModuleBase::Vector3<double>> kvec_d(
        1, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    hamilt::Overlap<hamilt::OperatorLCAO<double, double>> overlap_stress(
        nullptr, kvec_d, nullptr, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb.get());
    ModuleBase::matrix force_s(ucell.nat, 3);
    ModuleBase::matrix stress_s(3, 3);
    overlap_stress.cal_force_stress(false, true, energy_dm.get(), force_s,
                                    stress_s);
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            // ABACUS stress = -(1/Omega) dE/d eps, while the Overlap kernel
            // returns +Tr(E dS/d eps)/Omega for this adjoint.
            stress(a, b) -= stress_s(a, b);
        }
    }
    return true;
}

} // namespace hamilt
