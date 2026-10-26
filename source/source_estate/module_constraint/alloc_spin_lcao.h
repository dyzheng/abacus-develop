#ifndef ALLOC_SPIN_LCAO_H
#define ALLOC_SPIN_LCAO_H

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "source_base/matrix.h"
#include "source_base/vector3.h"
#include "source_estate/module_constraint/alloc_constraint.h"
#include "source_estate/module_constraint/constraint_io.h"

class UnitCell;
class Parallel_Orbitals;

namespace ModuleGint
{
class GintInfo;
}

namespace hamilt
{
template <typename T>
class HContainer;
}

namespace alloc
{

/**
 * @brief Production LCAO state for the S4.3 non-collinear allocation path.
 *
 * Scope: no-SOC, LCAO, one Gamma k-point, nspin=4.  The scalar allocation
 * algebra is built from the spin-diagonal block of the complex S(R) and from
 * scalar constraint-weight matrices W_i(R); the injected operator is
 *
 *   Delta H(k=Gamma) = sum_i K_i tensor (lambda_i . sigma)
 *
 * for spin constraints and K_i tensor I for charge constraints.  Only the
 * R=0 block is populated, which is exact for the single Gamma point because
 * folding sums every R block at k=0.  Multi-k runs are refused.
 */
class AllocSpinConstraintLcao
{
  public:
    static AllocSpinConstraintLcao& instance();

    void reset();
    bool active() const { return active_; }

    /**
     * @brief Build the scalar target space, W_i and K_i and expand each
     * constraint into the complex spinor Hamiltonian layout.
     *
     * cw[alpha] is the Becke weight field on the dense grid and
     * constraint_atoms[alpha] lists the physical atoms of that fragment.
     * `kinds[alpha]` selects spin (lambda.sigma) or charge (identity) spin
     * structure.  lambda is one three-vector per constraint.
     */
    bool build(const UnitCell& ucell,
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
               std::string& error);

    // Replace the three-component multipliers and rebuild the spinor kernels.
    // Fixed-vector production currently calls this once; the API is kept
    // separate so a future vector schedule can advance lambda in-SCF.
    bool set_lambda(const std::vector<ModuleBase::Vector3<double>>& lambda,
                    std::string& error);

    const std::vector<ModuleBase::Vector3<double>>& lambda() const
    {
        return lambda_;
    }
    const std::vector<hamilt::HContainer<std::complex<double>>>& kernels() const
    {
        return kernels_;
    }

    int nconstraint() const { return nconstraint_; }
    int norb() const { return norb_; }
    int ntarget() const { return ntarget_; }
    int rank() const { return space_.target.rank; }
    double cond() const { return space_.target.cond; }

    /**
     * @brief Read n_i = Tr[rho K_i] and m_i = Tr[rho_alpha K_i].
     *
     * dmr[0] is the real nspin=4 Pauli DMR (rho0/rhox/rhoy/rhoz in its four
     * interleaved slots).  The contraction uses the same scalar K_i as the
     * injected operator and the dense Pauli read-out in alloc_spin.cpp.
     */
    bool project(
        const UnitCell& ucell,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        std::vector<double>& n_alloc,
        std::vector<ModuleBase::Vector3<double>>& m_alloc,
        std::string& error) const;

    /**
     * @brief Build one complementary grid density per constraint.
     *
     * The scalar density of constraint alpha is
     *
     *   D_alpha = lambda_alpha . (rho_x, rho_y, rho_z)   (spin)
     *   D_alpha = lambda_alpha.z * rho_0                 (charge)
     *
     * and the returned grid field is sum_R Q_c D_alpha(R) Q_c.  The spinor
     * DMR retains one block per periodic image, so every block is projected
     * at its own R before Gint folds the real-space density; projecting only
     * the Gamma-folded sum would lose the periodic decomposition.  lambda is
     * already folded into the density, so the grid force/stress kernel must
     * use a unit coefficient for every entry.  dmr[0] is the real nspin=4
     * Pauli DMR; the four output functions are cheap to consume in the shared
     * weight-derivative kernels without going through a collinear channel
     * profile.
     *
     * The complementary density is the contraction coefficient of dW_i in
     * Tr(D dK_i), so W_i also contributes through the basis-function
     * derivative at fixed weight values.  That Pulay term is the local
     * potential force of w_i acting on the same complementary density; it is
     * returned in weight_pulay_force and must be added to the grid
     * weight-derivative force.
     */
    bool complement_densities(
        const UnitCell& ucell,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nrxx,
        std::vector<std::vector<double>>& rho,
        ModuleBase::matrix& weight_pulay_force,
        std::string& error) const;

    /**
     * @brief Energy-density adjoint of the non-collinear allocation energy.
     *
     * Returns E = sum_alpha E_alpha with
     * Tr(D_alpha dK_alpha) = sum_{mu,nu} E[mu,nu] dS[mu,nu] at fixed W.
     * D_alpha uses the same lambda-folded Pauli combination as the grid
     * density above.  The result is contracted with the two-center overlap
     * derivative by cal_alloc_constraint_dS_force/stress().
     */
    bool s_derivative_force_density(
        const UnitCell& ucell,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        std::vector<double>& energy_density,
        std::string& error) const;

  private:
    AllocSpinConstraintLcao() = default;

    bool rebuild_kernels(std::string& error);

    bool active_ = false;
    int nconstraint_ = 0;
    int norb_ = 0;
    int ntarget_ = 0;
    double rank_tol_ = 1e-10;
    const UnitCell* ucell_ = nullptr;
    AllocConstraintAlgebra space_;
    std::vector<constraint::ConstraintKind> kinds_;
    // Scalar overlap at the build geometry.  The dS adjoint differentiates
    // only this matrix; the weight matrices W_i are fixed in the frozen
    // target-space force/stress term.
    std::vector<double> S_dense_;
    std::vector<std::vector<double>> W_dense_;
    std::vector<std::vector<double>> K_dense_;
    // Constraint weight fields and the Gint layout used by the W_i Pulay
    // force.  Both are geometry-fixed and set by build().
    std::vector<std::vector<double>> cw_;
    ModuleGint::GintInfo* gint_info_ = nullptr;
    std::vector<ModuleBase::Vector3<double>> lambda_;
    std::vector<int> scalar_iat2iwt_;
    // Zeroed complex layout used as the template for every spinor kernel.
    std::unique_ptr<hamilt::HContainer<std::complex<double>>> layout_template_;
    std::vector<hamilt::HContainer<std::complex<double>>> kernels_;
};

/**
 * @brief Sum the spin-diagonal block of a complex S(R) into a scalar S.
 *
 * Exact only at a single Gamma k-point: the returned matrix is sum_R S(R).
 */
bool extract_spinor_scalar_overlap(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& S,
    const int norb,
    std::vector<double>& S_scalar,
    std::string& error);

/**
 * @brief Read the four Pauli components of a real nspin=4 DMR.
 *
 * The DMR stores (rho0, rho_x, rho_y, rho_z) in the four interleaved 2x2
 * spinor slots written by DensityMatrix_Tools::func_xyz_to_updown.
 */
bool extract_pauli_density(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& dmr,
    const int norb,
    std::vector<double>& rho0,
    std::vector<double>& rhox,
    std::vector<double>& rhoy,
    std::vector<double>& rhoz,
    std::string& error);

/**
 * @brief Expand one scalar K into a complex spinor HContainer at R=0.
 *
 * For Spin the block is K tensor (lambda.sigma); for Charge it is K tensor I.
 * All other R blocks stay zero, which is exact for a single Gamma point.
 */
bool build_spinor_kernel_container(
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& layout,
    const std::vector<int>& scalar_iat2iwt,
    const std::vector<double>& K,
    const int norb,
    const constraint::ConstraintKind kind,
    const ModuleBase::Vector3<double>& lambda,
    hamilt::HContainer<std::complex<double>>& out,
    std::string& error);

} // namespace alloc

#endif
