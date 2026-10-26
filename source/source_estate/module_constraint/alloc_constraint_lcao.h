#ifndef ALLOC_CONSTRAINT_LCAO_H
#define ALLOC_CONSTRAINT_LCAO_H

#include <string>
#include <vector>

#include "source_estate/module_constraint/alloc_lcao.h"

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
 * @brief Production LCAO state for the S3 allocation constraint.
 *
 * The operator is linear in the Lagrange multiplier for a fixed geometry.
 * build() precomputes one unit-multiplier HContainer K_alpha per constraint;
 * every SCF iteration then only performs a mu-weighted sparse add.  The same
 * object owns the orthonormal target space used by the ALM read-out, so the
 * injected operator and the observed allocation trace are built from one C'.
 */
class AllocConstraintLcao
{
  public:
    static AllocConstraintLcao& instance();

    // Clear all state (new geometry / unit tests).
    void reset();
    bool active() const { return active_; }

    /**
     * @brief Build the target space and one K_alpha per constraint.
     *
     * cw[alpha] is the Becke weight field of constraint alpha on the dense
     * grid. constraint_atoms[alpha] lists the physical atoms owned by that
     * constraint. Only Gamma-only real HContainers are accepted.
     */
    bool build(const UnitCell& ucell,
               const hamilt::HContainer<double>& S,
               const hamilt::HContainer<double>& h_layout,
               const std::vector<std::vector<double>>& cw,
               const std::vector<std::vector<int>>& constraint_atoms,
               ModuleGint::GintInfo* gint_info,
               const Parallel_Orbitals* paraV,
               const int target_l,
               const double rank_tol,
               std::string& error);

    // Multiplier applied by the Hamiltonian operator.  Sized to the number of
    // constraints after build(); before that the operator is a no-op.
    void set_mu_eff(const std::vector<double>& mu_eff);
    const std::vector<double>& mu_eff() const { return mu_eff_; }

    // Per-constraint spin-channel signs used by the Hamiltonian operator
    // (charge: +1/+1, spin: +1/-1).  Defaults to the spin profile.
    void set_injection_signs(const std::vector<double>& up,
                             const std::vector<double>& down);
    double injection_sign(const int alpha, const int spin) const;

    const std::vector<hamilt::HContainer<double>>& kernels() const
    {
        return kernels_;
    }

    int nconstraint() const { return nconstraint_; }
    int ntarget() const { return ntarget_; }
    int norb() const { return norb_; }
    int rank() const { return target_rank_; }
    double cond() const { return target_cond_; }
    const std::vector<int>& target_constraint() const
    {
        return target_constraint_;
    }

    /**
     * @brief Read the diagnostic target-subspace n_T/m_T per constraint.
     *
     * dmr contains one Gamma-only real HContainer per spin channel.  The
     * Outputs n_T and m_T aggregated over the target orbitals owned by each
     * constraint.  These are diagnostics only; the ALM uses
     * project_allocations() so that the observable matches the injected
     * K_i operator exactly.
     */
    bool project_moments(
        const UnitCell& ucell,
        const hamilt::HContainer<double>& S,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nspin,
        std::vector<double>& n_T_constraint,
        std::vector<double>& m_T_constraint,
        std::string& error) const;

    /**
     * @brief Read the full allocation observables Tr(DMR K_alpha).
     *
     * Unlike project_moments(), which reports the target-subspace n_T/m_T,
     * these are the quantities constrained by the Hamiltonian term
     * sum_alpha mu_alpha K_alpha: n_i = Tr[(D_up+D_dn) K_i] for charge and
     * m_i = Tr[(D_up-D_dn) K_i] for spin.  The kernels are geometry-fixed
     * and identical to the injected operator.
     */
    bool project_allocations(
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nspin,
        std::vector<double>& n_alloc,
        std::vector<double>& m_alloc,
        std::string& error) const;

    /**
     * @brief Build the complementary density Q_c D Q_c on the Gint grid.
     *
     * The frozen-target force uses this density in the existing weight-
     * derivative kernel.  DMR channels are up/down for nspin=2.  `nrxx` is
     * the dense pw_rhod grid length on this rank.
     */
    bool complement_density(
        const UnitCell& ucell,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nspin,
        const int nrxx,
        std::vector<std::vector<double>>& rho,
        std::string& error) const;

    /**
     * @brief Energy-density matrix for the explicit dK_i/dS force term.
     *
     * Returns E = sum_i mu_i E_i, where
     *   Tr(D_i dK_i) = sum_{mu,nu} E_i[mu,nu] dS[mu,nu]
     * at fixed weight matrices W_i.  The caller contracts E with the
     * two-center overlap derivatives, which produces
     *   F_S = -Tr(E dS).
     *
     * D_i is the per-constraint channel combination
     * read_up * D_up + read_dn * D_down, matching the injected signs for
     * charge and spin constraints.
     */
    bool s_derivative_force_density(
        const UnitCell& ucell,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nspin,
        const std::vector<double>& mu,
        std::vector<double>& energy_density,
        std::string& error) const;

  private:
    AllocConstraintLcao() = default;

    bool active_ = false;
    int nconstraint_ = 0;
    int ntarget_ = 0;
    int norb_ = 0;
    int target_rank_ = 0;
    double target_cond_ = 1.0;
    LcaoTargetSpace target_;
    std::vector<int> target_constraint_;
    std::vector<double> SB_; // norb x ntarget
    std::vector<double> S_dense_; // norb x norb, overlap at build geometry
    std::vector<std::vector<double>> W_dense_; // per-constraint weight matrices
    double rank_tol_ = 1e-10;
    std::vector<hamilt::HContainer<double>> kernels_;
    std::vector<double> mu_eff_;
    std::vector<double> injection_up_;
    std::vector<double> injection_down_;
    ModuleGint::GintInfo* gint_info_ = nullptr;
};

} // namespace alloc

#endif
