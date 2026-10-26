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
 * injected operator and the observed m_T are built from one C'.
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
     * @brief Read one m_T value per constraint from the current DMR.
     *
     * dmr contains one Gamma-only real HContainer per spin channel.  The
     * output is the ALM observable s_alpha = sum_{a in alpha} m_T[a].
     */
    bool project_moments(
        const UnitCell& ucell,
        const hamilt::HContainer<double>& S,
        const std::vector<const hamilt::HContainer<double>*>& dmr,
        const int nspin,
        std::vector<double>& m_T_constraint,
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
    std::vector<hamilt::HContainer<double>> kernels_;
    std::vector<double> mu_eff_;
};

} // namespace alloc

#endif
