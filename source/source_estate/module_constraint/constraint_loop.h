#ifndef CONSTRAINT_LOOP_H
#define CONSTRAINT_LOOP_H

#include <memory>
#include <string>
#include <vector>

#include "constraint_accounting.h"
#include "constraint_io.h"
#include "constraint_observe.h"
#include "mu_solver.h"
#include "source_base/matrix.h"
#include "source_cell/unitcell.h"
#include "weight_grid.h"

namespace constraint
{

enum class LoopPhase
{
    IDLE,        // not initialized or disabled
    REFERENCE,   // first (mu = 0) SCF: recording the reference charges
    CONSTRAINED, // outer secant loop active
    DONE         // CONVERGED or UNREACHABLE: no further action
};

/**
 * @brief Outer-loop controller of the real-space weight constraint (M8).
 *
 * Two-stage gating (DeltaP 4.3 / deltaspin sc_scf_thr_mode lineage):
 *   phase REFERENCE: the first SCF runs unconstrained (mu = 0, injection is
 *     a no-op); at its convergence the reference charges Q_ref are recorded
 *     and the targets are built (delta mode: t = Q_ref + delta; absolute
 *     mode: t = file values).  The first M4.step runs immediately (the
 *     reference point IS the mu = 0 observation), and the SCF is forced to
 *     continue into the constrained phase.
 *   phase CONSTRAINED: at every SCF convergence the charges Q(mu) are read
 *     (M2), M4.step advances mu (secant + guards), the constraint potential
 *     is re-injected for the next iteration (M3a), and the audit line (M5)
 *     is printed.  CONVERGED ends the SCF; UNREACHABLE fuses the run and
 *     reports the Q(mu) endpoint.
 *
 * The loop shares one WeightGrid instance with the observer and the
 * injector: observable == injection operator by construction.
 */
class ConstraintLoop
{
  public:
    static ConstraintLoop& instance();

    // Build M1 weights and arm the loop. Call once per geometry (PW
    // before_scf).  Re-initialization replaces all state.
    // Legacy entry: derives a homogeneous spec list from the single-type cfg
    // (kept for pre-A4 callers / unit tests).  The specs-driven overload
    // below is the production entry (esolvers).
    void init(const UnitCell& ucell,
              const ModulePW::PW_Basis* rho_basis,
              const ConstraintConfig& cfg,
              const std::vector<double>& radii,
              const double nelec);

    // Stage-A entry: arm the loop from the fully validated per-constraint
    // list (M7).  The loop stores 'specs' verbatim: the weight-fragment map,
    // the per-constraint density channels (M2/M3a) and the per-constraint
    // mu_max caps (M4, A0 D3) all come from the specs, never from the
    // legacy single-type cfg mirror.  'cfg' still supplies the run-level
    // switches (enabled / weight_type / target_mode / thr).
    void init(const UnitCell& ucell,
              const ModulePW::PW_Basis* rho_basis,
              const ConstraintConfig& cfg,
              const std::vector<ConstraintSpec>& specs,
              const std::vector<double>& radii,
              const double nelec);

    // Inject the current mu-weighted potential into v_eff and veff_smooth
    // before the diagonalization of SCF iteration 'iter' (1-based).
    // No-op unless the loop is active.
    void inject_potential(const int iter,
                          ModuleBase::matrix& v_eff,
                          ModuleBase::matrix& veff_smooth);

    // LCAO variant: inject the current mu-weighted potential into the
    // dense-grid effective potential v_eff only.  The LCAO Hamiltonian is
    // built from v_eff by the Veff operator (cal_gint_vl), while
    // veff_smooth lives on the FFT grid in LCAO and is never read by the
    // Hamiltonian — the PW path passes both.
    void inject_potential_lcao(const int iter, ModuleBase::matrix& v_eff);

    // Add the currently injected constraint potential back into a potential
    // matrix (same sign/layout as the injector).  The vnew snapshot taken by
    // Potential::get_vnew() holds v_phys(out) - [v_phys(in) + mu*w]; the SCC
    // force integral must see the physical difference only, so the callers
    // add the mu*w term back before the core-correction integral (no-op when
    // the loop is disabled or mu is all zero).
    void add_back_constraint_potential(ModuleBase::matrix& veff) const;

    // Read the constraint charges Q from the (mixed) rho of iteration 'iter'.
    void observe(const int iter, const double* const* rho, const int nspin);

    // SCF-convergence bookkeeping.  May override conv_esolver to false so
    // the SCF continues with the updated mu; leaves it true when the outer
    // loop is done (CONVERGED / UNREACHABLE) or inactive.
    void on_scf_converged(const int iter, bool& conv_esolver);

    // Print the final audit report (called from PW after_scf).
    void final_report();

    // Constraint contribution to the atomic forces (M6): the shared weight
    // field times the current multipliers, integrated against the density
    // by the grid force kernel (constraint_deriv — the same kernel the LCAO
    // path calls, so PW and LCAO forces are identical by construction).
    // rho is the per-spin density on the weight grid (nspin = 1/2); every
    // constraint folds the density with its own channel (A5 per-constraint
    // kernel: charge uses rho_up + rho_dn, spin uses rho_up - rho_dn), so a
    // mixed charge+spin list is one call.  Accumulates into forcecon
    // (nat x 3, Ry/Bohr).  No-op when the loop is disabled.
    void compute_force(const double* const* rho,
                       const int nspin,
                       ModuleBase::matrix& forcecon);

    // Reset all state (unit tests / re-init).
    void reset();

    bool enabled() const { return cfg_.enabled && phase_ != LoopPhase::IDLE; }
    const std::string& type() const { return cfg_.type; }
    LoopPhase phase() const { return phase_; }
    MuStatus status() const { return status_; }
    int outer_steps() const { return outer_steps_; }
    double mu_norm() const;
    const std::vector<double>& mu() const { return mu_; }
    const std::vector<double>& targets() const { return targets_; }
    const std::vector<double>& charges() const { return Q_; }
    // The shared weight field (M3b runtime audit / force kernel inputs).
    // Valid while enabled(): the grid is built in init().
    const WeightGrid& weight_grid() const { return *wg_; }
    // True once the outer loop reached CONVERGED or UNREACHABLE (the final
    // audited state; the M3b runtime audit runs once at this point).
    bool done() const { return phase_ == LoopPhase::DONE; }
    const ConstraintAudit& last_audit() const { return audit_; }
    const std::string& last_audit_line() const { return last_audit_line_; }

  private:
    ConstraintLoop() = default;
    // Run one outer step from the current charges Q_ against targets_.
    void outer_step(const int iter, bool& conv_esolver);
    void print_audit(const int iter);

    ConstraintConfig cfg_;
    // Stage-A per-constraint list and the derived per-component fields
    // (parallel vectors, filled in init()).
    std::vector<ConstraintSpec> specs_;
    std::vector<ConstraintKind> kinds_;
    std::vector<ChannelProfile> channels_;
    std::vector<double> mu_caps_; // per-constraint |mu| fuse caps (A0 D3)
    std::unique_ptr<WeightGrid> wg_;
    MuSolver mu_solver_;
    std::vector<double> mu_;
    std::vector<double> Q_;
    std::vector<double> Q_ref_;
    std::vector<double> targets_;
    double nelec_ = 0.0;
    LoopPhase phase_ = LoopPhase::IDLE;
    MuStatus status_ = MuStatus::RUNNING;
    int outer_steps_ = 0;
    ConstraintAudit audit_;
    std::string last_audit_line_;
    // Experiment switch (default off): when ABA_CONSTRAINT_FIXED_MU is set,
    // the multiplier is frozen at the env value and the outer secant loop
    // is disabled (constraint potential acts as a fixed external potential).
    bool fixed_mu_ = false;
};

} // namespace constraint

#endif
