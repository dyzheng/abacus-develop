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
    DONE         // CONVERGED / UNREACHABLE / BRANCH_FLIP: no further action
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
 *     reports the Q(mu) endpoint; with constraint_branch_tol > 0 the energy
 *     branch guard additionally fuses a converged SCF whose constrained
 *     energy dropped below the reference (BRANCH_FLIP).
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
    // loop is done (CONVERGED / UNREACHABLE / BRANCH_FLIP) or inactive.
    void on_scf_converged(const int iter, bool& conv_esolver);

    // Dual-iteration schedule hook (plan 2026-09-11-dual-iteration-strategy.md),
    // called by the esolver once per SCF iteration, after observe() and before
    // on_scf_converged().  No-op unless constraint_mu_schedule=inner, the loop
    // is in the constrained phase and the density gate drho < inner_thr is
    // passed.  On a real mu update the observable is read (observe()), M4 runs
    // one step, the audit line is printed, and the function returns true: the
    // caller MUST reset the charge-mixing history (mix_reset) so the Broyden
    // cache does not survive the change of fixed-point map.  Returns false in
    // every other case (no update -> no reset).
    bool on_iteration(const int iter, const double drho);

    // Energy-branch-guard input (L10 lineage; consumed only when
    // constraint_branch_tol > 0).  The esolver feeds the plain (constraint-
    // correction-free) KS total energy [Ry] of the SCF iteration it is about
    // to report as converged; the loop adds its own sum_a mu_a (Q_a - t_a)
    // so both sides of the comparison use one energy convention.  Called
    // right before on_scf_converged() every iteration; only consumed at a
    // genuine SCF convergence.  If the guard is armed but no energy was ever
    // supplied the loop WARNING_QUITs rather than running unguarded.
    void set_scf_energy(const double etot_ks);

    // Optional per-atom on-site projected moments (indexed by global atom,
    // supplied by the esolver from the DFT+U occupation matrices).  When
    // non-empty the audit line adds `onsite=` per constraint: the fragment sum
    // of these moments, i.e. the on-site counterpart of the Becke-weighted
    // q.  Purely informational (never enters the loop logic); leave empty and
    // the audit line is unchanged (II-1b instrument).
    void set_onsite_moments(const std::vector<double>& per_atom);

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
    // Energy branch guard (L10 lineage): armed iff constraint_branch_tol > 0.
    // reference_energy() is the constrained energy of the mu = 0 reference
    // SCF (valid once recorded); guard_energy() is the constrained energy
    // evaluated at the last guard check.
    bool branch_guard_armed() const { return cfg_.branch_tol > 0.0; }
    bool reference_recorded() const { return e_ref_valid_; }
    double reference_energy() const { return e_ref_; }
    double guard_energy() const { return e_guard_; }
    // Per-atom on-site moments as supplied by the esolver (empty when none).
    const std::vector<double>& onsite_moments() const { return onsite_atom_; }
    // Dual-iteration schedule diagnostics (measurement objects of the plan):
    // how many mu updates went through the INNER path, how many settle checks
    // bounced, and whether the inner schedule is still in effect (it degrades
    // to OUTER when constraint_inner_nmax is exhausted or the settle check
    // fails twice threshold).  schedule() reports the configured value.
    const std::string& schedule() const { return cfg_.mu_schedule; }
    bool inner_active() const { return inner_active_; }
    int inner_steps() const { return inner_steps_; }
    int settle_failures() const { return settle_fail_; }
    double mu_norm() const;
    const std::vector<double>& mu() const { return mu_; }
    const std::vector<double>& targets() const { return targets_; }
    const std::vector<double>& charges() const { return Q_; }
    // The shared weight field (M3b runtime audit / force kernel inputs).
    // Valid while enabled(): the grid is built in init().
    const WeightGrid& weight_grid() const { return *wg_; }
    // True once the outer loop reached CONVERGED, UNREACHABLE or BRANCH_FLIP
    // (the final audited state; the M3b runtime audit runs once here).
    bool done() const { return phase_ == LoopPhase::DONE; }
    const ConstraintAudit& last_audit() const { return audit_; }
    const std::string& last_audit_line() const { return last_audit_line_; }

  private:
    ConstraintLoop() = default;
    // Run one outer step from the current charges Q_ against targets_.
    void outer_step(const int iter, bool& conv_esolver);
    // One INNER-schedule mu update from the current observation.  Returns true
    // when mu actually changed (the caller must reset the mixing history).
    bool take_inner_step(const int iter);
    // Log once and hand the remainder of the run back to the OUTER schedule
    // (used when the inner budget is exhausted or the settle check bounced
    // twice).  Never silently continues: the reason is printed.
    void degrade_to_outer(const char* reason);
    // Largest |Q_a - target_a| over the constraint list (settle-check gate).
    double max_residual() const;
    void print_audit(const int iter);
    // Shared audit emitter: 'label' is "outer" / "inner" / "settle".
    void print_audit_line(const int iter, const int step, const char* label);

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
    // Energy branch guard (L10 lineage) state.  scf_energy_ is the
    // esolver-supplied plain KS energy of the current SCF iteration; e_ref_
    // the reference (mu = 0) constrained energy; e_guard_ the constrained
    // energy evaluated at the last check.  All stay at their defaults while
    // the guard is off (the default configuration).
    double scf_energy_ = 0.0;
    bool scf_energy_set_ = false;
    double e_ref_ = 0.0;
    bool e_ref_valid_ = false;
    double e_guard_ = 0.0;
    // Per-atom on-site moments supplied by the esolver (empty = not
    // available; audit line then omits the onsite= token).
    std::vector<double> onsite_atom_;
    // Dual-iteration schedule state (plan 2026-09-11-dual-iteration-strategy.md).
    // inner_schedule_ mirrors cfg_.mu_schedule at init; inner_active_ is the
    // live flag (false = OUTER dynamics, set false permanently on degrade).
    // settle_armed_ is set when the inner M4 step reports CONVERGED: mu is then
    // frozen and the SCF settles, and on_scf_converged re-verifies the target
    // on the settled density (the "inner converged but the relaxation bounced"
    // trap of 2026-07-20).  last_reset_iter_ measures the mixing-reset cost.
    bool inner_schedule_ = false;
    bool inner_active_ = false;
    double inner_thr_ = 1e-3;
    int inner_nmax_ = 20;
    int inner_steps_ = 0;
    int settle_fail_ = 0;
    bool settle_armed_ = false;
    bool inner_step_this_iter_ = false;
    int last_reset_iter_ = 0;
    // Experiment switch (default off): when ABA_CONSTRAINT_FIXED_MU is set,
    // the multiplier is frozen at the env value and the outer secant loop
    // is disabled (constraint potential acts as a fixed external potential).
    bool fixed_mu_ = false;
};

} // namespace constraint

#endif
