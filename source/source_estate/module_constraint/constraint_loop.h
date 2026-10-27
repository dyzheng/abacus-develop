#ifndef CONSTRAINT_LOOP_H
#define CONSTRAINT_LOOP_H

#include <memory>
#include <string>
#include <vector>

#include "constraint_accounting.h"
#include "constraint_alm.h"
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
              const double nelec,
              const int istep = 0);

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
              const double nelec,
              const int istep = 0);

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

    // S3 allocation observation: n_alloc/m_alloc are per constraint and are
    // the full allocation readings Tr(DMR K_i).  Charge constraints read
    // n_alloc, spin constraints read m_alloc; the remaining reference/
    // target/ALM bookkeeping is shared with observe().
    void observe_allocation(const int iter,
                            const std::vector<double>& n_alloc,
                            const std::vector<double>& m_alloc);

    // S4.3 non-collinear allocation observation.  Spin components arrive as
    // one three-vector per constraint; charge components keep using the
    // scalar n_alloc entry.  Fixed-vector mode is the currently supported
    // production closure (adaptive vector scheduling is refused at config
    // time), so this updates the audit/accounting state without advancing a
    // scalar secant step.
    void observe_allocation_spin(
        const int iter,
        const std::vector<double>& n_alloc,
        const std::vector<ModuleBase::Vector3<double>>& m_alloc);

    // SCF-convergence bookkeeping.  May override conv_esolver to false so
    // the SCF continues with the updated mu; leaves it true when the outer
    // loop is done (CONVERGED / UNREACHABLE / BRANCH_FLIP) or inactive.
    void on_scf_converged(const int iter, bool& conv_esolver);

    // M3 C3: drho threshold for this SCF iteration.  During the outer probe
    // (CONSTRAINED, OUTER schedule, neither fixed-mu nor a warm-start probe)
    // it relaxes to max(1e-5, 100*strict); once the residual is within
    // 10*constraint_thr the next SCF is forced to the strict threshold for a
    // final, fully converged density.  Every other phase returns 'strict'.
    double scf_threshold(double strict_thr) const;

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

    // Constraint energy added to the KS total energy by the esolver.  In the
    // legacy/ALM paths this is the audited functional mu.(Q-t) (plus the ALM
    // penalty); in allocation mode the Hamiltonian already contributes
    // +sum mu.Q to eband, so only -sum mu.target is added here.  The audit
    // line always reports mu.(Q-t), independently of this output correction.
    double output_energy_correction() const;

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

    // Grid-based constraint stress (S4.2), the strain counterpart of
    // compute_force(): the same per-constraint density channels and the
    // WeightGrid strain derivative.  Allocation mode consumes the
    // complementary density Q_c D Q_c stored by the esolver; the target-space
    // S derivative is added separately by the LCAO overlap kernel.
    // Accumulates into stresscon (3 x 3, Ry/Bohr^3).  No-op when disabled.
    void compute_stress(const double* const* rho,
                        const int nspin,
                        ModuleBase::matrix& stresscon);

    // Allocation-mode force density: the esolver stores the complementary
    // Q_c D Q_c density here before FORCE_STRESS calls compute_force().
    void set_allocation_force_density(std::vector<std::vector<double>> rho);

    // Complete target-projector derivative contribution (dQ/dP and the
    // S^{-1/2} Fréchet term) for the PW allocation force.  The matrix is in
    // Ry/Bohr and is added after the frozen Q_c dW Q_c grid kernel.
    void set_allocation_target_force(const ModuleBase::matrix& force);

    // Non-collinear allocation force densities.  Each entry already carries
    // its constraint multiplier and lambda.sigma contraction, so the grid
    // kernel uses a unit coefficient per entry.
    void set_allocation_spin_force_density(
        std::vector<std::vector<double>> rho);

    // Non-collinear allocation W_i Pulay force.  complement_densities()
    // evaluates the basis-derivative half on the already lambda-folded
    // complementary density; it must join forcecon before FORCE_STRESS
    // prints TOTAL-FORCE and before a relaxation consumes the force.
    void set_allocation_spin_pulay_force(const ModuleBase::matrix& force);

    // Reset all state (unit tests / re-init).
    void reset();

    bool enabled() const { return cfg_.enabled && phase_ != LoopPhase::IDLE; }
    const std::string& type() const { return cfg_.type; }
    LoopPhase phase() const { return phase_; }
    MuStatus status() const { return status_; }
    bool fixed_mu_mode() const { return fixed_mu_; }
    bool allocation_mode() const { return allocation_mode_; }
    const std::vector<ChannelProfile>& channels() const { return channels_; }
    bool dipole_constraint() const { return dipole_constraint_; }
    int outer_steps() const { return outer_steps_; }
    // M4 scheduling pack B diagnostics: the SCF converged on the last frame
    // but mu was frozen because another update could not fit scf_nmax.
    bool budget_frozen() const { return budget_frozen_; }
    bool warm_started_last_init() const { return warm_started_last_init_; }
    // Test-only diagnostic: the solver's retained slope of component i.
    double solver_slope(const int i) const { return mu_solver_.slope(i); }
    // M3 C2 diagnostics: switch, current trust radius [Ry] and cumulative
    // rejected-trial count.  The radius is 0 while the trust path is off.
    bool trust_enabled() const { return cfg_.trust_region; }
    double trust_radius() const { return mu_solver_.trust_radius(); }
    int trust_rejections() const { return mu_solver_.trust_reject_count(); }
    // Consume the one-shot "the fixed-point map changed" request.  The
    // esolver calls this right after on_scf_converged() and resets the
    // charge-mixing history when true.  Set on a rejected trust trial and on
    // every ALM multiplier move (probe seed / Newton re-seed / outer update):
    // the Broyden/DIIS cache extrapolates across the old potential, and R2
    // showed a 0.25 drho transient without the reset.
    bool consume_trust_rollback()
    {
        const bool pending = trust_rollback_pending_;
        trust_rollback_pending_ = false;
        return pending;
    }
    // M3 E1 diagnostics: number of steps rejected because the total
    // magnetization jumped away from the reference branch.
    int mag_rejections() const { return mag_rejections_; }
    // Final status token (one line, no trailing newline).  final_report()
    // prints exactly this string; tests and logs can reuse it.
    std::string final_status_line() const;
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
    const std::vector<ModuleBase::Vector3<double>>& mu_vec() const
    {
        return mu_vec_;
    }
    const std::vector<double>& targets() const { return targets_; }
    const std::vector<ModuleBase::Vector3<double>>& target_vec() const
    {
        return target_vec_;
    }
    const std::vector<double>& charges() const { return Q_; }
    const std::vector<ConstraintKind>& kinds() const { return kinds_; }
    const std::vector<ModuleBase::Vector3<double>>& charge_vec() const
    {
        return Q_vec_;
    }
    // ---- M4 ALM schedule (module D) ---------------------------------------
    // mu_eff is the multiplier the injector/force actually apply:
    // mu_a + k_a(Q_a - t_a) when the ALM is armed, else plain mu (probe or
    // outer schedules).  mu_phys(i) is the constant-potential label of
    // component i (== mu_eff at convergence).  alm_armed() is true once k
    // was set (probe measurement or carried warm-start response).
    const std::vector<double>& mu_eff() const { return mu_eff_; }
    double mu_phys(const int i) const { return mu_eff_.at(i); }
    bool alm_armed() const { return alm_armed_; }
    double alm_k(const int i) const { return alm_.k().at(i); }
    // ---- Module A (M1) passive dipole report -----------------------------
    // Pure observation: armed by cfg.dipole_report, refreshed by observe(),
    // never read by the loop logic.  dipole_ref() is the REFERENCE-phase
    // snapshot (the q_dip origin, delta semantics; valid only after the
    // reference phase closes).  q_dip(valid) is the dipole-derived charge
    // transfer of the constrained fragment vs the rest [e]; valid=false
    // when the constraint list can not define a fragment pair (0 or >2
    // constraints) or the report is off.
    const ModuleBase::Vector3<double>& dipole() const { return dipole_latest_; }
    const ModuleBase::Vector3<double>& dipole_ref() const { return dipole_ref_; }
    bool dipole_ref_valid() const { return dipole_ref_valid_; }
    double delta_v() const { return delta_v_latest_; } // saw frame, Hartree
    double q_dip(bool& valid) const;
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
    void build_geometry(const UnitCell& ucell,
                        const ModulePW::PW_Basis* rho_basis,
                        const std::vector<double>& radii,
                        const std::vector<ConstraintSpec>& specs);
    bool warm_start_eligible(const ConstraintConfig& cfg,
                             const std::vector<ConstraintSpec>& specs,
                             const int istep) const;
    void warm_reinit(const UnitCell& ucell,
                     const ModulePW::PW_Basis* rho_basis,
                     const ConstraintConfig& cfg,
                     const std::vector<ConstraintSpec>& specs,
                     const std::vector<double>& radii,
                     const double nelec);
    void finish_warm_start();
    void finish_warm_start_alm();
    double constraint_energy_functional() const;
    void alm_arm(const std::vector<double>& chi);
    void alm_adjudicate(const int iter, bool& conv_esolver);
    // Run one outer step from the current charges Q_ against targets_.
    void outer_step(const int iter, bool& conv_esolver);
    // One INNER-schedule mu update from the current observation.  Returns true
    // when mu actually changed (the caller must reset the mixing history).
    bool take_inner_step(const int iter);
    // Shared ALM/effective-multiplier update after Q_ has been refreshed.
    void finish_observation();
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
    // S4.3: non-collinear spin constraints carry a three-vector multiplier
    // and observable.  Charge entries keep their vector mirror at zero.
    std::vector<ModuleBase::Vector3<double>> mu_vec_;
    // M4 ALM state.
    AlmController alm_;
    bool alm_mode_ = false;          // cfg_.mu_schedule == "alm"
    bool alm_armed_ = false;         // k set (probe / carried warm response)
    // Penalty feedback switch.  False right after the probe / a warm rebind:
    // the Newton (or carried) mu is run as a plain fixed-potential SCF first.
    // Engaging the penalty on the stale probe density overshoots (R1: the
    // first post-probe potential jumped to 5x the fixed point and cost the
    // frame its 100-iteration budget).  adjudicate turns it on only when the
    // settled density still misses the target.
    bool alm_penalty_live_ = false;
    bool alm_probe_pending_ = false; // cold probe SCF in flight
    // Warm-start secant state: the carried slope can be stale when the new
    // geometry's gap closes (R2 frame 2: carried |chi|=0.58, true 1.49).  The
    // first warm settle anchors (mu0, Q0); the next one measures the secant
    // slope on THIS geometry and takes one Newton step with it.
    bool alm_warm_secant_pending_ = false;
    std::vector<double> alm_warm_mu0_;
    std::vector<double> alm_warm_q0_;
    // Ry: chi-measurement probe multiplier.  0.02 (the original plan value)
    // costs a full-length probe SCF plus a settle whose combined length left
    // no room for the ALM correction inside scf_nmax=100 on the delta=0.01
    // R2 case; 0.005 is still far above SCF noise (dQ ~ 3e-3 e) and keeps
    // both the probe and the post-probe settle short.
    double alm_probe_mu_ = 0.005;
    std::vector<double> mu_eff_;     // injector / force multiplier
    std::vector<double> drho_trend_; // last drho values for the dampen guard
    std::vector<double> Q_;
    std::vector<ModuleBase::Vector3<double>> Q_vec_;
    std::vector<double> Q_ref_;
    std::vector<double> targets_;
    std::vector<ModuleBase::Vector3<double>> target_vec_;
    bool spin_vector_mode_ = false;
    double nelec_ = 0.0;
    LoopPhase phase_ = LoopPhase::IDLE;
    MuStatus status_ = MuStatus::RUNNING;
    int outer_steps_ = 0;
    bool warm_started_last_init_ = false;
    bool warm_pending_ = false;        // rebind (+ delta Q_ref) at next convergence
    bool geometry_state_valid_ = false;
    bool warm_guard_notice_done_ = false; // once-per-cold-geometry log
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
    // Set by outer_step()/take_inner_step() when the trust solver rejected a
    // trial; consumed once by the esolver to reset the mixing history.
    bool trust_rollback_pending_ = false;
    // M3 C3 progressive-SCF state: tighten_pending_ is set when the outer
    // residual is close enough to the target that the next SCF must use the
    // strict threshold; tighten_notice_done_ keeps the log one-shot.
    bool tighten_pending_ = false;
    bool tighten_notice_done_ = false;
    // M4 scheduling pack B: rebuild-cost estimate for the budget freeze.
    bool budget_frozen_ = false;
    int last_step_iter_ = 0;
    int last_rebuild_cost_ = 0;
    // M3 E1 magnetic branch anchor: TMAG recorded at the reference (or the
    // first warm-geometry convergence) and the number of rejected jumps.
    double mag_ref_ = 0.0;
    bool mag_ref_valid_ = false;
    int mag_rejections_ = 0;
    // Module A reporter state (see the public accessors above).
    void update_dipole(const double* const* rho, const int nspin);
    void print_dipole_report() const;
    bool dipole_report_ = false;
    std::string dipole_frame_ = "molecular";
    double dipole_saw_pos_max_ = -1.0; // -1 = auto (lowest-density plane)
    double dipole_saw_pos_dec_ = 0.1;
    bool saw_resolved_ = false;        // auto frame resolved this geometry
    double saw_pos_max_live_ = 0.5;    // resolved pos_max actually in use
    const UnitCell* ucell_ = nullptr;  // per-geometry, set in init()
    ModuleBase::Vector3<double> dipole_origin_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> dipole_latest_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> dipole_ref_{0.0, 0.0, 0.0};
    bool dipole_ref_valid_ = false;
    double delta_v_latest_ = 0.0;
    ModuleBase::Vector3<double> frag_a_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> frag_b_{0.0, 0.0, 0.0};
    bool frag_valid_ = false;
    // M5 dipole-coordinate state.  The WeightGrid itself carries the frame;
    // these flags only govern lazy auto plane resolution and the log/report.
    bool dipole_constraint_ = false;
    bool dipole_constraint_auto_saw_ = false;
    bool dipole_constraint_saw_resolved_ = true;
    double dipole_constraint_saw_live_ = 0.5;
    // Experiment switch (default off): when ABA_CONSTRAINT_FIXED_MU is set,
    // the multiplier is frozen at the env value and the outer secant loop
    // is disabled (constraint potential acts as a fixed external potential).
    bool fixed_mu_ = false;
    bool json_fixed_source_ = false; // fixed_mu came from the v2 target file
    bool allocation_mode_ = false;
    std::vector<std::vector<double>> allocation_force_rho_;
    ModuleBase::matrix allocation_target_force_;
    std::vector<std::vector<double>> allocation_spin_force_rho_;
    ModuleBase::matrix allocation_spin_pulay_force_;
};

} // namespace constraint

#endif
