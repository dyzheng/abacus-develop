#include "constraint_loop.h"

#include <cmath>
#include <cstdlib>
#include <numeric>

#include "constraint_deriv.h"
#include "constraint_inject_pw.h"
#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"

namespace constraint
{

namespace
{
// Diagnostic experiment switch (default off; the production behaviour always
// resets).  ABA_CONSTRAINT_INNER_NO_RESET suppresses the charge-mixing reset
// that normally follows every INNER mu update, so the mixing-damage question
// (is the Broyden/DIIS cache corruption the real cost of an in-SCF mu change?)
// can be answered against a no-reset control (plan 2026-09-11 §3.3-3).  The mu
// update itself is untouched: only the history reset is skipped, and the run
// logs MIX_RESET SUPPRESSED so a control run is never mistaken for a normal one.
bool inner_no_reset_experiment()
{
    static const bool enabled
        = (std::getenv("ABA_CONSTRAINT_INNER_NO_RESET") != nullptr);
    return enabled;
}
} // anonymous namespace

ConstraintLoop& ConstraintLoop::instance()
{
    static ConstraintLoop loop;
    return loop;
}

void ConstraintLoop::reset()
{
    cfg_ = ConstraintConfig();
    specs_.clear();
    kinds_.clear();
    channels_.clear();
    mu_caps_.clear();
    wg_.reset();
    mu_solver_.reset();
    mu_.clear();
    Q_.clear();
    Q_ref_.clear();
    targets_.clear();
    nelec_ = 0.0;
    fixed_mu_ = false;
    phase_ = LoopPhase::IDLE;
    status_ = MuStatus::RUNNING;
    outer_steps_ = 0;
    audit_ = ConstraintAudit();
    last_audit_line_.clear();
    scf_energy_ = 0.0;
    scf_energy_set_ = false;
    e_ref_ = 0.0;
    e_ref_valid_ = false;
    e_guard_ = 0.0;
    onsite_atom_.clear();
    inner_schedule_ = false;
    inner_active_ = false;
    inner_thr_ = 1e-3;
    inner_nmax_ = 20;
    inner_steps_ = 0;
    settle_fail_ = 0;
    settle_armed_ = false;
    inner_step_this_iter_ = false;
    last_reset_iter_ = 0;
}

void ConstraintLoop::init(const UnitCell& ucell,
                          const ModulePW::PW_Basis* rho_basis,
                          const ConstraintConfig& cfg,
                          const std::vector<double>& radii,
                          const double nelec)
{
    // Legacy entry (pre-A4 callers / unit tests): derive a homogeneous
    // per-constraint list from the single-type cfg so every consumer below
    // reads only the specs — there is no second loop path to drift.
    std::vector<ConstraintSpec> specs;
    const ConstraintKind kind = (cfg.type == "spin")
                                    ? ConstraintKind::Spin
                                    : ConstraintKind::Charge;
    specs.reserve(cfg.targets.size());
    for (const ConstraintTarget& t : cfg.targets)
    {
        ConstraintSpec spec;
        spec.kind = kind;
        spec.atoms = t.atoms;
        spec.chan = build_channel_profile(kind);
        spec.target = t.value;
        spec.mu_max = cfg.mu_max;
        specs.push_back(spec);
    }
    init(ucell, rho_basis, cfg, specs, radii, nelec);
}

void ConstraintLoop::init(const UnitCell& ucell,
                          const ModulePW::PW_Basis* rho_basis,
                          const ConstraintConfig& cfg,
                          const std::vector<ConstraintSpec>& specs,
                          const std::vector<double>& radii,
                          const double nelec)
{
    reset();
    cfg_ = cfg;
    specs_ = specs;
    nelec_ = nelec;
    // Dual-iteration schedule (plan 2026-09-11-dual-iteration-strategy.md):
    // the config switch is mirrored into live state.  inner_active_ is what
    // on_iteration() gates on, so a degrade can switch the run back to OUTER
    // dynamics without touching the configuration.
    inner_schedule_ = (cfg_.mu_schedule == "inner");
    inner_active_ = false; // armed below only when the loop really runs
    inner_thr_ = cfg_.inner_thr;
    inner_nmax_ = cfg_.inner_nmax;
    if (!cfg_.enabled)
    {
        return;
    }
    // The loop really runs: arm the INNER schedule (if selected).  A degrade
    // below clears inner_active_ without changing inner_schedule_.
    inner_active_ = inner_schedule_;
    // Derive the per-constraint fields (M7 -> M2/M3/M4/M5): observable kind,
    // density channel and fuse cap per spec.  All consumers below iterate
    // these parallel vectors, so a mixed charge+spin list needs no special
    // casing in the observers / injectors / accounting.
    kinds_.reserve(specs_.size());
    channels_.reserve(specs_.size());
    mu_caps_.reserve(specs_.size());
    for (const ConstraintSpec& spec : specs_)
    {
        kinds_.push_back(spec.kind);
        channels_.push_back(spec.chan);
        mu_caps_.push_back(spec.mu_max);
    }
    // Build the shared weight field (M1) once per geometry.  The fragment
    // map comes from the per-constraint fragments of the spec list.
    wg_.reset(new WeightGrid(ucell, rho_basis, radii, WeightType::Becke));
    std::vector<std::vector<int>> atom_map;
    for (const ConstraintSpec& spec : specs_)
    {
        atom_map.push_back(spec.atoms);
    }
    wg_->set_constraint_atoms(atom_map);
    wg_->build();

    // mu starts at zero: the first SCF is the unconstrained reference.
    mu_.assign(specs_.size(), 0.0);
    // Experiment branch (default off): freeze the multiplier at a constant
    // read from the environment so the constraint potential acts as an
    // ordinary fixed external potential (no outer secant loop, no reference
    // SCF).  This is the attribution harness for the force FD round: with
    // mu constant, the printed SCF energy slope and the printed force must
    // agree if the energy and force derivations are mutually consistent.
    const char* fixed_mu_env = std::getenv("ABA_CONSTRAINT_FIXED_MU");
    if (fixed_mu_env != nullptr && std::strlen(fixed_mu_env) > 0)
    {
        // Compatibility guard: the energy branch guard compares every
        // constrained energy against the mu = 0 reference SCF energy, which
        // the fixed-mu experiment never runs.  Refuse the combination instead
        // of silently running a partially armed (trigger-free) guard.
        if (cfg_.branch_tol > 0.0)
        {
            ModuleBase::WARNING_QUIT("ConstraintLoop::init",
                "constraint_branch_tol > 0 is incompatible with the "
                "ABA_CONSTRAINT_FIXED_MU experiment (no reference SCF energy "
                "is available); unset the env variable or set "
                "constraint_branch_tol = 0");
        }
        const double mu_fixed = std::atof(fixed_mu_env);
        std::fill(mu_.begin(), mu_.end(), mu_fixed);
        fixed_mu_ = true;
        // Fixed-mu experiment: the multiplier never changes, so there is no
        // inner update to schedule; refuse to pretend otherwise.
        inner_active_ = false;
        targets_.resize(specs_.size());
        for (size_t a = 0; a < targets_.size(); ++a)
        {
            targets_[a] = specs_[a].target;
        }
        phase_ = LoopPhase::CONSTRAINED;
        status_ = MuStatus::RUNNING;
        return;
    }
    MuSolverParams params;
    // Per-constraint fuse caps (A0 D3): every spec carries its own mu_max;
    // the scalar field stays as a fallback only (legacy tests construct
    // homogeneous lists through the cfg entry above).
    params.mu_max_per_component = mu_caps_;
    params.conv_tol = cfg_.thr;
    // Outer-step caps (II-1): step_max bounds every step; step_probe bounds
    // only the first, history-free step, where the solver has no measured
    // slope and would otherwise always sit at the cap.  Both default to the
    // pre-II-1 behaviour when the INPUT keys are left alone.
    params.step_max = cfg_.step_max;
    params.step_probe = cfg_.step_probe;
    // Response sign: both channels respond negatively.  For the spin
    // channel the split injection (V_up += mu*w, V_dn -= mu*w) repels
    // spin-up from / attracts spin-down to the fragment for mu > 0, so the
    // magnetization reading m = rho_up - rho_dn decreases with mu — the
    // same negative density response as charge, verified empirically in the
    // 212_PW_constraint_h2o_spin integration case (dQ/dmu ~ -1.4 e/Ry).
    params.response_sign = -1;
    mu_solver_ = MuSolver(params);
    phase_ = LoopPhase::REFERENCE;
    status_ = MuStatus::RUNNING;
    outer_steps_ = 0;
}

void ConstraintLoop::inject_potential(const int iter,
                                      ModuleBase::matrix& v_eff,
                                      ModuleBase::matrix& veff_smooth)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: outer loop finished (CONVERGED / UNREACHABLE / BRANCH_FLIP):
    // the potential stays as the last injected one — nothing more to add.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Branch B: active loop.  In the reference phase mu is all zero, so the
    // injection is a no-op by value and the first SCF stays unconstrained.
    // Every alpha injects through its own channel profile (stage A: a mixed
    // charge+spin list reaches both potentials in one call); the injector
    // rejects a mu/weight/profile length mismatch — per its contract the
    // caller must WARNING_QUIT rather than silently run without the
    // constraint potential (review P2).
    if (!ConstraintInjectPW::inject(*wg_, mu_, channels_, v_eff)
        || !ConstraintInjectPW::inject(*wg_, mu_, channels_, veff_smooth))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential",
            "mu length does not match the constraint count (wiring bug)");
    }
    (void)iter;
}

void ConstraintLoop::inject_potential_lcao(const int iter,
                                           ModuleBase::matrix& v_eff)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: outer loop finished (CONVERGED / UNREACHABLE / BRANCH_FLIP):
    // the potential stays as the last injected one — nothing more to add.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Branch B: active loop.  In the reference phase mu is all zero, so the
    // injection is a no-op by value and the first SCF stays unconstrained.
    // Every alpha injects through its own channel profile (stage A: a mixed
    // charge+spin list reaches the dense-grid potential in one call); the
    // injector rejects a mu/weight/profile length mismatch — per its
    // contract the caller must WARNING_QUIT rather than silently run without
    // the constraint potential (review P2).
    if (!ConstraintInjectPW::inject(*wg_, mu_, channels_, v_eff))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential_lcao",
            "mu length does not match the constraint count (wiring bug)");
    }
    (void)iter;
}

void ConstraintLoop::add_back_constraint_potential(ModuleBase::matrix& veff) const
{
    if (!enabled() || !wg_ || mu_.empty())
    {
        return; // constraint off or not armed: nothing was injected
    }
    // Mirror of the injector (ConstraintInjectPW::inject): the same
    // per-channel sum mu_alpha * w_alpha that entered the vnew snapshot is
    // added back, restoring the physical potential difference for the SCC
    // force integral (see header comment).  Per-constraint channels keep the
    // mixed charge+spin potential in sync with the forward injection.
    if (!ConstraintInjectPW::inject(*wg_, mu_, channels_, veff))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::add_back_constraint_potential",
            "mu/veff length mismatch (wiring bug)");
    }
}

void ConstraintLoop::observe(const int iter, const double* const* rho,
                             const int nspin)
{
    if (!enabled())
    {
        return;
    }
    // Stage-A reading: every alpha reads with its own channel signs, so a
    // mixed charge+spin list is observed in one integral (A2 core).
    ConstraintObserver::observe(*wg_, rho, nspin, channels_, Q_);
    (void)iter;
}

void ConstraintLoop::outer_step(const int iter, bool& conv_esolver)
{
    // Experiment branch (fixed mu): the multiplier never changes, so the
    // first genuine SCF convergence closes the run.  No secant step runs;
    // the audit reports the fixed-mu residual Q(mu) - t for the record.
    if (fixed_mu_)
    {
        ++outer_steps_;
        print_audit(iter);
        status_ = MuStatus::CONVERGED;
        phase_ = LoopPhase::DONE;
        return;
    }
    if (phase_ == LoopPhase::REFERENCE)
    {
        // Reference convergence: record Q_ref and build the targets.
        Q_ref_ = Q_;
        targets_.resize(Q_.size());
        for (size_t a = 0; a < Q_.size(); ++a)
        {
            // Branch A: absolute mode — target is the file value directly
            // (calibration scale ~0.2-0.3 e, warned at configure time).
            if (cfg_.target_mode == "absolute")
            {
                targets_[a] = specs_[a].target;
            }
            else
            {
                // Branch B: delta mode — target is the reference charge plus
                // the requested shift (calibration scale ~e).
                targets_[a] = Q_ref_[a] + specs_[a].target;
            }
        }
        // The reference point is the mu = 0 observation: run the first
        // secant step immediately.  A delta = 0 target converges here and
        // the run ends; otherwise the SCF continues into the constrained
        // phase with the updated mu.
        phase_ = LoopPhase::CONSTRAINED;
        ++outer_steps_;
        status_ = mu_solver_.step(Q_, targets_, mu_);
        print_audit(iter);
        if (status_ == MuStatus::CONVERGED || status_ == MuStatus::UNREACHABLE)
        {
            phase_ = LoopPhase::DONE;
            return;
        }
        // Keep the SCF running so the new mu is felt by the density.
        conv_esolver = false;
        return;
    }

    // Branch B: constrained phase — advance mu from the observed Q(mu).
    ++outer_steps_;
    status_ = mu_solver_.step(Q_, targets_, mu_);
    print_audit(iter);
    if (status_ == MuStatus::CONVERGED || status_ == MuStatus::UNREACHABLE)
    {
        phase_ = LoopPhase::DONE;
        return;
    }
    conv_esolver = false;
}

void ConstraintLoop::on_scf_converged(const int iter, bool& conv_esolver)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: already finished — nothing to do.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Consume the per-iteration marker set by take_inner_step(): an INNER mu
    // update happened this SCF iteration.  Cleared here (every iteration) so
    // the OUTER path can never see a stale flag.
    const bool inner_step = inner_step_this_iter_;
    inner_step_this_iter_ = false;
    // Branch S: INNER settle check (plan 2026-09-11-dual-iteration-strategy.md
    // section 2.4).  The inner solver announced CONVERGED (this or an earlier
    // iteration) and mu has been frozen since; the density is now settled
    // (conv_esolver).  Re-verify the target on the settled density: the
    // historical trap is an inner optimisation that holds on the frozen
    // density but bounces once the density relaxes (2026-07-20 lesson).
    if (inner_active_ && phase_ == LoopPhase::CONSTRAINED && settle_armed_
        && conv_esolver)
    {
        const double res = max_residual();
        if (res < cfg_.thr)
        {
            // Branch S1: the target survived the relaxation -> genuine
            // CONVERGED.  Re-emit the audit on the settled density so the
            // reported e_con / Q are the settled values, not the frozen ones.
            status_ = MuStatus::CONVERGED;
            // Emit the audit BEFORE the phase flips to DONE: the audit line
            // carries the phase token, and the settle line describes a
            // constrained-phase decision (mirrors outer_step, which also
            // prints while phase_ is still CONSTRAINED).
            print_audit_line(iter, inner_steps_, "settle");
            phase_ = LoopPhase::DONE;
            GlobalV::ofs_running
                << "[constraint] settle check PASSED: residual " << res
                << " < " << cfg_.thr
                << " on the settled density (INNER schedule CONVERGED)\n";
            return; // conv_esolver stays true: the run is done
        }
        // Branch S2: the inner convergence claim did not survive the density
        // relaxation -> withdraw it (status back to RUNNING: the loop must
        // not report CONVERGED while the run continues) and resume the inner
        // loop.
        settle_armed_ = false;
        status_ = MuStatus::RUNNING;
        ++settle_fail_;
        GlobalV::ofs_running
            << "[constraint] settle check FAILED (x" << settle_fail_
            << "): residual " << res << " >= " << cfg_.thr
            << " on the settled density (the inner target did not survive "
               "the relaxation)\n";
        if (settle_fail_ >= 2)
        {
            // Branch S2a: two bounces -> stop pretending INNER works here;
            // fall through to the OUTER path (the SCF is settled, so an outer
            // step on this density is valid).
            degrade_to_outer("settle check failed twice");
        }
        else
        {
            // Branch S2b: first bounce -> keep the SCF running so the inner
            // loop can correct mu again.
            conv_esolver = false;
            return;
        }
    }
    // Branch B: SCF not converged yet — keep iterating; the observable is
    // only read at genuine SCF convergence (two-stage gating).  Without this
    // gate the outer step would run on the unconverged density and the
    // secant would chase the mixing noise instead of Q(mu).
    if (!conv_esolver)
    {
        return;
    }
    // Branch I: an INNER mu update was made this iteration but the target is
    // not reached yet — keep the SCF running so the density feels the new mu
    // (the mixing history was reset by the caller in the same iteration).
    if (inner_step)
    {
        conv_esolver = false;
        return;
    }
    // Online energy branch guard (L10 lineage, opt-in via
    // constraint_branch_tol > 0; default off = this whole block is skipped).
    // A converged SCF whose constrained energy lies BELOW the mu = 0
    // reference by more than the tolerance is not a constrained solution:
    // the SCF switched to another self-consistent branch (or magnetic state)
    // that happens to satisfy the weighted observable.  The targets are then
    // meaningless, so the run is fused here (BRANCH_FLIP) instead of being
    // reported as CONVERGED (II-1a/II-1b: a flipped Q was once recorded as
    // CONVERGED by the residual test alone).
    if (cfg_.branch_tol > 0.0)
    {
        // Wiring guard: the guard is only meaningful with an energy input;
        // fail loud rather than silently accepting an unguarded run.
        if (!scf_energy_set_)
        {
            ModuleBase::WARNING_QUIT("ConstraintLoop::on_scf_converged",
                "constraint_branch_tol > 0 but no SCF total energy was "
                "supplied (set_scf_energy); the esolver wiring is incomplete");
        }
        // Branch G1: reference phase — mu is identical zero, so the
        // constraint energy correction is exactly zero and the plain KS
        // energy IS the constrained reference energy.  Record it; nothing
        // can have flipped yet.
        if (phase_ == LoopPhase::REFERENCE)
        {
            e_ref_ = scf_energy_;
            e_ref_valid_ = true;
            GlobalV::ofs_running
                << "[constraint] branch guard armed: constraint_branch_tol="
                << cfg_.branch_tol << " Ry, e_ref=" << e_ref_
                << " Ry (mu = 0 reference energy)\n";
        }
        else
        {
            // Branch G2: constrained phase — compare the constrained energy
            // evaluated at the multiplier the SCF actually felt.  mu_ is
            // still the in-SCF value here (outer_step advances it below),
            // and Q_ was read from the just-converged density, so
            //   E_tot = E_KS + sum_a mu_a (Q_a - t_a)
            // is the energy the SCF minimized; e_ref_ uses the same
            // convention (its correction term is zero).
            if (Q_.size() != targets_.size())
            {
                // Defensive (wiring bug): indexing one of the two parallel
                // vectors with the other's size would read out of bounds.
                ModuleBase::WARNING_QUIT("ConstraintLoop::on_scf_converged",
                    "branch guard: observed-charge/target size mismatch");
            }
            double e_con = 0.0;
            for (size_t a = 0; a < mu_.size(); ++a)
            {
                e_con += mu_[a] * (Q_[a] - targets_[a]);
            }
            e_guard_ = scf_energy_ + e_con;
            const double de = e_guard_ - e_ref_;
            GlobalV::ofs_running
                << "[constraint] branch guard: e_tot=" << e_guard_
                << " Ry, de=e_tot-e_ref=" << de
                << " Ry (tol=" << cfg_.branch_tol << " Ry)\n";
            if (de < -cfg_.branch_tol)
            {
                status_ = MuStatus::BRANCH_FLIP;
                phase_ = LoopPhase::DONE;
                GlobalV::ofs_running
                    << "[constraint] BRANCH_FLIP: constrained energy is "
                    << -de << " Ry (" << -de * ModuleBase::Ry_to_eV
                    << " eV) BELOW the reference, beyond "
                       "constraint_branch_tol="
                    << cfg_.branch_tol << " Ry -> the SCF left the reference "
                       "electronic branch; fusing and reporting BRANCH_FLIP "
                       "(the constraint targets are NOT validated)\n";
                // Leave conv_esolver true: the SCF itself did converge — the
                // fuse is the outer-loop verdict, exactly like UNREACHABLE.
                return;
            }
        }
    }
    // Branch C: converged SCF with an active outer loop — outer step.
    outer_step(iter, conv_esolver);
}

void ConstraintLoop::set_scf_energy(const double etot_ks)
{
    // Guard input: stored unconditionally (cheap) and consumed only by the
    // branch guard above, so the default-off configuration is a pure no-op.
    scf_energy_ = etot_ks;
    scf_energy_set_ = true;
}

void ConstraintLoop::set_onsite_moments(const std::vector<double>& per_atom)
{
    // Informational instrument (II-1b): stored verbatim and folded into the
    // audit line only; never read by the injection / solver / force paths, so
    // a missing or stale value can not change the physics.
    onsite_atom_ = per_atom;
}

double ConstraintLoop::max_residual() const
{
    // Settle-check gate: the same residual the M4 convergence test uses
    // (|Q_a - t_a|), evaluated on whatever density Q_ currently describes.
    double worst = 0.0;
    const size_t n = std::min(Q_.size(), targets_.size());
    for (size_t a = 0; a < n; ++a)
    {
        worst = std::max(worst, std::abs(Q_[a] - targets_[a]));
    }
    return worst;
}

void ConstraintLoop::degrade_to_outer(const char* reason)
{
    if (!inner_active_)
    {
        return; // already degraded (or never inner): log exactly once
    }
    inner_active_ = false;
    settle_armed_ = false;
    // The inner convergence claim is withdrawn; the outer secant loop resumes
    // from the current mu/Q state and must be free to continue.
    if (status_ == MuStatus::CONVERGED)
    {
        status_ = MuStatus::RUNNING;
    }
    GlobalV::ofs_running
        << "[constraint] INNER schedule degraded to OUTER (" << reason
        << "): the in-SCF mu updates stop, the outer secant loop takes over "
           "at SCF convergence\n";
}

bool ConstraintLoop::take_inner_step(const int iter)
{
    // Budget guard (constraint_inner_nmax): the inner loop must not churn
    // forever.  When the budget is spent, hand the run back to the OUTER
    // schedule and report it loudly -- never spin, never silently continue.
    if (inner_steps_ >= inner_nmax_)
    {
        degrade_to_outer("constraint_inner_nmax exhausted");
        return false;
    }
    ++inner_steps_;
    // A real update happened this SCF iteration: on_scf_converged() must keep
    // the SCF running so the new mu is felt by the density.
    inner_step_this_iter_ = true;
    // M4 secant step on the freshly observed Q at the current mu.  The same
    // solver, caps and guards as the outer path (this is a schedule change,
    // not a solver change).
    status_ = mu_solver_.step(Q_, targets_, mu_);
    print_audit_line(iter, inner_steps_, "inner");
    // Branch A: the inner solver reached the target.  mu is frozen from now
    // on; convergence is NOT declared here -- the density must settle first
    // and the settle check re-verifies the target (historical trap: an inner
    // optimisation against the frozen density can bounce once the density
    // relaxes).
    if (status_ == MuStatus::CONVERGED)
    {
        settle_armed_ = true;
        return false;
    }
    // Branch B: a component fused at its mu cap.  Fuse the run exactly like
    // the outer path (no mixing reset: the run is over).
    if (status_ == MuStatus::UNREACHABLE)
    {
        phase_ = LoopPhase::DONE;
        return false;
    }
    // Branch C: mu moved -> the fixed-point map changed, so the caller must
    // reset the mixing history before the next SCF iteration.
    return true;
}

bool ConstraintLoop::on_iteration(const int iter, const double drho)
{
    // Guard: without an enabled loop there is no schedule at all.
    if (!enabled())
    {
        return false;
    }
    // Branch A: the loop already finished (CONVERGED / UNREACHABLE /
    // BRANCH_FLIP) -- no further updates.
    if (phase_ == LoopPhase::DONE)
    {
        return false;
    }
    // Branch B: OUTER schedule (default) or a degraded INNER run -- the
    // per-iteration hook is inert, exactly as before this feature.
    if (!inner_active_)
    {
        return false;
    }
    // Branch C: the reference SCF must stay the clean mu = 0 observation;
    // inner updates start only after it has recorded Q_ref.
    if (phase_ != LoopPhase::CONSTRAINED || fixed_mu_)
    {
        return false;
    }
    // Branch D: the settle phase -- mu is frozen and the density is allowed
    // to relax; the settle check in on_scf_converged() adjudicates.
    if (settle_armed_)
    {
        return false;
    }
    // Density gate (two-stage gating, deltaspin lineage): only update mu when
    // drho is small enough that the observed Q is a response to mu rather
    // than mixing noise.
    if (!(drho < inner_thr_))
    {
        return false;
    }
    // Reset-cost accounting: the first gated iteration after a reset closes
    // the previous update cycle (iterations spent re-converging the density).
    if (last_reset_iter_ > 0)
    {
        GlobalV::ofs_running
            << "[constraint] mixing recovered after "
            << (iter - last_reset_iter_) << " SCF iteration(s) (reset cost)\n";
        last_reset_iter_ = 0;
    }
    const bool reset = take_inner_step(iter);
    // Branch R1: mu did not change (CONVERGED / UNREACHABLE / budget
    // exhausted) -> the fixed-point map is untouched and no reset is needed.
    if (!reset)
    {
        return false;
    }
    // Branch R2 (diagnostic, default off): the no-reset control of the
    // mixing-damage experiment (plan 2026-09-11 §3.3-3).  The mu update stands;
    // only the history reset is skipped, and the suppression is logged.
    if (inner_no_reset_experiment())
    {
        GlobalV::ofs_running
            << "[constraint] MIX_RESET SUPPRESSED at SCF iteration " << iter
            << " (ABA_CONSTRAINT_INNER_NO_RESET experiment)\n";
        return false;
    }
    // Branch R3 (production): every mu update that changes the fixed-point map
    // is paired with exactly one reset.  The iteration of the reset is kept so
    // the next gated iteration can report the reset cost (mixing damage
    // measurement, plan §3.2).
    last_reset_iter_ = iter;
    GlobalV::ofs_running << "[constraint] MIX_RESET at SCF iteration " << iter
                         << " (mu update)\n";
    return true;
}

void ConstraintLoop::print_audit(const int iter)
{
    // OUTER-schedule audit: the historical label ("outer step N") is kept
    // byte-identical so legacy runs and their parsers are unaffected.
    print_audit_line(iter, outer_steps_, "outer");
}

void ConstraintLoop::print_audit_line(const int iter, const int step,
                                      const char* label)
{
    // Stage-A audit: per-constraint kinds label every detail line (M5:
    // c[i] kind=charge / kind=spin), so a mixed run stays machine
    // readable per channel.
    // On-site moments (when supplied) ride along in the same audit record:
    // per-constraint fragment sum, the on-site counterpart of the
    // Becke-weighted q (II-1b instrument).
    audit_ = ConstraintAccounting::audit(*wg_, mu_, Q_, targets_, nelec_,
                                         kinds_, onsite_atom_);
    last_audit_line_ = ConstraintAccounting::audit_line(audit_);
    GlobalV::ofs_running << "\n[constraint] " << label << " step " << step
                         << " after SCF iteration " << iter << " (phase="
                         << (phase_ == LoopPhase::CONSTRAINED ? "constrained"
                                                              : "reference")
                         << ")\n"
                         << last_audit_line_ << "\n";
}

void ConstraintLoop::final_report()
{
    if (!enabled() || last_audit_line_.empty())
    {
        return;
    }
    GlobalV::ofs_running << "\n[constraint] final status: ";
    if (fixed_mu_)
    {
        GlobalV::ofs_running
            << "FIXED_MU (experiment switch; multiplier frozen, targets not enforced)\n";
    }
    else if (status_ == MuStatus::CONVERGED)
    {
        GlobalV::ofs_running << "CONVERGED (targets reached within "
                             << cfg_.thr << " e)\n";
    }
    else if (status_ == MuStatus::UNREACHABLE)
    {
        // Report the cap of the component that actually fused: with
        // per-constraint caps (A0 D3) the scalar cfg.mu_max mirror may not
        // be the triggering limit.
        double fused_cap = cfg_.mu_max;
        const int fc = mu_solver_.fuse_component();
        if (fc >= 0 && static_cast<size_t>(fc) < mu_caps_.size())
        {
            fused_cap = mu_caps_[fc];
        }
        GlobalV::ofs_running << "UNREACHABLE (fused at mu cap "
                             << fused_cap << " Ry)\n";
    }
    else if (status_ == MuStatus::BRANCH_FLIP)
    {
        // Online energy branch guard verdict: report the two energies in Ry
        // and eV so the fuse is auditable without re-parsing the run.
        const double de = e_guard_ - e_ref_;
        GlobalV::ofs_running
            << "BRANCH_FLIP (fused: constrained energy " << e_guard_
            << " Ry is " << -de << " Ry (" << -de * ModuleBase::Ry_to_eV
            << " eV) below the reference " << e_ref_
            << " Ry at constraint_branch_tol=" << cfg_.branch_tol
            << " Ry — the SCF left the reference electronic branch; the "
               "targets are NOT validated)\n";
    }
    else
    {
        GlobalV::ofs_running << "RUNNING (SCF ended before the outer loop "
                                "converged — raise scf_nmax)\n";
    }
    GlobalV::ofs_running << last_audit_line_ << "\n";
}

double ConstraintLoop::mu_norm() const
{
    return std::accumulate(mu_.begin(), mu_.end(), 0.0,
                           [](const double s, const double v) {
                               return s + std::abs(v);
                           });
}

void ConstraintLoop::compute_force(const double* const* rho,
                                   const int nspin,
                                   ModuleBase::matrix& forcecon)
{
    if (!enabled() || !wg_)
    {
        return; // constraint off: no constraint force
    }
    // Build the position-derivative grid lazily (M6 kernel input); the
    // loop is one-per-geometry, so this happens once per force evaluation.
    if (!wg_->derivatives_built())
    {
        wg_->build_derivatives();
    }
    // Stationary-point guard (envelope-theorem premise, plan section 5.1):
    // the force is the exact derivative of the constrained energy only at
    // the converged outer loop; with a transient mu the residual is
    // O(|Q - t|) and the force must not be silently reported as exact.
    // Branch A: the branch guard fused the run — the residual may be small
    // (Q ~ t) yet the underlying state is not the constrained reference-branch
    // solution, so the envelope-theorem premise fails for a different reason.
    if (status_ == MuStatus::BRANCH_FLIP)
    {
        ModuleBase::WARNING("ConstraintLoop::compute_force",
                            "constraint force from a BRANCH_FLIP run: the SCF "
                            "left the reference branch, the force is not the "
                            "derivative of the intended constrained energy");
    }
    // Branch B: transient multiplier (outer loop not converged) — residual
    // O(|Q - t|).
    else if (mu_norm() > 0.0 && status_ != MuStatus::CONVERGED)
    {
        ModuleBase::WARNING("ConstraintLoop::compute_force",
                            "constraint force from an unconverged outer loop: "
                            "residual O(|Q - t|)");
    }
    // Stage-A per-constraint call (A5): the force kernel folds every alpha
    // with its own density channel (d_alpha = read_up*rho_up +
    // read_dn*rho_dn), so a mixed charge+spin list reaches the kernel in one
    // call — the A4 loop-side two-pass masking is retired.  The kernel
    // aborts loudly on a mu/channel/buffer mismatch.
    constraint::constraint_force(*wg_, rho, nspin, channels_, mu_, forcecon);
}

} // namespace constraint
