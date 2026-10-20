#ifndef CONSTRAINT_MU_SOLVER_H
#define CONSTRAINT_MU_SOLVER_H

#include <vector>

namespace constraint
{

struct MuSolverParams
{
    double step_max = 0.05;  // Ry: largest |dmu| per outer step per component
    // Ry: largest |dmu| for the FIRST step of a component only, i.e. the one
    // step taken before any secant slope has been measured.  Without a
    // history the solver falls back to kappa_min, so the first step always
    // sits at the step_max cap regardless of how far the target is -- a fixed
    // overshoot.  On a stiff correlated channel that overshoot is enough to
    // tear the SCF out of the reference magnetic branch (II-1 FeO: dQ/dmu
    // ~ 24 e/Ry, so one 0.05 Ry step moves Q by > 1 e).  A small probe lets
    // the secant measure a local slope first, after which step_max governs.
    // 0 (default) keeps the legacy behaviour (probe = step_max).
    double step_probe = 0.0;
    double mu_max = 5.0;     // Ry: hard cap on |mu| (fuse trigger boundary)
    // Per-constraint hard caps (stage A, A0 decision D3): parallel to the
    // component list; empty means every component uses the scalar 'mu_max'
    // above.  Different channels may need different fuse limits (soft
    // charge channels tolerate large caps, hard spin caps must fuse early).
    std::vector<double> mu_max_per_component;
    double kappa_min = 0.3;  // smallest kept |dQ/dmu| (secant guard)
    double kappa_max = 20.0; // largest kept |dQ/dmu| (secant guard)
    double conv_tol = 1e-4;  // e: per-component |Q - target| convergence
    int plateau_window = 3;  // fuse look-back: steps with <1% improvement
    // M3 C2: adaptive trust-region outer solver.  Off (default) keeps the
    // legacy component-secant path bit-identical.  When on, each trial step is
    // scored by rho = actual/predicted residual reduction:
    //   rho > eta_good -> radius *= grow (cap radius_max)
    //   rho < eta_bad  -> radius *= shrink and the step is rolled back
    // A multi-component list uses a Broyden rank-one Jacobian (when
    // broyden=true) instead of independent component secants.
    bool trust_region = false;
    bool broyden = true;
    double trust_radius_max = 1.0;   // Ry
    double trust_eta_good = 0.5;
    double trust_eta_bad = 0.1;
    double trust_shrink = 0.5;
    double trust_grow = 2.0;
    double trust_radius_min = 1e-8;  // Ry: numerical floor on the radius
    // After this many consecutive rejections the solver falls back to the
    // legacy secant path instead of shrinking forever (a noisy channel can
    // otherwise livelock on rejected sub-noise steps).
    int trust_max_rejects = 20;
    // Expected sign of the observed response dQ/dmu.  Both the charge and
    // the spin channel respond negatively (-1): a positive potential on the
    // fragment repels the coupled density (charge: rho; spin: m = rho_up -
    // rho_dn, because the split injection V_up += mu*w / V_dn -= mu*w
    // repels spin-up and attracts spin-down for mu > 0).  The sign sets both
    // the Newton-secant slope and the sign-flip guard reference; +1 is kept
    // as a parameter for channels with an inverted (non-standard) coupling.
    int response_sign = -1;
};

enum class MuStatus
{
    RUNNING,
    CONVERGED,
    UNREACHABLE,
    // Online energy branch guard (L10 lineage, set by ConstraintLoop, never
    // returned by MuSolver::step): the SCF converged at some mu, but the
    // constrained total energy fell below the unconstrained reference energy
    // by more than constraint_branch_tol.  That is the signature of a branch
    // flip — a different self-consistent solution (or a different magnetic
    // state) satisfied the weighted observable instead of the target being
    // reached along the reference branch.  BRANCH_FLIP fuses the run: the
    // targets are NOT validated, so the result must not be read as CONVERGED.
    BRANCH_FLIP
};

/**
 * @brief Outer-loop Lagrange multiplier solver (architecture layer M4).
 *
 * Component-wise secant on the observed response Q(mu).  Every component is
 * advanced independently with its own (mu_prev, Q_prev) history:
 *
 *   kappa_i = response_sign * clamp(|dQ_i/dmu_i|, kappa_min, kappa_max)
 *   dmu_i   = -(Q_i - target_i) / kappa_i,  |dmu_i| <= step_max
 *
 * Guards (each branch documented against its historical failure mode):
 *  - Sign flip (secant slope opposing the channel's expected response):
 *    fall back to kappa = response_sign * kappa_min and count the event;
 *    the update stays bounded and directional (T-4a').
 *  - Hard mu cap |mu_i| <= mu_max; a component pinned at the cap whose
 *    residual has not improved by >= 1% over the last plateau_window steps
 *    fuses the run with status UNREACHABLE (dead-channel, T-5').
 *  - Anti-fake convergence: Q == target at the very first step converges
 *    immediately; Q != target never reports CONVERGED.
 */
class MuSolver
{
  public:
    explicit MuSolver(const MuSolverParams& params = MuSolverParams());

    // Advance one outer step from the observed Q; mu is updated in place.
    // Q and target must keep the same size across calls.
    MuStatus step(const std::vector<double>& Q,
                  const std::vector<double>& target,
                  std::vector<double>& mu);

    // Forget all history (fresh outer run with a new target).
    void reset();

    // M2 warm start: anchor the secant history at a new geometry's first
    // converged observation (mu_at_obs, Q_at_obs).  The retained slope from
    // the previous geometry survives; mu_prev/Q_prev are replaced so the
    // geometry change can not be mistaken for a mu response.  The residual
    // window / fuse state are cleared (a new geometry is a new problem).
    void rebind(const std::vector<double>& mu,
                const std::vector<double>& Q);

    // Last accepted dQ/dmu of component i (signed, AFTER the
    // kappa_min/kappa_max clamp -- the value the update actually used;
    // negative for charge/spin), or the conservative
    // response_sign*kappa_min when none was measured.  A sign-flip fallback
    // value is recorded too (deliberate conservative choice: one bad step
    // degrades the retained slope to kappa_min instead of keeping a stale
    // measurement across geometries).
    double slope(const int i) const;

    // M3 C2/E1: force-reject the pending trial step and re-walk it with a
    // smaller trust radius.  Used by the magnetic branch anchor, which knows
    // the step was physically unacceptable even if its residual improved.
    // With no pending trial the last accepted (mu_prev_, Q_prev_) anchor is
    // used.  Returns RUNNING; the caller must rerun the SCF at the new mu.
    MuStatus reject(const std::vector<double>& Q,
                    const std::vector<double>& target,
                    std::vector<double>& mu);

    int nsteps() const { return nsteps_; }
    int sign_flip_count() const { return sign_flip_count_; }

    // M3 C2 diagnostics (zero/empty when trust_region is off).
    double trust_radius() const { return trust_radius_; }
    int trust_accept_count() const { return trust_accept_count_; }
    int trust_reject_count() const { return trust_reject_count_; }
    double last_rho() const { return last_rho_; }
    bool trust_fallback() const { return trust_fallback_; }
    double jacobian(const int i, const int j) const;

    // Fuse diagnostics: component that triggered UNREACHABLE, the mu value
    // and the residual at the fuse point (Q(mu) endpoint report).
    int fuse_component() const { return fuse_component_; }
    double fuse_mu() const { return fuse_mu_; }
    double fuse_residual() const { return fuse_residual_; }

  private:
    // Legacy component-secant path (pre-M3, must stay bit-identical).
    MuStatus step_secant(const std::vector<double>& Q,
                         const std::vector<double>& target,
                         std::vector<double>& mu);
    // M3 C2 trust-region/Broyden path.
    MuStatus step_trust(const std::vector<double>& Q,
                        const std::vector<double>& target,
                        std::vector<double>& mu);
    void ensure_history(const int n);
    double cap_for(const int i, const int n) const;
    // Solve J_active * dmu = -res_active; false on a singular system.
    bool solve_active(const std::vector<double>& res,
                      const std::vector<bool>& active,
                      std::vector<double>& dmu) const;
    std::vector<bool> active_for(const std::vector<double>& res) const;
    void broyden_update(const std::vector<double>& dmu,
                        const std::vector<double>& dQ);
    // Feed a (dmu, dQ) observation into the trust model.  The sign guard
    // keeps a pathological anti-response from flipping the solve direction;
    // accepted and bad-rho-rejected trials both use it (the rejected trial's
    // Q is still a converged observation, and without the update the retry
    // can repeat the identical step when the radius was not binding).
    void update_trial_model(const std::vector<double>& dmu,
                            const std::vector<double>& dQ);
    // Store a new pending trial from the current (restored) mu and the
    // residual at that mu.  Returns the step actually taken.
    void propose_trial(const std::vector<double>& res,
                       const std::vector<bool>& active,
                       std::vector<double>& mu);
    void reset_trust_state();

    MuSolverParams params_;
    std::vector<bool> has_history_;
    std::vector<double> mu_prev_;
    std::vector<double> Q_prev_;
    std::vector<double> last_slope_; // retained across rebind()
    std::vector<std::vector<double>> residual_window_;
    int nsteps_ = 0;
    int sign_flip_count_ = 0;
    int fuse_component_ = -1;
    double fuse_mu_ = 0.0;
    double fuse_residual_ = 0.0;
    // M3 C2 trust-region/Broyden state.
    std::vector<std::vector<double>> jacobian_;
    double trust_radius_ = 0.0;
    bool trust_pending_ = false;
    std::vector<double> mu_before_step_;
    std::vector<double> Q_before_step_;
    std::vector<double> res_before_step_;
    std::vector<double> pred_res_step_;
    int trust_accept_count_ = 0;
    int trust_reject_count_ = 0;
    int trust_consecutive_rejects_ = 0;
    double last_rho_ = 1.0;
    bool trust_fallback_ = false;
};

} // namespace constraint

#endif
