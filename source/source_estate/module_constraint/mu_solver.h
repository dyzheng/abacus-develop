#ifndef CONSTRAINT_MU_SOLVER_H
#define CONSTRAINT_MU_SOLVER_H

#include <vector>

namespace constraint
{

struct MuSolverParams
{
    double step_max = 0.05;  // Ry: largest |dmu| per outer step per component
    double mu_max = 5.0;     // Ry: hard cap on |mu| (fuse trigger boundary)
    double kappa_min = 0.3;  // smallest kept |dQ/dmu| (secant guard)
    double kappa_max = 20.0; // largest kept |dQ/dmu| (secant guard)
    double conv_tol = 1e-4;  // e: per-component |Q - target| convergence
    int plateau_window = 3;  // fuse look-back: steps with <1% improvement
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
    UNREACHABLE
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

    int nsteps() const { return nsteps_; }
    int sign_flip_count() const { return sign_flip_count_; }

    // Fuse diagnostics: component that triggered UNREACHABLE, the mu value
    // and the residual at the fuse point (Q(mu) endpoint report).
    int fuse_component() const { return fuse_component_; }
    double fuse_mu() const { return fuse_mu_; }
    double fuse_residual() const { return fuse_residual_; }

  private:
    MuSolverParams params_;
    std::vector<bool> has_history_;
    std::vector<double> mu_prev_;
    std::vector<double> Q_prev_;
    std::vector<std::vector<double>> residual_window_;
    int nsteps_ = 0;
    int sign_flip_count_ = 0;
    int fuse_component_ = -1;
    double fuse_mu_ = 0.0;
    double fuse_residual_ = 0.0;
};

} // namespace constraint

#endif
