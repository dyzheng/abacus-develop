#include "mu_solver.h"

#include <algorithm>
#include <cmath>

namespace constraint
{

namespace
{
// C++14-compatible clamp (std::clamp is C++17).
double clamp_value(const double v, const double lo, const double hi)
{
    return std::max(lo, std::min(hi, v));
}
} // anonymous namespace

MuSolver::MuSolver(const MuSolverParams& params) : params_(params)
{
}

void MuSolver::reset()
{
    has_history_.clear();
    mu_prev_.clear();
    Q_prev_.clear();
    residual_window_.clear();
    nsteps_ = 0;
    sign_flip_count_ = 0;
    fuse_component_ = -1;
    fuse_mu_ = 0.0;
    fuse_residual_ = 0.0;
}

MuStatus MuSolver::step(const std::vector<double>& Q,
                        const std::vector<double>& target,
                        std::vector<double>& mu)
{
    const int n = static_cast<int>(Q.size());
    if (static_cast<int>(has_history_.size()) != n)
    {
        // First call with this component count: initialize history.
        has_history_.assign(n, false);
        mu_prev_.assign(n, 0.0);
        Q_prev_.assign(n, 0.0);
        residual_window_.assign(n, std::vector<double>());
    }

    bool all_converged = true;
    for (int i = 0; i < n; ++i)
    {
        const double mu_at_obs = mu[i]; // mu at which Q[i] was observed
        const double res = Q[i] - target[i];

        // Convergence check precedes any update: a component already at the
        // target (including the first step with mu=0) converges immediately,
        // and a nonzero residual can never report CONVERGED (anti-fake).
        if (std::abs(res) < params_.conv_tol)
        {
            has_history_[i] = true;
            mu_prev_[i] = mu_at_obs;
            Q_prev_[i] = Q[i];
            continue;
        }
        all_converged = false;

        // Secant slope.  Without history, or on a degenerate (zero) observed
        // step, fall back to a conservative negative response.
        double kappa = -params_.kappa_min;
        if (has_history_[i])
        {
            const double dmu_obs = mu_at_obs - mu_prev_[i];
            const double dQ_obs = Q[i] - Q_prev_[i];
            if (dmu_obs != 0.0)
            {
                const double raw = dQ_obs / dmu_obs;
                if (raw > 0.0)
                {
                    // Sign flip: the observed response is non-monotonic
                    // (historical secant oscillation / dead-channel).  Fall
                    // back to the conservative negative slope and keep the
                    // step bounded; the run may fuse later at the mu cap.
                    ++sign_flip_count_;
                    kappa = -params_.kappa_min;
                }
                else
                {
                    // Clamp the magnitude into [kappa_min, kappa_max] so a
                    // stiff (large |kappa|) or soft (small |kappa|) response
                    // cannot produce an unbounded or frozen update.
                    const double mag = clamp_value(-raw,
                                              params_.kappa_min,
                                              params_.kappa_max);
                    kappa = -mag;
                }
            }
        }

        // Newton-secant step toward the target, capped in magnitude.
        double dmu = -res / kappa;
        dmu = clamp_value(dmu, -params_.step_max, params_.step_max);
        mu[i] += dmu;

        // Hard cap: |mu| never leaves [-mu_max, mu_max].  A component pinned
        // at the cap is a fuse candidate below.
        if (mu[i] > params_.mu_max)
        {
            mu[i] = params_.mu_max;
        }
        else if (mu[i] < -params_.mu_max)
        {
            mu[i] = -params_.mu_max;
        }

        has_history_[i] = true;
        mu_prev_[i] = mu_at_obs;
        Q_prev_[i] = Q[i];

        // Residual ring for the plateau fuse check.
        std::vector<double>& window = residual_window_[i];
        window.push_back(std::abs(res));
        if (static_cast<int>(window.size()) > params_.plateau_window)
        {
            window.erase(window.begin());
        }

        // Fuse: pinned at the mu cap with a flat residual plateau over the
        // look-back window means the constraint is unreachable in this
        // channel (R4/R3 dead channel); report the Q(mu) endpoint instead of
        // marching on forever.
        const bool pinned = (mu[i] == params_.mu_max) || (mu[i] == -params_.mu_max);
        if (pinned && static_cast<int>(window.size()) == params_.plateau_window)
        {
            const double base = std::max(window.front(), 1e-30);
            const double improvement = (window.front() - window.back()) / base;
            if (improvement < 0.01)
            {
                fuse_component_ = i;
                fuse_mu_ = mu[i];
                fuse_residual_ = res;
                return MuStatus::UNREACHABLE;
            }
        }
    }

    if (all_converged)
    {
        return MuStatus::CONVERGED;
    }
    ++nsteps_;
    return MuStatus::RUNNING;
}

} // namespace constraint
