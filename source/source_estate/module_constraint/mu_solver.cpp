#include "mu_solver.h"

#include <algorithm>
#include <cmath>

#include "source_base/tool_quit.h"

namespace constraint
{

namespace
{
// C++14-compatible clamp (std::clamp is C++17).
double clamp_value(const double v, const double lo, const double hi)
{
    return std::max(lo, std::min(hi, v));
}

double max_abs(const std::vector<double>& v)
{
    double out = 0.0;
    for (const double x : v)
    {
        out = std::max(out, std::abs(x));
    }
    return out;
}

double norm2(const std::vector<double>& v)
{
    double out = 0.0;
    for (const double x : v)
    {
        out += x * x;
    }
    return std::sqrt(out);
}
} // anonymous namespace

MuSolver::MuSolver(const MuSolverParams& params) : params_(params)
{
    // Branch: the trust path owns a radius before the first observation;
    // expose the initial step_max cap so diagnostics are meaningful even
    // before step() is called.  The secant path leaves it at zero.
    if (params_.trust_region)
    {
        trust_radius_ = std::max(
            params_.trust_radius_min,
            std::min(params_.step_max, params_.trust_radius_max));
    }
}

void MuSolver::reset()
{
    has_history_.clear();
    mu_prev_.clear();
    Q_prev_.clear();
    last_slope_.clear();
    residual_window_.clear();
    nsteps_ = 0;
    sign_flip_count_ = 0;
    fuse_component_ = -1;
    fuse_mu_ = 0.0;
    fuse_residual_ = 0.0;
    reset_trust_state();
    carry_jacobian_ = false;
}

MuStatus MuSolver::step(const std::vector<double>& Q,
                        const std::vector<double>& target,
                        std::vector<double>& mu)
{
    // Branch A: trust region disabled (the default) or the run permanently
    // fell back after repeated rejected sub-noise steps.  The legacy secant
    // path is kept bit-identical for both cases.
    if (!params_.trust_region || trust_fallback_)
    {
        return step_secant(Q, target, mu);
    }
    // Branch B: adaptive trust-region path (M3 C2).
    return step_trust(Q, target, mu);
}

MuStatus MuSolver::step_secant(const std::vector<double>& Q,
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
        const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
        last_slope_.assign(n, resp * params_.kappa_min);
        residual_window_.assign(n, std::vector<double>());
    }
    // Guard: a heterogeneous per-constraint cap list must stay parallel to
    // the component list.  A size mismatch would silently pair a component
    // with the wrong fuse limit (wiring bug).
    if (!params_.mu_max_per_component.empty()
        && static_cast<int>(params_.mu_max_per_component.size()) != n)
    {
        ModuleBase::WARNING_QUIT(
            "MuSolver::step",
            "mu_max_per_component size does not match the component count");
    }

    // Expected response sign of the channel (MuSolverParams::response_sign):
    // both charge and spin respond negatively — a positive potential on the
    // fragment repels the coupled density (for spin, V_up += mu*w repels
    // spin-up and V_dn -= mu*w attracts spin-down, so m = rho_up - rho_dn
    // falls with mu).  The sign flips both the Newton-secant slope and the
    // sign-flip guard reference.
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;

    // First-step cap.  A component with no accepted observation yet has no
    // measured secant slope, so kappa falls back to kappa_min and the
    // Newton step would always be clipped at step_max -- the fixed overshoot
    // that step_probe exists to avoid.  step_probe == 0 means "no probe":
    // fall back to step_max, i.e. the pre-probe behaviour.
    const double probe_cap = (params_.step_probe > 0.0) ? params_.step_probe
                                                        : params_.step_max;

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
        // step, fall back to the channel's conservative response.
        double kappa = slope(i);
        if (has_history_[i])
        {
            const double dmu_obs = mu_at_obs - mu_prev_[i];
            const double dQ_obs = Q[i] - Q_prev_[i];
            if (dmu_obs != 0.0)
            {
                const double raw = dQ_obs / dmu_obs;
                if (raw * resp < 0.0)
                {
                    // Sign flip: the observed response opposes the channel's
                    // expected sign (historical secant oscillation /
                    // dead-channel).  Fall back to the conservative slope
                    // and keep the step bounded; the run may fuse later at
                    // the mu cap.
                    ++sign_flip_count_;
                    kappa = resp * params_.kappa_min;
                }
                else
                {
                    // Clamp the magnitude into [kappa_min, kappa_max] so a
                    // stiff (large |kappa|) or soft (small |kappa|) response
                    // cannot produce an unbounded or frozen update.
                    const double mag = clamp_value(std::abs(raw),
                                              params_.kappa_min,
                                              params_.kappa_max);
                    kappa = resp * mag;
                }
                // Retain the value the update actually used: this is the
                // slope handed to the next ionic geometry by rebind().
                last_slope_[i] = kappa;
            }
        }

        // Newton-secant step toward the target, capped in magnitude.
        double dmu = -res / kappa;
        // Branch A: history-free component -- the probe cap applies (the
        // step is a slope measurement, not a Newton move).
        // Branch B: at least one accepted observation exists -- the secant
        // slope is real, so the full step_max cap applies.
        const double step_cap = has_history_[i] ? params_.step_max : probe_cap;
        dmu = clamp_value(dmu, -step_cap, step_cap);
        mu[i] += dmu;

        // Per-constraint hard cap (A0 D3): |mu_i| never leaves
        // [-cap_i, cap_i].  The per-component list overrides the scalar
        // fallback so a mixed run honors each constraint's own fuse limit; a
        // component pinned at its cap is a fuse candidate below.
        const double cap_i = (static_cast<int>(
                                  params_.mu_max_per_component.size())
                              == n)
                                 ? params_.mu_max_per_component[i]
                                 : params_.mu_max;
        if (mu[i] > cap_i)
        {
            mu[i] = cap_i;
        }
        else if (mu[i] < -cap_i)
        {
            mu[i] = -cap_i;
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
        const bool pinned = (mu[i] == cap_i) || (mu[i] == -cap_i);
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

void MuSolver::ensure_history(const int n)
{
    if (static_cast<int>(has_history_.size()) == n)
    {
        // A trust run keeps one Jacobian / radius per component list.  A
        // previous secant run (or reset) may have left them empty.
        if (static_cast<int>(jacobian_.size()) != n)
        {
            jacobian_.assign(n, std::vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
            {
                jacobian_[i][i] = slope(i);
            }
        }
        if (trust_radius_ <= 0.0)
        {
            trust_radius_ = std::max(
                params_.trust_radius_min,
                std::min(params_.step_max, params_.trust_radius_max));
        }
        return;
    }

    has_history_.assign(n, false);
    mu_prev_.assign(n, 0.0);
    Q_prev_.assign(n, 0.0);
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
    last_slope_.assign(n, resp * params_.kappa_min);
    residual_window_.assign(n, std::vector<double>());
    reset_trust_state();
    jacobian_.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        jacobian_[i][i] = last_slope_[i];
    }
    trust_radius_ = std::max(
        params_.trust_radius_min,
        std::min(params_.step_max, params_.trust_radius_max));
}

double MuSolver::cap_for(const int i, const int n) const
{
    return (static_cast<int>(params_.mu_max_per_component.size()) == n)
               ? params_.mu_max_per_component[i]
               : params_.mu_max;
}

double MuSolver::jacobian(const int i, const int j) const
{
    if (i < 0 || j < 0 || i >= static_cast<int>(jacobian_.size())
        || j >= static_cast<int>(jacobian_[i].size()))
    {
        return 0.0;
    }
    return jacobian_[i][j];
}

void MuSolver::reset_trust_state()
{
    jacobian_.clear();
    trust_radius_ = 0.0;
    trust_pending_ = false;
    mu_before_step_.clear();
    Q_before_step_.clear();
    res_before_step_.clear();
    pred_res_step_.clear();
    trust_accept_count_ = 0;
    trust_reject_count_ = 0;
    trust_consecutive_rejects_ = 0;
    last_rho_ = 1.0;
    trust_fallback_ = false;
}

bool MuSolver::solve_active(const std::vector<double>& res,
                            const std::vector<bool>& active,
                            std::vector<double>& dmu) const
{
    const int n = static_cast<int>(res.size());
    dmu.assign(n, 0.0);
    std::vector<int> idx;
    idx.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        if (active[i])
        {
            idx.push_back(i);
        }
    }
    const int m = static_cast<int>(idx.size());
    if (m == 0)
    {
        return true;
    }
    // Reduced dense system A dmu = -res.  The constraint count is tiny
    // (one per fragment): a partial-pivoted elimination is sufficient.
    std::vector<std::vector<double>> a(m, std::vector<double>(m + 1, 0.0));
    for (int r = 0; r < m; ++r)
    {
        for (int c = 0; c < m; ++c)
        {
            a[r][c] = jacobian_[idx[r]][idx[c]];
        }
        a[r][m] = -res[idx[r]];
    }
    for (int col = 0; col < m; ++col)
    {
        int pivot = col;
        for (int r = col + 1; r < m; ++r)
        {
            if (std::abs(a[r][col]) > std::abs(a[pivot][col]))
            {
                pivot = r;
            }
        }
        // Branch A: singular (or numerically dead) reduced Jacobian.  The
        // caller falls back to a diagonal secant step.
        if (std::abs(a[pivot][col]) < 1e-12)
        {
            return false;
        }
        // Branch B: regular pivot -- swap into place and eliminate below.
        if (pivot != col)
        {
            std::swap(a[pivot], a[col]);
        }
        for (int r = col + 1; r < m; ++r)
        {
            const double factor = a[r][col] / a[col][col];
            for (int c = col; c <= m; ++c)
            {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    for (int r = m - 1; r >= 0; --r)
    {
        double rhs = a[r][m];
        for (int c = r + 1; c < m; ++c)
        {
            rhs -= a[r][c] * dmu[idx[c]];
        }
        dmu[idx[r]] = rhs / a[r][r];
    }
    return true;
}

void MuSolver::broyden_update(const std::vector<double>& dmu,
                              const std::vector<double>& dQ)
{
    const int n = static_cast<int>(dmu.size());
    if (n == 0 || static_cast<int>(jacobian_.size()) != n
        || static_cast<int>(dQ.size()) != n)
    {
        return;
    }
    double denom = 0.0;
    for (int j = 0; j < n; ++j)
    {
        denom += dmu[j] * dmu[j];
    }
    // A zero-length trial carries no curvature information.
    if (denom < 1e-24)
    {
        return;
    }
    std::vector<double> Jdmu(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            Jdmu[i] += jacobian_[i][j] * dmu[j];
        }
    }
    for (int i = 0; i < n; ++i)
    {
        const double coeff = (dQ[i] - Jdmu[i]) / denom;
        // Branch A: non-finite curvature (a degenerate observation) is
        // discarded rather than poisoning every later solve.
        if (!std::isfinite(coeff))
        {
            continue;
        }
        // Branch B: standard good-Broyden rank-one update.
        for (int j = 0; j < n; ++j)
        {
            jacobian_[i][j] += coeff * dmu[j];
        }
    }
}

void MuSolver::update_trial_model(const std::vector<double>& dmu,
                                  const std::vector<double>& dQ)
{
    const int n = static_cast<int>(dmu.size());
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
    bool sign_ok = true;
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(dmu[i]) <= 1e-15)
        {
            continue;
        }
        const double raw = dQ[i] / dmu[i];
        if (raw * resp < 0.0)
        {
            sign_ok = false;
            ++sign_flip_count_;
        }
    }
    // Branch A: at least one component opposes the channel's expected
    // response.  Keep the conservative diagonal and skip the rank-one
    // update: feeding a branch-inverted observation into J would flip the
    // next Newton direction (see TrustRegionRejectsBadStep).
    if (!sign_ok)
    {
        for (int i = 0; i < n; ++i)
        {
            if (std::abs(dmu[i]) <= 1e-15)
            {
                continue;
            }
            last_slope_[i] = resp * params_.kappa_min;
            if (!params_.broyden)
            {
                jacobian_[i][i] = last_slope_[i];
            }
        }
        return;
    }
    // Branch B: physically consistent response -> coupled rank-one update
    // (or the diagonal-only variant when broyden=false).
    if (params_.broyden)
    {
        broyden_update(dmu, dQ);
    }
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(dmu[i]) <= 1e-15)
        {
            continue;
        }
        const double raw = dQ[i] / dmu[i];
        const double kappa = resp * clamp_value(std::abs(raw), params_.kappa_min,
                                                params_.kappa_max);
        last_slope_[i] = kappa;
        if (!params_.broyden)
        {
            jacobian_[i][i] = kappa;
        }
    }
}

void MuSolver::propose_trial(const std::vector<double>& res,
                             const std::vector<bool>& active,
                             std::vector<double>& mu)
{
    const int n = static_cast<int>(res.size());
    mu_before_step_ = mu;
    res_before_step_ = res;

    std::vector<double> dmu;
    if (!solve_active(res, active, dmu))
    {
        // Fallback: decoupled diagonal secant when the reduced system is
        // singular (e.g. two identical constraints before any coupling is
        // learned).  The Broyden update can restore a regular J later.
        dmu.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            if (!active[i])
            {
                continue;
            }
            const double jii = (static_cast<int>(jacobian_.size()) == n)
                                   ? jacobian_[i][i]
                                   : slope(i);
            const double denom = (std::abs(jii) > 1e-12) ? jii : slope(i);
            dmu[i] = -res[i] / denom;
        }
    }

    // Trust-region length cap in mu space.  A single-component system
    // reduces to |dmu| <= radius, the classical one-dimensional rule.
    const double len = norm2(dmu);
    if (len > trust_radius_ && len > 0.0)
    {
        const double scale = trust_radius_ / len;
        for (double& x : dmu)
        {
            x *= scale;
        }
    }

    // Per-component hard cap (A0 D3) and the predicted residual at the
    // actually-taken (possibly clamped) step.
    for (int i = 0; i < n; ++i)
    {
        mu[i] += dmu[i];
        const double cap_i = cap_for(i, n);
        mu[i] = clamp_value(mu[i], -cap_i, cap_i);
        dmu[i] = mu[i] - mu_before_step_[i];
    }
    pred_res_step_.assign(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        pred_res_step_[i] = res[i];
        for (int j = 0; j < n; ++j)
        {
            pred_res_step_[i] += jacobian_[i][j] * dmu[j];
        }
    }
    trust_pending_ = true;
}

MuStatus MuSolver::step_trust(const std::vector<double>& Q,
                              const std::vector<double>& target,
                              std::vector<double>& mu)
{
    const int n = static_cast<int>(Q.size());
    ensure_history(n);
    // Guard: the per-constraint cap list must stay parallel to Q/target.
    if (!params_.mu_max_per_component.empty()
        && static_cast<int>(params_.mu_max_per_component.size()) != n)
    {
        ModuleBase::WARNING_QUIT(
            "MuSolver::step_trust",
            "mu_max_per_component size does not match the component count");
    }

    std::vector<double> res(n, 0.0);
    std::vector<bool> active(n, false);
    bool all_converged = true;
    for (int i = 0; i < n; ++i)
    {
        res[i] = Q[i] - target[i];
        active[i] = std::abs(res[i]) >= params_.conv_tol;
        all_converged = all_converged && !active[i];
    }

    // Branch A: no trial pending (first call, or after a forced fallback).
    // Take the first trust step from the current observation.
    if (!trust_pending_)
    {
        if (all_converged)
        {
            for (int i = 0; i < n; ++i)
            {
                has_history_[i] = true;
                mu_prev_[i] = mu[i];
                Q_prev_[i] = Q[i];
            }
            return MuStatus::CONVERGED;
        }
        Q_before_step_ = Q;
        propose_trial(res, active, mu);
        ++nsteps_;
        return MuStatus::RUNNING;
    }

    // Branch B: score the pending trial by the actual/predicted residual
    // reduction.  The residual norm is the max norm, matching the
    // per-component convergence test.
    std::vector<double> dmu(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        dmu[i] = mu[i] - mu_before_step_[i];
    }
    // Branch B1: a zero-length trial (mu pinned at its cap) cannot be
    // scored; hand the remainder of the run to the secant path, whose
    // plateau fuse reports the unreachable channel.
    if (norm2(dmu) < 1e-15)
    {
        trust_pending_ = false;
        trust_fallback_ = true;
        trust_consecutive_rejects_ = 0;
        return step_secant(Q, target, mu);
    }

    const double r_before = max_abs(res_before_step_);
    const double r_pred = max_abs(pred_res_step_);
    const double r_now = max_abs(res);
    const double actual_reduction = r_before - r_now;
    const double predicted_reduction = r_before - r_pred;
    double rho = 0.0;
    if (predicted_reduction > 1e-30)
    {
        rho = actual_reduction / predicted_reduction;
    }
    else
    {
        // The model predicted no improvement.  A real improvement is then
        // an unexpectedly good step; otherwise the trial is rejected.
        rho = (actual_reduction > 0.0) ? 1.0 : -1.0;
    }
    last_rho_ = rho;

    // Branch C: bad step.  Roll mu back to the pre-trial anchor and re-walk
    // with a smaller radius.  Convergence is respected even if the model
    // predicted poorly: a target already met is never thrown away.
    if (!all_converged && rho < params_.trust_eta_bad)
    {
        ++trust_reject_count_;
        ++trust_consecutive_rejects_;
        trust_radius_ = std::max(params_.trust_radius_min,
                                 trust_radius_ * params_.trust_shrink);
        // The rejected trial is still a converged Q(mu) observation.  Fold it
        // into the model before restoring mu; otherwise a step that was not
        // radius-limited is proposed again verbatim and rejected forever
        // (integration 2026-10-20: 4 identical rejects per outer step).
        std::vector<double> dQ_obs(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            dQ_obs[i] = Q[i] - Q_before_step_[i];
        }
        update_trial_model(dmu, dQ_obs);
        mu = mu_before_step_;
        trust_pending_ = false;
        if (trust_consecutive_rejects_ >= params_.trust_max_rejects)
        {
            // Branch C1: rejected too many times in a row -- a noisy or
            // discontinuous channel.  Permanently fall back to the legacy
            // secant path, whose history/fuse guards are designed for it.
            trust_fallback_ = true;
            trust_consecutive_rejects_ = 0;
            return step_secant(Q_before_step_, target, mu);
        }
        // Branch C2: re-propose immediately from the restored anchor.  The
        // caller reruns the SCF at this new, smaller mu.
        propose_trial(res_before_step_, active_for(res_before_step_), mu);
        ++nsteps_;
        return MuStatus::RUNNING;
    }

    // Branch D: accepted step.  Grow the radius only for a clearly good
    // model, update the Jacobian with the observed response, then propose
    // the next trial.
    ++trust_accept_count_;
    trust_consecutive_rejects_ = 0;
    if (rho > params_.trust_eta_good)
    {
        trust_radius_ = std::min(params_.trust_radius_max,
                                 trust_radius_ * params_.trust_grow);
    }
    std::vector<double> dQ_obs(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        dQ_obs[i] = Q[i] - Q_before_step_[i];
    }
    // Coupled Broyden model (default) or the diagonal-only variant.  The
    // helper applies the response-sign guard before touching J.
    update_trial_model(dmu, dQ_obs);
    for (int i = 0; i < n; ++i)
    {
        has_history_[i] = true;
        mu_prev_[i] = mu[i];
        Q_prev_[i] = Q[i];
        std::vector<double>& window = residual_window_[i];
        window.push_back(std::abs(res[i]));
        if (static_cast<int>(window.size()) > params_.plateau_window)
        {
            window.erase(window.begin());
        }
    }
    trust_pending_ = false;
    if (all_converged)
    {
        return MuStatus::CONVERGED;
    }
    Q_before_step_ = Q;
    propose_trial(res, active, mu);
    ++nsteps_;
    return MuStatus::RUNNING;
}

std::vector<bool> MuSolver::active_for(const std::vector<double>& res) const
{
    std::vector<bool> active(res.size(), false);
    for (size_t i = 0; i < res.size(); ++i)
    {
        active[i] = std::abs(res[i]) >= params_.conv_tol;
    }
    return active;
}

MuStatus MuSolver::reject(const std::vector<double>& Q,
                          const std::vector<double>& target,
                          std::vector<double>& mu)
{
    const int n = static_cast<int>(Q.size());
    // Guard: the magnetic anchor is only meaningful with the trust region
    // rollback machinery; IO enforces this, but keep the API safe.
    if (!params_.trust_region)
    {
        ModuleBase::WARNING_QUIT(
            "MuSolver::reject",
            "reject() requires MuSolverParams::trust_region=true");
    }
    ensure_history(n);
    if (!trust_pending_)
    {
        // Branch A: no pending trial.  Restore the last accepted anchor and
        // re-walk from it with the shrunken radius.
        if (static_cast<int>(mu_prev_.size()) == n)
        {
            mu = mu_prev_;
        }
        const std::vector<double> Q_anchor =
            (static_cast<int>(Q_prev_.size()) == n) ? Q_prev_ : Q;
        std::vector<double> res(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            res[i] = Q_anchor[i] - target[i];
        }
        ++trust_reject_count_;
        ++trust_consecutive_rejects_;
        trust_radius_ = std::max(params_.trust_radius_min,
                                 trust_radius_ * params_.trust_shrink);
        if (trust_consecutive_rejects_ >= params_.trust_max_rejects)
        {
            trust_fallback_ = true;
            trust_consecutive_rejects_ = 0;
            return step_secant(Q_anchor, target, mu);
        }
        Q_before_step_ = Q_anchor;
        propose_trial(res, active_for(res), mu);
        ++nsteps_;
        return MuStatus::RUNNING;
    }

    // Branch B: a trial is pending -- force the same rollback/re-walk as a
    // bad rho, regardless of the residual improvement.
    ++trust_reject_count_;
    ++trust_consecutive_rejects_;
    trust_radius_ = std::max(params_.trust_radius_min,
                             trust_radius_ * params_.trust_shrink);
    mu = mu_before_step_;
    trust_pending_ = false;
    if (trust_consecutive_rejects_ >= params_.trust_max_rejects)
    {
        trust_fallback_ = true;
        trust_consecutive_rejects_ = 0;
        return step_secant(Q_before_step_, target, mu);
    }
    propose_trial(res_before_step_, active_for(res_before_step_), mu);
    ++nsteps_;
    return MuStatus::RUNNING;
}

void MuSolver::rebind(const std::vector<double>& mu,
                      const std::vector<double>& Q)
{
    const int n = static_cast<int>(Q.size());
    // Guard: the anchor is one (mu, Q) pair per component.  A size mismatch
    // would make later history reads pair observations from different
    // components (wiring bug), so refuse loudly.
    if (static_cast<int>(mu.size()) != n)
    {
        ModuleBase::WARNING_QUIT(
            "MuSolver::rebind",
            "mu and Q sizes differ while anchoring a new geometry");
    }
    has_history_.assign(n, true);
    mu_prev_ = mu;
    Q_prev_ = Q;
    residual_window_.assign(n, std::vector<double>());
    fuse_component_ = -1;
    fuse_mu_ = 0.0;
    fuse_residual_ = 0.0;
    // last_slope_ intentionally NOT cleared: it is the kappa carried across
    // ionic steps (C1).
    if (static_cast<int>(last_slope_.size()) != n)
    {
        const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
        last_slope_.assign(n, resp * params_.kappa_min);
    }
    // Branch: a new geometry with the trust path on starts with the carried
    // diagonal response, no pending trial, a fresh radius and no permanent
    // fallback (the new fixed-point problem may be smoother).
    if (params_.trust_region)
    {
        // Branch: carry the full Jacobian across geometries (M4 scheduling
        // pack A, opt-in).  The off-diagonal fragment couplings cost outer
        // steps to relearn every frame; the trust-region rho score rejects
        // and relearns a stale model, so carrying is safe.  Legacy (flag
        // off): diagonal reset from the carried slopes.
        if (!carry_jacobian_)
        {
            jacobian_.assign(n, std::vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
            {
                jacobian_[i][i] = last_slope_[i];
            }
        }
        else if (static_cast<int>(jacobian_.size()) != n)
        {
            // Guard: a changed component count has no compatible old J, so
            // seed the carried diagonal instead of reusing a wrong shape.
            jacobian_.assign(n, std::vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
            {
                jacobian_[i][i] = last_slope_[i];
            }
        }
        trust_radius_ = std::max(
            params_.trust_radius_min,
            std::min(params_.step_max, params_.trust_radius_max));
        trust_pending_ = false;
        mu_before_step_.clear();
        Q_before_step_.clear();
        res_before_step_.clear();
        pred_res_step_.clear();
        trust_consecutive_rejects_ = 0;
        trust_fallback_ = false;
    }
}

double MuSolver::slope(const int i) const
{
    if (i >= 0 && i < static_cast<int>(last_slope_.size()))
    {
        return last_slope_[i];
    }
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
    return resp * params_.kappa_min;
}

void MuSolver::seed_slope(const std::vector<double>& chi)
{
    const int n = static_cast<int>(chi.size());
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
    last_slope_.assign(n, resp * params_.kappa_min);
    for (int i = 0; i < n; ++i)
    {
        // Branch: a response with the expected sign is retained; the wrong
        // sign or a dead channel keeps the conservative fallback (the same
        // guard as the secant path).
        if (chi[i] * resp > 0.0 && std::abs(chi[i]) >= 1e-6)
        {
            const double mag = clamp_value(std::abs(chi[i]), params_.kappa_min,
                                           params_.kappa_max);
            last_slope_[i] = resp * mag;
        }
    }
    // A probe is not an accepted Newton step: keep the history empty so a
    // later legacy step() cannot pair this response with a cross-iteration
    // delta it never observed.
    has_history_.assign(n, false);
    mu_prev_.assign(n, 0.0);
    Q_prev_.assign(n, 0.0);
    residual_window_.assign(n, std::vector<double>());
    fuse_component_ = -1;
    fuse_mu_ = 0.0;
    fuse_residual_ = 0.0;
    // The trust model, when enabled, starts from the seeded diagonal.
    if (params_.trust_region)
    {
        jacobian_.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
        {
            jacobian_[i][i] = last_slope_[i];
        }
    }
}

} // namespace constraint
