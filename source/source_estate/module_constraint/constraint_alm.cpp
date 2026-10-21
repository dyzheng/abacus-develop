#include "constraint_alm.h"

#include <algorithm>
#include <cmath>

namespace constraint
{

void AlmController::resize(const int n)
{
    const int count = std::max(0, n);
    k_.assign(count, 0.0);
    residual_window_.assign(count, {});
    armed_ = false;
    fuse_component_ = -1;
    fuse_residual_ = 0.0;
}

void AlmController::set_k_override(const double k)
{
    // Branch: <= 0 is the auto sentinel (also tolerates a negative value
    // here; the INPUT guard rejects it before the controller is armed).
    k_override_ = (k > 0.0) ? std::max(k_min_, std::min(k_max_, k)) : 0.0;
}

void AlmController::arm(const std::vector<double>& chi)
{
    const size_t n = k_.size();
    for (size_t a = 0; a < n; ++a)
    {
        // Branch: explicit k overrides the response-derived gain.
        if (k_override_ > 0.0)
        {
            k_[a] = k_override_;
            continue;
        }
        const double c = (a < chi.size()) ? chi[a] : 0.0;
        // Branch: a non-negative or numerically flat response cannot define
        // a stable penalty gain -> nominal |chi| = 1 fallback.
        if (c >= 0.0 || std::abs(c) < 1e-6)
        {
            k_[a] = kappa_target_;
            ++arm_fallback_;
        }
        else
        {
            k_[a] = std::max(k_min_, std::min(k_max_, kappa_target_ / std::abs(c)));
        }
    }
    armed_ = (n > 0);
    reset_fuse();
}

void AlmController::mu_eff(const std::vector<double>& mu,
                           const std::vector<double>& Q,
                           const std::vector<double>& target,
                           std::vector<double>& out) const
{
    const size_t n = mu.size();
    out.resize(n);
    for (size_t a = 0; a < n; ++a)
    {
        // Branch: before arming the penalty is inactive and the effective
        // multiplier is the plain Lagrange multiplier.
        if (!armed_ || a >= k_.size())
        {
            out[a] = mu[a];
        }
        else
        {
            const double res = (a < Q.size() ? Q[a] : 0.0)
                               - (a < target.size() ? target[a] : 0.0);
            out[a] = mu[a] + k_[a] * res;
        }
    }
}

double AlmController::update(const std::vector<double>& Q,
                             const std::vector<double>& target,
                             const std::vector<double>& caps,
                             std::vector<double>& mu)
{
    const size_t n = mu.size();
    double worst = 0.0;
    for (size_t a = 0; a < n; ++a)
    {
        const double res = (a < Q.size() ? Q[a] : 0.0)
                           - (a < target.size() ? target[a] : 0.0);
        worst = std::max(worst, std::abs(res));
        if (a >= k_.size())
        {
            continue;
        }
        // Plateau bookkeeping: retain the last plateau_window |residual|s
        // for this component (bounded memory, oldest dropped first).
        residual_window_[a].push_back(std::abs(res));
        if (residual_window_[a].size()
            > static_cast<size_t>(plateau_window_))
        {
            residual_window_[a].erase(residual_window_[a].begin());
        }
        const double cap = (a < caps.size()) ? std::abs(caps[a]) : 0.0;
        const double trial = mu[a] + k_[a] * res;
        mu[a] = std::max(-cap, std::min(cap, trial));
        // Branch: only a multiplier pinned at its cap whose residual has
        // stopped improving is genuinely unreachable (a moving residual is
        // still making progress and must keep walking).
        if (std::abs(std::abs(mu[a]) - cap) <= 1e-12
            && residual_window_[a].size()
                   == static_cast<size_t>(plateau_window_))
        {
            const double first = residual_window_[a].front();
            const double last = residual_window_[a].back();
            const double improvement = first - last;
            if (improvement <= 0.01 * std::max(first, 1e-30))
            {
                fuse_component_ = static_cast<int>(a);
                fuse_residual_ = std::abs(res);
            }
        }
    }
    return worst;
}

double AlmController::energy_correction(const std::vector<double>& mu,
                                        const std::vector<double>& Q,
                                        const std::vector<double>& target) const
{
    double e = 0.0;
    for (size_t a = 0; a < mu.size(); ++a)
    {
        const double res = (a < Q.size() ? Q[a] : 0.0)
                           - (a < target.size() ? target[a] : 0.0);
        e += mu[a] * res;
        // Branch: unarmed has no penalty term; this also keeps the probe
        // phase's correction identical to the legacy accounting formula.
        if (armed_ && a < k_.size())
        {
            e += 0.5 * k_[a] * res * res;
        }
    }
    return e;
}

void AlmController::dampen()
{
    for (double& kv : k_)
    {
        kv = std::max(k_min_, 0.5 * kv);
    }
    ++dampen_count_;
}

void AlmController::reset_fuse()
{
    for (auto& window : residual_window_)
    {
        window.clear();
    }
    fuse_component_ = -1;
    fuse_residual_ = 0.0;
}

} // namespace constraint
