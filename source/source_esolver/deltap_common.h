#ifndef DELTAP_COMMON_H
#define DELTAP_COMMON_H
/**
 * @file deltap_common.h
 * @brief Basis-independent DeltaP constraint math (pure functions, no state).
 *
 * Single implementation of the residual / gradient-descent / constraint-space
 * conversions shared by the LCAO and PW SCF drivers (DeltapScfSolver).
 * All functions are stateless and unit-testable.
 */

#include <vector>
#include <cmath>
#include <algorithm>

#include "source_base/constants.h"

namespace deltap_common {

/**
 * Compute constraint residual r = C·γ − t (matrix mode) or r_i = γ_i − t_i
 * (per-atom mode; missing targets treated as 0).
 *
 * @param C      Constraint matrix (m × nat).  Empty → per-atom mode.
 * @param t      Targets: [m] for matrix mode, [nat] for per-atom mode.
 * @param gamma  Per-atom gamma [nat].
 * @return       Residual [m] or [nat].
 */
inline std::vector<double> compute_residual(
    const std::vector<std::vector<double>>& C,
    const std::vector<double>& t,
    const std::vector<double>& gamma)
{
    const int nat = static_cast<int>(gamma.size());
    const int m = static_cast<int>(C.size());
    std::vector<double> r;
    if (m > 0)
    {
        r.assign(m, 0.0);
        for (int a = 0; a < m; ++a)
        {
            for (int i = 0; i < nat; ++i)
                r[a] += C[a][i] * gamma[i];
            r[a] -= t[a];
        }
    }
    else
    {
        r.resize(nat, 0.0);
        for (int i = 0; i < nat; ++i)
            r[i] = gamma[i] - (i < static_cast<int>(t.size()) ? t[i] : 0.0);
    }
    return r;
}

/**
 * Max-norm of a residual vector (for convergence checks / reporting).
 */
inline double max_norm(const std::vector<double>& r)
{
    double mx = 0.0;
    for (double v : r)
        mx = std::max(mx, std::abs(v));
    return mx;
}

/**
 * Nearest-branch 2π unwrap: shift each γ to the 2π branch closest to the
 * previous branch-selected value, raw − round((raw−prev)/2π)·2π.  Used for
 * cross-SCF-step branch tracking in the PW path (LCAO does target-aware
 * branch selection inside module_deltap instead).  Empty `prev` (first
 * measurement of a SCF cycle) is returned unchanged.
 */
inline std::vector<double> unwrap_2pi(const std::vector<double>& gamma,
                                      const std::vector<double>& prev)
{
    std::vector<double> out = gamma;
    if (prev.empty())
        return out;
    const int n = std::min(static_cast<int>(out.size()), static_cast<int>(prev.size()));
    for (int i = 0; i < n; ++i)
    {
        const double n2pi = std::round((out[i] - prev[i]) / (2.0 * ModuleBase::PI));
        out[i] -= n2pi * 2.0 * ModuleBase::PI;
    }
    return out;
}

/**
 * Gradient-descent λ update with damping: λ ← mixing·(λ + step·r) + (1−mixing)·λ.
 * Honours the per-atom constrain mask (free atoms are left untouched).
 */
inline void gd_update(std::vector<double>& lambda,
                      const std::vector<double>& r,
                      const std::vector<int>& constrain,
                      double step,
                      double mixing)
{
    const int n = static_cast<int>(std::min(lambda.size(), r.size()));
    for (int i = 0; i < n; ++i)
    {
        bool constrained = (constrain.empty()
                            || i >= static_cast<int>(constrain.size())
                            || constrain[i] != 0);
        if (!constrained)
            continue;
        lambda[i] = mixing * (lambda[i] + step * r[i]) + (1.0 - mixing) * lambda[i];
    }
}

/**
 * Total-mode gradient descent: one shared step Σ_i r_i applied to every atom,
 * then damped with mixing.  No per-atom mask (matches total-constraint
 * semantics where all atoms share a single λ shift).
 */
inline void gd_update_total(std::vector<double>& lambda,
                            const std::vector<double>& r,
                            double step,
                            double mixing)
{
    double sum_r = 0.0;
    for (double v : r)
        sum_r += v;
    const double delta = step * sum_r;
    for (double& l : lambda)
        l = mixing * (l + delta) + (1.0 - mixing) * l;
}

/**
 * Convert constraint-space λ to effective per-atom λ: λ_eff[i] = Σ_a λ[a]·C[a][i].
 * Empty C → identity (per-atom mode).
 */
inline std::vector<double> to_effective_lambda(
    const std::vector<double>& lambda_cstr,
    const std::vector<std::vector<double>>& C,
    int nat)
{
    std::vector<double> out(nat, 0.0);
    const int m = static_cast<int>(C.size());
    if (m > 0)
    {
        for (int i = 0; i < nat; ++i)
            for (int a = 0; a < m; ++a)
                out[i] += lambda_cstr[a] * C[a][i];
    }
    else
    {
        for (int i = 0; i < std::min(nat, static_cast<int>(lambda_cstr.size())); ++i)
            out[i] = lambda_cstr[i];
    }
    return out;
}

/**
 * Compute dp_escon = -Σ λ_i · γ_i  (constraint energy correction, Ry).
 */
inline double compute_dp_escon(
    const std::vector<double>& lambda_eff,
    const std::vector<double>& gamma_I)
{
    double escon = 0.0;
    int n = std::min(static_cast<int>(lambda_eff.size()), static_cast<int>(gamma_I.size()));
    for (int i = 0; i < n; ++i)
        escon -= lambda_eff[i] * gamma_I[i];
    return escon;
}

} // namespace deltap_common
#endif // DELTAP_COMMON_H
