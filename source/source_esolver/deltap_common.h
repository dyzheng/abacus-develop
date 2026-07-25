#ifndef DELTAP_COMMON_H
#define DELTAP_COMMON_H
/**
 * @file deltap_common.h
 * @brief Basis-independent DeltaP constraint logic (shared between LCAO and PW).
 *
 * Separates the linear algebra of lambda updates, constraint matrix handling,
 * and residual computation from the basis-specific Hamiltonian application
 * and gamma computation.
 *
 * Currently only compute_dp_escon() is used (by LCAO).  The other functions
 * are kept for future constraint-matrix / total-mode refactoring.
 */

#include <vector>
#include <cmath>
#include <algorithm>

namespace deltap_common {

/**
 * Update constraint-space lambda vector using gradient descent.
 *
 * @param C            Constraint matrix (m × n_atoms).  Empty for per_atom mode.
 * @param t            Constraint targets (m).  Empty for per_atom mode.
 * @param gamma_I      Per-atom gamma values [n_atoms].
 * @param lambda_cstr  Constraint-space lambda [m] (in/out).  For per_atom: same as per-atom lambda.
 * @param constrain    Per-atom constrain flags [n_atoms] (empty = all constrained).
 * @param step         Gradient descent step size.
 * @param mixing       Damping factor (0 = no mixing, 1 = full step).
 * @param lambda_out   Effective per-atom lambda [n_atoms] (output).
 * @return             Max residual norm (for convergence check).
 * @note This function is currently unused — both LCAO and PW paths
 *   implement lambda updates inline.  Kept for future total/constraint-matrix
 *   refactoring.
 */
inline double update_lambda(
    const std::vector<std::vector<double>>& C,
    const std::vector<double>& t,
    const std::vector<double>& gamma_I,
    std::vector<double>& lambda_cstr,
    const std::vector<int>& constrain,
    double step,
    double mixing,
    std::vector<double>& lambda_out)
{
    int nat = static_cast<int>(gamma_I.size());
    int m = static_cast<int>(C.size());
    bool use_constraint_matrix = (m > 0);

    if (use_constraint_matrix)
    {
        // Constraint matrix mode: r[α] = Σ_i C[α][i]·γ_i - t[α]
        lambda_cstr.resize(m, 0.0);
        std::vector<double> lambda_raw = lambda_cstr;
        double max_res = 0.0;
        for (int a = 0; a < m; ++a)
        {
            double residual = 0.0;
            for (int i = 0; i < nat; ++i)
                residual += C[a][i] * gamma_I[i];
            residual -= t[a];
            max_res = std::max(max_res, std::abs(residual));
            lambda_raw[a] += step * residual;
        }
        for (int a = 0; a < m; ++a)
            lambda_cstr[a] = mixing * lambda_raw[a] + (1.0 - mixing) * lambda_cstr[a];

        // Convert to effective per-atom lambda: λ_eff[i] = Σ_a λ[a]·C[a][i]
        lambda_out.assign(nat, 0.0);
        for (int i = 0; i < nat; ++i)
            for (int a = 0; a < m; ++a)
                lambda_out[i] += lambda_cstr[a] * C[a][i];
        return max_res;
    }
    else
    {
        // Per-atom mode: r[i] = γ_i - t_i
        lambda_out = lambda_cstr;  // lambda_cstr IS the per-atom lambda
        lambda_cstr.resize(nat, 0.0);
        std::vector<double> lambda_raw = lambda_out;
        double max_res = 0.0;
        for (int i = 0; i < nat; ++i)
        {
            bool constrained = (constrain.empty() || static_cast<size_t>(i) >= constrain.size() || constrain[i] != 0);
            if (!constrained || t.empty()) continue;
            double residual = gamma_I[i] - t[i];
            max_res = std::max(max_res, std::abs(residual));
            lambda_raw[i] += step * residual;
        }
        for (int i = 0; i < nat; ++i)
            lambda_out[i] = mixing * lambda_raw[i] + (1.0 - mixing) * lambda_out[i];
        lambda_cstr = lambda_out;
        return max_res;
    }
}

/**
 * Convert constraint-space lambda to effective per-atom lambda for Hamiltonian.
 *
 * @param C            Constraint matrix (m × n_atoms).  Empty → identity.
 * @param lambda_cstr  Constraint-space lambda [m].
 * @param lambda_eff   Effective per-atom lambda [n_atoms] (output).
 */
inline void to_effective_lambda(
    const std::vector<std::vector<double>>& C,
    const std::vector<double>& lambda_cstr,
    int nat,
    std::vector<double>& lambda_eff)
{
    int m = static_cast<int>(C.size());
    lambda_eff.assign(nat, 0.0);
    if (m > 0)
    {
        for (int i = 0; i < nat; ++i)
            for (int a = 0; a < m; ++a)
                lambda_eff[i] += lambda_cstr[a] * C[a][i];
    }
    else
    {
        lambda_eff = lambda_cstr;
        lambda_eff.resize(nat, 0.0);
        for (int i = 0; i < std::min(nat, static_cast<int>(lambda_cstr.size())); ++i)
            lambda_eff[i] = lambda_cstr[i];
    }
}

/**
 * Compute constraint residual max-norm for convergence check.
 *
 * @param C        Constraint matrix (empty = per_atom).
 * @param t        Target values.
 * @param gamma_I  Current gamma values.
 * @return         max |residual|.
 */
inline double compute_max_residual(
    const std::vector<std::vector<double>>& C,
    const std::vector<double>& t,
    const std::vector<double>& gamma_I)
{
    int nat = static_cast<int>(gamma_I.size());
    int m = static_cast<int>(C.size());
    double max_dev = 0.0;

    if (m > 0)
    {
        for (int a = 0; a < m; ++a)
        {
            double cv = 0.0;
            for (int i = 0; i < nat; ++i)
                cv += C[a][i] * gamma_I[i];
            max_dev = std::max(max_dev, std::abs(cv - t[a]));
        }
    }
    else if (!t.empty())
    {
        for (int i = 0; i < nat; ++i)
            max_dev = std::max(max_dev, std::abs(gamma_I[i] - t[i]));
    }
    return max_dev;
}

/**
 * Compute dp_escon = -Σ λ_i · γ_i  (constraint energy correction).
 *
 * @param lambda_eff  Effective per-atom lambda [n_atoms].
 * @param gamma_I     Per-atom gamma [n_atoms].
 * @return            dp_escon value (Ry).
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
