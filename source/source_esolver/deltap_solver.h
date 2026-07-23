#ifndef DELTAP_SOLVER_H
#define DELTAP_SOLVER_H
/**
 * @file deltap_solver.h
 * @brief Solver callback interface for DeltaP inner loop.
 *
 * The inner BFGS loop needs to apply trial lambda values and re-solve
 * the KS equation with frozen charge density.  This is basis-specific:
 *   LCAO: dp_op->set_lambda() + HSolverLCAO::solve(skip_charge=true)
 *   PW:   onsite_proj->set_lambda() + HSolverPW::solve(skip_charge=true)
 *
 * The callback abstracts this so the shared inner-loop logic works for both.
 */

#include <vector>
#include <functional>

namespace deltap_solver {

/**
 * Callback type: apply lambda and re-solve.
 *
 * @param lambda_eff   Effective per-atom lambda to apply [n_atoms].
 * @param skip_charge  If true, do not update charge density (frozen-density mode).
 * @return             0 on success, non-zero on failure.
 */
using ApplyLambdaFunc = std::function<int(
    const std::vector<double>& lambda_eff,
    bool skip_charge)>;

/**
 * Callback type: compute per-atom gamma after re-solve.
 *
 * @param gamma_I_out  Output: per-atom gamma values [n_atoms].
 * @return             0 on success.
 */
using ComputeGammaFunc = std::function<int(
    std::vector<double>& gamma_I_out)>;

/**
 * Callback type: get current effective per-atom lambda.
 *
 * @return  Current lambda values [n_atoms].
 */
using GetLambdaFunc = std::function<std::vector<double>()>;

/**
 * Callback type: get current per-atom constrain flags.
 *
 * @return  Constrain flags [n_atoms] (empty = all constrained).
 */
using GetConstrainFunc = std::function<std::vector<int>()>;

} // namespace deltap_solver
#endif // DELTAP_SOLVER_H
