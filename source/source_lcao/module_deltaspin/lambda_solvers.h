#ifndef LAMBDA_SOLVERS_H
#define LAMBDA_SOLVERS_H

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "source_base/vector3.h"
#include "source_hsolver/hsolver_lcao_subspace.h"

namespace spinconstrain
{

// Forward declaration - full definition comes from spin_constrain.h
template <typename TK>
class SpinConstrain;

/// @brief Solver type enum
enum class LambdaSolverType { BFGS, ChiGuided, Subspace, FDCG };

/// @brief Result of a lambda solver run
struct LambdaSolverResult
{
    bool converged = false;
    double rms_error = 0.0;
    double total_time = 0.0;
    int n_inner_steps = 0;
};

/// @brief Convert string to LambdaSolverType (case-insensitive)
LambdaSolverType lambda_solver_type_from_string(const std::string& s);

/// @brief Convert LambdaSolverType to string
std::string lambda_solver_type_to_string(LambdaSolverType t);

/// @brief Abstract base class for inner lambda loop solvers
///
/// Replaces the entire inner lambda loop of SpinConstrain.
/// Each concrete solver implements a different algorithm
/// (BFGS, Chi-Guided Newton, Subspace, etc.) for converging
/// the atomic magnetic moments to the target values.
class LambdaSolver
{
  public:
    virtual ~LambdaSolver() = default;

    /// @brief Run the full inner lambda loop
    /// @param outer_step Current outer SCF step
    virtual LambdaSolverResult run(int outer_step) = 0;

    virtual std::string name() const = 0;
    virtual LambdaSolverType type() const = 0;
};

/// @brief BFGS solver with line search (original algorithm)
///
/// Works for both PW and LCAO basis. Each inner step performs
/// a full SCF diagonalization via cal_mw_from_lambda.
/// Uses adaptive alpha_trial and gradient decay as convergence criteria.
class BFGSLambdaSolver : public LambdaSolver
{
  public:
    explicit BFGSLambdaSolver(SpinConstrain<std::complex<double>>& sc)
        : sc_(sc) {}

    LambdaSolverResult run(int outer_step) override;
    std::string name() const override { return "BFGS"; }
    LambdaSolverType type() const override { return LambdaSolverType::BFGS; }

  private:
    SpinConstrain<std::complex<double>>& sc_;
};

/// @brief Chi-guided Newton solver with full diagonalization
///
/// Designed for LCAO nspin=2. Phase 1: full diag to get Mi.
/// Phase 2: compute analytical chi = dM/dlambda.
/// Phase 3+: Newton step with full diag + secant chi update.
/// Guarantees SCF consistency but more expensive per step than BFGS.
class ChiGuidedLambdaSolver : public LambdaSolver
{
  public:
    explicit ChiGuidedLambdaSolver(SpinConstrain<std::complex<double>>& sc)
        : sc_(sc) {}

    LambdaSolverResult run(int outer_step) override;
    std::string name() const override { return "ChiGuided"; }
    LambdaSolverType type() const override { return LambdaSolverType::ChiGuided; }

  private:
    SpinConstrain<std::complex<double>>& sc_;
};

/// @brief Subspace diagonalization solver
///
/// Uses LCAO subspace diagonalization for fast lambda optimization.
/// Phase 1: full diag to build subspace cache (H0_sub, S_sub, P_I_sub).
/// Phase 2+: solve in nbands×nbands subspace instead of NLOCAL×NLOCAL.
/// Supports persistence across MD ionic steps via set_persistent().
class SubspaceLambdaSolver : public LambdaSolver
{
  public:
    explicit SubspaceLambdaSolver(SpinConstrain<std::complex<double>>& sc,
                                   hsolver::SubspaceMode mode = hsolver::SubspaceMode::Subspace,
                                   hsolver::SubspacePrecision precision = hsolver::SubspacePrecision::fp64);

    LambdaSolverResult run(int outer_step) override;
    std::string name() const override { return "Subspace"; }
    LambdaSolverType type() const override { return LambdaSolverType::Subspace; }

    /// @brief Set subspace diagonalization mode
    void set_mode(hsolver::SubspaceMode mode);

    /// @brief Set execution precision
    void set_precision(hsolver::SubspacePrecision prec);

    /// @brief Enable/disable persistent cache for MD ionic step reuse
    void set_persistent(bool persistent);

    /// @brief Check if subspace cache is valid
    bool has_subspace() const;

    /// @brief Clear subspace cache
    void clear_subspace();

    /// @brief Get the underlying HSolverLCAOSubspace for direct access
    hsolver::HSolverLCAOSubspace* get_solver() { return subspace_solver_.get(); }

  private:
    SpinConstrain<std::complex<double>>& sc_;
    std::unique_ptr<hsolver::HSolverLCAOSubspace> subspace_solver_;
};

/// @brief Finite-Difference Jacobian + Conjugate Gradient solver
///
/// Most reliable convergence method for spin-constrained DFT.
/// Step 0: evaluate M(lambda_0) — 1 diag
/// Step 1: FD Jacobian via simultaneous perturbation — 1 diag
/// Step 2+: CG iteration with Polak-Ribiere+ beta, analytical step size,
///          and secant Jacobian update — 1 diag per step
/// Works for both LCAO and PW, both nspin=2 and nspin=4.
class FDCGLambdaSolver : public LambdaSolver
{
  public:
    explicit FDCGLambdaSolver(SpinConstrain<std::complex<double>>& sc)
        : sc_(sc) {}

    LambdaSolverResult run(int outer_step) override;
    std::string name() const override { return "FDCG"; }
    LambdaSolverType type() const override { return LambdaSolverType::FDCG; }

  private:
    SpinConstrain<std::complex<double>>& sc_;
};

/// @brief Create a lambda solver instance for the given type
std::unique_ptr<LambdaSolver> create_lambda_solver(
    LambdaSolverType solver_type,
    SpinConstrain<std::complex<double>>& sc,
    hsolver::SubspaceMode subspace_mode = hsolver::SubspaceMode::Subspace,
    hsolver::SubspacePrecision subspace_precision = hsolver::SubspacePrecision::fp64);

} // namespace spinconstrain

#endif // LAMBDA_SOLVERS_H
