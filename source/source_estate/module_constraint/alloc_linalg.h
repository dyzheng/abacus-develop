#ifndef ALLOC_LINALG_H
#define ALLOC_LINALG_H

#include <vector>

namespace alloc
{

/**
 * @brief Symmetric eigendecomposition of a dense row-major real matrix.
 *
 * Computes A = V diag(w) V^T for a symmetric n x n matrix A.  V is
 * row-major with eigenvectors in its columns, and w is sorted ascending.
 * The implementation is a cyclic Jacobi sweep and intentionally avoids an
 * Eigen dependency (the allocation target dimension R is small).
 */
void sym_eig(const std::vector<double>& A, const int n,
             std::vector<double>& w, std::vector<double>& V);

/**
 * @brief Symmetric inverse square root with rank truncation.
 *
 * For a symmetric positive semi-definite matrix S, returns S^{-1/2} =
 * V diag(f) V^T with f_i = 1/sqrt(w_i) for eigenvalues above rank_tol and
 * f_i = 0 otherwise.  The caller is responsible for auditing the condition
 * number implied by the truncation.
 */
void sym_inv_sqrt(const std::vector<double>& S, const int n,
                  std::vector<double>& Sinv_root, const double rank_tol);

/**
 * @brief Fréchet derivative of the rank-truncated symmetric inverse square
 * root, d(S^{-1/2}) for a symmetric perturbation dS.
 *
 * With S = V diag(w) V^T and C = V^T dS V, the derivative is
 * V (F o C) V^T, where F_ij = (w_i^{-1/2} - w_j^{-1/2}) / (w_i - w_j)
 * off the diagonal and F_ii = -w_i^{-3/2}/2.  Directions dropped by the
 * same rank_tol as sym_inv_sqrt() contribute zero, so the analytic
 * derivative is the derivative of the actual truncated inverse root.
 */
void sym_inv_sqrt_derivative(const std::vector<double>& S,
                             const std::vector<double>& dS,
                             const int n,
                             std::vector<double>& dSinv_root,
                             const double rank_tol);

} // namespace alloc

#endif
