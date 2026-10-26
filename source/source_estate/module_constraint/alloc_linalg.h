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

} // namespace alloc

#endif
