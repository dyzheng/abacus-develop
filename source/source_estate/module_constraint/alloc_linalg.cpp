#include "source_estate/module_constraint/alloc_linalg.h"

#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace alloc
{

void sym_eig(const std::vector<double>& A, const int n,
             std::vector<double>& w, std::vector<double>& V)
{
    w.assign(n, 0.0);
    V.assign(n * n, 0.0);
    // Branch: an empty matrix has no eigenvalues or eigenvectors.
    if (n == 0)
    {
        return;
    }

    std::vector<double> a = A;
    for (int i = 0; i < n; ++i)
    {
        V[i * n + i] = 1.0;
    }

    // Cyclic Jacobi sweeps.  Each sweep applies O(n^2) plane rotations to
    // the working matrix and accumulates the same rotations in V.
    for (int sweep = 0; sweep < 100; ++sweep)
    {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        // Branch: off-diagonal norm is at round-off level, so the working
        // matrix is diagonal to numerical precision.
        if (off <= 1e-30)
        {
            break;
        }

        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const double apq = a[p * n + q];
                // Branch: this pair is already diagonal to round-off; skip
                // its rotation to avoid dividing by an effectively zero apq.
                if (std::abs(apq) <= 1e-300)
                {
                    continue;
                }

                const double theta
                    = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t
                    = ((theta >= 0.0) ? 1.0 : -1.0)
                      / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                // Apply A <- J^T A J: rows first, then columns.  The same
                // rotation accumulates eigenvectors as V <- V J.
                for (int k = 0; k < n; ++k)
                {
                    const double akp = a[k * n + p];
                    const double akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double apk = a[p * n + k];
                    const double aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double vkp = V[k * n + p];
                    const double vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        w[i] = a[i * n + i];
    }

    // Sort eigenvalues ascending while permuting the eigenvector columns.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&w](const int lhs, const int rhs) { return w[lhs] < w[rhs]; });

    std::vector<double> w_sorted(n);
    std::vector<double> V_sorted(n * n);
    for (int i = 0; i < n; ++i)
    {
        w_sorted[i] = w[order[i]];
        for (int k = 0; k < n; ++k)
        {
            V_sorted[k * n + i] = V[k * n + order[i]];
        }
    }
    w.swap(w_sorted);
    V.swap(V_sorted);
}

void sym_inv_sqrt(const std::vector<double>& S, const int n,
                  std::vector<double>& Sinv_root, const double rank_tol)
{
    std::vector<double> w;
    std::vector<double> V;
    sym_eig(S, n, w, V);
    Sinv_root.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        // Branch: drop eigenvalues at or below the rank threshold.  This is
        // the only place numerical rank is changed, so callers can audit the
        // truncation through the returned condition number.
        const double f = (w[i] > rank_tol) ? 1.0 / std::sqrt(w[i]) : 0.0;
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                Sinv_root[row * n + col]
                    += V[row * n + i] * f * V[col * n + i];
            }
        }
    }
}

void sym_inv_sqrt_derivative(const std::vector<double>& S,
                             const std::vector<double>& dS,
                             const int n,
                             std::vector<double>& dSinv_root,
                             const double rank_tol)
{
    // Branch: the derivative of an empty matrix is empty.
    if (n <= 0)
    {
        dSinv_root.clear();
        return;
    }
    // Guard: the perturbation must describe the same symmetric space as S.
    if (static_cast<int>(S.size()) != n * n
        || static_cast<int>(dS.size()) != n * n)
    {
        throw std::invalid_argument("inverse-sqrt derivative dimensions differ");
    }

    std::vector<double> w;
    std::vector<double> V;
    sym_eig(S, n, w, V);

    // C = V^T dS V in the eigenbasis.  Symmetrizing dS here makes the
    // derivative insensitive to round-off from its producer.
    std::vector<double> C(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double value = 0.0;
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    const double ds = 0.5
                        * (dS[static_cast<size_t>(a) * n + b]
                           + dS[static_cast<size_t>(b) * n + a]);
                    value += V[static_cast<size_t>(a) * n + i] * ds
                             * V[static_cast<size_t>(b) * n + j];
                }
            }
            C[static_cast<size_t>(i) * n + j] = value;
        }
    }

    std::vector<double> F(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // Branch: truncated directions are outside the represented
            // subspace, exactly as in sym_inv_sqrt().
            if (w[i] <= rank_tol || w[j] <= rank_tol)
            {
                continue;
            }
            const double fi = 1.0 / std::sqrt(w[i]);
            const double fj = 1.0 / std::sqrt(w[j]);
            // Branch: a degenerate pair needs the limit of the divided
            // difference, not a numerically unstable ratio.
            if (i == j
                || std::abs(w[i] - w[j])
                       <= 1e-14 * (std::abs(w[i]) + std::abs(w[j])))
            {
                const double midpoint = 0.5 * (w[i] + w[j]);
                F[static_cast<size_t>(i) * n + j]
                    = -0.5 / (midpoint * std::sqrt(midpoint));
            }
            else
            {
                F[static_cast<size_t>(i) * n + j]
                    = (fi - fj) / (w[i] - w[j]);
            }
        }
    }

    dSinv_root.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double value = 0.0;
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    value += V[static_cast<size_t>(i) * n + a]
                             * F[static_cast<size_t>(a) * n + b]
                             * C[static_cast<size_t>(a) * n + b]
                             * V[static_cast<size_t>(j) * n + b];
                }
            }
            dSinv_root[static_cast<size_t>(i) * n + j] = value;
        }
    }
}

void herm_eig(const std::vector<std::complex<double>>& A,
              const int n,
              std::vector<double>& w,
              std::vector<std::complex<double>>& V)
{
    w.assign(n, 0.0);
    V.assign(static_cast<size_t>(n) * n, std::complex<double>(0.0, 0.0));
    // Branch: an empty matrix has no eigenpairs.
    if (n == 0)
    {
        return;
    }
    // Guard: the input must be a complete row-major square matrix.
    if (static_cast<int>(A.size()) != n * n)
    {
        throw std::invalid_argument("Hermitian eigendecomposition dimensions differ");
    }

    // LAPACK uses column-major storage.  Copying the Hermitian transpose is
    // equivalent to copying A itself after Hermitian symmetrization.
    std::vector<std::complex<double>> work_matrix(static_cast<size_t>(n) * n);
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            const std::complex<double> value
                = 0.5 * (A[static_cast<size_t>(row) * n + col]
                         + std::conj(A[static_cast<size_t>(col) * n + row]));
            work_matrix[static_cast<size_t>(col) * n + row] = value;
        }
    }

    const char jobz = 'V';
    const char uplo = 'U';
    int info = 0;
    int lwork = -1;
    std::complex<double> query(0.0, 0.0);
    std::vector<double> rwork(std::max(1, 3 * n - 2), 0.0);
    zheev_(&jobz, &uplo, &n, work_matrix.data(), &n, w.data(), &query,
           &lwork, rwork.data(), &info);
    // Guard: the workspace query itself must succeed.
    if (info != 0)
    {
        throw std::runtime_error("Hermitian eigendecomposition query failed");
    }
    lwork = std::max(1, static_cast<int>(query.real()));
    std::vector<std::complex<double>> work(lwork);
    zheev_(&jobz, &uplo, &n, work_matrix.data(), &n, w.data(), work.data(),
           &lwork, rwork.data(), &info);
    // Guard: a failed diagonalization cannot define a target space.
    if (info != 0)
    {
        throw std::runtime_error("Hermitian eigendecomposition failed");
    }

    // Eigenvector i is column i in the column-major LAPACK output.  Convert
    // it to column i of the row-major eigenvector matrix.
    for (int i = 0; i < n; ++i)
    {
        for (int row = 0; row < n; ++row)
        {
            V[static_cast<size_t>(row) * n + i]
                = work_matrix[static_cast<size_t>(i) * n + row];
        }
    }
}

void herm_inv_sqrt(const std::vector<std::complex<double>>& S,
                   const int n,
                   std::vector<std::complex<double>>& Sinv_root,
                   const double rank_tol)
{
    std::vector<double> w;
    std::vector<std::complex<double>> V;
    herm_eig(S, n, w, V);
    Sinv_root.assign(static_cast<size_t>(n) * n,
                     std::complex<double>(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        // Branch: drop directions below the same numerical-rank threshold
        // used by the real symmetric backend.
        const double f = (w[i] > rank_tol) ? 1.0 / std::sqrt(w[i]) : 0.0;
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                Sinv_root[static_cast<size_t>(row) * n + col]
                    += V[static_cast<size_t>(row) * n + i] * f
                       * std::conj(V[static_cast<size_t>(col) * n + i]);
            }
        }
    }
}

void herm_inv_sqrt_derivative(const std::vector<std::complex<double>>& S,
                              const std::vector<std::complex<double>>& dS,
                              const int n,
                              std::vector<std::complex<double>>& dSinv_root,
                              const double rank_tol)
{
    // Branch: the derivative of an empty matrix is empty.
    if (n <= 0)
    {
        dSinv_root.clear();
        return;
    }
    // Guard: S and dS must describe the same Hermitian matrix space.
    if (static_cast<int>(S.size()) != n * n
        || static_cast<int>(dS.size()) != n * n)
    {
        throw std::invalid_argument(
            "Hermitian inverse-sqrt derivative dimensions differ");
    }

    std::vector<double> w;
    std::vector<std::complex<double>> V;
    herm_eig(S, n, w, V);

    // C = V^H dS V in the eigenbasis.  Explicit Hermitian symmetrization
    // removes round-off skew from the projector-derivative producer.
    std::vector<std::complex<double>> C(static_cast<size_t>(n) * n,
                                        std::complex<double>(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::complex<double> value(0.0, 0.0);
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    const std::complex<double> ds
                        = 0.5
                          * (dS[static_cast<size_t>(a) * n + b]
                             + std::conj(dS[static_cast<size_t>(b) * n + a]));
                    value += std::conj(V[static_cast<size_t>(a) * n + i]) * ds
                             * V[static_cast<size_t>(b) * n + j];
                }
            }
            C[static_cast<size_t>(i) * n + j] = value;
        }
    }

    std::vector<double> F(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // Branch: truncated directions are outside the represented
            // subspace, exactly as in herm_inv_sqrt().
            if (w[i] <= rank_tol || w[j] <= rank_tol)
            {
                continue;
            }
            const double fi = 1.0 / std::sqrt(w[i]);
            const double fj = 1.0 / std::sqrt(w[j]);
            // Branch: a degenerate pair uses the analytic limit.
            if (i == j
                || std::abs(w[i] - w[j])
                       <= 1e-14 * (std::abs(w[i]) + std::abs(w[j])))
            {
                const double midpoint = 0.5 * (w[i] + w[j]);
                F[static_cast<size_t>(i) * n + j]
                    = -0.5 / (midpoint * std::sqrt(midpoint));
            }
            else
            {
                F[static_cast<size_t>(i) * n + j]
                    = (fi - fj) / (w[i] - w[j]);
            }
        }
    }

    dSinv_root.assign(static_cast<size_t>(n) * n,
                      std::complex<double>(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::complex<double> value(0.0, 0.0);
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    value += V[static_cast<size_t>(i) * n + a]
                             * F[static_cast<size_t>(a) * n + b]
                             * C[static_cast<size_t>(a) * n + b]
                             * std::conj(V[static_cast<size_t>(j) * n + b]);
                }
            }
            dSinv_root[static_cast<size_t>(i) * n + j] = value;
        }
    }
}

} // namespace alloc
