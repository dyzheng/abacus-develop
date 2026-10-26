#include "source_estate/module_constraint/alloc_linalg.h"

#include <algorithm>
#include <cmath>
#include <numeric>

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

} // namespace alloc
