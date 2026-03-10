#ifndef DIAGO_COND_CHECK_H
#define DIAGO_COND_CHECK_H

#include "module_base/macros.h"
#include "module_base/lapack_connector.h"
#include <vector>
#include <iostream>
#include <algorithm>

#if defined(__CUDA) || defined(__ROCM)
#include <cuda_runtime.h>
#endif

namespace hsolver
{

// Helper function to check matrix condition number
// Uses diagonal ratio to estimate if scc matrix is well-conditioned
// More robust than Cholesky for near-singular matrices
template <typename T>
bool check_matrix_condition_number(const T* scc_gpu,
                                   const int nbase,
                                   const double threshold = 1e12)
{
    using Real = typename GetTypeReal<T>::type;

    // 1. D2H copy scc matrix
    std::vector<T> scc_cpu(nbase * nbase);
#if defined(__CUDA) || defined(__ROCM)
    cudaMemcpy(scc_cpu.data(), scc_gpu,
               sizeof(T) * nbase * nbase,
               cudaMemcpyDeviceToHost);
#else
    // If not GPU, scc_gpu is already on CPU
    std::copy(scc_gpu, scc_gpu + nbase * nbase, scc_cpu.begin());
#endif

    // 2. Check diagonal elements for near-zero or negative values
    Real max_diag = 0.0;
    Real min_diag = 1e100;
    bool has_negative = false;
    bool has_zero = false;

    for (int i = 0; i < nbase; i++) {
        Real diag_val = std::real(scc_cpu[i * nbase + i]);
        if (diag_val <= 0.0) {
            if (diag_val < -1e-10) {
                has_negative = true;
            } else {
                has_zero = true;
            }
        }
        Real abs_diag = std::abs(diag_val);
        max_diag = std::max(max_diag, abs_diag);
        if (abs_diag > 1e-16) {
            min_diag = std::min(min_diag, abs_diag);
        }
    }

    // 3. Estimate condition number from diagonal ratio
    // For overlap matrix S, cond(S) is roughly bounded by max_diag/min_diag
    double diag_ratio = (min_diag > 1e-16) ? (max_diag / min_diag) : 1e16;

    // 4. Decision: only fall back to CPU for severe numerical issues
    // GPU cusolver is more robust than Cholesky for near-singular matrices
    if (has_negative) {
        std::cout << "WARNING: scc matrix has negative diagonal, falling back to CPU dngvd for numerical stability" << std::endl;
        return true;
    }

    if (diag_ratio > threshold) {
        std::cout << "WARNING: scc matrix diagonal ratio " << diag_ratio
                  << " exceeds threshold " << threshold
                  << ", falling back to CPU dngvd for numerical stability" << std::endl;
        return true;
    }

    return false;  // Use GPU
}

} // namespace hsolver

#endif // DIAGO_COND_CHECK_H
