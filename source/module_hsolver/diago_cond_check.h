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
// Uses Cholesky decomposition to test if scc matrix is well-conditioned
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

    // 2. Try Cholesky decomposition (scc should be positive definite)
    // If it fails, matrix is not positive definite -> use CPU
    int info = 0;
    LapackConnector::potrf('U', nbase, scc_cpu.data(), nbase, info);

    if (info != 0) {
        // Cholesky failed - matrix is not positive definite or is ill-conditioned
        std::cout << "DIAGO_COND_CHECK: scc matrix Cholesky decomposition failed (info=" << info
                  << "), using CPU dngvd for numerical stability" << std::endl;
        return true;  // Use CPU
    }

    // 3. Estimate condition number from diagonal elements of Cholesky factor
    // For a positive definite matrix A = L*L^T, cond(A) >= (max(diag(L))/min(diag(L)))^2
    Real max_diag = 0.0;
    Real min_diag = 1e100;
    for (int i = 0; i < nbase; i++) {
        Real diag_val = std::abs(scc_cpu[i * nbase + i]);
        max_diag = std::max(max_diag, diag_val);
        min_diag = std::min(min_diag, diag_val);
    }

    double cond_estimate = (min_diag > 1e-16) ? (max_diag / min_diag) * (max_diag / min_diag) : 1e16;

    std::cout << "DIAGO_COND_CHECK: scc matrix condition number estimate = " << cond_estimate
              << " (threshold = " << threshold << ")" << std::endl;
    std::cout.flush();

    if (cond_estimate > threshold) {
        std::cout << "DIAGO_COND_CHECK: condition number exceeds threshold, using CPU dngvd"
                  << std::endl;
        std::cout.flush();
        return true;  // Use CPU
    }

    std::cout << "DIAGO_COND_CHECK: condition number OK, using GPU dngvd" << std::endl;
    std::cout.flush();
    return false;  // Use GPU
}

} // namespace hsolver

#endif // DIAGO_COND_CHECK_H
