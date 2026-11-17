#ifndef MODULE_MBIS_ALM_GPU_H
#define MODULE_MBIS_ALM_GPU_H

/**
 * @file mbis_alm_gpu.h
 * @brief Device (CUDA/HIP) backend for the MBIS fixed point.
 *
 * The MBIS fixed point is dominated by the per-grid-point weight build:
 *   pro_A(x) = N_A zeta_A^3/(8 pi) sum_T exp(-zeta_A |x-R_A-T|),
 *   w_A(x)   = pro_A / sum_B pro_B,
 *   N_A = int w_A rho,   r_A = int |x-R_A| w_A rho,   zeta_A = 3 N_A / r_A.
 * This is embarrassingly parallel over the real-space grid, so the host loops
 * in mbis_alm.cpp delegate here when MbisConfig::use_gpu is set and a device
 * backend is compiled in.
 *
 * The declarations are always visible so callers do not need `#ifdef __CUDA`.
 * A CPU stub (mbis_alm_gpu_stub.cpp) provides the "unavailable" definitions
 * when neither CUDA nor ROCm is enabled.  The GPU implementations
 * (cuda/mbis_alm_gpu.cu, rocm/mbis_alm_gpu.hip.cu) provide the real ones.
 */

#include "mbis_alm.h"

namespace mbis_alm
{

/// True when a device backend is compiled in AND a usable GPU is visible.
bool mbis_gpu_available();

/**
 * @brief GPU implementation of compute_mbis().
 *
 * Returns false when the GPU path is unavailable or a device/runtime error
 * occurs; the caller then falls back to the CPU fixed point (never silently
 * returns a wrong state).
 */
bool compute_mbis_gpu(const std::vector<double>& rho,
                      const Grid& grid,
                      const std::vector<Atom>& atoms,
                      const MbisConfig& cfg,
                      MbisState& out);

} // namespace mbis_alm

#endif
