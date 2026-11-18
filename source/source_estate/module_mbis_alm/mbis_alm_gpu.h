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

/**
 * @brief One distributed MBIS fixed-point step on a rank-local grid slice.
 *
 * The MBIS per-point weights are local; only the populations N_A and mean
 * radii int|r-R_A| w_A rho need a global sum.  This entry computes the local
 * weights for the given (N,zeta) and the local N/MR partial sums; the caller
 * allreduces them, damps, and iterates.  `nplane`/`startz` describe the
 * rank-local z-plane block of the global grid (ABACUS PW layout).
 *
 * @param want_weights  false -> skip the device-to-host weight copy (only the
 *                      final call in the distributed loop needs the weights).
 * @return false on unavailable backend / device error; the caller must abort
 *         rather than silently continue with an inconsistent state.
 */
bool mbis_gpu_local_step(const std::vector<double>& rho_local,
                         const Grid& grid,
                         const int nplane,
                         const int startz,
                         const std::vector<Atom>& atoms,
                         const MbisConfig& cfg,
                         const std::vector<double>& N,
                         const std::vector<double>& zeta,
                         const bool want_weights,
                         std::vector<std::vector<double>>& w,
                         std::vector<double>& Nout,
                         std::vector<double>& MRout);

/**
 * @brief Persistent device workspace for the distributed MBIS fixed point.
 *
 * The rank-local density, atom centres and periodic-image translations do not
 * change during one fixed-point solve.  Creating the workspace uploads them
 * once; each step() only uploads the current (N,zeta) and launches the
 * pro/weight/reduction kernels.  This removes the per-iteration cudaMalloc +
 * H2D traffic from the MPI distributed loop.
 *
 * The class is a pimpl so the header stays free of CUDA/HIP types.  The device
 * implementation lives in cuda|rocm/mbis_alm_gpu.*; the CPU stub always
 * reports failure.
 */
class MbisGpuLocalWorkspace
{
  public:
    MbisGpuLocalWorkspace();
    ~MbisGpuLocalWorkspace();
    MbisGpuLocalWorkspace(const MbisGpuLocalWorkspace&) = delete;
    MbisGpuLocalWorkspace& operator=(const MbisGpuLocalWorkspace&) = delete;

    bool init(const std::vector<double>& rho_local,
              const Grid& grid,
              const int nplane,
              const int startz,
              const std::vector<Atom>& atoms,
              const MbisConfig& cfg);

    bool step(const std::vector<double>& N,
              const std::vector<double>& zeta,
              const bool want_weights,
              std::vector<std::vector<double>>& w,
              std::vector<double>& Nout,
              std::vector<double>& MRout);

    bool ready() const;

  private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace mbis_alm

#endif
