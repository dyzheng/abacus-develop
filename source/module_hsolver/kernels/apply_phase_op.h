#ifndef MODULE_HSOLVER_APPLY_PHASE_OP_H
#define MODULE_HSOLVER_APPLY_PHASE_OP_H

#include "module_base/macros.h"
#include "module_base/module_device/types.h"

namespace hsolver {

// Apply phase factor exp(i*phase) to wavefunction in real space
// porter[ir] *= exp(i * 2π * dk · r)
template <typename T, typename Device>
struct apply_phase_op {
    using Real = typename GetTypeReal<T>::type;

    /// @brief Apply phase factor exp(i*2π*dk·r) to wavefunction in real space
    ///
    /// Input Parameters
    /// \param d : the type of computing device
    /// \param nrxx : number of real-space grid points
    /// \param nx, ny, nz : FFT grid dimensions
    /// \param startz : starting z index for this MPI rank
    /// \param dk_x, dk_y, dk_z : k-point difference in Cartesian coordinates
    /// \param porter : wavefunction in real space (input/output)
    ///
    /// Output Parameters
    /// \param porter : wavefunction with phase factor applied
    void operator()(const Device* d,
                   const int nrxx,
                   const int nx,
                   const int ny,
                   const int nz,
                   const int startz,
                   const Real dk_x,
                   const Real dk_y,
                   const Real dk_z,
                   T* porter);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

template <typename T>
struct apply_phase_op<T, base_device::DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;

    void operator()(const base_device::DEVICE_GPU* d,
                   const int nrxx,
                   const int nx,
                   const int ny,
                   const int nz,
                   const int startz,
                   const Real dk_x,
                   const Real dk_y,
                   const Real dk_z,
                   T* porter);
};

#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

} // namespace hsolver

#endif // MODULE_HSOLVER_APPLY_PHASE_OP_H
