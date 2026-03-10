#include "../apply_phase_op.h"
#include "module_base/constants.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace hsolver {

// CUDA kernel to apply phase factor for float
__global__ void apply_phase_kernel_float(
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const float dk_x,
    const float dk_y,
    const float dk_z,
    cuFloatComplex* porter)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrxx) return;

    // Calculate grid indices from linear index
    const int iz = ir / (nx * ny);
    const int ixy = ir % (nx * ny);
    const int iy = ixy / nx;
    const int ix = ixy % nx;

    // Calculate fractional coordinates
    const float fx = static_cast<float>(ix) / nx;
    const float fy = static_cast<float>(iy) / ny;
    const float fz = static_cast<float>(iz + startz) / nz;

    // dk is in Cartesian coordinates (2pi/a units)
    // phase = 2π * dk · f where f is fractional coordinate
    float phase = dk_x * fx + dk_y * fy + dk_z * fz;
    phase *= ModuleBase::TWO_PI;

    // Apply phase factor: exp(i*phase)
    float cos_phase, sin_phase;
    sincosf(phase, &sin_phase, &cos_phase);

    cuFloatComplex phase_factor = make_cuFloatComplex(cos_phase, sin_phase);
    porter[ir] = cuCmulf(porter[ir], phase_factor);
}

// CUDA kernel to apply phase factor for double
__global__ void apply_phase_kernel_double(
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const double dk_x,
    const double dk_y,
    const double dk_z,
    cuDoubleComplex* porter)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrxx) return;

    // Calculate grid indices from linear index
    const int iz = ir / (nx * ny);
    const int ixy = ir % (nx * ny);
    const int iy = ixy / nx;
    const int ix = ixy % nx;

    // Calculate fractional coordinates
    const double fx = static_cast<double>(ix) / nx;
    const double fy = static_cast<double>(iy) / ny;
    const double fz = static_cast<double>(iz + startz) / nz;

    // dk is in Cartesian coordinates (2pi/a units)
    // phase = 2π * dk · f where f is fractional coordinate
    double phase = dk_x * fx + dk_y * fy + dk_z * fz;
    phase *= ModuleBase::TWO_PI;

    // Apply phase factor: exp(i*phase)
    double cos_phase, sin_phase;
    sincos(phase, &sin_phase, &cos_phase);

    cuDoubleComplex phase_factor = make_cuDoubleComplex(cos_phase, sin_phase);
    porter[ir] = cuCmul(porter[ir], phase_factor);
}

// GPU implementation for float
template <>
void apply_phase_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* d,
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const float dk_x,
    const float dk_y,
    const float dk_z,
    std::complex<float>* porter)
{
    const int block_size = 256;
    const int grid_size = (nrxx + block_size - 1) / block_size;

    apply_phase_kernel_float<<<grid_size, block_size>>>(
        nrxx, nx, ny, nz, startz,
        dk_x, dk_y, dk_z,
        reinterpret_cast<cuFloatComplex*>(porter));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in apply_phase_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

// GPU implementation for double
template <>
void apply_phase_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* d,
    const int nrxx,
    const int nx,
    const int ny,
    const int nz,
    const int startz,
    const double dk_x,
    const double dk_y,
    const double dk_z,
    std::complex<double>* porter)
{
    const int block_size = 256;
    const int grid_size = (nrxx + block_size - 1) / block_size;

    apply_phase_kernel_double<<<grid_size, block_size>>>(
        nrxx, nx, ny, nz, startz,
        dk_x, dk_y, dk_z,
        reinterpret_cast<cuDoubleComplex*>(porter));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in apply_phase_kernel_double: %s\n", cudaGetErrorString(err));
    }
}

} // namespace hsolver
