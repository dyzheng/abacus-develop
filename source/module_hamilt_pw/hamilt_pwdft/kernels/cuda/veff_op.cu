#include "module_hamilt_pw/hamilt_pwdft/kernels/veff_op.h"

#include <complex>

#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <base/macros/macros.h>

namespace hamilt {

#define THREADS_PER_BLOCK 256

template <typename FPTYPE>
__global__ void veff_pw(
    const int size,
    thrust::complex<FPTYPE>* out,
    const FPTYPE* in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    out[idx] *= in[idx];
}

template <typename FPTYPE>
__global__ void veff_pw_batch(
    const int size,
    thrust::complex<FPTYPE>* out,
    int ld_out,
    const FPTYPE* in,
    int batchSize)
{
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size || batch >= batchSize) {return;}
    out[batch * ld_out + idx] *= in[idx];
}

template <typename FPTYPE>
__global__ void veff_pw(
    const int size,
    thrust::complex<FPTYPE>* out,
    thrust::complex<FPTYPE>* out1,
    const FPTYPE* in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    thrust::complex<FPTYPE> sup =
        out[idx] * (in[0 * size + idx] + in[3 * size + idx])
            + out1[idx] * (in[1 * size + idx] - thrust::complex<FPTYPE>(0.0, 1.0) * in[2 * size + idx]);
    thrust::complex<FPTYPE> sdown =
        out1[idx] * (in[0 * size + idx] - in[3 * size + idx])
            + out[idx] * (in[1 * size + idx] + thrust::complex<FPTYPE>(0.0, 1.0) * in[2 * size + idx]);
    out[idx] = sup;
    out1[idx] = sdown;
}

template <typename FPTYPE>
__global__ void veff_pw_batch(
    const int size,
    thrust::complex<FPTYPE>* out,
    int ld_out,
    thrust::complex<FPTYPE>* out1,
    int ld_out1,
    const FPTYPE* in,
    int batchSize)
{
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size || batch >= batchSize) {return;}

    thrust::complex<FPTYPE> *out_batch = out + batch * ld_out;
    thrust::complex<FPTYPE> *out1_batch = out1 + batch * ld_out1;
    thrust::complex<FPTYPE> sup =
        out_batch[idx] * (in[0 * size + idx] + in[3 * size + idx])
            + out1_batch[idx] * (in[1 * size + idx] - thrust::complex<FPTYPE>(0.0, 1.0) * in[2 * size + idx]);
    thrust::complex<FPTYPE> sdown =
        out1_batch[idx] * (in[0 * size + idx] - in[3 * size + idx])
            + out_batch[idx] * (in[1 * size + idx] + thrust::complex<FPTYPE>(0.0, 1.0) * in[2 * size + idx]);
    out_batch[idx] = sup;
    out1_batch[idx] = sdown;
}

template <typename FPTYPE>
void veff_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                             const int& size,
                                                             std::complex<FPTYPE>* out,
                                                             const FPTYPE* in)
{
    const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    veff_pw<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        size, // control params
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), // array of data
        in); // array of data

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void veff_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                             const int& size,
                                                             std::complex<FPTYPE>* out,
                                                             std::complex<FPTYPE>* out1,
                                                             const FPTYPE** in)
{
    const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    veff_pw<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        size, // control params
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), // array of data
        reinterpret_cast<thrust::complex<FPTYPE>*>(out1), // array of data
        in[0]); // array of data

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void veff_pw_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                             const int& size,
                                                             std::complex<FPTYPE>* out,
                                                             int ld_out,
                                                             const FPTYPE* in,
                                                             int batchSize)
{
    dim3 block((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, batchSize);
    veff_pw_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        size, // control params
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), // array of data
        ld_out,
        in,// array of data
        batchSize
        );

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void veff_pw_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                             const int& size,
                                                             std::complex<FPTYPE>* out,
                                                             int ld_out,
                                                             std::complex<FPTYPE>* out1,
                                                             int ld_out1,
                                                             const FPTYPE** in,
                                                             int batchSize)
{
    dim3 block((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, batchSize);
    veff_pw_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        size, // control params
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), // array of data
        ld_out,
        reinterpret_cast<thrust::complex<FPTYPE>*>(out1), // array of data
        ld_out1,
        in[0],// array of data
        batchSize
        );

    cudaCheckOnDebug();
}

template struct veff_pw_op<float, base_device::DEVICE_GPU>;
template struct veff_pw_op<double, base_device::DEVICE_GPU>;
template struct veff_pw_batch_op<float, base_device::DEVICE_GPU>;
template struct veff_pw_batch_op<double, base_device::DEVICE_GPU>;

}  // namespace hamilt