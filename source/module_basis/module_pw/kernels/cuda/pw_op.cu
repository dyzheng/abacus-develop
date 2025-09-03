#include "module_basis/module_pw/kernels/pw_op.h"

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <base/macros/macros.h>

namespace ModulePW {

#define THREADS_PER_BLOCK 256

template<class FPTYPE>
__global__ void set_3d_fft_box(
    const int npwk,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < npwk)
    {
        int xx = box_index[idx];
        out[xx] = in[idx];
    }
}

template<class FPTYPE>
__global__ void set_3d_fft_box_batch(
    const int npwk,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    const int ld_in,
    thrust::complex<FPTYPE>* out,
    const int ld_out,
    const int batchSize)
{
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < npwk && batch < batchSize)
    {
        int xx = box_index[idx];
        out[batch * ld_out + xx] = in[batch * ld_in + idx];
    }
}

template<class FPTYPE>
__global__ void set_recip_to_real_output(
    const int nrxx,
    const bool add,
    const FPTYPE factor,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nrxx) {return;}
    if(add) {
        out[idx] += factor * in[idx];
    }
    else {
        out[idx] = in[idx];
    }
}

template<class FPTYPE>
__global__ void set_recip_to_real_output_batch(
    const int nrxx,
    const bool add,
    const FPTYPE factor,
    const thrust::complex<FPTYPE>* in,
    const int ld_in,
    thrust::complex<FPTYPE>* out,
    const int ld_out,
    const int batchSize)
{
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nrxx || batch >= batchSize) {return;}
    if(add) {
        out[batch * ld_out + idx] += factor * in[batch * ld_in + idx];
    }
    else {
        out[batch * ld_out + idx] = in[batch * ld_in + idx];
    }
}

template<class FPTYPE>
__global__ void set_real_to_recip_output(
    const int npwk,
    const int nxyz,
    const bool add,
    const FPTYPE factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= npwk) {return;}
    if(add) {
        out[idx] += factor / nxyz * in[box_index[idx]];
    }
    else {
        out[idx] = in[box_index[idx]] / nxyz;
    }
}

template<class FPTYPE>
__global__ void set_real_to_recip_output_batch(
    const int npwk,
    const int nxyz,
    const bool add,
    const FPTYPE factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    const int ld_in,
    thrust::complex<FPTYPE>* out,
    const int ld_out,
    const int batchSize)
{
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= npwk || batch >= batchSize) {return;}
    if(add) {
        out[batch * ld_out + idx] += factor / nxyz * in[batch * ld_in + box_index[idx]];
    }
    else {
        out[batch * ld_out + idx] = in[batch * ld_in + box_index[idx]] / nxyz;
    }
}

template <typename FPTYPE>
void set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                    const int npwk,
                                                                    const int* box_index,
                                                                    const std::complex<FPTYPE>* in,
                                                                    std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_3d_fft_box<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_3d_fft_box_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                    const int npwk,
                                                                    const int* box_index,
                                                                    const std::complex<FPTYPE>* in,
                                                                    const int ld_in,
                                                                    std::complex<FPTYPE>* out,
                                                                    const int ld_out,
                                                                    const int batchSize)
{
    dim3 block((npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, batchSize);
    set_3d_fft_box_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in), ld_in,
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), ld_out, batchSize);

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                              const int nrxx,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_recip_to_real_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_recip_to_real_output_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                              const int nrxx,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              const int ld_in,
                                                                              std::complex<FPTYPE>* out,
                                                                              const int ld_out,
                                                                              const int batchSize)
{

    dim3 block((nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, batchSize);
    set_recip_to_real_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in), ld_in,
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), ld_out, batchSize);

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                              const int npwk,
                                                                              const int nxyz,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_real_to_recip_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        nxyz,
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_real_to_recip_output_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* /*dev*/,
                                                                              const int npwk,
                                                                              const int nxyz,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              const int ld_in,
                                                                              std::complex<FPTYPE>* out,
                                                                              const int ld_out,
                                                                              const int batchSize)
{
    dim3 block((npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, batchSize);
    set_real_to_recip_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        nxyz,
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in), ld_in,
        reinterpret_cast<thrust::complex<FPTYPE>*>(out), ld_out, batchSize);

    cudaCheckOnDebug();
}

template struct set_3d_fft_box_op<float, base_device::DEVICE_GPU>;
template struct set_3d_fft_box_batch_op<float, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_op<float, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_batch_op<float, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_op<float, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_batch_op<float, base_device::DEVICE_GPU>;
template struct set_3d_fft_box_op<double, base_device::DEVICE_GPU>;
template struct set_3d_fft_box_batch_op<double, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_op<double, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_batch_op<double, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_op<double, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_batch_op<double, base_device::DEVICE_GPU>;

}  // namespace ModulePW
