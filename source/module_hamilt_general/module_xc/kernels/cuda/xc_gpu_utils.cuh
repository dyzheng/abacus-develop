// Shared GPU utility functions for XC kernels
// Used by both xc_functional_gpu_launcher.cu and xc_functional_gradcorr_gpu.cu
#ifndef XC_GPU_UTILS_CUH
#define XC_GPU_UTILS_CUH

#include <cuda_runtime.h>

namespace XC_GPU
{

// Double atomicAdd fallback for sm < 60
__device__ inline double atomicAdd_double(double* address, double val)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// Warp-level reduction using shuffle
__device__ inline double warp_reduce_sum(double val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory
// Reduces val_e and val_v across the block, atomically adds to global buffers
__device__ inline void block_reduce_add(double val_e, double val_v,
                                        double* etxc_buf, double* vtxc_buf)
{
    __shared__ double shared_e[8]; // max 8 warps per block (256 threads)
    __shared__ double shared_v[8];

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val_e = warp_reduce_sum(val_e);
    val_v = warp_reduce_sum(val_v);

    if (lane == 0) { shared_e[wid] = val_e; shared_v[wid] = val_v; }
    __syncthreads();

    int nwarps = blockDim.x / 32;
    if (threadIdx.x < (unsigned)nwarps)
    {
        val_e = shared_e[threadIdx.x];
        val_v = shared_v[threadIdx.x];
    }
    else
    {
        val_e = 0.0;
        val_v = 0.0;
    }

    if (wid == 0)
    {
        val_e = warp_reduce_sum(val_e);
        val_v = warp_reduce_sum(val_v);
        if (lane == 0)
        {
            atomicAdd_double(etxc_buf, val_e);
            atomicAdd_double(vtxc_buf, val_v);
        }
    }
}

} // namespace XC_GPU

#endif // XC_GPU_UTILS_CUH
