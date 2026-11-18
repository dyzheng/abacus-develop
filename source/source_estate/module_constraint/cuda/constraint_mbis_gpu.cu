#include "source_estate/module_constraint/constraint_mbis_gpu.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace constraint
{
namespace
{

constexpr int kThreads = 256;

__device__ inline void grid_fraction(const MbisDerivGrid& g, long long ir, double frac[3])
{
    const long long ij = static_cast<long long>(g.ny) * g.nplane;
    const long long i = ir / ij;
    const long long j = ir / g.nplane - i * g.ny;
    const long long k = ir % g.nplane + g.startz_current;
    frac[0] = static_cast<double>(i) / g.nx;
    frac[1] = static_cast<double>(j) / g.ny;
    frac[2] = static_cast<double>(k) / g.nz;
}

__device__ inline double wrap_frac(double v)
{
    return v - floor(v + 0.5);
}

// Minimum-image displacement (Bohr) from atom J toward grid point (r - R_J),
// matching WeightGrid::min_image_displacement (rfrac - taud, row-vector * cell).
__device__ inline void min_image_disp(const MbisDerivGrid& g, const double frac[3], const int J, double disp[3])
{
    const double fx = wrap_frac(frac[0] - g.taud[3 * J + 0]);
    const double fy = wrap_frac(frac[1] - g.taud[3 * J + 1]);
    const double fz = wrap_frac(frac[2] - g.taud[3 * J + 2]);
    disp[0] = fx * g.cell[0] + fy * g.cell[3] + fz * g.cell[6];
    disp[1] = fx * g.cell[1] + fy * g.cell[4] + fz * g.cell[7];
    disp[2] = fx * g.cell[2] + fy * g.cell[5] + fz * g.cell[8];
}

__device__ inline void block_reduce_atomic(double value, double* shared, double* target)
{
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(target, shared[0]);
    }
}

// Same as above but with the -dV factor folded in before the reduction.
__global__ void mbis_force_kernel_scaled(const MbisDerivGrid g, double* __restrict__ f)
{
    __shared__ double ssum[kThreads];
    const int J = blockIdx.y;
    const int d = blockIdx.z;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    double sum = 0.0;
    for (long long ir = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; ir < g.nrxx; ir += stride)
    {
        double frac[3];
        grid_fraction(g, ir, frac);
        double disp[3];
        min_image_disp(g, frac, J, disp);
        const double dist = sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]);
        if (dist < 1e-12)
        {
            continue;
        }
        const double pref = g.zeta[J] * g.w[static_cast<long long>(J) * g.nrxx + ir] * disp[d] / dist;
        double acc = 0.0;
        for (int alpha = 0; alpha < g.nalpha; ++alpha)
        {
            const double coef = static_cast<double>(g.in_alpha[static_cast<long long>(J) * g.nalpha + alpha])
                                - g.cw[static_cast<long long>(alpha) * g.nrxx + ir];
            acc += g.mu[alpha] * g.dens[static_cast<long long>(alpha) * g.nrxx + ir] * coef;
        }
        sum += acc * pref;
    }
    block_reduce_atomic(-g.dV * sum, ssum, &f[J * 3 + d]);
}

// One (alpha, a*3+b) pair per block slice.
__global__ void mbis_stress_kernel(const MbisDerivGrid g, double* __restrict__ s)
{
    __shared__ double ssum[kThreads];
    const int alpha = blockIdx.y;
    const int ab = blockIdx.z;
    const int a = ab / 3;
    const int b = ab % 3;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    double sum = 0.0;
    for (long long ir = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; ir < g.nrxx; ir += stride)
    {
        double frac[3];
        grid_fraction(g, ir, frac);
        const double dens = g.dens[static_cast<long long>(alpha) * g.nrxx + ir];
        for (int J = 0; J < g.nat; ++J)
        {
            double disp[3];
            min_image_disp(g, frac, J, disp);
            const double dist = sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]);
            if (dist < 1e-12)
            {
                continue;
            }
            const double coef = static_cast<double>(g.in_alpha[static_cast<long long>(J) * g.nalpha + alpha])
                                - g.cw[static_cast<long long>(alpha) * g.nrxx + ir];
            // dwe factor: coef * (-zeta_J w_J / d) * disp_a disp_b
            const double pref = -g.zeta[J] * g.w[static_cast<long long>(J) * g.nrxx + ir] / dist;
            sum += dens * coef * pref * disp[a] * disp[b];
        }
    }
    block_reduce_atomic(-g.mu[alpha] * g.dV * sum, ssum, &s[ab]);
}

struct DevBuf
{
    void* ptr = nullptr;
    ~DevBuf()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
    template <typename T>
    bool alloc(std::size_t n)
    {
        return cudaMalloc(&ptr, sizeof(T) * (n > 0 ? n : 1)) == cudaSuccess;
    }
    template <typename T>
    T* as()
    {
        return static_cast<T*>(ptr);
    }
};

bool upload(const void* src, void* dst, const std::size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool gpu_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

int reduce_blocks(const int nrxx)
{
    return std::min(1024, (nrxx + kThreads - 1) / kThreads);
}

} // namespace

bool mbis_force_gpu(const MbisDerivGrid& g, std::vector<double>& force)
{
    if (!gpu_available() || g.nat <= 0 || g.nalpha <= 0 || g.nrxx <= 0)
    {
        return false;
    }
    const std::size_t nat = g.nat;
    const std::size_t nalpha = g.nalpha;
    const std::size_t nrxx = g.nrxx;

    DevBuf d_taud, d_w, d_cw, d_dens, d_zeta, d_in, d_mu, d_f;
    if (!d_taud.alloc<double>(nat * 3) || !d_w.alloc<double>(nat * nrxx) || !d_cw.alloc<double>(nalpha * nrxx)
        || !d_dens.alloc<double>(nalpha * nrxx) || !d_zeta.alloc<double>(nat) || !d_in.alloc<int>(nat * nalpha)
        || !d_mu.alloc<double>(nalpha) || !d_f.alloc<double>(nat * 3))
    {
        return false;
    }
    if (!upload(g.taud, d_taud.ptr, sizeof(double) * nat * 3) || !upload(g.w, d_w.ptr, sizeof(double) * nat * nrxx)
        || !upload(g.cw, d_cw.ptr, sizeof(double) * nalpha * nrxx)
        || !upload(g.dens, d_dens.ptr, sizeof(double) * nalpha * nrxx)
        || !upload(g.zeta, d_zeta.ptr, sizeof(double) * nat)
        || !upload(g.in_alpha, d_in.ptr, sizeof(int) * nat * nalpha) || !upload(g.mu, d_mu.ptr, sizeof(double) * nalpha)
        || cudaMemset(d_f.ptr, 0, sizeof(double) * nat * 3) != cudaSuccess)
    {
        return false;
    }

    MbisDerivGrid dg = g;
    dg.taud = d_taud.as<double>();
    dg.w = d_w.as<double>();
    dg.cw = d_cw.as<double>();
    dg.dens = d_dens.as<double>();
    dg.zeta = d_zeta.as<double>();
    dg.in_alpha = d_in.as<int>();
    dg.mu = d_mu.as<double>();

    mbis_force_kernel_scaled<<<dim3(reduce_blocks(g.nrxx), g.nat, 3), kThreads>>>(dg, d_f.as<double>());
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    force.assign(nat * 3, 0.0);
    return cudaMemcpy(force.data(), d_f.ptr, sizeof(double) * nat * 3, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool mbis_stress_gpu(const MbisDerivGrid& g, std::vector<double>& stress)
{
    if (!gpu_available() || g.nat <= 0 || g.nalpha <= 0 || g.nrxx <= 0)
    {
        return false;
    }
    const std::size_t nat = g.nat;
    const std::size_t nalpha = g.nalpha;
    const std::size_t nrxx = g.nrxx;

    DevBuf d_taud, d_w, d_cw, d_dens, d_zeta, d_in, d_mu, d_s;
    if (!d_taud.alloc<double>(nat * 3) || !d_w.alloc<double>(nat * nrxx) || !d_cw.alloc<double>(nalpha * nrxx)
        || !d_dens.alloc<double>(nalpha * nrxx) || !d_zeta.alloc<double>(nat) || !d_in.alloc<int>(nat * nalpha)
        || !d_mu.alloc<double>(nalpha) || !d_s.alloc<double>(9))
    {
        return false;
    }
    if (!upload(g.taud, d_taud.ptr, sizeof(double) * nat * 3) || !upload(g.w, d_w.ptr, sizeof(double) * nat * nrxx)
        || !upload(g.cw, d_cw.ptr, sizeof(double) * nalpha * nrxx)
        || !upload(g.dens, d_dens.ptr, sizeof(double) * nalpha * nrxx)
        || !upload(g.zeta, d_zeta.ptr, sizeof(double) * nat)
        || !upload(g.in_alpha, d_in.ptr, sizeof(int) * nat * nalpha) || !upload(g.mu, d_mu.ptr, sizeof(double) * nalpha)
        || cudaMemset(d_s.ptr, 0, sizeof(double) * 9) != cudaSuccess)
    {
        return false;
    }

    MbisDerivGrid dg = g;
    dg.taud = d_taud.as<double>();
    dg.w = d_w.as<double>();
    dg.cw = d_cw.as<double>();
    dg.dens = d_dens.as<double>();
    dg.zeta = d_zeta.as<double>();
    dg.in_alpha = d_in.as<int>();
    dg.mu = d_mu.as<double>();

    mbis_stress_kernel<<<dim3(reduce_blocks(g.nrxx), g.nalpha, 9), kThreads>>>(dg, d_s.as<double>());
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    stress.assign(9, 0.0);
    return cudaMemcpy(stress.data(), d_s.ptr, sizeof(double) * 9, cudaMemcpyDeviceToHost) == cudaSuccess;
}

} // namespace constraint
