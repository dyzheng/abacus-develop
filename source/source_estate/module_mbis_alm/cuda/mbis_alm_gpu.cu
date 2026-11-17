#include "source_estate/module_mbis_alm/mbis_alm_gpu.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace mbis_alm
{
namespace
{

constexpr int kThreads = 256;
constexpr double kInv8Pi = 1.0 / (8.0 * 3.14159265358979323846);

// Plain-old-data grid description passed by value to the kernels (no host
// pointer indirection, so the device never touches std::vector internals).
struct GpuGrid
{
    long long n0, n1, n2;
    double ox, oy, oz;
    double dv00, dv01, dv02;
    double dv10, dv11, dv12;
    double dv20, dv21, dv22;
};

__device__ inline void grid_point(const GpuGrid& g, long long ig, double& x, double& y, double& z)
{
    const long long n12 = g.n1 * g.n2;
    const long long ix = ig / n12;
    const long long rem = ig - ix * n12;
    const long long iy = rem / g.n2;
    const long long iz = rem - iy * g.n2;
    x = g.ox + static_cast<double>(ix) * g.dv00 + static_cast<double>(iy) * g.dv10 + static_cast<double>(iz) * g.dv20;
    y = g.oy + static_cast<double>(ix) * g.dv01 + static_cast<double>(iy) * g.dv11 + static_cast<double>(iz) * g.dv21;
    z = g.oz + static_cast<double>(ix) * g.dv02 + static_cast<double>(iy) * g.dv12 + static_cast<double>(iz) * g.dv22;
}

// Pass 1 (fused): weight w_A = pro_A/P per atom and the partition function P.
// One thread owns one grid point, so it can form P locally and immediately
// normalise every atom's pro-density without a second global read/write pass.
// The atom/image loops keep the CPU reference order, so the resulting sums are
// bit-comparable up to round-off.
__global__ void pro_weights_kernel(const GpuGrid g,
                                   const long long nxyz,
                                   const int nat,
                                   const int nimg,
                                   const double* __restrict__ Rx,
                                   const double* __restrict__ Ry,
                                   const double* __restrict__ Rz,
                                   const double* __restrict__ Npop,
                                   const double* __restrict__ zeta,
                                   const double* __restrict__ Tx,
                                   const double* __restrict__ Ty,
                                   const double* __restrict__ Tz,
                                   const double cutoff,
                                   double* __restrict__ w,
                                   double* __restrict__ P,
                                   double* __restrict__ rmin_out)
{
    const long long ig = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ig >= nxyz)
    {
        return;
    }
    double px, py, pz;
    grid_point(g, ig, px, py, pz);
    double psum = 0.0;
    for (int a = 0; a < nat; ++a)
    {
        const double z = zeta[a];
        const double rx = Rx[a], ry = Ry[a], rz = Rz[a];
        double acc = 0.0;
        double rmin = 1e300;
        for (int t = 0; t < nimg; ++t)
        {
            const double dx = px - (rx + Tx[t]);
            const double dy = py - (ry + Ty[t]);
            const double dz = pz - (rz + Tz[t]);
            const double r = sqrt(dx * dx + dy * dy + dz * dz);
            // Branch: only images inside the radial cutoff contribute, exactly
            // like the CPU pro_density_at.
            if (r <= cutoff)
            {
                acc += exp(-z * r);
            }
            // Mean-radius distance uses the minimum over ALL images (no
            // cutoff), matching the CPU `r` grid.  It is free here because the
            // distance is already computed for the pro sum.
            rmin = (r < rmin) ? r : rmin;
        }
        const long long idx = static_cast<long long>(a) * nxyz + ig;
        const double v = Npop[a] * z * z * z * kInv8Pi * acc;
        w[idx] = v;
        rmin_out[idx] = rmin;
        psum += v;
    }
    P[ig] = psum;
    // Branch: empty partition -> all weights zero; otherwise normalise in
    // place (same guard as the CPU `P > 1e-300 ? pro/P : 0`).
    if (psum > 1e-300)
    {
        for (int a = 0; a < nat; ++a)
        {
            w[static_cast<long long>(a) * nxyz + ig] /= psum;
        }
    }
    else
    {
        for (int a = 0; a < nat; ++a)
        {
            w[static_cast<long long>(a) * nxyz + ig] = 0.0;
        }
    }
}

// Pass 2: per-atom reductions N_A = int w_A rho dV and
// r_A = int |x-R_A|_min w_A rho dV.  The minimum-image distance is produced by
// the fused pro/weight pass, so this kernel is pure memory + FLOPs.  gridDim.y
// selects the atom, so there are no contended global atomics between atoms.
__global__ void reduce_atom_kernel(const long long nxyz,
                                   const double* __restrict__ w,
                                   const double* __restrict__ rmin,
                                   const double* __restrict__ rho,
                                   const double dV,
                                   double* __restrict__ Nout,
                                   double* __restrict__ MRout)
{
    __shared__ double sN[kThreads];
    __shared__ double sMR[kThreads];
    const int tid = threadIdx.x;
    const int atom = blockIdx.y; // one grid slice per atom
    const long long base = static_cast<long long>(atom) * nxyz;
    double nsum = 0.0;
    double mrsum = 0.0;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long ig = static_cast<long long>(blockIdx.x) * blockDim.x + tid; ig < nxyz; ig += stride)
    {
        const double wa = w[base + ig];
        // Branch: a zero weight contributes nothing to either reduction.
        if (wa == 0.0)
        {
            continue;
        }
        const double wd = wa * rho[ig] * dV;
        nsum += wd;
        mrsum += rmin[base + ig] * wd;
    }
    sN[tid] = nsum;
    sMR[tid] = mrsum;
    __syncthreads();
    // Tree reduction of the two per-block partial sums.
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sN[tid] += sN[tid + s];
            sMR[tid] += sMR[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(&Nout[atom], sN[0]);
        atomicAdd(&MRout[atom], sMR[0]);
    }
}

// KL diagnostic D = sum_A int pro_A [ln pro_A - ln(w_A rho)] dV.  Using
// pro_A = w_A P avoids storing pro separately from the final weights.
__global__ void kl_kernel(const long long nxyz,
                          const int nat,
                          const double* __restrict__ w,
                          const double* __restrict__ P,
                          const double* __restrict__ rho,
                          const double dV,
                          double* __restrict__ Dout)
{
    __shared__ double sD[kThreads];
    const int tid = threadIdx.x;
    double dsum = 0.0;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long ig = static_cast<long long>(blockIdx.x) * blockDim.x + tid; ig < nxyz; ig += stride)
    {
        const double p = P[ig];
        for (int a = 0; a < nat; ++a)
        {
            const double wa = w[static_cast<long long>(a) * nxyz + ig];
            const double pa = wa * p;
            // Branch: keep the same degenerate-density guards as the CPU loop.
            if (pa <= 1e-30)
            {
                continue;
            }
            const double rhoa = wa * rho[ig];
            if (rhoa > 1e-300)
            {
                dsum += pa * (log(pa) - log(rhoa)) * dV;
            }
        }
    }
    sD[tid] = dsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sD[tid] += sD[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(Dout, sD[0]);
    }
}

// RAII device buffer so every early `return false` path frees cleanly.
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

bool gpu_ok(cudaError_t err)
{
    return err == cudaSuccess;
}

} // namespace

bool mbis_gpu_available()
{
    // Query on demand (cheap): a cached failure would stick if this ran before
    // DeviceContext initialised the process/device binding.
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

bool compute_mbis_gpu(const std::vector<double>& rho,
                      const Grid& grid,
                      const std::vector<Atom>& atoms,
                      const MbisConfig& cfg,
                      MbisState& out)
{
    if (!mbis_gpu_available())
    {
        return false;
    }
    // Compute the grid size / cell volume locally: the device object library
    // must not depend on Grid's out-of-line member functions (CPU-only tests
    // link it without mbis_alm.cpp).
    const long long nxyz
        = static_cast<long long>(grid.n[0]) * static_cast<long long>(grid.n[1]) * static_cast<long long>(grid.n[2]);
    const int nat = static_cast<int>(atoms.size());
    if (nxyz <= 0 || nat <= 0 || rho.size() != static_cast<std::size_t>(nxyz))
    {
        return false;
    }

    // Periodic image translations, identical to image_translations() on the
    // host (inclusive shell of radius `images` along each grid direction).
    const Vec3 c0 = static_cast<double>(grid.n[0]) * grid.dv[0];
    const Vec3 c1 = static_cast<double>(grid.n[1]) * grid.dv[1];
    const Vec3 c2 = static_cast<double>(grid.n[2]) * grid.dv[2];
    const int R = std::max(0, cfg.images);
    std::vector<double> Tx, Ty, Tz;
    Tx.reserve(static_cast<std::size_t>(2 * R + 1) * (2 * R + 1) * (2 * R + 1));
    Ty.reserve(Tx.capacity());
    Tz.reserve(Tx.capacity());
    for (int lx = -R; lx <= R; ++lx)
        for (int ly = -R; ly <= R; ++ly)
            for (int lz = -R; lz <= R; ++lz)
            {
                const Vec3 t
                    = static_cast<double>(lx) * c0 + static_cast<double>(ly) * c1 + static_cast<double>(lz) * c2;
                Tx.push_back(t.x);
                Ty.push_back(t.y);
                Tz.push_back(t.z);
            }
    const int nimg = static_cast<int>(Tx.size());

    // Host-side MBIS state (same warm-start rules as the CPU implementation).
    std::vector<double> N(nat), zeta(nat);
    const bool have_warm
        = (static_cast<int>(cfg.warm_N.size()) == nat && static_cast<int>(cfg.warm_zeta.size()) == nat);
    for (int a = 0; a < nat; ++a)
    {
        N[a] = have_warm ? cfg.warm_N[a] : atoms[a].Zval;
        zeta[a] = have_warm ? cfg.warm_zeta[a] : cfg.init_zeta;
    }

    std::vector<double> Rx(nat), Ry(nat), Rz(nat);
    for (int a = 0; a < nat; ++a)
    {
        Rx[a] = atoms[a].R.x;
        Ry[a] = atoms[a].R.y;
        Rz[a] = atoms[a].R.z;
    }

    GpuGrid g;
    g.n0 = grid.n[0];
    g.n1 = grid.n[1];
    g.n2 = grid.n[2];
    g.ox = grid.origin.x;
    g.oy = grid.origin.y;
    g.oz = grid.origin.z;
    g.dv00 = grid.dv[0].x;
    g.dv01 = grid.dv[0].y;
    g.dv02 = grid.dv[0].z;
    g.dv10 = grid.dv[1].x;
    g.dv11 = grid.dv[1].y;
    g.dv12 = grid.dv[1].z;
    g.dv20 = grid.dv[2].x;
    g.dv21 = grid.dv[2].y;
    g.dv22 = grid.dv[2].z;

    const Vec3& a0 = grid.dv[0];
    const Vec3& a1 = grid.dv[1];
    const Vec3& a2 = grid.dv[2];
    const double dV = std::abs(a0.x * (a1.y * a2.z - a1.z * a2.y) - a0.y * (a1.x * a2.z - a1.z * a2.x)
                               + a0.z * (a1.x * a2.y - a1.y * a2.x));
    const std::size_t nw = static_cast<std::size_t>(nat) * static_cast<std::size_t>(nxyz);

    DevBuf d_rho, d_pro, d_rmin, d_P, d_Rx, d_Ry, d_Rz, d_N, d_zeta, d_Tx, d_Ty, d_Tz, d_red, d_D;

    if (!d_rho.alloc<double>(nxyz) || !d_pro.alloc<double>(nw) || !d_rmin.alloc<double>(nw) || !d_P.alloc<double>(nxyz)
        || !d_Rx.alloc<double>(nat) || !d_Ry.alloc<double>(nat) || !d_Rz.alloc<double>(nat) || !d_N.alloc<double>(nat)
        || !d_zeta.alloc<double>(nat) || !d_Tx.alloc<double>(nimg) || !d_Ty.alloc<double>(nimg)
        || !d_Tz.alloc<double>(nimg) || !d_red.alloc<double>(2 * nat) || !d_D.alloc<double>(1))
    {
        return false;
    }

    if (!gpu_ok(cudaMemcpy(d_rho.as<double>(), rho.data(), sizeof(double) * nxyz, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Rx.as<double>(), Rx.data(), sizeof(double) * nat, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Ry.as<double>(), Ry.data(), sizeof(double) * nat, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Rz.as<double>(), Rz.data(), sizeof(double) * nat, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Tx.as<double>(), Tx.data(), sizeof(double) * nimg, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Ty.as<double>(), Ty.data(), sizeof(double) * nimg, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_Tz.as<double>(), Tz.data(), sizeof(double) * nimg, cudaMemcpyHostToDevice)))
    {
        return false;
    }

    const int pro_blocks = static_cast<int>((nxyz + kThreads - 1) / kThreads);
    const int red_blocks = std::min(1024, static_cast<int>((nxyz + kThreads - 1) / kThreads));
    std::vector<double> red(2 * nat, 0.0);

    for (int it = 0; it < cfg.max_iter; ++it)
    {
        if (!gpu_ok(cudaMemcpy(d_N.as<double>(), N.data(), sizeof(double) * nat, cudaMemcpyHostToDevice))
            || !gpu_ok(cudaMemcpy(d_zeta.as<double>(), zeta.data(), sizeof(double) * nat, cudaMemcpyHostToDevice)))
        {
            return false;
        }
        pro_weights_kernel<<<pro_blocks, kThreads>>>(g,
                                                     nxyz,
                                                     nat,
                                                     nimg,
                                                     d_Rx.as<double>(),
                                                     d_Ry.as<double>(),
                                                     d_Rz.as<double>(),
                                                     d_N.as<double>(),
                                                     d_zeta.as<double>(),
                                                     d_Tx.as<double>(),
                                                     d_Ty.as<double>(),
                                                     d_Tz.as<double>(),
                                                     cfg.cutoff,
                                                     d_pro.as<double>(),
                                                     d_P.as<double>(),
                                                     d_rmin.as<double>());
        if (!gpu_ok(cudaGetLastError()))
        {
            return false;
        }
        if (!gpu_ok(cudaMemset(d_red.ptr, 0, sizeof(double) * 2 * nat)))
        {
            return false;
        }
        // gridDim.y selects the atom, so every atom reduces independently with
        // one atomicAdd per block (no cross-atom contention).
        reduce_atom_kernel<<<dim3(red_blocks, nat), kThreads>>>(nxyz,
                                                                d_pro.as<double>(),
                                                                d_rmin.as<double>(),
                                                                d_rho.as<double>(),
                                                                dV,
                                                                d_red.as<double>(),
                                                                d_red.as<double>() + nat);
        if (!gpu_ok(cudaGetLastError()))
        {
            return false;
        }
        if (!gpu_ok(cudaMemcpy(red.data(), d_red.ptr, sizeof(double) * 2 * nat, cudaMemcpyDeviceToHost)))
        {
            return false;
        }

        double res = 0.0;
        std::vector<double> Nn(nat), zn(nat);
        for (int a = 0; a < nat; ++a)
        {
            const double na = red[a];
            const double mr = red[nat + a];
            Nn[a] = na;
            zn[a] = (na > 1e-12 && mr > 1e-12) ? 3.0 * na / mr : zeta[a];
            zn[a] = std::max(zn[a], 1e-2);
            res = std::max(res, std::abs(na - N[a]));
        }
        for (int a = 0; a < nat; ++a)
        {
            N[a] = (1.0 - cfg.damp) * N[a] + cfg.damp * Nn[a];
            zeta[a] = (1.0 - cfg.damp) * zeta[a] + cfg.damp * zn[a];
        }
        if (res < cfg.tol)
        {
            break;
        }
    }

    // Final fill with the converged (damped) parameters, matching the CPU path
    // which calls fill_weights once more after the loop.
    if (!gpu_ok(cudaMemcpy(d_N.as<double>(), N.data(), sizeof(double) * nat, cudaMemcpyHostToDevice))
        || !gpu_ok(cudaMemcpy(d_zeta.as<double>(), zeta.data(), sizeof(double) * nat, cudaMemcpyHostToDevice)))
    {
        return false;
    }
    pro_weights_kernel<<<pro_blocks, kThreads>>>(g,
                                                 nxyz,
                                                 nat,
                                                 nimg,
                                                 d_Rx.as<double>(),
                                                 d_Ry.as<double>(),
                                                 d_Rz.as<double>(),
                                                 d_N.as<double>(),
                                                 d_zeta.as<double>(),
                                                 d_Tx.as<double>(),
                                                 d_Ty.as<double>(),
                                                 d_Tz.as<double>(),
                                                 cfg.cutoff,
                                                 d_pro.as<double>(),
                                                 d_P.as<double>(),
                                                 d_rmin.as<double>());
    if (!gpu_ok(cudaGetLastError()))
    {
        return false;
    }

    std::vector<double> w_flat(nw, 0.0);
    if (!gpu_ok(cudaMemcpy(w_flat.data(), d_pro.ptr, sizeof(double) * nw, cudaMemcpyDeviceToHost)))
    {
        return false;
    }

    // KL diagnostic on the device grid.
    if (!gpu_ok(cudaMemset(d_D.ptr, 0, sizeof(double))))
    {
        return false;
    }
    kl_kernel<<<red_blocks, kThreads>>>(nxyz,
                                        nat,
                                        d_pro.as<double>(),
                                        d_P.as<double>(),
                                        d_rho.as<double>(),
                                        dV,
                                        d_D.as<double>());
    double D = 0.0;
    if (!gpu_ok(cudaGetLastError()) || !gpu_ok(cudaMemcpy(&D, d_D.ptr, sizeof(double), cudaMemcpyDeviceToHost)))
    {
        return false;
    }

    out.N = N;
    out.zeta = zeta;
    out.D = D;
    out.w.assign(nat, std::vector<double>(static_cast<std::size_t>(nxyz), 0.0));
    for (int a = 0; a < nat; ++a)
        for (long long ig = 0; ig < nxyz; ++ig)
            out.w[a][static_cast<std::size_t>(ig)] = w_flat[static_cast<std::size_t>(a) * nxyz + ig];
    return true;
}

} // namespace mbis_alm
