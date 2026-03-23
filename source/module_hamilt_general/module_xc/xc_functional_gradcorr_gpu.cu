// GPU implementation of gradcorr: GGA gradient correction on device
// Uses PW_Basis GPU FFT for grad_rho and grad_dot operations

#include "xc_functional_gpu.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/memory.h"
#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <complex>
#include <cstring>

// Include shared GPU utilities, LDA and GGA device functions
#include "kernels/cuda/xc_gpu_utils.cuh"
#include "kernels/cuda/xc_lda_gpu.cuh"
#include "kernels/cuda/xc_gga_gpu.cuh"

#if defined(__CUDA) || defined(__ROCM)

namespace XC_GPU
{

// ============================================================
// Helper kernels for complex-valued FFT operations
// std::complex<double> is stored as interleaved (re, im) pairs
// which is identical to double2 / cuDoubleComplex layout
// ============================================================

// Kernel: gdrtmp[ig] = i * rhog[ig] * gcar[ig][idir]
__global__ void mul_iG_complex_kernel(const int npw,
                                       const double2* __restrict__ rhog,
                                       const double* __restrict__ gcar_flat, // [npw*3]
                                       const int idir,
                                       double2* __restrict__ out)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw)
    {
        double g = gcar_flat[ig * 3 + idir];
        double re = rhog[ig].x;
        double im = rhog[ig].y;
        // i * (re + i*im) * g = (-im*g) + i*(re*g)
        out[ig].x = -im * g;
        out[ig].y = re * g;
    }
}

// Kernel: extract real part of complex array * tpiba into gdr
// gdr layout: gdr[ir * 3 + idir]
__global__ void extract_gdr_kernel(const int nrxx,
                                    const double2* __restrict__ cdata,
                                    const double tpiba,
                                    double* __restrict__ gdr,
                                    const int idir)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nrxx)
    {
        gdr[ir * 3 + idir] = cdata[ir].x * tpiba;
    }
}

// Kernel: set complex from h component for grad_dot
// aux[ir] = complex(h[ir*3+idir], 0)
__global__ void set_complex_from_h_kernel(const int nrxx,
                                           const double* __restrict__ h,
                                           const int idir,
                                           double2* __restrict__ aux)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nrxx)
    {
        aux[ir].x = h[ir * 3 + idir];
        aux[ir].y = 0.0;
    }
}

// Kernel: gaux[ig] = i * aux_g[ig] * gcar[ig][idir] (first direction, overwrite)
__global__ void set_iG_complex_kernel(const int npw,
                                       const double2* __restrict__ aux_g,
                                       const double* __restrict__ gcar_flat,
                                       const int idir,
                                       double2* __restrict__ gaux)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw)
    {
        double g = gcar_flat[ig * 3 + idir];
        double re = aux_g[ig].x;
        double im = aux_g[ig].y;
        gaux[ig].x = -im * g;
        gaux[ig].y = re * g;
    }
}

// Kernel: gaux[ig] += i * aux_g[ig] * gcar[ig][idir] (subsequent directions, accumulate)
__global__ void accum_iG_complex_kernel(const int npw,
                                         const double2* __restrict__ aux_g,
                                         const double* __restrict__ gcar_flat,
                                         const int idir,
                                         double2* __restrict__ gaux)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < npw)
    {
        double g = gcar_flat[ig * 3 + idir];
        double re = aux_g[ig].x;
        double im = aux_g[ig].y;
        gaux[ig].x += -im * g;
        gaux[ig].y += re * g;
    }
}

// Kernel: dh[ir] = real(aux[ir]) * tpiba
__global__ void extract_dh_real_kernel(const int nrxx,
                                        const double2* __restrict__ aux,
                                        const double tpiba,
                                        double* __restrict__ dh)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nrxx) dh[ir] = aux[ir].x * tpiba;
}

// Kernel: rhotmp = rho + fac * rho_core
__global__ void add_core_kernel(const int n, const double* __restrict__ rho,
                                 const double* __restrict__ rho_core,
                                 const double fac, double* __restrict__ out)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < n) out[ir] = rho[ir] + fac * rho_core[ir];
}

// Kernel: rho_real -> complex for FFT
__global__ void real_to_complex_kernel(const int n, const double* __restrict__ rho,
                                        double2* __restrict__ out)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < n) { out[ir].x = rho[ir]; out[ir].y = 0.0; }
}

// Kernel: v_xc[ir] -= dh[ir]
__global__ void sub_dh_gpu_kernel(const int n, double* __restrict__ v_xc,
                                   const double* __restrict__ dh)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < n) v_xc[ir] -= dh[ir];
}

// Kernel: rhotmp[ir] -= fac * rho_core[ir]
__global__ void sub_core_kernel(const int n, double* __restrict__ rhotmp,
                                 const double* __restrict__ rho_core, const double fac)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < n) rhotmp[ir] -= fac * rho_core[ir];
}

// Kernel: compute vtxc correction = -sum(dh[ir] * rhotmp[ir])
// Negative because CPU code does vtxcgc -= sum(dh * rhotmp)
// Uses block reduction, result atomically added to vtxc_buf
__global__ void vtxc_correction_kernel(const int n,
                                        const double* __restrict__ dh,
                                        const double* __restrict__ rhotmp,
                                        double* __restrict__ vtxc_buf)
{
    double lv = 0.0;
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < n) lv = -dh[ir] * rhotmp[ir]; // negative for subtraction

    // Block reduction
    __shared__ double shared_v[8];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    lv = warp_reduce_sum(lv);
    if (lane == 0) shared_v[wid] = lv;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    lv = (threadIdx.x < (unsigned)nwarps) ? shared_v[threadIdx.x] : 0.0;
    if (wid == 0)
    {
        lv = warp_reduce_sum(lv);
        if (lane == 0) atomicAdd_double(vtxc_buf, lv);
    }
}

// ============================================================
// GGA XC kernels (must be in this file to access GGA device functions)
// ============================================================

// GGA XC kernel for nspin=1
__global__ void gga_xc_nspin1_kernel(
    const int nrxx,
    const double* __restrict__ rhotmp,
    const double* __restrict__ rho_core,
    const double* __restrict__ gdr,
    double* __restrict__ v_xc,
    double* __restrict__ h,
    double* __restrict__ etxc_buf,
    double* __restrict__ vtxc_buf,
    const int* __restrict__ func_ids, const int nfunc,
    const double hybrid_alpha, const double fac)
{
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double le = 0.0, lv = 0.0;
    if (ir < nrxx)
    {
        double arho = fabs(rhotmp[ir]);
        h[ir * 3] = h[ir * 3 + 1] = h[ir * 3 + 2] = 0.0;
        if (arho > epsr)
        {
            double gx = gdr[ir * 3], gy = gdr[ir * 3 + 1], gz = gdr[ir * 3 + 2];
            double grho2 = gx * gx + gy * gy + gz * gz;
            double segno = (rhotmp[ir] >= 0.0) ? 1.0 : -1.0;
            double sxc, v1xc, v2xc;
            d_gcxc(func_ids, nfunc, arho, grho2, sxc, v1xc, v2xc, hybrid_alpha);
            v_xc[ir] += e2 * v1xc;
            h[ir * 3]     = e2 * v2xc * gx;
            h[ir * 3 + 1] = e2 * v2xc * gy;
            h[ir * 3 + 2] = e2 * v2xc * gz;
            lv += e2 * v1xc * (rhotmp[ir] - fac * rho_core[ir]);
            le += e2 * sxc * segno;
        }
    }
    block_reduce_add(le, lv, etxc_buf, vtxc_buf);
}

// GGA XC kernel for nspin=2
__global__ void gga_xc_nspin2_kernel(
    const int nrxx,
    const double* __restrict__ rhotmp1,
    const double* __restrict__ rhotmp2,
    const double* __restrict__ rho_core,
    const double* __restrict__ gdr1,
    const double* __restrict__ gdr2,
    double* __restrict__ v_xc_up,
    double* __restrict__ v_xc_dw,
    double* __restrict__ h1,
    double* __restrict__ h2,
    double* __restrict__ etxc_buf,
    double* __restrict__ vtxc_buf,
    const int* __restrict__ func_ids, const int nfunc,
    const double hybrid_alpha, const double fac)
{
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double le = 0.0, lv = 0.0;
    if (ir < nrxx)
    {
        h1[ir * 3] = h1[ir * 3 + 1] = h1[ir * 3 + 2] = 0.0;
        h2[ir * 3] = h2[ir * 3 + 1] = h2[ir * 3 + 2] = 0.0;

        double g1x = gdr1[ir * 3], g1y = gdr1[ir * 3 + 1], g1z = gdr1[ir * 3 + 2];
        double g2x = gdr2[ir * 3], g2y = gdr2[ir * 3 + 1], g2z = gdr2[ir * 3 + 2];
        double grho2a = g1x * g1x + g1y * g1y + g1z * g1z;
        double grho2b = g2x * g2x + g2y * g2y + g2z * g2z;

        double sx, v1xup, v1xdw, v2xup, v2xdw;
        d_gcx_spin(func_ids[0], rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
                   sx, v1xup, v1xdw, v2xup, v2xdw, hybrid_alpha);

        double sc = 0.0, v1cup = 0.0, v1cdw = 0.0, v2c = 0.0;
        double rh = rhotmp1[ir] + rhotmp2[ir];
        if (rh > epsr)
        {
            double zeta = (rhotmp1[ir] - rhotmp2[ir]) / rh;
            double ghx = g1x + g2x, ghy = g1y + g2y, ghz = g1z + g2z;
            double grh2 = ghx * ghx + ghy * ghy + ghz * ghz;
            d_gcc_spin(func_ids[0], (nfunc > 1 ? func_ids[1] : 0),
                       rh, zeta, grh2, sc, v1cup, v1cdw, v2c, hybrid_alpha);
        }

        double v2cup = v2c, v2cdw = v2c, v2cud = v2c;

        v_xc_up[ir] += e2 * (v1xup + v1cup);
        v_xc_dw[ir] += e2 * (v1xdw + v1cdw);

        h1[ir * 3]     = e2 * ((v2xup + v2cup) * g1x + v2cud * g2x);
        h1[ir * 3 + 1] = e2 * ((v2xup + v2cup) * g1y + v2cud * g2y);
        h1[ir * 3 + 2] = e2 * ((v2xup + v2cup) * g1z + v2cud * g2z);
        h2[ir * 3]     = e2 * ((v2xdw + v2cdw) * g2x + v2cud * g1x);
        h2[ir * 3 + 1] = e2 * ((v2xdw + v2cdw) * g2y + v2cud * g1y);
        h2[ir * 3 + 2] = e2 * ((v2xdw + v2cdw) * g2z + v2cud * g1z);

        lv += e2 * (v1xup + v1cup) * (rhotmp1[ir] - rho_core[ir] * fac);
        lv += e2 * (v1xdw + v1cdw) * (rhotmp2[ir] - rho_core[ir] * fac);
        le += e2 * (sx + sc);
    }
    block_reduce_add(le, lv, etxc_buf, vtxc_buf);
}

// ============================================================
// CPU FFT fallback for multi-process: D2H -> CPU FFT -> H2D
// ============================================================
static void fft_real_to_recip_via_cpu(
    const std::complex<double>* d_in,  // [nrxx] on GPU
    std::complex<double>* d_out,       // [npw] on GPU
    ModulePW::PW_Basis* rhopw)
{
    const int nrxx = rhopw->nrxx;
    const int npw = rhopw->npw;
    std::vector<std::complex<double>> h_in(nrxx);
    std::vector<std::complex<double>> h_out(npw);
    cudaMemcpy(h_in.data(), d_in, sizeof(std::complex<double>) * nrxx, cudaMemcpyDeviceToHost);
    rhopw->real2recip(h_in.data(), h_out.data());
    cudaMemcpy(d_out, h_out.data(), sizeof(std::complex<double>) * npw, cudaMemcpyHostToDevice);
}

static void fft_recip_to_real_via_cpu(
    const std::complex<double>* d_in,  // [npw] on GPU
    std::complex<double>* d_out,       // [nrxx] on GPU
    ModulePW::PW_Basis* rhopw)
{
    const int nrxx = rhopw->nrxx;
    const int npw = rhopw->npw;
    std::vector<std::complex<double>> h_in(npw);
    std::vector<std::complex<double>> h_out(nrxx);
    cudaMemcpy(h_in.data(), d_in, sizeof(std::complex<double>) * npw, cudaMemcpyDeviceToHost);
    rhopw->recip2real(h_in.data(), h_out.data());
    cudaMemcpy(d_out, h_out.data(), sizeof(std::complex<double>) * nrxx, cudaMemcpyHostToDevice);
}

// Wrapper: choose GPU FFT or CPU fallback based on gpu_fft_bundle availability
static void do_real_to_recip(
    const std::complex<double>* d_in,
    std::complex<double>* d_out,
    ModulePW::PW_Basis* rhopw)
{
    if (rhopw->gpu_fft_bundle != nullptr)
    {
        base_device::DEVICE_GPU* gpu_ctx = nullptr;
        rhopw->real_to_recip(gpu_ctx, d_in, d_out);
    }
    else
    {
        fft_real_to_recip_via_cpu(d_in, d_out, rhopw);
    }
}

static void do_recip_to_real(
    const std::complex<double>* d_in,
    std::complex<double>* d_out,
    ModulePW::PW_Basis* rhopw)
{
    if (rhopw->gpu_fft_bundle != nullptr)
    {
        base_device::DEVICE_GPU* gpu_ctx = nullptr;
        rhopw->recip_to_real(gpu_ctx, d_in, d_out);
    }
    else
    {
        fft_recip_to_real_via_cpu(d_in, d_out, rhopw);
    }
}

// ============================================================
// Host helper: GPU grad_rho
// ============================================================
static void grad_rho_on_gpu(
    const std::complex<double>* d_rhog, // [npw] on GPU
    double* d_gdr,                       // [nrxx*3] output on GPU
    const double* d_gcar_flat,           // [npw*3] on GPU
    ModulePW::PW_Basis* rhopw,
    const double tpiba)
{
    const int npw = rhopw->npw;
    const int nrxx = rhopw->nrxx;
    const int bs = 256;

    // Temp arrays for FFT
    std::complex<double>* d_gtmp = nullptr;
    std::complex<double>* d_rtmp = nullptr;
    cudaMallocCheck(&d_gtmp, sizeof(std::complex<double>) * npw, "gradcorr::grad_rho::d_gtmp");
    cudaMallocCheck(&d_rtmp, sizeof(std::complex<double>) * nrxx, "gradcorr::grad_rho::d_rtmp");

    for (int idir = 0; idir < 3; ++idir)
    {
        // gdrtmp[ig] = i * rhog[ig] * gcar[ig][idir]
        mul_iG_complex_kernel<<<(npw + bs - 1) / bs, bs>>>(
            npw,
            reinterpret_cast<const double2*>(d_rhog),
            d_gcar_flat, idir,
            reinterpret_cast<double2*>(d_gtmp));

        // IFFT: G -> R
        do_recip_to_real(d_gtmp, d_rtmp, rhopw);

        // Extract real part * tpiba into gdr[ir*3+idir]
        extract_gdr_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
            nrxx,
            reinterpret_cast<const double2*>(d_rtmp),
            tpiba, d_gdr, idir);
    }

    cudaFree(d_gtmp);
    cudaFree(d_rtmp);
}

// ============================================================
// Host helper: GPU grad_dot
// ============================================================
static void grad_dot_on_gpu(
    const double* d_h,    // [nrxx*3] on GPU
    double* d_dh,         // [nrxx] output on GPU
    const double* d_gcar_flat,
    ModulePW::PW_Basis* rhopw,
    const double tpiba)
{
    const int npw = rhopw->npw;
    const int nrxx = rhopw->nrxx;
    const int bs = 256;

    std::complex<double>* d_aux = nullptr;   // [nrxx] complex, real-space
    std::complex<double>* d_aux_g = nullptr; // [npw] complex, G-space
    std::complex<double>* d_gaux = nullptr;  // [npw] complex, accumulated
    cudaMallocCheck(&d_aux, sizeof(std::complex<double>) * nrxx, "gradcorr::grad_dot::d_aux");
    cudaMallocCheck(&d_aux_g, sizeof(std::complex<double>) * npw, "gradcorr::grad_dot::d_aux_g");
    cudaMallocCheck(&d_gaux, sizeof(std::complex<double>) * npw, "gradcorr::grad_dot::d_gaux");

    for (int idir = 0; idir < 3; ++idir)
    {
        // aux[ir] = complex(h[ir*3+idir], 0)
        set_complex_from_h_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
            nrxx, d_h, idir,
            reinterpret_cast<double2*>(d_aux));

        // FFT: R -> G
        do_real_to_recip(d_aux, d_aux_g, rhopw);

        // gaux[ig] += i * aux_g[ig] * gcar[ig][idir]
        if (idir == 0)
        {
            set_iG_complex_kernel<<<(npw + bs - 1) / bs, bs>>>(
                npw,
                reinterpret_cast<const double2*>(d_aux_g),
                d_gcar_flat, idir,
                reinterpret_cast<double2*>(d_gaux));
        }
        else
        {
            accum_iG_complex_kernel<<<(npw + bs - 1) / bs, bs>>>(
                npw,
                reinterpret_cast<const double2*>(d_aux_g),
                d_gcar_flat, idir,
                reinterpret_cast<double2*>(d_gaux));
        }
    }

    // IFFT: G -> R
    {
        do_recip_to_real(d_gaux, d_aux, rhopw);
    }

    // dh[ir] = real(aux[ir]) * tpiba
    extract_dh_real_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
        nrxx,
        reinterpret_cast<const double2*>(d_aux),
        tpiba, d_dh);

    cudaFree(d_aux);
    cudaFree(d_aux_g);
    cudaFree(d_gaux);
}

// ============================================================
// Main gradcorr_gpu function
// ============================================================
void gradcorr_gpu(const int nrxx, const int npw,
                  const int nspin0, const int nspin,
                  double** d_rho, const double* d_rho_core,
                  double** d_v_xc,
                  double& etxc, double& vtxc,
                  const std::vector<int>& func_ids,
                  const double hybrid_alpha,
                  const int func_type,
                  ModulePW::PW_Basis* rhopw,
                  const double tpiba,
                  const bool domag, const bool domag_z,
                  const double* ux_, const bool lsign_)
{
    ModuleBase::timer::tick("XC_Functional", "gradcorr_gpu");

    // Skip if not GGA
    if (func_type == 0 || func_type == 1)
    {
        ModuleBase::timer::tick("XC_Functional", "gradcorr_gpu");
        return;
    }

    const int bs = 256;
    const double fac = 1.0 / nspin0;

    // Upload gcar to GPU as flat array [npw*3]
    double* d_gcar_flat = nullptr;
    {
        double* h_gcar_flat = new double[npw * 3];
        for (int ig = 0; ig < npw; ++ig)
        {
            h_gcar_flat[ig * 3 + 0] = rhopw->gcar[ig].x;
            h_gcar_flat[ig * 3 + 1] = rhopw->gcar[ig].y;
            h_gcar_flat[ig * 3 + 2] = rhopw->gcar[ig].z;
        }
        cudaMallocCheck(&d_gcar_flat, sizeof(double) * npw * 3, "gradcorr::d_gcar_flat");
        cudaMemcpy(d_gcar_flat, h_gcar_flat, sizeof(double) * npw * 3, cudaMemcpyHostToDevice);
        delete[] h_gcar_flat;
    }

    // Upload func_ids to GPU
    int* d_fids = nullptr;
    cudaMallocCheck(&d_fids, func_ids.size() * sizeof(int), "gradcorr::d_fids");
    cudaMemcpy(d_fids, func_ids.data(), func_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate etxc/vtxc reduction buffers on GPU
    double* d_etxc = nullptr;
    double* d_vtxc = nullptr;
    cudaMallocCheck(&d_etxc, sizeof(double), "gradcorr::d_etxc");
    cudaMallocCheck(&d_vtxc, sizeof(double), "gradcorr::d_vtxc");
    cudaMemset(d_etxc, 0, sizeof(double));
    cudaMemset(d_vtxc, 0, sizeof(double));

    // Step 1: FFT rho -> rhog on GPU
    // We need complex arrays for FFT
    std::complex<double>* d_rho_complex = nullptr;
    std::complex<double>* d_rhog[2] = {nullptr, nullptr};
    std::complex<double>* d_rho_core_complex = nullptr;
    std::complex<double>* d_rhog_core = nullptr;

    cudaMallocCheck(&d_rho_complex, sizeof(std::complex<double>) * nrxx, "gradcorr::d_rho_complex");
    cudaMallocCheck(&d_rhog[0], sizeof(std::complex<double>) * npw, "gradcorr::d_rhog[0]");
    cudaMallocCheck(&d_rho_core_complex, sizeof(std::complex<double>) * nrxx, "gradcorr::d_rho_core_complex");
    cudaMallocCheck(&d_rhog_core, sizeof(std::complex<double>) * npw, "gradcorr::d_rhog_core");

    // FFT rho[0] -> rhog[0]
    real_to_complex_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rho[0],
        reinterpret_cast<double2*>(d_rho_complex));
    do_real_to_recip(d_rho_complex, d_rhog[0], rhopw);

    // FFT rho_core -> rhog_core
    real_to_complex_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rho_core,
        reinterpret_cast<double2*>(d_rho_core_complex));
    do_real_to_recip(d_rho_core_complex, d_rhog_core, rhopw);

    if (nspin == 2)
    {
        cudaMallocCheck(&d_rhog[1], sizeof(std::complex<double>) * npw, "gradcorr::d_rhog[1]");
        real_to_complex_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rho[1],
            reinterpret_cast<double2*>(d_rho_complex));
        do_real_to_recip(d_rho_complex, d_rhog[1], rhopw);
    }

    // Step 2: Compute rhotmp and rhogsum, then grad_rho
    double* d_rhotmp1 = nullptr;
    double* d_rhotmp2 = nullptr;
    std::complex<double>* d_rhogsum1 = nullptr;
    std::complex<double>* d_rhogsum2 = nullptr;
    double* d_gdr1 = nullptr;
    double* d_gdr2 = nullptr;

    cudaMallocCheck(&d_rhotmp1, sizeof(double) * nrxx, "gradcorr::d_rhotmp1");
    cudaMallocCheck(&d_rhogsum1, sizeof(std::complex<double>) * npw, "gradcorr::d_rhogsum1");
    cudaMallocCheck(&d_gdr1, sizeof(double) * nrxx * 3, "gradcorr::d_gdr1");

    // rhotmp1 = rho[0] + fac * rho_core
    add_core_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rho[0], d_rho_core, fac, d_rhotmp1);

    // rhogsum1 = rhog[0] + fac * rhog_core
    // Treat complex<double> as 2 interleaved doubles for the addition
    add_core_kernel<<<(2 * npw + bs - 1) / bs, bs>>>(
        2 * npw,
        reinterpret_cast<const double*>(d_rhog[0]),
        reinterpret_cast<const double*>(d_rhog_core),
        fac,
        reinterpret_cast<double*>(d_rhogsum1));

    // grad_rho for spin channel 1
    grad_rho_on_gpu(d_rhogsum1, d_gdr1, d_gcar_flat, rhopw, tpiba);

    if (nspin0 == 2)
    {
        cudaMallocCheck(&d_rhotmp2, sizeof(double) * nrxx, "gradcorr::d_rhotmp2");
        cudaMallocCheck(&d_rhogsum2, sizeof(std::complex<double>) * npw, "gradcorr::d_rhogsum2");
        cudaMallocCheck(&d_gdr2, sizeof(double) * nrxx * 3, "gradcorr::d_gdr2");

        add_core_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rho[1], d_rho_core, fac, d_rhotmp2);
        add_core_kernel<<<(2 * npw + bs - 1) / bs, bs>>>(
            2 * npw,
            reinterpret_cast<const double*>(d_rhog[1]),
            reinterpret_cast<const double*>(d_rhog_core),
            fac,
            reinterpret_cast<double*>(d_rhogsum2));

        grad_rho_on_gpu(d_rhogsum2, d_gdr2, d_gcar_flat, rhopw, tpiba);
    }

    // Step 3: GGA XC kernel
    double* d_h1 = nullptr;
    double* d_h2 = nullptr;
    cudaMallocCheck(&d_h1, sizeof(double) * nrxx * 3, "gradcorr::d_h1");

    if (nspin0 == 1)
    {
        gga_xc_nspin1_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
            nrxx, d_rhotmp1, d_rho_core, d_gdr1,
            d_v_xc[0], d_h1, d_etxc, d_vtxc,
            d_fids, (int)func_ids.size(), hybrid_alpha, fac);
    }
    else
    {
        cudaMallocCheck(&d_h2, sizeof(double) * nrxx * 3, "gradcorr::d_h2");
        gga_xc_nspin2_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
            nrxx, d_rhotmp1, d_rhotmp2, d_rho_core,
            d_gdr1, d_gdr2,
            d_v_xc[0], d_v_xc[1], d_h1, d_h2,
            d_etxc, d_vtxc,
            d_fids, (int)func_ids.size(), hybrid_alpha, fac);
    }

    // Step 4: Subtract rho_core from rhotmp (for vtxc correction)
    sub_core_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rhotmp1, d_rho_core, fac);
    if (nspin0 == 2)
    {
        sub_core_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_rhotmp2, d_rho_core, fac);
    }

    // Step 5: grad_dot(h) -> dh, then v_xc -= dh, vtxc -= sum(dh * rhotmp)
    double* d_dh = nullptr;
    cudaMallocCheck(&d_dh, sizeof(double) * nrxx, "gradcorr::d_dh");

    for (int is = 0; is < nspin0; ++is)
    {
        double* d_h_cur = (is == 0) ? d_h1 : d_h2;
        grad_dot_on_gpu(d_h_cur, d_dh, d_gcar_flat, rhopw, tpiba);

        sub_dh_gpu_kernel<<<(nrxx + bs - 1) / bs, bs>>>(nrxx, d_v_xc[is], d_dh);

        double* d_rhotmp_cur = (is == 0) ? d_rhotmp1 : d_rhotmp2;
        vtxc_correction_kernel<<<(nrxx + bs - 1) / bs, bs>>>(
            nrxx, d_dh, d_rhotmp_cur, d_vtxc);
    }

    // Step 6: Copy etxc/vtxc back to host
    double etxcgc = 0.0, vtxcgc = 0.0;
    cudaMemcpy(&etxcgc, d_etxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vtxcgc, d_vtxc, sizeof(double), cudaMemcpyDeviceToHost);

    // vtxc correction is subtracted
    etxc += etxcgc;
    vtxc += vtxcgc;

    // Cleanup
    cudaFree(d_gcar_flat);
    cudaFree(d_fids);
    cudaFree(d_etxc);
    cudaFree(d_vtxc);
    cudaFree(d_rho_complex);
    cudaFree(d_rhog[0]);
    if (d_rhog[1]) cudaFree(d_rhog[1]);
    cudaFree(d_rho_core_complex);
    cudaFree(d_rhog_core);
    cudaFree(d_rhotmp1);
    if (d_rhotmp2) cudaFree(d_rhotmp2);
    cudaFree(d_rhogsum1);
    if (d_rhogsum2) cudaFree(d_rhogsum2);
    cudaFree(d_gdr1);
    if (d_gdr2) cudaFree(d_gdr2);
    cudaFree(d_h1);
    if (d_h2) cudaFree(d_h2);
    cudaFree(d_dh);

    ModuleBase::timer::tick("XC_Functional", "gradcorr_gpu");
}

} // namespace XC_GPU

#endif // __CUDA || __ROCM
