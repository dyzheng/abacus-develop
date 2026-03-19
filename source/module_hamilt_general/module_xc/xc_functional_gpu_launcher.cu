// GPU XC kernel launcher: contains both kernel definitions and host launchers
#include "xc_functional_gpu.h"
#include <cuda_runtime.h>

// Include shared GPU utilities and LDA device functions
#include "kernels/cuda/xc_gpu_utils.cuh"
#include "kernels/cuda/xc_lda_gpu.cuh"

namespace XC_GPU
{

// ============================================================
// Kernel definitions
// ============================================================

__global__ void xc_lda_nspin1_kernel(
    const int nrxx,
    const double* __restrict__ rho,
    const double* __restrict__ rho_core,
    double* __restrict__ v_xc,
    double* __restrict__ etxc_buf,
    double* __restrict__ vtxc_buf,
    const int* __restrict__ func_ids,
    const int nfunc,
    const double hybrid_alpha)
{
    const double e2 = 2.0;
    const double vanishing_charge = 1.0e-10;
    const double third = 1.0 / 3.0;
    const double pi34 = 0.6203504908994;

    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double local_etxc = 0.0, local_vtxc = 0.0;

    if (ir < nrxx)
    {
        double rhox = rho[ir] + rho_core[ir];
        double arhox = fabs(rhox);
        if (arhox > vanishing_charge)
        {
            double rs = pi34 / pow(arhox, third);
            double exc = 0.0, vxc = 0.0;
            for (int i = 0; i < nfunc; ++i)
            {
                double e_tmp, v_tmp;
                d_xc_single(func_ids[i], rs, e_tmp, v_tmp, hybrid_alpha);
                exc += e_tmp; vxc += v_tmp;
            }
            v_xc[ir] = e2 * vxc;
            local_etxc = e2 * exc * rhox;
            local_vtxc = v_xc[ir] * rho[ir];
        }
        else { v_xc[ir] = 0.0; }
    }
    block_reduce_add(local_etxc, local_vtxc, etxc_buf, vtxc_buf);
}
__global__ void xc_lda_nspin2_kernel(
    const int nrxx,
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dw,
    const double* __restrict__ rho_core,
    double* __restrict__ v_xc_up,
    double* __restrict__ v_xc_dw,
    double* __restrict__ etxc_buf,
    double* __restrict__ vtxc_buf,
    const int* __restrict__ func_ids,
    const int nfunc,
    const double hybrid_alpha)
{
    const double e2 = 2.0, vanishing_charge = 1.0e-10;
    const double third = 1.0 / 3.0, pi34 = 0.62035049089940;
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double le = 0.0, lv = 0.0;
    if (ir < nrxx)
    {
        double rhox = rho_up[ir] + rho_dw[ir] + rho_core[ir];
        double arhox = fabs(rhox);
        if (arhox > vanishing_charge)
        {
            double zeta = (rho_up[ir] - rho_dw[ir]) / arhox;
            if (fabs(zeta) > 1.0) zeta = (zeta > 0.0) ? 1.0 : -1.0;
            double rs = pi34 / pow(arhox, third);
            double exc = 0.0, vu = 0.0, vd = 0.0;
            for (int i = 0; i < nfunc; ++i)
            {
                double et, vut, vdt;
                d_xc_spin_single(func_ids[i], rs, arhox, zeta, et, vut, vdt, hybrid_alpha);
                exc += et; vu += vut; vd += vdt;
            }
            v_xc_up[ir] = e2 * vu; v_xc_dw[ir] = e2 * vd;
            le = e2 * exc * rhox;
            lv = v_xc_up[ir] * rho_up[ir] + v_xc_dw[ir] * rho_dw[ir];
        }
        else { v_xc_up[ir] = 0.0; v_xc_dw[ir] = 0.0; }
    }
    block_reduce_add(le, lv, etxc_buf, vtxc_buf);
}

__global__ void xc_lda_nspin4_kernel(
    const int nrxx,
    const double* __restrict__ rho0, const double* __restrict__ rho1,
    const double* __restrict__ rho2, const double* __restrict__ rho3,
    const double* __restrict__ rho_core,
    double* __restrict__ vxc0, double* __restrict__ vxc1,
    double* __restrict__ vxc2, double* __restrict__ vxc3,
    double* __restrict__ etxc_buf, double* __restrict__ vtxc_buf,
    const int* __restrict__ func_ids, const int nfunc, const double hybrid_alpha)
{
    const double e2 = 2.0, vanishing_charge = 1.0e-10;
    const double third = 1.0 / 3.0, pi34 = 0.62035049089940;
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double le = 0.0, lv = 0.0;
    if (ir < nrxx)
    {
        double amag = sqrt(rho1[ir]*rho1[ir] + rho2[ir]*rho2[ir] + rho3[ir]*rho3[ir]);
        double rhox = rho0[ir] + rho_core[ir];
        double arhox = fabs(rhox);
        if (arhox > vanishing_charge)
        {
            double zeta = amag / arhox;
            if (fabs(zeta) > 1.0) zeta = (zeta > 0.0) ? 1.0 : -1.0;
            double rs = pi34 / pow(arhox, third);
            double exc = 0.0, vu = 0.0, vd = 0.0;
            for (int i = 0; i < nfunc; ++i)
            {
                double et, vut, vdt;
                d_xc_spin_single(func_ids[i], rs, arhox, zeta, et, vut, vdt, hybrid_alpha);
                exc += et; vu += vut; vd += vdt;
            }
            le = e2 * exc * rhox;
            vxc0[ir] = e2 * 0.5 * (vu + vd);
            lv = vxc0[ir] * rho0[ir];
            double vs = 0.5 * (vu - vd);
            if (amag > vanishing_charge)
            {
                vxc1[ir] = e2 * vs * rho1[ir] / amag;
                vxc2[ir] = e2 * vs * rho2[ir] / amag;
                vxc3[ir] = e2 * vs * rho3[ir] / amag;
                lv += vxc1[ir]*rho1[ir] + vxc2[ir]*rho2[ir] + vxc3[ir]*rho3[ir];
            }
            else { vxc1[ir] = 0.0; vxc2[ir] = 0.0; vxc3[ir] = 0.0; }
        }
        else { vxc0[ir]=0; vxc1[ir]=0; vxc2[ir]=0; vxc3[ir]=0; }
    }
    block_reduce_add(le, lv, etxc_buf, vtxc_buf);
}

// ============================================================
// Host launcher functions
// ============================================================

void v_xc_lda_gpu_nspin1(const int nrxx,
                          const double* d_rho, const double* d_rho_core,
                          double* d_v_xc, double& etxc, double& vtxc,
                          const std::vector<int>& func_ids, const double hybrid_alpha)
{
    int* d_fids = nullptr;
    cudaMalloc(&d_fids, func_ids.size() * sizeof(int));
    cudaMemcpy(d_fids, func_ids.data(), func_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    double* d_etxc = nullptr; double* d_vtxc = nullptr;
    cudaMalloc(&d_etxc, sizeof(double)); cudaMalloc(&d_vtxc, sizeof(double));
    cudaMemset(d_etxc, 0, sizeof(double)); cudaMemset(d_vtxc, 0, sizeof(double));
    const int bs = 256, gs = (nrxx + bs - 1) / bs;
    xc_lda_nspin1_kernel<<<gs, bs>>>(nrxx, d_rho, d_rho_core, d_v_xc,
                                      d_etxc, d_vtxc, d_fids, (int)func_ids.size(), hybrid_alpha);
    cudaMemcpy(&etxc, d_etxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vtxc, d_vtxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_fids); cudaFree(d_etxc); cudaFree(d_vtxc);
}

void v_xc_lda_gpu_nspin2(const int nrxx,
                          const double* d_rho_up, const double* d_rho_dw,
                          const double* d_rho_core,
                          double* d_v_xc_up, double* d_v_xc_dw,
                          double& etxc, double& vtxc,
                          const std::vector<int>& func_ids, const double hybrid_alpha)
{
    int* d_fids = nullptr;
    cudaMalloc(&d_fids, func_ids.size() * sizeof(int));
    cudaMemcpy(d_fids, func_ids.data(), func_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    double* d_etxc = nullptr; double* d_vtxc = nullptr;
    cudaMalloc(&d_etxc, sizeof(double)); cudaMalloc(&d_vtxc, sizeof(double));
    cudaMemset(d_etxc, 0, sizeof(double)); cudaMemset(d_vtxc, 0, sizeof(double));
    const int bs = 256, gs = (nrxx + bs - 1) / bs;
    xc_lda_nspin2_kernel<<<gs, bs>>>(nrxx, d_rho_up, d_rho_dw, d_rho_core,
                                      d_v_xc_up, d_v_xc_dw, d_etxc, d_vtxc,
                                      d_fids, (int)func_ids.size(), hybrid_alpha);
    cudaMemcpy(&etxc, d_etxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vtxc, d_vtxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_fids); cudaFree(d_etxc); cudaFree(d_vtxc);
}

void v_xc_lda_gpu_nspin4(const int nrxx,
                          const double* d_rho0, const double* d_rho1,
                          const double* d_rho2, const double* d_rho3,
                          const double* d_rho_core,
                          double* d_v_xc0, double* d_v_xc1,
                          double* d_v_xc2, double* d_v_xc3,
                          double& etxc, double& vtxc,
                          const std::vector<int>& func_ids, const double hybrid_alpha)
{
    int* d_fids = nullptr;
    cudaMalloc(&d_fids, func_ids.size() * sizeof(int));
    cudaMemcpy(d_fids, func_ids.data(), func_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    double* d_etxc = nullptr; double* d_vtxc = nullptr;
    cudaMalloc(&d_etxc, sizeof(double)); cudaMalloc(&d_vtxc, sizeof(double));
    cudaMemset(d_etxc, 0, sizeof(double)); cudaMemset(d_vtxc, 0, sizeof(double));
    const int bs = 256, gs = (nrxx + bs - 1) / bs;
    xc_lda_nspin4_kernel<<<gs, bs>>>(nrxx, d_rho0, d_rho1, d_rho2, d_rho3, d_rho_core,
                                      d_v_xc0, d_v_xc1, d_v_xc2, d_v_xc3,
                                      d_etxc, d_vtxc, d_fids, (int)func_ids.size(), hybrid_alpha);
    cudaMemcpy(&etxc, d_etxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vtxc, d_vtxc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_fids); cudaFree(d_etxc); cudaFree(d_vtxc);
}

} // namespace XC_GPU
