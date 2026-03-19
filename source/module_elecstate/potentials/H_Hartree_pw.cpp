#include "H_Hartree_pw.h"

#include "module_parameter/parameter.h"
#include "module_base/constants.h"
#include "module_base/timer.h"
#include "module_base/parallel_reduce.h"
#if defined(__CUDA) || defined(__ROCM)
#include <cuda_runtime.h>
#endif

namespace elecstate
{

double H_Hartree_pw::hartree_energy = 0.0;

//--------------------------------------------------------------------
// Transform charge density to hartree potential.
//--------------------------------------------------------------------
ModuleBase::matrix H_Hartree_pw::v_hartree(const UnitCell &cell,
                                           ModulePW::PW_Basis *rho_basis,
                                           const int &nspin,
                                           const double *const *const rho)
{
    ModuleBase::TITLE("H_Hartree_pw", "v_hartree");
    ModuleBase::timer::tick("H_Hartree_pw", "v_hartree");

    //  Hartree potential VH(r) from n(r)
    std::vector<std::complex<double>> Porter(rho_basis->nmaxgr);
    const int nspin0 = (nspin == 2) ? 2 : 1;
    for (int is = 0; is < nspin0; is++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
            Porter[ir] += std::complex<double>(rho[is][ir], 0.0);
    }
    //=============================
    //  bring rho (aux) to G space
    //=============================
#if defined(__CUDA) || defined(__ROCM)
    if (rho_basis->gpu_fft_bundle != nullptr)
    {
        // GPU FFT path: H2D -> cuFFT -> D2H
        std::complex<double>* d_porter = nullptr;
        std::complex<double>* d_rhog = nullptr;
        cudaMalloc(&d_porter, sizeof(std::complex<double>) * rho_basis->nrxx);
        cudaMalloc(&d_rhog, sizeof(std::complex<double>) * rho_basis->npw);
        cudaMemcpy(d_porter, Porter.data(), sizeof(std::complex<double>) * rho_basis->nrxx, cudaMemcpyHostToDevice);

        base_device::DEVICE_GPU* gpu_ctx = nullptr;
        rho_basis->real_to_recip(gpu_ctx, d_porter, d_rhog);

        cudaMemcpy(Porter.data(), d_rhog, sizeof(std::complex<double>) * rho_basis->npw, cudaMemcpyDeviceToHost);
        cudaFree(d_porter);
        cudaFree(d_rhog);
    }
    else
#endif
    {
        rho_basis->real2recip(Porter.data(), Porter.data());
    }

    //=======================================================
    // calculate hartree potential in G-space (NB: V(G=0)=0 )
    //=======================================================

    double ehart = 0.0;

    std::vector<std::complex<double>> vh_g(rho_basis->npw);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:ehart)
#endif
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if (rho_basis->gg[ig] >= 1.0e-8) // LiuXh 20180410
        {
            const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (cell.tpiba2 * rho_basis->gg[ig]);

            ehart += (conj(Porter[ig]) * Porter[ig]).real() * fac;
            vh_g[ig] = fac * Porter[ig];
        }
    }

    Parallel_Reduce::reduce_pool(ehart);
    ehart *= 0.5 * cell.omega;
    // std::cout << " ehart=" << ehart << std::endl;
    H_Hartree_pw::hartree_energy = ehart;

    //==========================================
    // transform hartree potential to real space
    //==========================================
#if defined(__CUDA) || defined(__ROCM)
    if (rho_basis->gpu_fft_bundle != nullptr)
    {
        // GPU IFFT path: H2D -> cuFFT -> D2H
        std::complex<double>* d_vhg = nullptr;
        std::complex<double>* d_porter = nullptr;
        cudaMalloc(&d_vhg, sizeof(std::complex<double>) * rho_basis->npw);
        cudaMalloc(&d_porter, sizeof(std::complex<double>) * rho_basis->nrxx);
        cudaMemcpy(d_vhg, vh_g.data(), sizeof(std::complex<double>) * rho_basis->npw, cudaMemcpyHostToDevice);

        base_device::DEVICE_GPU* gpu_ctx = nullptr;
        rho_basis->recip_to_real(gpu_ctx, d_vhg, d_porter);

        cudaMemcpy(Porter.data(), d_porter, sizeof(std::complex<double>) * rho_basis->nrxx, cudaMemcpyDeviceToHost);
        cudaFree(d_vhg);
        cudaFree(d_porter);
    }
    else
#endif
    {
        rho_basis->recip2real(vh_g.data(), Porter.data());
    }

    //==========================================
    // Add hartree potential to the xc potential
    //==========================================
    ModuleBase::matrix v(nspin, rho_basis->nrxx);
    if (nspin == 4)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
            v(0, ir) = Porter[ir].real();
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 512)
#endif
        for (int is = 0; is < nspin; is++)
            for (int ir = 0; ir < rho_basis->nrxx; ir++)
                v(is, ir) = Porter[ir].real();
    }

    ModuleBase::timer::tick("H_Hartree_pw", "v_hartree");
    return v;
} // end subroutine v_h

PotHartree::PotHartree(const ModulePW::PW_Basis* rho_basis_in)
{
    this->rho_basis_ = rho_basis_in;
    this->dynamic_mode = true;
    this->fixed_mode = false;
}

void PotHartree::cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff)
{
    if(PARAM.inp.use_paw)
    {
        double ** rho_tmp;
        rho_tmp = new double*[chg->nspin];
        for(int is = 0; is < chg->nspin; is++)
        {
            rho_tmp[is] = new double[rho_basis_->nrxx];
            for(int ir = 0; ir < rho_basis_->nrxx; ir++)
            {
                rho_tmp[is][ir] = chg->rho[is][ir] + chg->nhat[is][ir];
            }
        }
        v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, rho_tmp);

        for(int is = 0; is < chg->nspin; is++)
        {
            delete[] rho_tmp[is];
        }
        delete[] rho_tmp;
    }
    else
    {
        v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, chg->rho);
    }
    return;
}

} // namespace elecstate
