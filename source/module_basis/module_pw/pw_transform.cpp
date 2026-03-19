#include "module_fft/fft_bundle.h"
#include <complex>
#include "pw_basis.h"
#include <cassert>
#include "module_base/global_function.h"
#include "module_base/timer.h"
#include "pw_gatherscatter.h"
#include "module_basis/module_pw/kernels/pw_op.h"

namespace ModulePW {
/**
 * @brief transform real space to reciprocal space
 * @details c(g)=\int dr*f(r)*exp(-ig*r)
 *          Here we calculate c(g)
 * @param in: (nplane,ny,nx), complex<double> data
 * @param out: (nz, ns),  complex<double> data
 */
template <typename FPTYPE>
void PW_Basis::real2recip(const std::complex<FPTYPE>* in,
                          std::complex<FPTYPE>* out,
                          const bool add,
                          const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");

    assert(this->gamma_only == false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
    for(int ir = 0 ; ir < this->nrxx ; ++ir)
    {
        this->fft_bundle.get_auxr_data<FPTYPE>()[ir] = in[ir];
    }
    this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(),fft_bundle.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());
    
    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(),fft_bundle.get_auxg_data<FPTYPE>());

    if(add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ig = 0 ; ig < this->npw ; ++ig)
        {
            out[ig] += tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ig = 0 ; ig < this->npw ; ++ig)
        {
            out[ig] = tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform real space to reciprocal space
 * @details c(g)=\int dr*f(r)*exp(-ig*r)
 *          Here we calculate c(g)
 * @param in: (nplane,ny,nx), double data
 * @param out: (nz, ns),  complex<double> data
 */
template <typename FPTYPE>
void PW_Basis::real2recip(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");
    if(this->gamma_only)
    {
        const int npy = this->ny * this->nplane;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ix = 0 ; ix < this->nx ; ++ix)
        {
            for(int ipy = 0 ; ipy < npy ; ++ipy)
            {
                this->fft_bundle.get_rspace_data<FPTYPE>()[ix*npy + ipy] = in[ix*npy + ipy];
            }
        }

        this->fft_bundle.fftxyr2c(fft_bundle.get_rspace_data<FPTYPE>(),fft_bundle.get_auxr_data<FPTYPE>());
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ir = 0 ; ir < this->nrxx ; ++ir)
        {
            this->fft_bundle.get_auxr_data<FPTYPE>()[ir] = std::complex<FPTYPE>(in[ir],0);
        }
        this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(),fft_bundle.get_auxr_data<FPTYPE>());
    }
    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());
    
    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(),fft_bundle.get_auxg_data<FPTYPE>());

    if(add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ig = 0 ; ig < this->npw ; ++ig)
        {
            out[ig] += tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ig = 0 ; ig < this->npw ; ++ig)
        {
            out[ig] = tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform reciprocal space to real space
 * @details f(r)=1/V * \sum_{g} c(g)*exp(ig*r)
 *          Here we calculate f(r)
 * @param in: (nz,ns), complex<double>
 * @param out: (nplane, ny, nx), complex<double>
 */
template <typename FPTYPE>
void PW_Basis::recip2real(const std::complex<FPTYPE>* in,
                          std::complex<FPTYPE>* out,
                          const bool add,
                          const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
    for(int i = 0 ; i < this->nst * this->nz ; ++i)
    {
        fft_bundle.get_auxg_data<FPTYPE>()[i] = std::complex<FPTYPE>(0, 0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
    for(int ig = 0 ; ig < this->npw ; ++ig)
    {
        this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]] = in[ig];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(),this->fft_bundle.get_auxr_data<FPTYPE>());

    this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(),fft_bundle.get_auxr_data<FPTYPE>());
    
    if(add)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ir = 0 ; ir < this->nrxx ; ++ir)
        {
            out[ir] += factor * this->fft_bundle.get_auxr_data<FPTYPE>()[ir];
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
        for(int ir = 0 ; ir < this->nrxx ; ++ir)
        {
            out[ir] = this->fft_bundle.get_auxr_data<FPTYPE>()[ir];
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

/**
 * @brief transform reciprocal space to real space
 * @details f(r)=1/V * \sum_{g} c(g)*exp(ig*r)
 *          Here we calculate f(r)
 * @param in: (nz,ns), complex<double>
 * @param out: (nplane, ny, nx), double
 */
template <typename FPTYPE>
void PW_Basis::recip2real(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
    for(int i = 0 ; i < this->nst * this->nz ; ++i)
    {
        fft_bundle.get_auxg_data<FPTYPE>()[i] = std::complex<double>(0, 0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
    for(int ig = 0 ; ig < this->npw ; ++ig)
    {
        this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]] = in[ig];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    if(this->gamma_only)
    {
        this->fft_bundle.fftxyc2r(fft_bundle.get_auxr_data<FPTYPE>(),fft_bundle.get_rspace_data<FPTYPE>());

        // r2c in place
        const int npy = this->ny * this->nplane;

        if(add)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096/sizeof(FPTYPE))
#endif
            for(int ix = 0 ; ix < this->nx ; ++ix)
            {
                for(int ipy = 0 ; ipy < npy ; ++ipy)
                {
                    out[ix*npy + ipy] += factor * this->fft_bundle.get_rspace_data<FPTYPE>()[ix*npy + ipy];
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096/sizeof(FPTYPE))
#endif
            for(int ix = 0 ; ix < this->nx ; ++ix)
            {
                for(int ipy = 0 ; ipy < npy ; ++ipy)
                {
                    out[ix*npy + ipy] = this->fft_bundle.get_rspace_data<FPTYPE>()[ix*npy + ipy];
                }
            }
        }
    }
    else
    {
        this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(),fft_bundle.get_auxr_data<FPTYPE>());
        if(add)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
            for(int ir = 0 ; ir < this->nrxx ; ++ir)
            {
                out[ir] += factor * this->fft_bundle.get_auxr_data<FPTYPE>()[ir].real();
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(FPTYPE))
#endif
            for(int ir = 0 ; ir < this->nrxx ; ++ir)
            {
                out[ir] = this->fft_bundle.get_auxr_data<FPTYPE>()[ir].real();
            }
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

template void PW_Basis::real2recip<float>(const float* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::real2recip<float>(const std::complex<float>* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::recip2real<float>(const std::complex<float>* in,
                                          float* out,
                                          const bool add,
                                          const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis::recip2real<float>(const std::complex<float>* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const;

template void PW_Basis::real2recip<double>(const double* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::real2recip<double>(const std::complex<double>* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::recip2real<double>(const std::complex<double>* in,
                                           double* out,
                                           const bool add,
                                           const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis::recip2real<double>(const std::complex<double>* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const;

// ============================================================
// Device-aware FFT: CPU specializations (delegate to existing)
// ============================================================

template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_CPU* /*ctx*/,
                              const double* in,
                              std::complex<double>* out,
                              const bool add,
                              const double factor) const
{
    this->real2recip(in, out, add, factor);
}
template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_CPU* /*ctx*/,
                              const float* in,
                              std::complex<float>* out,
                              const bool add,
                              const float factor) const
{
    this->real2recip(in, out, add, factor);
}
template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<double>* in,
                              std::complex<double>* out,
                              const bool add,
                              const double factor) const
{
    this->real2recip(in, out, add, factor);
}
template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<float>* in,
                              std::complex<float>* out,
                              const bool add,
                              const float factor) const
{
    this->real2recip(in, out, add, factor);
}

template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<double>* in,
                              double* out,
                              const bool add,
                              const double factor) const
{
    this->recip2real(in, out, add, factor);
}
template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<float>* in,
                              float* out,
                              const bool add,
                              const float factor) const
{
    this->recip2real(in, out, add, factor);
}
template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<double>* in,
                              std::complex<double>* out,
                              const bool add,
                              const double factor) const
{
    this->recip2real(in, out, add, factor);
}
template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_CPU* /*ctx*/,
                              const std::complex<float>* in,
                              std::complex<float>* out,
                              const bool add,
                              const float factor) const
{
    this->recip2real(in, out, add, factor);
}

// ============================================================
// Device-aware FFT: GPU specializations (3D cuFFT)
// ============================================================
#if defined(__CUDA) || defined(__ROCM)

// real_to_recip: complex input -> complex output (GPU)
template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_GPU* ctx,
                              const std::complex<double>* in,
                              std::complex<double>* out,
                              const bool add,
                              const double factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->gpu_fft_bundle != nullptr);

    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(ctx, ctx,
                                                   this->gpu_fft_bundle->get_auxr_3d_data<double>(),
                                                   in, this->nrxx);

    this->gpu_fft_bundle->fft3D_forward(ctx,
        this->gpu_fft_bundle->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle->get_auxr_3d_data<double>());

    set_real_to_recip_output_op<double, base_device::DEVICE_GPU>()(ctx,
        this->npw, this->nxyz, add, factor,
        this->ig2ixyz,
        this->gpu_fft_bundle->get_auxr_3d_data<double>(),
        out);

    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_GPU* ctx,
                              const std::complex<float>* in,
                              std::complex<float>* out,
                              const bool add,
                              const float factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->gpu_fft_bundle != nullptr);

    base_device::memory::synchronize_memory_op<std::complex<float>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(ctx, ctx,
                                                   this->gpu_fft_bundle->get_auxr_3d_data<float>(),
                                                   in, this->nrxx);

    this->gpu_fft_bundle->fft3D_forward(ctx,
        this->gpu_fft_bundle->get_auxr_3d_data<float>(),
        this->gpu_fft_bundle->get_auxr_3d_data<float>());

    set_real_to_recip_output_op<float, base_device::DEVICE_GPU>()(ctx,
        this->npw, this->nxyz, add, factor,
        this->ig2ixyz,
        this->gpu_fft_bundle->get_auxr_3d_data<float>(),
        out);

    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

// real_to_recip: real input -> complex output (GPU)
// Not yet implemented for GPU — real-input FFT on GPU requires a real-to-complex kernel.
// For GPU XC path, convert real to complex on GPU before calling the complex overload.
template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_GPU* /*ctx*/,
                              const double* /*in*/,
                              std::complex<double>* /*out*/,
                              const bool /*add*/,
                              const double /*factor*/) const
{
    ModuleBase::WARNING_QUIT("PW_Basis::real_to_recip",
        "GPU real(double)->recip not implemented. Use complex input overload.");
}

template <>
void PW_Basis::real_to_recip(const base_device::DEVICE_GPU* /*ctx*/,
                              const float* /*in*/,
                              std::complex<float>* /*out*/,
                              const bool /*add*/,
                              const float /*factor*/) const
{
    ModuleBase::WARNING_QUIT("PW_Basis::real_to_recip",
        "GPU real(float)->recip not implemented. Use complex input overload.");
}

// recip_to_real: complex input -> complex output (GPU)
template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_GPU* ctx,
                              const std::complex<double>* in,
                              std::complex<double>* out,
                              const bool add,
                              const double factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->gpu_fft_bundle != nullptr);

    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        ctx, this->gpu_fft_bundle->get_auxr_3d_data<double>(), 0, this->nxyz);

    set_3d_fft_box_op<double, base_device::DEVICE_GPU>()(ctx,
        this->npw, this->ig2ixyz, in,
        this->gpu_fft_bundle->get_auxr_3d_data<double>());

    this->gpu_fft_bundle->fft3D_backward(ctx,
        this->gpu_fft_bundle->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle->get_auxr_3d_data<double>());

    set_recip_to_real_output_op<double, base_device::DEVICE_GPU>()(ctx,
        this->nrxx, add, factor,
        this->gpu_fft_bundle->get_auxr_3d_data<double>(),
        out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_GPU* ctx,
                              const std::complex<float>* in,
                              std::complex<float>* out,
                              const bool add,
                              const float factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->gpu_fft_bundle != nullptr);

    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(
        ctx, this->gpu_fft_bundle->get_auxr_3d_data<float>(), 0, this->nxyz);

    set_3d_fft_box_op<float, base_device::DEVICE_GPU>()(ctx,
        this->npw, this->ig2ixyz, in,
        this->gpu_fft_bundle->get_auxr_3d_data<float>());

    this->gpu_fft_bundle->fft3D_backward(ctx,
        this->gpu_fft_bundle->get_auxr_3d_data<float>(),
        this->gpu_fft_bundle->get_auxr_3d_data<float>());

    set_recip_to_real_output_op<float, base_device::DEVICE_GPU>()(ctx,
        this->nrxx, add, factor,
        this->gpu_fft_bundle->get_auxr_3d_data<float>(),
        out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

// recip_to_real: complex input -> real output (GPU)
// Not yet implemented — extracting real part on GPU requires a dedicated kernel.
template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_GPU* /*ctx*/,
                              const std::complex<double>* /*in*/,
                              double* /*out*/,
                              const bool /*add*/,
                              const double /*factor*/) const
{
    ModuleBase::WARNING_QUIT("PW_Basis::recip_to_real",
        "GPU recip->real(double) not implemented. Use complex output overload.");
}

template <>
void PW_Basis::recip_to_real(const base_device::DEVICE_GPU* /*ctx*/,
                              const std::complex<float>* /*in*/,
                              float* /*out*/,
                              const bool /*add*/,
                              const float /*factor*/) const
{
    ModuleBase::WARNING_QUIT("PW_Basis::recip_to_real",
        "GPU recip->real(float) not implemented. Use complex output overload.");
}

#endif // __CUDA || __ROCM

}