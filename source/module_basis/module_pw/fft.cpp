#include "fft.h"
#if defined(__CUFFTDX)
#include "kernels/fused_veff.h"
#endif

#include "module_base/memory.h"
#include "module_base/tool_quit.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace ModulePW
{

FFT::FFT()
{
}

FFT::~FFT()
{
    this->clear();
}
void FFT::clear()
{
    this->cleanFFT();
    if (z_auxg != nullptr)
    {
        fftw_free(z_auxg);
        z_auxg = nullptr;
    }
    if (z_auxr != nullptr)
    {
        fftw_free(z_auxr);
        z_auxr = nullptr;
    }
    d_rspace = nullptr;
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
        if (c_auxr_3d != nullptr)
        {
            delmem_cd_op()(gpu_ctx, c_auxr_3d);
            c_auxr_3d = nullptr;
        }
        if (z_auxr_3d != nullptr)
        {
            delmem_zd_op()(gpu_ctx, z_auxr_3d);
            z_auxr_3d = nullptr;
        }
    }
#endif // defined(__CUDA) || defined(__ROCM)
#if defined(__ENABLE_FLOAT_FFTW)
    if (this->precision == "single")
    {
        this->cleanfFFT();
        if (c_auxg != nullptr)
        {
            fftw_free(c_auxg);
            c_auxg = nullptr;
        }
        if (c_auxr != nullptr)
        {
            fftw_free(c_auxr);
            c_auxr = nullptr;
        }
        s_rspace = nullptr;
    }
#endif // defined(__ENABLE_FLOAT_FFTW)
}

void FFT::initfft(int nx_in, int ny_in, int nz_in, int lixy_in, int rixy_in, int ns_in, int nplane_in, int nproc_in,
                  bool gamma_only_in, bool xprime_in, bool mpifft_in)
{
    this->gamma_only = gamma_only_in;
    this->xprime = xprime_in;
    this->fftnx = this->nx = nx_in;
    this->fftny = this->ny = ny_in;
    if (this->gamma_only)
    {
        if (xprime)
            this->fftnx = int(nx / 2) + 1;
        else
            this->fftny = int(ny / 2) + 1;
    }
    this->nz = nz_in;
    this->ns = ns_in;
    this->lixy = lixy_in;
    this->rixy = rixy_in;
    this->nplane = nplane_in;
    this->nproc = nproc_in;
    this->mpifft = mpifft_in;
    this->nxy = this->nx * this->ny;
    this->fftnxy = this->fftnx * this->fftny;
    // this->maxgrids = (this->nz * this->ns > this->nxy * nplane) ? this->nz * this->ns : this->nxy * nplane;
    const int nrxx = this->nxy * this->nplane;
    const int nsz = this->nz * this->ns;
    int maxgrids = (nsz > nrxx) ? nsz : nrxx;
    if (!this->mpifft)
    {
        z_auxg = (std::complex<double>*)fftw_malloc(sizeof(fftw_complex) * maxgrids);
        z_auxr = (std::complex<double>*)fftw_malloc(sizeof(fftw_complex) * maxgrids);
        ModuleBase::Memory::record("FFT::grid", 2 * sizeof(fftw_complex) * maxgrids);
        d_rspace = (double*)z_auxg;
        // auxr_3d = static_cast<std::complex<double> *>(
        //     fftw_malloc(sizeof(fftw_complex) * (this->nx * this->ny * this->nz)));
#if defined(__ENABLE_FLOAT_FFTW)
        if (this->precision == "single")
        {
            c_auxg = (std::complex<float>*)fftw_malloc(sizeof(fftwf_complex) * maxgrids);
            c_auxr = (std::complex<float>*)fftw_malloc(sizeof(fftwf_complex) * maxgrids);
            ModuleBase::Memory::record("FFT::grid_s", 2 * sizeof(fftwf_complex) * maxgrids);
            s_rspace = (float*)c_auxg;
        }
#endif // defined(__ENABLE_FLOAT_FFTW)
    }
    else
    {
    }
}

void FFT::setupFFT()
{
    unsigned int flag = FFTW_ESTIMATE;
    switch (this->fft_mode)
    {
    case 0:
        flag = FFTW_ESTIMATE;
        break;
    case 1:
        flag = FFTW_MEASURE;
        break;
    case 2:
        flag = FFTW_PATIENT;
        break;
    case 3:
        flag = FFTW_EXHAUSTIVE;
        break;
    default:
        break;
    }
    if (!this->mpifft)
    {
        this->initplan(flag);
#if defined(__ENABLE_FLOAT_FFTW)
        if (this->precision == "single")
        {
            this->initplanf(flag);
        }
#endif // defined(__ENABLE_FLOAT_FFTW)
    }
#if defined(__FFTW3_MPI) && defined(__MPI)
    else
    {
        // this->initplan_mpi();
        // if (this->precision == "single") {
        //     this->initplanf_mpi();
        // }
    }
#endif
    return;
}

void FFT ::initplan(const unsigned int& flag)
{
    //---------------------------------------------------------
    //                              1 D - Z
    //---------------------------------------------------------

    //               fftw_plan_many_dft(int rank,          const int *n,       int howmany,
    //					                fftw_complex *in,  const int *inembed, int istride, int idist,
    //					                fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned
    //flags);

    this->planzfor = fftw_plan_many_dft(1, &this->nz, this->ns, (fftw_complex*)z_auxg, &this->nz, 1, this->nz,
                                        (fftw_complex*)z_auxg, &this->nz, 1, this->nz, FFTW_FORWARD, flag);

    this->planzbac = fftw_plan_many_dft(1, &this->nz, this->ns, (fftw_complex*)z_auxg, &this->nz, 1, this->nz,
                                        (fftw_complex*)z_auxg, &this->nz, 1, this->nz, FFTW_BACKWARD, flag);

    //---------------------------------------------------------
    //                              2 D - XY
    //---------------------------------------------------------
    // 1D+1D is much faster than 2D FFT!
    // in-place fft is better for c2c and out-of-place fft is better for c2r
    int* embed = nullptr;
    int npy = this->nplane * this->ny;
    // Batch the independent x slabs and z planes in one y-transform plan.
    // The array layout is [x][y][z], with z distributed across MPI ranks.
    auto plan_y = [&](int first_x, int count_x, int sign) {
        fftw_iodim transform = {this->ny, this->nplane, this->nplane};
        fftw_iodim batches[2] = {{this->nplane, 1, 1}, {count_x, npy, npy}};
        auto* data = reinterpret_cast<fftw_complex*>(z_auxr + first_x * npy);
        return fftw_plan_guru_dft(1, &transform, 2, batches, data, data, sign, flag);
    };
    if (this->xprime)
    {
        this->planyfor = plan_y(0, this->lixy + 1, FFTW_FORWARD);
        this->planybac = plan_y(0, this->lixy + 1, FFTW_BACKWARD);
        if (!this->gamma_only && this->rixy < this->nx)
        {
            this->planyfor2 = plan_y(this->rixy, this->nx - this->rixy, FFTW_FORWARD);
            this->planybac2 = plan_y(this->rixy, this->nx - this->rixy, FFTW_BACKWARD);
        }
        if (this->gamma_only)
        {
            this->planxr2c = fftw_plan_many_dft_r2c(1, &this->nx, npy, d_rspace, embed, npy, 1, (fftw_complex*)z_auxr,
                                                    embed, npy, 1, flag);
            this->planxc2r = fftw_plan_many_dft_c2r(1, &this->nx, npy, (fftw_complex*)z_auxr, embed, npy, 1, d_rspace,
                                                    embed, npy, 1, flag);
        }
        else
        {
            this->planxfor1 = fftw_plan_many_dft(1, &this->nx, npy, (fftw_complex*)z_auxr, embed, npy, 1,
                                                 (fftw_complex*)z_auxr, embed, npy, 1, FFTW_FORWARD, flag);
            this->planxbac1 = fftw_plan_many_dft(1, &this->nx, npy, (fftw_complex*)z_auxr, embed, npy, 1,
                                                 (fftw_complex*)z_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
        }
    }
    else
    {
        this->planxfor1 = fftw_plan_many_dft(1, &this->nx, this->nplane * (lixy + 1), (fftw_complex*)z_auxr, embed, npy,
                                             1, (fftw_complex*)z_auxr, embed, npy, 1, FFTW_FORWARD, flag);
        this->planxbac1 = fftw_plan_many_dft(1, &this->nx, this->nplane * (lixy + 1), (fftw_complex*)z_auxr, embed, npy,
                                             1, (fftw_complex*)z_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
        if (this->gamma_only)
        {
            this->planyr2c = fftw_plan_many_dft_r2c(1, &this->ny, this->nplane, d_rspace, embed, this->nplane, 1,
                                                    (fftw_complex*)z_auxr, embed, this->nplane, 1, flag);
            this->planyc2r = fftw_plan_many_dft_c2r(1, &this->ny, this->nplane, (fftw_complex*)z_auxr, embed,
                                                    this->nplane, 1, d_rspace, embed, this->nplane, 1, flag);
        }
        else
        {

            this->planxfor2 = fftw_plan_many_dft(1, &this->nx, this->nplane * (ny - rixy), (fftw_complex*)z_auxr, embed,
                                                 npy, 1, (fftw_complex*)z_auxr, embed, npy, 1, FFTW_FORWARD, flag);
            this->planxbac2 = fftw_plan_many_dft(1, &this->nx, this->nplane * (ny - rixy), (fftw_complex*)z_auxr, embed,
                                                 npy, 1, (fftw_complex*)z_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
            this->planyfor = plan_y(0, this->nx, FFTW_FORWARD);
            this->planybac = plan_y(0, this->nx, FFTW_BACKWARD);
        }
    }

    //---------------------------------------------------------
    //                              3 D - XYZ
    //---------------------------------------------------------
    // in-place fft test
    // this->plan3dforward = fftw_plan_dft_3d(
    //    this->nx, this->ny, this->nz,
    //    reinterpret_cast<fftw_complex *>(auxr_3d),
    //    reinterpret_cast<fftw_complex *>(auxr_3d),
    //    FFTW_FORWARD, flag);
    // this->plan3dbackward = fftw_plan_dft_3d(
    //    this->nx, this->ny, this->nz,
    //    reinterpret_cast<fftw_complex *>(auxr_3d),
    //    reinterpret_cast<fftw_complex *>(auxr_3d),
    //    FFTW_BACKWARD, flag);

}

#if defined(__ENABLE_FLOAT_FFTW)
void FFT ::initplanf(const unsigned int& flag)
{
    //---------------------------------------------------------
    //                              1 D
    //---------------------------------------------------------

    //               fftw_plan_many_dft(int rank,          const int *n,       int howmany,
    //					                fftw_complex *in,  const int *inembed, int istride, int idist,
    //					                fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned
    //flags);

    this->planfzfor = fftwf_plan_many_dft(1, &this->nz, this->ns, (fftwf_complex*)c_auxg, &this->nz, 1, this->nz,
                                          (fftwf_complex*)c_auxg, &this->nz, 1, this->nz, FFTW_FORWARD, flag);

    this->planfzbac = fftwf_plan_many_dft(1, &this->nz, this->ns, (fftwf_complex*)c_auxg, &this->nz, 1, this->nz,
                                          (fftwf_complex*)c_auxg, &this->nz, 1, this->nz, FFTW_BACKWARD, flag);
    //---------------------------------------------------------
    //                              2 D
    //---------------------------------------------------------

    int* embed = nullptr;
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        this->planfyfor = fftwf_plan_many_dft(1, &this->ny, this->nplane, (fftwf_complex*)c_auxr, embed, nplane, 1,
                                              (fftwf_complex*)c_auxr, embed, nplane, 1, FFTW_FORWARD, flag);
        this->planfybac = fftwf_plan_many_dft(1, &this->ny, this->nplane, (fftwf_complex*)c_auxr, embed, nplane, 1,
                                              (fftwf_complex*)c_auxr, embed, nplane, 1, FFTW_BACKWARD, flag);
        if (this->gamma_only)
        {
            this->planfxr2c = fftwf_plan_many_dft_r2c(1, &this->nx, npy, s_rspace, embed, npy, 1,
                                                      (fftwf_complex*)c_auxr, embed, npy, 1, flag);
            this->planfxc2r = fftwf_plan_many_dft_c2r(1, &this->nx, npy, (fftwf_complex*)c_auxr, embed, npy, 1,
                                                      s_rspace, embed, npy, 1, flag);
        }
        else
        {
            this->planfxfor1 = fftwf_plan_many_dft(1, &this->nx, npy, (fftwf_complex*)c_auxr, embed, npy, 1,
                                                   (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_FORWARD, flag);
            this->planfxbac1 = fftwf_plan_many_dft(1, &this->nx, npy, (fftwf_complex*)c_auxr, embed, npy, 1,
                                                   (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
        }
    }
    else
    {
        this->planfxfor1 = fftwf_plan_many_dft(1, &this->nx, this->nplane * (lixy + 1), (fftwf_complex*)c_auxr, embed,
                                               npy, 1, (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_FORWARD, flag);
        this->planfxbac1 = fftwf_plan_many_dft(1, &this->nx, this->nplane * (lixy + 1), (fftwf_complex*)c_auxr, embed,
                                               npy, 1, (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
        if (this->gamma_only)
        {
            this->planfyr2c = fftwf_plan_many_dft_r2c(1, &this->ny, this->nplane, s_rspace, embed, this->nplane, 1,
                                                      (fftwf_complex*)c_auxr, embed, this->nplane, 1, flag);
            this->planfyc2r = fftwf_plan_many_dft_c2r(1, &this->ny, this->nplane, (fftwf_complex*)c_auxr, embed,
                                                      this->nplane, 1, s_rspace, embed, this->nplane, 1, flag);
        }
        else
        {
            this->planfxfor2
                = fftwf_plan_many_dft(1, &this->nx, this->nplane * (this->ny - rixy), (fftwf_complex*)c_auxr, embed,
                                      npy, 1, (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_FORWARD, flag);
            this->planfxbac2
                = fftwf_plan_many_dft(1, &this->nx, this->nplane * (this->ny - rixy), (fftwf_complex*)c_auxr, embed,
                                      npy, 1, (fftwf_complex*)c_auxr, embed, npy, 1, FFTW_BACKWARD, flag);
            this->planfyfor
                = fftwf_plan_many_dft(1, &this->ny, this->nplane, (fftwf_complex*)c_auxr, embed, this->nplane, 1,
                                      (fftwf_complex*)c_auxr, embed, this->nplane, 1, FFTW_FORWARD, flag);
            this->planfybac
                = fftwf_plan_many_dft(1, &this->ny, this->nplane, (fftwf_complex*)c_auxr, embed, this->nplane, 1,
                                      (fftwf_complex*)c_auxr, embed, this->nplane, 1, FFTW_BACKWARD, flag);
        }
    }
}
#endif // defined(__ENABLE_FLOAT_FFTW)
// void FFT :: initplan_mpi()
// {

// }

// void FFT :: initplanf_mpi()
// {

// }

void FFT::cleanFFT()
{
    if (planzfor)
    {
        fftw_destroy_plan(planzfor);
        planzfor = NULL;
    }
    if (planzbac)
    {
        fftw_destroy_plan(planzbac);
        planzbac = NULL;
    }
    if (planxfor1)
    {
        fftw_destroy_plan(planxfor1);
        planxfor1 = NULL;
    }
    if (planxbac1)
    {
        fftw_destroy_plan(planxbac1);
        planxbac1 = NULL;
    }
    if (planxfor2)
    {
        fftw_destroy_plan(planxfor2);
        planxfor2 = NULL;
    }
    if (planxbac2)
    {
        fftw_destroy_plan(planxbac2);
        planxbac2 = NULL;
    }
    if (planyfor)
    {
        fftw_destroy_plan(planyfor);
        planyfor = NULL;
    }
    if (planybac)
    {
        fftw_destroy_plan(planybac);
        planybac = NULL;
    }
    if (planyfor2)
    {
        fftw_destroy_plan(planyfor2);
        planyfor2 = NULL;
    }
    if (planybac2)
    {
        fftw_destroy_plan(planybac2);
        planybac2 = NULL;
    }
    if (planxr2c)
    {
        fftw_destroy_plan(planxr2c);
        planxr2c = NULL;
    }
    if (planxc2r)
    {
        fftw_destroy_plan(planxc2r);
        planxc2r = NULL;
    }
    if (planyr2c)
    {
        fftw_destroy_plan(planyr2c);
        planyr2c = NULL;
    }
    if (planyc2r)
    {
        fftw_destroy_plan(planyc2r);
        planyc2r = NULL;
    }

    // fftw_destroy_plan(this->plan3dforward);
    // fftw_destroy_plan(this->plan3dbackward);
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
#if defined(__CUDA)
        if (c_handle)
        {
            cufftDestroy(c_handle);
            c_handle = {};
        }
        if (z_handle)
        {
            cufftDestroy(z_handle);
            z_handle = {};
        }
        for (auto* handle : {&z_x_handle, &z_yz_left_handle, &z_yz_right_handle})
        {
            if (*handle)
            {
                cufftDestroy(*handle);
                *handle = {};
            }
        }
#elif defined(__ROCM)
        if (c_handle)
        {
            hipfftDestroy(c_handle);
            c_handle = {};
        }
        if (z_handle)
        {
            hipfftDestroy(z_handle);
            z_handle = {};
        }
#endif
    }
#endif
}

#if defined(__ENABLE_FLOAT_FFTW)
void FFT::cleanfFFT()
{
    if (planfzfor)
    {
        fftwf_destroy_plan(planfzfor);
        planfzfor = NULL;
    }
    if (planfzbac)
    {
        fftwf_destroy_plan(planfzbac);
        planfzbac = NULL;
    }
    if (planfxfor1)
    {
        fftwf_destroy_plan(planfxfor1);
        planfxfor1 = NULL;
    }
    if (planfxbac1)
    {
        fftwf_destroy_plan(planfxbac1);
        planfxbac1 = NULL;
    }
    if (planfxfor2)
    {
        fftwf_destroy_plan(planfxfor2);
        planfxfor2 = NULL;
    }
    if (planfxbac2)
    {
        fftwf_destroy_plan(planfxbac2);
        planfxbac2 = NULL;
    }
    if (planfyfor)
    {
        fftwf_destroy_plan(planfyfor);
        planfyfor = NULL;
    }
    if (planfybac)
    {
        fftwf_destroy_plan(planfybac);
        planfybac = NULL;
    }
    if (planfxr2c)
    {
        fftwf_destroy_plan(planfxr2c);
        planfxr2c = NULL;
    }
    if (planfxc2r)
    {
        fftwf_destroy_plan(planfxc2r);
        planfxc2r = NULL;
    }
    if (planfyr2c)
    {
        fftwf_destroy_plan(planfyr2c);
        planfyr2c = NULL;
    }
    if (planfyc2r)
    {
        fftwf_destroy_plan(planfyc2r);
        planfyc2r = NULL;
    }
    return;
}
#endif // defined(__ENABLE_FLOAT_FFTW)

template <>
void FFT::fftzfor(std::complex<float>* in, std::complex<float>* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    fftwf_execute_dft(this->planfzfor, (fftwf_complex*)in, (fftwf_complex*)out);
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftzfor(std::complex<double>* in, std::complex<double>* out) const
{
    fftw_execute_dft(this->planzfor, (fftw_complex*)in, (fftw_complex*)out);
}

template <>
void FFT::fftzbac(std::complex<float>* in, std::complex<float>* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    fftwf_execute_dft(this->planfzbac, (fftwf_complex*)in, (fftwf_complex*)out);
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftzbac(std::complex<double>* in, std::complex<double>* out) const
{
    fftw_execute_dft(this->planzbac, (fftw_complex*)in, (fftw_complex*)out);
}

template <>
void FFT::fftxyfor(std::complex<float>* in, std::complex<float>* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftwf_execute_dft(this->planfxfor1, (fftwf_complex*)in, (fftwf_complex*)out);

        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftwf_execute_dft(this->planfyfor, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }
        for (int i = rixy; i < this->nx; ++i)
        {
            fftwf_execute_dft(this->planfyfor, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }
    }
    else
    {
        for (int i = 0; i < this->nx; ++i)
        {
            fftwf_execute_dft(this->planfyfor, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }

        fftwf_execute_dft(this->planfxfor1, (fftwf_complex*)in, (fftwf_complex*)out);
        fftwf_execute_dft(this->planfxfor2, (fftwf_complex*)&in[rixy * nplane], (fftwf_complex*)&out[rixy * nplane]);
    }
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftxyfor(std::complex<double>* in, std::complex<double>* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftw_execute_dft(this->planxfor1, (fftw_complex*)in, (fftw_complex*)out);

        fftw_execute_dft(this->planyfor, (fftw_complex*)in, (fftw_complex*)out);
        if (this->planyfor2)
        {
            fftw_execute_dft(this->planyfor2, (fftw_complex*)&in[rixy * npy], (fftw_complex*)&out[rixy * npy]);
        }
    }
    else
    {
        fftw_execute_dft(this->planyfor, (fftw_complex*)in, (fftw_complex*)out);

        fftw_execute_dft(this->planxfor1, (fftw_complex*)in, (fftw_complex*)out);
        fftw_execute_dft(this->planxfor2, (fftw_complex*)&in[rixy * nplane], (fftw_complex*)&out[rixy * nplane]);
    }
}

template <>
void FFT::fftxybac(std::complex<float>* in, std::complex<float>* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftwf_execute_dft(this->planfybac, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }
        for (int i = rixy; i < this->nx; ++i)
        {
            fftwf_execute_dft(this->planfybac, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }

        fftwf_execute_dft(this->planfxbac1, (fftwf_complex*)in, (fftwf_complex*)out);
    }
    else
    {
        fftwf_execute_dft(this->planfxbac1, (fftwf_complex*)in, (fftwf_complex*)out);
        fftwf_execute_dft(this->planfxbac2, (fftwf_complex*)&in[rixy * nplane], (fftwf_complex*)&out[rixy * nplane]);

        for (int i = 0; i < this->nx; ++i)
        {
            fftwf_execute_dft(this->planfybac, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&out[i * npy]);
        }
    }
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftxybac(std::complex<double>* in, std::complex<double>* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftw_execute_dft(this->planybac, (fftw_complex*)in, (fftw_complex*)out);
        if (this->planybac2)
        {
            fftw_execute_dft(this->planybac2, (fftw_complex*)&in[rixy * npy], (fftw_complex*)&out[rixy * npy]);
        }

        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)out);
    }
    else
    {
        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)out);
        fftw_execute_dft(this->planxbac2, (fftw_complex*)&in[rixy * nplane], (fftw_complex*)&out[rixy * nplane]);

        fftw_execute_dft(this->planybac, (fftw_complex*)in, (fftw_complex*)out);
    }
}

template <>
void FFT::fftxyr2c(float* in, std::complex<float>* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftwf_execute_dft_r2c(this->planfxr2c, in, (fftwf_complex*)out);

        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftwf_execute_dft(this->planfyfor, (fftwf_complex*)&out[i * npy], (fftwf_complex*)&out[i * npy]);
        }
    }
    else
    {
        for (int i = 0; i < this->nx; ++i)
        {
            fftwf_execute_dft_r2c(this->planfyr2c, &in[i * npy], (fftwf_complex*)&out[i * npy]);
        }

        fftwf_execute_dft(this->planfxfor1, (fftwf_complex*)out, (fftwf_complex*)out);
    }
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftxyr2c(double* in, std::complex<double>* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftw_execute_dft_r2c(this->planxr2c, in, (fftw_complex*)out);

        fftw_execute_dft(this->planyfor, (fftw_complex*)out, (fftw_complex*)out);
    }
    else
    {
        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft_r2c(this->planyr2c, &in[i * npy], (fftw_complex*)&out[i * npy]);
        }

        fftw_execute_dft(this->planxfor1, (fftw_complex*)out, (fftw_complex*)out);
    }
}

template <>
void FFT::fftxyc2r(std::complex<float>* in, float* out) const
{
#if defined(__ENABLE_FLOAT_FFTW)
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftwf_execute_dft(this->planfybac, (fftwf_complex*)&in[i * npy], (fftwf_complex*)&in[i * npy]);
        }

        fftwf_execute_dft_c2r(this->planfxc2r, (fftwf_complex*)in, out);
    }
    else
    {
        fftwf_execute_dft(this->planfxbac1, (fftwf_complex*)in, (fftwf_complex*)in);

        for (int i = 0; i < this->nx; ++i)
        {
            fftwf_execute_dft_c2r(this->planfyc2r, (fftwf_complex*)&in[i * npy], &out[i * npy]);
        }
    }
#else
    ModuleBase::WARNING_QUIT("fft", "Please compile ABACUS using the ENABLE_FLOAT_FFTW flag!");
#endif // defined(__ENABLE_FLOAT_FFTW)
}

template <>
void FFT::fftxyc2r(std::complex<double>* in, double* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftw_execute_dft(this->planybac, (fftw_complex*)in, (fftw_complex*)in);

        fftw_execute_dft_c2r(this->planxc2r, (fftw_complex*)in, out);
    }
    else
    {
        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)in);

        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft_c2r(this->planyc2r, (fftw_complex*)&in[i * npy], &out[i * npy]);
        }
    }
}

#if defined(__CUDA)
bool FFT::use_pruned_gpu_fft() const
{
    // Small grids are dominated by the extra kernel launches. Sparse large
    // grids avoid the y-z transforms in the unoccupied middle x slabs.
    return this->xprime && !this->gamma_only && this->nproc == 1
           && this->nx * this->ny * this->nz >= 1048576
           && 4 * (this->lixy + 1 + this->nx - this->rixy) <= 3 * this->nx;
}

void FFT::init_gpu_pruned_plan() const
{
    if (this->z_x_handle)
    {
        return;
    }
    int x_size = this->nx;
    int yz_size[2] = {this->ny, this->nz};
    const int plane = this->ny * this->nz;
    CHECK_CUFFT(cufftPlanMany(&this->z_x_handle, 1, &x_size, &x_size, plane, 1,
                             &x_size, plane, 1, CUFFT_Z2Z, plane));
    CHECK_CUFFT(cufftPlanMany(&this->z_yz_left_handle, 2, yz_size, yz_size, 1, plane,
                             yz_size, 1, plane, CUFFT_Z2Z, this->lixy + 1));
    if (this->rixy < this->nx)
    {
        CHECK_CUFFT(cufftPlanMany(&this->z_yz_right_handle, 2, yz_size, yz_size, 1, plane,
                                 yz_size, 1, plane, CUFFT_Z2Z, this->nx - this->rixy));
    }
}
#endif

#if defined(__CUDA) || defined(__ROCM)
template <>
void FFT::init_gpu_plan<float>() const
{
    if (!c_handle)
    {
#if defined(__CUDA)
        CHECK_CUFFT(cufftPlan3d(&c_handle, this->nx, this->ny, this->nz, CUFFT_C2C));
#elif defined(__ROCM)
        CHECK_CUFFT(hipfftPlan3d(&c_handle, this->nx, this->ny, this->nz, HIPFFT_C2C));
#endif
    }
}

template <>
void FFT::init_gpu_plan<double>() const
{
    if (!z_handle)
    {
#if defined(__CUDA)
        CHECK_CUFFT(cufftPlan3d(&z_handle, this->nx, this->ny, this->nz, CUFFT_Z2Z));
#elif defined(__ROCM)
        CHECK_CUFFT(hipfftPlan3d(&z_handle, this->nx, this->ny, this->nz, HIPFFT_Z2Z));
#endif
    }
}

template <>
void FFT::fft3D_forward(const base_device::DEVICE_GPU* /*ctx*/, std::complex<float>* in, std::complex<float>* out) const
{
    this->init_gpu_plan<float>();
#if defined(__CUDA)
    CHECK_CUFFT(cufftExecC2C(this->c_handle, reinterpret_cast<cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
#elif defined(__ROCM)
    CHECK_CUFFT(hipfftExecC2C(this->c_handle, reinterpret_cast<hipfftComplex*>(in),
                              reinterpret_cast<hipfftComplex*>(out), HIPFFT_FORWARD));
#endif
}
template <>
void FFT::fft3D_forward(const base_device::DEVICE_GPU* /*ctx*/, std::complex<double>* in,
                        std::complex<double>* out) const
{
    this->init_gpu_plan<double>();
#if defined(__CUDA)
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle, reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out), CUFFT_FORWARD));
#elif defined(__ROCM)
    CHECK_CUFFT(hipfftExecZ2Z(this->z_handle, reinterpret_cast<hipfftDoubleComplex*>(in),
                              reinterpret_cast<hipfftDoubleComplex*>(out), HIPFFT_FORWARD));
#endif
}

template <>
void FFT::fft3D_backward(const base_device::DEVICE_GPU* /*ctx*/, std::complex<float>* in,
                         std::complex<float>* out) const
{
    this->init_gpu_plan<float>();
#if defined(__CUDA)
    CHECK_CUFFT(cufftExecC2C(this->c_handle, reinterpret_cast<cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
#elif defined(__ROCM)
    CHECK_CUFFT(hipfftExecC2C(this->c_handle, reinterpret_cast<hipfftComplex*>(in),
                              reinterpret_cast<hipfftComplex*>(out), HIPFFT_BACKWARD));
#endif
}
template <>
void FFT::fft3D_backward(const base_device::DEVICE_GPU* /*ctx*/, std::complex<double>* in,
                         std::complex<double>* out) const
{
    this->init_gpu_plan<double>();
#if defined(__CUDA)
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle, reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out), CUFFT_INVERSE));
#elif defined(__ROCM)
    CHECK_CUFFT(hipfftExecZ2Z(this->z_handle, reinterpret_cast<hipfftDoubleComplex*>(in),
                              reinterpret_cast<hipfftDoubleComplex*>(out), HIPFFT_BACKWARD));
#endif
}
#endif

#if defined(__CUDA)
void FFT::fft3D_forward_pw(std::complex<double>* in, std::complex<double>* out) const
{
    if (!this->use_pruned_gpu_fft())
    {
        this->fft3D_forward(gpu_ctx, in, out);
        return;
    }
    this->init_gpu_pruned_plan();
    auto* input = reinterpret_cast<cufftDoubleComplex*>(in);
    auto* output = reinterpret_cast<cufftDoubleComplex*>(out);
    CHECK_CUFFT(cufftExecZ2Z(this->z_x_handle, input, output, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecZ2Z(this->z_yz_left_handle, output, output, CUFFT_FORWARD));
    if (this->z_yz_right_handle)
    {
        auto* right = output + this->rixy * this->ny * this->nz;
        CHECK_CUFFT(cufftExecZ2Z(this->z_yz_right_handle, right, right, CUFFT_FORWARD));
    }
}

void FFT::fft3D_backward_pw(std::complex<double>* in, std::complex<double>* out) const
{
    if (!this->use_pruned_gpu_fft())
    {
        this->fft3D_backward(gpu_ctx, in, out);
        return;
    }
    this->init_gpu_pruned_plan();
    auto* input = reinterpret_cast<cufftDoubleComplex*>(in);
    auto* output = reinterpret_cast<cufftDoubleComplex*>(out);
    CHECK_CUFFT(cufftExecZ2Z(this->z_yz_left_handle, input, input, CUFFT_INVERSE));
    if (this->z_yz_right_handle)
    {
        auto* right = input + this->rixy * this->ny * this->nz;
        CHECK_CUFFT(cufftExecZ2Z(this->z_yz_right_handle, right, right, CUFFT_INVERSE));
    }
    CHECK_CUFFT(cufftExecZ2Z(this->z_x_handle, input, output, CUFFT_INVERSE));
}
#endif

#if defined(__CUFFTDX)
int FFT::fused_veff_grid_size() const
{
    return this->nx == 200 && this->use_pruned_gpu_fft()
               ? (this->lixy + 1 + this->nx - this->rixy) * this->ny * this->nz : 0;
}

void FFT::apply_spinor_potential(const double* potential, std::complex<double>* up,
                                std::complex<double>* down) const
{
    assert(this->fused_veff_grid_size() > 0);
    this->init_gpu_pruned_plan();
    const int plane = this->ny * this->nz;
    const int left = this->lixy + 1;
    auto transform_yz = [&](std::complex<double>* grid, int direction)
    {
        auto* data = reinterpret_cast<cufftDoubleComplex*>(grid);
        CHECK_CUFFT(cufftExecZ2Z(this->z_yz_left_handle, data, data, direction));
        if (this->z_yz_right_handle)
        {
            data += left * plane;
            CHECK_CUFFT(cufftExecZ2Z(this->z_yz_right_handle, data, data, direction));
        }
    };
    transform_yz(up, CUFFT_INVERSE);
    transform_yz(down, CUFFT_INVERSE);
    fused_veff_x(plane, left, this->rixy, potential, up, down);
    transform_yz(up, CUFFT_FORWARD);
    transform_yz(down, CUFFT_FORWARD);
}
#endif

template <>
float* FFT::get_rspace_data() const
{
    return this->s_rspace;
}
template <>
double* FFT::get_rspace_data() const
{
    return this->d_rspace;
}

template <>
std::complex<float>* FFT::get_auxr_data() const
{
    return this->c_auxr;
}
template <>
std::complex<double>* FFT::get_auxr_data() const
{
    return this->z_auxr;
}

template <>
std::complex<float>* FFT::get_auxg_data() const
{
    return this->c_auxg;
}
template <>
std::complex<double>* FFT::get_auxg_data() const
{
    return this->z_auxg;
}

#if defined(__CUDA) || defined(__ROCM)
template <>
std::complex<float>* FFT::get_auxr_3d_data() const
{
    if (this->c_auxr_3d == nullptr)
    {
        resmem_cd_op()(gpu_ctx, this->c_auxr_3d, this->nx * this->ny * this->nz);
    }
    return this->c_auxr_3d;
}
template <>
std::complex<double>* FFT::get_auxr_3d_data() const
{
    if (this->z_auxr_3d == nullptr)
    {
        resmem_zd_op()(gpu_ctx, this->z_auxr_3d, this->nx * this->ny * this->nz);
    }
    return this->z_auxr_3d;
}
#endif

void FFT::set_device(std::string device_)
{
    this->device = std::move(device_);
}

void FFT::set_precision(std::string precision_)
{
    this->precision = std::move(precision_);
}

} // namespace ModulePW
