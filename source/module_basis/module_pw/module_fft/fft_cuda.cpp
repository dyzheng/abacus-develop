#include "fft_cuda.h"
#include "module_base/memory.h"
#include "module_base/module_device/memory_op.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace ModulePW
{
template <typename FPTYPE>
void FFT_CUDA<FPTYPE>::initfft(int nx_in, 
                               int ny_in, 
                               int nz_in)
{
    this->nx = nx_in;
    this->ny = ny_in;
    this->nz = nz_in;
}
template <>
void FFT_CUDA<float>::setupFFT()
{
    cufftPlan3d(&c_handle, this->nx, this->ny, this->nz, CUFFT_C2C);
    resmem_cd_op()(gpu_ctx, this->c_auxr_3d, this->nx * this->ny * this->nz);
    ModuleBase::Memory::record_gpu("FFT3D::c_auxr", sizeof(std::complex<float>) * this->nx * this->ny * this->nz);
        
}
template <>  
void FFT_CUDA<double>::setupFFT()
{
    cufftPlan3d(&z_handle, this->nx, this->ny, this->nz, CUFFT_Z2Z);
    resmem_zd_op()(gpu_ctx, this->z_auxr_3d, this->nx * this->ny * this->nz);
    ModuleBase::Memory::record_gpu("FFT3D::z_auxr", sizeof(std::complex<double>) * this->nx * this->ny * this->nz);
}
template <>
void FFT_CUDA<float>::cleanFFT()
{
    if (c_handle)
    {
        cufftDestroy(c_handle);
        c_handle = {};
    }
    if (c_xy_handle)
    {
        cufftDestroy(c_xy_handle);
        c_xy_handle = {};
    }
    if (c_z_handle)
    {
        cufftDestroy(c_z_handle);
        c_z_handle = {};
    }
}
template <>
void FFT_CUDA<double>::cleanFFT()
{
    if (z_handle)
    {
        cufftDestroy(z_handle);
        z_handle = {};
    }
    if (z_xy_handle)
    {
        cufftDestroy(z_xy_handle);
        z_xy_handle = {};
    }
    if (z_z_handle)
    {
        cufftDestroy(z_z_handle);
        z_z_handle = {};
    }
}
template <>
void FFT_CUDA<float>::clear()
{
    this->cleanFFT();
    if (c_auxr_3d != nullptr)
    {
        delmem_cd_op()(gpu_ctx, c_auxr_3d);
        c_auxr_3d = nullptr;
    }
}
template <>
void FFT_CUDA<double>::clear()
{
    this->cleanFFT();
    if (z_auxr_3d != nullptr)
    {
        delmem_zd_op()(gpu_ctx, z_auxr_3d);
        z_auxr_3d = nullptr;
    }
}

template <>
void FFT_CUDA<float>::fft3D_forward(std::complex<float>* in, 
                                    std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle, 
                             reinterpret_cast<cufftComplex*>(in), 
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<double>::fft3D_forward(std::complex<double>* in, 
                                     std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle, 
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out), 
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<float>::fft3D_backward(std::complex<float>* in, 
                                     std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle, 
                             reinterpret_cast<cufftComplex*>(in), 
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}

template <>
void FFT_CUDA<double>::fft3D_backward(std::complex<double>* in, 
                                      std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle, 
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out), 
                             CUFFT_INVERSE));
}
template <> std::complex<float>*
FFT_CUDA<float>::get_auxr_3d_data()  const {return this->c_auxr_3d;}
template <> std::complex<double>*
FFT_CUDA<double>::get_auxr_3d_data() const {return this->z_auxr_3d;}

// -----------------------------------------------------------------------
// initfft_split: create XY (2D, batched) and Z (1D, batched) cufft plans
// -----------------------------------------------------------------------
template <>
void FFT_CUDA<float>::initfft_split(int nx_in, int ny_in, int nz_in,
                                    int nplane_in, int nst_in, int chunk_sz)
{
    nplane_ = nplane_in;
    nst_    = nst_in;

    int xy_batch = std::min(chunk_sz, nplane_in);

    // XY plan: batch of xy_batch 2D C2C FFTs over (nx_in × ny_in)
    {
        int dims[2]      = {ny_in, nx_in};
        int inembed[2]   = {ny_in, nx_in};
        int onembed[2]   = {ny_in, nx_in};
        CHECK_CUFFT(cufftPlanMany(&c_xy_handle,
                                  2, dims,
                                  inembed, 1, nx_in * ny_in,
                                  onembed, 1, nx_in * ny_in,
                                  CUFFT_C2C, xy_batch));
    }

    // Z plan: batch of nst_in 1D C2C FFTs (process all sticks at once)
    {
        int dims[1]    = {nz_in};
        int inembed[1] = {nz_in};
        int onembed[1] = {nz_in};
        CHECK_CUFFT(cufftPlanMany(&c_z_handle,
                                  1, dims,
                                  inembed, 1, nz_in,
                                  onembed, 1, nz_in,
                                  CUFFT_C2C, nst_in));
    }
}

template <>
void FFT_CUDA<double>::initfft_split(int nx_in, int ny_in, int nz_in,
                                     int nplane_in, int nst_in, int chunk_sz)
{
    nplane_ = nplane_in;
    nst_    = nst_in;

    int xy_batch = std::min(chunk_sz, nplane_in);

    // XY plan (double)
    {
        int dims[2]    = {ny_in, nx_in};
        int inembed[2] = {ny_in, nx_in};
        int onembed[2] = {ny_in, nx_in};
        CHECK_CUFFT(cufftPlanMany(&z_xy_handle,
                                  2, dims,
                                  inembed, 1, nx_in * ny_in,
                                  onembed, 1, nx_in * ny_in,
                                  CUFFT_Z2Z, xy_batch));
    }

    // Z plan (double): batch of nst_in (process all sticks at once)
    {
        int dims[1]    = {nz_in};
        int inembed[1] = {nz_in};
        int onembed[1] = {nz_in};
        CHECK_CUFFT(cufftPlanMany(&z_z_handle,
                                  1, dims,
                                  inembed, 1, nz_in,
                                  onembed, 1, nz_in,
                                  CUFFT_Z2Z, nst_in));
    }
}

// -----------------------------------------------------------------------
// fftxy_forward / fftxy_backward  (async, on caller-supplied stream)
// -----------------------------------------------------------------------
template <>
void FFT_CUDA<float>::fftxy_forward(std::complex<float>* in,
                                    std::complex<float>* out,
                                    cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(c_xy_handle, stream));
    CHECK_CUFFT(cufftExecC2C(c_xy_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}

template <>
void FFT_CUDA<double>::fftxy_forward(std::complex<double>* in,
                                     std::complex<double>* out,
                                     cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(z_xy_handle, stream));
    CHECK_CUFFT(cufftExecZ2Z(z_xy_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}

template <>
void FFT_CUDA<float>::fftxy_backward(std::complex<float>* in,
                                     std::complex<float>* out,
                                     cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(c_xy_handle, stream));
    CHECK_CUFFT(cufftExecC2C(c_xy_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}

template <>
void FFT_CUDA<double>::fftxy_backward(std::complex<double>* in,
                                      std::complex<double>* out,
                                      cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(z_xy_handle, stream));
    CHECK_CUFFT(cufftExecZ2Z(z_xy_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}

// -----------------------------------------------------------------------
// fftz_forward / fftz_backward  (async, on caller-supplied stream)
// -----------------------------------------------------------------------
template <>
void FFT_CUDA<float>::fftz_forward(std::complex<float>* in,
                                   std::complex<float>* out,
                                   cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(c_z_handle, stream));
    CHECK_CUFFT(cufftExecC2C(c_z_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}

template <>
void FFT_CUDA<double>::fftz_forward(std::complex<double>* in,
                                    std::complex<double>* out,
                                    cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(z_z_handle, stream));
    CHECK_CUFFT(cufftExecZ2Z(z_z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}

template <>
void FFT_CUDA<float>::fftz_backward(std::complex<float>* in,
                                    std::complex<float>* out,
                                    cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(c_z_handle, stream));
    CHECK_CUFFT(cufftExecC2C(c_z_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}

template <>
void FFT_CUDA<double>::fftz_backward(std::complex<double>* in,
                                     std::complex<double>* out,
                                     cudaStream_t stream) const
{
    CHECK_CUFFT(cufftSetStream(z_z_handle, stream));
    CHECK_CUFFT(cufftExecZ2Z(z_z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}

template FFT_CUDA<float>::FFT_CUDA();
template FFT_CUDA<float>::~FFT_CUDA();
template FFT_CUDA<double>::FFT_CUDA();
template FFT_CUDA<double>::~FFT_CUDA();
}// namespace ModulePW