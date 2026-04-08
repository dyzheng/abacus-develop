#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft_base.h"
#include "cufft.h"
#include "cuda_runtime.h"
namespace ModulePW
{
template <typename FPTYPE>
class FFT_CUDA : public FFT_BASE<FPTYPE>
{
    public:
        FFT_CUDA(){};
        ~FFT_CUDA(){};

	    void setupFFT() override;

        void clear() override;

        void cleanFFT() override;

        /**
        * @brief Initialize the fft parameters for single-process 3D FFT
        * @param nx_in  number of grid points in x direction
        * @param ny_in  number of grid points in y direction
        * @param nz_in  number of grid points in z direction
        *
        */
        void initfft(int nx_in,
                     int ny_in,
                     int nz_in) override;

        /**
         * @brief Initialize the split XY+Z FFT plans used in multi-process mode.
         * @param nx_in   grid points in x
         * @param ny_in   grid points in y
         * @param nz_in   grid points in z
         * @param nplane_in  number of XY planes owned by this rank (batch for fftxy)
         * @param nst_in     number of sticks  owned by this rank (batch for fftz)
         * @param chunk_sz   pipeline chunk size (default 16)
         *
         * Must be called after initfft(nx,ny,nz) when poolnproc > 1.
         */
        void initfft_split(int nx_in, int ny_in, int nz_in,
                           int nplane_in, int nst_in, int chunk_sz = 16);

        /**
         * @brief Get the real space data
         * @return real space data
         */
        std::complex<FPTYPE>* get_auxr_3d_data() const override;

        /**
         * @brief Forward FFT in 3D (single-process path, synchronous)
         */
        void fft3D_forward(std::complex<FPTYPE>* in,
                           std::complex<FPTYPE>* out) const override;
        /**
         * @brief Backward FFT in 3D (single-process path, synchronous)
         */
        void fft3D_backward(std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out) const override;

        // ----------------------------------------------------------------
        // Split FFT interface for multi-process GPU path
        // ----------------------------------------------------------------

        /**
         * @brief Forward 2D FFT over XY planes, async on given stream.
         * @param in    input  (nplane, ny, nx) on device
         * @param out   output (nplane, ny, nx) on device
         * @param stream  CUDA stream; pass 0 for default stream
         */
        void fftxy_forward(std::complex<FPTYPE>* in,
                           std::complex<FPTYPE>* out,
                           cudaStream_t stream = 0) const;

        /**
         * @brief Backward 2D IFFT over XY planes, async on given stream.
         */
        void fftxy_backward(std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out,
                            cudaStream_t stream = 0) const;

        /**
         * @brief Forward 1D FFT over Z sticks, async on given stream.
         * @param in    input  (nst, nz) on device
         * @param out   output (nst, nz) on device
         * @param stream  CUDA stream; pass 0 for default stream
         */
        void fftz_forward(std::complex<FPTYPE>* in,
                          std::complex<FPTYPE>* out,
                          cudaStream_t stream = 0) const;

        /**
         * @brief Backward 1D IFFT over Z sticks, async on given stream.
         */
        void fftz_backward(std::complex<FPTYPE>* in,
                           std::complex<FPTYPE>* out,
                           cudaStream_t stream = 0) const;

    private:
        // ---- 3D plan (single-process) ----
        cufftHandle c_handle = {};
        cufftHandle z_handle = {};
        std::complex<float>*  c_auxr_3d = nullptr;
        std::complex<double>* z_auxr_3d = nullptr;

        // ---- Split plans (multi-process) ----
        // XY: batch of nplane 2D FFTs over (nx × ny)
        cufftHandle c_xy_handle = {};
        cufftHandle z_xy_handle = {};
        // Z:  batch of nst 1D FFTs over nz
        cufftHandle c_z_handle = {};
        cufftHandle z_z_handle = {};

        int nplane_ = 0;  // batch size for fftxy plans
        int nst_    = 0;  // batch size for fftz  plans
};

} // namespace ModulePW
#endif