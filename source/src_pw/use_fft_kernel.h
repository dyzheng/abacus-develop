#ifndef USE_FFT_KERNEL_H
#define USE_FFT_KERNEL_H

#ifdef __CUDA
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

void RoundTrip_kernel(const float2 *psi, const float *vr, const int *fft_index, const int &max_g,
    	const int &max_r, float2 *psic);
void RoundTrip_kernel(const double2 *psi, const double *vr, const int *fft_index, const int &max_g,
    	const int &max_r, double2 *psic);

#endif

#ifdef __ROCM
#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "hipfft.h"

void RoundTrip_kernel(const hipblasComplex *psi, const float *vr, const int *fft_index, const int &max_g,
    	const int &max_r, hipblasComplex *psic);
void RoundTrip_kernel(const hipblasDoubleComplex *psi, const double *vr, const int *fft_index, const int &max_g,
    	const int &max_r, hipblasDoubleComplex *psic);

#endif

#endif