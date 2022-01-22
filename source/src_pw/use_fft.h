#ifndef USE_FFT_H
#define USE_FFT_H

#include "../module_base/global_function.h"
#include "../module_base/global_variable.h"
#include "../module_base/matrix.h"
#include "../module_base/complexmatrix.h"
#include <vector>

#ifdef __CUDA
#include "cufft.h"
#include "use_fft_kernel.h"
#endif

#ifdef __ROCM
#include "hipfft.h"
#include "use_fft_kernel.h"
#endif 

class Use_FFT
{
	public:

	Use_FFT();
	~Use_FFT();

/// This is a memory space for FFT operation
private:
	static std::vector<std::complex<double>> porter;
public:
	static std::complex<double>* get_porter(const int &begin, const int &end);

	void allocate(void);

	//---------------------------------------------------------------------
	
	// From G space to real space. ModuleBase::ComplexMatrix
	void ToRealSpace(const int &is, const ModuleBase::ComplexMatrix &vg, double *vr);
	void ToRealSpace_psi(const int &ik, const std::complex<double> *psig, std::complex<double> *psir);
	void ToRealSpace_psi(const int &ik, const int &ib, const ModuleBase::ComplexMatrix &evc, std::complex<double> *psir);
	
	// From G space to real space. charge/MLWF
	void ToRealSpace(const std::complex<double> *vg, double *vr);

	// From G space to real space. wave functions.
	void ToRealSpace(const std::complex<double> *vg, std::complex<double> *vr);

	void ToRealSpace(const int &is, const ModuleBase::ComplexMatrix &vg, ModuleBase::matrix &v);

	//---------------------------------------------------------------------

	// From real space to G space.
	void ToReciSpace(const double* vr, std::complex<double> *vg);


	//---------------------------------------------------------------------

	void RoundTrip(
	    const std::complex<double> *psi,
		const double *vr,
		const int *_index,
		const int &max_g,
    	const int &max_r,
		std::complex<double> *psic);

#ifdef __CUDA
	double2 *d_porter;
	cufftHandle fft_handle;
	void RoundTrip(const float2 *psi, const float *vr, const int *fft_index, const int &max_g,
    	const int &max_r, float2 *psic)
	{
		RoundTrip_kernel(psi, vr, fft_index, max_g, max_r, psic);
	}
	void RoundTrip(const double2 *psi, const double *vr, const int *fft_index, const int &max_g,
    	const int &max_r, double2 *psic)
	{
		RoundTrip_kernel(psi, vr, fft_index, max_g, max_r, psic);
	}
#endif

#ifdef __ROCM
	hipfftHandle fft_handle;
	void RoundTrip(const hipblasComplex *psi, const float *vr, const int *fft_index, const int &max_g,
    	const int &max_r, hipblasComplex *psic)
	{
		RoundTrip_kernel(psi, vr, fft_index, max_g, max_r, psic);
	}
	void RoundTrip(const hipblasDoubleComplex *psi, const double *vr, const int *fft_index, const int &max_g,
    	const int &max_r, hipblasDoubleComplex *psic)
	{
		RoundTrip_kernel(psi, vr, fft_index, max_g, max_r, psic);
	}
#endif

private:

};

#endif
