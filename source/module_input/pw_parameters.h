#ifndef PW_PARAMETERS_H
#define PW_PARAMETERS_H

#include <string>
namespace ABACUS
{
class PW_parameters
{
public:
    PW_parameters()
    {
        nx=0;
        ny=0;
        nz=0;
        ncx=0;
        ncy=0;
        ncz=0;
        ecut[0]=0.0;
        ecut[1]=0.0;
        diag_thr_e=1.0e-2;
        diag_cg_nmax=50;
        diag_dav_n=4;
    };
    ~PW_parameters(){};
    ///
    /// three dimension of FFT wavefunc
    ///
    int nx;
    int ny;
    int nz;
    ///
    /// three dimension of FFT charge/grid
    ///
    int ncx;
    int ncy;
    int ncz;
    /// energy cutoff for 0: wavefunction 1: charge density 
    double ecut[2];
	/// threshold used in cg method
	double diag_thr_e;
	/// max dimension for davidson
	int diag_cg_nmax;
	/// max iteration number for cg
	int diag_dav_n;

};
}

#endif