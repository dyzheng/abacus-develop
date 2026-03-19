// Host-side interface for GPU XC calculations
#ifndef XC_FUNCTIONAL_GPU_H
#define XC_FUNCTIONAL_GPU_H

#include <vector>

namespace ModulePW { class PW_Basis; }

namespace XC_GPU
{

/// Launch LDA XC kernel on GPU for nspin=1
/// @param nrxx number of real-space grid points
/// @param d_rho device pointer to rho[nrxx]
/// @param d_rho_core device pointer to rho_core[nrxx]
/// @param d_v_xc device pointer to output vxc[nrxx]
/// @param etxc output: exchange-correlation energy (host scalar)
/// @param vtxc output: vtxc (host scalar)
/// @param func_ids functional IDs (host vector)
/// @param hybrid_alpha hybrid mixing parameter
void v_xc_lda_gpu_nspin1(const int nrxx,
                          const double* d_rho,
                          const double* d_rho_core,
                          double* d_v_xc,
                          double& etxc,
                          double& vtxc,
                          const std::vector<int>& func_ids,
                          const double hybrid_alpha);

/// Launch LDA XC kernel on GPU for nspin=2
void v_xc_lda_gpu_nspin2(const int nrxx,
                          const double* d_rho_up,
                          const double* d_rho_dw,
                          const double* d_rho_core,
                          double* d_v_xc_up,
                          double* d_v_xc_dw,
                          double& etxc,
                          double& vtxc,
                          const std::vector<int>& func_ids,
                          const double hybrid_alpha);

/// Launch LDA XC kernel on GPU for nspin=4 (non-collinear)
void v_xc_lda_gpu_nspin4(const int nrxx,
                          const double* d_rho0,
                          const double* d_rho1,
                          const double* d_rho2,
                          const double* d_rho3,
                          const double* d_rho_core,
                          double* d_v_xc0,
                          double* d_v_xc1,
                          double* d_v_xc2,
                          double* d_v_xc3,
                          double& etxc,
                          double& vtxc,
                          const std::vector<int>& func_ids,
                          const double hybrid_alpha);

/// GPU gradcorr: compute GGA gradient correction entirely on GPU
/// Requires PW_Basis with GPU FFT support (ig2ixyz initialized)
/// @param nrxx number of real-space grid points
/// @param npw number of plane waves
/// @param nspin0 effective spin (1 or 2)
/// @param nspin actual nspin parameter
/// @param d_rho device pointer to rho[nspin_rho][nrxx]
/// @param d_rho_core device pointer to rho_core[nrxx]
/// @param d_v_xc device pointer to output vxc[nspin][nrxx], accumulated
/// @param etxc accumulated exchange-correlation energy (host)
/// @param vtxc accumulated vtxc (host)
/// @param func_ids functional IDs (host vector)
/// @param hybrid_alpha hybrid mixing parameter
/// @param rhopw PW_Basis pointer for FFT
/// @param tpiba 2*pi/lat0
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
                  const double* ux_, const bool lsign_);

} // namespace XC_GPU

#endif // XC_FUNCTIONAL_GPU_H
