#include "module_hamilt_pw/hamilt_pwdft/kernels/force_op.h"
// #include "module_psi/kernels/device.h"
#include "module_base/module_device/types.h"

#include <complex>

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <base/macros/macros.h>
#include <module_base/module_device/device.h>

#define THREADS_PER_BLOCK 256

namespace hamilt {


template <typename FPTYPE>
__global__ void cal_vkb1_nl(
        const int npwx,
        const int vkb_nc,
        const int nbasis,
        const int ipol,
        const thrust::complex<FPTYPE> NEG_IMAG_UNIT,
        const thrust::complex<FPTYPE> *vkb,
        const FPTYPE *gcar,
        thrust::complex<FPTYPE> *vkb1)
{
    thrust::complex<FPTYPE> *pvkb1 = vkb1 + blockIdx.x * npwx;
    const thrust::complex<FPTYPE> *pvkb = vkb + blockIdx.x * vkb_nc;
    for (int ig = threadIdx.x; ig < nbasis; ig += blockDim.x) {
        pvkb1[ig] = pvkb[ig] * NEG_IMAG_UNIT * gcar[ig * 3 + ipol];
    }
}

template <typename FPTYPE>
__global__ void cal_force_nl(
        const bool nondiagonal,
        const int wg_nc,
        const int ntype,
        const int spin,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int forcenl_nc,
        const int nbands,
        const int ik,
        const int nkb,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE tpiba,
        const FPTYPE *d_wg,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE *deeq,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *force)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;

    int iat = 0, sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    int Nprojs = atom_nh[it];
    FPTYPE fac = d_wg[ik * wg_nc + ib] * 2.0 * tpiba;
    FPTYPE ekb_now = d_ekb[ik * wg_nc + ib];
    for (int ia = 0; ia < atom_na[it]; ia++) {
        for (int ip = threadIdx.x; ip < Nprojs; ip += blockDim.x) {
            // FPTYPE ps = GlobalC::ppcell.deeq[spin, iat, ip, ip];
            FPTYPE ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip]
                        - ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip * deeq_4 + ip];
            const int inkb = sum + ip;
            //out<<"\n ps = "<<ps;

            for (int ipol = 0; ipol < 3; ipol++) {
                const FPTYPE dbb = (conj(dbecp[ipol * nbands * nkb + ib * nkb + inkb]) *
                                    becp[ib * nkb + inkb]).real();
                // force[iat * forcenl_nc + ipol] -= ps * fac * dbb;
                atomicAdd(force + iat * forcenl_nc + ipol, -ps * fac * dbb);
                //cf[iat*3+ipol] += ps * fac * dbb;
            }

            if (nondiagonal) {
                //for (int ip2=0; ip2<Nprojs; ip2++)
                for (int ip2 = 0; ip2 < Nprojs; ip2++) {
                    if (ip != ip2) {
                        const int jnkb = sum + ip2;
                        ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip) * deeq_4 + ip2]
                             - ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip * deeq_4 + ip2];
                        for (int ipol = 0; ipol < 3; ipol++) {
                            const FPTYPE dbb = (conj(dbecp[ipol * nbands * nkb + ib * nkb + inkb]) *
                                                becp[ib * nkb + jnkb]).real();
                            atomicAdd(force + iat * forcenl_nc + ipol, -ps * fac * dbb);
                        }
                    }
                }
            }
        }
        iat += 1;
        sum += Nprojs;
    }
}

template <typename FPTYPE>
void cal_vkb1_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                 const int& nkb,
                                                                 const int& npwx,
                                                                 const int& vkb_nc,
                                                                 const int& nbasis,
                                                                 const int& ipol,
                                                                 const std::complex<FPTYPE>& NEG_IMAG_UNIT,
                                                                 const std::complex<FPTYPE>* vkb,
                                                                 const FPTYPE* gcar,
                                                                 std::complex<FPTYPE>* vkb1)
{
    cal_vkb1_nl<FPTYPE><<<nkb, THREADS_PER_BLOCK>>>(
            npwx,
            vkb_nc,
            nbasis,
            ipol,
            static_cast<const thrust::complex<FPTYPE>>(NEG_IMAG_UNIT), // array of data
            reinterpret_cast<const thrust::complex<FPTYPE>*>(vkb),
            gcar,// array of data
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb1)); // array of data

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void cal_force_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                  const bool& nondiagonal,
                                                                  const int& nbands_occ,
                                                                  const int& wg_nc,
                                                                  const int& ntype,
                                                                  const int& spin,
                                                                  const int& deeq_2,
                                                                  const int& deeq_3,
                                                                  const int& deeq_4,
                                                                  const int& forcenl_nc,
                                                                  const int& nbands,
                                                                  const int& ik,
                                                                  const int& nkb,
                                                                  const int* atom_nh,
                                                                  const int* atom_na,
                                                                  const FPTYPE& tpiba,
                                                                  const FPTYPE* d_wg,
                                                                  const FPTYPE* d_ekb,
                                                                  const FPTYPE* qq_nt,
                                                                  const FPTYPE* deeq,
                                                                  const std::complex<FPTYPE>* becp,
                                                                  const std::complex<FPTYPE>* dbecp,
                                                                  FPTYPE* force)
{
    cal_force_nl<FPTYPE><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
            nondiagonal,
            wg_nc, ntype, spin,
            deeq_2, deeq_3, deeq_4,
            forcenl_nc, nbands, ik, nkb,
            atom_nh, atom_na,
            tpiba,
            d_wg, d_ekb, qq_nt, deeq,
            reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
            reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
            force);// array of data

    cudaCheckOnDebug();
}

template <typename FPTYPE>
__global__ void saveVkbValues(
    const int *gcar_zero_ptrs, 
    const std::complex<FPTYPE> *vkb_ptr, 
    std::complex<FPTYPE> *vkb_save_ptr, 
    int nkb, 
    int npw, 
    size_t n_total_gcar_zeros)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index
    int ikb = index / n_total_gcar_zeros;              // index of nkb
    int icount = index % n_total_gcar_zeros;           // index of n_total_gcar_zeros
    
    // check if the index is valid
    if(ikb < nkb && icount < n_total_gcar_zeros)
    {
        int ig = gcar_zero_ptrs[icount]; // get ig from gcar_zero_ptrs
        // use the flat index to get the saved position, pay attention to the relationship between ikb and npw,
        vkb_save_ptr[index] = vkb_ptr[ikb * npw + ig];    // save the value
    }
}

template <typename FPTYPE>
__global__ void revertVkbValues(
    const int *gcar_zero_ptrs, 
    std::complex<FPTYPE> *vkb_ptr, 
    const std::complex<FPTYPE> *vkb_save_ptr, 
    int nkb, 
    int npw, 
    size_t n_total_gcar_zeros,
    const std::complex<FPTYPE> coeff)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index
    int ikb = index / n_total_gcar_zeros;              // index of nkb
    int icount = index % n_total_gcar_zeros;           // index of n_total_gcar_zeros
    
    // check if the index is valid
    if(ikb < nkb && icount < n_total_gcar_zeros)
    {
        int ig = gcar_zero_ptrs[icount]; // get ig from gcar_zero_ptrs
        // use the flat index to get the saved position, pay attention to the relationship between ikb and npw,
        vkb_ptr[ikb * npw + ig] = vkb_save_ptr[index] * coeff;    // revert the values
    }
}

// for revertVkbValues functions instantiation
template void revertVkbValues<float>(const int *gcar_zero_ptrs, std::complex<float> *vkb_ptr, const std::complex<float> *vkb_save_ptr, int nkb, int npw, size_t n_total_gcar_zeros);
template void revertVkbValues<double>(const int *gcar_zero_ptrs, std::complex<double> *vkb_ptr, const std::complex<double> *vkb_save_ptr, int nkb, int npw, size_t n_total_gcar_zeros);
// for saveVkbValues functions instantiation
template void saveVkbValues<float>(const int *gcar_zero_ptrs, const std::complex<float> *vkb_ptr, std::complex<float> *vkb_save_ptr, int nkb, int npw, size_t n_total_gcar_zeros);
template void saveVkbValues<double>(const int *gcar_zero_ptrs, const std::complex<double> *vkb_ptr, std::complex<double> *vkb_save_ptr, int nkb, int npw, size_t n_total_gcar_zeros);

template struct cal_vkb1_nl_op<float, base_device::DEVICE_GPU>;
template struct cal_force_nl_op<float, base_device::DEVICE_GPU>;

template struct cal_vkb1_nl_op<double, base_device::DEVICE_GPU>;
template struct cal_force_nl_op<double, base_device::DEVICE_GPU>;

}  // namespace hamilt