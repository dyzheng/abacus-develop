#include "module_basis/module_pw/kernels/pw_op.h"

namespace ModulePW {

template <typename FPTYPE>
struct set_3d_fft_box_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
                    const int npwk,
                    const int* box_index,
                    const std::complex<FPTYPE>* in,
                    std::complex<FPTYPE>* out)
    {
        for (int ig = 0; ig < npwk; ++ig)
        {
            out[box_index[ig]] = in[ig];
        }
    }
};

template <typename FPTYPE>
struct set_3d_fft_box_batch_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* dev,
                    const int npwk,
                    const int* box_index,
                    const std::complex<FPTYPE>* in,
                    const int ld_in,
                    std::complex<FPTYPE>* out,
                    const int ld_out,
                    const int batchSize)
    {
        for (int i = 0; i < batchSize; ++i)
        {
            set_3d_fft_box_op<FPTYPE, base_device::DEVICE_CPU>()(dev, npwk, box_index, in + i * ld_in, out + i * ld_out);
        }
    }
};

template <typename FPTYPE>
struct set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
                    const int nrxx,
                    const bool add,
                    const FPTYPE factor,
                    const std::complex<FPTYPE>* in,
                    std::complex<FPTYPE>* out)
    {
        if(add) {
            for(int ir = 0; ir < nrxx ; ++ir) {
                out[ir] += factor * in[ir];
            }
        }
        else {
            for(int ir = 0; ir < nrxx ; ++ir) {
                out[ir] = in[ir];
            }
        }
    }
};

template <typename FPTYPE>
struct set_recip_to_real_output_batch_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* dev,
                    const int nrxx,
                    const bool add,
                    const FPTYPE factor,
                    const std::complex<FPTYPE>* in,
                    const int ld_in,
                    std::complex<FPTYPE>* out,
                    const int ld_out,
                    const int batchSize)
    {
        for (int i = 0; i < batchSize; ++i)
        {
            set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_CPU>()(dev, nrxx, add, factor,
                in + i * ld_in, out + i * ld_out);
        }
    }
};

template <typename FPTYPE>
struct set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
                    const int npw_k,
                    const int nxyz,
                    const bool add,
                    const FPTYPE factor,
                    const int* box_index,
                    const std::complex<FPTYPE>* in,
                    std::complex<FPTYPE>* out)
    {
        if(add) {
            for(int ig = 0; ig < npw_k; ++ig) {
                out[ig] += factor / static_cast<FPTYPE>(nxyz) * in[box_index[ig]];
            }
        }
        else {
            for(int ig = 0; ig < npw_k; ++ig) {
                out[ig] = in[box_index[ig]] / static_cast<FPTYPE>(nxyz);
            }
        }
    }
};

template <typename FPTYPE>
struct set_real_to_recip_output_batch_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* dev,
                    const int npw_k,
                    const int nxyz,
                    const bool add,
                    const FPTYPE factor,
                    const int* box_index,
                    const std::complex<FPTYPE>* in,
                    const int ld_in,
                    std::complex<FPTYPE>* out,
                    const int ld_out,
                    const int batchSize)
    {
        for (int i = 0; i < batchSize; ++i)
        {
            set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_CPU>()(dev, npw_k, nxyz, add, factor,
                box_index, in + i * ld_in, out + i * ld_out);
        }
    }
};

template struct set_3d_fft_box_op<float, base_device::DEVICE_CPU>;
template struct set_3d_fft_box_batch_op<float, base_device::DEVICE_CPU>;
template struct set_recip_to_real_output_op<float, base_device::DEVICE_CPU>;
template struct set_recip_to_real_output_batch_op<float, base_device::DEVICE_CPU>;
template struct set_real_to_recip_output_op<float, base_device::DEVICE_CPU>;
template struct set_real_to_recip_output_batch_op<float, base_device::DEVICE_CPU>;
template struct set_3d_fft_box_op<double, base_device::DEVICE_CPU>;
template struct set_3d_fft_box_batch_op<double, base_device::DEVICE_CPU>;
template struct set_recip_to_real_output_op<double, base_device::DEVICE_CPU>;
template struct set_recip_to_real_output_batch_op<double, base_device::DEVICE_CPU>;
template struct set_real_to_recip_output_op<double, base_device::DEVICE_CPU>;
template struct set_real_to_recip_output_batch_op<double, base_device::DEVICE_CPU>;

}  // namespace ModulePW

