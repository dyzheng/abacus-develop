#include "source_pw/module_pwdft/dftu_pw.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/module_device/device.h"

namespace pw
{

template <typename T, typename Device>
void iter_init_dftu_pw(const int iter,
                        const int istep,
                        Plus_U& dftu,
                        const void* psi,
                        const ModuleBase::matrix& wg,
                        const UnitCell& ucell,
                        Charge_Mixing* p_chgmix,
                        const int* isk)
{
    if (!p_chgmix || !PARAM.inp.dft_plus_u)
    {
        return;
    }

    if (iter == 1 && istep == 0)
    {
        return;
    }

    if (dftu.omc != 2)
    {
        dftu.template cal_occ_pw<T, Device>(iter, psi, wg, ucell, p_chgmix, isk);
    }
    dftu.output(ucell, PARAM.inp.out_chg[0], PARAM.globalv.global_out_dir, PARAM.inp.nspin, PARAM.globalv.npol);
}

template void iter_init_dftu_pw<std::complex<double>, base_device::DEVICE_CPU>(const int iter,
                                                                               const int istep,
                                                                               Plus_U& dftu,
                                                                               const void* psi,
                                                                               const ModuleBase::matrix& wg,
                                                                               const UnitCell& ucell,
                                                                               Charge_Mixing* p_chgmix,
                                                                               const int* isk);
template void iter_init_dftu_pw<std::complex<float>, base_device::DEVICE_CPU>(const int iter,
                                                                              const int istep,
                                                                              Plus_U& dftu,
                                                                              const void* psi,
                                                                              const ModuleBase::matrix& wg,
                                                                              const UnitCell& ucell,
                                                                              Charge_Mixing* p_chgmix,
                                                                              const int* isk);
#if defined(__CUDA) || defined(__ROCM)
template void iter_init_dftu_pw<std::complex<double>, base_device::DEVICE_GPU>(const int iter,
                                                                               const int istep,
                                                                               Plus_U& dftu,
                                                                               const void* psi,
                                                                               const ModuleBase::matrix& wg,
                                                                               const UnitCell& ucell,
                                                                               Charge_Mixing* p_chgmix,
                                                                               const int* isk);
template void iter_init_dftu_pw<std::complex<float>, base_device::DEVICE_GPU>(const int iter,
                                                                              const int istep,
                                                                              Plus_U& dftu,
                                                                              const void* psi,
                                                                              const ModuleBase::matrix& wg,
                                                                              const UnitCell& ucell,
                                                                              Charge_Mixing* p_chgmix,
                                                                              const int* isk);
#endif

}
