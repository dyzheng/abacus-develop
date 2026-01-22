#include "FORCE_STRESS.h"

#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_func.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/global_variable.h"

template <typename T>
void Force_Stress_LCAO<T>::calForcePwPart(UnitCell& ucell,
                                          ModuleBase::matrix& fvl_dvl,
                                          ModuleBase::matrix& fewalds,
                                          ModuleBase::matrix& fcc,
                                          ModuleBase::matrix& fscc,
                                          const double& etxc,
                                          const ModuleBase::matrix& vnew,
                                          const bool vnew_exist,
                                          const Charge* const chr,
                                          ModulePW::PW_Basis* rhopw,
                                          const pseudopot_cell_vl& locpp,
                                          const Structure_Factor& sf)
{
    ModuleBase::TITLE("Force_Stress_LCAO", "calForcePwPart");
#ifdef __CUDA
    if(PARAM.inp.device == "gpu")
    {
        Forces<double, base_device::DEVICE_GPU> f_pw(nat);
        f_pw.cal_force_loc(ucell, fvl_dvl, rhopw, locpp.vloc, chr);
        f_pw.cal_force_ew(ucell, fewalds, rhopw, &sf);
        f_pw.cal_force_cc(fcc, rhopw, chr, locpp.numeric, ucell);
        f_pw.cal_force_scc(fscc, rhopw, vnew, vnew_exist, locpp.numeric, ucell);
    }
    else
#endif
    {
        Forces<double, base_device::DEVICE_CPU> f_pw(nat);
        f_pw.cal_force_loc(ucell, fvl_dvl, rhopw, locpp.vloc, chr);
        f_pw.cal_force_ew(ucell, fewalds, rhopw, &sf);
        f_pw.cal_force_cc(fcc, rhopw, chr, locpp.numeric, ucell);
        f_pw.cal_force_scc(fscc, rhopw, vnew, vnew_exist, locpp.numeric, ucell);
    }

    return;
}

template <typename T>
void Force_Stress_LCAO<T>::calStressPwPart(UnitCell& ucell,
                                           ModuleBase::matrix& sigmadvl,
                                           ModuleBase::matrix& sigmahar,
                                           ModuleBase::matrix& sigmaewa,
                                           ModuleBase::matrix& sigmacc,
                                           ModuleBase::matrix& sigmaxc,
                                           const double& etxc,
                                           const Charge* const chr,
                                           ModulePW::PW_Basis* rhopw,
                                           const pseudopot_cell_vl& locpp,
                                           const Structure_Factor& sf)
{
    ModuleBase::TITLE("Force_Stress_LCAO", "calStressPwPart");

    sc_pw.stress_loc(ucell, sigmadvl, rhopw, locpp.vloc, &sf, 0, chr);

    sc_pw.stress_har(ucell, sigmahar, rhopw, 0, chr);

    sc_pw.stress_ewa(ucell, sigmaewa, rhopw, 0);

    sc_pw.stress_cc(sigmacc, rhopw, ucell, &sf, 0, locpp.numeric, chr);

    for (int i = 0; i < 3; i++)
    {
        sigmaxc(i, i) = -etxc / ucell.omega;
    }
    sc_pw.stress_gga(ucell, sigmaxc, rhopw, chr);

    return;
}

template class Force_Stress_LCAO<double>;
template class Force_Stress_LCAO<std::complex<double>>;

