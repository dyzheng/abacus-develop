#include "FORCE_STRESS.h"
#include "source_pw/module_pwdft/global.h"
#include "source_io/module_parameter/parameter.h"
#include "source_hamilt/module_vdw/vdw.h"
#include "source_estate/module_pot/efield.h"
#include "source_estate/module_pot/gatefield.h"
#include "source_estate/module_pot/H_TDDFT_pw.h"
#include "source_lcao/module_operator_lcao/dftu_lcao.h"
#include "source_lcao/module_operator_lcao/dspin_lcao.h"

template <typename T>
void Force_Stress_LCAO<T>::calculate_vdw_force_stress(
    const UnitCell& ucell,
    const bool isforce,
    const bool isstress,
    ModuleBase::matrix& force_vdw,
    ModuleBase::matrix& stress_vdw)
{
    if (!isforce && !isstress)
    {
        return;
    }

    auto vdw_solver = vdw::make_vdw(ucell, PARAM.inp);
    if (vdw_solver != nullptr)
    {
        if (isforce)
        {
            force_vdw.create(ucell.nat, 3);
            const std::vector<ModuleBase::Vector3<double>>& force_vdw_temp = vdw_solver->get_force();
            for (int iat = 0; iat < ucell.nat; ++iat)
            {
                force_vdw(iat, 0) = force_vdw_temp[iat].x;
                force_vdw(iat, 1) = force_vdw_temp[iat].y;
                force_vdw(iat, 2) = force_vdw_temp[iat].z;
            }
        }
        if (isstress)
        {
            stress_vdw = vdw_solver->get_stress().to_matrix();
        }
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_efield_force(
    const UnitCell& ucell,
    const bool isforce,
    ModuleBase::matrix& fefield)
{
    if (PARAM.inp.efield_flag && isforce)
    {
        fefield.create(ucell.nat, 3);
        elecstate::Efield::compute_force(ucell, fefield);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_tddft_efield_force(
    const UnitCell& ucell,
    const bool isforce,
    ModuleBase::matrix& fefield_tddft)
{
    if (PARAM.inp.esolver_type == "tddft" && isforce)
    {
        fefield_tddft.create(ucell.nat, 3);
        elecstate::H_TDDFT_pw::compute_force(ucell, fefield_tddft);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_gatefield_force(
    const UnitCell& ucell,
    const bool isforce,
    ModuleBase::matrix& fgate)
{
    if (PARAM.inp.gate_flag && isforce)
    {
        fgate.create(ucell.nat, 3);
        elecstate::Gatefield::compute_force(ucell, fgate);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_solvent_force(
    UnitCell& ucell,
    const bool isforce,
    ModulePW::PW_Basis* rhopw,
    const pseudopot_cell_vl& locpp,
    surchem& solvent,
    ModuleBase::matrix& fsol)
{
    if (PARAM.inp.imp_sol && isforce)
    {
        fsol.create(ucell.nat, 3);
        solvent.cal_force_sol(ucell, rhopw, locpp.vloc, fsol);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_dftu_force_stress(
    const UnitCell& ucell,
    const Grid_Driver& gd,
    LCAO_domain::Setup_DM<T>& dmat,
    Parallel_Orbitals& pv,
    const K_Vectors& kv,
    const bool isforce,
    const bool isstress,
    const TwoCenterBundle& two_center_bundle,
    const LCAO_Orbitals& orb,
    Plus_U& dftu,
    ModuleBase::matrix& force_u,
    ModuleBase::matrix& stress_u)
{
    if (!PARAM.inp.dft_plus_u)
    {
        return;
    }

    if (isforce)
    {
        force_u.create(ucell.nat, 3);
    }
    if (isstress)
    {
        stress_u.create(3, 3);
    }

    if (PARAM.inp.dft_plus_u == 2)
    {
        ForceStressArrays fsr_dftu;
        std::vector<std::vector<double>>* dmk_d = nullptr;
        std::vector<std::vector<std::complex<double>>>* dmk_c = nullptr;
        assign_dmk_ptr<T>(dmat.dm, dmk_d, dmk_c, PARAM.globalv.gamma_only_local);
        dftu.force_stress(ucell, gd, dmk_d, dmk_c, pv, fsr_dftu, force_u, stress_u, kv);
    }
    else
    {
        hamilt::DFTU<hamilt::OperatorLCAO<T, double>> tmpu(nullptr,
                                                               kv.kvec_d,
                                                               nullptr,
                                                               ucell,
                                                               &gd,
                                                               two_center_bundle.overlap_orb_onsite.get(),
                                                               orb.cutoffs(),
                                                               &dftu);
        tmpu.cal_force_stress(isforce, isstress, force_u, stress_u);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_deltaspin_force_stress(
    const UnitCell& ucell,
    const Grid_Driver& gd,
    LCAO_domain::Setup_DM<T>& dmat,
    const K_Vectors& kv,
    const bool isforce,
    const bool isstress,
    const TwoCenterBundle& two_center_bundle,
    const LCAO_Orbitals& orb,
    ModuleBase::matrix& force_dspin,
    ModuleBase::matrix& stress_dspin)
{
    if (!PARAM.inp.sc_mag_switch)
    {
        return;
    }

    if (isforce)
    {
        force_dspin.create(ucell.nat, 3);
    }
    if (isstress)
    {
        stress_dspin.create(3, 3);
    }

    hamilt::DeltaSpin<hamilt::OperatorLCAO<T, double>> tmp_dspin(nullptr,
                                                                 kv.kvec_d,
                                                                 nullptr,
                                                                 ucell,
                                                                 &gd,
                                                                 two_center_bundle.overlap_orb_onsite.get(),
                                                                 orb.cutoffs());

    if (PARAM.inp.nspin == 2)
    {
        dmat.dm->switch_dmr(2);
    }
    const hamilt::HContainer<double>* dmr = dmat.dm->get_DMR_pointer(1);
    tmp_dspin.cal_force_stress(isforce, isstress, dmr, force_dspin, stress_dspin);
    if (PARAM.inp.nspin == 2)
    {
        dmat.dm->switch_dmr(0);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_exx_force_stress(
    const UnitCell& ucell,
    const bool isforce,
    const bool isstress,
    Exx_NAO<T>& exx_nao,
    ModuleBase::matrix& force_exx,
    ModuleBase::matrix& stress_exx)
{
#ifdef __EXX
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        if (isforce)
        {
            if (GlobalC::exx_info.info_ri.real_number)
            {
                exx_nao.exd->cal_exx_force(ucell.nat);
                force_exx = GlobalC::exx_info.info_global.hybrid_alpha * exx_nao.exd->get_force();
            }
            else
            {
                exx_nao.exc->cal_exx_force(ucell.nat);
                force_exx = GlobalC::exx_info.info_global.hybrid_alpha * exx_nao.exc->get_force();
            }
        }
        if (isstress)
        {
            if (GlobalC::exx_info.info_ri.real_number)
            {
                exx_nao.exd->cal_exx_stress(ucell.omega, ucell.lat0);
                stress_exx = GlobalC::exx_info.info_global.hybrid_alpha * exx_nao.exd->get_stress();
            }
            else
            {
                exx_nao.exc->cal_exx_stress(ucell.omega, ucell.lat0);
                stress_exx = GlobalC::exx_info.info_global.hybrid_alpha * exx_nao.exc->get_stress();
            }
        }
    }
#else
    (void)ucell;
    (void)isforce;
    (void)isstress;
    (void)exx_nao;
    (void)force_exx;
    (void)stress_exx;
#endif
}

template class Force_Stress_LCAO<double>;
template class Force_Stress_LCAO<std::complex<double>>;

