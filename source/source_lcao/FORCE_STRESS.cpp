#include "FORCE_STRESS.h"

#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/global.h"
#include "source_io/output_log.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_estate/elecstate_lcao.h"
#include "source_estate/module_pot/H_TDDFT_pw.h"
#include "source_estate/module_pot/efield.h"
#include "source_estate/module_pot/gatefield.h"
#include "source_hamilt/module_surchem/surchem.h"
#include "source_hamilt/module_vdw/vdw.h"
#include "source_lcao/setup_deepks.h"
#include "source_lcao/setup_exx.h"
#include "source_lcao/module_operator_lcao/dftu_lcao.h"
#include "source_lcao/module_operator_lcao/dspin_lcao.h"
#include "source_lcao/module_operator_lcao/nonlocal_new.h"
#include "source_lcao/module_operator_lcao/ekinetic_new.h"
#include "source_lcao/module_operator_lcao/overlap_new.h"
#include "source_lcao/pulay_fs.h"

template <>
void assign_dmk_ptr<double>(
    elecstate::DensityMatrix<double,double>* dm,
    std::vector<std::vector<double>>*& dmk_d,
    std::vector<std::vector<std::complex<double>>>*& dmk_c,
    bool gamma_only_local
)
{
    auto& dmk_tmp = dm->get_DMK_vector();
    dmk_d = &dmk_tmp;
    dmk_c = nullptr;
}

template <>
void assign_dmk_ptr<std::complex<double>>(
    elecstate::DensityMatrix<std::complex<double>,double>* dm,
    std::vector<std::vector<double>>*& dmk_d,
    std::vector<std::vector<std::complex<double>>>*& dmk_c,
    bool gamma_only_local
)
{
    auto& dmk_tmp = dm->get_DMK_vector();
    dmk_c = &dmk_tmp;
    dmk_d = nullptr;
}

template <typename T>
Force_Stress_LCAO<T>::Force_Stress_LCAO(Record_adj& ra, const int nat_in) : RA(&ra), nat(nat_in)
{
}

template <typename T>
Force_Stress_LCAO<T>::~Force_Stress_LCAO()
{
}

template <typename T>
void Force_Stress_LCAO<T>::getForceStress(UnitCell& ucell,
                                          const bool isforce,
                                          const bool isstress,
                                          const bool istestf,
                                          const bool istests,
                                          const Grid_Driver& gd,
                                          Parallel_Orbitals& pv,
                                          const elecstate::ElecState* pelec,
                                          LCAO_domain::Setup_DM<T> &dmat,
                                          const psi::Psi<T>* psi,
                                          const TwoCenterBundle& two_center_bundle,
                                          const LCAO_Orbitals& orb,
                                          ModuleBase::matrix& fcs,
                                          ModuleBase::matrix& scs,
                                          const pseudopot_cell_vl& locpp,
                                          const Structure_Factor& sf,
                                          const K_Vectors& kv,
                                          ModulePW::PW_Basis* rhopw,
                                          surchem& solvent,
                                          Plus_U &dftu,
                                          Setup_DeePKS<T>& deepks,
                                          Exx_NAO<T> &exx_nao,
                                          ModuleSymmetry::Symmetry* symm)
{
    ModuleBase::TITLE("Force_Stress_LCAO", "getForceStress");
    ModuleBase::timer::tick("Force_Stress_LCAO", "getForceStress");

    if (!isforce && !isstress)
    {
        ModuleBase::timer::tick("Force_Stress_LCAO", "getForceStress");
        return;
    }

    const int nat = ucell.nat;

    ModuleBase::matrix foverlap;
    ModuleBase::matrix ftvnl_dphi;
    ModuleBase::matrix fvnl_dbeta;
    ModuleBase::matrix fvl_dphi;
    ModuleBase::matrix fvl_dvl;
    ModuleBase::matrix fewalds;
    ModuleBase::matrix fcc;
    ModuleBase::matrix fscc;
    ModuleBase::matrix fvnl_dalpha;
    ModuleBase::matrix soverlap;
    ModuleBase::matrix stvnl_dphi;
    ModuleBase::matrix svnl_dbeta;
    ModuleBase::matrix svl_dphi;
    ModuleBase::matrix svnl_dalpha;
    ModuleBase::matrix sigmacc;
    ModuleBase::matrix sigmadvl;
    ModuleBase::matrix sigmaewa;
    ModuleBase::matrix sigmaxc;
    ModuleBase::matrix sigmahar;

    fvl_dphi.create(nat, 3);

    if (isforce)
    {
        fcs.create(nat, 3);
        foverlap.create(nat, 3);
        ftvnl_dphi.create(nat, 3);
        fvnl_dbeta.create(nat, 3);
        fvl_dvl.create(nat, 3);
        fewalds.create(nat, 3);
        fcc.create(nat, 3);
        fscc.create(nat, 3);
        fvnl_dalpha.create(nat, 3);

        this->calForcePwPart(ucell, fvl_dvl, fewalds, fcc, fscc, pelec->f_en.etxc,
              pelec->vnew, pelec->vnew_exist, pelec->charge, rhopw, locpp, sf);
    }

    if (isstress)
    {
        scs.create(3, 3);
        sigmacc.create(3, 3);
        sigmadvl.create(3, 3);
        sigmaewa.create(3, 3);
        sigmaxc.create(3, 3);
        sigmahar.create(3, 3);
        soverlap.create(3, 3);
        stvnl_dphi.create(3, 3);
        svnl_dbeta.create(3, 3);
        svl_dphi.create(3, 3);
        svnl_dalpha.create(3, 3);

        this->calStressPwPart(ucell, sigmadvl, sigmahar, sigmaewa, sigmacc,
          sigmaxc, pelec->f_en.etxc, pelec->charge, rhopw, locpp, sf);
    }

    elecstate::DensityMatrix<T, double> edm = flk.cal_edm(pelec, *psi, *dmat.dm, kv, pv,
                                                           PARAM.inp.nspin, PARAM.inp.nbands, ucell, *this->RA);

    this->calculate_operator_force_stress(isforce, isstress, ucell, gd, kv, two_center_bundle, orb,
                                          dmat, edm, foverlap, ftvnl_dphi, fvnl_dbeta, fvl_dphi,
                                          soverlap, stvnl_dphi, svnl_dbeta, svl_dphi);

    if (isforce)
    {
        Parallel_Reduce::reduce_pool(fvl_dphi.c, fvl_dphi.nr * fvl_dphi.nc);
    }
    if (isstress)
    {
        Parallel_Reduce::reduce_pool(svl_dphi.c, svl_dphi.nr * svl_dphi.nc);
    }

    this->calculate_deepks_force_stress(ucell, isforce, isstress, gd, kv, orb, deepks, fvnl_dalpha, svnl_dalpha);

    ModuleBase::matrix force_vdw;
    ModuleBase::matrix stress_vdw;
    auto vdw_solver = vdw::make_vdw(ucell, PARAM.inp);
    this->calculate_vdw_force_stress(ucell, isforce, isstress, force_vdw, stress_vdw);

    ModuleBase::matrix fefield;
    this->calculate_efield_force(ucell, isforce, fefield);

    ModuleBase::matrix fefield_tddft;
    this->calculate_tddft_efield_force(ucell, isforce, fefield_tddft);

    ModuleBase::matrix fgate;
    this->calculate_gatefield_force(ucell, isforce, fgate);

    ModuleBase::matrix fsol;
    this->calculate_solvent_force(ucell, isforce, rhopw, locpp, solvent, fsol);

    ModuleBase::matrix force_u;
    ModuleBase::matrix stress_u;
    this->calculate_dftu_force_stress(ucell, gd, dmat, pv, kv, isforce, isstress,
                                       two_center_bundle, orb, dftu, force_u, stress_u);

    ModuleBase::matrix force_dspin;
    ModuleBase::matrix stress_dspin;
    this->calculate_deltaspin_force_stress(ucell, gd, dmat, kv, isforce, isstress,
                                             two_center_bundle, orb, force_dspin, stress_dspin);

    ModuleBase::matrix force_exx;
    ModuleBase::matrix stress_exx;
    this->calculate_exx_force_stress(ucell, isforce, isstress, exx_nao, force_exx, stress_exx);

    if (isforce)
    {
        this->aggregate_forces(ucell, nat, fcs, foverlap, ftvnl_dphi, fvnl_dbeta, fvl_dphi,
                               fvl_dvl, fewalds, fcc, fscc, force_vdw, fefield, fefield_tddft,
                               fgate, fsol, force_u, force_dspin, force_exx, fvnl_dalpha);

        if (ModuleSymmetry::Symmetry::symm_flag == 1)
        {
            this->forceSymmetry(ucell, fcs, symm);
        }

        deepks.write_forces(fcs, fvnl_dalpha, PARAM.inp);

        this->print_force_test_output(ucell, nat, istestf, foverlap, ftvnl_dphi, fvnl_dbeta,
                                     fvl_dphi, fvl_dvl, fewalds, fcc, fscc, fefield,
                                     fefield_tddft, fgate, fsol, force_vdw, force_u,
                                     force_dspin, fvnl_dalpha, vdw_solver);

        ModuleIO::print_force(GlobalV::ofs_running, ucell, "TOTAL-FORCE (eV/Angstrom)", fcs, false);

        if (istestf)
        {
            GlobalV::ofs_running << "\n FORCE INVALID TABLE." << std::endl;
            GlobalV::ofs_running << " " << std::setw(8) << "atom" << std::setw(5) << "x" << std::setw(5) << "y"
                                 << std::setw(5) << "z" << std::endl;
            for (int iat = 0; iat < ucell.nat; iat++)
            {
                GlobalV::ofs_running << " " << std::setw(8) << iat;
                for (int i = 0; i < 3; i++)
                {
                    if (std::abs(fcs(iat, i) * ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A)
                        < Force_Stress_LCAO::force_invalid_threshold_ev)
                    {
                        fcs(iat, i) = 0.0;
                        GlobalV::ofs_running << std::setw(5) << "1";
                    }
                    else
                    {
                        GlobalV::ofs_running << std::setw(5) << "0";
                    }
                }
                GlobalV::ofs_running << std::endl;
            }
        }
    }

    if (isstress)
    {
        this->aggregate_stresses(scs, soverlap, stvnl_dphi, svnl_dbeta, svl_dphi, sigmadvl,
                                sigmaewa, sigmacc, sigmaxc, sigmahar, stress_vdw, stress_u,
                                stress_dspin, stress_exx, svnl_dalpha);

        if (ModuleSymmetry::Symmetry::symm_flag == 1)
        {
            symm->symmetrize_mat3(scs, ucell.lat);
        }

        deepks.write_stress(scs, svnl_dalpha, ucell.omega, PARAM.inp);

        this->print_stress_test_output(istests, soverlap, stvnl_dphi, svnl_dbeta, svl_dphi,
                                      sigmadvl, sigmahar, sigmaewa, sigmacc, sigmaxc,
                                      stress_vdw, stress_u, stress_dspin, scs, vdw_solver);

        bool screen_normal = true;
        bool ry = false;
        ModuleIO::print_stress("TOTAL-STRESS", scs, screen_normal, ry, GlobalV::ofs_running);

        double unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
        double external_stress[3] = {PARAM.inp.press1, PARAM.inp.press2, PARAM.inp.press3};

        for (int i = 0; i < 3; i++)
        {
            scs(i, i) -= external_stress[i] / unit_transform;
        }
    }

    ModuleBase::timer::tick("Force_Stress_LCAO", "getForceStress");
    return;
}

template Force_Stress_LCAO<double>::Force_Stress_LCAO(Record_adj&, int);
template Force_Stress_LCAO<std::complex<double>>::Force_Stress_LCAO(Record_adj&, int);
template Force_Stress_LCAO<double>::~Force_Stress_LCAO();
template Force_Stress_LCAO<std::complex<double>>::~Force_Stress_LCAO();
template void Force_Stress_LCAO<double>::getForceStress(UnitCell&, bool, bool, bool, bool, Grid_Driver const&, Parallel_Orbitals&, elecstate::ElecState const*, LCAO_domain::Setup_DM<double>&, psi::Psi<double, base_device::DEVICE_CPU> const*, TwoCenterBundle const&, LCAO_Orbitals const&, ModuleBase::matrix&, ModuleBase::matrix&, pseudopot_cell_vl const&, Structure_Factor const&, K_Vectors const&, ModulePW::PW_Basis*, surchem&, Plus_U&, Setup_DeePKS<double>&, Exx_NAO<double>&, ModuleSymmetry::Symmetry*);
template void Force_Stress_LCAO<std::complex<double>>::getForceStress(UnitCell&, bool, bool, bool, bool, Grid_Driver const&, Parallel_Orbitals&, elecstate::ElecState const*, LCAO_domain::Setup_DM<std::complex<double>>&, psi::Psi<std::complex<double>, base_device::DEVICE_CPU> const*, TwoCenterBundle const&, LCAO_Orbitals const&, ModuleBase::matrix&, ModuleBase::matrix&, pseudopot_cell_vl const&, Structure_Factor const&, K_Vectors const&, ModulePW::PW_Basis*, surchem&, Plus_U&, Setup_DeePKS<std::complex<double>>&, Exx_NAO<std::complex<double>>&, ModuleSymmetry::Symmetry*);

