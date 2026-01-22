#ifndef FORCE_STRESS_LCAO_H
#define FORCE_STRESS_LCAO_H

#include "FORCE.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/matrix.h"
#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_func.h"
#include "source_pw/module_pwdft/structure_factor.h"
#include "source_io/input_conv.h"
#include "source_psi/psi.h"
#ifdef __EXX
#include "source_lcao/module_ri/Exx_LRI_interface.h"
#endif
#include "force_stress_arrays.h"
#include "source_lcao/setup_exx.h" // for exx, mohan add 20251008
#include "source_lcao/setup_deepks.h" // for deepks, mohan add 20251010
#include "source_lcao/setup_dm.h" // mohan add 2025-11-03
#include "source_lcao/module_dftu/dftu.h" // mohan add 2025-11-07
#include "source_hamilt/module_vdw/vdw.h" // mohan add 2026-01-22


template <typename T>
class Force_Stress_LCAO
{
    // mohan add 2021-02-09
    friend class md;
    friend void Input_Conv::Convert();
    friend class ions;

  public:
    Force_Stress_LCAO(Record_adj& ra, const int nat_in);
    ~Force_Stress_LCAO();

    void getForceStress(UnitCell& ucell,
                        const bool isforce,
                        const bool isstress,
                        const bool istestf,
                        const bool istests,
                        const Grid_Driver& gd,
                        Parallel_Orbitals& pv,
                        const elecstate::ElecState* pelec,
                        LCAO_domain::Setup_DM<T> &dmat, // mohan add 2025-11-03
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
						Plus_U &dftu, // mohan add 2025-11-07
                        Setup_DeePKS<T> &deepks,
                        Exx_NAO<T> &exx_nao,
                        ModuleSymmetry::Symmetry* symm);

  private:
    int nat;
    Record_adj* RA;
    Force_LCAO<T> flk;
    Stress_Func<double> sc_pw;

    void forceSymmetry(const UnitCell& ucell, ModuleBase::matrix& fcs, ModuleSymmetry::Symmetry* symm);

    void calForcePwPart(UnitCell& ucell,
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
                        const Structure_Factor& sf);

    void integral_part(const bool isGammaOnly,
                       const bool isforce,
                       const bool isstress,
                       const UnitCell& ucell,
                       const Grid_Driver& gd,
                       ForceStressArrays& fsr, // mohan add 2024-06-15
					   const elecstate::ElecState* pelec,
					   const elecstate::DensityMatrix<T, double>* dm, // mohan add 2025-11-04
					   const psi::Psi<T>* psi,
                       ModuleBase::matrix& foverlap,
                       ModuleBase::matrix& ftvnl_dphi,
                       ModuleBase::matrix& fvnl_dbeta,
                       ModuleBase::matrix& fvl_dphi,
                       ModuleBase::matrix& soverlap,
                       ModuleBase::matrix& stvnl_dphi,
                       ModuleBase::matrix& svnl_dbeta,
                       ModuleBase::matrix& svl_dphi,
                       ModuleBase::matrix& fvnl_dalpha,
                       ModuleBase::matrix& svnl_dalpha,
                       Setup_DeePKS<T>& deepks,
                       const TwoCenterBundle& two_center_bundle,
                       const LCAO_Orbitals& orb,
                       const Parallel_Orbitals& pv,
                       const K_Vectors& kv);

    void calStressPwPart(UnitCell& ucell,
                         ModuleBase::matrix& sigmadvl,
                         ModuleBase::matrix& sigmahar,
                         ModuleBase::matrix& sigmaewa,
                         ModuleBase::matrix& sigmacc,
                         ModuleBase::matrix& sigmaxc,
                         const double& etxc,
                         const Charge* const chr,
                         ModulePW::PW_Basis* rhopw,
                         const pseudopot_cell_vl& locpp,
                         const Structure_Factor& sf);

    void calculate_operator_force_stress(
        const bool isforce,
        const bool isstress,
        const UnitCell& ucell,
        const Grid_Driver& gd,
        const K_Vectors& kv,
        const TwoCenterBundle& two_center_bundle,
        const LCAO_Orbitals& orb,
        LCAO_domain::Setup_DM<T>& dmat,
        const elecstate::DensityMatrix<T, double>& edm,
        ModuleBase::matrix& foverlap,
        ModuleBase::matrix& ftvnl_dphi,
        ModuleBase::matrix& fvnl_dbeta,
        ModuleBase::matrix& fvl_dphi,
        ModuleBase::matrix& soverlap,
        ModuleBase::matrix& stvnl_dphi,
        ModuleBase::matrix& svnl_dbeta,
        ModuleBase::matrix& svl_dphi);

    void calculate_operator_force_stress_spin12(
        const bool isforce,
        const bool isstress,
        const UnitCell& ucell,
        const Grid_Driver& gd,
        const K_Vectors& kv,
        const TwoCenterBundle& two_center_bundle,
        const LCAO_Orbitals& orb,
        LCAO_domain::Setup_DM<T>& dmat,
        const elecstate::DensityMatrix<T, double>& edm,
        ModuleBase::matrix& foverlap,
        ModuleBase::matrix& ftvnl_dphi,
        ModuleBase::matrix& fvnl_dbeta,
        ModuleBase::matrix& fvl_dphi,
        ModuleBase::matrix& soverlap,
        ModuleBase::matrix& stvnl_dphi,
        ModuleBase::matrix& svnl_dbeta,
        ModuleBase::matrix& svl_dphi);

    void calculate_operator_force_stress_spin4(
        const bool isforce,
        const bool isstress,
        const UnitCell& ucell,
        const Grid_Driver& gd,
        const K_Vectors& kv,
        const TwoCenterBundle& two_center_bundle,
        const LCAO_Orbitals& orb,
        LCAO_domain::Setup_DM<T>& dmat,
        const elecstate::DensityMatrix<T, double>& edm,
        ModuleBase::matrix& foverlap,
        ModuleBase::matrix& ftvnl_dphi,
        ModuleBase::matrix& fvnl_dbeta,
        ModuleBase::matrix& fvl_dphi,
        ModuleBase::matrix& soverlap,
        ModuleBase::matrix& stvnl_dphi,
        ModuleBase::matrix& svnl_dbeta,
        ModuleBase::matrix& svl_dphi);

    void calculate_deepks_force_stress(
        const UnitCell& ucell,
        const bool isforce,
        const bool isstress,
        const Grid_Driver& gd,
        const K_Vectors& kv,
        const LCAO_Orbitals& orb,
        Setup_DeePKS<T>& deepks,
        ModuleBase::matrix& fvnl_dalpha,
        ModuleBase::matrix& svnl_dalpha);

    void calculate_vdw_force_stress(
        const UnitCell& ucell,
        const bool isforce,
        const bool isstress,
        ModuleBase::matrix& force_vdw,
        ModuleBase::matrix& stress_vdw);

    void calculate_efield_force(
        const UnitCell& ucell,
        const bool isforce,
        ModuleBase::matrix& fefield);

    void calculate_tddft_efield_force(
        const UnitCell& ucell,
        const bool isforce,
        ModuleBase::matrix& fefield_tddft);

    void calculate_gatefield_force(
        const UnitCell& ucell,
        const bool isforce,
        ModuleBase::matrix& fgate);

    void calculate_solvent_force(
        UnitCell& ucell,
        const bool isforce,
        ModulePW::PW_Basis* rhopw,
        const pseudopot_cell_vl& locpp,
        surchem& solvent,
        ModuleBase::matrix& fsol);

    void calculate_dftu_force_stress(
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
        ModuleBase::matrix& stress_u);

    void calculate_deltaspin_force_stress(
        const UnitCell& ucell,
        const Grid_Driver& gd,
        LCAO_domain::Setup_DM<T>& dmat,
        const K_Vectors& kv,
        const bool isforce,
        const bool isstress,
        const TwoCenterBundle& two_center_bundle,
        const LCAO_Orbitals& orb,
        ModuleBase::matrix& force_dspin,
        ModuleBase::matrix& stress_dspin);

    void calculate_exx_force_stress(
        const UnitCell& ucell,
        const bool isforce,
        const bool isstress,
        Exx_NAO<T>& exx_nao,
        ModuleBase::matrix& force_exx,
        ModuleBase::matrix& stress_exx);

    void aggregate_forces(
        const UnitCell& ucell,
        const int nat,
        ModuleBase::matrix& fcs,
        const ModuleBase::matrix& foverlap,
        const ModuleBase::matrix& ftvnl_dphi,
        const ModuleBase::matrix& fvnl_dbeta,
        const ModuleBase::matrix& fvl_dphi,
        const ModuleBase::matrix& fvl_dvl,
        const ModuleBase::matrix& fewalds,
        const ModuleBase::matrix& fcc,
        const ModuleBase::matrix& fscc,
        const ModuleBase::matrix& force_vdw,
        const ModuleBase::matrix& fefield,
        const ModuleBase::matrix& fefield_tddft,
        const ModuleBase::matrix& fgate,
        const ModuleBase::matrix& fsol,
        const ModuleBase::matrix& force_u,
        const ModuleBase::matrix& force_dspin,
        const ModuleBase::matrix& force_exx,
        const ModuleBase::matrix& fvnl_dalpha);

    void aggregate_stresses(
        ModuleBase::matrix& scs,
        const ModuleBase::matrix& soverlap,
        const ModuleBase::matrix& stvnl_dphi,
        const ModuleBase::matrix& svnl_dbeta,
        const ModuleBase::matrix& svl_dphi,
        const ModuleBase::matrix& sigmadvl,
        const ModuleBase::matrix& sigmaewa,
        const ModuleBase::matrix& sigmacc,
        const ModuleBase::matrix& sigmaxc,
        const ModuleBase::matrix& sigmahar,
        const ModuleBase::matrix& stress_vdw,
        const ModuleBase::matrix& stress_u,
        const ModuleBase::matrix& stress_dspin,
        const ModuleBase::matrix& stress_exx,
        const ModuleBase::matrix& svnl_dalpha);

    void print_force_test_output(
        const UnitCell& ucell,
        const int nat,
        const bool istestf,
        const ModuleBase::matrix& foverlap,
        const ModuleBase::matrix& ftvnl_dphi,
        const ModuleBase::matrix& fvnl_dbeta,
        const ModuleBase::matrix& fvl_dphi,
        const ModuleBase::matrix& fvl_dvl,
        const ModuleBase::matrix& fewalds,
        const ModuleBase::matrix& fcc,
        const ModuleBase::matrix& fscc,
        const ModuleBase::matrix& fefield,
        const ModuleBase::matrix& fefield_tddft,
        const ModuleBase::matrix& fgate,
        const ModuleBase::matrix& fsol,
        const ModuleBase::matrix& force_vdw,
        const ModuleBase::matrix& force_u,
        const ModuleBase::matrix& force_dspin,
        const ModuleBase::matrix& fvnl_dalpha,
        const std::unique_ptr<vdw::Vdw>& vdw_solver);

    void print_stress_test_output(
        const int istests,
        const ModuleBase::matrix& soverlap,
        const ModuleBase::matrix& stvnl_dphi,
        const ModuleBase::matrix& svnl_dbeta,
        const ModuleBase::matrix& svl_dphi,
        const ModuleBase::matrix& sigmadvl,
        const ModuleBase::matrix& sigmahar,
        const ModuleBase::matrix& sigmaewa,
        const ModuleBase::matrix& sigmacc,
        const ModuleBase::matrix& sigmaxc,
        const ModuleBase::matrix& stress_vdw,
        const ModuleBase::matrix& stress_u,
        const ModuleBase::matrix& stress_dspin,
        const ModuleBase::matrix& scs,
        const std::unique_ptr<vdw::Vdw>& vdw_solver);

    static double force_invalid_threshold_ev;
};

template <typename T>
double Force_Stress_LCAO<T>::force_invalid_threshold_ev = 0.00;

template <typename T>
void assign_dmk_ptr(
    elecstate::DensityMatrix<T,double>* dm,
    std::vector<std::vector<double>>*& dmk_d,
    std::vector<std::vector<std::complex<double>>>*& dmk_c,
    bool gamma_only_local
);

#endif
