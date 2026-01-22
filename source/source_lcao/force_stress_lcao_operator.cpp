#include "FORCE_STRESS.h"
#include "source_lcao/module_operator_lcao/ekinetic_new.h"
#include "source_lcao/module_operator_lcao/overlap_new.h"
#include "source_lcao/module_operator_lcao/nonlocal_new.h"
#include "source_lcao/pulay_fs.h"

template <typename T>
void Force_Stress_LCAO<T>::calculate_operator_force_stress(
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
    ModuleBase::matrix& svl_dphi)
{
    if (!isforce && !isstress)
    {
        return;
    }

    if (PARAM.inp.nspin == 1 || PARAM.inp.nspin == 2)
    {
        calculate_operator_force_stress_spin12(isforce, isstress, ucell, gd, kv, two_center_bundle,
                                               orb, dmat, edm, foverlap, ftvnl_dphi, fvnl_dbeta,
                                               fvl_dphi, soverlap, stvnl_dphi, svnl_dbeta, svl_dphi);
    }
    else if (PARAM.inp.nspin == 4)
    {
        calculate_operator_force_stress_spin4(isforce, isstress, ucell, gd, kv, two_center_bundle,
                                              orb, dmat, edm, foverlap, ftvnl_dphi, fvnl_dbeta,
                                              fvl_dphi, soverlap, stvnl_dphi, svnl_dbeta, svl_dphi);
    }
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_operator_force_stress_spin12(
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
    ModuleBase::matrix& svl_dphi)
{
    if (PARAM.inp.nspin == 2)
    {
        dmat.dm->switch_dmr(1);
        const_cast<elecstate::DensityMatrix<T, double>&>(edm).switch_dmr(1);
    }

    const hamilt::HContainer<double>* dmR = dmat.dm->get_DMR_pointer(1);
    const hamilt::HContainer<double>* edmR = edm.get_DMR_pointer(1);

    if (PARAM.inp.t_in_h)
    {
        hamilt::EkineticNew<hamilt::OperatorLCAO<T, double>> tmp_ekinetic(
            nullptr, kv.kvec_d, nullptr, &ucell, orb.cutoffs(), &gd,
            two_center_bundle.kinetic_orb.get());
        tmp_ekinetic.cal_force_stress(isforce, isstress, dmR, ftvnl_dphi, stvnl_dphi);
    }

    hamilt::OverlapNew<hamilt::OperatorLCAO<T, double>> tmp_overlap(
        nullptr, kv.kvec_d, nullptr, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb.get());
    tmp_overlap.cal_force_stress(isforce, isstress, edmR, foverlap, soverlap);

    hamilt::NonlocalNew<hamilt::OperatorLCAO<T, double>> tmp_nonlocal(
        nullptr, kv.kvec_d, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb_beta.get());
    tmp_nonlocal.cal_force_stress(isforce, isstress, dmR, fvnl_dbeta, svnl_dbeta);

    if (PARAM.inp.nspin == 2)
    {
        dmat.dm->switch_dmr(0);
        const_cast<elecstate::DensityMatrix<T, double>&>(edm).switch_dmr(0);
    }

    flk.ParaV = dmat.dm->get_paraV_pointer();
    PulayForceStress::cal_pulay_fs(fvl_dphi, svl_dphi, *dmat.dm, ucell, nullptr,
                                   isforce, isstress, false);
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_operator_force_stress_spin4(
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
    ModuleBase::matrix& svl_dphi)
{
    if (PARAM.inp.t_in_h)
    {
        hamilt::EkineticNew<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>> tmp_ekinetic(
            nullptr, kv.kvec_d, nullptr, &ucell, orb.cutoffs(), &gd,
            two_center_bundle.kinetic_orb.get());
        tmp_ekinetic.cal_force_stress(isforce, isstress, dmat.dm->get_DMR_pointer(1), ftvnl_dphi, stvnl_dphi);
    }

    hamilt::OverlapNew<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>> tmp_overlap(
        nullptr, kv.kvec_d, nullptr, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb.get());
    tmp_overlap.cal_force_stress(isforce, isstress, edm.get_DMR_pointer(1), foverlap, soverlap);

    hamilt::HContainer<std::complex<double>> tmp_dmr(dmat.dm->get_DMR_pointer(1)->get_paraV());
    std::vector<int> ijrs = dmat.dm->get_DMR_pointer(1)->get_ijr_info();
    tmp_dmr.insert_ijrs(&ijrs);
    tmp_dmr.allocate();
    dmat.dm->cal_DMR_full(&tmp_dmr);

    hamilt::NonlocalNew<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>> tmp_nonlocal(
        nullptr, kv.kvec_d, nullptr, &ucell, orb.cutoffs(), &gd,
        two_center_bundle.overlap_orb_beta.get());
    tmp_nonlocal.cal_force_stress(isforce, isstress, &tmp_dmr, fvnl_dbeta, svnl_dbeta);

    flk.ParaV = dmat.dm->get_paraV_pointer();
    PulayForceStress::cal_pulay_fs(fvl_dphi, svl_dphi, *dmat.dm, ucell, nullptr,
                                   isforce, isstress, false);
}

template <typename T>
void Force_Stress_LCAO<T>::calculate_deepks_force_stress(
    const UnitCell& ucell,
    const bool isforce,
    const bool isstress,
    const Grid_Driver& gd,
    const K_Vectors& kv,
    const LCAO_Orbitals& orb,
    Setup_DeePKS<T>& deepks,
    ModuleBase::matrix& fvnl_dalpha,
    ModuleBase::matrix& svnl_dalpha)
{
#ifdef __MLALGO
    if (!PARAM.inp.deepks_scf)
    {
        return;
    }

    const int nks = (PARAM.inp.nspin == 1 || PARAM.inp.nspin == 2) ? 1 : kv.get_nks();
    if (PARAM.globalv.gamma_only_local)
    {
        DeePKS_domain::cal_f_delta<double>(deepks.ld.dm_r, ucell, orb, gd,
                                           *flk.ParaV, nks, deepks.ld.deepks_param,
                                           kv.kvec_d, deepks.ld.phialpha, deepks.ld.gedm,
                                           fvnl_dalpha, isstress, svnl_dalpha);
    }
    else
    {
        DeePKS_domain::cal_f_delta<std::complex<double>>(deepks.ld.dm_r, ucell, orb, gd,
                                                         *flk.ParaV, nks, deepks.ld.deepks_param,
                                                         kv.kvec_d, deepks.ld.phialpha, deepks.ld.gedm,
                                                         fvnl_dalpha, isstress, svnl_dalpha);
    }

    if (isforce)
    {
        Parallel_Reduce::reduce_pool(fvnl_dalpha.c, fvnl_dalpha.nr * fvnl_dalpha.nc);
    }
    if (isstress)
    {
        Parallel_Reduce::reduce_pool(svnl_dalpha.c, svnl_dalpha.nr * svnl_dalpha.nc);
    }
#else
    (void)ucell;
    (void)isforce;
    (void)isstress;
    (void)gd;
    (void)kv;
    (void)orb;
    (void)deepks;
    (void)fvnl_dalpha;
    (void)svnl_dalpha;
#endif
}

template class Force_Stress_LCAO<double>;
template class Force_Stress_LCAO<std::complex<double>>;

