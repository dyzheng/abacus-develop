#include "FORCE_STRESS.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/output_log.h"

template <typename T>
void Force_Stress_LCAO<T>::aggregate_forces(
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
    const ModuleBase::matrix& fvnl_dalpha)
{
    for (int i = 0; i < 3; i++)
    {
        double sum = 0.0;

        for (int iat = 0; iat < nat; iat++)
        {
            fcs(iat, i) += foverlap(iat, i) + ftvnl_dphi(iat, i) + fvnl_dbeta(iat, i) + fvl_dphi(iat, i)
                           + fvl_dvl(iat, i)
                           + fewalds(iat, i)
                           + fcc(iat, i)
                           + fscc(iat, i);

            if (PARAM.inp.dft_plus_u)
            {
                fcs(iat, i) += force_u(iat, i);
            }
            if (PARAM.inp.sc_mag_switch)
            {
                fcs(iat, i) += force_dspin(iat, i);
            }
#ifdef __EXX
            if (GlobalC::exx_info.info_global.cal_exx)
            {
                fcs(iat, i) += force_exx(iat, i);
            }
#endif
            if (force_vdw.nr > 0)
            {
                fcs(iat, i) += force_vdw(iat, i);
            }
            if (PARAM.inp.efield_flag)
            {
                fcs(iat, i) += fefield(iat, i);
            }
            if (PARAM.inp.esolver_type == "tddft")
            {
                fcs(iat, i) += fefield_tddft(iat, i);
            }
            if (PARAM.inp.gate_flag)
            {
                fcs(iat, i) += fgate(iat, i);
            }
            if (PARAM.inp.imp_sol)
            {
                fcs(iat, i) += fsol(iat, i);
            }
#ifdef __MLALGO
            if (PARAM.inp.deepks_scf)
            {
                fcs(iat, i) += fvnl_dalpha(iat, i);
            }
#endif
            sum += fcs(iat, i);
        }

        if (!(PARAM.inp.gate_flag || PARAM.inp.efield_flag))
        {
            for (int iat = 0; iat < nat; ++iat)
            {
                fcs(iat, i) -= sum / nat;
            }
        }
    }

    if (PARAM.inp.gate_flag || PARAM.inp.efield_flag)
    {
        GlobalV::ofs_running << "Atomic forces are not shifted if gate_flag or efield_flag == true!" << std::endl;
    }
}

template <typename T>
void Force_Stress_LCAO<T>::aggregate_stresses(
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
    const ModuleBase::matrix& svnl_dalpha)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            scs(i, j) += soverlap(i, j) + stvnl_dphi(i, j) + svnl_dbeta(i, j) + svl_dphi(i, j)
                         + sigmadvl(i, j)
                         + sigmaewa(i, j)
                         + sigmacc(i, j)
                         + sigmaxc(i, j)
                         + sigmahar(i, j);

            if (stress_vdw.nr > 0)
            {
                scs(i, j) += stress_vdw(i, j);
            }
            if (PARAM.inp.dft_plus_u)
            {
                scs(i, j) += stress_u(i, j);
            }
            if (PARAM.inp.sc_mag_switch)
            {
                scs(i, j) += stress_dspin(i, j);
            }
#ifdef __EXX
            if (GlobalC::exx_info.info_global.cal_exx)
            {
                scs(i, j) += stress_exx(i, j);
            }
#endif
#ifdef __MLALGO
            if (PARAM.inp.deepks_scf)
            {
                scs(i, j) += svnl_dalpha(i, j);
            }
#endif
        }
    }
}

template <typename T>
void Force_Stress_LCAO<T>::print_force_test_output(
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
    const std::unique_ptr<vdw::Vdw>& vdw_solver)
{
    if (!istestf)
    {
        return;
    }

    ModuleBase::matrix ftvnl;
    ftvnl.create(nat, 3);
    for (int iat = 0; iat < nat; iat++)
    {
        for (int i = 0; i < 3; i++)
        {
            ftvnl(iat, i) = ftvnl_dphi(iat, i) + fvnl_dbeta(iat, i);
        }
    }

    GlobalV::ofs_running << "\n PARTS OF FORCE: " << std::endl;
    GlobalV::ofs_running << std::setiosflags(std::ios::showpos);
    GlobalV::ofs_running << std::setiosflags(std::ios::fixed) << std::setprecision(8) << std::endl;

    ModuleIO::print_force(GlobalV::ofs_running, ucell, "OVERLAP    FORCE", foverlap, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "TVNL_DPHI  force", ftvnl_dphi, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "VNL_DBETA  force", fvnl_dbeta, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "T_VNL      FORCE", ftvnl, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "VL_dPHI    FORCE", fvl_dphi, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "VL_dVL     FORCE", fvl_dvl, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "EWALD      FORCE", fewalds, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "NLCC       FORCE", fcc, false);
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "SCC        FORCE", fscc, false);

    if (PARAM.inp.efield_flag)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "EFIELD     FORCE", fefield, false);
    }
    if (PARAM.inp.esolver_type == "tddft")
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "EFIELD_TDDFT     FORCE", fefield_tddft, false);
    }
    if (PARAM.inp.gate_flag)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "GATEFIELD     FORCE", fgate, false);
    }
    if (PARAM.inp.imp_sol)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "IMP_SOL     FORCE", fsol, false);
    }
    if (vdw_solver != nullptr)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "VDW        FORCE", force_vdw, false);
    }
    if (PARAM.inp.dft_plus_u)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "DFT+U      FORCE", force_u, false);
    }
    if (PARAM.inp.sc_mag_switch)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "DeltaSpin  FORCE", force_dspin, false);
    }
#ifdef __MLALGO
    if (PARAM.inp.deepks_scf)
    {
        ModuleIO::print_force(GlobalV::ofs_running, ucell, "DeePKS 	FORCE", fvnl_dalpha, true);
    }
#endif

    GlobalV::ofs_running << std::setiosflags(std::ios::left);
}

template <typename T>
void Force_Stress_LCAO<T>::print_stress_test_output(
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
    const std::unique_ptr<vdw::Vdw>& vdw_solver)
{
    if (!istests)
    {
        return;
    }

    ModuleBase::matrix svlocal;
    svlocal.create(3, 3);
    ModuleBase::matrix stvnl;
    stvnl.create(3, 3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            svlocal(i, j) = svl_dphi(i, j) + sigmadvl(i, j);
            stvnl(i, j) = stvnl_dphi(i, j) + svnl_dbeta(i, j);
        }
    }

    const bool screen = PARAM.inp.test_stress;
    const bool ry = false;

    GlobalV::ofs_running << "\n PARTS OF STRESS: " << std::endl;
    GlobalV::ofs_running << std::setiosflags(std::ios::showpos);
    GlobalV::ofs_running << std::setiosflags(std::ios::fixed) << std::setprecision(10) << std::endl;
    ModuleIO::print_stress("OVERLAP  STRESS", soverlap, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("T        STRESS", stvnl_dphi, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("VNL      STRESS", svnl_dbeta, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("T_VNL    STRESS", stvnl, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("VL_dPHI  STRESS", svl_dphi, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("VL_dVL   STRESS", sigmadvl, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("HAR      STRESS", sigmahar, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("EWALD    STRESS", sigmaewa, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("cc       STRESS", sigmacc, screen, ry, GlobalV::ofs_running);
    ModuleIO::print_stress("XC       STRESS", sigmaxc, screen, ry, GlobalV::ofs_running);
    if (vdw_solver != nullptr)
    {
        ModuleIO::print_stress("VDW      STRESS", sigmaxc, screen, ry, GlobalV::ofs_running);
    }
    if (PARAM.inp.dft_plus_u)
    {
        ModuleIO::print_stress("DFTU     STRESS", stress_u, screen, ry, GlobalV::ofs_running);
    }
    if (PARAM.inp.sc_mag_switch)
    {
        ModuleIO::print_stress("DeltaSpin  STRESS", stress_dspin, screen, ry, GlobalV::ofs_running);
    }
    ModuleIO::print_stress("TOTAL    STRESS", scs, screen, ry, GlobalV::ofs_running);

    GlobalV::ofs_running << std::setiosflags(std::ios::left);
}

template class Force_Stress_LCAO<double>;
template class Force_Stress_LCAO<std::complex<double>>;

