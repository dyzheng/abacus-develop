#include "esolver_ks_pw.h"

#include "source_estate/cal_ux.h"
#include "source_estate/elecstate_pw.h"
#include "source_estate/module_charge/symmetry_rho.h"

#include "source_hsolver/diago_iter_assist.h"
#include "source_hsolver/hsolver_pw.h"
#include "source_hsolver/diago_params.h"

#include "source_hsolver/kernels/hegvd_op.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/vsep_pw.h"
#include "source_pw/module_pwdft/hamilt_pw.h"

#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_pw.h"
#include "source_hamilt/module_xc/xc_functional.h" // use XC_Functional

#ifdef __DSP
#include "source_base/kernels/dsp/dsp_connector.h"
#endif

#include "source_pw/module_pwdft/setup_pot.h" // mohan add 20250929
#include "source_estate/setup_estate_pw.h" // mohan add 20251005
#include "source_io/module_ctrl/ctrl_output_pw.h" // mohan add 20250927
#include "source_estate/module_charge/chgmixing.h" // use charge mixing, mohan add 20251006 
#include "source_estate/update_pot.h" // mohan add 20251016
#include "source_pw/module_pwdft/update_cell_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/dftu_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/deltaspin_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include "source_estate/module_constraint/constraint_loop.h"

#include "source_hamilt/module_xc/exx_info.h" // use GlobalC::exx_info

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::ESolver_KS_PW()
{
    this->classname = "ESolver_KS_PW";
    this->basisname = "PW";
}

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::~ESolver_KS_PW()
{
    //****************************************************
    // do not add any codes in this deconstructor funcion
    //****************************************************
    // delete Hamilt
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    // delete exx_helper
    if (this->exx_helper != nullptr)
    {
        delete this->exx_helper;
        this->exx_helper = nullptr;
    }

    // mohan add 2025-10-12
    this->stp.clean();
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::allocate_hamilt(const UnitCell& ucell)
{
	this->p_hamilt = new hamilt::HamiltPW<T, Device>(
			this->pelec->pot, 
			this->pw_wfc, 
			&this->kv, 
			&this->ppcell, 
			&this->dftu,
			&ucell);
}



template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ESolver_KS::before_all_runners(ucell, inp);

    //! setup and allocation for pelec, potentials, etc. 
    elecstate::setup_estate_pw(ucell, this->kv, this->sf, this->pelec, this->chr,
      this->locpp, this->ppcell, this->vsep_cell, this->pw_wfc, this->pw_rho,
      this->pw_rhod, this->pw_big, this->solvent, inp);

    this->stp.before_runner(ucell, this->kv, this->sf, *this->pw_wfc, this->ppcell, PARAM.inp);

    // Initialize DeltaP PW: snapshot INPUT + STRU into the shared
    // DeltapScfSolver state machine (owned by pw_deltap).  The backend
    // closes over psi/kv/wfcpw/rhopw, which are stable for the whole run.
    if (PARAM.inp.deltap_switch)
    {
        pw_deltap::deltap_init(ucell, PARAM.inp,
                               this->stp.psi_cpu, &this->kv,
                               this->pw_wfc, this->pw_rho,
                               &this->pelec->wg);
    }

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT BASIS");

    //! Create exx_helper based on device and precision
    const bool is_gpu = (inp.device == "gpu");
    const bool is_single = (inp.precision == "single");

#if ((defined __CUDA) || (defined __ROCM))
    if (is_gpu)
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_GPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_GPU>();
        }
    }
    else
#endif
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }

    //! Initialize exx pw
    this->exx_helper->init(ucell, inp, this->pelec->wg);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "before_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "before_scf");

    ESolver_KS::before_scf(ucell, istep);

    //! Init variables (once the cell has changed)
    pw::update_cell_pw(ucell, this->ppcell, this->kv, this->pw_wfc, PARAM.inp);

    if (ucell.cell_parameter_updated)
    {
        this->stp.p_psi_init->prepare_init(PARAM.inp.pw_seed);
    }

    //! Init Hamiltonian (cell changed)
    //! Operators in HamiltPW should be reallocated once cell changed
    //! delete Hamilt if not first scf
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    //! Allocate HamiltPW
    this->allocate_hamilt(ucell);

    // Per-SCF-cycle reset: allow one DeltaP lambda update per ionic step.
    pw_deltap::reset_deltap_pw_scf_cycle();

    //! Setup potentials (local, non-local, sc, +U, DFT-1/2)
    // note: init DFT+U is done here for pw basis for every scf iteration, however, 
    // init DFT+U is done in "before_all_runners" in LCAO basis. This should be refactored, mohan note 2025-11-06
    pw::setup_pot(istep, ucell, this->kv, this->sf, this->pelec, this->Pgrid,
              this->chr, this->locpp, this->ppcell, this->dftu, this->vsep_cell,
              this->stp.template get_psi_t<T, Device>(), 
	      this->p_hamilt, 
	      this->pw_wfc, this->pw_rhod, PARAM.inp);

    // setup psi (electronic wave functions)
    this->stp.init(this->p_hamilt);

    //! Setup EXX helper for Hamiltonian and psi
    exx_helper->before_scf(this->p_hamilt, this->stp.template get_psi_t<T, Device>(), PARAM.inp);

    // Real-space weight constraint (phase 1/2, PW + Becke): configure from
    // INPUT via the shared module function (PW and LCAO channels must
    // observe identical guards, defaults and partition radii), then build
    // the shared weight field (M1) and arm the outer loop.
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintConfig cfg;
        std::vector<constraint::ConstraintSpec> specs;
        std::vector<double> radii;
        std::string error;
        const constraint::ConfigStatus st = constraint::configure_from_inputs(
            cfg, specs, ucell, radii, error);
        if (st == constraint::ConfigStatus::ERROR)
        {
            ModuleBase::WARNING_QUIT("ESolver_KS_PW::before_scf", error);
        }
        constraint::ConstraintLoop::instance().init(
            ucell, this->pw_rhod, cfg, specs, radii, PARAM.inp.nelec);
    }

    ModuleBase::timer::end("ESolver_KS_PW", "before_scf");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_init(UnitCell& ucell, const int istep, const int iter)
{
    ESolver_KS::iter_init(ucell, istep, iter);

    module_charge::chgmixing_ks_pw(iter, this->p_chgmix, this->dftu, PARAM.inp);

    // mohan move harris functional here, 2012-06-05
    // use 'rho(in)' and 'v_h and v_xc'(in)
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband(ucell);

    // update local occupations for DFT+U
    // should before lambda loop in DeltaSpin
    pw::iter_init_dftu_pw(iter, istep, this->dftu, this->stp.template get_psi_t<T, Device>(), this->pelec->wg, ucell, this->p_chgmix, this->kv.isk.data());
}

// Temporary, it should be replaced by hsolver later.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2rho_single(UnitCell& ucell, const int istep, const int iter, const double ethr)
{
    ModuleBase::timer::start("ESolver_KS_PW", "hamilt2rho_single");

    // reset energy
    this->pelec->f_en.eband = 0.0;
    this->pelec->f_en.demet = 0.0;

    // setup diagonalization parameters
    hsolver::setup_diago_params_pw<T, Device>(istep, iter, ethr, PARAM.inp);

    bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;

    // Real-space weight constraint (phase 1): inject the current mu-weighted
    // potential into v_eff and veff_smooth before the diagonalization.  In
    // the reference phase mu is zero, so the first SCF stays unconstrained.
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintLoop::instance().inject_potential(
            iter, this->pelec->pot->get_eff_v(),
            this->pelec->pot->get_veff_smooth());
    }

    // run the inner lambda loop to contrain atomic moments with the DeltaSpin method
    bool skip_solve = pw::run_deltaspin_lambda_loop(iter - 1, this->drho, PARAM.inp);

    // DeltaP (Phase A): no inner lambda loop in PW — lambda is updated
    // synchronously in iter_finish (deltap_iter_finish) after charge
    // convergence, so the normal HSolver below always runs.

    if (!skip_solve)
    {
        hsolver::HSolverPW<T, Device> hsolver_pw_obj(this->pw_wfc,
                                                     PARAM.inp.calculation,
                                                     PARAM.inp.basis_type,
                                                     PARAM.inp.ks_solver,
                                                     PARAM.globalv.use_uspp,
                                                     PARAM.inp.nspin,
                                                     hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                     hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                     PARAM.inp.use_k_continuity);

        hsolver_pw_obj.solve(static_cast<hamilt::Hamilt<T, Device>*>(this->p_hamilt), *this->stp.template get_psi_t<T, Device>(), this->pelec, this->pelec->ekb.c,
          GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL, skip_charge, ucell.tpiba, ucell.nat);
    }

    // symmetrize the charge density
    Symmetry_rho::symmetrize_rho(PARAM.inp.nspin, this->chr, this->pw_rhod, ucell.symm);

    ModuleBase::timer::end("ESolver_KS_PW", "hamilt2rho_single");
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    // DeltaP (F-7b, 2026-08-17): refresh the escon bookkeeping on the current
    // wavefunctions before the energy evaluation below, so the iteration's
    // total energy (eband + ... + dp_escon) carries an escon measured on the
    // same ψ as its eigenvalues.  Without this, the one-shot escon from the
    // first drho<deltap_inner_thr iteration enters FINAL_ETOT and leaves a
    // ~0.1–0.3% stale Γ (E'(λ) flatness artifact).
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    {
        pw_deltap::refresh_pw_escon(ucell, this->stp.psi_cpu, this->pelec->wg);
        this->pelec->f_en.dp_escon = pw_deltap::get_deltap_pw_escon();
    }

    // Related to EXX
    if (GlobalC::exx_info.info_global.cal_exx && !exx_helper->get_op_first_iter())
    {
        this->pelec->set_exx(exx_helper->cal_exx_energy(this->stp.template get_psi_t<T, Device>()));
    }

    // deband is calculated from "output" charge density
    this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);

    // Call iter_finish() of ESolver_KS
    ESolver_KS::iter_finish(ucell, istep, iter, conv_esolver);

    // D in USPP needs vloc, thus needs update when veff updated
    // calculate the effective coefficient matrix for non-local
    // pp projectors, liuyu 2023-10-24
    if (PARAM.globalv.use_uspp)
    {
        ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
        this->ppcell.cal_effective_D(veff, this->pw_rhod, ucell);
    }

    // Handle EXX-related operations after SCF iteration
    exx_helper->iter_finish(this->pelec, &this->chr, this->stp.template get_psi_t<T, Device>(), ucell, PARAM.inp, conv_esolver, iter);

    // check if oscillate for delta_spin method
    pw::check_deltaspin_oscillation(iter, this->drho, this->p_chgmix, PARAM.inp);

    // DeltaP: compute gamma and update lambda after SCF iteration
    pw_deltap::deltap_iter_finish(ucell, this->drho,
        this->stp.psi_cpu, this->kv, this->pw_wfc, this->pw_rho, PARAM.inp);
    // Apply DeltaP constraint energy correction to f_en
    // dp_escon is identical on every rank: γ is rank-0-synced inside the PW
    // backend's compute_gamma (T3 Bcast) and λ by sync_lambda, so the
    // rank-local assignment is consistent.
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
        this->pelec->f_en.dp_escon = pw_deltap::get_deltap_pw_escon();

    // Real-space weight constraint (phase 1): read the constraint charges
    // from the mixed density and run the outer-loop bookkeeping (reference
    // recording / M4 secant step / audit line).  The hook may override
    // conv_esolver to keep the SCF running until the constraint converges or
    // fuses (two-stage gating, DeltaP lineage).
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintLoop& cloop
            = constraint::ConstraintLoop::instance();
        cloop.observe(iter, this->chr.rho, PARAM.inp.nspin);
        cloop.on_scf_converged(iter, conv_esolver);
        // Refresh the total energy with the constraint correction
        // (cc_escon), mirroring the dp_escon path.
        this->pelec->f_en.cc_escon = cloop.last_audit().e_con;
        if (cloop.enabled())
        {
            this->pelec->f_en.calculate_etot();
        }
    }

    // the output quantities
    ModuleIO::ctrl_iter_pw(istep, iter, conv_esolver, this->stp.psi_cpu, 
              this->kv, this->pw_wfc, PARAM.inp);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_scf(UnitCell& ucell, const int istep, const bool conv_esolver)
{
    ModuleBase::TITLE("ESolver_KS_PW", "after_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "after_scf");

    // Calculate kinetic energy density tau for ELF if needed
    if (PARAM.inp.out_elf[0] > 0)
    {
        this->pelec->cal_tau(*(this->stp.psi_cpu));
    }

    ESolver_KS::after_scf(ucell, istep, conv_esolver);

    // Output quantities
    ModuleIO::ctrl_scf_pw<T, Device>(istep, ucell, this->pelec, this->chr, this->kv, this->pw_wfc,
              this->pw_rho, this->pw_rhod, this->pw_big, this->stp,
              this->Pgrid, PARAM.inp);

    // Real-space weight constraint (phase 1): final audit report.
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintLoop::instance().final_report();
    }

    ModuleBase::timer::end("ESolver_KS_PW", "after_scf");
}

template <typename T, typename Device>
double ESolver_KS_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    Forces<double, Device> ff(ucell.nat);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    // Calculate forces
    ff.cal_force(ucell, force, *this->pelec, this->pw_rhod, &ucell.symm,
                 &this->sf, this->solvent, &this->dftu, &this->locpp, &this->ppcell, 
                 &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>());
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    Stress_PW<double, Device> ss(this->pelec);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    ss.cal_stress(stress, ucell, this->dftu, this->locpp, this->ppcell, this->pw_rhod,
                  &ucell.symm, &this->sf, &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>());

    // external stress
    double unit_transform = 0.0;
    unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    double external_stress[3] = {PARAM.inp.press1, PARAM.inp.press2, PARAM.inp.press3};
    for (int i = 0; i < 3; i++)
    {
        stress(i, i) -= external_stress[i] / unit_transform;
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_all_runners(UnitCell& ucell)
{
    ESolver_KS::after_all_runners(ucell);

    ModuleIO::ctrl_runner_pw<T, Device>(ucell, this->pelec, this->pw_wfc, 
            this->pw_rho, this->pw_rhod, this->chr, this->kv, this->stp, 
            this->sf, this->ppcell, this->solvent, this->Pgrid, PARAM.inp); 

    elecstate::teardown_estate_pw(this->pelec, this->vsep_cell);
    
}

template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver
