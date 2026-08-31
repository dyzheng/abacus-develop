#include "esolver_ks_lcao.h"
#include "source_base/module_external/blacs_connector.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "source_estate/elecstate_tools.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_lcao/module_deltaspin/deltaspin_lcao.h"
#include "source_lcao/dftu_lcao.h"
#include "source_lcao/hs_matrix_k.hpp" // there may be multiple definitions if using hpp
#include "source_estate/module_charge/symmetry_rho.h"
#include "source_lcao/LCAO_domain.h" // need DeePKS_init
#include "source_lcao/FORCE_STRESS.h"
#include "source_estate/elecstate_lcao.h"
#include "source_lcao/hamilt_lcao.h"
#include "source_lcao/module_operator_lcao/deltap_lcao.h"
#include "source_lcao/module_deltap/deltap.h"
#include "deltap_common.h"
#include "source_io/module_unk/unk_overlap_lcao.h"
#include "source_io/module_hs/cal_r_overlap_R.h"
#include "source_hsolver/hsolver_lcao.h"
#include <iomanip>
#include <cmath>
#ifdef __EXX
#include "../source_lcao/module_ri/exx_opt_orb.h"
#endif
#include "source_lcao/module_rdmft/rdmft.h"
#include "source_estate/module_charge/chgmixing.h" // use charge mixing, mohan add 20251006
#include "source_estate/module_dm/init_dm.h" // init dm from electronic wave functions
#include "source_io/module_ctrl/ctrl_runner_lcao.h" // use ctrl_runner_lcao() 
#include "source_io/module_ctrl/ctrl_iter_lcao.h" // use ctrl_iter_lcao() 
#include "source_io/module_ctrl/ctrl_scf_lcao.h" // use ctrl_scf_lcao()
#include "source_io/module_output/print_info.h"
#include "source_lcao/rho_tau_lcao.h" // mohan add 20251024
#include "source_lcao/LCAO_set.h" // mohan add 20251111
#include "source_psi/setup_psi.h" // use Setup_Psi for deallocate_psi
#include "source_estate/module_constraint/constraint_inject_lcao.h"
#include "source_estate/module_constraint/constraint_loop.h"

namespace ModuleESolver
{

template <typename TK, typename TR>
ESolver_KS_LCAO<TK, TR>::ESolver_KS_LCAO()
{
    this->classname = "ESolver_KS_LCAO";
    this->basisname = "LCAO";
    this->exx_nao.init(); // mohan add 20251008
    this->deltap_scf_solver_ = std::make_unique<deltap_scf::DeltapScfSolver>();
}

template <typename TK, typename TR>
ESolver_KS_LCAO<TK, TR>::~ESolver_KS_LCAO()
{
	//****************************************************
	// do not add any codes in this deconstructor funcion
	//****************************************************
    Setup_Psi<TK>::deallocate_psi(this->psi);
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "before_all_runners");
    ModuleBase::timer::start("ESolver_KS_LCAO", "before_all_runners");

    // 1) before_all_runners in ESolver_KS
    ESolver_KS::before_all_runners(ucell, inp);

    // 1b) DeltaP constraint needs multi-k (complex<double>) wavefunctions.
    // Warn once at init (instead of every SCF iteration) for real instances.
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    {
        if constexpr (!std::is_same<TK, std::complex<double>>::value)
        {
            if (GlobalV::MY_RANK == 0)
                ModuleBase::WARNING("ESolver_KS_LCAO::before_all_runners",
                    "deltap_corr only supports multi-k (complex<double>) calculations");
        }
    }

    // 2) autoset nbands in ElecState before init_basis (for Psi 2d division)
    if (this->pelec == nullptr)
    {
        // TK stands for double and std::complex<double>?
        this->pelec = new elecstate::ElecStateLCAO<TK>(&(this->chr), &(this->kv),
          this->kv.get_nks(), this->pw_big);
    }

    // 3) read LCAO orbitals/projectors and construct the interpolation tables.
    LCAO_domain::init_basis_lcao(this->pv, inp.onsite_radius, inp.lcao_ecut,
      inp.lcao_dk, inp.lcao_dr, inp.lcao_rmax, ucell, two_center_bundle_, orb_);

    // 4) setup EXX calculations
    if (inp.calculation == "gen_opt_abfs")
    {
#ifdef __EXX
        Exx_Opt_Orb exx_opt_orb;
        exx_opt_orb.generate_matrix(GlobalC::exx_info.info_opt_abfs, this->kv, ucell, this->orb_);
#else
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::before_all_runners", "calculation=gen_opt_abfs must compile __EXX");
#endif
        return;
    }

    LCAO_domain::set_psi_occ_dm_chg<TK>(this->kv, this->psi, this->pv, this->pelec,
      this->dmat, this->chr, inp);

    LCAO_domain::set_pot<TK>(ucell, this->kv, this->sf, *this->pw_rho, *this->pw_rhod,
      this->pelec, this->orb_, this->pv, this->locpp, this->dftu,
      this->solvent, this->exx_nao, this->deepks, inp);

    //! if kpar is not divisible by nks, print a warning
    ModuleIO::print_kpar(this->kv.get_nks(), PARAM.globalv.kpar_lcao);

    //! init rdmft, added by jghan
    if (inp.rdmft == true)
    {
        rdmft_solver.init(this->pv, ucell,
          this->gd, this->kv, *(this->pelec), this->orb_,
          two_center_bundle_, inp.dft_functional, inp.rdmft_power_alpha);
    }

    ModuleBase::timer::end("ESolver_KS_LCAO", "before_all_runners");
    return;
}


template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "before_scf");
    ModuleBase::timer::start("ESolver_KS_LCAO", "before_scf");

    //! 1) call before_scf() of ESolver_KS.
    ESolver_KS::before_scf(ucell, istep);

    //! 2) find search radius
    double search_radius = atom_arrange::set_sr_NL(GlobalV::ofs_running,
      PARAM.inp.out_level, orb_.get_rcutmax_Phi(), ucell.infoNL.get_rcutmax_Beta(),
      PARAM.globalv.gamma_only_local);

    //! 3) use search_radius to search adj atoms
    atom_arrange::search(PARAM.globalv.search_pbc, GlobalV::ofs_running,
      this->gd, ucell, search_radius, PARAM.inp.test_atom_input);

    //! 4) initialize NAO basis set
    // here new is a unique pointer, which will be deleted automatically
    gint_info_.reset(
        new ModuleGint::GintInfo(
        this->pw_big->nbx, this->pw_big->nby, this->pw_big->nbz,
        this->pw_rho->nx, this->pw_rho->ny, this->pw_rho->nz,
        0, 0, this->pw_big->nbzp_start,
        this->pw_big->nbx, this->pw_big->nby, this->pw_big->nbzp,
        orb_.Phi, ucell, this->gd));
    ModuleGint::Gint::set_gint_info(gint_info_.get());

    // 7) For each atom, calculate the adjacent atoms in different cells
    // and allocate the space for H(R) and S(R).
    // If k point is used here, allocate HlocR after atom_arrange.
    this->RA.for_2d(ucell, this->gd, this->pv, PARAM.globalv.gamma_only_local, orb_.cutoffs());

    // 8) initialize the Hamiltonian operators
    // if atom moves, then delete old pointer and add a new one
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }
    if (this->p_hamilt == nullptr)
    {
        this->p_hamilt = new hamilt::HamiltLCAO<TK, TR>(
            ucell, this->gd, &this->pv, this->pelec->pot, this->kv,
            two_center_bundle_, orb_, this->dmat.dm, &this->dftu, this->deepks, istep, exx_nao);
    }

    // Branch A: DeltaP multi-ionic-step state (B-3/B-5).
    // The p_hamilt rebuild above creates a fresh DeltaPOperator whose
    // lambda_ was reset to deltap_lambda_init.  The SCF state machine owns
    // the converged λ across steps (apply_lambda persists it into
    // state_.lambda_eff); seed the new operator so contributeHR() re-adds
    // the full current λ to the rebuilt hR and get_lambda() returns the
    // previous step's value instead of zero.
    if (deltap_scf_initialized_ && PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    {
        auto* new_hamilt = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
        auto* new_dp_op = new_hamilt ? new_hamilt->get_dp_operator() : nullptr;
        const auto& lam = deltap_scf_solver_->state().lambda_eff;
        if (new_dp_op != nullptr && !lam.empty())
        {
            new_dp_op->set_lambda(lam);
        }
    }

    // 9) for each ionic step, the overlap <phi|alpha> must be rebuilt
    // since it depends on ionic positions
    this->deepks.build_overlap(ucell, orb_, pv, gd, *(two_center_bundle_.overlap_orb_alpha), PARAM.inp);

    // 10) prepare sc calculation
    init_deltaspin_lcao<TK>(ucell, PARAM.inp, &(this->pv), this->kv, this->p_hamilt, this->psi, this->dmat.dm, this->pelec);

    // 11) set xc type before the first cal of xc in pelec->init_scf, Peize Lin add 2016-12-03
    this->exx_nao.before_scf(ucell, this->kv, orb_, this->p_chgmix, istep, PARAM.inp);

    // 12) initalize DM(R), which has the same size with Hamiltonian(R)
    auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);

    if(!hamilt_lcao)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::before_scf","p_hamilt does not exist");
    }
    this->dmat.dm->init_DMR(*hamilt_lcao->getHR());

    // 13.1) decide the strategy for initializing DMR and HR
    if(istep == 0)//if the first scf step, readin DMR from file,
    {
        //calculate or readin the density matrix DMR
        if(PARAM.inp.init_chg == "dm")
        {
            //! 13.1.1) init charge density from density matrix file
            LCAO_domain::init_chg_dm<TK>(PARAM.globalv.global_readin_dir, PARAM.inp.nspin,
                this->dmat, ucell, &(this->pv), this->pelec->charge);
        }
        if(PARAM.inp.init_chg == "hr")
        {
            //! 13.1.2) init charge density from Hamiltonian matrix file
            LCAO_domain::init_chg_hr<TK, TR>(PARAM.globalv.global_readin_dir, PARAM.inp.nspin,
                static_cast<hamilt::Hamilt<TK>*>(this->p_hamilt), ucell, &(this->pv), this->psi[0], this->pelec, *this->dmat.dm,
                this->chr, PARAM.inp.ks_solver);
        }
    }
    else if(PARAM.inp.esolver_type!="tddft")//if not, use the DMR calculated from last step
    {
        // 13.1.2) two cases are considered:
        // 1. DMK in DensityMatrix is not empty (istep > 0), then DMR is initialized by DMK
        // 2. DMK in DensityMatrix is empty (istep == 0), then DMR is initialized by zeros
        this->dmat.dm->cal_DMR();
    }
    // 13.2) init_scf, should be before_scf? mohan add 2025-03-10
    this->pelec->init_scf(ucell, this->Pgrid, this->sf.strucFac, this->locpp.numeric, ucell.symm);

#ifdef __MLALGO
    // 14) initialize DM2(R) of DeePKS, the DM2(R) is different from DM(R)
    this->deepks.ld.init_DMR(ucell, orb_, this->pv, this->gd);
#endif

    // 16) the electron charge density should be symmetrized,
    Symmetry_rho::symmetrize_rho(PARAM.inp.nspin, this->chr, this->pw_rho, ucell.symm);

    // 17) update of RDMFT, added by jghan
    if (PARAM.inp.rdmft == true)
    {
        rdmft_solver.update_ion(ucell, *(this->pw_rho), this->locpp.vloc, this->sf.strucFac);
    }

    // Real-space weight constraint (phase 2, LCAO + Becke): configure from
    // INPUT via the shared module function (PW and LCAO channels must
    // observe identical guards, defaults and partition radii), then build
    // the shared weight field (M1) on the dense grid and arm the outer
    // loop.  The dense grid is the same pw_rhod the Veff operator
    // integrates into H(R) and the same grid chr.rho lives on, so the
    // observable and the injection operator share one WeightGrid instance.
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintConfig cfg;
        std::vector<double> radii;
        std::string error;
        const constraint::ConfigStatus st = constraint::configure_from_inputs(
            cfg, ucell, radii, error);
        if (st == constraint::ConfigStatus::ERROR)
        {
            ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::before_scf", error);
        }
        constraint::ConstraintLoop::instance().init(
            ucell, this->pw_rhod, cfg, radii, PARAM.inp.nelec);
        constraint_audit_done_ = false;
    }

    ModuleBase::timer::end("ESolver_KS_LCAO", "before_scf");
    return;
}


template <typename TK, typename TR>
double ESolver_KS_LCAO<TK, TR>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "cal_force");
    ModuleBase::timer::start("ESolver_KS_LCAO", "cal_force");

    Force_Stress_LCAO<TK> fsl(this->RA, ucell.nat);

    // Store DeltaP lambda for force/stress computation
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    {
        auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
        if (hamilt_lcao != nullptr)
        {
            auto* dp_op = hamilt_lcao->get_dp_operator();
            if (dp_op != nullptr)
                hamilt::DeltaPOperator<TK, TR>::store_lambda_for_force(dp_op->get_lambda());
        }
    }

    // DeltaP H_HK (Berry-connection) analytic force/stress — T7-c B-7 and
    // F-8 (2026-08-17).  The H_HR projector force/stress is added inside
    // FORCE_STRESS; the k-space H_HK term has no real-space dR/dε expression
    // there, so it is computed here from the converged wavefunctions with C
    // frozen (the constrained SCF energy is variational in C, so the
    // Hellmann-Feynman theorem applies).  The contributions are stored
    // statically and added inside FORCE_STRESS so the printed TOTAL-FORCE /
    // total stress include them.  Serial-only for the stress (see
    // deltap::DeltaP::compute_hk_force); must run before getForceStress.
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr
        && (PARAM.inp.cal_force || PARAM.inp.cal_stress))
    {
        if constexpr (std::is_same<TK, std::complex<double>>::value)
        {
            // Clear any stale H_HK force/stress from a previous ionic step
            // before recomputing, so a failed/disabled computation cannot
            // leak it.
            hamilt::DeltaPOperator<TK, TR>::store_hk_force_for_force({}, 0.0);
            hamilt::DeltaPOperator<TK, TR>::store_hk_stress_for_stress({});
            auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
            if (hamilt_lcao != nullptr)
            {
                auto* dp_op = hamilt_lcao->get_dp_operator();
                if (dp_op != nullptr && !dp_op->get_lambda().empty() && this->dp_scf_)
                {
                    std::vector<double> f_hk;
                    std::vector<double> s_hk;
                    double e_hk = 0.0;
                    if (this->dp_scf_->compute_hk_force(ucell, this->psi, this->pelec,
                                                        dp_op->get_lambda(), f_hk, e_hk,
                                                        PARAM.inp.cal_stress ? &s_hk : nullptr))
                    {
                        hamilt::DeltaPOperator<TK, TR>::store_hk_force_for_force(f_hk, e_hk);
                        if (PARAM.inp.cal_stress)
                        {
                            hamilt::DeltaPOperator<TK, TR>::store_hk_stress_for_stress(s_hk);
                        }
                        double fmax = 0.0;
                        for (size_t i = 0; i < f_hk.size(); ++i)
                        {
                            fmax = std::max(fmax, std::abs(f_hk[i]));
                        }
                        std::cout << " [DeltaP HK-force] E_HK=" << std::setprecision(10)
                                  << e_hk << " Ry  F_HK_max=" << std::setprecision(6)
                                  << fmax << " Ry/Bohr";
                        for (size_t i = 0; i < f_hk.size(); ++i)
                        {
                            std::cout << " " << std::setprecision(5) << f_hk[i];
                        }
                        if (PARAM.inp.cal_stress && s_hk.size() == 6)
                        {
                            std::cout << "  sigma_HK=[";
                            for (size_t i = 0; i < 6; ++i)
                            {
                                std::cout << std::setprecision(10) << s_hk[i] << (i < 5 ? "," : "");
                            }
                            std::cout << "] Ry/Bohr^3";
                        }
                        std::cout << std::endl;
                    }
                }
            }
        }
    }

    deepks.dpks_out_type = "tot";  // for deepks method

    fsl.getForceStress(ucell, PARAM.inp.cal_force, PARAM.inp.cal_stress, 
                       PARAM.inp.test_force, PARAM.inp.test_stress,
                       this->gd, this->pv, this->pelec, this->dmat, this->psi,
                       two_center_bundle_, orb_, force, this->scs,
                       this->locpp, this->sf, this->kv,
                       this->pw_rho, this->solvent, this->dftu, this->deepks,
                       this->exx_nao, &ucell.symm);

    // delete RA after cal_force
    this->RA.delete_grid();

    this->have_force = true;

    ModuleBase::timer::end("ESolver_KS_LCAO", "cal_force");
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "cal_stress");
    ModuleBase::timer::start("ESolver_KS_LCAO", "cal_stress");

    if (!this->have_force)
    {
        ModuleBase::matrix fcs;
        this->cal_force(ucell, fcs);
    }

    // the stress has been calculated in 'cal_force'
    stress = this->scs;
    this->have_force = false;

    ModuleBase::timer::end("ESolver_KS_LCAO", "cal_stress");
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::after_all_runners(UnitCell& ucell)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "after_all_runners");
    ModuleBase::timer::start("ESolver_KS_LCAO", "after_all_runners");

    ESolver_KS::after_all_runners(ucell);

    auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
    if(!hamilt_lcao)
    {
	    ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::after_all_runners","p_hamilt does not exist");
    }

    ModuleIO::ctrl_runner_lcao<TK, TR>(ucell,
		    PARAM.inp, this->kv, this->pelec, this->dmat, this->pv, this->Pgrid, 
		    this->gd, this->psi, this->chr, hamilt_lcao,
		    this->two_center_bundle_,
		    this->orb_, this->pw_rho, this->pw_rhod,
		    this->sf, this->locpp.vloc, this->exx_nao, this->solvent);


#ifdef __MPI
#ifdef __LCAO
    // Exit BLACS environment for LCAO calculations
    Cblacs_exit(1);
#endif
#endif

    ModuleBase::timer::end("ESolver_KS_LCAO", "after_all_runners");
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::iter_init(UnitCell& ucell, const int istep, const int iter)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "iter_init");

    // call iter_init() of ESolver_KS
    ESolver_KS::iter_init(ucell, istep, iter);

    module_charge::chgmixing_ks_lcao(iter, this->p_chgmix, this->dftu, 
      this->dmat.dm->get_DMR_pointer(1)->get_nnr(), PARAM.inp); 

    if (iter == 1)
    {
        this->gint_precision_controller_.set_mode(PARAM.inp.gint_precision);
        this->gint_precision_controller_.reset_for_new_scf();
        this->gint_info_->set_exec_precision(this->gint_precision_controller_.current_precision());
        if (PARAM.inp.gint_precision == "mix")
        {
            GlobalV::ofs_running << "\n >> Gint mixed-precision mode: starting SCF with fp32"
                                 << " (will switch to fp64 when drho is small enough)" << std::endl;
            std::cout << " >> NOTICE: Gint grid-integration starts with fp32 (mixed-precision mode)" << std::endl;
        }
        else if (PARAM.inp.gint_precision == "single")
        {
            std::cout << " >> NOTICE: Gint grid-integration runs in fp32 (single-precision mode)" << std::endl;
        }

        if (PARAM.inp.sc_mag_switch)
        {
            spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
            sc.set_subspace_exec_precision(this->gint_precision_controller_.current_precision());
        }
        else if (PARAM.inp.gint_precision == "single")
        {
            GlobalV::ofs_running << "\n >> Gint single-precision mode: using fp32 throughout SCF" << std::endl;
            std::cout << " >> NOTICE: Gint grid-integration uses fp32 throughout SCF (single-precision mode)" << std::endl;
        }
    }

    // mohan update 2012-06-05
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband(ucell);

    if (istep == 0 && PARAM.inp.init_wfc == "file")
	{
		int exx_two_level_step = 0;
#ifdef __EXX
		if (GlobalC::exx_info.info_global.cal_exx)
		{
			// the following steps are only needed in the first outer exx loop
			exx_two_level_step
				= GlobalC::exx_info.info_ri.real_number ? 
                  this->exx_nao.exd->two_level_step : this->exx_nao.exc->two_level_step;
		}
#endif
		elecstate::init_dm<TK>(ucell, this->pelec, this->dmat, this->psi, this->chr, iter, exx_two_level_step);
	}

#ifdef __EXX
    // calculate exact-exchange
    if (PARAM.inp.calculation != "nscf")
    {
        if (GlobalC::exx_info.info_ri.real_number)
        {
            this->exx_nao.exd->exx_eachiterinit(istep, ucell, *this->dmat.dm, this->kv, iter);
        }
        else
        {
            this->exx_nao.exc->exx_eachiterinit(istep, ucell, *this->dmat.dm, this->kv, iter);
        }
    }
#endif

    init_dftu_lcao<TK>(istep, iter, PARAM.inp, &(this->dftu), this->dmat.dm, ucell, this->chr.rho, this->pw_rho->nrxx);

#ifdef __MLALGO
    // the density matrixes of DeePKS have been updated in each iter
    this->deepks.ld.set_hr_cal(true);

    // HR in HamiltLCAO should be recalculate
    if (PARAM.inp.deepks_scf)
    {
        this->p_hamilt->refresh();
    }
#endif

    if (PARAM.inp.vl_in_h)
    {
        // update real space Hamiltonian
        this->p_hamilt->refresh();
    }

    // save density matrix DMR for mixing
    if (PARAM.inp.mixing_restart > 0 && PARAM.inp.mixing_dmr && this->p_chgmix->mixing_restart_count > 0)
    {
        this->dmat.dm->save_DMR();
    }
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::hamilt2rho_single(UnitCell& ucell, int istep, int iter, double ethr)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "hamilt2rho_single");

    // 1) reset energy
    this->pelec->f_en.eband = 0.0;
    this->pelec->f_en.demet = 0.0;
    bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;

    // =====================================================================
    // 2) DeltaSpin: inner lambda loop to constrain atomic magnetic moments
    // =====================================================================
    // The DeltaSpin method implements constrained LSDA via Lagrange multipliers:
    //   E'[rho] = E[rho] - sum_i lambda_i . (M_i - M_target_i)
    //
    // The constrained energy functional adds a penalty term that drives each
    // atom's magnetic moment M_i toward its target value M_target_i.
    // The Lagrange multiplier lambda_i acts as a "magnetic force" (eV/uB).
    //
    // Code paths by spin type:
    // ---------------------------------------------------------------
    // nspin=2 (collinear):
    //   - Only z-component of magnetization is constrained (M_z per atom)
    //   - H_delta = lambda_z * sigma_z (diagonal, opposite sign per spin channel)
    //   - DMR: uses switch_dmr(2) -> spin-difference density (rho_up - rho_dn)
    //   - cal_coeff_lambda: coefficients[spin] = +/- lambda_z
    //
    // nspin=4 (non-collinear):
    //   - Full 3D magnetization vector constrained (Mx, My, Mz per atom)
    //   - H_delta = lambda . sigma (full 2x2 Pauli matrix with spin-flip terms)
    //   - DMR: spinor density matrix (2x2 blocks interleaved)
    //   - cal_coeff_lambda: 4 coeffs for 2x2 spinor block
    //
    // direction_only mode:
    //   - Designed for non-collinear: removes parallel lambda component so
    //     only transverse (directional) constraint remains
    //   - CRITICAL: for nspin=2, direction_only projects lambda to ZERO because
    //     the only constrained direction (z) IS the parallel direction.
    //     Therefore direction_only MUST be disabled during Phase 1 BFGS.
    //
    // sc_scf_thr_mode parameter:
    //   - "threshold" (default): lambda loop activates when drho < sc_scf_thr
    //   - "immediate": lambda loop activates from iter>=2 (for PW basis)
    //   - "off": lambda loop never activates (lambda used as constant constraint)
    //   - For "threshold" mode, sc_scf_thr should be 10-100x larger than scf_thr
    //   - mixing_restart is auto-set based on sc_scf_thr_mode
    // =====================================================================
    bool skip_solve = false;
    if (PARAM.inp.sc_mag_switch)
    {
        spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();

        if (PARAM.inp.sc_lambda_strategy == "linear_scan")
        {
            sc.set_drho(this->drho);
            sc.run_lambda_linear_scan(iter - 1);

            skip_solve = true;
        }
        else if (PARAM.inp.sc_scf_thr_mode == "off")
        {
            // "off" mode: never activate the lambda loop.
            // Lambda values are loaded from STRU and used as constant constraints.
            // Replaces the old convention of setting sc_scf_thr=1e-10.
        }
        else if (PARAM.inp.sc_direction_only && PARAM.inp.nspin == 2)
        {
            // ================================================================
            // Collinear direction_only: two-phase strategy
            // ================================================================
            // For nspin=2, direction_only projection zeroes lambda entirely
            // (see lambda_loop.cpp). The two-phase strategy works around this:
            //
            // Phase 1 (iter 1..sc_dir_phase1_steps): BFGS with direction_only
            //   temporarily disabled, constraining moment MAGNITUDE to target.
            //   skip_solve=true: BFGS inner loop handles diagonalization.
            //
            // Phase 2 (iter > sc_dir_phase1_steps): Lambda decays gradually,
            //   normal SCF runs, system relaxes to magnetic ground state.
            // ================================================================
            if (iter <= PARAM.inp.sc_dir_phase1_steps)
            {
                sc.set_drho(this->drho);
                sc.set_direction_only(false);
                sc.run_lambda_loop(iter - 1);
                sc.set_direction_only(true);
                skip_solve = true;
            }
            else
            {
                if (iter == PARAM.inp.sc_dir_phase1_steps + 1)
                {
                    // Reset mixing at Phase 1->2 transition.
                    // Phase 1 BFGS updates DM directly without charge mixing,
                    // so Broyden history is incompatible with Phase 2 SCF.
                    // Also reset mixing_restart_count and mixing_restart_step
                    // to avoid polluting mixing_dmr logic.
                    this->p_chgmix->mix_reset();
                    this->p_chgmix->mixing_restart_count = 0;
                    this->p_chgmix->mixing_restart_step = PARAM.inp.scf_nmax + 1;
                }

                // Gradual lambda decay: factor = 0.5^(1/3) per step
                // (~halves every 3 steps). Gradual decay avoids discontinuous
                // Hamiltonian change that would cause charge density oscillations.
                int nat = sc.get_nat();
                auto lambda = sc.get_sc_lambda();
                const double DECAY = std::pow(0.5, 1.0 / 3.0);
                for (int ia = 0; ia < nat; ++ia)
                    for (int ic = 0; ic < 3; ++ic)
                        lambda[ia][ic] *= DECAY;
                sc.set_lambda(lambda);
            }
        }
        else if (PARAM.inp.sc_direction_only && PARAM.inp.nspin == 4)
        {
            // Non-collinear direction_only: direction_only projection works
            // correctly for nspin=4 (only removes parallel component, leaving
            // transverse constraint). Use standard sc_scf_thr_mode gate.
            if (PARAM.inp.sc_scf_thr_mode == "immediate")
            {
                if (iter > 1)
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    if (!sc.mag_converged()) { sc.set_mag_converged(true); }
                    skip_solve = true;
                }
            }
            else // "threshold"
            {
                if (!sc.mag_converged() && this->drho > 0 && this->drho < PARAM.inp.sc_scf_thr)
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    sc.set_mag_converged(true);
                    skip_solve = true;
                }
                else if (sc.mag_converged())
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    skip_solve = true;
                }
            }
        }
        else
        {
            // Standard DeltaSpin (no direction_only)
            if (PARAM.inp.sc_scf_thr_mode == "immediate")
            {
                // "immediate" mode: activate lambda loop from iter>=2.
                // iter=1 is skipped because initial wavefunctions are not
                // available to compute initial magnetic moments.
                if (iter > 1)
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    if (!sc.mag_converged()) { sc.set_mag_converged(true); }
                    skip_solve = true;
                }
            }
            else // "threshold"
            {
                // "threshold" mode: activate when drho < sc_scf_thr.
                // drho > 0 excludes iter=1 where drho has not been computed yet.
                if (!sc.mag_converged() && this->drho > 0 && this->drho < PARAM.inp.sc_scf_thr)
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    sc.set_mag_converged(true);
                    skip_solve = true;
                }
                else if (sc.mag_converged())
                {
                    sc.set_drho(this->drho);
                    sc.run_lambda_loop(iter - 1);
                    skip_solve = true;
                }
            }
        }

        // Run trace vs DMR diagnostic once near SCF convergence
        if (PARAM.inp.nspin == 2 && PARAM.inp.basis_type == "lcao"
            && this->drho > 0 && this->drho < 1e-3
            && PARAM.inp.sc_acceleration_mode != "off"
            && !sc.local_diag_run_)
        {
            double lambda_ref_ry = 0.0;
            for (int ia = 0; ia < sc.get_nat(); ia++) {
                if (sc.get_constrain()[ia].z != 0) {
                    lambda_ref_ry = sc.get_sc_lambda()[ia].z;
                    break;
                }
            }
            sc.run_trace_vs_dmr_diagnostic(iter - 1, lambda_ref_ry);
            sc.local_diag_run_ = true;
        }
    }

    // =====================================================================
    // 2b) DeltaP: inner lambda loop to constrain atomic polarization
    // =====================================================================
    // The constrained energy functional adds a penalty term with
    // Lagrange multiplier lambda for each atom:
    //   E'[rho] = E[rho] + sum_I lambda_I * (gamma_I - gamma_target_I)
    //
    // The inner loop runs BFGS-CG optimization on lambda within a single
    // SCF iteration, WITHOUT updating the charge density.  This follows
    // the same pattern as DeltaSpin (above).
    // =====================================================================
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr && deltap_scf_initialized_)
    {
        skip_solve = deltap_scf_solver_->inner_loop(this->drho);
    }

    // Real-space weight constraint (phase 2, LCAO): inject the current
    // mu-weighted constraint potential into the dense-grid effective
    // potential before the Hamiltonian is built.  The Veff operator
    // integrates v_eff into H(R) via cal_gint_vl inside the HSolver loop,
    // so H receives exactly the PW-identical operator Σ_α μ_α w_α(r) on
    // the same grid the observable is read from (observable == injection
    // operator by construction).  In the reference phase mu is all zero, so
    // the first SCF stays unconstrained.  Skipped on inner-loop iterations
    // (DeltaSpin/DeltaP) because HSolver does not run there and no
    // Hamiltonian is built.
    if (PARAM.inp.constraint && !skip_solve)
    {
        constraint::ConstraintLoop::instance().inject_potential_lcao(
            iter, this->pelec->pot->get_eff_v());
    }

    // 3) run Hsolver
    if (!skip_solve)
    {
        hsolver::HSolverLCAO<TK> hsolver_lcao_obj(&(this->pv), PARAM.inp.ks_solver);
        hsolver_lcao_obj.solve(static_cast<hamilt::Hamilt<TK>*>(this->p_hamilt), this->psi[0], this->pelec, *this->dmat.dm, 
          this->chr, PARAM.inp.nspin, skip_charge);
    }
    else
    {
        // Lambda loop updated the density matrix (DM) but not the real-space charge density.
        // HSolver was skipped, so we need to sync rho from DM manually.
        LCAO_domain::dm2rho(this->dmat.dm->get_DMR_vector(), PARAM.inp.nspin, &this->chr);
    }

    // 4) EXX
#ifdef __EXX
    if (PARAM.inp.calculation != "nscf")
    {
        if (GlobalC::exx_info.info_ri.real_number)
        {
            this->exx_nao.exd->exx_hamilt2rho(*this->pelec, this->pv, iter);
        }
        else
        {
            this->exx_nao.exc->exx_hamilt2rho(*this->pelec, this->pv, iter);
        }
    }
#endif

    // 5) symmetrize the charge density
    Symmetry_rho::symmetrize_rho(PARAM.inp.nspin, this->chr, this->pw_rho, ucell.symm);

    // 6) calculate delta energy
    this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);
}


template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "iter_finish");

    auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);

    if(!hamilt_lcao)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::iter_finish","p_hamilt does not exist");
    }

	const std::vector<std::vector<TK>>& dm_vec = this->dmat.dm->get_DMK_vector();

    // 1) calculate the local occupation number matrix and energy correction in DFT+U
    finish_dftu_lcao<TK>(iter, conv_esolver, PARAM.inp, &(this->dftu), ucell, dm_vec, this->kv, this->p_chgmix->get_mixing_beta(), hamilt_lcao);

    // 2) for deepks, calculate delta_e, output labels during electronic steps
    this->deepks.delta_e(ucell, this->kv, this->orb_, this->pv, this->gd, dm_vec, this->pelec->f_en, PARAM.inp);

    // 3) for delta spin
    cal_mi_lcao_wrapper<TK>(iter, PARAM.inp);

    // 3c) DeltaP SCF constraint: compute gamma^I, update lambda
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    {
        if constexpr (std::is_same<TK, std::complex<double>>::value)
        {
            if (!deltap_scf_initialized_)
            {
                deltap_init(ucell);
            }
            if (iter == 1)
            {
                deltap_scf_solver_->reset_ionic_step();
            }
            // The SCF convergence flag is only set later by
            // ESolver_KS::iter_finish (after charge mixing), so evaluate the
            // same drho < scf_thr criterion here for the DeltaP single-point
            // outer-loop secant hook (secant_at_convergence).
            const bool deltap_scf_converged = (this->drho < PARAM.inp.scf_thr);
            deltap_scf_solver_->iter_finish(iter, this->drho, deltap_scf_converged);
            // Phase 0.3-lite: at SCF convergence, freeze the continuity anchor
            // to the converged gamma reading so the next outer-loop SCF run
            // anchors its branch selection to it (not to the per-iteration
            // drifting W_prev_, which locks early iterations onto a wrong
            // branch).  No-op for gamma mode / non-converged iterations.
            if (deltap_scf_converged
                && PARAM.inp.deltap_observable == "operator")
            {
                dp_scf_->freeze_branch_ref();
                // L1.2/L1.3 (2026-08-16): Route A+ applicability gauges —
                // SMO completeness ⟨η⟩ (band/k average and per-band max) and
                // per-atom θ-spread (H_HR proxy error control ∝ λ_I·spread_I,
                // RouteA++ §1.4 推论 1).  Printed once per converged SCF.
                if (GlobalV::MY_RANK == 0)
                {
                    const double eta_avg = dp_scf_->get_last_eta_avg();
                    if (!std::isnan(eta_avg))
                    {
                        const auto& spr = dp_scf_->get_last_spread_I();
                        std::cout << " [DeltaP L1] <eta>=" << std::fixed
                                  << std::setprecision(6) << eta_avg
                                  << " (max " << dp_scf_->get_last_eta_max()
                                  << ") spread_I/atom=(";
                        for (size_t i = 0; i < spr.size(); ++i)
                        {
                            if (i > 0) std::cout << ", ";
                            std::cout << spr[i];
                        }
                        std::cout << ")" << std::endl;
                    }
                }
            }
            // dp_escon is identical on every rank: γ is rank-0-synced by
            // module_deltap (compute_gamma_scf Bcast) and λ by sync_lambda
            // write-back (T1), so the rank-local assignment is consistent.
            this->pelec->f_en.dp_escon = deltap_scf_solver_->state().dp_escon;
        }
    }

    // 3b) direction_only: report magnetic moment status
    if (PARAM.inp.sc_direction_only && PARAM.inp.sc_mag_switch)
    {
        spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
        const int nat = sc.get_nat();
        const auto& Mi = sc.get_Mi();
        const auto& target = sc.get_target_mag();
        const auto& constrain = sc.get_constrain();
        auto lambda = sc.get_sc_lambda();

        double lambda_abs = 0;
        for (int ia = 0; ia < nat; ++ia)
            for (int ic = 0; ic < 3; ++ic)
                if (constrain[ia][ic] != 0)
                    lambda_abs += std::abs(lambda[ia][ic] * ModuleBase::Ry_to_eV);

        GlobalV::ofs_running << " [DS-dir] iter " << iter << "  |lambda|=" << lambda_abs << " eV/uB" << std::endl;
        for (int ia = 0; ia < nat; ++ia)
            for (int ic = 0; ic < 3; ++ic)
                if (constrain[ia][ic] != 0)
                    GlobalV::ofs_running << "   Atom " << ia << " comp " << ic << ": Mi=" << Mi[ia][ic]
                                         << "  T=" << target[ia][ic] << std::endl;
    }

    // call iter_finish() of ESolver_KS, where band gap is printed,
    // eig and occ are printed, magnetization is calculated,
    // charge mixing is performed, potential is updated, 
    // HF and kS energies are computed, meta-GGA, Jason and restart
    ESolver_KS::iter_finish(ucell, istep, iter, conv_esolver);
    // Real-space weight constraint (phase 2, LCAO): read the constraint
    // charges from the mixed density and run the outer-loop bookkeeping
    // (reference recording / M4 secant step / audit line), mirroring the PW
    // channel.  The hook may override conv_esolver to keep the SCF running
    // until the constraint converges or fuses (two-stage gating, DeltaP
    // lineage).
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
        // M3b runtime audit (Task 2.6): once per geometry, when the outer
        // loop reaches its final state, cross-check the matrix-level
        // observable Tr[W^alpha . DM] (W built by the Gint vlocal kernel)
        // against the grid observable int w_alpha rho dr that the loop
        // reports.  The two paths share the same weight field and grid by
        // construction, so any non-negligible deviation is a wiring bug.
        if (!constraint_audit_done_ && cloop.enabled() && cloop.done())
        {
            constraint_audit_done_ = true;
            std::vector<std::vector<double>> cw;
            const constraint::WeightGrid& wg = cloop.weight_grid();
            cw.reserve(wg.nconstraint());
            for (int a = 0; a < wg.nconstraint(); ++a)
            {
                cw.push_back(wg.constraint_weight(a));
            }
            // Channel -> DM mode: charge reads the total density matrix,
            // spin reads the magnetization density matrix (m = up - dn),
            // mirroring the observable the loop observes.
            const bool spin = (cloop.type() == "spin");
            if (PARAM.inp.nspin == 2)
            {
                this->dmat.dm->switch_dmr(spin ? 2 : 1);
            }
            const hamilt::HContainer<double>* dmr
                = this->dmat.dm->get_DMR_pointer(1);
            const double dev = constraint::ConstraintInjectLCAO::
                audit_weighted_trace(cw, this->gint_info_.get(), dmr,
                                     cloop.charges(), &this->pv);
            if (PARAM.inp.nspin == 2)
            {
                this->dmat.dm->switch_dmr(0);
            }
            if (dev < 0.0)
            {
                GlobalV::ofs_running
                    << "[constraint] M3b runtime audit: SKIPPED "
                       "(W/DM layout or count mismatch)" << std::endl;
            }
            else
            {
                GlobalV::ofs_running
                    << "[constraint] M3b runtime audit: max |Tr[W.DM] - "
                       "int w rho| = " << std::setprecision(12) << dev
                    << " e" << std::endl;
            }
        }
    }
    // Route A+ fixed-geometry outer loop (scf + deltap_outer_nmax > 0): the
    // secant updated t_Γ at the previous convergence but |γ−t_γ| is still
    // above tolerance — continue the SCF loop with the new proxy target
    // instead of terminating the run.  The H_c change re-disturbs drho, so
    // the next DeltaP secant fires at the next convergence crossing.
    if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr
        && deltap_scf_solver_->consume_outer_redrive())
    {
        conv_esolver = false;
    }
    const bool precision_switched = this->gint_precision_controller_.update_after_iteration(this->drho, this->scf_thr);
    this->gint_info_->set_exec_precision(this->gint_precision_controller_.current_precision());
    if (precision_switched)
    {
        GlobalV::ofs_running << "\n >> Gint precision switched: fp32 -> fp64 (drho = "
                             << this->drho << ")" << std::endl;
        std::cout << " >> NOTICE: Gint grid-integration precision switched from fp32 to fp64" << std::endl;
    }

    if (PARAM.inp.sc_mag_switch)
    {
        spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
        sc.set_subspace_exec_precision(this->gint_precision_controller_.current_precision());
    }

    // mix density matrix if mixing_restart + mixing_dmr + not first
    // mixing_restart at every iter except the last iter
    if(iter != PARAM.inp.scf_nmax && !conv_esolver)
    {
        if (PARAM.inp.mixing_restart > 0 && this->p_chgmix->mixing_restart_count > 0 && PARAM.inp.mixing_dmr)
        {
            this->p_chgmix->mix_dmr(this->dmat.dm);
        }
    }

    // control the output related to the finished iteration
    ModuleIO::ctrl_iter_lcao<TK, TR>(ucell, PARAM.inp, this->kv, this->pelec, *this->dmat.dm,
      this->pv, this->gd, this->psi, this->chr, this->p_chgmix, 
      hamilt_lcao, this->orb_, this->deepks, 
      this->exx_nao, iter, istep, conv_esolver, this->scf_ene_thr);
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::after_scf(UnitCell& ucell, const int istep, const bool conv_esolver)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "after_scf");
    ModuleBase::timer::start("ESolver_KS_LCAO", "after_scf");

    auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);

    if(!hamilt_lcao)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::after_scf","p_hamilt does not exist");
    }

    if (PARAM.inp.out_elf[0] > 0)
	{
		LCAO_domain::dm2tau(this->dmat.dm->get_DMR_vector(), PARAM.inp.nspin, this->pelec->charge);
	}

    //! 1) call after_scf() of ESolver_KS
    ESolver_KS::after_scf(ucell, istep, conv_esolver);

    //! 2) output of lcao every few ionic steps
    ModuleIO::ctrl_scf_lcao<TK, TR>(ucell,
            PARAM.inp, this->kv, this->pelec, this->dmat.dm, this->pv,
            this->gd, this->psi, hamilt_lcao, this->dftu, this->two_center_bundle_,
            this->orb_, this->pw_wfc, this->pw_rho, this->pw_big, this->sf,
            this->rdmft_solver, this->deepks, this->exx_nao,
            this->conv_esolver, this->scf_nmax_flag, istep);

    // Real-space weight constraint (phase 2, LCAO): final audit report.
    if (PARAM.inp.constraint)
    {
        constraint::ConstraintLoop::instance().final_report();
    }

    //! 3) Clean up RA, which is used to serach for adjacent atoms
    if (!PARAM.inp.cal_force && !PARAM.inp.cal_stress)
    {
        this->RA.delete_grid();
    }

    ModuleBase::timer::end("ESolver_KS_LCAO", "after_scf");
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::deltap_init(UnitCell& ucell)
{
    // Branch A: build Wilson-loop infrastructure (first call only).
    // Build orb_onsite if not already built.
    if (!two_center_bundle_.overlap_orb_onsite)
    {
        two_center_bundle_.build_orb_onsite(PARAM.inp.deltap_rm);
        two_center_bundle_.tabulate();
    }
    // Build position matrix and berry overlap calculators.
    r_overlap_scf_ = std::make_unique<cal_r_overlap_R>();
    r_overlap_scf_->init(ucell, pv, orb_);
    berry_ovl_scf_ = std::make_unique<unkOverlap_lcao>();
    berry_ovl_scf_->init(ucell, kv.get_nkstot(), orb_);
    berry_ovl_scf_->cal_R_number(ucell, gd);
    berry_ovl_scf_->cal_orb_overlap(ucell);
    // Create the DeltaP numerical object.
    dp_scf_ = std::make_unique<deltap::DeltaP>();
    auto* dp = dp_scf_.get();
    dp->init(ucell, gd, kv,
              two_center_bundle_.overlap_orb_onsite.get(),
              two_center_bundle_.overlap_orb.get(),
              two_center_bundle_.overlap_onsite_onsite.get(),
              orb_.cutoffs(),
              PARAM.inp.deltap_rm, PARAM.inp.deltap_gdir,
              &pv, r_overlap_scf_.get(), berry_ovl_scf_.get());
    dp->load_branch();
    dp->init_inner_loop();

    // Snapshot INPUT into the basis-independent SCF state machine.
    deltap_scf::DeltapParams p;
    p.nat = ucell.nat;
    p.gdir = PARAM.inp.deltap_gdir;
    p.inner_thr = PARAM.inp.deltap_inner_thr;
    p.lambda_step = PARAM.inp.deltap_lambda_step;
    p.lambda_mixing = PARAM.inp.deltap_lambda_mixing;
    p.lambda_init = PARAM.inp.deltap_lambda_init;
    p.conv_thr = PARAM.inp.deltap_conv_thr;
    p.nscf = PARAM.inp.deltap_inner_nmax;
    p.inner_scheme = PARAM.inp.deltap_inner_scheme;
    p.total_mode = (PARAM.inp.deltap_constraint_mode == "total");
    p.target_file = PARAM.inp.deltap_target_file;
    p.constraint_matrix_file = PARAM.inp.deltap_constraint_matrix;
    p.observable_mode = PARAM.inp.deltap_observable;
    // T-6' (EFC L3.1): the constraint operator in operator mode is either
    // the τ_α·P̂ geometric proxy (historical Route A+) or the exact Ô_w =
    // θ_n·P̂ weight-channel operator.  Legacy gamma mode is locked to proxy
    // (its Hamiltonian path predates Route A+; zero regression).
    p.operator_mode = (PARAM.inp.deltap_observable == "operator")
                          ? PARAM.inp.deltap_operator_mode
                          : "proxy";
    // Legacy gamma mode is LOCKED to gamma-drive (its λ residual is built on
    // the target-aware γ report — the load-bearing wall; the Route A+ drive
    // switch is operator-mode-only, zero regression).
    p.drive = (PARAM.inp.deltap_observable == "operator") ? PARAM.inp.deltap_drive
                                                          : "gamma";
    // Route A+ outer-loop secant: single-point runs update t_Γ once at SCF
    // convergence (iter_finish); relax runs update at each new ionic step
    // (reset_ionic_step).
    p.secant_at_convergence = (PARAM.inp.calculation == "scf");
    p.secant_enabled = (PARAM.inp.deltap_secant == "on");
    p.proxy_target_file = PARAM.inp.deltap_proxy_target_file;
    p.outer_nmax = PARAM.inp.deltap_outer_nmax;
    p.outer_thr = PARAM.inp.deltap_outer_thr;
    p.constrain = ucell.get_dp_constrain();

    // STRU-based per-atom targets (DeltaSpin-style keywords).  Used only when
    // no target file is given; the state machine loads the file in init().
    if (p.target_file.empty())
    {
        p.target = ucell.get_dp_target();
        bool has_any_target = false;
        for (int iat = 0; iat < ucell.nat; ++iat)
            if (p.constrain[iat] != 0 && p.target[iat] != 0.0)
                has_any_target = true;
        // Report the STRU-target load once on the root rank only.  (Historical
        // code printed a contradictory "No targets" line on non-root ranks —
        // dangling-else bug; R4 cleanup.)
        if (has_any_target && GlobalV::MY_RANK == 0)
            std::cout << " [DeltaP] Loaded targets from STRU (dp_target/dp_constrain)" << std::endl;
        // No target anywhere (no file, STRU dp_target all unset/zero):
        // leave p.target EMPTY so the run is genuinely free.  Without this,
        // get_dp_target() returns an all-zero {0,0,0} vector, and the λ
        // update residual gate (!tgt.empty()) treats it as a "constrain
        // Γ → 0" target — the free run silently drives λ ≠ 0 and
        // contaminates the relax forces (root cause of the old "F_H1z
        // 限制" misattribution in 4.1; fixed 2026-08-17, Stage 4.3).
        if (!has_any_target)
            p.target.clear();
    }
    // When no target is specified, p.target is now truly empty (see above).
    // This allows ground-state γ determination without target-aware branch
    // selection, while the constraint (deltap_corr=1) still applies the
    // Hamiltonian correction with the current (possibly zero) λ.

    deltap_scf_solver_->init(p, deltap_make_backend(ucell));

    // Sync targets / constraint matrix into the DeltaP numerical object for
    // target-aware branch selection.
    dp->set_target_gamma(deltap_scf_solver_->params().target);
    // Phase 0.3-lite branch anchor: Route A+ operator mode reads γ on a
    // branch-continuous anchor (previous measurement / branch.dat, INPUT
    // deltap_branch_anchor); legacy gamma mode is LOCKED to the target-aware
    // selection (its λ residual is built on γ_report — the load-bearing wall
    // must not change, zero regression).
    const std::string branch_anchor
        = (PARAM.inp.deltap_observable == "operator") ? PARAM.inp.deltap_branch_anchor
                                                      : "target";
    dp->set_branch_anchor(branch_anchor);
    // T-6' (EFC L3.1): the constraint operator in operator mode is either
    // the τ_α·P̂ geometric proxy (historical Route A+) or the exact Ô_w =
    // θ_n·P̂ weight-channel operator.  Legacy gamma mode is locked to proxy
    // (its Hamiltonian path predates Route A+; zero regression).
    dp->set_operator_mode((PARAM.inp.deltap_observable == "operator")
                              ? PARAM.inp.deltap_operator_mode
                              : "proxy");
    // T-9' branch-state write guard: multi-geometry / FD runs keep the
    // calibrated deltap_branch.dat reference (silent overwrite broke A/B
    // comparability twice: 07-30, T4a).
    dp->set_branch_write(PARAM.inp.deltap_branch_write);
    // Phase 0.3-lite frozen continuity anchor: seeded at init by the
    // dp->load_branch() call above (load_branch mirrors into ref_gamma_), so
    // the lambda=0 first measurement anchors to the natural gamma reference
    // of a previous converged run instead of following t_gamma (T4a branch
    // hopping, 2026-08-05).  Legacy gamma mode (target anchor) never reads
    // ref_gamma_ — zero regression.
    if (!deltap_scf_solver_->params().C.empty())
        dp->set_constraint_matrix(deltap_scf_solver_->params().C, deltap_scf_solver_->params().t);

    deltap_scf_initialized_ = true;
}

template <typename TK, typename TR>
typename deltap_scf::DeltapScfSolver::Backend ESolver_KS_LCAO<TK, TR>::deltap_make_backend(UnitCell& ucell)
{
    auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
    if (!hamilt_lcao)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::deltap_init", "p_hamilt does not exist");
    }
    auto* dp_op = hamilt_lcao->get_dp_operator();
    if (!dp_op)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::deltap_init",
            "dp_operator is null but deltap_corr=1");
    }
    auto* dp = dp_scf_.get();
    const int alpha = PARAM.inp.deltap_gdir - 1;

    // Resolve the DeltaP operator from the *current* p_hamilt on every call.
    // p_hamilt (and its operator) is rebuilt at each ionic step
    // (before_scf), so a pointer captured here would dangle after step 1
    // (B-5 use-after-free, observed as bad_alloc in relax step 2).
    auto get_dp_op = [this]() -> hamilt::DeltaPOperator<TK, TR>* {
        auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
        return hamilt_lcao ? hamilt_lcao->get_dp_operator() : nullptr;
    };

    deltap_scf::DeltapScfSolver::Backend b;
    b.set_lambda = [get_dp_op](const std::vector<double>& lam) {
        auto* op = get_dp_op();
        if (op != nullptr) op->set_lambda(lam);
    };
    b.get_lambda = [get_dp_op]() {
        auto* op = get_dp_op();
        return op != nullptr ? op->get_lambda() : std::vector<double>();
    };
    // DeltaP gamma / HK-correction calls are complex<double>-only (multi-k);
    // the branches are discarded for the real instantiation.
    if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        b.compute_gamma = [this, dp, alpha, &ucell]() {
            dp->compute_gamma_scf(ucell, this->psi, this->pelec);
            const auto& g = dp->get_results().gamma_I;
            std::vector<double> out(g.size());
            for (size_t i = 0; i < g.size(); ++i)
                out[i] = g[i][alpha];
            return out;
        };
        b.compute_gamma_raw = [dp, alpha]() {
            const auto& r = dp->get_results().gamma_I_raw;
            if (r.empty())
                return std::vector<double>();
            std::vector<double> out(r.size());
            for (size_t i = 0; i < r.size(); ++i)
                out[i] = r[i][alpha];
            return out;
        };
        // T-7'' (2026-08-13): freeze the Stage-B branch shift at inner-loop
        // entry so the reported γ follows the raw response instead of being
        // pinned to the anchor (which swallowed the raw and made the
        // γ-drive inner-loop residual a constant).  Gamma-drive only.
        b.freeze_branch_shift = [dp]() { dp->freeze_branch_shift(); };
        // Route A+ operator observable: Γ_I = Γ_I^HR + Γ_I^HK measured at the
        // current wavefunctions.  HR comes from compute_gamma_scf (INPUT gdir,
        // τ_α(I)·⟨P̂_I⟩); HK from compute_gamma_op_hk (λ-independent).
        b.compute_gamma_op = [this, dp, &ucell]() {
            dp->compute_gamma_scf(ucell, this->psi, this->pelec);
            dp->compute_gamma_op_hk(ucell, this->psi, this->pelec);
            return dp->compute_operator_observable();
        };
        b.apply_hk_correction = [this, dp, get_dp_op, &ucell](const std::vector<double>& lam) {
            std::unordered_map<int, std::vector<std::complex<double>>> hk_corr;
            dp->compute_hk_correction(ucell, this->psi, this->pelec, lam, hk_corr);
            auto* op = get_dp_op();
            if (op != nullptr) op->set_hk_correction(hk_corr);
        };
    }
    b.solve_frozen = [this]() {
        hsolver::HSolverLCAO<TK> hsolver_lcao_obj(&(this->pv), PARAM.inp.ks_solver);
        hsolver_lcao_obj.solve(static_cast<hamilt::Hamilt<TK>*>(this->p_hamilt),
            this->psi[0], this->pelec, *this->dmat.dm, this->chr,
            PARAM.inp.nspin, true); // skip_charge = true (frozen density)
    };
    b.sync_lambda = [this, get_dp_op](std::vector<double>& lam) {
        if (lam.empty())
            return;
#ifdef __MPI
        if (this->pv.comm() != MPI_COMM_NULL)
        {
            int nproc = 1;
            MPI_Comm_size(this->pv.comm(), &nproc);
            if (nproc > 1)
                MPI_Bcast(lam.data(), static_cast<int>(lam.size()), MPI_DOUBLE, 0, this->pv.comm());
        }
#endif
        // Write the (rank-0) λ back into the operator so the operator λ and the
        // resulting escon are identical on every rank (mirrors the PW
        // backend's s_lambda write-back after Bcast).
        auto* op = get_dp_op();
        if (op != nullptr) op->set_lambda(lam);
    };
    b.on_phase2 = [dp, this]() {
        dp->start_cooldown(1);
        this->p_chgmix->mix_reset();
        // T-17 (V-H8, S1): the P2 λ update is an "edge" for the frozen Ô_w
        // operator kernel — the next Γ measurement re-captures θ and rebuilds
        // H_ow from the settled density (the kernel λ-scales exactly, but the
        // θ/ψ snapshot must refresh after the density has changed; without
        // this the operator would stay frozen on the P1 state forever).
        dp->mark_ow_kernel_stale();
    };
    b.get_optimizer = [dp]() -> ModuleOptimizer::FletcherReevesCG& { return dp->bfgs(); };
    b.lattice_period = [&ucell]() {
        double a_alpha = ucell.lat0;
        if (PARAM.inp.deltap_gdir == 1)      a_alpha *= ucell.a1.norm();
        else if (PARAM.inp.deltap_gdir == 2) a_alpha *= ucell.a2.norm();
        else                                  a_alpha *= ucell.a3.norm();
        return a_alpha;
    };
    return b;
}

template class ESolver_KS_LCAO<double, double>;
template class ESolver_KS_LCAO<std::complex<double>, double>;
template class ESolver_KS_LCAO<std::complex<double>, std::complex<double>>;
} // namespace ModuleESolver
