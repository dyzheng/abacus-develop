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
#include "source_io/module_unk/unk_overlap_lcao.h"
#include "source_io/module_hs/cal_r_overlap_R.h"
#include "source_hsolver/hsolver_lcao.h"
#include <iomanip>
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

namespace ModuleESolver
{

template <typename TK, typename TR>
ESolver_KS_LCAO<TK, TR>::ESolver_KS_LCAO()
{
    this->classname = "ESolver_KS_LCAO";
    this->basisname = "LCAO";
    this->exx_nao.init(); // mohan add 20251008
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
        deltap_inner_loop(ucell, iter, skip_solve);
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
            auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
            if (!hamilt_lcao)
            {
                ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::iter_finish", "p_hamilt does not exist");
            }

            auto* dp_op = hamilt_lcao->get_dp_operator();
            if (!dp_op)
            {
                ModuleBase::WARNING_QUIT("ESolver_KS_LCAO::iter_finish", "dp_operator is null but deltap_corr=1");
            }

            // Initialize Wilson loop infrastructure (first call only)
            if (!deltap_scf_initialized_)
            {
                deltap_init(ucell);
            }

            // Compute gamma (post inner-loop or for diagnostic)
            double max_dev = deltap_compute_gamma(ucell, iter);

            // Fallback: if inner loop is inactive (nscf==0), use gradient descent
            deltap_update_lambda(ucell, iter);

            // Print status with phase indicator
            auto* dp = static_cast<deltap::DeltaP*>(dp_scf_);
            // Diagnostic: raw gamma (pre-branch) for Born effective charge
            {
                const auto& gamma_raw = dp->get_results().gamma_I_raw;
                if (!gamma_raw.empty())
                {
                    const int a = PARAM.inp.deltap_gdir - 1;
                    double Sg_raw = 0.0;
                    for (int iat = 0; iat < ucell.nat; ++iat)
                        Sg_raw += gamma_raw[iat][a];
                    std::cout << "  [rawG] Σγ_raw=" << Sg_raw;
                    for (int iat = 0; iat < ucell.nat; ++iat)
                        std::cout << " γ" << iat << "=" << gamma_raw[iat][a];
                    std::cout << std::endl;
                }
            }
            const int alpha = PARAM.inp.deltap_gdir - 1;
            const auto& gamma_I = dp->get_results().gamma_I;
            std::vector<double> lambda = dp_op->get_lambda();

            bool total_mode = (PARAM.inp.deltap_constraint_mode == "total");
            bool use_constraint_matrix = !deltap_constraint_matrix_.empty();
            std::string phase = deltap_lambda_set_ ? "P3" : "P1";

            std::cout << " [DeltaP " << phase << "] iter=" << std::setw(3) << iter
                      << " γ=(" << std::fixed << std::setprecision(3);

            if (total_mode)
            {
                double total_g = 0.0, total_t = 0.0;
                for (int iat = 0; iat < ucell.nat; ++iat)
                    total_g += gamma_I[iat][alpha];
                std::cout << total_g << ") Σγ=" << total_g;
                // show single λ
                std::cout << " λ=";
                if (std::abs(lambda[0]) < 1e-10)
                    std::cout << std::scientific << std::setprecision(1) << lambda[0];
                else
                    std::cout << std::scientific << std::setprecision(2) << lambda[0];
            }
            else
            {
                for (int iat = 0; iat < ucell.nat; ++iat)
                {
                    if (iat > 0) std::cout << ", ";
                    std::cout << gamma_I[iat][alpha];
                }
                std::cout << ") λ=(";
                for (int iat = 0; iat < ucell.nat; ++iat)
                {
                    if (iat > 0) std::cout << ", ";
                    if (std::abs(lambda[iat]) < 1e-10)
                        std::cout << std::scientific << std::setprecision(1) << lambda[iat];
                    else
                        std::cout << std::scientific << std::setprecision(2) << lambda[iat];
                }
                std::cout << ")";
            }
            std::cout << " |γ-t|=" << std::scientific << std::setprecision(3) << max_dev
                      << "\n";

            // DeltaP constraint energy correction (analogous to DeltaSpin's escon):
            // H_corr contributes ~Σλ·γ to eband. Subtract it to get physical E_DFT.
            double dp_escon = 0.0;
            if (use_constraint_matrix)
            {
                for (int a = 0; a < static_cast<int>(deltap_constraint_matrix_.size()); ++a)
                {
                    double cv = 0.0;
                    for (int i = 0; i < ucell.nat; ++i)
                        cv += deltap_constraint_matrix_[a][i] * gamma_I[i][alpha];
                    dp_escon -= deltap_constraint_lambda_[a] * cv;
                }
            }
            else
            {
                for (int iat = 0; iat < ucell.nat; ++iat)
                    dp_escon -= lambda[iat] * gamma_I[iat][alpha];
            }
            this->pelec->f_en.dp_escon = dp_escon;

            // Effective electric field: E_eff = -λ_avg × π / a_alpha (a.u.)
            // Convert: 1 a.u. = 51.42 V/Å
            double a_alpha = ucell.lat0;
            if (PARAM.inp.deltap_gdir == 1)      a_alpha *= ucell.a1.norm();
            else if (PARAM.inp.deltap_gdir == 2) a_alpha *= ucell.a2.norm();
            else                                  a_alpha *= ucell.a3.norm();
            double lam_avg = 0.0;
            for (int iat = 0; iat < ucell.nat; ++iat) lam_avg += lambda[iat];
            lam_avg /= ucell.nat;
            double E_eff_au = -lam_avg * ModuleBase::PI / a_alpha;  // Hartree/(e·Bohr)
            double E_eff_V_per_A = E_eff_au * 51.422;                // V/Å
            std::cout << "   [E-field] E_eff=" << std::scientific << std::setprecision(3)
                      << E_eff_V_per_A << " V/Angstrom  (λ_avg=" << lam_avg << " Ry)" << std::endl;
        }
        else
        {
            ModuleBase::WARNING("ESolver_KS_LCAO::iter_finish",
                "deltap_corr only supports multi-k (complex<double>) calculations");
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
    // Branch A: Initialize Wilson loop infrastructure (first call only)
    // Build orb_onsite if not already built
    if (!two_center_bundle_.overlap_orb_onsite)
    {
        two_center_bundle_.build_orb_onsite(PARAM.inp.deltap_rm);
        two_center_bundle_.tabulate();
    }
    // Build position matrix and berry overlap calculators
    r_overlap_scf_ = new cal_r_overlap_R();
    static_cast<cal_r_overlap_R*>(r_overlap_scf_)->init(ucell, pv, orb_);
    berry_ovl_scf_ = new unkOverlap_lcao();
    static_cast<unkOverlap_lcao*>(berry_ovl_scf_)->init(ucell, kv.get_nkstot(), orb_);
    static_cast<unkOverlap_lcao*>(berry_ovl_scf_)->cal_R_number(ucell, gd);
    static_cast<unkOverlap_lcao*>(berry_ovl_scf_)->cal_orb_overlap(ucell);
    // Create DeltaP object
    auto* dp = new deltap::DeltaP();
    dp_scf_ = dp;
    dp->init(ucell, gd, kv,
              two_center_bundle_.overlap_orb_onsite.get(),
              two_center_bundle_.overlap_orb.get(),
              two_center_bundle_.overlap_onsite_onsite.get(),
              orb_.cutoffs(),
              PARAM.inp.deltap_rm, PARAM.inp.deltap_gdir,
              &pv, static_cast<cal_r_overlap_R*>(r_overlap_scf_),
              static_cast<unkOverlap_lcao*>(berry_ovl_scf_));
    dp->load_branch();
    dp->init_inner_loop();
    // Read target file (or STRU if not specified)
    if (!PARAM.inp.deltap_target_file.empty())
    {
        std::ifstream ifs(PARAM.inp.deltap_target_file);
        if (ifs.is_open())
        {
            bool total_mode = (PARAM.inp.deltap_constraint_mode == "total");
            if (total_mode)
            {
                deltap_target_.resize(ucell.nat, 0.0);
                double total_target = 0.0;
                ifs >> total_target;
                for (int iat = 0; iat < ucell.nat; ++iat)
                    deltap_target_[iat] = total_target / ucell.nat;
                std::cout << " [DeltaP] Loaded total target Σγ=" << total_target
                          << " → per-atom=" << total_target / ucell.nat << std::endl;
            }
            else
            {
                deltap_target_.assign(ucell.nat, 0.0);
                for (int iat = 0; iat < ucell.nat; ++iat)
                    ifs >> deltap_target_[iat];
                std::cout << " [DeltaP] Loaded target from " << PARAM.inp.deltap_target_file << std::endl;
            }
        }
    }
    else
    {
        // Read targets from STRU (DeltaSpin-style per-atom keywords)
        deltap_target_ = ucell.get_dp_target();
        deltap_constrain_ = ucell.get_dp_constrain();
        bool has_any_target = false;
        for (int iat = 0; iat < ucell.nat; ++iat)
            if (deltap_constrain_[iat] != 0 && deltap_target_[iat] != 0.0)
                has_any_target = true;
        if (has_any_target)
            std::cout << " [DeltaP] Loaded targets from STRU (dp_target/dp_constrain)" << std::endl;
        else
            std::cout << " [DeltaP] No targets specified; γ measured without constraint" << std::endl;
    }
    // When no target is specified, leave deltap_target_ empty.
    // This allows ground-state γ determination without target-aware
    // branch selection, while constraint (deltap_corr=1) still applies
    // the Hamiltonian correction with the current (possibly zero) λ.
    // Set target on DeltaP object for target-aware branch selection
    static_cast<deltap::DeltaP*>(dp_scf_)->set_target_gamma(deltap_target_);

    // Load constraint matrix (overrides constraint_mode when set)
    if (!PARAM.inp.deltap_constraint_matrix.empty())
    {
        std::ifstream ifs(PARAM.inp.deltap_constraint_matrix);
        if (ifs.is_open())
        {
            int m = 0, n = 0;
            ifs >> m >> n;
            if (n == ucell.nat && m > 0)
            {
                deltap_constraint_matrix_.resize(m, std::vector<double>(n, 0.0));
                deltap_constraint_target_.resize(m, 0.0);
                for (int a = 0; a < m; ++a)
                {
                    for (int i = 0; i < n; ++i)
                        ifs >> deltap_constraint_matrix_[a][i];
                    ifs >> deltap_constraint_target_[a];
                }
                deltap_constraint_lambda_.assign(m, PARAM.inp.deltap_lambda_init);
                static_cast<deltap::DeltaP*>(dp_scf_)->set_constraint_matrix(
                    deltap_constraint_matrix_, deltap_constraint_target_);
                std::cout << " [DeltaP] Loaded constraint matrix " << m << "x" << n
                          << " from " << PARAM.inp.deltap_constraint_matrix << std::endl;
            }
            else
            {
                std::cerr << "DeltaP: constraint matrix size mismatch (expected "
                          << ucell.nat << " columns, got " << n << ")" << std::endl;
            }
        }
    }
    // Initialize constraint lambda vector
    if (deltap_constraint_lambda_.empty())
        deltap_constraint_lambda_.assign(ucell.nat, 0.0);

    deltap_scf_initialized_ = true;
    deltap_inner_loop_done_ = false;
}

template <typename TK, typename TR>
double ESolver_KS_LCAO<TK, TR>::deltap_compute_gamma(UnitCell& ucell, const int iter)
{
    if constexpr (!std::is_same<TK, std::complex<double>>::value)
    {
        // Branch A: DeltaP compute gamma only supports complex<double> (multi-k)
        return 0.0;
    }
    else
    {
        // Compute gamma (post inner-loop or for diagnostic)
        auto* dp = static_cast<deltap::DeltaP*>(dp_scf_);
        dp->compute_gamma_scf(ucell, psi, this->pelec);
        const int alpha = PARAM.inp.deltap_gdir - 1;
        const auto& gamma_I = dp->get_results().gamma_I;

        double max_dev = 0.0;
        if (!deltap_constraint_matrix_.empty())
        {
            for (int a = 0; a < static_cast<int>(deltap_constraint_matrix_.size()); ++a)
            {
                double cv = 0.0;
                for (int i = 0; i < ucell.nat; ++i)
                    cv += deltap_constraint_matrix_[a][i] * gamma_I[i][alpha];
                max_dev = std::max(max_dev, std::abs(cv - deltap_constraint_target_[a]));
            }
        }
        else if (!deltap_target_.empty())
            for (int iat = 0; iat < ucell.nat; ++iat)
                max_dev = std::max(max_dev, std::abs(gamma_I[iat][alpha] - deltap_target_[iat]));

        return max_dev;
    }
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::deltap_inner_loop(UnitCell& ucell, const int iter, bool& skip_solve)
{
    if constexpr (!std::is_same<TK, std::complex<double>>::value) { return; }
    else
    {
        auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
        if (!hamilt_lcao || dp_scf_ == nullptr) { std::cout << " [DEBUG] hamilt or dp null" << std::endl; return; }
        auto* dp_op = hamilt_lcao->get_dp_operator();
        auto* dp = static_cast<deltap::DeltaP*>(dp_scf_);
        if (!dp_op || !dp || !dp->inner_loop_active()) { return; }

        // Gating: activate only when density is converged (Phase 1 done)
        if (this->drho > PARAM.inp.deltap_inner_thr) { return; }
        // Skip if inner loop already converged in a previous SCF iteration
        if (deltap_inner_loop_done_) { return; }

        // Measure current gamma
        dp->compute_gamma_scf(ucell, psi, this->pelec);
        const auto& gamma_I = dp->get_results().gamma_I;
        const int alpha = PARAM.inp.deltap_gdir - 1;

        // Compute residual: per_atom or constraint_matrix mode
        std::vector<double> residual;
        bool use_constraint_matrix = !deltap_constraint_matrix_.empty();

        if (use_constraint_matrix)
        {
            int m = static_cast<int>(deltap_constraint_matrix_.size());
            residual.resize(m, 0.0);
            for (int a = 0; a < m; ++a)
            {
                double cv = 0.0;
                for (int i = 0; i < ucell.nat; ++i)
                    cv += deltap_constraint_matrix_[a][i] * gamma_I[i][alpha];
                residual[a] = cv - deltap_constraint_target_[a];
            }
        }
        else
        {
            residual.resize(ucell.nat, 0.0);
            for (int iat = 0; iat < ucell.nat; ++iat)
                residual[iat] = gamma_I[iat][alpha] - deltap_target_[iat];
        }

        // BFGS-CG inner loop: optimize lambda with frozen charge density.
        // Each inner iteration re-diagonalizes with trial lambda (no density mixing),
        // which is ~10× faster than a full SCF step.
        int n_inner_atoms = use_constraint_matrix
            ? static_cast<int>(deltap_constraint_matrix_.size()) : ucell.nat;

        auto& bfgs = dp->bfgs();
        bfgs.init(n_inner_atoms, 0.5, PARAM.inp.deltap_conv_thr, 2, 0.01, 0.005);

        std::vector<double> lambda_inner;
        if (use_constraint_matrix)
        {
            lambda_inner = deltap_constraint_lambda_;
            bfgs.start_outer(lambda_inner);
        }
        else
        {
            lambda_inner = dp_op->get_lambda();
            bfgs.start_outer(lambda_inner);
        }

        bool bfgs_converged = false;
        const int nscf = dp->inner_loop_nscf();

        hsolver::HSolverLCAO<TK> hsolver_lcao_obj(&(this->pv), PARAM.inp.ks_solver);

        std::cout << " [DeltaP] inner loop start: nscf=" << nscf
                  << " rms=" << std::scientific << std::setprecision(4)
                  << bfgs.get_rms() << std::endl;

        for (int inner = 0; inner < nscf && !bfgs_converged; ++inner)
        {
            std::vector<double> lam_trial = lambda_inner;
            bfgs.step(residual, inner, lam_trial, bfgs_converged);
            if (bfgs_converged) break;

            // Apply trial lambda (convert to effective per-atom lambda)
            std::vector<double> lam_eff(ucell.nat, 0.0);
            if (use_constraint_matrix)
            {
                int m = static_cast<int>(deltap_constraint_matrix_.size());
                for (int i = 0; i < ucell.nat; ++i)
                    for (int a = 0; a < m; ++a)
                        lam_eff[i] += lam_trial[a] * deltap_constraint_matrix_[a][i];
            }
            else
            {
                lam_eff = lam_trial;
            }
            dp_op->set_lambda(lam_eff);

            std::unordered_map<int, std::vector<std::complex<double>>> hk_corr;
            dp->compute_hk_correction(ucell, psi, lam_eff, hk_corr);
            dp_op->set_hk_correction(hk_corr);

            // Re-solve with trial lambda (charge density frozen)
            hsolver_lcao_obj.solve(static_cast<hamilt::Hamilt<TK>*>(this->p_hamilt),
                this->psi[0], this->pelec, *this->dmat.dm, this->chr,
                PARAM.inp.nspin, true);  // skip_charge = true

            // Measure residual at trial point
            dp->compute_gamma_scf(ucell, psi, this->pelec);
            const auto& gamma_trial = dp->get_results().gamma_I;
            if (use_constraint_matrix)
            {
                int m = static_cast<int>(deltap_constraint_matrix_.size());
                for (int a = 0; a < m; ++a)
                {
                    double cv = 0.0;
                    for (int i = 0; i < ucell.nat; ++i)
                        cv += deltap_constraint_matrix_[a][i] * gamma_trial[i][alpha];
                    residual[a] = cv - deltap_constraint_target_[a];
                }
            }
            else
            {
                for (int iat = 0; iat < ucell.nat; ++iat)
                    residual[iat] = gamma_trial[iat][alpha] - deltap_target_[iat];
            }

            double alpha_opt = bfgs.accept_trial(residual);
            lambda_inner = lam_trial;

            std::cout << " [DeltaP]   inner=" << inner
                      << " rms=" << std::scientific << std::setprecision(4)
                      << bfgs.get_rms() << " alpha_opt=" << alpha_opt << std::endl;
        }

        // Set final lambda and reconstruct HK correction
        std::vector<double> lam_final(ucell.nat, 0.0);
        if (use_constraint_matrix)
        {
            int m = static_cast<int>(deltap_constraint_matrix_.size());
            for (int i = 0; i < ucell.nat; ++i)
                for (int a = 0; a < m; ++a)
                    lam_final[i] += lambda_inner[a] * deltap_constraint_matrix_[a][i];
            deltap_constraint_lambda_ = lambda_inner;
        }
        else
        {
            lam_final = lambda_inner;
        }
        dp_op->set_lambda(lam_final);
        std::unordered_map<int, std::vector<std::complex<double>>> hk_corr_final;
        dp->compute_hk_correction(ucell, psi, lam_final, hk_corr_final);
        dp_op->set_hk_correction(hk_corr_final);
        skip_solve = true;  // inner loop already solved

        std::cout << " [DeltaP] inner loop done: final";
        for (int iat = 0; iat < ucell.nat; ++iat)
            std::cout << " l" << iat << "=" << lam_final[iat];
        std::cout << std::endl;
        deltap_inner_loop_done_ = true;
    }
}

template <typename TK, typename TR>
void ESolver_KS_LCAO<TK, TR>::deltap_update_lambda(UnitCell& ucell, const int iter)
{
    if constexpr (!std::is_same<TK, std::complex<double>>::value)
    {
        // Branch B: DeltaP lambda update only supports complex<double> (multi-k)
        return;
    }
    else
    {
        auto* dp = static_cast<deltap::DeltaP*>(dp_scf_);
        if (!dp || dp->inner_loop_active())
        {
            // Branch C: If inner loop is active, lambda update is handled in deltap_inner_loop
            return;
        }

        auto* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
        if (!hamilt_lcao)
        {
            return;
        }

        auto* dp_op = hamilt_lcao->get_dp_operator();
        if (!dp_op)
        {
            return;
        }

        // Branch D: Two-phase strategy (nscf==0).
        // Phase 1: SCF converges with λ=0.  Gamma is computed (target-aware
        // branch selection runs) but λ remains zero — updating λ against
        // the crude atomic-guess charge density (drho~0.5) produces wrong λ.
        // Phase 2: once drho drops below deltap_inner_thr, the charge density
        // is converged enough for a single gradient-descent λ update.  λ is
        // then frozen for all remaining iterations.
        if (!deltap_lambda_set_ && this->drho > 0.0
            && this->drho < PARAM.inp.deltap_inner_thr)
        {
            deltap_lambda_set_ = true;

            const int alpha = PARAM.inp.deltap_gdir - 1;
            const auto& gamma_I = dp->get_results().gamma_I;
            std::vector<double> lambda = dp_op->get_lambda();

            double step = PARAM.inp.deltap_lambda_step;
            double mixing = PARAM.inp.deltap_lambda_mixing;
            if (mixing < 0.0) mixing = 0.0;
            if (mixing > 1.0) mixing = 1.0;
            if (mixing == 0.0) mixing = 1.0;

            std::vector<double> lambda_raw = lambda;
            bool total_mode = (PARAM.inp.deltap_constraint_mode == "total");
            bool use_constraint_matrix = !deltap_constraint_matrix_.empty();

            if (use_constraint_matrix)
            {
                // Constraint matrix mode: update λ in constraint space
                // r[α] = Σ_i C[α][i]·γ_i - t[α]
                // λ[α] += step · r[α]
                // Then convert to effective per-atom lambda
                int m = static_cast<int>(deltap_constraint_matrix_.size());
                deltap_constraint_lambda_.resize(m);
                std::vector<double> lambda_raw_cstr = deltap_constraint_lambda_;
                for (int a = 0; a < m; ++a)
                {
                    double residual = 0.0;
                    for (int i = 0; i < ucell.nat; ++i)
                        residual += deltap_constraint_matrix_[a][i] * gamma_I[i][alpha];
                    residual -= deltap_constraint_target_[a];
                    lambda_raw_cstr[a] += step * residual;
                }
                for (int a = 0; a < m; ++a)
                    deltap_constraint_lambda_[a] = mixing * lambda_raw_cstr[a]
                                                 + (1.0 - mixing) * deltap_constraint_lambda_[a];

                // Convert to effective per-atom lambda: λ_eff[i] = Σ_a λ[a]·C[a][i]
                lambda.assign(ucell.nat, 0.0);
                for (int i = 0; i < ucell.nat; ++i)
                    for (int a = 0; a < m; ++a)
                        lambda[i] += deltap_constraint_lambda_[a] * deltap_constraint_matrix_[a][i];
            }
            else if (!deltap_target_.empty())
            {
            if (use_constraint_matrix)
            {
                // Show per-atom γ and constraint-space λ
                for (int iat = 0; iat < ucell.nat; ++iat)
                {
                    if (iat > 0) std::cout << ", ";
                    std::cout << gamma_I[iat][alpha];
                }
                std::cout << ") C·γ=(" << std::setprecision(4);
                int m = static_cast<int>(deltap_constraint_matrix_.size());
                for (int a = 0; a < m; ++a)
                {
                    if (a > 0) std::cout << ", ";
                    double cv = 0.0;
                    for (int i = 0; i < ucell.nat; ++i)
                        cv += deltap_constraint_matrix_[a][i] * gamma_I[i][alpha];
                    std::cout << cv;
                }
                std::cout << ") λ=" << std::setprecision(2);
                for (int a = 0; a < m; ++a)
                {
                    if (a > 0) std::cout << ", ";
                    std::cout << std::scientific << deltap_constraint_lambda_[a];
                }
            }
            else if (total_mode)
                {
                    // total mode: one λ shared by all atoms
                    // λ_new = λ + step * (Σγ_actual - Σγ_target)
                    double total_actual = 0.0, total_target = 0.0;
                    for (int iat = 0; iat < ucell.nat; ++iat)
                    {
                        total_actual += gamma_I[iat][alpha];
                        total_target += deltap_target_[iat];
                    }
                    double delta = step * (total_actual - total_target);
                    for (int iat = 0; iat < ucell.nat; ++iat)
                        lambda_raw[iat] += delta;
                }
                else
                {
                    for (int iat = 0; iat < ucell.nat; ++iat)
                    {
                        bool constrained = (deltap_constrain_.empty()
                            || static_cast<size_t>(iat) >= deltap_constrain_.size()
                            || deltap_constrain_[iat] != 0);
                        if (constrained)
                            lambda_raw[iat] += step * (gamma_I[iat][alpha] - deltap_target_[iat]);
                    }
                }
            }
            if (!use_constraint_matrix)
            {
                for (int iat = 0; iat < ucell.nat; ++iat)
                    lambda[iat] = mixing * lambda_raw[iat] + (1.0 - mixing) * lambda[iat];
            }

            dp_op->set_lambda(lambda);
            dp->start_cooldown(1);

            // Reset charge mixing history: Broyden's approximate Jacobian
            // from the unconstrained iter=1 is invalid for the constrained
            // Hamiltonian.  Without this reset, the stale history causes
            // charge sloshing (drho oscillation at 3e-4~5e-4 level).
            // Reference: same pattern used in DeltaSpin Phase 1→2 transition.
            this->p_chgmix->mix_reset();

            // Phase 2 transition: print summary
            std::cout << " [DeltaP P2] iter=" << iter << " drho=" << std::scientific
                      << std::setprecision(2) << this->drho << " < " << PARAM.inp.deltap_inner_thr
                      << " → λ updated, mix_reset()\n";
        }

        // Always recompute HK correction with latest wavefunctions
        std::vector<double> lambda = dp_op->get_lambda();
        std::unordered_map<int, std::vector<std::complex<double>>> hk_corr;
        dp->compute_hk_correction(ucell, psi, lambda, hk_corr);
        dp_op->set_hk_correction(hk_corr);
    }
}

template class ESolver_KS_LCAO<double, double>;
template class ESolver_KS_LCAO<std::complex<double>, double>;
template class ESolver_KS_LCAO<std::complex<double>, std::complex<double>>;
} // namespace ModuleESolver
