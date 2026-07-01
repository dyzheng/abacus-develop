#include "deltaspin_lcao.h"
#include "spin_constrain.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_lcao/hamilt_lcao.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_estate/elecstate.h"

/**
 * @file deltaspin_lcao.cpp
 * @brief Wrapper/facade layer between ESolver and DeltaSpin module.
 *
 * @par Purpose
 * Provides a simplified interface to the ESolver layer, hiding the
 * SpinConstrain Singleton details. The ESolver calls these functions
 * rather than accessing SpinConstrain directly.
 *
 * @par Design rationale
 * - Template functions: Support both TK=double (nspin=2) and TK=complex<double> (nspin=4)
 * - Early returns: If sc_mag_switch is false, all functions return immediately
 *   without any overhead
 * - #ifdef __LCAO: The density matrix pointer is only available in LCAO builds
 *
 * @par Workflow
 * 1. ESolver calls init_deltaspin_lcao() at start of calculation
 * 2. Each SCF iteration:
 *    a. ESolver calls cal_mi_lcao_wrapper() to compute magnetic moments
 *    b. ESolver calls run_deltaspin_lambda_loop_lcao() to optimize lambda
 *    c. If skip_solve=true, ESolver skips the Hamiltonian solve (lambda loop already did it)
 */

namespace ModuleESolver
{

/**
 * @brief Initialize the SpinConstrain singleton with all input parameters.
 *
 * @details Called once at the start of a DeltaSpin calculation. Checks
 * sc_mag_switch first; if disabled, returns immediately without any action.
 *
 * @par Conditional compilation
 * The density matrix pointer (dm) is only available when __LCAO is defined.
 * For non-LCAO builds (PW-only), init_sc() is called without the dm parameter.
 */
template <typename TK>
void init_deltaspin_lcao(const UnitCell& ucell,
                          const Input_para& inp,
                          void* pv,
                          const K_Vectors& kv,
                          void* p_hamilt,
                          void* psi,
                          void* dm,
                          void* pelec,
                          void* gridD,
                          void* intor,
                          const std::vector<double>& orb_cutoff,
                          void* hR)
{
    // Early exit if neither DeltaSpin nor DeltaQ is enabled
    if (!inp.sc_mag_switch && !inp.sc_charge_switch)
    {
        return;
    }

    spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();

    // Derive acceleration parameters from sc_strategy
    std::string accel_mode = inp.sc_acceleration_mode;
    double accel_rms_thr = inp.sc_acceleration_rms_thr;

    bool user_overrode_accel_mode = (inp.sc_acceleration_mode != "off");
    bool user_overrode_rms_thr = (inp.sc_acceleration_rms_thr > 0.0);

    if (!user_overrode_accel_mode || !user_overrode_rms_thr)
    {
        if (inp.sc_strategy == "fast")
        {
            accel_mode = "subspace";
            accel_rms_thr = 1e10;
        }
        else if (inp.sc_strategy == "accuracy")
        {
            accel_mode = "off";
            accel_rms_thr = -1.0;
        }
        else // normal
        {
            accel_mode = "subspace";
            if (!user_overrode_rms_thr)
            {
                accel_rms_thr = 1e-2;
            }
        }
    }

    if (inp.sc_charge_switch)
    {
        accel_mode = "off";
        accel_rms_thr = -1.0;
        std::cout << "[DeltaQS] Subspace acceleration disabled for charge constraint mode" << std::endl;
    }

#ifdef __LCAO
    // LCAO build: pass density matrix pointer
    sc.init_sc(inp.sc_thr, inp.nsc, inp.nsc_min, inp.alpha_trial,
               inp.sccut, inp.sc_drop_thr,
               accel_mode, accel_rms_thr,
               ucell, inp.sc_direction_only,
               static_cast<Parallel_Orbitals*>(pv),
               inp.nspin, kv, p_hamilt, psi,
               static_cast<elecstate::DensityMatrix<TK, double>*>(dm),
               static_cast<elecstate::ElecState*>(pelec));
#else
    // Non-LCAO build: no density matrix
    sc.init_sc(inp.sc_thr, inp.nsc, inp.nsc_min, inp.alpha_trial,
               inp.sccut, inp.sc_drop_thr,
               accel_mode, accel_rms_thr,
               ucell, inp.sc_direction_only,
               static_cast<Parallel_Orbitals*>(pv),
               inp.nspin, kv, p_hamilt, psi,
               static_cast<elecstate::ElecState*>(pelec));
#endif

    // Initialize DeltaQS charge constraint data
    sc.init_deltaqs(ucell,
                    inp.sc_charge_switch,
                    inp.sc_qs_mode,
                    inp.sc_charge_mode,
                    inp.sc_charge_thr,
                    inp.sc_charge_alpha,
                    inp.sc_charge_sccut,
                    inp.sc_ground_state_search,
                    inp.sc_outer_max_iter,
                    inp.sc_outer_thr,
                    inp.sc_gradient_output,
                    gridD,
                    intor,
                    orb_cutoff,
                    hR);
}

/**
 * @brief Wrapper: calculate magnetic moments for current SCF iteration.
 *
 * @details If DeltaSpin is enabled, calls SpinConstrain::cal_mi_lcao().
 * The moments are stored in Mi_ and can be retrieved via get_target_mag().
 */
template <typename TK>
void cal_mi_lcao_wrapper(const int iter, const Input_para& inp)
{
    if (!inp.sc_mag_switch && !inp.sc_charge_switch)
    {
        return;
    }

#ifdef __LCAO
    spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
    if (inp.sc_mag_switch)
    {
        sc.cal_mi_lcao(iter);
    }
    if (inp.sc_charge_switch)
    {
        sc.cal_ni_lcao(iter, true);
    }
#endif
}

/**
 * @brief Wrapper: run the lambda optimization loop.
 *
 * @details Decision logic for when to run the lambda loop:
 *
 *   Case 1: NOT converged AND charge density is close enough (drho < sc_scf_thr)
 *   -> Run lambda loop, mark as converged, skip_solve = true
 *   Rationale: The charge density is stable enough to optimize lambda.
 *   The lambda loop does its own diagonalization, so skip the outer solve.
 *
 *   Case 2: Already converged
 *   -> Still run lambda loop (to refine for the current charge density)
 *   -> skip_solve = true
 *   Rationale: Even if converged, the charge density may have changed
 *   slightly, requiring lambda refinement.
 *
 *   Case 3: NOT converged AND charge density is NOT close enough (drho >= sc_scf_thr)
 *   -> Do nothing, skip_solve = false
 *   Rationale: The charge density is still changing significantly, so
 *   optimizing lambda would be premature. Wait for SCF to stabilize first.
 *
 * @param iter Current SCF iteration number
 * @param drho Charge density convergence criterion (max|drho|)
 * @param inp Input parameters
 * @return true if the ESolver should skip the Hamiltonian solve
 */
template <typename TK>
bool run_deltaspin_lambda_loop_lcao(const int iter,
                                      const double drho,
                                      const Input_para& inp)
{
    bool skip_solve = false;

    if (inp.sc_mag_switch || inp.sc_charge_switch)
    {
        spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
        bool use_qs = sc.is_charge_constraint_enabled();

        if (!sc.mag_converged() && drho > 0 && drho < inp.sc_scf_thr)
        {
            sc.set_drho(drho);
            if (use_qs)
                sc.run_qs_lambda_loop(iter);
            else if (inp.sc_mag_switch)
                sc.run_lambda_loop(iter);
            sc.set_mag_converged(true);
            skip_solve = true;
        }
        else if (sc.mag_converged())
        {
            sc.set_drho(drho);
            if (use_qs)
                sc.run_qs_lambda_loop(iter);
            else if (inp.sc_mag_switch)
                sc.run_lambda_loop(iter);
            skip_solve = true;
        }
    }

    return skip_solve;
}

template <typename TK>
void run_deltaspin_scan_diagnostic_lcao(const int iter, const Input_para& inp)
{
    if (!inp.sc_mag_switch)
    {
        return;
    }

#ifdef __LCAO
    spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
    sc.run_lambda_scan_diagnostic(iter);
#endif
}

/// Template instantiations for both spin types
template void init_deltaspin_lcao<double>(const UnitCell& ucell,
                                           const Input_para& inp,
                                           void* pv,
                                           const K_Vectors& kv,
                                           void* p_hamilt,
                                           void* psi,
                                           void* dm,
                                           void* pelec,
                                           void* gridD,
                                           void* intor,
                                           const std::vector<double>& orb_cutoff,
                                           void* hR);
template void init_deltaspin_lcao<std::complex<double>>(const UnitCell& ucell,
                                                          const Input_para& inp,
                                                          void* pv,
                                                          const K_Vectors& kv,
                                                          void* p_hamilt,
                                                          void* psi,
                                                          void* dm,
                                                          void* pelec,
                                                          void* gridD,
                                                          void* intor,
                                                          const std::vector<double>& orb_cutoff,
                                                          void* hR);

template void cal_mi_lcao_wrapper<double>(const int iter, const Input_para& inp);
template void cal_mi_lcao_wrapper<std::complex<double>>(const int iter, const Input_para& inp);

template bool run_deltaspin_lambda_loop_lcao<double>(const int iter,
                                                      const double drho,
                                                      const Input_para& inp);
template bool run_deltaspin_lambda_loop_lcao<std::complex<double>>(const int iter,
                                                                      const double drho,
                                                                      const Input_para& inp);

template void run_deltaspin_scan_diagnostic_lcao<double>(const int iter, const Input_para& inp);
template void run_deltaspin_scan_diagnostic_lcao<std::complex<double>>(const int iter, const Input_para& inp);

} // namespace ModuleESolver
