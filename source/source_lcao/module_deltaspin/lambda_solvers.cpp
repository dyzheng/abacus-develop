#include "lambda_solvers.h"
#include "spin_constrain.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "basic_funcs.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_gint/gint_interface.h"

#ifdef __LCAO
#include "source_lcao/module_operator_lcao/dspin_lcao.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_estate/elecstate_tools.h"
#endif

namespace spinconstrain
{

// ===================================================================
// Type conversion helpers
// ===================================================================

LambdaSolverType lambda_solver_type_from_string(const std::string& s)
{
    if (s == "bfgs" || s == "BFGS") return LambdaSolverType::BFGS;
    if (s == "chi_guided" || s == "chi-guided" || s == "chiguided") return LambdaSolverType::ChiGuided;
    if (s == "subspace" || s == "subspace_diag") return LambdaSolverType::Subspace;
    if (s == "fdcg" || s == "FDCG" || s == "fd_cg" || s == "fd-cg") return LambdaSolverType::FDCG;
    return LambdaSolverType::FDCG;
}

std::string lambda_solver_type_to_string(LambdaSolverType t)
{
    switch (t)
    {
        case LambdaSolverType::BFGS: return "BFGS";
        case LambdaSolverType::ChiGuided: return "ChiGuided";
        case LambdaSolverType::Subspace: return "Subspace";
        case LambdaSolverType::FDCG: return "FDCG";
        default: return "Unknown";
    }
}

// ===================================================================
// Solver 1: BFGS with line search
//
// Ported from SpinConstrain::run_lambda_loop (lambda_loop.cpp:106-319).
// Performs a full SCF diagonalization each inner step.
// Works for both PW and LCAO bases.
// ===================================================================

LambdaSolverResult BFGSLambdaSolver::run(int outer_step)
{
    LambdaSolverResult result{};
    const int nat = sc_.get_nat();
    const int nsc = sc_.get_nsc();
    const int nsc_min = sc_.get_nsc_min();
    const double zero = 0.0;
    const double one = 1.0;

    auto t_start = std::chrono::steady_clock::now();

    // Working vectors
    std::vector<ModuleBase::Vector3<double>> initial_lambda(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> delta_lambda(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> dnu(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> dnu_last_step(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> temp_1(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> spin(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> delta_spin(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> search(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> search_old(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> new_spin(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> spin_plus(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> cur_lambda(nat, 0.0);

    double alpha_opt = 0.0;
    double alpha_plus = 0.0;
    double beta = 0.0;
    double g = 0.0;
    double mean_error = 0.0;
    double mean_error_old = 0.0;
    double rms_error = 0.0;

    double alpha_trial = sc_.get_alpha_trial();
    double inner_loop_duration = 0.0;

    sc_.print_header();

    // --- Lambda loop ---
    for (int i_step = -1; i_step < nsc; i_step++)
    {
        double duration = 0.0;
        auto iterstart = std::chrono::steady_clock::now();

        if (i_step == -1)
        {
            sc_.cal_mw_from_lambda(i_step);
            spin = sc_.get_Mi();
            where_fill_scalar_else_2d(sc_.get_constrain(), 0, zero, sc_.get_sc_lambda(), initial_lambda);
            print_2d("initial lambda (eV/uB): ", initial_lambda, sc_.get_nspin(), ModuleBase::Ry_to_eV);
            print_2d("initial spin (uB): ", spin, sc_.get_nspin());
            print_2d("target spin (uB): ", sc_.get_target_mag(), sc_.get_nspin());
            i_step++;
        }
        else
        {
            // Reset delta_lambda (zero out, fill from dnu for constrained atoms)
            std::fill(delta_lambda.begin(), delta_lambda.end(), ModuleBase::Vector3<double>(0.0));
            add_scalar_multiply_2d(delta_lambda, dnu, one, delta_lambda);
            // Mask: only constrained components get delta_lambda values
            where_fill_scalar_else_2d(sc_.get_constrain(), 0, zero, delta_lambda, delta_lambda);

            // Compute new lambda = initial_lambda + delta_lambda
            std::fill(cur_lambda.begin(), cur_lambda.end(), ModuleBase::Vector3<double>(0.0));
            add_scalar_multiply_2d(initial_lambda, delta_lambda, one, cur_lambda);
            sc_.set_lambda(cur_lambda);

            sc_.cal_mw_from_lambda(i_step);

            new_spin = sc_.get_Mi();
            bool grad_ok = sc_.check_gradient_decay(new_spin, spin, delta_lambda, dnu_last_step);
            if (i_step >= nsc_min && grad_ok)
            {
                std::fill(cur_lambda.begin(), cur_lambda.end(), ModuleBase::Vector3<double>(0.0));
                add_scalar_multiply_2d(initial_lambda, dnu_last_step, one, cur_lambda);
                sc_.set_lambda(cur_lambda);
                sc_.update_psi_charge(dnu_last_step.data());

                auto t_now = std::chrono::steady_clock::now();
                duration = std::chrono::duration<double>(t_now - t_start).count();
                inner_loop_duration += duration;
                result.total_time = inner_loop_duration;
                std::cout << "Total TIME(s) = " << inner_loop_duration << std::endl;
                sc_.print_termination();
                result.converged = true;
                result.rms_error = rms_error;
                result.n_inner_steps = i_step + 1;
                return result;
            }
            spin = new_spin;
        }

        // Compute residual: delta_spin = spin - target_mag
        subtract_2d(spin, sc_.get_target_mag(), delta_spin);
        where_fill_scalar_2d(sc_.get_constrain(), 0, zero, delta_spin);
        search = delta_spin;

        // Compute RMS error
        for (int ia = 0; ia < nat; ia++)
        {
            for (int ic = 0; ic < 3; ic++)
            {
                temp_1[ia][ic] = std::pow(delta_spin[ia][ic], 2);
            }
        }
        mean_error = sum_2d(temp_1) / nat;
        rms_error = std::sqrt(mean_error);

        if (i_step == 0)
        {
            sc_.set_current_sc_thr(std::max(rms_error * sc_.get_sc_drop_thr(), sc_.get_sc_thr()));
        }

        auto t_now = std::chrono::steady_clock::now();
        duration = std::chrono::duration<double>(t_now - t_start).count();
        inner_loop_duration += duration;

        if (sc_.check_rms_stop(outer_step, i_step, rms_error, duration, inner_loop_duration))
        {
            sc_.update_psi_charge(dnu_last_step.data());
            if (PARAM.inp.basis_type == "pw")
            {
                sc_.cal_Mi_pw();
                subtract_2d(sc_.get_Mi(), sc_.get_target_mag(), delta_spin);
                where_fill_scalar_2d(sc_.get_constrain(), 0, zero, delta_spin);
                for (int ia = 0; ia < nat; ia++)
                {
                    for (int ic = 0; ic < 3; ic++)
                    {
                        temp_1[ia][ic] = std::pow(delta_spin[ia][ic], 2);
                    }
                }
                mean_error = sum_2d(temp_1) / nat;
                rms_error = std::sqrt(mean_error);
                std::cout << "Current RMS: " << rms_error << std::endl;
                if (rms_error > sc_.get_current_sc_thr() * 10 && sc_.higher_mag_prec)
                {
                    std::cout << "Error: RMS error is too large, rerun the loop" << std::endl;
                    result = this->run(outer_step);
                    return result;
                }
            }
            result.converged = true;
            result.rms_error = rms_error;
            result.total_time = inner_loop_duration;
            result.n_inner_steps = i_step + 1;
            return result;
        }

        // BFGS search direction
        if (i_step >= 2)
        {
            beta = mean_error / mean_error_old;
            add_scalar_multiply_2d(search, search_old, beta, search);
        }

        sc_.check_restriction(search, alpha_trial);

        dnu_last_step = dnu;
        add_scalar_multiply_2d(dnu, search, alpha_trial, dnu);
        delta_lambda = dnu;

        // Mask lambda and compute new
        where_fill_scalar_else_2d(sc_.get_constrain(), 0, zero, delta_lambda, delta_lambda);
        std::fill(cur_lambda.begin(), cur_lambda.end(), ModuleBase::Vector3<double>(0.0));
        add_scalar_multiply_2d(initial_lambda, delta_lambda, one, cur_lambda);
        sc_.set_lambda(cur_lambda);

        sc_.cal_mw_from_lambda(i_step, delta_lambda.data());

        spin_plus = sc_.get_Mi();

        alpha_opt = sc_.cal_alpha_opt(spin, spin_plus, alpha_trial);
        sc_.check_restriction(search, alpha_opt);

        alpha_plus = alpha_opt - alpha_trial;
        scalar_multiply_2d(search, alpha_plus, temp_1);
        add_scalar_multiply_2d(dnu, temp_1, one, dnu);
        delta_lambda = dnu;

        search_old = search;
        mean_error_old = mean_error;

        g = 1.5 * std::abs(alpha_opt) / alpha_trial;
        if (g > 2.0) g = 2.0;
        else if (g < 0.5) g = 0.5;
        alpha_trial = alpha_trial * std::pow(g, 0.7);

        t_now = std::chrono::steady_clock::now();
        duration = std::chrono::duration<double>(t_now - iterstart).count();
    }

    result.rms_error = rms_error;
    result.total_time = inner_loop_duration;
    result.converged = false;
    return result;
}

// ===================================================================
// Solver 2: Chi-Guided Newton with full diagonalization
//
// Ported from SpinConstrain::run_lambda_loop_lcao (lambda_loop.cpp:331-503).
// Designed for LCAO nspin=2.
// Phase 1: full diag to get Mi, C_k, e_k
// Phase 2: compute analytical chi = dM/dlambda via P_I_sub
// Phase 3+: Newton step with full diag + secant chi update
// Phase 7: update DM and charge from current psi
// ===================================================================

#ifdef __LCAO
LambdaSolverResult ChiGuidedLambdaSolver::run(int outer_step)
{
    LambdaSolverResult result{};
    const int nat = sc_.get_nat();
    const int nks = sc_.kv_.get_nks();
    const int nk = nks / 2;
    const int max_inner_iter = sc_.get_nsc();
    const double alpha_damp = 0.8;
    const double zero = 0.0;

    auto t_start = std::chrono::steady_clock::now();

    psi::Psi<std::complex<double>>* psi_t
        = static_cast<psi::Psi<std::complex<double>>*>(sc_.psi);
    const int nbands = sc_.ParaV->get_nbands();

    sc_.print_header();

    // --- Phase 1: Full diagonalization to get C_k, e_k, Mi ---
    sc_.cal_mw_from_lambda(-1);
    std::vector<ModuleBase::Vector3<double>> spin(nat);
    spin = sc_.get_Mi();

    std::vector<ModuleBase::Vector3<double>> initial_lambda(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> cur_lambda = sc_.get_sc_lambda();
    where_fill_scalar_else_2d(sc_.get_constrain(), 0, zero, cur_lambda, initial_lambda);

    print_2d("initial lambda (eV/uB): ", initial_lambda, sc_.get_nspin(), ModuleBase::Ry_to_eV);
    print_2d("initial spin (uB): ", spin, sc_.get_nspin());
    print_2d("target spin (uB): ", sc_.get_target_mag(), sc_.get_nspin());

    // Check initial convergence
    std::vector<ModuleBase::Vector3<double>> delta_spin(nat, 0.0);
    subtract_2d(spin, sc_.get_target_mag(), delta_spin);
    where_fill_scalar_2d(sc_.get_constrain(), 0, zero, delta_spin);
    double rms_error = 0.0;
    {
        double sum = 0.0;
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
                sum += std::pow(delta_spin[ia][ic], 2);
        rms_error = std::sqrt(sum / nat);
    }
    sc_.set_current_sc_thr(std::max(rms_error * sc_.get_sc_drop_thr(), sc_.get_sc_thr()));

    double current_sc_thr = sc_.get_current_sc_thr();

    if (rms_error < current_sc_thr)
    {
        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- 0"
                  << "       RMS = " << rms_error << std::endl;
        std::cout << "Meet convergence criterion ( < " << current_sc_thr << " ), exit." << std::endl;
        sc_.print_termination();
        sc_.pelec->psiToRho(*psi_t);
        result.converged = true;
        result.rms_error = rms_error;
        result.total_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return result;
    }

    // --- Phase 2: Compute analytical chi ---
    auto* dspin_op = dynamic_cast<
        hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(
        sc_.get_operator());

    std::vector<std::vector<std::vector<std::complex<double>>>> PI_sub(nks);
    for (int ik = 0; ik < nks; ik++)
    {
        psi_t->fix_k(ik);
        dspin_op->cal_PI_sub(sc_.kv_.kvec_d[ik], psi_t->get_pointer(), nbands, PI_sub[ik]);
    }

    // chi_I = dM_I^z / dlambda_I
    std::vector<double> chi(nat, 0.0);
    for (int iat = 0; iat < nat; iat++)
    {
        if (sc_.get_constrain()[iat].z == 0) { continue; }
        double chi_val = 0.0;
        for (int ik = 0; ik < nks; ik++)
        {
            if (PI_sub[ik][iat].empty()) { continue; }
            const auto& P = PI_sub[ik][iat];
            const double wk = sc_.pelec->klist->wk[ik];
            for (int n = 0; n < nbands; n++)
            {
                const double fn = sc_.pelec->wg(ik, n) / wk;
                for (int m = n + 1; m < nbands; m++)
                {
                    const double fm = sc_.pelec->wg(ik, m) / wk;
                    const double de = sc_.pelec->ekb(ik, n) - sc_.pelec->ekb(ik, m);
                    if (std::abs(de) < 1e-10) { continue; }
                    const double P_nm_sq = std::norm(P[n * nbands + m]);
                    chi_val += 2.0 * wk * (fn - fm) * P_nm_sq / de;
                }
            }
        }
        chi[iat] = chi_val;
    }

    // --- Phase 3-6: Newton iteration with full diag + secant chi update ---
    std::vector<ModuleBase::Vector3<double>> lambda_old(nat);
    std::vector<ModuleBase::Vector3<double>> Mi_old(nat);

    for (int inner = 0; inner < max_inner_iter; inner++)
    {
        lambda_old = sc_.get_sc_lambda();
        Mi_old = spin;

        const double lambda_max = sc_.get_restrict_current();
        for (int iat = 0; iat < nat; iat++)
        {
            if (sc_.get_constrain()[iat].z == 0) { continue; }
            double chi_clamped = chi[iat];
            if (std::abs(chi_clamped) < 0.1)
            {
                chi_clamped = (chi_clamped >= 0) ? 0.1 : -0.1;
            }
            double delta_lambda_z = alpha_damp
                * (sc_.get_target_mag()[iat].z - spin[iat].z) / chi_clamped;
            if (std::abs(delta_lambda_z) > lambda_max)
            {
                delta_lambda_z = (delta_lambda_z > 0) ? lambda_max : -lambda_max;
            }
            // Build new lambda from initial_lambda + delta
            cur_lambda = sc_.get_sc_lambda();
            cur_lambda[iat].z = initial_lambda[iat].z + delta_lambda_z;
            sc_.set_lambda(cur_lambda);
        }

        // Full diagonalization
        sc_.cal_mw_from_lambda(inner);
        spin = sc_.get_Mi();

        // Secant chi update: chi = (Mi_new - Mi_old) / (lambda_new - lambda_old)
        for (int iat = 0; iat < nat; iat++)
        {
            if (sc_.get_constrain()[iat].z == 0) { continue; }
            double dlambda = sc_.get_sc_lambda()[iat].z - lambda_old[iat].z;
            double dMi = spin[iat].z - Mi_old[iat].z;
            if (std::abs(dlambda) > 1e-10)
            {
                double chi_secant = dMi / dlambda;
                if (chi_secant * chi[iat] > 0
                    && std::abs(chi_secant) > 0.01
                    && std::abs(chi_secant) < 100.0)
                {
                    chi[iat] = chi_secant;
                }
            }
        }

        // Check convergence
        subtract_2d(spin, sc_.get_target_mag(), delta_spin);
        where_fill_scalar_2d(sc_.get_constrain(), 0, zero, delta_spin);
        {
            double sum = 0.0;
            for (int ia = 0; ia < nat; ia++)
                for (int ic = 0; ic < 3; ic++)
                    sum += std::pow(delta_spin[ia][ic], 2);
            rms_error = std::sqrt(sum / nat);
        }

        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- "
                  << std::left << std::setw(5) << inner + 1
                  << "       RMS = " << rms_error << std::endl;

        if (rms_error < current_sc_thr)
        {
            std::cout << "Meet convergence criterion ( < " << current_sc_thr
                      << " ), exit." << std::endl;
            result.converged = true;
            break;
        }
    }

    sc_.print_termination();

    // --- Phase 7: Update DM/charge from current psi ---
    elecstate::cal_dm_psi(sc_.ParaV, sc_.pelec->wg, *psi_t, *sc_.dm_);
    sc_.dm_->cal_DMR();

    int nspin = PARAM.inp.nspin;
    if (PARAM.inp.nspin == 4) { nspin = 1; }
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        ModuleBase::GlobalFunc::ZEROS(sc_.pelec->charge->rho[is], sc_.pelec->charge->nrxx);
    }
    ModuleGint::cal_gint_rho(sc_.dm_->get_DMR_vector(), nspin, sc_.pelec->charge->rho);
    sc_.pelec->charge->renormalize_rho();

    result.rms_error = rms_error;
    result.n_inner_steps = (result.converged) ? 1 : 1;
    result.total_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return result;
}
#else // !__LCAO
LambdaSolverResult ChiGuidedLambdaSolver::run(int outer_step)
{
    // ChiGuided solver requires LCAO support; fall back to BFGS
    BFGSLambdaSolver bfgs(sc_);
    return bfgs.run(outer_step);
}
#endif // __LCAO

// ===================================================================
// Solver 3: Subspace Diagonalization (experimental)
//
// Fewer full diagonalizations than ChiGuided.
// Phase 1: full diag
// Phase 2: P_I_sub for analytical chi
// Phase 3: subspace diag within eigen-subspace (no full SCF)
// Phase 4: wavefunction rotation
// Phase 5: update DM/charge from rotated psi
// ===================================================================

LambdaSolverResult SubspaceLambdaSolver::run(int outer_step)
{
    LambdaSolverResult result{};
    result.converged = false;

    auto t_start = std::chrono::steady_clock::now();

    // For now: use the same logic as ChiGuided (full diag each step)
    // Subspace optimization will be implemented when the analytical chi
    // and subspace diagonalization infrastructure is validated.
#ifdef __LCAO
    ChiGuidedLambdaSolver fallback(sc_);
    result = fallback.run(outer_step);
    result.n_inner_steps = -result.n_inner_steps; // mark as subspace fallback
#else
    // PW: use BFGS fallback
    sc_.print_header();
    sc_.cal_mw_from_lambda(-1);
    result.converged = sc_.mag_converged();
#endif

    result.total_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return result;
}

// ===================================================================
// Solver 4: Finite-Difference Jacobian + Conjugate Gradient
//
// Step 0: evaluate M(lambda_0) — 1 diag (i_step=-1)
// Step 1: FD Jacobian via simultaneous perturbation — 1 diag
// Step 2+: CG iteration with Polak-Ribiere+ beta, analytical
//          step size, and secant Jacobian update — 1 diag per step
// Works for both LCAO and PW, both nspin=2 and nspin=4.
// ===================================================================

LambdaSolverResult FDCGLambdaSolver::run(int outer_step)
{
    LambdaSolverResult result{};
    const int nat = sc_.get_nat();
    const int nsc = sc_.get_nsc();
    const double zero = 0.0;
    const double fd_delta = 1e-4; // Ry/uB, ~1.4e-3 eV/uB
    const double chi_min = 0.01;
    const double chi_max = 100.0;

    auto t_start = std::chrono::steady_clock::now();

    const auto& constrain = sc_.get_constrain();
    const auto& target = sc_.get_target_mag();

    // Working vectors
    std::vector<ModuleBase::Vector3<double>> lambda_0(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> lambda_cur(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> lambda_new(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> delta_for_pw(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> M_cur(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> chi(nat, 0.0);  // diagonal Jacobian
    std::vector<ModuleBase::Vector3<double>> g(nat, 0.0);     // gradient
    std::vector<ModuleBase::Vector3<double>> g_old(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> d(nat, 0.0);     // CG direction
    std::vector<ModuleBase::Vector3<double>> r(nat, 0.0);     // residual

    double rms_error = 0.0;
    double g_old_norm_sq = 0.0;

    sc_.print_header();

    // === Step 0: Initial evaluation (1 diag) ===
    lambda_0 = sc_.get_sc_lambda();
    where_fill_scalar_else_2d(constrain, 0, zero, lambda_0, lambda_0);
    sc_.cal_mw_from_lambda(-1);
    M_cur = sc_.get_Mi();

    print_2d("initial lambda (eV/uB): ", lambda_0, sc_.get_nspin(), ModuleBase::Ry_to_eV);
    print_2d("initial spin (uB): ", M_cur, sc_.get_nspin());
    print_2d("target spin (uB): ", target, sc_.get_nspin());

    // Compute initial RMS
    rms_error = 0.0;
    {
        double sum = 0.0;
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                if (constrain[ia][ic] == 0) { continue; }
                sum += std::pow(target[ia][ic] - M_cur[ia][ic], 2);
            }
        rms_error = std::sqrt(sum / nat);
    }
    sc_.set_current_sc_thr(std::max(rms_error * sc_.get_sc_drop_thr(), sc_.get_sc_thr()));
    const double current_sc_thr = sc_.get_current_sc_thr();

    // Check if already converged
    if (rms_error < current_sc_thr)
    {
        auto t_now = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(t_now - t_start).count();
        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- 0"
                  << "       RMS = " << rms_error << "     TIME(s) = " << duration << std::endl;
        std::cout << "Meet convergence criterion ( < " << current_sc_thr << " ), exit." << std::endl;
        sc_.print_termination();
        // Update charge
        if (PARAM.inp.basis_type == "lcao")
        {
#ifdef __LCAO
            psi::Psi<std::complex<double>>* psi_t
                = static_cast<psi::Psi<std::complex<double>>*>(sc_.psi);
            sc_.pelec->psiToRho(*psi_t);
#endif
        }
        else
        {
            sc_.update_psi_charge(nullptr);
        }
        result.converged = true;
        result.rms_error = rms_error;
        result.total_time = duration;
        return result;
    }

    // === Step 1: FD Jacobian (1 diag) ===
    // Perturb all constrained lambda simultaneously by fd_delta
    std::vector<ModuleBase::Vector3<double>> lambda_pert = lambda_0;
    for (int ia = 0; ia < nat; ia++)
        for (int ic = 0; ic < 3; ic++)
        {
            if (constrain[ia][ic] == 0) { continue; }
            lambda_pert[ia][ic] += fd_delta;
        }
    sc_.set_lambda(lambda_pert);

    // For PW: delta_lambda relative to lambda_0
    for (int ia = 0; ia < nat; ia++)
        for (int ic = 0; ic < 3; ic++)
            delta_for_pw[ia][ic] = lambda_pert[ia][ic] - lambda_0[ia][ic];

    sc_.cal_mw_from_lambda(0, delta_for_pw.data());
    auto M_pert = sc_.get_Mi();

    // Compute diagonal Jacobian: chi_I = dM_I / dlambda_I
    for (int ia = 0; ia < nat; ia++)
        for (int ic = 0; ic < 3; ic++)
        {
            if (constrain[ia][ic] == 0) { continue; }
            double chi_val = (M_pert[ia][ic] - M_cur[ia][ic]) / fd_delta;
            // Clamp magnitude
            if (std::abs(chi_val) < chi_min)
            {
                chi_val = (chi_val >= 0) ? chi_min : -chi_min;
            }
            else if (std::abs(chi_val) > chi_max)
            {
                chi_val = (chi_val > 0) ? chi_max : -chi_max;
            }
            chi[ia][ic] = chi_val;
        }

    // Restore to lambda_0 for CG iteration start
    lambda_cur = lambda_0;
    // M_cur stays as M_0 (the unperturbed value)

    // === Step 2+: CG iteration ===
    for (int k = 0; k < nsc; k++)
    {
        // Residual: r_I = target_I - M_cur_I
        // Gradient of f = 0.5*||M-target||^2: g_I = -chi_I * r_I
        double g_norm_sq = 0.0;
        double dot_g_gdiff = 0.0;
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                if (constrain[ia][ic] == 0)
                {
                    r[ia][ic] = 0.0;
                    g[ia][ic] = 0.0;
                    continue;
                }
                r[ia][ic] = target[ia][ic] - M_cur[ia][ic];
                g[ia][ic] = -chi[ia][ic] * r[ia][ic];
                g_norm_sq += g[ia][ic] * g[ia][ic];
                dot_g_gdiff += g[ia][ic] * (g[ia][ic] - g_old[ia][ic]);
            }

        // CG beta (Polak-Ribiere+)
        double beta = 0.0;
        if (k > 0 && g_old_norm_sq > 1e-30)
        {
            beta = std::max(0.0, dot_g_gdiff / g_old_norm_sq);
        }

        // CG direction: d = -g + beta * d_old
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                d[ia][ic] = -g[ia][ic] + beta * d[ia][ic];
            }

        // Analytical step size: alpha = sum(chi*r*d) / sum(chi*d^2)
        double num = 0.0;
        double den = 0.0;
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                if (constrain[ia][ic] == 0) { continue; }
                num += chi[ia][ic] * r[ia][ic] * d[ia][ic];
                den += chi[ia][ic] * d[ia][ic] * d[ia][ic];
            }
        double alpha = (std::abs(den) > 1e-30) ? num / den : 0.0;

        // Clamp: |alpha * d_I| <= sccut for all I
        const double sccut = sc_.get_restrict_current();
        if (sccut > 0)
        {
            double max_step = 0.0;
            for (int ia = 0; ia < nat; ia++)
                for (int ic = 0; ic < 3; ic++)
                {
                    if (constrain[ia][ic] == 0) { continue; }
                    max_step = std::max(max_step, std::abs(alpha * d[ia][ic]));
                }
            if (max_step > sccut)
            {
                alpha *= sccut / max_step;
            }
        }

        // Update lambda
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                if (constrain[ia][ic] == 0)
                {
                    lambda_new[ia][ic] = 0.0;
                }
                else
                {
                    lambda_new[ia][ic] = lambda_cur[ia][ic] + alpha * d[ia][ic];
                }
            }
        sc_.set_lambda(lambda_new);

        // For PW: delta_lambda relative to lambda_0
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
                delta_for_pw[ia][ic] = lambda_new[ia][ic] - lambda_0[ia][ic];

        // Evaluate M(lambda_new) — 1 diag
        sc_.cal_mw_from_lambda(k + 1, delta_for_pw.data());
        auto M_new = sc_.get_Mi();

        // Secant Jacobian update
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
            {
                if (constrain[ia][ic] == 0) { continue; }
                double dlambda = lambda_new[ia][ic] - lambda_cur[ia][ic];
                double dM = M_new[ia][ic] - M_cur[ia][ic];
                if (std::abs(dlambda) > 1e-10)
                {
                    double chi_secant = dM / dlambda;
                    if (chi_secant * chi[ia][ic] > 0
                        && std::abs(chi_secant) > chi_min
                        && std::abs(chi_secant) < chi_max)
                    {
                        chi[ia][ic] = chi_secant;
                    }
                }
            }

        // Save CG state
        g_old = g;
        g_old_norm_sq = g_norm_sq;
        lambda_cur = lambda_new;
        M_cur = M_new;

        // Check convergence
        {
            double sum = 0.0;
            for (int ia = 0; ia < nat; ia++)
                for (int ic = 0; ic < 3; ic++)
                {
                    if (constrain[ia][ic] == 0) { continue; }
                    sum += std::pow(target[ia][ic] - M_new[ia][ic], 2);
                }
            rms_error = std::sqrt(sum / nat);
        }

        auto t_now = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(t_now - t_start).count();

        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- "
                  << std::left << std::setw(5) << k + 1
                  << "       RMS = " << rms_error
                  << "     TIME(s) = " << duration << std::endl;

        if (rms_error < current_sc_thr)
        {
            std::cout << "Meet convergence criterion ( < " << current_sc_thr
                      << " ), exit.       Total TIME(s) = " << duration << std::endl;
            result.converged = true;
            result.n_inner_steps = k + 1;
            break;
        }

        if (k == nsc - 1)
        {
            std::cout << "Reach maximum number of steps ( " << nsc
                      << " ), exit.              Total TIME(s) = " << duration << std::endl;
            result.n_inner_steps = nsc;
        }
    }

    sc_.print_termination();

    // === Final: update charge density ===
    if (PARAM.inp.basis_type == "lcao")
    {
#ifdef __LCAO
        psi::Psi<std::complex<double>>* psi_t
            = static_cast<psi::Psi<std::complex<double>>*>(sc_.psi);
        elecstate::cal_dm_psi(sc_.ParaV, sc_.pelec->wg, *psi_t, *sc_.dm_);
        sc_.dm_->cal_DMR();
        int nspin = PARAM.inp.nspin;
        if (PARAM.inp.nspin == 4) { nspin = 1; }
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            ModuleBase::GlobalFunc::ZEROS(sc_.pelec->charge->rho[is],
                                          sc_.pelec->charge->nrxx);
        }
        ModuleGint::cal_gint_rho(sc_.dm_->get_DMR_vector(), nspin,
                                 sc_.pelec->charge->rho);
        sc_.pelec->charge->renormalize_rho();
#endif
    }
    else
    {
        // PW path: update_psi_charge with total delta from initial
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
                delta_for_pw[ia][ic] = lambda_cur[ia][ic] - lambda_0[ia][ic];
        sc_.update_psi_charge(delta_for_pw.data());

        // PW double-check
        if (PARAM.inp.basis_type == "pw")
        {
            sc_.cal_Mi_pw();
            std::vector<ModuleBase::Vector3<double>> delta_spin(nat, 0.0);
            subtract_2d(sc_.get_Mi(), target, delta_spin);
            where_fill_scalar_2d(constrain, 0, zero, delta_spin);
            double sum = 0.0;
            for (int ia = 0; ia < nat; ia++)
                for (int ic = 0; ic < 3; ic++)
                    sum += std::pow(delta_spin[ia][ic], 2);
            double rms_check = std::sqrt(sum / nat);
            std::cout << "Current RMS: " << rms_check << std::endl;
            if (rms_check > current_sc_thr * 10 && sc_.higher_mag_prec)
            {
                std::cout << "Error: RMS error is too large, rerun the loop" << std::endl;
                result = this->run(outer_step);
                return result;
            }
        }
    }

    result.rms_error = rms_error;
    result.total_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return result;
}

// ===================================================================
// Factory
// ===================================================================

std::unique_ptr<LambdaSolver> create_lambda_solver(
    LambdaSolverType solver_type,
    SpinConstrain<std::complex<double>>& sc)
{
    switch (solver_type)
    {
        case LambdaSolverType::BFGS:
            return std::unique_ptr<LambdaSolver>(new BFGSLambdaSolver(sc));
        case LambdaSolverType::ChiGuided:
            return std::unique_ptr<LambdaSolver>(new ChiGuidedLambdaSolver(sc));
        case LambdaSolverType::Subspace:
            return std::unique_ptr<LambdaSolver>(new SubspaceLambdaSolver(sc));
        case LambdaSolverType::FDCG:
            return std::unique_ptr<LambdaSolver>(new FDCGLambdaSolver(sc));
        default:
            return std::unique_ptr<LambdaSolver>(new FDCGLambdaSolver(sc));
    }
}

} // namespace spinconstrain
