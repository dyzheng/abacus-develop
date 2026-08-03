#include "deltap_scf.h"
#include "deltap_common.h"
#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_common.h"
#include "module_optimizer/bfgs.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace deltap_scf
{

// ---------------------------------------------------------------------------
// init: snapshot params, load target / constraint-matrix files, seed state.
// ---------------------------------------------------------------------------
void DeltapScfSolver::init(const DeltapParams& p, Backend b)
{
    params_ = p;
    backend_ = std::move(b);
    state_ = DeltapState();

    // Load per-atom gamma targets from file (overrides STRU values).
    // Rank 0 reads the file, then the parsed data is broadcast so every MPI
    // rank sees identical targets (C-23).  A missing/unreadable file keeps
    // the STRU-derived targets, matching the historical behavior.
    if (!params_.target_file.empty())
    {
        bool loaded = false;
        const bool total_mode = params_.total_mode;
        double total_target = 0.0;
        std::vector<double> file_target(params_.nat, 0.0);
        if (GlobalV::MY_RANK == 0)
        {
            std::ifstream ifs(params_.target_file);
            if (ifs.is_open())
            {
                loaded = true;
                if (total_mode)
                {
                    ifs >> total_target;
                }
                else
                {
                    for (int iat = 0; iat < params_.nat; ++iat)
                        ifs >> file_target[iat];
                }
            }
        }
#ifdef __MPI
        Parallel_Common::bcast_bool(loaded);
        Parallel_Common::bcast_double(total_target);
        Parallel_Common::bcast_double(file_target.data(), params_.nat);
#endif
        if (loaded)
        {
            if (total_mode)
            {
                params_.target.assign(params_.nat, total_target / params_.nat);
                if (params_.verbose && GlobalV::MY_RANK == 0)
                    std::cout << " [DeltaP] Loaded total target Σγ=" << total_target
                              << " → per-atom=" << total_target / params_.nat << std::endl;
            }
            else
            {
                params_.target = file_target;
                if (params_.verbose && GlobalV::MY_RANK == 0)
                    std::cout << " [DeltaP] Loaded target from " << params_.target_file << std::endl;
            }
        }
    }

    // Load linear constraint matrix C·γ = t (overrides constraint_mode when set).
    // Rank 0 reads the file; dimensions and entries are broadcast.  A size
    // mismatch is a configuration error and quits instead of silently
    // dropping the constraint.
    if (!params_.constraint_matrix_file.empty())
    {
        bool file_open = false;
        bool loaded = false;
        int m = 0, n = 0;
        std::vector<double> c_data, t_data;
        if (GlobalV::MY_RANK == 0)
        {
            std::ifstream ifs(params_.constraint_matrix_file);
            if (ifs.is_open())
            {
                file_open = true;
                ifs >> m >> n;
                if (n == params_.nat && m > 0)
                {
                    c_data.resize(static_cast<size_t>(m) * n, 0.0);
                    t_data.assign(m, 0.0);
                    for (int a = 0; a < m; ++a)
                    {
                        for (int i = 0; i < n; ++i)
                            ifs >> c_data[static_cast<size_t>(a) * n + i];
                        ifs >> t_data[a];
                    }
                    loaded = true;
                }
            }
        }
#ifdef __MPI
        Parallel_Common::bcast_bool(file_open);
        Parallel_Common::bcast_bool(loaded);
        Parallel_Common::bcast_int(m);
        Parallel_Common::bcast_int(n);
        if (loaded)
        {
            Parallel_Common::bcast_double(c_data.data(), m * n);
            Parallel_Common::bcast_double(t_data.data(), m);
        }
#endif
        if (loaded)
        {
            params_.C.assign(m, std::vector<double>(n, 0.0));
            for (int a = 0; a < m; ++a)
                for (int i = 0; i < n; ++i)
                    params_.C[a][i] = c_data[static_cast<size_t>(a) * n + i];
            params_.t = t_data;
            state_.lambda_cstr.assign(m, params_.lambda_init);
            if (params_.verbose && GlobalV::MY_RANK == 0)
                std::cout << " [DeltaP] Loaded constraint matrix " << m << "x" << n
                          << " from " << params_.constraint_matrix_file << std::endl;
        }
        else if (file_open && GlobalV::MY_RANK == 0)
        {
            ModuleBase::WARNING_QUIT("DeltapScfSolver::init",
                "DeltaP: constraint matrix size mismatch (expected "
                + std::to_string(params_.nat) + " columns, got " + std::to_string(n) + ")");
        }
    }

    // Constraint-space lambda: [m] in matrix mode, per-atom zeros otherwise.
    if (state_.lambda_cstr.empty())
    {
        state_.lambda_cstr.assign(use_constraint_matrix() ? static_cast<int>(params_.C.size())
                                                          : params_.nat,
                                  0.0);
    }
    state_.initialized = true;
}

void DeltapScfSolver::reset_ionic_step()
{
    // New ionic step restarts the SCF loop: allow λ to be re-updated and the
    // inner loop to re-run for the changed geometry.  Cross-iteration 2π
    // branch tracking restarts from an empty history.
    state_.lambda_set = false;
    state_.inner_loop_done = false;
    state_.gamma_prev.clear();
    state_.gamma_report.clear();
}

// ---------------------------------------------------------------------------
// inner_loop: frozen-density BFGS/FR-CG optimization of λ (nscf > 0).
// Returns true when the regular HSolver step must be skipped.
// ---------------------------------------------------------------------------
bool DeltapScfSolver::inner_loop(double drho)
{
    if (!state_.initialized || params_.nscf == 0)
        return false;
    // Gate: activate only when the density is converged (Phase 1 done).
    if (drho > params_.inner_thr)
        return false;
    // Skip if the inner loop already converged in a previous SCF iteration.
    if (state_.inner_loop_done)
        return false;

    // Measure current gamma and residual.
    state_.gamma_I = backend_.compute_gamma();
    std::vector<double> residual = deltap_common::compute_residual(
        params_.C, params_.t, state_.gamma_I);

    const int n_inner = use_constraint_matrix()
                            ? static_cast<int>(params_.C.size())
                            : params_.nat;
    auto& bfgs = backend_.get_optimizer();
    bfgs.init(n_inner, 0.5, params_.conv_thr, 2, 0.01, 0.005);

    // Initial λ: constraint-space (matrix mode) or operator λ (per-atom mode).
    std::vector<double> lambda_inner
        = use_constraint_matrix() ? state_.lambda_cstr : backend_.get_lambda();
    bfgs.start_outer(lambda_inner);

    bool bfgs_converged = false;
    const int nscf = params_.nscf;

    if (params_.verbose && GlobalV::MY_RANK == 0)
        std::cout << " [DeltaP] inner loop start: nscf=" << nscf
                  << " rms=" << std::scientific << std::setprecision(4)
                  << bfgs.get_rms() << std::endl;

    for (int inner = 0; inner < nscf && !bfgs_converged; ++inner)
    {
        std::vector<double> lam_trial = lambda_inner;
        bfgs.step(residual, inner, lam_trial, bfgs_converged);
        if (bfgs_converged)
            break;

        // Apply trial lambda (convert to effective per-atom lambda).
        const std::vector<double> lam_eff
            = deltap_common::to_effective_lambda(lam_trial, params_.C, params_.nat);
        apply_lambda(lam_eff);
        if (backend_.apply_hk_correction)
            backend_.apply_hk_correction(lam_eff);

        // Re-solve with trial lambda (charge density frozen).
        backend_.solve_frozen();

        // Measure residual at the trial point.
        const std::vector<double> gamma_trial = backend_.compute_gamma();
        residual = deltap_common::compute_residual(params_.C, params_.t, gamma_trial);

        const double alpha_opt = bfgs.accept_trial(residual);
        lambda_inner = lam_trial;

        if (params_.verbose && GlobalV::MY_RANK == 0)
            std::cout << " [DeltaP]   inner=" << inner
                      << " rms=" << std::scientific << std::setprecision(4)
                      << bfgs.get_rms() << " alpha_opt=" << alpha_opt << std::endl;
    }

    // Set final lambda and reconstruct the HK correction.
    const std::vector<double> lam_final
        = deltap_common::to_effective_lambda(lambda_inner, params_.C, params_.nat);
    if (use_constraint_matrix())
        state_.lambda_cstr = lambda_inner;
    apply_lambda(lam_final);
    if (backend_.apply_hk_correction)
        backend_.apply_hk_correction(lam_final);
    state_.inner_loop_done = true;

    if (params_.verbose && GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP] inner loop done: final";
        for (int iat = 0; iat < params_.nat; ++iat)
            std::cout << " l" << iat << "=" << lam_final[iat];
        std::cout << std::endl;
    }
    return true; // inner loop already solved; skip the regular HSolver step.
}

// ---------------------------------------------------------------------------
// apply_lambda: single sync point for every λ update (synchronous P2 update
// and inner-loop trials).  set_lambda applies the local value immediately;
// sync_lambda then broadcasts rank 0's λ and the backend writes it back into
// the operator, so all ranks keep the same operator λ / escon (D2).
// ---------------------------------------------------------------------------
void DeltapScfSolver::apply_lambda(const std::vector<double>& lambda)
{
    // Persist the effective per-atom λ in the state machine.  The LCAO
    // backend re-creates its operator at every ionic step (before_scf
    // rebuilds p_hamilt); the persisted value seeds the fresh operator so
    // the λ trajectory is not lost between steps (B-3).
    state_.lambda_eff = lambda;
    backend_.set_lambda(lambda);
    if (backend_.sync_lambda)
    {
        // sync_lambda may broadcast in place; hand it a mutable copy so the
        // caller's const vector is never modified.
        std::vector<double> lam = lambda;
        backend_.sync_lambda(lam);
    }
}

// ---------------------------------------------------------------------------
// iter_finish: per-SCF-iteration constraint update (synchronous mode).
// ---------------------------------------------------------------------------
void DeltapScfSolver::iter_finish(int iter, double drho)
{
    if (!state_.initialized || !backend_.compute_gamma)
        return;

    // 1) Measure per-atom gamma (raw).  The synchronous λ update below uses
    //    this measurement, matching the historical PW protocol where the
    //    update residual is built from the pre-update gamma.
    state_.gamma_I = backend_.compute_gamma();

    // 2) Synchronous lambda update (two-phase mode, nscf == 0).
    if (params_.nscf == 0)
        update_lambda_gd(iter, drho);

    // 3) Branch-selected gamma for residual / escon / report:
    //    - LCAO: target-aware branch selection already happened inside
    //      compute_gamma (module_deltap), so gamma_report == gamma_I.
    //    - PW: nearest-previous 2π unwrap across SCF steps.
    std::vector<double> gamma_report = state_.gamma_I;
    if (params_.unwrap_branch_2pi)
    {
        gamma_report = deltap_common::unwrap_2pi(gamma_report, state_.gamma_prev);
        state_.gamma_prev = gamma_report;
    }
    state_.gamma_report = std::move(gamma_report);

    // 4) Residual / constraint-energy correction with the operator's λ.
    const std::vector<double>& gr = state_.gamma_report;
    if (use_constraint_matrix())
        state_.max_res = deltap_common::max_norm(
            deltap_common::compute_residual(params_.C, params_.t, gr));
    else if (!params_.target.empty())
        state_.max_res = deltap_common::max_norm(
            deltap_common::compute_residual({}, params_.target, gr));
    else
        state_.max_res = 0.0;

    const std::vector<double> lambda
        = backend_.get_lambda ? backend_.get_lambda() : state_.lambda_eff;
    state_.dp_escon = deltap_common::compute_dp_escon(lambda, gr);
    if (backend_.apply_hk_correction)
        backend_.apply_hk_correction(lambda);

    // 5) Diagnostics (rank 0 only).
    if (params_.verbose && GlobalV::MY_RANK == 0)
        report(iter, lambda);
}

// ---------------------------------------------------------------------------
// update_lambda_gd: Phase-2 gradient-descent λ update, then freeze it.
// ---------------------------------------------------------------------------
void DeltapScfSolver::update_lambda_gd(int iter, double drho)
{
    // Branch D: two-phase strategy (nscf == 0).
    // Phase 1: SCF converges with λ=0.  Gamma is computed (target-aware
    // branch selection runs) but λ remains zero — updating λ against
    // the crude atomic-guess charge density (drho~0.5) produces wrong λ.
    // Phase 2: once drho drops below the threshold, take a single
    // gradient-descent λ update, then freeze λ for all remaining iterations.
    if (state_.lambda_set || !(drho > 0.0 && drho < params_.inner_thr))
        return;
    state_.lambda_set = true;

    std::vector<double> lambda
        = backend_.get_lambda ? backend_.get_lambda() : state_.lambda_eff;
    const double step = params_.lambda_step;
    double mixing = params_.lambda_mixing;
    if (mixing < 0.0)
        mixing = 0.0;
    if (mixing > 1.0)
        mixing = 1.0;

    if (use_constraint_matrix())
    {
        // Constraint-space update: r[α] = Σ_i C[α][i]·γ_i − t[α], then
        // λ_cstr ← mixing·(λ_cstr + step·r) + (1−mixing)·λ_cstr, and finally
        // λ_eff[i] = Σ_a λ_cstr[a]·C[a][i].
        const std::vector<double> residual = deltap_common::compute_residual(
            params_.C, params_.t, state_.gamma_I);
        deltap_common::gd_update(state_.lambda_cstr, residual, {}, step, mixing);
        lambda = deltap_common::to_effective_lambda(
            state_.lambda_cstr, params_.C, params_.nat);
    }
    else if (!params_.target.empty())
    {
        const std::vector<double> residual = deltap_common::compute_residual(
            {}, params_.target, state_.gamma_I);
        if (params_.total_mode)
            deltap_common::gd_update_total(lambda, residual, step, mixing);
        else
            deltap_common::gd_update(lambda, residual, params_.constrain, step, mixing);
    }

    apply_lambda(lambda);
    if (backend_.on_phase2)
        backend_.on_phase2();

    if (params_.verbose && GlobalV::MY_RANK == 0)
        std::cout << " [DeltaP P2] iter=" << iter << " drho=" << std::scientific
                  << std::setprecision(2) << drho << " < " << params_.inner_thr
                  << " → λ updated, mix_reset()\n";
}

// ---------------------------------------------------------------------------
// report: [rawG] / [DeltaP P1|P3] / [E-field] diagnostics (rank 0 only).
// ---------------------------------------------------------------------------
void DeltapScfSolver::report(int iter, const std::vector<double>& lambda) const
{
    // Raw (pre-branch) gamma diagnostic for Born effective charge.
    if (backend_.compute_gamma_raw)
    {
        const std::vector<double> raw = backend_.compute_gamma_raw();
        if (!raw.empty())
        {
            double sg_raw = 0.0;
            for (double v : raw)
                sg_raw += v;
            std::cout << "  [rawG] Σγ_raw=" << sg_raw;
            for (int iat = 0; iat < static_cast<int>(raw.size()); ++iat)
                std::cout << " γ" << iat << "=" << raw[iat];
            std::cout << std::endl;
        }
    }

    const std::string phase = state_.lambda_set ? "P3" : "P1";
    std::cout << " [DeltaP " << phase << "] iter=" << std::setw(3) << iter
              << " γ=(" << std::fixed << std::setprecision(3);

    if (params_.total_mode)
    {
        double total_g = 0.0;
        for (int iat = 0; iat < params_.nat; ++iat)
            total_g += state_.gamma_report[iat];
        std::cout << total_g << ") Σγ=" << total_g;
        // Show the single shared λ (6 sig figs: FD group-1 needs to freeze
        // the exact base-run λ).
        std::cout << " λ=";
        if (std::abs(lambda[0]) < 1e-10)
            std::cout << std::scientific << std::setprecision(1) << lambda[0];
        else
            std::cout << std::scientific << std::setprecision(6) << lambda[0];
    }
    else
    {
        for (int iat = 0; iat < params_.nat; ++iat)
        {
            if (iat > 0)
                std::cout << ", ";
            std::cout << state_.gamma_report[iat];
        }
        std::cout << ") λ=(";
        for (int iat = 0; iat < params_.nat; ++iat)
        {
            if (iat > 0)
                std::cout << ", ";
            if (std::abs(lambda[iat]) < 1e-10)
                std::cout << std::scientific << std::setprecision(1) << lambda[iat];
            else
                std::cout << std::scientific << std::setprecision(6) << lambda[iat];
        }
        std::cout << ")";
    }
    std::cout << " |γ-t|=" << std::scientific << std::setprecision(3) << state_.max_res
              << " escon=" << std::fixed << std::setprecision(6) << state_.dp_escon
              << " Ry\n";

    // Effective electric field: E_eff = -λ_avg × π / (2·a_alpha) (a.u.).
    // λ is in Ry; converting Ry → Hartree gives an extra factor of 1/2.
    // Convert: 1 a.u. = 51.422 V/Å.
    if (backend_.lattice_period)
    {
        const double a_alpha = backend_.lattice_period();
        double lam_avg = 0.0;
        for (int iat = 0; iat < params_.nat; ++iat)
            lam_avg += lambda[iat];
        lam_avg /= params_.nat;
        const double e_eff_au = -lam_avg * ModuleBase::PI / (2.0 * a_alpha);
        const double e_eff_v_per_a = e_eff_au * 51.422;
        std::cout << "   [E-field] E_eff=" << std::scientific << std::setprecision(3)
                  << e_eff_v_per_a << " V/Angstrom  (λ_avg=" << lam_avg << " Ry)"
                  << std::endl;
    }
}

} // namespace deltap_scf
