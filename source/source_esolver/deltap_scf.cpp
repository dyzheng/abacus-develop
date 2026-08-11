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
namespace
{
/// Route A+ observable selection: operator mode in proxy-drive returns Γ
/// (state_.gamma_op); operator mode in gamma-drive and legacy gamma mode
/// return the branch-selected γ (state_.gamma_report).  The λ-driving signal
/// (deltap_drive) selects WHICH observable drives λ; the escon accounting
/// always uses Γ in operator mode (see iter_finish).
const std::vector<double>& scf_observable(const deltap_scf::DeltapState& state,
                                          const deltap_scf::DeltapParams& params)
{
    if (params.observable_mode == "operator" && params.drive != "gamma"
        && !state.gamma_op.empty())
        return state.gamma_op;
    return state.gamma_report;
}
/// Route A+ target selection: operator mode in proxy-drive uses the proxy
/// t_Γ (state_.t_proxy); operator mode in gamma-drive and legacy gamma mode
/// use the user's t_γ (params_.target).
const std::vector<double>& scf_target(const deltap_scf::DeltapState& state,
                                      const deltap_scf::DeltapParams& params)
{
    if (params.observable_mode == "operator" && params.drive != "gamma"
        && !state.t_proxy.empty())
        return state.t_proxy;
    return params.target;
}
} // namespace


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
    // Route A+ proxy target: legacy single-fire mode starts t_Γ at the user's
    // t_γ (κ=1 first round, documented).  With the fixed-geometry outer loop
    // enabled (outer_nmax > 0) t_Γ starts EMPTY instead: the first SCF is a
    // free λ=0 run that measures the natural (Γ_nat, γ_nat), and the secant
    // starts from that measured natural point (Q2: t_Γ=t_γ is a κ=1 guess,
    // not a measurement — starting there would pin Γ to γ-scale values,
    // λ*≈O(1) Ry violent-perturbation territory for the first SCF).
    // Gamma-drive (deltap_drive=gamma) retires the t_Γ proxy layer entirely:
    // the λ residual drives γ against the user's t_γ directly, so no proxy
    // target is initialized or loaded (scf_target ignores t_proxy).
    state_.t_proxy = (params_.drive == "gamma" || params_.outer_nmax > 0)
                         ? std::vector<double>()
                         : params_.target;
    // Q3 (T3 protocol): an explicit proxy-target file freezes the calibrated
    // t_Γ* across geometries — the disp± legs then only re-converge λ against
    // the frozen t_Γ (no secant drift, no t_Γ(R) pollution in the FD).
    if (params_.observable_mode == "operator" && params_.drive != "gamma"
        && !params_.proxy_target_file.empty())
    {
        std::vector<double> proxy(params_.nat, 0.0);
        bool loaded = false;
        if (GlobalV::MY_RANK == 0)
        {
            std::ifstream ifs(params_.proxy_target_file);
            if (ifs.is_open())
            {
                loaded = true;
                for (int iat = 0; iat < params_.nat; ++iat)
                    ifs >> proxy[iat];
            }
        }
#ifdef __MPI
        Parallel_Common::bcast_bool(loaded);
        Parallel_Common::bcast_double(proxy.data(), params_.nat);
#endif
        if (loaded)
        {
            state_.t_proxy = proxy;
            if (params_.verbose && GlobalV::MY_RANK == 0)
                std::cout << " [DeltaP] operator mode: t_Γ loaded from "
                          << params_.proxy_target_file << " (frozen)"
                          << std::endl;
        }
        else if (GlobalV::MY_RANK == 0)
        {
            ModuleBase::WARNING_QUIT("DeltapScfSolver::init",
                "DeltaP: cannot open proxy target file "
                + params_.proxy_target_file);
        }
    }
    else if (params_.observable_mode == "operator" && params_.verbose
             && GlobalV::MY_RANK == 0)
    {
        if (params_.drive == "gamma")
            std::cout << " [DeltaP] operator mode: γ-drive (deltap_drive=gamma) — λ "
                      << "driven by the γ residual against t_γ directly; t_Γ/secant "
                      << "translation layer retired (force identity unchanged)"
                      << std::endl;
        else if (params_.outer_nmax > 0)
            std::cout << " [DeltaP] operator mode: fixed-geometry outer loop (nmax="
                      << params_.outer_nmax << ", thr=" << params_.outer_thr
                      << "); first SCF is a free λ=0 run, t_Γ starts at the "
                      << "measured natural Γ" << std::endl;
        else
            std::cout << " [DeltaP] operator mode: t_Γ initialized to t_γ (κ=1 first round)"
                      << std::endl;
    }
    state_.secant_at_conv_done = false;
    state_.initialized = true;
}

void DeltapScfSolver::reset_ionic_step()
{
    // Route A+ outer-loop secant (relax): the previous ionic step's final γ
    // (state_.gamma_report) is still in state here; update the proxy target
    // t_Γ before clearing the per-step history below.  No-op in gamma mode
    // or without a previous measurement (first ionic step).
    secant_update_proxy();
    // New ionic step restarts the SCF loop: allow λ to be re-updated and the
    // inner loop to re-run for the changed geometry.  Cross-iteration 2π
    // branch tracking restarts from an empty history.
    state_.lambda_set = false;
    state_.inner_loop_done = false;
    state_.gamma_prev.clear();
    state_.gamma_report.clear();
    state_.secant_at_conv_done = false;
}

// ---------------------------------------------------------------------------
// inner_loop: frozen-density BFGS/FR-CG optimization of λ (nscf > 0).
// Returns true when the regular HSolver step must be skipped.
// ---------------------------------------------------------------------------
bool DeltapScfSolver::inner_loop(double drho)
{
    if (!state_.initialized || params_.nscf == 0)
        return false;
    // Fixed-geometry outer-loop first pass (proxy-drive only): the first SCF
    // is a free λ=0 run (the secant starts from the measured natural point,
    // not from a guess).  Gamma-drive retires the outer loop — the inner loop
    // drives γ→t_γ directly and must not be blocked by the first-pass gate.
    if (params_.drive != "gamma" && params_.outer_nmax > 0
        && !state_.first_pass_done)
        return false;
    // Gate: activate only when the density is converged (Phase 1 done).
    if (drho > params_.inner_thr)
        return false;
    // Skip if the inner loop already converged in a previous SCF iteration.
    if (state_.inner_loop_done)
        return false;

    // Measure current gamma (and Γ in operator mode) and residual.
    state_.gamma_I = backend_.compute_gamma();
    if (params_.observable_mode == "operator" && backend_.compute_gamma_op)
        state_.gamma_op = backend_.compute_gamma_op();
    // Route A+ γ-drive: the λ-driving observable is the reported γ (γ residual
    // against t_γ directly); proxy-drive uses Γ.  Legacy gamma mode keeps
    // state_.gamma_I (zero regression).
    const std::vector<double>& obs0
        = (params_.observable_mode == "operator" && params_.drive != "gamma"
           && !state_.gamma_op.empty())
              ? state_.gamma_op
              : state_.gamma_I;
    // Route A+ operator mode: the inner-loop residual must target the proxy
    // t_Γ (scf_target), NOT params_.t (constraint-matrix targets, empty in
    // per-atom mode) — the pre-fix form r = Γ − 0 drove Γ→0 instead of
    // Γ→t_Γ* (T3 wiring, 2026-08-05).
    const std::vector<double>& tgt0 = scf_target(state_, params_);
    std::vector<double> residual = use_constraint_matrix()
        ? deltap_common::compute_residual(params_.C, params_.t, obs0)
        : deltap_common::compute_residual({}, tgt0, obs0);

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

        // Measure residual at the trial point (Γ in operator proxy-drive;
        // the reported γ in gamma-drive / legacy gamma mode).
        const std::vector<double> gamma_trial = backend_.compute_gamma();
        const std::vector<double> obs_trial
            = (params_.observable_mode == "operator" && params_.drive != "gamma"
               && backend_.compute_gamma_op)
                  ? backend_.compute_gamma_op()
                  : gamma_trial;
        const std::vector<double>& tgt_trial = scf_target(state_, params_);
        residual = use_constraint_matrix()
            ? deltap_common::compute_residual(params_.C, params_.t, obs_trial)
            : deltap_common::compute_residual({}, tgt_trial, obs_trial);

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
void DeltapScfSolver::iter_finish(int iter, double drho, bool conv_esolver)
{
    if (!state_.initialized || !backend_.compute_gamma)
        return;

    // 1) Measure per-atom gamma (raw).  The synchronous λ update below uses
    //    this measurement, matching the historical PW protocol where the
    //    update residual is built from the pre-update gamma.
    state_.gamma_I = backend_.compute_gamma();
    // Route A+ operator observable: measure Γ at the same wavefunctions.
    // Must happen before update_lambda_gd so the λ update residual can use Γ
    // in operator mode.
    if (params_.observable_mode == "operator" && backend_.compute_gamma_op)
        state_.gamma_op = backend_.compute_gamma_op();

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
    //    Route A+: the λ-driving observable is Γ vs the proxy t_Γ in
    //    proxy-drive and the reported γ vs t_γ in gamma-drive; legacy gamma
    //    mode keeps the γ vs t_γ residual (zero regression).  The escon
    //    accounting ALWAYS uses Γ in operator mode regardless of the driving
    //    signal — escon = −λ·Γ is the Route A+ identity that carries the
    //    force consistency E' = E_KS(ψ*), and it is a property of the
    //    accounting, not of what drives λ (2026-08-11 review, T-4').
    const std::vector<double>& gr = scf_observable(state_, params_);
    const std::vector<double>& tgt = scf_target(state_, params_);
    if (use_constraint_matrix())
        state_.max_res = deltap_common::max_norm(
            deltap_common::compute_residual(params_.C, params_.t, gr));
    else if (!tgt.empty())
        state_.max_res = deltap_common::max_norm(
            deltap_common::compute_residual({}, tgt, gr));
    else
        state_.max_res = 0.0;

    const std::vector<double> lambda
        = backend_.get_lambda ? backend_.get_lambda() : state_.lambda_eff;
    const std::vector<double>& escon_obs
        = (params_.observable_mode == "operator" && !state_.gamma_op.empty())
              ? state_.gamma_op
              : state_.gamma_report;
    state_.dp_escon = deltap_common::compute_dp_escon(lambda, escon_obs);
    if (backend_.apply_hk_correction)
        backend_.apply_hk_correction(lambda);

    // 5) Diagnostics (rank 0 only).
    if (params_.verbose && GlobalV::MY_RANK == 0)
        report(iter, lambda);

    // Route A+ outer-loop secant (single-point, proxy-drive only): one t_Γ
    // update after SCF convergence.  Relax runs do this in reset_ionic_step
    // instead, so the update is never applied twice to the same measurement.
    // Gamma-drive retires the secant/t_Γ layer entirely (the γ residual
    // drives λ directly; the outer loop would have nothing to update).
    if (params_.drive != "gamma" && conv_esolver
        && params_.secant_at_convergence && !state_.secant_at_conv_done)
    {
        // Fixed-geometry outer-loop first pass: the first convergence is the
        // free λ=0 natural run.  Seed the secant history at the measured
        // natural point (t_Γ=Γ_nat) so the first update is the κ=1 guess
        // FROM NATURAL (Q2: t_Γ=t_γ is a guess, not a measurement).
        if (params_.outer_nmax > 0 && !state_.first_pass_done)
        {
            state_.first_pass_done = true;
            state_.t_proxy = state_.gamma_op;
            state_.t_proxy_prev = state_.t_proxy;
            state_.gamma_meas_prev = gamma_report;
            state_.secant_prev_err = -1.0;
            state_.secant_bad_steps = 0;
        }
        secant_update_proxy();
        state_.secant_at_conv_done = true;
        state_.outer_steps++;
        // Fixed-geometry outer loop: if |γ−t_γ|∞ is still above tolerance and
        // the step budget remains, request the SCF loop to continue with the
        // new t_Γ (the H_c change re-disturbs drho, so the next secant fires
        // at the next convergence crossing).  Re-arm the inner λ BFGS so it
        // re-runs for the new proxy target.
        if (params_.outer_nmax > 0 && state_.outer_steps < params_.outer_nmax
            && state_.outer_err > params_.outer_thr)
        {
            state_.outer_redrive = true;
            state_.inner_loop_done = false;
        }
    }
    // Re-arm the single-point secant on re-drive iterations: once the density
    // is disturbed by the new t_Γ (drho rises above scf_thr), the next SCF
    // convergence re-fires the secant for the new measurement.
    else if (params_.outer_nmax > 0 && state_.secant_at_conv_done && !conv_esolver)
    {
        state_.secant_at_conv_done = false;
    }
}

// consume_outer_redrive: called by the ESolver right after iter_finish; if
// the fixed-geometry outer loop requested another SCF pass with the updated
// t_Γ, override conv_esolver so the run continues instead of terminating.
bool DeltapScfSolver::consume_outer_redrive()
{
    const bool redrive = state_.outer_redrive;
    state_.outer_redrive = false;
    return redrive;
}

// ---------------------------------------------------------------------------
// secant_update_proxy: Route A+ outer-loop calibration of the proxy target
// t_Γ so that the measured γ converges to the user's t_γ (D7, derivations §7).
//   t_Γ^(k+1) = t_Γ^(k) + κ^(k)·(t_γ − γ^(k))
//   κ^(0) = 1;  κ^(k≥1) = Δt_Γ/Δγ from the last two (t_Γ, γ) pairs,
//   clamped to [0.3, 20] (sign preserved);  |Δt_Γ| per step clamped to
//   1.0 rad;  two consecutive |γ−t_γ| increases halve κ, and a still-worse
//   step keeps the current t_Γ with a WARNING (relax never aborts).
//   T4 fixes (2026-08-05 review): with the measured γ↔t_Γ slope ≈0.07
//   (κ≈14), the old [0.3, 3] clamp made the "≤5 步" outer-loop prediction
//   structurally impossible — after the first measured pair κ is the secant
//   prediction step; the 1.0 rad step limit and the divergence guard remain
//   as the safety net.
// ---------------------------------------------------------------------------
void DeltapScfSolver::secant_update_proxy()
{
    // Gamma-drive (deltap_drive=gamma) retires the t_Γ/secant translation
    // layer entirely — the γ residual drives λ directly (T-4').
    if (params_.drive == "gamma")
        return;
    // Operator mode only (gamma mode constrains γ directly, no proxy).
    if (params_.observable_mode != "operator")
        return;
    // Q3: secant disabled → t_Γ stays frozen (T3 disp± legs).
    if (!params_.secant_enabled)
        return;
    // Requires an actual Γ measurement path (PW has none) and a measurement
    // from the just-completed SCF (none on the first SCF of a run).
    if (!backend_.compute_gamma_op)
        return;
    if (state_.gamma_report.empty() || state_.t_proxy.empty())
        return;
    if (params_.target.empty())
        return; // no physical t_γ to converge toward

    const std::vector<double>& t_gamma = params_.target;
    const std::vector<double>& gamma_meas = state_.gamma_report;
    const int nat = params_.nat;

    // κ: secant slope from the previous (t_Γ, γ) pair; first round κ=1.
    std::vector<double> kappa(nat, 1.0);
    if (state_.gamma_meas_prev.size() == static_cast<size_t>(nat)
        && state_.t_proxy_prev.size() == static_cast<size_t>(nat))
    {
        for (int iat = 0; iat < nat; ++iat)
        {
            const double dg = gamma_meas[iat] - state_.gamma_meas_prev[iat];
            const double dt = state_.t_proxy[iat] - state_.t_proxy_prev[iat];
            if (std::abs(dg) > 1e-12)
            {
                kappa[iat] = dt / dg;
                // Q2 (D7): κ = Δt_Γ/Δγ carries the SIGN of the measured
                // γ↔t_Γ response (γ ≈ −Γ + c gives κ < 0).  Clamp the
                // magnitude to [0.3, 3] but preserve the sign — a wrong sign
                // shows up as |γ−t_γ| increasing, and the next secant step
                // then flips κ automatically via the measured slope.
                const double abs_k = std::abs(kappa[iat]);
                if (abs_k < 0.3) kappa[iat] = (kappa[iat] >= 0.0) ? 0.3 : -0.3;
                else if (abs_k > 20.0) kappa[iat] = (kappa[iat] >= 0.0) ? 20.0 : -20.0;
            }
        }
    }

    // Divergence guard: two consecutive |γ−t_γ| increases halve κ; a third
    // consecutive increase keeps t_Γ (WARNING, relax continues unconstrained).
    double err = 0.0;
    for (int iat = 0; iat < nat; ++iat)
        err = std::max(err, std::abs(gamma_meas[iat] - t_gamma[iat]));
    // Fixed-geometry outer-loop convergence: last |γ−t_γ|∞ (consumed by the
    // re-drive decision in iter_finish).
    state_.outer_err = err;
    if (state_.secant_prev_err > 0.0 && err > state_.secant_prev_err)
    {
        state_.secant_bad_steps++;
        if (state_.secant_bad_steps == 2)
        {
            for (int iat = 0; iat < nat; ++iat)
                kappa[iat] *= 0.5;
        }
        else if (state_.secant_bad_steps >= 3)
        {
            if (GlobalV::MY_RANK == 0)
                std::cout << " [DeltaP] WARNING: outer-loop secant diverging "
                          << "(|γ−t_γ|∞=" << err << " > prev="
                          << state_.secant_prev_err << " for "
                          << state_.secant_bad_steps
                          << " steps); keeping t_Γ for this step" << std::endl;
            state_.secant_prev_err = err;
            return;
        }
    }
    else
    {
        state_.secant_bad_steps = 0;
    }

    // Store history, then apply the clamped update.
    state_.t_proxy_prev = state_.t_proxy;
    state_.gamma_meas_prev = gamma_meas;
    for (int iat = 0; iat < nat; ++iat)
    {
        double dt = kappa[iat] * (t_gamma[iat] - gamma_meas[iat]);
        if (dt > 1.0) dt = 1.0;
        if (dt < -1.0) dt = -1.0;
        state_.t_proxy[iat] += dt;
    }
    state_.secant_prev_err = err;

    if (params_.verbose && GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP Secant] |γ−t_γ|∞=" << std::scientific
                  << std::setprecision(3) << err << " κ=(" << std::fixed
                  << std::setprecision(2);
        for (int iat = 0; iat < nat; ++iat)
        {
            if (iat > 0) std::cout << ", ";
            std::cout << kappa[iat];
        }
        std::cout << ") γ=(" << std::setprecision(3);
        for (int iat = 0; iat < nat; ++iat)
        {
            if (iat > 0) std::cout << ", ";
            std::cout << gamma_meas[iat];
        }
        std::cout << ") t_Γ=(" << std::setprecision(3);
        for (int iat = 0; iat < nat; ++iat)
        {
            if (iat > 0) std::cout << ", ";
            std::cout << state_.t_proxy[iat];
        }
        std::cout << ")" << std::endl;
    }
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
    // Fixed-geometry outer-loop first pass (proxy-drive only): keep λ=0 (free
    // natural run).  Gamma-drive retires the outer loop — the synchronous λ
    // update drives γ→t_γ directly.
    if (params_.drive != "gamma" && params_.outer_nmax > 0
        && !state_.first_pass_done)
        return;
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

    // Route A+: the λ-update residual uses the same observable/target as the
    // convergence residual — Γ vs t_Γ in operator proxy-drive, the reported γ
    // vs t_γ in operator gamma-drive, γ vs t_γ in legacy gamma mode (which
    // keeps state_.gamma_I — the historical signal — for zero regression).
    const std::vector<double>& obs
        = (params_.observable_mode == "operator" && params_.drive != "gamma"
           && !state_.gamma_op.empty())
              ? state_.gamma_op
              : state_.gamma_I;
    const std::vector<double>& tgt = scf_target(state_, params_);
    if (use_constraint_matrix())
    {
        // Constraint-space update: r[α] = Σ_i C[α][i]·γ_i − t[α], then
        // λ_cstr ← mixing·(λ_cstr + step·r) + (1−mixing)·λ_cstr, and finally
        // λ_eff[i] = Σ_a λ_cstr[a]·C[a][i].
        const std::vector<double> residual = deltap_common::compute_residual(
            params_.C, params_.t, obs);
        deltap_common::gd_update(state_.lambda_cstr, residual, {}, step, mixing);
        lambda = deltap_common::to_effective_lambda(
            state_.lambda_cstr, params_.C, params_.nat);
    }
    else if (!tgt.empty())
    {
        const std::vector<double> residual = deltap_common::compute_residual(
            {}, tgt, obs);
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
    // Route A+ operator column: measured Γ at the same wavefunctions as γ.
    if (params_.observable_mode == "operator" && !state_.gamma_op.empty())
    {
        std::cout << " Γ=(" << std::fixed << std::setprecision(3);
        for (int iat = 0; iat < params_.nat; ++iat)
        {
            if (iat > 0)
                std::cout << ", ";
            std::cout << state_.gamma_op[iat];
        }
        std::cout << ")";
    }
    std::cout << " |γ-t|=" << std::scientific << std::setprecision(3) << state_.max_res
              << " escon=" << std::fixed << std::setprecision(6) << state_.dp_escon
              << " Ry\n";

    // Effective electric field.  Gamma mode: legacy λ–γ conjugate formula
    // E_eff = −λ_avg·π/(2·a_alpha) a.u. (retired for operator mode, see
    // derivations §1.1).  Operator mode (Route A+): the constraint operator
    // IS a ramp potential, E_eff = λ_avg/(2·a_alpha) a.u. with no π
    // (derivations §1.2; sign/factor pinned by V1, efield comparison).
    // λ is in Ry; converting Ry → Hartree gives an extra factor of 1/2.
    // Convert: 1 a.u. = 51.422 V/Å.
    if (backend_.lattice_period)
    {
        const double a_alpha = backend_.lattice_period();
        double lam_avg = 0.0;
        for (int iat = 0; iat < params_.nat; ++iat)
            lam_avg += lambda[iat];
        lam_avg /= params_.nat;
        const bool op_mode = (params_.observable_mode == "operator");
        const double e_eff_au = op_mode ? lam_avg / (2.0 * a_alpha)
                                        : -lam_avg * ModuleBase::PI / (2.0 * a_alpha);
        const double e_eff_v_per_a = e_eff_au * 51.422;
        std::cout << "   [E-field" << (op_mode ? " operator-ramp" : "") << "] E_eff="
                  << std::scientific << std::setprecision(3)
                  << e_eff_v_per_a << " V/Angstrom  (λ_avg=" << lam_avg << " Ry)"
                  << std::endl;
    }
}

} // namespace deltap_scf
