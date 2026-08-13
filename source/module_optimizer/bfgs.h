#ifndef FLETCHER_REEVES_CG_H
#define FLETCHER_REEVES_CG_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace ModuleOptimizer {

/**
 * @brief Fletcher-Reeves conjugate gradient optimizer for vector-valued residual.
 *
 * Minimizes a vector-valued residual function r(λ) = f(λ) - target,
 * where f: R^n → R^n is the physical observable (e.g. magnetic moments or
 * Wannier polarization per atom), and target ∈ R^n is the constraint target.
 *
 * The optimizer uses a nested iteration:
 *   - At each step, it proposes a trial λ and expects the caller to
 *     evaluate r(λ_trial) by re-diagonalizing the Hamiltonian.
 *   - It then uses linear interpolation (secant method) to estimate the
 *     optimal step size and updates λ.
 *
 * ## Optimization algorithm (Fletcher-Reeves CG):
 *
 *   r_k        = f(λ_k) - target                  residual at step k
 *   |r_k|²     = Σ_i r_k[i]²                      squared L₂ norm
 *
 *   β_k        = |r_k|² / |r_{k-1}|²              Fletcher-Reeves β
 *   d_k        = r_k + β_k · d_{k-1}              conjugate search direction
 *
 * ## Line search (linear interpolation / secant method):
 *
 *   sum_k      = Σ_i (target[i] - r_cur[i]) · (r_trial[i] - r_cur[i])
 *   sum_k2     = Σ_i |r_cur[i] - r_trial[i]|²
 *   α_opt      = α_trial · sum_k / sum_k2          optimal step size
 *
 *   λ_new      = λ_cur + α_opt · d_k               update λ
 *
 * ## Adaptive step size:
 *
 *   γ          = 1.0 · |α_opt| / α_trial                   neutral adaptation
 *   γ          = clip(γ, 0.5, 2.0)                        limit adaptation rate
 *   α_trial'   = α_trial · γ^{0.7}                 update for next step
 *
 * ## Step restriction:
 *
 *   boundary   = |α_trial| · max_i |d_k[i]|
 *   if boundary > max_step:
 *       α_trial = sign(α_trial) · max_step / max_i |d_k[i]|
 *
 * This caps the maximum change in any component to max_step, preventing
 * the line search from taking unrealistically large steps.
 *
 * ## Convergence:
 *
 *   rms_k      = sqrt(|r_k|² / n_dim)
 *   if rms_k < conv_thr: converged
 *
 * ## Gradient decay check (early termination):
 *
 *   dλ         = α_k · d_k                          effective λ change
 *   dM_eff      = |r_{k-1} - r_k| · min(1, max_steps/step)
 *   if dM_eff < decay_grad · |r_k|: gradient too flat → stop
 *
 * ## Usage pattern:
 *
 *   FletcherReevesCG opt;
 *   opt.init(n_dim, alpha_init, conv_thr, nsc_min, decay_grad, max_step);
 *
 *   for (int outer = 0; outer < max_outer; ++outer) {
 *       opt.start_outer();
 *       for (int inner = 0; inner < nsc_max; ++inner) {
 *           // 1. Get residual from caller
 *           std::vector<double> residual = compute_residual(lambda);
 *
 *           // 2. Feed to optimizer
 *           bool converged;
 *           opt.step(residual, inner, (out) lambda, (out) converged);
 *           if (converged) break;
 *
 *           // 3. Apply new lambda and evaluate
 *           apply_lambda(lambda);
 *       }
 *   }
 */
class FletcherReevesCG
{
  public:
    FletcherReevesCG() = default;

    /**
     * @brief Initialize the optimizer.
     * @param n_dim       number of constrained components (e.g. nat for DeltaP)
     * @param alpha_init  initial trial step size (physical units)
     * @param conv_thr    convergence threshold on RMS residual
     * @param nsc_min     minimum number of inner steps before checking gradient decay
     * @param decay_grad  threshold for gradient decay early exit
     * @param max_step    maximum allowed change in any component per step
     */
    void init(int n_dim, double alpha_init, double conv_thr, int nsc_min,
              double decay_grad, double max_step)
    {
        n_dim_ = n_dim;
        alpha_trial_ = alpha_init;
        conv_thr_ = conv_thr;
        nsc_min_ = nsc_min;
        decay_grad_ = decay_grad;
        max_step_ = max_step;

        initial_lambda_.assign(n_dim, 0.0);
        delta_lambda_.assign(n_dim, 0.0);
        dnu_.assign(n_dim, 0.0);
        dnu_last_.assign(n_dim, 0.0);
        search_.assign(n_dim, 0.0);
        search_old_.assign(n_dim, 0.0);
        residual_.assign(n_dim, 0.0);
        residual_old_.assign(n_dim, 0.0);
        lambda_.assign(n_dim, 0.0);
        // T-7' componentwise mode: per-component trial steps.
        if (componentwise_)
        {
            alpha_trial_vec_.assign(n_dim, alpha_init);
            current_trial_alpha_vec_.assign(n_dim, alpha_init);
        }
    }

    /**
     * @brief Enable the per-component secant (diagonal-Jacobian) mode.
     *
     * T-7' (2026-08-13): the scalar-α line search mixes opposite-sign
     * per-atom residual components (O wants λ up, H wants λ down), so the
     * single α_opt = α_trial·Σ(−r_i·Δr_i)/ΣΔr_i² can flip sign or collapse
     * (T-2 root cause ②, T3' a2: α −0.249→+0.197→−0.139).  In componentwise
     * mode each component i gets its own secant step
     *   α_opt[i] = α_trial[i]·(−r_i·Δr_i)/(Δr_i² + ε)
     * with a per-component max-step clamp — the diagonal-Jacobian
     * approximation J_ii = Δr_i/Δλ_i, no cross-component mixing.  Both
     * drive modes (proxy Γ / gamma γ) benefit.  Call after init().
     */
    void set_componentwise(bool v)
    {
        componentwise_ = v;
        if (componentwise_ && n_dim_ > 0)
        {
            alpha_trial_vec_.assign(n_dim_, alpha_trial_);
            current_trial_alpha_vec_.assign(n_dim_, alpha_trial_);
        }
    }

    bool componentwise() const { return componentwise_; }

    /**
     * @brief Start a new outer SCF iteration.
     *
     * Resets the conjugate gradient history. Call once per outer SCF step
     * before the inner loop. The initial_lambda is set to the current lambda
     * (passed in) so the inner loop optimizes from this reference point.
     *
     * @param initial_lam  current λ values (n_dim), will be saved as reference
     */
    void start_outer(const std::vector<double>& initial_lam)
    {
        initial_lambda_ = initial_lam;
        std::fill(dnu_.begin(), dnu_.end(), 0.0);
        std::fill(dnu_last_.begin(), dnu_last_.end(), 0.0);
        std::fill(search_old_.begin(), search_old_.end(), 0.0);
        step_count_ = 0;
    }

    /**
     * @brief Perform one inner optimization step.
     *
     * Caller must provide the current residual r = f(λ) - target.
     * The optimizer computes:
     *   1. RMS error and convergence check
     *   2. Search direction (steepest descent or conjugate gradient)
     *   3. Step restriction
     *   4. Updated lambda = initial_lambda + dnu
     *   5. Gradient decay check
     *
     * After this call, the caller should apply lambda and re-evaluate residual.
     *
     * @param[in]  residual_in  current residual r = f(λ) - target  (n_dim)
     * @param[in]  step         inner step index (0-based)
     * @param[out] lambda_out   total λ = initial_lambda_ + dnu_  (n_dim)
     * @param[out] converged    true if converged or gradient decayed
     */
    void step(const std::vector<double>& residual_in, int step,
              std::vector<double>& lambda_out, bool& converged)
    {
        converged = false;
        residual_ = residual_in;
        step_count_ = step + 1; // track for gradient decay check

        // ---- RMS error computation ----
        // rms = sqrt(Σ r[i]² / n_dim)
        double rms_val = 0.0;
        for (int i = 0; i < n_dim_; ++i)
            rms_val += residual_[i] * residual_[i];
        rms_val = std::sqrt(rms_val / n_dim_);
        latest_rms_ = rms_val;

        // ---- Convergence check ----
        if (rms_val < conv_thr_)
        {
            converged = true;
            return;
        }

        // ---- Search direction ----
        // T-7' componentwise mode: per-component steepest descent
        // s[i] = −r[i]; each component carries its own step α_trial_vec_[i].
        if (componentwise_)
        {
            for (int i = 0; i < n_dim_; ++i)
                search_[i] = -residual_[i];
            residual_old_ = residual_;
            search_old_ = search_;
            dnu_last_ = dnu_;
            for (int i = 0; i < n_dim_; ++i)
            {
                // Per-component step restriction: |α[i]·s[i]| ≤ max_step.
                if (max_step_ > 0.0 && std::abs(search_[i]) > 1e-30
                    && std::abs(alpha_trial_vec_[i] * search_[i]) > max_step_)
                {
                    alpha_trial_vec_[i]
                        = (alpha_trial_vec_[i] > 0 ? 1.0 : -1.0) * max_step_
                          / std::abs(search_[i]);
                }
                dnu_[i] += alpha_trial_vec_[i] * search_[i];
            }
            for (int i = 0; i < n_dim_; ++i)
                lambda_out[i] = initial_lambda_[i] + dnu_[i];
            current_trial_alpha_vec_ = alpha_trial_vec_;
            current_search_ = search_;
            current_trial_alpha_ = 0.0;
            for (int i = 0; i < n_dim_; ++i)
                current_trial_alpha_ = std::max(current_trial_alpha_,
                                                std::abs(alpha_trial_vec_[i]));
            return;
        }

        // ---- Search direction ----
        // d_k = r_k + β_k · d_{k-1}   (Fletcher-Reeves CG)
        // β_k = |r_k|² / |r_{k-1}|²
        search_ = residual_;
        if (step >= 2)  // need at least 2 steps for CG history
        {
            double r2_new = 0.0, r2_old = 0.0;
            for (int i = 0; i < n_dim_; ++i)
            {
                r2_new += residual_[i] * residual_[i];
                r2_old += residual_old_[i] * residual_old_[i];
            }
            double beta = (r2_old > 1e-30) ? (r2_new / r2_old) : 0.0;
            for (int i = 0; i < n_dim_; ++i)
                search_[i] = residual_[i] + beta * search_old_[i];
        }
        residual_old_ = residual_;
        search_old_ = search_;

        // ---- Step restriction ----
        // cap |α · max(search)| ≤ max_step
        check_restriction(search_, alpha_trial_);

        // ---- Cumulative step: dnu += α_trial · search ----
        dnu_last_ = dnu_;
        for (int i = 0; i < n_dim_; ++i)
            dnu_[i] += alpha_trial_ * search_[i];

        // ---- Output lambda = initial_lambda + dnu ----
        for (int i = 0; i < n_dim_; ++i)
            lambda_out[i] = initial_lambda_[i] + dnu_[i];
        current_trial_alpha_ = alpha_trial_;
        current_search_ = search_;
    }

    /**
     * @brief Accept/reject the trial step and compute optimal step size.
     *
     * After calling step() and evaluating the residual at the trial λ,
     * call this to compute the optimal step via linear interpolation.
     *
     * @param residual_trial  residual at the trial λ (after step())
     * @return                optimal step size α_opt
     */
    double accept_trial(const std::vector<double>& residual_trial)
    {
        // ---- Linear interpolation (secant method) ----
        // α_opt = α_trial · sum_k / sum_k2
        // sum_k  = Σ (target - r_cur) · (r_trial - r_cur)
        //        = Σ (-r_cur) · (r_trial - r_cur)
        //        = Σ (-r_cur) · Δr
        // sum_k2 = Σ |r_cur - r_trial|² = Σ Δr²
        double sum_k = 0.0, sum_k2 = 0.0;
        for (int i = 0; i < n_dim_; ++i)
        {
            double dr = residual_trial[i] - residual_[i];  // r_trial - r_cur
            sum_k += (-residual_[i]) * dr;  // (target - r_cur) · Δr = -r_cur · Δr
            sum_k2 += dr * dr;
        }

        // T-7' componentwise mode: per-component secant (diagonal Jacobian).
        // α_opt[i] = α_trial[i]·(−r_i·Δr_i)/Δr_i² — each component's step is
        // estimated from its own response only, so opposite-sign residual
        // components no longer contaminate each other's step size or sign.
        if (componentwise_)
        {
            double alpha_opt_max = 0.0;
            for (int i = 0; i < n_dim_; ++i)
            {
                double dr = residual_trial[i] - residual_[i];
                double denom = dr * dr;
                double alpha_opt = current_trial_alpha_vec_[i];
                if (denom > 1e-30)
                {
                    // (−r_cur)·Δr / Δr² gives the per-component secant step.
                    alpha_opt = current_trial_alpha_vec_[i]
                                * (-residual_[i] * dr) / denom;
                }
                // Per-component restriction: |α_opt[i]·s[i]| ≤ max_step.
                if (max_step_ > 0.0 && std::abs(current_search_[i]) > 1e-30
                    && std::abs(alpha_opt * current_search_[i]) > max_step_)
                {
                    alpha_opt = (alpha_opt > 0 ? 1.0 : -1.0) * max_step_
                                / std::abs(current_search_[i]);
                }
                dnu_[i] += (alpha_opt - current_trial_alpha_vec_[i])
                           * current_search_[i];
                // Per-component adaptive step (same γ rule as scalar mode).
                double g = 1.0 * std::abs(alpha_opt) / current_trial_alpha_vec_[i];
                g = std::max(0.5, std::min(2.0, g));
                alpha_trial_vec_[i] *= std::pow(g, 0.7);
                alpha_opt_max = std::max(alpha_opt_max, std::abs(alpha_opt));
            }
            return alpha_opt_max;
        }

        double alpha_opt = current_trial_alpha_;
        if (std::abs(sum_k2) > 1e-30)
            alpha_opt = current_trial_alpha_ * sum_k / sum_k2;

        // ---- Restrict optimal step ----
        check_restriction(current_search_, alpha_opt);

        // ---- Correct dnu: dnu += (α_opt - α_trial) · search ----
        double alpha_corr = alpha_opt - current_trial_alpha_;
        for (int i = 0; i < n_dim_; ++i)
            dnu_[i] += alpha_corr * current_search_[i];

        // ---- Adaptive step size update ----
        // γ = clip(1.0 · |α_opt| / α_trial, 0.5, 2.0)
        // α_trial' = α_trial · γ^{0.7}
        double g = 1.0 * std::abs(alpha_opt) / current_trial_alpha_;
        g = std::max(0.5, std::min(2.0, g));
        alpha_trial_ *= std::pow(g, 0.7);

        return alpha_opt;
    }

    /**
     * @brief Get the total lambda = initial_lambda + dnu after accept_trial().
     */
    void get_lambda(std::vector<double>& lam_out) const
    {
        for (int i = 0; i < n_dim_; ++i)
            lam_out[i] = initial_lambda_[i] + dnu_[i];
    }

    /**
     * @brief Check if gradient d(residual)/d(lambda) has decayed below threshold.
     *
     * This provides an early exit when further optimization yields diminishing
     * returns — the gradient magnitude has become too flat to make progress.
     *
     * dM_eff = |r_cur - r_last| · min(1, nsc_max / step_count)
     * if dM_eff < decay_grad · |r_cur|: gradient decayed
     *
     * @param residual_last  residual from the PREVIOUS step (r_{k-1})
     * @return true if gradient has decayed and early exit is recommended
     */
    bool gradient_decayed(const std::vector<double>& residual_last) const
    {
        if (step_count_ < nsc_min_)
            return false;

        double dr2 = 0.0, r2 = 0.0;
        for (int i = 0; i < n_dim_; ++i)
        {
            double dr = residual_[i] - residual_last[i];
            dr2 += dr * dr;
            r2 += residual_[i] * residual_[i];
        }
        double dM_eff = std::sqrt(dr2) * std::min(1.0,
                            static_cast<double>(nsc_min_) / step_count_);
        double bound = decay_grad_ * std::sqrt(r2);

        return (dM_eff < bound);
    }

    /**
     * @brief Get the latest RMS error.
     */
    double get_rms() const { return latest_rms_; }

    /**
     * @brief Get the current trial step size.
     */
    double get_alpha() const { return alpha_trial_; }

    /**
     * @brief Get the number of completed inner steps.
     */
    int get_step_count() const { return step_count_; }

    /**
     * @brief Set the convergence threshold (may be tightened during optimization).
     */
    void set_conv_thr(double thr) { conv_thr_ = thr; }

  private:
    /**
     * @brief Cap the step size: |α · max_i(search[i])| ≤ max_step_.
     */
    void check_restriction(const std::vector<double>& search, double& alpha)
    {
        // boundary = |α| · max_i |search[i]|
        double smax = 0.0;
        for (int i = 0; i < n_dim_; ++i)
            smax = std::max(smax, std::abs(search[i]));
        double boundary = std::abs(alpha) * smax;

        if (max_step_ > 0.0 && boundary > max_step_)
        {
            // α = sign(α) · max_step / smax
            alpha = (alpha > 0 ? 1.0 : -1.0) * max_step_ / smax;
        }
    }

    int n_dim_ = 0;
    double alpha_trial_ = 1.0;
    double conv_thr_ = 1e-3;
    int nsc_min_ = 2;
    double decay_grad_ = 0.1;
    double max_step_ = 10.0;
    int step_count_ = 0;
    double latest_rms_ = 0.0;
    double current_trial_alpha_ = 0.0;
    bool componentwise_ = false;                    ///< T-7' per-component secant
    std::vector<double> alpha_trial_vec_;           ///< per-component trial α
    std::vector<double> current_trial_alpha_vec_;   ///< per-component α of trial

    // State vectors
    std::vector<double> initial_lambda_;   // λ₀ — reference point
    std::vector<double> delta_lambda_;     // Δλ — current change from λ₀
    std::vector<double> dnu_;              // cumulative search path integral
    std::vector<double> dnu_last_;         // dnu from previous step
    std::vector<double> search_;           // current search direction d_k
    std::vector<double> search_old_;       // previous search direction d_{k-1}
    std::vector<double> residual_;         // current residual r_k
    std::vector<double> residual_old_;     // previous residual r_{k-1}
    std::vector<double> lambda_;           // workspace: λ = λ₀ + dnu
    std::vector<double> current_search_;   // search direction used for trial
};

} // namespace ModuleOptimizer

#endif // FLETCHER_REEVES_CG_H
