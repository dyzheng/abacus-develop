#include "gtest/gtest.h"
#include "module_optimizer/bfgs.h"
#include <vector>
#include <cmath>
#include <iostream>

using ModuleOptimizer::FletcherReevesCG;

/**
 * Helper: evaluate the physical observable f(lambda) and return residual.
 * For test purposes, f can be any of: f(λ)=λ, f(λ)=(λ-2)², f(λ)=A·λ, etc.
 * @param lam      current lambda
 * @param f        callback: f(lam, residual) fills residual = f(lam) - target
 */
static void run_bfgs(int n, double alpha_init, double conv_thr, int nsc_min,
                     double decay_grad, double max_step, int max_steps,
                     const std::vector<double>& initial_lam,
                     void (*f)(const std::vector<double>&, std::vector<double>&),
                     std::vector<double>& final_lam)
{
    FletcherReevesCG bfgs;
    bfgs.init(n, alpha_init, conv_thr, nsc_min, decay_grad, max_step);

    std::vector<double> lambda = initial_lam;
    bfgs.start_outer(lambda);

    std::vector<double> residual(n), lam_trial(n);

    for (int step = 0; step < max_steps; ++step)
    {
        // Compute residual at current lambda
        f(lambda, residual);

        // Check gradient decay (early exit)
        if (step >= 1) {
            // Get gradient decay status from previous step's residual
            // (stored in residual_old inside FletcherReevesCG)
        }

        // Propose trial lambda
        bool converged = false;
        bfgs.step(residual, step, lam_trial, converged);
        if (converged) break;

        // Evaluate residual at trial lambda
        std::vector<double> residual_trial(n);
        f(lam_trial, residual_trial);

        // Accept trial and get optimal step
        bfgs.accept_trial(residual_trial);
        bfgs.get_lambda(lambda);

        // Check convergence after correction
        double rms = 0.0;
        for (int i = 0; i < n; ++i)
            rms += residual[i] * residual[i];
        rms = std::sqrt(rms / n);
        if (rms < conv_thr) break;
    }
    final_lam = lambda;
}

/**
 * Test 1: f(λ) = λ (identity), target = 0.
 * Solution: λ = 0.
 * r(λ) = λ - 0 = λ.
 */
static void f_linear(const std::vector<double>& lam, std::vector<double>& r)
{
    for (size_t i = 0; i < lam.size(); ++i)
        r[i] = lam[i] - 0.0;
}

TEST(FletcherReevesCGTest, LinearResidual)
{
    int n = 1;
    std::vector<double> initial = {2.0};
    std::vector<double> final(n);
    run_bfgs(n, 1.0, 1e-6, 2, 0.01, 10.0, 20, initial, f_linear, final);
    EXPECT_NEAR(final[0], 0.0, 1e-5);
}

/**
 * Test 2: f(λ) = (λ-2)², target = 0.
 * Solution: λ = 2 (where f=0).
 * r(λ) = (λ-2)²
 */
static void f_quadratic(const std::vector<double>& lam, std::vector<double>& r)
{
    r[0] = (lam[0] - 2.0) * (lam[0] - 2.0);
}

TEST(FletcherReevesCGTest, QuadraticResidual)
{
    int n = 1;
    std::vector<double> initial = {0.0};
    std::vector<double> final(n);
    run_bfgs(n, 0.5, 1e-4, 3, 0.05, 5.0, 30, initial, f_quadratic, final);
    EXPECT_NEAR(final[0], 2.0, 0.05);
}

/**
 * Test 3: f₁ = λ₁+0.5λ₂, f₂ = 0.5λ₁+λ₂, target=[1,1].
 * Solution: λ₁ = λ₂ = 2/3.
 */
static void f_coupled2d(const std::vector<double>& lam, std::vector<double>& r)
{
    r[0] = (lam[0] + 0.5 * lam[1]) - 1.0;
    r[1] = (0.5 * lam[0] + lam[1]) - 1.0;
}

TEST(FletcherReevesCGTest, Coupled2D)
{
    int n = 2;
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> final(n);
    run_bfgs(n, 0.3, 1e-6, 2, 0.01, 5.0, 50, initial, f_coupled2d, final);
    EXPECT_NEAR(final[0], 2.0 / 3.0, 1e-4);
    EXPECT_NEAR(final[1], 2.0 / 3.0, 1e-4);
}

/**
 * Test 4: Restriction cap.
 * max_step=0.1 limits step to 0.1 per component.
 */
TEST(FletcherReevesCGTest, RestrictionCap)
{
    FletcherReevesCG bfgs;
    int n = 1;
    bfgs.init(n, 10.0, 1e-6, 2, 0.01, 0.1);
    std::vector<double> lambda = {0.0};
    bfgs.start_outer(lambda);

    std::vector<double> residual = {1.0}, lam_out(n);
    bool cv = false;
    bfgs.step(residual, 0, lam_out, cv);

    // search = [1.0], alpha capped to 0.1/1.0 = 0.1
    // dnu = 0.1 * 1.0 = 0.1, lambda = 0 + 0.1 = 0.1
    EXPECT_NEAR(lam_out[0], 0.1, 1e-10);
}

/**
 * Test 5: accept_trial() adjusts dnu correctly.
 * After step, accept_trial computes correct alpha_opt.
 */
TEST(FletcherReevesCGTest, AcceptTrial)
{
    FletcherReevesCG bfgs;
    int n = 1;
    bfgs.init(n, 1.0, 1e-10, 2, 0.01, 10.0);
    std::vector<double> lambda = {2.0};
    bfgs.start_outer(lambda);

    std::vector<double> residual = {2.0}; // r = f(2)-0 = 2
    std::vector<double> lam_trial(n);
    bool cv = false;
    bfgs.step(residual, 0, lam_trial, cv);
    // trial: lambda = 2 + 1.0*2.0 = 4.0
    EXPECT_NEAR(lam_trial[0], 4.0, 1e-10);

    // Evaluate at trial: r_trial = 4.0
    std::vector<double> residual_trial = {4.0};
    // accept_trial: alpha_opt = 1.0 * sum_k/sum_k2
    // sum_k = (-2)*(4-2) = -4, sum_k2 = (2-4)² = 4
    // alpha_opt = -1.0, correction = -1.0 - 1.0 = -2.0
    // dnu = 2.0 + (-2.0)*2.0 = -2.0
    // final lambda = 2 + (-2.0) = 0.0 (exact)
    bfgs.accept_trial(residual_trial);

    std::vector<double> lam_final(n);
    bfgs.get_lambda(lam_final);
    EXPECT_NEAR(lam_final[0], 0.0, 1e-10);
}

/**
 * Test 6: Convergence within max_steps for a well-behaved problem.
 */
static void f_slow_linear(const std::vector<double>& lam, std::vector<double>& r)
{
    r[0] = 0.1 * lam[0] - 1.0; // f=0.1λ, target=1.0 → solution λ=10
}

TEST(FletcherReevesCGTest, SlowConvergence)
{
    int n = 1;
    std::vector<double> initial = {0.0};
    std::vector<double> final(n);
    run_bfgs(n, 2.0, 1e-6, 2, 0.01, 10.0, 30, initial, f_slow_linear, final);
    EXPECT_NEAR(final[0], 10.0, 1e-4);
}

/**
 * T-7' tests: per-component secant (diagonal Jacobian) mode.
 *
 * The motivating failure (T-2 root cause ②, T3' a2): a scalar α
 * α_opt = α_trial·Σ(−r_i·Δr_i)/ΣΔr_i² mixes opposite-sign per-atom residual
 * components, so O (wants λ up) and H (wants λ down) contaminate each other's
 * step — α flips sign and the RMS plateaus.  Componentwise mode gives each
 * component its own secant step α_opt[i] = α_trial[i]·(−r_i·Δr_i)/Δr_i².
 */

/**
 * Diagonal system with opposite-sign responses and opposite-sign targets:
 *   r₁ = 2·λ₁ − 1   (solution λ₁ = 0.5)
 *   r₂ = −3·λ₂ + 1  (solution λ₂ = 1/3)
 * The single-α secant mixes the two slopes (d r₁/dλ₁=+2, d r₂/dλ₂=−3).
 */
static void f_opposite_sign(const std::vector<double>& lam, std::vector<double>& r)
{
    r[0] = 2.0 * lam[0] - 1.0;
    r[1] = -3.0 * lam[1] + 1.0;
}

TEST(FletcherReevesCGTest, ComponentwiseOppositeSignConverges)
{
    const int n = 2;
    std::vector<double> initial = {0.0, 0.0};
    FletcherReevesCG cg;
    cg.init(n, 0.5, 1e-8, 2, 0.01, 5.0);
    cg.set_componentwise(true);
    EXPECT_TRUE(cg.componentwise());
    std::vector<double> lambda = initial, lam_trial(n), residual(n), residual_trial(n);
    cg.start_outer(lambda);
    bool converged = false;
    for (int step = 0; step < 200 && !converged; ++step)
    {
        f_opposite_sign(lambda, residual);
        cg.step(residual, step, lam_trial, converged);
        if (converged)
            break;
        f_opposite_sign(lam_trial, residual_trial);
        cg.accept_trial(residual_trial);
        cg.get_lambda(lambda);
        double rms = 0.0;
        for (int i = 0; i < n; ++i)
            rms += residual[i] * residual[i];
        rms = std::sqrt(rms / n);
        if (rms < 1e-8)
            break;
    }
    EXPECT_NEAR(lambda[0], 0.5, 1e-4);
    EXPECT_NEAR(lambda[1], 1.0 / 3.0, 1e-4);
}

/**
 * Same system, scalar-CG reference.  Contract: componentwise converges within
 * the budget; when the scalar path also converges it must not be faster.
 * (On the sign-opposite system the scalar-α secant mixes the two slopes and
 * typically cannot converge — that is the T-2/T3' a2 failure mode this mode
 * fixes; the assertion is kept one-sided so a future scalar-path improvement
 * does not break the test.)
 */
TEST(FletcherReevesCGTest, ComponentwiseBeatsScalarOnOppositeSign)
{
    const int n = 2;
    const int max_steps = 60;
    std::vector<double> final_cw(n), final_sc(n);
    int steps_cw = max_steps, steps_sc = max_steps;

    auto run_scheme = [&](bool cw, std::vector<double>& final_lam, int& steps_out) {
        std::vector<double> initial = {0.0, 0.0};
        FletcherReevesCG cg;
        cg.init(n, 0.5, 1e-8, 2, 0.01, 5.0);
        cg.set_componentwise(cw);
        std::vector<double> lambda = initial, lam_trial(n), residual(n), rt(n);
        cg.start_outer(lambda);
        bool converged = false;
        int steps = 0;
        for (; steps < max_steps && !converged; ++steps)
        {
            f_opposite_sign(lambda, residual);
            cg.step(residual, steps, lam_trial, converged);
            if (converged)
                break;
            f_opposite_sign(lam_trial, rt);
            cg.accept_trial(rt);
            cg.get_lambda(lambda);
        }
        steps_out = steps;
        final_lam = lambda;
        return converged;
    };

    bool cw_ok = run_scheme(true, final_cw, steps_cw);
    bool sc_ok = run_scheme(false, final_sc, steps_sc);
    // Componentwise must converge within budget (and to the right point).
    EXPECT_TRUE(cw_ok);
    EXPECT_NEAR(final_cw[0], 0.5, 1e-3);
    EXPECT_NEAR(final_cw[1], 1.0 / 3.0, 1e-3);
    // When both converge, componentwise must not be slower than scalar CG.
    if (cw_ok && sc_ok)
    {
        EXPECT_LE(steps_cw, steps_sc) << "cw=" << steps_cw << " sc=" << steps_sc;
    }
    std::cout << "[info] opposite-sign: cw_ok=" << cw_ok << " steps=" << steps_cw
              << " sc_ok=" << sc_ok << " steps=" << steps_sc << std::endl;
}
