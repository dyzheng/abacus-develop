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
