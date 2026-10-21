#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "source_estate/module_constraint/mu_solver.h"

// Synthetic response Q(mu) = q0 - chi * mu (chi > 0 is the standard
// negative linear response; chi < 0 models a pathological non-monotonic
// channel for the sign-flip guard).
struct MockResponse
{
    double q0;
    double chi;
    double operator()(const double mu) const
    {
        return q0 - chi * mu;
    }
};

// Drive the solver until it leaves RUNNING or max_steps elapse.
static constraint::MuStatus drive(constraint::MuSolver& solver,
                                  const MockResponse& mock,
                                  const std::vector<double>& target,
                                  std::vector<double>& mu,
                                  const int max_steps)
{
    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < max_steps; ++k)
    {
        std::vector<double> Q(mu.size());
        for (size_t i = 0; i < mu.size(); ++i)
        {
            Q[i] = mock(mu[i]);
        }
        st = solver.step(Q, target, mu);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    return st;
}

TEST(MuSolverTest, SecantKnownRoot)
{
    // Single component: Q(mu) = 2.0 - 0.5 mu, target 1.9 -> root mu* = 0.2.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 5.0;
    params.kappa_min = 0.3;
    params.kappa_max = 20.0;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.5};

    std::vector<double> mu = {0.0};
    std::vector<double> target = {1.9};
    std::vector<double> mu_trace;
    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < 6; ++k)
    {
        const double Q = mock(mu[0]);
        st = solver.step({Q}, target, mu);
        mu_trace.push_back(mu[0]);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    EXPECT_EQ(st, constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(mock(mu[0]), 1.9, 1e-6);
    // mu approaches the root monotonically from below (step-limited).
    for (size_t k = 1; k < mu_trace.size(); ++k)
    {
        EXPECT_GE(mu_trace[k], mu_trace[k - 1]);
    }
    EXPECT_LE(mu[0], 0.2 + 1e-12);
}

TEST(MuSolverTest, SecantKnownRootTwoComponents)
{
    // Two independent channels: mu0* = 0.2, mu1* = 0.05.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock0{2.0, 0.5};
    MockResponse mock1{3.0, 2.0};

    std::vector<double> mu = {0.0, 0.0};
    const std::vector<double> target = {1.9, 2.9};
    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < 8; ++k)
    {
        const std::vector<double> Q = {mock0(mu[0]), mock1(mu[1])};
        st = solver.step(Q, target, mu);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    EXPECT_EQ(st, constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(mu[0], 0.2, 1e-3);
    EXPECT_NEAR(mu[1], 0.05, 1e-3);
}

TEST(MuSolverTest, SignFlipGuardAndFuse)
{
    // Non-monotonic channel Q(mu) = 2.0 + 0.5 mu: the secant slope flips
    // sign every step; the guard must keep the run bounded (mu pinned at the
    // cap, never beyond) and fuse it as UNREACHABLE instead of diverging.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 0.5;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    params.plateau_window = 3;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, -0.5};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};
    constraint::MuStatus st = drive(solver, mock, target, mu, 40);
    EXPECT_EQ(st, constraint::MuStatus::UNREACHABLE);
    EXPECT_GT(solver.sign_flip_count(), 0);
    EXPECT_DOUBLE_EQ(mu[0], params.mu_max);
    EXPECT_EQ(solver.fuse_component(), 0);
    EXPECT_DOUBLE_EQ(solver.fuse_mu(), params.mu_max);
    EXPECT_NEAR(solver.fuse_residual(), mock(mu[0]) - target[0], 1e-12);
    EXPECT_TRUE(std::isfinite(mu[0]));
}

TEST(MuSolverTest, PositiveResponseWithPositiveSignParam)
{
    // Parameter machinery test (not the spin channel's physics): a channel
    // with an inverted (non-standard) positive response Q(mu) = 1.0 + 2.0 mu
    // converges only when response_sign = +1 tells the secant to expect it.
    // With the default -1 the same mock is a sign-flip channel (see
    // SignFlipGuardAndFuse) and fuses, so the parameter is the
    // discriminator, not the test tolerance.  The real spin channel responds
    // negatively like charge (V_up += mu*w repels spin-up), so the default
    // -1 applies there too (SpinChannelConvergesOnLinearResponse).
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 5.0;
    params.kappa_min = 0.3;
    params.kappa_max = 20.0;
    params.conv_tol = 1e-4;
    params.response_sign = 1;
    constraint::MuSolver solver(params);
    MockResponse mock{1.0, -2.0}; // Q = 1 + 2 mu, root mu* = 0.05 for t=1.1

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.1};
    std::vector<double> mu_trace;
    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < 10; ++k)
    {
        const double Q = mock(mu[0]);
        st = solver.step({Q}, target, mu);
        mu_trace.push_back(mu[0]);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    EXPECT_EQ(st, constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(mu[0], 0.05, 1e-3);
    EXPECT_NEAR(mock(mu[0]), 1.1, 1e-6);
    // Approaches the root monotonically from below (step-limited).
    for (size_t k = 1; k < mu_trace.size(); ++k)
    {
        EXPECT_GE(mu_trace[k], mu_trace[k - 1]);
    }
}

TEST(MuSolverTest, FlatChannelFuse)
{
    // Both-sides-unreachable channel Q(mu) = Q0 constant: mu marches to the
    // cap and the flat residual plateau fuses the run (dead channel, T-5'),
    // rather than marching on or growing without bound.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 0.5;
    params.conv_tol = 1e-4;
    params.plateau_window = 3;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.0};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};
    constraint::MuStatus st = drive(solver, mock, target, mu, 40);
    EXPECT_EQ(st, constraint::MuStatus::UNREACHABLE);
    EXPECT_DOUBLE_EQ(mu[0], params.mu_max);
    EXPECT_NEAR(solver.fuse_residual(), 0.1, 1e-12);
    // Every intermediate mu stays inside the cap.
    EXPECT_LE(mu[0], params.mu_max + 1e-12);
}

TEST(MuSolverTest, SoftChannelFuseAtCap)
{
    // Soft response Q(mu) = 2.0 - 0.001 mu: the true root (mu = 100) lies
    // beyond mu_max; the kappa_min clamp keeps the march finite and the run
    // fuses at the cap with the endpoint residual reported.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 0.5;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.001};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};
    constraint::MuStatus st = drive(solver, mock, target, mu, 40);
    EXPECT_EQ(st, constraint::MuStatus::UNREACHABLE);
    EXPECT_DOUBLE_EQ(mu[0], params.mu_max);
    EXPECT_NEAR(solver.fuse_residual(), mock(mu[0]) - target[0], 1e-12);
}

TEST(MuSolverTest, AntiFakeConvergence)
{
    // mu = 0 with Q == target converges on the very first step.
    constraint::MuSolver solver;
    std::vector<double> mu = {0.0};
    EXPECT_EQ(solver.step({1.9}, {1.9}, mu), constraint::MuStatus::CONVERGED);

    // Q != target must never report CONVERGED; with a flat response it
    // eventually fuses as UNREACHABLE.
    constraint::MuSolverParams params;
    params.mu_max = 0.3;
    constraint::MuSolver solver2(params);
    MockResponse mock{2.0, 0.0};
    std::vector<double> mu2 = {0.0};
    constraint::MuStatus st = drive(solver2, mock, {1.9}, mu2, 30);
    EXPECT_NE(st, constraint::MuStatus::CONVERGED);
    EXPECT_EQ(st, constraint::MuStatus::UNREACHABLE);
}

TEST(MuSolverTest, StiffChannelBoundedLimitCycle)
{
    // Stiff response Q(mu) = 2.0 - 100 mu with a step-limited secant: the
    // root (0.001) is below the step resolution, so the run enters a bounded
    // limit cycle — never diverges, never fakes convergence.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.mu_max = 5.0;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 100.0};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};
    constraint::MuStatus st = drive(solver, mock, target, mu, 50);
    EXPECT_EQ(st, constraint::MuStatus::RUNNING);
    EXPECT_TRUE(std::isfinite(mu[0]));
    EXPECT_LE(std::abs(mu[0]), params.mu_max);
    EXPECT_LE(std::abs(mock(mu[0]) - target[0]), 5.0); // residual stays bounded
}

TEST(MuSolverTest, StepProbeCapsOnlyTheFirstStep)
{
    // II-1: the first step of a component has no measured slope, so it would
    // always sit at the step_max cap.  step_probe must cap that step alone
    // and leave every later step on step_max.
    constraint::MuSolverParams params;
    params.step_max = 0.05;
    params.step_probe = 0.004;
    params.mu_max = 5.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.5};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};

    // First step: history-free -> the probe cap, not the step_max cap.
    EXPECT_EQ(solver.step({mock(0.0)}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_DOUBLE_EQ(mu[0], params.step_probe);

    // Second step: the secant slope is real now (chi = 0.5 is inside the
    // clamp window), so the move is -res/kappa = 0.18/0.5 = 0.36 Ry, capped
    // at step_max -- i.e. strictly larger than the probe.
    EXPECT_EQ(solver.step({mock(mu[0])}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_DOUBLE_EQ(mu[0], params.step_probe + params.step_max);

    // Legacy: probe == 0 means "use step_max", i.e. the first step is the
    // full cap, exactly as before this parameter existed.
    constraint::MuSolverParams legacy;
    legacy.step_max = 0.05;
    constraint::MuSolver solver2(legacy);
    std::vector<double> mu2 = {0.0};
    EXPECT_EQ(solver2.step({mock(0.0)}, target, mu2),
              constraint::MuStatus::RUNNING);
    EXPECT_DOUBLE_EQ(mu2[0], legacy.step_max);
}

TEST(MuSolverTest, RebindRetainsSlope)
{
    constraint::MuSolverParams p;
    p.step_max = 0.05;
    p.kappa_min = 0.3;
    p.response_sign = -1;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0};
    // First observation at mu = 0 with Q = 1.0 above the target 0: the
    // negative-response channel needs +mu to repel charge,
    // dmu = -(Q-t)/kappa = -(1.0)/(-0.3) = +3.33, capped to +0.05.
    EXPECT_EQ(s.step({1.0}, {0.0}, mu), constraint::MuStatus::RUNNING);
    const double mu1 = mu[0];
    EXPECT_DOUBLE_EQ(mu1, 0.05);
    // Second observation Q = 0.9 (fell by 0.1 under +0.05 Ry): the measured
    // slope is dQ/dmu = -0.1/+0.05 = -2 (physical negative response).
    EXPECT_EQ(s.step({0.9}, {0.0}, mu), constraint::MuStatus::RUNNING);
    const double slope = s.slope(0);
    EXPECT_NEAR(slope, -2.0, 1e-12);

    // New geometry: anchor at the carried mu with a fresh Q.  The retained
    // slope must survive rebind, while the stale cross-geometry Q history
    // must not.
    s.rebind({mu1}, {0.95});
    EXPECT_NEAR(s.slope(0), slope, 1e-12);
    const double mu_before = mu[0];
    s.step({0.95}, {0.0}, mu); // dmu_obs == 0 -> retained slope path
    EXPECT_NE(mu[0], mu_before);
}

TEST(MuSolverTest, RebindCarriesJacobianWhenEnabled)
{
    constraint::MuSolverParams p;
    p.trust_region = true;
    p.broyden = true;
    p.step_max = 0.05;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0, 0.0};
    // Drive several accepted steps so an off-diagonal coupling is learned.
    s.step({1.0, 0.5}, {0.0, 0.0}, mu);
    s.step({0.9, 0.55}, {0.0, 0.0}, mu);
    s.step({0.81, 0.605}, {0.0, 0.0}, mu);
    const double j01 = s.jacobian(0, 1);
    // Carry on: the full Jacobian survives the rebind.
    s.set_carry_jacobian(true);
    s.rebind(mu, {0.8, 0.6});
    EXPECT_NEAR(s.jacobian(0, 1), j01, 1e-12);
    // Carry off (legacy): diagonal reset wipes the coupling.
    constraint::MuSolver s2(p);
    std::vector<double> mu2 = {0.0, 0.0};
    s2.step({1.0, 0.5}, {0.0, 0.0}, mu2);
    s2.step({0.9, 0.55}, {0.0, 0.0}, mu2);
    s2.step({0.81, 0.605}, {0.0, 0.0}, mu2);
    s2.rebind(mu2, {0.8, 0.6});
    EXPECT_DOUBLE_EQ(s2.jacobian(0, 1), 0.0);
}

TEST(MuSolverTest, RebindResetsFuseWindow)
{
    constraint::MuSolverParams p;
    p.step_max = 0.05;
    p.mu_max = 0.05;
    p.plateau_window = 2;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0};
    // Drive the component onto the cap with a flat residual.
    s.step({1.0}, {0.0}, mu);
    s.step({1.0}, {0.0}, mu);
    // A stale plateau must not fuse the new geometry immediately.
    s.rebind(mu, {1.0});
    const constraint::MuStatus st = s.step({1.0}, {0.0}, mu);
    EXPECT_EQ(st, constraint::MuStatus::RUNNING);
}

TEST(MuSolverTest, TrustRegionExpandsOnGoodStep)
{
    // Linear response with a perfect model after the first observation:
    // rho ~= 1 on every step, so the radius must double and let the next
    // step exceed the initial step_max cap.
    constraint::MuSolverParams params;
    params.trust_region = true;
    params.step_max = 0.05;
    params.trust_radius_max = 1.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.5};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {0.5}; // root mu = 3.0

    EXPECT_EQ(solver.step({mock(0.0)}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_NEAR(mu[0], params.step_max, 1e-12);
    EXPECT_NEAR(solver.trust_radius(), params.step_max, 1e-12);

    EXPECT_EQ(solver.step({mock(mu[0])}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_NEAR(solver.trust_radius(), 2.0 * params.step_max, 1e-12);
    // The second trial is twice the first one: the expanded radius -- not
    // step_max -- is the binding cap.
    EXPECT_NEAR(mu[0], 3.0 * params.step_max, 1e-12);
    EXPECT_GT(solver.trust_accept_count(), 0);
}

TEST(MuSolverTest, TrustRegionRejectsBadStep)
{
    // Q(mu) = 2 - 0.5 mu + 100 mu^2: the linear model predicts the first
    // step improves the residual, but the quadratic term makes it worse.
    // The solver must reject, halve the radius, restore mu and re-walk.
    constraint::MuSolverParams params;
    params.trust_region = true;
    params.step_max = 0.05;
    params.trust_radius_max = 1.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    auto mock = [](const double m) { return 2.0 - 0.5 * m + 100.0 * m * m; };

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.0};
    EXPECT_EQ(solver.step({mock(0.0)}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_NEAR(mu[0], params.step_max, 1e-12);

    const double rejected_mu = mu[0];
    EXPECT_EQ(solver.step({mock(mu[0])}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_EQ(solver.trust_reject_count(), 1);
    EXPECT_NEAR(solver.trust_radius(), 0.5 * params.step_max, 1e-12);
    // Re-walked from the pre-trial anchor: the new trial is smaller than the
    // rejected one and points in the same direction.
    EXPECT_GT(mu[0], 0.0);
    EXPECT_LT(mu[0], rejected_mu);
    EXPECT_NEAR(mu[0], 0.5 * params.step_max, 1e-12);
}

TEST(MuSolverTest, BroydenCapturesCoupling)
{
    // Coupled two-channel response Q = Q0 + J mu with J = [[-1, 0.5],
    // [0.5, -1]].  The first noisy rank-one update already leaves a
    // nonzero off-diagonal; the coupled solve must then reach the root.
    constraint::MuSolverParams params;
    params.trust_region = true;
    params.broyden = true;
    params.step_max = 0.05;
    params.trust_radius_max = 1.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);

    const std::vector<double> q0 = {1.0, 1.0};
    const std::vector<double> target = {0.0, 0.0};
    std::vector<double> mu = {0.0, 0.0};
    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < 30; ++k)
    {
        const std::vector<double> Q = {q0[0] - mu[0] + 0.5 * mu[1],
                                       q0[1] + 0.5 * mu[0] - mu[1]};
        st = solver.step(Q, target, mu);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    EXPECT_EQ(st, constraint::MuStatus::CONVERGED);
    EXPECT_NE(solver.jacobian(0, 1), 0.0);
    EXPECT_NEAR(mu[0], 2.0, 1e-3);
    EXPECT_NEAR(mu[1], 2.0, 1e-3);
}

TEST(MuSolverTest, RejectStepRestoresMu)
{
    // A forced rejection (E1 magnetic anchor) must shrink the radius and
    // re-walk from the pre-trial anchor instead of accepting the pending mu.
    constraint::MuSolverParams params;
    params.trust_region = true;
    params.step_max = 0.05;
    params.trust_radius_max = 1.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);
    MockResponse mock{2.0, 0.5};

    std::vector<double> mu = {0.0};
    const std::vector<double> target = {1.9};
    EXPECT_EQ(solver.step({mock(0.0)}, target, mu),
              constraint::MuStatus::RUNNING);
    const double rejected_mu = mu[0];
    EXPECT_DOUBLE_EQ(rejected_mu, params.step_max);

    EXPECT_EQ(solver.reject({mock(mu[0])}, target, mu),
              constraint::MuStatus::RUNNING);
    EXPECT_EQ(solver.trust_reject_count(), 1);
    EXPECT_NEAR(solver.trust_radius(), 0.5 * params.step_max, 1e-12);
    EXPECT_GT(mu[0], 0.0);
    EXPECT_LT(mu[0], rejected_mu);
    EXPECT_NEAR(mu[0], 0.5 * params.step_max, 1e-12);
}

TEST(MuSolverTest, BadRhoUpdatesModelInsteadOfRepeatingTheStep)
{
    // Two independent charge-like channels (dQ/dmu = -0.58) with opposite
    // targets.  The first trial (+/-0.00333 Ry) is not radius-limited; rho is
    // bad because the initial kappa=-0.3 over-predicts the improvement.  The
    // rejected observation must update J so the re-walk is smaller instead
    // of repeating the identical +/-0.00333 step.
    constraint::MuSolverParams params;
    params.trust_region = true;
    params.broyden = true;
    params.step_max = 0.05;
    params.trust_radius_max = 1.0;
    params.kappa_min = 0.3;
    params.conv_tol = 1e-4;
    constraint::MuSolver solver(params);

    const double q0 = 10.0;
    const double q1 = 10.0;
    const double chi = 0.58;
    const std::vector<double> target = {q0 + 0.001, q1 - 0.001};
    std::vector<double> mu = {0.0, 0.0};
    std::vector<double> Q = {q0, q1};
    EXPECT_EQ(solver.step(Q, target, mu), constraint::MuStatus::RUNNING);
    EXPECT_NEAR(mu[0], -0.001 / 0.3, 1e-12);
    EXPECT_NEAR(mu[1], 0.001 / 0.3, 1e-12);

    Q = {q0 - chi * mu[0], q1 - chi * mu[1]};
    EXPECT_EQ(solver.step(Q, target, mu), constraint::MuStatus::RUNNING);
    EXPECT_EQ(solver.trust_reject_count(), 1);
    // The re-walk used the measured chi, not the stale kappa.
    EXPECT_NEAR(mu[0], -0.001 / chi, 1e-6);
    EXPECT_NEAR(mu[1], 0.001 / chi, 1e-6);

    constraint::MuStatus st = constraint::MuStatus::RUNNING;
    for (int k = 0; k < 10; ++k)
    {
        Q = {q0 - chi * mu[0], q1 - chi * mu[1]};
        st = solver.step(Q, target, mu);
        if (st != constraint::MuStatus::RUNNING)
        {
            break;
        }
    }
    EXPECT_EQ(st, constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(mu[0], -0.001 / chi, 1e-6);
    EXPECT_NEAR(mu[1], 0.001 / chi, 1e-6);
}
