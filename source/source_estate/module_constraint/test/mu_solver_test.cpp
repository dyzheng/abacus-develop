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
