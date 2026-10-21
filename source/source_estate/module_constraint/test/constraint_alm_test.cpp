#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "source_estate/module_constraint/constraint_alm.h"

// Arm: k_a = kappa_target / |chi_a|, clamped to [k_min, k_max]; zero or
// sign-uncertain chi falls back to kappa_target (chi = -1 nominal).
TEST(ConstraintAlmTest, ArmFromChi)
{
    constraint::AlmController alm;
    alm.resize(2);
    EXPECT_FALSE(alm.armed());
    alm.arm({-0.65, -24.0});
    ASSERT_TRUE(alm.armed());
    EXPECT_NEAR(alm.k()[0], 0.5 / 0.65, 1e-12);
    // 0.5/24 = 0.0208 is below k_min: the auto gain is clamped to 0.1.
    EXPECT_NEAR(alm.k()[1], 0.1, 1e-12);
}

// mu_eff: before arming it is plain mu; after arming it is mu + k(Q - t).
TEST(ConstraintAlmTest, MuEffBeforeAndAfterArming)
{
    constraint::AlmController alm;
    alm.resize(1);
    std::vector<double> eff;
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_DOUBLE_EQ(eff[0], 0.2); // unarmed: penalty inactive
    alm.arm({-1.0}); // k = 0.5
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_NEAR(eff[0], 0.2 + 0.5 * 1.0, 1e-12);
}

// Outer update: mu += k(Q - t), capped per component; a pinned component
// with a flat residual fuses (UNREACHABLE semantics mirrored from MuSolver).
TEST(ConstraintAlmTest, UpdateCapsAndFuses)
{
    constraint::AlmController alm;
    alm.resize(1);
    // Explicit strong gain so one update (k*1.0 = 10) hits the cap; the
    // fuse bookkeeping is independent of the auto-gain target.
    alm.set_k_override(10.0);
    alm.arm({-1.0});
    std::vector<double> mu = {0.0};
    double res = alm.update({1.0}, {0.0}, {5.0}, mu);
    EXPECT_NEAR(res, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(mu[0], 5.0);
    EXPECT_EQ(alm.fuse_component(), -1); // first pin: no fuse yet
    alm.update({1.0}, {0.0}, {5.0}, mu);
    alm.update({1.0}, {0.0}, {5.0}, mu);
    EXPECT_EQ(alm.fuse_component(), 0);
    EXPECT_NEAR(alm.fuse_residual(), 1.0, 1e-12);
}

// Energy correction: sum_a [ mu_a(Q-t) + (k_a/2)(Q-t)^2 ].
TEST(ConstraintAlmTest, EnergyCorrection)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-2.0}); // k = 0.25
    const double e = alm.energy_correction({0.5}, {1.2}, {1.0});
    EXPECT_NEAR(e, 0.5 * 0.2 + 0.5 * 0.25 * 0.04, 1e-12);
}

// Linear-model contraction: with the implemented k*|chi| = 0.5 the
// implicit outer update contracts |mu - mu*| by 1/|1 - k*chi| = 1/1.5 per
// step; mu_phys after convergence equals the exact multiplier.
TEST(ConstraintAlmTest, ContractionAndMuPhysExact)
{
    constraint::AlmController alm;
    alm.resize(1);
    const double chi = -0.65;      // LiCl near-neutral response (e/Ry)
    const double q_ref = 3.0959;   // Li fragment free charge
    const double t = q_ref + 0.1;  // delta = +0.1
    const double mu_star = (t - q_ref) / chi; // exact multiplier
    alm.arm({chi});
    std::vector<double> mu = {0.0};
    for (int step = 0; step < 40; ++step)
    {
        const double k = alm.k()[0];
        const double q = (q_ref + chi * (mu[0] - k * t)) / (1.0 - chi * k);
        const double res = std::abs(q - t);
        if (res < 1e-4)
        {
            break;
        }
        alm.update({q}, {t}, {50.0}, mu);
        // 1/1.5 per step from an O(0.15) start needs ~17 updates.
        ASSERT_LT(step, 25);
    }
    std::vector<double> eff;
    const double k = alm.k()[0];
    const double q = (q_ref + chi * (mu[0] - k * t)) / (1.0 - chi * k);
    alm.mu_eff(mu, {q}, {t}, eff);
    EXPECT_NEAR(eff[0], mu_star, 1e-3); // mu_phys ~= exact multiplier
}

// Dampen halves k with a floor at k_min.
TEST(ConstraintAlmTest, DampenFloors)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-0.004}); // 0.5/0.004 = 125 -> at the k_max cap
    EXPECT_NEAR(alm.k()[0], 100.0, 1e-12);
    for (int i = 0; i < 20; ++i)
    {
        alm.dampen();
    }
    EXPECT_NEAR(alm.k()[0], 0.1, 1e-12); // k_min floor
}

// Explicit INPUT gain overrides the response-derived k but is clamped.
TEST(ConstraintAlmTest, ExplicitGainOverride)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.set_k_override(3.0);
    alm.arm({-0.1}); // auto would be 5.0, explicit wins
    EXPECT_NEAR(alm.k()[0], 3.0, 1e-12);
    alm.set_k_override(1000.0);
    alm.arm({-0.1});
    EXPECT_NEAR(alm.k()[0], 100.0, 1e-12);
}

// Lagged potential feedback (R1 regression).  The single-layer loop updates
//   v <- mu + k(Q(v)-t),  Q(v) = q_ref + chi*v
// once per SCF iteration, but the density at iteration n was produced by the
// PREVIOUS potential.  For the lagged map the mode amplitude is
// sqrt(K/(1+K)) with K = k|chi|: 0.94 at the design's K=7.5 (R1 never settles
// inside scf_nmax=100; drho froze at 0.194) versus 0.58 at the implemented
// K=0.5.  This test pins both the damping factor and the settling rate.
TEST(ConstraintAlmTest, LaggedPotentialMapRate)
{
    const double chi = -0.58;
    const double q_ref = 10.0;
    const double t = q_ref + 0.1;
    const double mu = 0.0;

    // Helper: iterate the lagged damped map for n steps; returns |v - v*|.
    auto lagged_error = [&](const double k, const double alpha, const int n) {
        // Fixed point v* = mu + k(Q(v*)-t): v*(1-k*chi) = -k*0.1.
        const double vstar = -k * 0.1 / (1.0 - k * chi);
        double v_prev = 0.0; // density initially at Q(v_prev)
        double v = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double q = q_ref + chi * v_prev;
            const double desired = mu + k * (q - t);
            const double v_next = v + alpha * (desired - v);
            v_prev = v;
            v = v_next;
        }
        return std::abs(v - vstar);
    };

    // Implemented auto gain: K = 0.5, alpha = 2/3, mode sqrt(1/3).
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({chi});
    const double k_auto = alm.k()[0];
    EXPECT_NEAR(k_auto * std::abs(chi), 0.5, 1e-12);
    EXPECT_NEAR(alm.damping(0), 2.0 / 3.0, 1e-12);
    const double e_auto = lagged_error(k_auto, alm.damping(0), 30);
    EXPECT_LT(e_auto, 1e-5);

    // Design's K = 7.5 via an explicit override: alpha = 1/8.5 and the
    // lagged mode sqrt(7.5/8.5) = 0.94 is still an order of magnitude away
    // from the fixed point after the same 30 iterations.
    alm.set_k_override(7.5 / std::abs(chi));
    alm.arm({chi});
    EXPECT_NEAR(alm.k()[0] * std::abs(chi), 7.5, 1e-12);
    EXPECT_NEAR(alm.damping(0), 1.0 / 8.5, 1e-12);
    const double e_stiff = lagged_error(alm.k()[0], alm.damping(0), 30);
    EXPECT_GT(e_stiff, 100.0 * e_auto);

    // The raw (alpha = 1) map at K = 7.5 diverges outright: the two-step
    // recurrence v_{n+1} = K v_{n-1} + const amplifies the error by K = 7.5
    // every two iterations (sqrt(K) = 2.74 per iteration).
    const double e_raw0 = lagged_error(alm.k()[0], 1.0, 1);
    const double e_raw3 = lagged_error(alm.k()[0], 1.0, 3);
    EXPECT_GT(e_raw3, 5.0 * e_raw0);
}

// An unarmed controller has no penalty feedback: damping must be the
// identity so probe / legacy outer phases pass the potential through.
TEST(ConstraintAlmTest, DampingIdentityWhenUnarmed)
{
    constraint::AlmController alm;
    alm.resize(2);
    EXPECT_DOUBLE_EQ(alm.damping(0), 1.0);
    EXPECT_DOUBLE_EQ(alm.damping(1), 1.0);
}
