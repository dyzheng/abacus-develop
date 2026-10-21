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
    EXPECT_NEAR(alm.k()[0], 7.5 / 0.65, 1e-12);
    EXPECT_NEAR(alm.k()[1], 7.5 / 24.0, 1e-12);
}

// mu_eff: before arming it is plain mu; after arming it is mu + k(Q - t).
TEST(ConstraintAlmTest, MuEffBeforeAndAfterArming)
{
    constraint::AlmController alm;
    alm.resize(1);
    std::vector<double> eff;
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_DOUBLE_EQ(eff[0], 0.2); // unarmed: penalty inactive
    alm.arm({-1.0}); // k = 7.5
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_NEAR(eff[0], 0.2 + 7.5 * 1.0, 1e-12);
}

// Outer update: mu += k(Q - t), capped per component; a pinned component
// with a flat residual fuses (UNREACHABLE semantics mirrored from MuSolver).
TEST(ConstraintAlmTest, UpdateCapsAndFuses)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-1.0}); // k = 7.5
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
    alm.arm({-2.0}); // k = 3.75
    const double e = alm.energy_correction({0.5}, {1.2}, {1.0});
    EXPECT_NEAR(e, 0.5 * 0.2 + 0.5 * 3.75 * 0.04, 1e-12);
}

// Linear-model contraction (design sec. 6.2): with k*chi = 7.5 the outer
// update contracts |mu - mu*| by 1/|1 - k*chi| = 1/6.5 per step; mu_phys
// after convergence equals the exact multiplier.
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
    for (int step = 0; step < 10; ++step)
    {
        const double k = alm.k()[0];
        const double q = (q_ref + chi * (mu[0] - k * t)) / (1.0 - chi * k);
        const double res = std::abs(q - t);
        if (res < 1e-4)
        {
            break;
        }
        alm.update({q}, {t}, {50.0}, mu);
        ASSERT_LT(step, 5); // must converge in <= 3-4 updates
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
    alm.arm({-0.075}); // k = 100 -> at the k_max cap
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
    alm.arm({-0.1}); // auto would be 75, explicit wins
    EXPECT_NEAR(alm.k()[0], 3.0, 1e-12);
    alm.set_k_override(1000.0);
    alm.arm({-0.1});
    EXPECT_NEAR(alm.k()[0], 100.0, 1e-12);
}
