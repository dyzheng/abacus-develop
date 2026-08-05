#include "gtest/gtest.h"

#include "../deltap_common.h"

/************************************************
 *  unit tests of deltap_common.h (pure functions)
 ***********************************************/

TEST(DeltapCommonTest, ComputeResidualPerAtom)
{
    const std::vector<double> gamma = {1.0, 2.0, 3.0};
    const std::vector<double> target = {1.5, 2.0};
    const auto r = deltap_common::compute_residual({}, target, gamma);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_DOUBLE_EQ(r[0], -0.5);
    EXPECT_DOUBLE_EQ(r[1], 0.0);
    EXPECT_DOUBLE_EQ(r[2], 3.0); // missing target treated as 0
}

TEST(DeltapCommonTest, ComputeResidualMatrix)
{
    const std::vector<std::vector<double>> C = {{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}};
    const std::vector<double> t = {1.0, 4.0};
    const auto r = deltap_common::compute_residual(C, t, {1.0, 2.0, 3.0});
    ASSERT_EQ(r.size(), 2u);
    EXPECT_DOUBLE_EQ(r[0], 0.0);
    EXPECT_DOUBLE_EQ(r[1], 1.0);
}

TEST(DeltapCommonTest, MaxNorm)
{
    EXPECT_DOUBLE_EQ(deltap_common::max_norm({}), 0.0);
    EXPECT_DOUBLE_EQ(deltap_common::max_norm({1.0, -4.0, 2.0}), 4.0);
}

TEST(DeltapCommonTest, GdUpdateMask)
{
    std::vector<double> lambda = {0.0, 0.0, 0.0};
    const std::vector<double> r = {1.0, 1.0, 1.0};
    const std::vector<int> constrain = {1, 0, 1}; // atom 1 locked
    deltap_common::gd_update(lambda, r, constrain, 0.5, 1.0);
    EXPECT_DOUBLE_EQ(lambda[0], 0.5);
    EXPECT_DOUBLE_EQ(lambda[1], 0.0);
    EXPECT_DOUBLE_EQ(lambda[2], 0.5);
}

TEST(DeltapCommonTest, GdUpdateMixing)
{
    std::vector<double> lambda = {10.0};
    deltap_common::gd_update(lambda, {2.0}, {}, 0.5, 0.2);
    EXPECT_DOUBLE_EQ(lambda[0], 10.2); // 0.2*(10+1) + 0.8*10
}

TEST(DeltapCommonTest, GdUpdateTotal)
{
    std::vector<double> lambda = {0.0, 0.0};
    deltap_common::gd_update_total(lambda, {1.0, 3.0}, 0.5, 1.0); // delta = 0.5*4
    EXPECT_DOUBLE_EQ(lambda[0], 2.0);
    EXPECT_DOUBLE_EQ(lambda[1], 2.0);
}

TEST(DeltapCommonTest, ToEffectiveLambdaIdentity)
{
    const auto out = deltap_common::to_effective_lambda({1.0, 2.0}, {}, 2);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 2.0);
}

TEST(DeltapCommonTest, ToEffectiveLambdaMatrix)
{
    const std::vector<std::vector<double>> C = {{1.0, 0.0}, {0.5, 0.5}};
    const auto out = deltap_common::to_effective_lambda({2.0, 4.0}, C, 2);
    EXPECT_DOUBLE_EQ(out[0], 4.0);
    EXPECT_DOUBLE_EQ(out[1], 2.0);
}

TEST(DeltapCommonTest, ComputeDpEscon)
{
    EXPECT_DOUBLE_EQ(deltap_common::compute_dp_escon({1.0, 2.0}, {3.0, -1.0}), -1.0);
}

// Route A+ operator mode: escon = −Σ_I λ_I·Γ_I (same function, Γ input
// vector instead of γ).  The proxy target t_Γ is not part of escon — it only
// enters the λ update residual.
TEST(DeltapCommonTest, ComputeDpEsconOperatorObservable)
{
    const std::vector<double> lambda = {0.5, -0.25, 1.0};
    const std::vector<double> gamma_op = {2.0, 4.0, -3.0};
    // escon = −(0.5·2.0 + (−0.25)·4.0 + 1.0·(−3.0)) = −(−3.0) = 3.0
    EXPECT_DOUBLE_EQ(deltap_common::compute_dp_escon(lambda, gamma_op), 3.0);
    // Per-atom Γ zero → escon zero regardless of λ.
    EXPECT_DOUBLE_EQ(deltap_common::compute_dp_escon(lambda, {0.0, 0.0, 0.0}), 0.0);
}

TEST(DeltapCommonTest, Unwrap2Pi)
{
    const double pi = ModuleBase::PI;
    // Raw values on neighbouring 2π branches; prev anchors the nearest branch.
    const auto out = deltap_common::unwrap_2pi({1.0 + 2 * pi, 2.0 - 2 * pi}, {1.1, 2.1});
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], 2.0, 1e-12);
    // Empty prev (first measurement of a SCF cycle) → unchanged.
    const auto out2 = deltap_common::unwrap_2pi({1.0, 2.0}, {});
    EXPECT_DOUBLE_EQ(out2[0], 1.0);
    EXPECT_DOUBLE_EQ(out2[1], 2.0);
}
