#include "gtest/gtest.h"
#include "source_esolver/deltap_common.h"
#include <vector>
#include <cmath>

// L1.2/L1.3 formula unit tests (2026-08-17).
//
// L1.3: compute_theta_spread — weighted std-dev of the band-resolved Wilson
// phases.  Acceptance: uniform θ gives spread = 0 exactly; a spread-out θ
// gives the closed-form weighted std-dev.
//
// L1.2: compute_smo_leakage — η = max(0, 1 − Σ_I w_In).  Acceptance:
// complete SMO (Σw = 1) gives 0; a leaky SMO (Σw < 1) gives 1 − Σw;
// an over-complete projection (Σw > 1) is clamped to 0.

namespace {

// Weighted std-dev by the direct definition.
double reference_spread(const std::vector<double>& w, const std::vector<double>& th)
{
    double wsum = 0.0, mean = 0.0;
    for (size_t i = 0; i < w.size(); ++i)
    {
        wsum += w[i];
        mean += w[i] * th[i];
    }
    mean /= wsum;
    double var = 0.0;
    for (size_t i = 0; i < w.size(); ++i)
        var += w[i] * (th[i] - mean) * (th[i] - mean);
    return std::sqrt(var / wsum);
}

} // namespace

TEST(DeltaPL1, SpreadUniformThetaIsZero)
{
    // Uniform θ (all bands carry the same Wilson phase): the H_HR proxy
    // θ_n → τ_α(I) replacement is then exact, so spread_I = 0 by definition
    // (Route A++ §1.4 推论 1 acceptance for L1.3).
    const std::vector<double> w = {0.6, 0.3, 0.1};
    const std::vector<double> theta = {1.234, 1.234, 1.234};
    EXPECT_DOUBLE_EQ(0.0, deltap_common::compute_theta_spread(w, theta));
}

TEST(DeltaPL1, SpreadMatchesWeightedStdDev)
{
    const std::vector<double> w = {0.5, 0.3, 0.2};
    const std::vector<double> theta = {0.1, 0.5, 1.3};
    const double ref = reference_spread(w, theta);
    EXPECT_NEAR(ref, deltap_common::compute_theta_spread(w, theta), 1e-14);
    // Positive-definite sanity: spread of a spread-out distribution > 0.
    EXPECT_GT(deltap_common::compute_theta_spread(w, theta), 0.0);
}

TEST(DeltaPL1, SpreadZeroWeights)
{
    // Degenerate input: zero total weight must not produce NaN.
    EXPECT_DOUBLE_EQ(0.0, deltap_common::compute_theta_spread({}, {}));
    EXPECT_DOUBLE_EQ(0.0, deltap_common::compute_theta_spread({0.0, 0.0}, {1.0, 2.0}));
}

TEST(DeltaPL1, SmoLeakage)
{
    // Complete SMO subspace: Σ_I w_In = 1 → no leakage.
    EXPECT_DOUBLE_EQ(0.0, deltap_common::compute_smo_leakage(1.0));
    // Leaky SMO: Σ_I w_In < 1 → leakage is the missing weight.
    EXPECT_NEAR(0.05, deltap_common::compute_smo_leakage(0.95), 1e-12);
    // Over-complete (non-orthogonal SMO channels, numerical Σw > 1):
    // clamped to zero leakage.
    EXPECT_DOUBLE_EQ(0.0, deltap_common::compute_smo_leakage(1.5));
    // Zero projection weight: the band is entirely outside the SMO
    // subspace → 100% leakage.
    EXPECT_DOUBLE_EQ(1.0, deltap_common::compute_smo_leakage(0.0));
}
