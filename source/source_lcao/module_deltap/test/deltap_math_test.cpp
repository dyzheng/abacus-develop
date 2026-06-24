#include "gtest/gtest.h"
#include "source_base/vector3.h"
#include "source_base/constants.h"
#include <complex>
#include <vector>
#include <cmath>

// Test the phase-summing formula:
// S(k) = sum_R e^{ikR} * overlap(R)
// dS(k,alpha) = sum_R i*R_alpha*e^{ikR} * overlap(R)
//
// We verify: dS(k,alpha) analytic matches finite-difference of S(k)

namespace {

std::complex<double> compute_S(double k, const std::vector<std::pair<double, double>>& overlaps)
{
    std::complex<double> s = {0.0, 0.0};
    for (size_t i = 0; i < overlaps.size(); ++i)
    {
        double R = overlaps[i].first;
        double val = overlaps[i].second;
        double arg = k * R;
        s += std::complex<double>(std::cos(arg), std::sin(arg)) * val;
    }
    return s;
}

std::complex<double> compute_dS_analytic(double k, int alpha,
    const std::vector<std::pair<double, std::vector<double>>>& overlaps)
{
    std::complex<double> ds = {0.0, 0.0};
    for (size_t i = 0; i < overlaps.size(); ++i)
    {
        double R = overlaps[i].first;
        const std::vector<double>& val = overlaps[i].second;
        double arg = k * R;
        std::complex<double> phase(std::cos(arg), std::sin(arg));
        std::complex<double> i_R(0.0, R);
        ds += i_R * phase * val[alpha];
    }
    return ds;
}

std::complex<double> compute_dS_finite_diff(double k, double dk,
    const std::vector<std::pair<double, double>>& overlaps)
{
    std::complex<double> s_plus = compute_S(k + dk, overlaps);
    std::complex<double> s_minus = compute_S(k - dk, overlaps);
    return (s_plus - s_minus) / (2.0 * dk);
}

} // anonymous namespace

class DeltaPMathTest : public testing::Test
{
protected:
    std::vector<std::pair<double, double>> overlaps_1d;
    double dk_;

    void SetUp() override
    {
        overlaps_1d.push_back(std::make_pair(-2.0, 0.01));
        overlaps_1d.push_back(std::make_pair(-1.0, 0.15));
        overlaps_1d.push_back(std::make_pair(0.0, 1.00));
        overlaps_1d.push_back(std::make_pair(1.0, 0.15));
        overlaps_1d.push_back(std::make_pair(2.0, 0.01));
        dk_ = 1e-6;
    }
};

TEST_F(DeltaPMathTest, SSumConsistency)
{
    double s0_real = compute_S(0.0, overlaps_1d).real();
    double expected = 0.01 + 0.15 + 1.00 + 0.15 + 0.01;
    EXPECT_NEAR(s0_real, expected, 1e-12);
}

TEST_F(DeltaPMathTest, AnalyticDSMatchesFiniteDiff)
{
    std::vector<std::pair<double, std::vector<double>>> overlaps_vec;
    for (size_t i = 0; i < overlaps_1d.size(); ++i)
    {
        std::vector<double> v;
        v.push_back(overlaps_1d[i].second);
        v.push_back(0.0);
        v.push_back(0.0);
        overlaps_vec.push_back(std::make_pair(overlaps_1d[i].first, v));
    }

    double k_tests[] = {0.1, 0.5, 1.0, 1.5, 2.0, 3.14159};
    for (int t = 0; t < 6; ++t)
    {
        double k = k_tests[t];
        std::complex<double> ds_analytic = compute_dS_analytic(k, 0, overlaps_vec);
        std::complex<double> ds_fd = compute_dS_finite_diff(k, dk_, overlaps_1d);

        double abs_err = std::abs(ds_analytic - ds_fd);
        double tol = 1e-8 * (std::abs(ds_fd) + 1.0);
        EXPECT_LT(abs_err, tol)
            << "k=" << k << " analytic=" << ds_analytic << " fd=" << ds_fd
            << " abs_err=" << abs_err << " tol=" << tol;
    }
}

TEST_F(DeltaPMathTest, BerryConnectionGaugeInvariance)
{
    std::complex<double> d_alpha_psi(0.2, -0.1);
    std::complex<double> alpha_psi(0.5, 0.2);

    std::complex<double> A = std::conj(d_alpha_psi) * alpha_psi
                           + std::conj(alpha_psi) * d_alpha_psi;

    double phi = 0.37;
    std::complex<double> phase(std::cos(phi), std::sin(phi));
    std::complex<double> d_alpha_psi_g = phase * d_alpha_psi;
    std::complex<double> alpha_psi_g = phase * alpha_psi;

    std::complex<double> A_g = std::conj(d_alpha_psi_g) * alpha_psi_g
                             + std::conj(alpha_psi_g) * d_alpha_psi_g;

    EXPECT_NEAR(A.imag(), 0.0, 1e-12);
    EXPECT_NEAR(A_g.imag(), 0.0, 1e-12);
    EXPECT_NEAR(A.real(), A_g.real(), 1e-12);
}
