#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_linalg.h"

namespace
{

std::vector<double> random_spd(const int n, std::mt19937& gen)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> B(n * n);
    for (double& x : B)
    {
        x = nd(gen);
    }
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
            {
                s += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = s + ((i == j) ? 1.0 : 0.0);
        }
    }
    return A;
}

std::vector<std::complex<double>> random_hpd(const int n, std::mt19937& gen)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::complex<double>> B(n * n);
    for (std::complex<double>& value : B)
    {
        value = std::complex<double>(nd(gen), nd(gen));
    }
    std::vector<std::complex<double>> A(n * n,
                                        std::complex<double>(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::complex<double> value(0.0, 0.0);
            for (int k = 0; k < n; ++k)
            {
                value += B[i * n + k] * std::conj(B[j * n + k]);
            }
            A[i * n + j] = value + ((i == j) ? 1.0 : 0.0);
        }
    }
    return A;
}

} // namespace

TEST(AllocLinalgTest, SymEigReconstruction)
{
    std::mt19937 gen(7);
    const int n = 8;
    const std::vector<double> A = random_spd(n, gen);
    std::vector<double> w;
    std::vector<double> V;
    alloc::sym_eig(A, n, w, V);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
            {
                s += V[i * n + k] * w[k] * V[j * n + k];
            }
            EXPECT_NEAR(s, A[i * n + j], 1e-10);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
            {
                s += V[k * n + i] * V[k * n + j];
            }
            EXPECT_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
    // The public contract fixes ascending eigenvalues.
    for (int i = 1; i < n; ++i)
    {
        EXPECT_LE(w[i - 1], w[i]);
    }
}

TEST(AllocLinalgTest, InvSqrt)
{
    std::mt19937 gen(11);
    const int n = 6;
    const std::vector<double> S = random_spd(n, gen);
    std::vector<double> Si;
    alloc::sym_inv_sqrt(S, n, Si, 1e-10);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    s += Si[i * n + a] * S[a * n + b] * Si[b * n + j];
                }
            }
            EXPECT_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }
}

TEST(AllocLinalgTest, RankTruncationDropsNullDirection)
{
    const int n = 2;
    // S = diag(4, 0): the null direction must be dropped, not inverted.
    const std::vector<double> S = {4.0, 0.0, 0.0, 0.0};
    std::vector<double> Si;
    alloc::sym_inv_sqrt(S, n, Si, 1e-12);
    EXPECT_NEAR(Si[0], 0.5, 1e-12);
    EXPECT_NEAR(Si[1], 0.0, 1e-15);
    EXPECT_NEAR(Si[2], 0.0, 1e-15);
    EXPECT_NEAR(Si[3], 0.0, 1e-15);
}

TEST(AllocLinalgTest, InvSqrtDerivativeMatchesFiniteDifference)
{
    std::mt19937 gen(19);
    const int n = 5;
    const std::vector<double> S = random_spd(n, gen);
    std::vector<double> dS = random_spd(n, gen);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dS[i * n + j] = 0.5 * (dS[i * n + j] + dS[j * n + i]);
        }
    }

    std::vector<double> analytic;
    alloc::sym_inv_sqrt_derivative(S, dS, n, analytic, 1e-10);

    const double h = 1e-5;
    std::vector<double> Sp(S.size());
    std::vector<double> Sm(S.size());
    for (size_t i = 0; i < S.size(); ++i)
    {
        Sp[i] = S[i] + h * dS[i];
        Sm[i] = S[i] - h * dS[i];
    }
    std::vector<double> Gp;
    std::vector<double> Gm;
    alloc::sym_inv_sqrt(Sp, n, Gp, 1e-10);
    alloc::sym_inv_sqrt(Sm, n, Gm, 1e-10);
    for (int i = 0; i < n * n; ++i)
    {
        const double fd = (Gp[i] - Gm[i]) / (2.0 * h);
        EXPECT_NEAR(analytic[i], fd, 2e-8);
    }
}

TEST(AllocLinalgTest, EmptyAndScalarInputs)
{
    std::vector<double> w;
    std::vector<double> V;
    alloc::sym_eig({}, 0, w, V);
    EXPECT_TRUE(w.empty());
    EXPECT_TRUE(V.empty());

    alloc::sym_eig({9.0}, 1, w, V);
    ASSERT_EQ(w.size(), 1u);
    ASSERT_EQ(V.size(), 1u);
    EXPECT_DOUBLE_EQ(w[0], 9.0);
    EXPECT_DOUBLE_EQ(V[0], 1.0);
}

TEST(AllocLinalgTest, HermitianEigReconstruction)
{
    std::mt19937 gen(23);
    const int n = 6;
    const std::vector<std::complex<double>> A = random_hpd(n, gen);
    std::vector<double> w;
    std::vector<std::complex<double>> V;
    alloc::herm_eig(A, n, w, V);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::complex<double> value(0.0, 0.0);
            for (int k = 0; k < n; ++k)
            {
                value += V[i * n + k] * w[k]
                         * std::conj(V[j * n + k]);
            }
            EXPECT_NEAR(std::abs(value - A[i * n + j]), 0.0, 1e-10);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::complex<double> overlap(0.0, 0.0);
            for (int k = 0; k < n; ++k)
            {
                overlap += std::conj(V[k * n + i]) * V[k * n + j];
            }
            EXPECT_NEAR(std::abs(overlap - std::complex<double>(
                                             (i == j) ? 1.0 : 0.0, 0.0)),
                        0.0, 1e-10);
        }
    }
    for (int i = 1; i < n; ++i)
    {
        EXPECT_LE(w[i - 1], w[i]);
    }
}

TEST(AllocLinalgTest, HermitianInvSqrtDerivativeMatchesFiniteDifference)
{
    std::mt19937 gen(31);
    const int n = 5;
    const std::vector<std::complex<double>> S = random_hpd(n, gen);
    std::vector<std::complex<double>> dS = random_hpd(n, gen);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dS[i * n + j] = 0.5
                            * (dS[i * n + j]
                               + std::conj(dS[j * n + i]));
        }
    }

    std::vector<std::complex<double>> analytic;
    alloc::herm_inv_sqrt_derivative(S, dS, n, analytic, 1e-10);

    const double h = 1e-5;
    std::vector<std::complex<double>> Sp(S.size());
    std::vector<std::complex<double>> Sm(S.size());
    for (size_t i = 0; i < S.size(); ++i)
    {
        Sp[i] = S[i] + h * dS[i];
        Sm[i] = S[i] - h * dS[i];
    }
    std::vector<std::complex<double>> Gp;
    std::vector<std::complex<double>> Gm;
    alloc::herm_inv_sqrt(Sp, n, Gp, 1e-10);
    alloc::herm_inv_sqrt(Sm, n, Gm, 1e-10);
    for (int i = 0; i < n * n; ++i)
    {
        const std::complex<double> fd = (Gp[i] - Gm[i]) / (2.0 * h);
        EXPECT_NEAR(std::abs(analytic[i] - fd), 0.0, 2e-8);
    }
}
