#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_constraint.h"
#include "source_estate/module_constraint/alloc_linalg.h"

namespace
{

std::vector<double> random_spd(const int n, std::mt19937& gen)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> B(static_cast<size_t>(n) * n);
    for (double& x : B)
    {
        x = nd(gen);
    }
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
            {
                s += B[static_cast<size_t>(i) * n + k]
                     * B[static_cast<size_t>(j) * n + k];
            }
            A[static_cast<size_t>(i) * n + j]
                = s + (i == j ? 0.7 : 0.0);
        }
    }
    return A;
}

std::vector<double> inverse_spd(const std::vector<double>& A, const int n)
{
    std::vector<double> w;
    std::vector<double> V;
    alloc::sym_eig(A, n, w, V);
    std::vector<double> inv(static_cast<size_t>(n) * n, 0.0);
    for (int a = 0; a < n; ++a)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                inv[static_cast<size_t>(i) * n + j]
                    += V[static_cast<size_t>(i) * n + a] / w[a]
                       * V[static_cast<size_t>(j) * n + a];
            }
        }
    }
    return inv;
}

std::vector<double> matmul(const std::vector<double>& A,
                           const std::vector<double>& B,
                           const int n,
                           const int m,
                           const int p)
{
    std::vector<double> C(static_cast<size_t>(n) * p, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < m; ++k)
        {
            const double a = A[static_cast<size_t>(i) * m + k];
            for (int j = 0; j < p; ++j)
            {
                C[static_cast<size_t>(i) * p + j]
                    += a * B[static_cast<size_t>(k) * p + j];
            }
        }
    }
    return C;
}

std::vector<double> projector(const std::vector<double>& S,
                              const std::vector<double>& B,
                              const int norb,
                              const int ntarget,
                              const std::vector<int>& columns = {})
{
    // P = S B B^T S; an empty column list means all target columns.
    std::vector<double> Cprime = B;
    if (!columns.empty())
    {
        Cprime.assign(static_cast<size_t>(norb) * columns.size(), 0.0);
        for (int a = 0; a < static_cast<int>(columns.size()); ++a)
        {
            for (int i = 0; i < norb; ++i)
            {
                Cprime[static_cast<size_t>(i) * columns.size() + a]
                    = B[static_cast<size_t>(i) * ntarget + columns[a]];
            }
        }
    }
    const int ncol = static_cast<int>(Cprime.size() / norb);
    const std::vector<double> SB = matmul(S, Cprime, norb, norb, ncol);
    std::vector<double> BtS(static_cast<size_t>(ncol) * norb, 0.0);
    for (int a = 0; a < ncol; ++a)
    {
        for (int j = 0; j < norb; ++j)
        {
            BtS[static_cast<size_t>(a) * norb + j]
                = SB[static_cast<size_t>(j) * ncol + a];
        }
    }
    return matmul(SB, BtS, norb, ncol, norb);
}

} // namespace

TEST(AllocConstraintTest, KernelMatchesDenseOperator)
{
    std::mt19937 gen(31);
    const int norb = 7;
    const std::vector<int> orbital = {1, 3};
    const std::vector<int> atom = {0, 1};
    const std::vector<int> owner = {0, 1};
    const int ntarget = 2;
    const std::vector<double> S = random_spd(norb, gen);
    std::vector<double> ST = {S[1 * norb + 1], S[1 * norb + 3],
                              S[3 * norb + 1], S[3 * norb + 3]};

    alloc::AllocConstraintAlgebra space;
    alloc::build_alloc_constraint_algebra(S, norb, orbital, atom, owner, ST,
                                          2, space, 1e-12);
    std::vector<double> W(static_cast<size_t>(norb) * norb);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < norb; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            const double value = nd(gen);
            W[static_cast<size_t>(i) * norb + j] = value;
            W[static_cast<size_t>(j) * norb + i] = value;
        }
    }

    std::vector<double> K;
    alloc::build_alloc_constraint_kernel(space, W, 0, K);
    const std::vector<double> Sinv = inverse_spd(S, norb);
    const std::vector<double> P = projector(S, space.target.Cprime, norb,
                                            ntarget);
    const std::vector<double> Qi = projector(S, space.target.Cprime, norb,
                                             ntarget, {0});
    const std::vector<double> PSW = matmul(P, matmul(Sinv, W, norb, norb, norb),
                                           norb, norb, norb);
    const std::vector<double> WSP = matmul(matmul(W, Sinv, norb, norb, norb), P,
                                           norb, norb, norb);
    const std::vector<double> PSWSP = matmul(
        matmul(P, matmul(Sinv, matmul(W, Sinv, norb, norb, norb), norb, norb,
                         norb),
               norb, norb, norb),
        P, norb, norb, norb);
    for (int i = 0; i < norb; ++i)
    {
        for (int j = 0; j < norb; ++j)
        {
            const size_t ij = static_cast<size_t>(i) * norb + j;
            const double ref = W[ij] - PSW[ij] - WSP[ij] + PSWSP[ij] + Qi[ij];
            EXPECT_NEAR(K[ij], ref, 1e-10);
            EXPECT_NEAR(K[ij], K[static_cast<size_t>(j) * norb + i], 1e-10);
        }
    }
}

TEST(AllocConstraintTest, KernelSumIsOverlap)
{
    std::mt19937 gen(41);
    const int norb = 6;
    const std::vector<int> orbital = {0, 2, 4};
    const std::vector<int> atom = {0, 1, 2};
    const std::vector<int> owner = {0, 1, 2};
    const int ntarget = 3;
    const std::vector<double> S = random_spd(norb, gen);
    std::vector<double> ST(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            ST[static_cast<size_t>(a) * ntarget + b]
                = S[static_cast<size_t>(orbital[a]) * norb + orbital[b]];
        }
    }
    alloc::AllocConstraintAlgebra space;
    alloc::build_alloc_constraint_algebra(S, norb, orbital, atom, owner, ST,
                                          3, space, 1e-12);

    // A partition of unity in operator space: sum_i W_i = S.  Then
    // sum_i K_i = S, independent of the target set.
    std::vector<std::vector<double>> W(3,
        std::vector<double>(static_cast<size_t>(norb) * norb, 0.0));
    for (int i = 0; i < norb; ++i)
    {
        for (int j = 0; j < norb; ++j)
        {
            const double s = S[static_cast<size_t>(i) * norb + j];
            W[0][static_cast<size_t>(i) * norb + j] = 0.2 * s;
            W[1][static_cast<size_t>(i) * norb + j] = 0.3 * s;
            W[2][static_cast<size_t>(i) * norb + j] = 0.5 * s;
        }
    }
    const std::vector<double> lambda = {1.0, 1.0, 1.0};
    std::vector<double> delta_h;
    alloc::build_alloc_delta_h(space, W, lambda, delta_h);
    for (size_t i = 0; i < S.size(); ++i)
    {
        EXPECT_NEAR(delta_h[i], S[i], 1e-10);
    }
}

TEST(AllocConstraintTest, ZeroMultiplierIsZeroCorrection)
{
    std::mt19937 gen(43);
    const int norb = 5;
    const std::vector<int> orbital = {2};
    const std::vector<int> atom = {0};
    const std::vector<int> owner = {0};
    const std::vector<double> S = random_spd(norb, gen);
    std::vector<double> ST = {S[2 * norb + 2]};
    alloc::AllocConstraintAlgebra space;
    alloc::build_alloc_constraint_algebra(S, norb, orbital, atom, owner, ST,
                                          1, space, 1e-12);
    std::vector<std::vector<double>> W(1,
        std::vector<double>(static_cast<size_t>(norb) * norb, 0.0));
    std::vector<double> delta_h;
    alloc::build_alloc_delta_h(space, W, {0.0}, delta_h);
    for (const double value : delta_h)
    {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}
