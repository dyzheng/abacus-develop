#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_lcao.h"

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
                = s + (i == j ? 0.5 : 0.0);
        }
    }
    return A;
}

std::vector<double> selection(const int norb, const std::vector<int>& orbital)
{
    const int ntarget = static_cast<int>(orbital.size());
    std::vector<double> C(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        C[static_cast<size_t>(orbital[a]) * ntarget + a] = 1.0;
    }
    return C;
}

} // namespace

TEST(AllocLcaoTest, TargetProjectionMatchesStateSum)
{
    std::mt19937 gen(17);
    const int norb = 8;
    const std::vector<int> orbital = {1, 5};
    const std::vector<int> atom = {0, 1};
    const int ntarget = static_cast<int>(orbital.size());
    const std::vector<double> S = random_spd(norb, gen);
    const std::vector<double> C = selection(norb, orbital);

    alloc::LcaoTargetSpace target;
    alloc::build_lcao_target(S, norb, C, ntarget, atom, target, 1e-12);
    ASSERT_EQ(target.ntarget, ntarget);
    ASSERT_EQ(target.rank, ntarget);
    EXPECT_LT(target.cond, 1e6);

    // Three arbitrary states define the model density matrices.  The state
    // coefficients are deliberately not normalized; the reference formula
    // must still agree element by element.
    const int nstate = 3;
    std::vector<std::vector<double>> states(nstate,
        std::vector<double>(norb, 0.0));
    std::vector<double> occ(nstate, 0.0);
    for (int n = 0; n < nstate; ++n)
    {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (double& c : states[n])
        {
            c = nd(gen);
        }
        occ[n] = 0.4 + 0.3 * n;
    }
    std::vector<double> D_total(static_cast<size_t>(norb) * norb, 0.0);
    std::vector<double> D_diff(static_cast<size_t>(norb) * norb, 0.0);
    for (int n = 0; n < nstate; ++n)
    {
        const double spin = (n == 1) ? -1.0 : 1.0;
        for (int i = 0; i < norb; ++i)
        {
            for (int j = 0; j < norb; ++j)
            {
                const double value = occ[n] * states[n][i] * states[n][j];
                D_total[static_cast<size_t>(i) * norb + j] += value;
                D_diff[static_cast<size_t>(i) * norb + j] += spin * value;
            }
        }
    }

    std::vector<double> n_T;
    std::vector<double> m_T;
    alloc::observe_lcao_target(target, S, D_total, D_diff, n_T, m_T);

    for (int a = 0; a < ntarget; ++a)
    {
        double n_ref = 0.0;
        double m_ref = 0.0;
        for (int n = 0; n < nstate; ++n)
        {
            double ov = 0.0;
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    ov += target.Cprime[static_cast<size_t>(i) * ntarget + a]
                          * S[static_cast<size_t>(i) * norb + j]
                          * states[n][j];
                }
            }
            const double spin = (n == 1) ? -1.0 : 1.0;
            n_ref += occ[n] * ov * ov;
            m_ref += spin * occ[n] * ov * ov;
        }
        EXPECT_NEAR(n_T[a], n_ref, 1e-10);
        EXPECT_NEAR(m_T[a], m_ref, 1e-10);
    }
}

TEST(AllocLcaoTest, SelectionTargetReconstructsOverlap)
{
    std::mt19937 gen(23);
    const int norb = 7;
    const std::vector<int> orbital = {0, 2, 4};
    const std::vector<int> atom = {0, 0, 2};
    const std::vector<double> S = random_spd(norb, gen);
    const std::vector<double> C = selection(norb, orbital);
    const int ntarget = static_cast<int>(orbital.size());

    std::vector<double> ST(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            ST[static_cast<size_t>(a) * ntarget + b]
                = S[static_cast<size_t>(orbital[a]) * norb + orbital[b]];
        }
    }

    alloc::LcaoTargetSpace target;
    alloc::finalize_lcao_target(norb, orbital, atom, ST, target, 1e-12);
    ASSERT_EQ(target.orbital, orbital);
    ASSERT_EQ(target.atom, atom);

    // Cprime^T S Cprime must be the identity in the retained target space.
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    value += target.Cprime[static_cast<size_t>(i) * ntarget + a]
                             * S[static_cast<size_t>(i) * norb + j]
                             * target.Cprime[static_cast<size_t>(j) * ntarget + b];
                }
            }
            EXPECT_NEAR(value, (a == b) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(AllocLcaoTest, RankTruncationAndCondition)
{
    const int norb = 3;
    const std::vector<int> orbital = {0, 1};
    const std::vector<int> atom = {0, 1};
    const std::vector<double> S = {4.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0};
    std::vector<double> ST = {4.0, 0.0, 0.0, 0.0};

    alloc::LcaoTargetSpace target;
    alloc::finalize_lcao_target(norb, orbital, atom, ST, target, 1e-10);
    EXPECT_EQ(target.rank, 1);
    EXPECT_TRUE(std::isinf(target.cond));
    for (int i = 0; i < norb; ++i)
    {
        for (int a = 0; a < 2; ++a)
        {
            // Branch: the singular target direction is truncated, so the
            // corresponding Cprime column must be exactly zero.
            if (a == 1)
            {
                EXPECT_DOUBLE_EQ(target.Cprime[static_cast<size_t>(i) * 2 + a],
                                 0.0);
            }
        }
    }
}
