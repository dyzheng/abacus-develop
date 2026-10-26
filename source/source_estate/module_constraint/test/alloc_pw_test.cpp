#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_constraint.h"
#include "source_estate/module_constraint/alloc_pw.h"

namespace
{

std::vector<double> random_spd(const int n, const unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> B(n * n);
    for (double& value : B)
    {
        value = nd(gen);
    }
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                A[i * n + j] += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] += (i == j) ? 1.0 : 0.0;
        }
    }
    return A;
}

std::vector<double> selection(const int norb,
                              const std::vector<int>& orbitals)
{
    const int r = static_cast<int>(orbitals.size());
    std::vector<double> C(norb * r, 0.0);
    for (int a = 0; a < r; ++a)
    {
        C[orbitals[a] * r + a] = 1.0;
    }
    return C;
}

std::vector<double> target_overlap(const std::vector<double>& S,
                                   const std::vector<double>& C,
                                   const int norb,
                                   const int ntarget)
{
    std::vector<double> ST(ntarget * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    ST[a * ntarget + b]
                        += C[i * ntarget + a] * S[i * norb + j]
                           * C[j * ntarget + b];
                }
            }
        }
    }
    return ST;
}

std::vector<double> matvec(const std::vector<double>& A,
                           const std::vector<double>& x,
                           const int n)
{
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            y[i] += A[i * n + j] * x[j];
        }
    }
    return y;
}

std::vector<double> transpose_matvec(const std::vector<double>& A,
                                     const std::vector<double>& x,
                                     const int rows,
                                     const int cols)
{
    std::vector<double> y(cols, 0.0);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            y[j] += A[i * cols + j] * x[i];
        }
    }
    return y;
}

std::vector<double> symmetric_random(const int n, const unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i; j < n; ++j)
        {
            const double value = nd(gen);
            A[i * n + j] = value;
            A[j * n + i] = value;
        }
    }
    return A;
}

} // namespace

TEST(AllocPwTest, ReadoutMatchesLcaoTargetSpace)
{
    const int norb = 8;
    const int nband = 4;
    const std::vector<double> S = random_spd(norb, 17);
    const std::vector<int> orbitals = {0, 2, 5};
    const std::vector<int> atom = {0, 0, 1};
    const std::vector<int> constraint = {0, 0, 1};
    const int ntarget = static_cast<int>(orbitals.size());
    const int nconstraint = 2;
    const std::vector<double> C = selection(norb, orbitals);
    const std::vector<double> ST = target_overlap(S, C, norb, ntarget);

    alloc::LcaoTargetSpace lcao_target;
    alloc::build_lcao_target(S, norb, C, ntarget, atom, lcao_target);
    alloc::PwTargetSpace pw_target;
    alloc::build_pw_target(ST, ntarget, atom, constraint, nconstraint,
                           pw_target);

    std::mt19937 gen(29);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> up(nband, std::vector<double>(norb));
    std::vector<std::vector<double>> dn(nband, std::vector<double>(norb));
    for (int band = 0; band < nband; ++band)
    {
        for (int i = 0; i < norb; ++i)
        {
            up[band][i] = nd(gen);
            dn[band][i] = nd(gen);
        }
    }
    const std::vector<double> occ_up = {1.0, 0.7, 0.2, 0.0};
    const std::vector<double> occ_dn = {0.8, 0.3, 0.0, 0.0};

    std::vector<double> D_total(norb * norb, 0.0);
    std::vector<double> D_diff(norb * norb, 0.0);
    for (int band = 0; band < nband; ++band)
    {
        for (int i = 0; i < norb; ++i)
        {
            for (int j = 0; j < norb; ++j)
            {
                D_total[i * norb + j]
                    += occ_up[band] * up[band][i] * up[band][j];
                D_total[i * norb + j]
                    += occ_dn[band] * dn[band][i] * dn[band][j];
                D_diff[i * norb + j]
                    += occ_up[band] * up[band][i] * up[band][j];
                D_diff[i * norb + j]
                    -= occ_dn[band] * dn[band][i] * dn[band][j];
            }
        }
    }
    std::vector<double> lcao_nT;
    std::vector<double> lcao_mT;
    alloc::observe_lcao_target(lcao_target, S, D_total, D_diff, lcao_nT,
                               lcao_mT);

    std::vector<std::vector<std::complex<double>>> becp_up(
        nband, std::vector<std::complex<double>>(ntarget, 0.0));
    std::vector<std::vector<std::complex<double>>> becp_dn(
        nband, std::vector<std::complex<double>>(ntarget, 0.0));
    for (int band = 0; band < nband; ++band)
    {
        const std::vector<double> raw_up
            = transpose_matvec(S, up[band], norb, norb);
        const std::vector<double> raw_dn
            = transpose_matvec(S, dn[band], norb, norb);
        for (int a = 0; a < ntarget; ++a)
        {
            double ou = 0.0;
            double od = 0.0;
            for (int i = 0; i < norb; ++i)
            {
                ou += C[i * ntarget + a] * raw_up[i];
                od += C[i * ntarget + a] * raw_dn[i];
            }
            becp_up[band][a] = ou;
            becp_dn[band][a] = od;
        }
    }
    const alloc::PwTargetMoments pw
        = alloc::observe_pw_target_moments(pw_target, becp_up, occ_up,
                                           becp_dn, occ_dn);

    double max_dn = 0.0;
    double max_dm = 0.0;
    for (int a = 0; a < ntarget; ++a)
    {
        max_dn = std::max(max_dn, std::abs(pw.n_T_target[a] - lcao_nT[a]));
        max_dm = std::max(max_dm, std::abs(pw.m_T_target[a] - lcao_mT[a]));
    }
    printf("alloc_pw parity: max_dnT=%.3e max_dmT=%.3e\n", max_dn, max_dm);
    EXPECT_LT(max_dn, 1e-10);
    EXPECT_LT(max_dm, 1e-10);
    // The S5 gate is intentionally looser than the algebraic equality above.
    EXPECT_LT(max_dm, 0.02);
}

TEST(AllocPwTest, CompleteTargetSumRules)
{
    const int norb = 7;
    const int nband = norb;
    const std::vector<double> S = random_spd(norb, 41);
    std::vector<int> orbitals(norb);
    std::vector<int> atom(norb, 0);
    std::vector<int> constraint(norb, 0);
    for (int i = 0; i < norb; ++i)
    {
        orbitals[i] = i;
    }
    const std::vector<double> C = selection(norb, orbitals);
    const std::vector<double> ST = target_overlap(S, C, norb, norb);
    alloc::PwTargetSpace space;
    alloc::build_pw_target(ST, norb, atom, constraint, 1, space);

    std::mt19937 gen(53);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<std::complex<double>>> becp_up(
        nband, std::vector<std::complex<double>>(norb, 0.0));
    std::vector<std::vector<std::complex<double>>> becp_dn(
        nband, std::vector<std::complex<double>>(norb, 0.0));
    std::vector<double> occ_up(nband);
    std::vector<double> occ_dn(nband);
    double expected_n = 0.0;
    double expected_m = 0.0;
    for (int band = 0; band < nband; ++band)
    {
        std::vector<double> cu(norb);
        std::vector<double> cd(norb);
        for (int i = 0; i < norb; ++i)
        {
            cu[i] = nd(gen);
            cd[i] = nd(gen);
        }
        occ_up[band] = 0.2 + 0.8 * (band + 1) / nband;
        occ_dn[band] = 0.1 + 0.5 * (band + 1) / nband;
        std::vector<double> Su = matvec(S, cu, norb);
        std::vector<double> Sd = matvec(S, cd, norb);
        double nu = 0.0;
        double ndn = 0.0;
        for (int i = 0; i < norb; ++i)
        {
            nu += cu[i] * Su[i];
            ndn += cd[i] * Sd[i];
            becp_up[band][i] = Su[i];
            becp_dn[band][i] = Sd[i];
        }
        expected_n += occ_up[band] * nu + occ_dn[band] * ndn;
        expected_m += occ_up[band] * nu - occ_dn[band] * ndn;
    }
    const alloc::PwTargetMoments pw
        = alloc::observe_pw_target_moments(space, becp_up, occ_up, becp_dn,
                                           occ_dn);
    printf("alloc_pw sum: n=%.12e/%.12e m=%.12e/%.12e dsum=%.3e\n",
           pw.n_T_total, expected_n, pw.m_T_total, expected_m,
           std::abs(pw.n_T_constraint[0] - pw.n_T_total));
    EXPECT_NEAR(pw.n_T_total, expected_n, 1e-10);
    EXPECT_NEAR(pw.m_T_total, expected_m, 1e-10);
    EXPECT_NEAR(pw.n_T_constraint[0], pw.n_T_total, 1e-12);
    EXPECT_NEAR(pw.m_T_constraint[0], pw.m_T_total, 1e-12);
}

TEST(AllocPwTest, ApplyDeltaHMatchesLcaoKernel)
{
    const int norb = 7;
    const std::vector<int> orbitals = {0, 2, 4};
    const std::vector<int> atom = {0, 0, 1};
    const std::vector<int> constraint = {0, 0, 1};
    const int ntarget = static_cast<int>(orbitals.size());
    const int nconstraint = 2;
    const std::vector<double> S(norb * norb, 0.0);
    std::vector<double> Sn = S;
    for (int i = 0; i < norb; ++i)
    {
        Sn[i * norb + i] = 1.0;
    }
    const std::vector<double> C = selection(norb, orbitals);
    const std::vector<double> ST = target_overlap(Sn, C, norb, ntarget);

    alloc::AllocConstraintAlgebra algebra;
    alloc::build_alloc_constraint_algebra(Sn, norb, orbitals, atom, constraint,
                                          ST, nconstraint, algebra);
    alloc::PwTargetSpace pw_space;
    alloc::build_pw_target(ST, ntarget, atom, constraint, nconstraint, pw_space);

    const std::vector<std::vector<double>> W = {
        symmetric_random(norb, 61), symmetric_random(norb, 67)};
    const std::vector<double> lambda = {0.13, -0.07};
    std::vector<double> K;
    alloc::build_alloc_delta_h(algebra, W, lambda, K);

    std::mt19937 gen(71);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> psi(norb);
    std::vector<double> V(norb * norb, 0.0);
    for (int i = 0; i < norb; ++i)
    {
        psi[i] = nd(gen);
        for (int j = 0; j < norb; ++j)
        {
            V[i * norb + j] = lambda[0] * W[0][i * norb + j]
                              + lambda[1] * W[1][i * norb + j];
        }
    }
    const std::vector<double> expected = matvec(K, psi, norb);

    std::vector<std::complex<double>> psi_c(norb);
    std::vector<std::complex<double>> vpsi_c(norb);
    const std::vector<double> vpsi = matvec(V, psi, norb);
    for (int i = 0; i < norb; ++i)
    {
        psi_c[i] = psi[i];
        vpsi_c[i] = vpsi[i];
    }
    std::vector<std::complex<double>> raw_overlap(ntarget, 0.0);
    std::vector<std::vector<std::complex<double>>> phi(
        ntarget, std::vector<std::complex<double>>(norb, 0.0));
    std::vector<std::vector<std::complex<double>>> vphi(
        ntarget, std::vector<std::complex<double>>(norb, 0.0));
    std::vector<std::complex<double>> m_raw(ntarget * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        raw_overlap[a] = psi[orbitals[a]];
        for (int i = 0; i < norb; ++i)
        {
            phi[a][i] = C[i * ntarget + a];
            for (int j = 0; j < norb; ++j)
            {
                vphi[a][i] += V[i * norb + j] * C[j * ntarget + a];
            }
        }
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    value += C[i * ntarget + a] * V[i * norb + j]
                             * C[j * ntarget + b];
                }
            }
            m_raw[a * ntarget + b] = value;
        }
    }

    std::vector<std::complex<double>> delta;
    alloc::apply_pw_alloc_delta_h(pw_space, phi, psi_c, raw_overlap, vpsi_c,
                                  vphi, m_raw, lambda, delta);
    double max_diff = 0.0;
    for (int i = 0; i < norb; ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(delta[i] - std::complex<double>(expected[i], 0.0)));
    }
    printf("alloc_pw apply: max_diff=%.3e\n", max_diff);
    EXPECT_LT(max_diff, 1e-10);
}

TEST(AllocPwTest, ProjectorTabulationBuildsOverlapAndBecp)
{
    const int npw = 11;
    const int ntarget = 3;
    const int nband = 2;
    const int ld_psi = 14;
    std::mt19937 gen(83);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::complex<double>> tab(ntarget * npw);
    std::vector<std::complex<double>> psi(nband * ld_psi,
                                          std::complex<double>(0.0, 0.0));
    for (std::complex<double>& value : tab)
    {
        value = std::complex<double>(nd(gen), nd(gen));
    }
    for (int band = 0; band < nband; ++band)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            psi[band * ld_psi + ig] = std::complex<double>(nd(gen), nd(gen));
        }
    }

    std::vector<double> ST;
    alloc::build_pw_projector_overlap(tab.data(), npw, ntarget, ST);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < npw; ++ig)
            {
                value += std::conj(tab[a * npw + ig]) * tab[b * npw + ig];
            }
            EXPECT_NEAR(ST[a * ntarget + b], value.real(), 1e-13);
        }
    }

    std::vector<std::vector<std::complex<double>>> becp;
    alloc::pw_becp_from_projectors(tab.data(), npw, ntarget, psi.data(),
                                   ld_psi, nband, becp);
    for (int band = 0; band < nband; ++band)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < npw; ++ig)
            {
                value += std::conj(tab[a * npw + ig])
                         * psi[band * ld_psi + ig];
            }
            EXPECT_NEAR(std::abs(becp[band][a] - value), 0.0, 1e-12);
        }
    }
}
