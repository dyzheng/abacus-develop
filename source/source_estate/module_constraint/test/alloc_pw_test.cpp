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

TEST(AllocPwTest, SelectOnsiteTargetShell)
{
    // OnsiteProjector uses nh = (nwl+1)^2 rows per atom.  O (nwl=2) therefore
    // carries l=0,1,2 while H (nwl=1) carries only l=0,1.
    const std::vector<int> nh = {9, 4, 4};
    const alloc::PwTargetShell d_shell
        = alloc::select_pw_target_shell(nh, 2);
    ASSERT_EQ(d_shell.row.size(), 5);
    ASSERT_EQ(d_shell.atom.size(), 5);
    for (int m = 0; m < 5; ++m)
    {
        EXPECT_EQ(d_shell.row[m], 4 + m);
        EXPECT_EQ(d_shell.atom[m], 0);
    }

    const alloc::PwTargetShell p_shell
        = alloc::select_pw_target_shell(nh, 1);
    ASSERT_EQ(p_shell.row.size(), 9);
    const std::vector<int> expected_rows = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    const std::vector<int> expected_atoms = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    EXPECT_EQ(p_shell.row, expected_rows);
    EXPECT_EQ(p_shell.atom, expected_atoms);

    // A shell beyond every atom's lmax produces the empty P=0 target set.
    const alloc::PwTargetShell f_shell
        = alloc::select_pw_target_shell(nh, 3);
    EXPECT_TRUE(f_shell.row.empty());
    EXPECT_TRUE(f_shell.atom.empty());
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
    alloc::PwAllocOrthogonalContext orthogonal;
    alloc::build_pw_alloc_orthogonal_context(pw_space, phi, vphi, m_raw,
                                              orthogonal);
    std::vector<std::complex<double>> delta_orthogonal;
    alloc::apply_pw_alloc_delta_h_orthogonal(
        pw_space, orthogonal, vphi, psi_c, raw_overlap, vpsi_c, lambda,
        delta_orthogonal);
    double max_diff = 0.0;
    double max_prepared_diff = 0.0;
    for (int i = 0; i < norb; ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(delta[i] - std::complex<double>(expected[i], 0.0)));
        max_prepared_diff = std::max(
            max_prepared_diff,
            std::abs(delta_orthogonal[i] - delta[i]));
    }
    printf("alloc_pw apply: max_diff=%.3e\n", max_diff);
    printf("alloc_pw prepared apply: max_diff=%.3e\n", max_prepared_diff);
    EXPECT_LT(max_diff, 1e-10);
    EXPECT_LT(max_prepared_diff, 1e-13);
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

TEST(AllocPwTest, TargetDerivativeMatchesFiniteDifference)
{
    const int ntarget = 4;
    const int nbasis = 7;
    std::mt19937 gen(97);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<std::complex<double>>> phi(
        ntarget, std::vector<std::complex<double>>(nbasis));
    std::vector<std::vector<std::complex<double>>> dphi(
        ntarget, std::vector<std::complex<double>>(nbasis));
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            // The current target representation is the real Gamma-point
            // overlap, so both the projector and its displacement are real.
            phi[a][i] = std::complex<double>(nd(gen), 0.0);
            dphi[a][i] = std::complex<double>(nd(gen), 0.0);
        }
    }

    std::vector<std::complex<double>> derivative_overlap(ntarget * ntarget,
                                                        std::complex<double>(0.0, 0.0));
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                derivative_overlap[a * ntarget + b]
                    += std::conj(dphi[a][i]) * phi[b][i];
            }
        }
    }

    std::vector<std::complex<double>> flat(ntarget * nbasis);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            flat[a * nbasis + i] = phi[a][i];
        }
    }
    std::vector<double> ST;
    alloc::build_pw_projector_overlap(flat.data(), nbasis, ntarget, ST);
    alloc::PwTargetSpace space;
    alloc::build_pw_target(ST, ntarget, {0, 0, 1, 1}, {0, 0, 1, 1}, 2,
                           space);

    std::vector<double> dG;
    alloc::build_pw_target_derivative(space, derivative_overlap, dG);

    std::vector<std::complex<double>> raw(ntarget);
    std::vector<std::complex<double>> d_raw(ntarget);
    std::vector<std::complex<double>> z(nbasis);
    for (int i = 0; i < nbasis; ++i)
    {
        z[i] = std::complex<double>(nd(gen), nd(gen));
    }
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            raw[a] += std::conj(phi[a][i]) * z[i];
            d_raw[a] += std::conj(dphi[a][i]) * z[i];
        }
    }
    std::vector<std::complex<double>> orth;
    std::vector<std::complex<double>> d_orth;
    alloc::orthogonalize_pw_overlaps_derivative(space, dG, raw, d_raw, orth,
                                                 d_orth);

    const double h = 1e-6;
    std::vector<std::vector<std::complex<double>>> phi_p = phi;
    std::vector<std::vector<std::complex<double>>> phi_m = phi;
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            phi_p[a][i] += h * dphi[a][i];
            phi_m[a][i] -= h * dphi[a][i];
        }
    }
    auto orthogonal_fd = [&](const std::vector<std::vector<std::complex<double>>>& f,
                             std::vector<std::complex<double>>& out) {
        std::vector<std::complex<double>> tab(ntarget * nbasis);
        for (int a = 0; a < ntarget; ++a)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                tab[a * nbasis + i] = f[a][i];
            }
        }
        std::vector<double> st;
        alloc::build_pw_projector_overlap(tab.data(), nbasis, ntarget, st);
        alloc::PwTargetSpace sp;
        alloc::build_pw_target(st, ntarget, {0, 0, 1, 1}, {0, 0, 1, 1}, 2,
                               sp);
        std::vector<std::vector<std::complex<double>>> ph;
        alloc::build_pw_alloc_orthogonal_phi(sp, f, ph);
        out.assign(ntarget, std::complex<double>(0.0, 0.0));
        for (int a = 0; a < ntarget; ++a)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                out[a] += std::conj(ph[a][i]) * z[i];
            }
        }
    };
    std::vector<std::complex<double>> p_plus;
    std::vector<std::complex<double>> p_minus;
    orthogonal_fd(phi_p, p_plus);
    orthogonal_fd(phi_m, p_minus);
    double max_diff = 0.0;
    for (int a = 0; a < ntarget; ++a)
    {
        const std::complex<double> fd = (p_plus[a] - p_minus[a]) / (2.0 * h);
        max_diff = std::max(max_diff, std::abs(fd - d_orth[a]));
    }
    printf("alloc_pw target derivative: max_diff=%.3e\n", max_diff);
    EXPECT_LT(max_diff, 1e-8);
}

TEST(AllocPwTest, AllocationOperatorDerivativeDecomposition)
{
    const int ntarget = 3;
    const int nbasis = 6;
    const int nconstraint = 2;
    const std::vector<int> atom = {0, 0, 1};
    const std::vector<int> owner = {0, 0, 1};
    std::mt19937 gen(113);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<std::complex<double>>> f(
        ntarget, std::vector<std::complex<double>>(nbasis));
    std::vector<std::vector<std::complex<double>>> df(
        ntarget, std::vector<std::complex<double>>(nbasis));
    std::vector<std::complex<double>> psi(nbasis);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            f[a][i] = std::complex<double>(nd(gen), 0.0);
            df[a][i] = std::complex<double>(nd(gen), 0.0);
            // Branch: this test displaces only the first atom, so projectors
            // centred on the second atom have zero raw derivative.
            if (atom[a] != 0)
            {
                df[a][i] = std::complex<double>(0.0, 0.0);
            }
        }
    }
    for (int i = 0; i < nbasis; ++i)
    {
        psi[i] = std::complex<double>(nd(gen), nd(gen));
    }
    std::vector<double> W(static_cast<size_t>(nconstraint) * nbasis * nbasis,
                          0.0);
    for (int alpha = 0; alpha < nconstraint; ++alpha)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            W[(alpha * nbasis + i) * nbasis + i]
                = 0.5 + 0.1 * (alpha * nbasis + i + 1);
        }
    }

    std::vector<std::complex<double>> flat(ntarget * nbasis);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            flat[a * nbasis + i] = f[a][i];
        }
    }
    std::vector<double> ST;
    alloc::build_pw_projector_overlap(flat.data(), nbasis, ntarget, ST);
    alloc::PwTargetSpace space;
    alloc::build_pw_target(ST, ntarget, atom, owner, nconstraint, space);
    std::vector<std::vector<std::complex<double>>> phi;
    alloc::build_pw_alloc_orthogonal_phi(space, f, phi);

    std::vector<std::complex<double>> D(ntarget * ntarget,
                                       std::complex<double>(0.0, 0.0));
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: only projectors on the displaced atom move.
        if (atom[a] != 0)
        {
            continue;
        }
        for (int b = 0; b < ntarget; ++b)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                D[a * ntarget + b] += std::conj(df[a][i]) * f[b][i];
            }
        }
    }
    std::vector<double> dG;
    alloc::build_pw_target_derivative(space, D, dG);

    std::vector<std::complex<double>> raw(ntarget);
    std::vector<std::complex<double>> d_raw(ntarget);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            // raw is the overlap with the un-orthogonalized projector.  Using
            // phi here would apply the inverse square root a second time.
            raw[a] += std::conj(f[a][i]) * psi[i];
            d_raw[a] += std::conj(df[a][i]) * psi[i];
        }
    }
    std::vector<std::complex<double>> orth;
    std::vector<std::complex<double>> d_orth;
    alloc::orthogonalize_pw_overlaps_derivative(space, dG, raw, d_raw, orth,
                                                 d_orth);

    std::vector<std::complex<double>> q = psi;
    for (int a = 0; a < ntarget; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            q[i] -= phi[a][i] * orth[a];
        }
    }
    // dP psi = sum_a (o_a dphi_a + d_o_a phi_a); build dphi from
    // dphi = f G, so dphi = df G + f dG.  The rotation G changes every
    // orthonormal target, including those centred on the other atom.
    std::vector<std::complex<double>> dPpsi(nbasis,
                                            std::complex<double>(0.0, 0.0));
    for (int a = 0; a < ntarget; ++a)
    {
        std::vector<std::complex<double>> dphi(nbasis,
                                               std::complex<double>(0.0, 0.0));
        for (int b = 0; b < ntarget; ++b)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                dphi[i] += space.ST_inv_root[a * ntarget + b] * df[b][i];
                dphi[i] += dG[a * ntarget + b] * f[b][i];
            }
        }
        for (int i = 0; i < nbasis; ++i)
        {
            dPpsi[i] += orth[a] * dphi[i] + d_orth[a] * phi[a][i];
        }
    }

    for (int alpha = 0; alpha < nconstraint; ++alpha)
    {
        std::complex<double> cross(0.0, 0.0);
        for (int i = 0; i < nbasis; ++i)
        {
            std::complex<double> wq(0.0, 0.0);
            for (int j = 0; j < nbasis; ++j)
            {
                wq += W[(alpha * nbasis + i) * nbasis + j] * q[j];
            }
            cross += std::conj(dPpsi[i]) * wq;
        }
        double dQ = 0.0;
        for (int a = 0; a < ntarget; ++a)
        {
            // Branch: Q_i contains every orthonormal target owned by this
            // constraint, including targets rotated indirectly by dG.
            if (owner[a] == alpha)
            {
                dQ += (std::conj(d_orth[a]) * orth[a]).real();
            }
        }
        dQ *= 2.0;
        const double analytic = dQ - 2.0 * cross.real();

        auto expectation = [&](const std::vector<std::vector<std::complex<double>>>& ph) {
            std::vector<std::complex<double>> P(
                nbasis * nbasis, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < ntarget; ++a)
            {
                for (int i = 0; i < nbasis; ++i)
                {
                    for (int j = 0; j < nbasis; ++j)
                    {
                        P[i * nbasis + j] += ph[a][i] * std::conj(ph[a][j]);
                    }
                }
            }
            std::vector<std::complex<double>> A(
                nbasis * nbasis, std::complex<double>(0.0, 0.0));
            for (int i = 0; i < nbasis; ++i)
            {
                for (int j = 0; j < nbasis; ++j)
                {
                    std::complex<double> val = 0.0;
                    for (int a = 0; a < ntarget; ++a)
                    {
                        // Branch: Q_i contains only constraint alpha's targets.
                        if (owner[a] == alpha)
                        {
                            val += ph[a][i] * std::conj(ph[a][j]);
                        }
                    }
                    // Q_c W Q_c = W - P W - W P + P W P.
                    const double wij = W[(alpha * nbasis + i) * nbasis + j];
                    val += wij;
                    for (int k = 0; k < nbasis; ++k)
                    {
                        val -= P[i * nbasis + k] * W[(alpha * nbasis + k) * nbasis + j];
                        val -= W[(alpha * nbasis + i) * nbasis + k] * P[k * nbasis + j];
                        for (int l = 0; l < nbasis; ++l)
                        {
                            val += P[i * nbasis + k]
                                   * W[(alpha * nbasis + k) * nbasis + l]
                                   * P[l * nbasis + j];
                        }
                    }
                    A[i * nbasis + j] = val;
                }
            }
            std::complex<double> e(0.0, 0.0);
            for (int i = 0; i < nbasis; ++i)
            {
                for (int j = 0; j < nbasis; ++j)
                {
                    e += std::conj(psi[i]) * A[i * nbasis + j] * psi[j];
                }
            }
            return e.real();
        };

        const double h = 1e-6;
        std::vector<std::vector<std::complex<double>>> fp = f;
        std::vector<std::vector<std::complex<double>>> fm = f;
        for (int a = 0; a < ntarget; ++a)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                fp[a][i] += h * df[a][i];
                fm[a][i] -= h * df[a][i];
            }
        }
        auto phi_fd = [&](const std::vector<std::vector<std::complex<double>>>& ff) {
            std::vector<std::complex<double>> tab(ntarget * nbasis);
            for (int a = 0; a < ntarget; ++a)
            {
                for (int i = 0; i < nbasis; ++i)
                {
                    tab[a * nbasis + i] = ff[a][i];
                }
            }
            std::vector<double> st;
            alloc::build_pw_projector_overlap(tab.data(), nbasis, ntarget, st);
            alloc::PwTargetSpace sp;
            alloc::build_pw_target(st, ntarget, atom, owner, nconstraint, sp);
            std::vector<std::vector<std::complex<double>>> ph;
            alloc::build_pw_alloc_orthogonal_phi(sp, ff, ph);
            return ph;
        };
        const double fd = (expectation(phi_fd(fp)) - expectation(phi_fd(fm)))
                          / (2.0 * h);
        printf("alloc_pw operator derivative alpha=%d: analytic=%.6e "
               "fd=%.6e diff=%.3e\n",
               alpha, analytic, fd, analytic - fd);
        EXPECT_NEAR(analytic, fd, 2e-7);
    }
}
