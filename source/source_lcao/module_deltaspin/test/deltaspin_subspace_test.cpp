#include "gtest/gtest.h"
#include <complex>
#include <cmath>
#include <vector>

/***********************************************************************
 * Unit tests for DeltaSpin LCAO subspace rotation (run_lambda_loop_lcao)
 *
 * These tests verify the core mathematical formulas independently of
 * the full ABACUS framework:
 *   1. chi = analytical Jacobian dM/dlambda
 *   2. H_sub diagonalization and unitary property of V
 *   3. Mi_new computed from rotated basis (V^H P V)
 ***********************************************************************/

// Minimal LAPACK zheev declaration (Fortran interface)
extern "C"
{
    void zheev_(const char* jobz,
                const char* uplo,
                const int* n,
                std::complex<double>* a,
                const int* lda,
                double* w,
                std::complex<double>* work,
                const int* lwork,
                double* rwork,
                int* info);
}

class DeltaSpinSubspaceTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: diagonalize a Hermitian matrix with LAPACK zheev
    // Returns eigenvalues in e_new and eigenvectors in V (column-major, overwrites input)
    void diagonalize(std::vector<std::complex<double>>& H,
                     int n,
                     std::vector<double>& e_new,
                     std::vector<std::complex<double>>& V)
    {
        V = H;
        e_new.resize(n);
        int lwork = 2 * n;
        std::vector<std::complex<double>> work(lwork);
        std::vector<double> rwork(3 * n);
        int info = 0;
        zheev_("V", "U", &n, V.data(), &n, e_new.data(), work.data(), &lwork, rwork.data(), &info);
        EXPECT_EQ(info, 0) << "zheev failed with info=" << info;
    }

    // Helper: compute V^H * V and check it equals identity
    void check_unitary(const std::vector<std::complex<double>>& V, int n)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::complex<double> sum = {0.0, 0.0};
                for (int k = 0; k < n; k++)
                {
                    // V[k*n + i] is V_{k,i}, conj gives V^H_{i,k}
                    sum += std::conj(V[k * n + i]) * V[k * n + j];
                }
                if (i == j)
                {
                    EXPECT_NEAR(sum.real(), 1.0, 1e-12);
                    EXPECT_NEAR(sum.imag(), 0.0, 1e-12);
                }
                else
                {
                    EXPECT_NEAR(sum.real(), 0.0, 1e-12);
                    EXPECT_NEAR(sum.imag(), 0.0, 1e-12);
                }
            }
        }
    }
};

// =====================================================================
// 1. Chi calculation: analytical Jacobian dM/dlambda
// =====================================================================

TEST_F(DeltaSpinSubspaceTest, ChiCalculation_TwoBand)
{
    // P = [[0, c], [c*, 0]] with c = 0.5 + 0.3i
    const int nbands = 2;
    const std::complex<double> c(0.5, 0.3);
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = c;
    P[1 * nbands + 0] = std::conj(c);

    // Energies and occupations: e=[0,1], f=[1,0]
    const double e[2] = {0.0, 1.0};
    const double f[2] = {1.0, 0.0};

    // Expected chi = 2*(f0-f1)*|c|^2 / (e0-e1)
    // |c|^2 = 0.25 + 0.09 = 0.34
    // chi = 2*(1-0)*0.34 / (0-1) = -0.68
    const double expected_chi = -0.68;

    double chi_val = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        for (int m = n + 1; m < nbands; m++)
        {
            const double de = e[n] - e[m];
            if (std::abs(de) < 1e-10) continue;
            const double P_nm_sq = std::norm(P[n * nbands + m]);
            chi_val += 2.0 * (f[n] - f[m]) * P_nm_sq / de;
        }
    }

    EXPECT_NEAR(chi_val, expected_chi, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, ChiCalculation_DegenerateBands)
{
    // Degenerate case: e0 == e1, should skip the term
    const int nbands = 2;
    const std::complex<double> c(1.0, 0.0);
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = c;
    P[1 * nbands + 0] = std::conj(c);

    const double e[2] = {1.0, 1.0}; // degenerate
    const double f[2] = {1.0, 0.5};

    double chi_val = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        for (int m = n + 1; m < nbands; m++)
        {
            const double de = e[n] - e[m];
            if (std::abs(de) < 1e-10) continue;
            const double P_nm_sq = std::norm(P[n * nbands + m]);
            chi_val += 2.0 * (f[n] - f[m]) * P_nm_sq / de;
        }
    }

    // Skipped because de == 0
    EXPECT_NEAR(chi_val, 0.0, 1e-15);
}

TEST_F(DeltaSpinSubspaceTest, ChiCalculation_MultiChannel)
{
    // Verify that both spin-up and spin-down channels add constructively
    // when both have the same P and occupations
    const int nbands = 2;
    const std::complex<double> c(0.5, 0.0);
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = c;
    P[1 * nbands + 0] = std::conj(c);

    const double e[2] = {0.0, 2.0};
    const double f[2] = {1.0, 0.0};

    // Single channel contribution
    double chi_single = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        for (int m = n + 1; m < nbands; m++)
        {
            const double de = e[n] - e[m];
            if (std::abs(de) < 1e-10) continue;
            const double P_nm_sq = std::norm(P[n * nbands + m]);
            chi_single += 2.0 * (f[n] - f[m]) * P_nm_sq / de;
        }
    }

    // Expected: 2*(1-0)*0.25 / (0-2) = 0.5 / (-2) = -0.25
    EXPECT_NEAR(chi_single, -0.25, 1e-12);

    // Two channels (spin-up + spin-down) with same P, e, f
    // In the code both channels add because sign*sign = 1
    double chi_total = 0.0;
    for (int ik = 0; ik < 2; ik++)
    {
        const double sign = (ik == 0) ? 1.0 : -1.0;
        for (int n = 0; n < nbands; n++)
        {
            for (int m = n + 1; m < nbands; m++)
            {
                const double de = e[n] - e[m];
                if (std::abs(de) < 1e-10) continue;
                const double P_nm_sq = std::norm(P[n * nbands + m]);
                // Note: code uses 2*(fn-fm)*P_nm_sq/de without extra sign factor
                // The sign only affects dlambda in H_sub, not chi itself
                chi_total += 2.0 * (f[n] - f[m]) * P_nm_sq / de;
            }
        }
    }
    EXPECT_NEAR(chi_total, 2.0 * chi_single, 1e-12);
}

// =====================================================================
// 2. Subspace diagonalization: H_sub = diag(e) + dlambda * P
// =====================================================================

TEST_F(DeltaSpinSubspaceTest, SubspaceDiag_SigmaX)
{
    // P = sigma_x = [[0, 1], [1, 0]], e = [0, 0], dlambda = 1.0
    // H_sub = [[0, 1], [1, 0]]
    // Eigenvalues: -1, +1
    // Eigenvectors: [-1/sqrt(2), 1/sqrt(2)] and [1/sqrt(2), 1/sqrt(2)] (columns)
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {1.0, 0.0};
    P[1 * nbands + 0] = {1.0, 0.0};

    const double dlambda = 1.0;
    std::vector<std::complex<double>> H_sub(nbands * nbands, {0.0, 0.0});
    H_sub[0 * nbands + 0] = {0.0, 0.0};
    H_sub[1 * nbands + 1] = {0.0, 0.0};
    for (int i = 0; i < nbands * nbands; i++)
    {
        H_sub[i] += dlambda * P[i];
    }

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(H_sub, nbands, e_new, V);

    EXPECT_NEAR(e_new[0], -1.0, 1e-12);
    EXPECT_NEAR(e_new[1], +1.0, 1e-12);

    check_unitary(V, nbands);

    // Verify eigenvectors manually
    // V[:,0] should be proportional to [-1, 1]
    std::complex<double> v00 = V[0 * nbands + 0]; // V_{0,0}
    std::complex<double> v10 = V[1 * nbands + 0]; // V_{1,0}
    EXPECT_NEAR(std::abs(v00), std::abs(v10), 1e-12);
    EXPECT_NEAR((v00 + v10).real(), 0.0, 1e-12);

    // V[:,1] should be proportional to [1, 1]
    std::complex<double> v01 = V[0 * nbands + 1];
    std::complex<double> v11 = V[1 * nbands + 1];
    EXPECT_NEAR(std::abs(v01), std::abs(v11), 1e-12);
    EXPECT_NEAR((v01 - v11).real(), 0.0, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, SubspaceDiag_ComplexP)
{
    // P = [[0, i], [-i, 0]] (sigma_y), e = [1, 2], dlambda = 0.5
    // H_sub = [[1, 0.5i], [-0.5i, 2]]
    // Eigenvalues: 1.5 +/- sqrt(0.25 + 0.25) = 1.5 +/- sqrt(0.5)
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {0.0, 1.0};
    P[1 * nbands + 0] = {0.0, -1.0};

    std::vector<std::complex<double>> H_sub(nbands * nbands, {0.0, 0.0});
    H_sub[0 * nbands + 0] = {1.0, 0.0};
    H_sub[1 * nbands + 1] = {2.0, 0.0};
    const double dlambda = 0.5;
    for (int i = 0; i < nbands * nbands; i++)
    {
        H_sub[i] += dlambda * P[i];
    }

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(H_sub, nbands, e_new, V);

    const double expected[2] = {1.5 - std::sqrt(0.5), 1.5 + std::sqrt(0.5)};
    EXPECT_NEAR(e_new[0], expected[0], 1e-12);
    EXPECT_NEAR(e_new[1], expected[1], 1e-12);

    check_unitary(V, nbands);
}

TEST_F(DeltaSpinSubspaceTest, SubspaceDiag_ThreeBand)
{
    // 3x3 Hermitian matrix: P = diag(1, -1, 2), e = [0, 0, 0], dlambda = 0.5
    // H_sub = diag(0.5, -0.5, 1.0)
    const int nbands = 3;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 0] = {1.0, 0.0};
    P[1 * nbands + 1] = {-1.0, 0.0};
    P[2 * nbands + 2] = {2.0, 0.0};

    std::vector<std::complex<double>> H_sub(nbands * nbands, {0.0, 0.0});
    const double dlambda = 0.5;
    for (int i = 0; i < nbands * nbands; i++)
    {
        H_sub[i] += dlambda * P[i];
    }

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(H_sub, nbands, e_new, V);

    // Already diagonal, eigenvalues should be -0.5, 0.5, 1.0
    EXPECT_NEAR(e_new[0], -0.5, 1e-12);
    EXPECT_NEAR(e_new[1], 0.5, 1e-12);
    EXPECT_NEAR(e_new[2], 1.0, 1e-12);

    check_unitary(V, nbands);

    // V^H P V should be diagonal with eigenvalues of P (not H_sub)
    // Since H_sub = dlambda * P and dlambda = 0.5, eigenvalues of P = e_new / dlambda
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        EXPECT_NEAR(pnn.real(), e_new[n] / 0.5, 1e-12);
        EXPECT_NEAR(pnn.imag(), 0.0, 1e-12);
    }
}

// =====================================================================
// 3. Mi_new from rotated basis: Mi_z = sum_n wg(n) * (V^H P V)_{nn}
// =====================================================================

TEST_F(DeltaSpinSubspaceTest, MiNew_SigmaX_OccupyLower)
{
    // P = sigma_x, diagonalized by V: V^H P V = diag(-1, +1)
    // If only lower eigenstate occupied (wg=[1,0]), Mi_z = -1
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {1.0, 0.0};
    P[1 * nbands + 0] = {1.0, 0.0};

    // Get V from diagonalization of P itself
    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(P, nbands, e_new, V);

    // Verify V^H P V = diag(-1, 1)
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        EXPECT_NEAR(pnn.real(), e_new[n], 1e-12);
        EXPECT_NEAR(pnn.imag(), 0.0, 1e-12);
    }

    // Compute Mi_new with wg = [1, 0]
    const double wg[2] = {1.0, 0.0};
    const double sign = 1.0;
    double mi_z = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        mi_z += sign * wg[n] * pnn.real();
    }

    EXPECT_NEAR(mi_z, -1.0, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, MiNew_SigmaX_OccupyUpper)
{
    // Same as above but wg = [0, 1]: Mi_z = +1
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {1.0, 0.0};
    P[1 * nbands + 0] = {1.0, 0.0};

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(P, nbands, e_new, V);

    const double wg[2] = {0.0, 1.0};
    const double sign = 1.0;
    double mi_z = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        mi_z += sign * wg[n] * pnn.real();
    }

    EXPECT_NEAR(mi_z, 1.0, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, MiNew_SigmaX_SignFlip)
{
    // Same as OccupyLower but sign = -1 (spin-down channel)
    // Expected: Mi_z = -(-1) = +1
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {1.0, 0.0};
    P[1 * nbands + 0] = {1.0, 0.0};

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(P, nbands, e_new, V);

    const double wg[2] = {1.0, 0.0};
    const double sign = -1.0;
    double mi_z = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        mi_z += sign * wg[n] * pnn.real();
    }

    EXPECT_NEAR(mi_z, 1.0, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, MiNew_ComplexP)
{
    // P = sigma_y, e_new from diagonalization should be [-1, +1]
    // With wg=[0.5, 0.5], Mi_z = 0
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 1] = {0.0, 1.0};
    P[1 * nbands + 0] = {0.0, -1.0};

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(P, nbands, e_new, V);

    const double wg[2] = {0.5, 0.5};
    const double sign = 1.0;
    double mi_z = 0.0;
    for (int n = 0; n < nbands; n++)
    {
        std::complex<double> pnn = {0.0, 0.0};
        for (int a = 0; a < nbands; a++)
        {
            std::complex<double> tmp = {0.0, 0.0};
            for (int b = 0; b < nbands; b++)
            {
                tmp += P[a * nbands + b] * V[b * nbands + n];
            }
            pnn += std::conj(V[a * nbands + n]) * tmp;
        }
        mi_z += sign * wg[n] * pnn.real();
    }

    EXPECT_NEAR(mi_z, 0.0, 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, MiNew_EmptyAndFull)
{
    // P = diag(1, -1), V = I (already diagonal)
    // wg = [1, 1]: Mi_z = 1 + (-1) = 0
    // wg = [1, 0]: Mi_z = 1
    // wg = [0, 1]: Mi_z = -1
    const int nbands = 2;
    std::vector<std::complex<double>> P(nbands * nbands, {0.0, 0.0});
    P[0 * nbands + 0] = {1.0, 0.0};
    P[1 * nbands + 1] = {-1.0, 0.0};

    std::vector<double> e_new;
    std::vector<std::complex<double>> V;
    diagonalize(P, nbands, e_new, V);

    auto compute_mi = [&](const double wg[2]) -> double
    {
        double mi_z = 0.0;
        for (int n = 0; n < nbands; n++)
        {
            std::complex<double> pnn = {0.0, 0.0};
            for (int a = 0; a < nbands; a++)
            {
                std::complex<double> tmp = {0.0, 0.0};
                for (int b = 0; b < nbands; b++)
                {
                    tmp += P[a * nbands + b] * V[b * nbands + n];
                }
                pnn += std::conj(V[a * nbands + n]) * tmp;
            }
            mi_z += wg[n] * pnn.real();
        }
        return mi_z;
    };

    // For diagonal P, V^H P V = diag(e_new), so Mi_z = sum_n wg[n] * e_new[n]
    const double wg1[2] = {1.0, 1.0};
    EXPECT_NEAR(compute_mi(wg1), e_new[0] + e_new[1], 1e-12);

    const double wg2[2] = {1.0, 0.0};
    EXPECT_NEAR(compute_mi(wg2), e_new[0], 1e-12);

    const double wg3[2] = {0.0, 1.0};
    EXPECT_NEAR(compute_mi(wg3), e_new[1], 1e-12);
}

TEST_F(DeltaSpinSubspaceTest, NewtonStep_Convergence)
{
    // End-to-end math test of the Newton update logic:
    //   delta_lambda = alpha * (target - current) / chi
    // Verify that if chi is correct, one Newton step gives exact lambda
    const double target_mag = 2.0;
    const double current_mag = 1.0;
    const double alpha = 1.0;
    const double chi = 1.0; // dM/dlambda = 1

    const double delta_lambda = alpha * (target_mag - current_mag) / chi;
    const double lambda_new = 0.0 + delta_lambda; // initial_lambda = 0

    // If M(lambda) = chi * lambda + offset, and current M = 1 at lambda=0,
    // then offset = 1. After lambda_new = 1, M = 1*1 + 1 = 2 = target.
    EXPECT_NEAR(lambda_new, 1.0, 1e-12);
}
