#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_spin.h"

namespace
{

using Complex = std::complex<double>;
using SpinVector = ModuleBase::Vector3<double>;

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
            for (int k = 0; k < n; ++k)
            {
                A[static_cast<size_t>(i) * n + j]
                    += B[static_cast<size_t>(i) * n + k]
                       * B[static_cast<size_t>(j) * n + k];
            }
            // Branch: add a diagonal shift only once to keep the matrix SPD.
            if (i == j)
            {
                A[static_cast<size_t>(i) * n + j] += 0.7;
            }
        }
    }
    return A;
}

std::vector<double> random_symmetric(const int n, std::mt19937& gen)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            const double value = nd(gen);
            A[static_cast<size_t>(i) * n + j] = value;
            A[static_cast<size_t>(j) * n + i] = value;
        }
    }
    return A;
}

std::vector<double> target_overlap(const std::vector<double>& S,
                                   const int norb,
                                   const std::vector<int>& orbital)
{
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
    return ST;
}

double dense_trace_product(const std::vector<double>& A,
                           const std::vector<double>& B,
                           const int n)
{
    double value = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            value += A[static_cast<size_t>(i) * n + j]
                     * B[static_cast<size_t>(j) * n + i];
        }
    }
    return value;
}

Complex pauli(const int row, const int col, const int axis)
{
    // Branch: sigma_z is diagonal and real.
    if (axis == 2)
    {
        return Complex(row == col ? (row == 0 ? 1.0 : -1.0) : 0.0, 0.0);
    }
    // Branch: sigma_x is real and off-diagonal.
    if (axis == 0)
    {
        return Complex(row == col ? 0.0 : 1.0, 0.0);
    }
    // Branch: sigma_y is imaginary; the upper entry is -i.
    return Complex(0.0, row == col ? 0.0 : (row == 0 ? -1.0 : 1.0));
}

std::vector<Complex> spinor_density(const std::vector<double>& rho0,
                                    const std::vector<double>& rhox,
                                    const std::vector<double>& rhoy,
                                    const std::vector<double>& rhoz,
                                    const int norb)
{
    const int nspinor = 2 * norb;
    std::vector<Complex> D(static_cast<size_t>(nspinor) * nspinor,
                           Complex(0.0, 0.0));
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const size_t orbital = static_cast<size_t>(mu) * norb + nu;
            for (int s = 0; s < 2; ++s)
            {
                for (int t = 0; t < 2; ++t)
                {
                    D[static_cast<size_t>(2 * mu + s) * nspinor + 2 * nu + t]
                        = 0.5
                          * (rho0[orbital] * (s == t ? 1.0 : 0.0)
                             + rhox[orbital] * pauli(s, t, 0)
                             + rhoy[orbital] * pauli(s, t, 1)
                             + rhoz[orbital] * pauli(s, t, 2));
                }
            }
        }
    }
    return D;
}

double spinor_trace_product(const std::vector<Complex>& D,
                            const std::vector<Complex>& H,
                            const int nspinor)
{
    Complex value(0.0, 0.0);
    for (int i = 0; i < nspinor; ++i)
    {
        for (int j = 0; j < nspinor; ++j)
        {
            value += D[static_cast<size_t>(j) * nspinor + i]
                     * H[static_cast<size_t>(i) * nspinor + j];
        }
    }
    return value.real();
}

void rotation_matrix(const SpinVector& axis,
                     const double angle,
                     std::vector<double>& R)
{
    const double norm = std::sqrt(axis.x * axis.x + axis.y * axis.y
                                  + axis.z * axis.z);
    const double nx = axis.x / norm;
    const double ny = axis.y / norm;
    const double nz = axis.z / norm;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double one_c = 1.0 - c;
    R = {c + one_c * nx * nx, one_c * nx * ny - s * nz,
         one_c * nx * nz + s * ny,
         one_c * ny * nx + s * nz, c + one_c * ny * ny,
         one_c * ny * nz - s * nx,
         one_c * nz * nx - s * ny, one_c * nz * ny + s * nx,
         c + one_c * nz * nz};
}

SpinVector rotate_vector(const std::vector<double>& R, const SpinVector& v)
{
    return SpinVector(R[0] * v.x + R[1] * v.y + R[2] * v.z,
                      R[3] * v.x + R[4] * v.y + R[5] * v.z,
                      R[6] * v.x + R[7] * v.y + R[8] * v.z);
}

std::vector<Complex> spinor_unitary(const SpinVector& axis,
                                    const double angle)
{
    const double norm = std::sqrt(axis.x * axis.x + axis.y * axis.y
                                  + axis.z * axis.z);
    const SpinVector n(axis.x / norm, axis.y / norm, axis.z / norm);
    const double c = std::cos(0.5 * angle);
    const double s = std::sin(0.5 * angle);
    return {Complex(c, -s * n.z), Complex(-s * n.y, -s * n.x),
            Complex(s * n.y, -s * n.x), Complex(c, s * n.z)};
}

std::vector<Complex> spinor_apply(const std::vector<Complex>& U,
                                  const std::vector<Complex>& D,
                                  const int n)
{
    std::vector<Complex> UD(static_cast<size_t>(n) * n, Complex(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            for (int j = 0; j < n; ++j)
            {
                UD[static_cast<size_t>(i) * n + j]
                    += U[static_cast<size_t>(i) * n + k]
                       * D[static_cast<size_t>(k) * n + j];
            }
        }
    }
    std::vector<Complex> out(static_cast<size_t>(n) * n, Complex(0.0, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            for (int j = 0; j < n; ++j)
            {
                out[static_cast<size_t>(i) * n + j]
                    += UD[static_cast<size_t>(i) * n + k]
                       * std::conj(U[static_cast<size_t>(j) * n + k]);
            }
        }
    }
    return out;
}

} // namespace

TEST(AllocSpinTest, SpinorAndPauliReadoutsAgree)
{
    std::mt19937 gen(71);
    const int norb = 5;
    const std::vector<double> K = random_symmetric(norb, gen);
    const std::vector<double> rho0 = random_symmetric(norb, gen);
    const std::vector<double> rhox = random_symmetric(norb, gen);
    const std::vector<double> rhoy = random_symmetric(norb, gen);
    const std::vector<double> rhoz = random_symmetric(norb, gen);
    const std::vector<Complex> D = spinor_density(rho0, rhox, rhoy, rhoz,
                                                  norb);

    double charge = 0.0;
    SpinVector moment;
    alloc::observe_alloc_spinor_moments(K, norb, D, charge, moment);
    const double n_ref = dense_trace_product(rho0, K, norb);
    const double mx_ref = dense_trace_product(rhox, K, norb);
    const double my_ref = dense_trace_product(rhoy, K, norb);
    const double mz_ref = dense_trace_product(rhoz, K, norb);
    EXPECT_NEAR(charge, n_ref, 1e-12);
    EXPECT_NEAR(moment.x, mx_ref, 1e-12);
    EXPECT_NEAR(moment.y, my_ref, 1e-12);
    EXPECT_NEAR(moment.z, mz_ref, 1e-12);
    const double max_readout_dev = std::max(
        std::max(std::abs(charge - n_ref), std::abs(moment.x - mx_ref)),
        std::max(std::abs(moment.y - my_ref), std::abs(moment.z - mz_ref)));

    const SpinVector lambda{0.31, -0.72, 1.13};
    std::vector<Complex> H;
    alloc::build_alloc_spin_kernel(K, norb, lambda, H);
    const double energy = spinor_trace_product(D, H, 2 * norb);
    EXPECT_NEAR(energy,
                lambda.x * moment.x + lambda.y * moment.y
                    + lambda.z * moment.z,
                1e-12);
    printf("alloc_spin readout: max_dev=%.3e energy_dev=%.3e\n",
           max_readout_dev,
           std::abs(energy - (lambda.x * moment.x + lambda.y * moment.y
                              + lambda.z * moment.z)));
}

TEST(AllocSpinTest, NoSocGlobalSpinRotationInvariant)
{
    std::mt19937 gen(73);
    const int norb = 4;
    const std::vector<double> K0 = random_symmetric(norb, gen);
    const std::vector<double> Kc = random_symmetric(norb, gen);
    const std::vector<double> rho0 = random_spd(norb, gen);
    const std::vector<double> rhox = random_symmetric(norb, gen);
    const std::vector<double> rhoy = random_symmetric(norb, gen);
    const std::vector<double> rhoz = random_symmetric(norb, gen);
    const std::vector<Complex> D = spinor_density(rho0, rhox, rhoy, rhoz,
                                                  norb);
    const SpinVector lambda{0.22, -0.41, 0.67};

    double charge = 0.0;
    SpinVector moment;
    alloc::observe_alloc_spinor_moments(Kc, norb, D, charge, moment);
    std::vector<Complex> H0;
    alloc::build_alloc_spin_kernel(K0, norb, SpinVector(0.0, 0.0, 0.0), H0);
    // Branch: the scalar K0 does not enter K tensor (lambda.sigma); add the
    // identity spin block explicitly for the Kohn-Sham part.
    std::vector<Complex> Hks(static_cast<size_t>(2 * norb) * 2 * norb,
                             Complex(0.0, 0.0));
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const double value = K0[static_cast<size_t>(mu) * norb + nu];
            Hks[static_cast<size_t>(2 * mu) * 2 * norb + 2 * nu] = value;
            Hks[static_cast<size_t>(2 * mu + 1) * 2 * norb + 2 * nu + 1]
                = value;
        }
    }
    const double e_ks = spinor_trace_product(D, Hks, 2 * norb);
    const double e_con = lambda.x * moment.x + lambda.y * moment.y
                         + lambda.z * moment.z;

    const SpinVector axis{0.3, -0.8, 0.5};
    const double angle = 0.71;
    std::vector<double> R;
    rotation_matrix(axis, angle, R);
    const SpinVector lambda_rot = rotate_vector(R, lambda);
    const std::vector<Complex> U2 = spinor_unitary(axis, angle);
    const int nspinor = 2 * norb;
    std::vector<Complex> U(static_cast<size_t>(nspinor) * nspinor,
                           Complex(0.0, 0.0));
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            // Branch: the scalar orbital basis is untouched; only the 2x2
            // spin block rotates.
            if (mu == nu)
            {
                for (int s = 0; s < 2; ++s)
                {
                    for (int t = 0; t < 2; ++t)
                    {
                        U[static_cast<size_t>(2 * mu + s) * nspinor
                          + 2 * nu + t]
                            = U2[static_cast<size_t>(s) * 2 + t];
                    }
                }
            }
        }
    }
    const std::vector<Complex> D_rot = spinor_apply(U, D, nspinor);
    SpinVector moment_rot;
    alloc::observe_alloc_spinor_moments(Kc, norb, D_rot, charge, moment_rot);
    const SpinVector expected = rotate_vector(R, moment);
    EXPECT_NEAR(moment_rot.x, expected.x, 1e-12);
    EXPECT_NEAR(moment_rot.y, expected.y, 1e-12);
    EXPECT_NEAR(moment_rot.z, expected.z, 1e-12);
    const double max_moment_dev
        = std::max(std::abs(moment_rot.x - expected.x),
                   std::max(std::abs(moment_rot.y - expected.y),
                            std::abs(moment_rot.z - expected.z)));

    std::vector<Complex> Hc_rot;
    alloc::build_alloc_spin_kernel(Kc, norb, lambda_rot, Hc_rot);
    const double e_rot = spinor_trace_product(D_rot, Hks, nspinor)
                         + spinor_trace_product(D_rot, Hc_rot, nspinor);
    EXPECT_NEAR(e_rot, e_ks + e_con, 1e-12);
    EXPECT_LT(std::abs(e_rot - (e_ks + e_con)), 1e-6);
    printf("alloc_spin rotation: max_moment_dev=%.3e dE=%.3e Ry\n",
           max_moment_dev, std::abs(e_rot - (e_ks + e_con)));
}

TEST(AllocSpinTest, SAdjointMatchesSpinFiniteDifference)
{
    std::mt19937 gen(79);
    const int norb = 9;
    const int nconstraint = 2;
    const std::vector<int> orbital = {1, 3, 5};
    const std::vector<int> atom = {0, 1, 2};
    const std::vector<int> owner = {0, 1, 0};
    const std::vector<double> S = random_spd(norb, gen);
    const std::vector<double> dS = random_symmetric(norb, gen);
    const std::vector<double> rhox = random_symmetric(norb, gen);
    const std::vector<double> rhoy = random_symmetric(norb, gen);
    const std::vector<double> rhoz = random_symmetric(norb, gen);
    const std::vector<double> rho0(
        static_cast<size_t>(norb) * norb, 0.0);
    const std::vector<SpinVector> lambda{SpinVector(0.17, -0.23, 0.41),
                                         SpinVector(-0.11, 0.29, -0.37)};

    std::vector<std::vector<double>> W(
        nconstraint, std::vector<double>(static_cast<size_t>(norb) * norb,
                                         0.0));
    for (std::vector<double>& matrix : W)
    {
        matrix = random_symmetric(norb, gen);
    }

    alloc::AllocConstraintAlgebra space;
    alloc::build_alloc_constraint_algebra(
        S, norb, orbital, atom, owner, target_overlap(S, norb, orbital),
        nconstraint, space, 1e-12);
    std::vector<double> E;
    alloc::build_alloc_spin_s_adjoint(space, S, W, lambda, rhox, rhoy, rhoz,
                                      E);

    const auto energy = [&](const std::vector<double>& overlap) {
        alloc::AllocConstraintAlgebra local;
        alloc::build_alloc_constraint_algebra(
            overlap, norb, orbital, atom, owner,
            target_overlap(overlap, norb, orbital), nconstraint, local,
            1e-12);
        std::vector<double> charges;
        std::vector<SpinVector> moments;
        alloc::observe_alloc_spin_moments(local, W, rho0, rhox, rhoy, rhoz,
                                          charges, moments);
        double value = 0.0;
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            value += lambda[alpha].x * moments[alpha].x
                     + lambda[alpha].y * moments[alpha].y
                     + lambda[alpha].z * moments[alpha].z;
        }
        return value;
    };

    const double eps = 1e-6;
    std::vector<double> Sp = S;
    std::vector<double> Sm = S;
    for (size_t i = 0; i < S.size(); ++i)
    {
        Sp[i] += eps * dS[i];
        Sm[i] -= eps * dS[i];
    }
    const double fd = (energy(Sp) - energy(Sm)) / (2.0 * eps);
    double predicted = 0.0;
    for (size_t i = 0; i < E.size(); ++i)
    {
        predicted += E[i] * dS[i];
    }
    EXPECT_NEAR(predicted, fd, 2e-7);
    printf("alloc_spin s-adjoint: predicted=%.12e fd=%.12e diff=%.3e\n",
           predicted, fd, std::abs(predicted - fd));
}
