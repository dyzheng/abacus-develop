#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "constraint_test_utils.h"
#include "source_estate/module_constraint/alloc_operator.h"
#include "source_estate/module_constraint/weight_grid.h"

// Mock symbols required by the cell_info object library in unit tests.
Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}

namespace
{

// Normalized Gaussian on the grid.  The R x R orthogonalization removes the
// discrete normalization error; this helper only shapes the target function.
std::vector<double> gaussian_grid(const ModulePW::PW_Basis& pw,
                                  const UnitCell& ucell,
                                  const std::array<double, 3>& center,
                                  const double sigma)
{
    std::vector<double> f(pw.nrxx, 0.0);
    const double pi = std::acos(-1.0);
    const double norm = std::pow(2.0 * pi * sigma * sigma, 1.5);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const int i = ir / (pw.ny * pw.nplane);
        const int j = ir / pw.nplane - i * pw.ny;
        const int k = ir % pw.nplane + pw.startz_current;
        const ModuleBase::Vector3<double> rf(
            static_cast<double>(i) / pw.nx,
            static_cast<double>(j) / pw.ny,
            static_cast<double>(k) / pw.nz);
        const ModuleBase::Vector3<double> r = rf * ucell.latvec * ucell.lat0;
        const double r2 = (r.x - center[0]) * (r.x - center[0])
                        + (r.y - center[1]) * (r.y - center[1])
                        + (r.z - center[2]) * (r.z - center[2]);
        f[ir] = std::exp(-r2 / (2.0 * sigma * sigma)) / norm;
    }
    return f;
}

} // namespace

TEST(AllocOperatorTest, SumRuleAndPositivity)
{
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    const double dV = ucell->omega / static_cast<double>(pw.nxyz);
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();

    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    std::vector<std::vector<double>> phi0;
    phi0.push_back(gaussian_grid(pw, *ucell, pos[0], 1.0));
    phi0.push_back(gaussian_grid(pw, *ucell, pos[1], 1.0));
    alloc::AllocOperator op;
    const auto t0 = std::chrono::high_resolution_clock::now();
    op.build_grid(wg, {0, 1}, phi0);
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("alloc_operator: ntarget=%d nrxx=%d cond=%.6e build=%.3f ms\n",
           op.ntarget(), op.nrxx(), op.cond(), build_ms);
    EXPECT_LT(op.cond(), 1e6);

    std::mt19937 gen(3);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(pw.nrxx);
    std::vector<double> y;
    std::vector<double> acc(pw.nrxx, 0.0);
    for (double& v : x)
    {
        v = nd(gen);
    }

    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        op.apply_A(atom, x, y);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            acc[ir] += y[ir];
        }
    }
    double max_sum_error = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        max_sum_error = std::max(max_sum_error, std::abs(acc[ir] - x[ir]));
        EXPECT_NEAR(acc[ir], x[ir], 1e-10);
    }

    double min_q = 1e300;
    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        op.apply_A(atom, x, y);
        double q = 0.0;
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            q += x[ir] * y[ir] * dV;
        }
        min_q = std::min(min_q, q);
        EXPECT_GE(q, -1e-12);
    }
    printf("alloc_operator audit: max_sum_error=%.3e min_q=%.6e\n",
           max_sum_error, min_q);
}

TEST(AllocOperatorTest, InterpolationLimits)
{
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();

    std::mt19937 gen(5);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(pw.nrxx);
    std::vector<double> y;
    for (double& v : x)
    {
        v = nd(gen);
    }

    // Empty target set: A_i reduces to multiplication by the Becke weight.
    alloc::AllocOperator op0;
    op0.build_grid(wg, {}, {});
    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        op0.apply_A(atom, x, y);
        const std::vector<double>& weight = wg.constraint_weight(atom);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            EXPECT_NEAR(y[ir], weight[ir] * x[ir], 1e-12);
        }
    }

    // Complete orthonormal target set: a delta at every grid point spans the
    // represented space, so Qc=0 and A_i is the atom's diagonal mask.
    ModulePW::PW_Basis pw4;
    pw4.initgrids(1.0, ucell->latvec, 4, 4, 4);
    pw4.distribute_r();
    const double dV4 = ucell->omega / static_cast<double>(pw4.nxyz);
    constraint::WeightGrid wg4(*ucell, &pw4, {1.5, 0.5, 0.5});
    wg4.set_constraint_atoms({{0}, {1}, {2}});
    wg4.build();

    const std::vector<std::array<double, 3>> pos4 = h2o_positions(*ucell);
    std::vector<std::vector<double>> deltas(
        pw4.nrxx, std::vector<double>(pw4.nrxx, 0.0));
    std::vector<int> which_atom(pw4.nrxx, 0);
    for (int a = 0; a < pw4.nrxx; ++a)
    {
        deltas[a][a] = 1.0 / std::sqrt(dV4);
        const int i = a / (pw4.ny * pw4.nplane);
        const int j = a / pw4.nplane - i * pw4.ny;
        const int k = a % pw4.nplane + pw4.startz_current;
        const ModuleBase::Vector3<double> rf(
            static_cast<double>(i) / pw4.nx,
            static_cast<double>(j) / pw4.ny,
            static_cast<double>(k) / pw4.nz);
        const ModuleBase::Vector3<double> r = rf * ucell->latvec * ucell->lat0;
        double best = 1e30;
        for (int atom = 0; atom < 3; ++atom)
        {
            const double d2
                = (r.x - pos4[atom][0]) * (r.x - pos4[atom][0])
                  + (r.y - pos4[atom][1]) * (r.y - pos4[atom][1])
                  + (r.z - pos4[atom][2]) * (r.z - pos4[atom][2]);
            // Branch: a closer atom takes ownership of this grid point.
            if (d2 < best)
            {
                best = d2;
                which_atom[a] = atom;
            }
        }
    }
    alloc::AllocOperator op_complete;
    op_complete.build_grid(wg4, which_atom, deltas);
    std::vector<double> x4(pw4.nrxx);
    for (double& v : x4)
    {
        v = nd(gen);
    }
    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        op_complete.apply_A(atom, x4, y);
        for (int a = 0; a < pw4.nrxx; ++a)
        {
            // Branch: only grid points assigned to this atom survive.
            const double expected = (which_atom[a] == atom) ? x4[a] : 0.0;
            EXPECT_NEAR(y[a], expected, 1e-10);
        }
    }
}

TEST(AllocOperatorTest, DenseDefinitionMatchesApply)
{
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 4, 4, 4);
    pw.distribute_r();
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();

    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    std::vector<std::vector<double>> phi0;
    phi0.push_back(gaussian_grid(pw, *ucell, pos[0], 1.0));
    phi0.push_back(gaussian_grid(pw, *ucell, pos[1], 1.0));
    alloc::AllocOperator op;
    op.build_grid(wg, {0, 1}, phi0);

    const int n = pw.nrxx;
    std::vector<double> P(n * n, 0.0);
    std::vector<double> Qc(n * n, 0.0);
    for (int col = 0; col < n; ++col)
    {
        std::vector<double> basis(n, 0.0);
        basis[col] = 1.0;
        std::vector<double> pcol;
        op.apply_P(basis, pcol);
        std::vector<double> qccol;
        op.apply_Qc(basis, qccol);
        for (int row = 0; row < n; ++row)
        {
            P[row * n + col] = pcol[row];
            Qc[row * n + col] = qccol[row];
        }
    }

    std::mt19937 gen(17);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(n);
    for (double& v : x)
    {
        v = nd(gen);
    }

    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        std::vector<double> Qi(n * n, 0.0);
        for (int col = 0; col < n; ++col)
        {
            std::vector<double> basis(n, 0.0);
            basis[col] = 1.0;
            std::vector<double> qicol;
            op.apply_Qi(atom, basis, qicol);
            for (int row = 0; row < n; ++row)
            {
                Qi[row * n + col] = qicol[row];
            }
        }

        // Dense assembly A_i = Q_i + Qc W_i Qc.
        std::vector<double> dense(n * n, 0.0);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                double val = Qi[row * n + col];
                for (int k = 0; k < n; ++k)
                {
                    val += Qc[row * n + k] * op.weight(atom)[k]
                           * Qc[k * n + col];
                }
                dense[row * n + col] = val;
            }
        }

        std::vector<double> expected(n, 0.0);
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                expected[row] += dense[row * n + col] * x[col];
            }
        }
        std::vector<double> applied;
        op.apply_A(atom, x, applied);
        for (int ir = 0; ir < n; ++ir)
        {
            EXPECT_NEAR(applied[ir], expected[ir], 1e-10);
        }
    }
}
