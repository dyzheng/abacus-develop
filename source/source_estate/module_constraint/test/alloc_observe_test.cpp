#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <vector>

#include "constraint_test_utils.h"
#include "source_estate/module_constraint/alloc_observe.h"
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

TEST(AllocObserveTest, SumRulesAndComplement)
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
    alloc::AllocOperator op;
    op.build_grid(wg, {0, 1},
                  {gaussian_grid(pw, *ucell, pos[0], 1.0),
                   gaussian_grid(pw, *ucell, pos[1], 1.0)});

    std::vector<std::vector<double>> up{
        gaussian_grid(pw, *ucell, pos[0], 1.0)};
    std::vector<std::vector<double>> dn{
        gaussian_grid(pw, *ucell, pos[2], 1.0)};
    const std::vector<double> occ_up{1.0};
    const std::vector<double> occ_dn{1.0};

    alloc::AllocObserve obs(op);
    const alloc::AllocResult result = obs.observe(up, occ_up, dn, occ_dn);

    double nref = 0.0;
    double mref = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        double rho_up = 0.0;
        double rho_dn = 0.0;
        for (size_t band = 0; band < up.size(); ++band)
        {
            rho_up += occ_up[band] * up[band][ir] * up[band][ir];
        }
        for (size_t band = 0; band < dn.size(); ++band)
        {
            rho_dn += occ_dn[band] * dn[band][ir] * dn[band][ir];
        }
        nref += (rho_up + rho_dn) * dV;
        mref += (rho_up - rho_dn) * dV;
    }

    double nsum = 0.0;
    double msum = 0.0;
    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        nsum += result.n_tot[atom];
        msum += result.m_tot[atom];
        EXPECT_NEAR(result.n_tot[atom],
                    result.n_T[atom] + result.n_c[atom], 1e-10);
        EXPECT_NEAR(result.m_tot[atom],
                    result.m_T[atom] + result.m_c[atom], 1e-10);
    }
    EXPECT_NEAR(nsum, nref, 1e-8);
    EXPECT_NEAR(msum, mref, 1e-8);
    printf("alloc_observe: cond=%.6e sum_n=%.12e/%.12e "
           "sum_m=%.12e/%.12e m_c=(%.6e %.6e %.6e)\n",
           result.cond, nsum, nref, msum, mref, result.m_c[0],
           result.m_c[1], result.m_c[2]);
    for (const double value : result.n_c_grid)
    {
        EXPECT_GE(value, -1e-14);
    }
    EXPECT_LT(result.cond, 1e6);
}

TEST(AllocObserveTest, EmptyTargetSetEqualsBecke)
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
    std::vector<std::vector<double>> up{
        gaussian_grid(pw, *ucell, pos[0], 1.0)};
    std::vector<std::vector<double>> dn{
        gaussian_grid(pw, *ucell, pos[1], 1.0)};
    const std::vector<double> occ_up{1.0};
    const std::vector<double> occ_dn{1.0};

    alloc::AllocOperator op0;
    op0.build_grid(wg, {}, {});
    alloc::AllocObserve obs0(op0);
    const alloc::AllocResult result = obs0.observe(up, occ_up, dn, occ_dn);

    for (int atom = 0; atom < ucell->nat; ++atom)
    {
        double n_direct = 0.0;
        double m_direct = 0.0;
        const std::vector<double>& weight = wg.constraint_weight(atom);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            double rho_up = 0.0;
            double rho_dn = 0.0;
            for (size_t band = 0; band < up.size(); ++band)
            {
                rho_up += occ_up[band] * up[band][ir] * up[band][ir];
            }
            for (size_t band = 0; band < dn.size(); ++band)
            {
                rho_dn += occ_dn[band] * dn[band][ir] * dn[band][ir];
            }
            n_direct += weight[ir] * (rho_up + rho_dn) * dV;
            m_direct += weight[ir] * (rho_up - rho_dn) * dV;
        }
        EXPECT_NEAR(result.n_tot[atom], n_direct, 1e-13);
        EXPECT_NEAR(result.m_tot[atom], m_direct, 1e-13);
    }
}
