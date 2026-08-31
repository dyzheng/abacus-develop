#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

#include "constraint_test_utils.h"

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
#include "source_base/matrix.h"
#include "source_estate/module_constraint/constraint_inject_pw.h"
#include "source_estate/module_constraint/constraint_observe.h"
#include "source_estate/module_constraint/weight_grid.h"

using ModuleBase::PI;

// Normalized Gaussian centered at 'center' with width sigma (Bohr).
static double gaussian(const std::array<double, 3>& r,
                       const std::array<double, 3>& center,
                       const double sigma)
{
    const double dx = r[0] - center[0];
    const double dy = r[1] - center[1];
    const double dz = r[2] - center[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double norm = std::pow(2.0 * PI * sigma * sigma, 1.5);
    return std::exp(-r2 / (2.0 * sigma * sigma)) / norm;
}

class ConstraintInjectPWTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;
    std::vector<std::array<double, 3>> pos;
    std::vector<double> nelec_atom;
    std::vector<double> rho;

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40);
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
        pos = h2o_positions(*ucell);
        nelec_atom = {8.0, 1.0, 1.0};
        // Atomic superposition charge density on the grid.
        const double sigma = 1.0;
        rho.assign(rhopw->nrxx, 0.0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const int i = ir / (rhopw->ny * rhopw->nplane);
            const int j = ir / rhopw->nplane - i * rhopw->ny;
            const int k = ir % rhopw->nplane + rhopw->startz_current;
            const ModuleBase::Vector3<double> rfrac(
                static_cast<double>(i) / rhopw->nx,
                static_cast<double>(j) / rhopw->ny,
                static_cast<double>(k) / rhopw->nz);
            const ModuleBase::Vector3<double> rc =
                rfrac * ucell->latvec * ucell->lat0;
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                rho[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
            }
        }
    }

    void TearDown() override
    {
        delete rhopw;
    }
};

TEST_F(ConstraintInjectPWTest, PointwiseInjectionNspin1)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii);
    wg.build();
    const int nrxx = rhopw->nrxx;
    const int nalpha = wg.nconstraint();

    // Fixed veff and mu; assert the pointwise identity
    //   veff_new(ir) == veff_old(ir) + sum_alpha mu[alpha]*w[alpha][ir]
    // at machine precision.
    ModuleBase::matrix veff(1, nrxx);
    std::vector<double> mu(nalpha);
    for (int a = 0; a < nalpha; ++a)
    {
        mu[a] = 0.1 * (a + 1);
    }
    for (int ir = 0; ir < nrxx; ++ir)
    {
        veff(0, ir) = std::sin(0.01 * ir); // nontrivial background
    }
    ModuleBase::matrix veff_ref = veff;

    ASSERT_TRUE(constraint::ConstraintInjectPW::inject(wg, mu, constraint::DensityChannel::Charge, veff));

    for (int ir = 0; ir < nrxx; ++ir)
    {
        double dv = 0.0;
        for (int a = 0; a < nalpha; ++a)
        {
            dv += mu[a] * wg.constraint_weight(a)[ir];
        }
        // Sum order differs between the injector (sequential += per alpha)
        // and the reference (single dv sum), so compare at machine precision.
        EXPECT_NEAR(veff(0, ir), veff_ref(0, ir) + dv, 1e-12);
    }
}

TEST_F(ConstraintInjectPWTest, ChargeChannelOnlyNspin2)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii);
    wg.build();
    const int nrxx = rhopw->nrxx;
    const int nalpha = wg.nconstraint();

    ModuleBase::matrix veff(2, nrxx);
    std::vector<double> mu(nalpha, 0.3);
    for (int is = 0; is < 2; ++is)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            veff(is, ir) = 1.0 + is;
        }
    }
    ModuleBase::matrix veff_ref = veff;

    ASSERT_TRUE(constraint::ConstraintInjectPW::inject(wg, mu, constraint::DensityChannel::Charge, veff));

    for (int ir = 0; ir < nrxx; ++ir)
    {
        double dv = 0.0;
        for (int a = 0; a < nalpha; ++a)
        {
            dv += mu[a] * wg.constraint_weight(a)[ir];
        }
        // Charge channel coupling: both spin channels gain the same +dV.
        EXPECT_NEAR(veff(0, ir), veff_ref(0, ir) + dv, 1e-12);
        EXPECT_NEAR(veff(1, ir), veff_ref(1, ir) + dv, 1e-12);
        // The spin-difference potential (up +dV, down -dV) is phase 2 and
        // must not appear: the difference of the two channels is unchanged.
        EXPECT_NEAR(veff(1, ir) - veff(0, ir), veff_ref(1, ir) - veff_ref(0, ir), 1e-12);
    }
}

TEST_F(ConstraintInjectPWTest, ObservableEqualsInjectionOperator)
{
    // Shared-instance principle: the observer and the injector consume the
    // very same WeightGrid, so the measured observable equals the injected
    // operator.  Verify with one constraint (mu, w0) that
    //   Q_0 = int w0*rho dr  ==  (1/mu) * int rho * dV dr,  dV = mu*w0.
    constraint::WeightGrid wg(*ucell, rhopw, radii);
    wg.build();
    const int nrxx = rhopw->nrxx;
    const double dV_cell = rhopw->omega / static_cast<double>(rhopw->nxyz);

    // Single-constraint fragment (all three atoms) so mu has one component.
    wg.set_constraint_atoms({{0, 1, 2}});
    std::vector<double> Q;
    const double* rho_ptr[1] = {rho.data()};
    constraint::ConstraintObserver::observe(wg, rho_ptr, 1, constraint::DensityChannel::Charge, Q);
    ASSERT_EQ(Q.size(), 1u);

    const double mu = 0.25;
    ModuleBase::matrix veff(1, nrxx);
    ASSERT_TRUE(constraint::ConstraintInjectPW::inject(wg, {mu}, constraint::DensityChannel::Charge, veff));

    double integral = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        integral += rho[ir] * veff(0, ir);
    }
    integral *= dV_cell;
    // int rho * (mu*w0) dr = mu * Q_0 up to quadrature round-off.
    EXPECT_NEAR(integral, mu * Q[0], 1e-10 * std::abs(Q[0]));

    // Two-constraint linearity: int rho * (mu0 w0 + mu1 w1) dr = sum mu_a Q_a.
    constraint::WeightGrid wg2(*ucell, rhopw, radii);
    wg2.build();
    wg2.set_constraint_atoms({{0}, {1, 2}});
    std::vector<double> mu2 = {0.1, -0.2};
    ModuleBase::matrix veff2(1, nrxx);
    ASSERT_TRUE(constraint::ConstraintInjectPW::inject(wg2, mu2, constraint::DensityChannel::Charge, veff2));
    std::vector<double> Q2;
    constraint::ConstraintObserver::observe(wg2, rho_ptr, 1, constraint::DensityChannel::Charge, Q2);
    ASSERT_EQ(Q2.size(), 2u);
    double integral2 = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        integral2 += rho[ir] * veff2(0, ir);
    }
    integral2 *= dV_cell;
    EXPECT_NEAR(integral2, mu2[0] * Q2[0] + mu2[1] * Q2[1], 1e-10 * std::max(std::abs(Q2[0]), std::abs(Q2[1])));
}

TEST_F(ConstraintInjectPWTest, SizeMismatchGuard)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii);
    wg.build();
    ModuleBase::matrix veff(1, rhopw->nrxx);
    ModuleBase::matrix veff_ref = veff;
    // Wrong mu length: reject and leave veff untouched.
    EXPECT_FALSE(constraint::ConstraintInjectPW::inject(wg, {0.1}, constraint::DensityChannel::Charge, veff));
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(veff(0, ir), veff_ref(0, ir));
    }
}

TEST_F(ConstraintInjectPWTest, SplitInjectionSpin)
{
    // Phase-2 spin channel (DeltaSpin +/- lambda semantics):
    //   V_up += mu*w, V_dn -= mu*w
    // driving the spin-difference density m = rho_up - rho_dn toward the
    // target.  The injected operator must equal the measured observable
    // (shared WeightGrid), and a single-channel buffer (nspin=1) must be
    // rejected rather than silently run a wrong operator.
    constraint::WeightGrid wg(*ucell, rhopw, radii);
    wg.build();
    const int nrxx = rhopw->nrxx;
    const int nalpha = wg.nconstraint();

    ModuleBase::matrix veff(2, nrxx);
    std::vector<double> mu(nalpha, 0.3);
    for (int is = 0; is < 2; ++is)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            veff(is, ir) = 2.0 + is;
        }
    }
    ModuleBase::matrix veff_ref = veff;

    ASSERT_TRUE(constraint::ConstraintInjectPW::inject(
        wg, mu, constraint::DensityChannel::Spin, veff));

    for (int ir = 0; ir < nrxx; ++ir)
    {
        double dv = 0.0;
        for (int a = 0; a < nalpha; ++a)
        {
            dv += mu[a] * wg.constraint_weight(a)[ir];
        }
        // Split injection: up gains +dV, down loses dV.
        EXPECT_NEAR(veff(0, ir), veff_ref(0, ir) + dv, 1e-12);
        EXPECT_NEAR(veff(1, ir), veff_ref(1, ir) - dv, 1e-12);
        // The spin-difference potential must actually change (this is the
        // discriminant against the charge channel which leaves it fixed).
        EXPECT_NEAR(veff(1, ir) - veff(0, ir),
                    veff_ref(1, ir) - veff_ref(0, ir) - 2.0 * dv, 1e-12);
    }

    // Guard: spin injection into a single-channel (nspin=1) buffer fails
    // and leaves the buffer untouched.
    ModuleBase::matrix veff1(1, nrxx);
    ModuleBase::matrix veff1_ref = veff1;
    EXPECT_FALSE(constraint::ConstraintInjectPW::inject(
        wg, mu, constraint::DensityChannel::Spin, veff1));
    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(veff1(0, ir), veff1_ref(0, ir));
    }
}
