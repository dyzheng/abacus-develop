#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>

#define private public
#define protected public
#include "source_base/global_variable.h"
#include "source_base/module_grid/partition.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_constraint/constraint_dipole.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "../../test/prepare_unitcell.h"
#undef private
#undef protected

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

// Local GlobalV defaults for a serial PW unit test.
static void Set_GlobalV_Default()
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    GlobalV::NPROC_IN_POOL = 1;
}

// H2O in a 10 Bohr cubic box, O at the center, H's mirrored about x=5 Bohr.
static std::unique_ptr<UnitCell> make_h2o_ucell()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0, // lat0 in Bohr; tau (Bohr) == Cartesian coordinates
        {10, 0, 0, 0, 10, 0, 0, 0, 10}, // 10 Bohr box
        {"O", "H"}, {"O.upf", "H.upf"}, {"upf201", "upf201"}, {"", ""},
        {1, 2}, {16.0, 1.0}, "Cartesian",
        {5.0, 5.0, 5.0, 6.2, 5.0, 5.0, 3.8, 5.0, 5.0}, // O, H1, H2 (Bohr)
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

// Cartesian (Bohr) atomic positions in the same global atom order used by
// WeightGrid.  Kept local because this file defines its own 10 Bohr H2O
// fixture instead of using the shared 20 Bohr constraint test helper.
static std::vector<std::array<double, 3>> h2o_test_positions(const UnitCell& ucell)
{
    std::vector<std::array<double, 3>> pos(ucell.nat);
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double> p =
                ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
            pos[iat] = {p.x, p.y, p.z};
            ++iat;
        }
    }
    return pos;
}

class WeightGridTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40); // 0.5 Bohr spacing
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
    }

    void TearDown() override
    {
        delete rhopw;
    }

    // Global (serial) index of the mirror image of ir under x -> L - x.
    int mirror_index(const int ir) const
    {
        const int nz = rhopw->nz;
        const int ny = rhopw->ny;
        const int nx = rhopw->nx;
        const int i = ir / (ny * nz);
        const int j = ir / nz - i * ny;
        const int k = ir % nz;
        const int i2 = (nx - i) % nx;
        return (i2 * ny + j) * nz + k;
    }
};

TEST_F(WeightGridTest, PartitionOfUnity)
{
    using iclock = std::chrono::high_resolution_clock;
    const iclock::time_point t0 = iclock::now();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    const iclock::duration dur = iclock::now() - t0;

    // Per-point partition of unity on the local grid.
    double local_max = 0.0;
    const auto& cw = wg.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        double sum = 0.0;
        for (int alpha = 0; alpha < wg.nconstraint(); ++alpha)
        {
            sum += cw[alpha][ir];
        }
        local_max = std::max(local_max, std::abs(sum - 1.0));
    }
    EXPECT_LT(local_max, 1e-10);
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);

    printf("weight_grid: ngrid=%d nat=%d build_time=%.3e s (per point %.2e s)\n",
           rhopw->nrxx, wg.nat(),
           std::chrono::duration<double>(dur).count(),
           std::chrono::duration<double>(dur).count() / rhopw->nrxx);
}

// Three-centre partition: the normalized Becke product must sum to one at
// every grid point.  The pairwise identity s(mu)+s(-mu)=1 is not enough for
// more than two centres; normalization is what makes it a partition.
TEST_F(WeightGridTest, ThreeCentrePartitionOfUnity)
{
    // Move the two H atoms to an equilateral triangle around O.  The three
    // original H2O atoms are collinear, so no point is equidistant from all
    // of them and the 3/4 counterexample cannot be realized there.
    const double ox = 5.0;
    const double oy = 5.0;
    const double oz = 5.0;
    const double side = 1.5; // O-H distance, Bohr
    const std::array<std::array<double, 3>, 2> h_cart = {{
        {ox + side, oy, oz},
        {ox + 0.5 * side, oy + 0.5 * std::sqrt(3.0) * side, oz}
    }};
    for (int ia = 0; ia < 2; ++ia)
    {
        // The 10 Bohr cubic fixture has lat0=1, so direct = Cartesian/10.
        ucell->atoms[1].taud[ia] = ModuleBase::Vector3<double>(
            h_cart[ia][0] / 10.0, h_cart[ia][1] / 10.0, h_cart[ia][2] / 10.0);
    }

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    EXPECT_LT(wg.max_partition_deviation(), 1e-12);

    // Counter-check at the equilateral centroid, which is equidistant from
    // all three atoms.  Every pair then has mu=0 and s(mu)=1/2, so the
    // unnormalized three-centre product sums to 3/4; normalization is what
    // turns it into a partition.
    const std::vector<std::array<double, 3>> pos = h2o_test_positions(*ucell);
    const double px = (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0;
    const double py = (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0;
    const double pz = (pos[0][2] + pos[1][2] + pos[2][2]) / 3.0;
    double sum_raw = 0.0;
    for (int I = 0; I < 3; ++I)
    {
        double p = 1.0;
        for (int J = 0; J < 3; ++J)
        {
            // Branch: a centre does not exclude itself from the product.
            if (J == I)
            {
                continue;
            }
            const double rI = std::sqrt(std::pow(px - pos[I][0], 2)
                + std::pow(py - pos[I][1], 2)
                + std::pow(pz - pos[I][2], 2));
            const double rJ = std::sqrt(std::pow(px - pos[J][0], 2)
                + std::pow(py - pos[J][1], 2)
                + std::pow(pz - pos[J][2], 2));
            const double RIJ = std::sqrt(std::pow(pos[I][0] - pos[J][0], 2)
                + std::pow(pos[I][1] - pos[J][1], 2)
                + std::pow(pos[I][2] - pos[J][2], 2));
            p *= Grid::Partition::s_becke((rI - rJ) / RIJ);
        }
        sum_raw += p;
    }
    // The equilateral-centre limit is exactly 3/4 before normalization.
    EXPECT_NEAR(sum_raw, 0.75, 1e-12);
}

// Sum over atoms of the analytic weight derivative must vanish at every
// target atom/axis/grid point because sum_i w_i = 1 identically.  This is
// the derivative counterpart of the partition-of-unity audit.
TEST_F(WeightGridTest, DerivativeSumRule)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    const int nat = wg.nat();
    const int nrxx = wg.nrxx();
    double max_dsum = 0.0;
    int violations = 0;
    for (int J = 0; J < nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                double s = 0.0;
                for (int a = 0; a < nat; ++a)
                {
                    s += wg.weight_derivative(a, J, d, ir);
                }
                max_dsum = std::max(max_dsum, std::abs(s));
                // Branch: count any point that breaks the analytic sum rule.
                // Counting (rather than aborting on the first point) makes
                // the mirror-switch-plane baseline quantifiable.
                if (std::abs(s) > 1e-10)
                {
                    ++violations;
                }
            }
        }
    }
    printf("weight_grid derivative sum rule: max_dsum=%.3e violations=%d\n",
           max_dsum, violations);
    EXPECT_EQ(violations, 0);
    EXPECT_LT(max_dsum, 1e-10);
}

TEST_F(WeightGridTest, HirshfeldPeriodicPartitionAndRemainder)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::HirshfeldPeriodic);
    wg.build();
    ASSERT_TRUE(wg.is_hirshfeld());
    ASSERT_EQ(wg.nconstraint(), ucell->nat);

    double local_max = 0.0;
    double max_rem = 0.0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        double sum = wg.remainder_weight(ir);
        for (int iat = 0; iat < wg.nat(); ++iat)
        {
            sum += wg.atom_weight(iat)[ir];
        }
        local_max = std::max(local_max, std::abs(sum - 1.0));
        max_rem = std::max(max_rem, wg.remainder_weight(ir));
    }
    printf("hirshfeld partition: maxdev=%.3e max_remainder=%.6f\n",
           local_max, max_rem);
    EXPECT_LT(local_max, 1e-12);
    EXPECT_LT(wg.max_partition_deviation(), 1e-12);
    EXPECT_NEAR(max_rem, wg.max_remainder_weight(), 0.0);
    // The 10 Bohr box has vacuum corners farther than every compact window;
    // there the explicit remainder must take the whole weight.
    EXPECT_GT(max_rem, 0.99);
}

TEST_F(WeightGridTest, HirshfeldDerivativeSumAndFiniteDifference)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::HirshfeldPeriodic);
    wg.build();
    wg.build_derivatives();
    ASSERT_TRUE(wg.derivatives_built());

    double max_total = 0.0;
    double max_covered_atoms = 0.0;
    for (int J = 0; J < wg.nat(); ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            for (int ir = 0; ir < rhopw->nrxx; ++ir)
            {
                double sum = wg.remainder_derivative(J, d, ir);
                for (int iat = 0; iat < wg.nat(); ++iat)
                {
                    sum += wg.weight_derivative(iat, J, d, ir);
                }
                max_total = std::max(max_total, std::abs(sum));
                // Branch: in the fully covered region q_rem is exactly zero,
                // so the atom-only quotient derivative must also sum to zero.
                if (wg.remainder_weight(ir) == 0.0)
                {
                    double atom_sum = 0.0;
                    for (int iat = 0; iat < wg.nat(); ++iat)
                    {
                        atom_sum += wg.weight_derivative(iat, J, d, ir);
                    }
                    max_covered_atoms
                        = std::max(max_covered_atoms, std::abs(atom_sum));
                }
            }
        }
    }
    printf("hirshfeld derivative sum: total=%.3e covered_atoms=%.3e\n",
           max_total, max_covered_atoms);
    EXPECT_LT(max_total, 1e-10);
    EXPECT_LT(max_covered_atoms, 1e-10);

    // Central-difference one atom in Cartesian z.  The periodic sum and the
    // C2 window are smooth, so the analytic derivative must agree across the
    // old minimum-image switch planes rather than jump there.
    const double delta = 1.0e-5;
    const double dz_frac
        = delta / (ucell->latvec.e33 * ucell->lat0);
    const int J = 0;
    const double z0 = ucell->atoms[0].taud[0].z;
    ucell->atoms[0].taud[0].z = z0 + dz_frac;
    constraint::WeightGrid plus(*ucell, rhopw, radii,
                                constraint::WeightType::HirshfeldPeriodic);
    plus.build();
    const auto w_plus = plus.atom_weight(0);
    ucell->atoms[0].taud[0].z = z0 - dz_frac;
    constraint::WeightGrid minus(*ucell, rhopw, radii,
                                 constraint::WeightType::HirshfeldPeriodic);
    minus.build();
    const auto w_minus = minus.atom_weight(0);
    ucell->atoms[0].taud[0].z = z0;

    double max_abs = 0.0;
    double max_fd = 0.0;
    int max_ir = 0;
    double max_fd_value = 0.0;
    double max_analytic = 0.0;
    const auto& w_base = wg.atom_weight(0);
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        // Branch: skip the compact-window endpoint and the pure-vacuum
        // remainder region.  Both have a vanishing atomic density, where the
        // quotient is dominated by the numerical floor rather than by the
        // smooth physical window.  Interior points exercise the same smooth
        // derivative across periodic images.
        if (w_base[ir] < 1e-9 || w_base[ir] > 1.0 - 1e-9
            || w_plus[ir] < 1e-9 || w_minus[ir] < 1e-9)
        {
            continue;
        }
        const double fd = (w_plus[ir] - w_minus[ir]) / (2.0 * delta);
        const double analytic = wg.weight_derivative(0, J, 2, ir);
        if (std::abs(fd - analytic) > max_abs)
        {
            max_abs = std::abs(fd - analytic);
            max_ir = ir;
            max_fd_value = fd;
            max_analytic = analytic;
        }
        max_fd = std::max(max_fd, std::abs(fd));
    }
    printf("hirshfeld position FD: max_abs=%.3e max_deriv=%.3e ir=%d "
           "fd=%.9e analytic=%.9e\n",
           max_abs, max_fd, max_ir, max_fd_value, max_analytic);
    {
        const int i = max_ir / (rhopw->ny * rhopw->nplane);
        const int j = max_ir / rhopw->nplane - i * rhopw->ny;
        const int k = max_ir % rhopw->nplane + rhopw->startz_current;
        printf("  max point ijk=(%d,%d,%d) wplus=%.9e wminus=%.9e\n",
               i, j, k, w_plus[max_ir], w_minus[max_ir]);
        printf("  plus weights=(%.6e %.6e %.6e) rem=%.6e; minus weights="
               "(%.6e %.6e %.6e) rem=%.6e\n",
               plus.atom_weight(0)[max_ir], plus.atom_weight(1)[max_ir],
               plus.atom_weight(2)[max_ir], plus.remainder_weight(max_ir),
               minus.atom_weight(0)[max_ir], minus.atom_weight(1)[max_ir],
               minus.atom_weight(2)[max_ir], minus.remainder_weight(max_ir));
    }
    EXPECT_LT(max_abs, 1e-6 * std::max(1.0, max_fd));
}

TEST_F(WeightGridTest, HirshfeldDerivativeAcrossPeriodicBoundary)
{
    // Put O close to the x=0/10 periodic boundary.  The nearest Beeke image
    // would switch at the cell face, whereas the Hirshfeld lattice sum keeps
    // the reference density and its derivative continuous there.
    const double x0 = ucell->atoms[0].taud[0].x;
    ucell->atoms[0].taud[0].x = 0.98;
    constraint::WeightGrid base(*ucell, rhopw, radii,
                                constraint::WeightType::HirshfeldPeriodic);
    base.build();
    base.build_derivatives();

    const double delta = 1.0e-5;
    const double dx_frac = delta / (ucell->latvec.e11 * ucell->lat0);
    ucell->atoms[0].taud[0].x = 0.98 + dx_frac;
    constraint::WeightGrid plus(*ucell, rhopw, radii,
                                constraint::WeightType::HirshfeldPeriodic);
    plus.build();
    ucell->atoms[0].taud[0].x = 0.98 - dx_frac;
    constraint::WeightGrid minus(*ucell, rhopw, radii,
                                 constraint::WeightType::HirshfeldPeriodic);
    minus.build();
    ucell->atoms[0].taud[0].x = x0;

    const auto& w_base = base.atom_weight(0);
    const auto& w_plus = plus.atom_weight(0);
    const auto& w_minus = minus.atom_weight(0);
    double max_abs = 0.0;
    double max_fd = 0.0;
    int checked = 0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        // Branch: only interior points avoid the compact-window endpoint,
        // where the analytic density itself vanishes to high order.
        if (w_base[ir] < 1e-9 || w_base[ir] > 1.0 - 1e-9
            || w_plus[ir] < 1e-9 || w_minus[ir] < 1e-9)
        {
            continue;
        }
        const double fd = (w_plus[ir] - w_minus[ir]) / (2.0 * delta);
        const double analytic = base.weight_derivative(0, 0, 0, ir);
        max_abs = std::max(max_abs, std::abs(fd - analytic));
        max_fd = std::max(max_fd, std::abs(fd));
        ++checked;
    }
    printf("hirshfeld boundary FD: checked=%d max_abs=%.3e max_deriv=%.3e\n",
           checked, max_abs, max_fd);
    EXPECT_GT(checked, 0);
    EXPECT_LT(max_abs, 1e-6 * std::max(1.0, max_fd));
}

// M5: the dipole density weight plus its ionic offset must reproduce the
// passive DipoleObserver's saw dipole exactly (same saw function, same grid
// quadrature).  A uniform toy density keeps the check independent of any
// atom-centred integration error.
TEST_F(WeightGridTest, DipoleSawWeightMatchesObserver)
{
    ucell->atoms[0].ncpp.zv = 6.0;
    ucell->atoms[1].ncpp.zv = 1.0;
    const double pos_max = 0.7;
    const double pos_dec = 0.1;
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::DipoleSaw, 0.0, 2,
                              pos_max, pos_dec);
    wg.build();
    ASSERT_EQ(wg.nconstraint(), 1);
    ASSERT_TRUE(wg.is_dipole());

    std::vector<double> rho(rhopw->nrxx, 0.3);
    const double* rp[1] = {rho.data()};
    const double expected = constraint::DipoleObserver::saw_dipole(
        *ucell, rhopw, rp, 1, pos_max, pos_dec);
    const double dV = rhopw->omega / static_cast<double>(rhopw->nxyz);
    double qe = 0.0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        qe += wg.constraint_weight(0)[ir] * rho[ir];
    }
    const double q = wg.observable_offset(0) + qe * dV;
    EXPECT_NEAR(q, expected, 1e-9);

    // The density weight is geometry-independent: the derivative grid is
    // exactly zero, while the ionic offset derivative is Z*saw'.
    wg.build_derivatives();
    ASSERT_TRUE(wg.derivatives_built());
    double dmax = 0.0;
    for (const double v : wg.weight_derivatives()[0])
    {
        dmax = std::max(dmax, std::fabs(v));
    }
    EXPECT_DOUBLE_EQ(dmax, 0.0);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 0, 0), 0.0);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 0, 1), 0.0);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 0, 2), 6.0);

    // Moving the saw plane rebuilds both the weight and the offset.
    const double old_offset = wg.observable_offset(0);
    wg.set_dipole_saw_pos(0.2);
    EXPECT_DOUBLE_EQ(wg.dipole_saw_pos(), 0.2);
    EXPECT_NE(wg.observable_offset(0), old_offset);
}

// Molecular dipole weight: w = -r_axis at every grid point and the ionic
// offset is sum Z*R_axis.
TEST_F(WeightGridTest, DipoleMolecularWeightAndOffset)
{
    ucell->atoms[0].ncpp.zv = 6.0;
    ucell->atoms[1].ncpp.zv = 1.0;
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::DipoleMolecular, 0.0,
                              2);
    wg.build();
    for (int ir = 0; ir < rhopw->nrxx; ir += 37)
    {
        EXPECT_DOUBLE_EQ(wg.constraint_weight(0)[ir],
                         -wg.grid_position(ir).z);
    }
    // O at z=5, both H at z=5 -> offset = (6+1+1)*5 = 40 e*Bohr.
    EXPECT_NEAR(wg.observable_offset(0), 40.0, 1e-12);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 0, 2), 6.0);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(wg.dipole_atom_offset_derivative(0, 1, 0), 0.0);
}

TEST_F(WeightGridTest, SymmetryMirror)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();

    // Mirror reflection x -> L - x maps H1 <-> H2 and O -> O, so per-atom
    // weights at mirrored grid points must agree pointwise.
    const auto& w = wg.constraint_weights();
    double max_h = 0.0, max_o = 0.0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        const int ir2 = mirror_index(ir);
        max_h = std::max(max_h, std::abs(w[1][ir] - w[2][ir2]));
        max_o = std::max(max_o, std::abs(w[0][ir] - w[0][ir2]));
    }
    EXPECT_LT(max_h, 1e-12);
    EXPECT_LT(max_o, 1e-12);
}

TEST_F(WeightGridTest, DeterministicRebuild)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    const auto snapshot = wg.constraint_weights();
    wg.build(); // rebuild on the same geometry -> bit-identical
    const auto& again = wg.constraint_weights();
    for (int alpha = 0; alpha < wg.nconstraint(); ++alpha)
    {
        EXPECT_TRUE(snapshot[alpha] == again[alpha]);
    }
}

TEST_F(WeightGridTest, ScreeningKeepsPartition)
{
    // Aggressive cutoff (3 Bohr) screens far atoms at many grid points; the
    // partition of unity over the involved set must still hold exactly.
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke, 3.0);
    wg.build();
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);

    // At points where every atom is within the cutoff the result must equal
    // the unscreened weight (screening is a pure truncation of the neighbor
    // list, not a different recipe).
    constraint::WeightGrid full(*ucell, rhopw, radii, constraint::WeightType::Becke);
    full.build();
    double max_diff = 0.0;
    const auto& w_sc = wg.constraint_weights();
    const auto& w_full = full.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        max_diff = std::max(max_diff, std::abs(w_sc[0][ir] - w_full[0][ir]));
    }
    EXPECT_GT(max_diff, 0.0); // screening does change far-away points
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);
}

TEST_F(WeightGridTest, FragmentConstraint)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    // Fragment constraint {O, H1}: weight = w_O + w_H1 pointwise.
    wg.set_constraint_atoms({{0, 1}});
    ASSERT_EQ(wg.nconstraint(), 1);
    const auto& cw = wg.constraint_weights();
    // Reference: per-atom weights on the default constraint map.
    constraint::WeightGrid ref(*ucell, rhopw, radii, constraint::WeightType::Becke);
    ref.build();
    const auto& wref = ref.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(cw[0][ir], wref[0][ir] + wref[1][ir]);
    }
}

TEST_F(WeightGridTest, SetAtomsBeforeBuild)
{
    // Fragment map may be set before build(): build() must derive cw_ from
    // the pre-set constraint map (defensive branch A in set_constraint_atoms).
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.set_constraint_atoms({{0, 1}});
    wg.build();
    ASSERT_EQ(wg.nconstraint(), 1);
    // Same result as setting the map after build().
    constraint::WeightGrid ref(*ucell, rhopw, radii, constraint::WeightType::Becke);
    ref.build();
    ref.set_constraint_atoms({{0, 1}});
    const auto& cw = wg.constraint_weights();
    const auto& cref = ref.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(cw[0][ir], cref[0][ir]);
    }
}

// ---------------------------------------------------------------------------
// 2.5.1 M1 derivative grid: analytic vs direct kernel, translation
// invariance (strong chain-rule check), coincident-point zero derivative.
// ---------------------------------------------------------------------------

// Minimum-image fractional displacement wrap, replicated independently from
// WeightGrid::min_image_displacement (deliberately not reusing it).
static double wrap_frac(const double v)
{
    return v - std::floor(v + 0.5);
}

// Becke input arrays (drR, eR, dRR) for the 10 Bohr cubic test box
// (Cartesian == fractional * 10, lat0 = 1) at a given fractional grid point.
struct PointGeom
{
    std::vector<double> drR;
    std::vector<double> eR;
    std::vector<double> dRR;
    std::vector<double> pair_dir;
};
static PointGeom eval_geom(const std::vector<std::array<double, 3>>& taud,
                           const std::array<double, 3>& rfrac)
{
    const int nat = static_cast<int>(taud.size());
    PointGeom g;
    g.drR.assign(nat, 0.0);
    g.eR.assign(3 * nat, 0.0);
    g.dRR.assign(nat * nat, 0.0);
    g.pair_dir.assign(3 * nat * nat, 0.0);
    for (int I = 0; I < nat; ++I)
    {
        // Direction cosines must point from the grid point toward the atom
        // (M0 kernel convention).  WeightGrid builds them as the negated
        // atom -> grid-point minimum-image displacement; we replicate that
        // exactly — wrap is not odd at the +/-0.5 tie-break boundary, so
        // plain wrap(taud - rfrac) would pick the opposite image there.
        std::array<double, 3> disp{};
        for (int d = 0; d < 3; ++d)
        {
            disp[d] = -wrap_frac(rfrac[d] - taud[I][d]) * 10.0;
        }
        g.drR[I] = std::sqrt(disp[0] * disp[0] + disp[1] * disp[1]
                             + disp[2] * disp[2]);
        for (int d = 0; d < 3; ++d)
        {
            g.eR[3 * I + d] = disp[d] / g.drR[I];
        }
        for (int J = I + 1; J < nat; ++J)
        {
            std::array<double, 3> dij{};
            for (int d = 0; d < 3; ++d)
            {
                dij[d] = wrap_frac(taud[I][d] - taud[J][d]) * 10.0;
            }
            const double dIJ = std::sqrt(dij[0] * dij[0] + dij[1] * dij[1]
                                         + dij[2] * dij[2]);
            g.dRR[I * nat + J] = dIJ;
            g.dRR[J * nat + I] = dIJ;
            for (int d = 0; d < 3; ++d)
            {
                const double dir = dij[d] / dIJ;
                g.pair_dir[(I * nat + J) * 3 + d] = dir;
                g.pair_dir[(J * nat + I) * 3 + d] = -dir;
            }
        }
    }
    return g;
}

// Fractional coordinates of every atom in the cell (global atom index).
static std::vector<std::array<double, 3>> cell_taud(const UnitCell& ucell)
{
    std::vector<std::array<double, 3>> taud(ucell.nat);
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            taud[iat] = {ucell.atoms[it].taud[ia].x,
                         ucell.atoms[it].taud[ia].y,
                         ucell.atoms[it].taud[ia].z};
            ++iat;
        }
    }
    return taud;
}

// Fractional coordinate of the local grid point ir (serial: nplane == nz).
static std::array<double, 3> grid_frac(const ModulePW::PW_Basis& pw,
                                       const int ir)
{
    const int i = ir / (pw.ny * pw.nz);
    const int j = ir / pw.nz - i * pw.ny;
    const int k = ir % pw.nz;
    return {static_cast<double>(i) / pw.nx,
            static_cast<double>(j) / pw.ny,
            static_cast<double>(k) / pw.nz};
}

TEST_F(WeightGridTest, DerivGridAnalytic)
{
    // The assembled derivative grid must equal a direct per-point call of
    // the M0 kernel w_becke_adjusted_deriv (independent geometry rebuild in
    // the test, same min-image convention) for the O weight vs every atom.
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    ASSERT_TRUE(wg.derivatives_built());

    const auto taud = cell_taud(*ucell);
    std::vector<int> iR_all = {0, 1, 2};

    int checked = 0;
    for (int probe = 0; probe < 400 && checked < 20; ++probe)
    {
        const int ir = (probe * 997 + 123) % rhopw->nrxx;
        const auto rfrac = grid_frac(*rhopw, ir);
        const PointGeom g = eval_geom(taud, rfrac);
        // Skip points coinciding (or nearly coinciding) with an atom: the
        // coincident case is exact-zero and covered by its own test.
        bool near_atom = false;
        for (const double d : g.drR)
        {
            near_atom = near_atom || d < 1e-3;
        }
        if (near_atom)
        {
            continue;
        }
        for (int J = 0; J < 3; ++J)
        {
            double dw[3] = {0.0, 0.0, 0.0};
            Grid::Partition::w_becke_adjusted_deriv(
                3, g.drR.data(), g.dRR.data(), radii.data(), g.eR.data(),
                g.pair_dir.data(), 3, iR_all.data(), 0, J, dw);
            for (int d = 0; d < 3; ++d)
            {
                EXPECT_NEAR(wg.weight_derivative(0, J, d, ir), dw[d], 1e-12)
                    << "probe " << probe << " J " << J << " d " << d;
            }
        }
        ++checked;
    }
    EXPECT_GT(checked, 10);
}

TEST_F(WeightGridTest, PeriodicImagePositionDerivative)
{
    // Regression for the periodic-image mismatch: at this grid point the
    // grid-nearest images of O and H are not the minimum-image atom pair, so
    // reconstructing (R_I - R_K) from drR/eR gives the wrong dRR direction.
    // The kernel must consume the explicit pair direction and still match a
    // finite difference with respect to each atom position.
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();

    const auto taud = cell_taud(*ucell);
    const int ir = 1659; // r = (0.25, 0.25, 4.75) Bohr on this 40^3 grid
    const auto rfrac = grid_frac(*rhopw, ir);
    const PointGeom g = eval_geom(taud, rfrac);
    const std::vector<int> iR_all = {0, 1, 2};

    double max_image_mismatch = 0.0;
    for (int I = 0; I < 3; ++I)
    {
        for (int K = I + 1; K < 3; ++K)
        {
            for (int d = 0; d < 3; ++d)
            {
                const double reconstructed
                    = (g.drR[I] * g.eR[3 * I + d]
                       - g.drR[K] * g.eR[3 * K + d])
                      / g.dRR[I * 3 + K];
                max_image_mismatch
                    = std::max(max_image_mismatch,
                               std::abs(reconstructed
                                        - g.pair_dir[(I * 3 + K) * 3 + d]));
            }
        }
    }
    // Guard: the fixture must actually exercise the cross-image branch.
    EXPECT_GT(max_image_mismatch, 0.1);

    const double h = 1e-4; // Bohr
    for (int J = 0; J < 3; ++J)
    {
        auto plus = taud;
        plus[J][0] += h / 10.0; // this fixture has Cartesian = frac * 10
        auto minus = taud;
        minus[J][0] -= h / 10.0;
        const PointGeom gp = eval_geom(plus, rfrac);
        const PointGeom gm = eval_geom(minus, rfrac);
        const double wp = Grid::Partition::w_becke_adjusted(
            3, gp.drR.data(), gp.dRR.data(), radii.data(), 3, iR_all.data(), 0);
        const double wm = Grid::Partition::w_becke_adjusted(
            3, gm.drR.data(), gm.dRR.data(), radii.data(), 3, iR_all.data(), 0);
        const double fd = (wp - wm) / (2.0 * h);

        double dw[3] = {0.0, 0.0, 0.0};
        Grid::Partition::w_becke_adjusted_deriv(
            3, g.drR.data(), g.dRR.data(), radii.data(), g.eR.data(),
            g.pair_dir.data(), 3, iR_all.data(), 0, J, dw);
        EXPECT_NEAR(dw[0], fd, 1e-7) << "atom " << J;
        EXPECT_NEAR(wg.weight_derivative(0, J, 0, ir), fd, 1e-7)
            << "cached atom " << J;
    }

    // The strain kernel also consumes pair_dir, so exercise the same
    // cross-image point under a homogeneous lattice deformation.  Scaling
    // Cartesian coordinates inside a fixed box would not transform the
    // periodic images consistently, hence rebuild the weight field on the
    // strained lattice with fixed fractional coordinates.
    const ModuleBase::Matrix3 base_latvec = ucell->latvec;
    auto base_element = [&base_latvec](const int row, const int col) {
        return (&base_latvec.e11)[3 * row + col];
    };
    auto cell_element = [this](const int row, const int col) -> double& {
        return (&ucell->latvec.e11)[3 * row + col];
    };
    const double strain_step = 1e-6;
    const std::array<std::array<int, 2>, 3> strain_components = {{
        {0, 0}, {0, 2}, {2, 2}}};
    for (const auto& component : strain_components)
    {
        const int a = component[0];
        const int b = component[1];
        auto strained_weight = [&](const double step) {
            ucell->latvec = base_latvec;
            for (int row = 0; row < 3; ++row)
            {
                // Branch: perturb one deformation-gradient component of the
                // row-vector lattice matrix.
                for (int col = 0; col < 3; ++col)
                {
                    if (col == a)
                    {
                        cell_element(row, col)
                            += base_element(row, b) * step;
                    }
                }
            }
            constraint::WeightGrid strained(*ucell, rhopw, radii,
                                            constraint::WeightType::Becke);
            strained.build();
            const double value = strained.constraint_weight(0)[ir];
            ucell->latvec = base_latvec;
            return value;
        };
        const double wp = strained_weight(strain_step);
        const double wm = strained_weight(-strain_step);
        const double strain_fd = (wp - wm) / (2.0 * strain_step);

        EXPECT_NEAR(wg.weight_strain_derivative(0, a, b, ir), strain_fd,
                    1e-7)
            << "cached strain component " << a << b;
    }
}

TEST_F(WeightGridTest, DerivGridCoincidentPointZero)
{
    // O sits exactly on a grid point ((5,5,5) Bohr <-> frac (0.5,0.5,0.5)
    // on the 40^3 / 10 Bohr grid).  At r = R_c every Becke cell function
    // factor is s(+-1) = {1, 0} with s'(+-1) = 0, so all per-atom weight
    // derivatives vanish identically (the raw kernel would hit a 0/0 at
    // mu = +-1, so the build must short-circuit the point).
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();

    const int nx = rhopw->nx, ny = rhopw->ny, nz = rhopw->nz;
    const int ir = ((nx / 2) * ny + ny / 2) * nz + nz / 2;
    for (int alpha = 0; alpha < wg.nconstraint(); ++alpha)
    {
        for (int J = 0; J < wg.nat(); ++J)
        {
            for (int d = 0; d < 3; ++d)
            {
                EXPECT_DOUBLE_EQ(wg.weight_derivative(alpha, J, d, ir), 0.0);
            }
        }
    }
}

TEST_F(WeightGridTest, DerivGridTranslationInvariance)
{
    // Strong chain-rule self-check: under a rigid translation of the whole
    // system (grid point and all atoms together) the weight is unchanged,
    // so  sum_J d w_alpha/d R_J + d w_alpha/d r = 0  pointwise.  The
    // d w/d r term is obtained independently by a 5-point central
    // difference of the analytic weight on shifted grid points.
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();

    const auto taud = cell_taud(*ucell);
    std::vector<int> iR_all = {0, 1, 2};
    const double h = 1e-3; // Bohr; 5-point stencil error ~ h^4 |w^(5)|

    int checked = 0;
    for (int probe = 0; probe < 400 && checked < 20; ++probe)
    {
        const int ir = (probe * 701 + 5) % rhopw->nrxx;
        const auto rfrac0 = grid_frac(*rhopw, ir);
        const PointGeom g0 = eval_geom(taud, rfrac0);
        bool near_atom = false;
        for (const double d : g0.drR)
        {
            near_atom = near_atom || d < 1e-3;
        }
        if (near_atom)
        {
            continue;
        }
        // Skip probes near a minimum-image tie-break boundary: when any atom
        // sits at |frac displacement| >= 0.49 from the grid point (~0.1 Bohr
        // from the +/-0.5 image-jump boundary on this 0.25 Bohr grid), w as
        // a function of the atom position is non-differentiable there — the
        // wrapped image jumps as the atom crosses, so the analytic chain-rule
        // derivative (single side) and the stencil (which crosses the jump)
        // cannot agree.  The set has measure zero and no bearing on the force
        // volume integral; 0.49 leaves a conservative 0.1 Bohr margin for the
        // 1e-3 Bohr stencil.
        bool tie_break = false;
        for (int I = 0; I < wg.nat(); ++I)
        {
            for (int d = 0; d < 3; ++d)
            {
                if (std::abs(wrap_frac(rfrac0[d] - taud[I][d])) >= 0.49)
                {
                    tie_break = true;
                }
            }
        }
        if (tie_break)
        {
            continue;
        }
        for (int dd = 0; dd < 3; ++dd)
        {
            // Grid sum: sum_J d w_O / d R_J (component dd).
            double S = 0.0;
            for (int J = 0; J < 3; ++J)
            {
                S += wg.weight_derivative(0, J, dd, ir);
            }
            // Independent d w/d r_dd via the 5-point stencil.
            auto w_at_shift = [&](const double s) {
                std::array<double, 3> rf = rfrac0;
                rf[dd] += s / 10.0; // Cartesian shift s (Bohr) -> frac
                const PointGeom g = eval_geom(taud, rf);
                return Grid::Partition::w_becke_adjusted(
                    3, g.drR.data(), g.dRR.data(), radii.data(), 3,
                    iR_all.data(), 0);
            };
            const double wm2 = w_at_shift(-2.0 * h);
            const double wm1 = w_at_shift(-h);
            const double wp1 = w_at_shift(h);
            const double wp2 = w_at_shift(2.0 * h);
            const double dwdr = (wm2 - 8.0 * wm1 + 8.0 * wp1 - wp2)
                                / (12.0 * h);
            EXPECT_NEAR(S, -dwdr, 1e-10)
                << "probe " << probe << " component " << dd;
        }
        ++checked;
    }
    EXPECT_GT(checked, 10);
}

TEST_F(WeightGridTest, StrainDerivativeMatchesFiniteDifference)
{
    // The S4.2 weight stress uses
    //   d w/d eps_{ab} = sum_J (R_J - r)_b d w/d R_{J,a}
    // for a homogeneous deformation with fixed fractional coordinates.
    // Check it against a central difference of w at the same local grid
    // points, rebuilding the weight field on the strained cell.
    constraint::WeightGrid wg(*ucell, rhopw, radii,
                              constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();

    const ModuleBase::Matrix3 base = ucell->latvec;
    auto base_element = [&base](const int row, const int col) {
        return (&base.e11)[3 * row + col];
    };
    auto cell_element = [this](const int row, const int col) -> double& {
        return (&ucell->latvec.e11)[3 * row + col];
    };
    const double eps = 1e-6;
    // One diagonal, one mixed, and one diagonal on the last axis: this
    // catches an index transpose without paying for all nine components.
    const std::array<std::array<int, 2>, 4> components = {{
        {0, 0}, {0, 1}, {1, 2}, {2, 2}}};

    auto make_strained_weights = [&](const int a, const int b,
                                     const double step) {
        ucell->latvec = base;
        for (int row = 0; row < 3; ++row)
        {
            // Branch: a single deformation-gradient component changes only
            // the a-th Cartesian column by step times the b-th base column.
            for (int col = 0; col < 3; ++col)
            {
                if (col == a)
                {
                    cell_element(row, col) += base_element(row, b) * step;
                }
            }
        }
        constraint::WeightGrid strained(*ucell, rhopw, radii,
                                        constraint::WeightType::Becke);
        strained.build();
        // Copy: constraint_weight() returns a reference into `strained`,
        // which dies at the end of this lambda.
        return std::vector<double>(strained.constraint_weight(0));
    };

    int checked = 0;
    for (const auto& comp : components)
    {
        const int a = comp[0];
        const int b = comp[1];
        int component_checks = 0;
        const std::vector<double> w_plus
            = make_strained_weights(a, b, eps);
        const std::vector<double> w_minus
            = make_strained_weights(a, b, -eps);
        // Restore the base cell before the analytic evaluation (the cached
        // derivative grid was built there).
        ucell->latvec = base;

        for (int probe = 0; probe < 400 && component_checks < 6; ++probe)
        {
            const int ir = (probe * 811 + 37) % rhopw->nrxx;
            // Skip points within 0.05 Bohr of an atom or close to a
            // minimum-image switch plane where the weight is non-smooth.
            bool skip = false;
            const auto rf = grid_frac(*rhopw, ir);
            const auto taud = cell_taud(*ucell);
            for (int I = 0; I < wg.nat(); ++I)
            {
                double dr2 = 0.0;
                for (int d = 0; d < 3; ++d)
                {
                    const double df = wrap_frac(rf[d] - taud[I][d]);
                    dr2 += df * df;
                    skip = skip || std::abs(df) >= 0.49;
                }
                skip = skip || std::sqrt(dr2) * 10.0 < 0.05;
            }
            if (skip)
            {
                continue;
            }
            const double fd = (w_plus[ir] - w_minus[ir]) / (2.0 * eps);
            const double analytic
                = wg.weight_strain_derivative(0, a, b, ir);
            EXPECT_NEAR(analytic, fd, 1e-7)
                << "component " << a << b << " probe " << probe;
            ++checked;
            ++component_checks;
        }
    }
    ucell->latvec = base;
    EXPECT_GT(checked, 20);
}

TEST_F(WeightGridTest, DerivFragmentConstraint)
{
    // Per-constraint derivative = sum of the per-atom derivatives of the
    // fragment; setting the constraint map after build_derivatives must
    // re-derive the cached grid (set_constraint_atoms branch B).
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    wg.set_constraint_atoms({{0, 1}});
    ASSERT_EQ(wg.nconstraint(), 1);

    // Direct check: build a fresh grid with the fragment map set BEFORE
    // build_derivatives; both orders must agree.
    constraint::WeightGrid ref(*ucell, rhopw, radii, constraint::WeightType::Becke);
    ref.set_constraint_atoms({{0, 1}});
    ref.build();
    ref.build_derivatives();
    for (int J = 0; J < ref.nat(); ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            for (int ir = 0; ir < rhopw->nrxx; ++ir)
            {
                EXPECT_DOUBLE_EQ(wg.weight_derivative(0, J, d, ir),
                                 ref.weight_derivative(0, J, d, ir));
            }
        }
    }
}
