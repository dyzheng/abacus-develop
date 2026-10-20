#include "gtest/gtest.h"

#include <algorithm>
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
#include "source_estate/module_constraint/constraint_dipole.h"

using ModuleBase::PI;
using ModuleBase::Vector3;

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

class ConstraintDipoleTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<std::array<double, 3>> pos;
    // Grid Cartesian position of local point ir (mirrors the established
    // efield.cpp / constraint test indexing).
    Vector3<double> grid_pos(const int ir) const
    {
        const int i = ir / (rhopw->ny * rhopw->nplane);
        const int j = ir / rhopw->nplane - i * rhopw->ny;
        const int k = ir % rhopw->nplane + rhopw->startz_current;
        const Vector3<double> rfrac(static_cast<double>(i) / rhopw->nx,
                                    static_cast<double>(j) / rhopw->ny,
                                    static_cast<double>(k) / rhopw->nz);
        return rfrac * ucell->latvec * ucell->lat0;
    }
    // Toy density: per-atom Gaussians with electron counts 'nelec_atom',
    // optionally centered off-atom via 'centers' (empty = on-atom).
    void fill_rho(const std::vector<double>& nelec_atom,
                  const std::vector<std::array<double, 3>>& centers,
                  std::vector<double>& rho) const
    {
        const double sigma = 1.0;
        rho.assign(rhopw->nrxx, 0.0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const Vector3<double> rc = grid_pos(ir);
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                const std::array<double, 3>& c
                    = centers.empty() ? pos[I] : centers[I];
                rho[ir] += nelec_atom[I] * gaussian(r, c, sigma);
            }
        }
    }

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell(); // O at (10,10,10), H at (11.2,10,10),(8.8,10,10) Bohr
        // The test fixture does not read a pseudopotential: supply the ionic
        // valence charges used by the dipole formulas (O 6, H 1).
        ucell->atoms[0].ncpp.zv = 6.0;
        ucell->atoms[1].ncpp.zv = 1.0;
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40);
        rhopw->distribute_r();
        pos = h2o_positions(*ucell);
    }
    void TearDown() override { delete rhopw; }
};

// Neutral cell (N_I = Z*_I = {6,1,1}), O Gaussian shifted by +1 Bohr in z:
//   p = sum Z* (R_I - c_I) = 6 * (R_O - (R_O + z)) = (0, 0, -6) e*Bohr,
// independent of the origin.
TEST_F(ConstraintDipoleTest, MolecularNeutralOriginIndependent)
{
    std::vector<double> rho;
    std::vector<std::array<double, 3>> centers = pos;
    centers[0][2] += 1.0; // O electron cloud displaced +1 Bohr in z
    fill_rho({6.0, 1.0, 1.0}, centers, rho);
    const double* rp[1] = {rho.data()};

    const Vector3<double> o1(0.0, 0.0, 0.0);
    const Vector3<double> o2(3.3, -1.7, 5.1);
    const Vector3<double> p1 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o1);
    const Vector3<double> p2 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o2);
    EXPECT_NEAR(p1.z, -6.0, 1e-6);
    EXPECT_NEAR(p1.x, 0.0, 1e-6);
    EXPECT_NEAR(p1.y, 0.0, 1e-6);
    EXPECT_NEAR(p2.z, -6.0, 1e-6); // origin-independent for a neutral cell
    EXPECT_NEAR(p2.x, 0.0, 1e-6);
}

// Charged toy (N_I = {8,1,1} vs Z* = {6,1,1}, net electron excess 2):
// on-atom Gaussians -> p(o) = p(0) - Q_net * o with Q_net = sum Z* - sum N
// = -2, i.e. shifting the origin by d shifts p by +2*d.
TEST_F(ConstraintDipoleTest, MolecularChargedOriginShift)
{
    std::vector<double> rho;
    fill_rho({8.0, 1.0, 1.0}, {}, rho);
    const double* rp[1] = {rho.data()};
    const Vector3<double> o1(0.0, 0.0, 0.0);
    const Vector3<double> o2(1.0, 2.0, 3.0);
    const Vector3<double> p1 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o1);
    const Vector3<double> p2 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o2);
    // On-atom density about the ionic positions: p(0) = (ΣZ - N) R_ion.
    EXPECT_NEAR(p1.x, -20.0, 1e-6);
    EXPECT_NEAR(p1.y, -20.0, 1e-6);
    EXPECT_NEAR(p1.z, -20.0, 1e-6);
    // p(o2) - p(o1) = -Q_net (o2 - o1) = (2, 4, 6).
    EXPECT_NEAR(p2.x - p1.x, 2.0, 1e-6);
    EXPECT_NEAR(p2.y - p1.y, 4.0, 1e-6);
    EXPECT_NEAR(p2.z - p1.z, 6.0, 1e-6);
}

// Z*-weighted ionic center: fragment {0} -> R_O; fragment {1,2} -> midpoint.
TEST_F(ConstraintDipoleTest, IonicCenter)
{
    const Vector3<double> c0 = constraint::DipoleObserver::ionic_center(
        *ucell, {0});
    EXPECT_NEAR(c0.x, pos[0][0], 1e-12);
    EXPECT_NEAR(c0.z, pos[0][2], 1e-12);
    const Vector3<double> c12 = constraint::DipoleObserver::ionic_center(
        *ucell, {1, 2});
    EXPECT_NEAR(c12.x, 0.5 * (pos[1][0] + pos[2][0]), 1e-12);
    EXPECT_NEAR(c12.z, 0.5 * (pos[1][2] + pos[2][2]), 1e-12);
}

// q_dip = (p - p_ref) . (R_a - R_b) / |R_a - R_b|^2; electrons moved from
// fragment a to fragment b by delta give q_dip = +delta (a's excess positive
// charge; the LiCl convention, property-1 sec. 5.2).
TEST_F(ConstraintDipoleTest, ChargeTransferSign)
{
    const Vector3<double> p_ref(0.0, 0.0, 0.0);
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(0.0, 0.0, 4.0);
    // Moving +2 e of ELECTRON charge from a to b changes p by
    // -2 * (R_b - R_a) = (0, 0, -8).
    const Vector3<double> p(0.0, 0.0, -8.0);
    const double q = constraint::DipoleObserver::charge_transfer(p, p_ref, a, b);
    EXPECT_NEAR(q, 2.0, 1e-12);
    // Reverse direction flips the sign.
    const double q_rev = constraint::DipoleObserver::charge_transfer(
        p, p_ref, b, a);
    EXPECT_NEAR(q_rev, -2.0, 1e-12);
}

// saw() shape: periodic, continuous at pos_max and pos_max+pos_dec, unit
// ramp slope outside the decay window (locks the Efield mirror semantics).
TEST_F(ConstraintDipoleTest, SawShape)
{
    const double a = 0.4, b = 0.1; // pos_max, pos_dec
    const double fac = 1.0 - b;
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, 0.0),
                -a + 0.5 * fac, 1e-15);
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, a), 0.5 * fac, 1e-15);
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, a + b), -0.5 * fac,
                1e-15);
    // Periodicity: saw(0) == saw(1-eps) in the limit.
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, 0.0),
                constraint::DipoleObserver::saw(a, b, 1.0 - 1e-12), 1e-9);
    // Unit ramp slope in the rising region: saw(x+h) - saw(x) = h.
    const double x = 0.2, h = 1e-6;
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, x + h)
                    - constraint::DipoleObserver::saw(a, b, x),
                h, 1e-12);
}

// Saw dipole of a charge-transfer toy: neutral atomic background (N_I = Z*,
// cancels exactly) + 2 e moved from the z = 5 Bohr plane to the z = 15 Bohr
// plane (Gaussians in z, uniform in x-y).  Both planes sit in the ramp
// region (slope 1), so p_z = -2 * (15 - 5) = -20 e*Bohr, and
// dV = 4 pi p_z / A < 0 (electrons moved up -> potential higher at the
// bottom).  Box: 20 x 20 x 20 Bohr -> A = 400 Bohr^2.
TEST_F(ConstraintDipoleTest, SawDipoleAndPotentialDrop)
{
    // Density: neutral on-atom background + transfer pair.
    std::vector<double> rho(rhopw->nrxx, 0.0);
    const double sigma = 1.0;
    const std::vector<double> nelec_atom = {6.0, 1.0, 1.0};
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        const Vector3<double> rc = grid_pos(ir);
        const std::array<double, 3> r{rc.x, rc.y, rc.z};
        for (size_t I = 0; I < pos.size(); ++I)
        {
            rho[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
        }
        // Transfer pair: +2 e at z = 15, -2 e at z = 5 (electrons), each a
        // z-Gaussian smeared uniformly over x-y (amplitude normalized in z
        // only; the x-y integral is supplied by the uniform factor 1/A_xy
        // with A_xy = 400 Bohr^2).
        const double gz_up = std::exp(-std::pow((rc.z - 15.0) / sigma, 2) / 2.0)
                             / (std::sqrt(2.0 * PI) * sigma) / 400.0;
        const double gz_dn = std::exp(-std::pow((rc.z - 5.0) / sigma, 2) / 2.0)
                             / (std::sqrt(2.0 * PI) * sigma) / 400.0;
        rho[ir] += 2.0 * (gz_up - gz_dn);
    }
    const double* rp[1] = {rho.data()};
    // Frame: decay window [0.95, 1.0] in fractional z = [19, 20] Bohr, far
    // from both sheets and from the atoms (which sit at z = 10 Bohr).
    const double p = constraint::DipoleObserver::saw_dipole(
        *ucell, rhopw, rp, 1, 0.95, 0.05);
    // Atoms at z = 10 Bohr (x_frac = 0.5): saw = 0.5 - 0.95 + 0.5*0.95 =
    // 0.025 for both background and ions -> they cancel (N_I = Z*_I).
    // Sheets at z = 5/15 Bohr (x_frac 0.25/0.75, both in the ramp):
    // saw(0.25) = 0.25 - 0.95 + 0.475 = -0.225; saw(0.75) = 0.75 - 0.95 +
    // 0.475 = 0.275.  p_z = L_z * [0 - 2*(saw(0.75) - saw(0.25))] =
    // 20 * (-2 * 0.5) = -20 e*Bohr.
    // The 40^3 grid integrates the normalized z-Gaussians to ~1e-5, so the
    // toy cancellation leaves a ~2e-4 residual on the -20 e*Bohr signal.
    EXPECT_NEAR(p, -20.0, 5e-4);
    const double dv = constraint::DipoleObserver::potential_drop(p, *ucell);
    // dV = 4 pi p / A = 4 pi (-20) / 400 Hartree.
    EXPECT_NEAR(dv, -20.0 * ModuleBase::FOUR_PI / 400.0, 1e-5);
    EXPECT_LT(dv, 0.0); // electrons up -> potential higher at the bottom
}

// Auto placement: the lowest plane-averaged density of the toy is far from
// the atoms at z_frac = 0.5.
TEST_F(ConstraintDipoleTest, LowestDensityPlane)
{
    std::vector<double> rho;
    fill_rho({6.0, 1.0, 1.0}, {}, rho);
    const double* rp[1] = {rho.data()};
    const double x = constraint::DipoleObserver::lowest_density_plane(
        rhopw, rp, 1);
    // Atoms sit at z_frac = 0.5; the minimum must be far from them.
    const double d = std::fabs(x - 0.5);
    const double dwrap = std::min(d, 1.0 - d); // periodic distance
    EXPECT_GT(dwrap, 0.1);
    EXPECT_GE(x, 0.0);
    EXPECT_LT(x, 1.0);
}

// Decay-window guard: atoms at z = 10 Bohr; window [9.5, 10.5] Bohr
// (pos_max 0.475, pos_dec 0.05) contains them -> distance 0; window
// [19, 20] Bohr -> distance ~9 Bohr.
TEST_F(ConstraintDipoleTest, DecayWindowGuard)
{
    EXPECT_NEAR(constraint::DipoleObserver::min_atom_decay_distance(
                    *ucell, 0.475, 0.05),
                0.0, 1e-12);
    const double d = constraint::DipoleObserver::min_atom_decay_distance(
        *ucell, 0.95, 0.05);
    EXPECT_GT(d, 8.9);
    EXPECT_LT(d, 9.1);
}
