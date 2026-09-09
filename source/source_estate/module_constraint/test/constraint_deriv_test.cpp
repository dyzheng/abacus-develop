#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "constraint_test_utils.h"
#include "source_base/module_grid/delley.h"
#include "source_base/module_grid/partition.h"
#include "source_base/module_grid/radial.h"
#include "source_estate/module_constraint/constraint_deriv.h"
#include "source_estate/module_constraint/constraint_observe.h"

using ModuleBase::PI;

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

// Minimum-image fractional wrap, replicated from
// WeightGrid::min_image_displacement (deliberately not reusing it).
static double wrap_frac(const double v)
{
    return v - std::floor(v + 0.5);
}

// Minimum-image displacement (Bohr) from 'from' toward 'to' (Cartesian).
// The 20 Bohr cubic test box has Cartesian == fractional * 20 (lat0 = 1).
static std::array<double, 3> min_image_disp(const std::array<double, 3>& from,
                                            const std::array<double, 3>& to)
{
    std::array<double, 3> d{};
    for (int k = 0; k < 3; ++k)
    {
        d[k] = wrap_frac((to[k] - from[k]) / 20.0) * 20.0;
    }
    return d;
}

// H2O-like geometry with all atoms OFF the minimum-image tie-break
// boundaries (no fractional coordinate within 0.05 of 0.0/0.5), so the
// finite-difference reference on atom shifts never crosses an image jump.
static std::unique_ptr<UnitCell> make_h2o_ucell_offboundary()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0, // lat0 in Bohr; tau (Bohr) == Cartesian coordinates
        {20, 0, 0, 0, 20, 0, 0, 0, 20}, // 20 Bohr box
        {"O", "H"}, {"O.upf", "H.upf"}, {"upf201", "upf201"}, {"", ""},
        {1, 2}, {16.0, 1.0}, "Cartesian",
        // O--H bond 3.2 Bohr; atoms off the minimum-image tie-break
        // boundaries (no fractional coordinate within 0.03 of 0.0/0.5).
        {9.4, 9.4, 9.4, 12.6, 9.4, 9.4, 6.2, 9.4, 9.4}, // O, H1, H2 (Bohr)
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

class ConstraintDerivTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;
    std::vector<std::array<double, 3>> pos;
    std::vector<double> nelec_atom;

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell_offboundary();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 120, 120, 120); // 1/6 Bohr spacing
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
        pos = h2o_positions(*ucell);
        nelec_atom = {8.0, 1.0, 1.0};
    }

    void TearDown() override
    {
        delete rhopw;
    }

    // Atomic superposition density on the local grid (e/Bohr^3).
    void fill_rho(const double sigma, std::vector<double>& rho) const
    {
        fill_rho_basis(*rhopw, sigma, rho);
    }

    // Same, on an arbitrary PW basis (used by tests that run on a local
    // coarser grid).
    void fill_rho_basis(const ModulePW::PW_Basis& pw, const double sigma,
                        std::vector<double>& rho) const
    {
        rho.assign(pw.nrxx, 0.0);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            const int i = ir / (pw.ny * pw.nplane);
            const int j = ir / pw.nplane - i * pw.ny;
            const int k = ir % pw.nplane + pw.startz_current;
            const ModuleBase::Vector3<double> rfrac(
                static_cast<double>(i) / pw.nx,
                static_cast<double>(j) / pw.ny,
                static_cast<double>(k) / pw.nz);
            const ModuleBase::Vector3<double> rc =
                rfrac * ucell->latvec * ucell->lat0;
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                rho[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
            }
        }
    }

    // Spin-resolved atomic-superposition densities with independent per-spin
    // electron counts (up + dn need not equal nelec_atom): used by the mixed
    // charge+spin force tests (charge density = up + dn, magnetization =
    // up - dn are both smooth Gaussian superpositions).
    void fill_rho_spin_resolved(const ModulePW::PW_Basis& pw,
                                const double sigma,
                                const std::vector<double>& nelec_up,
                                const std::vector<double>& nelec_dn,
                                std::vector<double>& rho_up,
                                std::vector<double>& rho_dn) const
    {
        rho_up.assign(pw.nrxx, 0.0);
        rho_dn.assign(pw.nrxx, 0.0);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            const int i = ir / (pw.ny * pw.nplane);
            const int j = ir / pw.nplane - i * pw.ny;
            const int k = ir % pw.nplane + pw.startz_current;
            const ModuleBase::Vector3<double> rfrac(
                static_cast<double>(i) / pw.nx,
                static_cast<double>(j) / pw.ny,
                static_cast<double>(k) / pw.nz);
            const ModuleBase::Vector3<double> rc =
                rfrac * ucell->latvec * ucell->lat0;
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                rho_up[ir] += nelec_up[I] * gaussian(r, pos[I], sigma);
                rho_dn[ir] += nelec_dn[I] * gaussian(r, pos[I], sigma);
            }
        }
    }

    // Mixed fragments {{0}, {1, 2}} with their per-kind channel profiles and
    // the spin-resolved density arrays in the fixture layout.
    void fill_mixed_channels(std::vector<constraint::ChannelProfile>& channels) const
    {
        channels.push_back(constraint::build_channel_profile(
            constraint::ConstraintKind::Charge)); // fragment {0}
        channels.push_back(constraint::build_channel_profile(
            constraint::ConstraintKind::Spin)); // fragment {1, 2}
    }

    // High-order continuous quadrature of Q_alpha = int w_alpha rho dr,
    // independent of the density grid.  wpos are the atom positions used by
    // the weight function (possibly shifted); rhopos are the fixed density
    // centers.
    std::vector<double> q_quadrature(
        const std::vector<std::array<double, 3>>& wpos,
        const std::vector<std::array<double, 3>>& rhopos,
        const double sigma) const
    {
        const int nat = static_cast<int>(wpos.size());
        std::vector<double> r_ang, w_ang;
        int lmax = 35;
        Grid::Angular::delley(lmax, r_ang, w_ang);
        std::vector<double> r_rad, w_rad;
        Grid::Radial::baker(120, 8.0, r_rad, w_rad, 2);

        std::vector<double> dRR(nat * nat, 0.0);
        for (int I = 0; I < nat; ++I)
        {
            for (int J = I + 1; J < nat; ++J)
            {
                const auto dij = min_image_disp(wpos[I], wpos[J]);
                const double d = std::sqrt(dij[0] * dij[0] + dij[1] * dij[1]
                                           + dij[2] * dij[2]);
                dRR[I * nat + J] = d;
                dRR[J * nat + I] = d;
            }
        }

        std::vector<double> Q(nat, 0.0);
        std::vector<int> iR(nat);
        std::iota(iR.begin(), iR.end(), 0);
        std::vector<double> drR(nat);
        for (int I = 0; I < nat; ++I)
        {
            for (size_t irad = 0; irad < r_rad.size(); ++irad)
            {
                for (size_t iang = 0; iang < w_ang.size(); ++iang)
                {
                    const std::array<double, 3> rf = {
                        wpos[I][0] + r_rad[irad] * r_ang[3 * iang],
                        wpos[I][1] + r_rad[irad] * r_ang[3 * iang + 1],
                        wpos[I][2] + r_rad[irad] * r_ang[3 * iang + 2]};
                    double rho_val = 0.0;
                    for (int J = 0; J < nat; ++J)
                    {
                        // Min-image distance from the atom to the point (the
                        // grid convention); the quadrature points of the
                        // one-center expansions may fall outside the cell or
                        // across an image boundary, where the direct
                        // distance is the wrong branch of the periodic
                        // weight.
                        const auto disp = min_image_disp(wpos[J], rf);
                        drR[J] = std::sqrt(disp[0] * disp[0]
                                           + disp[1] * disp[1]
                                           + disp[2] * disp[2]);
                        rho_val += nelec_atom[J]
                                   * gaussian(rf, rhopos[J], sigma);
                    }
                    const double w = Grid::Partition::w_becke_adjusted(
                        nat, drR.data(), dRR.data(), radii.data(), nat,
                        iR.data(), I);
                    Q[I] += w * rho_val * w_rad[irad] * w_ang[iang] * 4.0 * PI;
                }
            }
        }
        return Q;
    }

    // Continuous quadrature of F_J,d = -sum_alpha mu_alpha
    // int rho d w_alpha/d R_J,d dr using the M0 analytic derivative kernel
    // at FIXED quadrature points (one-center expansion around the original
    // O position, independent of the WeightGrid grid assembly).  The points
    // do not move with the atoms, so d/dR of the quadrature is exactly the
    // sum of the M0 kernel values (no moving-center extra terms).
    // 'dens_coef' holds the per-atom Gaussian amplitude of the density
    // channel being integrated (total charge nelec_atom for the charge
    // channel, per-atom magnetization for the spin channel), so the same
    // fixed-point machinery serves the per-component mixed references.
    std::vector<double> f_quadrature(
        const std::vector<std::array<double, 3>>& wpos,
        const std::vector<std::array<double, 3>>& rhopos,
        const double sigma,
        const std::vector<double>& mu,
        const std::vector<double>& dens_coef) const
    {
        const int nat = static_cast<int>(wpos.size());
        std::vector<double> r_ang, w_ang;
        int lmax = 35;
        Grid::Angular::delley(lmax, r_ang, w_ang);
        std::vector<double> r_rad, w_rad;
        Grid::Radial::baker(300, 14.0, r_rad, w_rad, 2);

        std::vector<double> dRR(nat * nat, 0.0);
        for (int I = 0; I < nat; ++I)
        {
            for (int J = I + 1; J < nat; ++J)
            {
                const auto dij = min_image_disp(wpos[I], wpos[J]);
                const double d = std::sqrt(dij[0] * dij[0] + dij[1] * dij[1]
                                           + dij[2] * dij[2]);
                dRR[I * nat + J] = d;
                dRR[J * nat + I] = d;
            }
        }

        std::vector<double> F(static_cast<size_t>(nat) * 3, 0.0);
        std::vector<int> iR(nat);
        std::iota(iR.begin(), iR.end(), 0);
        std::vector<double> drR(nat), eR(3 * nat);
        // One-center expansion around the ORIGINAL O position: the fixed
        // quadrature points cover the density support (Gaussian tails below
        // 1e-8 at 6 sigma) and the Becke weight of the active constraint.
        const std::array<double, 3> center = rhopos[0];
        for (size_t irad = 0; irad < r_rad.size(); ++irad)
        {
            for (size_t iang = 0; iang < w_ang.size(); ++iang)
            {
                const std::array<double, 3> rf = {
                    center[0] + r_rad[irad] * r_ang[3 * iang],
                    center[1] + r_rad[irad] * r_ang[3 * iang + 1],
                    center[2] + r_rad[irad] * r_ang[3 * iang + 2]};
                double rho_val = 0.0;
                for (int J = 0; J < nat; ++J)
                {
                    // Min-image distances and direction cosines (grid
                    // convention); eR points from the quadrature point
                    // toward the atom, i.e. the negated atom -> point
                    // min-image displacement.
                    const auto disp = min_image_disp(wpos[J], rf);
                    drR[J] = std::sqrt(disp[0] * disp[0]
                                       + disp[1] * disp[1]
                                       + disp[2] * disp[2]);
                    eR[3 * J] = -disp[0] / drR[J];
                    eR[3 * J + 1] = -disp[1] / drR[J];
                    eR[3 * J + 2] = -disp[2] / drR[J];
                    rho_val += dens_coef[J] * gaussian(rf, rhopos[J], sigma);
                }
                const double wq = w_rad[irad] * w_ang[iang] * 4.0 * PI;
                for (int alpha = 0; alpha < nat; ++alpha)
                {
                    if (mu[alpha] == 0.0)
                    {
                        continue;
                    }
                    for (int J = 0; J < nat; ++J)
                    {
                        double dw[3] = {0.0, 0.0, 0.0};
                        Grid::Partition::w_becke_adjusted_deriv(
                            nat, drR.data(), dRR.data(), radii.data(),
                            eR.data(), nat, iR.data(), alpha, J, dw);
                        for (int d = 0; d < 3; ++d)
                        {
                            F[J * 3 + d] -= mu[alpha] * rho_val * dw[d] * wq;
                        }
                    }
                }
            }
        }
        return F;
    }

    // 5-point central FD of Q_0 w.r.t. R_J,d: independent of the M0
    // derivative kernel, pins the physical identity F_J = -mu dQ/dR_J.
    double fd_reference(const int J,
                        const int d,
                        const double sigma,
                        const std::vector<double>& mu) const
    {
        const double h = 1e-3; // Bohr; 5-point error ~ h^4 |Q^(5)|
        double q[4];
        const double shift[4] = {-2.0 * h, -h, h, 2.0 * h};
        for (int k = 0; k < 4; ++k)
        {
            auto wpos = pos;
            wpos[J][d] += shift[k];
            q[k] = q_quadrature(wpos, pos, sigma)[0];
        }
        const double dQ = (q[0] - 8.0 * q[1] + 8.0 * q[2] - q[3])
                          / (12.0 * h);
        return -mu[0] * dQ;
    }
};

TEST_F(ConstraintDerivTest, ForceOnSyntheticDensity)
{
    // Gaussian superposition density, mu = {0.1, 0, 0}: only the O
    // constraint is active, so F_J = -mu_0 dQ_0/dR_J exactly.  sigma = 1.5
    // keeps the grid quadrature error (h/sigma = 1/6, cusp planes
    // weighted by rho ~ 1e-11) below 1e-8.
    const double sigma = 1.5;
    std::vector<double> rho;
    fill_rho(sigma, rho);
    const double* rho_ptr = rho.data();
    const double* rho_arr[2] = {rho_ptr, rho_ptr};

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    std::vector<double> mu = {0.1, 0.0, 0.0};

    // Kernel force (grid integration).
    ModuleBase::matrix F(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge, mu, F);

    // Reference 1: continuous M0-kernel quadrature of the same integral.
    const std::vector<double> Fref = f_quadrature(pos, pos, sigma, mu,
                                                  nelec_atom);
    // Reference 2: 5-point FD of the quadrature Q_0 (no derivative kernel).
    for (int J = 0; J < 3; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_NEAR(F(J, d), Fref[J * 3 + d], 1e-8)
                << "M0-quadrature reference J " << J << " d " << d;
            EXPECT_NEAR(F(J, d), fd_reference(J, d, sigma, mu), 1e-8)
                << "FD-of-Q reference J " << J << " d " << d;
        }
    }
}

TEST_F(ConstraintDerivTest, NewtonThirdLaw)
{
    // sum_J F_J = -sum_alpha mu_alpha dQ_alpha/dt under a rigid translation
    // of all atoms (the plan's "weights move rigidly with the atoms" global
    // check): pointwise identity sum_J d w_alpha/dR_J = -d w_alpha/dr
    // integrated against the density.  For a non-constant density the net
    // atom force is not zero (the constraint also pushes on the density
    // field), so the reference is the right-hand side, computed by a 5-point
    // FD of the GRID observable Q_alpha (ConstraintObserver, weight VALUES
    // only — no derivative kernel) under rigid atomic translations, on the
    // same grid.  A continuous quadrature of int w_alpha grad rho dr is NOT
    // converged to 1e-8 for the sharp H-weight integrands (measured 1.7e-6
    // floor, flat from nrad 140 to 300), while the grid-observer FD agrees
    // with the kernel sum to 7e-13.  The literal "sum_J F_J = 0" form holds
    // only for constant rho, where the grid sum is contaminated by the
    // min-image tie-break cusp (O(dx), measure-zero in the continuous
    // integral) — hence the exact identity form.
    const double sigma = 1.5;
    // The identity is grid-consistent (both sides on the same grid), so a
    // local coarser grid keeps the 1e-9 assertion while halving the cost of
    // the twelve shifted-geometry rebuilds below.
    ModulePW::PW_Basis local_pw;
    local_pw.initgrids(1.0, ucell->latvec, 64, 64, 64);
    local_pw.distribute_r();
    std::vector<double> rho;
    fill_rho_basis(local_pw, sigma, rho);
    const double* rho_ptr = rho.data();
    const double* rho_arr[2] = {rho_ptr, rho_ptr};

    constraint::WeightGrid wg(*ucell, &local_pw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    const std::vector<double> mu = {0.1, -0.05, 0.2};

    ModuleBase::matrix F(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge, mu, F);

    // FD sweep of the grid observable under rigid translations: every shift
    // gives dQ_alpha/dt for all alpha at once (5-point stencil, h = 1e-3
    // Bohr; the cubic 20 Bohr box maps a Cartesian shift s to frac s/20).
    const int nat = static_cast<int>(pos.size());
    const double h = 1e-3;
    std::vector<std::vector<double>> dQdt(nat, std::vector<double>(3, 0.0));
    const double shift[4] = {-2.0 * h, -h, h, 2.0 * h};
    for (int d = 0; d < 3; ++d)
    {
        std::vector<double> q[4];
        for (int k = 0; k < 4; ++k)
        {
            for (int it = 0; it < ucell->ntype; ++it)
            {
                for (int ia = 0; ia < ucell->atoms[it].na; ++ia)
                {
                    if (d == 0) ucell->atoms[it].taud[ia].x += shift[k] / 20.0;
                    if (d == 1) ucell->atoms[it].taud[ia].y += shift[k] / 20.0;
                    if (d == 2) ucell->atoms[it].taud[ia].z += shift[k] / 20.0;
                }
            }
            constraint::WeightGrid wgs(*ucell, &local_pw, radii, constraint::WeightType::Becke);
            wgs.build();
            constraint::ConstraintObserver::observe(wgs, rho_arr, 1,
                                                    constraint::DensityChannel::Charge,
                                                    q[k]);
            for (int it = 0; it < ucell->ntype; ++it)
            {
                for (int ia = 0; ia < ucell->atoms[it].na; ++ia)
                {
                    if (d == 0) ucell->atoms[it].taud[ia].x -= shift[k] / 20.0;
                    if (d == 1) ucell->atoms[it].taud[ia].y -= shift[k] / 20.0;
                    if (d == 2) ucell->atoms[it].taud[ia].z -= shift[k] / 20.0;
                }
            }
        }
        for (int alpha = 0; alpha < nat; ++alpha)
        {
            dQdt[alpha][d] = (q[0][alpha] - 8.0 * q[1][alpha] + 8.0 * q[2][alpha]
                              - q[3][alpha])
                             / (12.0 * h);
        }
    }

    for (int d = 0; d < 3; ++d)
    {
        double sum = 0.0;
        for (int J = 0; J < nat; ++J)
        {
            sum += F(J, d);
        }
        double ref = 0.0;
        for (int alpha = 0; alpha < nat; ++alpha)
        {
            ref -= mu[alpha] * dQdt[alpha][d];
        }
        EXPECT_NEAR(sum, ref, 1e-9) << "component " << d;
    }
}

TEST_F(ConstraintDerivTest, ForceLinearInMu)
{
    // The kernel is linear in mu: F(mu1+mu2) = F(mu1)+F(mu2), and mu = 0
    // contributes exactly nothing (accumulate semantics; zero multiplier
    // short-circuits).
    const double sigma = 1.5;
    std::vector<double> rho;
    fill_rho(sigma, rho);
    const double* rho_ptr = rho.data();
    const double* rho_arr[2] = {rho_ptr, rho_ptr};

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();

    const std::vector<double> mu1 = {0.1, 0.0, 0.0};
    const std::vector<double> mu2 = {0.0, 0.3, -0.1};
    std::vector<double> mu12 = {0.1, 0.3, -0.1};

    ModuleBase::matrix F1(ucell->nat, 3), F2(ucell->nat, 3), F12(ucell->nat, 3), F0(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge, mu1, F1);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge, mu2, F2);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge, mu12, F12);
    constraint::constraint_force(wg, rho_arr, 1,
                                 constraint::DensityChannel::Charge,
                                 std::vector<double>(3, 0.0), F0);
    for (int J = 0; J < 3; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_NEAR(F12(J, d), F1(J, d) + F2(J, d), 1e-12)
                << "J " << J << " d " << d;
            EXPECT_DOUBLE_EQ(F0(J, d), 0.0);
        }
    }
}

TEST_F(ConstraintDerivTest, SpinChannelReadsMagnetization)
{
    // Spin channel reads m = rho_up - rho_dn: with rho_dn = 0 the force must
    // equal the charge-channel force on rho_up; with rho_up = rho_dn it must
    // vanish.
    const double sigma = 1.5;
    std::vector<double> rho_up, rho_dn;
    fill_rho(sigma, rho_up);
    rho_dn.assign(rhopw->nrxx, 0.0);
    const double* up = rho_up.data();
    const double* dn = rho_dn.data();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    wg.build_derivatives();
    const std::vector<double> mu = {0.1, 0.0, 0.0};

    // Spin force with rho_dn = 0 == charge force on rho_up (nspin=1 folds
    // rho[0] only; spin folds rho[0]-rho[1]).
    const double* spin_rho[2] = {up, dn};
    ModuleBase::matrix Fspin(ucell->nat, 3);
    constraint::constraint_force(wg, spin_rho, 2,
                                 constraint::DensityChannel::Spin, mu, Fspin);
    const double* chg_rho[2] = {up, up};
    ModuleBase::matrix Fchg(ucell->nat, 3);
    constraint::constraint_force(wg, chg_rho, 1,
                                 constraint::DensityChannel::Charge, mu, Fchg);
    for (int J = 0; J < 3; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_NEAR(Fspin(J, d), Fchg(J, d), 1e-12)
                << "J " << J << " d " << d;
        }
    }

    // rho_up = rho_dn -> m = 0 -> spin force exactly zero.
    const double* equal[2] = {up, up};
    ModuleBase::matrix Fzero(ucell->nat, 3);
    constraint::constraint_force(wg, equal, 2,
                                 constraint::DensityChannel::Spin, mu, Fzero);
    for (int J = 0; J < 3; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_DOUBLE_EQ(Fzero(J, d), 0.0);
        }
    }
}

TEST_F(ConstraintDerivTest, MixedChannelForce)
{
    // Per-constraint mixed kernel (A5) on a fragment map {{0}, {1, 2}} with
    // a spin-resolved density.  Checks:
    //  1. component short-circuit: F(mu_c, mu_s) == F(mu_c, 0) + F(0, mu_s),
    //     grid-exact to 1e-12 (the kernel is linear in mu and a zero
    //     multiplier contributes exactly nothing);
    //  2. the mixed call reproduces the legacy single-channel superposition
    //     bit-for-bit, so the A5 refactor does not move the homogeneous-list
    //     semantics the loop tests pin;
    //  3. per-component analytic anchors by the independent M0-kernel
    //     quadrature: the charge component over rho_up + rho_dn (== the
    //     nelec_atom Gaussian superposition) on fragment {0} and the spin
    //     component over rho_up - rho_dn on fragment {1, 2} (the fragment
    //     derivative grid is the sum of the per-atom derivatives, so the
    //     quadrature spreads the spin multiplier over the two H atoms).
    const double sigma = 1.5;
    // Up/down electron counts per atom: up + dn == nelec_atom {8, 1, 1}, so
    // the charge channel is exactly the reference Gaussian superposition,
    // while the magnetization {4, 0.6, 0.8} is NOT proportional to it — the
    // spin fold is a non-trivial rho_up - rho_dn combination.
    std::vector<double> rho_up, rho_dn;
    fill_rho_spin_resolved(*rhopw, sigma, {6.0, 0.8, 0.9}, {2.0, 0.2, 0.1},
                           rho_up, rho_dn);
    const double* up = rho_up.data();
    const double* dn = rho_dn.data();
    const double* rho_arr[2] = {up, dn};

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.set_constraint_atoms({{0}, {1, 2}});
    wg.build();
    wg.build_derivatives();
    std::vector<constraint::ChannelProfile> channels;
    fill_mixed_channels(channels); // fragment {0} charge, {1, 2} spin

    ModuleBase::matrix F_mix(ucell->nat, 3), F_c(ucell->nat, 3),
        F_s(ucell->nat, 3), F0(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 2, channels, {0.1, -0.05},
                                 F_mix);
    constraint::constraint_force(wg, rho_arr, 2, channels, {0.1, 0.0}, F_c);
    constraint::constraint_force(wg, rho_arr, 2, channels, {0.0, -0.05}, F_s);
    constraint::constraint_force(wg, rho_arr, 2, channels, {0.0, 0.0}, F0);

    // Legacy single-channel superposition (homogeneous profiles).
    ModuleBase::matrix F_c_legacy(ucell->nat, 3), F_s_legacy(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 2,
                                 constraint::DensityChannel::Charge,
                                 {0.1, 0.0}, F_c_legacy);
    constraint::constraint_force(wg, rho_arr, 2,
                                 constraint::DensityChannel::Spin,
                                 {0.0, -0.05}, F_s_legacy);

    // Independent M0-kernel quadrature anchors per component (density
    // amplitudes: charge = up + dn, spin = up - dn; the spin fragment spans
    // both H atoms, hence the multiplier spread over {1, 2}).
    const std::vector<double> Fref_c = f_quadrature(pos, pos, sigma,
                                                    {0.1, 0.0, 0.0},
                                                    {8.0, 1.0, 1.0});
    const std::vector<double> Fref_s = f_quadrature(pos, pos, sigma,
                                                    {0.0, -0.05, -0.05},
                                                    {4.0, 0.6, 0.8});
    for (int J = 0; J < ucell->nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_NEAR(F_mix(J, d), F_c(J, d) + F_s(J, d), 1e-12)
                << "component short-circuit J " << J << " d " << d;
            EXPECT_NEAR(F_mix(J, d), F_c_legacy(J, d) + F_s_legacy(J, d), 1e-12)
                << "legacy superposition J " << J << " d " << d;
            EXPECT_NEAR(F_c(J, d), F_c_legacy(J, d), 1e-12)
                << "charge homogeneous path J " << J << " d " << d;
            EXPECT_NEAR(F_s(J, d), F_s_legacy(J, d), 1e-12)
                << "spin homogeneous path J " << J << " d " << d;
            EXPECT_DOUBLE_EQ(F0(J, d), 0.0);
            EXPECT_NEAR(F_c_legacy(J, d), Fref_c[J * 3 + d], 1e-8)
                << "charge quadrature anchor J " << J << " d " << d;
            EXPECT_NEAR(F_s_legacy(J, d), Fref_s[J * 3 + d], 1e-8)
                << "spin quadrature anchor J " << J << " d " << d;
        }
    }
}

TEST_F(ConstraintDerivTest, MixedForceNewtonThirdLaw)
{
    // Native mixed third law under a rigid global translation: for every
    // alpha the pointwise grid identity sum_J d w_alpha/dR_J = -d w_alpha/dr
    // gives sum_J F_J = -sum_alpha mu_alpha dQ_alpha/dt.  The RHS comes from
    // a 5-point FD of the MIXED grid observer (observe.cpp folds every alpha
    // with its own channel signs — a different code path from the force
    // kernel), so a kernel channel-folding bug (e.g. the spin alpha reading
    // rho_up + rho_dn instead of rho_up - rho_dn) breaks the identity: this
    // test is the A5 per-channel folding falsifier.  The single-channel
    // NewtonThirdLaw above already pins the charge identity; here both
    // components sit on one list.
    const double sigma = 1.5;
    // The identity is grid-consistent (both sides on the same grid), so a
    // local coarser grid keeps the 1e-9 assertion while halving the cost of
    // the twelve shifted-geometry rebuilds below.
    ModulePW::PW_Basis local_pw;
    local_pw.initgrids(1.0, ucell->latvec, 64, 64, 64);
    local_pw.distribute_r();
    std::vector<double> rho_up, rho_dn;
    fill_rho_spin_resolved(local_pw, sigma, {6.0, 0.8, 0.9}, {2.0, 0.2, 0.1},
                           rho_up, rho_dn);
    const double* up = rho_up.data();
    const double* dn = rho_dn.data();
    const double* rho_arr[2] = {up, dn};

    constraint::WeightGrid wg(*ucell, &local_pw, radii, constraint::WeightType::Becke);
    wg.set_constraint_atoms({{0}, {1, 2}});
    wg.build();
    wg.build_derivatives();
    std::vector<constraint::ChannelProfile> channels;
    fill_mixed_channels(channels);
    const std::vector<double> mu = {0.1, -0.05};

    ModuleBase::matrix F(ucell->nat, 3);
    constraint::constraint_force(wg, rho_arr, 2, channels, mu, F);

    // FD sweep of the mixed grid observable under rigid translations: every
    // shift gives dQ_alpha/dt for both components at once (5-point stencil,
    // h = 1e-3 Bohr; the cubic 20 Bohr box maps a Cartesian shift s to frac
    // s/20).
    const int nalpha = 2;
    const double h = 1e-3;
    std::vector<std::vector<double>> dQdt(nalpha, std::vector<double>(3, 0.0));
    const double shift[4] = {-2.0 * h, -h, h, 2.0 * h};
    for (int d = 0; d < 3; ++d)
    {
        std::vector<double> q[4];
        for (int k = 0; k < 4; ++k)
        {
            for (int it = 0; it < ucell->ntype; ++it)
            {
                for (int ia = 0; ia < ucell->atoms[it].na; ++ia)
                {
                    if (d == 0) ucell->atoms[it].taud[ia].x += shift[k] / 20.0;
                    if (d == 1) ucell->atoms[it].taud[ia].y += shift[k] / 20.0;
                    if (d == 2) ucell->atoms[it].taud[ia].z += shift[k] / 20.0;
                }
            }
            constraint::WeightGrid wgs(*ucell, &local_pw, radii,
                                       constraint::WeightType::Becke);
            wgs.set_constraint_atoms({{0}, {1, 2}});
            wgs.build();
            constraint::ConstraintObserver::observe(wgs, rho_arr, 2, channels,
                                                    q[k]);
            for (int it = 0; it < ucell->ntype; ++it)
            {
                for (int ia = 0; ia < ucell->atoms[it].na; ++ia)
                {
                    if (d == 0) ucell->atoms[it].taud[ia].x -= shift[k] / 20.0;
                    if (d == 1) ucell->atoms[it].taud[ia].y -= shift[k] / 20.0;
                    if (d == 2) ucell->atoms[it].taud[ia].z -= shift[k] / 20.0;
                }
            }
        }
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            dQdt[alpha][d] = (q[0][alpha] - 8.0 * q[1][alpha]
                              + 8.0 * q[2][alpha] - q[3][alpha])
                             / (12.0 * h);
        }
    }

    for (int d = 0; d < 3; ++d)
    {
        double sum = 0.0;
        for (int J = 0; J < ucell->nat; ++J)
        {
            sum += F(J, d);
        }
        double ref = 0.0;
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            ref -= mu[alpha] * dQdt[alpha][d];
        }
        EXPECT_NEAR(sum, ref, 1e-9) << "component " << d;
    }
}
