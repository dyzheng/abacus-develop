#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <numeric>
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
#include "source_base/module_grid/delley.h"
#include "source_base/module_grid/partition.h"
#include "source_base/module_grid/radial.h"
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

// Atomic superposition density: rho(r) = sum_I N_I * gauss_I(r).
class ConstraintObserveTest : public ::testing::Test
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
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40); // 0.5 Bohr spacing, 20 Bohr box
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
        pos = h2o_positions(*ucell);
        nelec_atom = {8.0, 1.0, 1.0};
    }

    void TearDown() override
    {
        delete rhopw;
    }

    void fill_rho_coarse(const ModulePW::PW_Basis& pw, const double sigma,
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

    void fill_rho(const double sigma, std::vector<double>& rho) const
    {
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

    // High-order one-center quadrature reference of Q_I = int w_I * rho dr.
    std::vector<double> reference_charges(const double sigma,
                                          const int nrad,
                                          const int lmax,
                                          const double rcut) const
    {
        std::vector<double> r_ang, w_ang;
        int lmax_in = lmax;
        Grid::Angular::delley(lmax_in, r_ang, w_ang);
        std::vector<double> r_rad, w_rad;
        Grid::Radial::baker(nrad, rcut, r_rad, w_rad, 2);

        std::vector<double> dRR(pos.size() * pos.size(), 0.0);
        for (size_t I = 0; I < pos.size(); ++I)
        {
            for (size_t J = I + 1; J < pos.size(); ++J)
            {
                const double dx = pos[I][0] - pos[J][0];
                const double dy = pos[I][1] - pos[J][1];
                const double dz = pos[I][2] - pos[J][2];
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                dRR[I * pos.size() + J] = d;
                dRR[J * pos.size() + I] = d;
            }
        }

        std::vector<double> Q(pos.size(), 0.0);
        std::vector<int> iR(pos.size());
        for (size_t I = 0; I < pos.size(); ++I)
        {
            iR[I] = static_cast<int>(I);
        }
        for (size_t I = 0; I < pos.size(); ++I)
        {
            std::vector<double> drR(pos.size());
            for (size_t irad = 0; irad < r_rad.size(); ++irad)
            {
                for (size_t iang = 0; iang < w_ang.size(); ++iang)
                {
                    const std::array<double, 3> rf = {
                        pos[I][0] + r_rad[irad] * r_ang[3 * iang],
                        pos[I][1] + r_rad[irad] * r_ang[3 * iang + 1],
                        pos[I][2] + r_rad[irad] * r_ang[3 * iang + 2]};
                    double rho_val = 0.0;
                    for (size_t J = 0; J < pos.size(); ++J)
                    {
                        const double dx = rf[0] - pos[J][0];
                        const double dy = rf[1] - pos[J][1];
                        const double dz = rf[2] - pos[J][2];
                        drR[J] = std::sqrt(dx * dx + dy * dy + dz * dz);
                        rho_val += nelec_atom[J] * gaussian(rf, pos[J], sigma);
                    }
                    const double w = Grid::Partition::w_becke_adjusted(
                        static_cast<int>(pos.size()), drR.data(), dRR.data(),
                        radii.data(), static_cast<int>(pos.size()), iR.data(),
                        static_cast<int>(I));
                    Q[I] += w * rho_val * w_rad[irad] * w_ang[iang] * 4.0 * PI;
                }
            }
        }
        return Q;
    }
};

TEST_F(ConstraintObserveTest, AtomicSuperposition)
{
    const double sigma = 1.0; // Bohr; density scale resolved by the grid
    std::vector<double> rho;
    fill_rho(sigma, rho);
    const double* rho_ptr = rho.data();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    std::vector<double> Q;
    constraint::ConstraintObserver::observe(wg, &rho_ptr, 1, Q);

    // Sum rule: sum_alpha Q_alpha == N_el.  The partition of unity is exact
    // pointwise; the 1e-8 tolerance absorbs the floating-point accumulation
    // floor (~N_g * 2e-15 ~ 1e-10 on 6.4e4 grid points).
    const double nel = 10.0;
    double qsum = 0.0;
    for (const double q : Q)
    {
        qsum += q;
    }
    EXPECT_NEAR(qsum, nel, 1e-8);

    // Per-atom benchmark against a high-order independent quadrature of the
    // same partition rule.  The per-atom SPLIT has a grid-quadrature floor
    // of ~0.035 e at h/sigma = 0.5 (measured; both sums equal N_el exactly),
    // so the assertion is a bounded split-sensitivity cross-check; the exact
    // per-point pins are PointwiseDeltaReading and the sum rule above.
    const std::vector<double> Qref = reference_charges(sigma, 120, 35, 8.0);
    double err_half = 0.0;
    for (size_t alpha = 0; alpha < Q.size(); ++alpha)
    {
        err_half = std::max(err_half, std::abs(Q[alpha] - Qref[alpha]));
    }
    EXPECT_LT(err_half, 0.05);

    // Convergence toward the reference: halving the spacing must roughly
    // quarter the split error, so the coarser 1 Bohr grid must be farther
    // from the reference than the 0.5 Bohr grid.
    ModulePW::PW_Basis coarse_pw;
    coarse_pw.initgrids(1.0, ucell->latvec, 20, 20, 20);
    coarse_pw.distribute_r();
    constraint::WeightGrid wg_c(*ucell, &coarse_pw, radii, constraint::WeightType::Becke);
    wg_c.build();
    std::vector<double> rho_c;
    fill_rho_coarse(coarse_pw, sigma, rho_c);
    const double* rho_c_ptr = rho_c.data();
    std::vector<double> Q_c;
    constraint::ConstraintObserver::observe(wg_c, &rho_c_ptr, 1, Q_c);
    double err_one = 0.0;
    for (size_t alpha = 0; alpha < Q_c.size(); ++alpha)
    {
        err_one = std::max(err_one, std::abs(Q_c[alpha] - Qref[alpha]));
    }
    EXPECT_GT(err_one, err_half);
}

TEST_F(ConstraintObserveTest, PointwiseDeltaReading)
{
    // Grid delta rho(g*) = 1/dV at a single point g*: the reading must
    // return exactly w_alpha(g*), with no quadrature error.  This pins the
    // index mapping, the dV factor and the per-point weights simultaneously.
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    const double dV = rhopw->omega / static_cast<double>(rhopw->nxyz);
    const int nalpha = wg.nconstraint();
    std::vector<double> rho(rhopw->nrxx, 0.0);
    const double* rho_ptr = rho.data();

    for (int probe = 0; probe < 20; ++probe)
    {
        const int ir = (probe * 997) % rhopw->nrxx;
        std::fill(rho.begin(), rho.end(), 0.0);
        rho[ir] = 1.0 / dV;
        std::vector<double> Q;
        constraint::ConstraintObserver::observe(wg, &rho_ptr, 1, Q);
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            EXPECT_NEAR(Q[alpha], wg.constraint_weight(alpha)[ir], 1e-12)
                << "probe " << probe << " alpha " << alpha;
        }
    }
}

TEST_F(ConstraintObserveTest, ConstantDensitySum)
{
    // Constant density rho = 1: sum_alpha Q_alpha == cell volume exactly.
    std::vector<double> rho(rhopw->nrxx, 1.0);
    const double* rho_ptr = rho.data();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    std::vector<double> Q;
    constraint::ConstraintObserver::observe(wg, &rho_ptr, 1, Q);

    double qsum = 0.0;
    for (const double q : Q)
    {
        qsum += q;
    }
    EXPECT_NEAR(qsum, ucell->omega, 1e-8);
}


// ---------------------------------------------------------------------------
// V2b: independent Becke 1988 reference (deliberately independent code path).
// Reimplements the partition rule from the original paper (p(x) iterated 3
// times, the cell function s(mu), the per-center product and the final
// normalization) WITHOUT calling Grid::Partition::w_becke* / w_becke_adjusted,
// so a shared implementation bug cannot hide behind a shared code path.
// ---------------------------------------------------------------------------

// p(x) = (3x - x^3)/2 iterated 3 times (Becke 1988, Eq. 4).
static double ref_becke_p3(const double x)
{
    double p = 0.5 * x * (3.0 - x * x);
    p = 0.5 * p * (3.0 - p * p);
    p = 0.5 * p * (3.0 - p * p);
    return p;
}

// Cell function s(mu) = 0.5 * (1 - p3(mu)) (Becke 1988, Eq. 5).
static double ref_becke_s(const double mu)
{
    return 0.5 * (1.0 - ref_becke_p3(mu));
}

// Becke weight of center c at a grid point: P_c = prod_{J != c} s(mu_cJ),
// w_c = P_c / sum_k P_k, with mu_cJ = (|r-R_c| - |r-R_J|) / |R_c - R_J|
// (Becke 1988, Eq. 6-7).
static double ref_becke_weight(const int nat,
                               const double* drR,
                               const double* dRR,
                               const int c)
{
    std::vector<double> P(nat, 1.0);
    for (int I = 0; I < nat; ++I)
    {
        for (int J = I + 1; J < nat; ++J)
        {
            const double mu = (drR[I] - drR[J]) / dRR[I * nat + J];
            const double s = ref_becke_s(mu);
            P[I] *= s;
            P[J] *= (1.0 - s); // s(-mu) = 1 - s(mu)
        }
    }
    const double sum = std::accumulate(P.begin(), P.end(), 0.0);
    return P[c] / sum;
}

TEST_F(ConstraintObserveTest, IndependentReferenceBecke)
{
    // Homonuclear radii: the production heteronuclear size adjustment
    // vanishes for equal radii (a_ij = 0), so w_becke_adjusted degenerates to
    // plain Becke and the independent 1988 reference must agree pointwise.
    radii = {1.5, 1.5, 1.5};

    const double sigma = 1.0;
    std::vector<double> rho;
    fill_rho(sigma, rho);
    const double* rho_ptr = rho.data();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    std::vector<double> Q_prod;
    constraint::ConstraintObserver::observe(wg, &rho_ptr, 1, Q_prod);

    // Interatomic distance matrix (all atoms inside the cell, no wrapping).
    std::vector<double> dRR(pos.size() * pos.size(), 0.0);
    for (size_t I = 0; I < pos.size(); ++I)
    {
        for (size_t J = I + 1; J < pos.size(); ++J)
        {
            const double dx = pos[I][0] - pos[J][0];
            const double dy = pos[I][1] - pos[J][1];
            const double dz = pos[I][2] - pos[J][2];
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            dRR[I * pos.size() + J] = d;
            dRR[J * pos.size() + I] = d;
        }
    }

    // Same grid quadrature, independent weights: the only difference from
    // the production chain is the weight implementation itself.
    const ModulePW::PW_Basis* pw = rhopw;
    const double dV = pw->omega / static_cast<double>(pw->nxyz);
    std::vector<double> Q_ref(pos.size(), 0.0);
    std::vector<double> drR(pos.size());
    for (int ir = 0; ir < pw->nrxx; ++ir)
    {
        const int i = ir / (pw->ny * pw->nplane);
        const int j = ir / pw->nplane - i * pw->ny;
        const int k = ir % pw->nplane + pw->startz_current;
        const ModuleBase::Vector3<double> rfrac(
            static_cast<double>(i) / pw->nx,
            static_cast<double>(j) / pw->ny,
            static_cast<double>(k) / pw->nz);
        const ModuleBase::Vector3<double> rc =
            rfrac * ucell->latvec * ucell->lat0;
        const std::array<double, 3> r{rc.x, rc.y, rc.z};
        for (size_t I = 0; I < pos.size(); ++I)
        {
            const double dx = r[0] - pos[I][0];
            const double dy = r[1] - pos[I][1];
            const double dz = r[2] - pos[I][2];
            drR[I] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        for (size_t I = 0; I < pos.size(); ++I)
        {
            const double w = ref_becke_weight(static_cast<int>(pos.size()),
                                              drR.data(), dRR.data(),
                                              static_cast<int>(I));
            Q_ref[I] += w * rho[ir] * dV;
        }
    }

    for (size_t I = 0; I < Q_prod.size(); ++I)
    {
        EXPECT_NEAR(Q_prod[I], Q_ref[I], 1e-8)
            << "per-atom charge mismatch at atom " << I;
    }
}
