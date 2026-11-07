#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "source_estate/module_mbis_alm/mbis_alm.h"
#include "source_estate/module_mbis_alm/pw_grid_bridge.h"

using namespace mbis_alm;

namespace
{

struct Toy
{
    Grid grid;
    std::vector<Atom> atoms;
    std::vector<double> rho;
    std::vector<double> spin;
};

Toy make_toy()
{
    Toy t;
    t.grid.n = {{24, 24, 24}};
    t.grid.origin = Vec3{0.0, 0.0, 0.0};
    t.grid.dv = {{Vec3{0.4, 0.0, 0.0}, Vec3{0.0, 0.4, 0.0}, Vec3{0.0, 0.0, 0.4}}};
    t.atoms = {{Vec3{2.4, 4.8, 4.8}, 8.0}, {Vec3{7.2, 4.8, 4.8}, 8.0}};
    const std::size_t n = t.grid.size();
    t.rho.assign(n, 0.0);
    t.spin.assign(n, 0.0);
    for (int ix = 0; ix < 24; ++ix)
        for (int iy = 0; iy < 24; ++iy)
            for (int iz = 0; iz < 24; ++iz)
            {
                const std::size_t ig = (static_cast<std::size_t>(ix) * 24 + iy) * 24 + iz;
                const Vec3 p = t.grid.point(ix, iy, iz);
                double rho = 0.0, spin = 0.0;
                for (int a = 0; a < 2; ++a)
                {
                    const Vec3 d = p - t.atoms[a].R;
                    const double g = std::exp(-dot(d, d) / (2.0 * 1.0 * 1.0));
                    rho += 8.0 * g;
                    spin += (a == 0 ? 1.0 : -1.0) * 4.0 * g;  // toy AFM
                }
                t.rho[ig] = rho;
                t.spin[ig] = spin;
            }
    return t;
}

MbisConfig cfg()
{
    MbisConfig c;
    c.cutoff = 1.0e6;  // no cutoff -> smooth finite-difference
    c.max_iter = 800;
    c.tol = 1e-11;
    c.damp = 0.3;
    c.images = 0;
    c.init_zeta = 1.0;
    return c;
}

} // namespace

TEST(MbisAlmTest, PartitionOfUnity)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, cfg());
    double maxdev = 0.0;
    for (std::size_t ig = 0; ig < t.grid.size(); ++ig)
    {
        double s = 0.0;
        for (int a = 0; a < 2; ++a)
            s += st.w[a][ig];
        maxdev = std::max(maxdev, std::abs(s - 1.0));
    }
    EXPECT_LT(maxdev, 1e-12);
}

TEST(MbisAlmTest, ChargeSumRule)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, cfg());
    const Observables o = observe(st, t.rho, t.spin, t.grid);
    double ne = 0.0;
    for (double v : t.rho)
        ne += v * t.grid.dV();
    EXPECT_NEAR(o.N[0] + o.N[1], ne, 1e-9);
    double mtot = 0.0;
    for (double v : t.spin)
        mtot += v * t.grid.dV();
    EXPECT_NEAR(o.M[0] + o.M[1], mtot, 1e-9);
}

TEST(MbisAlmTest, MbisMeanRadiusFixedPoint)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, cfg());
    // zeta_a should equal 3 N_a / <r>_a with <r> from the converged weights.
    for (int a = 0; a < 2; ++a)
    {
        double na = 0.0, mr = 0.0;
        for (int ix = 0; ix < 24; ++ix)
            for (int iy = 0; iy < 24; ++iy)
                for (int iz = 0; iz < 24; ++iz)
                {
                    const std::size_t ig = (static_cast<std::size_t>(ix) * 24 + iy) * 24 + iz;
                    const Vec3 p = t.grid.point(ix, iy, iz);
                    const double r = norm(p - t.atoms[a].R);
                    na += st.w[a][ig] * t.rho[ig] * t.grid.dV();
                    mr += r * st.w[a][ig] * t.rho[ig] * t.grid.dV();
                }
        EXPECT_NEAR(st.zeta[a], 3.0 * na / mr, 1e-6);
        EXPECT_NEAR(st.N[a], na, 1e-6);
    }
}

TEST(MbisAlmTest, ConstraintPotential)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, cfg());
    const std::vector<double> coef{0.7, -0.2};
    const std::vector<double> v = constraint_potential(st, coef, t.grid);
    for (std::size_t ig = 0; ig < t.grid.size(); ig += 101)
    {
        EXPECT_NEAR(v[ig], coef[0] * st.w[0][ig] + coef[1] * st.w[1][ig], 1e-14);
    }
}

TEST(MbisAlmTest, ForceFiniteDifference)
{
    const Toy t = make_toy();
    const MbisConfig c = cfg();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, c);
    const std::vector<double> mu_c{0.7, -0.4};
    const std::vector<double> mu_s{0.3, -0.2};
    const std::vector<Vec3> F = constraint_force(st, t.grid, t.atoms, t.rho, t.spin, mu_c, mu_s);

    const double h = 1e-4;
    for (int J = 0; J < 2; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            std::vector<Atom> ap = t.atoms, am = t.atoms;
            double* rp = (d == 0) ? &ap[J].R.x : (d == 1) ? &ap[J].R.y : &ap[J].R.z;
            double* rm = (d == 0) ? &am[J].R.x : (d == 1) ? &am[J].R.y : &am[J].R.z;
            *rp += h;
            *rm -= h;
            const MbisState sp = weights_from_params(t.rho, t.grid, ap, st.N, st.zeta, c.images, c.cutoff);
            const MbisState sm = weights_from_params(t.rho, t.grid, am, st.N, st.zeta, c.images, c.cutoff);
            const Observables op = observe(sp, t.rho, t.spin, t.grid);
            const Observables om = observe(sm, t.rho, t.spin, t.grid);
            double ep = 0.0, em = 0.0;
            for (int a = 0; a < 2; ++a)
            {
                ep += mu_c[a] * op.N[a] + mu_s[a] * op.M[a];
                em += mu_c[a] * om.N[a] + mu_s[a] * om.M[a];
            }
            const double ffd = (ep - em) / (2.0 * h);
            const double fana = (d == 0) ? F[J].x : (d == 1) ? F[J].y : F[J].z;
            EXPECT_NEAR(fana, ffd, 2e-4 * std::max(1.0, std::abs(ffd)));
        }
    }
}

TEST(MbisAlmTest, AlmConvergesExactlyOneStep)
{
    // model Q(mu) = Q0 - chi*mu ; target t ; mu* = (Q0-t)/chi.
    const double chi = 0.5, Q0 = 1.0, t = 0.2;
    Alm alm;
    alm.resize(1);
    alm.set_target(0, t);
    alm.arm({-chi});  // dQ/dmu = -chi
    EXPECT_NEAR(alm.comp(0).k, 2.0, 1e-12);       // kappa/|chi| = 1/0.5
    EXPECT_NEAR(alm.damping(0), 0.5, 1e-12);      // 1/(1+k|chi|)
    for (int n = 0; n < 5; ++n)
    {
        const double Q = Q0 - chi * alm.comp(0).mu;
        alm.update({Q});
    }
    const double Q = Q0 - chi * alm.comp(0).mu;
    EXPECT_NEAR(Q, t, 1e-12);
    EXPECT_NEAR(alm.comp(0).mu, (Q0 - t) / chi, 1e-10);
    EXPECT_NEAR(alm.energy_correction({Q}), 0.0, 1e-12);
}

TEST(MbisAlmTest, PenaltyOnlyHasBias)
{
    // penalty-only fixed point of v = k (Q(v) - t), Q = Q0 - chi v,
    // gives Q != t for finite k (this is why the Lagrange update is needed).
    const double chi = 0.5, Q0 = 1.0, t = 0.2, k = 2.0;
    const double v = k * (Q0 - t) / (1.0 + k * chi);
    const double Q = Q0 - chi * v;
    // finite-k penalty leaves a bias |Q - t| ~ chi*(Q0-t)/(1+k*chi);
    // the Lagrange/ALM update removes it (tested in AlmConvergesExactlyOneStep).
    EXPECT_GT(std::abs(Q - t), 0.3);
}

namespace
{
// bcc conventional cell (skewed voxels) with two equivalent Fe-like sites:
// the raw image-sum partition breaks the site equivalence, so --symmetrize
// must restore equal N and zeta (regression for the in-place-average bug).
Toy make_bcc()
{
    Toy t;
    t.grid.n = {{30, 30, 30}};
    t.grid.origin = Vec3{0.0, 0.0, 0.0};
    t.grid.dv = {{Vec3{0.273, 0.1365, 0.1365}, Vec3{0.1365, 0.273, 0.1365},
                  Vec3{0.1365, 0.1365, 0.273}}};
    const Vec3 c0 = 30.0 * t.grid.dv[0];
    const Vec3 c1 = 30.0 * t.grid.dv[1];
    const Vec3 c2 = 30.0 * t.grid.dv[2];
    const Vec3 center = 0.5 * (c0 + c1 + c2);
    t.atoms = {{Vec3{0.0, 0.0, 0.0}, 16.0}, {center, 16.0}};
    const std::size_t n = t.grid.size();
    t.rho.assign(n, 0.0);
    t.spin.assign(n, 0.0);
    for (int ix = 0; ix < 30; ++ix)
        for (int iy = 0; iy < 30; ++iy)
            for (int iz = 0; iz < 30; ++iz)
            {
                const std::size_t ig = (static_cast<std::size_t>(ix) * 30 + iy) * 30 + iz;
                const Vec3 p = t.grid.point(ix, iy, iz);
                double rho = 0.0, spin = 0.0;
                for (int a = 0; a < 2; ++a)
                {
                    const Vec3 d = p - t.atoms[a].R;
                    const double g = std::exp(-dot(d, d) / (2.0 * 1.0 * 1.0));
                    rho += 16.0 * g;
                    spin += (a == 0 ? 1.0 : -1.0) * 8.0 * g;
                }
                t.rho[ig] = rho;
                t.spin[ig] = spin;
            }
    return t;
}
} // namespace

TEST(MbisAlmTest, SymmetrizeRestoresEquivalence)
{
    const Toy t = make_bcc();
    MbisConfig c = cfg();
    c.images = 1;
    c.max_iter = 400;
    c.tol = 1e-9;
    c.damp = 0.5;
    c.symmetrize = false;
    const MbisState raw = compute_mbis(t.rho, t.grid, t.atoms, c);
    c.symmetrize = true;
    const MbisState sym = compute_mbis(t.rho, t.grid, t.atoms, c);

    // the raw skewed-cell partition is asymmetric...
    EXPECT_GT(std::abs(raw.N[0] - raw.N[1]), 1e-3);
    // ...and the symmetry projection restores equivalent sites.
    EXPECT_LT(std::abs(sym.N[0] - sym.N[1]), 1e-8);
    EXPECT_LT(std::abs(sym.zeta[0] - sym.zeta[1]), 1e-8);
    EXPECT_NEAR(sym.N[0] + sym.N[1], [&] {
        double ne = 0.0;
        for (double v : t.rho)
            ne += v * t.grid.dV();
        return ne;
    }(), 1e-8);
}

TEST(MbisAlmTest, GridBridgeMatchesLattice)
{
    auto det3 = [](const Vec3 a[3]) {
        return a[0].x * (a[1].y * a[2].z - a[1].z * a[2].y)
             - a[0].y * (a[1].x * a[2].z - a[1].z * a[2].x)
             + a[0].z * (a[1].x * a[2].y - a[1].y * a[2].x);
    };
    // Fe2 bcc conventional cell (Bohr): a_i = LATTICE_CONSTANT * vectors.
    {
        int n[3] = {30, 30, 30};
        Vec3 a[3] = {{8.19, 4.095, 4.095}, {4.095, 8.19, 4.095}, {4.095, 4.095, 8.19}};
        const Grid g = make_grid(n, a);
        EXPECT_EQ(g.n[0], 30);
        EXPECT_EQ(g.n[2], 30);
        EXPECT_NEAR(g.dV() * 30.0 * 30.0 * 30.0, std::abs(det3(a)), 1e-9);
        const Vec3 p = g.point(29, 0, 0);
        EXPECT_NEAR(p.x, 29.0 * 8.19 / 30.0, 1e-12);
        EXPECT_NEAR(p.y, 29.0 * 4.095 / 30.0, 1e-12);
    }
    // LiCl cubic box (Bohr): L = 15 * LATTICE_CONSTANT.
    {
        int n[3] = {192, 192, 192};
        const double L = 15.0 * 1.8897261254578284;
        Vec3 a[3] = {{L, 0.0, 0.0}, {0.0, L, 0.0}, {0.0, 0.0, L}};
        const Grid g = make_grid(n, a);
        EXPECT_NEAR(g.dV() * 192.0 * 192.0 * 192.0, L * L * L, 1e-3);
    }
    const auto at = make_atoms({{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}}, {16.0, 7.0});
    EXPECT_DOUBLE_EQ(at[0].Zval, 16.0);
    EXPECT_DOUBLE_EQ(at[1].R.z, 3.0);
}
