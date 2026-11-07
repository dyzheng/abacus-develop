#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "source_estate/module_mbis_alm/mbis_alm_adapter.h"

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
                    const double g = std::exp(-dot(d, d) / 2.0);
                    rho += 8.0 * g;
                    spin += (a == 0 ? 1.0 : -1.0) * 4.0 * g;
                }
                t.rho[ig] = rho;
                t.spin[ig] = spin;
            }
    return t;
}

MbisConfig mbis_cfg()
{
    MbisConfig c;
    c.cutoff = 1.0e6;
    c.max_iter = 800;
    c.tol = 1e-11;
    c.damp = 0.3;
    c.images = 0;
    c.init_zeta = 1.0;
    return c;
}

bool solve_linear(std::vector<std::vector<double>> A, std::vector<double> b,
                  std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    for (int i = 0; i < n; ++i)
    {
        int p = i;
        for (int r = i + 1; r < n; ++r)
            if (std::abs(A[r][i]) > std::abs(A[p][i]))
                p = r;
        if (std::abs(A[p][i]) < 1e-14)
            return false;
        std::swap(A[i], A[p]);
        std::swap(b[i], b[p]);
        for (int r = i + 1; r < n; ++r)
        {
            const double f = A[r][i] / A[i][i];
            for (int c = i; c < n; ++c)
                A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double s = b[i];
        for (int c = i + 1; c < n; ++c)
            s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return true;
}

// Mock host solver: the inner problem is the linear response
//   Q = Q0 + J * coef,   coef = mu + k (Q - t)   (ALM potential iteration)
// which is solved exactly by  (I - K J) coef = mu + k (Q0 - t).
struct MockSolver : public SolverAdapter
{
    std::vector<double> Qc0, Qs0;
    std::vector<std::vector<double>> Jc, Js;

    MockSolver(const std::vector<double>& qc, const std::vector<double>& qs,
               const std::vector<std::vector<double>>& jc,
               const std::vector<std::vector<double>>& js)
        : Qc0(qc), Qs0(qs), Jc(jc), Js(js)
    {
    }

    Observables solve(const MbisState& /*st*/, const Alm& ac, const Alm& as) override
    {
        const int nat = static_cast<int>(Qc0.size());
        Observables o;
        o.N = Qc0;
        o.M = Qs0;

        auto inner = [&](const Alm& alm, const std::vector<double>& Q0,
                         const std::vector<std::vector<double>>& J,
                         std::vector<double>& out) {
            std::vector<std::vector<double>> A(nat, std::vector<double>(nat, 0.0));
            std::vector<double> b(nat, 0.0);
            for (int i = 0; i < nat; ++i)
            {
                const double k = alm.comp(i).k;
                const double mu = alm.comp(i).mu;
                const double t = alm.comp(i).target;
                b[i] = mu + k * (Q0[i] - t);
                for (int j = 0; j < nat; ++j)
                    A[i][j] = (i == j ? 1.0 : 0.0) - k * J[i][j];
            }
            std::vector<double> coef;
            solve_linear(A, b, coef);
            for (int i = 0; i < nat; ++i)
            {
                out[i] = Q0[i];
                for (int j = 0; j < nat; ++j)
                    out[i] += J[i][j] * coef[j];
            }
        };
        inner(ac, Qc0, Jc, o.N);
        inner(as, Qs0, Js, o.M);
        return o;
    }
};

} // namespace

TEST(MbisAlmAdapterTest, EndToEndAlmConverges)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, mbis_cfg());
    const Observables q0 = observe(st, t.rho, t.spin, t.grid);
    const int nat = 2;
    const double dV = t.grid.dV();

    std::vector<std::vector<double>> O(nat, std::vector<double>(nat, 0.0));
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (std::size_t ig = 0; ig < t.grid.size(); ++ig)
                O[a][b] += st.w[a][ig] * st.w[b][ig] * t.rho[ig] * dV;

    const double g = 0.1;
    std::vector<std::vector<double>> Jc(nat, std::vector<double>(nat, 0.0));
    std::vector<std::vector<double>> Js = Jc;
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (int c = 0; c < nat; ++c)
            {
                Jc[a][b] += (-g * O[a][c]) * O[c][b];
                Js[a][b] += (-g * O[a][c]) * O[c][b];
            }

    std::vector<double> chi(nat, 0.0);
    for (int a = 0; a < nat; ++a)
        chi[a] = Jc[a][a];

    const std::vector<double> t_c{q0.N[0] - 0.8, q0.N[1] - 0.8};
    const std::vector<double> t_s{q0.M[0] - 0.5, q0.M[1] - 0.5};

    Alm alm_c, alm_s;
    alm_c.resize(nat);
    alm_s.resize(nat);
    for (int a = 0; a < nat; ++a)
    {
        alm_c.set_target(a, t_c[a]);
        alm_s.set_target(a, t_s[a]);
    }
    alm_c.arm(chi, 1.0, 1e-10, 1e9);
    alm_s.arm(chi, 1.0, 1e-10, 1e9);

    MockSolver mock(q0.N, q0.M, Jc, Js);
    DriverConfig cfg;
    cfg.mbis = mbis_cfg();
    cfg.max_outer = 500;
    cfg.tol = 1e-8;
    const DriverResult r = run_alm_mbis(t.grid, t.atoms, t.rho, t.spin, alm_c, alm_s, mock, cfg);

    EXPECT_TRUE(r.converged) << "outer=" << r.outer_iters << " res=" << r.residual;
    for (int a = 0; a < nat; ++a)
    {
        EXPECT_NEAR(r.obs.N[a], t_c[a], 1e-6);
        EXPECT_NEAR(r.obs.M[a], t_s[a], 1e-6);
    }
}

TEST(MbisAlmAdapterTest, ChargeAndSpinBothConstrained)
{
    const Toy t = make_toy();
    const MbisState st = compute_mbis(t.rho, t.grid, t.atoms, mbis_cfg());
    const Observables q0 = observe(st, t.rho, t.spin, t.grid);
    const int nat = 2;
    const double dV = t.grid.dV();
    std::vector<std::vector<double>> O(nat, std::vector<double>(nat, 0.0));
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (std::size_t ig = 0; ig < t.grid.size(); ++ig)
                O[a][b] += st.w[a][ig] * st.w[b][ig] * t.rho[ig] * dV;
    std::vector<std::vector<double>> J(nat, std::vector<double>(nat, 0.0));
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (int c = 0; c < nat; ++c)
                J[a][b] += (-0.1 * O[a][c]) * O[c][b];
    std::vector<double> chi(nat, 0.0);
    for (int a = 0; a < nat; ++a)
        chi[a] = J[a][a];

    const std::vector<double> t_c{q0.N[0] - 0.4, q0.N[1] - 0.4};
    const std::vector<double> t_s{q0.M[0] + 0.6, q0.M[1] - 0.6};

    Alm alm_c, alm_s;
    alm_c.resize(nat);
    alm_s.resize(nat);
    for (int a = 0; a < nat; ++a)
    {
        alm_c.set_target(a, t_c[a]);
        alm_s.set_target(a, t_s[a]);
    }
    alm_c.arm(chi, 1.0, 1e-10, 1e9);
    alm_s.arm(chi, 1.0, 1e-10, 1e9);
    MockSolver mock(q0.N, q0.M, J, J);

    DriverConfig cfg;
    cfg.mbis = mbis_cfg();
    cfg.max_outer = 500;
    cfg.tol = 1e-8;
    const DriverResult r = run_alm_mbis(t.grid, t.atoms, t.rho, t.spin, alm_c, alm_s, mock, cfg);

    EXPECT_TRUE(r.converged) << "outer=" << r.outer_iters << " res=" << r.residual;
    for (int a = 0; a < nat; ++a)
    {
        EXPECT_NEAR(r.obs.N[a], t_c[a], 1e-6);
        EXPECT_NEAR(r.obs.M[a], t_s[a], 1e-6);
    }
}
