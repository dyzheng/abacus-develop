/**
 * @file mbis_alm_e2e.cpp
 * @brief Semi-real end-to-end ALM on a real ABACUS density cube.
 *
 * Pipeline: cube -> MBIS weights -> constraint response J built from the real
 * MBIS weight overlap O_AB = int w_A w_B rho -> automatic k calibration from the
 * response -> ALM charge+spin constraint, compared with a global-step Lagrange
 * (Uzawa) and a penalty-only scheme on the same response.
 *
 * The "solver" here is a linear-response model Q = Q0 + J * coef (the inner
 * ALM self-consistency is solved exactly); the *coupling* J comes from the real
 * density/weights, so the iteration counts reflect the real spatial overlap.
 * It is the semi-real step between the pure linear model and a full KS adapter.
 *
 * Build (no ABACUS dependency):
 *   g++ -std=c++14 -O2 -I source tools/mbis_alm_e2e.cpp \
 *       source/source_estate/module_mbis_alm/mbis_alm.cpp \
 *       source/source_estate/module_mbis_alm/mbis_alm_adapter.cpp \
 *       -o /tmp/mbis_alm_e2e
 * Run:
 *   mbis_alm_e2e chg.cube [--images N] [--symmetrize] [--dc 0.2] [--ds 0.2]
 *   mbis_alm_e2e chgs1.cube chgs2.cube [--images N] [--symmetrize]
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "mbis_cube_io.h"
#include "source_estate/module_mbis_alm/mbis_alm_adapter.h"

using namespace mbis_alm;
using mbis_cube::Cube;

namespace
{

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

struct LinearSolver : public SolverAdapter
{
    std::vector<double> Qc0, Qs0;
    std::vector<std::vector<double>> Jc, Js;
    LinearSolver(const std::vector<double>& qc, const std::vector<double>& qs,
                 const std::vector<std::vector<double>>& jc,
                 const std::vector<std::vector<double>>& js)
        : Qc0(qc), Qs0(qs), Jc(jc), Js(js)
    {
    }
    Observables solve(const MbisState&, const Alm& ac, const Alm& as) override
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
                b[i] = alm.comp(i).mu + k * (Q0[i] - alm.comp(i).target);
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

struct Conv
{
    int iters = 0;
    double res = 0.0;
    bool converged = false;
};

void matvec(const std::vector<std::vector<double>>& J, const std::vector<double>& x,
            std::vector<double>& y)
{
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i)
    {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j)
            y[i] += J[i][j] * x[j];
    }
}

Conv lagrange_channel(const std::vector<double>& Q0,
                      const std::vector<std::vector<double>>& J,
                      const std::vector<double>& t, int maxit, double tol)
{
    const int n = static_cast<int>(Q0.size());
    double lmax = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += std::abs(J[i][j]);
        lmax = std::max(lmax, s);
    }
    const double alpha = (lmax > 0.0) ? 0.9 / lmax : 0.0;
    std::vector<double> mu(n, 0.0), coef(n, 0.0), Q(n, 0.0);
    Conv c;
    for (int it = 0; it < maxit; ++it)
    {
        matvec(J, coef, Q);
        double res = 0.0;
        for (int i = 0; i < n; ++i)
        {
            Q[i] += Q0[i];
            res = std::max(res, std::abs(Q[i] - t[i]));
        }
        c.iters = it + 1;
        c.res = res;
        if (res < tol)
        {
            c.converged = true;
            return c;
        }
        for (int i = 0; i < n; ++i)
        {
            mu[i] += alpha * (Q[i] - t[i]);
            coef[i] = mu[i];
        }
    }
    return c;
}

double penalty_bias_channel(const std::vector<double>& Q0,
                            const std::vector<std::vector<double>>& J,
                            const std::vector<double>& t,
                            const std::vector<double>& chi, double kappa)
{
    const int n = static_cast<int>(Q0.size());
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        const double a = std::abs(chi[i]);
        const double k = (a > 1e-12) ? kappa / a : kappa;
        b[i] = k * (Q0[i] - t[i]);
        for (int j = 0; j < n; ++j)
            A[i][j] = (i == j ? 1.0 : 0.0) - k * J[i][j];
    }
    std::vector<double> coef, Q = Q0;
    solve_linear(A, b, coef);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Q[i] += J[i][j] * coef[j];
    double res = 0.0;
    for (int i = 0; i < n; ++i)
        res = std::max(res, std::abs(Q[i] - t[i]));
    return res;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> cubes;
    int images = 0;
    bool symmetrize = false;
    double dc = 0.2, ds = 0.2;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--images") == 0 && i + 1 < argc)
            images = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--symmetrize") == 0)
            symmetrize = true;
        else if (std::strcmp(argv[i], "--dc") == 0 && i + 1 < argc)
            dc = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ds") == 0 && i + 1 < argc)
            ds = std::atof(argv[++i]);
        else
            cubes.push_back(argv[i]);
    }
    if (cubes.empty())
    {
        std::fprintf(stderr, "usage: %s cube1 [cube2] [--images N] [--symmetrize] "
                             "[--dc x] [--ds y]\n", argv[0]);
        return 2;
    }

    const Cube c1 = mbis_cube::read_cube(cubes[0]);
    std::vector<double> rho = c1.rho, spin;
    if (cubes.size() > 1)
    {
        const Cube c2 = mbis_cube::read_cube(cubes[1]);
        spin.assign(rho.size(), 0.0);
        for (std::size_t i = 0; i < rho.size(); ++i)
        {
            spin[i] = rho[i] - c2.rho[i];
            rho[i] += c2.rho[i];
        }
    }

    MbisConfig cfg;
    cfg.cutoff = 1.0e6;
    cfg.max_iter = 2000;
    cfg.tol = 1e-11;
    cfg.damp = 0.3;
    cfg.images = images;
    cfg.init_zeta = 1.0;
    cfg.symmetrize = symmetrize;

    const MbisState st = compute_mbis(rho, c1.grid, c1.atoms, cfg);
    const Observables q0 = observe(st, rho, spin, c1.grid);
    const int nat = static_cast<int>(c1.atoms.size());
    const double dV = c1.grid.dV();

    std::vector<std::vector<double>> O(nat, std::vector<double>(nat, 0.0));
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (std::size_t ig = 0; ig < c1.grid.size(); ++ig)
                O[a][b] += st.w[a][ig] * st.w[b][ig] * rho[ig] * dV;
    const double g = 0.1;
    std::vector<std::vector<double>> J(nat, std::vector<double>(nat, 0.0));
    for (int a = 0; a < nat; ++a)
        for (int b = 0; b < nat; ++b)
            for (int c = 0; c < nat; ++c)
                J[a][b] += (-g * O[a][c]) * O[c][b];
    std::vector<double> chi(nat, 0.0);
    for (int a = 0; a < nat; ++a)
        chi[a] = J[a][a];

    std::printf("grid=%dx%dx%d nat=%d\n", c1.grid.n[0], c1.grid.n[1], c1.grid.n[2], nat);
    for (int a = 0; a < nat; ++a)
        std::printf("atom %d: N0=%.6f M0=%+.6f chi=%.6e\n", a, q0.N[a], q0.M[a], chi[a]);

    // targets
    std::vector<double> t_c(nat, 0.0), t_s(nat, 0.0);
    for (int a = 0; a < nat; ++a)
    {
        t_c[a] = q0.N[a] - dc;
        t_s[a] = q0.M[a] - ds;
    }

    // ---- ALM (charge+spin), k armed from the response, wide clamp ----
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

    LinearSolver solver(q0.N, q0.M, J, J);
    DriverConfig dcfg;
    dcfg.mbis = cfg;
    dcfg.max_outer = 2000;
    dcfg.tol = 1e-8;
    const DriverResult r = run_alm_mbis(c1.grid, c1.atoms, rho, spin, alm_c, alm_s,
                                        solver, dcfg);

    // calibrated k (charge)
    double kmin = 1e18, kmax = 0.0;
    for (int a = 0; a < nat; ++a)
    {
        kmin = std::min(kmin, alm_c.comp(a).k);
        kmax = std::max(kmax, alm_c.comp(a).k);
    }
    std::printf("\nALM (charge+spin): converged=%s iters=%d residual=%.3e  k in [%.3e,%.3e]\n",
                r.converged ? "yes" : "no", r.outer_iters, r.residual, kmin, kmax);

    // ---- Lagrange (global step) and penalty-only, same response ----
    const int maxit = 200000;
    const Conv lagc = lagrange_channel(q0.N, J, t_c, maxit, 1e-8);
    const Conv lags = (spin.empty()) ? Conv{} : lagrange_channel(q0.M, J, t_s, maxit, 1e-8);
    const double penc = penalty_bias_channel(q0.N, J, t_c, chi, 1.0);
    const double pens = (spin.empty()) ? 0.0 : penalty_bias_channel(q0.M, J, t_s, chi, 1.0);

    std::printf("%-26s %12s %14s\n", "scheme (charge)", "iters", "max|Q-t|");
    std::printf("%-26s %12d %14.3e\n", "Lagrange (global alpha)", lagc.iters, lagc.res);
    std::printf("%-26s %12s %14.3e\n", "penalty-only", "-", penc);
    if (!spin.empty())
    {
        std::printf("%-26s %12s %14s\n", "scheme (spin)", "iters", "max|Q-t|");
        std::printf("%-26s %12d %14.3e\n", "Lagrange (global alpha)", lags.iters, lags.res);
        std::printf("%-26s %12s %14.3e\n", "penalty-only", "-", pens);
    }
    return 0;
}
