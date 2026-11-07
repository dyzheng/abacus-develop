/**
 * @file mbis_alm_compare.cpp
 * @brief Standalone comparison of constraint solvers on a linear-response model.
 *
 * Model (per constraint component a):
 *     Q_a(v) = Q0_a - chi_a * v_a          (v = constraint potential)
 * Target t_a. The three schemes:
 *   - Lagrange (Uzawa):  v = mu,            mu += alpha (Q - t)      (global alpha)
 *   - ALM:               v = mu + k(Q - t), mu += k (Q - t)          (per-mode k = kappa/chi)
 *   - penalty only:      v = k(Q - t),      no multiplier (biased for finite k)
 *
 * A near-null response mode (chi -> 0) mimics the charge soft modes seen in the
 * LCAO near-null-mode investigation; the comparison shows why a global-step
 * Uzawa stalls while the per-mode augmented Lagrangian removes the bias quickly.
 *
 * Build/run (independent of the ABACUS build):
 *   g++ -std=c++14 -O2 -I source tools/mbis_alm_compare.cpp \
 *       source/source_estate/module_mbis_alm/mbis_alm.cpp -o /tmp/mbis_alm_compare
 *   /tmp/mbis_alm_compare
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "source_estate/module_mbis_alm/mbis_alm.h"

using namespace mbis_alm;

namespace
{

struct Result
{
    int iters = 0;
    double residual = 0.0;
    bool converged = false;
};

// Q(v) = Q0 - chi v  (component-wise)
void model_Q(const std::vector<double>& chi, const std::vector<double>& Q0,
             const std::vector<double>& v, std::vector<double>& Q)
{
    for (std::size_t a = 0; a < Q.size(); ++a)
        Q[a] = Q0[a] - chi[a] * v[a];
}

double max_residual(const std::vector<double>& Q, const std::vector<double>& t)
{
    double m = 0.0;
    for (std::size_t a = 0; a < Q.size(); ++a)
        m = std::max(m, std::abs(Q[a] - t[a]));
    return m;
}

Result run_lagrange(const std::vector<double>& chi, const std::vector<double>& Q0,
                    const std::vector<double>& t, double alpha, int maxit, double tol)
{
    const int nc = static_cast<int>(chi.size());
    std::vector<double> mu(nc, 0.0), v(nc, 0.0), Q(nc, 0.0);
    Result r;
    for (int it = 0; it < maxit; ++it)
    {
        v = mu;  // v = mu
        model_Q(chi, Q0, v, Q);
        r.residual = max_residual(Q, t);
        r.iters = it + 1;
        if (r.residual < tol)
        {
            r.converged = true;
            return r;
        }
        for (int a = 0; a < nc; ++a)
            mu[a] += alpha * (Q[a] - t[a]);
    }
    return r;
}

Result run_alm(const std::vector<double>& chi, const std::vector<double>& Q0,
               const std::vector<double>& t, double kappa, int maxit, double tol)
{
    const int nc = static_cast<int>(chi.size());
    Alm alm;
    alm.resize(nc);
    std::vector<double> chic(nc, 0.0);
    for (int a = 0; a < nc; ++a)
    {
        alm.set_target(a, t[a]);
        chic[a] = -chi[a];  // dQ/dmu = -chi
    }
    alm.arm(chic, kappa, 1e-8, 1e9);

    std::vector<double> v(nc, 0.0), Q(nc, 0.0);
    Result r;
    for (int it = 0; it < maxit; ++it)
    {
        // inner SCF (exact model solve): v = mu + k (Q(v) - t)
        for (int a = 0; a < nc; ++a)
        {
            const double mu = alm.comp(a).mu;
            const double k = alm.comp(a).k;
            v[a] = (mu + k * (Q0[a] - t[a])) / (1.0 + k * chi[a]);
        }
        model_Q(chi, Q0, v, Q);
        r.residual = max_residual(Q, t);
        r.iters = it + 1;
        if (r.residual < tol)
        {
            r.converged = true;
            return r;
        }
        alm.update(Q);
    }
    return r;
}

Result run_penalty(const std::vector<double>& chi, const std::vector<double>& Q0,
                   const std::vector<double>& t, double kappa)
{
    const int nc = static_cast<int>(chi.size());
    std::vector<double> v(nc, 0.0), Q(nc, 0.0);
    for (int a = 0; a < nc; ++a)
    {
        const double k = kappa / chi[a];
        v[a] = k * (Q0[a] - t[a]) / (1.0 + k * chi[a]);
    }
    model_Q(chi, Q0, v, Q);
    Result r;
    r.iters = 1;  // single inner fixed point
    r.residual = max_residual(Q, t);
    r.converged = r.residual < 1e-10;
    return r;
}

} // namespace

int main()
{
    // One stiff mode, two moderate, one near-null (chi -> 0).
    const std::vector<double> chi{10.0, 2.0, 0.5, 0.01};
    std::vector<double> Q0(4, 1.0);
    std::vector<double> t(4, 0.0);
    const double tol = 1e-10;
    const int maxit = 200000;

    // global-step Uzawa: stable requires alpha < 2/chi_max = 0.2
    const double alpha = 0.1;

    const Result lag = run_lagrange(chi, Q0, t, alpha, maxit, tol);
    const Result alm1 = run_alm(chi, Q0, t, 1.0, maxit, tol);
    const Result alm10 = run_alm(chi, Q0, t, 10.0, maxit, tol);
    const Result pen1 = run_penalty(chi, Q0, t, 1.0);
    const Result pen10 = run_penalty(chi, Q0, t, 10.0);

    std::printf("linear-response model, chi = {");
    for (std::size_t a = 0; a < chi.size(); ++a)
        std::printf("%s%.3g", a ? ", " : "", chi[a]);
    std::printf("}, tol=%.0e\n\n", tol);

    std::printf("%-22s %10s %14s %8s\n", "scheme", "iters", "max|Q-t|", "conv");
    auto row = [](const char* name, const Result& r) {
        std::printf("%-22s %10d %14.3e %8s\n", name, r.iters, r.residual,
                    r.converged ? "yes" : "no");
    };
    row("Lagrange (global a)", lag);
    row("ALM k=1/chi", alm1);
    row("ALM k=10/chi", alm10);
    row("penalty-only k=1/chi", pen1);
    row("penalty-only k=10/chi", pen10);

    // interpretation / sanity assertions
    const bool ok = alm1.converged && alm10.converged
                    && (alm1.iters < lag.iters)
                    && (pen1.residual > 1e-6) && (pen10.residual < pen1.residual);
    std::printf("\n%-22s %s\n", "consistency",
                ok ? "PASS (ALM fast+exact; penalty biased; Lagrange stalls on "
                     "near-null with global step)" : "FAIL");
    return ok ? 0 : 1;
}
