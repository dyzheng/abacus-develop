#include "mbis_alm_adapter.h"

#include <algorithm>
#include <cmath>

namespace mbis_alm
{

DriverResult run_alm_mbis(const Grid& grid, const std::vector<Atom>& atoms,
                          const std::vector<double>& rho0,
                          const std::vector<double>& spin0,
                          Alm& alm_c, Alm& alm_s, SolverAdapter& solver,
                          const DriverConfig& cfg)
{
    DriverResult r;
    const int nat = static_cast<int>(atoms.size());
    r.state = compute_mbis(rho0, grid, atoms, cfg.mbis);

    for (int it = 0; it < cfg.max_outer; ++it)
    {
        r.obs = solver.solve(r.state, alm_c, alm_s);

        double res = 0.0;
        for (int a = 0; a < nat; ++a)
        {
            res = std::max(res, std::abs(r.obs.N[a] - alm_c.comp(a).target));
            res = std::max(res, std::abs(r.obs.M[a] - alm_s.comp(a).target));
        }
        r.outer_iters = it + 1;
        r.residual = res;
        if (res < cfg.tol)
        {
            r.converged = true;
            break;
        }
        alm_c.update(r.obs.N);
        alm_s.update(r.obs.M);
    }
    return r;
}

} // namespace mbis_alm
