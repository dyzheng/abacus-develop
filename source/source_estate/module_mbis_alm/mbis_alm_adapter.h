#ifndef MODULE_MBIS_ALM_ADAPTER_H
#define MODULE_MBIS_ALM_ADAPTER_H

/**
 * @file mbis_alm_adapter.h
 * @brief Thin, decoupled interface between the standalone MBIS+ALM core and a
 *        host Kohn-Sham solver.
 *
 * The host implements SolverAdapter::solve(): given the frozen MBIS weights and
 * the ALM controllers, it converges its own inner (fixed-multiplier) problem,
 * updating the constraint potential from mu_eff = mu + k(Q - t) each SCF step,
 * and returns the self-consistent per-atom charge/spin observables.
 *
 * run_alm_mbis() then performs only the outer augmented-Lagrangian update
 *   mu += k (Q - target)
 * and repeats until |Q - target| < tol.
 */

#include "mbis_alm.h"

namespace mbis_alm
{

class SolverAdapter
{
  public:
    virtual ~SolverAdapter() = default;
    /// Converge the inner (fixed-multiplier) problem at the frozen weights and
    /// return the self-consistent per-atom charge/spin observables.
    virtual Observables solve(const MbisState& st, const Alm& alm_c, const Alm& alm_s) = 0;
};

struct DriverConfig
{
    MbisConfig mbis;
    int max_outer = 200;
    double tol = 1e-8;
};

struct DriverResult
{
    MbisState state;
    Observables obs;
    int outer_iters = 0;
    bool converged = false;
    double residual = 0.0;
};

/**
 * @brief Probe-based k calibration: measure chi_a = dQ_a/dmu_a with a
 * finite-difference solve at k=0, then arm both ALM controllers.
 * This is what a real host solver calls before the outer ALM loop.
 */
void calibrate_alm(SolverAdapter& solver, const MbisState& st,
                   Alm& alm_c, Alm& alm_s, const Observables& Q0,
                   double delta = 1e-3, double kappa = 1.0,
                   double kmin = 1e-8, double kmax = 1e9);

DriverResult run_alm_mbis(const Grid& grid, const std::vector<Atom>& atoms,
                          const std::vector<double>& rho0,
                          const std::vector<double>& spin0,
                          Alm& alm_c, Alm& alm_s, SolverAdapter& solver,
                          const DriverConfig& cfg);

} // namespace mbis_alm

#endif
