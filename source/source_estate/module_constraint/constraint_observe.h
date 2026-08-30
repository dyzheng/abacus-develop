#ifndef CONSTRAINT_OBSERVE_H
#define CONSTRAINT_OBSERVE_H

#include <vector>

#include "weight_grid.h"

namespace constraint
{

/**
 * @brief Grid reading of constraint observables (architecture layer M2).
 *
 *   Q_alpha = sum_g w_alpha(g) * rho(g) * dV,   dV = omega / nxyz
 *
 * The weight field used here is the very same WeightGrid instance that the
 * potential injector (M3a) reads ("observable == injection operator", so the
 * dspin identity premise holds by construction).
 *
 * Charge channel: rho is the charge density; for nspin == 2 the two spin
 * channels are summed.  The spin channel m = rho_up - rho_dn is reserved for
 * phase 2 (magnetic moment constraints) and is not wired here.
 */
class ConstraintObserver
{
  public:
    // Q is resized to nconstraint() and filled with the rank-reduced values
    // (Parallel_Reduce::reduce_pool over the PW pool; no-op in serial).
    static void observe(const WeightGrid& wg,
                        const double* const* rho,
                        const int nspin,
                        std::vector<double>& Q);
};

} // namespace constraint

#endif
