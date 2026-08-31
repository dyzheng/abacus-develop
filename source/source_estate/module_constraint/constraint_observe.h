#ifndef CONSTRAINT_OBSERVE_H
#define CONSTRAINT_OBSERVE_H

#include <string>
#include <vector>

#include "weight_grid.h"

namespace constraint
{

// Observable channel of the constraint reading/injection (M2/M3).
enum class DensityChannel
{
    Charge, // Q_alpha = int w_alpha (rho_up + rho_dn); nspin == 1 reads rho[0]
    Spin    // Q_alpha = int w_alpha (rho_up - rho_dn); requires nspin == 2
};

// Map a validated constraint_type string ("charge" | "spin") to the channel.
inline DensityChannel channel_from_type(const std::string& type)
{
    return type == "spin" ? DensityChannel::Spin : DensityChannel::Charge;
}

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
 * channels are summed.  Spin channel: m = rho_up - rho_dn (magnetic moment
 * constraints, phase 2) and requires nspin == 2 (guarded).
 */
class ConstraintObserver
{
  public:
    // Q is resized to nconstraint() and filled with the rank-reduced values
    // (Parallel_Reduce::reduce_pool over the PW pool; no-op in serial).
    static void observe(const WeightGrid& wg,
                        const double* const* rho,
                        const int nspin,
                        const DensityChannel channel,
                        std::vector<double>& Q);
};

} // namespace constraint

#endif
