#ifndef CONSTRAINT_INJECT_PW_H
#define CONSTRAINT_INJECT_PW_H

#include <vector>

#include "source_base/matrix.h"
#include "weight_grid.h"

namespace constraint
{

/**
 * @brief PW effective-potential injection of the constraint operator (M3a).
 *
 *   veff(ispin, ir) += sum_alpha mu[alpha] * w_alpha(ir)
 *
 * Phase 1 wires the charge channel only.  For nspin == 1 the single channel
 * receives the potential; for nspin == 2 the same potential is added to both
 * spin channels so that it couples to the total charge rho_up + rho_dn.  The
 * spin-difference coupling (+mu on up, -mu on down) is the phase-2 magnetic
 * extension and is deliberately not implemented here.
 *
 * The injector reads the very same WeightGrid instance as the observer (M2):
 * the injected operator equals the measured observable by construction
 * ("observable == injection operator", architecture principle 2).
 *
 * @return false when mu.size() != wg.nconstraint() (the veff is left
 *         untouched); the outer loop must WARNING_QUIT in that case.
 */
class ConstraintInjectPW
{
  public:
    static bool inject(const WeightGrid& wg,
                       const std::vector<double>& mu,
                       ModuleBase::matrix& veff);
};

} // namespace constraint

#endif
