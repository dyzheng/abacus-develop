#ifndef CONSTRAINT_INJECT_PW_H
#define CONSTRAINT_INJECT_PW_H

#include <vector>

#include "source_base/matrix.h"
#include "constraint_observe.h"
#include "weight_grid.h"

namespace constraint
{

/**
 * @brief PW effective-potential injection of the constraint operator (M3a).
 *
 * Charge channel: veff(ispin, ir) += sum_alpha mu[alpha] * w_alpha(ir) on
 * every spin channel (couples to the total charge rho_up + rho_dn; nspin==1
 * is the single channel).
 * Spin channel: veff(0, ir) += mu*w, veff(1, ir) -= mu*w (couples to the
 * magnetization rho_up - rho_dn, DeltaSpin +/- lambda semantics); requires
 * an nspin == 2 buffer (guarded by a false return).
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
                       const DensityChannel channel,
                       ModuleBase::matrix& veff);

    // Per-constraint injection (stage A): every alpha injects with its own
    // channel signs, so one call can mix charge and spin components.  The
    // profile list must be parallel to wg's constraint list and to mu; any
    // mismatch returns false and leaves veff untouched.
    static bool inject(const WeightGrid& wg,
                       const std::vector<double>& mu,
                       const std::vector<ChannelProfile>& channels,
                       ModuleBase::matrix& veff);
};

} // namespace constraint

#endif
