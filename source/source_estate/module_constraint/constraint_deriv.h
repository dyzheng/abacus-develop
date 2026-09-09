#ifndef CONSTRAINT_DERIV_H
#define CONSTRAINT_DERIV_H

#include <vector>

#include "constraint_observe.h"
#include "source_base/matrix.h"
#include "weight_grid.h"

namespace constraint
{

/**
 * @brief Grid-based constraint force kernel (architecture layer M6).
 *
 *   F_J^d = -sum_alpha mu_alpha * sum_g rho(g) * d w_alpha(g)/d R_J^d * dV
 *
 * Pure grid operation on the PW density grid: the density (charge or
 * magnetization) times the position-derivative field produced by
 * WeightGrid::build_derivatives().  Both fields are basis-set independent
 * (the LCAO density is read on the same pw_rhod grid), so this single
 * kernel serves both the PW (Forces) and LCAO (Force_LCAO) force paths —
 * the architecture dividend of the realspace formulation.
 *
 * Units: mu in Ry, rho in e/Bohr^3, d w/dR in 1/Bohr, dV in Bohr^3
 * -> F in Ry/Bohr (e = 1 in atomic units).
 *
 * @param wg      Weight field with build_derivatives() already called.
 * @param rho     Per-spin densities on the same grid as wg (rho[0] = up;
 *                rho[1] = down, read only for nspin == 2).
 * @param nspin   1 or 2.
 * @param channel Charge (total density rho_up + rho_dn) or Spin
 *                (magnetization m = rho_up - rho_dn; requires nspin == 2).
 * @param mu      Per-constraint multipliers, size nconstraint() (Ry).
 * @param force   Accumulated into (nat x 3, row-major iat*3+d).  The caller
 *                zeroes the buffer before calling; the pool reduction over
 *                the density-grid ranks happens inside this kernel.
 */
void constraint_force(const WeightGrid& wg,
                      const double* const* rho,
                      const int nspin,
                      const DensityChannel channel,
                      const std::vector<double>& mu,
                      ModuleBase::matrix& force);

/**
 * @brief Per-constraint force kernel (stage A, architecture layer M6).
 *
 * Same integral as the single-channel overload, but every constraint alpha
 * folds the density with ITS OWN channel signs (d_alpha = read_up * rho_up
 * + read_dn * rho_dn), so one call serves a mixed charge+spin list — the
 * observable == injection operator premise carried over to the force
 * channel (A4 wiring consumed the per-constraint profiles; A5 moves the
 * per-alpha combination into the kernel itself, retiring the loop-side
 * two-pass masking).
 *
 * Guards (never silently run the wrong channel):
 *  - channels.size() != nconstraint() -> abort (would mis-pair weights);
 *  - a spin-like profile (read_dn != read_up) under nspin == 1 -> abort
 *    (only rho[0] exists; same contract as the observer / injector);
 *  - mu.size() / force-buffer mismatches abort (as the single-channel call).
 *
 * A zero multiplier short-circuits that alpha exactly (mu = 0 contributes
 * nothing to the force).
 *
 * @param channels Per-constraint channel profiles, parallel to mu and to
 *                 wg's constraint list (factory-built, never hand-filled).
 */
void constraint_force(const WeightGrid& wg,
                      const double* const* rho,
                      const int nspin,
                      const std::vector<ChannelProfile>& channels,
                      const std::vector<double>& mu,
                      ModuleBase::matrix& force);

} // namespace constraint

#endif
