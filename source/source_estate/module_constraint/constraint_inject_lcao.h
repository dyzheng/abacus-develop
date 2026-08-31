#pragma once

#include <vector>
#include "source_lcao/module_gint/gint_info.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

namespace constraint
{

// M3b: LCAO constraint matrices W^alpha_mu,nu = ∫ phi_mu(r) w_alpha(r)
// phi_nu(r) dr, computed with the existing module_gint vlocal kernel (the
// constraint weight field w_alpha plays the role of the local potential in
// the same kernel that builds the local part of H).  No new integration
// framework: this is a thin reuse of the production Gint grid-basis loops.
//
// Per-geometry once: build() precomputes one HContainer per constraint.
// Linear in the Lagrange multipliers mu, so each SCF iteration only performs
// a mu-weighted sparse add (add_weighted) into the working Hamiltonian.
//
// Production wiring note (Task 2.3): the LCAO esolver channel injects the
// constraint potential at the v_eff grid level — the Veff operator
// integrates v_eff into H(R) via the very same cal_gint_vl kernel, so H
// contains sum_alpha mu_alpha W^alpha by linearity without calling build()
// here.  This class therefore currently serves as a continuously-executed
// unit-test audit instrument (W^alpha <-> direct-grid quadrature, sum-rule
// sum_alpha W^alpha == S) rather than the production path.  Task 2.6 may
// promote it to a runtime audit (Tr W^alpha . DM vs int w_alpha rho dr
// cross-check) or the phase-2 wrap-up review decides on removal — it is not
// dead code and must not be deleted without re-auditing those checks.
class ConstraintInjectLCAO
{
public:
    // Compute W^alpha for every constraint.  cw[alpha] is the weight field
    // of constraint alpha sampled on the Gint mesh (pw_rho layout, the same
    // grid the esolver passes as the local potential).  gint_info must be
    // the active GintInfo of the LCAO esolver (it is also registered as the
    // shared ModuleGint::Gint info).  Returns per-constraint HContainers
    // with the same layout as the LCAO Hamiltonian (gamma: real HContainer).
    static std::vector<hamilt::HContainer<double>> build(
        const std::vector<std::vector<double>>& cw,
        ModuleGint::GintInfo* gint_info);

    // H += sum_alpha mu_alpha * W^alpha in place.  W must have been built by
    // build() with the same gint_info (identical HContainer layout).  A zero
    // multiplier is skipped (no-op by value).
    static void add_weighted(
        const std::vector<double>& mu,
        const std::vector<hamilt::HContainer<double>>& W,
        hamilt::HContainer<double>* H);
};

} // namespace constraint
