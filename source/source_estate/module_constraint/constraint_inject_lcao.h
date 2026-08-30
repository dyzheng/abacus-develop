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
