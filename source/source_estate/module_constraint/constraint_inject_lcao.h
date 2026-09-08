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
    // paraV: when non-null (MPI runs), the returned HContainers carry the
    // Parallel_Orbitals distribution so the Gint kernel's transfer
    // (transferSerials2Parallels) can scatter the serial grid result into
    // the per-rank layout — the same requirement the production Hamiltonian
    // HR satisfies.  Pass the esolver's Parallel_Orbitals.
    // dm_layout: optional MPI-only layout reference (pass the production DM
    // container).  When set, every returned HContainer twins its structure
    // exactly, which guarantees a non-empty per-rank target (a grid-derived
    // IJR list can be empty on a rank whose real-space sub-domain overlaps no
    // atom) and bit-identical trace() pairing with that DM.
    static std::vector<hamilt::HContainer<double>> build(
        const std::vector<std::vector<double>>& cw,
        ModuleGint::GintInfo* gint_info,
        const Parallel_Orbitals* paraV = nullptr,
        const hamilt::HContainer<double>* dm_layout = nullptr);

    // H += sum_alpha mu_alpha * W^alpha in place.  W must have been built by
    // build() with the same gint_info (identical HContainer layout).  A zero
    // multiplier is skipped (no-op by value).
    static void add_weighted(
        const std::vector<double>& mu,
        const std::vector<hamilt::HContainer<double>>& W,
        hamilt::HContainer<double>* H);

    // Real-space trace of the product of two same-layout HContainers, in the
    // ABACUS energy convention: each (iat1, iat2, R) block of A is paired
    // with the corresponding block of B and the flat products are summed
    // (the density matrix is stored with exactly the operator's layout).
    // Returns false and leaves trace untouched when the layouts are not
    // bit-identical (nnr and ijr info mismatch), so a mis-wired caller can
    // never silently read a garbage trace.
    static bool trace(const hamilt::HContainer<double>& A,
                      const hamilt::HContainer<double>& B,
                      double& trace_out);

    // M3b runtime audit (Task 2.6 fate decision: promote to production):
    // build W^alpha from the weight fields cw via the Gint vlocal kernel and
    // compare the matrix-level observable Tr[W^alpha . DM] with the grid
    // observable int w_alpha rho dr supplied in q_grid (the constraint
    // loop's last observed charges).  Returns the maximum absolute deviation
    // over alpha, or -1.0 when the audit cannot run (null DM, count
    // mismatch, or layout incompatibility — the caller should then skip the
    // check rather than abort).
    static double audit_weighted_trace(
        const std::vector<std::vector<double>>& cw,
        ModuleGint::GintInfo* gint_info,
        const hamilt::HContainer<double>* dmr,
        const std::vector<double>& q_grid,
        const Parallel_Orbitals* paraV = nullptr);
};

} // namespace constraint
