#ifndef ALLOC_PW_H
#define ALLOC_PW_H

#include <complex>
#include <vector>

namespace alloc
{

/**
 * @brief Orthonormalized target-projector space for the PW backend.
 *
 * The raw projectors alpha_a have an R x R Hermitian overlap
 * ST_ab = <alpha_a|alpha_b>.
 * They are orthonormalized with G = ST^{-1/2}; the lower-level PW adapter
 * supplies becp = <alpha_a|psi> and local products V alpha_a.  This mirrors
 * the LCAO target space without ever forming the full plane-wave basis.
 */
struct PwTargetSpace
{
    int ntarget = 0;
    int nconstraint = 0;
    std::vector<int> atom;       // target -> global atom
    std::vector<int> constraint; // target -> constraint, -1 for unowned
    std::vector<std::complex<double>> ST; // ntarget x ntarget, row-major
    std::vector<std::complex<double>> ST_inv_root; // ntarget x ntarget
    int rank = 0;
    double cond = 1.0;
};

/**
 * @brief Select one angular-momentum shell from the onsite projectors.
 *
 * OnsiteProjector stores, for each atom, the rows of l = 0, 1, ..., lmax in
 * the ABACUS real-spherical-harmonic order.  The row offset of shell l is
 * therefore l^2 and its width is 2*l+1.  nh_per_atom is the total number of
 * onsite rows on each atom ((nwl+1)^2); an atom without the requested shell is
 * skipped.
 */
struct PwTargetShell
{
    std::vector<int> row;  // compact target -> onsite-projector row
    std::vector<int> atom; // compact target -> global atom
};

PwTargetShell select_pw_target_shell(const std::vector<int>& nh_per_atom,
                                     const int target_l);

/**
 * @brief Build the PW target space from projector-projector overlaps.
 *
 * ST is Hermitian-symmetrized and truncated by the same rank threshold as the
 * LCAO backend.  target_constraint may contain -1 for a target that belongs
 * to the global P but not to a constrained Q_i.
 */
void build_pw_target(const std::vector<std::complex<double>>& ST,
                     const int ntarget,
                     const std::vector<int>& target_atom,
                     const std::vector<int>& target_constraint,
                     const int nconstraint,
                     PwTargetSpace& out,
                     const double rank_tol = 1e-10);

/** @brief Real-overlap compatibility overload (Gamma-point inputs). */
void build_pw_target(const std::vector<double>& ST,
                     const int ntarget,
                     const std::vector<int>& target_atom,
                     const std::vector<int>& target_constraint,
                     const int nconstraint,
                     PwTargetSpace& out,
                     const double rank_tol = 1e-10);

/**
 * @brief Build ST = <alpha_a|alpha_b> from the OnsiteProjector tabulation.
 *
 * tab_atomic is laid out [target * npw + ig] and stores <G+k|alpha_target>
 * on the rank-local plane-wave partition, matching OnsiteProjector.  The
 * small overlap matrix is pooled before return.
 */
void build_pw_projector_overlap(const std::complex<double>* tab_atomic,
                                const int npw,
                                const int ntarget,
                                std::vector<std::complex<double>>& ST);

/**
 * @brief Compute becp[band][target] = <alpha_target|psi_band> from the
 * OnsiteProjector tabulation and plane-wave coefficients.
 *
 * psi uses a leading dimension ld_psi (npwx) and each band starts at
 * band * ld_psi.  The result is pooled so every rank sees the same overlaps.
 */
void pw_becp_from_projectors(
    const std::complex<double>* tab_atomic,
    const int npw,
    const int ntarget,
    const std::complex<double>* psi,
    const int ld_psi,
    const int nband,
    std::vector<std::vector<std::complex<double>>>& becp);

/**
 * @brief Apply G = ST^{-1/2} to one vector of raw overlaps.
 */
void orthogonalize_pw_overlaps(const PwTargetSpace& space,
                               const std::vector<std::complex<double>>& raw,
                               std::vector<std::complex<double>>& orth);

/**
 * @brief Derivative of the target inverse-square-root overlap.
 *
 * derivative_overlap[a,b] is <d alpha_a|alpha_b> for one atomic displacement
 * direction.  The raw overlap perturbation is dST = D + D^dagger, and the
 * returned matrix is d(ST^{-1/2}) from the Hermitian Fréchet derivative with
 * the target space's rank cutoff.
 */
void build_pw_target_derivative(
    const PwTargetSpace& space,
    const std::vector<std::complex<double>>& derivative_overlap,
    std::vector<std::complex<double>>& dST_inv_root);

/**
 * @brief Rotate both a raw overlap and its atomic-displacement derivative
 * into the orthonormal target basis.
 *
 * raw[b] = <alpha_b|v> and d_raw[b] = <d alpha_b|v>.  With phi = alpha G,
 * the returned derivative is <d phi_a|v> = sum_b G_ab d_raw[b]
 *                                        + sum_b dG_ab raw[b].
 */
void orthogonalize_pw_overlaps_derivative(
    const PwTargetSpace& space,
    const std::vector<std::complex<double>>& dST_inv_root,
    const std::vector<std::complex<double>>& raw,
    const std::vector<std::complex<double>>& d_raw,
    std::vector<std::complex<double>>& orth,
    std::vector<std::complex<double>>& d_orth);

struct PwTargetMoments
{
    std::vector<double> n_T_target;
    std::vector<double> m_T_target;
    std::vector<double> n_T_constraint;
    std::vector<double> m_T_constraint;
    // Sum over every target (not necessarily the full Hilbert space).  These
    // equal the total charge/spin when the target set is complete.
    double n_T_total = 0.0;
    double m_T_total = 0.0;
};

/**
 * @brief Collinear PW target read-out from becp for both spin channels.
 *
 * becp_up/dn are [band][target] arrays of <alpha_a|psi_band>.  Occupations
 * are in electrons; the magnetization uses the same mu_B normalization as the
 * LCAO backend (spin-up minus spin-down).
 */
PwTargetMoments observe_pw_target_moments(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& becp_up,
    const std::vector<double>& occ_up,
    const std::vector<std::vector<std::complex<double>>>& becp_dn,
    const std::vector<double>& occ_dn);

/**
 * @brief Pre-orthogonalized target-space data for repeated band applications.
 *
 * phi_orth = phi G, vphi_orth = V phi G, m_orth = G^T m_raw G.  These objects
 * depend on the target space and the local potential, but not on the band, so
 * the PW operator can prepare them once per (ik, spin, lambda) and reuse them
 * for every band and every Hamiltonian application in one SCF frame.
 */
struct PwAllocOrthogonalContext
{
    std::vector<std::vector<std::complex<double>>> phi_orth;
    std::vector<std::vector<std::complex<double>>> vphi_orth;
    std::vector<std::complex<double>> m_orth;
};

/**
 * @brief Form the orthonormal target vectors phi_orth = phi G.
 *
 * This is the band-independent half needed by the frozen-target force, where
 * only Q_c = I - sum_a |phi_orth_a><phi_orth_a| is required.  Keeping it
 * separate avoids constructing V phi G and the R x R potential matrix when
 * the caller only needs the complement projector.
 */
void build_pw_alloc_orthogonal_phi(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    std::vector<std::vector<std::complex<double>>>& phi_orth);

/**
 * @brief Build the band-independent orthonormal target products.
 *
 * The inputs use the raw-projector basis described by apply_pw_alloc_delta_h.
 * This routine performs the R x R algebra once; apply_pw_alloc_delta_h_orthogonal
 * then only evaluates the per-band overlaps and vector combinations.
 */
void build_pw_alloc_orthogonal_context(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& m_raw,
    PwAllocOrthogonalContext& context);

/**
 * @brief Apply the design eq. (5.2) allocation correction to one PW band.
 *
 * Inputs are in the raw-projector basis:
 *   raw_overlap[a] = <alpha_a|psi>
 *   vpsi           = V|psi>
 *   vphi[a]        = V|alpha_a>
 *   m_raw[a,b]     = <alpha_a|V|alpha_b>
 *
 * phi[a] is the raw target coefficient vector (plane-wave coefficients).
 * The routine forms phi' = phi G, o' = G o, Vphi' = Vphi G,
 * m' = G^T m_raw G and returns
 *
 *   |delta> = V|psi> - P V|psi> - V P|psi> + P V P|psi>
 *             + sum_i lambda_i Q_i|psi>.
 *
 * This is the PW analogue of build_alloc_delta_h(); no full-basis dense
 * matrix is formed.
 */
void apply_pw_alloc_delta_h(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    const std::vector<std::complex<double>>& psi,
    const std::vector<std::complex<double>>& raw_overlap,
    const std::vector<std::complex<double>>& vpsi,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& m_raw,
    const std::vector<double>& lambda,
    std::vector<std::complex<double>>& delta);

/**
 * @brief Per-band stage of apply_pw_alloc_delta_h with prepared target data.
 *
 * vphi remains the raw V|alpha> vector because <alpha|V|psi> is evaluated for
 * each band; the P, V P and P V P terms use context.phi_orth, context.vphi_orth
 * and context.m_orth.  All pool reductions are batched into one collective per
 * overlap vector instead of one collective per target.
 */
void apply_pw_alloc_delta_h_orthogonal(
    const PwTargetSpace& space,
    const PwAllocOrthogonalContext& context,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& psi,
    const std::vector<std::complex<double>>& raw_overlap,
    const std::vector<std::complex<double>>& vpsi,
    const std::vector<double>& lambda,
    std::vector<std::complex<double>>& delta);

} // namespace alloc

#endif
