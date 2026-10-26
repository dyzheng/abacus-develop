#ifndef ALLOC_PW_H
#define ALLOC_PW_H

#include <complex>
#include <vector>

namespace alloc
{

/**
 * @brief Orthonormalized target-projector space for the PW backend.
 *
 * The raw projectors alpha_a have an R x R overlap ST_ab = <alpha_a|alpha_b>.
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
    std::vector<double> ST;      // ntarget x ntarget, row-major
    std::vector<double> ST_inv_root; // ntarget x ntarget, row-major
    int rank = 0;
    double cond = 1.0;
};

/**
 * @brief Build the PW target space from projector-projector overlaps.
 *
 * ST is symmetrized (projector overlaps are real symmetric) and truncated by
 * the same rank threshold as the LCAO backend.  target_constraint may contain
 * -1 for a target that belongs to the global P but not to a constrained Q_i.
 */
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
                                std::vector<double>& ST);

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

} // namespace alloc

#endif
