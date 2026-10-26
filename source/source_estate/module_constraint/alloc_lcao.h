#ifndef ALLOC_LCAO_H
#define ALLOC_LCAO_H

#include <vector>

namespace alloc
{

/**
 * @brief Orthonormalized target orbital space in an LCAO basis.
 *
 * Cprime is the row-major norb x ntarget matrix C S_T^{-1/2}, where
 * S_T = C^T S C.  The caller supplies C (the selected target columns in
 * the AO basis) for build_lcao_target(), or the selection indices for
 * finalize_lcao_target().  Keeping this core independent of HContainer makes
 * the S2.1 dense DMR/state-sum audit reusable by the production adapter.
 */
struct LcaoTargetSpace
{
    int norb = 0;
    int ntarget = 0;
    std::vector<int> orbital; // target AO indices (selection form)
    std::vector<int> atom;    // per-target global atom label
    std::vector<double> ST;   // ntarget x ntarget, row-major
    std::vector<double> ST_inv_root; // ntarget x ntarget, row-major
    std::vector<double> Cprime;      // norb x ntarget, row-major
    double cond = 1.0;
    int rank = 0;
};

/**
 * @brief Build the target space from an explicit coefficient matrix C.
 *
 * C is norb x ntarget row-major.  The target overlap S_T = C^T S C is
 * orthonormalized with the same rank-truncated symmetric inverse square root
 * used by the grid backend.  Every target must carry an atom label.
 */
void build_lcao_target(const std::vector<double>& S,
                       const int norb,
                       const std::vector<double>& C,
                       const int ntarget,
                       const std::vector<int>& target_atom,
                       LcaoTargetSpace& out,
                       const double rank_tol = 1e-10);

/**
 * @brief Build the target space from selected AO columns and a target overlap.
 *
 * This is the production/container entry: the caller has already accumulated
 * S_T from the local HContainer blocks and passes the selected AO indices.
 * Cprime is then the selection matrix times S_T^{-1/2}.
 */
void finalize_lcao_target(const int norb,
                          const std::vector<int>& target_orbital,
                          const std::vector<int>& target_atom,
                          const std::vector<double>& ST,
                          LcaoTargetSpace& out,
                          const double rank_tol = 1e-10);

/**
 * @brief Read target occupations from dense total and spin-difference DMRs.
 *
 * For every target a, u_a = S Cprime_a and n_T[a] = u_a^T D u_a.  The spin
 * difference follows the same contraction with D_diff; pass an empty D_diff
 * for a non-spin-polarized run.  This is the direct dense version of the
 * LCAO read-out, used by the S2.1 state-sum cross-check.
 */
void observe_lcao_target(const LcaoTargetSpace& target,
                         const std::vector<double>& S,
                         const std::vector<double>& D_total,
                         const std::vector<double>& D_diff,
                         std::vector<double>& n_T_target,
                         std::vector<double>& m_T_target);

} // namespace alloc

#endif
