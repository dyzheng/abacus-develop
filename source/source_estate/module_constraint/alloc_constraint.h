#ifndef ALLOC_CONSTRAINT_H
#define ALLOC_CONSTRAINT_H

#include <vector>

#include "source_estate/module_constraint/alloc_lcao.h"

namespace alloc
{

/**
 * @brief Dense algebra of the LCAO unified-allocation constraint operator.
 *
 * The operator is linear in the multiplier for a fixed geometry.  For one
 * constrained component i with unit multiplier,
 *
 *   K_i = W_i - P S^{-1} W_i - W_i S^{-1} P
 *           + P S^{-1} W_i S^{-1} P + Q_i,
 *
 * where P is the projector onto all selected target orbitals and
 * Q_i = sum_{a in i}|phi_a><phi_a|.  The implementation never forms S^{-1}:
 * with B = C' (the orthonormalized target coefficients), P S^{-1} = S B B^T
 * and the three metric factors collapse to dense matrix products.
 */
struct AllocConstraintAlgebra
{
    int norb = 0;
    int ntarget = 0;
    int nconstraint = 0;
    LcaoTargetSpace target;
    std::vector<int> target_constraint; // target orbital -> constraint index
    std::vector<double> SB;             // norb x ntarget, S*B
    double rank_tol = 1e-10;            // target-overlap truncation threshold
};

/**
 * @brief Build the shared target space and S*B contraction.
 *
 * `target_atom` labels each target orbital; `target_constraint[a]` is the
 * constraint that owns that target (or -1 for an unconstrained target that
 * still belongs to the global projector P).
 */
void build_alloc_constraint_algebra(
    const std::vector<double>& S,
    const int norb,
    const std::vector<int>& target_orbital,
    const std::vector<int>& target_atom,
    const std::vector<int>& target_constraint,
    const std::vector<double>& ST,
    const int nconstraint,
    AllocConstraintAlgebra& out,
    const double rank_tol = 1e-10);

/**
 * @brief Build one dense unit-multiplier kernel K_i.
 *
 * W must be symmetric and have size norb x norb.  K is returned symmetric
 * up to the floating-point round-off of the matrix products.
 */
void build_alloc_constraint_kernel(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& W,
    const int constraint,
    std::vector<double>& K);

/**
 * @brief Adjoint of the fixed-weight geometry derivative Tr(D dK_i).
 *
 * Returns a symmetric matrix E_i such that, for every symmetric variation
 * dS of the overlap matrix at fixed W_i,
 *
 *   Tr(D dK_i) = sum_{mu,nu} E_i[mu,nu] dS[mu,nu].
 *
 * Only the explicit S dependence of K_i is differentiated here; the weight
 * derivative dW_i is handled by the real-space force kernel.  The routine
 * uses reverse-mode matrix calculus and an R x R Frechet derivative of
 * S_T^{-1/2}, so its cost stays at the same dense-contraction order as one
 * K_i build.
 */
void build_alloc_constraint_s_adjoint(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<double>& W,
    const int constraint,
    const std::vector<double>& D,
    std::vector<double>& E);

/**
 * @brief Sum lambda_i K_i into a dense Hamiltonian correction.
 */
void build_alloc_delta_h(const AllocConstraintAlgebra& space,
                         const std::vector<std::vector<double>>& W,
                         const std::vector<double>& lambda,
                         std::vector<double>& delta_h);

/**
 * @brief Project target occupations and aggregate them by constraint.
 *
 * `n_T_target`/`m_T_target` are per target orbital; the returned
 * `n_T_constraint`/`m_T_constraint` are one value per constraint.  Targets
 * with target_constraint < 0 do not contribute to either constraint sum.
 */
void observe_alloc_constraint_moments(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<double>& D_total,
    const std::vector<double>& D_diff,
    std::vector<double>& n_T_target,
    std::vector<double>& m_T_target,
    std::vector<double>& n_T_constraint,
    std::vector<double>& m_T_constraint);

} // namespace alloc

#endif
