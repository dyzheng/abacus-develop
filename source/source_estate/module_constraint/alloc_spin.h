#ifndef ALLOC_SPIN_H
#define ALLOC_SPIN_H

#include <complex>
#include <vector>

#include "source_base/vector3.h"
#include "source_estate/module_constraint/alloc_constraint.h"

namespace alloc
{

/**
 * @brief Non-collinear (spin-vector) extension of the allocation constraint.
 *
 * A spin allocation multiplier is a three-vector lambda_i.  The Hamiltonian
 * correction is
 *
 *   Delta H = sum_i A_i tensor (lambda_i . sigma),
 *
 * and the matching observable is m_i = Tr[rho (A_i tensor sigma)].  The
 * routines here implement the spin algebra in the same dense scalar-AO basis
 * as alloc_constraint.cpp.  They do not modify the collinear scalar path.
 */

/**
 * @brief Expand one scalar allocation kernel into a 2 x 2 spin block.
 *
 * The returned H is row-major in the interleaved spinor basis
 * (orbital mu, spin s) -> index 2*mu+s, and equals K tensor (lambda . sigma).
 */
void build_alloc_spin_kernel(const std::vector<double>& K,
                             const int norb,
                             const ModuleBase::Vector3<double>& lambda,
                             std::vector<std::complex<double>>& H);

/**
 * @brief Charge and spin-vector allocation observables per constraint.
 *
 * rho0/rhox/rhoy/rhoz are dense norb x norb Pauli density matrices in the
 * ABACUS convention.  The spin moments use the dimensionless normalization
 * of the existing allocation path (m = Tr[rho sigma A], i.e. mu_B units).
 */
void observe_alloc_spin_moments(
    const AllocConstraintAlgebra& space,
    const std::vector<std::vector<double>>& W,
    const std::vector<double>& rho0,
    const std::vector<double>& rhox,
    const std::vector<double>& rhoy,
    const std::vector<double>& rhoz,
    std::vector<double>& charges,
    std::vector<ModuleBase::Vector3<double>>& moments);

/**
 * @brief Direct spinor trace of one scalar allocation kernel.
 *
 * D is the full row-major 2*norb x 2*norb spinor density matrix.  This is the
 * reference implementation used to cross-check the Pauli-component read-out.
 */
void observe_alloc_spinor_moments(const std::vector<double>& K,
                                  const int norb,
                                  const std::vector<std::complex<double>>& D,
                                  double& charge,
                                  ModuleBase::Vector3<double>& moment);

/**
 * @brief Dense adjoint of the spin-vector constraint energy.
 *
 * For fixed weight matrices W_i, the derivative of
 *   E_c = sum_i lambda_i . m_i
 * with respect to the overlap is Tr(E dS), with
 *
 *   E = sum_i adjoint(K_i, lambda_i . rho).
 *
 * This is the non-collinear counterpart of build_alloc_constraint_s_adjoint()
 * and is consumed by the same two-center force/stress kernels.
 */
void build_alloc_spin_s_adjoint(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<std::vector<double>>& W,
    const std::vector<ModuleBase::Vector3<double>>& lambda,
    const std::vector<double>& rhox,
    const std::vector<double>& rhoy,
    const std::vector<double>& rhoz,
    std::vector<double>& E);

} // namespace alloc

#endif
