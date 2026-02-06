#ifndef ORBITAL_MAG_DM_H
#define ORBITAL_MAG_DM_H

#include "source_base/vector3.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

namespace hamilt {

/**
 * @brief Calculate orbital magnetic moment using density matrix method
 *
 * This class implements the density matrix-based formula for orbital magnetization:
 * M_itin(k) = (1/2) * Im * Tr[ P * (P_x * V_tilde_y - P_y * V_tilde_x) ]
 *
 * Where:
 * - V_tilde_α = V_α - S_α * W (gauge-corrected velocity)
 * - W = P * H (energy density matrix)
 * - P_x, P_y = k-derivatives of density matrix (finite difference)
 * - V_α = ∂H/∂k_α, S_α = ∂S/∂k_α (analytical k-derivatives)
 *
 * This method avoids the need for eigenvectors, making it suitable for rt-TDDFT.
 *
 * @tparam TK Type for k-space matrices (std::complex<double> for multi-k)
 */
template <typename TK>
class OrbitalMagDM {
public:
    /**
     * @brief Constructor
     *
     * @param ucell_in Unit cell structure
     * @param kv_in K-point vectors and weights
     * @param dm_in Density matrix object
     * @param hR_in Hamiltonian in real space H(R)
     * @param sR_in Overlap matrix in real space S(R)
     * @param pv_in Parallel orbital distribution info (nullptr for serial)
     * @param At_in Vector potential A(t) in Cartesian coordinates (default: zero)
     */
    OrbitalMagDM(const UnitCell& ucell_in,
                 const K_Vectors& kv_in,
                 const elecstate::DensityMatrix<TK, double>& dm_in,
                 const HContainer<double>* hR_in,
                 const HContainer<double>* sR_in,
                 const Parallel_Orbitals* pv_in,
                 const LCAO_Orbitals& orb_in,
                 const ModuleBase::Vector3<double>& At_in = {0.0, 0.0, 0.0});

    ~OrbitalMagDM();

    /**
     * @brief Calculate orbital magnetic moment vector
     *
     * @return ModuleBase::Vector3<double> Orbital magnetic moment (M_x, M_y, M_z)
     */
    ModuleBase::Vector3<double> calculate_orbital_moment();

    /**
     * @brief Calculate local (on-site) orbital moment: Tr[P(k) * L_local]
     *
     * Builds the on-site angular momentum matrix L_local using
     * cal_r_overlap_R::get_psi_L_psi() for each atom, then computes
     * sum_k wk * Tr[P(k) * L_local(k)] where L_local(k) is the
     * Fourier transform of the on-site L matrix.
     *
     * @return ModuleBase::Vector3<double> Local orbital moment (M_x, M_y, M_z)
     */
    ModuleBase::Vector3<double> calculate_local_moment();

    /**
     * @brief Set vector potential A(t) for rt-TDDFT calculations
     *
     * @param At Vector potential in Cartesian coordinates
     */
    void set_vector_potential(const ModuleBase::Vector3<double>& At);

private:
    /**
     * @brief Fold real-space matrix to k-space with optional A(t)
     *
     * M(k) = sum_R M(R) * exp(i*(k+A)*R)
     *
     * @param ik K-point index
     * @param mR Real-space matrix container
     * @param mk Output k-space matrix
     */
    void fold_to_k(int ik, const HContainer<double>* mR, TK* mk);

    /**
     * @brief Compute k-derivative of real-space matrix
     *
     * dM/dk_α = i * sum_R R_α * M(R) * exp(i*(k+A)*R)
     *
     * @param ik K-point index
     * @param alpha Cartesian direction (0=x, 1=y, 2=z)
     * @param mR Real-space matrix container
     * @param dm_dk Output k-derivative matrix
     */
    void compute_k_derivative(int ik, int alpha, const HContainer<double>* mR, TK* dm_dk);

    /**
     * @brief Compute density matrix derivative using finite difference
     *
     * dP/dk_α = [P(k+dk_α) - P(k-dk_α)] / (2*dk)
     *
     * @param ik K-point index
     * @param alpha Cartesian direction (0=x, 1=y, 2=z)
     * @param dP_dk Output density matrix derivative
     */
    void compute_dm_derivative(int ik, int alpha, TK* dP_dk);

    /**
     * @brief Find k-point neighbors for finite difference with periodic BZ wrapping
     *
     * @param ik K-point index
     * @param alpha Cartesian direction (0=x, 1=y, 2=z)
     * @return std::pair<int, int> (ik_plus, ik_minus) indices, or (-1, -1) if not found
     */
    std::pair<int, int> find_k_neighbors(int ik, int alpha);

    /**
     * @brief Compute energy density matrix W = P * H
     *
     * @param P Density matrix at k-point
     * @param H Hamiltonian at k-point
     * @param W Output energy density matrix
     */
    void compute_energy_density(const TK* P, const TK* H, TK* W);

    /**
     * @brief Compute gauge-corrected velocity V_tilde = V - S_alpha * W
     *
     * @param V Velocity matrix (dH/dk_alpha)
     * @param S_alpha Overlap derivative (dS/dk_alpha)
     * @param W Energy density matrix
     * @param V_tilde Output corrected velocity matrix
     */
    void compute_corrected_velocity(const TK* V, const TK* S_alpha, const TK* W, TK* V_tilde);

    /**
     * @brief Compute trace term Tr[P * P_x * V_tilde_y]
     *
     * @param P Density matrix
     * @param P_x Density matrix derivative in x-direction
     * @param V_tilde_y Corrected velocity in y-direction
     * @return std::complex<double> Trace value
     */
    std::complex<double> compute_trace_term(const TK* P, const TK* P_x, const TK* V_tilde_y);

    const UnitCell& ucell;
    const K_Vectors& kv;
    const elecstate::DensityMatrix<TK, double>& dm;
    const HContainer<double>* hR;
    const HContainer<double>* sR;
    const Parallel_Orbitals* pv;
    const LCAO_Orbitals& orb;
    ModuleBase::Vector3<double> cart_At;  ///< Vector potential A(t) in Cartesian coords

    int nlocal;   ///< Number of global orbitals (basis size)
    double dk_finite = 0.001;  ///< Finite difference step size in reciprocal lattice units

    /**
     * @brief Check if running in parallel mode
     */
    bool is_parallel() const { return pv != nullptr && !pv->is_serial; }

    /**
     * @brief Get local row size for distributed matrices
     */
    int get_local_nrow() const { return is_parallel() ? pv->nrow : nlocal; }

    /**
     * @brief Get local column size for distributed matrices
     */
    int get_local_ncol() const { return is_parallel() ? pv->ncol : nlocal; }

    /**
     * @brief Get local matrix size (nrow * ncol)
     */
    int64_t get_local_size() const { return is_parallel() ? pv->nloc : static_cast<int64_t>(nlocal) * nlocal; }
};

} // namespace hamilt

#endif // ORBITAL_MAG_DM_H
