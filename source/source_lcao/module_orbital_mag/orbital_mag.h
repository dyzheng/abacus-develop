#ifndef ORBITAL_MAG_H
#define ORBITAL_MAG_H

#include "source_base/vector3.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_estate/elecstate.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_psi/psi.h"

namespace hamilt {

/**
 * @brief Calculate orbital magnetic moment using Modern Theory of Orbital Magnetization
 *
 * This class implements the Sum-Over-States (SOS) formula for orbital magnetization
 * to avoid numerical instability from direct k-derivative calculations.
 *
 * Reference: Ceresoli et al., PRB 74, 024408 (2006)
 */
class OrbitalMag {
public:
    /**
     * @brief Constructor
     *
     * @param ucell_in Unit cell structure
     * @param kv_in K-point vectors and weights
     * @param pelec_in Electronic state (eigenvalues, weights, Fermi energy)
     * @param hR_in Hamiltonian in real space H(R)
     * @param sR_in Overlap matrix in real space S(R)
     * @param psi_in Wavefunctions (eigenvectors)
     * @param pv_in Parallel orbital distribution info (nullptr for serial)
     */
    OrbitalMag(const UnitCell& ucell_in,
               const K_Vectors& kv_in,
               const elecstate::ElecState& pelec_in,
               const HContainer<double>* hR_in,
               const HContainer<double>* sR_in,
               const psi::Psi<std::complex<double>>* psi_in,
               const Parallel_Orbitals* pv_in = nullptr);

    ~OrbitalMag();

    /**
     * @brief Calculate orbital magnetic moment vector
     *
     * @return ModuleBase::Vector3<double> Orbital magnetic moment (M_x, M_y, M_z)
     */
    ModuleBase::Vector3<double> calculate_orbital_moment();

private:
    /**
     * @brief Compute k-derivative of H(R) or S(R): dH/dk_α or dS/dk_α
     *
     * Uses analytical formula: dH/dk_α = i * Σ_R H(R) * R_α * exp(i*k*R)
     *
     * @param ik K-point index
     * @param alpha Cartesian direction (0=x, 1=y, 2=z)
     * @param hR Input H(R) or S(R) container
     * @param dk_matrix Output dH/dk or dS/dk matrix (nlocal × nlocal)
     */
    void compute_k_derivative(const int ik,
                              const int alpha,
                              const HContainer<double>* hR,
                              std::complex<double>* dk_matrix);

    /**
     * @brief Transform velocity operator to eigenstate basis
     *
     * Computes: V_nm = C_n† * [dH/dk - E_avg * dS/dk] * C_m
     * where E_avg = (E_n + E_m)/2 for numerical stability
     *
     * @param ik K-point index
     * @param dH_dk Derivative dH/dk_α in AO basis
     * @param dS_dk Derivative dS/dk_α in AO basis
     * @param eigenvalues Band energies at this k-point
     * @param eigenvectors Wavefunctions at this k-point (nlocal × nbands)
     * @param V_nm Output velocity matrix in band basis (nbands × nbands)
     */
    void compute_velocity_matrix(const int ik,
                                 const std::complex<double>* dH_dk,
                                 const std::complex<double>* dS_dk,
                                 const double* eigenvalues,
                                 const std::complex<double>* eigenvectors,
                                 std::complex<double>* V_nm);

    /**
     * @brief Compute sum-over-states contribution to M_z from one k-point
     *
     * Formula: M_z_k = (1/2) * Im * Σ_{n∈occ} Σ_{m≠n}
     *          [<n|v_x|m><m|v_y|n>] / (E_n - E_m)² * (E_m + E_n - 2μ)
     *
     * @param ik K-point index
     * @param V_x Velocity matrix in x-direction (nbands × nbands)
     * @param V_y Velocity matrix in y-direction (nbands × nbands)
     * @param eigenvalues Band energies at this k-point
     * @param mu Fermi energy
     * @return double Contribution to M_z from this k-point
     */
    double compute_sos_contribution(const int ik,
                                    const std::complex<double>* V_x,
                                    const std::complex<double>* V_y,
                                    const double* eigenvalues,
                                    const double mu);

    const UnitCell& ucell;
    const K_Vectors& kv;
    const elecstate::ElecState& pelec;
    const HContainer<double>* hR;
    const HContainer<double>* sR;
    const psi::Psi<std::complex<double>>* psi;
    const Parallel_Orbitals* pv;  ///< Parallel orbital distribution (nullptr for serial)

    int nbands;  ///< Number of bands
    int nlocal;  ///< Number of global orbitals (basis size)

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

#endif
