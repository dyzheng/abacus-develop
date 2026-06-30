/**
 * @file deltaqs_projector.h
 * @brief Complete Single-Zeta (CSZ) projector for DeltaQS charge constraints.
 *
 * @par Purpose
 * Builds the pre_hr projection matrices using ALL zeta orbitals per angular
 * momentum channel (up to n_zeta_per_l from ValenceConfig), instead of only
 * the first zeta as in DeltaSpin.
 *
 * @par Comparison with DeltaSpin projector
 * DeltaSpin cal_pre_HR():
 *   - Only first zeta per l: nlm_target has (l_max+1)^2 entries
 *   - Index: l^2 + l + m
 *
 * DeltaQS cal_pre_hr_csz():
 *   - All n_zeta zetas per l: nlm_target has sum_l(n_zeta_l * (2l+1)) entries
 *   - Index: sequential over (zeta, l, m)
 *
 * @par Architecture
 * This class is independent of the DeltaSpin operator. It produces HContainer
 * objects that can be used with the existing cal_moment() function to compute
 * projected charges.
 */
#ifndef DELTAQS_PROJECTOR_H
#define DELTAQS_PROJECTOR_H

#include <vector>
#include <string>
#include <unordered_map>

#include "source_cell/unitcell.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_lcao/module_deltaqs/upf_valence_parser.h"

namespace deltaqs {

/**
 * @brief Adjacent atom data for CSZ projection (extended nlm).
 *
 * Similar to DeltaSpin's BI_AdjacentData but with extended nlm that
 * includes multiple zetas per angular momentum channel.
 */
struct CSZ_AdjacentData {
    int iat_adj;                                          ///< global atom index of adjacent atom
    ModuleBase::Vector3<int> R_index;                     ///< cell index of adjacent atom
    std::unordered_map<int, std::vector<double>> nlm;     ///< iw_global -> <phi_iw|alpha_I_zeta_lm>
};

/**
 * @brief CSZ projector: builds pre_hr matrices with all zetas per l.
 */
class CSZProjector {
public:
    CSZProjector() = default;
    ~CSZProjector();

    /**
     * @brief Initialize and build CSZ projection matrices for all constrained atoms.
     *
     * @param ucell Unit cell with atom positions and orbital info
     * @param configs CSZ basis config per element type (from determine_csz_basis)
     * @param paraV Parallel orbital distribution
     * @param gridD Grid driver for neighbor search
     * @param intor Two-center integrator for <phi|alpha> overlaps
     * @param orb_cutoff Orbital cutoff radii per element type
     * @param hR Reference Hamiltonian container (for atom pair topology)
     * @param constraint_atoms Boolean mask: which atoms are charge-constrained
     */
    void build(const UnitCell& ucell,
               const std::vector<ValenceConfig>& configs,
               const Parallel_Orbitals* paraV,
               const Grid_Driver* gridD,
               const TwoCenterIntegrator* intor,
               const std::vector<double>& orb_cutoff,
               const hamilt::HContainer<double>* hR,
               const std::vector<bool>& constraint_atoms);

    /**
     * @brief Get pre_hr for atom iat.
     * @return Pointer to HContainer<double>, or nullptr if not constrained.
     */
    const hamilt::HContainer<double>* get_pre_hr(int iat) const;

    /**
     * @brief Get total number of projector functions for atom iat.
     */
    int get_nproj(int iat) const;

    /**
     * @brief Get constraint atom list.
     */
    const std::vector<bool>& get_constraint_atom_list() const { return constraint_atom_list_; }

    /**
     * @brief Compute projected charge for all constrained atoms.
     *
     * Uses the CSZ pre_hr matrices and the total charge density matrix (dmR)
     * to compute N_I = Tr(dmR * pre_hr_I) for each constrained atom I.
     *
     * @param dmR Total charge density matrix (rho_up + rho_down)
     * @return Vector of projected charges per atom (0 for unconstrained atoms)
     */
    std::vector<double> cal_charge(const hamilt::HContainer<double>* dmR) const;

    /**
     * @brief Compute projected magnetic moment for all constrained atoms.
     *
     * @param dmR_diff Spin difference density matrix (rho_up - rho_down)
     * @return Vector of projected moments per atom
     */
    std::vector<double> cal_moment(const hamilt::HContainer<double>* dmR_diff) const;

private:
    std::vector<hamilt::HContainer<double>*> pre_hr_;  ///< pre_hr per atom
    std::vector<int> nproj_;                            ///< number of projectors per atom
    std::vector<bool> constraint_atom_list_;            ///< constraint mask
    const UnitCell* ucell_ = nullptr;
    const Parallel_Orbitals* paraV_ = nullptr;

    void cal_hr_ijr(int iat1, int iat2,
                    const std::unordered_map<int, std::vector<double>>& nlm1_all,
                    const std::unordered_map<int, std::vector<double>>& nlm2_all,
                    double* data_pointer);
};

} // namespace deltaqs

#endif // DELTAQS_PROJECTOR_H
