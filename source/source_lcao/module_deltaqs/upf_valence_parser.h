/**
 * @file upf_valence_parser.h
 * @brief Determine Complete Single-Zeta (CSZ) projection basis for DeltaQS.
 *
 * @par Purpose
 * Determines how many zeta orbitals per angular momentum channel (l) are needed
 * for the Complete Single-Zeta projection in DeltaQS charge constraints.
 *
 * @par Algorithm
 * For each element:
 *   1. Read zv (total valence electrons) from pseudopotential header
 *   2. Read orbital file to get number of zetas per l channel
 *   3. Use chemical knowledge to determine valence configuration
 *   4. Compute n_zeta^l = ceil(n_e^l / (2l+1)) for each l
 *
 * @par Why not use pp.oc?
 * ABACUS filters relativistic j=l+0.5 channels in read_pp.cpp and averages
 * wavefunctions, but does NOT update occupations. This causes pp.oc to be
 * incorrect after filtering.
 *
 * @par Example (Fe, Z_val=16)
 * Valence configuration: 3s²3p⁶3d⁶4s²
 *   l=0 (s): 2+2=4 electrons, orbital has 4 zetas -> use 4
 *   l=1 (p): 6 electrons, orbital has 2 zetas -> use 2
 *   l=2 (d): 6 electrons, orbital has 2 zetas -> use 2
 * CSZ uses all available zetas from orbital file.
 */
#ifndef UPF_VALENCE_PARSER_H
#define UPF_VALENCE_PARSER_H

#include <vector>
#include <map>
#include <string>
#include "source_cell/unitcell.h"

namespace deltaqs {

/**
 * @brief Valence electron configuration for a single element.
 */
struct ValenceConfig {
    double zv_total = 0.0; ///< Total valence electrons from pseudopotential
    std::map<int, int> orbital_zetas_per_l; ///< l -> number of zetas in orbital file
    std::map<int, int> csz_per_l; ///< l -> number of zetas to use for CSZ projection
    int l_max = 0; ///< Maximum angular momentum
    int total_csz_orbitals = 0; ///< Total CSZ projection orbitals
};

/**
 * @brief Determine CSZ projection basis from pseudopotential and orbital file.
 *
 * @param ucell UnitCell containing pseudopotential and orbital information
 * @return Vector of ValenceConfig, one per element type
 */
std::vector<ValenceConfig> determine_csz_basis(const UnitCell& ucell);

/**
 * @brief Validate that orbital file has sufficient zetas.
 *
 * @param ucell UnitCell
 * @param configs Vector of ValenceConfig from determine_csz_basis()
 * @return true if valid, false otherwise
 */
bool validate_csz_orbitals(const UnitCell& ucell, const std::vector<ValenceConfig>& configs);

/**
 * @brief Print CSZ configuration for debugging.
 */
void print_csz_configs(const std::vector<ValenceConfig>& configs, const UnitCell& ucell);

} // namespace deltaqs

#endif // UPF_VALENCE_PARSER_H
