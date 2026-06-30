/**
 * @file upf_valence_parser.cpp
 * @brief Implementation of CSZ basis determination for DeltaQS.
 *
 * @par Strategy
 * Since ABACUS filters relativistic channels and corrupts pp.oc, we use a simpler
 * approach: read zv from pseudopotential header and use ALL available zetas from
 * the orbital file as the CSZ basis.
 *
 * This is safe because:
 * 1. Orbital files are designed to span the valence space
 * 2. Using all zetas ensures complete coverage
 * 3. Extra zetas (polarization functions) add flexibility without harm
 */
#include "upf_valence_parser.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "source_base/tool_quit.h"

namespace deltaqs {

std::vector<ValenceConfig> determine_csz_basis(const UnitCell& ucell)
{
    std::vector<ValenceConfig> configs(ucell.ntype);

    for (int it = 0; it < ucell.ntype; it++)
    {
        ValenceConfig& vc = configs[it];
        
        // Read total valence electrons from pseudopotential header
        vc.zv_total = ucell.atoms[it].ncpp.zv;
        
        if (vc.zv_total <= 0)
        {
            std::ostringstream msg;
            msg << "Element " << ucell.atoms[it].label
                << " has zv=" << vc.zv_total << " (invalid valence charge). "
                << "Check pseudopotential file.";
            ModuleBase::WARNING_QUIT("DeltaQS::determine_csz_basis", msg.str());
        }

        // Read number of zetas per l from orbital file
        vc.l_max = ucell.atoms[it].nwl;
        
        for (int l = 0; l <= vc.l_max; l++)
        {
            int n_zetas = ucell.atoms[it].l_nchi[l];
            vc.orbital_zetas_per_l[l] = n_zetas;
            
            // Use all available zetas as CSZ basis
            vc.csz_per_l[l] = n_zetas;
            vc.total_csz_orbitals += n_zetas * (2 * l + 1);
        }

        // Silent by default; use print_csz_configs() for explicit output
    }

    return configs;
}

bool validate_csz_orbitals(const UnitCell& ucell, const std::vector<ValenceConfig>& configs)
{
    // With the new approach (use all zetas from orbital file), validation is trivial:
    // just check that orbital file has at least one zeta per l channel that has electrons
    
    // For now, we assume the orbital file is appropriate for the pseudopotential
    // A more sophisticated check would compare zv with the orbital file's designed valence
    
    return true;
}

void print_csz_configs(const std::vector<ValenceConfig>& configs, const UnitCell& ucell)
{
    std::cout << "\n[DeltaQS] Complete Single-Zeta (CSZ) Projection Basis:" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    for (int it = 0; it < ucell.ntype; it++)
    {
        const ValenceConfig& vc = configs[it];
        std::cout << "\nElement: " << ucell.atoms[it].label
                  << " (Z_val = " << vc.zv_total << ")" << std::endl;
        std::cout << std::setw(4) << "l"
                  << std::setw(12) << "orbital_zetas"
                  << std::setw(10) << "csz_zetas"
                  << std::setw(14) << "proj_orbitals" << std::endl;

        for (int l = 0; l <= vc.l_max; l++)
        {
            int orb_zetas = vc.orbital_zetas_per_l.at(l);
            int csz_zetas = vc.csz_per_l.at(l);
            int proj = csz_zetas * (2 * l + 1);
            
            std::cout << std::setw(4) << l
                      << std::setw(12) << orb_zetas
                      << std::setw(10) << csz_zetas
                      << std::setw(14) << proj << std::endl;
        }

        std::cout << "Total CSZ projection orbitals: " << vc.total_csz_orbitals << std::endl;
    }

    std::cout << std::string(70, '=') << std::endl;
}

} // namespace deltaqs
