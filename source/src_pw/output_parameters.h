#ifndef OUTPUT_PARAMETERS_H
#define OUTPUT_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Output_parameters
{
public:
    Output_parameters()
    {
        out_dm = 0;
        out_band = 0;
        out_stru = 0;
        out_level = "ie";
        out_hamiltonian = 0;
        out_overlap = 0;
        out_element_info = 0;
        out_r_matrix = 0;
        out_mul = 0;
        out_chg = 0;
        out_pot = 0;
        out_wfc_pw = 0;
        out_wfc_lcao = 0;
        out_dos = 0;
        dos_emax = 15;
        dos_emin = -15;
        dos_edelta = 0.01;
        dos_scale = 0.01;
        dos_sigma = 0.07;
    };
    ~Output_parameters(){};

    /// >0 output density matrix
    bool out_dm;
    /// output energy and band structure
    bool out_band;
    /// output the structure files after each ion step
    bool out_stru;
    /// ie(for electrons); i(for ions); m(minimal level outputs)
    std::string out_level;
    /// output hamiltonian matrix, 1 for H(k), 2 for H(R)
    int out_hamiltonian;
    /// output overlap matrix, 1 for S(k), 2 for S(R)
    int out_overlap;
    /// output detail element info
    bool out_element_info;
    /// output r(R) matrix
    bool out_r_matrix;
    /// mulliken charge or not
    bool out_mul;
    /// >0 output charge density for selected electron steps
    bool out_chg;
    /// output realspace potential
    bool out_pot;
    /// output PW wave functions
    bool out_wfc_pw;
    /// output LCAO wave functions
    bool out_wfc_lcao;
    /// output energy and dos
    bool out_dos;
    /// maximal range for dos
    double dos_emax;
    /// minimal range for dos
    double dos_emin;
    /// delta energy for dos
    double dos_edelta;
    /// scale dos range by
    double dos_scale;
    /// gauss b coefficeinet(default=0.07)
    double dos_sigma;


};
}

#endif