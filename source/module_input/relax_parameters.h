#ifndef RELAX_PARAMETERS_H
#define RELAX_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Relax_parameters
{
public:
    Relax_parameters()
    {
        press[0] = 0.0;
        press[1] = 0.0;
        press[2] = 0.0;
        cell_factor = 1.2;
        ion_step = 0;
        relax_bfgs_rmax = 0.8;
        relax_bfgs_rmin = 1.0e-5;
        relax_bfgs_ini = 0.5;
        relax_bfgs_w1 = 0.01;
        relax_bfgs_w2 = 0.5;
        relax_fthr = 1.0e-3;
        relax_fthr_ev2 = 0.0;
        relax_sthr = 1.0e-2;
        relax_cell_fix = "None";
        relax_method = "cg";
        move_cg_thr = 0.5;
    };
    ~Relax_parameters(){};
    /// external pressure, for xx, yy, zz direction
    double press[3];
    /// Used in the construction of the pseudopotential tables. 
    /// It should exceed the maximum linear contraction of the cell during a simulation.
    double cell_factor;
    /// number of max ionic iter
    int ion_step;
    /// threshold for ionic force in relax calculation
    double relax_fthr;
    /// invalid force threshold
    double relax_fthr_ev2;
    /// wolfe condition 1
    double relax_bfgs_w1;
    /// wolfe condition 2
    double relax_bfgs_w2;
    /// trust radius max , unit : Bohr
    double relax_bfgs_rmax;
    /// trust radius min
    double relax_bfgs_rmin;
    /// initial move , unit : Bohr
    double relax_bfgs_ini;
    /// threshold for stress tensor in cell-relax calculation
    double relax_sthr;
    /// which axes are fixed
    std::string relax_cell_fix;
    /// methods to move_ion: sd, bfgs, cg
    std::string relax_method;
    /// threshold when cg to bfgs
    double move_cg_thr;

};
}

#endif