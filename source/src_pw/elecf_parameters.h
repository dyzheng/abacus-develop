#ifndef ELECF_PARAMETERS_H
#define ELECF_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Elecf_parameters
{
public:
    Elecf_parameters()
    {
        elec_field = 0;
        elec_dir = 1;
        elec_maxpos = 0.5;
        elec_opreg = 0.1;
        elec_amp = 0.001;
    };
    ~Elecf_parameters(){};

    /// add electric field
    int elec_field;
    /// directory for electric field
    int elec_dir;
    /// maximal position of efield [0,1)
    double elec_maxpos;
    /// where sawlike potential decrease
    double elec_opreg;
    /// amplitute of the efield, unit is a.u.
    double elec_amp;

};
}

#endif