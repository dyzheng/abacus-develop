#ifndef LCAO_PARAMETERS_H
#define LCAO_PARAMETERS_H

#include <string>
namespace ABACUS
{
class LCAO_parameters
{
public:
    LCAO_parameters()
    {
        nb2d = 1;
        lmaxmax = 2;
        lcao_dk = 0.01;
        lcao_dr = 0.01;
        lcao_rmax = 30;
        gamma_only = 0;
        bx = 2;
        by = 2;
        bz = 2;
        lcao_ecut = 0.0;
    };
    ~LCAO_parameters(){};

    /// 2d distribution of atoms
    int nb2d;
    /// maximum of l channels used
    int lmaxmax;
    /// energy cutoff for LCAO
    double lcao_ecut;
    /// delta k for 1D integration in LCAO
    double lcao_dk;
    /// delta r for 1D integration in LCAO
    double lcao_dr;
    /// max R for 1D two-center integration table
    double lcao_rmax;
    /// gamma only, only used in LCAO basis
    bool gamma_only;
    ///
    /// big mesh ball
    ///
    int bx;
    int by;
    int bz;

};
}

#endif