#ifndef SCF_PARAMETERS_H
#define SCF_PARAMETERS_H

#include <string>
namespace ABACUS
{
class SCF_parameters
{
public:
    SCF_parameters()
    {
        ks_solver = "default";
        mixing_type = "pulay";
        mixing_beta = 0.7;
        mixing_ndim = 8;
        mixing_gg0 = 0.0;
        ocp = 0;
        ocp_set = "none";
        scf_thr = 1e-9;
        scf_nmin = 0;
        scf_nmax = 40;
        smearing_method = "fixed";
        smearing_sigma = 0.01;
    };
    ~SCF_parameters(){};
    
    /// choose method to solve KS-equation, PW: cg, dav, LCAO: elpa, scalapack, hpseps
    std::string ks_solver;
    /// choose method for charge density mixing method in scf: 
    /// plain, broyden, pulay, kerker, pulay-kerker
    std::string mixing_type;
    /// parameter for charge density mixing
    double mixing_beta;
    /// used in Broyden method
    int mixing_ndim;
    /// used in kerker method
    double mixing_gg0;
    /// 
    int ocp;
    /// 
    std::string ocp_set;
    /// charge density error
    double scf_thr;
    /// number of min elec iter
    int scf_nmin;
    /// number of max elec iter
    int scf_nmax;
    /// "gaussian",
    /// "mp","methfessel-paxton"
	/// "mv","marzari-vanderbilt","cold"
	/// "fd","fermi-dirac"
    std::string smearing_method;
    /// parameter for smearing method, unit: Ry
    double smearing_sigma;

};
}

#endif