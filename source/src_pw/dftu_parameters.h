#ifndef DFTU_PARAMETERS_H
#define DFTU_PARAMETERS_H

#include <string>
#include <vector>
namespace ABACUS
{
class Dftu_parameters
{
public:
    Dftu_parameters()
    {
		dft_plus_u = false; // 1:DFT+U correction; 0：standard DFT calcullation
		yukawa_potential = false;
        yukawa_lambda = 0.0;
		double_counting = 1;
		omc = false;
		dftu_type = 2;
		dft_plus_dmft = false;
	};
    ~Dftu_parameters(){};

	/// true:DFT+U correction; false：standard DFT calcullation(default)
	bool dft_plus_u; 
	/// which correlated orbitals need corrected ; d:2 ,f:3, do not need correction:-1
	std::vector<int> orbital_corr; 
	/// Hubbard Coulomb interaction parameter U(ev)
	std::vector<double> hubbard_u; 
	/// Hund exchange parameter J(ev)
	std::vector<double> hund_j; 
	/// whether turn on occupation matrix control method or not
	bool omc; 
	/// default:false
	bool yukawa_potential; 
	/// default:0.0
	double yukawa_lambda; 

	/// The two parameters below are not usable currently
	/// 1:rotationally invarient formalism; 2:simplified form(default)
	int dftu_type;
	/// 1:FLL(fully localized limit)(default); 2:AMF(around mean field)
	int double_counting; 

	///==========================================================
	///    DFT+DMFT       Xin Qu added on 2021-08
	///==========================================================
	bool dft_plus_dmft; // true:DFT+U correction; false：standard DFT calcullation(default)
};
}

#endif