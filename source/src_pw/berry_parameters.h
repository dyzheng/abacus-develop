#ifndef Berry_PARAMETERS_H
#define Berry_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Berry_parameters
{
public:
    Berry_parameters()
    {
		berry_phase = false;
		gdir = 3;
		towannier90 = false;
		nnkp = "seedname.nnkp";
		wannier_spin = "up";
	};
    ~Berry_parameters(){};

	/// calculate berry phase or not
	bool berry_phase;
	/// calculate the polarization in the direction of the lattice vector
	int gdir;

	///==========================================================
	/// Wannier functions
	///==========================================================
	/// use wannier90 code interface or not
	bool towannier90;
	/// the wannier90 code nnkp file name
	std::string nnkp;
	/// calculate spin in wannier90 code interface
	std::string wannier_spin; 
};
}

#endif