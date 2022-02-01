#ifndef DEEPKS_PARAMETERS_H
#define DEEPKS_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Deepks_parameters
{
public:
    Deepks_parameters()
    {
        out_descriptor = 0;
        deepks_lmax = 2;
        deepks_mixin = 0;
        deepks_file = "model.ptg";
    };
    ~Deepks_parameters(){};

	//==========================================================
	// DeepKS -- added by caoyu and mohan
	//==========================================================
	/// (need libnpy) output descritpor for deepks.
	bool out_descriptor;
	/// lmax used in descriptor
	int deepks_lmax;
	/// (need libnpy and libtorch) if set 1, a trained model would be needed to cal V_delta and F_delta
	bool deepks_mixin;
	/// needed when deepks_scf=1
	std::string deepks_file; 
};
}

#endif