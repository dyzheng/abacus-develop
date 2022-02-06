#ifndef TDDFT_PARAMETERS_H
#define TDDFT_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Tddft_parameters
{
public:
    Tddft_parameters()
    {
		tddft = 0;
		td_dr2 = 1e-9;
		td_dt = 0.02;
		td_force_dt = 0.02;
		td_val_elec_01 = 1;
		td_val_elec_02 = 1;
		td_val_elec_03 = 1;
		td_vext = 0;
		td_vext_dire = 1;

		td_timescale = 0.5;
		td_vexttype = 1;
		td_vextout = 0;
		td_dipoleout = 0;
	};
    ~Tddft_parameters(){};

	/// calculate tddft or not
	int tddft;
	/// threshold for electronic iteration of tddft
	double td_dr2;
	/// "fs"
	double td_dt;
	/// "fs"
	double td_force_dt;
	/// valence electron 01
	int td_val_elec_01;
	/// valence electron 02
	int td_val_elec_02;
	/// valence electron 03
	int td_val_elec_03;
	/// add extern potential or not
	int td_vext;
	/// vext direction
	int td_vext_dire;
	/// "fs"
	double td_timescale; 
	int td_vexttype;
	/// output the electronic potential or not
	int td_vextout;
	/// output the dipole or not
	int td_dipoleout; 
};
}

#endif