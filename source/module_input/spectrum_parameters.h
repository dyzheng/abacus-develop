#ifndef SPECTRUM_PARAMETERS_H
#define SPECTRUM_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Spectrum_parameters
{
public:
    Spectrum_parameters()
    {
		spectral_type = "None";
		spectral_method = 0;
		kernel_type = "rpa";
		eels_method = 0;
		absorption_method = 0;
		system_type = "bulk";
		eta = 0.05;
		domega = 0.01;
		nomega = 300;
		ecut_chi = 1;
		// oband=1;
		q_start[0] = 0.1;
		q_start[1] = 0.1;
		q_start[2] = 0.1;
		q_direct[0] = 1;
		q_direct[1] = 0;
		q_direct[2] = 0;
		// start_q=1;
		// interval_q=1;
		nq = 1;
		out_epsilon = true;
		out_chi = false;
		out_chi0 = false;
		fermi_level = 0.0;
		coulomb_cutoff = false;

		kmesh_interpolation = false;
		for (int i = 0; i < 100; i++)
		{
			qcar[i][0] = 0.0;
			qcar[i][1] = 0.0;
			qcar[i][2] = 0.0;
		}

		lcao_box[0] = 10;
		lcao_box[1] = 10;
		lcao_box[2] = 10;

		// epsilon0 = false;
		// intersmear = 0.01;
		intrasmear = 0.0;
		shift = 0.0;
		metalcalc = false;
		eps_degauss = 0.01;
	};
    ~Spectrum_parameters(){};

	std::string spectral_type; // the type of the calculated spectrum
	int spectral_method; // 0: tddft(linear response)
	int eels_method; // 0: hilbert_transform method; 1: standard method
	int absorption_method; // 0: vasp's method  1: pwscf's method
	// int		 epsilon_choice;         // 0: hilbert_transform method; 1: standard method
	std::string kernel_type; // the kernel type: rpa, tdlda ...
	std::string system_type; // bulk or surface
	double eta; // unit(Ry)
	double domega; // unit(Ry)
	int nomega;
	int ecut_chi; // the dimension of G
	// int     oband;                 // the number of "occupied" bands
	double q_start[3]; // the position of the first q point in direct coordinate
	double q_direct[3]; // the q direction
	// int     start_q;               // the serial number of the start qpoint
	// int     interval_q;            // the interval of the qpoints
	int nq; // the total number of qpoints for calculation
	bool out_epsilon; // output epsilon or not
	bool out_chi; // output chi or not
	bool out_chi0; // output chi0 or not
	double fermi_level; // the change the fermi level(Ry)
	bool coulomb_cutoff; // turn on or off the Coulomb_cutoff 0/1
	// bool     epsilon0;              // calculate the macroscopic dielectric constant or not
	// double   intersmear;            // eta
	double intrasmear; // Eta
	double shift;
	bool metalcalc; // metal or not
	double eps_degauss; // degauss
	// int	epsilon0_choice;             // 0: vasp's method  1: pwscf's method
	bool kmesh_interpolation; // calculting <i,0|j,R>
	double qcar[100][3]; // the Cartesian position of q points(unit: 2*PI/lat0)
	int lcao_box[3]; // the scale for searching the existence of the overlap <i,0|j,R>
};
}

#endif