#ifndef TEST_PARAMETERS_H
#define TEST_PARAMETERS_H

#include <string>
namespace ABACUS
{
class Test_parameters
{
public:
    Test_parameters()
    {
		printe = 100;
		out_alllog = false;
		nurse = 0;
		colour = false;
		t_in_h = 1;
		vl_in_h = 1;
		vnl_in_h = 1;
		vh_in_h = 1;
		vxc_in_h = 1;
		vion_in_h = 1;
		test_force = false;
		test_stress = false;
		new_dm = 1;
		test_just_neighbor = false;
		symmetry_prec = 1.0e-5;
	};
    ~Test_parameters(){};

    ///print detail energy in some scf unconvergent step.
	int printe;
	/// output all logs.
	bool out_alllog;
	/// used for debug.
	int nurse;
	/// used for fun.
	bool colour;
	/// calculate the T or not.
	int t_in_h;
	/// calculate the vloc or not.
	int vl_in_h;
	/// calculate the vnl or not.
	int vnl_in_h;
	/// calculate the hartree potential or not
	int vh_in_h;
	/// calculate the xc potential or not
	int vxc_in_h;
	/// calculate the local ionic potential or not
	int vion_in_h;
	/// test the force.
	bool test_force;
	/// test the stress.
	bool test_stress; 

	///==========================================================
	/// new DM algorithm test
	/// add by Shen Yu @ 2019/5/9
	/// newDM values:
	///  0: not use new DM algorithm;
	///  1: use new DM algorithm and show no debug information
	///  2: use new DM algorithm and only show key debug information
	///  3: use new DM algorithm and show all detail debug information
	///==========================================================
	int new_dm;
	///==========================================================
	/// variables for test only
	///==========================================================
	bool test_just_neighbor;
	/// accuracy for symmetry
	double symmetry_prec; 
};
}

#endif