#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include<streambuf>
#ifdef __MPI
#include "mpi.h"
#endif

/************************************************
 *  unit test of Atom_pseudo
 ***********************************************/

/**
 * - Tested Functions:
 *   - Atom()
 *     - constructor of class Atom
 *   - ~Atom()
 *     - deconstructor of class Atom
 *   - print_Atom()
 *     - print atomic info to file
 *   - set_index()
 *     - index between numerical atomic orbtials and quantum numbers L,m
 *   - bcast_atom()
 *     - bcast basic atomic info to all processes
 *   - bcast_atom2()
 *     - bcast norm-conserving pseudopotential info to all processes
 */

#define private public
#include "module_cell/read_pp.h"
#include "module_cell/pseudo_nc.h"
#include "module_cell/atom_pseudo.h"
#include "module_cell/atom_spec.h"

class AtomSpecTest : public testing::Test
{
protected:
	Atom atom;
	Pseudopot_upf upf;
	std::ofstream ofs;
	std::ifstream ifs;
};

TEST_F(AtomSpecTest, PrintAtom)
{
	ofs.open("tmp_atom_info");
	atom.label = "C";
	atom.type = 1;
	atom.na = 2;
	atom.nwl = 2;
	atom.Rcut = 1.1;
	atom.nw = 14;
	atom.stapos_wf = 0;
	atom.mass = 12.0;
	delete[] atom.tau;
	atom.tau = new ModuleBase::Vector3<double>[atom.na];
	atom.tau[0].x = 0.2;
	atom.tau[0].y = 0.2;
	atom.tau[0].z = 0.2;
	atom.tau[1].x = 0.4;
	atom.tau[1].y = 0.4;
	atom.tau[1].z = 0.4;
	atom.print_Atom(ofs);
	ofs.close();
	ifs.open("tmp_atom_info");
	std::string str((std::istreambuf_iterator<char>(ifs)),std::istreambuf_iterator<char>());
    	EXPECT_THAT(str, testing::HasSubstr("label = C"));
    	EXPECT_THAT(str, testing::HasSubstr("type = 1"));
    	EXPECT_THAT(str, testing::HasSubstr("na = 2"));
    	EXPECT_THAT(str, testing::HasSubstr("nwl = 2"));
    	EXPECT_THAT(str, testing::HasSubstr("Rcut = 1.1"));
    	EXPECT_THAT(str, testing::HasSubstr("nw = 14"));
    	EXPECT_THAT(str, testing::HasSubstr("stapos_wf = 0"));
    	EXPECT_THAT(str, testing::HasSubstr("mass = 12"));
    	EXPECT_THAT(str, testing::HasSubstr("atom_position(cartesian) Dimension = 2"));
	ifs.close();
	remove("tmp_atom_info");

}

TEST_F(AtomSpecTest, SetIndex)
{
	ifs.open("./support/C.upf");
	GlobalV::PSEUDORCUT = 15.0;
	upf.read_pseudo_upf201(ifs);
	atom.ncpp.set_pseudo_nc(upf);
	ifs.close();
	EXPECT_TRUE(atom.ncpp.has_so);
	atom.nw = 0;
	atom.nwl = 2;
	delete[] atom.l_nchi;
	atom.l_nchi = new int[atom.nwl];
	atom.l_nchi[0] = 2;
	atom.nw += atom.l_nchi[0];
	atom.l_nchi[1] = 4;
	atom.nw += 3*atom.l_nchi[1];
	atom.set_index();
	EXPECT_EQ(atom.iw2l[13],1);
	EXPECT_EQ(atom.iw2n[13],3);
	EXPECT_EQ(atom.iw2m[13],2);
	EXPECT_EQ(atom.iw2_ylm[13],3);
	EXPECT_TRUE(atom.iw2_new[11]);
}

#ifdef __MPI
TEST_F(AtomSpecTest, BcastAtom)
{
	GlobalV::test_atom = 1;
	if(GlobalV::MY_RANK==0)
	{
		atom.label = "C";
		atom.type = 1;
		atom.na = 2;
		atom.nwl = 2;
		atom.Rcut = 1.1;
		atom.nw = 14;
		atom.stapos_wf = 0;
		atom.mass = 12.0;
		delete[] atom.tau;
		atom.tau = new ModuleBase::Vector3<double>[atom.na];
		atom.tau[0].x = 0.2;
		atom.tau[0].y = 0.2;
		atom.tau[0].z = 0.2;
		atom.tau[1].x = 0.4;
		atom.tau[1].y = 0.4;
		atom.tau[1].z = 0.4;
	}
	atom.bcast_atom();
	if(GlobalV::MY_RANK!=0)
	{
		EXPECT_EQ(atom.label,"C");
		EXPECT_EQ(atom.type,1);
		EXPECT_EQ(atom.na,2);
		EXPECT_EQ(atom.nwl,2);
		EXPECT_DOUBLE_EQ(atom.Rcut,1.1);
		EXPECT_EQ(atom.nw,14);
		EXPECT_EQ(atom.stapos_wf,0);
		EXPECT_DOUBLE_EQ(atom.mass,12.0);
		EXPECT_DOUBLE_EQ(atom.tau[0].x,0.2);
		EXPECT_DOUBLE_EQ(atom.tau[1].z,0.4);
	}
}

TEST_F(AtomSpecTest, BcastAtom2)
{
	if(GlobalV::MY_RANK==0)
	{
		ifs.open("./support/C.upf");
		GlobalV::PSEUDORCUT = 15.0;
		upf.read_pseudo_upf201(ifs);
		atom.ncpp.set_pseudo_nc(upf);
		ifs.close();
		EXPECT_TRUE(atom.ncpp.has_so);
	}
	atom.bcast_atom2();
	if(GlobalV::MY_RANK!=0)
	{
		EXPECT_EQ(atom.ncpp.nbeta,6);
		EXPECT_EQ(atom.ncpp.nchi,3);
		EXPECT_DOUBLE_EQ(atom.ncpp.rho_atc[0],8.7234550809E-01);
	}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	testing::InitGoogleTest(&argc, argv);

	MPI_Comm_size(MPI_COMM_WORLD,&GlobalV::NPROC);
	MPI_Comm_rank(MPI_COMM_WORLD,&GlobalV::MY_RANK);
	int result = RUN_ALL_TESTS();
	
	MPI_Finalize();
	
	return result;
}
#endif
#undef private