#define private public
#include "../hcontainer.h"
#include "gtest/gtest.h"

using namespace hamilt;

//Unit test of HContainer with gtest framework
class HContainerTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        //set up a unitcell, with one element and three atoms, each atom has 2 orbitals
        ucell.ntype = 1;
        ucell.nat = 3;
        ucell.atoms = new Atom[ucell.ntype];
        ucell.iat2it = new int[ucell.nat];
        for(int iat = 0; iat < ucell.nat; iat++)
        {
            ucell.iat2it[iat] = 0;
        }
        ucell.atoms[0].nw = 2;

        //set up a HContainer with ucell
        HR = new HContainer<double>(ucell);
    }

    void TearDown() override
    {
        delete HR;
        delete[] ucell.atoms;
        delete[] ucell.iat2it;
    }

    UnitCell ucell;
    HContainer<double>* HR;
};

//using TEST_F to test HContainer::insert_pair
TEST_F(HContainerTest, insert_pair)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //insert atom_ij again
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //set up another AtomPair
    AtomPair<double> atom_kl(1, 0);
    atom_kl.set_size(2, 2);
    //insert atom_kl into HR
    HR->insert_pair(atom_kl);
    //check if atom_kl is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    EXPECT_EQ(HR->atom_pairs[1].get_atom_i(), 1);
    EXPECT_EQ(HR->atom_pairs[1].get_atom_j(), 0);
    EXPECT_EQ(HR->atom_pairs[1].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[1].get_col_size(), 2);
}

//using TEST_F to test HContainer::find_pair
TEST_F(HContainerTest, find_pair)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //find atom_ij in HR
    AtomPair<double>* atom_ij_ptr = HR->find_pair(0, 1);
    //check if atom_ij is found
    EXPECT_EQ(atom_ij_ptr, &HR->atom_pairs[0]);
    //find atom_kl in HR
    AtomPair<double>* atom_kl_ptr = HR->find_pair(1, 0);
    //check if atom_kl is found
    EXPECT_EQ(atom_kl_ptr, nullptr);
}

//using TEST_F to test HContainer::get_atom_pair, both with atom_i, atom_j and with index
TEST_F(HContainerTest, get_atom_pair)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //get atom_ij in HR
    AtomPair<double>& atom_ij_ref = HR->get_atom_pair(0, 1);
    //check if atom_ij is found
    EXPECT_EQ(&atom_ij_ref, &HR->atom_pairs[0]);
    //get atom_kl in HR
    AtomPair<double>& atom_kl_ref = HR->get_atom_pair(1, 0);
    //check if atom_kl is found
    EXPECT_EQ(&atom_kl_ref, &HR->atom_pairs[0]);
}

//using TEST_F to test HContainer::operator()
TEST_F(HContainerTest, operator)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //get atom_ij in HR
    AtomPair<double>& atom_ij_ref = HR->get_atom_pair(0, 1);
    //check if atom_ij is found
    EXPECT_EQ(&atom_ij_ref, &(HR->atom_pairs[0]));
    //get atom_kl in HR
    AtomPair<double>& atom_kl_ref = HR->get_atom_pair(1, 0);
    //check if atom_kl is found
    EXPECT_EQ(&atom_kl_ref, &HR->atom_pairs[0]);
}

//using TEST_F to test HContainer::fix_R and unfix_R
TEST_F(HContainerTest, fix_R)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //fix R
    HR->fix_R(0, 0, 0);
    //check if R is fixed
    EXPECT_EQ(HR->current_R, 0);
    //fix R again
    HR->fix_R(0, 0, 0);
    //check if R is fixed
    EXPECT_EQ(HR->current_R, 0);
    //fix another R
    HR->fix_R(1, 0, 0);
    //check if R is fixed
    EXPECT_EQ(HR->current_R, 1);
    //unfix R
    HR->unfix_R();
    //check if R is unfixed
    EXPECT_EQ(HR->current_R, -1);
}

//using TEST_F to test HContainer::fix_gamma
TEST_F(HContainerTest, fix_gamma)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //fix gamma
    HR->fix_gamma();
    //check if gamma is fixed
    EXPECT_EQ(HR->gamma_only, true);
    //fix gamma again
    HR->fix_gamma();
    //check if gamma is fixed
    EXPECT_EQ(HR->gamma_only, true);
}

/**
 * using TEST_F to test HContainer::loop_R, 
 * step: 1. size_R_loop(), 2. for-loop, loop_R(), 3. fix_R(), 4. do something
 */
TEST_F(HContainerTest, loop_R)
{
    //1. size_R_loop()
    int size_for_loop_R = HR->size_R_loop();
    EXPECT_EQ(size_for_loop_R, 1);
    //2. for-loop, loop_R()
    int rx, ry, rz;
    for (int i = 0; i < size_for_loop_R; i++)
    {
        HR->loop_R(i, rx, ry, rz);
        EXPECT_EQ(rx, 0);
        EXPECT_EQ(ry, 0);
        EXPECT_EQ(rz, 0);
        //check if R is fixed
        EXPECT_EQ(HR->current_R, i);
        //4. do something


    }
}

//using TEST_F to test HContainer::size_atom_pairs
//1. test with R fixed
//2. test with R unfixed
TEST_F(HContainerTest, size_atom_pairs)
{
    //1. test with R fixed
    //fix R
    HR->fix_R(0, 0, 0);
    //check if R is fixed
    EXPECT_EQ(HR->current_R, 0);
    //get AP size
    int AP_size = HR->size_atom_pairs();
    EXPECT_EQ(AP_size, 0);
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //get AP size
    AP_size = HR->size_atom_pairs();
    EXPECT_EQ(AP_size, 1);
    //2. test with R unfixed
    //unfix R
    HR->unfix_R();
    //check if R is unfixed
    EXPECT_EQ(HR->current_R, -1);
    //get AP size
    AP_size = HR->size_atom_pairs();
    EXPECT_EQ(AP_size, 1);
}

//using TEST_F to test HContainer::data()
TEST_F(HContainerTest, data)
{
    //set up a AtomPair
    AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    //insert atom_ij into HR
    HR->insert_pair(atom_ij);
    //check if atom_ij is inserted into HR
    EXPECT_EQ(HR->atom_pairs.size(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_i(), 0);
    EXPECT_EQ(HR->atom_pairs[0].get_atom_j(), 1);
    EXPECT_EQ(HR->atom_pairs[0].get_row_size(), 2);
    EXPECT_EQ(HR->atom_pairs[0].get_col_size(), 2);
    //get data pointer
    double* data_ptr = HR->data(0, 1);
    //check if data pointer is correct
    EXPECT_EQ(data_ptr, HR->atom_pairs[0].get_pointer());
}

int main(int argc, char **argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD,&GlobalV::MY_RANK);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}