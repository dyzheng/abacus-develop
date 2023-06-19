#include "gtest/gtest.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"

/**
 * Unit test of HContainer
 * HContainer is a container of hamilt::AtomPair, in this test, we test the following functions:
 * 1. insert_pair
 * 2. find_pair
 * 3. get_atom_pair
 * 4. fix_R and unfix_R
 * 5. fix_gamma
 * 6. loop_R
 * 7. size_atom_pairs
 * 8. data
 *
 */

// Unit test of HContainer with gtest framework
class HContainerTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // set up a unitcell, with one element and three atoms, each atom has 2 orbitals
        ucell.ntype = 1;
        ucell.nat = 3;
        ucell.atoms = new Atom[ucell.ntype];
        ucell.iat2it = new int[ucell.nat];
        for (int iat = 0; iat < ucell.nat; iat++)
        {
            ucell.iat2it[iat] = 0;
        }
        ucell.atoms[0].nw = 2;

        // set up a HContainer with ucell
        HR = new hamilt::HContainer<double>(ucell);
    }

    void TearDown() override
    {
        delete HR;
        delete[] ucell.atoms;
        delete[] ucell.iat2it;
    }

    UnitCell ucell;
    hamilt::HContainer<double>* HR;
};

// using TEST_F to test HContainer::insert_pair
TEST_F(HContainerTest, insert_pair)
{
    // check HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_i(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_j(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_col_size(), 2);
    // set up a hamilt::AtomPair
    hamilt::AtomPair<double> atom_ij(0, 3);
    atom_ij.set_size(2, 2);
    // insert atom_ij into HR
    HR->insert_pair(atom_ij);
    // check if atom_ij is inserted into HR
    EXPECT_EQ(HR->size_atom_pairs(), 10);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_i(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_j(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_col_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_atom_j(), 3);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_col_size(), 2);
    // insert atom_ij again
    HR->insert_pair(atom_ij);
    // check if atom_ij is inserted into HR
    EXPECT_EQ(HR->size_atom_pairs(), 10);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_atom_j(), 3);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0, 3).get_col_size(), 2);
    // set up another hamilt::AtomPair
    hamilt::AtomPair<double> atom_kl(1, 0);
    atom_kl.set_size(2, 2);
    double tmp_array[4] = {1, 2, 3, 4};
    atom_kl.get_HR_values(1, 0, 0).add_array(&tmp_array[0]);
    // insert atom_kl into HR
    HR->insert_pair(atom_kl);
    // check if atom_kl is inserted into HR
    EXPECT_EQ(HR->size_atom_pairs(), 10);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_i(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_atom_j(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(2, 2).get_col_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(1, 0).get_atom_i(), 1);
    EXPECT_EQ(HR->get_atom_pair(1, 0).get_atom_j(), 0);
    EXPECT_EQ(HR->get_atom_pair(1, 0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(1, 0).get_col_size(), 2);
    // check if data is correct
    double* data_ptr = HR->get_atom_pair(1, 0).get_HR_values(1, 0, 0).get_pointer();
    EXPECT_EQ(data_ptr[0], 1);
    EXPECT_EQ(data_ptr[1], 2);
    EXPECT_EQ(data_ptr[2], 3);
    EXPECT_EQ(data_ptr[3], 4);
}

// using TEST_F to test HContainer::find_pair
TEST_F(HContainerTest, find_pair)
{
    // check HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_j(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0).get_col_size(), 2);
    // find atom_ij in HR
    hamilt::AtomPair<double>* atom_ij_ptr = HR->find_pair(0, 1);
    // check if atom_ij is found
    EXPECT_EQ(atom_ij_ptr->get_atom_i(), 0);
    EXPECT_EQ(atom_ij_ptr->get_atom_j(), 1);
    EXPECT_EQ(atom_ij_ptr->get_row_size(), 2);
    EXPECT_EQ(atom_ij_ptr->get_col_size(), 2);
    // find atom_kl in HR
    hamilt::AtomPair<double>* atom_kl_ptr = HR->find_pair(1, 0);
    // check if atom_kl is found
    EXPECT_EQ(atom_kl_ptr->get_atom_i(), 1);
    EXPECT_EQ(atom_kl_ptr->get_atom_j(), 0);
    EXPECT_EQ(atom_kl_ptr->get_row_size(), 2);
    EXPECT_EQ(atom_kl_ptr->get_col_size(), 2);
    // find atom_ij not in HR
    hamilt::AtomPair<double>* atom_ij_ptr2 = HR->find_pair(0, 3);
    // check if atom_ij is found
    EXPECT_EQ(atom_ij_ptr2, nullptr);
}

// using TEST_F to test HContainer::get_atom_pair, both with atom_i, atom_j and with index
TEST_F(HContainerTest, get_atom_pair)
{
    // check  HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_j(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0).get_col_size(), 2);
    // get atom_ij in HR
    hamilt::AtomPair<double>& atom_ij_ref = HR->get_atom_pair(0, 1);
    // check if atom_ij is found
    EXPECT_EQ(atom_ij_ref.get_atom_i(), 0);
    EXPECT_EQ(atom_ij_ref.get_atom_j(), 1);
    EXPECT_EQ(atom_ij_ref.get_row_size(), 2);
    EXPECT_EQ(atom_ij_ref.get_col_size(), 2);
    // get atom_kl in HR
    hamilt::AtomPair<double>& atom_kl_ref = HR->get_atom_pair(1, 0);
    // check if atom_kl is found
    EXPECT_EQ(atom_kl_ref.get_atom_i(), 1);
    EXPECT_EQ(atom_kl_ref.get_atom_j(), 0);
    EXPECT_EQ(atom_kl_ref.get_row_size(), 2);
    EXPECT_EQ(atom_kl_ref.get_col_size(), 2);
    // get atom_ij in HR with index
    hamilt::AtomPair<double>& atom_ij_ref2 = HR->get_atom_pair(0);
    // check if atom_ij is found
    EXPECT_EQ(atom_ij_ref2.get_atom_i(), 0);
    EXPECT_EQ(atom_ij_ref2.get_atom_j(), 0);
    EXPECT_EQ(atom_ij_ref2.get_row_size(), 2);
    EXPECT_EQ(atom_ij_ref2.get_col_size(), 2);
    // get atom_kl in HR with index
    hamilt::AtomPair<double>& atom_kl_ref2 = HR->get_atom_pair(8);
    // check if atom_kl is found
    EXPECT_EQ(atom_kl_ref2.get_atom_i(), 2);
    EXPECT_EQ(atom_kl_ref2.get_atom_j(), 2);
    EXPECT_EQ(atom_kl_ref2.get_row_size(), 2);
    EXPECT_EQ(atom_kl_ref2.get_col_size(), 2);
}

// using TEST_F to test HContainer::fix_R and unfix_R
TEST_F(HContainerTest, fix_R)
{
    // check HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_j(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0).get_col_size(), 2);
    EXPECT_EQ(HR->size_R_loop(), 1);
    // fix R
    EXPECT_EQ(HR->fix_R(0, 0, 0), true);
    // check if R is fixed
    EXPECT_EQ(HR->get_current_R(), 0);
    // fix R again
    EXPECT_EQ(HR->fix_R(0, 0, 0), true);
    // check if R is fixed
    EXPECT_EQ(HR->get_current_R(), 0);
    // fix another R
    EXPECT_EQ(HR->fix_R(1, 0, 0), false);
    // check if R is fixed
    EXPECT_EQ(HR->get_current_R(), -1);
    // unfix R
    HR->unfix_R();
    // check if R is unfixed
    EXPECT_EQ(HR->get_current_R(), -1);
}

// using TEST_F to test HContainer::fix_gamma
TEST_F(HContainerTest, fix_gamma)
{
    // check HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_j(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0).get_col_size(), 2);
    EXPECT_EQ(HR->size_R_loop(), 1);
    hamilt::AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    hamilt::BaseMatrix<double>& tmp = atom_ij.get_HR_values(1, 0, 0);
    double tmp_array[4] = {1, 2, 3, 4};
    tmp.add_array(tmp_array);
    // insert atom_ij into HR
    HR->insert_pair(atom_ij);
    EXPECT_EQ(HR->size_R_loop(), 2);
    // fix gamma
    EXPECT_EQ(HR->is_gamma_only(), false);
    HR->fix_gamma();
    // check if gamma is fixed
    EXPECT_EQ(HR->is_gamma_only(), true);
    EXPECT_EQ(HR->size_R_loop(), 1);
    // fix gamma again
    HR->fix_gamma();
    // check if gamma is fixed
    EXPECT_EQ(HR->is_gamma_only(), true);
}

/**
 * using TEST_F to test HContainer::loop_R,
 * step: 1. size_R_loop(), 2. for-loop, loop_R(), 3. fix_R(), 4. do something
 */
TEST_F(HContainerTest, loop_R)
{
    // 1. size_R_loop()
    int size_for_loop_R = HR->size_R_loop();
    EXPECT_EQ(size_for_loop_R, 1);
    // 2. for-loop, loop_R()
    int rx, ry, rz;
    for (int i = 0; i < size_for_loop_R; i++)
    {
        HR->loop_R(i, rx, ry, rz);
        EXPECT_EQ(rx, 0);
        EXPECT_EQ(ry, 0);
        EXPECT_EQ(rz, 0);
        HR->fix_R(rx, ry, rz);
        // check if R is fixed
        EXPECT_EQ(HR->get_current_R(), i);
        // 4. do something
    }
    HR->unfix_R();
    // check if R is unfixed
    EXPECT_EQ(HR->get_current_R(), -1);
}

// using TEST_F to test HContainer::size_atom_pairs
// 1. test with R fixed
// 2. test with R unfixed
TEST_F(HContainerTest, size_atom_pairs)
{
    // get size_R_loop
    int size_R_loop = HR->size_R_loop();
    EXPECT_EQ(size_R_loop, 1);
    // 1. test with R fixed
    // fix R
    EXPECT_EQ(HR->get_current_R(), -1);
    bool ok = HR->fix_R(0, 0, 0);
    // check if R is fixed
    EXPECT_EQ(ok, true);
    EXPECT_EQ(HR->get_current_R(), 0);
    // get AP size
    int AP_size = HR->size_atom_pairs();
    EXPECT_EQ(AP_size, 9);
    // fix to another R
    hamilt::AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    hamilt::BaseMatrix<double>& tmp = atom_ij.get_HR_values(1, 0, 0);
    double tmp_array[4] = {1, 2, 3, 4};
    tmp.add_array(tmp_array);
    // insert atom_ij into HR
    HR->insert_pair(atom_ij);
    // get size_R_loop again, it should be 2
    size_R_loop = HR->size_R_loop();
    EXPECT_EQ(size_R_loop, 2);
    ok = HR->fix_R(1, 0, 0);
    // check if R is fixed
    EXPECT_EQ(ok, true);
    EXPECT_EQ(HR->get_current_R(), 1);
    EXPECT_EQ(HR->size_atom_pairs(), 1);
    // check if tmp_atom_pairs is correct
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0).get_atom_j(), 1);
    EXPECT_EQ(HR->get_atom_pair(0).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0).get_col_size(), 2);
    int* R_ptr = HR->get_atom_pair(0).get_R_index();
    EXPECT_EQ(R_ptr[0], 1);
    EXPECT_EQ(R_ptr[1], 0);
    EXPECT_EQ(R_ptr[2], 0);
    // check if data is correct
    double* data_ptr = HR->get_atom_pair(0).get_pointer();
    EXPECT_EQ(data_ptr[0], 1);
    EXPECT_EQ(data_ptr[1], 2);
    EXPECT_EQ(data_ptr[2], 3);
    EXPECT_EQ(data_ptr[3], 4);
    // 2. test with R unfixed
    // unfix R
    HR->unfix_R();
    // check if R is unfixed
    EXPECT_EQ(HR->get_current_R(), -1);
    // get AP size
    AP_size = HR->size_atom_pairs();
    EXPECT_EQ(AP_size, 9);
    // fix to another R with no AP
    ok = HR->fix_R(2, 0, 0);
    // check if R is fixed
    EXPECT_EQ(ok, false);
    EXPECT_EQ(HR->get_current_R(), -1);
    EXPECT_EQ(HR->size_atom_pairs(), 9);
}

// using TEST_F to test HContainer::data()
TEST_F(HContainerTest, data)
{
    // set up a hamilt::AtomPair
    hamilt::AtomPair<double> atom_ij(0, 1);
    atom_ij.set_size(2, 2);
    hamilt::BaseMatrix<double>& tmp = atom_ij.get_HR_values(0, 0, 0);
    double tmp_array[4] = {1, 2, 3, 4};
    tmp.add_array(tmp_array);
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    // insert atom_ij into HR
    HR->insert_pair(atom_ij);
    // check if atom_ij is inserted into HR
    EXPECT_EQ(HR->size_atom_pairs(), 9);
    EXPECT_EQ(HR->get_atom_pair(0, 1).get_atom_i(), 0);
    EXPECT_EQ(HR->get_atom_pair(0, 1).get_atom_j(), 1);
    EXPECT_EQ(HR->get_atom_pair(0, 1).get_row_size(), 2);
    EXPECT_EQ(HR->get_atom_pair(0, 1).get_col_size(), 2);
    // get data pointer
    double* data_ptr = HR->data(0, 1);
    // check if data pointer is correct
    EXPECT_EQ(data_ptr, HR->get_atom_pair(0, 1).get_pointer());
    // check if data is correct
    EXPECT_EQ(data_ptr[0], 1);
    EXPECT_EQ(data_ptr[1], 2);
    EXPECT_EQ(data_ptr[2], 3);
    EXPECT_EQ(data_ptr[3], 4);
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}