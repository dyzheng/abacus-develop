#include "../read_txt_input_list.h"
#include "../read_txt_input_process.h"
#include "gtest/gtest.h"
#include "../../abacus_parameters.h"

class item_test : public testing::Test
{
protected:
    
    Read_Txt_Input::Input_List *input_list = nullptr;
    Read_Txt_Input::Input_Process *input_process = nullptr;

    void SetUp()
    {
        input_list = new Read_Txt_Input::Input_List;
        (*input_list).set_items();
        this->input_process = new Read_Txt_Input::Input_Process(*input_list);
	}

    void first_test()
    {
        generate_input();
		(*input_process).read_and_convert("INPUT");
    }
	void test_input(std::string input_name)
	{
		(*input_process).read_and_convert(input_name);
	}

    void generate_input()
    {
        std::ofstream in("INPUT");
        in<<"mixing_beta 0.5"<<std::endl;
        in<<"ecut 71.80 Ry"<<std::endl;
        in<<"scf_thr 3.5e-6"<<std::endl;
        //in<<""<<std::endl;
        //in<<""<<std::endl;

        in.close();
    }

    void TearDown()
    {
        delete[] input_process;
    }
};



TEST_F(item_test, first)
{
    first_test();
    EXPECT_DOUBLE_EQ(ABACUS::para.scfp.mixing_beta, 0.5);
    EXPECT_DOUBLE_EQ(ABACUS::para.pwp.ecut[0], 71.80);
    EXPECT_DOUBLE_EQ(ABACUS::para.scfp.scf_thr, 3.5e-6);
}

TEST_F(item_test, check_output)
{
	test_input("INPUT-new-0");
	EXPECT_DOUBLE_EQ(ABACUS::para.scfp.mixing_beta, 0.5);
	EXPECT_DOUBLE_EQ(ABACUS::para.pwp.ecut[0], 71.80);
    EXPECT_DOUBLE_EQ(ABACUS::para.scfp.scf_thr, 3.5e-6);
}

int main(int argc, char **argv)
{
#ifdef __MPI
	MPI_Init(&argc, &argv);
#endif
	testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
	MPI_Finalize();
#endif
	return result;
}