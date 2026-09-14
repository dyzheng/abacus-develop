#include "gmock/gmock.h"
#include "gtest/gtest.h"
#define private public
#include "source_io/module_parameter/parameter.h"
#undef private
#include "source_io/module_chgpot/rhog_io.h"
#ifdef __MPI
#include "source_basis/module_pw/test/test_tool.h"
#include "mpi.h"
#endif

/**
 * - Tested Functions:
 *  - read_rhog()
 */

class ReadRhogTest : public ::testing::Test
{
  protected:
    ModulePW::PW_Basis* rhopw = nullptr;
    std::complex<double>** rhog = nullptr;

    virtual void SetUp()
    {
        rhopw = new ModulePW::PW_Basis;
        rhog = new std::complex<double>*[1];
        rhog[0] = new std::complex<double>[1471];
    }
    virtual void TearDown()
    {
        if (rhopw != nullptr) {
            delete rhopw;
}
        if (rhog[0] != nullptr) {
            delete[] rhog[0];
}
        if (rhog != nullptr) {
            delete[] rhog;
}
    }
};

// Test the read_rhog function
TEST_F(ReadRhogTest, ReadRhog)
{
    std::string filename = "./support/charge-density.dat";
    PARAM.input.nspin = 1;
#ifdef __MPI
    rhopw->initmpi(GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL, MPI_COMM_WORLD);
#endif
    rhopw->initgrids(6.5, ModuleBase::Matrix3(-0.5, 0.0, 0.5, 0.0, 0.5, 0.5, -0.5, 0.5, 0.0), 120);
    rhopw->initparameters(false, 120);
    rhopw->setuptransform();
    rhopw->collect_local_pw();

    bool result = ModuleIO::read_rhog(filename, rhopw, rhog);

    EXPECT_TRUE(result);
    EXPECT_DOUBLE_EQ(rhog[0][0].real(), -1.0304462993299456e-05);
    EXPECT_DOUBLE_EQ(rhog[0][0].imag(), -1.2701788626185278e-13);
    EXPECT_DOUBLE_EQ(rhog[0][1].real(), -0.0003875762482855959);
    EXPECT_DOUBLE_EQ(rhog[0][1].imag(), -4.2556814316812048e-12);
    EXPECT_DOUBLE_EQ(rhog[0][1470].real(), -3.5683133614445107e-05);
    EXPECT_DOUBLE_EQ(rhog[0][1470].imag(), 1.6176615686863767e-12);
}

// Test the read_rhog function when the file is not found
TEST_F(ReadRhogTest, NotFoundFile)
{
    std::string filename = "notfound.txt";

    GlobalV::ofs_warning.open("test_read_rhog.txt");
    bool result = ModuleIO::read_rhog(filename, rhopw, rhog);
    GlobalV::ofs_warning.close();

    std::ifstream ifs_running("test_read_rhog.txt");
    std::stringstream ss;
    ss << ifs_running.rdbuf();
    std::string file_content = ss.str();
    ifs_running.close();

    std::string expected_content = " ModuleIO::read_rhog  warning : Can't open file notfound.txt\n";

    EXPECT_FALSE(result);
    EXPECT_EQ(file_content, expected_content);
    std::remove("test_read_rhog.txt");
}

// Test the read_rhog function when tgamma_only is inconsistent
TEST_F(ReadRhogTest, InconsistentGammaOnly)
{
    std::string filename = "./support/charge-density.dat";
    PARAM.input.nspin = 2;
    rhopw->gamma_only = true;

    GlobalV::ofs_warning.open("test_read_rhog.txt");
    bool result = ModuleIO::read_rhog(filename, rhopw, rhog);
    GlobalV::ofs_warning.close();

    std::ifstream ifs_running("test_read_rhog.txt");
    std::stringstream ss;
    ss << ifs_running.rdbuf();
    std::string file_content = ss.str();
    ifs_running.close();

    std::string expected_content
        = " ModuleIO::read_rhog  warning : some planewaves in file are not used\n ModuleIO::read_rhog  warning : some "
          "spin channels in file are missing\n ModuleIO::read_rhog  warning : gamma_only read from file is "
          "inconsistent with INPUT\n";

    EXPECT_FALSE(result);
    EXPECT_EQ(file_content, expected_content);
    std::remove("test_read_rhog.txt");
}

// Test the read_rhog function when some planewaves in file are missing
TEST_F(ReadRhogTest, SomePWMissing)
{
    std::string filename = "./support/charge-density.dat";
    PARAM.input.nspin = 1;
    rhopw->npwtot = 2000;

    GlobalV::ofs_warning.open("test_read_rhog.txt");
    bool result = ModuleIO::read_rhog(filename, rhopw, rhog);
    GlobalV::ofs_warning.close();

    std::ifstream ifs_running("test_read_rhog.txt");
    std::stringstream ss;
    ss << ifs_running.rdbuf();
    std::string file_content = ss.str();
    ifs_running.close();

    std::string expected_content = " ModuleIO::read_rhog  warning : some planewaves in file are missing\n";

    EXPECT_TRUE(result);
    EXPECT_EQ(file_content, expected_content);
    std::remove("test_read_rhog.txt");
}

// Regression test for C-29: a charge-density file written with a LARGER
// plane-wave basis (here ecutwfc = 120 Ry, npwtot = 1471) must not write
// outside the rhog buffer when it is read into a smaller basis.
//
// Inside the FFT box but outside the current basis set the map fftixyz2ig
// carries -1, and the unguarded code wrote rhog[is][-1] -- 16 bytes before
// the buffer, i.e. into the heap chunk header sitting there.  The damage is
// silent: glibc only notices much later, when that chunk is freed
// (Charge::destroy at teardown, "free(): invalid next size (normal)").
// The canary slot in front of the buffer makes the out-of-bounds write
// deterministic without ASAN.
TEST_F(ReadRhogTest, LargerBasisInFileDoesNotWriteBeforeBuffer)
{
    const std::string filename = "./support/charge-density.dat";
    PARAM.input.nspin = 1;

#ifdef __MPI
    rhopw->initmpi(GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL, MPI_COMM_WORLD);
#endif
    // Same cell and same FFT box (ecutrho = 120) as the file, but a much
    // smaller plane-wave sphere (ecutwfc = 30): most of the 1471 Miller
    // indices stored in the file are then inside the box yet outside the
    // basis set, which is exactly the C-29 trigger.
    rhopw->initgrids(6.5, ModuleBase::Matrix3(-0.5, 0.0, 0.5, 0.0, 0.5, 0.5, -0.5, 0.5, 0.0), 120);
    rhopw->initparameters(false, 30);
    rhopw->setuptransform();
    rhopw->collect_local_pw();
    ASSERT_GT(rhopw->npw, 0);
    ASSERT_LT(rhopw->npw, 1471); // the file must be the larger basis

    // Canary slot immediately in front of the buffer the reader fills.
    std::vector<std::complex<double>> guarded(rhopw->npw + 1);
    const std::complex<double> canary(3.25, -1.5);
    guarded[0] = canary;
    std::complex<double>* const fixture_buffer = rhog[0];
    rhog[0] = guarded.data() + 1;

    GlobalV::ofs_warning.open("test_read_rhog.txt");
    const bool result = ModuleIO::read_rhog(filename, rhopw, rhog);
    GlobalV::ofs_warning.close();

    rhog[0] = fixture_buffer; // restore the fixture allocation for TearDown

    // The scenario itself must be a larger-basis file, otherwise the test
    // would not exercise the guard.
    std::ifstream ifs_warning("test_read_rhog.txt");
    std::stringstream ss;
    ss << ifs_warning.rdbuf();
    ifs_warning.close();
    std::remove("test_read_rhog.txt");

    EXPECT_TRUE(result);
    EXPECT_THAT(ss.str(), ::testing::HasSubstr("some planewaves in file are not used"));
    // Branch check: the canary next to the buffer must be untouched, i.e. the
    // reader must not have written at ig == -1.
    EXPECT_EQ(guarded[0].real(), canary.real());
    EXPECT_EQ(guarded[0].imag(), canary.imag());
}

int main(int argc, char** argv)
{
#ifdef __MPI
    setupmpi(argc, argv, GlobalV::NPROC, GlobalV::MY_RANK);
    divide_pools(GlobalV::NPROC,
                 GlobalV::MY_RANK,
                 GlobalV::NPROC_IN_POOL,
                 GlobalV::KPAR,
                 GlobalV::MY_POOL,
                 GlobalV::RANK_IN_POOL);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    finishmpi();
#endif
    return result;
}
