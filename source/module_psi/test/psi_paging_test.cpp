#include "module_psi/psi.h"
#include "gtest/gtest.h"
#include <complex>

using namespace psi;

class PsiPagingTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
    }
};

TEST_F(PsiPagingTest, DefaultStorageMode)
{
    Psi<std::complex<double>> psi;
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::ALL_GPU);
}

TEST_F(PsiPagingTest, SetStorageMode)
{
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::PAGED_GPU);
}

TEST_F(PsiPagingTest, SetStorageModeAllCPU)
{
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::ALL_CPU);
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::ALL_CPU);
}

TEST_F(PsiPagingTest, SetSameModeTwice)
{
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::ALL_GPU);
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::ALL_GPU);
}
