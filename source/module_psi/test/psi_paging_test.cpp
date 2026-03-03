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

TEST_F(PsiPagingTest, PagedGPUResize)
{
    // Test PAGED_GPU mode allocation on CPU device
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
    psi.resize(10, 20, 100);

    // CPU pointer should be accessible
    auto* ptr = psi.get_cpu_pointer(0);
    EXPECT_NE(ptr, nullptr);

    // current_k_gpu should be -1 (nothing loaded yet)
    EXPECT_EQ(psi.get_current_k_gpu(), -1);
}

TEST_F(PsiPagingTest, LoadAndStoreKPoint)
{
    // Test load/store cycle on CPU device (acts as identity)
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
    psi.resize(4, 2, 3);

    // Write test data to CPU buffer for k=2
    auto* cpu_ptr = psi.get_cpu_pointer(2);
    for (int i = 0; i < 2 * 3; i++)
    {
        cpu_ptr[i] = std::complex<double>(2.0 + i, 1.0);
    }

    // Load k=2 to device
    psi.load_k_to_gpu(2);
    EXPECT_EQ(psi.get_current_k_gpu(), 2);

    // Store from device back to CPU k=2
    psi.store_k_from_gpu(2);

    // Verify data round-tripped correctly
    auto* verify_ptr = psi.get_cpu_pointer(2);
    for (int i = 0; i < 2 * 3; i++)
    {
        EXPECT_DOUBLE_EQ(verify_ptr[i].real(), 2.0 + i);
        EXPECT_DOUBLE_EQ(verify_ptr[i].imag(), 1.0);
    }
}

TEST_F(PsiPagingTest, EnsureKOnGPU)
{
    Psi<std::complex<double>> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
    psi.resize(5, 2, 3);

    // First call loads
    psi.ensure_k_on_gpu(3);
    EXPECT_EQ(psi.get_current_k_gpu(), 3);

    // Second call same k - should be no-op
    psi.ensure_k_on_gpu(3);
    EXPECT_EQ(psi.get_current_k_gpu(), 3);

    // Different k - should load new
    psi.ensure_k_on_gpu(1);
    EXPECT_EQ(psi.get_current_k_gpu(), 1);
}

TEST_F(PsiPagingTest, ProperCleanup)
{
    // Test that destructor doesn't crash
    {
        Psi<std::complex<double>> psi;
        psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
        psi.resize(10, 20, 100);
        psi.load_k_to_gpu(0);
    }
    SUCCEED();
}
