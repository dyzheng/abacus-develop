#include "module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h"
#include <gtest/gtest.h>
#include <complex>

using namespace hamilt;

class VKBBatchTest : public ::testing::Test {};

TEST_F(VKBBatchTest, ComputeOptimalBatchSize)
{
    // 200 atoms, 10 proj/atom, 10000 pw, 8GB
    int batch = compute_optimal_batch_size(200, 10, 10000, 8ULL * 1024 * 1024 * 1024);
    EXPECT_GT(batch, 0);
    EXPECT_LE(batch, 200);
}

TEST_F(VKBBatchTest, ComputeOptimalBatchSizeSmallMem)
{
    // Very small memory => should still give at least 1
    int batch = compute_optimal_batch_size(200, 10, 10000, 1024);
    EXPECT_EQ(batch, 1);
}

TEST_F(VKBBatchTest, ComputeOptimalBatchSizeLargeMem)
{
    // Very large memory => all atoms in one batch
    int batch = compute_optimal_batch_size(10, 5, 100, 1ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(batch, 10);
}

TEST_F(VKBBatchTest, ComputeOptimalBatchSizeZeroProj)
{
    // Zero projectors per atom => all atoms in one batch
    int batch = compute_optimal_batch_size(10, 0, 100, 1024);
    EXPECT_EQ(batch, 10);
}

TEST_F(VKBBatchTest, InitAutoDetect)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4, 4, 4, 8, 8, 8, 8, 8};
    // Use small memory to force multiple batches
    size_t small_mem = 10 * 100 * sizeof(std::complex<double>) * 2; // ~2 atoms worth
    mgr.init(10, nproj.data(), 100, small_mem);
    EXPECT_TRUE(mgr.is_initialized());
    EXPECT_GT(mgr.get_nbatch(), 1);

    // Verify all atoms are covered
    int total_atoms = 0;
    int total_nkb = 0;
    for (int i = 0; i < mgr.get_nbatch(); i++)
    {
        int start, end, nkb;
        mgr.get_batch_info(i, start, end, nkb);
        EXPECT_GE(start, 0);
        EXPECT_GT(end, start);
        total_atoms += (end - start);
        total_nkb += nkb;
    }
    EXPECT_EQ(total_atoms, 10);
    EXPECT_EQ(total_nkb, 60); // 5*4 + 5*8
}

TEST_F(VKBBatchTest, InitExplicitBatchSize)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    mgr.init(10, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024, 3); // 3 atoms per batch
    EXPECT_EQ(mgr.get_nbatch(), 4);                                  // 3+3+3+1

    int start, end, nkb;
    mgr.get_batch_info(0, start, end, nkb);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, 3);
    EXPECT_EQ(nkb, 12); // 3 * 4

    mgr.get_batch_info(3, start, end, nkb);
    EXPECT_EQ(start, 9);
    EXPECT_EQ(end, 10);
    EXPECT_EQ(nkb, 4); // 1 * 4
}

TEST_F(VKBBatchTest, SingleBatchLargeMem)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4};
    mgr.init(3, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(mgr.get_nbatch(), 1);

    int start, end, nkb;
    mgr.get_batch_info(0, start, end, nkb);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, 3);
    EXPECT_EQ(nkb, 12);
}

TEST_F(VKBBatchTest, MaxNkbBatch)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 8, 8, 4};
    mgr.init(5, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024, 2);
    // Batch 0: atoms 0,1 => nkb=8
    // Batch 1: atoms 2,3 => nkb=16
    // Batch 2: atom 4 => nkb=4
    EXPECT_EQ(mgr.get_max_nkb_batch(), 16);
}

TEST_F(VKBBatchTest, MaxBatchElements)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 8, 8, 4};
    mgr.init(5, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024, 2);
    // Batch 1 has nkb=16, npwx=100 => 1600 elements
    EXPECT_EQ(mgr.get_max_batch_elements(), 1600);
}

TEST_F(VKBBatchTest, FloatInstantiation)
{
    VKBBatchManager<std::complex<float>> mgr;
    std::vector<int> nproj = {4, 4, 4};
    mgr.init(3, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024);
    EXPECT_TRUE(mgr.is_initialized());
    EXPECT_EQ(mgr.get_nbatch(), 1);
}

TEST_F(VKBBatchTest, GetBatchInfoOutOfRange)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4};
    mgr.init(3, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024);

    int start, end, nkb;
    EXPECT_THROW(mgr.get_batch_info(-1, start, end, nkb), std::out_of_range);
    EXPECT_THROW(mgr.get_batch_info(1, start, end, nkb), std::out_of_range);
}

TEST_F(VKBBatchTest, JkbOffset)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 8, 8, 4};
    mgr.init(5, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024, 2);
    // Batch 0: atoms 0,1 => nkb=8, offset=0
    // Batch 1: atoms 2,3 => nkb=16, offset=8
    // Batch 2: atom 4 => nkb=4, offset=24
    EXPECT_EQ(mgr.get_jkb_offset(0), 0);
    EXPECT_EQ(mgr.get_jkb_offset(1), 8);
    EXPECT_EQ(mgr.get_jkb_offset(2), 24);
}

TEST_F(VKBBatchTest, JkbOffsetUniformProj)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    mgr.init(10, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024, 3);
    // 4 batches: 3+3+3+1
    // Batch 0: atoms 0-2 => nkb=12, offset=0
    // Batch 1: atoms 3-5 => nkb=12, offset=12
    // Batch 2: atoms 6-8 => nkb=12, offset=24
    // Batch 3: atom 9   => nkb=4,  offset=36
    EXPECT_EQ(mgr.get_jkb_offset(0), 0);
    EXPECT_EQ(mgr.get_jkb_offset(1), 12);
    EXPECT_EQ(mgr.get_jkb_offset(2), 24);
    EXPECT_EQ(mgr.get_jkb_offset(3), 36);
}

TEST_F(VKBBatchTest, JkbOffsetSingleBatch)
{
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 4, 4};
    mgr.init(3, nproj.data(), 100, 1ULL * 1024 * 1024 * 1024);
    // Single batch: offset must be 0
    EXPECT_EQ(mgr.get_nbatch(), 1);
    EXPECT_EQ(mgr.get_jkb_offset(0), 0);
}

TEST_F(VKBBatchTest, BatchContinuity)
{
    // Verify that batches are contiguous: end of batch i == start of batch i+1
    VKBBatchManager<std::complex<double>> mgr;
    std::vector<int> nproj = {4, 8, 4, 8, 4, 8, 4, 8};
    mgr.init(8, nproj.data(), 200, 1ULL * 1024 * 1024 * 1024, 3);

    for (int i = 0; i < mgr.get_nbatch() - 1; i++)
    {
        int start_i, end_i, nkb_i;
        int start_next, end_next, nkb_next;
        mgr.get_batch_info(i, start_i, end_i, nkb_i);
        mgr.get_batch_info(i + 1, start_next, end_next, nkb_next);
        EXPECT_EQ(end_i, start_next) << "Batch " << i << " end != batch " << i + 1 << " start";
    }
}
