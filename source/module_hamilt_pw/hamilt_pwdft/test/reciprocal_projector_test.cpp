#include "module_hamilt_pw/hamilt_pwdft/reciprocal_projector.h"
#include "module_hamilt_pw/hamilt_pwdft/projector_base.h"
#include "module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h"

#include <gtest/gtest.h>
#include <complex>
#include <cstring>
#include <vector>
#include <numeric>
#include <cmath>

#ifdef __MPI
#include <mpi.h>
#include "module_base/parallel_comm.h"
#endif

using CD = std::complex<double>;
using namespace hamilt;

/**
 * @brief Google Test Environment to handle MPI init/finalize.
 */
class MPIEnvironment : public ::testing::Environment
{
  public:
    void SetUp() override
    {
#ifdef __MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
        }
        // Set POOL_WORLD to MPI_COMM_WORLD for single-process tests
        POOL_WORLD = MPI_COMM_WORLD;
#endif
    }

    void TearDown() override
    {
#ifdef __MPI
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
        {
            MPI_Finalize();
        }
#endif
    }
};

// Register environment before main runs
static auto* const mpi_env_reg_ __attribute__((used)) =
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment());

/**
 * @brief Helper: fill vkb[j][g] = (j+1) * (g+1) as complex.
 *
 * vkb is stored row-major: vkb[j * npwx + g].
 */
static void fill_vkb(std::vector<CD>& vkb, int nkb, int npwx)
{
    vkb.assign(static_cast<size_t>(nkb) * npwx, CD(0.0, 0.0));
    for (int j = 0; j < nkb; j++)
    {
        for (int g = 0; g < npwx; g++)
        {
            vkb[j * npwx + g] = CD(static_cast<double>((j + 1) * (g + 1)), 0.0);
        }
    }
}

/**
 * @brief Helper: fill identity deeq.
 *
 * deeq[spin * b2 * b3 * b4 + iat * b3 * b4 + ip * b4 + ip2] = (ip == ip2) ? 1.0 : 0.0
 */
static void fill_identity_deeq(std::vector<double>& deeq, int nspin, int nat, int nhm)
{
    const size_t total = static_cast<size_t>(nspin) * nat * nhm * nhm;
    deeq.assign(total, 0.0);
    for (int s = 0; s < nspin; s++)
    {
        for (int ia = 0; ia < nat; ia++)
        {
            for (int ip = 0; ip < nhm; ip++)
            {
                deeq[((s * nat + ia) * nhm + ip) * nhm + ip] = 1.0;
            }
        }
    }
}

/**
 * @brief Helper: compute reference becp = vkb^H * psi manually.
 *
 * becp[band * nkb + j] = sum_g conj(vkb[j * npwx + g]) * psi[band * max_npw + g]
 * Only sums over g in [0, npw).
 */
static void ref_compute_becp(const std::vector<CD>& vkb, int nkb, int npwx,
                              const std::vector<CD>& psi, int nbands, int npw, int max_npw,
                              std::vector<CD>& becp)
{
    becp.assign(static_cast<size_t>(nbands) * nkb, CD(0.0, 0.0));
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int j = 0; j < nkb; j++)
        {
            CD sum(0.0, 0.0);
            for (int g = 0; g < npw; g++)
            {
                sum += std::conj(vkb[j * npwx + g]) * psi[ib * max_npw + g];
            }
            becp[ib * nkb + j] = sum;
        }
    }
}

/**
 * @brief Helper: apply nonlocal_op reference (identity D).
 *
 * With identity D, ps[j * nbands + band] = becp[band * nkb + j].
 * Then hpsi[band * max_npw + g] += sum_j vkb[j * npwx + g] * ps[j * nbands + band]
 */
static void ref_apply_deeq_identity(const std::vector<CD>& vkb, int nkb, int npwx,
                                     const std::vector<CD>& becp, int nbands, int npw, int max_npw,
                                     std::vector<CD>& hpsi)
{
    // With identity D, ps = becp (with transposed layout)
    // hpsi[band][g] += sum_j vkb[j][g] * becp[band][j]
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            CD val(0.0, 0.0);
            for (int j = 0; j < nkb; j++)
            {
                val += vkb[j * npwx + g] * becp[ib * nkb + j];
            }
            hpsi[ib * max_npw + g] += val;
        }
    }
}

/**
 * @brief Helper: apply nonlocal_op reference with non-trivial D matrix.
 *
 * ps[j * nbands + band] = sum_{j2} D[iat_of_j, j_local, j2_local] * becp[band * nkb + j2_global]
 * Then hpsi[band * max_npw + g] += sum_j vkb[j * npwx + g] * ps[j * nbands + band]
 *
 * We use the actual nonlocal_op indexing to be consistent:
 * For each atom type it, for each atom ii of that type:
 *   for each band jj:
 *     for each projector kk:
 *       ps[(sum + ii * nproj + kk) * nbands + jj] +=
 *           deeq[(spin * nat + iat + ii) * nhm * nhm + xx * nhm + kk]
 *           * becp[jj * nkb + sum + ii * nproj + xx]
 */
static void ref_apply_deeq_general(const std::vector<CD>& vkb, int nkb, int npwx,
                                    const std::vector<CD>& becp, int nbands, int npw, int max_npw,
                                    const std::vector<double>& deeq, int nat, int nhm,
                                    int ntype, const std::vector<int>& na_per_type,
                                    const std::vector<int>& nh_per_type,
                                    int spin,
                                    std::vector<CD>& hpsi)
{
    // Step 1: compute ps
    std::vector<CD> ps(static_cast<size_t>(nkb) * nbands, CD(0.0, 0.0));
    int sum = 0;
    int iat = 0;
    for (int it = 0; it < ntype; it++)
    {
        const int na = na_per_type[it];
        const int nproj = nh_per_type[it];
        for (int ii = 0; ii < na; ii++)
        {
            for (int jj = 0; jj < nbands; jj++)
            {
                for (int kk = 0; kk < nproj; kk++)
                {
                    for (int xx = 0; xx < nproj; xx++)
                    {
                        ps[(sum + ii * nproj + kk) * nbands + jj]
                            += deeq[((spin * nat + iat + ii) * nhm + xx) * nhm + kk]
                               * becp[jj * nkb + sum + ii * nproj + xx];
                    }
                }
            }
        }
        sum += na * nproj;
        iat += na;
    }

    // Step 2: hpsi += vkb * ps
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            CD val(0.0, 0.0);
            for (int j = 0; j < nkb; j++)
            {
                val += vkb[j * npwx + g] * ps[j * nbands + ib];
            }
            hpsi[ib * max_npw + g] += val;
        }
    }
}

// =====================================================================
// Test Fixture
// =====================================================================

class ReciprocalProjectorTest : public ::testing::Test
{
  protected:
    // Tolerance for floating-point comparisons
    static constexpr double tol = 1.0e-10;
};

// =====================================================================
// Test 1: Constructor and memory reporting in full mode
// =====================================================================
TEST_F(ReciprocalProjectorTest, ConstructorAndMemoryReporting)
{
    const int ntype = 1;
    const int nkb = 4;
    const int npwx = 8;
    std::vector<int> na_per_type = {2};
    std::vector<int> nh_per_type = {2};
    std::vector<int> isk = {0};
    const int ik = 0;
    const int nat = 2;
    const int nhm = 2;
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb(nkb * npwx, CD(1.0, 0.0));
    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), ik, deeq_bounds,
        nullptr); // full mode

    // In full mode, no batch buffers are allocated in constructor
    EXPECT_EQ(proj.get_memory_bytes(), 0u);
    // Destructor should not crash (implicit)
}

// =====================================================================
// Test 2: Constructor in batched mode
// =====================================================================
TEST_F(ReciprocalProjectorTest, ConstructorBatchedMode)
{
    const int ntype = 1;
    const int npwx = 8;
    const int nat = 6;
    const int nhm = 2;
    const int nkb = nat * nhm; // 12
    std::vector<int> na_per_type = {6};
    std::vector<int> nh_per_type = {2};
    std::vector<int> isk = {0};
    const int ik = 0;
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb(nkb * npwx, CD(1.0, 0.0));
    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // Create batch manager with 2 atoms per batch for 6 atoms => 3 batches
    VKBBatchManager<CD> mgr;
    std::vector<int> nproj_per_atom(nat, nhm); // each atom has 2 projectors
    mgr.init(nat, nproj_per_atom.data(), npwx, 1ULL << 30, 2);
    ASSERT_EQ(mgr.get_nbatch(), 3);

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        nullptr, vkb.data(), nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), ik, deeq_bounds,
        &mgr);

    // In batched mode with >1 batch, batch VKB buffer is allocated
    EXPECT_GT(proj.get_memory_bytes(), 0u);
}

// =====================================================================
// Test 3: compute_becp full mode single band
// =====================================================================
TEST_F(ReciprocalProjectorTest, ComputeBecpFullModeSingleBand)
{
    const int ntype = 1;
    const int nat = 2;
    const int nhm = 2;
    const int nkb = 4;   // 2 atoms * 2 proj
    const int npwx = 8;
    const int npw = 5;
    const int nbands = 1;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {nat};
    std::vector<int> nh_per_type = {nhm};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // psi[g] = g+1
    std::vector<CD> psi(max_npw, CD(0.0, 0.0));
    for (int g = 0; g < npw; g++)
    {
        psi[g] = CD(static_cast<double>(g + 1), 0.0);
    }

    // Expected becp
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);

    // Actual becp
    std::vector<CD> becp(nkb, CD(0.0, 0.0));

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, nullptr);

    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    for (int j = 0; j < nkb; j++)
    {
        EXPECT_NEAR(becp[j].real(), becp_ref[j].real(), tol) << "j=" << j;
        EXPECT_NEAR(becp[j].imag(), becp_ref[j].imag(), tol) << "j=" << j;
    }
}

// =====================================================================
// Test 4: compute_becp full mode multi-band
// =====================================================================
TEST_F(ReciprocalProjectorTest, ComputeBecpFullModeMultiBand)
{
    const int ntype = 1;
    const int nat = 2;
    const int nhm = 2;
    const int nkb = 4;
    const int npwx = 8;
    const int npw = 5;
    const int nbands = 3;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {nat};
    std::vector<int> nh_per_type = {nhm};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // psi[band][g] = (band+1) * (g+1)
    std::vector<CD> psi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            psi[ib * max_npw + g] = CD(static_cast<double>((ib + 1) * (g + 1)), 0.0);
        }
    }

    // Reference
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);

    // Actual
    std::vector<CD> becp(static_cast<size_t>(nbands) * nkb, CD(0.0, 0.0));

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, nullptr);

    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    for (int i = 0; i < nbands * nkb; i++)
    {
        EXPECT_NEAR(becp[i].real(), becp_ref[i].real(), tol) << "i=" << i;
        EXPECT_NEAR(becp[i].imag(), becp_ref[i].imag(), tol) << "i=" << i;
    }
}

// =====================================================================
// Test 5: apply_deeq full mode single band (identity D)
// =====================================================================
TEST_F(ReciprocalProjectorTest, ApplyDeeqFullModeSingleBand)
{
    const int ntype = 1;
    const int nat = 2;
    const int nhm = 2;
    const int nkb = 4;
    const int npwx = 8;
    const int npw = 5;
    const int nbands = 1;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {nat};
    std::vector<int> nh_per_type = {nhm};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // psi
    std::vector<CD> psi(max_npw, CD(0.0, 0.0));
    for (int g = 0; g < npw; g++)
    {
        psi[g] = CD(static_cast<double>(g + 1), 0.0);
    }

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, nullptr);

    // Step 1: compute becp
    std::vector<CD> becp(nkb, CD(0.0, 0.0));
    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    // Step 2: apply_deeq
    std::vector<CD> hpsi(max_npw, CD(0.0, 0.0));
    proj.apply_deeq_and_accumulate(becp.data(), hpsi.data(), nbands, npw, max_npw, npol);

    // Reference: with identity D, hpsi = vkb * becp
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);
    std::vector<CD> hpsi_ref(max_npw, CD(0.0, 0.0));
    ref_apply_deeq_identity(vkb, nkb, npwx, becp_ref, nbands, npw, max_npw, hpsi_ref);

    for (int g = 0; g < npw; g++)
    {
        EXPECT_NEAR(hpsi[g].real(), hpsi_ref[g].real(), tol) << "g=" << g;
        EXPECT_NEAR(hpsi[g].imag(), hpsi_ref[g].imag(), tol) << "g=" << g;
    }
    // Verify that elements beyond npw are still zero
    for (int g = npw; g < max_npw; g++)
    {
        EXPECT_NEAR(hpsi[g].real(), 0.0, tol) << "beyond npw: g=" << g;
        EXPECT_NEAR(hpsi[g].imag(), 0.0, tol) << "beyond npw: g=" << g;
    }
}

// =====================================================================
// Test 6: apply_deeq full mode multi-band (non-trivial D)
// =====================================================================
TEST_F(ReciprocalProjectorTest, ApplyDeeqFullModeMultiBand)
{
    const int ntype = 1;
    const int nat = 2;
    const int nhm = 2;
    const int nkb = 4;
    const int npwx = 8;
    const int npw = 5;
    const int nbands = 2;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {nat};
    std::vector<int> nh_per_type = {nhm};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    // Non-trivial D: D[iat][ip][ip2] with some off-diagonal terms
    // For atom 0: D = [[2.0, 0.5], [0.5, 1.0]]
    // For atom 1: D = [[1.5, 0.3], [0.3, 2.5]]
    const size_t deeq_size = static_cast<size_t>(1) * nat * nhm * nhm;
    std::vector<double> deeq(deeq_size, 0.0);
    // atom 0
    deeq[((0 * nat + 0) * nhm + 0) * nhm + 0] = 2.0;
    deeq[((0 * nat + 0) * nhm + 0) * nhm + 1] = 0.5;
    deeq[((0 * nat + 0) * nhm + 1) * nhm + 0] = 0.5;
    deeq[((0 * nat + 0) * nhm + 1) * nhm + 1] = 1.0;
    // atom 1
    deeq[((0 * nat + 1) * nhm + 0) * nhm + 0] = 1.5;
    deeq[((0 * nat + 1) * nhm + 0) * nhm + 1] = 0.3;
    deeq[((0 * nat + 1) * nhm + 1) * nhm + 0] = 0.3;
    deeq[((0 * nat + 1) * nhm + 1) * nhm + 1] = 2.5;

    // psi[band][g] = (band+1) * (g+1)
    std::vector<CD> psi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            psi[ib * max_npw + g] = CD(static_cast<double>((ib + 1) * (g + 1)), 0.0);
        }
    }

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, nullptr);

    // Step 1: compute becp
    std::vector<CD> becp(static_cast<size_t>(nbands) * nkb, CD(0.0, 0.0));
    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    // Step 2: apply_deeq
    std::vector<CD> hpsi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    proj.apply_deeq_and_accumulate(becp.data(), hpsi.data(), nbands, npw, max_npw, npol);

    // Reference (general D)
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);
    std::vector<CD> hpsi_ref(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    ref_apply_deeq_general(vkb, nkb, npwx, becp_ref, nbands, npw, max_npw,
                           deeq, nat, nhm, ntype, na_per_type, nh_per_type, 0, hpsi_ref);

    for (int i = 0; i < nbands * max_npw; i++)
    {
        EXPECT_NEAR(hpsi[i].real(), hpsi_ref[i].real(), tol) << "i=" << i;
        EXPECT_NEAR(hpsi[i].imag(), hpsi_ref[i].imag(), tol) << "i=" << i;
    }
}

// =====================================================================
// Test 7: Batched mode compute_becp
// =====================================================================
TEST_F(ReciprocalProjectorTest, BatchedModeComputeBecp)
{
    // 2 types: type A with nh=2, 2 atoms; type B with nh=3, 2 atoms
    const int ntype = 2;
    const int nat = 4;
    const int nhm = 3; // max(2, 3)
    const int nkb = 2 * 2 + 2 * 3; // = 10
    const int npwx = 8;
    const int npw = 6;
    const int nbands = 2;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {2, 2};
    std::vector<int> nh_per_type = {2, 3};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // psi[band][g] = (band+1) * (g+1)
    std::vector<CD> psi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            psi[ib * max_npw + g] = CD(static_cast<double>((ib + 1) * (g + 1)), 0.0);
        }
    }

    // Create batch manager with 2 atoms per batch
    // Atom layout: atom 0,1 (type A, nh=2), atom 2,3 (type B, nh=3)
    // nproj_per_atom: [2, 2, 3, 3]
    VKBBatchManager<CD> mgr;
    std::vector<int> nproj_per_atom = {2, 2, 3, 3};
    mgr.init(nat, nproj_per_atom.data(), npwx, 1ULL << 30, 2);
    ASSERT_EQ(mgr.get_nbatch(), 2);

    // Full-mode reference becp
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);

    // Batched mode becp
    std::vector<CD> becp(static_cast<size_t>(nbands) * nkb, CD(0.0, 0.0));

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        nullptr, vkb.data(), nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, &mgr);

    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    for (int i = 0; i < nbands * nkb; i++)
    {
        EXPECT_NEAR(becp[i].real(), becp_ref[i].real(), tol) << "i=" << i;
        EXPECT_NEAR(becp[i].imag(), becp_ref[i].imag(), tol) << "i=" << i;
    }
}

// =====================================================================
// Test 8: Batched mode end-to-end (becp + apply_deeq)
// =====================================================================
TEST_F(ReciprocalProjectorTest, BatchedModeEndToEnd)
{
    // Same setup as Test 7 but with full round-trip
    const int ntype = 2;
    const int nat = 4;
    const int nhm = 3;
    const int nkb = 2 * 2 + 2 * 3; // = 10
    const int npwx = 8;
    const int npw = 6;
    const int nbands = 2;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {2, 2};
    std::vector<int> nh_per_type = {2, 3};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    // psi
    std::vector<CD> psi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int g = 0; g < npw; g++)
        {
            psi[ib * max_npw + g] = CD(static_cast<double>((ib + 1) * (g + 1)), 0.0);
        }
    }

    // ===== Full mode reference =====
    std::vector<CD> becp_full_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_full_ref);
    std::vector<CD> hpsi_full_ref(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));
    ref_apply_deeq_identity(vkb, nkb, npwx, becp_full_ref, nbands, npw, max_npw, hpsi_full_ref);

    // ===== Batched mode =====
    VKBBatchManager<CD> mgr;
    std::vector<int> nproj_per_atom = {2, 2, 3, 3};
    mgr.init(nat, nproj_per_atom.data(), npwx, 1ULL << 30, 2);

    std::vector<CD> becp(static_cast<size_t>(nbands) * nkb, CD(0.0, 0.0));
    std::vector<CD> hpsi(static_cast<size_t>(nbands) * max_npw, CD(0.0, 0.0));

    ReciprocalProjector<CD, base_device::DEVICE_CPU> proj(
        nullptr, vkb.data(), nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, &mgr);

    proj.compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);
    proj.apply_deeq_and_accumulate(becp.data(), hpsi.data(), nbands, npw, max_npw, npol);

    // Compare against full-mode reference
    for (int i = 0; i < nbands * max_npw; i++)
    {
        EXPECT_NEAR(hpsi[i].real(), hpsi_full_ref[i].real(), tol) << "i=" << i;
        EXPECT_NEAR(hpsi[i].imag(), hpsi_full_ref[i].imag(), tol) << "i=" << i;
    }
}

// =====================================================================
// Test 9: Polymorphism through base pointer
// =====================================================================
TEST_F(ReciprocalProjectorTest, PolymorphismThroughBasePointer)
{
    const int ntype = 1;
    const int nat = 2;
    const int nhm = 2;
    const int nkb = 4;
    const int npwx = 8;
    const int npw = 5;
    const int nbands = 1;
    const int max_npw = npwx;
    const int npol = 1;
    std::vector<int> na_per_type = {nat};
    std::vector<int> nh_per_type = {nhm};
    std::vector<int> isk = {0};
    int deeq_bounds[3] = {nat, nhm, nhm};

    std::vector<CD> vkb;
    fill_vkb(vkb, nkb, npwx);

    std::vector<double> deeq;
    fill_identity_deeq(deeq, 1, nat, nhm);

    std::vector<CD> psi(max_npw, CD(0.0, 0.0));
    for (int g = 0; g < npw; g++)
    {
        psi[g] = CD(static_cast<double>(g + 1), 0.0);
    }

    // Create through base pointer
    ProjectorBase<CD>* base_ptr = new ReciprocalProjector<CD, base_device::DEVICE_CPU>(
        vkb.data(), nullptr, nkb, npwx,
        deeq.data(), nullptr,
        ntype, na_per_type.data(), nh_per_type.data(),
        isk.data(), 0, deeq_bounds, nullptr);

    // Call compute_becp through base pointer
    std::vector<CD> becp(nkb, CD(0.0, 0.0));
    base_ptr->compute_becp(psi.data(), becp.data(), nbands, npw, max_npw, npol);

    // Verify becp
    std::vector<CD> becp_ref;
    ref_compute_becp(vkb, nkb, npwx, psi, nbands, npw, max_npw, becp_ref);
    for (int j = 0; j < nkb; j++)
    {
        EXPECT_NEAR(becp[j].real(), becp_ref[j].real(), tol) << "j=" << j;
        EXPECT_NEAR(becp[j].imag(), becp_ref[j].imag(), tol) << "j=" << j;
    }

    // Call apply_deeq through base pointer
    std::vector<CD> hpsi(max_npw, CD(0.0, 0.0));
    base_ptr->apply_deeq_and_accumulate(becp.data(), hpsi.data(), nbands, npw, max_npw, npol);

    // Verify hpsi
    std::vector<CD> hpsi_ref(max_npw, CD(0.0, 0.0));
    ref_apply_deeq_identity(vkb, nkb, npwx, becp_ref, nbands, npw, max_npw, hpsi_ref);
    for (int g = 0; g < npw; g++)
    {
        EXPECT_NEAR(hpsi[g].real(), hpsi_ref[g].real(), tol) << "g=" << g;
        EXPECT_NEAR(hpsi[g].imag(), hpsi_ref[g].imag(), tol) << "g=" << g;
    }

    // Call get_memory_bytes through base pointer
    // In full mode, after apply_deeq the ps_ workspace is allocated:
    // ps_ size = nkb * nbands = 4 * 1 = 4 elements * sizeof(CD) = 64 bytes
    size_t mem = base_ptr->get_memory_bytes();
    const size_t expected_ps_bytes = static_cast<size_t>(nkb) * nbands * sizeof(CD);
    EXPECT_EQ(mem, expected_ps_bytes);

    delete base_ptr;
}
