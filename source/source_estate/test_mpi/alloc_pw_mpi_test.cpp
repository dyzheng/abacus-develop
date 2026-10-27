#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_estate/module_constraint/alloc_pw.h"
#include "source_io/module_parameter/parameter.h"

namespace
{

std::complex<double> tab_value(const int target, const int ig)
{
    return std::complex<double>(0.1 * (target + 1) - 0.03 * ig,
                                0.07 * (ig + 2) + 0.02 * target);
}

std::complex<double> psi_value(const int band, const int ig)
{
    return std::complex<double>(0.05 * (band + 1) + 0.02 * ig,
                                -0.04 * (ig + 3) + 0.01 * band);
}

std::complex<double> phi_value(const int target, const int ig)
{
    const double x = 0.1 * (ig - 8);
    // The PW target space stores a real projector overlap, so use three
    // linearly independent real vectors: 1, x, and a centered x^2 mode.
    if (target == 0)
    {
        return std::complex<double>(1.0 + 0.05 * x, 0.0);
    }
    if (target == 1)
    {
        return std::complex<double>(x, 0.0);
    }
    return std::complex<double>(x * x - 0.5, 0.0);
}

} // namespace

TEST(AllocPwMpiTest, ProjectorReductionsMatchGlobalReference)
{
    GlobalV::KPAR = 1;
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                PARAM.inp.bndpar,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);

    const int global_npw = 13;
    const int ntarget = 3;
    const int nband = 2;
    std::vector<int> local_ig;
    for (int ig = GlobalV::RANK_IN_POOL; ig < global_npw;
         ig += GlobalV::NPROC_IN_POOL)
    {
        local_ig.push_back(ig);
    }
    const int npw = static_cast<int>(local_ig.size());

    // Each rank owns a disjoint plane-wave slice, exactly like the real
    // projector tabulation passed to the reduction helpers.
    std::vector<std::complex<double>> tab(ntarget * npw);
    std::vector<std::complex<double>> psi(nband * npw);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            tab[a * npw + ig] = tab_value(a, local_ig[ig]);
        }
    }
    for (int band = 0; band < nband; ++band)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            psi[band * npw + ig] = psi_value(band, local_ig[ig]);
        }
    }

    std::vector<std::complex<double>> ST;
    alloc::build_pw_projector_overlap(tab.data(), npw, ntarget, ST);
    std::vector<std::vector<std::complex<double>>> becp;
    alloc::pw_becp_from_projectors(tab.data(), npw, ntarget, psi.data(), npw,
                                   nband, becp);

    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            std::complex<double> expected(0.0, 0.0);
            for (int ig = 0; ig < global_npw; ++ig)
            {
                expected += std::conj(tab_value(a, ig)) * tab_value(b, ig);
            }
            EXPECT_NEAR(std::abs(ST[a * ntarget + b] - expected), 0.0, 1e-12);
        }
    }
    for (int band = 0; band < nband; ++band)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            std::complex<double> expected(0.0, 0.0);
            for (int ig = 0; ig < global_npw; ++ig)
            {
                expected += std::conj(tab_value(a, ig))
                            * psi_value(band, ig);
            }
            EXPECT_NEAR(std::abs(becp[band][a] - expected), 0.0, 1e-12);
        }
    }
}

TEST(AllocPwMpiTest, OrthogonalPhiIsOrthonormalAcrossRanks)
{
    GlobalV::KPAR = 1;
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                PARAM.inp.bndpar,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);

    const int global_npw = 17;
    const int ntarget = 3;
    std::vector<int> local_ig;
    for (int ig = GlobalV::RANK_IN_POOL; ig < global_npw;
         ig += GlobalV::NPROC_IN_POOL)
    {
        local_ig.push_back(ig);
    }
    const int npw = static_cast<int>(local_ig.size());

    // Each rank owns the same disjoint plane-wave slice used by the real
    // target context; the overlap matrix is pooled back to every rank.
    std::vector<std::complex<double>> tab(ntarget * npw);
    std::vector<std::vector<std::complex<double>>> phi(
        ntarget, std::vector<std::complex<double>>(npw));
    for (int a = 0; a < ntarget; ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            tab[a * npw + ig] = phi_value(a, local_ig[ig]);
            phi[a][ig] = tab[a * npw + ig];
        }
    }

    std::vector<std::complex<double>> ST;
    alloc::build_pw_projector_overlap(tab.data(), npw, ntarget, ST);
    alloc::PwTargetSpace space;
    alloc::build_pw_target(ST, ntarget, {0, 0, 0}, {0, 0, 0}, 1, space);
    std::vector<std::vector<std::complex<double>>> phi_orth;
    alloc::build_pw_alloc_orthogonal_phi(space, phi, phi_orth);

    double max_dev = 0.0;
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            std::complex<double> dot(0.0, 0.0);
            for (int ig = 0; ig < npw; ++ig)
            {
                dot += std::conj(phi_orth[a][ig]) * phi_orth[b][ig];
            }
            Parallel_Reduce::reduce_pool(dot);
            const double expected = (a == b) ? 1.0 : 0.0;
            max_dev = std::max(max_dev, std::abs(dot.real() - expected));
            max_dev = std::max(max_dev, std::abs(dot.imag()));
        }
    }
    if (GlobalV::MY_RANK == 0)
    {
        printf("alloc_pw orthogonal phi across pool: max_dev=%.3e\n",
               max_dev);
    }
    EXPECT_LT(max_dev, 1e-13);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
