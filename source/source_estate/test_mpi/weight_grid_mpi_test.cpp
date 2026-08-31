#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_constraint/constraint_observe.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "../test/prepare_unitcell.h"
#undef private
#undef protected

// Mock symbols required by the cell_info object library in unit tests.
Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}

// H2O in a 10 Bohr cubic box, O at the center, H's mirrored about x=5 Bohr.
static std::unique_ptr<UnitCell> make_h2o_ucell()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0,
        {10, 0, 0, 0, 10, 0, 0, 0, 10},
        {"O", "H"}, {"O.upf", "H.upf"}, {"upf201", "upf201"}, {"", ""},
        {1, 2}, {16.0, 1.0}, "Cartesian",
        {5.0, 5.0, 5.0, 6.2, 5.0, 5.0, 3.8, 5.0, 5.0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

TEST(WeightGridMpiTest, DistributedMatchesSerial)
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                PARAM.input.bndpar,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);

    auto ucell = make_h2o_ucell();
    const std::vector<double> radii = {1.5, 0.5, 0.5};

    // Distributed density grid over the whole world pool (non-square mesh).
    ModulePW::PW_Basis rhopw;
    rhopw.initmpi(GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL, POOL_WORLD);
    rhopw.initgrids(1.0, ucell->latvec, 24, 16, 20);
    rhopw.distribute_r();

    constraint::WeightGrid wg(*ucell, &rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);

    // Rank 0 additionally builds the full serial reference grid.
    int nat = wg.nat();
    int nxyz = rhopw.nx * rhopw.ny * rhopw.nz;
    std::vector<double> ref(nat * nxyz, 0.0);
    if (GlobalV::MY_RANK == 0)
    {
        ModulePW::PW_Basis refpw;
        refpw.initmpi(1, 0, MPI_COMM_SELF);
        refpw.initgrids(1.0, ucell->latvec, rhopw.nx, rhopw.ny, rhopw.nz);
        refpw.distribute_r();
        constraint::WeightGrid wg_ref(*ucell, &refpw, radii, constraint::WeightType::Becke);
        wg_ref.build();
        const auto& cw = wg_ref.constraint_weights();
        for (int alpha = 0; alpha < nat; ++alpha)
        {
            for (int ir = 0; ir < nxyz; ++ir)
            {
                ref[alpha * nxyz + ir] = cw[alpha][ir];
            }
        }
    }
    MPI_Bcast(ref.data(), static_cast<int>(ref.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Every rank checks its local slice against the serial reference using
    // the global index gi = ixy * nz + iz_global.
    const auto& cw = wg.constraint_weights();
    const int nz = rhopw.nz;
    double max_diff = 0.0;
    for (int alpha = 0; alpha < nat; ++alpha)
    {
        for (int ir = 0; ir < rhopw.nrxx; ++ir)
        {
            const int ixy = ir / rhopw.nplane;
            const int iz = rhopw.startz_current + ir % rhopw.nplane;
            const int gi = ixy * nz + iz;
            max_diff = std::max(max_diff, std::abs(cw[alpha][ir] - ref[alpha * nxyz + gi]));
        }
    }
    EXPECT_LT(max_diff, 1e-12);
}

TEST(WeightGridMpiTest, ObserveReduceConsistent)
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                PARAM.input.bndpar,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);

    auto ucell = make_h2o_ucell();
    const std::vector<double> radii = {1.5, 0.5, 0.5};

    ModulePW::PW_Basis rhopw;
    rhopw.initmpi(GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL, POOL_WORLD);
    rhopw.initgrids(1.0, ucell->latvec, 24, 16, 20);
    rhopw.distribute_r();

    constraint::WeightGrid wg(*ucell, &rhopw, radii, constraint::WeightType::Becke);
    wg.build();

    // Constant density rho = 1: Q_alpha = int w_alpha dr, sum = cell volume.
    std::vector<double> rho(rhopw.nrxx, 1.0);
    const double* rho_ptr = rho.data();
    std::vector<double> Q;
    constraint::ConstraintObserver::observe(wg, &rho_ptr, 1, constraint::DensityChannel::Charge, Q);
    const double vol = ucell->omega;

    // reduce_pool is an allreduce: every rank must hold bit-identical Q.
    std::vector<double> Q_all(GlobalV::NPROC * Q.size(), 0.0);
    MPI_Allgather(Q.data(), static_cast<int>(Q.size()), MPI_DOUBLE,
                  Q_all.data(), static_cast<int>(Q.size()), MPI_DOUBLE,
                  MPI_COMM_WORLD);
    for (int r = 0; r < GlobalV::NPROC; ++r)
    {
        for (size_t alpha = 0; alpha < Q.size(); ++alpha)
        {
            EXPECT_DOUBLE_EQ(Q[alpha], Q_all[r * Q.size() + alpha]);
        }
    }
    double qsum = 0.0;
    for (const double q : Q)
    {
        qsum += q;
    }
    EXPECT_NEAR(qsum, vol, 1e-8);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
    int result = 0;
    testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
