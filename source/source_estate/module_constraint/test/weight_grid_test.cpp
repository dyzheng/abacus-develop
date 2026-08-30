#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <memory>

#define private public
#define protected public
#include "source_base/global_variable.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "../../test/prepare_unitcell.h"
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

// Local GlobalV defaults for a serial PW unit test.
static void Set_GlobalV_Default()
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 10.0;
    PARAM.input.basis_type = "pw";
    GlobalV::KPAR = 1;
    GlobalV::NPROC_IN_POOL = 1;
}

// H2O in a 10 Bohr cubic box, O at the center, H's mirrored about x=5 Bohr.
static std::unique_ptr<UnitCell> make_h2o_ucell()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0, // lat0 in Bohr; tau (Bohr) == Cartesian coordinates
        {10, 0, 0, 0, 10, 0, 0, 0, 10}, // 10 Bohr box
        {"O", "H"}, {"O.upf", "H.upf"}, {"upf201", "upf201"}, {"", ""},
        {1, 2}, {16.0, 1.0}, "Cartesian",
        {5.0, 5.0, 5.0, 6.2, 5.0, 5.0, 3.8, 5.0, 5.0}, // O, H1, H2 (Bohr)
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

class WeightGridTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<double> radii;

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell();
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 20, 20, 20); // 0.5 Bohr spacing
        rhopw->distribute_r();
        radii = {1.5, 0.5, 0.5}; // O, H1, H2 (Bohr)
    }

    void TearDown() override
    {
        delete rhopw;
    }

    // Global (serial) index of the mirror image of ir under x -> L - x.
    int mirror_index(const int ir) const
    {
        const int nz = rhopw->nz;
        const int ny = rhopw->ny;
        const int nx = rhopw->nx;
        const int i = ir / (ny * nz);
        const int j = ir / nz - i * ny;
        const int k = ir % nz;
        const int i2 = (nx - i) % nx;
        return (i2 * ny + j) * nz + k;
    }
};

TEST_F(WeightGridTest, PartitionOfUnity)
{
    using iclock = std::chrono::high_resolution_clock;
    const iclock::time_point t0 = iclock::now();

    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    const iclock::duration dur = iclock::now() - t0;

    // Per-point partition of unity on the local grid.
    double local_max = 0.0;
    const auto& cw = wg.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        double sum = 0.0;
        for (int alpha = 0; alpha < wg.nconstraint(); ++alpha)
        {
            sum += cw[alpha][ir];
        }
        local_max = std::max(local_max, std::abs(sum - 1.0));
    }
    EXPECT_LT(local_max, 1e-10);
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);

    printf("weight_grid: ngrid=%d nat=%d build_time=%.3e s (per point %.2e s)\n",
           rhopw->nrxx, wg.nat(),
           std::chrono::duration<double>(dur).count(),
           std::chrono::duration<double>(dur).count() / rhopw->nrxx);
}

TEST_F(WeightGridTest, SymmetryMirror)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();

    // Mirror reflection x -> L - x maps H1 <-> H2 and O -> O, so per-atom
    // weights at mirrored grid points must agree pointwise.
    const auto& w = wg.constraint_weights();
    double max_h = 0.0, max_o = 0.0;
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        const int ir2 = mirror_index(ir);
        max_h = std::max(max_h, std::abs(w[1][ir] - w[2][ir2]));
        max_o = std::max(max_o, std::abs(w[0][ir] - w[0][ir2]));
    }
    EXPECT_LT(max_h, 1e-12);
    EXPECT_LT(max_o, 1e-12);
}

TEST_F(WeightGridTest, DeterministicRebuild)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    const auto snapshot = wg.constraint_weights();
    wg.build(); // rebuild on the same geometry -> bit-identical
    const auto& again = wg.constraint_weights();
    for (int alpha = 0; alpha < wg.nconstraint(); ++alpha)
    {
        EXPECT_TRUE(snapshot[alpha] == again[alpha]);
    }
}

TEST_F(WeightGridTest, ScreeningKeepsPartition)
{
    // Aggressive cutoff (3 Bohr) screens far atoms at many grid points; the
    // partition of unity over the involved set must still hold exactly.
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke, 3.0);
    wg.build();
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);

    // At points where every atom is within the cutoff the result must equal
    // the unscreened weight (screening is a pure truncation of the neighbor
    // list, not a different recipe).
    constraint::WeightGrid full(*ucell, rhopw, radii, constraint::WeightType::Becke);
    full.build();
    double max_diff = 0.0;
    const auto& w_sc = wg.constraint_weights();
    const auto& w_full = full.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        max_diff = std::max(max_diff, std::abs(w_sc[0][ir] - w_full[0][ir]));
    }
    EXPECT_GT(max_diff, 0.0); // screening does change far-away points
    EXPECT_LT(wg.max_partition_deviation(), 1e-10);
}

TEST_F(WeightGridTest, FragmentConstraint)
{
    constraint::WeightGrid wg(*ucell, rhopw, radii, constraint::WeightType::Becke);
    wg.build();
    // Fragment constraint {O, H1}: weight = w_O + w_H1 pointwise.
    wg.set_constraint_atoms({{0, 1}});
    ASSERT_EQ(wg.nconstraint(), 1);
    const auto& cw = wg.constraint_weights();
    // Reference: per-atom weights on the default constraint map.
    constraint::WeightGrid ref(*ucell, rhopw, radii, constraint::WeightType::Becke);
    ref.build();
    const auto& wref = ref.constraint_weights();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(cw[0][ir], wref[0][ir] + wref[1][ir]);
    }
}
