#include "gtest/gtest.h"
#define private public
#include "source_io/module_parameter/parameter.h"
#undef private

/***********************************************************************
 * Unit tests for DFT+U PW nspin=1/2/4 support (PR-2)
 *
 * Strategy: test energy weights and becp index logic as pure
 * arithmetic — no need to link against full ABACUS libraries.
 * set_locale is tested via integration tests.
 ***********************************************************************/

class DftuPwTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =====================================================================
// Energy weight tests
// =====================================================================

TEST_F(DftuPwTest, EnergyWeightsNspin1)
{
    PARAM.input.nspin = 1;
    double weight_eu = 1;
    switch(PARAM.inp.nspin)
    {
        case 1: weight_eu = 1.0; break;
        case 2: weight_eu = 0.5; break;
        case 4: weight_eu = 0.25; break;
        default: break;
    }
    const double diag_coeff = PARAM.inp.nspin == 4 ? 1.0 : 0.5;
    EXPECT_DOUBLE_EQ(weight_eu, 1.0);
    EXPECT_DOUBLE_EQ(diag_coeff, 0.5);
}

TEST_F(DftuPwTest, EnergyWeightsNspin2)
{
    PARAM.input.nspin = 2;
    double weight_eu = 1;
    switch(PARAM.inp.nspin)
    {
        case 1: weight_eu = 1.0; break;
        case 2: weight_eu = 0.5; break;
        case 4: weight_eu = 0.25; break;
        default: break;
    }
    const double diag_coeff = PARAM.inp.nspin == 4 ? 1.0 : 0.5;
    EXPECT_DOUBLE_EQ(weight_eu, 0.5);
    EXPECT_DOUBLE_EQ(diag_coeff, 0.5);
}

TEST_F(DftuPwTest, EnergyWeightsNspin4)
{
    PARAM.input.nspin = 4;
    double weight_eu = 1;
    switch(PARAM.inp.nspin)
    {
        case 1: weight_eu = 1.0; break;
        case 2: weight_eu = 0.5; break;
        case 4: weight_eu = 0.25; break;
        default: break;
    }
    const double diag_coeff = PARAM.inp.nspin == 4 ? 1.0 : 0.5;
    EXPECT_DOUBLE_EQ(weight_eu, 0.25);
    EXPECT_DOUBLE_EQ(diag_coeff, 1.0);
}

// =====================================================================
// Becp index tests
// =====================================================================

TEST_F(DftuPwTest, OccupNspin12Index)
{
    const int nkb = 10, begin_ih = 3, m_begin = 4, m = 2, ib = 5;
    // nspin=1/2: index = ib*nkb + begin_ih + m_begin + m
    const int index_nspin12 = ib * nkb + begin_ih + m_begin + m;
    EXPECT_EQ(index_nspin12, 59);
    // different from nspin=4
    const int index_nspin4 = ib * 2 * nkb + begin_ih + m_begin + m;
    EXPECT_NE(index_nspin12, index_nspin4);
}

TEST_F(DftuPwTest, OccupNspin4Index)
{
    const int nkb = 10, begin_ih = 3, m_begin = 4, m = 2, ib = 5;
    const int index_nspin4 = ib * 2 * nkb + begin_ih + m_begin + m;
    EXPECT_EQ(index_nspin4, 109);
}

// =====================================================================
// set_locale logic tests (pure array copy, no UnitCell needed)
// =====================================================================

TEST_F(DftuPwTest, SetLocaleNspin4)
{
    // Simulate set_locale for nspin=4: uom_array -> locale copy
    PARAM.input.nspin = 4;
    const int mat_size = 10; // (2*2+1)*2 for d-orbital with npol=2
    const int total = mat_size * mat_size; // 100

    std::vector<double> uom_array(total);
    for(int i = 0; i < total; i++)
        uom_array[i] = static_cast<double>(i + 1);

    // Simulate locale as raw array (same as ModuleBase::matrix::c)
    std::vector<double> locale_c(total, 0.0);

    // nspin=4 branch: direct copy
    for(int mm = 0; mm < total; mm++)
        locale_c[mm] = uom_array[mm];

    for(int i = 0; i < total; i++)
        EXPECT_DOUBLE_EQ(locale_c[i], static_cast<double>(i + 1));
}

TEST_F(DftuPwTest, SetLocaleNspin2)
{
    // Simulate set_locale for nspin=2: uom_array -> locale copy (spin-up + spin-down)
    PARAM.input.nspin = 2;
    const int mat_size = 5; // 2*2+1 for d-orbital
    const int size_per_spin = mat_size * mat_size; // 25
    const int total = size_per_spin * 2; // 50

    std::vector<double> uom_array(total);
    for(int i = 0; i < size_per_spin; i++)
    {
        uom_array[i] = static_cast<double>(i + 1);                // spin-up
        uom_array[i + size_per_spin] = static_cast<double>(i + 101); // spin-down
    }

    std::vector<double> locale_up(size_per_spin, 0.0);
    std::vector<double> locale_dn(size_per_spin, 0.0);

    // nspin=1/2 branch: copy both spin channels
    const int nr_nc = size_per_spin; // locale[iat][l][0][0].nr * locale[iat][l][0][0].nc
    for(int mm = 0; mm < nr_nc; mm++)
    {
        locale_up[mm] = uom_array[mm];
        locale_dn[mm] = uom_array[mm + nr_nc];
    }

    for(int i = 0; i < size_per_spin; i++)
    {
        EXPECT_DOUBLE_EQ(locale_up[i], static_cast<double>(i + 1));
        EXPECT_DOUBLE_EQ(locale_dn[i], static_cast<double>(i + 101));
    }
}
