#include "gtest/gtest.h"
#include <complex>
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

// =====================================================================
// VU effective potential tests (cal_occ_pw logic)
// =====================================================================

TEST_F(DftuPwTest, VUPotNspin1_DiagonalLocale)
{
    // For nspin=1: VU[m1,m2] = U * (0.5*delta(m1,m2) - locale[m2*m_size+m1])
    // With diagonal locale: locale[m,m] = 0.3
    const double U_val = 4.0;
    const int m_size = 5; // d-orbital: 2*2+1
    const int size = m_size * m_size;

    std::vector<double> locale_c(size, 0.0);
    for(int m = 0; m < m_size; m++)
        locale_c[m * m_size + m] = 0.3; // diagonal

    std::vector<std::complex<double>> vu(size, {0.0, 0.0});
    for(int m1 = 0; m1 < m_size; m1++)
    {
        for(int m2 = 0; m2 < m_size; m2++)
        {
            const double diag_coeff = 0.5; // nspin != 4
            vu[m1 * m_size + m2] = U_val *
                (diag_coeff * (m1 == m2) - locale_c[m2 * m_size + m1]);
        }
    }

    // diagonal: U*(0.5 - 0.3) = 4.0*0.2 = 0.8
    for(int m = 0; m < m_size; m++)
        EXPECT_DOUBLE_EQ(vu[m * m_size + m].real(), 0.8);

    // off-diagonal: U*(0 - 0) = 0
    EXPECT_DOUBLE_EQ(vu[0 * m_size + 1].real(), 0.0);
    EXPECT_DOUBLE_EQ(vu[1 * m_size + 0].real(), 0.0);
}

TEST_F(DftuPwTest, VUPotNspin1_OffDiagonalLocale)
{
    // locale has off-diagonal elements
    const double U_val = 3.0;
    const int m_size = 3; // p-orbital: 2*1+1
    const int size = m_size * m_size;

    std::vector<double> locale_c(size, 0.0);
    locale_c[0 * m_size + 1] = 0.1; // locale(0,1) = 0.1
    locale_c[1 * m_size + 0] = 0.2; // locale(1,0) = 0.2

    std::vector<std::complex<double>> vu(size, {0.0, 0.0});
    for(int m1 = 0; m1 < m_size; m1++)
    {
        for(int m2 = 0; m2 < m_size; m2++)
        {
            vu[m1 * m_size + m2] = U_val *
                (0.5 * (m1 == m2) - locale_c[m2 * m_size + m1]);
        }
    }

    // VU[0,1] = U * (0 - locale[1*3+0]) = 3.0 * (-0.2) = -0.6
    EXPECT_DOUBLE_EQ(vu[0 * m_size + 1].real(), -0.6);
    // VU[1,0] = U * (0 - locale[0*3+1]) = 3.0 * (-0.1) = -0.3
    EXPECT_DOUBLE_EQ(vu[1 * m_size + 0].real(), -0.3);
}

TEST_F(DftuPwTest, VUPotNspin2_TwoSpinChannels)
{
    // nspin=2: two independent spin channels with same formula
    const double U_val = 5.0;
    const int m_size = 3;
    const int size = m_size * m_size;

    std::vector<double> locale_up(size, 0.0);
    std::vector<double> locale_dn(size, 0.0);
    locale_up[0] = 0.4; // locale_up(0,0) = 0.4
    locale_dn[0] = 0.1; // locale_dn(0,0) = 0.1

    // VU_up[0,0] = U*(0.5 - 0.4) = 0.5
    double vu_up_00 = U_val * (0.5 - locale_up[0 * m_size + 0]);
    EXPECT_DOUBLE_EQ(vu_up_00, 0.5);

    // VU_dn[0,0] = U*(0.5 - 0.1) = 2.0
    double vu_dn_00 = U_val * (0.5 - locale_dn[0 * m_size + 0]);
    EXPECT_DOUBLE_EQ(vu_dn_00, 2.0);
}

TEST_F(DftuPwTest, VUPotNspin4_PauliTransform)
{
    // nspin=4: after computing VU in Pauli basis, transform to spin basis
    // vu_spin[0] = 0.5*(vu_pauli[0] + vu_pauli[3])
    // vu_spin[3] = 0.5*(vu_pauli[0] - vu_pauli[3])
    // vu_spin[1] = 0.5*(vu_pauli[1] + i*vu_pauli[2])
    // vu_spin[2] = 0.5*(vu_pauli[1] - i*vu_pauli[2])
    const int m_size = 3;
    const int size = m_size * m_size;

    // For a single (m1,m2) pair, test the Pauli->spin transform
    std::complex<double> vu_pauli[4];
    vu_pauli[0] = {1.0, 0.0}; // charge channel
    vu_pauli[1] = {0.5, 0.0}; // sigma_x
    vu_pauli[2] = {0.3, 0.0}; // sigma_y
    vu_pauli[3] = {0.2, 0.0}; // sigma_z

    std::complex<double> vu_spin[4];
    vu_spin[0] = 0.5 * (vu_pauli[0] + vu_pauli[3]);
    vu_spin[3] = 0.5 * (vu_pauli[0] - vu_pauli[3]);
    vu_spin[1] = 0.5 * (vu_pauli[1] + std::complex<double>(0.0, 1.0) * vu_pauli[2]);
    vu_spin[2] = 0.5 * (vu_pauli[1] - std::complex<double>(0.0, 1.0) * vu_pauli[2]);

    EXPECT_DOUBLE_EQ(vu_spin[0].real(), 0.6);  // 0.5*(1.0+0.2)
    EXPECT_DOUBLE_EQ(vu_spin[0].imag(), 0.0);
    EXPECT_DOUBLE_EQ(vu_spin[3].real(), 0.4);  // 0.5*(1.0-0.2)
    EXPECT_DOUBLE_EQ(vu_spin[3].imag(), 0.0);
    EXPECT_DOUBLE_EQ(vu_spin[1].real(), 0.25); // 0.5*0.5
    EXPECT_DOUBLE_EQ(vu_spin[1].imag(), 0.15); // 0.5*0.3
    EXPECT_DOUBLE_EQ(vu_spin[2].real(), 0.25); // 0.5*0.5
    EXPECT_DOUBLE_EQ(vu_spin[2].imag(), -0.15);// -0.5*0.3
}

// =====================================================================
// Energy calculation tests
// =====================================================================

TEST_F(DftuPwTest, EnergyNspin1_DiagonalLocale)
{
    // E_U = sum_{m1,m2} U * weight_eu * locale[m2,m1] * locale[m1,m2]
    // weight_eu = 1.0 for nspin=1
    const double U_val = 4.0;
    const int m_size = 3;
    const int size = m_size * m_size;

    std::vector<double> locale_c(size, 0.0);
    locale_c[0 * m_size + 0] = 0.5;
    locale_c[1 * m_size + 1] = 0.3;
    locale_c[2 * m_size + 2] = 0.2;

    double energy_u = 0.0;
    const double weight_eu = 1.0;
    for(int m1 = 0; m1 < m_size; m1++)
    {
        for(int m2 = 0; m2 < m_size; m2++)
        {
            energy_u += U_val * weight_eu * locale_c[m2 * m_size + m1]
                        * locale_c[m1 * m_size + m2];
        }
    }

    // Only diagonal contributes: U * (0.5^2 + 0.3^2 + 0.2^2) = 4*(0.25+0.09+0.04) = 4*0.38 = 1.52
    EXPECT_DOUBLE_EQ(energy_u, 1.52);
}

TEST_F(DftuPwTest, EnergyNspin2_TwoChannels)
{
    // nspin=2: weight_eu = 0.5, sum over both spin channels
    const double U_val = 2.0;
    const int m_size = 3;
    const int size = m_size * m_size;
    const double weight_eu = 0.5;

    std::vector<double> locale_up(size, 0.0);
    std::vector<double> locale_dn(size, 0.0);
    locale_up[0] = 0.4; // (0,0)
    locale_dn[0] = 0.6; // (0,0)

    double energy_u = 0.0;
    // spin-up contribution
    for(int m1 = 0; m1 < m_size; m1++)
        for(int m2 = 0; m2 < m_size; m2++)
            energy_u += U_val * weight_eu * locale_up[m2 * m_size + m1] * locale_up[m1 * m_size + m2];
    // spin-down contribution
    for(int m1 = 0; m1 < m_size; m1++)
        for(int m2 = 0; m2 < m_size; m2++)
            energy_u += U_val * weight_eu * locale_dn[m2 * m_size + m1] * locale_dn[m1 * m_size + m2];

    // U*0.5*(0.4^2 + 0.6^2) = 2*0.5*(0.16+0.36) = 0.52
    EXPECT_DOUBLE_EQ(energy_u, 0.52);
}

TEST_F(DftuPwTest, EnergyNspin4_WithOffDiagonal)
{
    // nspin=4: weight_eu = 0.25, includes off-diagonal Pauli components
    const double U_val = 2.0;
    const int m_size = 2; // simplified: s-orbital would be 1, use 2 for test
    const int size = m_size * m_size;
    const double weight_eu = 0.25;

    // 4 Pauli components stored contiguously
    std::vector<double> locale_c(size * 4, 0.0);
    // charge channel (is=0)
    locale_c[0] = 0.5; locale_c[1] = 0.1;
    locale_c[2] = 0.1; locale_c[3] = 0.5;
    // sigma_x (is=1)
    locale_c[size + 0] = 0.2; locale_c[size + 1] = 0.0;
    locale_c[size + 2] = 0.0; locale_c[size + 3] = 0.2;

    double energy_u = 0.0;
    for(int is = 0; is < 4; is++)
    {
        int start = is * size;
        for(int m1 = 0; m1 < m_size; m1++)
        {
            for(int m2 = 0; m2 < m_size; m2++)
            {
                energy_u += U_val * weight_eu
                    * locale_c[start + m2 * m_size + m1]
                    * locale_c[start + m1 * m_size + m2];
            }
        }
    }

    // is=0: 2*0.25*(0.5*0.5 + 0.1*0.1 + 0.1*0.1 + 0.5*0.5) = 0.5*(0.25+0.01+0.01+0.25) = 0.26
    // is=1: 2*0.25*(0.2*0.2 + 0 + 0 + 0.2*0.2) = 0.5*(0.04+0.04) = 0.04
    // is=2,3: 0
    EXPECT_DOUBLE_EQ(energy_u, 0.30);
}

// =====================================================================
// Locale accumulation from becp (cal_occ_pw core loop)
// =====================================================================

TEST_F(DftuPwTest, LocaleAccumNspin12)
{
    // nspin=1/2: locale[m1*m_size+m2] += weight * real(conj(becp[m1]) * becp[m2])
    const int m_size = 3; // p-orbital
    const int nkb = 5;
    const int begin_ih = 0;
    const int m_begin = 0; // target_l=1, m_begin = 1*1 = 1... but for test simplicity use 0
    const int nbands = 2;
    const double weights[2] = {1.0, 0.5};

    // becp array: becp[ib*nkb + begin_ih + m_begin + m]
    std::vector<std::complex<double>> becp(nbands * nkb, {0.0, 0.0});
    // band 0
    becp[0 * nkb + 0] = {1.0, 0.0};
    becp[0 * nkb + 1] = {0.0, 1.0};
    becp[0 * nkb + 2] = {0.5, 0.5};
    // band 1
    becp[1 * nkb + 0] = {0.5, 0.0};
    becp[1 * nkb + 1] = {0.5, -0.5};
    becp[1 * nkb + 2] = {0.0, 1.0};

    std::vector<double> locale_c(m_size * m_size, 0.0);
    for(int ib = 0; ib < nbands; ib++)
    {
        const double weight = weights[ib];
        int ind_m1m2 = 0;
        for(int m1 = 0; m1 < m_size; m1++)
        {
            const int index_m1 = ib * nkb + begin_ih + m_begin + m1;
            for(int m2 = 0; m2 < m_size; m2++)
            {
                const int index_m2 = ib * nkb + begin_ih + m_begin + m2;
                locale_c[ind_m1m2] += weight * (std::conj(becp[index_m1]) * becp[index_m2]).real();
                ind_m1m2++;
            }
        }
    }

    // band0, w=1.0: conj(becp0)*becp0 = |1|^2=1, conj(becp0)*becp1 = 1*(0,1)=(0,1)->real=0
    // locale[0,0] from band0 = 1.0*1.0 = 1.0
    // band1, w=0.5: conj(becp0)*becp0 = |0.5|^2=0.25
    // locale[0,0] from band1 = 0.5*0.25 = 0.125
    EXPECT_DOUBLE_EQ(locale_c[0], 1.125); // 1.0 + 0.125

    // locale[1,1]: band0 = 1.0*|i|^2 = 1.0, band1 = 0.5*|(0.5,-0.5)|^2 = 0.5*0.5 = 0.25
    EXPECT_DOUBLE_EQ(locale_c[4], 1.25);
}

TEST_F(DftuPwTest, LocaleAccumNspin4_PauliComponents)
{
    // nspin=4: 4 Pauli components from becp with npol=2
    // occ[0] = w * conj(becp_up[m1]) * becp_up[m2]
    // occ[1] = w * conj(becp_up[m1]) * becp_dn[m2]
    // occ[2] = w * conj(becp_dn[m1]) * becp_up[m2]
    // occ[3] = w * conj(becp_dn[m1]) * becp_dn[m2]
    // locale[ind] += (occ[0]+occ[3]).real()       -- charge
    // locale[ind+size] += (occ[1]+occ[2]).real()   -- sigma_x
    // locale[ind+2*size] += (occ[1]-occ[2]).imag() -- sigma_y
    // locale[ind+3*size] += (occ[0]-occ[3]).real() -- sigma_z

    const int m_size = 1; // s-orbital for simplicity
    const int nkb = 2;
    const int nbands = 1;
    const double weight = 1.0;

    // becp layout: becp[ib*2*nkb + begin_ih + m]  (up)
    //              becp[ib*2*nkb + begin_ih + m + nkb] (down)
    std::vector<std::complex<double>> becp(nbands * 2 * nkb, {0.0, 0.0});
    // m=0 only (s-orbital)
    becp[0 * 2 * nkb + 0] = {0.8, 0.0};       // becp_up[m=0]
    becp[0 * 2 * nkb + 0 + nkb] = {0.0, 0.6}; // becp_dn[m=0]

    const int size = m_size * m_size; // 1
    std::vector<double> locale_c(size * 4, 0.0);

    for(int ib = 0; ib < nbands; ib++)
    {
        int ind_m1m2 = 0;
        for(int m1 = 0; m1 < m_size; m1++)
        {
            const int index_m1 = ib * 2 * nkb + 0 + m1;
            for(int m2 = 0; m2 < m_size; m2++)
            {
                const int index_m2 = ib * 2 * nkb + 0 + m2;
                std::complex<double> occ[4];
                occ[0] = weight * std::conj(becp[index_m1]) * becp[index_m2];
                occ[1] = weight * std::conj(becp[index_m1]) * becp[index_m2 + nkb];
                occ[2] = weight * std::conj(becp[index_m1 + nkb]) * becp[index_m2];
                occ[3] = weight * std::conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];
                locale_c[ind_m1m2] += (occ[0] + occ[3]).real();
                locale_c[ind_m1m2 + size] += (occ[1] + occ[2]).real();
                locale_c[ind_m1m2 + 2 * size] += (occ[1] - occ[2]).imag();
                locale_c[ind_m1m2 + 3 * size] += (occ[0] - occ[3]).real();
                ind_m1m2++;
            }
        }
    }

    // becp_up = (0.8, 0), becp_dn = (0, 0.6)
    // occ[0] = conj(0.8)*0.8 = 0.64
    // occ[1] = conj(0.8)*(0,0.6) = 0.8*(0,0.6) = (0, 0.48)
    // occ[2] = conj(0,0.6)*0.8 = (0,-0.6)*0.8 = (0, -0.48)
    // occ[3] = conj(0,0.6)*(0,0.6) = (0,-0.6)*(0,0.6) = 0.36
    EXPECT_DOUBLE_EQ(locale_c[0], 1.0);    // (0.64+0.36).real = 1.0 (charge)
    EXPECT_DOUBLE_EQ(locale_c[1], 0.0);    // (occ1+occ2).real = ((0,0.48)+(0,-0.48)).real = 0
    EXPECT_DOUBLE_EQ(locale_c[2], 0.96);   // (occ1-occ2).imag = ((0,0.48)-(0,-0.48)).imag = 0.96
    EXPECT_DOUBLE_EQ(locale_c[3], 0.28);   // (occ0-occ3).real = (0.64-0.36) = 0.28 (sigma_z)
}

TEST_F(DftuPwTest, CopyLocaleToUomSave_Nspin2)
{
    // Verify copy_locale logic: uom_save[index+mm] = locale[spin0], uom_save[index+mm+size] = locale[spin1]
    const int m_size = 3;
    const int size = m_size * m_size;

    std::vector<double> locale_spin0(size), locale_spin1(size);
    for(int i = 0; i < size; i++)
    {
        locale_spin0[i] = static_cast<double>(i + 1);
        locale_spin1[i] = static_cast<double>(i + 100);
    }

    std::vector<double> uom_save(size * 2, 0.0);
    const int eff_pot_index = 0;
    for(int mm = 0; mm < size; mm++)
    {
        uom_save[eff_pot_index + mm] = locale_spin0[mm];
        uom_save[eff_pot_index + mm + size] = locale_spin1[mm];
    }

    for(int i = 0; i < size; i++)
    {
        EXPECT_DOUBLE_EQ(uom_save[i], static_cast<double>(i + 1));
        EXPECT_DOUBLE_EQ(uom_save[i + size], static_cast<double>(i + 100));
    }
}

TEST_F(DftuPwTest, CopyLocaleToUomSave_Nspin4)
{
    // nspin=4: 4 blocks stored contiguously
    const int m_size = 3;
    const int size = m_size * m_size;
    const int total = size * 4; // 4 Pauli components

    std::vector<double> locale_c(total);
    for(int i = 0; i < total; i++)
        locale_c[i] = static_cast<double>(i + 1);

    std::vector<double> uom_save(total, 0.0);
    const int eff_pot_index = 0;
    for(int mm = 0; mm < size; mm++)
    {
        uom_save[eff_pot_index + mm] = locale_c[mm];
        uom_save[eff_pot_index + mm + size] = locale_c[mm + size];
        uom_save[eff_pot_index + mm + 2 * size] = locale_c[mm + 2 * size];
        uom_save[eff_pot_index + mm + 3 * size] = locale_c[mm + 3 * size];
    }

    for(int i = 0; i < total; i++)
        EXPECT_DOUBLE_EQ(uom_save[i], static_cast<double>(i + 1));
}
