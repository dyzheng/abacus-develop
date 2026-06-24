#include "gtest/gtest.h"
#include "source_base/vector3.h"
#include <complex>
#include <vector>
#include <cmath>

// Test SMO-anchored gauge fixing math:
// 1. After gauge, D_anchor is positive real at every k
// 2. Gauge phases are continuous (no pi jumps) along k-string
// 3. term1 = <psi|d_k alpha> * <alpha|psi> is gauge invariant

namespace {

struct SyntheticData {
    int nppstr;
    int nbands;
    int nat;
    int nproj;
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> D_I;
};

void compute_gauge(const SyntheticData& data,
                   std::vector<std::vector<std::complex<double>>>& gauge_phase,
                   std::vector<int>& anchor_iat,
                   std::vector<int>& anchor_lm)
{
    int nppstr = data.nppstr;
    int nbands = data.nbands;

    gauge_phase.resize(nppstr);
    for (int j = 0; j < nppstr; ++j)
        gauge_phase[j].resize(nbands, std::complex<double>(1.0, 0.0));

    anchor_iat.resize(nbands, -1);
    anchor_lm.resize(nbands, -1);

    for (int n = 0; n < nbands; ++n)
    {
        double max_proj = 0.0;
        for (int iat = 0; iat < data.nat; ++iat)
        {
            for (int lm = 0; lm < data.nproj; ++lm)
            {
                double proj = std::abs(data.D_I[0][iat][lm][n]);
                if (proj > max_proj)
                {
                    max_proj = proj;
                    anchor_iat[n] = iat;
                    anchor_lm[n] = lm;
                }
            }
        }
        if (anchor_iat[n] >= 0)
        {
            std::complex<double> D = data.D_I[0][anchor_iat[n]][anchor_lm[n]][n];
            double absD = std::abs(D);
            if (absD > 1e-15)
                gauge_phase[0][n] = std::conj(D) / absD;
        }
    }

    for (int j = 1; j < nppstr; ++j)
    {
        for (int n = 0; n < nbands; ++n)
        {
            if (anchor_iat[n] < 0) { gauge_phase[j][n] = gauge_phase[j-1][n]; continue; }
            std::complex<double> D = data.D_I[j][anchor_iat[n]][anchor_lm[n]][n];
            double absD = std::abs(D);
            std::complex<double> g(1.0, 0.0);
            if (absD > 1e-15)
                g = std::conj(D) / absD;
            std::complex<double> overlap = g * std::conj(gauge_phase[j-1][n]);
            if (overlap.real() < 0.0) g = -g;
            gauge_phase[j][n] = g;
        }
    }
}

} // anonymous namespace

class DeltaPGaugeTest : public testing::Test
{
protected:
    SyntheticData data_;

    void SetUp() override
    {
        data_.nppstr = 5;
        data_.nbands = 2;
        data_.nat = 2;
        data_.nproj = 2;

        data_.D_I.resize(data_.nppstr);
        for (int ik = 0; ik < data_.nppstr; ++ik)
        {
            data_.D_I[ik].resize(data_.nat);
            for (int iat = 0; iat < data_.nat; ++iat)
            {
                data_.D_I[ik][iat].resize(data_.nproj);
                for (int lm = 0; lm < data_.nproj; ++lm)
                {
                    data_.D_I[ik][iat][lm].resize(data_.nbands);
                }
            }
        }

        for (int ik = 0; ik < data_.nppstr; ++ik)
        {
            double k = 0.1 * ik;
            double phase = 0.3 * k + 0.1;
            double amp = 1.0 - 0.01 * ik;
            data_.D_I[ik][0][0][0] = std::polar(amp, phase);
            data_.D_I[ik][0][1][0] = std::polar(0.1, phase + 0.5);
            data_.D_I[ik][1][0][0] = std::polar(0.05, phase + 1.0);
            data_.D_I[ik][1][1][0] = std::polar(0.02, phase + 1.5);
        }

        for (int ik = 0; ik < data_.nppstr; ++ik)
        {
            double k = 0.1 * ik;
            double phase = -0.2 * k + 0.5;
            double amp = 0.8 + 0.01 * ik;
            data_.D_I[ik][1][1][1] = std::polar(amp, phase);
            data_.D_I[ik][1][0][1] = std::polar(0.1, phase + 0.3);
            data_.D_I[ik][0][1][1] = std::polar(0.05, phase + 0.7);
            data_.D_I[ik][0][0][1] = std::polar(0.02, phase + 1.1);
        }
    }
};

TEST_F(DeltaPGaugeTest, AnchorIsMaxProjection)
{
    std::vector<std::vector<std::complex<double>>> gauge;
    std::vector<int> anchor_iat, anchor_lm;
    compute_gauge(data_, gauge, anchor_iat, anchor_lm);

    EXPECT_EQ(anchor_iat[0], 0);
    EXPECT_EQ(anchor_lm[0], 0);
    EXPECT_EQ(anchor_iat[1], 1);
    EXPECT_EQ(anchor_lm[1], 1);
}

TEST_F(DeltaPGaugeTest, AnchorProjectionIsPositiveReal)
{
    std::vector<std::vector<std::complex<double>>> gauge;
    std::vector<int> anchor_iat, anchor_lm;
    compute_gauge(data_, gauge, anchor_iat, anchor_lm);

    for (int n = 0; n < data_.nbands; ++n)
    {
        for (int ik = 0; ik < data_.nppstr; ++ik)
        {
            std::complex<double> D = data_.D_I[ik][anchor_iat[n]][anchor_lm[n]][n];
            std::complex<double> D_gauge = D * gauge[ik][n];
            EXPECT_NEAR(D_gauge.imag(), 0.0, 1e-12)
                << "n=" << n << " ik=" << ik;
            EXPECT_GT(D_gauge.real(), 0.0)
                << "n=" << n << " ik=" << ik;
        }
    }
}

TEST_F(DeltaPGaugeTest, PhaseContinuityNoPiJumps)
{
    std::vector<std::vector<std::complex<double>>> gauge;
    std::vector<int> anchor_iat, anchor_lm;
    compute_gauge(data_, gauge, anchor_iat, anchor_lm);

    for (int n = 0; n < data_.nbands; ++n)
    {
        for (int ik = 1; ik < data_.nppstr; ++ik)
        {
            std::complex<double> overlap = gauge[ik][n] * std::conj(gauge[ik-1][n]);
            EXPECT_GT(overlap.real(), 0.0)
                << "Phase jump at n=" << n << " ik=" << ik;
        }
    }
}

TEST_F(DeltaPGaugeTest, Term1GaugeInvariance)
{
    std::complex<double> C(0.7, 0.3);
    std::complex<double> dS(0.5, -0.1);
    std::complex<double> S(0.4, 0.2);
    std::complex<double> D_I = std::conj(S) * C;

    std::complex<double> bra_grad = std::conj(C) * dS;
    std::complex<double> term1 = bra_grad * D_I;

    double phi = 0.37;
    std::complex<double> g(std::cos(phi), std::sin(phi));
    std::complex<double> C_g = C * g;
    std::complex<double> D_I_g = D_I * g;

    std::complex<double> bra_grad_g = std::conj(C_g) * dS;
    std::complex<double> term1_g = bra_grad_g * D_I_g;

    EXPECT_NEAR((term1 - term1_g).real(), 0.0, 1e-12);
    EXPECT_NEAR((term1 - term1_g).imag(), 0.0, 1e-12);
}
