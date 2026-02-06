#include <gtest/gtest.h>
#include <complex>
#include <cmath>

#define private public
#define protected public
#include "source_io/cal_r_overlap_R.h"
#include "source_io/module_parameter/parameter.h"
#undef protected
#undef private

#include "source_base/global_variable.h"

#ifdef __MPI
#include <mpi.h>
#endif

// Stubs for symbols pulled in by UnitCell and related headers
Atom_pseudo::Atom_pseudo() {}
Atom_pseudo::~Atom_pseudo() {}
#ifdef __MPI
void Atom_pseudo::bcast_atom_pseudo() {}
#endif
pseudo::pseudo() {}
pseudo::~pseudo() {}

Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}
#ifdef __LCAO
InfoNonlocal::InfoNonlocal() {}
InfoNonlocal::~InfoNonlocal() {}
#endif
void output::printM3(std::ofstream& ofs, const std::string& description, const ModuleBase::Matrix3& m) {}

#define DOUBLETHRESHOLD 1e-8

class CalROverlapRLTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Set global parameters needed by cal_r_overlap_R::init
        PARAM.input.cal_force = false;
        PARAM.sys.nlocal = 13; // Si: 2s2p1d = 2*1 + 2*3 + 1*5 = 13

        // Set up LCAO_Orbitals by reading a real orbital file
        orb_.read_in_flag = true;
        orb_.orbital_file.push_back("../../../../tests/PP_ORB/Si_gga_8au_100Ry_2s2p1d.orb");
        orb_.ecutwfc = 100.0;
        orb_.dk = 0.01;
        orb_.dR = 0.01;
        orb_.Rmax = 20.0;

        std::ofstream ofs_log("cal_r_overlap_R_L_test.log");
        orb_.Read_Orbitals(ofs_log, 1, 2, false, 0, false, 0);
        ofs_log.close();

        // Set up a minimal UnitCell with one Si atom
        ucell_.ntype = 1;
        ucell_.atoms = new Atom[1];
        ucell_.set_atom_flag = true;
        ucell_.atoms[0].na = 1;
        ucell_.atoms[0].tau.resize(1, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
        ucell_.atoms[0].nwl = orb_.Phi[0].getLmax();
        ucell_.atoms[0].l_nchi.resize(ucell_.atoms[0].nwl + 1);
        for (int L = 0; L <= ucell_.atoms[0].nwl; L++)
        {
            ucell_.atoms[0].l_nchi[L] = orb_.Phi[0].getNchi(L);
        }
        ucell_.lat0 = 1.0;

        // Set up a trivial Parallel_Orbitals (serial mode)
        pv_.set_serial(PARAM.globalv.nlocal, PARAM.globalv.nlocal);

        // Initialize cal_r_overlap_R
        rR_.init(ucell_, pv_, orb_);
    }

    void TearDown() override
    {
        std::remove("cal_r_overlap_R_L_test.log");
    }

    LCAO_Orbitals orb_;
    UnitCell ucell_;
    Parallel_Orbitals pv_;
    cal_r_overlap_R rR_;
};

// Test 1: <s|L|s> = 0 for on-site s-orbital pair
TEST_F(CalROverlapRLTest, SsOnSiteIsZero)
{
    ModuleBase::Vector3<double> R(0.0, 0.0, 0.0);
    // s-orbital: T=0, L=0, m=0, N=0 (first s) and N=1 (second s)
    for (int N1 = 0; N1 < 2; N1++)
    {
        for (int N2 = 0; N2 < 2; N2++)
        {
            auto L_vec = rR_.get_psi_L_psi(R, 0, 0, 0, N1, R, 0, 0, 0, N2);
            EXPECT_NEAR(L_vec.x.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.x.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.y.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.y.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.z.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.z.imag(), 0.0, DOUBLETHRESHOLD);
        }
    }
}

// Test 2: <s|L|p> = 0 for on-site (angular momentum doesn't couple s and p in this basis)
TEST_F(CalROverlapRLTest, SpOnSiteIsZero)
{
    ModuleBase::Vector3<double> R(0.0, 0.0, 0.0);
    // s: L=0, m=0; p: L=1, m=0,1,2
    for (int N_s = 0; N_s < 2; N_s++)
    {
        for (int m_p = 0; m_p < 3; m_p++)
        {
            auto L_vec = rR_.get_psi_L_psi(R, 0, 0, 0, N_s, R, 0, 1, m_p, 0);
            EXPECT_NEAR(L_vec.x.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.x.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.y.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.y.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.z.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_vec.z.imag(), 0.0, DOUBLETHRESHOLD);
        }
    }
}

// Test 3: Anti-Hermiticity of L matrix elements with real orbitals
// For real orbitals: <μ|L_α|ν> = -<ν|L_α|μ>*
// Since L is Hermitian and orbitals are real, the matrix elements are purely imaginary
// and antisymmetric: <μ|L_α|ν> = -<ν|L_α|μ>
TEST_F(CalROverlapRLTest, AntiHermiticity)
{
    ModuleBase::Vector3<double> R(0.0, 0.0, 0.0);
    // Test p-p pairs: L=1, different m values
    for (int m1 = 0; m1 < 3; m1++)
    {
        for (int m2 = 0; m2 < 3; m2++)
        {
            auto L_12 = rR_.get_psi_L_psi(R, 0, 1, m1, 0, R, 0, 1, m2, 0);
            auto L_21 = rR_.get_psi_L_psi(R, 0, 1, m2, 0, R, 0, 1, m1, 0);

            // <μ|L_α|ν> = -<ν|L_α|μ>* (Hermitian operator with real basis)
            // For purely imaginary values: -conj(i*a) = -(-i*a) = i*a, so imag parts are negated
            EXPECT_NEAR(L_12.x.real() + L_21.x.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.x.imag() + L_21.x.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.y.real() + L_21.y.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.y.imag() + L_21.y.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.z.real() + L_21.z.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.z.imag() + L_21.z.imag(), 0.0, DOUBLETHRESHOLD);
        }
    }

    // Test d-d pairs: L=2, different m values
    for (int m1 = 0; m1 < 5; m1++)
    {
        for (int m2 = 0; m2 < 5; m2++)
        {
            auto L_12 = rR_.get_psi_L_psi(R, 0, 2, m1, 0, R, 0, 2, m2, 0);
            auto L_21 = rR_.get_psi_L_psi(R, 0, 2, m2, 0, R, 0, 2, m1, 0);

            EXPECT_NEAR(L_12.x.real() + L_21.x.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.x.imag() + L_21.x.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.y.real() + L_21.y.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.y.imag() + L_21.y.imag(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.z.real() + L_21.z.real(), 0.0, DOUBLETHRESHOLD);
            EXPECT_NEAR(L_12.z.imag() + L_21.z.imag(), 0.0, DOUBLETHRESHOLD);
        }
    }
}

// Test 4: Finite-difference verification
// Compare analytical gradient (from cal_grad_overlap) with numerical gradient of get_psi_r_psi
TEST_F(CalROverlapRLTest, FiniteDifferenceGradient)
{
    // Use a displaced pair so the gradient is non-trivial
    ModuleBase::Vector3<double> R1(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> R2(2.0, 1.0, 0.5);

    const double delta = 1e-5;
    const double fd_tol = 1e-4; // finite-difference tolerance

    // Test with p-d pair: L1=1, m1=0, N1=0; L2=2, m2=1, N2=0
    int T = 0, L1 = 1, m1 = 0, N1 = 0, L2 = 2, m2 = 1, N2 = 0;

    // Get analytical L matrix element
    auto L_analytical = rR_.get_psi_L_psi(R1, T, L1, m1, N1, R2, T, L2, m2, N2);

    // Compute L via finite difference of get_psi_r_psi
    // L_α = i * ε_αβγ * ∂/∂R_γ <φ|(r-R_B)_β|φ>
    // where the gradient is w.r.t. R2

    // Get ∂/∂R_x, ∂/∂R_y, ∂/∂R_z of <φ|(r-R_B)_β|φ> for β=x,y,z
    // grad_r_beta[beta][gamma] = ∂/∂R2_gamma <φ|(r-R_B)_beta|φ>
    double grad_r[3][3]; // [beta][gamma]

    for (int gamma = 0; gamma < 3; gamma++)
    {
        ModuleBase::Vector3<double> R2_plus = R2;
        ModuleBase::Vector3<double> R2_minus = R2;
        if (gamma == 0) { R2_plus.x += delta; R2_minus.x -= delta; }
        else if (gamma == 1) { R2_plus.y += delta; R2_minus.y -= delta; }
        else { R2_plus.z += delta; R2_minus.z -= delta; }

        auto r_plus = rR_.get_psi_r_psi(R1, T, L1, m1, N1, R2_plus, T, L2, m2, N2);
        auto r_minus = rR_.get_psi_r_psi(R1, T, L1, m1, N1, R2_minus, T, L2, m2, N2);

        // Note: get_psi_r_psi returns <φ|(r)|φ> = <φ|(r-R_B)|φ> + R1 * S
        // We need <φ|(r-R_B)|φ>, so subtract R1 * S
        // But since R1 = (0,0,0), this is just get_psi_r_psi directly
        auto dr = (r_plus - r_minus) * (0.5 / delta);
        grad_r[0][gamma] = dr.x; // d<r_x>/dR_gamma
        grad_r[1][gamma] = dr.y; // d<r_y>/dR_gamma
        grad_r[2][gamma] = dr.z; // d<r_z>/dR_gamma
    }

    // L_x = i * (d<r_z>/dR_y - d<r_y>/dR_z)
    // L_y = i * (d<r_x>/dR_z - d<r_z>/dR_x)
    // L_z = i * (d<r_y>/dR_x - d<r_x>/dR_y)
    double Lx_fd = grad_r[2][1] - grad_r[1][2];
    double Ly_fd = grad_r[0][2] - grad_r[2][0];
    double Lz_fd = grad_r[1][0] - grad_r[0][1];

    // The analytical result should be purely imaginary (for real orbitals at non-zero distance)
    // L_analytical.x = i * (real part), so L_analytical.x.imag() should match Lx_fd
    EXPECT_NEAR(L_analytical.x.real(), 0.0, fd_tol);
    EXPECT_NEAR(L_analytical.x.imag(), Lx_fd, fd_tol);
    EXPECT_NEAR(L_analytical.y.real(), 0.0, fd_tol);
    EXPECT_NEAR(L_analytical.y.imag(), Ly_fd, fd_tol);
    EXPECT_NEAR(L_analytical.z.real(), 0.0, fd_tol);
    EXPECT_NEAR(L_analytical.z.imag(), Lz_fd, fd_tol);
}

// Test 5: Diagonal L matrix elements are zero (on-site, same orbital)
// <φ|L_α|φ> = 0 for any real orbital with itself
TEST_F(CalROverlapRLTest, DiagonalIsZero)
{
    ModuleBase::Vector3<double> R(0.0, 0.0, 0.0);

    // Test all orbitals: s, p, d
    // s orbitals
    for (int N = 0; N < 2; N++)
    {
        auto L_vec = rR_.get_psi_L_psi(R, 0, 0, 0, N, R, 0, 0, 0, N);
        EXPECT_NEAR(std::abs(L_vec.x), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.y), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.z), 0.0, DOUBLETHRESHOLD);
    }

    // p orbitals
    for (int m = 0; m < 3; m++)
    {
        auto L_vec = rR_.get_psi_L_psi(R, 0, 1, m, 0, R, 0, 1, m, 0);
        EXPECT_NEAR(std::abs(L_vec.x), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.y), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.z), 0.0, DOUBLETHRESHOLD);
    }

    // d orbitals
    for (int m = 0; m < 5; m++)
    {
        auto L_vec = rR_.get_psi_L_psi(R, 0, 2, m, 0, R, 0, 2, m, 0);
        EXPECT_NEAR(std::abs(L_vec.x), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.y), 0.0, DOUBLETHRESHOLD);
        EXPECT_NEAR(std::abs(L_vec.z), 0.0, DOUBLETHRESHOLD);
    }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}
