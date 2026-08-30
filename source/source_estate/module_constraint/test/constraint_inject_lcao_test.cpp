#include "gtest/gtest.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "source_base/global_variable.h"
#include "source_base/math_integral.h"
#include "source_base/ylm.h"
#include "source_basis/module_ao/ORB_atomic.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_sup.h"
#include "source_basis/module_pw/pw_basis_big.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/setup_nonlocal.h"
#include "source_cell/unitcell.h"
#include "source_estate/magnetism.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_gint/gint.h"
#include "source_lcao/module_gint/gint_info.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "../../test/prepare_unitcell.h"
#include "../constraint_inject_lcao.h"
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

namespace
{

// Cubic box edge (Bohr) and grid points per direction of the toy cell.
constexpr double kBox = 20.0;
constexpr int kGrid = 40;

// Local GlobalV/PARAM defaults for a serial LCAO gamma-only unit test.
void Set_GlobalV_Default()
{
    PARAM.input.device = "cpu";
    PARAM.input.precision = "double";
    PARAM.input.nspin = 1;
    PARAM.input.nelec = 2.0;
    PARAM.input.basis_type = "lcao";
    PARAM.input.gamma_only = true;
    PARAM.input.out_level = "m";
    PARAM.sys.nlocal = 2;
    PARAM.sys.npol = 1;
    PARAM.sys.gamma_only_local = true;
    PARAM.sys.search_pbc = true;
    GlobalV::KPAR = 1;
    GlobalV::NPROC_IN_POOL = 1;
}

// Two-H dimer: one s orbital per atom, atoms away from the cell boundary so
// no periodic image of the partner atom enters the neighbor search.
std::unique_ptr<UnitCell> make_dimer_ucell()
{
    UcellTestPrepare utp(
        "cubic", 2, false, false, false, "None",
        1.0, // lat0 in Bohr; tau (Bohr) == Cartesian coordinates
        {20, 0, 0, 0, 20, 0, 0, 0, 20}, // 20 Bohr box
        {"H"}, {"H.upf"}, {"upf201"}, {""},
        {2}, {1.0}, "Cartesian",
        {8.3, 10.0, 10.0, 12.3, 10.0, 10.0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    return utp.SetUcellInfo();
}

// Analytical Gaussian s-orbital tabulated on a uniform radial mesh and
// normalized like an ABACUS NAO (integral of r^2 * psi^2 = 1).
Numerical_Orbital make_s_orbital(const double rcut)
{
    const int nr = 501;      // odd, required by the Simpson normalization
    const double dr = 0.01;  // Bohr
    std::vector<double> r_radial(nr), rab(nr, dr), psi(nr);
    for (int ir = 0; ir < nr; ++ir)
    {
        r_radial[ir] = ir * dr;
        psi[ir] = std::exp(-r_radial[ir] * r_radial[ir]);
    }
    std::vector<double> integrand(nr);
    for (int ir = 0; ir < nr; ++ir)
    {
        integrand[ir] = std::pow(psi[ir] * r_radial[ir], 2);
    }
    double radint = 0.0;
    ModuleBase::Integral::Simpson_Integral(nr, integrand.data(), rab.data(), radint);
    for (int ir = 0; ir < nr; ++ir)
    {
        psi[ir] /= std::sqrt(radint);
    }

    const int nk = 201;  // odd, > 1 (radial Fourier mesh)
    const double dk = 0.01;
    const double dr_uniform = 0.005;

    Numerical_Orbital_Lm nolm;
    nolm.set_orbital_info(
        "H", 0, 0, 0, nr, rab.data(), r_radial.data(),
        Numerical_Orbital_Lm::Psi_Type::Psi, psi.data(), nk, dk, dr_uniform,
        false, true, true);

    Numerical_Orbital orb;
    orb.chi().push_back(nolm);
    const int nchi[1] = {1};
    orb.set_orbital_info(0, "H", 0, nchi, 1);
    EXPECT_DOUBLE_EQ(orb.getRcut(), rcut);
    return orb;
}

// Cartesian position of real-space grid point ir (serial FFT layout,
// z fastest) in Bohr.
ModuleBase::Vector3<double> grid_position(
    const ModulePW::PW_Basis_Big& pw, const int ir)
{
    const int iz = ir % pw.nz;
    const int iy = (ir / pw.nz) % pw.ny;
    const int ix = ir / (pw.ny * pw.nz);
    return ModuleBase::Vector3<double>(
               static_cast<double>(ix) / pw.nx,
               static_cast<double>(iy) / pw.ny,
               static_cast<double>(iz) / pw.nz)
           * pw.latvec * pw.lat0;
}

// Independent reference evaluation of the NAO at the displacement vector rel:
// replicates the cubic-Hermite interpolation of GintAtom::set_phi on the
// same psi_uniform/dpsi_uniform tables (l=0, single chi).  This is an
// independent quadrature path: the grid-point -> value mapping is rebuilt
// here from the interpolation tables instead of reusing Gint code.
double eval_orbital(const Numerical_Orbital& orb,
                    const ModuleBase::Vector3<double>& rel)
{
    const double dist = rel.norm() < 1e-9 ? 1e-9 : rel.norm();
    if (dist > orb.getRcut())
    {
        return 0.0;
    }
    const Numerical_Orbital_Lm& lm = orb.PhiLN(0, 0);
    const double* psi_u = lm.getPsiuniform();
    const double* dpsi_u = lm.getDpsiuniform();
    const double dr = lm.getDruniform();

    const double position = dist / dr;
    const int ip = static_cast<int>(position);
    const double dx = position - ip;
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;
    const double c3 = 3.0 * dx2 - 2.0 * dx3;
    const double c1 = 1.0 - c3;
    const double c2 = (dx - 2.0 * dx2 + dx3) * dr;
    const double c4 = (dx3 - dx2) * dr;
    const double psi = c1 * psi_u[ip] + c2 * dpsi_u[ip] + c3 * psi_u[ip + 1]
                       + c4 * dpsi_u[ip + 1];

    std::vector<double> ylma;
    ModuleBase::Ylm::sph_harm(0, rel.x / dist, rel.y / dist, rel.z / dist, ylma);
    return psi * ylma[0];
}

// Reference W^alpha_mu,nu = sum_ir phi_mu(r) w(r) phi_nu(r) dV on the same
// pw_rho mesh used by the Gint kernel.
struct DenseRef
{
    double m[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
};

DenseRef direct_integral(
    const ModulePW::PW_Basis_Big& pw,
    const Numerical_Orbital& orb,
    const std::array<ModuleBase::Vector3<double>, 2>& atom_pos,
    const std::vector<double>& w)
{
    DenseRef ref;
    const double dr3 = pw.omega / static_cast<double>(pw.nrxx);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const ModuleBase::Vector3<double> r = grid_position(pw, ir);
        double phi[2];
        for (int mu = 0; mu < 2; ++mu)
        {
            phi[mu] = eval_orbital(orb, r - atom_pos[mu]);
        }
        for (int mu = 0; mu < 2; ++mu)
        {
            for (int nu = 0; nu < 2; ++nu)
            {
                ref.m[mu][nu] += phi[mu] * w[ir] * phi[nu] * dr3;
            }
        }
    }
    return ref;
}

// Read a single matrix element of a constraint HContainer (one orbital per
// atom, R = 0).
double gint_element(const hamilt::HContainer<double>& W,
                    const int iat1, const int iat2)
{
    const auto* mat = W.find_matrix(iat1, iat2, 0, 0, 0);
    EXPECT_NE(mat, nullptr);
    if (mat == nullptr)
    {
        return 0.0;
    }
    EXPECT_EQ(mat->get_row_size(), 1);
    EXPECT_EQ(mat->get_col_size(), 1);
    return mat->get_value(0, 0);
}

} // namespace

class ConstraintInjectLCAOTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell_ = make_dimer_ucell();
        // Set the atom indexing tables required by GintInfo (iat2it/iat2ia
        // raw arrays plus the itia2iat IntArray, following the LCAO test
        // scaffold in module_hcontainer/test/prepare_unitcell.h).
        ucell_->namax = 2;
        delete[] ucell_->iat2it;
        delete[] ucell_->iat2ia;
        ucell_->iat2it = new int[ucell_->nat];
        ucell_->iat2ia = new int[ucell_->nat];
        ucell_->itia2iat.create(ucell_->ntype, ucell_->namax);
        {
            int iat = 0;
            for (int it = 0; it < ucell_->ntype; ++it)
            {
                for (int ia = 0; ia < ucell_->atoms[it].na; ++ia)
                {
                    ucell_->iat2it[iat] = it;
                    ucell_->iat2ia[iat] = ia;
                    ucell_->itia2iat(it, ia) = iat;
                    ++iat;
                }
            }
        }
        orb_ = make_s_orbital(5.0);

        // Collapse the toy basis to a single s orbital per atom and set the
        // per-atom Gint indexing tables (iw2l/iw2n/iw2m/iw2_ylm/iw2_new).
        ucell_->atoms[0].nwl = 0;
        ucell_->atoms[0].l_nchi = {1};
        ucell_->atoms[0].nw = 1;
        ucell_->atoms[0].Rcut = orb_.getRcut();
        ucell_->atoms[0].set_index();
        ucell_->set_iat2iwt(1);

        // Neighbor search (required by GintInfo::init_ijr_info_).
        std::ofstream ofs("./constraint_inject_lcao_test.log");
        const double sr = atom_arrange::set_sr_NL(
            ofs, PARAM.input.out_level, orb_.getRcut(),
            ucell_->infoNL.get_rcutmax_Beta(), PARAM.sys.gamma_only_local);
        atom_arrange::search(PARAM.sys.search_pbc, ofs, gd_, *ucell_, sr, 0);

        // FFT/real-space density grid (serial, explicit natural grids).
        pw_ = new ModulePW::PW_Basis_Big("cpu", "double");
        pw_->initgrids(ucell_->lat0, ucell_->latvec, kGrid, kGrid, kGrid);
        pw_->initparameters(false, 40.0);
        pw_->distribute_r();

        // Full GintInfo (real constructor path, same as the LCAO esolver).
        gint_info_ = new ModuleGint::GintInfo(
            pw_->nbx, pw_->nby, pw_->nbz, pw_->nx, pw_->ny, pw_->nz,
            0, 0, pw_->nbzp_start, pw_->nbx, pw_->nby, pw_->nbzp,
            &orb_, *ucell_, gd_);
        ModuleGint::Gint::set_gint_info(gint_info_);

        atom_pos_[0] = ucell_->atoms[0].taud[0] * ucell_->latvec * ucell_->lat0;
        atom_pos_[1] = ucell_->atoms[0].taud[1] * ucell_->latvec * ucell_->lat0;

        // Analytic partition weights on the pw_rho grid: w1 + w2 == 1 exactly
        // (w2 = 1 - w1), smooth and grid-resolved.
        const int nrxx = pw_->nrxx;
        w1_.assign(nrxx, 0.0);
        w2_.assign(nrxx, 0.0);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const double x = grid_position(*pw_, ir).x;
            w1_[ir] = 0.5 * (1.0 + x / kBox);
            w2_[ir] = 1.0 - w1_[ir];
        }
    }

    void TearDown() override
    {
        delete gint_info_;
        delete pw_;
    }

    std::unique_ptr<UnitCell> ucell_;
    Numerical_Orbital orb_;
    Grid_Driver gd_;
    ModulePW::PW_Basis_Big* pw_ = nullptr;
    ModuleGint::GintInfo* gint_info_ = nullptr;
    std::array<ModuleBase::Vector3<double>, 2> atom_pos_;
    std::vector<double> w1_, w2_;
};

// M3b core: W^alpha_μν from the Gint kernel == direct grid quadrature of
// phi_mu * w_alpha * phi_nu on the same mesh (relative diff < 1e-10).
TEST_F(ConstraintInjectLCAOTest, MatrixElementVsDirectGrid)
{
    const std::vector<std::vector<double>> cw = {w1_, w2_};
    const auto W = constraint::ConstraintInjectLCAO::build(cw, gint_info_);
    ASSERT_EQ(W.size(), 2);

    const DenseRef ref1 = direct_integral(*pw_, orb_, atom_pos_, w1_);
    const DenseRef ref2 = direct_integral(*pw_, orb_, atom_pos_, w2_);

    const double gint1[2][2] = {
        {gint_element(W[0], 0, 0), gint_element(W[0], 0, 1)},
        {gint_element(W[0], 1, 0), gint_element(W[0], 1, 1)}};
    const double gint2[2][2] = {
        {gint_element(W[1], 0, 0), gint_element(W[1], 0, 1)},
        {gint_element(W[1], 1, 0), gint_element(W[1], 1, 1)}};

    for (int mu = 0; mu < 2; ++mu)
    {
        for (int nu = 0; nu < 2; ++nu)
        {
            EXPECT_LT(std::abs(gint1[mu][nu] - ref1.m[mu][nu]),
                      std::max(std::abs(ref1.m[mu][nu]), 1e-12) * 1e-10);
            EXPECT_LT(std::abs(gint2[mu][nu] - ref2.m[mu][nu]),
                      std::max(std::abs(ref2.m[mu][nu]), 1e-12) * 1e-10);
        }
    }
}

// Partition-of-unity audit at the matrix-element level: sum_alpha W^alpha
// == overlap matrix S (machine-precision level, < 1e-10).
TEST_F(ConstraintInjectLCAOTest, PartitionSumRuleEqualsOverlap)
{
    const std::vector<std::vector<double>> cw = {w1_, w2_};
    const auto W = constraint::ConstraintInjectLCAO::build(cw, gint_info_);

    std::vector<double> ones(pw_->nrxx, 1.0);
    const DenseRef S_ref = direct_integral(*pw_, orb_, atom_pos_, ones);

    for (int mu = 0; mu < 2; ++mu)
    {
        for (int nu = 0; nu < 2; ++nu)
        {
            const double sum_w = gint_element(W[0], mu, nu) + gint_element(W[1], mu, nu);
            EXPECT_LT(std::abs(sum_w - S_ref.m[mu][nu]),
                      std::max(std::abs(S_ref.m[mu][nu]), 1e-12) * 1e-10);
        }
    }
}

// add_weighted: H += sum_alpha mu_alpha * W^alpha, elementwise.
TEST_F(ConstraintInjectLCAOTest, AddWeightedMatchesLinearCombination)
{
    const std::vector<std::vector<double>> cw = {w1_, w2_};
    const auto W = constraint::ConstraintInjectLCAO::build(cw, gint_info_);
    ASSERT_EQ(W.size(), 2);
    ASSERT_EQ(W[0].get_nnr(), W[1].get_nnr());

    const std::vector<double> mu = {0.5, -0.25};
    hamilt::HContainer<double> H = gint_info_->get_hr<double>();
    ASSERT_EQ(H.get_nnr(), W[0].get_nnr());
    constraint::ConstraintInjectLCAO::add_weighted(mu, W, &H);

    const double* w0 = W[0].get_wrapper();
    const double* w1 = W[1].get_wrapper();
    const double* h = H.get_wrapper();
    for (size_t i = 0; i < H.get_nnr(); ++i)
    {
        EXPECT_DOUBLE_EQ(h[i], mu[0] * w0[i] + mu[1] * w1[i]);
    }
}

// Zero multipliers are skipped: the resulting H is bit-identical to the
// manual combination with those terms dropped.
TEST_F(ConstraintInjectLCAOTest, ZeroMuIsNoOp)
{
    const std::vector<std::vector<double>> cw = {w1_, w2_};
    const auto W = constraint::ConstraintInjectLCAO::build(cw, gint_info_);

    const std::vector<double> mu = {0.0, 0.7};
    hamilt::HContainer<double> H = gint_info_->get_hr<double>();
    constraint::ConstraintInjectLCAO::add_weighted(mu, W, &H);

    const double* w1 = W[1].get_wrapper();
    const double* h = H.get_wrapper();
    for (size_t i = 0; i < H.get_nnr(); ++i)
    {
        EXPECT_DOUBLE_EQ(h[i], mu[1] * w1[i]);
    }
}
