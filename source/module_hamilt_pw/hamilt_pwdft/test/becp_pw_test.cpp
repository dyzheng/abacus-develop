#include <gtest/gtest.h>
#include "module_hamilt_pw/hamilt_pwdft/becp_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

Atom_pseudo::Atom_pseudo() {}
Atom_pseudo::~Atom_pseudo() {}
#ifdef __MPI
void Atom_pseudo::bcast_atom_pseudo() {}
#endif
pseudo::pseudo() {}
pseudo::~pseudo() {}

pseudopot_cell_vnl::pseudopot_cell_vnl() {}
pseudopot_cell_vnl::~pseudopot_cell_vnl()
{
}
pseudopot_cell_vl::pseudopot_cell_vl() {}
pseudopot_cell_vl::~pseudopot_cell_vl() {}
Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}
void output::printM3(std::ofstream &ofs, const std::string &description, const ModuleBase::Matrix3 &m) {}
#ifdef __LCAO
ORB_gaunt_table::ORB_gaunt_table() {}
ORB_gaunt_table::~ORB_gaunt_table() {}
InfoNonlocal::InfoNonlocal() {}
InfoNonlocal::~InfoNonlocal() {}
#endif
UnitCell::UnitCell() {}
UnitCell::~UnitCell() {}
Parallel_Grid::Parallel_Grid() {}
Parallel_Grid::~Parallel_Grid() {}
Parallel_Kpoints::Parallel_Kpoints() {}
Parallel_Kpoints::~Parallel_Kpoints() {}
void ModulePW::PW_Basis::distribution_method1() {}
void ModulePW::PW_Basis::distribution_method2() {}
Soc::~Soc() {}
Fcoef::~Fcoef() {}
#ifdef __MPI
void Parallel_Grid::zpiece_to_all(double *, const int &, double *) {}
#endif

TEST(TestBecpPW, TestInitProj)
{
    std::string orbital_dir = "../../../../../tests/PP_ORB";
    std::vector<std::string> orb_files = {"", "Ti_gga_8au_100Ry_4s2p2d1f.orb", "Al_gga_10au_100Ry_3s3p2d.orb"};
    std::vector<int> nproj = {0, 3, 0};
    std::vector<int> lproj = {0, 1, 2};
    std::vector<int> iproj = {0, 0, 0};
    std::vector<double> onsite_r = {3.0, 3.0, 3.0};
    std::vector<double> rgrid;
    std::vector<std::vector<double>> projs;
    std::vector<std::vector<int>> it2iproj;
    becp::init_proj(orbital_dir, orb_files, nproj, lproj, iproj, onsite_r, rgrid, projs, it2iproj);
    const int nr = rgrid.size();
    for (auto &proj : projs)
    {
        EXPECT_EQ(proj.size(), nr);
        // not all zeros
        bool not_all_zeros = false;
        for (auto &p : proj)
        {
            if (p != 0.0)
            {
                not_all_zeros = true;
                break;
            }
        }
        EXPECT_TRUE(not_all_zeros);
    }
    const std::vector<std::vector<int>> it2iprojref = {{}, {0, 1, 2}, {}};
    EXPECT_EQ(it2iproj.size(), it2iprojref.size());
    for (int i = 0; i < it2iproj.size(); i++)
    {
        EXPECT_EQ(it2iproj[i].size(), it2iprojref[i].size());
        for (int j = 0; j < it2iproj[i].size(); j++)
        {
            EXPECT_EQ(it2iproj[i][j], it2iprojref[i][j]);
        }
    }
}



int main(int argc, char **argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    int result = 0;
    result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}