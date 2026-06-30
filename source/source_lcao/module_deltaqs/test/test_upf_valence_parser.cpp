/**
 * @file test_upf_valence_parser.cpp
 * @brief Unit tests for CSZ basis determination.
 */
#include <gtest/gtest.h>
#include "source_lcao/module_deltaqs/upf_valence_parser.h"
#include "source_cell/unitcell.h"

class CSZBasisTest : public ::testing::Test {
protected:
    void SetUpFe(UnitCell& ucell) {
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "Fe";
        ucell.atoms[0].ncpp.zv = 16.0;
        ucell.atoms[0].na = 2;
        ucell.atoms[0].nwl = 3;
        ucell.atoms[0].l_nchi = {4, 2, 2, 1}; // 4s2p2d1f
    }

    void SetUpO(UnitCell& ucell) {
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "O";
        ucell.atoms[0].ncpp.zv = 6.0;
        ucell.atoms[0].na = 1;
        ucell.atoms[0].nwl = 2;
        ucell.atoms[0].l_nchi = {2, 2, 1}; // 2s2p1d
    }

    void TearDown_UnitCell(UnitCell& ucell) {
        delete[] ucell.atoms;
        ucell.atoms = nullptr;
    }
};

TEST_F(CSZBasisTest, FeCSZBasis) {
    UnitCell ucell;
    SetUpFe(ucell);

    auto configs = deltaqs::determine_csz_basis(ucell);

    ASSERT_EQ(configs.size(), 1);
    const auto& vc = configs[0];

    EXPECT_DOUBLE_EQ(vc.zv_total, 16.0);
    EXPECT_EQ(vc.l_max, 3);

    // Use all zetas from orbital file
    EXPECT_EQ(vc.csz_per_l.at(0), 4); // s: 4 zetas
    EXPECT_EQ(vc.csz_per_l.at(1), 2); // p: 2 zetas
    EXPECT_EQ(vc.csz_per_l.at(2), 2); // d: 2 zetas
    EXPECT_EQ(vc.csz_per_l.at(3), 1); // f: 1 zeta

    // Total: 4*1 + 2*3 + 2*5 + 1*7 = 4+6+10+7 = 27
    EXPECT_EQ(vc.total_csz_orbitals, 27);

    EXPECT_TRUE(deltaqs::validate_csz_orbitals(ucell, configs));

    TearDown_UnitCell(ucell);
}

TEST_F(CSZBasisTest, OCSZBasis) {
    UnitCell ucell;
    SetUpO(ucell);

    auto configs = deltaqs::determine_csz_basis(ucell);

    ASSERT_EQ(configs.size(), 1);
    const auto& vc = configs[0];

    EXPECT_DOUBLE_EQ(vc.zv_total, 6.0);
    EXPECT_EQ(vc.l_max, 2);

    EXPECT_EQ(vc.csz_per_l.at(0), 2); // s: 2 zetas
    EXPECT_EQ(vc.csz_per_l.at(1), 2); // p: 2 zetas
    EXPECT_EQ(vc.csz_per_l.at(2), 1); // d: 1 zeta

    // Total: 2*1 + 2*3 + 1*5 = 2+6+5 = 13
    EXPECT_EQ(vc.total_csz_orbitals, 13);

    TearDown_UnitCell(ucell);
}
