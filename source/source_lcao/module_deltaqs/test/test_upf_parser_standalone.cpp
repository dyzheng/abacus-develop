/**
 * @file test_upf_parser_standalone.cpp
 * @brief Standalone test for UPF valence parser using actual ABACUS pseudopotential files.
 *
 * Compile:
 *   g++ -std=c++14 -I/abacus-develop/source \
 *       test_upf_parser_standalone.cpp \
 *       /abacus-develop/build/source/source_lcao/module_deltaqs/CMakeFiles/deltaqs.dir/upf_valence_parser.cpp.o \
 *       -o test_upf_parser
 *
 * This test creates a mock UnitCell with data matching known UPF files
 * and verifies the parser produces correct CSZ configurations.
 */
#include <iostream>
#include <cassert>
#include <cmath>
#include "source_lcao/module_deltaqs/upf_valence_parser.h"

int main()
{
    std::cout << "=== DeltaQS UPF Valence Parser Standalone Test ===" << std::endl;

    // Test 1: Fe (3s²3p⁶3d⁶4s², Z_val=16)
    {
        UnitCell ucell;
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "Fe";
        ucell.atoms[0].ncpp.zv = 16.0;
        ucell.atoms[0].na = 2;
        ucell.atoms[0].nwl = 3;
        ucell.atoms[0].l_nchi = {4, 2, 2, 1}; // 4s2p2d1f orbital file

        // PP_PSWFC channels: 3s(2e), 3p(6e), 3d(6e), 4s(2e)
        ucell.atoms[0].ncpp.nchi = 4;
        ucell.atoms[0].ncpp.lchi = {0, 1, 2, 0}; // s, p, d, s
        ucell.atoms[0].ncpp.oc = {2.0, 6.0, 6.0, 2.0};

        auto configs = deltaqs::parse_valence_configs(ucell);
        deltaqs::print_valence_configs(configs, ucell);

        assert(configs.size() == 1);
        assert(std::abs(configs[0].zv_total - 16.0) < 0.01);
        assert(configs[0].csz_per_l.at(0) == 4); // s: 4 zetas
        assert(configs[0].csz_per_l.at(1) == 2); // p: 2 zetas
        assert(configs[0].csz_per_l.at(2) == 2); // d: 2 zetas
        assert(configs[0].total_csz_orbitals == 20); // 4+6+10

        bool valid = deltaqs::validate_csz_orbitals(ucell, configs);
        assert(valid);

        std::cout << "[PASS] Fe test" << std::endl;
        delete[] ucell.atoms;
    }

    // Test 2: O (2s²2p⁴, Z_val=6)
    {
        UnitCell ucell;
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "O";
        ucell.atoms[0].ncpp.zv = 6.0;
        ucell.atoms[0].na = 1;
        ucell.atoms[0].nwl = 2;
        ucell.atoms[0].l_nchi = {2, 2, 1}; // 2s2p1d orbital file

        // PP_PSWFC channels: 2s(2e), 2p(4e)
        ucell.atoms[0].ncpp.nchi = 2;
        ucell.atoms[0].ncpp.lchi = {0, 1}; // s, p
        ucell.atoms[0].ncpp.oc = {2.0, 4.0};

        auto configs = deltaqs::parse_valence_configs(ucell);
        deltaqs::print_valence_configs(configs, ucell);

        assert(configs.size() == 1);
        assert(std::abs(configs[0].zv_total - 6.0) < 0.01);
        assert(configs[0].csz_per_l.at(0) == 2); // s: 2 zetas
        assert(configs[0].csz_per_l.at(1) == 2); // p: ceil(4/3) = 2 zetas
        assert(configs[0].total_csz_orbitals == 8); // 2+6

        bool valid = deltaqs::validate_csz_orbitals(ucell, configs);
        assert(valid);

        std::cout << "[PASS] O test" << std::endl;
        delete[] ucell.atoms;
    }

    // Test 3: Validation failure (insufficient zetas)
    {
        UnitCell ucell;
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "Fe";
        ucell.atoms[0].ncpp.zv = 16.0;
        ucell.atoms[0].na = 1;
        ucell.atoms[0].nwl = 2;
        ucell.atoms[0].l_nchi = {1, 1, 1}; // Only 1s1p1d (insufficient!)

        ucell.atoms[0].ncpp.nchi = 4;
        ucell.atoms[0].ncpp.lchi = {0, 1, 2, 0};
        ucell.atoms[0].ncpp.oc = {2.0, 6.0, 6.0, 2.0};

        auto configs = deltaqs::parse_valence_configs(ucell);
        bool valid = deltaqs::validate_csz_orbitals(ucell, configs);
        assert(!valid); // Should fail

        std::cout << "[PASS] Validation failure test" << std::endl;
        delete[] ucell.atoms;
    }

    // Test 4: Ti (3d²4s², Z_val=4) - test with fewer electrons
    {
        UnitCell ucell;
        ucell.ntype = 1;
        ucell.atoms = new Atom[1];
        ucell.atoms[0].label = "Ti";
        ucell.atoms[0].ncpp.zv = 12.0; // Ti: 3s²3p⁶3d²4s²
        ucell.atoms[0].na = 1;
        ucell.atoms[0].nwl = 2;
        ucell.atoms[0].l_nchi = {3, 2, 2}; // 3s2p2d orbital file

        ucell.atoms[0].ncpp.nchi = 4;
        ucell.atoms[0].ncpp.lchi = {0, 1, 2, 0}; // 3s, 3p, 3d, 4s
        ucell.atoms[0].ncpp.oc = {2.0, 6.0, 2.0, 2.0};

        auto configs = deltaqs::parse_valence_configs(ucell);
        deltaqs::print_valence_configs(configs, ucell);

        assert(configs[0].csz_per_l.at(0) == 4); // s: ceil(4/1) = 4
        assert(configs[0].csz_per_l.at(1) == 2); // p: ceil(6/3) = 2
        assert(configs[0].csz_per_l.at(2) == 1); // d: ceil(2/5) = 1
        assert(configs[0].total_csz_orbitals == 4 + 6 + 5); // = 15

        std::cout << "[PASS] Ti test" << std::endl;
        delete[] ucell.atoms;
    }

    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}
