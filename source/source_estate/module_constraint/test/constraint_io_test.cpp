#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "constraint_test_utils.h"
#include "source_base/constants.h"
#include "source_estate/module_constraint/constraint_io.h"

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

TEST(ConstraintIOTest, DisabledByDefault)
{
    constraint::ConstraintConfig cfg;
    std::string error;
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, false, "charge", "becke", "delta", "{}", 5.0, 1e-4, 3, 1, error);
    EXPECT_EQ(st, constraint::ConfigStatus::DISABLED);
    EXPECT_FALSE(cfg.enabled);
}

TEST(ConstraintIOTest, DefaultsAndDeltaParsing)
{
    const std::string json = R"({"targets": [0.1, -0.1]})";
    constraint::ConstraintConfig cfg;
    std::string error;
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, true, "charge", "becke", "delta", json, 5.0, 1e-4, 3, 1, error);
    EXPECT_EQ(st, constraint::ConfigStatus::OK) << error;
    EXPECT_TRUE(cfg.enabled);
    EXPECT_EQ(cfg.weight_type, "becke");
    EXPECT_EQ(cfg.target_mode, "delta");
    EXPECT_DOUBLE_EQ(cfg.mu_max, 5.0); // default cap
    EXPECT_DOUBLE_EQ(cfg.thr, 1e-4);
    ASSERT_EQ(cfg.targets.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.targets[0].value, 0.1);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0})); // default atom i
    EXPECT_DOUBLE_EQ(cfg.targets[1].value, -0.1);
    EXPECT_EQ(cfg.targets[1].atoms, std::vector<int>({1}));
}

TEST(ConstraintIOTest, FragmentParsing)
{
    const std::string json =
        R"({"targets": [0.2], "atoms": [[0, 1, 2]]})";
    constraint::ConstraintConfig cfg;
    std::string error;
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, true, "charge", "becke", "delta", json, 5.0, 1e-4, 3, 1, error);
    EXPECT_EQ(st, constraint::ConfigStatus::OK) << error;
    ASSERT_EQ(cfg.targets.size(), 1u);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0, 1, 2}));

    // Flat "atoms" list.
    const std::string flat = R"({"targets": [0.1, 0.2], "atoms": [2, 0]})";
    constraint::ConstraintConfig cfg2;
    EXPECT_EQ(constraint::configure_constraint(cfg2, true, "charge", "becke",
                                               "delta", flat, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg2.targets[0].atoms, std::vector<int>({2}));
    EXPECT_EQ(cfg2.targets[1].atoms, std::vector<int>({0}));
}

TEST(ConstraintIOTest, AbsoluteModeWarnsButRuns)
{
    const std::string json = R"({"targets": [0.25, 0.25, 0.25]})";
    constraint::ConstraintConfig cfg;
    std::string error;
    // absolute mode is accepted with an explicit warning (written to the
    // (possibly unopened) ofs_warning stream; must not abort).
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, true, "charge", "becke", "absolute", json, 5.0, 1e-4, 3, 1, error);
    EXPECT_EQ(st, constraint::ConfigStatus::OK) << error;
    EXPECT_EQ(cfg.target_mode, "absolute");
}

TEST(ConstraintIOTest, Guards)
{
    std::string error;
    constraint::ConstraintConfig cfg;
    // hirshfeld weight: not implemented, never silently run.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "hirshfeld",
                                               "delta", "{}", 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("hirshfeld"), std::string::npos);
    // wrong type.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "magnet", "becke",
                                               "delta", "{}", 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    // missing target: no implicit constraint.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", "", 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("target"), std::string::npos);
    // bad mode.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "relax", "{}", 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    // fragment count mismatch.
    const std::string bad = R"({"targets": [0.1], "atoms": [[0], [1]]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", bad, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    // atom index out of range.
    const std::string oob = R"({"targets": [0.1], "atoms": [[7]]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", oob, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
}

TEST(ConstraintIOTest, MalformedJson)
{
    std::string error;
    constraint::ConstraintConfig cfg;
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", "not json", 5.0, 1e-4,
                                               3, 1, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta",
                                               R"({"targets": [0.1, })",
                                               5.0, 1e-4, 3, 1, error),
              constraint::ConfigStatus::ERROR);
}

TEST(ConstraintIOTest, SpinTypeGuard)
{
    // Phase-2 spin channel: type=spin requires nspin=2 (the reading and the
    // injection act on the spin-difference density rho_up - rho_dn).  A
    // nspin != 2 run must refuse loudly rather than silently run a wrong
    // constraint (historical false-convergence discipline, T4a').
    const std::string json = R"({"targets": [0.1], "atoms": [[0]]})";
    std::string error;
    constraint::ConstraintConfig cfg;
    // spin + nspin=1 -> ERROR.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "spin", "becke",
                                               "delta", json, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);
    // spin + nspin=2 -> OK, channel recorded in the config.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "spin", "becke",
                                               "delta", json, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg.type, "spin");
    // charge + nspin=2 still OK (charge couples to the total density).
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", json, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    // unknown type still ERROR regardless of nspin.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "magnet", "becke",
                                               "delta", json, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::ERROR);
}

TEST(ConstraintIOTest, NestedSingleElementFragments)
{
    // Nested fragments with one atom each (whitespace after the separators
    // must be tolerated): [[0], [1], [2]].
    const std::string json =
        R"({"targets": [0.1, -0.1, 0.2], "atoms": [[0], [1], [2]]})";
    constraint::ConstraintConfig cfg;
    std::string error;
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", json, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(cfg.targets.size(), 3u);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0}));
    EXPECT_EQ(cfg.targets[1].atoms, std::vector<int>({1}));
    EXPECT_EQ(cfg.targets[2].atoms, std::vector<int>({2}));
}

TEST(ConstraintIOTest, ConfigureFromInputsShared)
{
    auto ucell = make_h2o_ucell(); // O, H, H -> 3 atoms
    Set_GlobalV_Default();
    PARAM.input.constraint = false;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    PARAM.input.constraint_target_file = "not_there.json";
    constraint::ConstraintConfig cfg;
    std::vector<double> radii;
    std::string error;

    // Switch off -> DISABLED, nothing else touched.
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::DISABLED);
    EXPECT_FALSE(cfg.enabled);
    EXPECT_TRUE(radii.empty());

    // Unreadable target file -> ERROR.
    PARAM.input.constraint = true;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::ERROR);

    // Empty target file -> ERROR (no implicit constraint without target).
    PARAM.input.constraint_target_file = "";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::ERROR);

    // Guard: GPU device refuses to run.
    PARAM.input.device = "gpu";
    PARAM.input.constraint_target_file = "not_there.json";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.device = "cpu";

    // OK path: real target file, delta mode, one fragment on atom 0.
    const std::string path = "constraint_target_shared_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(cfg.targets.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.targets[0].value, 0.1);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0}));
    // Covalent radii (Angstrom -> Bohr): O = 0.64 A, H = 0.32 A.
    ASSERT_EQ(radii.size(), 3u);
    EXPECT_NEAR(radii[0], 0.64 / ModuleBase::BOHR_TO_A, 1e-12);
    EXPECT_NEAR(radii[1], 0.32 / ModuleBase::BOHR_TO_A, 1e-12);
    EXPECT_NEAR(radii[2], 0.32 / ModuleBase::BOHR_TO_A, 1e-12);

    // Spin guard through the shared path: PARAM nspin=1 + type=spin -> ERROR
    // (the spin channel needs a two-channel run); nspin=2 -> OK.
    PARAM.input.constraint_type = "spin";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);
    PARAM.input.nspin = 2;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, *ucell, radii, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg.type, "spin");
    PARAM.input.nspin = 1;
    PARAM.input.constraint_type = "charge";
    std::remove(path.c_str());
}
