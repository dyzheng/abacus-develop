#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "source_estate/module_constraint/constraint_io.h"

TEST(ConstraintIOTest, DisabledByDefault)
{
    constraint::ConstraintConfig cfg;
    std::string error;
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, false, "charge", "becke", "delta", "{}", 5.0, 1e-4, 3, error);
    EXPECT_EQ(st, constraint::ConfigStatus::DISABLED);
    EXPECT_FALSE(cfg.enabled);
}

TEST(ConstraintIOTest, DefaultsAndDeltaParsing)
{
    const std::string json = R"({"targets": [0.1, -0.1]})";
    constraint::ConstraintConfig cfg;
    std::string error;
    const constraint::ConfigStatus st = constraint::configure_constraint(
        cfg, true, "charge", "becke", "delta", json, 5.0, 1e-4, 3, error);
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
        cfg, true, "charge", "becke", "delta", json, 5.0, 1e-4, 3, error);
    EXPECT_EQ(st, constraint::ConfigStatus::OK) << error;
    ASSERT_EQ(cfg.targets.size(), 1u);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0, 1, 2}));

    // Flat "atoms" list.
    const std::string flat = R"({"targets": [0.1, 0.2], "atoms": [2, 0]})";
    constraint::ConstraintConfig cfg2;
    EXPECT_EQ(constraint::configure_constraint(cfg2, true, "charge", "becke",
                                               "delta", flat, 5.0, 1e-4, 3,
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
        cfg, true, "charge", "becke", "absolute", json, 5.0, 1e-4, 3, error);
    EXPECT_EQ(st, constraint::ConfigStatus::OK) << error;
    EXPECT_EQ(cfg.target_mode, "absolute");
}

TEST(ConstraintIOTest, Guards)
{
    std::string error;
    constraint::ConstraintConfig cfg;
    // hirshfeld weight: not implemented, never silently run.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "hirshfeld",
                                               "delta", "{}", 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("hirshfeld"), std::string::npos);
    // wrong type.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "magnet", "becke",
                                               "delta", "{}", 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
    // missing target: no implicit constraint.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", "", 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("target"), std::string::npos);
    // bad mode.
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "relax", "{}", 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
    // fragment count mismatch.
    const std::string bad = R"({"targets": [0.1], "atoms": [[0], [1]]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", bad, 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
    // atom index out of range.
    const std::string oob = R"({"targets": [0.1], "atoms": [[7]]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", oob, 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::ERROR);
}

TEST(ConstraintIOTest, MalformedJson)
{
    std::string error;
    constraint::ConstraintConfig cfg;
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", "not json", 5.0, 1e-4,
                                               3, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta",
                                               R"({"targets": [0.1, })",
                                               5.0, 1e-4, 3, error),
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
                                               "delta", json, 5.0, 1e-4, 3,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(cfg.targets.size(), 3u);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0}));
    EXPECT_EQ(cfg.targets[1].atoms, std::vector<int>({1}));
    EXPECT_EQ(cfg.targets[2].atoms, std::vector<int>({2}));
}
