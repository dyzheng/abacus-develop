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
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Switch off -> DISABLED, nothing else touched.
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::DISABLED);
    EXPECT_FALSE(cfg.enabled);
    EXPECT_TRUE(radii.empty());
    EXPECT_TRUE(specs.empty());

    // Unreadable target file -> ERROR.
    PARAM.input.constraint = true;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);

    // Empty target file -> ERROR (no implicit constraint without target).
    PARAM.input.constraint_target_file = "";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);

    // Guard: GPU device refuses to run.
    PARAM.input.device = "gpu";
    PARAM.input.constraint_target_file = "not_there.json";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.device = "cpu";

    // OK path: real target file, delta mode, one fragment on atom 0.
    const std::string path = "constraint_target_shared_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(cfg.targets.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.targets[0].value, 0.1);
    EXPECT_EQ(cfg.targets[0].atoms, std::vector<int>({0}));
    // The stage-A specs output is populated through the shared path (v1 file
    // + run-level type=charge -> homogeneous charge specs).
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Charge);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0}));
    EXPECT_DOUBLE_EQ(specs[0].target, 0.1);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 5.0); // run-level fallback
    // Covalent radii (Angstrom -> Bohr): O = 0.64 A, H = 0.32 A.
    ASSERT_EQ(radii.size(), 3u);
    EXPECT_NEAR(radii[0], 0.64 / ModuleBase::BOHR_TO_A, 1e-12);
    EXPECT_NEAR(radii[1], 0.32 / ModuleBase::BOHR_TO_A, 1e-12);
    EXPECT_NEAR(radii[2], 0.32 / ModuleBase::BOHR_TO_A, 1e-12);

    // Spin guard through the shared path: PARAM nspin=1 + type=spin -> ERROR
    // (the spin channel needs a two-channel run); nspin=2 -> OK.
    PARAM.input.constraint_type = "spin";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);
    PARAM.input.nspin = 2;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg.type, "spin");
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Spin);
    PARAM.input.nspin = 1;
    PARAM.input.constraint_type = "charge";
    std::remove(path.c_str());
}

namespace
{
// Compare two channel profiles field-by-field (no operator== on the struct).
bool same_channel(const constraint::ChannelProfile& a,
                  const constraint::ChannelProfile& b)
{
    return a.read_up == b.read_up && a.read_dn == b.read_dn
           && a.inj_up == b.inj_up && a.inj_dn == b.inj_dn;
}
bool warning_has(const std::vector<std::string>& warnings,
                 const std::string& marker)
{
    for (const std::string& w : warnings)
    {
        if (w.find(marker) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}
} // namespace

TEST(ConstraintIOTest, MixedConstraintListParsing)
{
    // v2 format: charge first (explicit fragment + per-constraint mu_max),
    // spin second (fragment defaults to the list index 1, mu_max falls back
    // to the run-level cap).  The parser must derive chan from the kind
    // factory, never a hand-filled default.
    const std::string v2 = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0], "mu_max": 3.0},
        {"type": "spin", "target": -0.05}]})";
    std::vector<constraint::ConstraintSpec> specs;
    constraint::ConstraintFileFormat fmt;
    std::vector<std::string> warnings;
    std::string error;
    ASSERT_TRUE(constraint::parse_constraint_file(v2, "charge", 5.0, 3, 2,
                                                  specs, fmt, warnings, error))
        << error;
    EXPECT_EQ(fmt, constraint::ConstraintFileFormat::V2);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Charge);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0}));
    EXPECT_DOUBLE_EQ(specs[0].target, 0.1);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 3.0); // per-constraint cap honored
    EXPECT_TRUE(same_channel(specs[0].chan,
                             constraint::build_channel_profile(
                                 constraint::ConstraintKind::Charge)));
    EXPECT_EQ(specs[1].kind, constraint::ConstraintKind::Spin);
    EXPECT_EQ(specs[1].atoms, std::vector<int>({1})); // default = list index
    EXPECT_DOUBLE_EQ(specs[1].target, -0.05);
    EXPECT_DOUBLE_EQ(specs[1].mu_max, 5.0); // run-level fallback
    EXPECT_TRUE(same_channel(specs[1].chan,
                             constraint::build_channel_profile(
                                 constraint::ConstraintKind::Spin)));

    // Factory contract (A0 D4): charge reads/sums both spins and injects
    // into both; spin reads the difference and injects with opposite signs.
    const constraint::ChannelProfile charge = constraint::build_channel_profile(
        constraint::ConstraintKind::Charge);
    EXPECT_EQ(charge.read_up, 1);
    EXPECT_EQ(charge.read_dn, 1);
    EXPECT_DOUBLE_EQ(charge.inj_up, 1.0);
    EXPECT_DOUBLE_EQ(charge.inj_dn, 1.0);
    const constraint::ChannelProfile spin = constraint::build_channel_profile(
        constraint::ConstraintKind::Spin);
    EXPECT_EQ(spin.read_up, 1);
    EXPECT_EQ(spin.read_dn, -1);
    EXPECT_DOUBLE_EQ(spin.inj_up, 1.0);
    EXPECT_DOUBLE_EQ(spin.inj_dn, -1.0);

    // Legacy v1 format + run-level type=spin -> auto-converted specs with a
    // deprecation warning (nested fragments preserved).
    const std::string v1 = R"({"targets": [0.1, -0.1], "atoms": [[0], [1, 2]]})";
    specs.clear();
    warnings.clear();
    error.clear();
    ASSERT_TRUE(constraint::parse_constraint_file(v1, "spin", 5.0, 3, 2,
                                                  specs, fmt, warnings, error))
        << error;
    EXPECT_EQ(fmt, constraint::ConstraintFileFormat::V1);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Spin);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0}));
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 5.0);
    EXPECT_EQ(specs[1].kind, constraint::ConstraintKind::Spin);
    EXPECT_EQ(specs[1].atoms, std::vector<int>({1, 2}));
    EXPECT_DOUBLE_EQ(specs[1].target, -0.1);
    EXPECT_TRUE(warning_has(warnings, "deprecat"));

    // v1 flat "atoms" list: one single-atom fragment per listed atom.
    const std::string flat = R"({"targets": [0.1, 0.2], "atoms": [2, 0]})";
    specs.clear();
    warnings.clear();
    error.clear();
    ASSERT_TRUE(constraint::parse_constraint_file(flat, "charge", 5.0, 3, 1,
                                                  specs, fmt, warnings, error))
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({2}));
    EXPECT_EQ(specs[1].atoms, std::vector<int>({0}));

    // v2 multi-atom fragment: "atoms": [0, 1] is ONE fragment (v2 contract),
    // unlike the v1 flat list where every atom is its own fragment.
    const std::string multi = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0, 1]}]})";
    specs.clear();
    warnings.clear();
    error.clear();
    ASSERT_TRUE(constraint::parse_constraint_file(multi, "charge", 5.0, 3, 1,
                                                  specs, fmt, warnings, error))
        << error;
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0, 1}));
}

TEST(ConstraintIOTest, MixedGuards)
{
    std::string error;
    constraint::ConstraintConfig cfg;

    // v2 spin constraint under nspin=1 -> ERROR (spin channel needs two).
    const std::string spin1 = R"({"constraints": [
        {"type": "spin", "target": 0.1, "atoms": [0]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", spin1, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);

    // "dipole" kind is not implemented in stage A -> ERROR.
    error.clear();
    const std::string dipole = R"({"constraints": [
        {"type": "dipole", "target": 0.1, "atoms": [0]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", dipole, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("dipole"), std::string::npos);
    EXPECT_NE(error.find("not implemented"), std::string::npos);

    // Empty "constraints" list -> ERROR (no implicit constraint).
    error.clear();
    const std::string empty = R"({"constraints": []})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", empty, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);

    // Atom index out of range -> ERROR.
    error.clear();
    const std::string oob = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [7]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", oob, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("out of range"), std::string::npos);

    // Empty fragment (atoms: []) -> ERROR.
    error.clear();
    const std::string empty_frag = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": []}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", empty_frag, 5.0, 1e-4,
                                               3, 1, error),
              constraint::ConfigStatus::ERROR);

    // Per-constraint mu_max <= 0 -> ERROR.
    error.clear();
    const std::string bad_mu = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0], "mu_max": 0.0}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", bad_mu, 5.0, 1e-4, 3,
                                               1, error),
              constraint::ConfigStatus::ERROR);

    // "constraints" and "targets" co-present -> ERROR (never guess).
    error.clear();
    const std::string both = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0]}], "targets": [0.1]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", both, 5.0, 1e-4, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);

    // Duplicate (kind, atoms): the same observable measured twice is
    // near-collinear -> WARNING, not a silent run.
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    constraint::ConstraintFileFormat fmt;
    error.clear();
    const std::string dup = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0]},
        {"type": "charge", "target": 0.2, "atoms": [0]}]})";
    ASSERT_TRUE(constraint::parse_constraint_file(dup, "charge", 5.0, 3, 1,
                                                  specs, fmt, warnings, error))
        << error;
    EXPECT_TRUE(warning_has(warnings, "duplicate"));

    // v2 mixed kinds parse at the spec layer ...
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string mixed = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0]},
        {"type": "spin", "target": 0.1, "atoms": [0]}]})";
    ASSERT_TRUE(constraint::parse_constraint_file(mixed, "charge", 5.0, 3, 2,
                                                  specs, fmt, warnings, error))
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Charge);
    EXPECT_EQ(specs[1].kind, constraint::ConstraintKind::Spin);
    // ... but the legacy 11-arg entry (which discards 'specs') still cannot
    // express a mixed run and refuses (the expressibility guard moved from
    // the extended core to this entry at A4 — the specs-bearing callers go
    // through configure_from_inputs and the stage-A loop instead).
    error.clear();
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", mixed, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("mixed"), std::string::npos);

    // A4: the extended core (specs out) SERVES a mixed kind run — the specs
    // list is what the stage-A loop consumes; this is the guard-removal
    // assertion (red while the staging guard rejects, green after A4).
    error.clear();
    specs.clear();
    warnings.clear();
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta",
                                               mixed, 5.0, 1e-4, 3, 2, error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Charge);
    EXPECT_EQ(specs[1].kind, constraint::ConstraintKind::Spin);

    // Heterogeneous per-constraint mu_max (A0 D3): served by the extended
    // core (each cap survives in its spec) ...
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string hetero = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0], "mu_max": 3.0},
        {"type": "charge", "target": -0.1, "atoms": [1], "mu_max": 0.2}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta",
                                               hetero, 5.0, 1e-4, 3, 1, error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 3.0);
    EXPECT_DOUBLE_EQ(specs[1].mu_max, 0.2);
    // ... and refused by the legacy single-cap entry (which can not carry
    // per-constraint caps).
    error.clear();
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", hetero, 5.0, 1e-4, 3,
                                               1, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("mu_max"), std::string::npos);

    // v2 + non-default run-level constraint_type -> WARNING "supersede"
    // (explicit run-level type is not silently ignored on a v2 file).
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string charge_only = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "spin", "becke", "delta",
                                               charge_only, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(warning_has(warnings, "supersede"));
    EXPECT_EQ(cfg.type, "charge"); // v2 per-constraint type wins

    // v2 single-kind runnable through the legacy cfg when expressible:
    // homogeneous per-constraint caps collapse onto the single cfg cap.
    error.clear();
    const std::string single_charge = R"({"constraints": [
        {"type": "charge", "target": 0.1, "atoms": [0]},
        {"type": "charge", "target": -0.1, "atoms": [1]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", single_charge, 5.0,
                                               1e-4, 3, 1, error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(cfg.targets.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.targets[0].value, 0.1);
    EXPECT_DOUBLE_EQ(cfg.targets[1].value, -0.1);
}
