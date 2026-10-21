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

TEST(ConstraintIOTest, OuterStepCapAndBranchGuardGuards)
{
    // II-1 / online branch guard: the run-level outer-step caps
    // (constraint_step_max / constraint_step_probe) and the energy branch
    // guard tolerance (constraint_branch_tol) live on the specs-bearing
    // overload.  They must land in the validated config, and a malformed
    // value must refuse loudly instead of silently running with an
    // unintended first step or an armed-but-broken branch guard.
    const std::string json = R"({"targets": [0.1], "atoms": [[0]]})";
    std::string error;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    // Defaults: legacy behaviour (step_max 0.05 Ry, probe off, guard off).
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.step_max, 0.05);
    EXPECT_DOUBLE_EQ(cfg.step_probe, 0.0);
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 0.0); // guard default off
    // An explicit probe is carried through verbatim.
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, 0.004, 0.0, 3, 1,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.step_probe, 0.004);
    // An explicit branch tolerance is carried through verbatim.
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, 0.0, 1e-3, 3, 1,
                                               error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 1e-3);
    // Probe larger than the cap: it would silently change which cap governs
    // the first step, so refuse.
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, 0.06, 0.0, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_step_probe"), std::string::npos);
    // Non-positive step cap: no step could ever be taken.
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.0, 0.0, 0.0, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_step_max"), std::string::npos);
    // Negative probe: refuse.
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, -1e-3, 0.0, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_step_probe"), std::string::npos);
    // Negative branch tolerance: refuse (0 is the explicit "guard off").
    EXPECT_EQ(constraint::configure_constraint(cfg, specs, warnings, true,
                                               "charge", "becke", "delta", json,
                                               5.0, 1e-4, 0.05, 0.0, -1e-9, 3, 1,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_branch_tol"), std::string::npos);
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

    // M5: the dipole coordinate is a single global constraint; an atom
    // fragment would silently turn it into an undefined fragment dipole.
    error.clear();
    const std::string dipole = R"({"constraints": [
        {"type": "dipole", "target": 0.1, "atoms": [0]}]})";
    EXPECT_EQ(constraint::configure_constraint(cfg, true, "charge", "becke",
                                               "delta", dipole, 5.0, 1e-4, 3, 2,
                                               error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("dipole"), std::string::npos);
    EXPECT_NE(error.find("atoms"), std::string::npos);

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
                                               mixed, 5.0, 1e-4, 0.05, 0.0, 0.0,
                                               3, 2, error),
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
                                               hetero, 5.0, 1e-4, 0.05, 0.0, 0.0,
                                               3, 1, error),
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
                                               charge_only, 5.0, 1e-4, 0.05,
                                               0.0, 0.0, 3, 2, error),
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

TEST(ConstraintIOTest, DipoleConstraintParsingAndGuards)
{
    std::string error;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    const std::string json = R"({"constraints": [
        {"type":"dipole","target":0.25,"axis":"z","frame":"saw",
         "saw_pos":0.7,"saw_pos_dec":0.08,"mu_max":0.05}]})";
    ASSERT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta", json,
                  5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].kind, constraint::ConstraintKind::Dipole);
    EXPECT_EQ(specs[0].atoms, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(specs[0].dipole_frame, "saw");
    EXPECT_EQ(specs[0].dipole_axis, 2);
    EXPECT_DOUBLE_EQ(specs[0].dipole_saw_pos, 0.7);
    EXPECT_DOUBLE_EQ(specs[0].dipole_saw_pos_dec, 0.08);
    EXPECT_DOUBLE_EQ(specs[0].mu_max, 0.05);
    EXPECT_EQ(cfg.type, "dipole");

    // Molecular x-axis is accepted and carries its axis index.
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string molecular = R"({"constraints": [
        {"type":"dipole","target":1.0,"axis":"x","frame":"molecular"}]})";
    ASSERT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta",
                  molecular, 5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(specs[0].dipole_axis, 0);
    EXPECT_EQ(specs[0].dipole_frame, "molecular");

    // Mixed charge+dipole -> ERROR (one global dipole coordinate only).
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string mixed = R"({"constraints": [
        {"type":"charge","target":0.1,"atoms":[0]},
        {"type":"dipole","target":0.25,"saw_pos":0.7}]})";
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta",
                  mixed, 5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("only constraint"), std::string::npos);

    // Saw frame currently only implements the z axis.
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string saw_x = R"({"constraints": [
        {"type":"dipole","target":0.25,"axis":"x","frame":"saw"}]})";
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta",
                  saw_x, 5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("axis"), std::string::npos);

    // V targets are meaningful only for the periodic saw frame.
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string volt_mol = R"({"constraints": [
        {"type":"dipole","target":0.25,"frame":"molecular","unit":"V"}]})";
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta",
                  volt_mol, 5.0, 1e-4, 0.05, 0.0, 0.0, 3, 1, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("V"), std::string::npos);

    // Automatic saw placement cannot serve the fixed-mu product mode: the
    // first injection happens before any density exists.
    error.clear();
    specs.clear();
    warnings.clear();
    const std::string fixed_auto = R"({"constraints": [
        {"type":"dipole","target":0.25,"fixed_mu":-0.01}]})";
    constraint::ConstraintFileFormat fmt;
    ASSERT_TRUE(constraint::parse_constraint_file(
        fixed_auto, "charge", 5.0, 3, 1, specs, fmt, warnings, error))
        << error;
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_dipole_fixed_auto.json";
    {
        std::ofstream ofs(path);
        ofs << fixed_auto;
    }
    PARAM.input.constraint_target_file = path;
    std::vector<double> radii;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("saw_pos"), std::string::npos);
    std::remove(path.c_str());
    PARAM.input.constraint = false;
    PARAM.input.constraint_target_file.clear();
    PARAM.input.efield_amp = 0.0;
    PARAM.input.dip_cor_flag = false;
}

TEST(ConstraintIOTest, DipoleVoltTargetConverted)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "absolute";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    PARAM.input.efield_amp = 0.0;
    PARAM.input.dip_cor_flag = false;
    const std::string path = "constraint_target_dipole_volt.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"dipole","target":1.0,"frame":"saw","saw_pos":0.9,
             "unit":"V"}]})";
    }
    PARAM.input.constraint_target_file = path;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;
    ASSERT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 1u);
    // 1 V potential drop -> p = A*dV/(4*pi*Hartree_to_eV).
    const double area = 20.0 * 20.0;
    const double expected = area
                            / (ModuleBase::FOUR_PI
                               * ModuleBase::Hartree_to_eV);
    EXPECT_NEAR(specs[0].target, expected, 1e-12);
    EXPECT_EQ(specs[0].dipole_unit, "ebohr");
    ASSERT_EQ(cfg.targets.size(), 1u);
    EXPECT_NEAR(cfg.targets[0].value, expected, 1e-12);
    std::remove(path.c_str());
    PARAM.input.constraint = false;
    PARAM.input.constraint_target_file.clear();
    PARAM.input.constraint_target_mode = "delta";
}

TEST(ConstraintIOTest, DipoleReportPlumbing)
{
    auto ucell = make_h2o_ucell(); // O, H, H -> 3 atoms
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_dipole_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;

    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Default off: the flag is false and the frames carry their defaults.
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.dipole_report);
    EXPECT_EQ(cfg.dipole_frame, "molecular");
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_max, -1.0);
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_dec, 0.1);

    // Enabled with saw frame: values flow through.
    PARAM.input.constraint_dipole_report = true;
    PARAM.input.constraint_dipole_frame = "saw";
    PARAM.input.constraint_dipole_saw_pos_max = 0.45;
    PARAM.input.constraint_dipole_saw_pos_dec = 0.08;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(cfg.dipole_report);
    EXPECT_EQ(cfg.dipole_frame, "saw");
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_max, 0.45);
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_dec, 0.08);

    // Guard: unknown frame -> ERROR.
    PARAM.input.constraint_dipole_frame = "circular";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("dipole_frame"), std::string::npos);

    // Guard: saw decay width out of (0, 0.5] -> ERROR.
    PARAM.input.constraint_dipole_frame = "saw";
    PARAM.input.constraint_dipole_saw_pos_dec = 0.6;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_dipole_saw_pos_dec = 0.1;

    // Guard: saw pos_max outside [-1, 1) -> ERROR.
    PARAM.input.constraint_dipole_saw_pos_max = 1.5;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_dipole_saw_pos_max = -1.0;

    // Guard: saw requires a z-aligned third lattice vector.
    ucell->latvec.e13 = 0.1;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("z-aligned"), std::string::npos);
    ucell->latvec.e13 = 0.0;

    // Guard: non-collinear spin not implemented -> ERROR.
    PARAM.input.nspin = 4;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);
    PARAM.input.nspin = 1;
    PARAM.input.constraint_dipole_report = false;
    PARAM.input.constraint_dipole_frame = "molecular";
    std::remove(path.c_str());
}

TEST(ConstraintIOTest, WarmStartPlumbing)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_warm_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Default: warm start is on (MD/relax only; the loop enforces the scope).
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(cfg.warm_start);
    PARAM.input.constraint_warm_start = false;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.warm_start);
    std::remove(path.c_str());
}

TEST(ConstraintIOTest, FixedMuParsingAndGuards)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "absolute";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_fixed_mu_test.json";
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // All constraints fixed -> values flow into the specs.
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.12},
            {"type":"charge","target":5.0,"atoms":[1,2],"fixed_mu":0.08}]})";
    }
    PARAM.input.constraint_target_file = path;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_TRUE(specs[0].fixed_mu);
    EXPECT_TRUE(specs[1].fixed_mu);
    EXPECT_DOUBLE_EQ(specs[0].fixed_mu_value, -0.12);
    EXPECT_DOUBLE_EQ(specs[1].fixed_mu_value, 0.08);

    // Mixed fixed/free -> ERROR (phase-1 C4 keeps one dynamics mode per run).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.12},
            {"type":"charge","target":5.0,"atoms":[1,2]}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("fixed_mu"), std::string::npos);

    // Fixed value outside the component cap -> ERROR (loud, no silent clamp).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-7.5}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("fixed_mu"), std::string::npos);

    // v1 files can not express fixed_mu: the legacy path stays untouched.
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_FALSE(specs[0].fixed_mu);
    std::remove(path.c_str());
}

TEST(ConstraintIOTest, BranchTolAutoForSpin)
{
    // E2: -1 is the auto sentinel.  A non-fixed spin list selects the
    // historical 1e-3 Ry guard; charge and all-fixed lists stay off.  An
    // explicit 0 always disables it, and other negative values are refused.
    const std::string v1 = R"({"targets": [0.1], "atoms": [[0]]})";
    const std::string v2_fixed = R"({"constraints": [
        {"type":"spin","target":0.1,"atoms":[0],"fixed_mu":0.1}]})";
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<std::string> warnings;
    std::string error;

    // Spin + auto -> 1e-3 Ry.
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "spin", "becke", "delta", v1,
                  5.0, 1e-4, 0.05, 0.0, -1.0, 3, 2, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 1e-3);

    // Charge + auto -> 0.
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "delta", v1,
                  5.0, 1e-4, 0.05, 0.0, -1.0, 3, 1, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 0.0);

    // Explicit 0 still disables the spin default.
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "spin", "becke", "delta", v1,
                  5.0, 1e-4, 0.05, 0.0, 0.0, 3, 2, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 0.0);

    // All-fixed spin (v2) + auto -> 0: fixed-mu product runs must not trip
    // the branch_tol/fixed_mu mutual-exclusion guard.
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "charge", "becke", "absolute",
                  v2_fixed, 5.0, 1e-4, 0.05, 0.0, -1.0, 3, 2, error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.branch_tol, 0.0);

    // Other negative sentinels are not "auto": refuse loudly.
    EXPECT_EQ(constraint::configure_constraint(
                  cfg, specs, warnings, true, "spin", "becke", "delta", v1,
                  5.0, 1e-4, 0.05, 0.0, -2.0, 3, 2, error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_branch_tol"), std::string::npos);
}

TEST(ConstraintIOTest, TrustProgressiveMagPlumbing)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    PARAM.input.nspin = 2;
    const std::string path = "constraint_target_m3_plumbing.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;

    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Defaults: all M3 behavior switches off.
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.trust_region);
    EXPECT_FALSE(cfg.progressive_scf);
    EXPECT_FALSE(cfg.mag_anchor);
    EXPECT_DOUBLE_EQ(cfg.mag_tol, 0.5);

    // Explicit values flow through the INPUT pipe.
    PARAM.input.constraint_trust_region = true;
    PARAM.input.constraint_progressive_scf = true;
    PARAM.input.constraint_mag_anchor = true;
    PARAM.input.constraint_mag_tol = 0.3;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(cfg.trust_region);
    EXPECT_TRUE(cfg.progressive_scf);
    EXPECT_TRUE(cfg.mag_anchor);
    EXPECT_DOUBLE_EQ(cfg.mag_tol, 0.3);

    // E1 requires the trust-region rollback machinery.
    PARAM.input.constraint_trust_region = false;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_mag_anchor"), std::string::npos);

    // A non-positive tolerance would reject numerical noise forever.
    PARAM.input.constraint_trust_region = true;
    PARAM.input.constraint_mag_tol = 0.0;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("constraint_mag_tol"), std::string::npos);

    std::remove(path.c_str());
}

TEST(ConstraintIOTest, AlmPlumbingAndGuards)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    // Isolate the global PARAM state from the preceding M3 plumbing test.
    PARAM.input.constraint_trust_region = false;
    PARAM.input.constraint_progressive_scf = false;
    PARAM.input.constraint_mag_anchor = false;
    PARAM.input.constraint_mag_tol = 0.5;
    const std::string path = "constraint_target_alm_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // mu_schedule = alm is accepted and flows through; k default = auto.
    PARAM.input.constraint_mu_schedule = "alm";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg.mu_schedule, "alm");
    EXPECT_DOUBLE_EQ(cfg.alm_k, 0.0); // 0 = auto (k = 0.5/|chi|)

    // Explicit k flows through.
    PARAM.input.constraint_alm_k = 3.0;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.alm_k, 3.0);
    PARAM.input.constraint_alm_k = 0.0;

    // Guard: alm + negative k -> ERROR.
    PARAM.input.constraint_alm_k = -1.0;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_alm_k = 0.0;

    // Guard: alm + json fixed_mu (all-fixed) -> ERROR (one dynamics mode).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.1}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("alm"), std::string::npos);

    // Scheduling flags flow through with defaults.
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.trust_carry_jacobian);
    EXPECT_TRUE(cfg.budget_freeze);
    EXPECT_TRUE(cfg.warm_start_running);
    PARAM.input.constraint_mu_schedule = "outer";
    std::remove(path.c_str());
}
