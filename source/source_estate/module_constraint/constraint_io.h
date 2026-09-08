#ifndef CONSTRAINT_IO_H
#define CONSTRAINT_IO_H

#include <string>
#include <vector>

#include "source_cell/unitcell.h"

namespace constraint
{

// One constraint (fragment): a target charge plus the atoms whose Becke
// weights define the constraint observable.
struct ConstraintTarget
{
    double value = 0.0;       // target in e (delta or absolute semantics)
    std::vector<int> atoms;   // fragment: global atom indices
};

// Fully validated constraint configuration (architecture layer M7).
struct ConstraintConfig
{
    bool enabled = false;
    std::string type = "charge";        // phase 1: only "charge"
    std::string weight_type = "becke";  // phase 1: only "becke"
    std::string target_mode = "delta";  // "delta" | "absolute"
    double mu_max = 5.0;                // Ry, cap on |mu|
    double thr = 1e-4;                  // e, per-constraint convergence
    std::vector<ConstraintTarget> targets;
};

enum class ConfigStatus
{
    DISABLED, // constraint switch off: nothing to do
    OK,       // validated, ready to run
    ERROR     // guard triggered: caller must WARNING_QUIT with 'error'
};

/**
 * @brief Build and guard the constraint configuration (M7).
 *
 * Semantic guards (never silently run a wrong calculation):
 *  - weight_type != "becke" -> ERROR ("not implemented in phase 1")
 *  - type != "charge"/"spin" -> ERROR (legacy v1 target files only: a v2
 *    file carries a per-constraint "type" and the run-level type only warns)
 *  - type == "spin" && nspin != 2 -> ERROR (the spin channel reads and
 *    injects the spin-difference density and requires a two-channel run)
 *  - no targets parsed      -> ERROR (no implicit constraint without target)
 *  - target_mode == "absolute" -> non-fatal WARNING (calibration scale
 *    differs from delta: charges ~0.2-0.3 e vs ~e shifts)
 *  - mu_max / thr sanity    -> ERROR
 *
 * @param target_file_content Content of the target JSON file (empty string
 *        when no file is configured).
 * @param nat Number of atoms, used to validate fragment indices.
 */
ConfigStatus configure_constraint(ConstraintConfig& cfg,
                                  const bool enabled,
                                  const std::string& type,
                                  const std::string& weight_type,
                                  const std::string& target_mode,
                                  const std::string& target_file_content,
                                  const double mu_max,
                                  const double thr,
                                  const int nat,
                                  const int nspin,
                                  std::string& error);

// Read a target file into a string; returns false with 'error' set on IO
// failure.
bool read_target_file(const std::string& path,
                      std::string& content,
                      std::string& error);

// Minimal JSON-subset parser for the target file:
//   { "targets": [v0, v1, ...],
//     "atoms": [[a, b], [c], ...] }        (nested fragments, optional)
// or "atoms": [a, b, c] (flat per-atom fragment list).
// Without "atoms", fragment i defaults to atom i.  Returns false with a
// message on structural errors.
bool parse_target_file(const std::string& content,
                       std::vector<ConstraintTarget>& targets,
                       std::string& error);

// Build the fully validated constraint configuration from the global INPUT
// (M7) plus the per-atom covalent-radius partition radii.  Shared by the PW
// and LCAO esolvers so both basis channels observe identical guards,
// defaults and radii (phase-1 technical debt: this ~50-line block used to
// be duplicated in each before_scf).
//
// Returns ConfigStatus::DISABLED when the constraint switch is off (caller
// may skip arming the loop), OK after a successful build, or ERROR with a
// human-readable message in 'error' (caller must WARNING_QUIT).  'radii'
// is filled with one radius per global atom (Bohr) only on success.
ConfigStatus configure_from_inputs(ConstraintConfig& cfg,
                                   const UnitCell& ucell,
                                   std::vector<double>& radii,
                                   std::string& error);


// ---------------------------------------------------------------------------
// Stage A mixed (charge + spin) constraint model (architecture layer M7).
// A single constraint list may now mix observable kinds; each entry carries
// its own fragment, density channel and per-constraint multiplier cap.
// ---------------------------------------------------------------------------

// Observable kind of one constraint.  Stage A implements only the total
// charge and the magnetization channels; "dipole" (z-weighted, redesign
// stage C) is rejected by the guards and never reaches this enum.
enum class ConstraintKind
{
    Charge, // Q = int w (rho_up + rho_dn)
    Spin    // Q = int w (rho_up - rho_dn); requires nspin == 2
};

// Per-constraint density read / potential injection signs.  Built only by
// build_channel_profile() and never hand-filled, so the three shared (w,
// chan) consumers (observer / injector / force kernel) can not drift apart.
struct ChannelProfile
{
    int read_up = 0;     // observer: coefficient of rho_up in Q
    int read_dn = 0;     // observer: coefficient of rho_dn in Q
    double inj_up = 0.0; // injector: coefficient of +mu*w on v_eff(up)
    double inj_dn = 0.0; // injector: coefficient of +mu*w on v_eff(dn)
};

// One validated constraint of the mixed list.  Internal order == JSON order
// (a v1 conversion keeps the "targets" order); the audit line keeps
// c[0..N-1] and gains a "kind=" tag in A4.
struct ConstraintSpec
{
    ConstraintKind kind = ConstraintKind::Charge;
    std::vector<int> atoms; // Becke fragment (global atom indices)
    ChannelProfile chan;    // == build_channel_profile(kind) (factory only)
    double target = 0.0;    // delta/absolute semantics set by the run-level
                            // target_mode (A0 D1: target_mode stays run-level)
    double mu_max = 5.0;    // per-constraint cap; omitted -> run-level cap
};

// File-format tag reported by the parser / configuration core.
enum class ConstraintFileFormat
{
    V1, // legacy {"targets": [...], "atoms": ...} + run-level type
    V2  // new {"constraints": [{type, target, atoms?, mu_max?}, ...]}
};

/**
 * @brief Derive the channel profile from a constraint kind (A0 D4 factory).
 *
 * charge: reads rho_up + rho_dn and injects +mu*w into both spin channels;
 * spin: reads rho_up - rho_dn and injects +mu*w / -mu*w into the up / down
 * channels (DeltaSpin semantics).  The factory is the single source of these
 * signs; ConstraintSpec.chan must always equal build_channel_profile(kind).
 */
ChannelProfile build_channel_profile(ConstraintKind kind);

/**
 * @brief Parse a target-file content into validated ConstraintSpecs.
 *
 * Accepts both file formats: the v2 "constraints" list natively, and the
 * legacy v1 "targets" file auto-converted (kind = channel of the run-level
 * "type", mu_max = the run-level cap; a deprecation warning is recorded).
 * Per-constraint guards: unknown type (incl. "dipole") -> error; spin under
 * nspin != 2 -> error; missing/empty/out-of-range atoms -> error; mu_max
 * <= 0 -> error; an empty list -> error.  Duplicate (kind, atoms) entries
 * (same observable measured twice, near-collinear) produce a warning.
 *
 * @param run_type    Run-level constraint_type.  Only meaningful for v1
 *                    files (kind of the whole list); must be "charge"/"spin".
 * @param run_mu_max  Fallback cap for constraints that omit "mu_max".
 * @param format      Filled with the detected file format.
 * @param warnings    Human-readable warnings (deprecation / duplicates),
 *                    one entry each; never fatal.
 * @return false with 'error' set on structural or guard failure.
 */
bool parse_constraint_file(const std::string& content,
                           const std::string& run_type,
                           const double run_mu_max,
                           const int nat,
                           const int nspin,
                           std::vector<ConstraintSpec>& specs,
                           ConstraintFileFormat& format,
                           std::vector<std::string>& warnings,
                           std::string& error);

// Extended core of configure_constraint: also emits the fully validated
// ConstraintSpec list and the non-fatal warnings (v1 deprecation, v2
// "supersede" of the run-level type, absolute-mode calibration, duplicate
// near-collinear entries).  Stage-A wiring: configure_from_inputs uses this
// core and prints the warnings once per run; the overload above is kept as
// the legacy single-channel entry (all pre-A1 tests exercise it unchanged).
ConfigStatus configure_constraint(ConstraintConfig& cfg,
                                  std::vector<ConstraintSpec>& specs,
                                  std::vector<std::string>& warnings,
                                  const bool enabled,
                                  const std::string& type,
                                  const std::string& weight_type,
                                  const std::string& target_mode,
                                  const std::string& target_file_content,
                                  const double mu_max,
                                  const double thr,
                                  const int nat,
                                  const int nspin,
                                  std::string& error);
} // namespace constraint
#endif
