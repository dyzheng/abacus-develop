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
 *  - type != "charge"/"spin" -> ERROR
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

} // namespace constraint

#endif
