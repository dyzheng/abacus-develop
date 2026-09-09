#include "constraint_io.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "source_base/constants.h"
#include "source_base/element_covalent_radius.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"
#include "source_io/module_parameter/parameter.h"

namespace constraint
{

bool read_target_file(const std::string& path,
                      std::string& content,
                      std::string& error)
{
    std::ifstream ifs(path);
    if (!ifs)
    {
        error = "cannot open constraint target file: " + path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    content = oss.str();
    return true;
}

namespace
{

// Skip spaces, tabs, newlines and carriage returns.
void skip_ws(const std::string& s, size_t& pos)
{
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
    {
        ++pos;
    }
}

// Parse a double from position pos; advances pos past the number.
bool parse_number(const std::string& s, size_t& pos, double& out)
{
    const size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+'))
    {
        ++pos;
    }
    bool digits = false;
    while (pos < s.size()
           && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.'))
    {
        if (std::isdigit(static_cast<unsigned char>(s[pos])))
        {
            digits = true;
        }
        ++pos;
    }
    if (!digits)
    {
        return false;
    }
    out = std::stod(s.substr(start, pos - start));
    return true;
}

// Parse an integer from position pos; advances pos past the number.
bool parse_int(const std::string& s, size_t& pos, int& out)
{
    const size_t start = pos;
    if (pos < s.size() && s[pos] == '-')
    {
        ++pos;
    }
    bool digits = false;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])))
    {
        digits = true;
        ++pos;
    }
    if (!digits)
    {
        return false;
    }
    out = std::stoi(s.substr(start, pos - start));
    return true;
}

// Locate the next '"key"' occurrence at or after pos.
bool find_key(const std::string& s, const std::string& key, size_t& pos)
{
    const std::string quoted = "\"" + key + "\"";
    const size_t hit = s.find(quoted, pos);
    if (hit == std::string::npos)
    {
        return false;
    }
    pos = hit + quoted.size();
    return true;
}

} // anonymous namespace

bool parse_target_file(const std::string& content,
                       std::vector<ConstraintTarget>& targets,
                       std::string& error)
{
    targets.clear();
    size_t pos = 0;

    // "targets" value: array of numbers.
    if (!find_key(content, "targets", pos))
    {
        error = "target file misses the \"targets\" array";
        return false;
    }
    skip_ws(content, pos);
    if (pos >= content.size() || content[pos] != ':')
    {
        error = "target file: expected ':' after \"targets\"";
        return false;
    }
    ++pos;
    skip_ws(content, pos);
    if (pos >= content.size() || content[pos] != '[')
    {
        error = "target file: expected '[' after \"targets\":";
        return false;
    }
    ++pos;
    std::vector<double> values;
    for (;;)
    {
        skip_ws(content, pos);
        if (pos >= content.size())
        {
            error = "target file: unterminated \"targets\" array";
            return false;
        }
        if (content[pos] == ']')
        {
            ++pos;
            break;
        }
        double v = 0.0;
        if (!parse_number(content, pos, v))
        {
            error = "target file: malformed number in \"targets\"";
            return false;
        }
        values.push_back(v);
        skip_ws(content, pos);
        if (pos < content.size() && content[pos] == ',')
        {
            ++pos;
        }
    }

    // Optional "atoms": nested or flat list of fragment atom indices.
    std::vector<std::vector<int>> fragments;
    pos = 0;
    if (find_key(content, "atoms", pos))
    {
        skip_ws(content, pos);
        if (pos >= content.size() || content[pos] != ':')
        {
            error = "target file: expected ':' after \"atoms\"";
            return false;
        }
        ++pos;
        skip_ws(content, pos);
        if (pos >= content.size() || content[pos] != '[')
        {
            error = "target file: expected '[' after \"atoms\":";
            return false;
        }
        ++pos;
        skip_ws(content, pos);
        if (pos < content.size() && content[pos] == '[')
        {
            // Nested fragments: [[a, b], [c], ...].
            while (true)
            {
                skip_ws(content, pos);
                if (pos >= content.size() || content[pos] != '[')
                {
                    break;
                }
                ++pos;
                std::vector<int> frag;
                skip_ws(content, pos);
                if (pos < content.size() && content[pos] == ']')
                {
                    // Empty fragment: rejected below as a validation error.
                    ++pos;
                    fragments.push_back(frag);
                    skip_ws(content, pos);
                    if (pos < content.size() && content[pos] == ',')
                    {
                        ++pos;
                    }
                    continue;
                }
                for (;;)
                {
                    skip_ws(content, pos);
                    int a = 0;
                    if (!parse_int(content, pos, a))
                    {
                        error = "target file: malformed atom index in \"atoms\"";
                        return false;
                    }
                    frag.push_back(a);
                    skip_ws(content, pos);
                    if (pos < content.size() && content[pos] == ',')
                    {
                        ++pos;
                    }
                    else
                    {
                        break;
                    }
                }
                if (pos >= content.size() || content[pos] != ']')
                {
                    error = "target file: unterminated fragment in \"atoms\"";
                    return false;
                }
                ++pos;
                fragments.push_back(frag);
                skip_ws(content, pos);
                if (pos < content.size() && content[pos] == ',')
                {
                    ++pos;
                }
            }
            if (pos >= content.size() || content[pos] != ']')
            {
                error = "target file: unterminated \"atoms\" array";
                return false;
            }
        }
        else
        {
            // Flat per-atom list: [a, b, c].
            for (;;)
            {
                skip_ws(content, pos);
                int a = 0;
                if (!parse_int(content, pos, a))
                {
                    error = "target file: malformed atom index in \"atoms\"";
                    return false;
                }
                fragments.push_back({a});
                skip_ws(content, pos);
                if (pos < content.size() && content[pos] == ',')
                {
                    ++pos;
                }
                else if (pos < content.size() && content[pos] == ']')
                {
                    ++pos;
                    break;
                }
                else
                {
                    error = "target file: unterminated flat \"atoms\" list";
                    return false;
                }
            }
        }
    }

    // Assemble targets.  Without fragments, constraint i defaults to atom i.
    if (values.empty())
    {
        error = "target file: \"targets\" must contain at least one value";
        return false;
    }
    if (!fragments.empty() && fragments.size() != values.size())
    {
        error = "target file: \"atoms\" fragment count must match \"targets\"";
        return false;
    }
    targets.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        targets[i].value = values[i];
        targets[i].atoms = fragments.empty() ? std::vector<int>{static_cast<int>(i)}
                                             : fragments[i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage A: mixed constraint parsing helpers.
// ---------------------------------------------------------------------------

// Kind <-> run-level type-string ("charge" | "spin").  The cfg.type kept for
// the legacy single-channel loop must stay on this vocabulary (the loop maps
// it through channel_from_type()); the audit serialization (M5) and
// diagnostics share this same mapping so the machine-readable kind token can
// never drift from the config vocabulary.
const char* kind_to_type_string(ConstraintKind kind)
{
    // Branch: spin channel string.
    if (kind == ConstraintKind::Spin)
    {
        return "spin";
    }
    // Branch: charge channel string (default).
    return "charge";
}

namespace
{

// Detect the file schema from the presence of the top-level keys; a file
// mixing "constraints" (v2) and "targets" (v1) is a hard error: guessing the
// intended semantics would silently run a wrong constraint list.
bool detect_format(const std::string& content,
                   ConstraintFileFormat& format,
                   std::string& error)
{
    size_t pos = 0;
    const bool has_constraints = find_key(content, "constraints", pos);
    pos = 0;
    const bool has_targets = find_key(content, "targets", pos);
    if (has_constraints && has_targets)
    {
        error = "constraint target file mixes the v2 \"constraints\" list with "
                "the legacy \"targets\" array; use exactly one format";
        return false;
    }
    if (has_constraints)
    {
        format = ConstraintFileFormat::V2;
        return true;
    }
    if (has_targets)
    {
        format = ConstraintFileFormat::V1;
        return true;
    }
    error = "constraint target file must contain a \"constraints\" list (v2) "
            "or a legacy \"targets\" array (v1)";
    return false;
}

// Read one quoted string token after the current position; advances past the
// closing quote.  The white-listed schema has no escaped quotes inside the
// token values, so a plain scan is sufficient.
bool parse_string_token(const std::string& s, size_t& pos, std::string& out)
{
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"')
    {
        return false;
    }
    ++pos;
    const size_t start = pos;
    while (pos < s.size() && s[pos] != '"')
    {
        ++pos;
    }
    if (pos >= s.size())
    {
        return false;
    }
    out = s.substr(start, pos - start);
    ++pos;
    return true;
}

// Validate one per-constraint object of the v2 list.  Fields may appear in
// any order; each one is located by an independent key scan of the object
// slice.  Guarded (never silently run a wrong calculation):
//   - "type" outside {charge, spin} (incl. "dipole") -> ERROR
//   - spin under nspin != 2 -> ERROR (the spin channel needs two channels)
//   - missing "type"/"target" -> ERROR
//   - "atoms": empty fragment, out-of-range index, or nested arrays -> ERROR
//   - "mu_max" <= 0 -> ERROR; omitted -> run-level fallback
bool parse_constraint_object(const std::string& obj,
                             const int index,
                             const int nat,
                             const int nspin,
                             const double run_mu_max,
                             ConstraintSpec& spec,
                             std::string& error)
{
    // Branch A: "type" key (required).
    size_t pos = 0;
    if (!find_key(obj, "type", pos))
    {
        error = "constraint c[" + std::to_string(index)
                + "] misses the required \"type\"";
        return false;
    }
    skip_ws(obj, pos);
    if (pos >= obj.size() || obj[pos] != ':')
    {
        error = "constraint c[" + std::to_string(index)
                + "]: expected ':' after \"type\"";
        return false;
    }
    ++pos;
    std::string type;
    if (!parse_string_token(obj, pos, type))
    {
        error = "constraint c[" + std::to_string(index)
                + "]: \"type\" must be a quoted string";
        return false;
    }
    // Branch B: implemented kinds only; anything else (incl. "dipole") is
    // refused loudly rather than guessed.
    if (type == "charge")
    {
        spec.kind = ConstraintKind::Charge;
    }
    else if (type == "spin")
    {
        spec.kind = ConstraintKind::Spin;
    }
    else
    {
        error = "constraint type \"" + type
                + "\" is not implemented in stage A (only \"charge\" and "
                  "\"spin\")";
        return false;
    }
    spec.chan = build_channel_profile(spec.kind);
    if (spec.kind == ConstraintKind::Spin && nspin != 2)
    {
        error = "constraint c[" + std::to_string(index)
                + "]: type=spin requires nspin=2 (the spin channel reads and "
                  "injects the spin-difference density rho_up - rho_dn)";
        return false;
    }

    // Branch C: "target" key (required).
    pos = 0;
    if (!find_key(obj, "target", pos))
    {
        error = "constraint c[" + std::to_string(index)
                + "] misses the required \"target\"";
        return false;
    }
    skip_ws(obj, pos);
    if (pos >= obj.size() || obj[pos] != ':')
    {
        error = "constraint c[" + std::to_string(index)
                + "]: expected ':' after \"target\"";
        return false;
    }
    ++pos;
    skip_ws(obj, pos);
    if (!parse_number(obj, pos, spec.target))
    {
        error = "constraint c[" + std::to_string(index)
                + "]: malformed number in \"target\"";
        return false;
    }

    // Branch D: optional "atoms" flat fragment list; default = atom 'index'
    // (the constraint position in the list), mirroring the v1 default.
    pos = 0;
    if (find_key(obj, "atoms", pos))
    {
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != ':')
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: expected ':' after \"atoms\"";
            return false;
        }
        ++pos;
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != '[')
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: expected '[' after \"atoms\":";
            return false;
        }
        ++pos;
        std::vector<int> atoms;
        for (;;)
        {
            skip_ws(obj, pos);
            if (pos >= obj.size())
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: unterminated \"atoms\" array";
                return false;
            }
            if (obj[pos] == ']')
            {
                ++pos;
                break;
            }
            int a = 0;
            if (!parse_int(obj, pos, a))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: \"atoms\" must be a flat list of atom indices "
                          "(nested fragments are a legacy v1 form)";
                return false;
            }
            atoms.push_back(a);
            skip_ws(obj, pos);
            if (pos < obj.size() && obj[pos] == ',')
            {
                ++pos;
            }
        }
        if (atoms.empty())
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: empty atom fragment";
            return false;
        }
        spec.atoms = atoms;
    }
    else
    {
        spec.atoms = {index};
    }
    for (const int iat : spec.atoms)
    {
        if (iat < 0 || iat >= nat)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: fragment atom index out of range: "
                    + std::to_string(iat) + " (nat=" + std::to_string(nat)
                    + ")";
            return false;
        }
    }

    // Branch E: optional per-constraint "mu_max"; omitted -> run-level cap.
    pos = 0;
    if (find_key(obj, "mu_max", pos))
    {
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != ':')
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: expected ':' after \"mu_max\"";
            return false;
        }
        ++pos;
        skip_ws(obj, pos);
        double cap = 0.0;
        if (!parse_number(obj, pos, cap))
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: malformed number in \"mu_max\"";
            return false;
        }
        if (cap <= 0.0)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: \"mu_max\" must be > 0";
            return false;
        }
        spec.mu_max = cap;
    }
    else
    {
        spec.mu_max = run_mu_max;
    }
    return true;
}

// Parse the v2 "constraints" object list into validated specs (JSON order).
// The white-listed schema allows objects with flat atoms arrays only; nested
// objects / arrays are rejected.
bool parse_v2_list(const std::string& content,
                   const int nat,
                   const int nspin,
                   const double run_mu_max,
                   std::vector<ConstraintSpec>& specs,
                   std::string& error)
{
    size_t pos = 0;
    if (!find_key(content, "constraints", pos))
    {
        error = "constraint file misses the \"constraints\" array";
        return false;
    }
    skip_ws(content, pos);
    if (pos >= content.size() || content[pos] != ':')
    {
        error = "constraint file: expected ':' after \"constraints\"";
        return false;
    }
    ++pos;
    skip_ws(content, pos);
    if (pos >= content.size() || content[pos] != '[')
    {
        error = "constraint file: expected '[' after \"constraints\":";
        return false;
    }
    ++pos;
    int index = 0;
    for (;;)
    {
        skip_ws(content, pos);
        if (pos >= content.size())
        {
            error = "constraint file: unterminated \"constraints\" array";
            return false;
        }
        if (content[pos] == ']')
        {
            ++pos;
            break;
        }
        if (content[pos] != '{')
        {
            error = "constraint file: expected a \"{...}\" constraint object";
            return false;
        }
        // Extract the matching object slice (quotes are skipped so a closing
        // brace inside a string token can not terminate the object early).
        const size_t begin = pos;
        bool in_string = false;
        while (pos < content.size())
        {
            const char c = content[pos];
            if (in_string)
            {
                if (c == '"')
                {
                    in_string = false;
                }
            }
            else if (c == '"')
            {
                in_string = true;
            }
            else if (c == '}')
            {
                break;
            }
            ++pos;
        }
        if (pos >= content.size())
        {
            error = "constraint file: unterminated constraint object c["
                    + std::to_string(index) + "]";
            return false;
        }
        const std::string obj = content.substr(begin, pos - begin + 1);
        ++pos;
        ConstraintSpec spec;
        if (!parse_constraint_object(obj, index, nat, nspin, run_mu_max, spec,
                                     error))
        {
            return false;
        }
        specs.push_back(spec);
        ++index;
        skip_ws(content, pos);
        if (pos >= content.size())
        {
            error = "constraint file: unterminated \"constraints\" array";
            return false;
        }
        if (content[pos] == ',')
        {
            ++pos;
        }
        else if (content[pos] != ']')
        {
            error = "constraint file: expected ',' or ']' after a constraint "
                    "object";
            return false;
        }
    }
    if (specs.empty())
    {
        error = "constraint file: the \"constraints\" list must not be empty";
        return false;
    }
    return true;
}

} // namespace

// Factory of the per-kind channel profile (A0 D4).  The signs are the single
// contract shared by the observer / injector / force kernel consumers; a
// ConstraintSpec.chan must always equal build_channel_profile(spec.kind).
ChannelProfile build_channel_profile(ConstraintKind kind)
{
    // Branch: charge channel couples to the total density rho_up + rho_dn.
    if (kind == ConstraintKind::Charge)
    {
        return ChannelProfile{1, 1, 1.0, 1.0};
    }
    // Branch: spin channel couples to the magnetization rho_up - rho_dn
    // (DeltaSpin semantics: up gets +mu*w, down gets -mu*w).
    return ChannelProfile{1, -1, 1.0, -1.0};
}

bool parse_constraint_file(const std::string& content,
                           const std::string& run_type,
                           const double run_mu_max,
                           const int nat,
                           const int nspin,
                           std::vector<ConstraintSpec>& specs,
                           ConstraintFileFormat& format,
                           std::vector<std::string>& warnings,
                           std::string& error)
{
    specs.clear();
    warnings.clear();
    ConstraintFileFormat fmt;
    if (!detect_format(content, fmt, error))
    {
        return false;
    }
    format = fmt;
    if (fmt == ConstraintFileFormat::V2)
    {
        // Branch V2: per-constraint objects carry their own type / atoms /
        // mu_max; the run-level constraint_type plays no role here.
        if (!parse_v2_list(content, nat, nspin, run_mu_max, specs, error))
        {
            return false;
        }
    }
    else
    {
        // Branch V1: the legacy file shares one channel given by the
        // run-level type; convert each parsed target into a spec and record
        // the deprecation (the file format still works unchanged).
        if (run_type != "charge" && run_type != "spin")
        {
            error = "constraint_type=\"" + run_type
                    + "\" is not implemented in phase 2 (only \"charge\" and "
                      "\"spin\"; legacy target files need a run-level "
                      "channel)";
            return false;
        }
        if (run_type == "spin" && nspin != 2)
        {
            error = "constraint_type=spin requires nspin=2 (the spin channel "
                    "reads and injects the spin-difference density rho_up - "
                    "rho_dn)";
            return false;
        }
        std::vector<ConstraintTarget> targets;
        if (!parse_target_file(content, targets, error))
        {
            return false;
        }
        const ConstraintKind kind = (run_type == "spin")
                                        ? ConstraintKind::Spin
                                        : ConstraintKind::Charge;
        for (size_t i = 0; i < targets.size(); ++i)
        {
            ConstraintSpec spec;
            spec.kind = kind;
            spec.chan = build_channel_profile(kind);
            spec.atoms = targets[i].atoms;
            spec.target = targets[i].value;
            spec.mu_max = run_mu_max;
            if (spec.atoms.empty())
            {
                error = "constraint target has an empty atom fragment";
                return false;
            }
            for (const int iat : spec.atoms)
            {
                if (iat < 0 || iat >= nat)
                {
                    error = "constraint fragment atom index out of range: "
                            + std::to_string(iat) + " (nat="
                            + std::to_string(nat) + ")";
                    return false;
                }
            }
            specs.push_back(spec);
        }
        if (specs.empty())
        {
            error = "constraint is enabled but no targets were parsed";
            return false;
        }
        warnings.push_back("deprecation: the legacy \"targets\" target-file "
                           "format is deprecated; migrate to the v2 "
                           "\"constraints\" list (per-constraint type and "
                           "mu_max)");
    }

    // Near-collinear warning: two constraints of the same kind on the same
    // fragment read the same observable (the secant system is singular).
    // Fragment order is normalized so [0, 1] and [1, 0] count as identical.
    for (size_t i = 0; i < specs.size(); ++i)
    {
        for (size_t j = i + 1; j < specs.size(); ++j)
        {
            if (specs[i].kind != specs[j].kind)
            {
                continue;
            }
            std::vector<int> a = specs[i].atoms;
            std::vector<int> b = specs[j].atoms;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a == b)
            {
                warnings.push_back("duplicate (kind, atoms) constraints c["
                                   + std::to_string(i) + "] and c["
                                   + std::to_string(j)
                                   + "]: identical fragments measure the same "
                                     "observable (near-collinear); check the "
                                     "constraint list");
            }
        }
    }
    return true;
}

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
                                  std::string& error)
{
    cfg = ConstraintConfig();
    specs.clear();
    warnings.clear();
    if (!enabled)
    {
        return ConfigStatus::DISABLED;
    }
    cfg.enabled = true;
    cfg.weight_type = weight_type;
    cfg.target_mode = target_mode;
    cfg.thr = thr;

    // Guard: phase-1/2 recipes only.  A wrong recipe must abort loudly
    // rather than silently run an unvalidated partition.
    if (weight_type != "becke")
    {
        error = "constraint_weight_type=\"" + weight_type
                + "\" is not implemented in phase 1 (only \"becke\")";
        return ConfigStatus::ERROR;
    }
    if (target_mode != "delta" && target_mode != "absolute")
    {
        error = "constraint_target_mode must be \"delta\" or \"absolute\"";
        return ConfigStatus::ERROR;
    }
    if (mu_max <= 0.0)
    {
        error = "constraint_mu_max must be > 0";
        return ConfigStatus::ERROR;
    }
    if (thr <= 0.0)
    {
        error = "constraint_thr must be > 0";
        return ConfigStatus::ERROR;
    }
    // Absolute mode: explicit warning that the calibration scale differs
    // from delta mode (charges ~0.2-0.3 e instead of ~e shifts).
    if (target_mode == "absolute")
    {
        warnings.push_back("constraint_target_mode=absolute: targets are "
                           "absolute charges in e; the calibration scale "
                           "differs from delta mode, verify your targets are "
                           "on this scale");
    }

    // A missing target file is a hard error: implicit constraints without an
    // explicit target are forbidden (DeltaP 4.3).
    if (target_file_content.empty())
    {
        error = "constraint is enabled but constraint_target_file is empty";
        return ConfigStatus::ERROR;
    }
    ConstraintFileFormat fmt;
    if (!detect_format(target_file_content, fmt, error))
    {
        return ConfigStatus::ERROR;
    }
    // Branch V1: the run-level channel guards apply to legacy files (their
    // whole list shares one channel); a v2 file carries per-constraint types
    // instead and only a "supersede" warning is recorded below.
    if (fmt == ConstraintFileFormat::V1)
    {
        if (type != "charge" && type != "spin")
        {
            error = "constraint_type=\"" + type
                    + "\" is not implemented in phase 2 (only \"charge\" and "
                      "\"spin\")";
            return ConfigStatus::ERROR;
        }
        if (type == "spin" && nspin != 2)
        {
            error = "constraint_type=spin requires nspin=2 (the spin channel "
                    "reads and injects the spin-difference density rho_up - "
                    "rho_dn)";
            return ConfigStatus::ERROR;
        }
    }
    if (!parse_constraint_file(target_file_content, type, mu_max, nat, nspin,
                               specs, fmt, warnings, error))
    {
        return ConfigStatus::ERROR;
    }
    if (fmt == ConstraintFileFormat::V1)
    {
        // Legacy cfg keeps the run-level channel and cap verbatim.
        cfg.type = type;
        cfg.mu_max = mu_max;
    }
    else
    {
        // Branch V2: per-constraint types supersede the run-level
        // constraint_type; an explicit non-default setting is reported, never
        // silently dropped.
        if (type != "charge")
        {
            warnings.push_back("per-constraint types supersede "
                               "constraint_type=\"" + type
                               + "\"; the run-level constraint_type only "
                                 "applies to legacy v1 target files");
        }
        bool saw_charge = false;
        bool saw_spin = false;
        for (const ConstraintSpec& spec : specs)
        {
            if (spec.kind == ConstraintKind::Spin)
            {
                saw_spin = true;
            }
            else
            {
                saw_charge = true;
            }
        }
        // Legacy single-type mirror (A4): cfg.type / cfg.mu_max summarize
        // the validated list for consumers that can not read 'specs'
        // (legacy loop entry, historical tests).  The specs output is the
        // authority for a mixed kind run and for heterogeneous per-constraint
        // caps; the mirror only reports a representative kind and cap.  The
        // legacy 11-arg entry below re-guards what it can not express.
        cfg.type = saw_spin ? "spin" : "charge";
        cfg.mu_max = specs.front().mu_max;
    }

    // Fill the legacy single-target list used by the (pre-A4) loop chain;
    // per-constraint channels live in 'specs'.
    for (const ConstraintSpec& spec : specs)
    {
        ConstraintTarget t;
        t.value = spec.target;
        t.atoms = spec.atoms;
        cfg.targets.push_back(t);
    }
    return ConfigStatus::OK;
}

// Legacy single-channel entry: pre-A1 callers keep the exact signature.  The
// specs and warnings produced by the extended core are discarded here; the
// esolver-facing configure_from_inputs uses the extended core and reports
// the warnings once per run.
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
                                  std::string& error)
{
    std::vector<ConstraintSpec> specs;
    std::vector<std::string> warnings;
    const ConfigStatus st = configure_constraint(
        cfg, specs, warnings, enabled, type, weight_type, target_mode,
        target_file_content, mu_max, thr, nat, nspin, error);
    if (st != ConfigStatus::OK)
    {
        return st;
    }
    // Expressibility guard (moved from the extended core at A4): this entry
    // returns only the legacy single-type cfg, which can not carry a mixed
    // kind list or heterogeneous per-constraint caps.  Refusing here keeps a
    // caller that discards 'specs' from silently running a wrong channel —
    // the specs-bearing configure_from_inputs / stage-A loop is required.
    bool saw_charge = false;
    bool saw_spin = false;
    const double cap = specs.front().mu_max;
    for (size_t i = 0; i < specs.size(); ++i)
    {
        // Branch A: spin entry in the list.
        if (specs[i].kind == ConstraintKind::Spin)
        {
            saw_spin = true;
        }
        else
        {
            // Branch B: charge entry.
            saw_charge = true;
        }
        if (i > 0 && specs[i].mu_max != cap)
        {
            error = "per-constraint \"mu_max\" differs between c[0] and c["
                    + std::to_string(i)
                    + "]: the legacy single-cap configure entry can not honor "
                      "it; use the specs-bearing configure_from_inputs "
                      "(stage-A loop wiring)";
            return ConfigStatus::ERROR;
        }
    }
    if (saw_charge && saw_spin)
    {
        error = "mixed charge+spin constraint runs can not be expressed by "
                "the legacy single-channel configure entry; use the "
                "specs-bearing configure_from_inputs (stage-A loop wiring)";
        return ConfigStatus::ERROR;
    }
    return st;
}

ConfigStatus configure_from_inputs(ConstraintConfig& cfg,
                                   std::vector<ConstraintSpec>& specs,
                                   const UnitCell& ucell,
                                   std::vector<double>& radii,
                                   std::string& error)
{
    cfg = ConstraintConfig();
    specs.clear();
    radii.clear();
    if (!PARAM.inp.constraint)
    {
        return ConfigStatus::DISABLED;
    }

    // Shared phase-1/2 guard: only the CPU / double / non-double-grid path
    // is wired.  Anything else refuses to run rather than silently
    // producing a wrong constraint potential (the injector writes the
    // in-place potential buffers read by the Hamiltonian builders of both
    // basis channels).
    if (PARAM.globalv.double_grid || PARAM.inp.precision == "single"
        || PARAM.inp.device == "gpu")
    {
        error = "constraint framework requires double precision on CPU "
                "without double_grid";
        return ConfigStatus::ERROR;
    }

    std::string content;
    if (!PARAM.inp.constraint_target_file.empty()
        && !read_target_file(PARAM.inp.constraint_target_file, content, error))
    {
        return ConfigStatus::ERROR;
    }
    std::vector<std::string> warnings;
    const ConfigStatus st = configure_constraint(
        cfg, specs, warnings, PARAM.inp.constraint,
        PARAM.inp.constraint_type, PARAM.inp.constraint_weight_type,
        PARAM.inp.constraint_target_mode, content, PARAM.inp.constraint_mu_max,
        PARAM.inp.constraint_thr, ucell.nat, PARAM.inp.nspin, error);
    if (st == ConfigStatus::ERROR)
    {
        return st;
    }
    // Non-fatal notices (format deprecation / superseded run-level type /
    // absolute-mode calibration / near-collinear duplicates) go to the
    // warning stream once per run and never touch the result output.
    for (const std::string& w : warnings)
    {
        ModuleBase::WARNING("ConstraintIO", w);
    }

    // Partition radii from the covalent-radius table (Angstrom -> Bohr).
    radii.assign(ucell.nat, 0.0);
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            const auto it_rad
                = ModuleBase::CovalentRadius.find(ucell.atoms[it].label);
            radii[iat] = (it_rad != ModuleBase::CovalentRadius.end())
                             ? it_rad->second / ModuleBase::BOHR_TO_A
                             : 1.0 / ModuleBase::BOHR_TO_A;
            ++iat;
        }
    }
    return st;
}

} // namespace constraint
