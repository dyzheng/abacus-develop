#include "constraint_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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

// Parse either a scalar number or a three-component JSON array.  A scalar is
// mirrored onto the z component (the collinear limit of a spin vector), while
// is_vector records which form the file actually supplied.
bool parse_number_or_vector(const std::string& s,
                            size_t& pos,
                            ModuleBase::Vector3<double>& out,
                            bool& is_vector)
{
    skip_ws(s, pos);
    // Branch: a leading '[' selects the vector form; anything else must be a
    // plain scalar so a malformed vector can not silently degrade to z-only.
    if (pos < s.size() && s[pos] == '[')
    {
        ++pos;
        double values[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a)
        {
            skip_ws(s, pos);
            // Guard: every component of the vector must be a valid number.
            if (!parse_number(s, pos, values[a]))
            {
                return false;
            }
            skip_ws(s, pos);
            // Branch: the first two components require a comma separator;
            // the third requires the closing bracket.
            if (a < 2)
            {
                if (pos >= s.size() || s[pos] != ',')
                {
                    return false;
                }
                ++pos;
            }
        }
        skip_ws(s, pos);
        // Guard: an unclosed vector would consume the following JSON fields.
        if (pos >= s.size() || s[pos] != ']')
        {
            return false;
        }
        ++pos;
        out = ModuleBase::Vector3<double>(values[0], values[1], values[2]);
        is_vector = true;
        return true;
    }
    double value = 0.0;
    // Branch: scalar form -> collinear z-axis target/multiplier.
    if (!parse_number(s, pos, value))
    {
        return false;
    }
    out = ModuleBase::Vector3<double>(0.0, 0.0, value);
    is_vector = false;
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
    // Branch: global dipole coordinate string.
    if (kind == ConstraintKind::Dipole)
    {
        return "dipole";
    }
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
//   - "type" outside {charge, spin, dipole} -> ERROR
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
    // Branch B: implemented kinds only; anything else is refused loudly
    // rather than guessed.
    if (type == "charge")
    {
        spec.kind = ConstraintKind::Charge;
    }
    else if (type == "spin")
    {
        spec.kind = ConstraintKind::Spin;
    }
    else if (type == "dipole")
    {
        spec.kind = ConstraintKind::Dipole;
    }
    else
    {
        error = "constraint type \"" + type
                + "\" is not implemented (only \"charge\", \"spin\" and "
                  "\"dipole\")";
        return false;
    }
    spec.chan = build_channel_profile(spec.kind);
    // Branch: collinear spin constraints use the two-channel density, while
    // non-collinear runs carry a full spin vector.  Other spin counts have no
    // spin channel at all and are refused.
    if (spec.kind == ConstraintKind::Spin && nspin != 2 && nspin != 4)
    {
        error = "constraint c[" + std::to_string(index)
                + "]: type=spin requires nspin=2 or nspin=4 (the spin channel "
                  "reads and injects a collinear difference or a non-collinear "
                  "vector moment)";
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
    if (!parse_number_or_vector(obj, pos, spec.target_vec,
                                spec.target_vector))
    {
        error = "constraint c[" + std::to_string(index)
                + "]: malformed number or [x,y,z] vector in \"target\"";
        return false;
    }
    // Guard: only a non-collinear spin constraint can consume a vector
    // target; keeping the scalar mirror avoids a silent reinterpretation in
    // every pre-existing charge/collinear consumer.
    if (spec.target_vector
        && !(spec.kind == ConstraintKind::Spin && nspin == 4))
    {
        error = "constraint c[" + std::to_string(index)
                + "]: vector \"target\" requires type=spin with nspin=4";
        return false;
    }
    spec.target = spec.target_vec.z;

    // Branch B2: dipole-only frame fields.  They are parsed here, before the
    // atom-list branch, so a dipole can never silently acquire a fragment
    // (a global dipole is the only allowed form; multi-fragment dipoles are
    // underdetermined).
    if (spec.kind == ConstraintKind::Dipole)
    {
        pos = 0;
        if (find_key(obj, "frame", pos))
        {
            skip_ws(obj, pos);
            if (pos >= obj.size() || obj[pos] != ':')
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: expected ':' after \"frame\"";
                return false;
            }
            ++pos;
            if (!parse_string_token(obj, pos, spec.dipole_frame))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: \"frame\" must be a quoted string";
                return false;
            }
        }
        pos = 0;
        if (find_key(obj, "axis", pos))
        {
            skip_ws(obj, pos);
            if (pos >= obj.size() || obj[pos] != ':')
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: expected ':' after \"axis\"";
                return false;
            }
            ++pos;
            std::string axis;
            if (!parse_string_token(obj, pos, axis))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: \"axis\" must be a quoted string";
                return false;
            }
            // Branch: x/y/z map to Cartesian component indices.
            if (axis == "x")
            {
                spec.dipole_axis = 0;
            }
            else if (axis == "y")
            {
                spec.dipole_axis = 1;
            }
            else if (axis == "z")
            {
                spec.dipole_axis = 2;
            }
            else
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: \"axis\" must be \"x\", \"y\" or \"z\"";
                return false;
            }
        }
        pos = 0;
        if (find_key(obj, "saw_pos", pos))
        {
            skip_ws(obj, pos);
            if (pos >= obj.size() || obj[pos] != ':')
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: expected ':' after \"saw_pos\"";
                return false;
            }
            ++pos;
            skip_ws(obj, pos);
            if (!parse_number(obj, pos, spec.dipole_saw_pos))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: malformed number in \"saw_pos\"";
                return false;
            }
        }
        pos = 0;
        if (find_key(obj, "saw_pos_dec", pos))
        {
            skip_ws(obj, pos);
            if (pos >= obj.size() || obj[pos] != ':')
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: expected ':' after \"saw_pos_dec\"";
                return false;
            }
            ++pos;
            skip_ws(obj, pos);
            if (!parse_number(obj, pos, spec.dipole_saw_pos_dec))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: malformed number in \"saw_pos_dec\"";
                return false;
            }
        }
        pos = 0;
        if (find_key(obj, "unit", pos))
        {
            skip_ws(obj, pos);
            if (pos >= obj.size() || obj[pos] != ':')
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: expected ':' after \"unit\"";
                return false;
            }
            ++pos;
            if (!parse_string_token(obj, pos, spec.dipole_unit))
            {
                error = "constraint c[" + std::to_string(index)
                        + "]: \"unit\" must be a quoted string";
                return false;
            }
        }
    }

    // Branch D: optional "atoms" flat fragment list; default = atom 'index'
    // (the constraint position in the list), mirroring the v1 default.
    pos = 0;
    if (find_key(obj, "atoms", pos))
    {
        // Guard: a dipole is one global observable.  A per-constraint atom
        // fragment would silently redefine it as a fragment dipole.
        if (spec.kind == ConstraintKind::Dipole)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: type=dipole is a global constraint and must not "
                      "carry \"atoms\"";
            return false;
        }
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
        // Branch: dipole covers the whole cell; other kinds keep the v2
        // default fragment (atom at the list position).
        if (spec.kind == ConstraintKind::Dipole)
        {
            spec.atoms.resize(nat);
            for (int iat = 0; iat < nat; ++iat)
            {
                spec.atoms[iat] = iat;
            }
        }
        else
        {
            spec.atoms = {index};
        }
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

    // Branch F: optional "fixed_mu" (M2 C4).  The value is a Lagrange
    // multiplier in Ry, not a target; |value| is validated against the
    // component cap once the cap is known.
    pos = 0;
    if (find_key(obj, "fixed_mu", pos))
    {
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != ':')
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: expected ':' after \"fixed_mu\"";
            return false;
        }
        ++pos;
        skip_ws(obj, pos);
        if (!parse_number_or_vector(obj, pos, spec.fixed_mu_vec,
                                    spec.fixed_mu_vector))
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: malformed number or [x,y,z] vector in \"fixed_mu\"";
            return false;
        }
        // Guard: vector multipliers are meaningful only for the
        // non-collinear spin channel.
        if (spec.fixed_mu_vector
            && !(spec.kind == ConstraintKind::Spin && nspin == 4))
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: vector \"fixed_mu\" requires type=spin with nspin=4";
            return false;
        }
        spec.fixed_mu_value = spec.fixed_mu_vec.z;
        spec.fixed_mu = true;
    }

    // Branch G: dipole semantic guards.  These are independent of the cell
    // shape (the latter is checked in configure_from_inputs, which has the
    // UnitCell); refusing an unknown frame/axis here keeps a malformed file
    // from reaching the weight builder.
    if (spec.kind == ConstraintKind::Dipole)
    {
        if (spec.dipole_frame != "saw" && spec.dipole_frame != "molecular")
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: \"frame\" must be \"saw\" or \"molecular\"";
            return false;
        }
        // Branch: the saw implementation is tied to the z-aligned third
        // lattice vector; other axes are refused rather than approximating.
        if (spec.dipole_frame == "saw" && spec.dipole_axis != 2)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: frame=saw currently supports only axis=\"z\"";
            return false;
        }
        if (spec.dipole_saw_pos_dec <= 0.0
            || spec.dipole_saw_pos_dec > 0.5)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: \"saw_pos_dec\" must be in (0, 0.5]";
            return false;
        }
        if (spec.dipole_saw_pos < -1.0 || spec.dipole_saw_pos >= 1.0)
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: \"saw_pos\" must be in [-1, 1) (-1 = auto)";
            return false;
        }
        if (spec.dipole_unit != "ebohr" && spec.dipole_unit != "V")
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: \"unit\" must be \"ebohr\" or \"V\"";
            return false;
        }
        // Branch: a potential-drop target has no meaning in the molecular
        // frame (there is no periodic saw), so refuse the combination.
        if (spec.dipole_unit == "V" && spec.dipole_frame != "saw")
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: unit=\"V\" requires frame=\"saw\"";
            return false;
        }
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

    // Guard (M2 C4): fixed mu is a run-level dynamics mode.  A mixed list
    // would need a two-mode outer loop; refuse instead of silently freezing
    // only part of the run.
    int n_fixed = 0;
    for (const ConstraintSpec& s : specs)
    {
        if (s.fixed_mu)
        {
            ++n_fixed;
        }
    }
    if (n_fixed > 0 && n_fixed != static_cast<int>(specs.size()))
    {
        error = "constraint file: \"fixed_mu\" must be set for all "
                "constraints or none (mixed fixed/free runs are not "
                "implemented)";
        return false;
    }
    // Guard: a fixed value beyond the component cap is a user error, not a
    // value to clamp silently (the audit would otherwise report a mu that
    // was never applied).
    for (const ConstraintSpec& s : specs)
    {
        const double fixed_norm = s.fixed_mu_vector ? s.fixed_mu_vec.norm()
                                                    : std::abs(s.fixed_mu_value);
        // Branch: the per-component cap must bound the actual injected vector
        // norm, not only its z mirror.
        if (s.fixed_mu && fixed_norm > s.mu_max)
        {
            error = "constraint file: |fixed_mu| = "
                    + std::to_string(fixed_norm)
                    + " exceeds the component mu_max = "
                    + std::to_string(s.mu_max);
            return false;
        }
    }

    // Guard (M5): the dipole is a single global coordinate.  A mixed list or
    // a second dipole would require a decomposition the real-space weight
    // formulation does not define, so refuse instead of silently dropping
    // one component.
    int n_dipole = 0;
    for (const ConstraintSpec& s : specs)
    {
        if (s.kind == ConstraintKind::Dipole)
        {
            ++n_dipole;
        }
    }
    if (n_dipole > 0 && static_cast<int>(specs.size()) != 1)
    {
        error = "constraint file: a dipole constraint must be the only "
                "constraint (global single-dipole runs only)";
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
    // The dipole weight is also built from the total density (the ionic
    // offset is added separately by the observer), so it shares this profile.
    if (kind == ConstraintKind::Charge || kind == ConstraintKind::Dipole)
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
        // Branch: non-collinear runs accept the legacy scalar spin target as
        // the z-component; v2 files can carry the full vector instead.
        if (run_type == "spin" && nspin != 2 && nspin != 4)
        {
            error = "constraint_type=spin requires nspin=2 or nspin=4 (the "
                    "spin channel reads and injects a collinear difference or "
                    "a non-collinear vector moment)";
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
            spec.target_vec = ModuleBase::Vector3<double>(0.0, 0.0,
                                                          targets[i].value);
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
                                  const double step_max,
                                  const double step_probe,
                                  const double branch_tol,
                                  const int nat,
                                  const int nspin,
                                  std::string& error,
                                  const std::string& mu_schedule,
                                  const double inner_thr,
                                  const int inner_nmax)
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
    cfg.step_max = step_max;
    cfg.step_probe = step_probe;
    cfg.branch_tol = branch_tol;
    cfg.mu_schedule = mu_schedule;
    cfg.inner_thr = inner_thr;
    cfg.inner_nmax = inner_nmax;

    // Guard: validated recipes only.  A wrong recipe must abort loudly
    // rather than silently run an unvalidated partition.
    if (weight_type != "becke" && weight_type != "hirshfeld"
        && weight_type != "hirshfeld_periodic")
    {
        error = "constraint_weight_type=\"" + weight_type
                + "\" is not implemented (use \"becke\" or \"hirshfeld\")";
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
    // Guard: the outer-step cap must be positive, and the first-step probe
    // must be a genuine sub-cap (larger than it would defeat its purpose and
    // silently change which cap governs the first step).
    if (step_max <= 0.0)
    {
        error = "constraint_step_max must be > 0";
        return ConfigStatus::ERROR;
    }
    if (step_probe < 0.0 || step_probe > step_max)
    {
        error = "constraint_step_probe must satisfy 0 <= step_probe <= "
                "constraint_step_max (0 selects the legacy probe = step_max)";
        return ConfigStatus::ERROR;
    }
    // Guard: -1 is the explicit "auto" sentinel (resolved after the specs
    // are parsed); any other negative value has no meaning.  0 is the
    // explicit "guard off" value and must stay reachable.
    if (branch_tol < 0.0 && branch_tol != -1.0)
    {
        error = "constraint_branch_tol must be >= 0, or -1 for auto (spin: "
                "1e-3 Ry; charge/all-fixed: 0)";
        return ConfigStatus::ERROR;
    }
    // Guard: the schedule is an explicit vocabulary switch.
    // Anything else must abort rather than silently fall back to outer (the
    // user asked for a different dynamics and would not get it).
    if (mu_schedule != "outer" && mu_schedule != "inner"
        && mu_schedule != "alm")
    {
        error = "constraint_mu_schedule must be \"outer\", \"inner\" or "
                "\"alm\"";
        return ConfigStatus::ERROR;
    }
    // Guard (INNER only): both inner knobs must be usable.  A non-positive
    // gate would disable the dynamics the user asked for, and a non-positive
    // budget would trip the fallback on the first update; both are refused up
    // front instead of silently behaving like outer.  The outer schedule keeps
    // ignoring both values (legacy runs stay bit-identical).
    if (mu_schedule == "inner")
    {
        if (inner_thr <= 0.0)
        {
            error = "constraint_inner_thr must be > 0 when "
                    "constraint_mu_schedule=inner";
            return ConfigStatus::ERROR;
        }
        if (inner_nmax <= 0)
        {
            error = "constraint_inner_nmax must be > 0 when "
                    "constraint_mu_schedule=inner";
            return ConfigStatus::ERROR;
        }
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
        // Branch: the legacy run-level type also allows nspin=4, where its
        // scalar target denotes the z-component.
        if (type == "spin" && nspin != 2 && nspin != 4)
        {
            error = "constraint_type=spin requires nspin=2 or nspin=4 (the "
                    "spin channel reads and injects a collinear difference or "
                    "a non-collinear vector moment)";
            return ConfigStatus::ERROR;
        }
    }
    if (!parse_constraint_file(target_file_content, type, mu_max, nat, nspin,
                               specs, fmt, warnings, error))
    {
        return ConfigStatus::ERROR;
    }
    // Branch (E2): resolve the auto sentinel after the full list is known.
    // A spin constraint (non-fixed) gets the historical guarded default
    // 1e-3 Ry; charge-only and all-fixed lists stay off, so fixed-mu product
    // runs never trip the branch_tol/fixed_mu mutual-exclusion guard.
    if (branch_tol < 0.0)
    {
        bool has_spin = false;
        bool all_fixed = !specs.empty();
        for (const ConstraintSpec& spec : specs)
        {
            has_spin = has_spin || (spec.kind == ConstraintKind::Spin);
            all_fixed = all_fixed && spec.fixed_mu;
        }
        cfg.branch_tol = (has_spin && !all_fixed) ? 1e-3 : 0.0;
        if (cfg.branch_tol > 0.0)
        {
            warnings.push_back(
                "constraint_branch_tol=-1 (auto): spin channel selected the "
                "default 1e-3 Ry energy branch guard; set 0 to disable");
        }
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
        bool saw_dipole = false;
        for (const ConstraintSpec& spec : specs)
        {
            if (spec.kind == ConstraintKind::Spin)
            {
                saw_spin = true;
            }
            else if (spec.kind == ConstraintKind::Dipole)
            {
                saw_dipole = true;
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
        cfg.type = saw_dipole ? "dipole" : (saw_spin ? "spin" : "charge");
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
    // Legacy entry: this overload predates the run-level outer-step caps, so
    // it forwards the ConstraintConfig defaults (step_max 0.05 Ry, probe off)
    // -- exactly the behaviour its pre-A1 callers were written against.
    const ConstraintConfig defaults;
    const ConfigStatus st = configure_constraint(
        cfg, specs, warnings, enabled, type, weight_type, target_mode,
        target_file_content, mu_max, thr, defaults.step_max,
        defaults.step_probe, defaults.branch_tol, nat, nspin, error);
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
    bool saw_dipole = false;
    const double cap = specs.front().mu_max;
    for (size_t i = 0; i < specs.size(); ++i)
    {
        // Branch A: spin entry in the list.
        if (specs[i].kind == ConstraintKind::Spin)
        {
            saw_spin = true;
        }
        else if (specs[i].kind == ConstraintKind::Dipole)
        {
            saw_dipole = true;
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
    // Guard: the legacy cfg carries no dipole frame/axis fields, so this
    // entry can not arm the dipole coordinate without losing its semantics.
    if (saw_dipole)
    {
        error = "dipole constraints must use the specs-bearing "
                "configure_from_inputs entry (the legacy single-channel cfg "
                "has no frame/axis fields)";
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
        PARAM.inp.constraint_thr, PARAM.inp.constraint_step_max,
        PARAM.inp.constraint_step_probe, PARAM.inp.constraint_branch_tol,
        ucell.nat, PARAM.inp.nspin, error,
        PARAM.inp.constraint_mu_schedule, PARAM.inp.constraint_inner_thr,
        PARAM.inp.constraint_inner_nmax);
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

    // Branch: the non-collinear production closure is the fixed-vector
    // allocation path.  Adaptive vector-multiplier scheduling has not been
    // defined yet, so refuse it explicitly instead of falling through to a
    // scalar loop whose residual/target semantics do not apply.
    if (PARAM.inp.nspin == 4)
    {
        if (!PARAM.inp.constraint_alloc_switch)
        {
            error = "nspin = 4 constraints require "
                    "constraint_alloc_switch = true (the grid observer is "
                    "collinear-only)";
            return ConfigStatus::ERROR;
        }
        for (const ConstraintSpec& spec : specs)
        {
            // Branch: the non-collinear closure currently fixes every
            // multiplier.  A mixed fixed/free list would run the scalar
            // outer loop against vector spin targets, which is undefined.
            if (!spec.fixed_mu)
            {
                error = "nspin = 4 constraints currently require fixed_mu "
                        "(a vector for spin, scalar for charge); adaptive "
                        "vector-multiplier scheduling is not implemented yet";
                return ConfigStatus::ERROR;
            }
        }
    }

    // M5 dipole coordinate: frame/unit/cell guards and the V -> e*Bohr
    // conversion need the UnitCell, so they live here rather than in the
    // cell-free parser.
    bool has_dipole = false;
    bool all_fixed = !specs.empty();
    for (const ConstraintSpec& spec : specs)
    {
        has_dipole = has_dipole || (spec.kind == ConstraintKind::Dipole);
        all_fixed = all_fixed && spec.fixed_mu;
    }
    // Branch: a dipole coordinate shares the periodic field with Efield's
    // saw potential and the dipole correction; running both would count the
    // same external field twice, so refuse instead of silently adding them.
    if (has_dipole
        && (PARAM.inp.efield_amp != 0.0 || PARAM.inp.dip_cor_flag))
    {
        error = "a dipole constraint is mutually exclusive with "
                "efield_amp != 0 and dip_cor_flag = true (both add the same "
                "periodic electric field)";
        return ConfigStatus::ERROR;
    }
    if (has_dipole)
    {
        // Branch: non-collinear densities store four components while the
        // observer integrates only the total (one/two-component) density.
        if (PARAM.inp.nspin == 4)
        {
            error = "dipole constraints are not implemented for nspin = 4";
            return ConfigStatus::ERROR;
        }
        for (ConstraintSpec& spec : specs)
        {
            // Branch: only the dipole entry needs frame-specific work.
            if (spec.kind != ConstraintKind::Dipole)
            {
                continue;
            }
            // Branch: saw frame requires the z-aligned third lattice vector
            // assumed by the dipole / deltaV formulas.
            if (spec.dipole_frame == "saw")
            {
                const ModuleBase::Matrix3& lv = ucell.latvec;
                const double off = std::abs(lv.e13) + std::abs(lv.e23)
                                   + std::abs(lv.e31)
                                   + std::abs(lv.e32);
                if (off > 1e-10 || lv.e33 <= 0.0)
                {
                    error = "a saw dipole constraint requires a z-aligned "
                            "third lattice vector orthogonal to a,b";
                    return ConfigStatus::ERROR;
                }
            }
            // Branch: potential-drop targets are converted through
            // deltaV = 4*pi*p/A (Hartree), with A the xy cell area.
            if (spec.dipole_unit == "V")
            {
                const ModuleBase::Matrix3& lv = ucell.latvec;
                const double axb = lv.e11 * lv.e22 - lv.e12 * lv.e21;
                const double area = std::abs(axb) * ucell.lat0 * ucell.lat0;
                spec.target = spec.target * area
                              / (ModuleBase::FOUR_PI
                                 * ModuleBase::Hartree_to_eV);
                spec.dipole_unit = "ebohr";
            }
        }
        // Mirror the (possibly unit-converted) dipole target into the legacy
        // cfg.targets view used by reports and historical tests.
        for (size_t a = 0; a < specs.size() && a < cfg.targets.size(); ++a)
        {
            cfg.targets[a].value = specs[a].target;
        }
        // Guard: an all-fixed run injects before the first density exists, so
        // an auto saw plane would seed a nonzero mu through an arbitrary
        // placeholder plane.  Require the user to place it explicitly.
        if (all_fixed)
        {
            for (const ConstraintSpec& spec : specs)
            {
                if (spec.kind == ConstraintKind::Dipole
                    && spec.dipole_frame == "saw"
                    && spec.dipole_saw_pos < 0.0)
                {
                    error = "fixed_mu dipole saw constraints require an "
                            "explicit \"saw_pos\" (no density is available "
                            "before the first injection)";
                    return ConfigStatus::ERROR;
                }
            }
        }
    }

    cfg.warm_start = PARAM.inp.constraint_warm_start;
    // M4 ALM + scheduling pack plumbing (guards at the same site as the M1
    // dipole guards below).
    cfg.alm_k = PARAM.inp.constraint_alm_k;
    cfg.trust_carry_jacobian = PARAM.inp.constraint_trust_carry_jacobian;
    cfg.budget_freeze = PARAM.inp.constraint_budget_freeze;
    cfg.warm_start_running = PARAM.inp.constraint_warm_start_running;
    // Branch: ALM requested -> validate its gain and reject a dynamics mode
    // that has no outer multiplier at all (all components fixed).
    if (cfg.mu_schedule == "alm")
    {
        // Branch: negative gain has no physical meaning; 0 is the auto
        // sentinel and must stay accepted.
        if (cfg.alm_k < 0.0)
        {
            error = "constraint_alm_k must be >= 0 (0 = auto)";
            return ConfigStatus::ERROR;
        }
        bool all_fixed = !specs.empty();
        for (const ConstraintSpec& s : specs)
        {
            all_fixed = all_fixed && s.fixed_mu;
        }
        // Branch: fixed-mu runs have no outer loop, so ALM can never arm.
        if (all_fixed)
        {
            error = "constraint_mu_schedule=alm is incompatible with "
                    "\"fixed_mu\" (fixed-mu runs have no outer loop; use "
                    "outer/inner or remove fixed_mu)";
            return ConfigStatus::ERROR;
        }
    }
    // M3 C2/C3/E1 switches: independent of the target list, so they are
    // copied after the validated core just like warm_start / dipole.
    cfg.trust_region = PARAM.inp.constraint_trust_region;
    cfg.progressive_scf = PARAM.inp.constraint_progressive_scf;
    cfg.mag_anchor = PARAM.inp.constraint_mag_anchor;
    cfg.mag_tol = PARAM.inp.constraint_mag_tol;
    // Guard (E1): the magnetic anchor is a trust-region rejection policy;
    // without a radius there is no rollback mechanism to invoke.
    if (cfg.mag_anchor && !cfg.trust_region)
    {
        error = "constraint_mag_anchor=true requires "
                "constraint_trust_region=true (the magnetic anchor rejects "
                "a step through the trust region)";
        return ConfigStatus::ERROR;
    }
    // Guard (E1): a non-positive magnetization tolerance would reject every
    // step (including numerical noise) and livelock the rollback loop.
    if (cfg.mag_anchor && cfg.mag_tol <= 0.0)
    {
        error = "constraint_mag_tol must be > 0 when "
                "constraint_mag_anchor=true";
        return ConfigStatus::ERROR;
    }
    // Guard (E1): the anchor monitors the collinear total magnetization,
    // which only exists for nspin=2.
    if (cfg.mag_anchor && PARAM.inp.nspin != 2)
    {
        error = "constraint_mag_anchor requires nspin=2 (it monitors the "
                "collinear total magnetization)";
        return ConfigStatus::ERROR;
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
    // Module A (M1) dipole-report plumbing: passive observation, filled
    // straight from INPUT (no interaction with the target list).  The
    // legacy single-type configure entry leaves the defaults (report off),
    // which is exactly its pre-M1 behaviour.
    cfg.dipole_report = PARAM.inp.constraint_dipole_report;
    cfg.dipole_frame = PARAM.inp.constraint_dipole_frame;
    cfg.dipole_saw_pos_max = PARAM.inp.constraint_dipole_saw_pos_max;
    cfg.dipole_saw_pos_dec = PARAM.inp.constraint_dipole_saw_pos_dec;
    // Branch: report requested -> the frame and saw parameters must be
    // valid; anything else refuses to run rather than silently reporting a
    // wrong dipole (the same "loud guard" policy as the rest of this file).
    if (cfg.dipole_report)
    {
        // Branch A: unknown frame.
        if (cfg.dipole_frame != "molecular" && cfg.dipole_frame != "saw")
        {
            error = "constraint_dipole_frame must be \"molecular\" or "
                    "\"saw\", got \"" + cfg.dipole_frame + "\"";
            return ConfigStatus::ERROR;
        }
        // Branch B: saw-frame parameter sanity.
        if (cfg.dipole_frame == "saw"
            && (cfg.dipole_saw_pos_dec <= 0.0 || cfg.dipole_saw_pos_dec > 0.5
                || cfg.dipole_saw_pos_max >= 1.0
                || cfg.dipole_saw_pos_max < -1.0))
        {
            error = "constraint_dipole_saw_pos_dec must be in (0, 0.5] and "
                    "constraint_dipole_saw_pos_max in [-1, 1)";
            return ConfigStatus::ERROR;
        }
        // Branch B2: the saw frame requires a z-aligned third lattice
        // vector orthogonal to a,b (the dipole and deltaV formulas use
        // latvec(2,2) as the exact cell height).
        if (cfg.dipole_frame == "saw")
        {
            const ModuleBase::Matrix3& lv = ucell.latvec;
            const double off = std::abs(lv.e13) + std::abs(lv.e23)
                               + std::abs(lv.e31) + std::abs(lv.e32);
            // Branch: cell shape violates the saw-frame assumption.
            if (off > 1e-10 || lv.e33 <= 0.0)
            {
                error = "constraint_dipole_frame = saw requires a z-aligned "
                        "third lattice vector orthogonal to a,b";
                return ConfigStatus::ERROR;
            }
        }
        // Branch C: non-collinear spin (nspin == 4) reads a 4-component
        // density the reporter is not wired for -> refuse loudly.
        if (PARAM.inp.nspin == 4)
        {
            error = "constraint_dipole_report is not implemented for "
                    "nspin = 4";
            return ConfigStatus::ERROR;
        }
    }
    return st;
}

} // namespace constraint
