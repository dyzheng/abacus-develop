#include "constraint_io.h"

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
    cfg = ConstraintConfig();
    if (!enabled)
    {
        return ConfigStatus::DISABLED;
    }
    cfg.enabled = true;
    cfg.type = type;
    cfg.weight_type = weight_type;
    cfg.target_mode = target_mode;
    cfg.mu_max = mu_max;
    cfg.thr = thr;

    // Guard: phase-1/2 recipes only.  A wrong recipe must abort loudly
    // rather than silently run an unvalidated partition.
    if (weight_type != "becke")
    {
        error = "constraint_weight_type=\"" + weight_type
                + "\" is not implemented in phase 1 (only \"becke\")";
        return ConfigStatus::ERROR;
    }
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
        ModuleBase::WARNING("ConstraintIO",
                            "constraint_target_mode=absolute: targets are absolute "
                            "charges in e; the calibration scale differs from delta "
                            "mode, verify your targets are on this scale");
    }

    // Parse the target file.  A missing target is a hard error: implicit
    // constraints without an explicit target are forbidden (DeltaP 4.3).
    if (target_file_content.empty())
    {
        error = "constraint is enabled but constraint_target_file is empty";
        return ConfigStatus::ERROR;
    }
    if (!parse_target_file(target_file_content, cfg.targets, error))
    {
        return ConfigStatus::ERROR;
    }
    if (cfg.targets.empty())
    {
        error = "constraint is enabled but no targets were parsed";
        return ConfigStatus::ERROR;
    }

    // Validate fragment atom indices against the cell.
    for (const ConstraintTarget& t : cfg.targets)
    {
        if (t.atoms.empty())
        {
            error = "constraint target has an empty atom fragment";
            return ConfigStatus::ERROR;
        }
        for (const int iat : t.atoms)
        {
            if (iat < 0 || iat >= nat)
            {
                error = "constraint fragment atom index out of range: "
                        + std::to_string(iat) + " (nat=" + std::to_string(nat)
                        + ")";
                return ConfigStatus::ERROR;
            }
        }
    }
    return ConfigStatus::OK;
}

ConfigStatus configure_from_inputs(ConstraintConfig& cfg,
                                   const UnitCell& ucell,
                                   std::vector<double>& radii,
                                   std::string& error)
{
    cfg = ConstraintConfig();
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
    const ConfigStatus st = configure_constraint(
        cfg, PARAM.inp.constraint, PARAM.inp.constraint_type,
        PARAM.inp.constraint_weight_type, PARAM.inp.constraint_target_mode,
        content, PARAM.inp.constraint_mu_max, PARAM.inp.constraint_thr,
        ucell.nat, PARAM.inp.nspin, error);
    if (st == ConfigStatus::ERROR)
    {
        return st;
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
