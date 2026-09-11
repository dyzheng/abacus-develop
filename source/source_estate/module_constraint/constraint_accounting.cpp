#include "constraint_accounting.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace constraint
{

ConstraintAudit ConstraintAccounting::audit(const WeightGrid& wg,
                                            const std::vector<double>& mu,
                                            const std::vector<double>& Q,
                                            const std::vector<double>& target,
                                            const double nelec)
{
    // Legacy entry: no per-constraint kind information (no kind= tokens).
    std::vector<ConstraintKind> kinds;
    return audit(wg, mu, Q, target, nelec, kinds);
}

ConstraintAudit ConstraintAccounting::audit(const WeightGrid& wg,
                                            const std::vector<double>& mu,
                                            const std::vector<double>& Q,
                                            const std::vector<double>& target,
                                            const double nelec,
                                            const std::vector<ConstraintKind>& kinds)
{
    // No on-site moments supplied: the onsite= token stays absent.
    return audit(wg, mu, Q, target, nelec, kinds, std::vector<double>());
}

ConstraintAudit ConstraintAccounting::audit(const WeightGrid& wg,
                                            const std::vector<double>& mu,
                                            const std::vector<double>& Q,
                                            const std::vector<double>& target,
                                            const double nelec,
                                            const std::vector<ConstraintKind>& kinds,
                                            const std::vector<double>& onsite_per_atom)
{
    ConstraintAudit a;
    a.nelec = nelec;
    a.maxdev = wg.max_partition_deviation();
    const int n = static_cast<int>(Q.size());
    a.Q = Q;
    a.target = target;
    a.mu = mu;
    a.kinds = kinds;
    a.residual.assign(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        const double res = Q[i] - target[i];
        a.residual[i] = res;
        a.e_con += mu[i] * res;
        a.max_residual = std::max(a.max_residual, std::abs(res));
        a.total_charge += Q[i];
    }
    // Optional on-site moment per constraint: the fragment sum mirrors how
    // Q_alpha is defined (a sum over the fragment's weight field), so the two
    // numbers describe the same atoms under the two different projectors.
    if (!onsite_per_atom.empty()
        && static_cast<int>(wg.constraint_atoms().size()) == n)
    {
        a.onsite.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (const int iat : wg.constraint_atoms()[i])
            {
                // Branch A: in-range atom index — accumulate its moment.
                if (iat >= 0 && static_cast<size_t>(iat) < onsite_per_atom.size())
                {
                    a.onsite[i] += onsite_per_atom[iat];
                }
                else
                {
                    // Branch B: out-of-range index (caller bug) — mark the
                    // whole token as unavailable instead of a partial sum.
                    a.onsite.clear();
                    break;
                }
            }
            if (a.onsite.empty())
            {
                break;
            }
        }
    }
    return a;
}

std::string ConstraintAccounting::audit_line(const ConstraintAudit& a)
{
    std::ostringstream os;
    os << "CONSTRAINT_AUDIT nconstraint=" << a.Q.size() << " ";
    os << "e_con=" << std::setprecision(10) << a.e_con << " ";
    os << "max_residual=" << std::setprecision(10) << a.max_residual << " ";
    os << "total_charge=" << std::setprecision(10) << a.total_charge << " ";
    os << "nelec=" << std::setprecision(10) << a.nelec << " ";
    os << "maxdev=" << std::setprecision(10) << a.maxdev;
    // Per-constraint detail lines follow on separate lines.
    for (size_t i = 0; i < a.Q.size(); ++i)
    {
        os << "\nCONSTRAINT_AUDIT c[" << i << "]";
        // Kind label (M5, stage A): present only when the caller supplied
        // the per-constraint kinds (the legacy audit call keeps the
        // historical output unchanged).
        if (a.kinds.size() == a.Q.size())
        {
            os << " kind=" << kind_to_type_string(a.kinds[i]);
        }
        os << " q=" << std::setprecision(10) << a.Q[i] << " t=" << a.target[i]
           << " mu=" << a.mu[i] << " res=" << a.residual[i];
        // On-site projected moment (fragment sum), present only when the
        // caller supplied per-atom moments.
        if (a.onsite.size() == a.Q.size())
        {
            os << " onsite=" << std::setprecision(10) << a.onsite[i];
        }
    }
    return os.str();
}

} // namespace constraint
