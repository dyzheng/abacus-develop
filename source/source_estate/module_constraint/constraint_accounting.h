#ifndef CONSTRAINT_ACCOUNTING_H
#define CONSTRAINT_ACCOUNTING_H

#include <string>
#include <vector>

#include "weight_grid.h"

namespace constraint
{

// Diagnostic summary of one constraint audit (M5).
struct ConstraintAudit
{
    double e_con = 0.0;        // E_con = sum_alpha mu_alpha (Q_alpha - t_alpha)
    double max_residual = 0.0; // max_alpha |Q_alpha - t_alpha|
    double total_charge = 0.0; // sum_alpha Q_alpha
    double nelec = 0.0;        // expected total charge (caller-supplied)
    double maxdev = 0.0;       // M1 partition-of-unity audit, max over grid
    std::vector<double> Q;     // observed charges (copy)
    std::vector<double> target;
    std::vector<double> mu;
    std::vector<double> residual;
};

/**
 * @brief Constraint energy accounting and audit line (architecture layer M5).
 *
 * The constraint energy correction
 *   E_con = sum_alpha mu_alpha (Q_alpha - t_alpha)
 * follows the DeltaSpin escon form and is added to the electronic total
 * energy by the outer loop.  The audit line is machine readable
 * (key=value) so validation scripts (V1/V2/V3) can parse it directly.
 *
 * total_charge = sum_alpha Q_alpha equals the true electron count sum_I N_I
 * only when the constraint fragments partition all atoms (the default
 * one-constraint-per-atom map does).
 */
class ConstraintAccounting
{
  public:
    // Build the audit record from the observed charges, targets and mu.
    static ConstraintAudit audit(const WeightGrid& wg,
                                 const std::vector<double>& mu,
                                 const std::vector<double>& Q,
                                 const std::vector<double>& target,
                                 const double nelec);

    // One-line machine-readable summary (key=value tokens).
    static std::string audit_line(const ConstraintAudit& a);
};

} // namespace constraint

#endif
