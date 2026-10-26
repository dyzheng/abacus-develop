#ifndef ALLOC_LCAO_REPORT_H
#define ALLOC_LCAO_REPORT_H

#include <iosfwd>
#include <string>
#include <vector>

#include "source_estate/module_constraint/alloc_lcao.h"

class UnitCell;

namespace constraint
{
class WeightGrid;
}

namespace hamilt
{
template <typename T>
class HContainer;
}

namespace alloc
{

/**
 * @brief Read-only LCAO allocation audit for one gamma-only geometry.
 *
 * The target space is selected by angular momentum (default l=2) on every
 * atom that carries at least one such NAO shell.  n_T/m_T come from the
 * density matrix through the orthonormalized target coefficients; n/m come
 * from the real-space density split by the per-atom Becke weights.  The
 * complementary channel is reported as n_c=n-n_T and m_c=m-m_T.
 */
struct LcaoAllocReport
{
    int frame = 0;
    int norb = 0;
    int ntarget = 0;
    int target_l = 2;
    int nat = 0;
    int rank = 0;
    double cond = 1.0;
    std::vector<int> target_atom;
    std::vector<int> target_orbital;
    std::vector<double> n_T;
    std::vector<double> m_T;
    std::vector<double> n_c;
    std::vector<double> m_c;
    std::vector<double> n_tot;
    std::vector<double> m_tot;
    double sum_n = 0.0;
    double nelec = 0.0;
    double dsum_n = 0.0;
    double sum_m = 0.0;
    double tot_mag = 0.0;
    double dsum_m = 0.0;
    bool supported = true;
    std::string skip_reason;
};

/**
 * @brief Build the LCAO allocation report from the current DMR and S(R).
 *
 * Only the gamma-only, collinear (nspin 1/2), real-HContainer path is
 * implemented in S2.  A periodic multi-k/non-collinear request returns a
 * report with supported=false and a reason instead of silently contracting
 * an R-nonzero HContainer as if it were a single dense matrix.
 */
LcaoAllocReport report_lcao_alloc(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& S,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const constraint::WeightGrid& wg,
    const double* const* rho,
    const int nspin,
    const double nelec,
    const int frame,
    const int target_l);

// Emit the ALLOC block to the running log.  Call on rank 0 after the report
// has been assembled from pool-reduced values.
void print_lcao_alloc_report(const LcaoAllocReport& report, std::ostream& os);

} // namespace alloc

#endif
