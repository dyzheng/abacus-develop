#ifndef ALLOC_PW_REPORT_H
#define ALLOC_PW_REPORT_H

#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

#include "source_psi/psi.h"

class UnitCell;
class K_Vectors;

namespace ModuleBase
{
class matrix;
}

namespace constraint
{
class WeightGrid;
}

namespace alloc
{

/**
 * @brief Read-only PW allocation audit for one converged wavefunction frame.
 *
 * Target rows are the onsite-projector shell selected by angular momentum on
 * every atom that carries it.  n_T/m_T are the orthonormalized target-space
 * occupations from the occupied PW bands; when a WeightGrid and density are
 * supplied, n/m are the matching per-atom real-space totals and n_c/m_c are
 * their complements.
 */
struct PwAllocReport
{
    int frame = 0;
    int target_l = 2;
    int nat = 0;
    int ntarget = 0;
    int nband = 0;
    int rank = 0;
    double cond = 1.0;
    std::vector<int> target_atom;
    std::vector<int> target_row;
    std::vector<double> n_T;
    std::vector<double> m_T;
    std::vector<double> n_c;
    std::vector<double> m_c;
    std::vector<double> n_tot;
    std::vector<double> m_tot;
    double sum_n_T = 0.0;
    double sum_m_T = 0.0;
    double sum_n = 0.0;
    double sum_m = 0.0;
    double nelec = 0.0;
    double dsum_n = 0.0;
    double tot_mag = 0.0;
    double dsum_m = 0.0;
    bool totals_available = false;
    bool supported = true;
    std::string skip_reason;
};

/**
 * @brief Build the PW allocation report from the current CPU wavefunctions.
 *
 * The collinear nspin=1/2 path is supported.  Non-collinear spinors are
 * reported as skipped because the target read-out currently sums scalar
 * spin channels only.  The onsite projector must have been initialized with a
 * positive onsite_radius and the same numerical orbitals used by LCAO.
 */
PwAllocReport report_pw_alloc(const UnitCell& ucell,
                              const psi::Psi<std::complex<double>>* psi,
                              const ModuleBase::matrix& occupations,
                              const K_Vectors& kv,
                              const int nspin,
                              const constraint::WeightGrid* weight_grid,
                              const double* const* rho,
                              const double nelec,
                              const int frame,
                              const int target_l);

// Emit the ALLOC-PW block.  Call on rank 0 after all pool reductions.
void print_pw_alloc_report(const PwAllocReport& report, std::ostream& os);

} // namespace alloc

#endif
