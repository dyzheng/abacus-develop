#include "source_pw/module_pwdft/alloc_pw_report.h"

#include "source_base/global_variable.h"
#include "source_base/matrix.h"
#include "source_base/parallel_reduce.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_pw.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/onsite_proj.h"

#include <cmath>
#include <complex>
#include <iomanip>
#include <ostream>
#include <vector>

namespace alloc
{
namespace
{

void set_unsupported(PwAllocReport& report, const std::string& reason)
{
    report.supported = false;
    report.skip_reason = reason;
}

void reduce_pool(std::vector<double>& values)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(values.data(),
                                 static_cast<int>(values.size()));
#endif
}

void reduce_pool(double& value)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(value);
#endif
}

void fill_real_space_totals(const UnitCell& ucell,
                            const constraint::WeightGrid& weight_grid,
                            const double* const* rho,
                            const int nspin,
                            const double nelec,
                            PwAllocReport& report)
{
    report.totals_available = true;
    report.nelec = nelec;
    const double dV = weight_grid.rho_basis()->omega
                      / static_cast<double>(weight_grid.rho_basis()->nxyz);
    std::vector<double> totals(static_cast<size_t>(2) * ucell.nat, 0.0);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const std::vector<double>& weight = weight_grid.atom_weight(iat);
        double n = 0.0;
        double m = 0.0;
        for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
        {
            n += weight[ir] * rho[0][ir];
            // Branch: the collinear spin-difference density is available only
            // for a two-channel run.
            if (nspin == 2)
            {
                n += weight[ir] * rho[1][ir];
                m += weight[ir] * (rho[0][ir] - rho[1][ir]);
            }
        }
        totals[2 * iat] = n * dV;
        totals[2 * iat + 1] = m * dV;
    }
    reduce_pool(totals);
    double global_m = 0.0;
    for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
    {
        // Branch: a non-magnetic density has no spin-difference channel.
        if (nspin == 2)
        {
            global_m += rho[0][ir] - rho[1][ir];
        }
    }
    reduce_pool(global_m);
    report.tot_mag = global_m * dV;

    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        report.n_tot[iat] = totals[2 * iat];
        report.m_tot[iat] = totals[2 * iat + 1];
        report.n_c[iat] = report.n_tot[iat] - report.n_T[iat];
        report.m_c[iat] = report.m_tot[iat] - report.m_T[iat];
        report.sum_n += report.n_tot[iat];
        report.sum_m += report.m_tot[iat];
    }
    report.dsum_n = report.sum_n - report.nelec;
    report.dsum_m = report.sum_m - report.tot_mag;
}

bool accumulate_k_point(
    projectors::OnsiteProjector<double, base_device::DEVICE_CPU>* onsite_p,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const int ik,
    const int spin_sign,
    const int nspin,
    PwAllocReport& report)
{
    psi->fix_k(ik);
    onsite_p->tabulate_atomic(ik);
    const int nband = psi->get_nbands();
    onsite_p->overlap_proj_psi(nband, psi->get_pointer());

    const std::complex<double>* tab = onsite_p->get_tab_atomic();
    const std::complex<double>* becp = onsite_p->get_h_becp();
    const int npw = onsite_p->get_npw();
    const int full_ntarget = onsite_p->get_tot_nproj();
    const int ntarget = report.ntarget;
    // Branch: a failed projector allocation cannot produce an overlap or a
    // band projection; report the condition instead of indexing null data.
    if (tab == nullptr || becp == nullptr || npw <= 0)
    {
        return false;
    }
    // Branch: full_ntarget is the total onsite row count; the selected rows
    // must all fall inside it.
    if (full_ntarget <= 0 || ntarget < 0)
    {
        return false;
    }
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: a selected row outside the projector table is a wiring bug.
        if (report.target_row[a] < 0 || report.target_row[a] >= full_ntarget)
        {
            return false;
        }
    }

    std::vector<std::complex<double>> compact_tab(
        static_cast<size_t>(ntarget) * npw);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            compact_tab[static_cast<size_t>(a) * npw + ig]
                = tab[static_cast<size_t>(report.target_row[a]) * npw + ig];
        }
    }
    std::vector<double> ST;
    build_pw_projector_overlap(compact_tab.data(), npw, ntarget, ST);
    std::vector<int> unowned(ntarget, -1);
    PwTargetSpace space;
    build_pw_target(ST, ntarget, report.target_atom, unowned, 0, space);
    report.rank = space.rank;
    report.cond = std::max(report.cond, space.cond);

    std::vector<std::complex<double>> raw(ntarget);
    std::vector<std::complex<double>> orth;
    for (int band = 0; band < nband; ++band)
    {
        const double weight = occupations(ik, band);
        // Branch: an unoccupied band contributes nothing to either moment.
        if (weight == 0.0)
        {
            continue;
        }
        for (int a = 0; a < ntarget; ++a)
        {
            raw[a] = becp[static_cast<size_t>(band) * full_ntarget
                          + report.target_row[a]];
        }
        orthogonalize_pw_overlaps(space, raw, orth);
        for (int a = 0; a < ntarget; ++a)
        {
            const double contribution = weight * std::norm(orth[a]);
            const int iat = report.target_atom[a];
            report.n_T[iat] += contribution;
            // Branch: only a collinear two-channel calculation carries a
            // spin moment; nspin=1 contributes charge only.
            if (nspin == 2)
            {
                report.m_T[iat] += spin_sign * contribution;
            }
        }
    }
    return true;
}

} // namespace

PwAllocReport report_pw_alloc(const UnitCell& ucell,
                              const psi::Psi<std::complex<double>>* psi,
                              const ModuleBase::matrix& occupations,
                              const K_Vectors& kv,
                              const int nspin,
                              const constraint::WeightGrid* weight_grid,
                              const double* const* rho,
                              const double nelec,
                              const int frame,
                              const int target_l)
{
    PwAllocReport report;
    report.frame = frame;
    report.target_l = target_l;
    report.nat = ucell.nat;
    report.n_T.assign(ucell.nat, 0.0);
    report.m_T.assign(ucell.nat, 0.0);
    report.n_c.assign(ucell.nat, 0.0);
    report.m_c.assign(ucell.nat, 0.0);
    report.n_tot.assign(ucell.nat, 0.0);
    report.m_tot.assign(ucell.nat, 0.0);

    // Branch: the S5 PW read-out is defined for the collinear scalar-band
    // path.  Spinor (nspin=4) bands need the separate Pauli contraction.
    if (nspin != 1 && nspin != 2)
    {
        set_unsupported(report, "PW allocation report supports only nspin 1/2");
        return report;
    }
    // Branch: without the CPU wavefunction there is no band projection.
    if (psi == nullptr)
    {
        set_unsupported(report, "CPU wavefunctions are unavailable");
        return report;
    }
    // Branch: non-collinear spinors have two components per band; the scalar
    // occupancy accumulation below would silently drop one of them.
    if (psi->get_npol() != 1)
    {
        set_unsupported(report, "non-collinear spinor bands are not wired");
        return report;
    }
    const int nks = psi->get_nk();
    const int nband = psi->get_nbands();
    report.nband = nband;
    // Branch: a run without k-points or bands has no occupied target space.
    if (nks <= 0 || nband <= 0)
    {
        set_unsupported(report, "wavefunction has no k-points or bands");
        return report;
    }

    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    // Branch: onsite projector initialization is optional across ABACUS; an
    // uninitialized table means onsite_radius/orbital inputs are missing.
    if (onsite_p == nullptr || onsite_p->get_tot_nproj() == 0)
    {
        set_unsupported(report,
                        "onsite projectors are not initialized (set "
                        "onsite_radius > 0 and supply NUMERICAL_ORBITAL)");
        return report;
    }

    std::vector<int> nh_per_atom(ucell.nat, 0);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        nh_per_atom[iat] = onsite_p->get_nh(iat);
    }
    const PwTargetShell selection
        = select_pw_target_shell(nh_per_atom, target_l);
    report.target_row = selection.row;
    report.target_atom = selection.atom;
    report.ntarget = static_cast<int>(selection.row.size());

    for (int ik = 0; ik < nks; ++ik)
    {
        // Branch: a collinear run supplies one spin label per k-point; using
        // a default sign would silently mislabel the magnetization channel.
        if (static_cast<int>(kv.isk.size()) <= ik)
        {
            set_unsupported(report, "k-point spin labels are unavailable");
            return report;
        }
        const int spin_sign = (kv.isk[ik] == 1) ? -1 : 1;
        if (!accumulate_k_point(onsite_p, psi, occupations, ik, spin_sign,
                                 nspin, report))
        {
            set_unsupported(report, "onsite projector tabulation is unavailable");
            return report;
        }
    }
    // Each k-pool owns a disjoint set of k-points.  n_T/m_T are already
    // identical within a pool after the projector reduction in cal_becp, so
    // divide by the pool size before summing the pool copies globally.
    Parallel_Reduce::reduce_double_allpool(
        PARAM.inp.kpar, GlobalV::NPROC_IN_POOL, report.n_T.data(),
        static_cast<int>(report.n_T.size()));
    Parallel_Reduce::reduce_double_allpool(
        PARAM.inp.kpar, GlobalV::NPROC_IN_POOL, report.m_T.data(),
        static_cast<int>(report.m_T.size()));
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        report.sum_n_T += report.n_T[iat];
        report.sum_m_T += report.m_T[iat];
    }

    // Branch: the real-space complement block is optional so the target-only
    // read-out can run in tests without a WeightGrid.
    if (weight_grid != nullptr && rho != nullptr && rho[0] != nullptr)
    {
        fill_real_space_totals(ucell, *weight_grid, rho, nspin, nelec,
                               report);
    }
    return report;
}

void print_pw_alloc_report(const PwAllocReport& report, std::ostream& os)
{
    os << "[constraint] ALLOC-PW frame=" << report.frame
       << " target_l=" << report.target_l << " ntarget=" << report.ntarget
       << " nband=" << report.nband << " cond=" << report.cond
       << " rank=" << report.rank;
    // Branch: a skipped report prints one explicit reason and no per-atom
    // values, so callers cannot mistake zeros for a real measurement.
    if (!report.supported)
    {
        os << " skipped reason=\"" << report.skip_reason << "\"\n";
        return;
    }
    os << "\n";
    os << std::scientific << std::setprecision(8);
    for (int iat = 0; iat < report.nat; ++iat)
    {
        os << "ALLOC-PW atom=" << iat << " n_T=" << report.n_T[iat]
           << " m_T=" << report.m_T[iat];
        // Branch: totals are unavailable when no WeightGrid/density was
        // supplied; omit rather than printing misleading zeros.
        if (report.totals_available)
        {
            os << " n=" << report.n_tot[iat] << " n_c=" << report.n_c[iat]
               << " m=" << report.m_tot[iat] << " m_c=" << report.m_c[iat];
        }
        os << "\n";
    }
    os << "ALLOC-PW audit sum_n_T=" << report.sum_n_T
       << " sum_m_T=" << report.sum_m_T;
    // Branch: the closing sum-rule fields exist only for a real-space sum.
    if (report.totals_available)
    {
        os << " sum_n=" << report.sum_n << " nelec=" << report.nelec
           << " dsum_n=" << report.dsum_n << " sum_m=" << report.sum_m
           << " tot_mag=" << report.tot_mag << " dsum_m=" << report.dsum_m;
    }
    os << "\n";
}

} // namespace alloc
