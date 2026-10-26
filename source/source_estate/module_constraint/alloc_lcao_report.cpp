#include "source_estate/module_constraint/alloc_lcao_report.h"

#include "source_base/parallel_reduce.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

namespace alloc
{
namespace
{

// Gamma-only contraction is the S2 contract: every stored block must be at
// R=(0,0,0).  A non-zero block means the caller asked for a periodic multi-k
// contraction, which must use S(k)/DM(k), not this dense-R0 adapter.
bool all_blocks_are_gamma(const hamilt::HContainer<double>& H)
{
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair = H.get_atom_pair(static_cast<int>(iap));
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
            // Branch: one non-zero cell vector makes the dense-R0 formula
            // invalid for the entire container.
            if (R.x != 0 || R.y != 0 || R.z != 0)
            {
                return false;
            }
        }
    }
    return true;
}

template <typename F>
void for_each_gamma_block(const hamilt::HContainer<double>& H,
                          const UnitCell& ucell,
                          F&& callback)
{
    const Parallel_Orbitals* pv = H.get_paraV();
    // Branch: a production HContainer always carries its distribution; an
    // uninitialized one cannot be mapped to global AO indices.
    if (pv == nullptr)
    {
        return;
    }
    for (size_t iap = 0; iap < H.size_atom_pairs(); ++iap)
    {
        const hamilt::AtomPair<double>& pair
            = H.get_atom_pair(static_cast<int>(iap));
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        const hamilt::BaseMatrix<double>* matrix = pair.find_matrix(0, 0, 0);
        // Branch: a pair without an R=0 block contributes nothing to this
        // gamma-only contraction.
        if (matrix == nullptr)
        {
            continue;
        }

        const std::vector<int> rows = pv->get_indexes_row(iat1);
        const std::vector<int> cols = pv->get_indexes_col(iat2);
        const int base_i = ucell.get_iat2iwt()[iat1];
        const int base_j = ucell.get_iat2iwt()[iat2];
        for (int i = 0; i < matrix->get_row_size(); ++i)
        {
            for (int j = 0; j < matrix->get_col_size(); ++j)
            {
                callback(base_i + rows[i], base_j + cols[j],
                         matrix->get_value(i, j));
            }
        }
    }
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

std::vector<int> select_targets(const UnitCell& ucell, const int target_l)
{
    std::vector<int> orbital;
    // Branch: a negative l cannot select a physical NAO shell.
    if (target_l < 0)
    {
        return orbital;
    }
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const int it = ucell.iat2it[iat];
        const Atom& atom = ucell.atoms[it];
        // Branch: this atom type has no target-l shell; it contributes no
        // target orbital and remains purely in the complementary channel.
        if (target_l > atom.nwl || target_l >= static_cast<int>(atom.l_nchi.size())
            || atom.l_nchi[target_l] <= 0)
        {
            continue;
        }
        int offset = 0;
        for (int L = 0; L < target_l; ++L)
        {
            offset += atom.l_nchi[L] * (2 * L + 1);
        }
        for (int n = 0; n < atom.l_nchi[target_l]; ++n)
        {
            for (int m = 0; m < 2 * target_l + 1; ++m)
            {
                orbital.push_back(ucell.get_iat2iwt()[iat] + offset
                                  + n * (2 * target_l + 1) + m);
            }
        }
    }
    return orbital;
}

std::vector<int> atom_labels(const std::vector<int>& target_orbital,
                             const UnitCell& ucell)
{
    std::vector<int> labels;
    labels.reserve(target_orbital.size());
    for (int orbital : target_orbital)
    {
        int iat = 0;
        // Branch: map the selected global AO index to its atom by the
        // contiguous iat2iwt table; this is the same layout used by the
        // production AO matrices.
        while (iat + 1 < ucell.nat
               && orbital >= ucell.get_iat2iwt()[iat + 1])
        {
            ++iat;
        }
        labels.push_back(iat);
    }
    return labels;
}

} // namespace

LcaoAllocReport report_lcao_alloc(
    const UnitCell& ucell,
    const hamilt::HContainer<double>& S,
    const std::vector<const hamilt::HContainer<double>*>& dmr,
    const constraint::WeightGrid& wg,
    const double* const* rho,
    const int nspin,
    const double nelec,
    const int frame,
    const int target_l)
{
    LcaoAllocReport report;
    report.frame = frame;
    report.target_l = target_l;
    report.nat = ucell.nat;
    report.nelec = nelec;
    report.n_T.assign(ucell.nat, 0.0);
    report.m_T.assign(ucell.nat, 0.0);
    report.n_c.assign(ucell.nat, 0.0);
    report.m_c.assign(ucell.nat, 0.0);
    report.n_tot.assign(ucell.nat, 0.0);
    report.m_tot.assign(ucell.nat, 0.0);

    // Branch: S2 only wires the real collinear LCAO path.  Non-collinear
    // spinors and missing density data are rejected explicitly.
    if (nspin != 1 && nspin != 2)
    {
        report.supported = false;
        report.skip_reason = "S2 LCAO report supports only nspin 1/2";
        return report;
    }
    if (rho == nullptr || rho[0] == nullptr)
    {
        report.supported = false;
        report.skip_reason = "real-space density is unavailable";
        return report;
    }
    if (S.get_paraV() == nullptr)
    {
        report.supported = false;
        report.skip_reason = "S(R) has no Parallel_Orbitals layout";
        return report;
    }
    if (!all_blocks_are_gamma(S))
    {
        report.supported = false;
        report.skip_reason = "periodic multi-k S(R) is deferred beyond S2";
        return report;
    }
    for (const hamilt::HContainer<double>* d : dmr)
    {
        // Branch: every DMR channel must share the gamma-only contract.
        if (d == nullptr || !all_blocks_are_gamma(*d))
        {
            report.supported = false;
            report.skip_reason = "periodic multi-k DMR is deferred beyond S2";
            return report;
        }
    }
    if (static_cast<int>(dmr.size()) != nspin)
    {
        report.supported = false;
        report.skip_reason = "DMR spin-channel count does not match nspin";
        return report;
    }
    if (wg.is_dipole())
    {
        report.supported = false;
        report.skip_reason = "dipole weights do not define per-atom Becke weights";
        return report;
    }
    if (wg.nat() != ucell.nat || static_cast<int>(wg.nrxx()) <= 0)
    {
        report.supported = false;
        report.skip_reason = "per-atom weight grid is unavailable";
        return report;
    }

    const Parallel_Orbitals* pv = S.get_paraV();
    report.norb = pv->get_global_row_size();
    report.target_orbital = select_targets(ucell, target_l);
    report.target_atom = atom_labels(report.target_orbital, ucell);
    report.ntarget = static_cast<int>(report.target_orbital.size());

    std::vector<int> target_pos(report.norb, -1);
    for (int a = 0; a < report.ntarget; ++a)
    {
        target_pos[report.target_orbital[a]] = a;
    }

    // S_T[a,b] = <phi_a|S|phi_b> for the selected AO columns.  Each local
    // block contributes its own elements; the pool reduction makes the small
    // R x R matrix identical on every rank.
    std::vector<double> ST(static_cast<size_t>(report.ntarget) * report.ntarget,
                           0.0);
    for_each_gamma_block(S, ucell, [&](const int gi, const int gj,
                                       const double value) {
        const int a = target_pos[gi];
        const int b = target_pos[gj];
        // Branch: only target-target blocks enter S_T.
        if (a >= 0 && b >= 0)
        {
            ST[static_cast<size_t>(a) * report.ntarget + b] += value;
        }
    });
    reduce_pool(ST);

    LcaoTargetSpace target;
    finalize_lcao_target(report.norb, report.target_orbital,
                         report.target_atom, ST, target);
    report.cond = target.cond;
    report.rank = target.rank;

    std::vector<double> n_T_target(report.ntarget, 0.0);
    std::vector<double> m_T_target(report.ntarget, 0.0);
    for (int a = 0; a < report.ntarget; ++a)
    {
        // u_a = S c'_a, gated by the target coefficient support.  Building u
        // once per target keeps the DMR contraction independent of the AO
        // layout and directly mirrors the dense S2.1 formula.
        std::vector<double> u(report.norb, 0.0);
        for_each_gamma_block(S, ucell, [&](const int gi, const int gj,
                                           const double value) {
            const int b = target_pos[gj];
            // Branch: only columns in the selected target space contribute
            // to S c'_a.
            if (b >= 0)
            {
                u[gi] += value * target.Cprime[static_cast<size_t>(gj)
                                                   * report.ntarget + a];
            }
        });
        reduce_pool(u);

        for (int spin = 0; spin < nspin; ++spin)
        {
            double local = 0.0;
            for_each_gamma_block(*dmr[spin], ucell,
                [&](const int gi, const int gj, const double value) {
                    local += u[gi] * value * u[gj];
                });
            reduce_pool(local);
            n_T_target[a] += local;
            // Branch: collinear spin difference is up minus down.
            if (nspin == 2)
            {
                m_T_target[a] += (spin == 0 ? local : -local);
            }
        }
        report.n_T[report.target_atom[a]] += n_T_target[a];
        report.m_T[report.target_atom[a]] += m_T_target[a];
    }

    // Total per-atom charge/moment from the same Becke partition used by the
    // constraint loop.  The weight grid is rank-local, so reduce the two
    // per-atom vectors together.
    const double dV = wg.rho_basis()->omega / static_cast<double>(wg.rho_basis()->nxyz);
    std::vector<double> totals(static_cast<size_t>(2) * ucell.nat, 0.0);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const std::vector<double>& w = wg.atom_weight(iat);
        double n = 0.0;
        double m = 0.0;
        for (int ir = 0; ir < wg.nrxx(); ++ir)
        {
            n += w[ir] * rho[0][ir];
            // Branch: spin-polarized grids carry the minority channel in
            // rho[1]; non-spin-polarized runs have no moment channel.
            if (nspin == 2)
            {
                n += w[ir] * rho[1][ir];
                m += w[ir] * (rho[0][ir] - rho[1][ir]);
            }
        }
        totals[2 * iat] = n * dV;
        totals[2 * iat + 1] = m * dV;
    }
    reduce_pool(totals);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        report.n_tot[iat] = totals[2 * iat];
        report.m_tot[iat] = totals[2 * iat + 1];
        report.n_c[iat] = report.n_tot[iat] - report.n_T[iat];
        report.m_c[iat] = report.m_tot[iat] - report.m_T[iat];
        report.sum_n += report.n_tot[iat];
        report.sum_m += report.m_tot[iat];
    }

    // The observable audit must compare against the same mixed density the
    // per-atom split used.  ucell.magnet.tot_mag is evaluated before charge
    // mixing and can differ by SCF-noise at the final frame, so recompute the
    // global spin integral here.  The independent per-atom weighted sum then
    // verifies the Becke partition rather than restating the input value.
    double global_m = 0.0;
    for (int ir = 0; ir < wg.nrxx(); ++ir)
    {
        // Branch: only a two-channel collinear density has a spin moment.
        if (nspin == 2)
        {
            global_m += rho[0][ir] - rho[1][ir];
        }
    }
    reduce_pool(global_m);
    report.tot_mag = global_m * dV;
    report.dsum_n = report.sum_n - report.nelec;
    report.dsum_m = report.sum_m - report.tot_mag;
    return report;
}

void print_lcao_alloc_report(const LcaoAllocReport& report, std::ostream& os)
{
    // Branch: unsupported configurations still get a one-line, explicit
    // notice; no silent fallback to a wrong periodic contraction.
    if (!report.supported)
    {
        os << "[constraint] ALLOC skipped frame=" << report.frame
           << " reason=\"" << report.skip_reason << "\"\n";
        return;
    }

    os << std::setprecision(10);
    os << "[constraint] ALLOC frame=" << report.frame
       << " target_l=" << report.target_l
       << " ntarget=" << report.ntarget
       << " norb=" << report.norb
       << " cond=" << report.cond
       << " rank=" << report.rank << "\n";
    os << "ALLOC targets atom:orbital=";
    for (int a = 0; a < report.ntarget; ++a)
    {
        if (a > 0)
        {
            os << ",";
        }
        os << report.target_atom[a] << ":" << report.target_orbital[a];
    }
    os << "\n";
    for (int iat = 0; iat < report.nat; ++iat)
    {
        os << "ALLOC c[" << iat << "] atom=" << iat
           << " n_T=" << report.n_T[iat]
           << " n_c=" << report.n_c[iat]
           << " n=" << report.n_tot[iat]
           << " m_T=" << report.m_T[iat]
           << " m_c=" << report.m_c[iat]
           << " m=" << report.m_tot[iat] << "\n";
    }
    os << "ALLOC audit sum_n=" << report.sum_n
       << " nelec=" << report.nelec
       << " dsum_n=" << report.dsum_n
       << " sum_m=" << report.sum_m
       << " tot_mag=" << report.tot_mag
       << " dsum_m=" << report.dsum_m << "\n";
}

} // namespace alloc
