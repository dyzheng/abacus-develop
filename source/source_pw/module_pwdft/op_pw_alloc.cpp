#include "op_pw_alloc.h"

#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_pw.h"
#include "source_estate/module_constraint/constraint_io.h"
#include "source_estate/module_constraint/constraint_loop.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/onsite_proj.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace alloc
{

struct PwTargetContext
{
    int npw = 0;
    PwTargetSpace space;
    std::vector<std::vector<std::complex<double>>> phi;
    // compact target -> global onsite-projector row
    std::vector<int> target_row;
};

namespace
{

void reduce_pool(std::complex<double>& value)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(value);
#endif
}

void reduce_pool(std::vector<double>& values)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(values.data(), static_cast<int>(values.size()));
#endif
}

void reduce_pool(std::vector<std::complex<double>>& values)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(values.data(), static_cast<int>(values.size()));
#endif
}

std::vector<std::complex<double>> apply_local_potential(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const std::complex<double>* input,
    const int npw)
{
    std::vector<std::complex<double>> work(pw.nmaxgr,
                                           std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> porter(pw.nmaxgr,
                                             std::complex<double>(0.0, 0.0));
    for (int ig = 0; ig < npw; ++ig)
    {
        work[ig] = input[ig];
    }
    pw.template recip_to_real<std::complex<double>, base_device::DEVICE_CPU>(
        work.data(), porter.data(), ik);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        porter[ir] *= potential[ir];
    }
    pw.template real_to_recip<std::complex<double>, base_device::DEVICE_CPU>(
        porter.data(), work.data(), ik);
    std::vector<std::complex<double>> result(npw,
                                             std::complex<double>(0.0, 0.0));
    for (int ig = 0; ig < npw; ++ig)
    {
        result[ig] = work[ig];
    }
    return result;
}

bool build_target_context(
    projectors::OnsiteProjector<double, base_device::DEVICE_CPU>* onsite_p,
    const constraint::WeightGrid& weight_grid,
    const int ik,
    const int target_l,
    const std::complex<double>* psi,
    const int nband,
    const int ld_psi,
    PwTargetContext& context,
    std::string& error)
{
    // Branch: allocation needs an initialized onsite-projector table and a
    // selected angular-momentum shell.  Both are configuration, not runtime
    // numerical failures, so report a precise reason.
    if (onsite_p == nullptr || onsite_p->get_tot_nproj() <= 0)
    {
        error = "PW allocation requires onsite projectors "
                "(onsite_radius > 0 and NUMERICAL_ORBITAL)";
        return false;
    }
    onsite_p->tabulate_atomic(ik);
    // tab_atomic_ is the OnsiteProjector vkb workspace and is only filled by
    // cal_becp.  Allocation may be the first onsite consumer in a PW run, so
    // explicitly populate it before reading the projector rows.  The result
    // is cached in context across all later Hamiltonian applications.
    if (psi != nullptr && nband > 0)
    {
        onsite_p->overlap_proj_psi(nband, psi, ld_psi);
    }
    const std::complex<double>* tab = onsite_p->get_tab_atomic();
    const int npw = onsite_p->get_npw();
    // Branch: a missing tabulation leaves no plane-wave target vectors.
    if (tab == nullptr || npw <= 0)
    {
        error = "PW allocation could not tabulate onsite projectors";
        return false;
    }

    const int nat = static_cast<int>(weight_grid.nat());
    std::vector<int> nh_per_atom(nat, 0);
    for (int iat = 0; iat < nat; ++iat)
    {
        nh_per_atom[iat] = onsite_p->get_nh(iat);
    }
    const PwTargetShell selection
        = select_pw_target_shell(nh_per_atom, target_l);
    // Branch: an empty shell would silently reduce the allocation operator to
    // the legacy local weight, which is exactly what S5 must replace.
    if (selection.row.empty())
    {
        error = "PW allocation selected zero target projectors";
        return false;
    }

    const int ntarget = static_cast<int>(selection.row.size());
    const int nconstraint = weight_grid.nconstraint();
    const auto& constraint_atoms = weight_grid.constraint_atoms();
    std::vector<int> target_constraint(ntarget, -1);
    for (int a = 0; a < ntarget; ++a)
    {
        int owner = -1;
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            // Branch: each target atom must belong to at most one constraint;
            // a duplicate owner makes Q_i ambiguous.
            if (std::find(constraint_atoms[alpha].begin(),
                          constraint_atoms[alpha].end(),
                          selection.atom[a])
                != constraint_atoms[alpha].end())
            {
                if (owner >= 0)
                {
                    error = "PW target atom belongs to more than one constraint";
                    return false;
                }
                owner = alpha;
            }
        }
        target_constraint[a] = owner;
    }

    std::vector<std::complex<double>> compact_tab(
        static_cast<size_t>(ntarget) * npw);
    context.phi.assign(
        ntarget,
        std::vector<std::complex<double>>(npw,
                                          std::complex<double>(0.0, 0.0)));
    context.target_row = selection.row;
    for (int a = 0; a < ntarget; ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            const std::complex<double> value
                = tab[static_cast<size_t>(selection.row[a]) * npw + ig];
            compact_tab[static_cast<size_t>(a) * npw + ig] = value;
            context.phi[a][ig] = value;
        }
    }
    std::vector<std::complex<double>> ST;
    build_pw_projector_overlap(compact_tab.data(), npw, ntarget, ST);
    build_pw_target(ST, ntarget, selection.atom, target_constraint,
                    nconstraint, context.space);
    // Branch: a zero-rank projector shell has no complement subspace to
    // constrain; this is a rank/configuration failure, not a no-op.
    if (context.space.rank == 0)
    {
        double max_st = 0.0;
        for (const std::complex<double> value : context.space.ST)
        {
            max_st = std::max(max_st, std::abs(value));
        }
        error = "PW allocation target projector has zero numerical rank "
                "(npw=" + std::to_string(npw)
                + ", ntarget=" + std::to_string(ntarget)
                + ", max_ST=" + std::to_string(max_st) + ")";
        return false;
    }
    context.npw = npw;
    return true;
}

bool build_mu_potential(const constraint::WeightGrid& weight_grid,
                        const constraint::ConstraintLoop& loop,
                        const int spin,
                        std::vector<double>& potential,
                        std::vector<double>& lambda,
                        std::string& error)
{
    const int nconstraint = weight_grid.nconstraint();
    const std::vector<double>& mu = loop.mu_eff();
    const std::vector<constraint::ChannelProfile>& channels = loop.channels();
    // Guard: the multiplier, channel and weight lists must stay parallel.
    if (static_cast<int>(mu.size()) != nconstraint
        || static_cast<int>(channels.size()) != nconstraint)
    {
        error = "PW allocation multiplier/constraint count mismatch";
        return false;
    }

    potential.assign(weight_grid.nrxx(), 0.0);
    lambda.assign(nconstraint, 0.0);
    const auto& weights = weight_grid.constraint_weights();
    for (int alpha = 0; alpha < nconstraint; ++alpha)
    {
        // Branch A: spin-up (or the single nspin=1 channel) uses inj_up.
        // Branch B: spin-down uses inj_dn, which is the opposite sign for a
        // spin constraint and the same sign for a charge constraint.
        const double sign = (spin == 0) ? channels[alpha].inj_up
                                       : channels[alpha].inj_dn;
        const double coefficient = mu[alpha] * sign;
        lambda[alpha] = coefficient;
        for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
        {
            potential[ir] += coefficient * weights[alpha][ir];
        }
    }
    return true;
}

bool build_target_products(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const PwTargetContext& context,
    std::vector<std::vector<std::complex<double>>>& vphi,
    std::vector<std::complex<double>>& m_raw,
    std::string& error)
{
    const int r = context.space.ntarget;
    vphi.assign(r,
                std::vector<std::complex<double>>(context.npw,
                                                  std::complex<double>(0.0, 0.0)));
    m_raw.assign(static_cast<size_t>(r) * r,
                 std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        vphi[a] = apply_local_potential(pw, ik, potential,
                                        context.phi[a].data(), context.npw);
    }
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < context.npw; ++ig)
            {
                value += std::conj(context.phi[a][ig]) * vphi[b][ig];
            }
            m_raw[static_cast<size_t>(a) * r + b] = value;
        }
    }
    // Pool the complete R x R matrix in one collective.  The per-matrix-element
    // reductions were a major MPI cost for repeated Hamiltonian applications.
    {
        std::vector<double> packed(2 * static_cast<size_t>(r) * r, 0.0);
        for (size_t i = 0; i < m_raw.size(); ++i)
        {
            packed[2 * i] = m_raw[i].real();
            packed[2 * i + 1] = m_raw[i].imag();
        }
        reduce_pool(packed);
        for (size_t i = 0; i < m_raw.size(); ++i)
        {
            m_raw[i] = std::complex<double>(packed[2 * i], packed[2 * i + 1]);
        }
    }
    (void)error;
    return true;
}

void build_raw_overlap(const PwTargetContext& context,
                       const std::complex<double>* psi,
                       std::vector<std::complex<double>>& raw)
{
    raw.assign(context.space.ntarget, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < context.space.ntarget; ++a)
    {
        std::complex<double> value(0.0, 0.0);
        for (int ig = 0; ig < context.npw; ++ig)
        {
            value += std::conj(context.phi[a][ig]) * psi[ig];
        }
        raw[a] = value;
    }
    // One collective for the whole raw target overlap vector.
    {
        std::vector<double> packed(2 * raw.size(), 0.0);
        for (size_t i = 0; i < raw.size(); ++i)
        {
            packed[2 * i] = raw[i].real();
            packed[2 * i + 1] = raw[i].imag();
        }
        reduce_pool(packed);
        for (size_t i = 0; i < raw.size(); ++i)
        {
            raw[i] = std::complex<double>(packed[2 * i], packed[2 * i + 1]);
        }
    }
}

struct PwAllocBandForceData
{
    int nocc = 0;
    int ntarget = 0;
    int nconstraint = 0;
    int npw = 0;
    int ld = 0;
    int nvec = 0;
    int target_offset = 0;
    std::vector<int> band;
    std::vector<double> occupation;
    std::vector<std::vector<std::complex<double>>> raw_overlap;
    std::vector<std::vector<std::complex<double>>> orth_overlap;
    // [occupied band / W_i q_band / raw target] vectors, padded to ld
    std::vector<std::complex<double>> custom_vectors;
    // [W_i q_band][target] raw overlap, indexed (i*nocc + s)*ntarget + a
    std::vector<std::complex<double>> wq_overlap;
};

bool build_band_force_data(
    const ModulePW::PW_Basis_K& wfcpw,
    const int ik,
    const std::complex<double>* psi_k,
    const ModuleBase::matrix& occupations,
    const int nband,
    const int ld,
    const constraint::WeightGrid& weight_grid,
    const PwTargetContext& context,
    PwAllocBandForceData& data,
    std::string& error)
{
    const int r = context.space.ntarget;
    const int nconstraint = context.space.nconstraint;
    data.ntarget = r;
    data.nconstraint = nconstraint;
    data.npw = context.npw;
    data.ld = ld;
    for (int band = 0; band < nband; ++band)
    {
        // Branch: only occupied bands contribute to the density matrix.
        if (occupations(ik, band) != 0.0)
        {
            data.band.push_back(band);
            data.occupation.push_back(occupations(ik, band));
        }
    }
    data.nocc = static_cast<int>(data.band.size());
    data.target_offset = data.nocc + nconstraint * data.nocc;
    data.nvec = data.target_offset + r;
    data.custom_vectors.assign(static_cast<size_t>(data.nvec) * ld,
                               std::complex<double>(0.0, 0.0));
    data.raw_overlap.assign(
        data.nocc, std::vector<std::complex<double>>(
                       r, std::complex<double>(0.0, 0.0)));
    data.orth_overlap.assign(
        data.nocc, std::vector<std::complex<double>>(
                       r, std::complex<double>(0.0, 0.0)));
    data.wq_overlap.assign(
        static_cast<size_t>(nconstraint) * data.nocc * r,
        std::complex<double>(0.0, 0.0));
    // Branch: no occupied bands means no derivative contraction to form.
    if (data.nocc == 0)
    {
        return true;
    }
    // Guard: the caller's leading dimension must hold every local G vector.
    if (ld < context.npw)
    {
        error = "PW allocation force wavefunction stride is too small";
        return false;
    }

    for (int s = 0; s < data.nocc; ++s)
    {
        const std::complex<double>* band_ptr
            = psi_k + static_cast<size_t>(data.band[s]) * ld;
        build_raw_overlap(context, band_ptr, data.raw_overlap[s]);
        for (int a = 0; a < r; ++a)
        {
            for (int b = 0; b < r; ++b)
            {
                data.orth_overlap[s][a]
                    += std::conj(
                           context.space.ST_inv_root[static_cast<size_t>(a) * r
                                                     + b])
                       * data.raw_overlap[s][b];
            }
        }

        // Q_c|psi> = |psi> - sum_a |phi_a> <phi_a|psi>.
        std::vector<std::complex<double>> complement(
            context.npw, std::complex<double>(0.0, 0.0));
        for (int ig = 0; ig < context.npw; ++ig)
        {
            complement[ig] = band_ptr[ig];
        }
        for (int a = 0; a < r; ++a)
        {
            for (int ig = 0; ig < context.npw; ++ig)
            {
                complement[ig] -= context.phi[a][ig]
                                  * data.orth_overlap[s][a];
            }
        }
        for (int ig = 0; ig < context.npw; ++ig)
        {
            data.custom_vectors[static_cast<size_t>(s) * ld + ig]
                = band_ptr[ig];
        }

        // W_i Q_c|psi> is needed both for the value overlap and for the
        // derivative contraction against it.
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            const std::vector<std::complex<double>> wq
                = apply_local_potential(wfcpw, ik,
                                        weight_grid.constraint_weight(alpha),
                                        complement.data(), context.npw);
            const int vec_index = data.nocc + alpha * data.nocc + s;
            for (int ig = 0; ig < context.npw; ++ig)
            {
                data.custom_vectors[static_cast<size_t>(vec_index) * ld + ig]
                    = wq[ig];
            }
            std::vector<std::complex<double>> overlap(
                r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    overlap[a] += std::conj(context.phi[a][ig]) * wq[ig];
                }
            }
            reduce_pool(overlap);
            for (int a = 0; a < r; ++a)
            {
                data.wq_overlap[(static_cast<size_t>(alpha) * data.nocc + s)
                                    * r + a] = overlap[a];
            }
        }
    }
    // Raw target vectors are the right-hand sides for dST and dG.
    for (int a = 0; a < r; ++a)
    {
        const int vec_index = data.target_offset + a;
        for (int ig = 0; ig < context.npw; ++ig)
        {
            data.custom_vectors[static_cast<size_t>(vec_index) * ld + ig]
                = context.phi[a][ig];
        }
    }
    return true;
}

void accumulate_target_force_direction(
    const PwTargetContext& context,
    const PwAllocBandForceData& data,
    const std::vector<std::complex<double>>& derivative_overlap,
    const int nkb,
    const int atom,
    const int ipol,
    const int spin,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    const double tpiba,
    std::vector<double>& force)
{
    const int r = context.space.ntarget;
    // Branch: an atom without selected target projectors has no target-space
    // derivative contribution.
    bool owns_target = false;
    for (int a = 0; a < r; ++a)
    {
        // Branch: only projectors centred on the displaced atom move.
        if (context.space.atom[a] == atom)
        {
            owns_target = true;
            break;
        }
    }
    if (!owns_target)
    {
        return;
    }

    const auto out = [&](const int a, const int vec) {
        return derivative_overlap[static_cast<size_t>(context.target_row[a])
                                      * data.nvec + vec];
    };

    // D[a,b] = <d alpha_a|alpha_b>.  Rows outside the displaced atom are
    // zero by construction, so dST = D + D^dagger contains every projector
    // on this atom exactly once.
    std::vector<std::complex<double>> D(static_cast<size_t>(r) * r,
                                        std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        // Branch: a projector on another atom does not move here.
        if (context.space.atom[a] != atom)
        {
            continue;
        }
        for (int b = 0; b < r; ++b)
        {
            D[static_cast<size_t>(a) * r + b]
                = out(a, data.target_offset + b);
        }
    }
    std::vector<std::complex<double>> dST_inv_root;
    build_pw_target_derivative(context.space, D, dST_inv_root);

    int owner = -1;
    for (int a = 0; a < r; ++a)
    {
        // Branch: all targets on one atom share the same optional constraint
        // (build_target_context rejects a duplicate owner).
        if (context.space.atom[a] == atom)
        {
            owner = context.space.constraint[a];
            break;
        }
    }

    for (int s = 0; s < data.nocc; ++s)
    {
        std::vector<std::complex<double>> d_raw(
            r, std::complex<double>(0.0, 0.0));
        for (int a = 0; a < r; ++a)
        {
            // Branch: keep only derivatives of projectors on this atom.
            if (context.space.atom[a] == atom)
            {
                d_raw[a] = out(a, s);
            }
        }
        std::vector<std::complex<double>> orth;
        std::vector<std::complex<double>> d_orth;
        orthogonalize_pw_overlaps_derivative(
            context.space, dST_inv_root, data.raw_overlap[s], d_raw, orth,
            d_orth);

        double dQ_owner = 0.0;
        for (int a = 0; a < r; ++a)
        {
            // Branch: Q_i contains every orthonormal target owned by this
            // constraint.  Even targets centred on other atoms rotate when
            // dG changes, so the sum must not be restricted to this atom.
            if (owner >= 0 && context.space.constraint[a] == owner)
            {
                dQ_owner += (std::conj(d_orth[a])
                             * data.orth_overlap[s][a]).real();
            }
        }
        dQ_owner *= 2.0;

        for (int alpha = 0; alpha < data.nconstraint; ++alpha)
        {
            // Branch: a zero multiplier contributes no constraint force.
            if (mu[alpha] == 0.0)
            {
                continue;
            }
            std::vector<std::complex<double>> z(
                r, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> dz(
                r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                z[a] = data.wq_overlap[(static_cast<size_t>(alpha)
                                            * data.nocc + s)
                                           * r + a];
                // Branch: d alpha_a is nonzero only on the displaced atom.
                if (context.space.atom[a] == atom)
                {
                    dz[a] = out(a, data.nocc + alpha * data.nocc + s);
                }
            }
            std::vector<std::complex<double>> orth_wq;
            std::vector<std::complex<double>> d_orth_wq;
            orthogonalize_pw_overlaps_derivative(context.space,
                                                 dST_inv_root, z, dz,
                                                 orth_wq, d_orth_wq);

            // -2 Re <dP psi|W_i Q_c|psi>.  dP rotates every orthonormal
            // target, not only projectors centred on the displaced atom.
            std::complex<double> cross(0.0, 0.0);
            for (int a = 0; a < r; ++a)
            {
                cross += std::conj(data.orth_overlap[s][a])
                             * d_orth_wq[a]
                         + std::conj(d_orth[a]) * orth_wq[a];
            }
            double derivative = -2.0 * cross.real();
            // Branch: only the constraint owning this atom receives the
            // explicit dQ_i term; every W_i still feels the dP cross term.
            if (owner == alpha)
            {
                derivative += dQ_owner;
            }
            const double channel
                = (spin == 0) ? channels[alpha].inj_up
                              : channels[alpha].inj_dn;
            force[static_cast<size_t>(atom) * 3 + ipol]
                -= data.occupation[s] * mu[alpha] * channel * derivative
                   * tpiba;
        }
    }
    (void)nkb;
}

void accumulate_target_stress_component(
    const PwTargetContext& context,
    const PwAllocBandForceData& data,
    const std::vector<std::complex<double>>& derivative_overlap,
    const int ipol,
    const int jpol,
    const int spin,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    std::vector<double>& stress_energy)
{
    const int r = context.space.ntarget;
    const int index = ipol * 3 + jpol;
    const auto out = [&](const int a, const int vec) {
        return derivative_overlap[static_cast<size_t>(context.target_row[a])
                                      * data.nvec + vec];
    };

    // A homogeneous strain moves every projector, so this D matrix contains
    // all target rows rather than the single atom used by the force path.
    std::vector<std::complex<double>> D(static_cast<size_t>(r) * r,
                                        std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            D[static_cast<size_t>(a) * r + b]
                = out(a, data.target_offset + b);
        }
    }
    std::vector<std::complex<double>> dST_inv_root;
    build_pw_target_derivative(context.space, D, dST_inv_root);

    for (int s = 0; s < data.nocc; ++s)
    {
        std::vector<std::complex<double>> d_raw(
            r, std::complex<double>(0.0, 0.0));
        for (int a = 0; a < r; ++a)
        {
            d_raw[a] = out(a, s);
        }
        std::vector<std::complex<double>> orth;
        std::vector<std::complex<double>> d_orth;
        orthogonalize_pw_overlaps_derivative(
            context.space, dST_inv_root, data.raw_overlap[s], d_raw, orth,
            d_orth);

        for (int alpha = 0; alpha < data.nconstraint; ++alpha)
        {
            // Branch: a zero multiplier contributes no target strain term.
            if (mu[alpha] == 0.0)
            {
                continue;
            }
            std::vector<std::complex<double>> z(
                r, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> dz(
                r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                z[a] = data.wq_overlap[(static_cast<size_t>(alpha)
                                            * data.nocc + s)
                                           * r + a];
                dz[a] = out(a, data.nocc + alpha * data.nocc + s);
            }
            std::vector<std::complex<double>> orth_wq;
            std::vector<std::complex<double>> d_orth_wq;
            orthogonalize_pw_overlaps_derivative(context.space,
                                                 dST_inv_root, z, dz,
                                                 orth_wq, d_orth_wq);

            std::complex<double> cross(0.0, 0.0);
            for (int a = 0; a < r; ++a)
            {
                cross += std::conj(data.orth_overlap[s][a])
                             * d_orth_wq[a]
                         + std::conj(d_orth[a]) * orth_wq[a];
            }
            const double cross_derivative = -2.0 * cross.real();
            double dq_derivative = 0.0;

            // Branch: Q_alpha contains exactly the orthonormal targets owned
            // by this constraint.  Every target still contributes through the
            // dP cross term above.
            for (int a = 0; a < r; ++a)
            {
                if (context.space.constraint[a] == alpha)
                {
                    dq_derivative += 2.0
                                     * (std::conj(d_orth[a])
                                        * data.orth_overlap[s][a])
                                           .real();
                }
            }
            const double channel
                = (spin == 0) ? channels[alpha].inj_up
                              : channels[alpha].inj_dn;
            const double prefactor
                = data.occupation[s] * mu[alpha] * channel;
            stress_energy[index] -= prefactor
                                    * (dq_derivative + cross_derivative);
        }
    }
}

} // namespace

bool compute_pw_alloc_observables(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<double>& n_alloc,
    std::vector<double>& m_alloc,
    std::string& error)
{
    const int nconstraint = weight_grid.nconstraint();
    n_alloc.assign(nconstraint, 0.0);
    m_alloc.assign(nconstraint, 0.0);
    // Branch: the observable is defined for the collinear scalar-band path.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation observable supports only nspin 1/2";
        return false;
    }
    // Branch: the spinor band layout has two components per band and needs a
    // separate Pauli contraction, which this collinear closure does not form.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation observable requires scalar CPU wavefunctions";
        return false;
    }
    const int nks = psi->get_nk();
    // Branch: no k-point means no occupied wavefunction to project.
    if (nks <= 0)
    {
        error = "PW allocation observable has no k-points";
        return false;
    }
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    for (int ik = 0; ik < nks; ++ik)
    {
        // Branch: every k-point has its own spin label and projector basis.
        if (static_cast<int>(kv.isk.size()) <= ik)
        {
            error = "PW allocation observable has no k-point spin label";
            return false;
        }
        const int nband = psi->get_nbands();
        const int ld_psi = psi->get_nbasis();
        psi->fix_k(ik);
        PwTargetContext context;
        if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                  psi->get_pointer(), nband, ld_psi,
                                  context, error))
        {
            return false;
        }
        const int spin_sign = (kv.isk[ik] == 1) ? -1 : 1;
        const std::complex<double>* psi_k = psi->get_pointer();
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            const std::vector<double>& potential
                = weight_grid.constraint_weight(alpha);
            std::vector<std::vector<std::complex<double>>> vphi;
            std::vector<std::complex<double>> m_raw;
            if (!build_target_products(wfcpw, ik, potential,
                                       context, vphi, m_raw, error))
            {
                return false;
            }
            PwAllocOrthogonalContext orthogonal;
            build_pw_alloc_orthogonal_context(context.space, context.phi,
                                              vphi, m_raw, orthogonal);
            std::vector<double> lambda(nconstraint, 0.0);
            lambda[alpha] = 1.0;
            for (int band = 0; band < nband; ++band)
            {
                const double occupation = occupations(ik, band);
                // Branch: an empty band contributes nothing to Tr(D K_i).
                if (occupation == 0.0)
                {
                    continue;
                }
                const std::complex<double>* band_ptr
                    = psi_k + static_cast<size_t>(band) * ld_psi;
                std::vector<std::complex<double>> raw;
                build_raw_overlap(context, band_ptr, raw);
                const std::vector<std::complex<double>> vpsi
                    = apply_local_potential(wfcpw, ik, potential,
                                            band_ptr, context.npw);
                std::vector<std::complex<double>> delta;
                apply_pw_alloc_delta_h_orthogonal(
                    context.space, orthogonal, vphi,
                    std::vector<std::complex<double>>(band_ptr,
                                                      band_ptr + context.npw),
                    raw, vpsi, lambda, delta);
                std::complex<double> trace(0.0, 0.0);
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    trace += std::conj(band_ptr[ig]) * delta[ig];
                }
                reduce_pool(trace);
                n_alloc[alpha] += occupation * trace.real();
                // Branch: only a two-channel collinear run has a spin
                // difference observable.
                if (nspin == 2)
                {
                    m_alloc[alpha] += spin_sign * occupation * trace.real();
                }
            }
        }
    }
    // Every rank in a pool owns the same local k list after the pool
    // reductions above.  Average the identical copies and sum the disjoint
    // k-point sets of different pools.
    const int npool = GlobalV::KPAR * PARAM.inp.bndpar;
    Parallel_Reduce::reduce_double_allpool(
        npool, GlobalV::NPROC_IN_POOL, n_alloc.data(), nconstraint);
    Parallel_Reduce::reduce_double_allpool(
        npool, GlobalV::NPROC_IN_POOL, m_alloc.data(), nconstraint);
    (void)ucell;
    return true;
}

bool compute_pw_alloc_complement_density(
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<std::vector<double>>& rho_c,
    std::string& error)
{
    // Branch: the frozen force uses the same collinear scalar-band density
    // channels as the allocation observable.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation force supports only nspin 1/2";
        return false;
    }
    // Branch: spinor wavefunctions require a Pauli contraction which this
    // frozen-target PW closure does not form.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation force requires scalar CPU wavefunctions";
        return false;
    }
    // Branch: the force density lives on the same real-space partition as the
    // weight field; an empty grid cannot be integrated.
    if (wfcpw.nrxx <= 0)
    {
        error = "PW allocation force has an empty density grid";
        return false;
    }
    const int nks = psi->get_nk();
    // Branch: no k-point means there is no occupied complement density.
    if (nks <= 0)
    {
        error = "PW allocation force has no k-points";
        return false;
    }
    rho_c.assign(nspin, std::vector<double>(wfcpw.nrxx, 0.0));
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    for (int ik = 0; ik < nks; ++ik)
    {
        // Branch: each k-point carries the spin label selecting its density
        // channel.
        if (static_cast<int>(kv.isk.size()) <= ik)
        {
            error = "PW allocation force has no k-point spin label";
            return false;
        }
        const int spin = (nspin == 2) ? kv.isk[ik] : 0;
        // Guard: an unexpected spin label would write outside rho_c.
        if (spin < 0 || spin >= nspin)
        {
            error = "PW allocation force has an invalid spin label";
            return false;
        }
        const int nband = psi->get_nbands();
        const int ld_psi = psi->get_nbasis();
        psi->fix_k(ik);
        const std::complex<double>* psi_k = psi->get_pointer();
        PwTargetContext context;
        if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                  psi_k, nband, ld_psi, context, error))
        {
            return false;
        }
        std::vector<std::vector<std::complex<double>>> phi_orth;
        try
        {
            build_pw_alloc_orthogonal_phi(context.space, context.phi,
                                          phi_orth);
        }
        catch (const std::invalid_argument& exc)
        {
            error = exc.what();
            return false;
        }
        const int r = context.space.ntarget;
        std::vector<std::complex<double>> work(wfcpw.nmaxgr,
                                               std::complex<double>(0.0, 0.0));
        std::vector<std::complex<double>> porter(wfcpw.nmaxgr,
                                                 std::complex<double>(0.0, 0.0));
        for (int band = 0; band < nband; ++band)
        {
            const double occupation = occupations(ik, band);
            // Branch: an unoccupied band contributes no density.
            if (occupation == 0.0)
            {
                continue;
            }
            const std::complex<double>* band_ptr
                = psi_k + static_cast<size_t>(band) * ld_psi;
            std::vector<std::complex<double>> overlap(
                r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    overlap[a] += std::conj(phi_orth[a][ig]) * band_ptr[ig];
                }
            }
            reduce_pool(overlap);
            std::fill(work.begin(), work.end(),
                      std::complex<double>(0.0, 0.0));
            for (int ig = 0; ig < context.npw; ++ig)
            {
                work[ig] = band_ptr[ig];
            }
            for (int a = 0; a < r; ++a)
            {
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    work[ig] -= phi_orth[a][ig] * overlap[a];
                }
            }
            wfcpw.template recip_to_real<std::complex<double>,
                                         base_device::DEVICE_CPU>(
                work.data(), porter.data(), ik);
            const double weight = occupation / wfcpw.omega;
            for (int ir = 0; ir < wfcpw.nrxx; ++ir)
            {
                rho_c[spin][ir] += weight * std::norm(porter[ir]);
            }
        }
    }
    // Each k-pool owns a disjoint set of k-points while the real-space grid
    // partition is identical across pools.  KP_WORLD therefore connects the
    // same local grid slice in every k-pool; summing there yields the global
    // complement density without dividing by the number of pool ranks.
    if (GlobalV::KPAR > 1)
    {
#ifdef __MPI
        // Guard: an uneven k-pool split has no KP_WORLD, so the local grid
        // slices cannot be matched safely across pools.
        if (KP_WORLD == MPI_COMM_NULL)
        {
            error = "PW allocation force requires evenly divided KPAR pools";
            return false;
        }
        for (int spin = 0; spin < nspin; ++spin)
        {
            MPI_Allreduce(MPI_IN_PLACE, rho_c[spin].data(), wfcpw.nrxx,
                          MPI_DOUBLE, MPI_SUM, KP_WORLD);
        }
#else
        error = "PW allocation force cannot combine KPAR without MPI";
        return false;
#endif
    }
    return true;
}

bool compute_pw_alloc_target_force(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    ModuleBase::matrix& force,
    std::string& error)
{
    // Branch: this closure is the collinear scalar-band force channel.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation target force supports only nspin 1/2";
        return false;
    }
    // Branch: spinor derivatives need a Pauli contraction which is not part
    // of the collinear closure.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation target force requires scalar CPU wavefunctions";
        return false;
    }
    const int nconstraint = weight_grid.nconstraint();
    // Guard: multipliers and density-channel signs define the same operator
    // as the Hamiltonian/force kernel; a mismatch is a wiring bug.
    if (static_cast<int>(mu.size()) != nconstraint
        || static_cast<int>(channels.size()) != nconstraint)
    {
        error = "PW allocation target force has a constraint-count mismatch";
        return false;
    }
    const int nks = psi->get_nk();
    // Branch: no k-point means no occupied target-state derivative.
    if (nks <= 0)
    {
        error = "PW allocation target force has no k-points";
        return false;
    }

    force.create(ucell.nat, 3);
    std::fill(force.c, force.c + ucell.nat * 3, 0.0);
    std::vector<double> force_local(static_cast<size_t>(ucell.nat) * 3, 0.0);
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    try
    {
        for (int ik = 0; ik < nks; ++ik)
        {
            // Branch: every k-point carries the spin label selecting the
            // Hamiltonian channel's multiplier sign.
            if (static_cast<int>(kv.isk.size()) <= ik)
            {
                error = "PW allocation target force has no k-point spin label";
                return false;
            }
            const int spin = (nspin == 2) ? kv.isk[ik] : 0;
            // Guard: an unexpected spin label has no channel coefficient.
            if (spin < 0 || spin >= nspin)
            {
                error = "PW allocation target force has an invalid spin label";
                return false;
            }
            const int nband = psi->get_nbands();
            const int ld_psi = psi->get_nbasis();
            psi->fix_k(ik);
            const std::complex<double>* psi_k = psi->get_pointer();
            PwTargetContext context;
            if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                      psi_k, nband, ld_psi, context, error))
            {
                return false;
            }
            PwAllocBandForceData data;
            if (!build_band_force_data(wfcpw, ik, psi_k, occupations, nband,
                                       ld_psi, weight_grid, context, data,
                                       error))
            {
                return false;
            }
            // Branch: an empty spectrum has no target-direction derivative.
            if (data.nvec == 0)
            {
                continue;
            }
            auto* fs_tools = onsite_p->get_fs_tools();
            // Guard: the custom derivative path needs the same projector
            // table that built the target context.
            if (fs_tools == nullptr || fs_tools->get_nkb() <= 0)
            {
                error = "PW allocation target force has no onsite VKB table";
                return false;
            }
            const int nkb = fs_tools->get_nkb();
            std::vector<std::complex<double>> derivative_overlap(
                static_cast<size_t>(nkb) * data.nvec,
                std::complex<double>(0.0, 0.0));
            for (int ipol = 0; ipol < 3; ++ipol)
            {
                // One derivative GEMM provides d<alpha|psi>, the dST
                // matrix and the d<alpha|W_i Q_c psi> contractions.
                fs_tools->cal_dbecp_f_custom(
                    ik, data.nvec, ipol, data.custom_vectors.data(), ld_psi,
                    derivative_overlap.data());
                // The projector derivative GEMM is plane-wave-partition local.
                // Pool all rows and vectors once before any contraction so
                // every rank forms the same target-space derivative.
                reduce_pool(derivative_overlap);
                for (int atom = 0; atom < ucell.nat; ++atom)
                {
                    accumulate_target_force_direction(
                        context, data, derivative_overlap, nkb, atom, ipol,
                        spin, mu, channels, ucell.tpiba, force_local);
                }
            }
        }
    }
    catch (const std::exception& exc)
    {
        error = std::string("PW allocation target force: ") + exc.what();
        return false;
    }

    // Every rank in a pool has the same reduced wavefunction contractions;
    // average those copies and sum the disjoint k-point sets of all pools.
    const int npool = GlobalV::KPAR * PARAM.inp.bndpar;
    Parallel_Reduce::reduce_double_allpool(
        npool, GlobalV::NPROC_IN_POOL, force_local.data(), ucell.nat * 3);
    for (int i = 0; i < ucell.nat * 3; ++i)
    {
        force.c[i] = force_local[i];
    }
    return true;
}

bool compute_pw_alloc_target_stress(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    ModuleBase::matrix& stress,
    std::string& error)
{
    // Branch: this closure is the collinear scalar-band stress channel.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation target stress supports only nspin 1/2";
        return false;
    }
    // Branch: spinor strain derivatives need a Pauli contraction which is not
    // part of the collinear closure.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation target stress requires scalar CPU wavefunctions";
        return false;
    }
    const int nconstraint = weight_grid.nconstraint();
    // Guard: multipliers and density-channel signs define the same operator
    // as the Hamiltonian/stress kernel; a mismatch is a wiring bug.
    if (static_cast<int>(mu.size()) != nconstraint
        || static_cast<int>(channels.size()) != nconstraint)
    {
        error = "PW allocation target stress has a constraint-count mismatch";
        return false;
    }
    const int nks = psi->get_nk();
    // Branch: no k-point means no occupied target-state strain derivative.
    if (nks <= 0)
    {
        error = "PW allocation target stress has no k-points";
        return false;
    }
    // Guard: the volume-normalized stress needs a positive cell volume.
    if (wfcpw.omega <= 0.0)
    {
        error = "PW allocation target stress has a non-positive cell volume";
        return false;
    }

    stress.create(3, 3);
    std::fill(stress.c, stress.c + 9, 0.0);
    std::vector<double> stress_energy(9, 0.0);
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    try
    {
        for (int ik = 0; ik < nks; ++ik)
        {
            // Branch: every k-point carries the spin label selecting the
            // Hamiltonian channel's multiplier sign.
            if (static_cast<int>(kv.isk.size()) <= ik)
            {
                error = "PW allocation target stress has no k-point spin label";
                return false;
            }
            const int spin = (nspin == 2) ? kv.isk[ik] : 0;
            // Guard: an unexpected spin label has no channel coefficient.
            if (spin < 0 || spin >= nspin)
            {
                error = "PW allocation target stress has an invalid spin label";
                return false;
            }
            const int nband = psi->get_nbands();
            const int ld_psi = psi->get_nbasis();
            psi->fix_k(ik);
            const std::complex<double>* psi_k = psi->get_pointer();
            PwTargetContext context;
            if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                      psi_k, nband, ld_psi, context, error))
            {
                return false;
            }
            PwAllocBandForceData data;
            if (!build_band_force_data(wfcpw, ik, psi_k, occupations, nband,
                                       ld_psi, weight_grid, context, data,
                                       error))
            {
                return false;
            }
            // Branch: an empty occupied spectrum has no strain derivative.
            if (data.nvec == 0)
            {
                continue;
            }
            auto* fs_tools = onsite_p->get_fs_tools();
            // Guard: the custom derivative path needs the same projector
            // table that built the target context.
            if (fs_tools == nullptr || fs_tools->get_nkb() <= 0)
            {
                error = "PW allocation target stress has no onsite VKB table";
                return false;
            }
            const int nkb = fs_tools->get_nkb();
            std::vector<std::complex<double>> derivative_overlap(
                static_cast<size_t>(nkb) * data.nvec,
                std::complex<double>(0.0, 0.0));
            for (int ipol = 0; ipol < 3; ++ipol)
            {
                for (int jpol = 0; jpol < 3; ++jpol)
                {
                    // Each component reuses the same custom-vector GEMM.
                    // Both off-diagonal directions are evaluated so the
                    // final tensor is explicitly symmetrized.
                    fs_tools->cal_dbecp_s_custom(
                        ik, data.nvec, ipol, jpol, data.custom_vectors.data(),
                        ld_psi, derivative_overlap.data());
                    // The projector derivative is plane-wave-partition
                    // local; pool it before contracting with the small target
                    // space so every rank forms the same strain derivative.
                    reduce_pool(derivative_overlap);
                    accumulate_target_stress_component(
                        context, data, derivative_overlap, ipol, jpol, spin,
                        mu, channels, stress_energy);
                }
            }
        }
    }
    catch (const std::exception& exc)
    {
        error = std::string("PW allocation target stress: ") + exc.what();
        return false;
    }

    // Every rank in a pool has the same reduced wavefunction contractions;
    // average those copies and sum the disjoint k-point sets of all pools.
    const int npool = GlobalV::KPAR * PARAM.inp.bndpar;
    Parallel_Reduce::reduce_double_allpool(
        npool, GlobalV::NPROC_IN_POOL, stress_energy.data(), 9);

    const double inv_omega = 1.0 / wfcpw.omega;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = a; b < 3; ++b)
        {
            // ABACUS reports stress as -(1/Omega) dE/d eps.  The accumulator
            // stores s = -dE/d eps (the same convention as the grid kernel),
            // so the physical symmetrized tensor is +sym[s]/Omega.  The
            // off-diagonal derivative is already the symmetric-strain one.
            const double value
                = 0.5
                  * (stress_energy[a * 3 + b] + stress_energy[b * 3 + a])
                  * inv_omega;
            stress(a, b) = value;
            // Branch: mirror the off-diagonal entry for the total-stress
            // consumer, matching the grid constraint kernel.
            if (a != b)
            {
                stress(b, a) = value;
            }
        }
    }
    return true;
}

} // namespace alloc

namespace hamilt
{

class AllocPwCache
{
  public:
    struct Target
    {
        bool valid = false;
        int ik = -1;
        const constraint::WeightGrid* weight_grid = nullptr;
        int target_l = -1;
        long generation = 0;
        alloc::PwTargetContext context;
    };

    struct Product
    {
        bool valid = false;
        int ik = -1;
        int spin = -1;
        long target_generation = -1;
        std::vector<double> lambda;
        std::vector<std::vector<std::complex<double>>> vphi;
        alloc::PwAllocOrthogonalContext orthogonal;
    };

    Target target;
    Product product;
};

template <typename T, typename Device>
AllocConstraintPW<OperatorPW<T, Device>>::AllocConstraintPW(
    const int* isk_in,
    const UnitCell* ucell_in,
    const ModulePW::PW_Basis_K* wfcpw_in)
    : isk(isk_in), ucell(ucell_in), wfcpw(wfcpw_in),
      cache(new AllocPwCache())
{
    this->classname = "AllocConstraintPW";
    this->cal_type = calculation_type::pw_onsite;
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
AllocConstraintPW<OperatorPW<T, Device>>::AllocConstraintPW(
    const AllocConstraintPW<OperatorPW<T_in, Device_in>>* op)
    : isk(op->get_isk()),
      ucell(op->get_ucell()),
      wfcpw(op->get_wfcpw()),
      cache(new AllocPwCache())
{
    this->classname = "AllocConstraintPW";
    this->cal_type = calculation_type::pw_onsite;
}

template <typename T, typename Device>
AllocConstraintPW<OperatorPW<T, Device>>::~AllocConstraintPW() = default;

template <typename T, typename Device>
void AllocConstraintPW<OperatorPW<T, Device>>::init(const int ik_in)
{
    // The allocation operator reads the same onsite tabulation as DFT+U and
    // DeltaSpin.  Refresh it here because this operator can be the only PW
    // onsite consumer in an allocation-only run.
    auto* onsite_p
        = projectors::OnsiteProjector<double, Device>::get_instance();
    // Branch: the projector singleton exists even before init; only a
    // tabulated instance has rows to refresh.
    if (onsite_p != nullptr && onsite_p->get_tot_nproj() > 0)
    {
        onsite_p->tabulate_atomic(ik_in);
    }
    OperatorPW<T, Device>::init(ik_in);
}

template <typename T, typename Device>
void AllocConstraintPW<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node) const
{
    (void)is_first_node;
    // Branch: only the double-precision CPU path owns the host projector and
    // FFT workspace used by the PW allocation algebra.  This is a runtime
    // branch (not if constexpr) to keep the project C++14-compatible.
    if (!std::is_same<T, std::complex<double>>::value
        || !std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        return;
    }
    // Branch: this operator is inert unless the allocation switch is on.
    if (!PARAM.inp.constraint || !PARAM.inp.constraint_alloc_switch)
    {
        return;
    }
    constraint::ConstraintLoop& loop
        = constraint::ConstraintLoop::instance();
    // Branch: before the loop is armed (or in the legacy grid mode) the
    // normal Hamiltonian must remain untouched.
    if (!loop.enabled() || !loop.allocation_mode())
    {
        return;
    }
    // Guard: the scalar PW allocation algebra has one component per band.
    if (npol != 1)
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation currently supports only collinear npol = 1");
    }
    // Guard: dipole weights are a global coordinate, not per-atom target
    // ownership; the allocation backend is not defined for them.
    if (loop.dipole_constraint())
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation does not support dipole constraints");
    }
    // Branch: an empty band range is a valid no-op.
    if (nbands <= 0)
    {
        return;
    }
    const int spin = (PARAM.inp.nspin == 2) ? this->isk[this->ik] : 0;
    // Guard: an unexpected spin label would silently apply the wrong
    // channel sign to a spin constraint.
    if (spin != 0 && spin != 1)
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation received an invalid spin label");
    }
    const constraint::WeightGrid& wg = loop.weight_grid();
    // Guard: the FFT potential and Wavefunction grids must coincide; the
    // constraint configuration already rejects double_grid.
    if (wg.nrxx() != this->wfcpw->nrxx)
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation weight and wavefunction grids differ");
    }

    std::vector<double> potential;
    std::vector<double> lambda;
    std::string error;
    if (!alloc::build_mu_potential(wg, loop, spin, potential, lambda,
                                   error))
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
    }
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    // The target space is geometry/ik keyed and independent of lambda.  Reuse
    // it across every band, CG/Davidson application and unchanged SCF frame.
    if (!cache->target.valid || cache->target.ik != this->ik
        || cache->target.weight_grid != &wg
        || cache->target.target_l != PARAM.inp.constraint_alloc_target_l)
    {
        if (!alloc::build_target_context(
                onsite_p, wg, this->ik,
                PARAM.inp.constraint_alloc_target_l,
                reinterpret_cast<const std::complex<double>*>(tmpsi_in),
                nbands, nbasis,
                cache->target.context, error))
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        cache->target.valid = true;
        cache->target.ik = this->ik;
        cache->target.weight_grid = &wg;
        cache->target.target_l = PARAM.inp.constraint_alloc_target_l;
        ++cache->target.generation;
        // A new target space invalidates every multiplier-dependent product.
        cache->product.valid = false;
    }
    const alloc::PwTargetContext& context = cache->target.context;
    const int npw = ngk_ik > 0 ? ngk_ik : context.npw;
    // Guard: the operator chain and onsite tabulation must describe one
    // plane-wave partition.
    if (npw != context.npw || npw > nbasis)
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation plane-wave partition mismatch");
    }
    // V|alpha>, <alpha|V|beta> and the orthonormal products depend only on the
    // local potential.  In a fixed-mu frame lambda is bitwise constant, so the
    // expensive FFTs and R x R algebra run once instead of once per band.
    if (!cache->product.valid || cache->product.ik != this->ik
        || cache->product.spin != spin
        || cache->product.target_generation != cache->target.generation
        || cache->product.lambda != lambda)
    {
        std::vector<std::complex<double>> m_raw;
        if (!alloc::build_target_products(*this->wfcpw, this->ik, potential,
                                          context, cache->product.vphi, m_raw,
                                          error))
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        alloc::build_pw_alloc_orthogonal_context(
            context.space, context.phi, cache->product.vphi, m_raw,
            cache->product.orthogonal);
        cache->product.valid = true;
        cache->product.ik = this->ik;
        cache->product.spin = spin;
        cache->product.target_generation = cache->target.generation;
        cache->product.lambda = lambda;
    }

    for (int ib = 0; ib < nbands; ++ib)
    {
        const std::complex<double>* psi_band
            = reinterpret_cast<const std::complex<double>*>(tmpsi_in)
              + static_cast<size_t>(ib) * nbasis;
        std::complex<double>* hpsi_band
            = reinterpret_cast<std::complex<double>*>(tmhpsi)
              + static_cast<size_t>(ib) * nbasis;
        std::vector<std::complex<double>> raw;
        alloc::build_raw_overlap(context, psi_band, raw);
        const std::vector<std::complex<double>> vpsi
            = alloc::apply_local_potential(*this->wfcpw, this->ik,
                                           potential, psi_band, npw);
        std::vector<std::complex<double>> delta;
        alloc::apply_pw_alloc_delta_h_orthogonal(
            context.space, cache->product.orthogonal, cache->product.vphi,
            std::vector<std::complex<double>>(psi_band, psi_band + npw),
            raw, vpsi, lambda, delta);
        for (int ig = 0; ig < npw; ++ig)
        {
            hpsi_band[ig] += delta[ig];
        }
    }
}

template class AllocConstraintPW<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class AllocConstraintPW<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;

#if ((defined __CUDA) || (defined __ROCM))
template class AllocConstraintPW<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class AllocConstraintPW<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
#endif

} // namespace hamilt
