#include "op_pw_alloc.h"

#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_base/timer.h"
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

void apply_local_potential_reuse(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const std::complex<double>* input,
    const int npw,
    std::vector<std::complex<double>>& porter,
    std::complex<double>* output)
{
    // The PW transform reads only the first npw reciprocal coefficients and
    // overwrites all nrxx real-space values.  Keeping the input and output in
    // place therefore avoids two nmaxgr-sized clears plus a full copy per band.
    pw.template recip_to_real<std::complex<double>, base_device::DEVICE_CPU>(
        input, porter.data(), ik);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        porter[ir] *= potential[ir];
    }
    pw.template real_to_recip<std::complex<double>, base_device::DEVICE_CPU>(
        porter.data(), output, ik);
}

std::vector<std::complex<double>> apply_local_potential(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const std::complex<double>* input,
    const int npw)
{
    std::vector<std::complex<double>> porter(pw.nmaxgr,
                                             std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> result(npw,
                                             std::complex<double>(0.0, 0.0));
    apply_local_potential_reuse(pw, ik, potential, input, npw, porter,
                                result.data());
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

bool build_mu_coefficients(const constraint::WeightGrid& weight_grid,
                           const constraint::ConstraintLoop& loop,
                           const int spin,
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

    lambda.assign(nconstraint, 0.0);
    for (int alpha = 0; alpha < nconstraint; ++alpha)
    {
        // Branch A: spin-up (or the single nspin=1 channel) uses inj_up.
        // Branch B: spin-down uses inj_dn, which is the opposite sign for a
        // spin constraint and the same sign for a charge constraint.
        const double sign = (spin == 0) ? channels[alpha].inj_up
                                       : channels[alpha].inj_dn;
        lambda[alpha] = mu[alpha] * sign;
    }
    return true;
}

void build_mu_potential(const constraint::WeightGrid& weight_grid,
                        const std::vector<double>& lambda,
                        std::vector<double>& potential)
{
    potential.assign(weight_grid.nrxx(), 0.0);
    const auto& weights = weight_grid.constraint_weights();
    for (int alpha = 0; alpha < weight_grid.nconstraint(); ++alpha)
    {
        // Branch: a zero multiplier leaves the accumulated potential unchanged.
        if (lambda[alpha] == 0.0)
        {
            continue;
        }
        for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
        {
            potential[ir] += lambda[alpha] * weights[alpha][ir];
        }
    }
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
    // All target FFTs use the same real-space work array; allocating it once
    // avoids R repeated nmaxgr allocations and clears on every lambda change.
    std::vector<std::complex<double>> porter(pw.nmaxgr,
                                             std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        apply_local_potential_reuse(pw, ik, potential, context.phi[a].data(),
                                    context.npw, porter, vphi[a].data());
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

void build_raw_overlap_local(const PwTargetContext& context,
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
}

void build_raw_overlap(const PwTargetContext& context,
                       const std::complex<double>* psi,
                       std::vector<std::complex<double>>& raw)
{
    build_raw_overlap_local(context, psi, raw);
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

bool build_pw_alloc_subspace_matrices_impl(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<std::vector<std::complex<double>>>& a_sub,
    std::vector<int>& spin_labels,
    std::string& error)
{
    (void)ucell;
    // Branch: the scalar PW allocation observable and operator are defined
    // only for collinear bands.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation subspace supports only nspin 1/2";
        return false;
    }
    // Branch: spinor bands have two components per band and need a Pauli
    // contraction, which this closure does not form.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation subspace requires scalar CPU wavefunctions";
        return false;
    }
    const int nks = psi->get_nk();
    const int nband = psi->get_nbands();
    const int ld_psi = psi->get_nbasis();
    // Branch: an empty band or k-point set has no subspace to constrain.
    if (nks <= 0 || nband <= 0)
    {
        error = "PW allocation subspace has no k-points or bands";
        return false;
    }
    // Guard: the target weights and wavefunction transforms must live on the
    // same real-space partition.
    if (weight_grid.nrxx() != wfcpw.nrxx)
    {
        error = "PW allocation subspace weight/wavefunction grid mismatch";
        return false;
    }
    const int nconstraint = weight_grid.nconstraint();
    // Branch: a run without constraints has no A_i matrices to build.
    if (nconstraint <= 0)
    {
        a_sub.clear();
        spin_labels.assign(nks, 0);
        return true;
    }

    a_sub.assign(nks, {});
    spin_labels.assign(nks, 0);
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    const double inverse_grid_size = 1.0 / static_cast<double>(wfcpw.nxyz);
    for (int ik = 0; ik < nks; ++ik)
    {
        // Guard: every k-point needs the spin label selecting its channel.
        if (static_cast<int>(kv.isk.size()) <= ik)
        {
            error = "PW allocation subspace has no k-point spin label";
            return false;
        }
        const int spin = (nspin == 2) ? kv.isk[ik] : 0;
        // Guard: an unexpected spin label would select the wrong channel.
        if (spin < 0 || spin >= nspin)
        {
            error = "PW allocation subspace has an invalid spin label";
            return false;
        }
        spin_labels[ik] = spin;
        psi->fix_k(ik);
        PwTargetContext context;
        if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                  psi->get_pointer(), nband, ld_psi, context,
                                  error))
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
            error = std::string("PW allocation subspace: ") + exc.what();
            return false;
        }
        const int r = context.space.ntarget;
        // Orthonormal target overlaps and complement vectors are band
        // independent inputs to all constraint matrices.  Keep them for the
        // current k-point only, so memory stays bounded by one k-block.
        std::vector<std::vector<std::complex<double>>> orth(
            nband, std::vector<std::complex<double>>(
                       r, std::complex<double>(0.0, 0.0)));
        std::vector<std::vector<std::complex<double>>> complement_real(
            nband, std::vector<std::complex<double>>(
                       wfcpw.nmaxgr, std::complex<double>(0.0, 0.0)));
        std::vector<std::complex<double>> complement(
            context.npw, std::complex<double>(0.0, 0.0));
        const std::complex<double>* psi_k = psi->get_pointer();
        for (int band = 0; band < nband; ++band)
        {
            const std::complex<double>* band_ptr
                = psi_k + static_cast<size_t>(band) * ld_psi;
            std::vector<std::complex<double>> raw;
            build_raw_overlap(context, band_ptr, raw);
            orthogonalize_pw_overlaps(context.space, raw, orth[band]);
            // Q_c|psi_n> = |psi_n> - sum_a |phi_a><phi_a|psi_n>.
            std::copy(band_ptr, band_ptr + context.npw, complement.begin());
            for (int a = 0; a < r; ++a)
            {
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    complement[ig] -= phi_orth[a][ig] * orth[band][a];
                }
            }
            wfcpw.template recip_to_real<std::complex<double>,
                                         base_device::DEVICE_CPU>(
                complement.data(), complement_real[band].data(), ik);
        }

        std::vector<std::complex<double>>& matrices = a_sub[ik];
        matrices.assign(static_cast<size_t>(nconstraint) * nband * nband,
                        std::complex<double>(0.0, 0.0));
        // The Q_c w_i Q_c block is diagonal in real space but not in the band
        // basis.  Accumulate the local-grid contribution first, then reduce
        // it before adding the already-global target-projector term.
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            std::complex<double>* block
                = matrices.data()
                  + static_cast<size_t>(alpha) * nband * nband;
            const std::vector<double>& weights
                = weight_grid.constraint_weight(alpha);
            for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
            {
                const double weight = weights[ir] * inverse_grid_size;
                // Branch: a zero weight contributes nothing to this grid row.
                if (weight == 0.0)
                {
                    continue;
                }
                for (int m = 0; m < nband; ++m)
                {
                    const std::complex<double> left
                        = std::conj(complement_real[m][ir]) * weight;
                    for (int n = 0; n < nband; ++n)
                    {
                        block[static_cast<size_t>(m) * nband + n]
                            += left * complement_real[n][ir];
                    }
                }
            }
        }
        reduce_pool(matrices);
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            std::complex<double>* block
                = matrices.data()
                  + static_cast<size_t>(alpha) * nband * nband;
            for (int a = 0; a < r; ++a)
            {
                // Branch: only target projectors owned by this constraint
                // enter Q_i; unowned rows belong to P only.
                if (context.space.constraint[a] != alpha)
                {
                    continue;
                }
                for (int m = 0; m < nband; ++m)
                {
                    const std::complex<double> left
                        = std::conj(orth[m][a]);
                    for (int n = 0; n < nband; ++n)
                    {
                        block[static_cast<size_t>(m) * nband + n]
                            += left * orth[n][a];
                    }
                }
            }
            // The analytic operator is Hermitian.  Remove roundoff asymmetry
            // before it enters the generalized eigensolver.
            for (int m = 0; m < nband; ++m)
            {
                for (int n = m + 1; n < nband; ++n)
                {
                    const std::complex<double> value
                        = 0.5
                          * (block[static_cast<size_t>(m) * nband + n]
                             + std::conj(block[static_cast<size_t>(n) * nband + m]));
                    block[static_cast<size_t>(m) * nband + n] = value;
                    block[static_cast<size_t>(n) * nband + m]
                        = std::conj(value);
                }
            }
        }
    }
    return true;
}

void build_vphi_overlap_local(
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::complex<double>* psi,
    const int npw,
    std::vector<std::complex<double>>& braw)
{
    braw.assign(vphi.size(), std::complex<double>(0.0, 0.0));
    for (size_t a = 0; a < vphi.size(); ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            braw[a] += std::conj(vphi[a][ig]) * psi[ig];
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

bool build_pw_alloc_subspace_matrices(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<std::vector<std::complex<double>>>& a_sub,
    std::vector<int>& spin_labels,
    std::string& error)
{
    // The implementation shares the target-context and overlap helpers with
    // the operator/observable paths; expose it through the public alloc API.
    return build_pw_alloc_subspace_matrices_impl(
        ucell, psi, kv, wfcpw, nspin, weight_grid, target_l, a_sub,
        spin_labels, error);
}

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
        // The normalized target basis is independent of the constrained
        // channel.  Build it once per k and reuse it for every band and every
        // alpha; the old observable loop rebuilt the full target products for
        // each alpha separately.
        std::vector<std::vector<std::complex<double>>> phi_orth;
        try
        {
            build_pw_alloc_orthogonal_phi(context.space, context.phi,
                                          phi_orth);
        }
        catch (const std::invalid_argument& exc)
        {
            error = std::string("PW allocation observable: ") + exc.what();
            return false;
        }
        const int spin_sign = (kv.isk[ik] == 1) ? -1 : 1;
        const std::complex<double>* psi_k = psi->get_pointer();
        const int npw = context.npw;
        // The complementary state is transformed to real space once per band.
        // Every constraint then reduces to a weighted norm on that same grid,
        // instead of paying one FFT pair per constraint.
        std::vector<std::complex<double>> complement_real(
            wfcpw.nmaxgr, std::complex<double>(0.0, 0.0));
        const double inverse_grid_size = 1.0 / static_cast<double>(wfcpw.nxyz);
        for (int band = 0; band < nband; ++band)
        {
            const double occupation = occupations(ik, band);
            // Branch: an empty band contributes nothing to q.
            if (occupation == 0.0)
            {
                continue;
            }
            const std::complex<double>* band_ptr
                = psi_k + static_cast<size_t>(band) * ld_psi;
            std::vector<std::complex<double>> raw;
            build_raw_overlap(context, band_ptr, raw);
            std::vector<std::complex<double>> overlap;
            orthogonalize_pw_overlaps(context.space, raw, overlap);

            // Q_c|psi> = |psi> - sum_a |phi_a><phi_a|psi>.
            std::vector<std::complex<double>> complement(
                npw, std::complex<double>(0.0, 0.0));
            for (int ig = 0; ig < npw; ++ig)
            {
                complement[ig] = band_ptr[ig];
            }
            for (int a = 0; a < context.space.ntarget; ++a)
            {
                for (int ig = 0; ig < npw; ++ig)
                {
                    complement[ig] -= phi_orth[a][ig] * overlap[a];
                }
            }

            std::vector<double> q_alpha(nconstraint, 0.0);
            std::vector<double> qc_alpha(nconstraint, 0.0);
            for (int a = 0; a < context.space.ntarget; ++a)
            {
                const int owner = context.space.constraint[a];
                // Branch: unowned target projectors belong to P but not to a
                // constrained Q_i, so they do not enter any q_alpha.  The raw
                // overlap was already pool-reduced, hence this contribution is
                // global and must not enter the later collective reduction.
                if (owner >= 0)
                {
                    q_alpha[owner] += std::norm(overlap[a]);
                }
            }
            // |Q_c psi(r)|^2 carries all the band dependence; the constraint
            // weights are diagonal in real space, so one transform suffices.
            wfcpw.template recip_to_real<std::complex<double>,
                                         base_device::DEVICE_CPU>(
                complement.data(), complement_real.data(), ik);
            for (int alpha = 0; alpha < nconstraint; ++alpha)
            {
                const std::vector<double>& weight
                    = weight_grid.constraint_weight(alpha);
                double qc = 0.0;
                for (int ir = 0; ir < wfcpw.nrxx; ++ir)
                {
                    qc += std::norm(complement_real[ir]) * weight[ir];
                }
                qc_alpha[alpha] += qc * inverse_grid_size;
            }
            // The Q_c contribution is rank-local, so reduce it separately
            // from the already global Q_i contribution.  This keeps the
            // collective count at one per band without double counting Q_i.
            reduce_pool(qc_alpha);
            for (int alpha = 0; alpha < nconstraint; ++alpha)
            {
                q_alpha[alpha] += qc_alpha[alpha];
                n_alloc[alpha] += occupation * q_alpha[alpha];
                // Branch: only a two-channel collinear run has a spin
                // difference observable.
                if (nspin == 2)
                {
                    m_alloc[alpha]
                        += spin_sign * occupation * q_alpha[alpha];
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

    struct LocalPotential
    {
        bool valid = false;
        const constraint::WeightGrid* weight_grid = nullptr;
        std::vector<double> lambda;
        std::vector<double> values;
    };

    // Targets are indexed by k-point.  Keeping one context per k-point avoids
    // rebuilding the onsite tabulation whenever the Hamiltonian switches k.
    std::vector<Target> targets;
    // Products are much larger than targets, so retain only a bounded window.
    // The window is keyed by ik; for small k sets every k keeps its product.
    std::vector<Product> products;
    // Monotonic generation for target rebuilds.  A product is valid only when
    // its generation matches the target context from the same k slot.
    long next_target_generation = 0;
    // nspin is at most 2 on the supported collinear path.  Separate caches
    // avoid rebuilding the O(nrxx) potential when k-points alternate spin.
    LocalPotential local_potential[2];
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
void AllocConstraintPW<OperatorPW<T, Device>>::enable_fold_potential(
    const int grid_size)
{
    // Branch: only the double-precision CPU allocation algebra can share the
    // host Veff real-space workspace.  Other instantiations keep the standalone
    // transform path.
    if (!std::is_same<T, std::complex<double>>::value
        || !std::is_same<Device, base_device::DEVICE_CPU>::value
        || grid_size <= 0)
    {
        return;
    }
#ifdef __DSP
    // The DSP Veff path uses a bundled convolution API and cannot accept an
    // extra real-space potential without falling back to a second transform.
    return;
#else
    this->fold_potential_grid = grid_size;
#endif
}

template <typename T, typename Device>
const std::vector<double>*
AllocConstraintPW<OperatorPW<T, Device>>::get_fold_potential(
    const int spin, bool& can_fold) const
{
    can_fold = false;
    // Branch: the operator is inactive before the constraint loop is armed or
    // when the user selected the legacy grid backend.
    if (!PARAM.inp.constraint || !PARAM.inp.constraint_alloc_switch)
    {
        return nullptr;
    }
    // Branch: only the host double-precision path can feed Veff's CPU FFT
    // workspace.  Returning early also avoids building the potential cache
    // when a device build queries this interface.
    if (!std::is_same<T, std::complex<double>>::value
        || !std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        return nullptr;
    }
    constraint::ConstraintLoop& loop
        = constraint::ConstraintLoop::instance();
    // Branch: an unarmed loop leaves the ordinary Hamiltonian untouched.
    if (!loop.enabled() || !loop.allocation_mode())
    {
        return nullptr;
    }
    // Guard: this cache is indexed by the two collinear spin channels.
    if (spin != 0 && spin != 1)
    {
        return nullptr;
    }
    const constraint::WeightGrid& wg = loop.weight_grid();
    // Branch: a Veff on a different real-space partition cannot consume the
    // allocation potential.  act() keeps the standalone path (and reports the
    // grid mismatch there).
    if (wg.nrxx() != this->wfcpw->nrxx)
    {
        return nullptr;
    }
    const std::vector<double>& mu = loop.mu_eff();
    // Branch: an identically zero multiplier makes every allocation term
    // vanish, so neither Veff nor act() should touch the potential cache.
    if (static_cast<int>(mu.size()) == wg.nconstraint() && !mu.empty()
        && std::all_of(mu.begin(), mu.end(),
                       [](const double value) { return value == 0.0; }))
    {
        return nullptr;
    }
    std::string error;
    std::vector<double> lambda;
    if (!alloc::build_mu_coefficients(wg, loop, spin, lambda, error))
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::get_fold_potential", error);
    }
    // Branch: channel signs can cancel every effective multiplier even when
    // the stored ALM multipliers are nonzero.
    if (std::all_of(lambda.begin(), lambda.end(),
                    [](const double value) { return value == 0.0; }))
    {
        return nullptr;
    }
    AllocPwCache::LocalPotential& cached = cache->local_potential[spin];
    // The potential depends only on the weight grid and effective lambda.  A
    // fixed-mu frame reuses this array across every band and every Hamiltonian
    // application instead of rebuilding it in each call.
    if (!cached.valid || cached.weight_grid != &wg
        || cached.lambda != lambda)
    {
        alloc::build_mu_potential(wg, lambda, cached.values);
        cached.valid = true;
        cached.weight_grid = &wg;
        cached.lambda = lambda;
    }
    // can_fold and act() use the same grid key, so Veff never adds V|psi> and
    // then leaves the allocation operator to add it a second time.
    can_fold = this->fold_potential_grid == wg.nrxx();
    return &cached.values;
}

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

    std::string error;
    bool folded_potential = false;
    const std::vector<double>* potential_ptr
        = get_fold_potential(spin, folded_potential);
    // Branch: zero multipliers or an unarmed loop leave the allocation
    // operator inert.  The potential cache is shared with Veff so this is the
    // same decision that controlled whether V|psi> was already added.
    if (potential_ptr == nullptr)
    {
        return;
    }
    const std::vector<double>& potential = *potential_ptr;
    std::vector<double> lambda;
    if (!alloc::build_mu_coefficients(wg, loop, spin, lambda, error))
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
    }
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    const int nks = std::max(1, this->wfcpw->nks);
    // Branch: the cache is first used after the wavefunction basis is known,
    // so allocate its per-k target slots lazily from the actual k count.
    if (static_cast<int>(cache->targets.size()) < nks)
    {
        cache->targets.resize(nks);
    }
    AllocPwCache::Target& target = cache->targets[this->ik];
    // The target space is geometry/ik keyed and independent of lambda.  Reuse
    // it across every band, CG/Davidson application and unchanged SCF frame.
    if (!target.valid || target.weight_grid != &wg
        || target.target_l != PARAM.inp.constraint_alloc_target_l)
    {
        ModuleBase::timer::start("AllocConstraintPW", "target_context");
        const bool built = alloc::build_target_context(
            onsite_p, wg, this->ik, PARAM.inp.constraint_alloc_target_l,
            reinterpret_cast<const std::complex<double>*>(tmpsi_in), nbands,
            nbasis, target.context, error);
        ModuleBase::timer::end("AllocConstraintPW", "target_context");
        if (!built)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        target.valid = true;
        target.ik = this->ik;
        target.weight_grid = &wg;
        target.target_l = PARAM.inp.constraint_alloc_target_l;
        target.generation = ++cache->next_target_generation;
    }
    const alloc::PwTargetContext& context = target.context;
    const int npw = ngk_ik > 0 ? ngk_ik : context.npw;
    // Guard: the operator chain and onsite tabulation must describe one
    // plane-wave partition.
    if (npw != context.npw || npw > nbasis)
    {
        ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
            "PW allocation plane-wave partition mismatch");
    }
    // Products are bounded in memory but keep one slot per k when nks is
    // small.  This is enough to survive the k alternation inside a Davidson
    // sweep while avoiding O(nks) copies of vphi on large calculations.
    constexpr int product_cache_limit = 8;
    const int product_slots
        = std::max(1, std::min(nks, product_cache_limit));
    // Branch: a change in the k count (or the first call) rebuilds the slot
    // table; existing products are invalidated naturally by their ik keys.
    if (static_cast<int>(cache->products.size()) != product_slots)
    {
        cache->products.resize(product_slots);
    }
    const int product_slot = this->ik % product_slots;
    AllocPwCache::Product& product = cache->products[product_slot];
    // V|alpha>, <alpha|V|beta> and the orthonormal products depend only on the
    // local potential.  In a fixed-mu frame lambda is bitwise constant, so the
    // expensive FFTs and R x R algebra run once instead of once per band.
    if (!product.valid || product.ik != this->ik
        || product.spin != spin
        || product.target_generation != target.generation
        || product.lambda != lambda)
    {
        std::vector<std::complex<double>> m_raw;
        ModuleBase::timer::start("AllocConstraintPW", "target_products");
        const bool built = alloc::build_target_products(
            *this->wfcpw, this->ik, potential, context, product.vphi, m_raw,
            error);
        ModuleBase::timer::end("AllocConstraintPW", "target_products");
        if (!built)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        alloc::build_pw_alloc_orthogonal_context(
            context.space, context.phi, product.vphi, m_raw,
            product.orthogonal);
        product.valid = true;
        product.ik = this->ik;
        product.spin = spin;
        product.target_generation = target.generation;
        product.lambda = lambda;
    }

    const int ntarget = context.space.ntarget;
    // Keep the batched target-overlap storage bounded for large band counts.
    // Eight bands still cut two reductions per band down to one per chunk.
    const int band_chunk = std::min(nbands, 8);
    std::vector<std::vector<std::complex<double>>> raw_chunk(band_chunk);
    std::vector<std::vector<std::complex<double>>> braw_chunk(band_chunk);
    std::vector<std::vector<std::complex<double>>> vpsi_chunk(band_chunk);
    // Branch A (folded): Veff already supplied V|psi> in tmhpsi, so neither
    // the band FFT workspace nor a local V|psi> array is needed.  Branch B
    // (standalone): retain the original per-band transform.
    std::vector<std::complex<double>> fft_porter;
    if (!folded_potential)
    {
        fft_porter.assign(this->wfcpw->nmaxgr,
                          std::complex<double>(0.0, 0.0));
    }
    const std::vector<std::complex<double>> empty_vpsi;
    std::vector<std::complex<double>> delta;
    for (int band_start = 0; band_start < nbands; band_start += band_chunk)
    {
        const int chunk_bands = std::min(band_chunk, nbands - band_start);
        const size_t overlap_count = static_cast<size_t>(chunk_bands) * ntarget;
        for (int local_band = 0; local_band < chunk_bands; ++local_band)
        {
            const int ib = band_start + local_band;
            const std::complex<double>* psi_band
                = reinterpret_cast<const std::complex<double>*>(tmpsi_in)
                  + static_cast<size_t>(ib) * nbasis;
            alloc::build_raw_overlap_local(context, psi_band,
                                           raw_chunk[local_band]);
            // Branch: the standalone path must form V|psi> here; the folded
            // path receives it from Veff and only forms the projector terms.
            if (!folded_potential)
            {
                vpsi_chunk[local_band].assign(
                    npw, std::complex<double>(0.0, 0.0));
                alloc::apply_local_potential_reuse(
                    *this->wfcpw, this->ik, potential, psi_band, npw,
                    fft_porter, vpsi_chunk[local_band].data());
            }
            alloc::build_vphi_overlap_local(product.vphi, psi_band, npw,
                                            braw_chunk[local_band]);
        }
        // Pack the chunk's <phi|psi> and <Vphi|psi> into one collective.  The
        // old path paid two pool reductions per band, which dominates small-R
        // PW operator applications with many bands.
        {
            std::vector<double> packed(4 * overlap_count, 0.0);
            for (int local_band = 0; local_band < chunk_bands; ++local_band)
            {
                for (int a = 0; a < ntarget; ++a)
                {
                    const size_t index
                        = static_cast<size_t>(local_band) * ntarget + a;
                    packed[2 * index] = raw_chunk[local_band][a].real();
                    packed[2 * index + 1] = raw_chunk[local_band][a].imag();
                }
            }
            const size_t braw_offset = 2 * overlap_count;
            for (int local_band = 0; local_band < chunk_bands; ++local_band)
            {
                for (int a = 0; a < ntarget; ++a)
                {
                    const size_t index
                        = static_cast<size_t>(local_band) * ntarget + a;
                    packed[braw_offset + 2 * index]
                        = braw_chunk[local_band][a].real();
                    packed[braw_offset + 2 * index + 1]
                        = braw_chunk[local_band][a].imag();
                }
            }
            alloc::reduce_pool(packed);
            for (int local_band = 0; local_band < chunk_bands; ++local_band)
            {
                for (int a = 0; a < ntarget; ++a)
                {
                    const size_t index
                        = static_cast<size_t>(local_band) * ntarget + a;
                    raw_chunk[local_band][a] = std::complex<double>(
                        packed[2 * index], packed[2 * index + 1]);
                    braw_chunk[local_band][a] = std::complex<double>(
                        packed[braw_offset + 2 * index],
                        packed[braw_offset + 2 * index + 1]);
                }
            }
        }
        for (int local_band = 0; local_band < chunk_bands; ++local_band)
        {
            const int ib = band_start + local_band;
            std::complex<double>* hpsi_band
                = reinterpret_cast<std::complex<double>*>(tmhpsi)
                  + static_cast<size_t>(ib) * nbasis;
            const std::vector<std::complex<double>>& vpsi_band
                = folded_potential ? empty_vpsi : vpsi_chunk[local_band];
            alloc::apply_pw_alloc_delta_h_orthogonal_reduced(
                context.space, product.orthogonal, product.vphi,
                npw, raw_chunk[local_band], braw_chunk[local_band],
                vpsi_band, lambda, delta, !folded_potential);
            for (int ig = 0; ig < npw; ++ig)
            {
                hpsi_band[ig] += delta[ig];
            }
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
