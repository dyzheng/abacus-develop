#include "constraint_deriv.h"

#include <vector>

#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"

namespace constraint
{
namespace
{

// Build the canonical combined density buffers a per-constraint channel list
// needs.  Factory profiles are charge (+1,+1) and spin (+1,-1), so two
// contiguous arrays cover every alpha without an O(nalpha x nrxx) temporary.
void build_combined_densities(const double* const* rho,
                              const int nspin,
                              const int nrxx,
                              const std::vector<ChannelProfile>& channels,
                              std::vector<double>& dens_charge,
                              std::vector<double>& dens_spin)
{
    bool need_charge = false;
    bool need_spin = false;
    for (const ChannelProfile& ch : channels)
    {
        // Branch A: spin-like profile (split read signs).
        if (ch.read_dn != ch.read_up)
        {
            need_spin = true;
        }
        else
        {
            // Branch B: charge-like profile (symmetric read signs).
            need_charge = true;
        }
    }
    if (nspin == 2)
    {
        // Branch: two-channel density; build only the combined arrays the
        // constraint list actually reads.
        if (need_charge)
        {
            dens_charge.assign(nrxx, 0.0);
            for (int ir = 0; ir < nrxx; ++ir)
            {
                dens_charge[ir] = rho[0][ir] + rho[1][ir];
            }
        }
        if (need_spin)
        {
            dens_spin.assign(nrxx, 0.0);
            for (int ir = 0; ir < nrxx; ++ir)
            {
                dens_spin[ir] = rho[0][ir] - rho[1][ir];
            }
        }
    }
    else if (need_charge)
    {
        // Branch: nspin == 1 stores the total density in rho[0]; the caller
        // has already rejected spin-like profiles in this case.
        dens_charge.assign(nrxx, 0.0);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            dens_charge[ir] = rho[0][ir];
        }
    }
}

} // namespace

void constraint_force(const WeightGrid& wg,
                      const double* const* rho,
                      const int nspin,
                      const DensityChannel channel,
                      const std::vector<double>& mu,
                      ModuleBase::matrix& force)
{
    // Legacy single-channel entry (pre-A5 callers / historical tests): the
    // whole constraint list shares one channel.  Expanding it to a
    // homogeneous per-constraint profile list keeps the force integral in a
    // single implementation (no second kernel path to drift).
    const ConstraintKind kind = (channel == DensityChannel::Spin)
                                    ? ConstraintKind::Spin
                                    : ConstraintKind::Charge;
    const ChannelProfile profile = build_channel_profile(kind);
    std::vector<ChannelProfile> channels(wg.nconstraint(), profile);
    constraint_force(wg, rho, nspin, channels, mu, force);
}

void constraint_force(const WeightGrid& wg,
                      const double* const* rho,
                      const int nspin,
                      const std::vector<ChannelProfile>& channels,
                      const std::vector<double>& mu,
                      ModuleBase::matrix& force)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nat = wg.nat();
    const int nalpha = wg.nconstraint();

    // Guard: the force needs the position-derivative grid; a caller that
    // forgot build_derivatives() would silently read an empty cache.
    if (!wg.derivatives_built())
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "weight derivative grid not built "
                                 "(call build_derivatives)");
    }
    // Guard: one multiplier and one channel profile per constraint; a
    // mismatch would silently pair a weight with the wrong multiplier or
    // density channel (wiring bug).
    if (static_cast<int>(mu.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "mu size != nconstraint");
    }
    if (static_cast<int>(channels.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "channel count != nconstraint");
    }
    // Guard: a profile that reads the down density (read_dn != read_up,
    // i.e. the spin channel) needs the two-channel density buffer.  Under
    // nspin == 1 only rho[0] (the total density) exists, so a spin reading
    // would silently integrate a nonsense observable (same contract as the
    // observer / injector).
    for (const ChannelProfile& ch : channels)
    {
        if (nspin != 2 && ch.read_dn != ch.read_up)
        {
            ModuleBase::WARNING_QUIT("constraint_force",
                                     "spin channel requires nspin == 2");
        }
    }
    if (force.nr != nat || force.nc != 3)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "force buffer must be nat x 3");
    }

    std::vector<double> dens_charge;
    std::vector<double> dens_spin;
    build_combined_densities(rho, nspin, nrxx, channels, dens_charge,
                             dens_spin);

    // Local accumulation buffer (nat*3, the PW force layout iat*3+ipol).
    std::vector<double> f(static_cast<size_t>(nat) * 3, 0.0);
    const auto& dw = wg.weight_derivatives();
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const double mua = mu[alpha];
        if (mua == 0.0)
        {
            continue; // zero multiplier: this constraint contributes nothing
        }
        // Per-constraint combined density (A5): every alpha folds rho with
        // its own channel signs; a mixed charge+spin list needs no loop-side
        // masking any more.
        const bool spinlike = channels[alpha].read_dn != channels[alpha].read_up;
        const double* dens = spinlike ? dens_spin.data() : dens_charge.data();
        for (int J = 0; J < nat; ++J)
        {
            for (int d = 0; d < 3; ++d)
            {
                const double* dwd = &dw[alpha][(J * 3 + d) * nrxx];
                double acc = 0.0;
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    acc += dens[ir] * dwd[ir];
                }
                // F_J^d -= mu_alpha * int rho d w_alpha/d R_J^d dr
                f[J * 3 + d] -= mua * acc * dV;
            }
        }
    }

    // Sum the partial grid integrals over the density-grid pool ranks
    // (allreduce; no-op in serial builds), then accumulate into the
    // caller's buffer.
#ifdef __MPI
    Parallel_Reduce::reduce_pool(f.data(), static_cast<int>(f.size()));
#endif
    // Branch: a dipole constraint carries an explicit-coordinate ionic
    // offset.  Its density weight is grid-fixed (dw == 0), so this offset
    // derivative is the complete force.  Add it AFTER the allreduce: every
    // rank knows the same atomic positions, so adding before would multiply
    // the term by the pool size.
    if (wg.is_dipole())
    {
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            for (int J = 0; J < nat; ++J)
            {
                for (int d = 0; d < 3; ++d)
                {
                    f[J * 3 + d] -= mu[alpha]
                        * wg.dipole_atom_offset_derivative(alpha, J, d);
                }
            }
        }
    }
    for (int i = 0; i < nat * 3; ++i)
    {
        force.c[i] += f[i];
    }
}

void constraint_stress(const WeightGrid& wg,
                       const double* const* rho,
                       const int nspin,
                       const std::vector<ChannelProfile>& channels,
                       const std::vector<double>& mu,
                       ModuleBase::matrix& stress)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nalpha = wg.nconstraint();

    // Guard: the strain derivative is built from the position-derivative
    // cache, so an unbuilt cache would silently report a zero stress.
    if (!wg.derivatives_built())
    {
        ModuleBase::WARNING_QUIT("constraint_stress",
                                 "weight derivative grid not built "
                                 "(call build_derivatives)");
    }
    // Guard: a dipole density weight is a coordinate function with a
    // different strain derivative; the Becke contraction below does not
    // represent it, so refuse instead of returning a wrong tensor.
    if (wg.is_dipole())
    {
        ModuleBase::WARNING_QUIT("constraint_stress",
                                 "dipole constraint stress is not implemented");
    }
    // Guard: the S5.2 Hirshfeld backend supplies the position derivative but
    // not yet the strain derivative; refuse instead of reporting zero.
    if (wg.is_hirshfeld())
    {
        ModuleBase::WARNING_QUIT("constraint_stress",
                                 "Hirshfeld constraint stress is not implemented");
    }
    // Guard: one multiplier and one channel profile per constraint.
    if (static_cast<int>(mu.size()) != nalpha
        || static_cast<int>(channels.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_stress",
                                 "mu/channel count != nconstraint");
    }
    // Guard: a spin-like profile needs both spin channels, exactly as in the
    // force kernel.
    for (const ChannelProfile& ch : channels)
    {
        if (nspin != 2 && ch.read_dn != ch.read_up)
        {
            ModuleBase::WARNING_QUIT("constraint_stress",
                                     "spin channel requires nspin == 2");
        }
    }
    // Guard: the caller supplies the full stress accumulator.
    if (stress.nr != 3 || stress.nc != 3)
    {
        ModuleBase::WARNING_QUIT("constraint_stress",
                                 "stress buffer must be 3 x 3");
    }

    std::vector<double> dens_charge;
    std::vector<double> dens_spin;
    build_combined_densities(rho, nspin, nrxx, channels, dens_charge,
                             dens_spin);

    // Local 3 x 3 accumulation.  The full (a,b) square is filled before the
    // final symmetrization so a tiny round-off asymmetry cannot survive.
    std::vector<double> s(9, 0.0);
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const double mua = mu[alpha];
        // Branch: a zero multiplier contributes no stress from this alpha.
        if (mua == 0.0)
        {
            continue;
        }
        const bool spinlike = channels[alpha].read_dn != channels[alpha].read_up;
        const double* dens = spinlike ? dens_spin.data() : dens_charge.data();
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                double acc = 0.0;
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    acc += dens[ir]
                           * wg.weight_strain_derivative(alpha, a, b, ir);
                }
                // sigma_ab -= mu_alpha * int rho d w_alpha/d eps_ab dr / Omega
                s[a * 3 + b] -= mua * acc * dV;
            }
        }
        // Volume/Jacobian term.  The weight matrix is an integral over the
        // cell, so d(int rho w dV)/d eps_ab carries rho*w*delta_ab in
        // addition to the explicit weight derivative.  The target projector
        // is dimensionless and its volume dependence is already carried by
        // the S adjoint, so this term uses only the weight-field density.
        double q_weight = 0.0;
        const std::vector<double>& w_alpha = wg.constraint_weight(alpha);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            q_weight += dens[ir] * w_alpha[ir] * dV;
        }
        for (int a = 0; a < 3; ++a)
        {
            s[a * 3 + a] -= mua * q_weight;
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(s.data(), static_cast<int>(s.size()));
#endif
    const double inv_omega = 1.0 / rho_basis->omega;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = a; b < 3; ++b)
        {
            // Symmetric strain: the physical stress is the symmetric part of
            // the chain-rule tensor.  Store the same value on both triangles.
            const double value
                = 0.5 * (s[a * 3 + b] + s[b * 3 + a]) * inv_omega;
            stress(a, b) += value;
            // Branch: off-diagonal entries must mirror to keep the tensor
            // symmetric for the total-stress consumer.
            if (a != b)
            {
                stress(b, a) += value;
            }
        }
    }
}

void constraint_force_densities(
    const WeightGrid& wg,
    const std::vector<std::vector<double>>& rho,
    ModuleBase::matrix& force)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nat = wg.nat();
    const int nalpha = wg.nconstraint();

    // Guard: the force needs the position-derivative grid.
    if (!wg.derivatives_built())
    {
        ModuleBase::WARNING_QUIT("constraint_force_densities",
                                 "weight derivative grid not built "
                                 "(call build_derivatives)");
    }
    // Guard: one already folded density per constraint; a mismatch would
    // silently pair a density with another fragment's weight derivative.
    if (static_cast<int>(rho.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_force_densities",
                                 "density count != nconstraint");
    }
    for (const std::vector<double>& density : rho)
    {
        // Guard: every density must live on the same dense grid as wg.
        if (static_cast<int>(density.size()) != nrxx)
        {
            ModuleBase::WARNING_QUIT("constraint_force_densities",
                                     "density grid size mismatch");
        }
    }
    // Guard: a dipole weight has an additional ionic coordinate derivative;
    // the allocation closure currently excludes it, so refuse instead of
    // returning an incomplete force.
    if (wg.is_dipole())
    {
        ModuleBase::WARNING_QUIT("constraint_force_densities",
                                 "dipole constraints are not supported by the "
                                 "non-collinear allocation force");
    }
    if (force.nr != nat || force.nc != 3)
    {
        ModuleBase::WARNING_QUIT("constraint_force_densities",
                                 "force buffer must be nat x 3");
    }

    std::vector<double> f(static_cast<size_t>(nat) * 3, 0.0);
    const auto& dw = wg.weight_derivatives();
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const double* density = rho[alpha].data();
        for (int J = 0; J < nat; ++J)
        {
            for (int d = 0; d < 3; ++d)
            {
                const double* dwd = &dw[alpha][(J * 3 + d) * nrxx];
                double acc = 0.0;
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    acc += density[ir] * dwd[ir];
                }
                f[J * 3 + d] -= acc * dV;
            }
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(f.data(), static_cast<int>(f.size()));
#endif
    for (int i = 0; i < nat * 3; ++i)
    {
        force.c[i] += f[i];
    }
}

void constraint_stress_densities(
    const WeightGrid& wg,
    const std::vector<std::vector<double>>& rho,
    ModuleBase::matrix& stress)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nalpha = wg.nconstraint();

    // Guard: the strain derivative is built from the same cache as the
    // position kernel.
    if (!wg.derivatives_built())
    {
        ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                 "weight derivative grid not built "
                                 "(call build_derivatives)");
    }
    // Guard: one folded density per constraint.
    if (static_cast<int>(rho.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                 "density count != nconstraint");
    }
    for (const std::vector<double>& density : rho)
    {
        // Guard: every density must live on the weight grid.
        if (static_cast<int>(density.size()) != nrxx)
        {
            ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                     "density grid size mismatch");
        }
    }
    // Guard: the dipole strain derivative is a different coordinate term.
    if (wg.is_dipole())
    {
        ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                 "dipole constraints are not supported by the "
                                 "non-collinear allocation stress");
    }
    // Guard: the Hirshfeld strain derivative is not part of S5.2.
    if (wg.is_hirshfeld())
    {
        ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                 "Hirshfeld constraint stress is not implemented");
    }
    // Guard: the caller supplies the full stress accumulator.
    if (stress.nr != 3 || stress.nc != 3)
    {
        ModuleBase::WARNING_QUIT("constraint_stress_densities",
                                 "stress buffer must be 3 x 3");
    }

    std::vector<double> s(9, 0.0);
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const double* density = rho[alpha].data();
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                double acc = 0.0;
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    acc += density[ir]
                           * wg.weight_strain_derivative(alpha, a, b, ir);
                }
                s[a * 3 + b] -= acc * dV;
            }
        }
        // Cell-integral Jacobian: d(int rho w dV)/d eps_ab carries the
        // isotropic rho*w term in addition to the explicit weight strain
        // derivative.
        double q_weight = 0.0;
        const std::vector<double>& w_alpha = wg.constraint_weight(alpha);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            q_weight += density[ir] * w_alpha[ir] * dV;
        }
        for (int a = 0; a < 3; ++a)
        {
            s[a * 3 + a] -= q_weight;
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(s.data(), static_cast<int>(s.size()));
#endif
    const double inv_omega = 1.0 / rho_basis->omega;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = a; b < 3; ++b)
        {
            // Symmetric strain: store the symmetrized tensor on both
            // triangles so the total-stress consumer sees one convention.
            const double value
                = 0.5 * (s[a * 3 + b] + s[b * 3 + a]) * inv_omega;
            stress(a, b) += value;
            // Branch: mirror the off-diagonal entry.
            if (a != b)
            {
                stress(b, a) += value;
            }
        }
    }
}

} // namespace constraint
