#include "constraint_deriv.h"

#include <vector>

#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"

namespace constraint
{

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

    // Scan which combined densities the list needs.  Factory profiles are
    // charge (+1,+1) and spin (+1,-1), so a profile with read_dn == read_up
    // folds rho_up + rho_dn (charge) and one with read_dn != read_up folds
    // rho_up - rho_dn (spin); building at most the two canonical buffers
    // keeps the inner loop over one contiguous array per alpha without an
    // O(nalpha x nrxx) intermediate.
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
    std::vector<double> dens_charge;
    std::vector<double> dens_spin;
    if (nspin == 2)
    {
        // Branch: two-channel density; build the combined arrays the list
        // needs (charge = total, spin = magnetization).
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
        // Branch: nspin == 1 stores the total density in rho[0]; only
        // charge-like profiles survive the guard above.
        dens_charge.assign(nrxx, 0.0);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            dens_charge[ir] = rho[0][ir];
        }
    }

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
    for (int i = 0; i < nat * 3; ++i)
    {
        force.c[i] += f[i];
    }
}

} // namespace constraint
