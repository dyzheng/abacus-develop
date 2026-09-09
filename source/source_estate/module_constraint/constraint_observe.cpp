#include "constraint_observe.h"

#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"

namespace constraint
{

void ConstraintObserver::observe(const WeightGrid& wg,
                                 const double* const* rho,
                                 const int nspin,
                                 const DensityChannel channel,
                                 std::vector<double>& Q)
{
    // Legacy single-channel entry (pre-A2 callers / historical tests): the
    // whole constraint list shares one channel.  Expanding it to a
    // homogeneous per-constraint profile list keeps the reading integral in a
    // single implementation (no second quadrature path to drift).
    if (channel == DensityChannel::Spin && nspin != 2)
    {
        // Guard: the spin channel reads rho_up - rho_dn and requires nspin
        // == 2.  A wrong channel/nspin combination must abort loudly (never
        // silently read a nonsense observable — phase-1 guard discipline).
        ModuleBase::WARNING_QUIT("ConstraintObserver::observe",
                                 "spin channel requires nspin == 2");
    }
    const ConstraintKind kind = (channel == DensityChannel::Spin)
                                    ? ConstraintKind::Spin
                                    : ConstraintKind::Charge;
    const ChannelProfile profile = build_channel_profile(kind);
    std::vector<ChannelProfile> channels(wg.nconstraint(), profile);
    observe(wg, rho, nspin, channels, Q);
}

void ConstraintObserver::observe(const WeightGrid& wg,
                                 const double* const* rho,
                                 const int nspin,
                                 const std::vector<ChannelProfile>& channels,
                                 std::vector<double>& Q)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nalpha = wg.nconstraint();
    const auto& cw = wg.constraint_weights();

    // Guard: the per-constraint channel list must be parallel to the weight
    // grid's constraint list (same order the WeightGrid was built from).  A
    // mismatch would silently pair each weight with the wrong channel —
    // abort loudly instead of reading a wrong observable.
    if (static_cast<int>(channels.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT(
            "ConstraintObserver::observe",
            "per-constraint channel count does not match the weight grid "
            "constraint count");
    }

    // Guard: a profile that couples to the down density (read_dn != read_up,
    // i.e. the spin channel) needs the two-channel density buffer.  Under
    // nspin == 1 only rho[0] (the total density) exists, so a spin reading
    // would silently return a wrong observable.
    for (const ChannelProfile& ch : channels)
    {
        if (nspin != 2 && ch.read_dn != ch.read_up)
        {
            ModuleBase::WARNING_QUIT("ConstraintObserver::observe",
                                     "spin channel requires nspin == 2");
        }
    }

    Q.assign(nalpha, 0.0);
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const ChannelProfile& ch = channels[alpha];
        const std::vector<double>& w = cw[alpha];
        double q = 0.0;
        if (nspin == 2)
        {
            // Branch: two-channel density; every alpha combines the two spin
            // channels with its own read signs.
            for (int ir = 0; ir < nrxx; ++ir)
            {
                q += w[ir] * (rho[0][ir] * static_cast<double>(ch.read_up)
                              + rho[1][ir] * static_cast<double>(ch.read_dn));
            }
        }
        else
        {
            // Branch: nspin == 1 stores the total density in rho[0]; the
            // read_up coefficient selects it (spin profiles are rejected
            // above, so read_dn is never silently dropped).
            for (int ir = 0; ir < nrxx; ++ir)
            {
                q += w[ir] * rho[0][ir] * static_cast<double>(ch.read_up);
            }
        }
        Q[alpha] = q * dV;
    }

    // Sum reduction over the PW pool (no-op in serial builds).
#ifdef __MPI
    Parallel_Reduce::reduce_pool(Q.data(), static_cast<int>(Q.size()));
#endif
}

} // namespace constraint
