#include "constraint_inject_pw.h"

namespace constraint
{

bool ConstraintInjectPW::inject(const WeightGrid& wg,
                                const std::vector<double>& mu,
                                const DensityChannel channel,
                                ModuleBase::matrix& veff)
{
    // Legacy single-channel entry (pre-A3 callers / historical tests): the
    // whole constraint list shares one channel.  Expanding it to a
    // homogeneous per-constraint profile list keeps the injection in a single
    // implementation (no second operator path to drift).
    if (static_cast<int>(mu.size()) != wg.nconstraint())
    {
        // Guard: a mu vector of the wrong length would silently corrupt the
        // potential.  The orchestrator sizes mu from wg.nconstraint(), so this
        // can only happen on a wiring bug.
        return false;
    }
    const ConstraintKind kind = (channel == DensityChannel::Spin)
                                    ? ConstraintKind::Spin
                                    : ConstraintKind::Charge;
    const ChannelProfile profile = build_channel_profile(kind);
    std::vector<ChannelProfile> channels(wg.nconstraint(), profile);
    return inject(wg, mu, channels, veff);
}

bool ConstraintInjectPW::inject(const WeightGrid& wg,
                                const std::vector<double>& mu,
                                const std::vector<ChannelProfile>& channels,
                                ModuleBase::matrix& veff)
{
    const int nalpha = wg.nconstraint();
    if (static_cast<int>(mu.size()) != nalpha)
    {
        // Guard: a mu vector of the wrong length would silently corrupt the
        // potential.  The orchestrator sizes mu from wg.nconstraint(), so this
        // can only happen on a wiring bug.
        return false;
    }
    if (static_cast<int>(channels.size()) != nalpha)
    {
        // Guard: the per-constraint channel list must be parallel to the
        // weight grid's constraint list and to mu.  A mismatch would silently
        // inject each multiplier through the wrong channel.
        return false;
    }

    const int nrxx = wg.nrxx();
    const int nspin = veff.nr; // matrix layout: (nspin, nrxx)
    const auto& cw = wg.constraint_weights();

    // Guard: a profile that splits the injection (inj_dn != inj_up, i.e. the
    // spin channel) needs two spin buffers; a single-channel buffer cannot
    // host it and must be rejected (never silently run the wrong operator).
    for (const ChannelProfile& ch : channels)
    {
        if (nspin != 2 && ch.inj_dn != ch.inj_up)
        {
            return false;
        }
    }

    if (nspin == 2)
    {
        // Branch A: two-channel buffer.  Every alpha injects its own
        // up/down signs: charge (+1, +1) couples to the total density, spin
        // (+1, -1) couples to the magnetization (DeltaSpin +/- lambda
        // semantics).
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            const double mu_a = mu[alpha];
            const double su = channels[alpha].inj_up;
            const double sd = channels[alpha].inj_dn;
            const std::vector<double>& w = cw[alpha];
            for (int ir = 0; ir < nrxx; ++ir)
            {
                veff(0, ir) += mu_a * su * w[ir];
                veff(1, ir) += mu_a * sd * w[ir];
            }
        }
    }
    else
    {
        // Branch B: single-channel buffer (nspin == 1).  Only the charge
        // profile passes the guard above, so su == +1 and only the total
        // density sees the constraint potential.
        for (int alpha = 0; alpha < nalpha; ++alpha)
        {
            const double mu_a = mu[alpha];
            const double su = channels[alpha].inj_up;
            const std::vector<double>& w = cw[alpha];
            for (int ir = 0; ir < nrxx; ++ir)
            {
                veff(0, ir) += mu_a * su * w[ir];
            }
        }
    }
    return true;
}

} // namespace constraint
