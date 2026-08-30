#include "constraint_inject_pw.h"

namespace constraint
{

bool ConstraintInjectPW::inject(const WeightGrid& wg,
                                const std::vector<double>& mu,
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

    const int nrxx = wg.nrxx();
    const int nspin = veff.nr; // matrix layout: (nspin, nrxx)
    const auto& cw = wg.constraint_weights();

    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        const double mu_a = mu[alpha];
        const std::vector<double>& w = cw[alpha];
        // Branch A: charge channel injection (phase 1).
        // The same +sum mu*w is applied to every spin channel: for nspin == 1
        // this is the single effective potential; for nspin == 2 both channels
        // receive it so the coupling is to the total charge only.
        for (int is = 0; is < nspin; ++is)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                veff(is, ir) += mu_a * w[ir];
            }
        }
    }
    return true;
}

} // namespace constraint
