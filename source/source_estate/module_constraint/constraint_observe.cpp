#include "constraint_observe.h"

#include "source_base/parallel_reduce.h"

namespace constraint
{

void ConstraintObserver::observe(const WeightGrid& wg,
                                 const double* const* rho,
                                 const int nspin,
                                 std::vector<double>& Q)
{
    const ModulePW::PW_Basis* rho_basis = wg.rho_basis();
    const double dV = rho_basis->omega / static_cast<double>(rho_basis->nxyz);
    const int nrxx = rho_basis->nrxx;
    const int nalpha = wg.nconstraint();
    const auto& cw = wg.constraint_weights();

    Q.assign(nalpha, 0.0);
    for (int alpha = 0; alpha < nalpha; ++alpha)
    {
        double q = 0.0;
        for (int ir = 0; ir < nrxx; ++ir)
        {
            // Charge channel: rho[0] for nspin=1, rho[0]+rho[1] for nspin=2.
            // The spin channel rho[0]-rho[1] is a phase-2 extension and does
            // not enter the charge reading.
            double r = rho[0][ir];
            if (nspin == 2)
            {
                r += rho[1][ir];
            }
            q += cw[alpha][ir] * r;
        }
        Q[alpha] = q * dV;
    }

    // Sum reduction over the PW pool (no-op in serial builds).
#ifdef __MPI
    Parallel_Reduce::reduce_pool(Q.data(), static_cast<int>(Q.size()));
#endif
}

} // namespace constraint
