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
    // Guard: the spin channel reads rho_up - rho_dn and requires nspin == 2
    // (same contract as ConstraintObserver; a wrong combination must abort
    // loudly, never silently read a nonsense observable).
    if (channel == DensityChannel::Spin && nspin != 2)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "spin channel requires nspin == 2");
    }
    // Guard: one multiplier per constraint and a nat x 3 accumulation
    // buffer; a mismatch means the caller is not wired to this kernel.
    if (static_cast<int>(mu.size()) != nalpha)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "mu size != nconstraint");
    }
    if (force.nr != nat || force.nc != 3)
    {
        ModuleBase::WARNING_QUIT("constraint_force",
                                 "force buffer must be nat x 3");
    }

    // Fold the channel choice into one per-point density array so the inner
    // loops below read a single contiguous buffer (the derivative grid is
    // laid out [alpha][(J*3+d)*nrxx+ir], so the innermost stride is ir).
    std::vector<double> dens(nrxx);
    if (channel == DensityChannel::Spin)
    {
        // Spin channel: magnetization density rho_up - rho_dn.
        for (int ir = 0; ir < nrxx; ++ir)
        {
            dens[ir] = rho[0][ir] - rho[1][ir];
        }
    }
    else if (nspin == 2)
    {
        // Charge channel, nspin == 2: total charge rho_up + rho_dn.
        for (int ir = 0; ir < nrxx; ++ir)
        {
            dens[ir] = rho[0][ir] + rho[1][ir];
        }
    }
    else
    {
        // Charge channel, nspin == 1: the single rho[0] density array.
        for (int ir = 0; ir < nrxx; ++ir)
        {
            dens[ir] = rho[0][ir];
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
