#include "source_estate/module_constraint/alloc_observe.h"

#include "source_base/parallel_reduce.h"

#include <stdexcept>

namespace alloc
{

void AllocObserve::accumulate_spin(
    const std::vector<std::vector<double>>& bands,
    const std::vector<double>& occ,
    const double spin_sign,
    std::vector<double>& n_T,
    std::vector<double>& m_T,
    std::vector<double>& n_c_grid) const
{
    // Branch: an empty spin channel contributes nothing; this permits a
    // spin-unpolarized caller to pass an empty down-spin set.
    if (bands.empty())
    {
        return;
    }
    // Branch: occupations and bands must describe the same channel.
    if (bands.size() != occ.size())
    {
        throw std::invalid_argument("band and occupation counts differ");
    }

    const int nrxx = op_.nrxx();
    std::vector<double> residual(nrxx, 0.0);
    for (size_t band = 0; band < bands.size(); ++band)
    {
        // Branch: a band carrying no occupation need not be projected.
        if (occ[band] == 0.0)
        {
            continue;
        }
        // Branch: every band function must live on the operator grid.
        if (static_cast<int>(bands[band].size()) != nrxx)
        {
            throw std::invalid_argument("band function has wrong grid size");
        }

        std::vector<double> o;
        op_.overlaps(bands[band], o);

        // Residual psi - sum_a phi_a <phi_a|psi> defines the complementary
        // density.  Its norm is evaluated directly to preserve positivity.
        std::vector<double> projected;
        op_.apply_P(bands[band], projected);
        for (int ir = 0; ir < nrxx; ++ir)
        {
            residual[ir] = bands[band][ir] - projected[ir];
            n_c_grid[ir] += occ[band] * residual[ir] * residual[ir];
        }

        for (int a = 0; a < op_.ntarget(); ++a)
        {
            const double weight = occ[band] * o[a] * o[a];
            const int atom = op_.atom_of(a);
            n_T[atom] += weight;
            m_T[atom] += spin_sign * weight;
        }
    }
}

AllocResult AllocObserve::observe(
    const std::vector<std::vector<double>>& up,
    const std::vector<double>& occ_up,
    const std::vector<std::vector<double>>& dn,
    const std::vector<double>& occ_dn) const
{
    const int nat = op_.nat();
    AllocResult result;
    result.n_T.assign(nat, 0.0);
    result.m_T.assign(nat, 0.0);
    result.n_c.assign(nat, 0.0);
    result.m_c.assign(nat, 0.0);
    result.n_tot.assign(nat, 0.0);
    result.m_tot.assign(nat, 0.0);
    result.n_c_grid.assign(op_.nrxx(), 0.0);
    result.cond = op_.cond();

    // Spin-up has sign +1 in the collinear spin density; spin-down has -1.
    accumulate_spin(up, occ_up, +1.0, result.n_T, result.m_T,
                    result.n_c_grid);
    std::vector<double> n_c_down_grid(op_.nrxx(), 0.0);
    accumulate_spin(dn, occ_dn, -1.0, result.n_T, result.m_T,
                    n_c_down_grid);

    // n_c_grid stores total charge.  The difference of the two spin
    // complement densities is integrated directly into m_c.
    for (int ir = 0; ir < op_.nrxx(); ++ir)
    {
        result.n_c_grid[ir] += n_c_down_grid[ir];
    }

    for (int atom = 0; atom < nat; ++atom)
    {
        double nc = 0.0;
        double mc = 0.0;
        const std::vector<double>& weight = op_.weight(atom);
        for (int ir = 0; ir < op_.nrxx(); ++ir)
        {
            nc += weight[ir] * result.n_c_grid[ir];
            mc += weight[ir]
                  * (result.n_c_grid[ir] - 2.0 * n_c_down_grid[ir]);
        }
        result.n_c[atom] = nc * op_.dV();
        result.m_c[atom] = mc * op_.dV();
    }
    // Integrals are rank-local sums; make the per-atom read-out identical on
    // every pool rank.  The grid output itself remains rank-local.  n_T/m_T
    // are deliberately not reduced: overlaps() already produced global
    // projections, so every rank holds the same target-subspace occupancy.
#ifdef __MPI
    Parallel_Reduce::reduce_pool(result.n_c.data(),
                                 static_cast<int>(result.n_c.size()));
    Parallel_Reduce::reduce_pool(result.m_c.data(),
                                 static_cast<int>(result.m_c.size()));
#endif

    for (int atom = 0; atom < nat; ++atom)
    {
        result.n_tot[atom] = result.n_T[atom] + result.n_c[atom];
        result.m_tot[atom] = result.m_T[atom] + result.m_c[atom];
    }
    return result;
}

} // namespace alloc
