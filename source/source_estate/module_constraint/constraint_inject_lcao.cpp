#include "constraint_inject_lcao.h"

#include <algorithm>

#include "source_base/parallel_reduce.h"
#include "source_lcao/module_gint/gint_interface.h"

namespace constraint
{

std::vector<hamilt::HContainer<double>> ConstraintInjectLCAO::build(
    const std::vector<std::vector<double>>& cw,
    ModuleGint::GintInfo* gint_info,
    const Parallel_Orbitals* paraV,
    const hamilt::HContainer<double>* dm_layout)
{
    // The vlocal kernel reads the shared GintInfo; make sure it points at the
    // esolver's active instance (normally already set by the LCAO esolver).
    ModuleGint::Gint::set_gint_info(gint_info);

    std::vector<hamilt::HContainer<double>> W;
    W.reserve(cw.size());
    for (size_t alpha = 0; alpha < cw.size(); ++alpha)
    {
        // Branch A: MPI — the Gint kernel transfers its serial grid result
        // into the target via transferSerials2Parallels, which requires the
        // target to carry the Parallel_Orbitals distribution (exactly like
        // the production Hamiltonian HR).
        if (paraV != nullptr)
        {
            // Branch A1: a production DM is supplied as the layout reference
            // (the runtime audit always passes one) — twin its structure
            // exactly.  The DM is assembled per-rank from the grid-independent
            // adjacency list, so every participating rank holds at least one
            // local block; a Gint grid-derived IJR list, by contrast, can be
            // empty on a rank whose real-space grid sub-domain overlaps no
            // atom, and transferSerials2Parallels dereferences an empty
            // target's atom-pair list (segfault).  Twinning the DM layout
            // also makes the per-rank W blocks bit-identical to the DM, which
            // is exactly what trace() requires afterwards.
            if (dm_layout != nullptr)
            {
                W.push_back(hamilt::HContainer<double>(*dm_layout));
            }
            else
            {
                // Branch A2: no reference layout available — fall back to
                // deriving the target from the shared GintInfo IJR structure
                // plus the esolver's distribution.  This can leave a rank
                // with an empty target when its grid sub-domain overlaps no
                // atom, so MPI callers should always pass the production DM
                // as dm_layout (Branch A1).
                W.push_back(hamilt::HContainer<double>(
                    paraV, nullptr, &gint_info->get_ijr_info()));
            }
        }
        else
        {
            // Branch B: serial — a plain nat-layout container suffices (the
            // kernel's single-rank add path needs no distribution).
            W.push_back(gint_info->get_hr<double>());
        }
        ModuleGint::cal_gint_vl(cw[alpha].data(), &W.back());
    }
    return W;
}

void ConstraintInjectLCAO::add_weighted(
    const std::vector<double>& mu,
    const std::vector<hamilt::HContainer<double>>& W,
    hamilt::HContainer<double>* H)
{
    // Branch A: no active constraints — nothing to add.
    if (W.empty() || mu.empty())
    {
        return;
    }
    for (size_t alpha = 0; alpha < W.size(); ++alpha)
    {
        // Branch B: zero multiplier — the contribution is a no-op by value.
        if (mu[alpha] == 0.0)
        {
            continue;
        }
        const double* w = W[alpha].get_wrapper();
        double* h = H->get_wrapper();
        for (size_t i = 0; i < H->get_nnr(); ++i)
        {
            h[i] += mu[alpha] * w[i];
        }
    }
}

bool ConstraintInjectLCAO::trace(const hamilt::HContainer<double>& A,
                                  const hamilt::HContainer<double>& B,
                                  double& trace_out)
{
    // Layout guard: a flat pairing is only meaningful on bit-identical
    // (pair, R, block) structures.  Compare both the total element count and
    // the full IJR structure so a mismatched caller fails loudly instead of
    // silently pairing blocks that do not correspond.
    if (A.get_nnr() != B.get_nnr())
    {
        return false;
    }
    const std::vector<int> ijr_a = A.get_ijr_info();
    const std::vector<int> ijr_b = B.get_ijr_info();
    if (ijr_a != ijr_b)
    {
        return false;
    }
    const double* a = A.get_wrapper();
    const double* b = B.get_wrapper();
    double s = 0.0;
    for (size_t i = 0; i < A.get_nnr(); ++i)
    {
        s += a[i] * b[i];
    }
    trace_out = s;
    return true;
}

double ConstraintInjectLCAO::audit_weighted_trace(
    const std::vector<std::vector<double>>& cw,
    ModuleGint::GintInfo* gint_info,
    const hamilt::HContainer<double>* dmr,
    const std::vector<double>& q_grid,
    const Parallel_Orbitals* paraV)
{
    if (dmr == nullptr)
    {
        return -1.0;
    }
    // Build W with the DM as the MPI layout reference so that (1) the target
    // is never empty on a rank and (2) the flat trace below pairs blocks that
    // are bit-identical by construction.
    const std::vector<hamilt::HContainer<double>> W
        = build(cw, gint_info, paraV, dmr);
    if (W.size() != q_grid.size())
    {
        return -1.0;
    }
    double max_dev = 0.0;
    for (size_t alpha = 0; alpha < W.size(); ++alpha)
    {
        double tr = 0.0;
        if (!trace(W[alpha], *dmr, tr))
        {
            return -1.0;
        }
        // The DM (and W) are distributed over the pool: each rank holds the
        // blocks of its local orbital rows, so the local flat trace is a
        // partial sum.  Reduce over the pool to obtain the full trace, the
        // same quantity the grid observable (q_grid) is reduced to.
#ifdef __MPI
        Parallel_Reduce::reduce_pool(tr);
#endif
        max_dev = std::max(max_dev, std::abs(tr - q_grid[alpha]));
    }
    return max_dev;
}

} // namespace constraint
