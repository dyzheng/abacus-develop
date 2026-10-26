#include "source_estate/module_constraint/alloc_constraint.h"

#include <stdexcept>
#include <string>

namespace alloc
{
namespace
{

void check_square(const std::vector<double>& A, const int n, const char* name)
{
    // Branch: every dense input matrix is row-major n x n; reject a dimension
    // error before any index arithmetic.
    if (n < 0 || static_cast<int>(A.size()) != n * n)
    {
        throw std::invalid_argument(std::string(name) + " has wrong size");
    }
}

std::vector<double> selection_matrix(const int norb,
                                     const std::vector<int>& orbital)
{
    const int ntarget = static_cast<int>(orbital.size());
    std::vector<double> C(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: a target AO index must address the full basis.
        if (orbital[a] < 0 || orbital[a] >= norb)
        {
            throw std::invalid_argument("target orbital index out of range");
        }
        C[static_cast<size_t>(orbital[a]) * ntarget + a] = 1.0;
    }
    return C;
}

} // namespace

void build_alloc_constraint_algebra(
    const std::vector<double>& S,
    const int norb,
    const std::vector<int>& target_orbital,
    const std::vector<int>& target_atom,
    const std::vector<int>& target_constraint,
    const std::vector<double>& ST,
    const int nconstraint,
    AllocConstraintAlgebra& out,
    const double rank_tol)
{
    check_square(S, norb, "S");
    const int ntarget = static_cast<int>(target_orbital.size());
    // Branch: the three target lists must describe the same R columns.
    if (static_cast<int>(target_atom.size()) != ntarget
        || static_cast<int>(target_constraint.size()) != ntarget)
    {
        throw std::invalid_argument("target list sizes differ");
    }
    check_square(ST, ntarget, "S_T");
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: target_constraint is either an owned component or -1 for
        // a target that belongs only to the global projector.
        if (target_constraint[a] < -1 || target_constraint[a] >= nconstraint)
        {
            throw std::invalid_argument("target constraint index out of range");
        }
    }

    out.norb = norb;
    out.ntarget = ntarget;
    out.nconstraint = nconstraint;
    out.target_constraint = target_constraint;
    finalize_lcao_target(norb, target_orbital, target_atom, ST, out.target,
                         rank_tol);

    // SB = S B, with B = C'.  This is the only norb x R contraction needed by
    // both the operator and the projection read-out.
    out.SB.assign(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int row = 0; row < norb; ++row)
    {
        for (int col = 0; col < norb; ++col)
        {
            const double s = S[static_cast<size_t>(row) * norb + col];
            for (int a = 0; a < ntarget; ++a)
            {
                out.SB[static_cast<size_t>(row) * ntarget + a]
                    += s * out.target.Cprime[static_cast<size_t>(col)
                                                 * ntarget + a];
            }
        }
    }
}

void build_alloc_constraint_kernel(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& W,
    const int constraint,
    std::vector<double>& K)
{
    const int norb = space.norb;
    const int ntarget = space.ntarget;
    check_square(W, norb, "W");
    // Branch: a kernel index must name an actual constrained component.
    if (constraint < 0 || constraint >= space.nconstraint)
    {
        throw std::invalid_argument("constraint index out of range");
    }

    // B^T W and W B.  The final kernel is an outer product of these small
    // norb x R / R x norb contractions with S B, so no S^{-1} is formed.
    std::vector<double> BtW(static_cast<size_t>(ntarget) * norb, 0.0);
    std::vector<double> WB(static_cast<size_t>(norb) * ntarget, 0.0);
    std::vector<double> BtWB(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const double w = W[static_cast<size_t>(mu) * norb + nu];
            for (int a = 0; a < ntarget; ++a)
            {
                const double b_mu = space.target.Cprime[static_cast<size_t>(mu)
                                                            * ntarget + a];
                BtW[static_cast<size_t>(a) * norb + nu] += b_mu * w;
                WB[static_cast<size_t>(mu) * ntarget + a]
                    += w * space.target.Cprime[static_cast<size_t>(nu)
                                                   * ntarget + a];
            }
        }
    }
    for (int a = 0; a < ntarget; ++a)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            const double b_nu = space.target.Cprime[static_cast<size_t>(nu)
                                                        * ntarget + a];
            for (int b = 0; b < ntarget; ++b)
            {
                BtWB[static_cast<size_t>(b) * ntarget + a]
                    += BtW[static_cast<size_t>(b) * norb + nu] * b_nu;
            }
        }
    }

    K.assign(static_cast<size_t>(norb) * norb, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            double value = W[static_cast<size_t>(mu) * norb + nu];
            for (int a = 0; a < ntarget; ++a)
            {
                const double sb_mu = space.SB[static_cast<size_t>(mu)
                                                  * ntarget + a];
                const double sb_nu = space.SB[static_cast<size_t>(nu)
                                                  * ntarget + a];
                value -= sb_mu * BtW[static_cast<size_t>(a) * norb + nu];
                value -= WB[static_cast<size_t>(mu) * ntarget + a] * sb_nu;
                for (int b = 0; b < ntarget; ++b)
                {
                    value += sb_mu
                             * BtWB[static_cast<size_t>(a) * ntarget + b]
                             * space.SB[static_cast<size_t>(nu) * ntarget + b];
                }
                // Branch: the target projector Q_i contains only targets
                // owned by this constraint.
                if (space.target_constraint[a] == constraint)
                {
                    value += sb_mu * sb_nu;
                }
            }
            K[static_cast<size_t>(mu) * norb + nu] = value;
        }
    }
}

void build_alloc_delta_h(const AllocConstraintAlgebra& space,
                         const std::vector<std::vector<double>>& W,
                         const std::vector<double>& lambda,
                         std::vector<double>& delta_h)
{
    // Branch: a multiplier/weight count mismatch would silently drop a
    // constrained component, so reject it before assembly.
    if (static_cast<int>(W.size()) != space.nconstraint
        || static_cast<int>(lambda.size()) != space.nconstraint)
    {
        throw std::invalid_argument("W/lambda count does not match constraints");
    }

    delta_h.assign(static_cast<size_t>(space.norb) * space.norb, 0.0);
    for (int alpha = 0; alpha < space.nconstraint; ++alpha)
    {
        // Branch: a zero multiplier contributes nothing.
        if (lambda[alpha] == 0.0)
        {
            continue;
        }
        std::vector<double> K;
        build_alloc_constraint_kernel(space, W[alpha], alpha, K);
        for (size_t i = 0; i < delta_h.size(); ++i)
        {
            delta_h[i] += lambda[alpha] * K[i];
        }
    }
}

void observe_alloc_constraint_moments(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<double>& D_total,
    const std::vector<double>& D_diff,
    std::vector<double>& n_T_target,
    std::vector<double>& m_T_target,
    std::vector<double>& n_T_constraint,
    std::vector<double>& m_T_constraint)
{
    observe_lcao_target(space.target, S, D_total, D_diff, n_T_target,
                        m_T_target);
    n_T_constraint.assign(space.nconstraint, 0.0);
    m_T_constraint.assign(space.nconstraint, 0.0);
    for (int a = 0; a < space.ntarget; ++a)
    {
        const int alpha = space.target_constraint[a];
        // Branch: unowned targets belong to P but not to any constraint sum.
        if (alpha >= 0)
        {
            n_T_constraint[alpha] += n_T_target[a];
            m_T_constraint[alpha] += m_T_target[a];
        }
    }
}

} // namespace alloc
