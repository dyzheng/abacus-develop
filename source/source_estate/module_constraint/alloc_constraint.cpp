#include "source_estate/module_constraint/alloc_constraint.h"

#include "source_estate/module_constraint/alloc_linalg.h"

#include <algorithm>
#include <cmath>
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
    out.rank_tol = rank_tol;
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

void build_alloc_constraint_s_adjoint(
    const AllocConstraintAlgebra& space,
    const std::vector<double>& S,
    const std::vector<double>& W,
    const int constraint,
    const std::vector<double>& D,
    std::vector<double>& E)
{
    const int norb = space.norb;
    const int ntarget = space.ntarget;
    check_square(S, norb, "S");
    check_square(W, norb, "W");
    check_square(D, norb, "D");
    // Branch: the adjoint belongs to one constrained component and uses the
    // target-space data built with it.
    if (constraint < 0 || constraint >= space.nconstraint)
    {
        throw std::invalid_argument("constraint index out of range");
    }
    // Guard: this reverse-mode derivation assumes the selection form of C,
    // where target.orbital names each orthonormalized column.
    if (static_cast<int>(space.target.orbital.size()) != ntarget
        || static_cast<int>(space.target.Cprime.size()) != norb * ntarget
        || static_cast<int>(space.SB.size()) != norb * ntarget)
    {
        throw std::invalid_argument("target space is not in selection form");
    }

    const auto idx = [](const int row, const int col, const int ncols) {
        return static_cast<size_t>(row) * ncols + col;
    };
    std::vector<double> C(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        const int orbital = space.target.orbital[a];
        // Branch: every selected target must address a full-basis AO.
        if (orbital < 0 || orbital >= norb)
        {
            throw std::invalid_argument("target orbital index out of range");
        }
        C[idx(orbital, a, ntarget)] = 1.0;
    }

    const std::vector<double>& B = space.target.Cprime;
    const std::vector<double>& U = space.SB;

    // H_i = B^T W B + diag(owner == i).  This is the small matrix that
    // multiplies U from both sides in the target-projector part of K_i.
    std::vector<double> BtW(static_cast<size_t>(ntarget) * norb, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            for (int mu = 0; mu < norb; ++mu)
            {
                BtW[idx(a, nu, norb)]
                    += B[idx(mu, a, ntarget)] * W[idx(mu, nu, norb)];
            }
        }
    }
    std::vector<double> H(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int nu = 0; nu < norb; ++nu)
            {
                value += BtW[idx(a, nu, norb)] * B[idx(nu, b, ntarget)];
            }
            // Branch: only targets owned by this constraint enter Q_i.
            if (a == b && space.target_constraint[a] == constraint)
            {
                value += 1.0;
            }
            H[idx(a, b, ntarget)] = value;
        }
    }

    // Reverse pass through K_i = W - U B^T W - W B U^T + U H_i U^T.
    // G_U/G_B are the adjoints of U and B, while G_M is the adjoint of the
    // small matrix M_i = B^T W B.  The two cross terms contribute equally,
    // hence the factors of two below.
    std::vector<double> WB(static_cast<size_t>(norb) * ntarget, 0.0);
    std::vector<double> DU(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            double wb = 0.0;
            double du = 0.0;
            for (int nu = 0; nu < norb; ++nu)
            {
                wb += W[idx(mu, nu, norb)] * B[idx(nu, a, ntarget)];
                du += D[idx(mu, nu, norb)] * U[idx(nu, a, ntarget)];
            }
            WB[idx(mu, a, ntarget)] = wb;
            DU[idx(mu, a, ntarget)] = du;
        }
    }

    std::vector<double> DWB(static_cast<size_t>(norb) * ntarget, 0.0);
    std::vector<double> DUH(static_cast<size_t>(norb) * ntarget, 0.0);
    std::vector<double> WDU(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            double dwb = 0.0;
            double duh = 0.0;
            double wdu = 0.0;
            for (int nu = 0; nu < norb; ++nu)
            {
                dwb += D[idx(mu, nu, norb)] * WB[idx(nu, a, ntarget)];
                wdu += W[idx(mu, nu, norb)] * DU[idx(nu, a, ntarget)];
            }
            for (int b = 0; b < ntarget; ++b)
            {
                duh += DU[idx(mu, b, ntarget)] * H[idx(b, a, ntarget)];
            }
            DWB[idx(mu, a, ntarget)] = dwb;
            DUH[idx(mu, a, ntarget)] = duh;
            WDU[idx(mu, a, ntarget)] = wdu;
        }
    }

    std::vector<double> G_U(static_cast<size_t>(norb) * ntarget, 0.0);
    for (size_t i = 0; i < G_U.size(); ++i)
    {
        G_U[i] = 2.0 * (DUH[i] - DWB[i]);
    }

    // G_M = U^T D U.  The M term adds B^T W B to H, so its adjoint enters
    // G_B through both B positions.
    std::vector<double> G_M(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int mu = 0; mu < norb; ++mu)
            {
                value += U[idx(mu, a, ntarget)] * DU[idx(mu, b, ntarget)];
            }
            G_M[idx(a, b, ntarget)] = value;
        }
    }

    std::vector<double> WBGM(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            double value = 0.0;
            for (int b = 0; b < ntarget; ++b)
            {
                value += WB[idx(mu, b, ntarget)] * G_M[idx(b, a, ntarget)];
            }
            WBGM[idx(mu, a, ntarget)] = value;
        }
    }

    std::vector<double> G_B(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            double value = -2.0 * WDU[idx(mu, a, ntarget)]
                           + 2.0 * WBGM[idx(mu, a, ntarget)];
            for (int nu = 0; nu < norb; ++nu)
            {
                value += S[idx(mu, nu, norb)] * G_U[idx(nu, a, ntarget)];
            }
            G_B[idx(mu, a, ntarget)] = value;
        }
    }

    std::vector<double> G_G(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int mu = 0; mu < norb; ++mu)
            {
                value += C[idx(mu, a, ntarget)] * G_B[idx(mu, b, ntarget)];
            }
            G_G[idx(a, b, ntarget)] = value;
        }
    }

    // Frechet adjoint of G = S_T^{-1/2}: with S_T = V diag(w) V^T,
    // dG = V [F o (V^T dS_T V)] V^T.  F is the divided difference of
    // f(w)=w^{-1/2}; truncated directions have zero derivative because the
    // current rank-truncation rule is piecewise constant there.
    std::vector<double> eig;
    std::vector<double> eigvec;
    sym_eig(space.target.ST, ntarget, eig, eigvec);
    std::vector<double> G_G_eig(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < ntarget; ++i)
            {
                for (int j = 0; j < ntarget; ++j)
                {
                    value += eigvec[idx(i, a, ntarget)]
                             * G_G[idx(i, j, ntarget)]
                             * eigvec[idx(j, b, ntarget)];
                }
            }
            double factor = 0.0;
            const bool retained = eig[a] > space.rank_tol && eig[b] > space.rank_tol;
            // Branch: both directions must survive the same truncation used
            // to build C'.  Otherwise the piecewise-constant branch has zero
            // derivative.
            if (retained)
            {
                const double scale = std::max(std::abs(eig[a]), std::abs(eig[b]));
                // Branch: coincident eigenvalues need the derivative limit,
                // not the divided difference (which is 0/0).
                if (std::abs(eig[a] - eig[b]) <= 1e-12 * std::max(1.0, scale))
                {
                    factor = -0.5 * std::pow(eig[a], -1.5);
                }
                else
                {
                    factor = (std::pow(eig[a], -0.5) - std::pow(eig[b], -0.5))
                             / (eig[a] - eig[b]);
                }
            }
            G_G_eig[idx(a, b, ntarget)] = factor * value;
        }
    }

    std::vector<double> G_ST(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < ntarget; ++i)
            {
                for (int j = 0; j < ntarget; ++j)
                {
                    value += eigvec[idx(a, i, ntarget)]
                             * G_G_eig[idx(i, j, ntarget)]
                             * eigvec[idx(b, j, ntarget)];
                }
            }
            G_ST[idx(a, b, ntarget)] = value;
        }
    }

    // Direct S dependence of U contributes G_U B^T; the target overlap S_T
    // contributes through C G_ST C^T.
    E.assign(static_cast<size_t>(norb) * norb, 0.0);
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = 0; nu < norb; ++nu)
        {
            double value = 0.0;
            for (int a = 0; a < ntarget; ++a)
            {
                value += G_U[idx(mu, a, ntarget)] * B[idx(nu, a, ntarget)];
            }
            for (int a = 0; a < ntarget; ++a)
            {
                for (int b = 0; b < ntarget; ++b)
                {
                    value += C[idx(mu, a, ntarget)]
                             * G_ST[idx(a, b, ntarget)]
                             * C[idx(nu, b, ntarget)];
                }
            }
            E[idx(mu, nu, norb)] = value;
        }
    }
    // Symmetric variations only: the antisymmetric part cannot be probed by
    // dS and must not contribute to the force.
    for (int mu = 0; mu < norb; ++mu)
    {
        for (int nu = mu + 1; nu < norb; ++nu)
        {
            const double sym = 0.5 * (E[idx(mu, nu, norb)]
                                      + E[idx(nu, mu, norb)]);
            E[idx(mu, nu, norb)] = sym;
            E[idx(nu, mu, norb)] = sym;
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
