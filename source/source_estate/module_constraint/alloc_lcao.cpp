#include "source_estate/module_constraint/alloc_lcao.h"

#include "source_estate/module_constraint/alloc_linalg.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace alloc
{
namespace
{

void check_square(const std::vector<double>& A, const int n, const char* name)
{
    // Branch: every dense matrix in this backend is a square n x n array;
    // reject a caller dimension error before any index arithmetic.
    if (n < 0 || static_cast<int>(A.size()) != n * n)
    {
        throw std::invalid_argument(std::string(name) + " has wrong size");
    }
}

void finish_target(const std::vector<double>& C,
                   const int norb,
                   const int ntarget,
                   const std::vector<double>& ST,
                   LcaoTargetSpace& out,
                   const double rank_tol)
{
    check_square(ST, ntarget, "S_T");

    out.ST = ST;
    out.cond = 1.0;
    out.rank = 0;
    // Branch: an empty target set has no overlap eigenproblem; leave the
    // identity condition/rank so the audit can still report zero target
    // occupations without a special case in the caller.
    if (ntarget > 0)
    {
        std::vector<double> eig;
        std::vector<double> eigvec;
        sym_eig(ST, ntarget, eig, eigvec);
        for (int i = 0; i < ntarget; ++i)
        {
            // Branch: count only eigenvalues above the same truncation
            // threshold used by sym_inv_sqrt.
            if (eig[i] > rank_tol)
            {
                ++out.rank;
            }
        }
        // Branch: a target space with a (near-)singular overlap cannot define
        // a finite condition number.  Keep the rank-truncated inverse so the
        // caller can still diagnose the selected set, but report infinity.
        if (eig.front() > rank_tol)
        {
            out.cond = eig.back() / eig.front();
        }
        else
        {
            out.cond = std::numeric_limits<double>::infinity();
        }
    }

    sym_inv_sqrt(ST, ntarget, out.ST_inv_root, rank_tol);

    // Cprime = C S_T^{-1/2}.  The loop order is row-major over C.
    out.Cprime.assign(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int i = 0; i < norb; ++i)
    {
        for (int a = 0; a < ntarget; ++a)
        {
            double value = 0.0;
            for (int b = 0; b < ntarget; ++b)
            {
                value += C[static_cast<size_t>(i) * ntarget + b]
                         * out.ST_inv_root[static_cast<size_t>(b) * ntarget + a];
            }
            out.Cprime[static_cast<size_t>(i) * ntarget + a] = value;
        }
    }
}

std::vector<double> selection_matrix(const int norb,
                                     const std::vector<int>& orbital,
                                     const int ntarget)
{
    std::vector<double> C(static_cast<size_t>(norb) * ntarget, 0.0);
    for (int b = 0; b < ntarget; ++b)
    {
        // Branch: a target AO index must address an actual global orbital.
        if (orbital[b] < 0 || orbital[b] >= norb)
        {
            throw std::invalid_argument("target orbital index out of range");
        }
        C[static_cast<size_t>(orbital[b]) * ntarget + b] = 1.0;
    }
    return C;
}

} // namespace

void build_lcao_target(const std::vector<double>& S,
                       const int norb,
                       const std::vector<double>& C,
                       const int ntarget,
                       const std::vector<int>& target_atom,
                       LcaoTargetSpace& out,
                       const double rank_tol)
{
    check_square(S, norb, "S");
    // Branch: C must be exactly the norb x ntarget coefficient matrix and
    // every target needs an atom label for the per-atom aggregation.
    if (static_cast<int>(C.size()) != norb * ntarget)
    {
        throw std::invalid_argument("C has wrong size");
    }
    if (static_cast<int>(target_atom.size()) != ntarget)
    {
        throw std::invalid_argument("target atom count does not match C");
    }
    for (int atom : target_atom)
    {
        // Branch: the dense core cannot aggregate an invalid atom label.
        if (atom < 0)
        {
            throw std::invalid_argument("target atom index is negative");
        }
    }

    out.norb = norb;
    out.ntarget = ntarget;
    out.atom = target_atom;
    out.orbital.clear();

    std::vector<double> ST(static_cast<size_t>(ntarget) * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            double value = 0.0;
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    value += C[static_cast<size_t>(i) * ntarget + a]
                             * S[static_cast<size_t>(i) * norb + j]
                             * C[static_cast<size_t>(j) * ntarget + b];
                }
            }
            ST[static_cast<size_t>(a) * ntarget + b] = value;
        }
    }
    finish_target(C, norb, ntarget, ST, out, rank_tol);
}

void finalize_lcao_target(const int norb,
                          const std::vector<int>& target_orbital,
                          const std::vector<int>& target_atom,
                          const std::vector<double>& ST,
                          LcaoTargetSpace& out,
                          const double rank_tol)
{
    const int ntarget = static_cast<int>(target_orbital.size());
    // Branch: the selection and atom lists must stay one-to-one.
    if (static_cast<int>(target_atom.size()) != ntarget)
    {
        throw std::invalid_argument(
            "target orbital and atom counts differ");
    }

    out.norb = norb;
    out.ntarget = ntarget;
    out.orbital = target_orbital;
    out.atom = target_atom;
    const std::vector<double> C = selection_matrix(norb, target_orbital, ntarget);
    finish_target(C, norb, ntarget, ST, out, rank_tol);
}

void observe_lcao_target(const LcaoTargetSpace& target,
                         const std::vector<double>& S,
                         const std::vector<double>& D_total,
                         const std::vector<double>& D_diff,
                         std::vector<double>& n_T_target,
                         std::vector<double>& m_T_target)
{
    const int norb = target.norb;
    const int ntarget = target.ntarget;
    check_square(S, norb, "S");
    check_square(D_total, norb, "D_total");
    // Branch: an empty spin-difference matrix is the non-magnetic channel.
    if (!D_diff.empty())
    {
        check_square(D_diff, norb, "D_diff");
    }
    // Branch: the coefficient matrix and the target labels must describe the
    // same target space.
    if (static_cast<int>(target.Cprime.size()) != norb * ntarget
        || static_cast<int>(target.atom.size()) != ntarget)
    {
        throw std::invalid_argument("target space is inconsistent");
    }

    n_T_target.assign(ntarget, 0.0);
    m_T_target.assign(ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        // u = S c'_a.  The same vector enters the total and spin-difference
        // contractions, so build it once per target.
        std::vector<double> u(norb, 0.0);
        for (int i = 0; i < norb; ++i)
        {
            for (int j = 0; j < norb; ++j)
            {
                u[i] += S[static_cast<size_t>(i) * norb + j]
                        * target.Cprime[static_cast<size_t>(j) * ntarget + a];
            }
        }

        double n = 0.0;
        double m = 0.0;
        for (int i = 0; i < norb; ++i)
        {
            for (int j = 0; j < norb; ++j)
            {
                const double uij = u[i] * u[j];
                n += uij * D_total[static_cast<size_t>(i) * norb + j];
                // Branch: no spin-difference channel was supplied (nspin=1).
                if (!D_diff.empty())
                {
                    m += uij * D_diff[static_cast<size_t>(i) * norb + j];
                }
            }
        }
        n_T_target[a] = n;
        m_T_target[a] = m;
    }
}

} // namespace alloc
