#include "source_estate/module_constraint/alloc_operator.h"

#include "source_base/parallel_reduce.h"
#include "source_estate/module_constraint/alloc_linalg.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace alloc
{

void AllocOperator::check_vector(const std::vector<double>& x,
                                  const char* name) const
{
    // Branch: reject a caller error before the grid loops index the vector.
    if (static_cast<int>(x.size()) != nrxx_)
    {
        throw std::invalid_argument(std::string(name) + " has wrong length");
    }
}

void AllocOperator::build_grid(const constraint::WeightGrid& wg,
                               const std::vector<int>& target_atom,
                               const std::vector<std::vector<double>>& phi0)
{
    nrxx_ = wg.nrxx();
    dV_ = wg.rho_basis()->omega / static_cast<double>(wg.rho_basis()->nxyz);
    cond_ = 1.0;
    phi_.clear();
    atom_of_.clear();
    w_.clear();

    // The allocation is per physical atom.  A fragment or overlapping
    // constraint map would leave an atom without a unique weight, so require
    // the canonical singleton map used by the S1 prototype.
    const std::vector<std::vector<int>>& constraints = wg.constraint_atoms();
    // Branch: non-singleton constraint maps are unsupported by this backend.
    if (static_cast<int>(constraints.size()) != wg.nat())
    {
        throw std::invalid_argument(
            "AllocOperator requires one singleton constraint per atom");
    }
    for (int atom = 0; atom < wg.nat(); ++atom)
    {
        // Branch: a multi-atom or misordered fragment cannot supply w_atom.
        if (constraints[atom].size() != 1
            || constraints[atom][0] != atom)
        {
            throw std::invalid_argument(
                "AllocOperator requires constraint atom order 0..nat-1");
        }
        w_.push_back(wg.constraint_weight(atom));
    }

    // Branch: empty target set.  Then P=0, Qc=I and A_i=diag(w_i).
    if (target_atom.empty())
    {
        // Branch: phi0 must also be empty; otherwise the caller gave a
        // target function with no atom label.
        if (!phi0.empty())
        {
            throw std::invalid_argument(
                "empty target_atom requires empty target orbitals");
        }
        return;
    }
    // Branch: one atom label is required for every target orbital.
    if (target_atom.size() != phi0.size())
    {
        throw std::invalid_argument(
            "target_atom and phi0 must have the same size");
    }

    const int ntarget = static_cast<int>(target_atom.size());
    atom_of_ = target_atom;
    phi_.resize(ntarget);
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: reject invalid atom labels before the weight table is used.
        if (atom_of_[a] < 0 || atom_of_[a] >= wg.nat())
        {
            throw std::invalid_argument("target atom index out of range");
        }
        // Branch: every target function must live on the shared grid.
        if (static_cast<int>(phi0[a].size()) != nrxx_)
        {
            throw std::invalid_argument("target orbital has wrong grid size");
        }
        phi_[a] = phi0[a];
    }

    // S_T[a,b] = <phi_a|phi_b> on the represented grid.
    std::vector<double> ST(ntarget * ntarget, 0.0);
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = a; b < ntarget; ++b)
        {
            double s = 0.0;
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                s += phi0[a][ir] * phi0[b][ir];
            }
            s *= dV_;
#ifdef __MPI
            Parallel_Reduce::reduce_pool(s);
#endif
            ST[a * ntarget + b] = s;
            ST[b * ntarget + a] = s;
        }
    }

    std::vector<double> eig;
    std::vector<double> eigvec;
    sym_eig(ST, ntarget, eig, eigvec);
    const double rank_tol = 1e-10;
    // Branch: a singular or negative overlap subspace has no finite
    // condition number; retain the inverse-sqrt truncation below but report
    // infinity so callers cannot silently use an ill-conditioned projector.
    if (eig.front() <= rank_tol)
    {
        cond_ = std::numeric_limits<double>::infinity();
    }
    else
    {
        cond_ = eig.back() / eig.front();
    }

    std::vector<double> ST_inv_sqrt;
    sym_inv_sqrt(ST, ntarget, ST_inv_sqrt, rank_tol);

    // phi' = S_T^{-1/2} phi, computed pointwise on the grid.  The overlap of
    // the resulting targets is the identity to the Jacobi solve tolerance.
    for (int a = 0; a < ntarget; ++a)
    {
        phi_[a].assign(nrxx_, 0.0);
        for (int b = 0; b < ntarget; ++b)
        {
            const double t = ST_inv_sqrt[a * ntarget + b];
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                phi_[a][ir] += t * phi0[b][ir];
            }
        }
    }
}

void AllocOperator::overlaps(const std::vector<double>& x,
                             std::vector<double>& o) const
{
    check_vector(x, "x");
    o.assign(phi_.size(), 0.0);
    for (size_t a = 0; a < phi_.size(); ++a)
    {
        double s = 0.0;
        for (int ir = 0; ir < nrxx_; ++ir)
        {
            s += phi_[a][ir] * x[ir];
        }
        s *= dV_;
#ifdef __MPI
        Parallel_Reduce::reduce_pool(s);
#endif
        o[a] = s;
    }
}

void AllocOperator::apply_P(const std::vector<double>& x,
                            std::vector<double>& out) const
{
    check_vector(x, "x");
    std::vector<double> o;
    overlaps(x, o);
    out.assign(nrxx_, 0.0);
    for (size_t a = 0; a < phi_.size(); ++a)
    {
        for (int ir = 0; ir < nrxx_; ++ir)
        {
            out[ir] += phi_[a][ir] * o[a];
        }
    }
}

void AllocOperator::apply_Qc(const std::vector<double>& x,
                             std::vector<double>& out) const
{
    check_vector(x, "x");
    apply_P(x, out);
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        out[ir] = x[ir] - out[ir];
    }
}

void AllocOperator::apply_Qi(const int atom, const std::vector<double>& x,
                             std::vector<double>& out) const
{
    check_vector(x, "x");
    // Branch: an out-of-range atom cannot have an allocation weight.
    if (atom < 0 || atom >= nat())
    {
        throw std::invalid_argument("allocation atom index out of range");
    }

    std::vector<double> o;
    overlaps(x, o);
    out.assign(nrxx_, 0.0);
    // Q_i x: only the target orbitals centred on this atom remain.
    for (int a = 0; a < ntarget(); ++a)
    {
        // Branch: target a belongs to the requested atom.
        if (atom_of_[a] == atom)
        {
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                out[ir] += phi_[a][ir] * o[a];
            }
        }
    }
}

void AllocOperator::apply_A(const int atom, const std::vector<double>& x,
                            std::vector<double>& out) const
{
    check_vector(x, "x");
    // Branch: an out-of-range atom cannot have an allocation weight.
    if (atom < 0 || atom >= nat())
    {
        throw std::invalid_argument("allocation atom index out of range");
    }

    apply_Qi(atom, x, out);

    // Qc W_i Qc x = (I-P) [w_i (I-P)x].  Compute Qc first so the weight is
    // applied only in the complementary subspace, matching design eq. (2.1).
    std::vector<double> qc_x;
    apply_Qc(x, qc_x);
    std::vector<double> weighted(nrxx_, 0.0);
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        weighted[ir] = w_[atom][ir] * qc_x[ir];
    }
    std::vector<double> qc_weighted;
    apply_Qc(weighted, qc_weighted);
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        out[ir] += qc_weighted[ir];
    }
}

} // namespace alloc
