#include "source_estate/module_constraint/alloc_pw.h"

#include "source_estate/module_constraint/alloc_linalg.h"

#include "source_base/parallel_reduce.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace alloc
{
namespace
{

void check_space(const PwTargetSpace& space)
{
    const int r = space.ntarget;
    // Branch: every target-space array must describe the same R x R / R map.
    if (static_cast<int>(space.ST.size()) != r * r
        || static_cast<int>(space.ST_inv_root.size()) != r * r
        || static_cast<int>(space.atom.size()) != r
        || static_cast<int>(space.constraint.size()) != r)
    {
        throw std::invalid_argument("PW target space is inconsistent");
    }
    for (int a = 0; a < r; ++a)
    {
        // Branch: an invalid ownership label cannot be aggregated.
        if (space.atom[a] < 0 || space.constraint[a] < -1
            || space.constraint[a] >= space.nconstraint)
        {
            throw std::invalid_argument("PW target ownership is out of range");
        }
    }
}

void check_bands(const std::vector<std::vector<std::complex<double>>>& becp,
                 const std::vector<double>& occ,
                 const int ntarget,
                 const char* name)
{
    // Branch: occupations and bands describe the same spin channel.
    if (becp.size() != occ.size())
    {
        throw std::invalid_argument(std::string(name)
                                    + " band/occupation counts differ");
    }
    for (const std::vector<std::complex<double>>& band : becp)
    {
        // Branch: every band stores one overlap per target projector.
        if (static_cast<int>(band.size()) != ntarget)
        {
            throw std::invalid_argument(std::string(name)
                                        + " becp has wrong target count");
        }
    }
}

} // namespace

PwTargetShell select_pw_target_shell(const std::vector<int>& nh_per_atom,
                                     const int target_l)
{
    PwTargetShell selection;
    // Branch: a negative angular momentum cannot select a physical shell.
    if (target_l < 0)
    {
        return selection;
    }
    const int width = 2 * target_l + 1;
    const int offset = target_l * target_l;
    // The Projector/onsite tables store every atom's (l,m) block
    // consecutively, so a shell row must carry the atom's global row base.
    // Returning the local offset would make every atom read the first
    // atom's projector and collapse the target space to that single atom.
    int row_base = 0;
    for (int iat = 0; iat < static_cast<int>(nh_per_atom.size()); ++iat)
    {
        // Branch: OnsiteProjector stores a complete (l,m) block per atom, so
        // the shell exists exactly when offset+width fits in nh_per_atom.
        if (nh_per_atom[iat] >= offset + width)
        {
            for (int m = 0; m < width; ++m)
            {
                selection.row.push_back(row_base + offset + m);
                selection.atom.push_back(iat);
            }
        }
        row_base += nh_per_atom[iat];
    }
    return selection;
}

void build_pw_target(const std::vector<std::complex<double>>& ST,
                     const int ntarget,
                     const std::vector<int>& target_atom,
                     const std::vector<int>& target_constraint,
                     const int nconstraint,
                     PwTargetSpace& out,
                     const double rank_tol)
{
    // Branch: the projector overlap must be exactly R x R and every target
    // needs both an atom and an optional constraint label.
    if (ntarget < 0 || static_cast<int>(ST.size()) != ntarget * ntarget
        || static_cast<int>(target_atom.size()) != ntarget
        || static_cast<int>(target_constraint.size()) != ntarget
        || nconstraint < 0)
    {
        throw std::invalid_argument("PW target input has wrong dimensions");
    }
    out.ntarget = ntarget;
    out.nconstraint = nconstraint;
    out.atom = target_atom;
    out.constraint = target_constraint;
    out.rank = 0;
    out.cond = 1.0;
    out.ST.assign(static_cast<size_t>(ntarget) * ntarget,
                  std::complex<double>(0.0, 0.0));
    out.ST_inv_root.assign(static_cast<size_t>(ntarget) * ntarget,
                           std::complex<double>(0.0, 0.0));
    for (int a = 0; a < ntarget; ++a)
    {
        // Branch: reject an ownership map that cannot be aggregated later.
        if (target_atom[a] < 0 || target_constraint[a] < -1
            || target_constraint[a] >= nconstraint)
        {
            throw std::invalid_argument("PW target ownership is out of range");
        }
        for (int b = 0; b < ntarget; ++b)
        {
            // The projector overlap is Hermitian by construction; an
            // explicit symmetrization removes numerical skew from the input.
            const std::complex<double> value
                = 0.5
                  * (ST[static_cast<size_t>(a) * ntarget + b]
                     + std::conj(ST[static_cast<size_t>(b) * ntarget + a]));
            out.ST[static_cast<size_t>(a) * ntarget + b] = value;
        }
    }
    // Branch: an empty target set has no eigenproblem and represents P=0.
    if (ntarget == 0)
    {
        return;
    }

    std::vector<double> eig;
    std::vector<std::complex<double>> eigvec;
    herm_eig(out.ST, ntarget, eig, eigvec);
    for (int i = 0; i < ntarget; ++i)
    {
        // Branch: count only directions retained by the inverse-sqrt cutoff.
        if (eig[i] > rank_tol)
        {
            ++out.rank;
        }
    }
    // Branch: a singular target overlap has no finite condition number.
    if (eig.front() <= rank_tol)
    {
        out.cond = std::numeric_limits<double>::infinity();
    }
    else
    {
        out.cond = eig.back() / eig.front();
    }
    herm_inv_sqrt(out.ST, ntarget, out.ST_inv_root, rank_tol);
}

void build_pw_target(const std::vector<double>& ST,
                     const int ntarget,
                     const std::vector<int>& target_atom,
                     const std::vector<int>& target_constraint,
                     const int nconstraint,
                     PwTargetSpace& out,
                     const double rank_tol)
{
    // Branch: preserve the Gamma-point API by lifting the real symmetric
    // overlap into the complex Hermitian representation.
    std::vector<std::complex<double>> complex_st(
        ST.size(), std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < ST.size(); ++i)
    {
        complex_st[i] = std::complex<double>(ST[i], 0.0);
    }
    build_pw_target(complex_st, ntarget, target_atom, target_constraint,
                    nconstraint, out, rank_tol);
}

void build_pw_projector_overlap(const std::complex<double>* tab_atomic,
                                const int npw,
                                const int ntarget,
                                std::vector<std::complex<double>>& ST)
{
    // Branch: the tabulation and target count must define a Hermitian R x R
    // overlap; an empty target set is a valid P=0 backend.
    if (npw < 0 || ntarget < 0 || (ntarget > 0 && tab_atomic == nullptr))
    {
        throw std::invalid_argument("PW projector overlap input is invalid");
    }
    ST.assign(static_cast<size_t>(ntarget) * ntarget,
              std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> local(
        static_cast<size_t>(ntarget) * ntarget,
        std::complex<double>(0.0, 0.0));
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = a; b < ntarget; ++b)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < npw; ++ig)
            {
                value += std::conj(tab_atomic[a * npw + ig])
                         * tab_atomic[b * npw + ig];
            }
            local[static_cast<size_t>(a) * ntarget + b] = value;
        }
    }
    // Reduce all R x R elements in one pool collective.  The previous
    // per-element reductions serialized R(R+1)/2 collectives per k-point.
#ifdef __MPI
    Parallel_Reduce::reduce_pool(local.data(), static_cast<int>(local.size()));
#endif
    for (int a = 0; a < ntarget; ++a)
    {
        for (int b = 0; b < ntarget; ++b)
        {
            // Only the upper triangle was formed locality-by-locality.  Use
            // the Hermitian partner so the reduced matrix is full, not a
            // triangular matrix with an implicit zero lower half.
            const int row = (a < b) ? a : b;
            const int col = (a < b) ? b : a;
            const std::complex<double> value
                = local[static_cast<size_t>(row) * ntarget + col];
            ST[static_cast<size_t>(a) * ntarget + b]
                = (a <= b) ? value : std::conj(value);
        }
    }
}

void pw_becp_from_projectors(
    const std::complex<double>* tab_atomic,
    const int npw,
    const int ntarget,
    const std::complex<double>* psi,
    const int ld_psi,
    const int nband,
    std::vector<std::vector<std::complex<double>>>& becp)
{
    // Branch: every band needs a valid start in the plane-wave buffer and the
    // tabulation must cover the requested projectors.
    if (npw < 0 || ntarget < 0 || nband < 0 || ld_psi < npw
        || (ntarget > 0 && tab_atomic == nullptr)
        || (nband > 0 && psi == nullptr))
    {
        throw std::invalid_argument("PW becp input is invalid");
    }
    becp.assign(nband,
                std::vector<std::complex<double>>(ntarget,
                                                  std::complex<double>(0.0, 0.0)));
    for (int band = 0; band < nband; ++band)
    {
        const std::complex<double>* p = psi + static_cast<size_t>(band) * ld_psi;
        for (int a = 0; a < ntarget; ++a)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < npw; ++ig)
            {
                value += std::conj(tab_atomic[a * npw + ig]) * p[ig];
            }
#ifdef __MPI
            Parallel_Reduce::reduce_pool(value);
#endif
            becp[band][a] = value;
        }
    }
}

void orthogonalize_pw_overlaps(const PwTargetSpace& space,
                               const std::vector<std::complex<double>>& raw,
                               std::vector<std::complex<double>>& orth)
{
    check_space(space);
    // Branch: one raw overlap per raw projector is required.
    if (static_cast<int>(raw.size()) != space.ntarget)
    {
        throw std::invalid_argument("PW overlap vector has wrong length");
    }
    orth.assign(space.ntarget, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < space.ntarget; ++a)
    {
        for (int b = 0; b < space.ntarget; ++b)
        {
            // orth = G^H raw because raw stores <alpha_b|psi>.  The
            // Hermitian G is not real at a general k-point.
            orth[a] += std::conj(space.ST_inv_root[static_cast<size_t>(a)
                                                      * space.ntarget + b])
                       * raw[b];
        }
    }
}

void build_pw_target_derivative(
    const PwTargetSpace& space,
    const std::vector<std::complex<double>>& derivative_overlap,
    std::vector<std::complex<double>>& dST_inv_root)
{
    check_space(space);
    const int r = space.ntarget;
    // Guard: one derivative overlap per raw target pair is required.
    if (static_cast<int>(derivative_overlap.size()) != r * r)
    {
        throw std::invalid_argument("PW target derivative has wrong dimensions");
    }
    // Branch: an empty target set has no inverse-root derivative.
    if (r == 0)
    {
        dST_inv_root.clear();
        return;
    }

    std::vector<std::complex<double>> dST(
        static_cast<size_t>(r) * r, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            // dST = D + D^dagger.  Keep the full Hermitian perturbation;
            // dropping the imaginary part is only valid at Gamma.
            dST[static_cast<size_t>(a) * r + b]
                = derivative_overlap[static_cast<size_t>(a) * r + b]
                  + std::conj(derivative_overlap[static_cast<size_t>(b) * r + a]);
        }
    }
    herm_inv_sqrt_derivative(space.ST, dST, r, dST_inv_root, 1e-10);
}

void orthogonalize_pw_overlaps_derivative(
    const PwTargetSpace& space,
    const std::vector<std::complex<double>>& dST_inv_root,
    const std::vector<std::complex<double>>& raw,
    const std::vector<std::complex<double>>& d_raw,
    std::vector<std::complex<double>>& orth,
    std::vector<std::complex<double>>& d_orth)
{
    check_space(space);
    const int r = space.ntarget;
    // Guard: the value, its derivative and dG must all describe the same
    // target space before any contraction is formed.
    if (static_cast<int>(raw.size()) != r
        || static_cast<int>(d_raw.size()) != r
        || static_cast<int>(dST_inv_root.size()) != r * r)
    {
        throw std::invalid_argument(
            "PW target derivative vectors have wrong dimensions");
    }
    orth.assign(r, std::complex<double>(0.0, 0.0));
    d_orth.assign(r, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            const std::complex<double> g
                = space.ST_inv_root[static_cast<size_t>(a) * r + b];
            const std::complex<double> dg
                = dST_inv_root[static_cast<size_t>(a) * r + b];
            // Overlap values rotate with G^H, while target vectors rotate
            // with G itself.
            orth[a] += std::conj(g) * raw[b];
            d_orth[a] += std::conj(g) * d_raw[b] + std::conj(dg) * raw[b];
        }
    }
}

PwTargetMoments observe_pw_target_moments(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& becp_up,
    const std::vector<double>& occ_up,
    const std::vector<std::vector<std::complex<double>>>& becp_dn,
    const std::vector<double>& occ_dn)
{
    check_space(space);
    check_bands(becp_up, occ_up, space.ntarget, "up");
    check_bands(becp_dn, occ_dn, space.ntarget, "down");

    PwTargetMoments result;
    result.n_T_target.assign(space.ntarget, 0.0);
    result.m_T_target.assign(space.ntarget, 0.0);
    result.n_T_constraint.assign(space.nconstraint, 0.0);
    result.m_T_constraint.assign(space.nconstraint, 0.0);

    const auto accumulate = [&](const std::vector<std::vector<std::complex<double>>>& becp,
                                const std::vector<double>& occ,
                                const double spin_sign) {
        for (size_t band = 0; band < becp.size(); ++band)
        {
            // Branch: an unoccupied band contributes nothing.
            if (occ[band] == 0.0)
            {
                continue;
            }
            std::vector<std::complex<double>> o;
            orthogonalize_pw_overlaps(space, becp[band], o);
            for (int a = 0; a < space.ntarget; ++a)
            {
                const double weight = occ[band] * std::norm(o[a]);
                result.n_T_target[a] += weight;
                result.m_T_target[a] += spin_sign * weight;
            }
        }
    };
    // Spin-up and spin-down share the orthonormal target projector; only the
    // magnetization sign distinguishes their contributions.
    accumulate(becp_up, occ_up, +1.0);
    accumulate(becp_dn, occ_dn, -1.0);

    for (int a = 0; a < space.ntarget; ++a)
    {
        result.n_T_total += result.n_T_target[a];
        result.m_T_total += result.m_T_target[a];
        const int alpha = space.constraint[a];
        // Branch: an unowned target enters P but not a constraint sum.
        if (alpha >= 0)
        {
            result.n_T_constraint[alpha] += result.n_T_target[a];
            result.m_T_constraint[alpha] += result.m_T_target[a];
        }
    }
    return result;
}

void build_pw_alloc_orthogonal_phi(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    std::vector<std::vector<std::complex<double>>>& phi_orth)
{
    check_space(space);
    const int r = space.ntarget;
    // Branch: an empty target set has no orthogonalized products to prepare.
    if (r == 0)
    {
        phi_orth.clear();
        return;
    }
    // Branch: every target vector must exist before it can be rotated into
    // the orthonormal basis.
    if (static_cast<int>(phi.size()) != r)
    {
        throw std::invalid_argument("PW allocation target input has wrong dimensions");
    }
    const int nbasis = static_cast<int>(phi[0].size());
    phi_orth.assign(
        r, std::vector<std::complex<double>>(nbasis,
                                             std::complex<double>(0.0, 0.0)));
    for (int a = 0; a < r; ++a)
    {
        // Branch: a target vector with a different basis length cannot be
        // rotated by the common R x R overlap factor.
        if (static_cast<int>(phi[a].size()) != nbasis)
        {
            throw std::invalid_argument("PW target coefficient vector has wrong length");
        }
        for (int b = 0; b < r; ++b)
        {
            const std::complex<double> g
                = space.ST_inv_root[static_cast<size_t>(a) * r + b];
            for (int i = 0; i < nbasis; ++i)
            {
                phi_orth[a][i] += g * phi[b][i];
            }
        }
    }
}

void build_pw_alloc_orthogonal_context(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& m_raw,
    PwAllocOrthogonalContext& context)
{
    check_space(space);
    const int r = space.ntarget;
    // Branch: an empty target set has no orthogonalized products to prepare.
    if (r == 0)
    {
        context.phi_orth.clear();
        context.vphi_orth.clear();
        context.m_orth.clear();
        return;
    }
    // Branch: all raw target vectors and the overlap matrix must describe one
    // R x R target space and one common plane-wave basis.
    if (static_cast<int>(phi.size()) != r
        || static_cast<int>(vphi.size()) != r
        || static_cast<int>(m_raw.size()) != r * r)
    {
        throw std::invalid_argument("PW allocation target input has wrong dimensions");
    }
    build_pw_alloc_orthogonal_phi(space, phi, context.phi_orth);
    const int nbasis = static_cast<int>(context.phi_orth[0].size());
    context.vphi_orth.assign(
        r, std::vector<std::complex<double>>(nbasis,
                                             std::complex<double>(0.0, 0.0)));
    for (int a = 0; a < r; ++a)
    {
        // Branch: V|alpha> must live in the same plane-wave partition as the
        // raw target that defines phi_orth.
        if (static_cast<int>(vphi[a].size()) != nbasis)
        {
            throw std::invalid_argument("PW target coefficient vector has wrong length");
        }
        for (int b = 0; b < r; ++b)
        {
            const std::complex<double> g
                = space.ST_inv_root[static_cast<size_t>(a) * r + b];
            for (int i = 0; i < nbasis; ++i)
            {
                context.vphi_orth[a][i] += g * vphi[b][i];
            }
        }
    }

    // m' = G^T m_raw G in the orthonormal target basis.  This R x R product is
    // band independent, so it is formed once per prepared local potential.
    context.m_orth.assign(static_cast<size_t>(r) * r,
                          std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            std::complex<double> value(0.0, 0.0);
            for (int c = 0; c < r; ++c)
            {
                for (int d = 0; d < r; ++d)
                {
                    value += space.ST_inv_root[static_cast<size_t>(a) * r + c]
                             * m_raw[static_cast<size_t>(c) * r + d]
                             * space.ST_inv_root[static_cast<size_t>(d) * r + b];
                }
            }
            context.m_orth[static_cast<size_t>(a) * r + b] = value;
        }
    }
}

void apply_pw_alloc_delta_h_orthogonal(
    const PwTargetSpace& space,
    const PwAllocOrthogonalContext& context,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& psi,
    const std::vector<std::complex<double>>& raw_overlap,
    const std::vector<std::complex<double>>& vpsi,
    const std::vector<double>& lambda,
    std::vector<std::complex<double>>& delta)
{
    check_space(space);
    const int r = space.ntarget;
    const int nbasis = static_cast<int>(psi.size());
    // Branch: every per-band vector and prepared target product must share the
    // current plane-wave basis and constraint count.
    if (static_cast<int>(vpsi.size()) != nbasis
        || static_cast<int>(raw_overlap.size()) != r
        || static_cast<int>(vphi.size()) != r
        || static_cast<int>(context.phi_orth.size()) != r
        || static_cast<int>(context.vphi_orth.size()) != r
        || static_cast<int>(context.m_orth.size()) != r * r
        || static_cast<int>(lambda.size()) != space.nconstraint)
    {
        throw std::invalid_argument("PW allocation apply input has wrong dimensions");
    }
    for (int a = 0; a < r; ++a)
    {
        // Branch: raw and prepared target vectors all live in the same basis.
        if (static_cast<int>(vphi[a].size()) != nbasis
            || static_cast<int>(context.phi_orth[a].size()) != nbasis
            || static_cast<int>(context.vphi_orth[a].size()) != nbasis)
        {
            throw std::invalid_argument("PW target coefficient vector has wrong length");
        }
    }

    // Raw <alpha|V|psi> is evaluated once for the whole target vector and then
    // pooled once.  The old per-target collectives dominated small-R PW runs.
    std::vector<std::complex<double>> braw(r, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            braw[a] += std::conj(vphi[a][i]) * psi[i];
        }
    }
#ifdef __MPI
    Parallel_Reduce::reduce_pool(braw.data(), r);
#endif
    std::vector<std::complex<double>> o;
    std::vector<std::complex<double>> borth;
    orthogonalize_pw_overlaps(space, raw_overlap, o);
    orthogonalize_pw_overlaps(space, braw, borth);

    std::vector<std::complex<double>> pvpsi(nbasis,
                                             std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> v_ppsi(nbasis,
                                              std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> pvp(nbasis,
                                          std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> qlambda(nbasis,
                                               std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        for (int i = 0; i < nbasis; ++i)
        {
            pvpsi[i] += context.phi_orth[a][i] * borth[a];
            v_ppsi[i] += context.vphi_orth[a][i] * o[a];
        }
    }
    for (int b = 0; b < r; ++b)
    {
        std::complex<double> coefficient(0.0, 0.0);
        for (int a = 0; a < r; ++a)
        {
            coefficient += context.m_orth[static_cast<size_t>(b) * r + a] * o[a];
        }
        for (int i = 0; i < nbasis; ++i)
        {
            pvp[i] += context.phi_orth[b][i] * coefficient;
        }
        const int alpha = space.constraint[b];
        // Branch: unowned targets have no Q_i term in the local multiplier.
        if (alpha >= 0)
        {
            for (int i = 0; i < nbasis; ++i)
            {
                qlambda[i] += lambda[alpha] * context.phi_orth[b][i] * o[b];
            }
        }
    }

    delta.assign(nbasis, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < nbasis; ++i)
    {
        delta[i] = vpsi[i] - pvpsi[i] - v_ppsi[i] + pvp[i] + qlambda[i];
    }
}

void apply_pw_alloc_delta_h(
    const PwTargetSpace& space,
    const std::vector<std::vector<std::complex<double>>>& phi,
    const std::vector<std::complex<double>>& psi,
    const std::vector<std::complex<double>>& raw_overlap,
    const std::vector<std::complex<double>>& vpsi,
    const std::vector<std::vector<std::complex<double>>>& vphi,
    const std::vector<std::complex<double>>& m_raw,
    const std::vector<double>& lambda,
    std::vector<std::complex<double>>& delta)
{
    PwAllocOrthogonalContext context;
    build_pw_alloc_orthogonal_context(space, phi, vphi, m_raw, context);
    apply_pw_alloc_delta_h_orthogonal(space, context, vphi, psi, raw_overlap,
                                      vpsi, lambda, delta);
}

} // namespace alloc
