#include "mbis_alm.h"

#include <algorithm>
#include <cmath>

namespace mbis_alm
{

std::size_t Grid::size() const
{
    return static_cast<std::size_t>(n[0]) * n[1] * n[2];
}

double Grid::dV() const
{
    const Vec3& a = dv[0];
    const Vec3& b = dv[1];
    const Vec3& c = dv[2];
    const double det = a.x * (b.y * c.z - b.z * c.y)
                     - a.y * (b.x * c.z - b.z * c.x)
                     + a.z * (b.x * c.y - b.y * c.x);
    return std::abs(det);
}

Vec3 Grid::point(int ix, int iy, int iz) const
{
    return origin + ix * dv[0] + iy * dv[1] + iz * dv[2];
}

namespace
{

std::vector<Vec3> grid_points(const Grid& grid)
{
    std::vector<Vec3> pts(grid.size());
    std::size_t k = 0;
    for (int ix = 0; ix < grid.n[0]; ++ix)
        for (int iy = 0; iy < grid.n[1]; ++iy)
            for (int iz = 0; iz < grid.n[2]; ++iz)
                pts[k++] = grid.point(ix, iy, iz);
    return pts;
}

std::vector<Vec3> image_translations(const Grid& grid, int images)
{
    std::vector<Vec3> T{Vec3{0.0, 0.0, 0.0}};
    if (images < 1)
        return T;
    T.clear();
    const Vec3 c0 = grid.n[0] * grid.dv[0];
    const Vec3 c1 = grid.n[1] * grid.dv[1];
    const Vec3 c2 = grid.n[2] * grid.dv[2];
    for (int lx = -images; lx <= images; ++lx)
        for (int ly = -images; ly <= images; ++ly)
            for (int lz = -images; lz <= images; ++lz)
                T.push_back(lx * c0 + ly * c1 + lz * c2);
    return T;
}

// Slater-s pro-density summed over periodic images (cutoff applied per image).
double pro_density_at(const Vec3& p, const Vec3& R, double N, double zeta,
                      const std::vector<Vec3>& images, double cutoff)
{
    double acc = 0.0;
    for (const Vec3& T : images)
    {
        const double r = norm(p - (R + T));
        if (r <= cutoff)
            acc += std::exp(-zeta * r);
    }
    return N * zeta * zeta * zeta / (8.0 * M_PI) * acc;
}

void fill_weights(const std::vector<Vec3>& pts, const Grid& grid,
                  const std::vector<Atom>& atoms, const std::vector<double>& N,
                  const std::vector<double>& zeta, int images, double cutoff,
                  MbisState& st)
{
    const std::size_t nxyz = pts.size();
    const int nat = static_cast<int>(atoms.size());
    const std::vector<Vec3> imgs = image_translations(grid, images);
    std::vector<double> pro(nat * nxyz, 0.0);
    std::vector<double> P(nxyz, 0.0);
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const double v = pro_density_at(pts[ig], atoms[a].R, N[a], zeta[a], imgs, cutoff);
            pro[a * nxyz + ig] = v;
            P[ig] += v;
        }
    st.w.assign(nat, std::vector<double>(nxyz, 0.0));
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
            st.w[a][ig] = (P[ig] > 1e-300) ? pro[a * nxyz + ig] / P[ig] : 0.0;
}

} // namespace

MbisState weights_from_params(const std::vector<double>& /*rho*/, const Grid& grid,
                              const std::vector<Atom>& atoms,
                              const std::vector<double>& N,
                              const std::vector<double>& zeta,
                              int images, double cutoff)
{
    MbisState st;
    st.N = N;
    st.zeta = zeta;
    fill_weights(grid_points(grid), grid, atoms, N, zeta, images, cutoff, st);
    return st;
}

MbisState compute_mbis(const std::vector<double>& rho, const Grid& grid,
                       const std::vector<Atom>& atoms, const MbisConfig& cfg)
{
    const std::vector<Vec3> pts = grid_points(grid);
    const std::size_t nxyz = pts.size();
    const int nat = static_cast<int>(atoms.size());
    const double dV = grid.dV();

    // direct distances (used for the mean-radius update)
    std::vector<std::vector<double>> r(nat, std::vector<double>(nxyz, 0.0));
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
            r[a][ig] = norm(pts[ig] - atoms[a].R);

    MbisState st;
    st.N.resize(nat);
    st.zeta.assign(nat, cfg.init_zeta);
    for (int a = 0; a < nat; ++a)
        st.N[a] = atoms[a].Zval;

    for (int it = 0; it < cfg.max_iter; ++it)
    {
        fill_weights(pts, grid, atoms, st.N, st.zeta, cfg.images, cfg.cutoff, st);
        double res = 0.0;
        std::vector<double> Nn(nat, 0.0), zn(nat, 0.0);
        for (int a = 0; a < nat; ++a)
        {
            double na = 0.0, mr = 0.0;
            for (std::size_t ig = 0; ig < nxyz; ++ig)
            {
                const double wa = st.w[a][ig];
                na += wa * rho[ig] * dV;
                mr += r[a][ig] * wa * rho[ig] * dV;
            }
            Nn[a] = na;
            zn[a] = (na > 1e-12 && mr > 1e-12) ? 3.0 * na / mr : st.zeta[a];
            res = std::max(res, std::abs(na - st.N[a]));
        }
        if (cfg.symmetrize)
        {
            // Symmetry projection for equivalent atoms (same valence charge):
            // average the population/exponent so a non-primitive cell cannot
            // break the physically equivalent sites.  Use temporaries so the
            // sum is not polluted by earlier in-place assignments.
            std::vector<double> Ns(nat, 0.0), zs(nat, 0.0);
            for (int a = 0; a < nat; ++a)
            {
                int na = 0;
                double nm = 0.0, zm = 0.0;
                for (int b = 0; b < nat; ++b)
                    if (std::abs(atoms[b].Zval - atoms[a].Zval) < 1e-12)
                    {
                        nm += Nn[b];
                        zm += zn[b];
                        ++na;
                    }
                Ns[a] = nm / na;
                zs[a] = zm / na;
            }
            Nn = Ns;
            zn = zs;
        }
        for (int a = 0; a < nat; ++a)
        {
            st.N[a] = (1.0 - cfg.damp) * st.N[a] + cfg.damp * Nn[a];
            st.zeta[a] = (1.0 - cfg.damp) * st.zeta[a] + cfg.damp * zn[a];
        }
        if (res < cfg.tol)
            break;
    }
    fill_weights(pts, grid, atoms, st.N, st.zeta, cfg.images, cfg.cutoff, st);
    return st;
}

Observables observe(const MbisState& st, const std::vector<double>& rho,
                    const std::vector<double>& spin, const Grid& grid)
{
    Observables o;
    const std::size_t nxyz = grid.size();
    const int nat = static_cast<int>(st.w.size());
    o.N.assign(nat, 0.0);
    o.M.assign(nat, 0.0);
    const double dV = grid.dV();
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            o.N[a] += st.w[a][ig] * rho[ig] * dV;
            if (!spin.empty())
                o.M[a] += st.w[a][ig] * spin[ig] * dV;
        }
    return o;
}

std::vector<double> constraint_potential(const MbisState& st,
                                         const std::vector<double>& coef,
                                         const Grid& grid)
{
    std::vector<double> v(grid.size(), 0.0);
    const int nat = static_cast<int>(st.w.size());
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < grid.size(); ++ig)
            v[ig] += coef[a] * st.w[a][ig];
    return v;
}

std::vector<Vec3> constraint_force(const MbisState& st, const Grid& grid,
                                   const std::vector<Atom>& atoms,
                                   const std::vector<double>& rho,
                                   const std::vector<double>& spin,
                                   const std::vector<double>& mu_c,
                                   const std::vector<double>& mu_s)
{
    const std::vector<Vec3> pts = grid_points(grid);
    const std::size_t nxyz = pts.size();
    const int nat = static_cast<int>(atoms.size());
    const double dV = grid.dV();

    // frozen-parameter pro-densities and their centre derivatives
    std::vector<double> pro(nat * nxyz, 0.0);
    std::vector<double> P(nxyz, 0.0);
    std::vector<std::vector<Vec3>> dpro(nat, std::vector<Vec3>(nxyz, Vec3{}));
    for (int a = 0; a < nat; ++a)
    {
        const double zeta = st.zeta[a];
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const Vec3 d = pts[ig] - atoms[a].R;
            const double ra = norm(d);
            const double pa = st.N[a] * zeta * zeta * zeta / (8.0 * M_PI) * std::exp(-zeta * ra);
            pro[a * nxyz + ig] = pa;
            P[ig] += pa;
            if (ra > 1e-12)
                dpro[a][ig] = (zeta / ra) * (1.0 / (8.0 * M_PI)) * st.N[a]
                              * zeta * zeta * zeta * std::exp(-zeta * ra) * d;
        }
    }

    std::vector<Vec3> F(nat, Vec3{});
    for (std::size_t ig = 0; ig < nxyz; ++ig)
    {
        if (P[ig] <= 1e-300)
            continue;
        // drive_a = mu_c[a] rho + mu_s[a] spin ; weighted sum
        std::vector<double> drive(nat, 0.0);
        double wdrive = 0.0;
        for (int a = 0; a < nat; ++a)
        {
            drive[a] = mu_c[a] * rho[ig] + (spin.empty() ? 0.0 : mu_s[a] * spin[ig]);
            wdrive += (pro[a * nxyz + ig] / P[ig]) * drive[a];
        }
        for (int J = 0; J < nat; ++J)
        {
            const double coef = (drive[J] - wdrive) / P[ig] * dV;
            F[J] = F[J] + coef * dpro[J][ig];
        }
    }
    return F;
}

// ---------------- ALM ----------------

void Alm::resize(int n)
{
    c_.assign(n, AlmComponent{});
}

void Alm::arm(const std::vector<double>& chi, double kappa, double kmin, double kmax)
{
    for (int i = 0; i < size(); ++i)
    {
        const double a = std::abs(chi[i]);
        double k = (a > 1e-12) ? kappa / a : kappa;
        k = std::min(std::max(k, kmin), kmax);
        c_[i].k = k;
        c_[i].chi = chi[i];
        c_[i].armed = true;
    }
}

std::vector<double> Alm::mu_eff(const std::vector<double>& Q) const
{
    std::vector<double> out(size(), 0.0);
    for (int i = 0; i < size(); ++i)
        out[i] = mu_eff(i, Q[i]);
    return out;
}

double Alm::damping(int i) const
{
    const double a = std::abs(c_[i].chi);
    return (a > 1e-12) ? 1.0 / (1.0 + c_[i].k * a) : 1.0;
}

double Alm::update(const std::vector<double>& Q, double cap)
{
    double mx = 0.0;
    for (int i = 0; i < size(); ++i)
    {
        double d = c_[i].k * (Q[i] - c_[i].target);
        if (std::abs(d) > cap)
            d = std::copysign(cap, d);
        c_[i].mu += d;
        mx = std::max(mx, std::abs(d));
    }
    return mx;
}

double Alm::energy_correction(const std::vector<double>& Q) const
{
    double e = 0.0;
    for (int i = 0; i < size(); ++i)
    {
        const double r = Q[i] - c_[i].target;
        e += c_[i].mu * r + 0.5 * c_[i].k * r * r;
    }
    return e;
}

} // namespace mbis_alm
