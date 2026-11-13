#include "mbis_alm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <complex>

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

bool invert3(const double a[9], double inv[9])
{
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7])
                     - a[1] * (a[3] * a[8] - a[5] * a[6])
                     + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (std::abs(det) < 1e-14)
        return false;
    const double id = 1.0 / det;
    inv[0] = (a[4] * a[8] - a[5] * a[7]) * id;
    inv[1] = (a[2] * a[7] - a[1] * a[8]) * id;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) * id;
    inv[3] = (a[5] * a[6] - a[3] * a[8]) * id;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) * id;
    inv[5] = (a[2] * a[3] - a[0] * a[5]) * id;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) * id;
    inv[7] = (a[1] * a[6] - a[0] * a[7]) * id;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) * id;
    return true;
}

// Bring a Cartesian displacement into the cell [-0.5,0.5) (row-major cell).
Vec3 modulo_cell(const Vec3& d, const double inv[9], const double cell[9])
{
    double f[3];
    for (int i = 0; i < 3; ++i)
        f[i] = d.x * inv[i] + d.y * inv[3 + i] + d.z * inv[6 + i];
    for (int i = 0; i < 3; ++i)
        f[i] -= std::round(f[i]);
    return Vec3{f[0] * cell[0] + f[1] * cell[3] + f[2] * cell[6],
                f[0] * cell[1] + f[1] * cell[4] + f[2] * cell[7],
                f[0] * cell[2] + f[1] * cell[5] + f[2] * cell[8]};
}

// Detect the translation-group cosets of the atomic configuration: a Cartesian
// vector tau is a lattice translation if (R_i + tau) maps every atom onto an
// atom of the same species (modulo the cell).  This generalises the image sum
// from the input cell lattice to the full crystal translation lattice (e.g. the
// bcc body-centre translation), WITHOUT any species-averaging heuristic.
std::vector<Vec3> detect_cosets(const std::vector<Atom>& atoms, const double cell[9],
                                double tol)
{
    std::vector<Vec3> cosets{Vec3{0.0, 0.0, 0.0}};
    double inv[9];
    if (!invert3(cell, inv))
        return cosets;
    auto is_translation = [&](const Vec3& tau) {
        for (std::size_t i = 0; i < atoms.size(); ++i)
        {
            const Vec3 p = atoms[i].R + tau;
            bool found = false;
            for (std::size_t j = 0; j < atoms.size(); ++j)
            {
                if (std::abs(atoms[j].Zval - atoms[i].Zval) > 1e-8)
                    continue;
                if (norm(modulo_cell(p - atoms[j].R, inv, cell)) < tol)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        return true;
    };
    for (std::size_t i = 0; i < atoms.size(); ++i)
        for (std::size_t j = 0; j < atoms.size(); ++j)
        {
            if (std::abs(atoms[j].Zval - atoms[i].Zval) > 1e-8)
                continue;
            const Vec3 tau = modulo_cell(atoms[j].R - atoms[i].R, inv, cell);
            bool dup = false;
            for (const Vec3& t : cosets)
                if (norm(modulo_cell(tau - t, inv, cell)) < tol)
                {
                    dup = true;
                    break;
                }
            if (!dup && is_translation(tau))
                cosets.push_back(tau);
        }
    return cosets;
}

std::vector<Vec3> image_translations(const Grid& grid, int images,
                                     const std::vector<Atom>& atoms)
{
    (void)atoms;
    const Vec3 c0 = grid.n[0] * grid.dv[0];
    const Vec3 c1 = grid.n[1] * grid.dv[1];
    const Vec3 c2 = grid.n[2] * grid.dv[2];
    // Full LATTICE sum (not cosets): a body-centre atom R1 = R0 + g has its
    // pro-density equal to R0's translated by g because 2g is a lattice vector
    // and the periodic pro is invariant, so equivalent atoms are treated
    // identically.  Summing the lattice (not a coset half-lattice) is what
    // restores the symmetry.  R >= 1 always includes the home cell.
    const int R = std::max(0, images);  // images=0 -> no periodic images (large box)
    std::vector<Vec3> T;
    T.reserve((2 * R + 1) * (2 * R + 1) * (2 * R + 1));
    for (int lx = -R; lx <= R; ++lx)
        for (int ly = -R; ly <= R; ++ly)
            for (int lz = -R; lz <= R; ++lz)
                T.push_back(lx * c0 + ly * c1 + lz * c2);
    return T;
}

// Slater-s pro-density summed over periodic images (cutoff applied per image).
double pro_density_at(const Vec3& p, const Vec3& R, double N, double zeta,
                      const std::vector<Vec3>& images, double cutoff)
{
    // Minimum-image distance (nearest periodic image), NOT a truncated lattice
    // sum: the nearest image is the correct *local* MBIS pro-density and is
    // exactly symmetric for equivalent atoms (R1 = R0 + g, 2g a lattice vector).
    // A truncated lattice sum is slowly convergent for a diffuse pro and breaks
    // the symmetry numerically.
    // SMOOTH periodic pro: full lattice sum over images (smooth in R, so its
    // analytic centre derivative matches the finite-difference of the energy).
    // The nearest-image form is discontinuous at wrap boundaries; the sum is not.
    double acc = 0.0;
    for (const Vec3& T : images)
    {
        const double r = norm(p - (R + T));
        if (r <= cutoff) acc += std::exp(-zeta * r);
    }
    if (acc <= 0.0) return 0.0;
    return N * zeta * zeta * zeta / (8.0 * M_PI) * acc;
}

void fill_weights(const std::vector<Vec3>& pts, const Grid& grid,
                  const std::vector<Atom>& atoms, const std::vector<double>& N,
                  const std::vector<double>& zeta, int images, double cutoff,
                  MbisState& st)
{
    const std::size_t nxyz = pts.size();
    const int nat = static_cast<int>(atoms.size());
    const std::vector<Vec3> imgs = image_translations(grid, images, atoms);
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

    // Mean-radius distance uses the SAME image set as the pro-density: direct
    // for images=0 (large box), min-image for images>=1 (small cell).  Equivalent
    // atoms then get identical mean radii / zeta when the periodic images are on.
    const std::vector<Vec3> imgs_r = image_translations(grid, cfg.images, atoms);
    std::vector<std::vector<double>> r(nat, std::vector<double>(nxyz, 0.0));
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            double rr = 1e300;
            for (const Vec3& T : imgs_r) rr = std::min(rr, norm(pts[ig] - atoms[a].R - T));
            r[a][ig] = rr;
        }

    MbisState st;
    st.N.resize(nat);
    st.zeta.assign(nat, cfg.init_zeta);
    const bool have_warm = (static_cast<int>(cfg.warm_N.size()) == nat
                            && static_cast<int>(cfg.warm_zeta.size()) == nat);
    for (int a = 0; a < nat; ++a)
    {
        st.N[a] = have_warm ? cfg.warm_N[a] : atoms[a].Zval;
        st.zeta[a] = have_warm ? cfg.warm_zeta[a] : cfg.init_zeta;
    }

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
            zn[a] = std::max(zn[a], 1e-2);  // exclude the degenerate flat-pro (zeta->0) root
            res = std::max(res, std::abs(na - st.N[a]));
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

    // KL objective D = sum_A int rho_A^pro [ ln rho_A^pro - ln rho_A ] dr,
    // rho_A = w_A rho.  Used to select the lowest-D self-consistent solution.
    {
        const std::vector<Vec3> imgs = image_translations(grid, cfg.images, atoms);
        std::vector<double> pro(nat * nxyz, 0.0);
        std::vector<double> P(nxyz, 0.0);
        for (int a = 0; a < nat; ++a)
            for (std::size_t ig = 0; ig < nxyz; ++ig)
            {
                const double v = pro_density_at(pts[ig], atoms[a].R, st.N[a],
                                                st.zeta[a], imgs, cfg.cutoff);
                pro[a * nxyz + ig] = v;
                P[ig] += v;
            }
        double D = 0.0;
        for (int a = 0; a < nat; ++a)
            for (std::size_t ig = 0; ig < nxyz; ++ig)
            {
                const double pa = pro[a * nxyz + ig];
                if (pa <= 1e-30)
                    continue;
                const double wa = (P[ig] > 1e-300) ? pa / P[ig] : 0.0;
                const double rhoa = wa * rho[ig];
                if (rhoa > 1e-300)
                    D += pa * (std::log(pa) - std::log(rhoa)) * dV;
            }
        st.D = D;
    }
    return st;
}

MbisState compute_mbis_unique(const std::vector<double>& rho, const Grid& grid,
                              const std::vector<Atom>& atoms, const MbisConfig& cfg,
                              int n_starts)
{
    const double scales[] = {0.5, 1.0, 2.0, 4.0};
    const int ns = std::max(1, std::min(n_starts, 4));
    MbisState best;
    best.D = std::numeric_limits<double>::max();
    for (int s = 0; s < ns; ++s)
    {
        MbisConfig c = cfg;
        c.init_zeta = scales[s];
        c.warm_N.clear();
        c.warm_zeta.clear();
        MbisState st = compute_mbis(rho, grid, atoms, c);
        if (st.D < best.D)
            best = st;
    }
    return best;
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

// Minimum-image (periodic) distance between a grid point and an atom centre.
double mbis_periodic_distance(const Vec3& p, const Vec3& R, const Grid& grid)
{
    const Vec3 c0 = grid.n[0] * grid.dv[0];
    const Vec3 c1 = grid.n[1] * grid.dv[1];
    const Vec3 c2 = grid.n[2] * grid.dv[2];
    const double cell[9] = {c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z};
    double inv[9];
    if (!invert3(cell, inv)) return norm(p - R);
    return norm(modulo_cell(p - R, inv, cell));
}

std::vector<Vec3> constraint_force(const MbisState& st, const Grid& grid,
                                   const std::vector<Atom>& atoms,
                                   const std::vector<double>& rho,
                                   const std::vector<double>& spin,
                                   const std::vector<double>& mu_c,
                                   const std::vector<double>& mu_s,
                                   int images)
{
    const std::vector<Vec3> pts = grid_points(grid);
    const std::size_t nxyz = pts.size();
    const int nat = static_cast<int>(atoms.size());
    const double dV = grid.dV();
    // Use the SAME smooth lattice-sum pro as pro_density_at, so the analytic
    // centre derivative is the exact gradient of the finite-difference energy
    // (the nearest-image form is discontinuous at wrap boundaries).
    const std::vector<Vec3> imgs = image_translations(grid, images, atoms);

    std::vector<double> pro(nat * nxyz, 0.0);
    std::vector<double> P(nxyz, 0.0);
    std::vector<std::vector<Vec3>> dpro(nat, std::vector<Vec3>(nxyz, Vec3{}));
    for (int a = 0; a < nat; ++a)
    {
        const double zeta = st.zeta[a];
        const double pre = st.N[a] * zeta * zeta * zeta / (8.0 * M_PI);
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            double pa = 0.0;
            Vec3 da{};
            for (const Vec3& T : imgs)
            {
                // d = x - R - T ; d/dR exp(-zeta|d|) = (zeta/|d|) exp(-zeta|d|) d
                const Vec3 d = pts[ig] - atoms[a].R - T;
                const double ra = norm(d);
                const double e = std::exp(-zeta * ra);
                pa += pre * e;
                if (ra > 1e-12) da = da + (zeta / ra) * pre * e * d;
            }
            pro[a * nxyz + ig] = pa;
            P[ig] += pa;
            dpro[a][ig] = da;
        }
    }

    std::vector<Vec3> F(nat, Vec3{});
    for (std::size_t ig = 0; ig < nxyz; ++ig)
    {
        if (P[ig] <= 1e-300) continue;
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

namespace mbis_alm
{
namespace
{
double kl_D_from_p(const std::vector<double>& p, int nat,
                   const std::vector<Vec3>& pts, const Grid& grid,
                   const std::vector<Atom>& atoms, const std::vector<Vec3>& imgs,
                   double cutoff, const std::vector<double>& rho)
{
    const std::size_t nxyz = pts.size();
    const double dV = grid.dV();
    std::vector<double> pro(nat * nxyz, 0.0), P(nxyz, 0.0);
    for (int a = 0; a < nat; ++a)
    {
        const double N = std::max(p[2 * a], 1e-12);
        const double z = std::max(p[2 * a + 1], 1e-12);
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const double v = pro_density_at(pts[ig], atoms[a].R, N, z, imgs, cutoff);
            pro[a * nxyz + ig] = v;
            P[ig] += v;
        }
    }
    double D = 0.0;
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const double pa = pro[a * nxyz + ig];
            if (pa <= 1e-30) continue;
            const double wa = (P[ig] > 1e-300) ? pa / P[ig] : 0.0;
            const double rhoa = wa * rho[ig];
            if (rhoa > 1e-300) D += pa * (std::log(pa) - std::log(rhoa)) * dV;
        }
    return D;
}

// Cyclic Jacobi eigenvalues of a small symmetric matrix.
std::vector<double> jacobi_eig(std::vector<std::vector<double>> a)
{
    const int n = static_cast<int>(a.size());
    for (int sweep = 0; sweep < 100; ++sweep)
    {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                off += a[i][j] * a[i][j];
        if (off < 1e-24) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
            {
                if (std::abs(a[p][q]) < 1e-18) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0)
                                 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < n; ++k)
                {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
    }
    std::vector<double> eig(n, 0.0);
    for (int i = 0; i < n; ++i) eig[i] = a[i][i];
    return eig;
}

// Jacobi eigenvalues + eigenvectors (V columns) of a small symmetric matrix.
void jacobi_eig_vec(std::vector<std::vector<double>> a, std::vector<double>& eig,
                    std::vector<std::vector<double>>& V)
{
    const int n = static_cast<int>(a.size());
    V.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep)
    {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += a[i][j] * a[i][j];
        if (off < 1e-26) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
            {
                if (std::abs(a[p][q]) < 1e-18) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0)
                    / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                for (int k = 0; k < n; ++k)
                {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - sn * akq;
                    a[k][q] = sn * akp + c * akq;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - sn * aqk;
                    a[q][k] = sn * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - sn * vkq;
                    V[k][q] = sn * vkp + c * vkq;
                }
            }
    }
    eig.assign(n, 0.0);
    for (int i = 0; i < n; ++i) eig[i] = a[i][i];
}

// Solve A x = b for a small dense system (Gaussian elimination, partial pivot).
bool solve_small(std::vector<std::vector<double>> A, std::vector<double> b,
                 std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    for (int i = 0; i < n; ++i)
    {
        int piv = i;
        for (int r = i + 1; r < n; ++r)
            if (std::abs(A[r][i]) > std::abs(A[piv][i])) piv = r;
        if (std::abs(A[piv][i]) < 1e-14) return false;
        std::swap(A[i], A[piv]); std::swap(b[i], b[piv]);
        for (int r = i + 1; r < n; ++r)
        {
            const double f = A[r][i] / A[i][i];
            for (int c = i; c < n; ++c) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double s = b[i];
        for (int c = i + 1; c < n; ++c) s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return true;
}
} // namespace

std::vector<double> mbis_hessian_eigenvalues(const std::vector<double>& rho,
                                             const Grid& grid,
                                             const std::vector<Atom>& atoms,
                                             const MbisConfig& cfg,
                                             const MbisState& state,
                                             double fd_step)
{
    const int nat = static_cast<int>(atoms.size());
    const int n = 2 * nat;
    const std::vector<Vec3> pts = grid_points(grid);
    const std::vector<Vec3> imgs = image_translations(grid, cfg.images, atoms);
    std::vector<double> p(n, 0.0);
    for (int a = 0; a < nat; ++a)
    {
        p[2 * a] = state.N[a];
        p[2 * a + 1] = state.zeta[a];
    }
    auto eval = [&](const std::vector<double>& pp) {
        return kl_D_from_p(pp, nat, pts, grid, atoms, imgs, cfg.cutoff, rho);
    };
    const double D0 = eval(p);
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        const double hi = fd_step * std::max(1.0, std::abs(p[i]));
        for (int j = i; j < n; ++j)
        {
            const double hj = fd_step * std::max(1.0, std::abs(p[j]));
            double v = 0.0;
            if (i == j)
            {
                auto pp1 = p; pp1[i] += hi;
                auto pp2 = p; pp2[i] -= hi;
                v = (eval(pp1) - 2.0 * D0 + eval(pp2)) / (hi * hi);
            }
            else
            {
                auto pp1 = p; pp1[i] += hi; pp1[j] += hj;
                auto pp2 = p; pp2[i] += hi; pp2[j] -= hj;
                auto pp3 = p; pp3[i] -= hi; pp3[j] += hj;
                auto pp4 = p; pp4[i] -= hi; pp4[j] -= hj;
                v = (eval(pp1) - eval(pp2) - eval(pp3) + eval(pp4)) / (4.0 * hi * hj);
            }
            H[i][j] = v;
            H[j][i] = v;
        }
    }
    return jacobi_eig(H);
}
} // namespace mbis_alm

namespace mbis_alm
{
std::string mbis_hessian_classification(const std::vector<double>& eig, double tol)
{
    bool any_neg = false, any_deg = false;
    for (double e : eig)
    {
        if (e < -tol) any_neg = true;
        if (std::abs(e) <= tol) any_deg = true;
    }
    if (any_neg) return "saddle";
    if (any_deg) return "degenerate";
    return "minimum";
}
} // namespace mbis_alm

namespace mbis_alm
{
namespace
{
// One Picard step F(p): N_A = int w_A rho, zeta_A = 3 N_A / int r w_A rho.
std::vector<double> picard_F(const std::vector<double>& p, int nat,
                             const std::vector<Vec3>& pts, const Grid& grid,
                             const std::vector<Atom>& atoms,
                             const std::vector<Vec3>& imgs, double cutoff,
                             const std::vector<double>& rho)
{
    const std::size_t nxyz = pts.size();
    const double dV = grid.dV();
    std::vector<double> pro(nat * nxyz, 0.0), P(nxyz, 0.0);
    for (int a = 0; a < nat; ++a)
    {
        const double N = std::max(p[2 * a], 1e-12);
        const double z = std::max(p[2 * a + 1], 1e-12);
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const double v = pro_density_at(pts[ig], atoms[a].R, N, z, imgs, cutoff);
            pro[a * nxyz + ig] = v;
            P[ig] += v;
        }
    }
    std::vector<double> F(2 * nat, 0.0);
    for (int a = 0; a < nat; ++a)
    {
        double na = 0.0, mr = 0.0;
        for (std::size_t ig = 0; ig < nxyz; ++ig)
        {
            const double wa = (P[ig] > 1e-300) ? pro[a * nxyz + ig] / P[ig] : 0.0;
            na += wa * rho[ig] * dV;
            double rr = 1e300;
            for (const Vec3& T : imgs) rr = std::min(rr, norm(pts[ig] - atoms[a].R - T));
            mr += rr * wa * rho[ig] * dV;
        }
        F[2 * a] = na;
        F[2 * a + 1] = (na > 1e-12 && mr > 1e-12) ? 3.0 * na / mr : p[2 * a + 1];
        F[2 * a + 1] = std::max(F[2 * a + 1], 1e-2);  // exclude zeta->0 degenerate root
    }
    return F;
}

// Solve small dense A x = b (Gaussian elimination, partial pivot).
bool solve_lin(std::vector<std::vector<double>> A, std::vector<double> b,
                 std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    for (int i = 0; i < n; ++i)
    {
        int piv = i;
        for (int r = i + 1; r < n; ++r)
            if (std::abs(A[r][i]) > std::abs(A[piv][i])) piv = r;
        if (std::abs(A[piv][i]) < 1e-14) return false;
        std::swap(A[i], A[piv]); std::swap(b[i], b[piv]);
        for (int r = i + 1; r < n; ++r)
        {
            const double f = A[r][i] / A[i][i];
            for (int c = i; c < n; ++c) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double sm = b[i];
        for (int c = i + 1; c < n; ++c) sm -= A[i][c] * x[c];
        x[i] = sm / A[i][i];
    }
    return true;
}

// FD Jacobian J = dF/dp of the Picard map.
std::vector<std::vector<double>> fp_jac(const std::vector<double>& p, int nat,
                                        const std::vector<Vec3>& pts, const Grid& grid,
                                        const std::vector<Atom>& atoms,
                                        const std::vector<Vec3>& imgs, double cutoff,
                                        const std::vector<double>& rho)
{
    const int n = 2 * nat;
    std::vector<std::vector<double>> J(n, std::vector<double>(n, 0.0));
    for (int j = 0; j < n; ++j)
    {
        const double hj = 1e-5 * std::max(1.0, std::abs(p[j]));
        auto p1 = p; p1[j] += hj;
        auto p2 = p; p2[j] -= hj;
        const auto F1 = picard_F(p1, nat, pts, grid, atoms, imgs, cutoff, rho);
        const auto F2 = picard_F(p2, nat, pts, grid, atoms, imgs, cutoff, rho);
        for (int i = 0; i < n; ++i) J[i][j] = (F1[i] - F2[i]) / (2.0 * hj);
    }
    return J;
}

// Newton on G(p) = F(p) - p = 0 (constrained fixed point; F preserves sum rule).
std::vector<double> newton_fp(std::vector<double> p, int nat,
                              const std::vector<Vec3>& pts, const Grid& grid,
                              const std::vector<Atom>& atoms,
                              const std::vector<Vec3>& imgs, double cutoff,
                              const std::vector<double>& rho, int maxit = 50)
{
    const int n = 2 * nat;
    for (int it = 0; it < maxit; ++it)
    {
        const auto F = picard_F(p, nat, pts, grid, atoms, imgs, cutoff, rho);
        std::vector<double> G(n, 0.0);
        double r = 0.0;
        for (int i = 0; i < n; ++i) { G[i] = F[i] - p[i]; r += G[i] * G[i]; }
        r = std::sqrt(r);
        if (r < 1e-9) break;
        auto J = fp_jac(p, nat, pts, grid, atoms, imgs, cutoff, rho);
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        std::vector<double> rhs(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j) A[i][j] = J[i][j] - (i == j ? 1.0 : 0.0);
            rhs[i] = -G[i];
        }
        std::vector<double> dx;
        // Guard: on a singular/size-mismatched system solve_lin leaves dx empty;
        // fall back to a (damped) gradient step, never index an empty vector.
        if (!solve_lin(A, rhs, dx) || static_cast<int>(dx.size()) != n)
        {
            dx.assign(n, 0.0);
            for (int i = 0; i < n; ++i) dx[i] = -G[i];
        }
        double alpha = 1.0;
        bool ok = false;
        for (int ls = 0; ls < 30; ++ls)
        {
            auto trial = p;
            for (int i = 0; i < n; ++i)
                trial[i] = std::max((i % 2 == 1) ? 5e-2 : 0.0, p[i] + alpha * dx[i]);
            const auto Ft = picard_F(trial, nat, pts, grid, atoms, imgs, cutoff, rho);
            double rt = 0.0;
            for (int i = 0; i < n; ++i) rt += (Ft[i] - trial[i]) * (Ft[i] - trial[i]);
            if (std::sqrt(rt) < r) { p = trial; ok = true; break; }
            alpha *= 0.5;
        }
        if (!ok)
        {
            // Newton direction failed (ill-conditioned FD Jacobian near the
            // fixed point).  Fall back to a Picard step: the Picard map is
            // contractive here (spectral radius < 1), so it still makes
            // progress toward the same fixed point.
            const auto Fp = picard_F(p, nat, pts, grid, atoms, imgs, cutoff, rho);
            double rp = 0.0;
            for (int i = 0; i < n; ++i) rp += (Fp[i] - p[i]) * (Fp[i] - p[i]);
            if (std::sqrt(rp) < r) p = Fp;
            else break;
        }
    }
    return p;
}
} // namespace

MbisState compute_mbis_newton_fp(const std::vector<double>& rho, const Grid& grid,
                                 const std::vector<Atom>& atoms, const MbisConfig& cfg,
                                 int n_starts)
{
    const double scales[] = {0.5, 1.0, 2.0, 4.0};
    const int ns = std::max(1, std::min(n_starts, 4));
    const int nat = static_cast<int>(atoms.size());
    const std::vector<Vec3> pts = grid_points(grid);
    const std::vector<Vec3> imgs = image_translations(grid, cfg.images, atoms);
    std::vector<double> best_p;
    double best_D = std::numeric_limits<double>::max();
    for (int s = 0; s < ns; ++s)
    {
        std::vector<double> p(2 * nat, 0.0);
        for (int a = 0; a < nat; ++a)
        {
            p[2 * a] = atoms[a].Zval;
            p[2 * a + 1] = scales[s];
        }
        const auto pn = newton_fp(p, nat, pts, grid, atoms, imgs, cfg.cutoff, rho);
        const double D = kl_D_from_p(pn, nat, pts, grid, atoms, imgs, cfg.cutoff, rho);
        if (D < best_D) { best_D = D; best_p = pn; }
    }
    // The KL objective is DEGENERATE: a uniform partition (zeta -> 0) makes the
    // pro-density position-independent, w_A = 1/nat, and D = 0 exactly, which is
    // the global minimum.  Hence "lowest D" cannot select a physical solution.
    // Reject degenerate Newton solutions and fall back to the canonical Picard
    // fixed point from a physical start (zeta ~ 1), which is stable and symmetric.
    bool degenerate = (best_D <= 0.0) || (static_cast<int>(best_p.size()) != 2 * nat);
    for (int a = 0; a < nat && !degenerate; ++a)
    {
        // flat-pro root (zeta at/near the clamp) or non-positive population
        if (best_p[2 * a + 1] < 0.05 || best_p[2 * a] <= 0.0) degenerate = true;
    }
    if (degenerate) return compute_mbis(rho, grid, atoms, cfg);

    std::vector<double> N(nat), z(nat);
    for (int a = 0; a < nat; ++a) { N[a] = best_p[2 * a]; z[a] = best_p[2 * a + 1]; }
    MbisState st = weights_from_params(rho, grid, atoms, N, z, cfg.images, cfg.cutoff);
    st.D = best_D;
    return st;
}

double mbis_fixedpoint_spectral_norm(const std::vector<double>& rho, const Grid& grid,
                                     const std::vector<Atom>& atoms,
                                     const MbisConfig& cfg, const MbisState& state)
{
    const int nat = static_cast<int>(atoms.size());
    const std::vector<Vec3> pts = grid_points(grid);
    const std::vector<Vec3> imgs = image_translations(grid, cfg.images, atoms);
    std::vector<double> p(2 * nat, 0.0);
    for (int a = 0; a < nat; ++a) { p[2 * a] = state.N[a]; p[2 * a + 1] = state.zeta[a]; }
    const auto J = fp_jac(p, nat, pts, grid, atoms, imgs, cfg.cutoff, rho);
    const int n = 2 * nat;
    std::vector<std::vector<double>> JtJ(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double v = 0.0;
            for (int k = 0; k < n; ++k) v += J[k][i] * J[k][j];
            JtJ[i][j] = v;
        }
    const auto eig = jacobi_eig(JtJ);
    double mx = 0.0;
    for (double e : eig) mx = std::max(mx, e);
    return std::sqrt(std::max(0.0, mx));
}
} // namespace mbis_alm

namespace mbis_alm
{
namespace
{
// Faddeev-LeVerrier: coefficients c[0..n] of det(lambda I - A)
//   = c[0] lambda^n + c[1] lambda^{n-1} + ... + c[n],  c[0]=1.
std::vector<double> char_poly(const std::vector<std::vector<double>>& A)
{
    const int n = static_cast<int>(A.size());
    std::vector<double> c(n + 1, 0.0);
    c[0] = 1.0;
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) M[i][i] = 1.0;
    for (int k = 1; k <= n; ++k)
    {
        std::vector<std::vector<double>> AM(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                double v = 0.0;
                for (int t = 0; t < n; ++t) v += A[i][t] * M[t][j];
                AM[i][j] = v;
            }
        double tr = 0.0;
        for (int i = 0; i < n; ++i) tr += AM[i][i];
        const double ck = -tr / k;
        c[k] = ck;
        M = AM;
        for (int i = 0; i < n; ++i) M[i][i] += ck;
    }
    return c;
}

// Durand-Kerner roots of the monic polynomial c (c[0]=1).
bool durand_kerner(const std::vector<double>& c,
                   std::vector<std::complex<double>>& roots)
{
    const int n = static_cast<int>(c.size()) - 1;
    auto poly = [&](std::complex<double> z) {
        std::complex<double> r(c[0], 0.0);
        for (int k = 1; k <= n; ++k) r = r * z + std::complex<double>(c[k], 0.0);
        return r;
    };
    const std::complex<double> seed(0.4, 0.9);
    roots.assign(n, std::complex<double>(1.0, 0.0));
    std::complex<double> acc(1.0, 0.0);
    for (int i = 0; i < n; ++i) { acc *= seed; roots[i] = acc; }
    for (int it = 0; it < 500; ++it)
    {
        double maxd = 0.0;
        for (int i = 0; i < n; ++i)
        {
            std::complex<double> den(1.0, 0.0);
            for (int j = 0; j < n; ++j)
                if (j != i) den *= (roots[i] - roots[j]);
            if (std::abs(den) < 1e-300) continue;
            const std::complex<double> dz = poly(roots[i]) / den;
            roots[i] -= dz;
            maxd = std::max(maxd, std::abs(dz));
        }
        if (maxd < 1e-12) return true;
    }
    return false;
}
} // namespace

double mbis_fixedpoint_spectral_radius(const std::vector<double>& rho,
                                       const Grid& grid,
                                       const std::vector<Atom>& atoms,
                                       const MbisConfig& cfg,
                                       const MbisState& state,
                                       bool* converged)
{
    const int nat = static_cast<int>(atoms.size());
    const std::vector<Vec3> pts = grid_points(grid);
    const std::vector<Vec3> imgs = image_translations(grid, cfg.images, atoms);
    std::vector<double> p(2 * nat, 0.0);
    for (int a = 0; a < nat; ++a) { p[2 * a] = state.N[a]; p[2 * a + 1] = state.zeta[a]; }
    const auto J = fp_jac(p, nat, pts, grid, atoms, imgs, cfg.cutoff, rho);
    const auto cp = char_poly(J);
    std::vector<std::complex<double>> roots;
    const bool ok = durand_kerner(cp, roots);
    if (converged) *converged = ok;
    double rho_max = 0.0;
    for (const auto& z : roots) rho_max = std::max(rho_max, std::abs(z));
    return rho_max;
}
} // namespace mbis_alm

namespace mbis_alm
{
int mbis_coset_count(const Grid& grid, const std::vector<Atom>& atoms)
{
    const Vec3 c0 = grid.n[0] * grid.dv[0];
    const Vec3 c1 = grid.n[1] * grid.dv[1];
    const Vec3 c2 = grid.n[2] * grid.dv[2];
    const double cell[9] = {c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z};
    return static_cast<int>(detect_cosets(atoms, cell, 1e-6).size());
}
} // namespace mbis_alm
