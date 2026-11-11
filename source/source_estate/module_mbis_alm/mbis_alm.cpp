#include "mbis_alm.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
    const Vec3 c0 = grid.n[0] * grid.dv[0];
    const Vec3 c1 = grid.n[1] * grid.dv[1];
    const Vec3 c2 = grid.n[2] * grid.dv[2];
    const double cell[9] = {c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z};
    const std::vector<Vec3> cosets = detect_cosets(atoms, cell, 1e-6);
    const int R = std::max(0, images);
    std::vector<Vec3> T;
    for (int lx = -R; lx <= R; ++lx)
        for (int ly = -R; ly <= R; ++ly)
            for (int lz = -R; lz <= R; ++lz)
                for (const Vec3& tau : cosets)
                    T.push_back(lx * c0 + ly * c1 + lz * c2 + tau);
    if (T.empty())
        T.push_back(Vec3{0.0, 0.0, 0.0});
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

    // direct distances (used for the mean-radius update)
    std::vector<std::vector<double>> r(nat, std::vector<double>(nxyz, 0.0));
    for (int a = 0; a < nat; ++a)
        for (std::size_t ig = 0; ig < nxyz; ++ig)
            r[a][ig] = norm(pts[ig] - atoms[a].R);

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
