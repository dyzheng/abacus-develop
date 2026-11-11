#ifndef MODULE_MBIS_ALM_H
#define MODULE_MBIS_ALM_H

/**
 * @file mbis_alm.h
 * @brief Standalone MBIS (iterative Slater stockholder) weight + augmented
 *        Lagrangian charge/spin constraint core.
 *
 * This module is deliberately independent of the ABACUS constraint/allocation
 * implementation: it only depends on the C++ standard library and works on
 * plain std::vector grids, so it can be unit-tested and compared with other
 * schemes (Mulliken / onsite projector / Hirshfeld) in isolation.
 *
 * Physics (see docs/superpowers/specs/
 * 2026-10-30-mbis-charge-spin-constraint-algorithm.md):
 *   rho_A^pro(r) = N_A zeta_A^3/(8 pi) exp(-zeta_A |r-R_A|)
 *   w_A(r)       = rho_A^pro / sum_B rho_B^pro
 *   N_A = int w_A rho,  M_A = int w_A (rho_up - rho_dn)
 *   E_ALM = E_KS + sum_a [ mu_a (Q_a - t_a) + (k_a/2)(Q_a - t_a)^2 ]
 *   v_con = sum_a (mu_a + k_a (Q_a - t_a)) w_a = sum_a mu_eff_a w_a
 */

#include <array>
#include <cmath>
#include <string>
#include <cstddef>
#include <vector>

namespace mbis_alm
{

struct Vec3
{
    double x = 0.0, y = 0.0, z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double s, const Vec3& a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

/// One atom: position (Bohr) and valence electron count (pro-density init).
struct Atom
{
    Vec3 R;
    double Zval = 0.0;
};

/// Uniform grid described by origin + three voxel vectors (rows); may be skewed.
struct Grid
{
    std::array<int, 3> n{{0, 0, 0}};
    Vec3 origin;
    std::array<Vec3, 3> dv{{}};

    std::size_t size() const;
    double dV() const;  ///< |det(dv)|
    Vec3 point(int ix, int iy, int iz) const;
};

struct MbisConfig
{
    double cutoff = 6.0;    ///< radial cutoff (Bohr) for the pro-density
    int max_iter = 300;
    double tol = 1e-9;
    double damp = 0.5;
    int images = 0;         ///< 0: direct atoms; N: +/- N periodic cells
    double init_zeta = 1.5;
    bool symmetrize = false; ///< average N,zeta over atoms of the same species
    std::vector<double> warm_N;    ///< optional per-atom warm-start populations
    std::vector<double> warm_zeta; ///< optional per-atom warm-start exponents
};

/// MBIS state: per-atom weights on the grid, populations and Slater exponents.
struct MbisState
{
    std::vector<std::vector<double>> w;  ///< [nat][grid]
    std::vector<double> N;               ///< electron populations
    std::vector<double> zeta;            ///< Slater exponents (Bohr^-1)
    double D = 0.0;                      ///< KL objective at the solution (for global-min selection)
};

/// Run the MBIS KL fixed point (population + mean-radius update).
MbisState compute_mbis(const std::vector<double>& rho, const Grid& grid,
                       const std::vector<Atom>& atoms, const MbisConfig& cfg);

/**
 * @brief Multi-start global-min selection: run the MBIS KL fixed point from
 * several initial widths and return the solution with the lowest KL objective
 * D.  This addresses the non-convexity / multiple self-consistent solutions
 * by selecting the lowest-D basin (the MBIS-defined minimum).
 */
MbisState compute_mbis_unique(const std::vector<double>& rho, const Grid& grid,
                              const std::vector<Atom>& atoms, const MbisConfig& cfg,
                              int n_starts = 4);

/**
 * @brief Finite-difference Hessian eigenvalues of the MBIS KL objective D(p)
 * (p = {N_A, zeta_A}) at a given state.  Used to certify a local minimum
 * (all eigenvalues > 0) versus a saddle (any eigenvalue < 0).  Small systems.
 */
std::vector<double> mbis_hessian_eigenvalues(const std::vector<double>& rho,
                                             const Grid& grid,
                                             const std::vector<Atom>& atoms,
                                             const MbisConfig& cfg,
                                             const MbisState& state,
                                             double fd_step = 1e-4);

/// Classify Hessian eigenvalues: "minimum" (all > tol), "saddle" (any < -tol),
/// "degenerate" (otherwise).  Certifies the MBIS solution kind.
std::string mbis_hessian_classification(const std::vector<double>& eig,
                                        double tol = 1e-6);

/// Frozen-parameter weights (for force finite-difference tests / warm starts).
MbisState weights_from_params(const std::vector<double>& rho, const Grid& grid,
                              const std::vector<Atom>& atoms,
                              const std::vector<double>& N,
                              const std::vector<double>& zeta,
                              int images, double cutoff);

struct Observables
{
    std::vector<double> N;  ///< charge (electrons)
    std::vector<double> M;  ///< spin (up - down)
};

Observables observe(const MbisState& st, const std::vector<double>& rho,
                    const std::vector<double>& spin, const Grid& grid);

/// Constraint local potential V[ig] = sum_A coef[A] * w[A][ig].
std::vector<double> constraint_potential(const MbisState& st,
                                         const std::vector<double>& coef,
                                         const Grid& grid);

/**
 * @brief Constraint force F_J = sum_A int (d w_A / dR_J) (mu_c[A] rho + mu_s[A] spin).
 *
 * Uses the frozen-parameter (MBIS zeta,N held fixed) weight derivative
 *   d w_A/dR_J = (delta_{AJ} - w_A) (d rho_J^pro/dR_J) / P,
 *   d rho_J^pro/dR_J = zeta_J (r-R_J)/|r-R_J| rho_J^pro.
 */
std::vector<Vec3> constraint_force(const MbisState& st, const Grid& grid,
                                   const std::vector<Atom>& atoms,
                                   const std::vector<double>& rho,
                                   const std::vector<double>& spin,
                                   const std::vector<double>& mu_c,
                                   const std::vector<double>& mu_s);

/// One augmented-Lagrangian component (a charge or spin constraint).
struct AlmComponent
{
    double target = 0.0;
    double mu = 0.0;
    double k = 0.0;
    double chi = -1.0;
    bool armed = false;
};

/**
 * @brief Standalone augmented-Lagrangian multiplier controller.
 *
 * E_ALM = E_KS + sum_a [ mu_a (Q_a - t_a) + (k_a/2)(Q_a - t_a)^2 ]
 * mu_eff,a = mu_a + k_a (Q_a - t_a)
 * update:  mu_a += k_a (Q_a - t_a)         (exactness / bias removal)
 * damping: alpha_a = 1/(1 + k_a |chi_a|)    (use for the in-SCF potential)
 */
class Alm
{
  public:
    void resize(int n);
    int size() const { return static_cast<int>(c_.size()); }
    AlmComponent& comp(int i) { return c_[i]; }
    const AlmComponent& comp(int i) const { return c_[i]; }

    void set_target(int i, double t) { c_[i].target = t; }
    void set_mu(int i, double mu) { c_[i].mu = mu; }
    void set_chi(int i, double chi) { c_[i].chi = chi; }

    /// Arm k = kappa/|chi| clamped to [kmin,kmax]; zero chi -> kappa.
    void arm(const std::vector<double>& chi, double kappa = 1.0,
             double kmin = 0.1, double kmax = 100.0);

    double mu_eff(int i, double Q) const { return c_[i].mu + c_[i].k * (Q - c_[i].target); }
    std::vector<double> mu_eff(const std::vector<double>& Q) const;
    double damping(int i) const;

    /// Outer update mu += k (Q - target); returns max |Delta mu| (capped).
    double update(const std::vector<double>& Q, double cap = 1e9);

    /// E_ALM - E_KS for the current Q (zero at the fixed point).
    double energy_correction(const std::vector<double>& Q) const;

  private:
    std::vector<AlmComponent> c_;
};

} // namespace mbis_alm

#endif
