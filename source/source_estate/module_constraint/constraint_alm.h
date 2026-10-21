#ifndef CONSTRAINT_ALM_H
#define CONSTRAINT_ALM_H

#include <vector>

namespace constraint
{

/**
 * @brief Augmented-Lagrangian state and pure update math (module D / M4).
 *
 *   E_ALM[rho] = E_KS[rho] + sum_a [mu_a(Q_a - t_a)
 *                                  + (k_a/2)(Q_a - t_a)^2]
 *   v_con(r)   = sum_a (mu_a + k_a(Q_a - t_a)) w_a(r)
 *              = sum_a mu_eff_a w_a(r)
 *
 * The penalty term makes the SCF converge the constrained problem directly
 * (no inject-rebuild cycle); the outer update mu += k(Q - t) contracts
 * |mu - mu*| by 1/|1 - k*chi| per step (design sec. 6.2).  The physical
 * multiplier reported for constant-potential labelling is
 * mu_phys = mu + k(Q - t) (= mu_eff at the converged state).
 */
class AlmController
{
  public:
    void resize(const int n);
    bool armed() const { return armed_; }
    // Arm k from the measured signed responses chi_a = dQ/dmu
    // (negative for charge/spin): k_a = kappa_target / |chi_a|, clamped to
    // [k_min, k_max].  A non-positive or absurd chi falls back to |chi| = 1
    // (k = kappa_target) with the event counted (the caller logs it).
    void arm(const std::vector<double>& chi);
    // Explicit global gain (INPUT constraint_alm_k > 0).  When set, arm()
    // uses this k for every component instead of kappa_target/|chi|; the value is
    // still clamped to [k_min, k_max].  0 restores automatic arming.
    void set_k_override(const double k);
    int arm_fallback_count() const { return arm_fallback_; }
    // mu_eff_a = mu_a + k_a (Q_a - t_a); unarmed -> plain mu (probe phase).
    void mu_eff(const std::vector<double>& mu,
                const std::vector<double>& Q,
                const std::vector<double>& target,
                std::vector<double>& out) const;
    // Damped per-SCF-iteration step for the penalty potential.  The raw map
    // v <- mu + k(Q(v)-t) has linearised derivative k*chi (negative, and
    // |k*chi| = kappa_target by construction), so iterating it explicitly
    // every SCF iteration diverges (R1: drho frozen at 0.194 in a 2-cycle).
    // alpha = 1/(1 + k|chi|) gives derivative 0 at the converged response
    // while leaving the fixed point v = mu + k(Q-t) unchanged.
    double damping(const int a) const;
    // Outer ALM update mu_a += k_a (Q_a - t_a), per-component caps; a
    // component pinned at its cap whose residual has not improved by >= 1%
    // over the last plateau_window updates fuses (UNREACHABLE mirror).
    // Returns max_a |Q_a - t_a| BEFORE the update.
    double update(const std::vector<double>& Q,
                  const std::vector<double>& target,
                  const std::vector<double>& caps,
                  std::vector<double>& mu);
    // sum_a [ mu_a (Q_a - t_a) + (k_a/2) (Q_a - t_a)^2 ].
    double energy_correction(const std::vector<double>& mu,
                             const std::vector<double>& Q,
                             const std::vector<double>& target) const;
    // Stiffness guard: halve every k (floor k_min) on drho growth.
    void dampen();
    int dampen_count() const { return dampen_count_; }
    const std::vector<double>& k() const { return k_; }
    int fuse_component() const { return fuse_component_; }
    double fuse_residual() const { return fuse_residual_; }
    // Reset fuse/plateau bookkeeping at a new geometry (warm rebind); k
    // survives (it is re-armed from the carried response anyway).
    void reset_fuse();

  private:
    std::vector<double> k_;
    // Measured |dQ/dmu| per component (nominal 1.0 on the arm fallback);
    // retained for the potential-step damping only.
    std::vector<double> chi_abs_;
    bool armed_ = false;
    // Auto-gain target k*|chi|.  The design asked for 5-10 (fast implicit
    // ALM outer contraction), but the implemented single-layer loop feeds the
    // penalty potential back once per SCF iteration, where the density lags
    // the potential by (at least) one iteration.  For that lagged map the
    // mode amplitude is sqrt(k|chi|/(1+k|chi|)): 0.94 at 7.5 (R1 never
    // settles inside scf_nmax=100) versus 0.58 at 0.5.  The outer update at
    // SCF convergence still drives the residual to zero; the Newton seed
    // from the probe supplies the starting multiplier.
    double kappa_target_ = 0.5;
    double k_min_ = 0.1;        // Ry/e
    double k_max_ = 100.0;      // Ry/e
    double k_override_ = 0.0;   // INPUT constraint_alm_k; 0 = auto
    int plateau_window_ = 3;
    int arm_fallback_ = 0;
    int dampen_count_ = 0;
    std::vector<std::vector<double>> residual_window_;
    int fuse_component_ = -1;
    double fuse_residual_ = 0.0;
};

} // namespace constraint

#endif
