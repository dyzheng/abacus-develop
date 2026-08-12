#ifndef DELTAP_SCF_H
#define DELTAP_SCF_H
/**
 * @file deltap_scf.h
 * @brief Basis-independent DeltaP SCF constraint state machine.
 *
 * Owns all mutable DeltaP SCF state (lambda / gamma / phase flags) and the
 * P1→P2→P3 two-phase control flow plus the frozen-density inner loop.
 * Basis-specific operations (Hamiltonian application, gamma measurement,
 * HSolver) are injected by the ESolver via Backend callbacks, so the same
 * state machine drives both LCAO and PW.
 */

#include <functional>
#include <string>
#include <vector>

namespace ModuleOptimizer
{
class FletcherReevesCG;
}

namespace deltap_scf
{

/// Basis-independent DeltaP SCF parameters (snapshot of INPUT + targets).
struct DeltapParams
{
    int nat = 0;                    ///< number of atoms
    int gdir = 3;                   ///< constraint direction (1=x, 2=y, 3=z)
    double inner_thr = 1.0e-3;      ///< drho gate for P2 / inner loop
    double lambda_step = 0.01;      ///< gradient-descent step (Ry)
    double lambda_mixing = 0.1;     ///< damping factor
    double lambda_init = 0.0;       ///< initial λ
    double conv_thr = 1.0e-6;       ///< inner-loop convergence threshold
    int nscf = 0;                   ///< inner-loop steps (0 = synchronous two-phase)
    bool total_mode = false;        ///< deltap_constraint_mode == "total"
    bool verbose = true;            ///< print [DeltaP] diagnostics
    bool unwrap_branch_2pi = false; ///< PW-style cross-SCF 2π branch tracking
    std::string target_file;             ///< per-atom γ target file (empty = none)
    std::string constraint_matrix_file;  ///< C·γ = t file (empty = none)
    std::vector<double> target;          ///< per-atom γ targets [nat]
    std::vector<int> constrain;          ///< per-atom constrain flags [nat]
    std::vector<std::vector<double>> C;  ///< constraint matrix (m×nat, empty = off)
    std::vector<double> t;               ///< constraint targets [m]
    std::string observable_mode = "operator";  ///< SCF constraint variable:
    ///< "operator" = Γ (Route A+, default), "gamma" = legacy Wilson-loop γ.
    /// Route A+ λ-driving signal (operator mode only): "proxy" (default) =
    /// the λ residual drives Γ against the t_Γ proxy target (calibrated by
    /// the outer-loop secant); "gamma" = the λ residual drives the reported
    /// γ directly against the user's t_γ — the t_Γ/secant translation layer
    /// is retired.  The escon accounting (escon = −λ·Γ) and the force
    /// consistency E' = E_KS(ψ*) are properties of the ACCOUNTING, not of
    /// the driving signal, so γ-drive preserves the O(λ) leakage structure
    /// (2026-08-11 review, T-4').  Legacy gamma mode is locked to "gamma".
    std::string drive = "proxy";
    /// Route A+ constraint operator (operator mode only): "proxy" = the
    /// τ_α·P̂ geometric proxy (H_HR, historical Route A+); "ow" = the exact
    /// weight-channel operator Ô_w = θ_n·P̂ (band-resolved Wilson phase θ_n,
    /// EFC L3.1, T-6').  Legacy gamma mode is locked to "proxy".
    std::string operator_mode = "proxy";
    bool secant_at_convergence = false; ///< single-point runs: one t_Γ secant
    ///< update in iter_finish after SCF convergence (relax uses
    ///< reset_ionic_step instead).
    bool secant_enabled = true;     ///< master switch for the t_Γ outer-loop
    ///< secant ("deltap_secant"; off = freeze t_Γ, e.g. T3 disp± legs).
    std::string proxy_target_file = ""; ///< file with per-atom t_Γ values
    ///< (overrides the t_Γ=t_γ first-round init; used to freeze the
    ///< calibrated t_Γ* across geometries for T3).
    int outer_nmax = 0;             ///< Route A+ fixed-geometry outer-loop steps
    ///< (scf + deltap_outer_nmax > 0: re-drive SCF after each t_Γ secant update
    ///< until |γ−t_γ|∞ ≤ outer_thr or the step budget is exhausted; 0 = legacy
    ///< single-fire secant at convergence).
    double outer_thr = 1.0e-2;      ///< outer-loop |γ−t_γ|∞ convergence (rad)
};

/// Mutable SCF state, fully owned by DeltapScfSolver.
struct DeltapState
{
    bool initialized = false;
    bool lambda_set = false;         ///< P2 λ updated and frozen
    bool inner_loop_done = false;    ///< inner loop converged once
    std::vector<double> lambda_eff;  ///< effective per-atom λ [nat]
    std::vector<double> lambda_cstr; ///< constraint-space λ [m] (per-atom: same as λ_eff)
    std::vector<double> gamma_I;     ///< latest per-atom γ (folded to gdir) [nat]
    std::vector<double> gamma_report;///< branch-selected γ for escon/report [nat]
    std::vector<double> gamma_prev;  ///< previous branch-selected γ (2π unwrap)
    /// Latest per-atom Γ (Route A+ operator observable) [nat], measured at
    /// the same wavefunctions as gamma_I (operator mode only).
    std::vector<double> gamma_op;
    /// Proxy target t_Γ [nat] for operator mode: initialized to the user's
    /// t_γ (κ=1 first round) and updated by the outer-loop secant
    /// (reset_ionic_step / post-convergence single point).
    std::vector<double> t_proxy;
    /// Secant history: previous measured γ and previous t_Γ (outer loop).
    std::vector<double> gamma_meas_prev;
    std::vector<double> t_proxy_prev;
    double secant_prev_err = -1.0; ///< previous |γ−t_γ|∞ for divergence guard
    int secant_bad_steps = 0;      ///< consecutive |γ−t_γ| increases
    bool secant_at_conv_done = false; ///< single-point secant fired this SCF
    /// Fixed-geometry outer loop (scf + outer_nmax > 0) state.
    bool first_pass_done = false;  ///< first (free λ=0) SCF measured natural Γ/γ
    int outer_steps = 0;           ///< outer secant updates applied
    double outer_err = -1.0;       ///< last |γ−t_γ|∞ (outer-loop convergence)
    bool outer_redrive = false;    ///< request the SCF loop to continue (new t_Γ)
    double max_res = 0.0;
    double dp_escon = 0.0;
};

class DeltapScfSolver
{
  public:
    /// Basis-specific operations injected by the ESolver.
    struct Backend
    {
        std::function<void(const std::vector<double>&)> set_lambda;           ///< operator / onsite_proj
        std::function<std::vector<double>()> get_lambda;                      ///< current operator λ
        std::function<void(const std::vector<double>&)> apply_hk_correction;  ///< recompute + set HK (LCAO)
        std::function<std::vector<double>()> compute_gamma;                   ///< folded per-atom γ [nat]
        std::function<std::vector<double>()> compute_gamma_op;                ///< per-atom Γ (operator observable) [nat]
        std::function<void()> solve_frozen;                                   ///< HSolver(skip_charge=true)
        std::function<void(std::vector<double>&)> sync_lambda;                ///< MPI Bcast (optional)
        std::function<void()> on_phase2;                                      ///< cooldown + mix_reset (optional)
        std::function<ModuleOptimizer::FletcherReevesCG&()> get_optimizer;    ///< inner-loop optimizer (optional)
        // Optional verbose diagnostics
        std::function<std::vector<double>()> compute_gamma_raw;               ///< pre-branch raw γ [nat]
        std::function<double()> lattice_period;                               ///< a_alpha in Bohr
    };

    void init(const DeltapParams& p, Backend b);
    void reset_ionic_step();
    /// Consume the fixed-geometry outer-loop re-drive request (called by the
    /// ESolver after iter_finish to keep the SCF loop running with the new t_Γ).
    bool consume_outer_redrive();

    /// Run the frozen-density inner loop (nscf > 0). Returns true if the
    /// regular HSolver step must be skipped (the inner loop already solved).
    bool inner_loop(double drho);

    /// Per-SCF-iteration constraint update: measure γ, update λ (synchronous
    /// mode), recompute escon / HK correction, report diagnostics.
    void iter_finish(int iter, double drho, bool conv_esolver = false);

    const DeltapState& state() const { return state_; }
    const DeltapParams& params() const { return params_; }

  private:
    bool use_constraint_matrix() const { return !params_.C.empty(); }
    /// Apply λ to the operator and (if present) broadcast it across ranks so
    /// every rank's operator / escon uses rank 0's value.  Single sync point
    /// for both the synchronous update and the inner-loop trials.
    void apply_lambda(const std::vector<double>& lambda);
    void update_lambda_gd(int iter, double drho);
    /// Route A+ outer-loop secant: update the proxy target t_Γ so that the
    /// measured γ converges to the user's t_γ (derivations §7, D7).
    void secant_update_proxy();
    void report(int iter, const std::vector<double>& lambda) const;

    DeltapParams params_;
    DeltapState state_;
    Backend backend_;
};

} // namespace deltap_scf
#endif // DELTAP_SCF_H
