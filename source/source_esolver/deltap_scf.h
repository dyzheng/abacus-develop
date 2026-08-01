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

    /// Run the frozen-density inner loop (nscf > 0). Returns true if the
    /// regular HSolver step must be skipped (the inner loop already solved).
    bool inner_loop(double drho);

    /// Per-SCF-iteration constraint update: measure γ, update λ (synchronous
    /// mode), recompute escon / HK correction, report diagnostics.
    void iter_finish(int iter, double drho);

    const DeltapState& state() const { return state_; }
    const DeltapParams& params() const { return params_; }

  private:
    bool use_constraint_matrix() const { return !params_.C.empty(); }
    void update_lambda_gd(int iter, double drho);
    void report(int iter, const std::vector<double>& lambda) const;

    DeltapParams params_;
    DeltapState state_;
    Backend backend_;
};

} // namespace deltap_scf
#endif // DELTAP_SCF_H
