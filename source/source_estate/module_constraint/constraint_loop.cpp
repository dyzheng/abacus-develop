#include "constraint_loop.h"

#include <cmath>
#include <cstdlib>
#include <numeric>

#include "constraint_deriv.h"
#include "constraint_inject_pw.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"

namespace constraint
{

ConstraintLoop& ConstraintLoop::instance()
{
    static ConstraintLoop loop;
    return loop;
}

void ConstraintLoop::reset()
{
    cfg_ = ConstraintConfig();
    wg_.reset();
    mu_solver_.reset();
    mu_.clear();
    Q_.clear();
    Q_ref_.clear();
    targets_.clear();
    nelec_ = 0.0;
    fixed_mu_ = false;
    phase_ = LoopPhase::IDLE;
    status_ = MuStatus::RUNNING;
    outer_steps_ = 0;
    audit_ = ConstraintAudit();
    last_audit_line_.clear();
}

void ConstraintLoop::init(const UnitCell& ucell,
                          const ModulePW::PW_Basis* rho_basis,
                          const ConstraintConfig& cfg,
                          const std::vector<double>& radii,
                          const double nelec)
{
    reset();
    cfg_ = cfg;
    nelec_ = nelec;
    if (!cfg_.enabled)
    {
        return;
    }
    // Build the shared weight field (M1) once per geometry.  The fragment
    // map comes from the parsed target file (default: one atom per
    // constraint).
    wg_.reset(new WeightGrid(ucell, rho_basis, radii, WeightType::Becke));
    std::vector<std::vector<int>> atom_map;
    for (const ConstraintTarget& t : cfg_.targets)
    {
        atom_map.push_back(t.atoms);
    }
    wg_->set_constraint_atoms(atom_map);
    wg_->build();

    // mu starts at zero: the first SCF is the unconstrained reference.
    mu_.assign(wg_->nconstraint(), 0.0);
    // Experiment branch (default off): freeze the multiplier at a constant
    // read from the environment so the constraint potential acts as an
    // ordinary fixed external potential (no outer secant loop, no reference
    // SCF).  This is the attribution harness for the force FD round: with
    // mu constant, the printed SCF energy slope and the printed force must
    // agree if the energy and force derivations are mutually consistent.
    const char* fixed_mu_env = std::getenv("ABA_CONSTRAINT_FIXED_MU");
    if (fixed_mu_env != nullptr && std::strlen(fixed_mu_env) > 0)
    {
        const double mu_fixed = std::atof(fixed_mu_env);
        std::fill(mu_.begin(), mu_.end(), mu_fixed);
        fixed_mu_ = true;
        targets_.resize(wg_->nconstraint());
        for (size_t a = 0; a < targets_.size(); ++a)
        {
            targets_[a] = cfg_.targets[a].value;
        }
        phase_ = LoopPhase::CONSTRAINED;
        status_ = MuStatus::RUNNING;
        return;
    }
    MuSolverParams params;
    params.mu_max = cfg_.mu_max;
    params.conv_tol = cfg_.thr;
    // Response sign: both channels respond negatively.  For the spin
    // channel the split injection (V_up += mu*w, V_dn -= mu*w) repels
    // spin-up from / attracts spin-down to the fragment for mu > 0, so the
    // magnetization reading m = rho_up - rho_dn decreases with mu — the
    // same negative density response as charge, verified empirically in the
    // 212_PW_constraint_h2o_spin integration case (dQ/dmu ~ -1.4 e/Ry).
    params.response_sign = -1;
    mu_solver_ = MuSolver(params);
    phase_ = LoopPhase::REFERENCE;
    status_ = MuStatus::RUNNING;
    outer_steps_ = 0;
}

void ConstraintLoop::inject_potential(const int iter,
                                      ModuleBase::matrix& v_eff,
                                      ModuleBase::matrix& veff_smooth)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: outer loop finished (CONVERGED / UNREACHABLE): the potential
    // stays as the last injected one — nothing more to add.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Branch B: active loop.  In the reference phase mu is all zero, so the
    // injection is a no-op by value and the first SCF stays unconstrained.
    // The injector rejects a mu/weight length mismatch; per its contract the
    // caller must WARNING_QUIT rather than silently run without the
    // constraint potential (review P2).
    const DensityChannel channel = channel_from_type(cfg_.type);
    if (!ConstraintInjectPW::inject(*wg_, mu_, channel, v_eff)
        || !ConstraintInjectPW::inject(*wg_, mu_, channel, veff_smooth))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential",
            "mu length does not match the constraint count (wiring bug)");
    }
    (void)iter;
}

void ConstraintLoop::inject_potential_lcao(const int iter,
                                           ModuleBase::matrix& v_eff)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: outer loop finished (CONVERGED / UNREACHABLE): the potential
    // stays as the last injected one — nothing more to add.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Branch B: active loop.  In the reference phase mu is all zero, so the
    // injection is a no-op by value and the first SCF stays unconstrained.
    // The injector rejects a mu/weight length mismatch; per its contract the
    // caller must WARNING_QUIT rather than silently run without the
    // constraint potential (review P2).
    const DensityChannel channel = channel_from_type(cfg_.type);
    if (!ConstraintInjectPW::inject(*wg_, mu_, channel, v_eff))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential_lcao",
            "mu length does not match the constraint count (wiring bug)");
    }
    (void)iter;
}

void ConstraintLoop::add_back_constraint_potential(ModuleBase::matrix& veff) const
{
    if (!enabled() || !wg_ || mu_.empty())
    {
        return; // constraint off or not armed: nothing was injected
    }
    // Mirror of the injector (ConstraintInjectPW::inject): the same
    // per-channel sum mu_alpha * w_alpha that entered the vnew snapshot is
    // added back, restoring the physical potential difference for the SCC
    // force integral (see header comment).
    if (!ConstraintInjectPW::inject(*wg_, mu_, channel_from_type(cfg_.type), veff))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::add_back_constraint_potential",
            "mu/veff length mismatch (wiring bug)");
    }
}

void ConstraintLoop::observe(const int iter, const double* const* rho,
                             const int nspin)
{
    if (!enabled())
    {
        return;
    }
    ConstraintObserver::observe(*wg_, rho, nspin,
                                channel_from_type(cfg_.type), Q_);
    (void)iter;
}

void ConstraintLoop::outer_step(const int iter, bool& conv_esolver)
{
    // Experiment branch (fixed mu): the multiplier never changes, so the
    // first genuine SCF convergence closes the run.  No secant step runs;
    // the audit reports the fixed-mu residual Q(mu) - t for the record.
    if (fixed_mu_)
    {
        ++outer_steps_;
        print_audit(iter);
        status_ = MuStatus::CONVERGED;
        phase_ = LoopPhase::DONE;
        return;
    }
    if (phase_ == LoopPhase::REFERENCE)
    {
        // Reference convergence: record Q_ref and build the targets.
        Q_ref_ = Q_;
        targets_.resize(Q_.size());
        for (size_t a = 0; a < Q_.size(); ++a)
        {
            // Branch A: absolute mode — target is the file value directly
            // (calibration scale ~0.2-0.3 e, warned at configure time).
            if (cfg_.target_mode == "absolute")
            {
                targets_[a] = cfg_.targets[a].value;
            }
            else
            {
                // Branch B: delta mode — target is the reference charge plus
                // the requested shift (calibration scale ~e).
                targets_[a] = Q_ref_[a] + cfg_.targets[a].value;
            }
        }
        // The reference point is the mu = 0 observation: run the first
        // secant step immediately.  A delta = 0 target converges here and
        // the run ends; otherwise the SCF continues into the constrained
        // phase with the updated mu.
        phase_ = LoopPhase::CONSTRAINED;
        ++outer_steps_;
        status_ = mu_solver_.step(Q_, targets_, mu_);
        print_audit(iter);
        if (status_ == MuStatus::CONVERGED || status_ == MuStatus::UNREACHABLE)
        {
            phase_ = LoopPhase::DONE;
            return;
        }
        // Keep the SCF running so the new mu is felt by the density.
        conv_esolver = false;
        return;
    }

    // Branch B: constrained phase — advance mu from the observed Q(mu).
    ++outer_steps_;
    status_ = mu_solver_.step(Q_, targets_, mu_);
    print_audit(iter);
    if (status_ == MuStatus::CONVERGED || status_ == MuStatus::UNREACHABLE)
    {
        phase_ = LoopPhase::DONE;
        return;
    }
    conv_esolver = false;
}

void ConstraintLoop::on_scf_converged(const int iter, bool& conv_esolver)
{
    if (!enabled())
    {
        return;
    }
    // Branch A: already finished — nothing to do.
    if (phase_ == LoopPhase::DONE)
    {
        return;
    }
    // Branch B: SCF not converged yet — keep iterating; the observable is
    // only read at genuine SCF convergence (two-stage gating).  Without this
    // gate the outer step would run on the unconverged density and the
    // secant would chase the mixing noise instead of Q(mu).
    if (!conv_esolver)
    {
        return;
    }
    // Branch C: converged SCF with an active outer loop — outer step.
    outer_step(iter, conv_esolver);
}

void ConstraintLoop::print_audit(const int iter)
{
    audit_ = ConstraintAccounting::audit(*wg_, mu_, Q_, targets_, nelec_);
    last_audit_line_ = ConstraintAccounting::audit_line(audit_);
    GlobalV::ofs_running << "\n[constraint] outer step " << outer_steps_
                         << " after SCF iteration " << iter << " (phase="
                         << (phase_ == LoopPhase::CONSTRAINED ? "constrained"
                                                              : "reference")
                         << ")\n"
                         << last_audit_line_ << "\n";
}

void ConstraintLoop::final_report()
{
    if (!enabled() || last_audit_line_.empty())
    {
        return;
    }
    GlobalV::ofs_running << "\n[constraint] final status: ";
    if (fixed_mu_)
    {
        GlobalV::ofs_running
            << "FIXED_MU (experiment switch; multiplier frozen, targets not enforced)\n";
    }
    else if (status_ == MuStatus::CONVERGED)
    {
        GlobalV::ofs_running << "CONVERGED (targets reached within "
                             << cfg_.thr << " e)\n";
    }
    else if (status_ == MuStatus::UNREACHABLE)
    {
        GlobalV::ofs_running << "UNREACHABLE (fused at mu cap "
                             << cfg_.mu_max << " Ry)\n";
    }
    else
    {
        GlobalV::ofs_running << "RUNNING (SCF ended before the outer loop "
                                "converged — raise scf_nmax)\n";
    }
    GlobalV::ofs_running << last_audit_line_ << "\n";
}

double ConstraintLoop::mu_norm() const
{
    return std::accumulate(mu_.begin(), mu_.end(), 0.0,
                           [](const double s, const double v) {
                               return s + std::abs(v);
                           });
}

void ConstraintLoop::compute_force(const double* const* rho,
                                   const int nspin,
                                   ModuleBase::matrix& forcecon)
{
    if (!enabled() || !wg_)
    {
        return; // constraint off: no constraint force
    }
    // Build the position-derivative grid lazily (M6 kernel input); the
    // loop is one-per-geometry, so this happens once per force evaluation.
    if (!wg_->derivatives_built())
    {
        wg_->build_derivatives();
    }
    // Stationary-point guard (envelope-theorem premise, plan section 5.1):
    // the force is the exact derivative of the constrained energy only at
    // the converged outer loop; with a transient mu the residual is
    // O(|Q - t|) and the force must not be silently reported as exact.
    if (mu_norm() > 0.0 && status_ != MuStatus::CONVERGED)
    {
        ModuleBase::WARNING("ConstraintLoop::compute_force",
                            "constraint force from an unconverged outer loop: "
                            "residual O(|Q - t|)");
    }
    constraint::constraint_force(*wg_, rho, nspin,
                                 channel_from_type(cfg_.type), mu_, forcecon);
}

} // namespace constraint
