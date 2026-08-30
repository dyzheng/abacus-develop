#include "constraint_loop.h"

#include <cmath>
#include <numeric>

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
    MuSolverParams params;
    params.mu_max = cfg_.mu_max;
    params.conv_tol = cfg_.thr;
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
    if (!ConstraintInjectPW::inject(*wg_, mu_, v_eff)
        || !ConstraintInjectPW::inject(*wg_, mu_, veff_smooth))
    {
        ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential",
            "mu length does not match the constraint count (wiring bug)");
    }
    (void)iter;
}

void ConstraintLoop::observe(const int iter, const double* const* rho,
                             const int nspin)
{
    if (!enabled())
    {
        return;
    }
    ConstraintObserver::observe(*wg_, rho, nspin, Q_);
    (void)iter;
}

void ConstraintLoop::outer_step(const int iter, bool& conv_esolver)
{
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
    if (status_ == MuStatus::CONVERGED)
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

} // namespace constraint
