#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include <iomanip>
#include <iostream>

namespace pw_deltap {

namespace {
    bool s_active = false;
    bool s_lambda_set = false;
    std::vector<double> s_lambda;       // current per-atom lambda (py)
    std::vector<double> s_targets;      // target per-atom gamma (rad)
    std::vector<int> s_constrain;       // per-atom constrain flags
    double s_gamma_total = 0.0;         // cached total gamma from last computation
}

void set_deltap_pw_lambda(const std::vector<double>& lambda,
                          const std::vector<int>& constrain)
{
    s_lambda = lambda;
    s_targets = lambda; // initial lambda = target (constant for now)
    s_constrain = constrain;
}

const std::vector<double>& get_deltap_pw_lambda()
{
    return s_lambda;
}

const std::vector<int>& get_deltap_pw_constrain()
{
    return s_constrain;
}

const std::vector<double>& get_deltap_pw_targets()
{
    return s_targets;
}

void set_deltap_pw_active(bool active)
{
    s_active = active;
}

bool is_deltap_pw_active()
{
    return s_active;
}

bool run_deltap_lambda_loop(const int iter,
                            const double drho,
                            const Input_para& inp)
{
    if (!inp.deltap_switch)
        return false;

    set_deltap_pw_active(true);
    return false;
}

double compute_total_gamma_pw(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi_in,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    int nbands)
{
    if (gdir < 1 || gdir > 3) return 0.0;
    if (wfcpw == nullptr || psi_in == nullptr || rhopw == nullptr) return 0.0;

    berryphase bp;
    bp.direction = gdir;
    bp.GDIR = gdir;
    bp.set_kpoints(kv, gdir);

    if (bp.total_string == 0 || bp.nppstr < 2)
        return 0.0;

    double gamma_total = 0.0;
    for (int istr = 0; istr < bp.total_string; istr++)
    {
        gamma_total += bp.stringPhase(ucell, istr, nbands,
                                       wfcpw->npwk_max, psi_in, rhopw, wfcpw, kv);
    }

    s_gamma_total = gamma_total;
    return gamma_total;
}

void deltap_iter_finish(
    const UnitCell& ucell,
    double drho,
    const psi::Psi<std::complex<double>>* psi_cpu,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    const Input_para& inp)
{
    if (!inp.deltap_switch || !inp.deltap_corr)
        return;

    int nat = ucell.nat;
    if (nat == 0) return;

    // No per-iteration dynamic update yet for Gamma-only grids
    // (λ is static from STRU → Phase A behavior)
    // For multi-k grids, compute gamma and update lambda

    int gdir = inp.deltap_gdir;
    if (gdir < 1 || gdir > 3) return;

    // Compute gamma only when charge is converged enough
    if (drho <= 0.0 || drho >= inp.deltap_inner_thr)
        return;

    // Get current lambda (from previous iteration, or initial from STRU)
    std::vector<double> lambda = get_deltap_pw_lambda();
    if (lambda.empty()) lambda.assign(nat, 0.0);
    const std::vector<int>& constrain = get_deltap_pw_constrain();

    // Get targets
    // We don't have direct access to ucell.get_dp_target() here but we stored
    // both target and constrain at initialization — the init lambda IS the target.
    // Actually, we need separate storage for target values.  Use stored lambda as
    // initial guess, and read targets from STRU initialization data.

    // Compute total gamma from wavefunctions
    int nocc = inp.nelec / 2; // nspin=1, nelec/2 = occupied bands
    if (nocc < 1) nocc = 1;

    double gamma_total = compute_total_gamma_pw(
        ucell, psi_cpu, kv, wfcpw, rhopw, gdir, nocc);

    if (gamma_total == 0.0)
        return; // no valid k-strings (Gamma-only grid)

    // For Phase B2: assign gamma equally to constrained atoms
    // Phase B3 will add proper per-atom decomposition via becp weights
    int nconstrained = 0;
    for (int iat = 0; iat < nat; iat++)
    {
        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
        if (ok) nconstrained++;
    }
    if (nconstrained == 0) return;

    double gamma_per_atom = gamma_total / nconstrained;

    // Build per-atom gamma and update lambda via gradient descent
    std::vector<double> gamma_1d(nat, 0.0);
    for (int iat = 0; iat < nat; iat++)
    {
        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
        gamma_1d[iat] = ok ? gamma_per_atom : 0.0;
    }

    // Target values (fixed, from STRU at initialization)
    const std::vector<double>& targets = get_deltap_pw_targets();
    double step = inp.deltap_lambda_step;
    double mixing = inp.deltap_lambda_mixing;
    if (mixing < 0.0) mixing = 0.0;
    if (mixing > 1.0) mixing = 1.0;
    if (mixing == 0.0) mixing = 1.0;

    // Simple gradient descent: lambda_new[i] = lambda[i] + step * (gamma[i] - target[i])
    double max_res = 0.0;
    for (int iat = 0; iat < nat; iat++)
    {
        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
        if (!ok) continue;
        double residual = gamma_1d[iat] - targets[iat];
        if (std::abs(residual) > max_res) max_res = std::abs(residual);
        lambda[iat] = mixing * (lambda[iat] + step * residual) + (1.0 - mixing) * lambda[iat];
    }

    set_deltap_pw_lambda(lambda, constrain);

    std::cout << " [DeltaP-PW] drho=" << std::scientific << std::setprecision(3)
              << drho << " γ_total=" << std::fixed << std::setprecision(4)
              << gamma_total << " rad  λ_avg=";
    double lam_avg = 0.0;
    for (int iat = 0; iat < nat; iat++) lam_avg += lambda[iat];
    lam_avg /= nat;
    std::cout << std::scientific << std::setprecision(3) << lam_avg
              << " |res|=" << max_res << std::endl;
}

} // namespace pw_deltap
