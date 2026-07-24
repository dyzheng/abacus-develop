#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_base/constants.h"
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
    double s_dp_escon = 0.0;            // cached dp_escon from last computation
    void* s_hamilt = nullptr;           // stored HamiltPW pointer for inner loop (Phase D.2)
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

double get_deltap_pw_escon()
{
    return s_dp_escon;
}

void set_deltap_pw_active(bool active)
{
    s_active = active;
}

bool is_deltap_pw_active()
{
    return s_active;
}

void set_deltap_pw_hamilt(void* hamilt)
{
    s_hamilt = hamilt;
    s_lambda_set = false;  // re-enable lambda update for new SCF cycle
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

    int gdir = inp.deltap_gdir;
    if (gdir < 1 || gdir > 3) return;

    // Gate: activate only when drho drops below threshold
    if (drho <= 0.0 || drho >= inp.deltap_inner_thr)
        return;

    if (s_lambda_set)
        return;
    s_lambda_set = true;

    std::vector<double> lambda = get_deltap_pw_lambda();
    if (lambda.empty()) lambda.assign(nat, 0.0);
    const std::vector<int>& constrain = get_deltap_pw_constrain();
    const std::vector<double>& targets = get_deltap_pw_targets();

    int nocc = static_cast<int>(inp.nelec / ModuleBase::DEGSPIN);
    if (nocc < 1) nocc = 1;

    double gamma_total = compute_total_gamma_pw(
        ucell, psi_cpu, kv, wfcpw, rhopw, gdir, nocc);

    if (gamma_total == 0.0)
        return;

    // Compute per-atom gamma at lambda=0 (baseline)
    std::vector<double> gamma_baseline(nat, 0.0);
    compute_per_atom_gamma_from_becp(ucell, nocc, gamma_total, gamma_baseline);

    double step = inp.deltap_lambda_step;
    double mixing = inp.deltap_lambda_mixing;
    if (mixing < 0.0) mixing = 0.0;
    if (mixing > 1.0) mixing = 1.0;
    if (mixing == 0.0) mixing = 1.0;

    int inner_nmax = inp.deltap_inner_nmax;
    bool inner_loop_ok = false;
    if (inner_nmax > 0 && s_hamilt != nullptr && psi_cpu != nullptr)
    {
        auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
        if (onsite_p != nullptr)
        {
            int nk = psi_cpu->get_nk();
            int nbands = psi_cpu->get_nbands();
            int npol = psi_cpu->get_npol();
            int nproj = onsite_p->get_tot_nproj();
            if (nk > 0 && nbands > 0 && nproj > 0)
            {
                inner_loop_ok = true;

                // Inner loop: re-compute gamma via becp re-weighting
                // Phase D.1: simple gradient descent (no subspace diag)
                // Phase D.2 (TODO): subspace diag with GEMM + diag_responce
                //   Requires: save H_sub/S_sub/becp (see git history ad25e6be8)
                //             then H_sub(λ) = H_sub(0) + becp†·ps via GEMM
                for (int inner = 0; inner < inner_nmax; inner++)
                {
                    std::vector<double> gamma_trial(nat, 0.0);
                    compute_per_atom_gamma_from_becp(ucell, nocc, gamma_total, gamma_trial);

                    double max_res_inner = 0.0;
                    for (int iat = 0; iat < nat; iat++)
                    {
                        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
                        if (!ok) continue;
                        double residual = gamma_trial[iat] - targets[iat];
                        if (std::abs(residual) > max_res_inner) max_res_inner = std::abs(residual);
                        lambda[iat] = mixing * (lambda[iat] + step * residual) + (1.0 - mixing) * lambda[iat];
                    }

                    if (max_res_inner < inp.deltap_inner_thr)
                        break;
                }
            }
        }
    }
    
    if (!inner_loop_ok)
    {
        // Synchronous mode: single gradient descent step
        double max_res = 0.0;
        for (int iat = 0; iat < nat; iat++)
        {
            bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
            if (!ok) continue;
            double residual = gamma_baseline[iat] - targets[iat];
            if (std::abs(residual) > max_res) max_res = std::abs(residual);
            lambda[iat] = mixing * (lambda[iat] + step * residual) + (1.0 - mixing) * lambda[iat];
        }
    }

    set_deltap_pw_lambda(lambda, constrain);

    // Compute final max_res and per-atom gamma for output
    std::vector<double> gamma_final(nat, 0.0);
    compute_per_atom_gamma_from_becp(ucell, nocc, gamma_total, gamma_final);
    double max_res = 0.0;
    for (int iat = 0; iat < nat; iat++)
    {
        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
        if (ok)
            max_res = std::max(max_res, std::abs(gamma_final[iat] - targets[iat]));
    }

    // Compute dp_escon = -Σ λ_I · γ_I (constraint energy correction)
    // Subtracted from band energy to recover physical DFT energy
    // Follows same formula as DeltaSpin's escon = -Σ λ·M
    double dp_escon = 0.0;
    for (int iat = 0; iat < nat; iat++)
        dp_escon -= lambda[iat] * gamma_final[iat];
    s_dp_escon = dp_escon;

    double lam_avg = 0.0;
    for (int iat = 0; iat < nat; iat++) lam_avg += lambda[iat];
    lam_avg /= nat;

    std::cout << " [DeltaP-PW] drho=" << std::scientific << std::setprecision(3)
              << drho << " γ_total=" << std::fixed << std::setprecision(4)
              << gamma_total << " rad  λ_avg=";
    std::cout << std::scientific << std::setprecision(3) << lam_avg
              << " |res|=" << max_res
              << " escon=" << std::fixed << std::setprecision(6) << dp_escon << " Ry"
              << " γ/atom=(";
    for (int iat = 0; iat < nat; iat++)
    {
        if (iat > 0) std::cout << ", ";
        std::cout << gamma_final[iat];
    }
    std::cout << ")" << std::endl;
}

void compute_per_atom_gamma_from_becp(
    const UnitCell& ucell,
    int nocc,
    double gamma_total,
    std::vector<double>& gamma_per_atom)
{
    int nat = ucell.nat;
    gamma_per_atom.assign(nat, 0.0);

    auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
    if (onsite_p == nullptr) return;

    int tot_nproj = onsite_p->get_tot_nproj();
    if (tot_nproj == 0) return;

    const std::complex<double>* becp = onsite_p->get_becp();
    if (becp == nullptr) return;

    // Compute per-atom weights: w[I] = Σ_n Σ_{α∈I} |<alpha|psi_n>|^2
    std::vector<double> w(nat, 0.0);
    int iproj = 0;
    for (int iat = 0; iat < nat; iat++)
    {
        int nh = onsite_p->get_nh(iat);
        for (int ip = 0; ip < nh; ip++)
        {
            double w_ip = 0.0;
            for (int ib = 0; ib < nocc; ib++)
            {
                std::complex<double> b = becp[ib * tot_nproj + iproj];
                w_ip += b.real() * b.real() + b.imag() * b.imag();
            }
            w[iat] += w_ip;
            iproj++;
        }
    }

    double w_total = 0.0;
    for (int iat = 0; iat < nat; iat++) w_total += w[iat];

    if (w_total < 1e-30)
    {
        // Fallback: equal division
        for (int iat = 0; iat < nat; iat++)
            gamma_per_atom[iat] = gamma_total / nat;
        return;
    }

    for (int iat = 0; iat < nat; iat++)
        gamma_per_atom[iat] = gamma_total * w[iat] / w_total;
}

} // namespace pw_deltap
