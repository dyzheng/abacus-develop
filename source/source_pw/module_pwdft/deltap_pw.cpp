#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_hamilt/hamilt.h"
#include "source_base/constants.h"
#include "source_base/kernels/math_kernel_op.h"
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
    
    // Subspace data for inner lambda loop (saved once per SCF, reused across inner steps)
    bool s_sub_saved = false;
    std::vector<std::complex<double>> s_sub_h;     // H_sub per k-point [nk*nbands*nbands]
    std::vector<std::complex<double>> s_sub_s;     // S_sub per k-point
    std::vector<std::complex<double>> s_becp;      // becp per k-point [nk*nproj*nbands*npol]
    int s_nk = 0, s_nbands = 0, s_nproj = 0, s_npol = 0;
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
    hamilt::Hamilt<std::complex<double>>* p_hamilt,
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

    // ---- Subspace inner loop (only if deltap_inner_nmax > 0) ----
    int inner_nmax = inp.deltap_inner_nmax;
    bool inner_loop_ok = false;
    if (inner_nmax > 0 && p_hamilt != nullptr && psi_cpu != nullptr)
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
                const int* nh_iat = &onsite_p->get_nh(0);

                // Save subspace data (once per SCF)
                if (!s_sub_saved)
                {
                    s_nk = nk; s_nbands = nbands; s_nproj = nproj; s_npol = npol;
                    s_sub_h.resize(nk * nbands * nbands);
                    s_sub_s.resize(nk * nbands * nbands);
                    int size_becp = nbands * nproj * npol;
                    s_becp.resize(nk * size_becp);

                    auto* psi_nc = const_cast<psi::Psi<std::complex<double>>*>(psi_cpu);
                    for (int ik = 0; ik < nk; ik++)
                    {
                        psi_nc->fix_k(ik);
                        auto* h_k = s_sub_h.data() + ik * nbands * nbands;
                        auto* s_k = s_sub_s.data() + ik * nbands * nbands;
                        auto* becp_k = s_becp.data() + ik * size_becp;
                        p_hamilt->updateHk(ik);
                        hsolver::DiagoIterAssist<std::complex<double>>::cal_hs_subspace(
                            p_hamilt, *psi_nc, h_k, s_k);
                        memcpy(becp_k, onsite_p->get_becp(),
                            sizeof(std::complex<double>) * size_becp);
                    }
                    s_sub_saved = true;
                }

                // Inner loop: gradient descent with subspace re-solve
                for (int inner = 0; inner < inner_nmax; inner++)
                {
                    int size_becp = s_nbands * s_nproj * s_npol;
                    std::vector<std::complex<double>> h_tmp(s_nbands * s_nbands);
                    std::vector<std::complex<double>> s_tmp(s_nbands * s_nbands);
                    std::vector<std::complex<double>> becp_tmp(size_becp);
                    std::vector<std::complex<double>> ps(size_becp, 0.0);
                    std::vector<double> w_tot(nat, 0.0);

                    for (int ik = 0; ik < s_nk; ik++)
                    {
                        auto* h_k = s_sub_h.data() + ik * s_nbands * s_nbands;
                        auto* s_k = s_sub_s.data() + ik * s_nbands * s_nbands;
                        auto* becp_k = s_becp.data() + ik * size_becp;

                        // Build ps = diag(lambda[atom_of(proj)]) * becp
                        std::fill(ps.begin(), ps.end(), std::complex<double>(0.0, 0.0));
                        int iproj = 0;
                        for (int iat = 0; iat < nat; iat++)
                        {
                            int nh = nh_iat[iat];
                            std::complex<double> coeff(lambda[iat], 0.0);
                            for (int ip = 0; ip < nh; ip++)
                            {
                                for (int ib = 0; ib < s_nbands; ib++)
                                    ps[ib * s_nproj + iproj] += coeff * becp_k[ib * s_nproj + iproj];
                                iproj++;
                            }
                        }

                        // H_sub(lambda) = H_sub(0) + becp† * ps
                        memcpy(h_tmp.data(), h_k, sizeof(std::complex<double>) * s_nbands * s_nbands);
                        memcpy(s_tmp.data(), s_k, sizeof(std::complex<double>) * s_nbands * s_nbands);
                        memcpy(becp_tmp.data(), becp_k, sizeof(std::complex<double>) * size_becp);

                        ModuleBase::gemm_op<std::complex<double>, base_device::DEVICE_CPU>()(
                            'C', 'N', s_nbands, s_nbands, s_nproj * s_npol,
                            &ModuleBase::ONE, becp_k, s_nproj * s_npol,
                            ps.data(), s_nproj * s_npol,
                            &ModuleBase::ONE, h_tmp.data(), s_nbands);

                        // Subspace diagonalization: H·V = S·V·E
                        hsolver::DiagoIterAssist<std::complex<double>>::diag_responce(
                            h_tmp.data(), s_tmp.data(), s_nbands,
                            becp_tmp.data(), becp_tmp.data(), s_nproj * s_npol, nullptr);

                        // Compute per-atom weights from rotated becp
                        iproj = 0;
                        for (int iat = 0; iat < nat; iat++)
                        {
                            int nh = nh_iat[iat];
                            for (int ip = 0; ip < nh; ip++)
                            {
                                for (int ib = 0; ib < nocc; ib++)
                                {
                                    auto b = becp_tmp[ib * s_nproj + iproj];
                                    w_tot[iat] += b.real() * b.real() + b.imag() * b.imag();
                                }
                                iproj++;
                            }
                        }
                    } // end k-point loop

                    // Normalize per-atom gamma
                    std::vector<double> gamma_trial(nat, 0.0);
                    double w_sum = 0.0;
                    for (int iat = 0; iat < nat; iat++) w_sum += w_tot[iat];
                    if (w_sum < 1e-30) break;
                    for (int iat = 0; iat < nat; iat++)
                        gamma_trial[iat] = gamma_total * w_tot[iat] / w_sum;

                    // Compute residual and update lambda
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
                } // end inner loop
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

    // Compute final max_res for output
    std::vector<double> gamma_final(nat, 0.0);
    compute_per_atom_gamma_from_becp(ucell, nocc, gamma_total, gamma_final);
    double max_res = 0.0;
    for (int iat = 0; iat < nat; iat++)
    {
        bool ok = (constrain.empty() || static_cast<size_t>(iat) >= constrain.size() || constrain[iat] != 0);
        if (ok)
            max_res = std::max(max_res, std::abs(gamma_final[iat] - targets[iat]));
    }

    double lam_avg = 0.0;
    for (int iat = 0; iat < nat; iat++) lam_avg += lambda[iat];
    lam_avg /= nat;

    std::cout << " [DeltaP-PW] drho=" << std::scientific << std::setprecision(3)
              << drho << " γ_total=" << std::fixed << std::setprecision(4)
              << gamma_total << " rad  λ_avg=";
    std::cout << std::scientific << std::setprecision(3) << lam_avg
              << " |res|=" << max_res
              << " γ/atom=(" << std::fixed << std::setprecision(4);
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
