#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_io/module_unk/unk_overlap_pw.h"
#include "source_base/constants.h"
#include "source_base/module_external/lapack_connector.h"
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

    // Compute per-atom gamma via Wilson loop decomposition
    std::vector<double> gamma_baseline(nat, 0.0);
    compute_per_atom_gamma_wilson(ucell, nocc, psi_cpu, wfcpw, rhopw, gdir, gamma_baseline);

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
                    compute_per_atom_gamma_wilson(ucell, nocc, psi_cpu, wfcpw, rhopw, gdir, gamma_trial);

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
    compute_per_atom_gamma_wilson(ucell, nocc, psi_cpu, wfcpw, rhopw, gdir, gamma_final);
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

void compute_per_atom_gamma_wilson(
    const UnitCell& ucell,
    int nocc,
    const psi::Psi<std::complex<double>>* psi_cpu,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    std::vector<double>& gamma_per_atom)
{
    int nat = ucell.nat;
    gamma_per_atom.assign(nat, 0.0);
    if (nocc < 1 || psi_cpu == nullptr || wfcpw == nullptr || rhopw == nullptr) return;

    int nk = psi_cpu->get_nk();
    int npol = psi_cpu->get_npol();

    // Build Wilson loop matrix M_{nm} = <u_n(k0)|e^{iG·r}|u_m(k0)>
    // For Gamma-only: single k-point, G-phase overlap gives the Berry phase matrix
    int m_dim = nocc;
    std::vector<std::complex<double>> M(m_dim * m_dim);
    unkOverlap_pw uw;

    ModuleBase::Vector3<double> G(0.0, 0.0, 0.0);
    if (gdir == 1) G = ModuleBase::Vector3<double>(1.0, 0.0, 0.0);
    else if (gdir == 2) G = ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
    else G = ModuleBase::Vector3<double>(0.0, 0.0, 1.0);

    for (int nb = 0; nb < m_dim; nb++)
        for (int mb = 0; mb < m_dim; mb++)
            M[nb * m_dim + mb] = uw.unkdotp_G0(rhopw, wfcpw, 0, 0, nb, mb, psi_cpu, G);

    // Diagonalize: M = V · diag(lambda) · V^{-1}
    // LAPACK zgeev: column-major layout (Fortran order)
    // M is n×n column-major: M(nb, mb) = M[nb * n + mb]
    // For right eigenvectors: M * V = V * lambda
    int info = 0;
    std::vector<std::complex<double>> eigenvalues(m_dim);
    std::vector<std::complex<double>> VR(m_dim * m_dim);
    std::vector<std::complex<double>> work(4 * m_dim);
    std::vector<double> rwork(2 * m_dim);

    char jobvl = 'N';
    char jobvr = 'V';
    int lwork = 4 * m_dim;
    zgeev_(&jobvl, &jobvr, &m_dim, M.data(), &m_dim, eigenvalues.data(),
            nullptr, &m_dim, VR.data(), &m_dim,
            work.data(), &lwork, rwork.data(), &info);

    if (info != 0) return;

    std::vector<double> theta(m_dim);
    for (int n = 0; n < m_dim; n++)
        theta[n] = atan2(eigenvalues[n].imag(), eigenvalues[n].real());

    // Get becp at k0 from OnsiteProjector
    auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
    if (onsite_p == nullptr) return;
    int tot_nproj = onsite_p->get_tot_nproj();
    if (tot_nproj == 0) return;
    const std::complex<double>* becp = onsite_p->get_becp();
    if (becp == nullptr) return;

    // Project becp onto Wilson loop eigenvectors: proj[α, n] = Σ_m VR[m,n] × becp[α,m]
    // VR is column-major: VR(m, n) = VR[n * m_dim + m]
    std::vector<std::vector<double>> w_atom_band(nat, std::vector<double>(m_dim, 0.0));

    int iproj = 0;
    for (int iat = 0; iat < nat; iat++)
    {
        int nh = onsite_p->get_nh(iat);
        for (int ip = 0; ip < nh; ip++)
        {
            for (int n = 0; n < m_dim; n++)
            {
                std::complex<double> proj_val(0.0, 0.0);
                for (int m = 0; m < m_dim; m++)
                {
                    std::complex<double> vr_mn = VR[n * m_dim + m];
                    std::complex<double> becp_val = becp[m * tot_nproj + iproj];
                    proj_val += vr_mn * std::conj(becp_val);
                }
                double weight = proj_val.real() * proj_val.real() + proj_val.imag() * proj_val.imag();
                w_atom_band[iat][n] += weight;
            }
            iproj++;
        }
    }

    // Normalize per-band weights and compute per-atom gamma
    for (int n = 0; n < m_dim; n++)
    {
        double w_tot = 0.0;
        for (int iat = 0; iat < nat; iat++) w_tot += w_atom_band[iat][n];
        if (w_tot < 1e-30) continue;
        for (int iat = 0; iat < nat; iat++)
            gamma_per_atom[iat] += w_atom_band[iat][n] * theta[n] / w_tot;
    }
}

} // namespace pw_deltap
