#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_esolver/deltap_scf.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_io/module_unk/unk_overlap_pw.h"
#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/parallel_common.h"
#include <iomanip>
#include <iostream>
#include <limits>

namespace pw_deltap {

namespace {

// ---------------------------------------------------------------------------
// Operator-state backing: written by the DeltapScfSolver backend's set_lambda
// callback and read by the on-site force / stress / Hamiltonian consumers
// (forces_onsite.cpp, stress_onsite.cpp, op_pw_proj.cpp).
// ---------------------------------------------------------------------------
std::vector<double> s_lambda;   // current per-atom lambda (Ry)
std::vector<int> s_constrain;   // per-atom constrain flags

// Shared SCF constraint state machine: owns lambda_set / gamma / escon /
// branch-tracking history (replaces the historical file-level singletons).
deltap_scf::DeltapScfSolver g_solver;

void store_lambda(const std::vector<double>& lambda, const std::vector<int>& constrain)
{
    s_lambda = lambda;
    s_constrain = constrain;
}

// ---------------------------------------------------------------------------
// Total Berry phase gamma along gdir (sum over k-strings of stringPhase).
// ---------------------------------------------------------------------------
double compute_total_gamma_pw(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi_in,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    int nbands)
{
    if (gdir < 1 || gdir > 3) return std::numeric_limits<double>::quiet_NaN();
    if (wfcpw == nullptr || psi_in == nullptr || rhopw == nullptr)
        return std::numeric_limits<double>::quiet_NaN();

    berryphase bp;
    bp.direction = gdir;
    bp.GDIR = gdir;
    bp.set_kpoints(kv, gdir);

    if (bp.total_string == 0 || bp.nppstr < 2)
        return std::numeric_limits<double>::quiet_NaN();

    double gamma_total = 0.0;
    for (int istr = 0; istr < bp.total_string; istr++)
    {
        gamma_total += bp.stringPhase(ucell, istr, nbands,
                                       wfcpw->npwk_max, psi_in, rhopw, wfcpw, kv);
    }
    return gamma_total;
}

// ---------------------------------------------------------------------------
// Per-atom gamma via k-string Wilson loop eigenvalue decomposition:
//  1. Build overlap matrices M_j = <u_n(k_j)|u_m(k_{j+1})> (unkdotp_G) with
//     G-phase for the boundary link (unkdotp_G0).
//  2. Wilson loop product M_total = Π_j M_j, diagonalized via zgeev.
//  3. Project becp at k0 onto the eigenvectors → per-atom weights, giving
//     gamma[I] += w[I,n] × θ_n / Σ_J w[J,n], averaged over k-strings.
// ---------------------------------------------------------------------------
void compute_per_atom_gamma_kstring(
    const UnitCell& ucell,
    int nocc,
    const psi::Psi<std::complex<double>>* psi_cpu,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    std::vector<double>& gamma_per_atom)
{
    int nat = ucell.nat;
    gamma_per_atom.assign(nat, 0.0);
    if (nocc < 1 || psi_cpu == nullptr || wfcpw == nullptr || rhopw == nullptr) return;

    int m_dim = nocc;
    unkOverlap_pw uw;

    // Set up k-strings
    berryphase bp;
    bp.direction = gdir;
    bp.GDIR = gdir;
    bp.set_kpoints(kv, gdir);
    if (bp.total_string == 0 || bp.nppstr < 2) return;

    ModuleBase::Vector3<double> G(0.0, 0.0, 0.0);
    if (gdir == 1)      G = ModuleBase::Vector3<double>(1.0, 0.0, 0.0);
    else if (gdir == 2) G = ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
    else                G = ModuleBase::Vector3<double>(0.0, 0.0, 1.0);

    auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
    if (onsite_p == nullptr) return;
    int tot_nproj = onsite_p->get_tot_nproj();
    if (tot_nproj == 0) return;
    const std::complex<double>* becp_k0 = onsite_p->get_becp();
    if (becp_k0 == nullptr) return;

    std::vector<std::vector<double>> theta_str(bp.total_string, std::vector<double>(m_dim, 0.0));

    for (int istr = 0; istr < bp.total_string; istr++)
    {
        // Wilson loop product M_total = Π_j M_j
        std::vector<std::complex<double>> M_total(m_dim * m_dim);
        for (int i = 0; i < m_dim; i++) M_total[i * m_dim + i] = 1.0;

        for (int ks = 0; ks < (bp.nppstr - 1); ks++)
        {
            int ik1 = bp.k_index[istr][ks];
            int ik2 = bp.k_index[istr][ks + 1];
            std::vector<std::complex<double>> M_j(m_dim * m_dim);

            if (ks == (bp.nppstr - 2))
                for (int nb = 0; nb < m_dim; nb++)
                    for (int mb = 0; mb < m_dim; mb++)
                        M_j[nb * m_dim + mb] = uw.unkdotp_G0(rhopw, wfcpw, ik1, ik2, nb, mb, psi_cpu, G);
            else
                for (int nb = 0; nb < m_dim; nb++)
                    for (int mb = 0; mb < m_dim; mb++)
                        M_j[nb * m_dim + mb] = uw.unkdotp_G(wfcpw, ik1, ik2, nb, mb, psi_cpu);

            // M_total = M_j · M_total
            std::vector<std::complex<double>> tmp(m_dim * m_dim);
            for (int i = 0; i < m_dim; i++)
                for (int j = 0; j < m_dim; j++)
                    for (int k = 0; k < m_dim; k++)
                        tmp[i * m_dim + j] += M_j[i * m_dim + k] * M_total[k * m_dim + j];
            M_total = tmp;
        }

        // Diagonalize
        std::vector<std::complex<double>> ev(m_dim), VR(m_dim * m_dim);
        std::vector<std::complex<double>> work(4 * m_dim);
        std::vector<double> rwork(2 * m_dim);
        char jl = 'N', jr = 'V';
        int lw = 4 * m_dim, info = 0;
        zgeev_(&jl, &jr, &m_dim, M_total.data(), &m_dim, ev.data(),
                nullptr, &m_dim, VR.data(), &m_dim, work.data(), &lw, rwork.data(), &info);
        if (info != 0) continue;

        for (int n = 0; n < m_dim; n++)
            theta_str[istr][n] = atan2(ev[n].imag(), ev[n].real());

        // Project becp onto eigenvectors → per-atom weights
        std::vector<std::vector<double>> w_ab(nat, std::vector<double>(m_dim, 0.0));
        int ip = 0;
        for (int iat = 0; iat < nat; iat++)
        {
            int nh = onsite_p->get_nh(iat);
            for (int ih = 0; ih < nh; ih++)
            {
                for (int n = 0; n < m_dim; n++)
                {
                    std::complex<double> p(0,0);
                    for (int m = 0; m < m_dim; m++)
                        p += VR[m + n * m_dim] * std::conj(becp_k0[m * tot_nproj + ip]);
                    w_ab[iat][n] += p.real()*p.real() + p.imag()*p.imag();
                }
                ip++;
            }
        }

        // Per-atom gamma for this string
        for (int n = 0; n < m_dim; n++)
        {
            double w_t = 0; for (int iat = 0; iat < nat; iat++) w_t += w_ab[iat][n];
            if (w_t < 1e-30) continue;
            for (int iat = 0; iat < nat; iat++)
                gamma_per_atom[iat] += w_ab[iat][n] * theta_str[istr][n] / w_t;
        }
    }

    // Average over k-strings
    for (int iat = 0; iat < nat; iat++)
        gamma_per_atom[iat] /= bp.total_string;
}

// ---------------------------------------------------------------------------
// PW backend for the shared DeltapScfSolver: per-atom γ measurement and the
// operator λ storage that the on-site Hamiltonian / forces read.
// ---------------------------------------------------------------------------
deltap_scf::DeltapScfSolver::Backend make_backend(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi_cpu, const K_Vectors* kv,
    const ModulePW::PW_Basis_K* wfcpw, const ModulePW::PW_Basis* rhopw,
    int gdir)
{
    deltap_scf::DeltapScfSolver::Backend b;
    b.set_lambda = [](const std::vector<double>& lambda) { store_lambda(lambda, s_constrain); };
    b.get_lambda = []() { return s_lambda; };
    // Keep the operator lambda consistent across MPI ranks after each update
    // (S-09): rank 0's value wins, then every rank's storage is refreshed.
    b.sync_lambda = [](std::vector<double>& lam) {
#ifdef __MPI
        if (GlobalV::NPROC > 1)
            Parallel_Common::bcast_double(lam.data(), static_cast<int>(lam.size()));
#endif
        s_lambda = lam;
    };
    // Per-atom gamma via Wilson-loop decomposition.  psi/kv/wfcpw/rhopw are
    // stable for the whole run (allocated in before_runner); nocc derives
    // from PARAM at call time, matching the historical computation.
    b.compute_gamma = [&ucell, psi_cpu, kv, wfcpw, rhopw, gdir]() {
        int nocc = static_cast<int>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
        if (nocc < 1) nocc = 1;
        std::vector<double> gamma(ucell.nat, 0.0);
        compute_per_atom_gamma_kstring(ucell, nocc, psi_cpu, *kv, wfcpw, rhopw, gdir, gamma);
#ifdef __MPI
        // Keep the per-atom γ measurement identical on every rank (rank 0
        // wins): gamma_report / escon then match everywhere, applying the
        // same policy as sync_lambda and the historical gamma sync
        // (750ee179d).
        if (GlobalV::NPROC > 1)
            Parallel_Common::bcast_double(gamma.data(), static_cast<int>(gamma.size()));
#endif
        return gamma;
    };
    return b;
}

// ---------------------------------------------------------------------------
// [DeltaP-PW] report line.  Format kept byte-identical to the historical
// output (tests may parse it): masked max residual and per-atom gamma come
// from the state machine's branch-selected gamma; gamma_total is the
// separate Wilson-loop total.
// ---------------------------------------------------------------------------
void report_pw(double drho, double gamma_total)
{
    if (GlobalV::MY_RANK != 0)
        return;
    const auto& st = g_solver.state();
    const auto& targets = g_solver.params().target;
    const int nat = g_solver.params().nat;

    double lam_avg = 0.0;
    for (int iat = 0; iat < nat; ++iat)
        lam_avg += s_lambda[iat];
    lam_avg /= nat;

    // Max residual over free atoms only; missing targets treated as 0
    // (historical PW semantics: no target = constrain γ to zero).
    double max_res = 0.0;
    for (int iat = 0; iat < nat; ++iat)
    {
        bool ok = (s_constrain.empty() || iat >= static_cast<int>(s_constrain.size())
                   || s_constrain[iat] != 0);
        if (!ok) continue;
        double t = (iat < static_cast<int>(targets.size())) ? targets[iat] : 0.0;
        max_res = std::max(max_res, std::abs(st.gamma_report[iat] - t));
    }

    std::cout << " [DeltaP-PW] drho=" << std::scientific << std::setprecision(3)
              << drho << " γ_total=" << std::fixed << std::setprecision(4)
              << gamma_total << " rad  λ_avg=";
    std::cout << std::scientific << std::setprecision(3) << lam_avg
              << " |res|=" << max_res
              << " escon=" << std::fixed << std::setprecision(6) << st.dp_escon << " Ry"
              << " γ/atom=(";
    for (int iat = 0; iat < nat; ++iat)
    {
        if (iat > 0) std::cout << ", ";
        std::cout << st.gamma_report[iat];
    }
    std::cout << ")" << std::endl;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Initialize DeltaP PW from INPUT + STRU into the shared SCF state machine.
// Called from ESolver_KS_PW::before_all_runners (once per run).
// ---------------------------------------------------------------------------
void deltap_init(const UnitCell& ucell, const Input_para& inp,
                 const psi::Psi<std::complex<double>>* psi_cpu,
                 const K_Vectors* kv, const ModulePW::PW_Basis_K* wfcpw,
                 const ModulePW::PW_Basis* rhopw)
{
    // Targets / constrain come from STRU; the initial lambda is a separate
    // parameter (Ry) and MUST NOT be conflated with the gamma targets (rad).
    const std::vector<double> dp_target = ucell.get_dp_target();
    const std::vector<int> dp_constrain = ucell.get_dp_constrain();
    s_lambda.assign(ucell.nat, inp.deltap_lambda_init);
    s_constrain = dp_constrain;

    deltap_scf::DeltapParams p;
    p.nat = ucell.nat;
    p.gdir = inp.deltap_gdir;
    p.inner_thr = inp.deltap_inner_thr;
    p.lambda_step = inp.deltap_lambda_step;
    p.lambda_mixing = inp.deltap_lambda_mixing;
    p.lambda_init = inp.deltap_lambda_init;
    p.conv_thr = inp.deltap_conv_thr;
    p.nscf = 0;                 // PW inner loop rejected in deltap_iter_finish
    p.total_mode = false;       // PW historically updates per-atom (INPUT total mode ignored)
    p.verbose = false;          // PW prints its own [DeltaP-PW] line
    p.unwrap_branch_2pi = true; // cross-SCF 2π branch tracking
    p.target = dp_target;
    // Historical PW semantics: no target means "constrain γ to zero", so
    // missing STRU targets become an explicit zero target vector (this also
    // fixes the historical out-of-bounds read on the empty target vector).
    if (p.target.empty())
        p.target.assign(ucell.nat, 0.0);
    p.constrain = dp_constrain;

    g_solver.init(p, make_backend(ucell, psi_cpu, kv, wfcpw, rhopw, inp.deltap_gdir));

    bool has_strutarget = false;
    for (double t : dp_target)
        if (std::abs(t) > 1e-12) { has_strutarget = true; break; }
    if (GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP-PW] Initialized with " << ucell.nat << " atoms";
        if (has_strutarget)
            std::cout << " (STRU targets)";
        else
            std::cout << " (no targets)";
        std::cout << " lambda_init=" << inp.deltap_lambda_init << " Ry";
        std::cout << std::endl;
    }
}

const std::vector<double>& get_deltap_pw_lambda()
{
    return s_lambda;
}

const std::vector<int>& get_deltap_pw_constrain()
{
    return s_constrain;
}

double get_deltap_pw_escon()
{
    return g_solver.state().dp_escon;
}

void reset_deltap_pw_scf_cycle()
{
    // Per-SCF-cycle reset: allow one lambda update per ionic step and restart
    // cross-iteration 2π branch tracking for the new geometry.
    g_solver.reset_ionic_step();
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
    if (ucell.nat == 0)
        return;
    if (inp.deltap_gdir < 1 || inp.deltap_gdir > 3)
        return;

    // Two-phase gate: act only once the charge density is converged enough.
    if (drho <= 0.0 || drho >= inp.deltap_inner_thr)
        return;
    // One λ update per SCF cycle (Phase 2), then λ stays frozen.
    if (g_solver.state().lambda_set)
        return;

    // The PW inner loop is not implemented: it would re-measure gamma from
    // the same wavefunctions without re-diagonalizing the Hamiltonian, so
    // every iteration would see the same residual and λ would overshoot by
    // inner_nmax×.  Use the synchronous two-phase mode (deltap_inner_nmax=0).
    if (inp.deltap_inner_nmax > 0)
    {
        ModuleBase::WARNING_QUIT("deltap_pw",
            "DeltaP-PW inner loop (deltap_inner_nmax > 0) is not implemented. "
            "Set deltap_inner_nmax to 0 for synchronous two-phase mode.");
    }

    // Total Berry phase along gdir (separate from the per-atom decomposition).
    int nocc = static_cast<int>(inp.nelec / ModuleBase::DEGSPIN);
    if (nocc < 1) nocc = 1;
    double gamma_total = compute_total_gamma_pw(ucell, psi_cpu, kv, wfcpw, rhopw,
                                                inp.deltap_gdir, nocc);
    if (std::isnan(gamma_total))
        return;

    // Measure per-atom γ, run the gradient-descent λ update, apply 2π branch
    // tracking and escon inside the shared state machine.
    g_solver.iter_finish(0, drho);

    report_pw(drho, gamma_total);
}

} // namespace pw_deltap
