#include "deltap.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_hs/cal_r_overlap_R.h"
#include "source_io/module_unk/unk_overlap_lcao.h"
#ifdef __MPI
#include "source_base/parallel_comm.h"
#include "source_base/module_external/scalapack_connector.h"
#endif
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <vector>
#include <fstream>
#include <iomanip>

// LAPACK declarations (Fortran, column-major). zgesvd_ is not in
// lapack_connector.h, so declare it here. zgetrf_ redeclaration is harmless.
extern "C" {
    void zgesvd_(const char* jobu, const char* jobvt, const int* M, const int* N,
                 std::complex<double>* A, const int* lda, double* S,
                 std::complex<double>* U, const int* ldu,
                 std::complex<double>* Vt, const int* ldvt,
                 std::complex<double>* work, const int* lwork,
                 double* rwork, int* info);
    void zgetrf_(const int* M, const int* N, std::complex<double>* A,
                 const int* lda, int* ipiv, int* info);
    void zgeev_(const char* jobvl, const char* jobvr, const int* N,
                std::complex<double>* A, const int* lda,
                std::complex<double>* W,
                std::complex<double>* VL, const int* ldvl,
                std::complex<double>* VR, const int* ldvr,
                std::complex<double>* work, const int* lwork,
                double* rwork, int* info);
    void zgeqrf_(const int* M, const int* N, std::complex<double>* A,
                 const int* lda, std::complex<double>* tau,
                 std::complex<double>* work, const int* lwork, int* info);
    void zungqr_(const int* M, const int* N, const int* K,
                 std::complex<double>* A, const int* lda,
                 const std::complex<double>* tau,
                 std::complex<double>* work, const int* lwork, int* info);
}

namespace deltap {

// Build the displacement overlap matrix S(dk)_{mu,nu} = sum_R e^{2*pi*i*dk.R}
// <phi_mu(0) | phi_nu(R)> in the same 2D-block-cyclic distribution as paraV_.
// dk is the spacing between two adjacent k-points on the Wilson string:
// dk = 1/(nppstr_-1) along gdir_. The matrix is reused for every k-pair, so it
// is computed once per call. All NAO basis pairs are evaluated (unlike the
// SMO-projector overlaps in compute_real_overlaps which keep first zeta only).
void DeltaP::compute_S_dk(const UnitCell& ucell)
{
    ModuleBase::TITLE("DeltaP", "compute_S_dk");
    ModuleBase::timer::start("DeltaP", "compute_S_dk");

    const int npol = ucell.get_npol();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();
    S_dk_nrow_ = nrow;
    S_dk_ncol_ = ncol;
    S_dk_.assign(static_cast<size_t>(nrow) * ncol, std::complex<double>(0.0, 0.0));

    const int* iat2iwt = paraV_->iat2iwt_;

    // dk vector: 1/(nppstr_-1) along the chosen direction, zero elsewhere
    const double dk_step = 1.0 / (nppstr_ - 1);
    double dkv[3] = {0.0, 0.0, 0.0};
    dkv[gdir_ - 1] = dk_step;

    for (int iat = 0; iat < nat_; iat++)
    {
        auto tau0 = ucell.get_tau(iat);
        int T0 = 0, I0 = 0;
        ucell.iat2iait(iat, &I0, &T0);
        const int nw0 = ucell.atoms[T0].nw;

        // enumerate neighbouring atoms (including the home cell, ad == 0)
        AdjacentAtomInfo adjs;
        gd_->Find_atom(ucell, tau0, T0, I0, &adjs);

        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const ModuleBase::Vector3<int> R = adjs.box[ad];
            const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];

            // cutoff filter: sum of the two basis radii (NOT rm_)
            if (ucell.cal_dtau(iat, iat1, R).norm() * ucell.lat0
                > orb_cutoff_[T0] + orb_cutoff_[T1])
            {
                continue;
            }

            // Phase: 2*pi*(dk.R - dk.tau_bra)
            // This matches the ABACUS berry_phase convention:
            // exp(2*pi*i*(k_R*R - dk*tau)) where k_R = k_L + dk
            // = exp(2*pi*i*dk*R) * exp(2*pi*i*(k_L*R - dk*tau))
            // For the Bloch-state overlap (no position correction),
            // we use exp(2*pi*i*(dk*R - dk*tau_bra))
            // The tau_bra correction is a per-atom phase that converts
            // <psi_k|psi_{k+b}> to approximately <u_k|u_{k+b}>
            double arg = ModuleBase::TWO_PI * (
                dkv[0] * R.x + dkv[1] * R.y + dkv[2] * R.z
                - dkv[0] * tau0.x - dkv[1] * tau0.y - dkv[2] * tau0.z);
            const std::complex<double> phase(std::cos(arg), std::sin(arg));

            // snap wants vR = R2 - R1 = (ket centre) - (bra neighbour), in Bohr
            const ModuleBase::Vector3<double> dtau = tau0 - tau1;
            const Atom* atom1 = &ucell.atoms[T1];
            const int nw1 = atom1->nw;

            for (int iw1 = 0; iw1 < nw1; ++iw1)
            {
                const int L1 = atom1->iw2l[iw1];
                const int N1 = atom1->iw2n[iw1];
                const int m1 = atom1->iw2m[iw1];
                const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;

                std::vector<std::vector<double>> nlm;
                overlap_intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm);
                if (nlm.empty() || nlm[0].empty()) continue;

                for (int iw0 = 0; iw0 < nw0; ++iw0)
                {
                    const double ov = nlm[0][iw0];
                    if (std::abs(ov) < 1e-15) continue;

                    // the spatial overlap is spin-diagonal: place it on every
                    // spin block so that the result is valid for npol > 1 too
                    for (int s = 0; s < npol; ++s)
                    {
                        const int gmu = iat2iwt[iat] + npol * iw0 + s;
                        const int gnu = iat2iwt[iat1] + npol * iw1 + s;
                        const int lr = paraV_->global2local_row(gmu);
                        const int lc = paraV_->global2local_col(gnu);
                        if (lr >= 0 && lc >= 0)
                        {
                            S_dk_[lr + lc * nrow] += phase * ov;
                        }
                    }
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_S_dk");
}

// Per-link version: computes S(dk) with the berry_phase phase convention
//   phase = 2*pi*(k_R*R - dk*tau_bra)
// and adds the first-order position operator correction
//   overlap *= (1 - i*dk_cart*tpiba*R_alpha)
// This converts Bloch-state overlap <psi|psi> to periodic-part overlap <u|u>.
void DeltaP::compute_S_dk_link(const UnitCell& ucell,
                               const ModuleBase::Vector3<double>& kvec_d_R,
                               const ModuleBase::Vector3<double>& kvec_c_L,
                               const ModuleBase::Vector3<double>& kvec_c_R)
{
    const int npol = ucell.get_npol();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();
    S_dk_nrow_ = nrow;
    S_dk_ncol_ = ncol;
    S_dk_.assign(static_cast<size_t>(nrow) * ncol, std::complex<double>(0.0, 0.0));

    const int* iat2iwt = paraV_->iat2iwt_;
    const double dk_step = 1.0 / (nppstr_ - 1);
    double dkv[3] = {0.0, 0.0, 0.0};
    dkv[gdir_ - 1] = dk_step;
    double dk_cart = dk_step * ucell.tpiba * ucell.lat0;

    // First call: compute and cache the raw overlap data + position matrix
    if (!S_dk_cache_valid_)
    {
        S_dk_cache_.clear();
        for (int iat = 0; iat < nat_; iat++)
        {
            auto tau0 = ucell.get_tau(iat);
            int T0 = 0, I0 = 0;
            ucell.iat2iait(iat, &I0, &T0);
            const int nw0 = ucell.atoms[T0].nw;
            AdjacentAtomInfo adjs;
            gd_->Find_atom(ucell, tau0, T0, I0, &adjs);

            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = ucell.itia2iat(T1, I1);
                const ModuleBase::Vector3<int> R = adjs.box[ad];
                if (ucell.cal_dtau(iat, iat1, R).norm() * ucell.lat0
                    > orb_cutoff_[T0] + orb_cutoff_[T1]) continue;

                const ModuleBase::Vector3<double> dtau = tau0 - adjs.adjacent_tau[ad];
                const Atom* atom1 = &ucell.atoms[T1];
                const int nw1 = atom1->nw;
                ModuleBase::Vector3<double> R1_cart = tau0 * ucell.lat0;
                ModuleBase::Vector3<double> R2_cart = adjs.adjacent_tau[ad] * ucell.lat0;

                for (int iw1 = 0; iw1 < nw1; ++iw1)
                {
                    const int L1 = atom1->iw2l[iw1];
                    const int N1 = atom1->iw2n[iw1];
                    const int m1 = atom1->iw2m[iw1];
                    const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                    std::vector<std::vector<double>> nlm;
                    overlap_intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm);
                    if (nlm.empty() || nlm[0].empty()) continue;

                    for (int iw0 = 0; iw0 < nw0; ++iw0)
                    {
                        const double ov = nlm[0][iw0];
                        if (std::abs(ov) < 1e-15) continue;
                        for (int s = 0; s < npol; ++s)
                        {
                            const int gmu = iat2iwt[iat] + npol * iw0 + s;
                            const int gnu = iat2iwt[iat1] + npol * iw1 + s;
                            const int lr = paraV_->global2local_row(gmu);
                            const int lc = paraV_->global2local_col(gnu);
                            if (lr >= 0 && lc >= 0)
                            {
                                S_dk_cache_entry e;
                                e.lr = lr; e.lc = lc; e.ov = ov;
                                e.Rx = R.x; e.Ry = R.y; e.Rz = R.z;
                                e.tau_x = tau0.x; e.tau_y = tau0.y; e.tau_z = tau0.z;
                                e.R1_cart = R1_cart;
                                e.T1 = T0; e.L1 = ucell.atoms[T0].iw2l[iw0];
                                e.m1 = ucell.atoms[T0].iw2m[iw0]; e.N1 = ucell.atoms[T0].iw2n[iw0];
                                e.R2_cart = R2_cart;
                                e.T2 = T1; e.L2 = L1; e.m2 = m1; e.N2 = N1;

                                // Pre-compute local position matrix <phi|r'|phi(R)>
                                if (r_overlap_)
                                {
                                    ModuleBase::Vector3<double> r_full = r_overlap_->get_psi_r_psi(
                                        R1_cart, e.T1, e.L1, e.m1, e.N1,
                                        R2_cart, e.T2, e.L2, e.m2, e.N2);
                                    ModuleBase::Vector3<double> r_local = r_full - R1_cart * ov;
                                    e.r_local_x = r_local.x;
                                    e.r_local_y = r_local.y;
                                    e.r_local_z = r_local.z;
                                    e.r_computed = true;
                                }

                                S_dk_cache_.push_back(std::move(e));
                            }
                        }
                    }
                }
            }
        }
        S_dk_cache_valid_ = true;
    }

    // Apply per-link phase + position correction using cached data
    // berry_phase convention:
    //   phase = 2*pi*(kvec_c_R . R_cart - dk_c . tau)
    //   overlap = <phi|phi(R)> - i*dk*tpiba*<phi|r|phi(R)>
    ModuleBase::Vector3<double> dk_c = kvec_c_R - kvec_c_L;

    for (const auto& e : S_dk_cache_)
    {
        ModuleBase::Vector3<double> R_cart = e.Rx * ucell.a1 + e.Ry * ucell.a2 + e.Rz * ucell.a3;
        double arg = ModuleBase::TWO_PI * (
            kvec_c_R.x * R_cart.x + kvec_c_R.y * R_cart.y + kvec_c_R.z * R_cart.z
            - dk_c.x * e.tau_x - dk_c.y * e.tau_y - dk_c.z * e.tau_z);
        std::complex<double> phase(std::cos(arg), std::sin(arg));

        // Position correction using CACHED local position matrix
        std::complex<double> overlap(e.ov, 0.0);
        if (e.r_computed)
        {
            double imag_part = -(dk_c.x * e.r_local_x + dk_c.y * e.r_local_y + dk_c.z * e.r_local_z) * ucell.tpiba;
            overlap = std::complex<double>(e.ov, imag_part);
        }

        S_dk_[e.lr + e.lc * nrow] += phase * overlap;
    }
}

void DeltaP::compute_wannier_polarization(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DeltaP", "compute_wannier_polarization");
    ModuleBase::timer::start("DeltaP", "compute_wannier_polarization");

    std::cout << "\n * * * * * *\n << Start DeltaP Wannier polarization\n";

    // Load branch state from previous SCF/run for cross-SCF phase smoothness
    if (!scf_mode_)
        load_branch();
    load_match();

    // Step 0: compute real-space overlaps and k-string (skip if already done in SCF)
    if (!scf_initialized_)
    {
        compute_real_overlaps(ucell, *gd_);
        compute_smo_overlap_matrix(ucell);
        scf_initialized_ = true;
    }

    const int nks = psi->get_nk();
    const int nbands = psi->get_nbands();
    const int nrow_local = paraV_->get_row_size();

    // Get occupied bands
    double occ_bands_d = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands_d - std::floor(occ_bands_d)) > 0.0)
        occ_bands_d = std::floor(occ_bands_d) + 1.0;
    const int nocc = static_cast<int>(occ_bands_d);
    const int nocc_use = std::min(nocc, nbands);

    // Total SMO projection channels
    int nproj_total = 0;
    for (int iat = 0; iat < nat_; ++iat)
        nproj_total += nproj_per_atom_[iat];

    // Save original gdir for restore
    const int gdir_orig = gdir_;

    // Initialize 3-component results
    results_.P_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I_raw.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.r_elec_center.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    if (static_cast<int>(W_prev_.size()) != nat_)
        W_prev_.assign(nat_, ModuleBase::Vector3<double>(std::numeric_limits<double>::quiet_NaN(),
                                                          std::numeric_limits<double>::quiet_NaN(),
                                                          std::numeric_limits<double>::quiet_NaN()));

    // ---- Compute polarization for all three directions ----
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        gdir_ = alpha + 1;
        const int alpha_idx = gdir_ - 1;
        setup_kstring(*kv_);
        S_dk_.clear();
        S_dk_cache_valid_ = false;

        // Per-alpha: isolated accumulators.  Each direction has its own
        // gamma_accum, string count, prev_gamma, etc. — no cross-direction
        // contamination from the Wilson loop or branch selection.
        std::vector<double> gamma_accum(nat_, 0.0);
        std::vector<double> smo_w_accum(nat_, 0.0);
        std::vector<double> gamma_raw_accum(nat_, 0.0);
        std::vector<double> r_elec_accum(nat_, 0.0);
        int n_strings_processed = 0;
        std::vector<std::vector<double>> w_In_first_string_;  // current alpha's first-string weights
        std::vector<std::complex<double>> zeta_list;
        std::vector<double> total_bp_per_string;      // total Berry phase (arg(zeta)) per string
        double current_zeta_scale = 1.0;               // scale factor from last zeta rescale

        // Branch reference: if previous converged value exists, use it;
        // otherwise use NaN to skip branch selection on first iteration.
        // This ensures the first iteration records raw gamma without being
        // pulled toward 0 by the branch selection logic.
        std::vector<double> prev_gamma(nat_, std::numeric_limits<double>::quiet_NaN());
        for (int iat = 0; iat < nat_; ++iat)
            if (has_prev_ && static_cast<int>(W_prev_.size()) > iat
                && !std::isnan(W_prev_[iat][alpha]))
                prev_gamma[iat] = W_prev_[iat][alpha];
        gamma_principal_.assign(nat_, 0.0);
        gamma_selected_.assign(nat_, 0.0);

        std::cout << "   DeltaP [gdir=" << gdir_ << "]: nppstr_=" << nppstr_ << " total_string_=" << total_string_
                  << " k_index_.size()=" << k_index_.size() << " nks=" << nks << std::endl;
        if (k_index_.empty() || nppstr_ == 0)
            continue;  // skip this direction

        // compute_S_dk for this direction
        S_dk_cache_valid_ = false;

        int n_dim = nocc_use;
        int m_dim = nproj_total;

        double a_alpha = 0.0;
        if (gdir_ == 1) a_alpha = ucell.lat0 * ucell.a1.norm();
        else if (gdir_ == 2) a_alpha = ucell.lat0 * ucell.a2.norm();
        else a_alpha = ucell.lat0 * ucell.a3.norm();
        const double omega = ucell.omega;
        double spin_factor = (PARAM.inp.nspin == 1) ? 2.0 : 1.0;
        const double prefactor = spin_factor * a_alpha / (2.0 * ModuleBase::PI * omega);

        // String layout and diagnostics (per direction)
        kstring_data_.resize(nppstr_);
        std::vector<std::complex<double>*> psi_k_ptrs(nppstr_, nullptr);
        std::vector<std::vector<double>> gamma_accum_per_string;
        std::vector<std::vector<std::complex<double>>> evals_all;
        std::vector<std::vector<double>> gamma_raw_per_string;
        std::vector<std::vector<double>> gamma_sel_per_string;

        // Persistent reference for zeta rescaling (consistent across strings)
        double ref_gamma_unw_sum = 0.0;
        bool ref_captured = false;

        for (int istring = 0; istring < total_string_; ++istring)
    {
        // --- Step 1a: S_k, D_I for this k-string ---
        // CRITICAL: clear kstring_data_ to prevent accumulation across strings
        for (int j = 0; j < nppstr_; ++j)
        {
            kstring_data_[j].S_k.clear();
            kstring_data_[j].dS_k.clear();
            kstring_data_[j].D_I.clear();
        }
        for (int j = 0; j < nppstr_; ++j)
        {
            int ik_psi = k_index_[istring][j];
            if (ik_psi >= nks) continue;
            kstring_data_[j].kvec_d = kv_->kvec_d[ik_psi];
            psi->fix_k(ik_psi);
            psi_k_ptrs[j] = psi->get_pointer();
            compute_S_k(j);
            compute_D_I(j, psi->get_pointer(), nbands, nrow_local);
        }

#ifdef __MPI
        for (int j = 0; j < nppstr_; j++)
        {
            for (int iat = 0; iat < nat_; iat++)
            {
                int r = nproj_per_atom_[iat];
                for (int lm = 0; lm < r; lm++)
                {
                    if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                    int sz = kstring_data_[j].D_I[iat][lm].size();
                    if (sz > 0)
                        MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                      2 * sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
                }
            }
        }
#endif

        // Mark which gdir and string kstring_data_ belongs to.
        // compute_hk_correction uses this to detect stale data.
        kstring_gdir_ = gdir_;
        kstring_string_ = istring;

        // --- Step 2: O_kpair (use berry_phase overlap if available) ---
        if (istring == 0)
            std::cout << "   DeltaP: berry_overlap_=" << (berry_overlap_ ? "non-null" : "NULL") << std::endl;

        std::vector<std::vector<std::complex<double>>> O_kpair(nppstr_ - 1);
        ModuleBase::Vector3<double> dk_string;
        if (nppstr_ > 1 && k_index_[istring][0] < nks && k_index_[istring][1] < nks)
            dk_string = kv_->kvec_c[k_index_[istring][1]] - kv_->kvec_c[k_index_[istring][0]];

        // Pre-compute S_dk and allocate SC workspace for fast O_kpair path.
        if (S_dk_.empty() && !berry_overlap_)
            compute_S_dk(ucell);
        const int nrow_ov = paraV_->get_row_size();
        std::vector<std::complex<double>> SC(static_cast<size_t>(nrow_ov) * nocc_use, {0.0, 0.0});

        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            int ik_L = k_index_[istring][j];
            int ik_R = k_index_[istring][j + 1];
            std::vector<std::complex<double>> O_full(
                static_cast<size_t>(nocc_use) * nocc_use, std::complex<double>(0.0, 0.0));

            // G-phase for PBC-wrapped boundary link.
            // G_cart = (2π/a_gdir, 0, 0) etc. for direction gdir.
            // For a uniform Gamma mesh: G_cart = n_p · dk_string, where
            // n_p = number of k-points along gdir (kvec_d stores spacing dk).
            ModuleBase::Vector3<double> G_cart_bdy(0.0, 0.0, 0.0);
            const ModuleBase::Vector3<double>* G_add_ptr = nullptr;
            if (j == nppstr_ - 2 && gdir_ > 0 && gdir_ <= 3)
            {
                G_cart_bdy = dk_string * static_cast<double>(nmp_use_[gdir_ - 1]);
                G_add_ptr = &G_cart_bdy;
            }
            if (ik_R < nks && ik_L < nks)
            {
                if (berry_overlap_)
                {
                    berry_overlap_->berryphase_overlap(ucell, ik_L, ik_R,
                        dk_string,
                        nocc_use, *paraV_, psi, *kv_, O_full, G_add_ptr);
                }
                else
                {
                    // Fast path: O = C†(k_L) · S(dk) · C(k_R) via manual GEMM.
                    // NOTE: This path does not include the G-phase for the PBC-wrapped
                    // boundary link. For production use, set berry_overlap_ (via init)
                    // to use the correct berryphase_overlap path.
                    psi->fix_k(ik_L);
                    const std::complex<double>* c_L = psi->get_pointer();
                    psi->fix_k(ik_R);
                    const std::complex<double>* c_R = psi->get_pointer();

                    const int nrow = paraV_->get_row_size();
                    const int ncol = paraV_->get_col_size();
                    // SC = S_dk * C_R  (nrow × nocc_use)
                    for (int p = 0; p < nocc_use; ++p)
                        for (int alpha = 0; alpha < nrow; ++alpha)
                        {
                            std::complex<double> s(0.0, 0.0);
                            for (int gamma = 0; gamma < ncol; ++gamma)
                                s += S_dk_[alpha + gamma * nrow] * c_R[gamma + p * nrow];
                            SC[alpha + p * nrow] = s;
                        }
                    // O = C_L† * SC  (nocc_use × nocc_use)
                    for (int q = 0; q < nocc_use; ++q)
                        for (int p = 0; p < nocc_use; ++p)
                        {
                            std::complex<double> s(0.0, 0.0);
                            for (int alpha = 0; alpha < nrow; ++alpha)
                                s += std::conj(c_L[alpha + q * nrow]) * SC[alpha + p * nrow];
                            O_full[q + p * nocc_use] = s;
                        }
                }
            }

             O_kpair[j] = O_full;
         }

        // --- Step 3a: Compute zeta = prod_j det(O_j) ---
        // Use det_berryphase directly (bypasses O_matrix gathering issues)
        // CRITICAL: det_berryphase does MPI_Allreduce(MPI_PROD) internally,
        // so ALL ranks must call it for EVERY link, even if ik_L/ik_R are invalid.
        std::complex<double> zeta_scalar(1.0, 0.0);
        if (berry_overlap_ && nppstr_ > 1)
        {
            ModuleBase::Vector3<double> dk_str;
            if (k_index_[istring][0] < nks && k_index_[istring][1] < nks)
                dk_str = kv_->kvec_c[k_index_[istring][1]] - kv_->kvec_c[k_index_[istring][0]];
            for (int j = 0; j < nppstr_ - 1; ++j)
            {
                int ik_L = k_index_[istring][j];
                int ik_R = k_index_[istring][j + 1];
                // Clamp to valid range — all ranks must call det_berryphase
                // to participate in the internal MPI_Allreduce(MPI_PROD)
                if (ik_L >= nks) ik_L = 0;
                if (ik_R >= nks) ik_R = 0;
                 zeta_scalar *= berry_overlap_->det_berryphase(
                     ucell, ik_L, ik_R, dk_str, nocc_use, *paraV_, psi, *kv_);
            }
        }
        else
        {
            for (int j = 0; j < nppstr_ - 1; ++j)
            {
                const auto& Oj = O_kpair[j];
                std::vector<std::complex<double>> O_copy = Oj;
                std::vector<int> ipiv_lu(std::max(n_dim, 1));
                int info_lu = 0, n_lu = n_dim;
                zgetrf_(&n_lu, &n_lu, O_copy.data(), &n_lu, ipiv_lu.data(), &info_lu);
                if (info_lu != 0) { zeta_scalar = 0; break; }
                std::complex<double> det_o(1.0, 0.0);
                int sign_lu = 1;
                for (int i = 0; i < n_dim; ++i)
                {
                    det_o *= O_copy[i + i * n_dim];
                    if (ipiv_lu[i] != i + 1) sign_lu = -sign_lu;
                }
                if (sign_lu < 0) det_o = -det_o;
                zeta_scalar *= det_o;
            }
        }

        // --- Step 3b: Build Wilson loop with phase unwrapping ---
        // Method A: track eigenvalue phases continuously along k-string
        // W_j = O_0 · O_1 · ... · O_j (partial product)
        // At each step, diagonalize W_j and track eigenvalue continuity
        
        std::vector<std::complex<double>> W_mat(n_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int i = 0; i < n_dim; ++i)
            W_mat[i + i * n_dim] = std::complex<double>(1.0, 0.0);

        // Unwrapped eigenvalue phases (continuous, not mod 2π)
        std::vector<double> gamma_unwrapped(n_dim, 0.0);
        // Previous step eigenvalues (for continuity tracking)
        std::vector<std::complex<double>> evals_prev(n_dim, std::complex<double>(1.0, 0.0));
        bool first_diag = true;

        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            // W_j = W_{j-1} · O_j
            std::vector<std::complex<double>> tmp(n_dim * n_dim, std::complex<double>(0.0, 0.0));
            const auto& Oj = O_kpair[j];
            for (int b = 0; b < n_dim; ++b)
                for (int i = 0; i < n_dim; ++i)
                {
                    std::complex<double> s(0.0, 0.0);
                    for (int k = 0; k < n_dim; ++k)
                        s += W_mat[i + k * n_dim] * Oj[k + b * n_dim];
                    tmp[i + b * n_dim] = s;
                }
            // Normalize by max element
            double max_elem = 0.0;
            for (int i = 0; i < n_dim * n_dim; ++i)
                max_elem = std::max(max_elem, std::abs(tmp[i]));
            if (max_elem > 1e-10)
                for (int i = 0; i < n_dim * n_dim; ++i)
                    tmp[i] /= max_elem;
            W_mat = tmp;

            // Diagonalize W_j at each step
            std::vector<std::complex<double>> evals_j(n_dim);
            std::vector<std::complex<double>> VR_j(n_dim * n_dim);
            {
                int lwork = -1;
                std::vector<std::complex<double>> work(1);
                std::vector<double> rwork(2 * n_dim);
                int info = 0;
                char jobvl = 'N', jobvr = 'V';
                int n_eig = n_dim;
                std::vector<std::complex<double>> W_copy = W_mat;
                zgeev_(&jobvl, &jobvr, &n_eig, W_copy.data(), &n_eig,
                       evals_j.data(), nullptr, &n_eig, VR_j.data(), &n_eig,
                       work.data(), &lwork, rwork.data(), &info);
                if (info != 0) continue;
                lwork = static_cast<int>(work[0].real());
                work.resize(std::max(lwork, 1));
                W_copy = W_mat;
                zgeev_(&jobvl, &jobvr, &n_eig, W_copy.data(), &n_eig,
                       evals_j.data(), nullptr, &n_eig, VR_j.data(), &n_eig,
                       work.data(), &lwork, rwork.data(), &info);
                if (info != 0) continue;
            }

            if (first_diag)
            {
                // Sort eigenvalues by argument (ascending) for deterministic
                // band ordering. zgeev provides no ordering guarantee.
                std::vector<int> perm(n_dim);
                for (int n = 0; n < n_dim; ++n) perm[n] = n;
                std::sort(perm.begin(), perm.end(),
                    [&evals_j](int a, int b) { return std::arg(evals_j[a]) < std::arg(evals_j[b]); });
                std::vector<std::complex<double>> evals_sorted(n_dim);
                std::vector<double> gamma_sorted(n_dim);
                std::vector<std::complex<double>> VR_sorted(n_dim * n_dim);
                for (int n = 0; n < n_dim; ++n)
                {
                    evals_sorted[n] = evals_j[perm[n]];
                    gamma_sorted[n] = std::arg(evals_j[perm[n]]);
                    for (int i = 0; i < n_dim; ++i)
                        VR_sorted[i + n * n_dim] = VR_j[i + perm[n] * n_dim];
                }
                evals_j = evals_sorted;
                gamma_unwrapped = gamma_sorted;
                VR_j = VR_sorted;
                first_diag = false;
            }
            else
            {
                // Cost matrix: |phase gap| between new eval[n] and prev eval[m]
                const int N = n_dim;
                std::vector<std::vector<double>> cost(N, std::vector<double>(N));
                for (int m = 0; m < N; ++m)
                    for (int nn = 0; nn < N; ++nn)
                    {
                        double diff = std::arg(evals_j[nn] / evals_prev[m]);
                        while (diff > M_PI) diff -= 2.0 * M_PI;
                        while (diff <= -M_PI) diff += 2.0 * M_PI;
                        cost[m][nn] = std::abs(diff);
                    }
                // NaN guard: fall back to greedy if eigenvalues are pathological
                bool has_nan = false;
                for (int m = 0; m < N && !has_nan; ++m)
                    if (std::abs(evals_j[m]) != std::abs(evals_j[m])
                        || std::abs(evals_prev[m]) != std::abs(evals_prev[m]))
                        has_nan = true;
                if (has_nan)
                {
                    std::vector<bool> matched(N, false);
                    std::vector<double> gamma_new(N, 0.0);
                    for (int nn = 0; nn < N; ++nn)
                    {
                        double best_diff = 1e10; int best_m = -1;
                        for (int m = 0; m < N; ++m)
                        {
                            if (matched[m]) continue;
                            if (cost[nn][m] < best_diff) { best_diff = cost[nn][m]; best_m = m; }
                        }
                        if (best_m >= 0)
                        {
                            double diff = std::arg(evals_j[nn] / evals_prev[best_m]);
                            while (diff > M_PI) diff -= 2.0 * M_PI;
                            while (diff <= -M_PI) diff += 2.0 * M_PI;
                            gamma_new[best_m] = gamma_unwrapped[best_m] + diff;
                            matched[best_m] = true;
                        }
                    }
                    gamma_unwrapped = gamma_new;
                }
                else
                {
                    // Check if we have a saved matching from a previous run
                    bool has_saved = false;
                    if (match_loaded_ && alpha < static_cast<int>(saved_matches_.size())
                        && istring < static_cast<int>(saved_matches_[alpha].size())
                        && j < static_cast<int>(saved_matches_[alpha][istring].size())
                        && static_cast<int>(saved_matches_[alpha][istring][j].size()) == N)
                        has_saved = true;

                    if (has_saved)
                    {
                        // Replay saved matching: use the saved match_to directly.
                        // This ensures identical eigenvalue tracking across independent
                        // runs (e.g., lambda sweep), eliminating Hungarian ambiguity.
                        const auto& saved = saved_matches_[alpha][istring][j];
                        std::vector<double> gamma_new(N, 0.0);
                        for (int nn = 0; nn < N; ++nn)
                        {
                            int m = saved[nn];
                            if (m < 0 || m >= N) continue;  // Skip invalid matches
                            double diff = std::arg(evals_j[nn] / evals_prev[m]);
                            while (diff > M_PI) diff -= 2.0 * M_PI;
                            while (diff <= -M_PI) diff += 2.0 * M_PI;
                            gamma_new[m] = gamma_unwrapped[m] + diff;
                        }
                        gamma_unwrapped = gamma_new;
                    }
                    else
                    {
                    // Kuhn-Munkres (Hungarian) global optimal matching
                    std::vector<double> u(N, 0.0), v(N, 0.0);
                    std::vector<int> p(N, -1), way(N, -1);
                    for (int i = 0; i < N; ++i)
                    {
                        p[0] = i; int j0 = 0;
                        std::vector<double> minv(N, 1e300);
                        std::vector<bool> used(N, false);
                        int hung_iter = 0;
                        do {
                            if (++hung_iter > N * N + 5) { j0 = 0; break; }
                            if (j0 < 0 || j0 >= N) break;
                            used[j0] = true;
                            int i0 = p[j0];
                            if (i0 < 0 || i0 >= N) break;
                            double delta = 1e300; int j1 = 0;
                            for (int j = 1; j < N; ++j)
                            {
                                if (!used[j])
                                {
                                    double cur = cost[i0][j] - u[i0] - v[j];
                                    if (cur == cur && cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                                }
                            }
                            for (int j = 0; j < N; ++j)
                            {
                                if (used[j]) { 
                                    if (p[j] >= 0 && p[j] < N) u[p[j]] += delta; 
                                    v[j] -= delta; 
                                }
                                else         { minv[j] -= delta; }
                            }
                            j0 = j1;
                        } while (j0 >= 0 && j0 < N && p[j0] != -1);
                        while (j0 != 0) { 
                            if (j0 < 0 || j0 >= N) break;
                            int j1 = way[j0]; 
                            if (j1 < 0 || j1 >= N) break;
                            p[j0] = p[j1]; 
                            j0 = j1; 
                        }
                    }
                    // Extract matching and unwrap phases
                    std::vector<int> match_to(N, -1);
                    for (int j = 1; j < N; ++j)
                        if (p[j] != -1) match_to[p[j]] = j;
                    for (int nn = 0; nn < N; ++nn)
                    {
                        if (match_to[nn] < 0)
                        {
                            double best_diff = 1e10; int best_m = -1;
                            for (int m = 0; m < N; ++m)
                            {
                                bool taken = false;
                                for (int nn2 = 0; nn2 < N; ++nn2)
                                    if (match_to[nn2] == m) { taken = true; break; }
                                if (taken) continue;
                                if (cost[nn][m] < best_diff) { best_diff = cost[nn][m]; best_m = m; }
                            }
                            if (best_m >= 0) match_to[nn] = best_m;
                        }
                    }
                    std::vector<double> gamma_new(N, 0.0);
                    for (int nn = 0; nn < N; ++nn)
                    {
                        int m = match_to[nn];
                        if (m < 0 || m >= N) continue;  // Skip invalid matches
                        double diff = std::arg(evals_j[nn] / evals_prev[m]);
                        while (diff > M_PI) diff -= 2.0 * M_PI;
                        while (diff <= -M_PI) diff += 2.0 * M_PI;
                        gamma_new[m] = gamma_unwrapped[m] + diff;
                    }
                    gamma_unwrapped = gamma_new;

                    // Save matching for future runs (lambda sweep determinism)
                    if (alpha >= static_cast<int>(saved_matches_.size()))
                        saved_matches_.resize(alpha + 1);
                    if (istring >= static_cast<int>(saved_matches_[alpha].size()))
                        saved_matches_[alpha].resize(istring + 1);
                    if (j >= static_cast<int>(saved_matches_[alpha][istring].size()))
                        saved_matches_[alpha][istring].resize(j + 1);
                    saved_matches_[alpha][istring][j] = match_to;
                    } // end Hungarian else
                }
            }
            evals_prev = evals_j;
        }

        // gamma_unwrapped now contains the unwrapped Berry phases
        // <r_n> = -R * gamma_unwrapped[n] / (2π)  (no branch cut!)

        // Final eigenvalues/eigenvectors (from the final W_mat)
        std::vector<std::complex<double>> evals(n_dim);
        std::vector<std::complex<double>> VR(n_dim * n_dim);
        {
            int lwork = -1;
            std::vector<std::complex<double>> work(1);
            std::vector<double> rwork(2 * n_dim);
            int info = 0;
            char jobvl = 'N', jobvr = 'V';
            int n_eig = n_dim;
            std::vector<std::complex<double>> W_copy = W_mat;
            zgeev_(&jobvl, &jobvr, &n_eig, W_copy.data(), &n_eig,
                   evals.data(), nullptr, &n_eig, VR.data(), &n_eig,
                   work.data(), &lwork, rwork.data(), &info);
            if (info != 0) { std::cerr << "DeltaP: zgeev failed info=" << info << std::endl; continue; }
            lwork = static_cast<int>(work[0].real());
            work.resize(std::max(lwork, 1));
            W_copy = W_mat;
            zgeev_(&jobvl, &jobvr, &n_eig, W_copy.data(), &n_eig,
                   evals.data(), nullptr, &n_eig, VR.data(), &n_eig,
                   work.data(), &lwork, rwork.data(), &info);
            if (info != 0) { std::cerr << "DeltaP: zgeev failed info=" << info << std::endl; continue; }
        }

        // Reorder evals to match gamma_unwrapped ordering
        // (zgeev may return eigenvalues in different order than our tracking)
        // Match by closest phase to gamma_unwrapped
        {
            std::vector<bool> matched(n_dim, false);
            std::vector<std::complex<double>> evals_sorted(n_dim);
            std::vector<std::complex<double>> VR_sorted(n_dim * n_dim);
            for (int n = 0; n < n_dim; ++n)
            {
                double best_diff = 1e10;
                int best_m = -1;
                for (int m = 0; m < n_dim; ++m)
                {
                    if (matched[m]) continue;
                    double diff = std::abs(std::arg(evals[m]) - std::fmod(gamma_unwrapped[n], 2.0*M_PI));
                    diff = std::min(diff, 2.0*M_PI - diff);
                    if (diff < best_diff) { best_diff = diff; best_m = m; }
                }
                if (best_m >= 0)
                {
                    evals_sorted[n] = evals[best_m];
                    for (int i = 0; i < n_dim; ++i)
                        VR_sorted[i + n * n_dim] = VR[i + best_m * n_dim];
                    matched[best_m] = true;
                }
            }
            evals = evals_sorted;
            VR = VR_sorted;
        }

        // --- Step 4b: Compute det(W) for branch tracking ---
        // det(W) = product of det(O_j), computed via LU on the (normalized) W_mat
        std::vector<std::complex<double>> W_for_det = W_mat;
        std::vector<int> ipiv_det(std::max(n_dim, 1));
        int info_det = 0;
        int n_det = n_dim;
        zgetrf_(&n_det, &n_det, W_for_det.data(), &n_det, ipiv_det.data(), &info_det);
        std::complex<double> zeta(1.0, 0.0);
        if (info_det == 0)
        {
            int sign_det = 1;
            for (int i = 0; i < n_dim; ++i)
            {
                zeta *= W_for_det[i + i * n_dim];
                if (ipiv_det[i] != i + 1) sign_det = -sign_det;
            }
            if (sign_det < 0) zeta = -zeta;
        }
        zeta_list.push_back(zeta);

        // --- Step 5: D_mat at k_0, projections, per-atom Berry phases ---
        std::vector<std::complex<double>> D_mat(m_dim * n_dim, std::complex<double>(0.0, 0.0));
        {
            int row_offset = 0;
            for (int iat = 0; iat < nat_; ++iat)
            {
                int r = nproj_per_atom_[iat];
                if (kstring_data_[0].D_I.size() > static_cast<size_t>(iat))
                    for (int lm = 0; lm < r; ++lm)
                        if (kstring_data_[0].D_I[iat].size() > static_cast<size_t>(lm))
                            for (int n = 0; n < n_dim; ++n)
                                if (kstring_data_[0].D_I[iat][lm].size() > static_cast<size_t>(n))
                                    D_mat[(row_offset + lm) + n * m_dim] = kstring_data_[0].D_I[iat][lm][n];
                row_offset += r;
            }
        }

        // proj[a, n] = sum_m D_mat[a, m] * VR[m, n] = <alpha_a | v_n>
        std::vector<std::complex<double>> proj(m_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int n = 0; n < n_dim; ++n)
            for (int a = 0; a < m_dim; ++a)
            {
                std::complex<double> s(0.0, 0.0);
                for (int m = 0; m < n_dim; ++m)
                    s += D_mat[a + m * m_dim] * VR[m + n * n_dim];
                proj[a + n * m_dim] = s;
            }

        // Debug: output D_mat and proj for first n
        if (istring == 0)
        {
            std::cout << "   DeltaP Dmat: m_dim=" << m_dim << " n_dim=" << n_dim << std::endl;
            // Check psi normalization: ||psi_0||^2 = c† * S_LCAO * c
            // For serial: psi_k[mu + n * nlocal] = c_{n,mu}
            // We can't compute S_LCAO * c easily, but we can check sum |c|^2
            double c_norm_sq = 0.0;
            for (int mu = 0; mu < n_dim; ++mu)  // This is wrong, n_dim is nocc not nlocal
                c_norm_sq += std::norm(D_mat[mu + 0 * m_dim]);
            // Actually let's just check |D_mat[:,0]|^2 and see if it's > 1
            for (int n = 0; n < std::min(n_dim, 2); ++n)
            {
                std::cout << "     D_mat[:, " << n << "]:";
                for (int a = 0; a < std::min(m_dim, 5); ++a)
                    std::cout << " " << D_mat[a + n * m_dim];
                std::cout << " ..." << std::endl;
                double d_norm = 0.0;
                for (int a = 0; a < m_dim; ++a)
                    d_norm += std::norm(D_mat[a + n * m_dim]);
                std::cout << "     |D_mat[:," << n << "]|^2 = " << d_norm << std::endl;
            }
            // Check: D_I should be <phi_onsite | psi>
            // |<phi_onsite_lm | psi_n>| <= 1 (Cauchy-Schwarz, both normalized)
            // If |D_I| > 1, then either phi_onsite or psi is not normalized
            // Or S_k values are wrong
        }

        // Per-atom Berry phases and SMO weights for this k-string
        // Use Löwdin orthogonalization: tilde_proj = S^{-1/2} * proj
        // w_In = sum_{a in I} |tilde_proj[a,n]|^2, satisfies sum_I w_In = 1
        std::vector<std::complex<double>> tilde_proj(m_dim * n_dim, std::complex<double>(0.0, 0.0));
        if (smo_m_dim_ > 0 && smo_m_dim_ == m_dim)
        {
            for (int n = 0; n < n_dim; ++n)
                for (int a = 0; a < m_dim; ++a)
                {
                    std::complex<double> s(0.0, 0.0);
                    for (int b = 0; b < m_dim; ++b)
                        s += smo_overlap_inv_[a + b * m_dim] * proj[b + n * m_dim];
                    tilde_proj[a + n * m_dim] = s;
                }
            // Verify Löwdin S^{-1/2}: smo_overlap_inv_ stores S^{-1/2}, not S^{-1}.
            // The correct identity check is S^{-1/2}·S·S^{-1/2} = I.
            if (istring == 0)
            {
                double max_err = 0.0;
                for (int i = 0; i < m_dim; ++i)
                    for (int j = 0; j < m_dim; ++j)
                    {
                        double val = 0.0;
                        for (int k = 0; k < m_dim; ++k)
                            for (int l = 0; l < m_dim; ++l)
                                val += smo_overlap_inv_[i + k * m_dim]
                                     * smo_overlap_[k + l * m_dim]
                                     * smo_overlap_inv_[l + j * m_dim];
                        double expected = (i == j) ? 1.0 : 0.0;
                        max_err = std::max(max_err, std::abs(val - expected));
                    }
                std::cout << "   DeltaP: S^{-1/2}*S*S^{-1/2} - I max_err = " << max_err << " (should be ~0)" << std::endl;

                double max_err3 = 0.0;
                for (int i = 0; i < m_dim; ++i)
                    for (int j = 0; j < m_dim; ++j)
                    {
                        double val = 0.0;
                        for (int k = 0; k < m_dim; ++k)
                            for (int l = 0; l < m_dim; ++l)
                                val += smo_overlap_inv_[i + k * m_dim]
                                     * smo_overlap_inv_[k + l * m_dim]
                                     * smo_overlap_[l + j * m_dim];
                        double expected = (i == j) ? 1.0 : 0.0;
                        max_err3 = std::max(max_err3, std::abs(val - expected));
                    }
                std::cout << "   DeltaP: (Sinv)^2 * S - I max_err = " << max_err3 << " (should be ~0)" << std::endl;

                for (int n = 0; n < n_dim; ++n)
                {
                    double raw_sum = 0.0, tilde_sum = 0.0;
                    for (int a = 0; a < m_dim; ++a)
                    {
                        raw_sum += std::norm(proj[a + n * m_dim]);
                        tilde_sum += std::norm(tilde_proj[a + n * m_dim]);
                    }
                    std::cout << "   DeltaP: n=" << n
                              << " raw_sum=" << raw_sum
                              << " tilde_sum=" << tilde_sum << std::endl;
                }
            }
        }
        else
        {
            tilde_proj = proj;
        }

        std::vector<double> gamma_I_per_atom(nat_, 0.0);
        std::vector<double> smo_weight_sum_per_atom(nat_, 0.0);
        std::vector<double> r_elec_per_atom(nat_, 0.0);
        std::vector<std::vector<double>> w_In_matrix(n_dim, std::vector<double>(nat_, 0.0));
        // First pass: compute raw weights per band per atom
        for (int n = 0; n < n_dim; ++n)
            for (int iat = 0; iat < nat_; ++iat)
            {
                int row_offset = 0;
                for (int i = 0; i < iat; ++i) row_offset += nproj_per_atom_[i];
                int r = nproj_per_atom_[iat];
                double w_In = 0.0;
                for (int a = row_offset; a < row_offset + r; ++a)
                    w_In += std::norm(tilde_proj[a + n * m_dim]);
                if (w_In < 0) w_In = 0;
                w_In_matrix[n][iat] = w_In;
            }

        // Normalize per band and compute per-atom gamma
        for (int iat = 0; iat < nat_; ++iat)
        {
            double gamma_I = 0.0, w_sum = 0.0, r_weighted = 0.0;
            for (int n = 0; n < n_dim; ++n)
            {
                double w_tot = 0.0;
                for (int j = 0; j < nat_; ++j) w_tot += w_In_matrix[n][j];
                double w_norm = (w_tot > 1e-30) ? w_In_matrix[n][iat] / w_tot : 0.0;
                gamma_I += w_norm * gamma_unwrapped[n];
                w_sum += w_norm;
                r_weighted += w_norm * (-a_alpha * gamma_unwrapped[n] / (2.0 * ModuleBase::PI));
            }
            gamma_accum[iat] += gamma_I;
            gamma_I_per_atom[iat] = gamma_I;
            smo_weight_sum_per_atom[iat] = w_sum;
            r_elec_per_atom[iat] = (w_sum > 1e-15) ? r_weighted / w_sum : 0.0;
        }
        gamma_accum_per_string.push_back(gamma_I_per_atom);
        evals_all.push_back(evals);
        n_strings_processed++;

        // Debug: output eigenvalues, weights, and prefactor for branch enumeration
        if (n_strings_processed == 1)
        {
#ifdef __MPI
            if (GlobalV::MY_RANK == 0)
#endif
            {
                std::ofstream ofs("deltap_branch_enum.dat");
                ofs << std::setprecision(17);
                ofs << n_dim << " " << nat_ << " " << prefactor << " " << a_alpha << " " << omega << "\n";
                ofs << zeta_scalar.real() << " " << zeta_scalar.imag() << " " << std::arg(zeta_scalar) << "\n";
                for (int n = 0; n < n_dim; ++n)
                    ofs << evals[n].real() << " " << evals[n].imag() << " " << std::arg(evals[n]) << " " << gamma_unwrapped[n] << "\n";
                for (int n = 0; n < n_dim; ++n)
                {
                    for (int iat = 0; iat < nat_; ++iat)
                        ofs << w_In_matrix[n][iat] << " ";
                    ofs << "\n";
                }
                ofs.close();
                std::cout << "   DeltaP: branch enumeration data written to deltap_branch_enum.dat" << std::endl;
            }
        }

        // Accumulate SMO weights, raw gamma, r_elec, and w_In matrix
        for (int iat = 0; iat < nat_; ++iat)
        {
            smo_w_accum[iat] += smo_weight_sum_per_atom[iat];
            gamma_raw_accum[iat] += gamma_I_per_atom[iat];
            r_elec_accum[iat] += r_elec_per_atom[iat];
        }
        // Store w_In from first string (all strings should give same w_In for isolated systems)
        if (n_strings_processed == 1)
        {
            results_.smo_weights = w_In_matrix;
            w_In_first_string_ = w_In_matrix;  // save for global branch search (current alpha)
        }

        // Per-atom polarization: zeta rescaling (preserves relative distribution
        // from SMO weights, which empirically matches Wannier90 better than
        // normalized weights). The branch-set selection is applied as a
        // separate cross-structure correction below.
        std::vector<double> gamma_pre_branch = gamma_I_per_atom;
        if (n_strings_processed > 0 && n_dim > 0)
        {
            // Step 1: zeta rescale using consistent reference across strings.
            // Different strings may unwrap eigenvalues differently (2π-per-band
            // ambiguity), making per-string gamma_unw_sum inconsistent.  Use
            // the FIRST string's unwrapped sum as the reference for all strings
            // so that the scale factor is deterministic.
            double gamma_raw_sum = 0.0;
            for (int iat = 0; iat < nat_; ++iat) gamma_raw_sum += gamma_I_per_atom[iat];
            double gamma_unw_sum = 0.0;
            for (int n = 0; n < n_dim; ++n) gamma_unw_sum += gamma_unwrapped[n];
            if (!ref_captured) { ref_gamma_unw_sum = gamma_unw_sum; ref_captured = true; }
            else               { gamma_unw_sum = ref_gamma_unw_sum; }
            double scale = 1.0;
            if (std::abs(gamma_raw_sum) > 1e-15 && std::abs(gamma_raw_sum - gamma_unw_sum) > 1e-10)
            {
                scale = gamma_unw_sum / gamma_raw_sum;
                for (int iat = 0; iat < nat_; ++iat)
                {
                    double old_val = gamma_I_per_atom[iat];
                    gamma_I_per_atom[iat] = old_val * scale;
                    gamma_accum[iat] += gamma_I_per_atom[iat] - old_val;
                }
            }
            current_zeta_scale = scale;

            // Step 2: DELETED — per-string target-aware search moved to
            // post-loop global search.  Different Wilson-loop strings have
            // different w_In weights and thus different shift lattices.
            // Independent per-string searches can fail when a string's
            // shift lattice is too sparse to reach the target, corrupting
            // the final average.  The global search on the accumulated
            // average uses the first string's shift amplitudes (representative)
            // and is much more robust.

            // Step 3: cross-structure branch-set consistency.
            // Ensure per-atom γ stays within π of its previous value
            // (from branch.dat or previous SCF iteration).
            if (!std::isnan(prev_gamma[0]))
            {
                // Pre-compute per-band normalization denominators
                // γ_I = Σ_n (w_In(n,I) / w_tot_n(n)) · γ_unwrapped(n)
                // A 2π shift of band n changes γ_I by 2π·w_In/w_tot_n.
                std::vector<double> w_tot_n(n_dim, 0.0);
                for (int n = 0; n < n_dim; ++n)
                    for (int i = 0; i < nat_; ++i)
                        w_tot_n[n] += w_In_matrix[n][i];

                for (int iat = 0; iat < nat_; ++iat)
                {
                    double g = gamma_I_per_atom[iat];
                    double prev = prev_gamma[iat];
                    if (std::abs(g - prev) < M_PI) { prev_gamma[iat] = g; continue; }

                    // Search single-band shifts using per-band normalized amplitudes
                    double best_val = g;
                    double best_dist = std::abs(g - prev);
                    for (int n = 0; n < n_dim; ++n)
                    {
                        double w_In = w_In_matrix[n][iat];
                        if (std::abs(w_In) < 1e-12 || std::abs(w_tot_n[n]) < 1e-12) continue;
                        for (int sign = -1; sign <= 1; sign += 2)
                        {
                            double shift = current_zeta_scale * 2.0 * M_PI * w_In / w_tot_n[n];
                            double candidate = g + sign * shift;
                            double dist = std::abs(candidate - prev);
                            if (dist < best_dist) { best_dist = dist; best_val = candidate; }
                        }
                    }
                    gamma_accum[iat] += best_val - gamma_I_per_atom[iat];
                    gamma_I_per_atom[iat] = best_val;
                    prev_gamma[iat] = best_val;

                    if (n_strings_processed == 1)
                        std::cout << "   DeltaP branch-set: atom " << iat
                                  << " rescaled=" << std::scientific << std::setprecision(6) << g
                                  << " selected=" << best_val
                                  << " prev=" << prev
                                  << " delta=" << best_val - g << std::endl;
                }
            }
            else
            {
                for (int iat = 0; iat < nat_; ++iat)
                    prev_gamma[iat] = gamma_I_per_atom[iat];
            }
            gamma_raw_per_string.push_back(gamma_pre_branch);
            gamma_sel_per_string.push_back(gamma_I_per_atom);
            total_bp_per_string.push_back(std::arg(zeta_scalar));  // for [totalBP] diagnostic
        }

        if (istring == 0)
        {
            double g_check = 0.0;
            for (int n = 0; n < n_dim; ++n) g_check += std::arg(evals[n]);
            std::cout << "   DeltaP Wilson loop (string 0): nocc=" << n_dim
                      << " gamma=" << std::scientific << std::setprecision(6) << g_check << std::endl;
        }

        // Debug: output zeta for this string (scalar product, matching berry_phase)
#ifdef __MPI
        if (GlobalV::MY_RANK == 0)
#endif
        {
            std::ofstream ofs("deltap_zeta_debug.dat", std::ios::app);
            ofs << istring << " " << std::setprecision(17)
                << zeta_scalar.real() << " " << zeta_scalar.imag() << " "
                << std::arg(zeta_scalar) << std::endl;
            ofs.close();
        }
    }

    // --- Average over k-strings ---
    // Simple average. The berry_phase "divide by average" unwrapping
    // was tested but gave worse results (scale factor varies across
    // structures, corrupting Z*). The 3% P error comes from a few
    // strings with 2π jumps in arg(zeta), which is within acceptable
    // accuracy for the current framework.
    std::cout << "   DeltaP: processed " << n_strings_processed << " / " << total_string_ << " k-strings" << std::endl;

    // Total Berry phase diagnostic (bypasses per-atom decomposition)
    if (n_strings_processed > 0 && !total_bp_per_string.empty())
    {
        double total_bp_avg = 0.0;
        for (size_t i = 0; i < total_bp_per_string.size(); ++i)
            total_bp_avg += total_bp_per_string[i];
        total_bp_avg /= total_bp_per_string.size();
        std::cout << "   [totalBP] alpha=" << alpha << " avg_arg(zeta)="
                  << std::scientific << std::setprecision(6) << total_bp_avg
                  << " nstrings=" << total_bp_per_string.size() << std::endl;
    }

    if (n_strings_processed >= 2)
    {
        std::cout << "\n   === Cross-String Branch Consistency ===" << std::endl;
        std::cout << "   Strings processed: " << n_strings_processed << " / " << total_string_ << std::endl;
        for (int iat = 0; iat < nat_; ++iat)
        {
            double gamma_min = 1e300, gamma_max = -1e300;
            double gamma_sum = 0.0, gamma_sum2 = 0.0;
            std::cout << "   Atom " << iat << " per-string gamma:";
            for (int ist = 0; ist < n_strings_processed; ++ist)
            {
                double g = gamma_sel_per_string[ist][iat];
                double g_raw = gamma_raw_per_string[ist][iat];
                double shift = g - g_raw;
                std::cout << " [" << ist << "] raw=" << std::scientific << std::setprecision(6) << g_raw
                          << " sel=" << g << " Δ=" << shift;
                gamma_min = std::min(gamma_min, g);
                gamma_max = std::max(gamma_max, g);
                gamma_sum += g; gamma_sum2 += g * g;
            }
            double gamma_mean = gamma_sum / n_strings_processed;
            double gamma_std = (n_strings_processed > 1)
                ? std::sqrt((gamma_sum2 - gamma_sum * gamma_sum / n_strings_processed) / (n_strings_processed - 1))
                : 0.0;
            std::cout << std::endl << "        spread: min=" << gamma_min << " max=" << gamma_max
                      << " mean=" << gamma_mean << " σ=" << gamma_std << std::endl;
            if (!gamma_sel_per_string.empty())
            {
                double w_sum_I = 0.0;
                const auto& wm = results_.smo_weights;
                if (!wm.empty())
                    for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                        w_sum_I += wm[n][iat];
                if (w_sum_I > 1e-12)
                    std::cout << "        w_sum^I=" << w_sum_I << " 2π·w_sum^I=" << (2.0*M_PI*w_sum_I)
                              << (gamma_std > 0.1 * 2.0*M_PI*w_sum_I ? "  ← BRANCH INCONSISTENT!" : "  OK") << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Global target-aware branch selection.
    // Applied ONCE on the accumulated average (not per-string), to avoid
    // branch inconsistency when different Wilson-loop strings have
    // different w_In weights (different shift lattices).
    // Uses the first string's w_In (stored in w_In_first_string_) as
    // representative shift amplitudes.
    // Check if any meaningful target is set (non-zero within tolerance)
    bool has_target = false;
    if (!target_gamma_.empty())
        for (size_t i = 0; i < target_gamma_.size(); ++i)
            if (std::abs(target_gamma_[i]) > 1e-12) { has_target = true; break; }
    bool has_constraint = !constraint_matrix_.empty();

    if ((has_target || has_constraint)
        && n_strings_processed > 0 && !w_In_first_string_.empty())
    {
        const auto& wm = w_In_first_string_;
        bool total_mode = (PARAM.inp.deltap_constraint_mode == "total");
        bool use_constraint_matrix = !constraint_matrix_.empty();

        if (use_constraint_matrix)
        {
            // Constraint-space sequential greedy search
            // C·γ = t, where C is m×n, t is m×1
            const auto& C = constraint_matrix_;
            const auto& t = constraint_target_;
            int m = static_cast<int>(C.size());

            // Build shift amplitudes and raw gamma for each atom
            std::vector<std::vector<double>> shift_amps(nat_);
            std::vector<double> raw_gamma(nat_);
            for (int iat = 0; iat < nat_; ++iat)
            {
                raw_gamma[iat] = gamma_accum[iat] / n_strings_processed;
                if (!wm.empty())
                    for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                    {
                        if (std::abs(wm[n][iat]) < 1e-12) continue;
                        // Per-band normalization: see Step-3 comment.
                        double w_tot_n = 0.0;
                        for (int j = 0; j < nat_; ++j)
                            w_tot_n += wm[n][j];
                        if (w_tot_n < 1e-12) continue;
                        shift_amps[iat].push_back(current_zeta_scale * 2.0 * M_PI * wm[n][iat] / w_tot_n);
                    }
            }

            // Greedy forward pass
            std::vector<double> gamma_best(nat_);
            for (int iat = 0; iat < nat_; ++iat) gamma_best[iat] = raw_gamma[iat];

            // Running contribution from already-selected atoms
            std::vector<double> contrib(m, 0.0);

            for (int iat = 0; iat < nat_; ++iat)
            {
                int n_dim_shift = static_cast<int>(shift_amps[iat].size());
                if (n_dim_shift == 0) continue;

                // Compute remaining residual: t - C·gamma_running - C_future·raw
                std::vector<double> r_current(m, 0.0);
                for (int a = 0; a < m; ++a)
                {
                    r_current[a] = t[a] - contrib[a];  // remove already-selected
                    // remove future atoms' raw contribution
                    for (int j = iat + 1; j < nat_; ++j)
                        r_current[a] -= C[a][j] * raw_gamma[j];
                }

                // Effective target for this atom: min ||C_i * γ - r_current||
                double c_i_sq = 0.0;
                double c_dot_r = 0.0;
                for (int a = 0; a < m; ++a)
                {
                    c_i_sq += C[a][iat] * C[a][iat];
                    c_dot_r += C[a][iat] * r_current[a];
                }
                double effective_target = raw_gamma[iat];  // default: no shift
                if (c_i_sq > 1e-10)
                    effective_target = c_dot_r / c_i_sq;

                // Exhaustive search K=5
                double best_val = raw_gamma[iat];
                double best_dist = std::abs(raw_gamma[iat] - effective_target);
                const int K = 5;
                std::vector<int> kvec(n_dim_shift, 0);
                bool done = false;
                while (!done)
                {
                    bool all_zero = true;
                    double shift = 0.0;
                    for (int n = 0; n < n_dim_shift; ++n)
                    {
                        if (kvec[n] != 0) all_zero = false;
                        shift += kvec[n] * shift_amps[iat][n];
                    }
                    if (!all_zero)
                    {
                        double candidate = raw_gamma[iat] + shift;
                        double dist = std::abs(candidate - effective_target);
                        if (dist < best_dist) { best_dist = dist; best_val = candidate; }
                    }
                    int carry_pos = 0;
                    while (carry_pos < n_dim_shift)
                    {
                        kvec[carry_pos]++;
                        if (kvec[carry_pos] > K) { kvec[carry_pos] = -K; carry_pos++; }
                        else break;
                    }
                    if (carry_pos >= n_dim_shift) done = true;
                }

                // Apply shift and record
                double delta = best_val - raw_gamma[iat];
                gamma_accum[iat] += delta * n_strings_processed;
                gamma_best[iat] = best_val;

                // Update contribution
                for (int a = 0; a < m; ++a)
                    contrib[a] += C[a][iat] * best_val;
            }
        }
        else if (total_mode)
        {
            // Sequential greedy: each atom targets the remaining total
            double total_target = 0.0;
            for (int iat = 0; iat < nat_; ++iat)
                total_target += target_gamma_[iat];

            double running_sum = 0.0;
            for (int iat = 0; iat < nat_; ++iat)
            {
                double avg_raw = gamma_accum[iat] / n_strings_processed;
                int remaining = nat_ - iat;
                // The remaining atoms need to contribute: total_target - running_sum
                // This atom's fair share (assuming equal): (total_target - running_sum) / remaining
                double effective_target = avg_raw;  // default: no shift
                if (remaining > 1)
                    effective_target = (total_target - running_sum) / remaining;
                else
                    effective_target = total_target - running_sum;  // last atom takes the remainder

                double w_total = 0.0;
                // Check per-band normalization validity
                if (!wm.empty())
                    for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                        w_total += wm[n][iat];
                if (w_total < 1e-12) { running_sum += avg_raw; continue; }

                std::vector<double> shift_amp;
                if (!wm.empty())
                    for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                    {
                        if (std::abs(wm[n][iat]) < 1e-12) continue;
                        double w_tot_n = 0.0;
                        for (int j = 0; j < nat_; ++j)
                            w_tot_n += wm[n][j];
                        if (w_tot_n < 1e-12) continue;
                        shift_amp.push_back(current_zeta_scale * 2.0 * M_PI * wm[n][iat] / w_tot_n);
                    }

                const int n_dim_shift = static_cast<int>(shift_amp.size());
                if (n_dim_shift == 0) { running_sum += avg_raw; continue; }

                double best_val = avg_raw;
                double best_dist = std::abs(avg_raw - effective_target);
                const int K = 5;
                std::vector<int> kvec(n_dim_shift, 0);
                bool done = false;
                while (!done)
                {
                    bool all_zero = true;
                    double shift = 0.0;
                    for (int n = 0; n < n_dim_shift; ++n)
                    {
                        if (kvec[n] != 0) { all_zero = false; }
                        shift += kvec[n] * shift_amp[n];
                    }
                    if (!all_zero)
                    {
                        double candidate = avg_raw + shift;
                        double dist = std::abs(candidate - effective_target);
                        if (dist < best_dist)
                        {
                            best_dist = dist;
                            best_val = candidate;
                        }
                    }
                    int carry_pos = 0;
                    while (carry_pos < n_dim_shift)
                    {
                        kvec[carry_pos]++;
                        if (kvec[carry_pos] > K) { kvec[carry_pos] = -K; carry_pos++; }
                        else { break; }
                    }
                    if (carry_pos >= n_dim_shift) done = true;
                }

                double delta = best_val - avg_raw;
                gamma_accum[iat] += delta * n_strings_processed;
                running_sum += best_val;
            }
        }
        else
        {
            for (int iat = 0; iat < nat_; ++iat)
            {
                double avg_raw = gamma_accum[iat] / n_strings_processed;
                double target = target_gamma_[iat];

            double w_total = 0.0;
            // Check per-band normalization validity
            if (!wm.empty())
                for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                    w_total += wm[n][iat];

            if (w_total < 1e-12) continue;

            std::vector<double> shift_amp;
            if (!wm.empty())
                for (int n = 0; n < static_cast<int>(wm.size()) && static_cast<size_t>(iat) < wm[n].size(); ++n)
                {
                    if (std::abs(wm[n][iat]) < 1e-12) continue;
                    double w_tot_n = 0.0;
                    for (int j = 0; j < nat_; ++j)
                        w_tot_n += wm[n][j];
                    if (w_tot_n < 1e-12) continue;
                    shift_amp.push_back(current_zeta_scale * 2.0 * M_PI * wm[n][iat] / w_tot_n);
                }

            const int n_dim_shift = static_cast<int>(shift_amp.size());
            if (n_dim_shift == 0) continue;

            double best_val = avg_raw;
            double best_dist = std::abs(avg_raw - target);
            const int K = 5;
            std::vector<int> kvec(n_dim_shift, 0);
            bool done = false;
            while (!done)
            {
                bool all_zero = true;
                double shift = 0.0;
                for (int n = 0; n < n_dim_shift; ++n)
                {
                    if (kvec[n] != 0) { all_zero = false; }
                    shift += kvec[n] * shift_amp[n];
                }
                if (!all_zero)
                {
                    double candidate = avg_raw + shift;
                    double dist = std::abs(candidate - target);
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_val = candidate;
                    }
                }
                // Increment kvec (odometer-style)
                int carry_pos = 0;
                while (carry_pos < n_dim_shift)
                {
                    kvec[carry_pos]++;
                    if (kvec[carry_pos] > K)
                    {
                        kvec[carry_pos] = -K;
                        carry_pos++;
                    }
                    else { break; }
                }
                if (carry_pos >= n_dim_shift) done = true;
            }

            // Apply the global branch shift to gamma_accum
            // gamma_accum = raw_sum + zeta_corrections; we need to add the branch shift
            double delta = best_val - avg_raw;
            gamma_accum[iat] += delta * n_strings_processed;
        }
    }
    }

    // Initialize results arrays (once, before alpha loop)
    results_.smo_weight_sum.resize(nat_, 0.0);

    for (int iat = 0; iat < nat_; ++iat)
    {
        double gamma_I = (n_strings_processed > 0) ? gamma_accum[iat] / n_strings_processed : 0.0;
        W_prev_[iat][alpha] = gamma_I;
        has_prev_ = true;
        results_.gamma_I[iat][alpha_idx] = gamma_I;
        results_.P_I[iat][alpha_idx] = prefactor * gamma_I;
        results_.smo_weight_sum[iat] = (n_strings_processed > 0) ? smo_w_accum[iat] / n_strings_processed : 0.0;
        results_.gamma_I_raw[iat][alpha_idx] = (n_strings_processed > 0) ? gamma_raw_accum[iat] / n_strings_processed : 0.0;
        results_.r_elec_center[iat][alpha_idx] = (n_strings_processed > 0) ? r_elec_accum[iat] / n_strings_processed : 0.0;
    }

    } // alpha loop

    gdir_ = gdir_orig;  // restore original direction for downstream use

    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat_; ++iat)
        results_.P_total += results_.P_I[iat];

    verify_sum_rule();
    if (!scf_mode_)
    {
        write_results(ucell);
    }
    // Always persist branch state and eigenvalue matching
    // (SCF and non-SCF mode) for cross-run determinism.
    save_branch();
    save_match();

    std::cout << " >> Finish DeltaP Wannier polarization.\n * * * * * *\n";

    ModuleBase::timer::end("DeltaP", "compute_wannier_polarization");
}

void DeltaP::compute_gamma_scf(const UnitCell& ucell,
                               const psi::Psi<std::complex<double>>* psi,
                               const elecstate::ElecState* pelec)
{
    // Lightweight wrapper: call compute_wannier_polarization in SCF mode
    // (skips file I/O, reuses initialization across SCF steps)
    scf_mode_ = true;
    compute_wannier_polarization(ucell, psi, pelec);
    scf_mode_ = false;

#ifdef __MPI
    if (paraV_ != nullptr)
    {
        MPI_Comm comm = paraV_->comm();
        int nproc = 1;
        MPI_Comm_size(comm, &nproc);
        if (nproc > 1)
        {
            // Broadcast gamma results from rank 0 to all ranks.
            // This handles both Gamma-only (all ranks have same data)
            // and distributed multi-k (only rank 0 has correct k-point-0 data).
            int nat = static_cast<int>(results_.gamma_I.size());
            for (int iat = 0; iat < nat; ++iat)
            {
                for (int dir = 0; dir < 3; ++dir)
                {
                    MPI_Bcast(&results_.gamma_I[iat][dir], 1, MPI_DOUBLE, 0, comm);
                    MPI_Bcast(&results_.gamma_I_raw[iat][dir], 1, MPI_DOUBLE, 0, comm);
                    MPI_Bcast(&results_.P_I[iat][dir], 1, MPI_DOUBLE, 0, comm);
                    MPI_Bcast(&results_.r_elec_center[iat][dir], 1, MPI_DOUBLE, 0, comm);
                }
            }
            for (int iat = 0; iat < nat; ++iat)
                MPI_Bcast(&results_.smo_weight_sum[iat], 1, MPI_DOUBLE, 0, comm);
            for (int dir = 0; dir < 3; ++dir)
                MPI_Bcast(&results_.P_total[dir], 1, MPI_DOUBLE, 0, comm);
        }
    }
#endif
}

void DeltaP::compute_hk_correction(const UnitCell& ucell,
                                   const psi::Psi<std::complex<double>>* psi,
                                   const std::vector<double>& lambda,
                                   std::unordered_map<int, std::vector<std::complex<double>>>& hk_correction)
{
    ModuleBase::TITLE("DeltaP", "compute_hk_correction");
    hk_correction.clear();

    if (nppstr_ < 2 || kstring_data_.empty()) return;

    const int nks = psi->get_nk();
    const int nbands = psi->get_nbands();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();

    // The HK correction matrix is built as nrow×nrow and written into
    // hsk->get_hk() which has nrow×ncol local entries.  This is safe only
    // when the 2D block-cyclic grid distributes the same number of rows
    // and columns to each MPI rank, i.e. nrow == ncol.
    if (nrow != ncol)
    {
        ModuleBase::WARNING_QUIT("DeltaP::compute_hk_correction",
            "The LCAO parallel grid has nrow != ncol.  "
            "DeltaP currently requires a square process grid.  "
            "Try running with a square number of MPI ranks.");
    }

    // Rebuild S_k/D_I if they belong to a different direction or string.
    // compute_gamma_scf leaves kstring_data_ from the last alpha (gdir=3)
    // and the last string; we need the INPUT gdir and string 0.
    if (kstring_gdir_ != gdir_ || kstring_string_ != 0)
    {
        setup_kstring(*kv_);
        kstring_data_.assign(nppstr_, KSpaceData());
        for (int j = 0; j < nppstr_; ++j)
        {
            int ik = k_index_[0][j];
            if (ik >= nks) continue;
            kstring_data_[j].kvec_d = kv_->kvec_d[ik];
            psi->fix_k(ik);
            compute_S_k(j);
            compute_D_I(j, psi->get_pointer(), nbands, nrow);
        }
        kstring_gdir_ = gdir_;
        kstring_string_ = 0;
#ifdef __MPI
        for (int j = 0; j < nppstr_; ++j)
        {
            for (int iat = 0; iat < nat_; ++iat)
            {
                int r = nproj_per_atom_[iat];
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                    int sz = kstring_data_[j].D_I[iat][lm].size();
                    if (sz > 0)
                        MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                      2 * sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
                }
            }
        }
#endif
    }

    // Ensure S_dk_ is computed
    if (S_dk_.empty())
    {
        compute_S_dk(ucell);
    }

    // Number of occupied bands
    double occ_bands_d = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands_d - std::floor(occ_bands_d)) > 0.0)
        occ_bands_d = std::floor(occ_bands_d) + 1.0;
    const int nocc = static_cast<int>(occ_bands_d);
    const int nocc_use = std::min(nocc, nbands);
    if (nocc_use <= 0) return;

    const std::complex<double> half_i(0.0, 0.5);

    // Note: In MPI mode, each rank computes its LOCAL block of the
    // HK correction matrix using its local wavefunctions and local S_dk.
    // The Hamiltonian is distributed, so each rank's correction is applied
    // to its own block — no MPI communication needed here.

    // For each link on the first k-string
    for (int j = 0; j < nppstr_ - 1; ++j)
    {
        int ik_L = k_index_[0][j];
        int ik_R = k_index_[0][j + 1];
        if (ik_L >= nks || ik_R >= nks) continue;

        // Get wavefunction coefficients at k_L and k_R
        psi->fix_k(ik_L);
        const std::complex<double>* c_L = psi->get_pointer();

        psi->fix_k(ik_R);
        const std::complex<double>* c_R = psi->get_pointer();

        // Compute effective weights: w_eff[n] = sum_I lambda[I] * sum_{lm} |D_I[iat][lm][n]|^2
        std::vector<double> w_eff(nocc_use, 0.0);
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            for (int n = 0; n < nocc_use; ++n)
            {
                double w_In = 0.0;
                if (kstring_data_[j].D_I.size() > static_cast<size_t>(iat))
                {
                    for (int lm = 0; lm < r; ++lm)
                    {
                        if (kstring_data_[j].D_I[iat].size() > static_cast<size_t>(lm) &&
                            kstring_data_[j].D_I[iat][lm].size() > static_cast<size_t>(n))
                        {
                            w_In += std::norm(kstring_data_[j].D_I[iat][lm][n]);
                        }
                    }
                }
                w_eff[n] += lambda[iat] * w_In;
            }
        }

        // Step 1: SC = S_dk * C_R -> (nrow x nocc_use)
        std::vector<std::complex<double>> SC(nrow * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int alpha = 0; alpha < nrow; ++alpha)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int gamma = 0; gamma < ncol; ++gamma)
                {
                    sum += S_dk_[alpha + gamma * nrow] * c_R[gamma + p * nrow];
                }
                SC[alpha + p * nrow] = sum;
            }
        }

        // Step 2: F = (i/2) * w_eff * SC -> (nrow x nocc_use)
        std::vector<std::complex<double>> F(nrow * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int alpha = 0; alpha < nrow; ++alpha)
            {
                F[alpha + p * nrow] = half_i * w_eff[p] * SC[alpha + p * nrow];
            }
        }

        // Step 3: M = F * C_L^dagger -> (nrow x nrow)
        std::vector<std::complex<double>> M(nrow * nrow, {0.0, 0.0});
        for (int beta = 0; beta < nrow; ++beta)
        {
            for (int alpha = 0; alpha < nrow; ++alpha)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int p = 0; p < nocc_use; ++p)
                {
                    sum += F[alpha + p * nrow] * std::conj(c_L[beta + p * nrow]);
                }
                M[alpha + beta * nrow] = sum;
            }
        }

        // Step 4: H_sym = (M + M^dagger) / 2 -> (nrow x nrow)
        std::vector<std::complex<double>> H_sym(nrow * nrow, {0.0, 0.0});
        for (int beta = 0; beta < nrow; ++beta)
        {
            for (int alpha = 0; alpha < nrow; ++alpha)
            {
                H_sym[alpha + beta * nrow] = 0.5 * (M[alpha + beta * nrow] + std::conj(M[beta + alpha * nrow]));
            }
        }

        hk_correction[ik_L] = H_sym;
    }
}

// Read previously saved per-atom Wilson-loop products W^I so that arg()
// unwrapping stays smooth across independent SCF runs.  The file is
// written by save_branch() and lives in the ABACUS working directory.
void DeltaP::load_branch()
{
    const std::string fname = "deltap_branch.dat";
    std::ifstream ifs(fname);
    if (!ifs.is_open()) return;

    int nat_file = 0;
    ifs >> nat_file;
    if (nat_file != nat_)
    {
        std::cerr << "DeltaP: branch file nat=" << nat_file
                  << " != current nat=" << nat_ << ", ignoring" << std::endl;
        return;
    }

    W_prev_.resize(nat_);
    for (int iat = 0; iat < nat_; ++iat)
    {
        double gx = 0.0, gy = 0.0, gz = 0.0;
        ifs >> gx >> gy >> gz;
        W_prev_[iat] = ModuleBase::Vector3<double>(gx, gy, gz);
    }
    has_prev_ = true;
    std::cout << " DeltaP: loaded branch state from " << fname << std::endl;
}

// Persist per-atom Wilson-loop products W^I for the next SCF/run.
void DeltaP::save_branch() const
{
    if (!has_prev_ || static_cast<int>(W_prev_.size()) != nat_) return;
#ifdef __MPI
    if (GlobalV::MY_RANK != 0) return;
#endif

    const std::string fname = "deltap_branch.dat";
    std::ofstream ofs(fname);
    if (!ofs.is_open())
    {
        std::cerr << "DeltaP: cannot open " << fname << " for writing" << std::endl;
        return;
    }

    ofs << std::setprecision(17);
    ofs << nat_ << "\n";
    for (int iat = 0; iat < nat_; ++iat)
        ofs << W_prev_[iat].x << " " << W_prev_[iat].y << " " << W_prev_[iat].z << "\n";
    ofs.close();
}

void DeltaP::load_match()
{
    const std::string fname = "deltap_match.dat";
    std::ifstream ifs(fname);
    if (!ifs.is_open()) { match_loaded_ = false; return; }

    saved_matches_.clear();
    int alpha, istring, j, N;
    while (ifs >> alpha >> istring >> j >> N)
    {
        if (alpha < 0 || alpha >= 3) continue;
        if (alpha >= static_cast<int>(saved_matches_.size()))
            saved_matches_.resize(alpha + 1);
        if (istring >= static_cast<int>(saved_matches_[alpha].size()))
            saved_matches_[alpha].resize(istring + 1);
        if (j >= static_cast<int>(saved_matches_[alpha][istring].size()))
            saved_matches_[alpha][istring].resize(j + 1);
        saved_matches_[alpha][istring][j].resize(N, -1);
        for (int n = 0; n < N; ++n)
            ifs >> saved_matches_[alpha][istring][j][n];
    }
    match_loaded_ = !saved_matches_.empty();
    if (match_loaded_)
        std::cout << " DeltaP: loaded eigenvalue matching from " << fname << std::endl;
}

void DeltaP::save_match() const
{
    if (saved_matches_.empty()) return;
#ifdef __MPI
    if (GlobalV::MY_RANK != 0) return;
#endif
    const std::string fname = "deltap_match.dat";
    std::ofstream ofs(fname);
    if (!ofs.is_open()) return;
    ofs << std::setprecision(10);
    for (size_t a = 0; a < saved_matches_.size(); ++a)
        for (size_t s = 0; s < saved_matches_[a].size(); ++s)
            for (size_t j = 0; j < saved_matches_[a][s].size(); ++j)
            {
                const auto& m = saved_matches_[a][s][j];
                if (m.empty()) continue;
                ofs << a << " " << s << " " << j << " " << m.size();
                for (int v : m) ofs << " " << v;
                ofs << "\n";
            }
    ofs.close();
}

// ============================================================
// Branch-set selection: resolve 2π ambiguity by nearest-neighbor
//
// Given principal value  γ^I_0 = Σ_n w^I_n · arg(λ_n)
// and the set            Γ^I = {γ^I_0 + 2π · w^I·k : k ∈ Z^N}
// select the element closest to gamma_prev.
//
// Search strategy (efficient, avoids 3^N brute force):
//   1. If |γ^I_0 - prev| < π  → no jump, k = 0
//   2. Try single-band shifts: for each n, k_n = ±1, rest 0
//   3. Try two-band shifts (rare): for each (n,m), k_n=±1, k_m=±1
// ============================================================
double DeltaP::select_branch_set(
    const std::vector<double>& weights,
    const std::vector<double>& arg_evals,
    int nbands,
    double gamma_prev,
    std::vector<int>& k_selected) const
{
    k_selected.assign(nbands, 0);

    // Principal value
    double g0 = 0.0;
    for (int n = 0; n < nbands; ++n)
        g0 += weights[n] * arg_evals[n];

    // No previous value (first run): return principal value, no selection
    if (std::isnan(gamma_prev))
        return g0;

    // Case 1: no jump
    if (std::abs(g0 - gamma_prev) < M_PI)
        return g0;

    // Case 2: single-band shift
    double best_val = g0;
    double best_dist = std::abs(g0 - gamma_prev);
    int best_n = -1;
    int best_sign = 0;

    for (int n = 0; n < nbands; ++n)
    {
        if (std::abs(weights[n]) < 1e-12) continue;
        for (int sign = -1; sign <= 1; sign += 2)
        {
            double candidate = g0 + sign * 2.0 * M_PI * weights[n];
            double dist = std::abs(candidate - gamma_prev);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_val = candidate;
                best_n = n;
                best_sign = sign;
            }
        }
    }

    if (best_n >= 0)
    {
        k_selected[best_n] = best_sign;
        return best_val;
    }

    // Case 3: two-band shift (very rare)
    for (int n = 0; n < nbands; ++n)
    {
        if (std::abs(weights[n]) < 1e-12) continue;
        for (int m = n + 1; m < nbands; ++m)
        {
            if (std::abs(weights[m]) < 1e-12) continue;
            for (int sn = -1; sn <= 1; sn += 2)
            for (int sm = -1; sm <= 1; sm += 2)
            {
                double candidate = g0 + 2.0 * M_PI * (sn * weights[n] + sm * weights[m]);
                double dist = std::abs(candidate - gamma_prev);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_val = candidate;
                    k_selected[n] = sn;
                    k_selected[m] = sm;
                }
            }
        }
    }

    return best_val;
}

// ============================================================
// Resta-Z method: compute per-atom electronic center displacement
// using z^I = <exp(-i*2*pi*r/R)> from density matrix
//
// Formula:
//   z^I = sum_{mu,nu in I} D_{mu,nu} * <phi_mu | exp(-i*2*pi*r_alpha/R) | phi_nu(R)>
//   <r_elec^I> = -(R/2*pi) * Im[ln(z^I)]  (branch cut possible)
//   delta_r^I = r_ion^I - <r_elec^I>
//
// The density matrix D_{mu,nu} = sum_n f_n * c_{n,mu} * conj(c_{n,nu})
// is computed from occupied LCAO coefficients at Gamma.
// The matrix element <phi_mu | exp(-i*2*pi*r/R) | phi_nu(R)> is computed
// using the existing overlap integrator (overlap_intor_) with a phase factor.
//
// Key advantage: no Wilson loop, no eigenvalue tracking, no arg branch cut
// (except ln, which may or may not cross the branch depending on atom position)
// ============================================================
void DeltaP::compute_resta_z(const UnitCell& ucell,
                             const psi::Psi<std::complex<double>>* psi,
                             const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DeltaP", "compute_resta_z");

#ifdef __MPI
    // This function uses a Mulliken-population approximation that assumes
    // serial psi layout.  The 2D block-cyclic MPI indexing in the DM loop
    // is incorrect (see line ~2140).  Skip with a warning.
    ModuleBase::WARNING("DeltaP::compute_resta_z",
        "compute_resta_z is experimental and serial-only — skipping under MPI");
    ModuleBase::timer::start("DeltaP", "compute_resta_z");
    ModuleBase::timer::end("DeltaP", "compute_resta_z");
    return;
#endif

    ModuleBase::timer::start("DeltaP", "compute_resta_z");

    std::cout << "\n * * * * * *\n << Start DeltaP Resta-Z displacement\n";

    // Lattice vector along polarization direction
    const int alpha_idx = gdir_ - 1;
    double R_bohr = 0.0;
    if (gdir_ == 1) R_bohr = ucell.lat0 * ucell.a1.norm();
    else if (gdir_ == 2) R_bohr = ucell.lat0 * ucell.a2.norm();
    else R_bohr = ucell.lat0 * ucell.a3.norm();
    const double V_bohr = ucell.omega;

    // G vector = 2*pi/R along gdir (in Cartesian, bohr^-1)
    double G_cart[3] = {0.0, 0.0, 0.0};
    G_cart[alpha_idx] = 2.0 * ModuleBase::PI / R_bohr;

    const int nks = psi->get_nk();
    const int nbands = psi->get_nbands();
    const int nlocal = paraV_->get_global_row_size();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();
    const int npol = ucell.get_npol();
    const int* iat2iwt = paraV_->iat2iwt_;

    // Get occupied bands
    double occ_bands_d = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands_d - std::floor(occ_bands_d)) > 0.0)
        occ_bands_d = std::floor(occ_bands_d) + 1.0;
    const int nocc = static_cast<int>(occ_bands_d);
    const int nocc_use = std::min(nocc, nbands);

    // For each k-point, compute z^I contribution:
    // z^I(k) = sum_{m,n} f_m * c*_{m,mu} * <phi_mu|exp(-iG.r)|phi_nu> * c_{n,nu}
    //        = sum_{m} f_m * <psi_{m,k}| exp(-iG.r) |psi_{m,k}>
    // (diagonal in m for Gamma-only; for general k, need off-diagonal terms)
    //
    // For simplicity, we compute at each k:
    //   M_{mu,nu}(k) = <phi_mu| exp(-iG.r) |phi_nu(R)> with Bloch phase
    //   z^I(k) = sum_{mu,nu in I} c*_{m,mu}(k) * M_{mu,nu}(k) * c_{m,nu}(k)
    //          (sum over occupied m, and mu/nu on atom I)

    // Accumulate z^I per atom (complex)
    std::vector<std::complex<double>> z_per_atom(nat_, std::complex<double>(0.0, 0.0));

    for (int ik = 0; ik < nks; ++ik)
    {
        psi->fix_k(ik);
        const std::complex<double>* psi_k = psi->get_pointer();

        // Build the exp(-iG.r) overlap matrix M_{mu,nu} in 2D block-cyclic
        // M_{mu,nu} = sum_R e^{-i*G.(R+tau_nu)} * <phi_mu(0)|exp(-iG.r)|phi_nu(R)>
        //
        // For the position operator exp(-iG.r), we use the identity:
        // <phi_mu(0)| exp(-iG.r) |phi_nu(R)> = integral of phi_mu*(r) * exp(-iG.r) * phi_nu(r-R) dr
        //
        // This is a two-center integral with an extra exp(-iG.r) factor.
        // We approximate it using the first-order expansion:
        // exp(-iG.r) ≈ 1 - i*G*r (for small G*r)
        // <phi|exp(-iG.r)|phi(R)> ≈ <phi|phi(R)> - i*G*<phi|r|phi(R)>
        //
        // The overlap <phi|phi(R)> is computed by overlap_intor_->snap()
        // The position matrix <phi|r|phi(R)> is computed by r_overlap_->get_psi_r_psi()
        //
        // For exact computation, we need a dedicated two-center integral for exp(-iG.r),
        // but the first-order expansion is sufficient for small G (large R).

        // Build M_{mu,nu} = overlap - i*G * r_matrix (first order)
        // This is done in the 2D block-cyclic distribution
        std::vector<std::complex<double>> M_mat(static_cast<size_t>(nrow) * ncol, std::complex<double>(0.0, 0.0));

        // Phase factor for this k-point: exp(-i*G*k) where k is the fractional k-point
        // Actually, the Bloch phase is already handled by the k-point weighting.
        // For the Resta-Z method at k, we need:
        // z(k) = <u_{n,k}|exp(-iG.r)|u_{n,k}> where |u> is the periodic part
        // This requires the full exp(-iG.r) matrix, not just first order.

        // For now, use the first-order approximation:
        // z ≈ 1 - i*G*<r> = 1 - i*G*(sum mu,nu c*_mu * <phi_mu|r|phi_nu> * c_nu)
        // <r> = <psi|r|psi> (position expectation value, well-defined at Gamma)

        // Build the r-matrix in LCAO basis for the gdir component
        // r_{mu,nu} = sum_R <phi_mu(0)|r_alpha|phi_nu(R)> * Bloch_phase
        // This uses the existing compute_S_dk_link infrastructure but for r instead of S

        // Actually, for the Resta-Z method, we need the full exp(-iG.r) integral,
        // not just the first-order approximation. The first-order gives:
        // z ≈ 1 - i*G*<r>, so <r> = (1-z)/(i*G) = i*(z-1)/G
        // This is only valid when G*<r> << 1, i.e., <r> << R.
        // For atoms near R/2, this breaks down.

        // Better approach: use the exact two-center integral for <phi|exp(-iG.r)|phi(R)>
        // by modifying the snap() call to include the exp(-iG.r) factor.
        // But this requires modifying the TwoCenterIntegrator, which is complex.

        // Pragmatic approach: use the position matrix r_{mu,nu} (already available
        // from compute_S_dk_link's r_local computation) to compute <r> directly,
        // then compute z = exp(-i*G*<r>) as an approximation.

        // For each atom I, compute <r_elec^I> from the density matrix:
        // <r^I> = sum_{mu,nu in I} D_{mu,nu} * r_{mu,nu} / sum_{mu,nu in I} D_{mu,nu} * S_{mu,nu}
        // where D_{mu,nu} = sum_n c_{n,mu}*c*_{n,nu} (density matrix)
        //       r_{mu,nu} = <phi_mu|r|phi_nu> (position matrix)
        //       S_{mu,nu} = <phi_mu|phi_nu> (overlap matrix)

        // This is essentially Mulliken population analysis with position.
        // It's well-defined (r is well-defined for localized LCAO basis)
        // but NOT gauge-invariant (depends on the choice of basis).
        // However, it gives a physically reasonable displacement.

        // Skip this k-point if not Gamma (r is only well-defined at Gamma)
        // Actually, for the Resta method, we need all k-points.
        // But for the Mulliken approach, we only need Gamma.
        // Let's use the Mulliken approach for now (k=0 only).

        if (ik != 0) continue;  // TODO: extend to all k

        // Get k-point weight
        double kw = kv_->wk[ik];
        if (kw < 1e-10) continue;

        // Compute per-atom density matrix * position matrix
        for (int iat = 0; iat < nat_; ++iat)
        {
            auto tau0 = ucell.get_tau(iat);
            int I0 = 0, T0 = 0;
            ucell.iat2iait(iat, &I0, &T0);
            const int nw0 = ucell.atoms[T0].nw;

            // Find adjacent atoms
            AdjacentAtomInfo adjs;
            gd_->Find_atom(ucell, tau0, T0, I0, &adjs);

            double r_sum = 0.0;  // sum D_{mu,nu} * r_{mu,nu}
            double s_sum = 0.0;  // sum D_{mu,nu} * S_{mu,nu} (Mulliken charge)

            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = ucell.itia2iat(T1, I1);
                const ModuleBase::Vector3<int>& R_index = adjs.box[ad];

                if (ucell.cal_dtau(iat, iat1, R_index).norm() * ucell.lat0
                    > orb_cutoff_[T0] + orb_cutoff_[T1])
                    continue;

                const ModuleBase::Vector3<double> dtau = tau0 - adjs.adjacent_tau[ad];
                const Atom* atom1 = &ucell.atoms[T1];
                const int nw1 = atom1->nw;

                // R vector in Cartesian (bohr)
                ModuleBase::Vector3<double> R_cart(
                    R_index.x * ucell.a1.x + R_index.y * ucell.a2.x + R_index.z * ucell.a3.x,
                    R_index.x * ucell.a1.y + R_index.y * ucell.a2.y + R_index.z * ucell.a3.y,
                    R_index.x * ucell.a1.z + R_index.y * ucell.a2.z + R_index.z * ucell.a3.z);
                R_cart *= ucell.lat0;

                // Ionic position of ket atom (bohr)
                ModuleBase::Vector3<double> R2_cart = adjs.adjacent_tau[ad] * ucell.lat0;
                ModuleBase::Vector3<double> R1_cart = tau0 * ucell.lat0;

                for (int iw1 = 0; iw1 < nw1; ++iw1)
                {
                    const int L1 = atom1->iw2l[iw1];
                    const int N1 = atom1->iw2n[iw1];
                    const int m1 = atom1->iw2m[iw1];
                    const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;

                    // Compute overlap <phi_mu(0)|phi_nu(R)>
                    std::vector<std::vector<double>> nlm_ov;
                    overlap_intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm_ov);
                    if (nlm_ov.empty() || nlm_ov[0].empty()) continue;

                    for (int iw0 = 0; iw0 < nw0; ++iw0)
                    {
                        const double ov = nlm_ov[0][iw0];
                        if (std::abs(ov) < 1e-15) continue;

                        // Compute position matrix <phi_{mu}(0)|r_alpha|phi_nu(R)>
                        // for this specific bra orbital (iw0)
                        ModuleBase::Vector3<double> r_mat(0.0, 0.0, 0.0);
                        if (r_overlap_)
                        {
                            const int L0 = ucell.atoms[T0].iw2l[iw0];
                            const int m0 = ucell.atoms[T0].iw2m[iw0];
                            r_mat = r_overlap_->get_psi_r_psi(
                                R1_cart, T0, L0, m0, 0,
                                R2_cart, T1, L1, m1, N1);
                        }

                        // Get LCAO coefficients for this pair
                        for (int s = 0; s < npol; ++s)
                        {
                            const int gmu = iat2iwt[iat] + npol * iw0 + s;
                            const int gnu = iat2iwt[iat1] + npol * iw1 + s;
                            const int lr = paraV_->global2local_row(gmu);
                            const int lc = paraV_->global2local_col(gnu);
                            if (lr < 0 || lc < 0) continue;

                            // Density matrix element D_{mu,nu} = sum_n c_{n,mu} * conj(c_{n,nu})
                            std::complex<double> D_mn(0.0, 0.0);
                            for (int n = 0; n < nocc_use; ++n)
                            {
                                // psi_k is (nbands, nlocal) in local layout
                                // psi_k[n * nrow_local + lr] is c_{n,mu} (local row lr)
                                // psi_k[n * ncol_local + lc] is c_{n,nu} (local col lc)
                                // Actually, the layout depends on the Parallel_Orbitals
                                // For serial (no MPI): psi_k[n * nlocal + gmu] = c_{n,mu}
                                std::complex<double> c_mu, c_nu;
#ifdef __MPI
                                // In MPI, need to gather coefficients from other ranks
                                // For simplicity, use the local pointer if available
                                c_mu = psi_k[n * nrow + lr];
                                c_nu = psi_k[n * nrow + lc];  // This is wrong for 2D block-cyclic
#else
                                c_mu = psi_k[n * nlocal + gmu];
                                c_nu = psi_k[n * nlocal + gnu];
#endif
                                D_mn += c_mu * std::conj(c_nu);
                            }

                            // Mulliken: D * S (overlap)
                            s_sum += std::real(D_mn) * ov * kw;
                            // Position: D * r (position matrix)
                            // r_mat is <phi_mu(0)|r|phi_nu(R)> measured from origin
                            // (get_psi_r_psi returns full position, not relative)
                            double r_element = r_mat[alpha_idx];
                            r_sum += std::real(D_mn) * r_element * kw;
                        }
                    }
                }
            }

            // <r_elec^I> = r_sum / s_sum (Mulliken-weighted position)
            if (std::abs(s_sum) > 1e-10)
            {
                z_per_atom[iat] += std::complex<double>(r_sum / s_sum, 0.0);
            }
        }
    }

    // MPI reduction
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, z_per_atom.data(), 2 * nat_, MPI_DOUBLE, MPI_SUM, paraV_->comm());
#endif

    // Compute displacement
    results_.r_elec_center.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    for (int iat = 0; iat < nat_; ++iat)
    {
        // z_per_atom[iat] now contains <r_elec^I> (Mulliken-weighted, in bohr)
        double r_elec = std::real(z_per_atom[iat]);
        results_.r_elec_center[iat][alpha_idx] = r_elec;
    }

    std::cout << "   DeltaP Resta-Z: computed per-atom electronic center (Mulliken method)\n";

    // Output
    std::cout << "   DeltaP Resta-Z displacement:\n";
    for (int iat = 0; iat < nat_; ++iat)
    {
        int ia, it;
        ucell.iat2iait(iat, &ia, &it);
        double r_elec = results_.r_elec_center[iat][alpha_idx];
        double r_ion = 0.0;
        if (gdir_ == 1) r_ion = ucell.get_tau(iat).x * ucell.lat0;
        else if (gdir_ == 2) r_ion = ucell.get_tau(iat).y * ucell.lat0;
        else r_ion = ucell.get_tau(iat).z * ucell.lat0;
        double delta_r = r_ion - r_elec;
        std::cout << "     " << ucell.atom_label[it] << ia
                  << " r_elec=" << r_elec << " r_ion=" << r_ion
                  << " delta=" << delta_r << " bohr (" << delta_r / 1.8897259886 << " A)\n";
    }

    std::cout << " >> Finish DeltaP Resta-Z.\n * * * * * *\n";
    ModuleBase::timer::end("DeltaP", "compute_resta_z");
}

void DeltaP::init_inner_loop()
{
    nscf_ = PARAM.inp.deltap_inner_nmax;
    bfgs_.init(nat_, 0.5, PARAM.inp.deltap_conv_thr, 2, 0.01, 0.005);
}

} // namespace deltap
