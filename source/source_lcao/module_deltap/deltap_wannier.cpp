#include "deltap.h"
#include "source_esolver/deltap_common.h"
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

#ifdef __MPI
// DeltaP band-pair completeness (TODO 3.2/3.3, 2026-08-13).  The A' scheme
// fills a band-space matrix entry (g1, g2) with the local-rows × local-bands
// partial and completes it by an Allreduce.  That is complete only when every
// band pair is local to some rank — with dim1 > 1 process columns the local
// band sets are disjoint, so pairs spanning two process columns would never
// be filled.  This helper gathers the FULL band set for this rank's own rows
// (ranks sharing coord[0] hold the same orbital rows but disjoint band-column
// blocks), so the caller can form its rows' partial for ALL band pairs.  The
// caller then completes the row sum with a single Allreduce, contributing
// once per row group (ranks with coord[1] != 0 zero their partials).
static void gather_band_columns(const Parallel_Orbitals* paraV,
                                const std::complex<double>* local,
                                std::vector<std::complex<double>>& full,
                                std::vector<int>& gpos)
{
    full.clear();
    gpos.clear();
    MPI_Comm comm = paraV->comm();
    if (comm == MPI_COMM_NULL) return;
    const int dim1 = paraV->dim1;
    const int nb = paraV->nb;
    const int nrow = paraV->get_row_size();
    const int ncol_b = paraV->ncol_bands;

    MPI_Comm row_comm;
    MPI_Comm_split(comm, paraV->coord[0], paraV->coord[1], &row_comm);
    std::vector<int> counts(dim1, 0), displs(dim1, 0);
    MPI_Allgather(&ncol_b, 1, MPI_INT, counts.data(), 1, MPI_INT, row_comm);
    for (int q = 1; q < dim1; ++q)
    {
        displs[q] = displs[q - 1] + counts[q - 1];
    }
    const int nbands_g = displs[dim1 - 1] + counts[dim1 - 1];
    full.assign(static_cast<size_t>(nrow) * nbands_g, {0.0, 0.0});
    std::vector<int> cnt2(dim1, 0), dsp2(dim1, 0);
    for (int q = 0; q < dim1; ++q)
    {
        // complex<double> = 2 MPI_DOUBLE per element: Allgatherv counts are
        // in units of the send type (MPI_DOUBLE), so the per-rank block is
        // 2·nrow·ncol_b doubles (TODO 3.2/3.3, 2026-08-13 — half-count bug
        // found by the h2o_asym Γ^HK mismatch).
        cnt2[q] = 2 * nrow * counts[q];
        dsp2[q] = 2 * nrow * displs[q];
    }
    MPI_Allgatherv(local, nrow * ncol_b, MPI_DOUBLE,
                   reinterpret_cast<double*>(full.data()),
                   cnt2.data(), dsp2.data(), MPI_DOUBLE, row_comm);
    MPI_Comm_free(&row_comm);

    // gpos[g] = position in the gathered buffer of global band g
    // (local2global_col(j) = (j/nb·dim1 + q)·nb + j%nb, q = (g/nb)%dim1).
    const int nbands_global = paraV->get_wfc_global_nbands();
    gpos.assign(nbands_global, -1);
    for (int g = 0; g < nbands_global; ++g)
    {
        const int q = (g / nb) % dim1;
        const int j = (g / nb / dim1) * nb + (g % nb);
        gpos[g] = displs[q] + j;
    }
}
#endif

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
        // GUARD (B-6 family): the S_dk phase needs the bra position in
        // Direct (fractional) coordinates — dk is a fractional reciprocal
        // step, so dk·τ_bra is dimensionless only when τ_bra is fractional.
        // get_tau() returns the lat0-unit Cartesian position (numerically
        // ~Å), which over-sizes the phase by ~L (the cell size in lat0
        // units).  Do NOT replace taud0 with get_tau() here; H_HR's τ_α
        // (deltap_force_stress.hpp) is intentionally still lat0-unit (B-6
        // body, open) and must NOT be mixed into this phase either.
        const ModuleBase::Vector3<double> taud0 = ucell.atoms[T0].taud[I0];

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
                - dkv[0] * taud0.x - dkv[1] * taud0.y - dkv[2] * taud0.z);
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

    // Load branch state from previous SCF/run for cross-SCF phase smoothness.
    // SCF mode skips the file I/O (legacy): the frozen continuity anchor
    // (ref_gamma_) is seeded ONCE by the esolver at init (deltap_init ->
    // load_branch, which mirrors into ref_gamma_) and updated only at SCF
    // convergence (freeze_branch_ref).  Re-loading here per computation would
    // re-read the branch.dat that save_branch() overwrote at the end of the
    // PREVIOUS computation, re-introducing the per-iteration anchor drift
    // that locks early unconverged iterations onto a wrong branch (T4a
    // finding 2026-08-06).  Legacy gamma mode keeps the old skip (zero
    // regression).
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
    // Global band count (A' scheme): D_I is indexed by global band, and
    // nocc_use must be rank-invariant.  psi->get_nbands() is the LOCAL
    // column count under MPI and would make D_I/nocc_use rank-dependent.
    const int nbands = paraV_->get_wfc_global_nbands();
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
    // Per-computation Stage-B shift record (frozen branch shift, Phase
    // 0.3-lite): zero every computation so atoms/skipped directions keep a
    // zero shift (report = raw); freeze_branch_ref() copies the converged
    // computation's shifts into branch_shift_ for the next outer-loop run.
    last_shift_.assign(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));

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
        // Route A+ operator observable: ⟨P̂_I⟩ = Σ_k w_k Σ_n f_n Σ_{lm∈I} |D_{I,lm,n}(k)|²
        // (raw SMO projection per k-point, no S^{-1/2}; wg already folds in w_k·f_n).
        std::vector<double> p_hat_accum(nat_, 0.0);
        // L1.2: completeness ⟨η⟩ accumulators (band- and k-average over the
        // INPUT gdir; eta_accum2 for the max bound).
        double eta_accum = 0.0;
        double eta_accum2 = 0.0;
        double eta_max = 0.0;
        long eta_count = 0;
        std::vector<double> spread_accum(nat_, 0.0);
        int spread_count = 0;
        int n_strings_processed = 0;
        std::vector<std::vector<double>> w_In_first_string_;  // current alpha's first-string weights
        std::vector<std::complex<double>> zeta_list;
        std::vector<double> total_bp_per_string;      // total Berry phase (arg(zeta)) per string
        double current_zeta_scale = 1.0;               // scale factor from last zeta rescale

        // Branch reference: if previous converged value exists, use it;
        // otherwise use NaN to skip branch selection on first iteration.
        // This ensures the first iteration records raw gamma without being
        // pulled toward 0 by the branch selection logic.
        // Phase 0.3-lite: in continuity mode the per-string Stage-A reference
        // is the FROZEN anchor (ref_gamma_) too — NOT the per-computation
        // drifting W_prev_.  A drifting Stage-A reference re-biases the
        // per-string shift path each iteration (string-0's raw-0 phase gets
        // shifted toward the previous report), and once the report lands on a
        // wrong lattice point the coupling locks it there even though the raw
        // gamma converges to the natural value (T4a finding 2026-08-06).
        // Legacy gamma mode keeps the W_prev_ behavior (zero regression).
        const bool have_stage_a_ref
            = (branch_anchor_ == "continuity") ? has_ref_gamma_ : has_prev_;
        const std::vector<ModuleBase::Vector3<double>>& stage_a_ref
            = (branch_anchor_ == "continuity" && has_ref_gamma_) ? ref_gamma_ : W_prev_;
        std::vector<double> prev_gamma(nat_, std::numeric_limits<double>::quiet_NaN());
        for (int iat = 0; iat < nat_; ++iat)
            if (have_stage_a_ref && static_cast<int>(stage_a_ref.size()) > iat
                && !std::isnan(stage_a_ref[iat][alpha]))
                prev_gamma[iat] = stage_a_ref[iat][alpha];
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
                    {
                        MPI_Comm comm = paraV_->comm();
                        if (comm != MPI_COMM_NULL)
                            MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                          2 * sz, MPI_DOUBLE, MPI_SUM, comm);
                    }
                }
            }
        }
#endif

        // Route A+ operator observable (Γ_I^HR): accumulate ⟨P̂_I⟩ over ALL
        // physical k-points of this string (each physical k-point belongs to
        // exactly one string per gdir).  The PBC-wrapped string stores the
        // last slot as k_0 + G ≡ k_0 (the Wilson loop adds a G-phase to the
        // boundary link), so only slots j = 0..nppstr_-2 are distinct physical
        // k-points — skipping the wrapped copy avoids double counting k_0.
        // Uses the raw per-k-point SMO projection D_I (not the
        // S^{-1/2}-rotated tilde_proj), matching the real-space Tr[ρ·P̂_I]
        // contraction (T0 cross-check).
        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            const int ik_psi = k_index_[istring][j];
            if (ik_psi >= nks) continue;
            for (int n = 0; n < nocc_use; ++n)
            {
                const double wg_val = pelec->wg(ik_psi, n);
                if (wg_val == 0.0) continue;
                double w_tot_n = 0.0;  // Σ_I w_In(k) for this k, band n (L1.2)
                for (int iat = 0; iat < nat_; ++iat)
                {
                    if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                    const int r = nproj_per_atom_[iat];
                    double w_In = 0.0;
                    for (int lm = 0; lm < r; ++lm)
                    {
                        if (kstring_data_[j].D_I[iat].size() > static_cast<size_t>(lm)
                            && kstring_data_[j].D_I[iat][lm].size() > static_cast<size_t>(n))
                        {
                            w_In += std::norm(kstring_data_[j].D_I[iat][lm][n]);
                        }
                    }
                    p_hat_accum[iat] += wg_val * w_In;
                    w_tot_n += w_In;
                }
                // L1.2: SMO projection leakage η_n(k) = max(0, 1 − Σ_I w_In(k))
                // (clamped at 0: a numerically >1 total is projection overlap
                // of non-orthogonal SMO channels, not negative leakage).
                // Accumulated on the INPUT gdir only, so each physical k is
                // counted once across the three gdir passes.
                if (alpha == gdir_orig - 1)
                {
                    const double eta = deltap_common::compute_smo_leakage(w_tot_n);
                    eta_accum += eta;
                    eta_accum2 += eta * eta;
                    if (eta > eta_max) eta_max = eta;
                    eta_count++;
                }
            }
        }

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
                    // Circular phase distance in [-pi, pi]: the naive
                    // min(|a-g|, 2pi-|a-g|) goes NEGATIVE when |a-g| > 2pi,
                    // and the greedy match then prefers the wrong band (the
                    // negative "distance" wins).  remainder() wraps correctly.
                    double diff = std::abs(std::remainder(
                        std::arg(evals[m]) - std::fmod(gamma_unwrapped[n], 2.0 * M_PI), 2.0 * M_PI));
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
        // L1.3: per-atom θ-spread (weighted std-dev of the band-resolved
        // Wilson phases), accumulated on the INPUT gdir only:
        //   spread_I ≡ [ Σ_n w̄_In (θ_n − θ̄_I)² / Σ_n w̄_In ]^{1/2},
        //   w̄_In = w_In/Σ_J w_Jn,  θ̄_I = Σ_n w̄_In·θ_n / Σ_n w̄_In
        // H_HR proxy error control ∝ λ_I·spread_I (RouteA++ §1.4 推论 1);
        // uniform θ (all bands equal) gives spread_I = 0 exactly.
        if (alpha == gdir_orig - 1 && n_dim > 0)
        {
            std::vector<double> w_tot_n(n_dim, 0.0);
            for (int n = 0; n < n_dim; ++n)
                for (int i = 0; i < nat_; ++i)
                    w_tot_n[n] += w_In_matrix[n][i];
            for (int iat = 0; iat < nat_; ++iat)
            {
                std::vector<double> w_norm(n_dim, 0.0);
                for (int n = 0; n < n_dim; ++n)
                    w_norm[n] = (w_tot_n[n] > 1e-30) ? w_In_matrix[n][iat] / w_tot_n[n] : 0.0;
                spread_accum[iat] += deltap_common::compute_theta_spread(w_norm, gamma_unwrapped);
            }
            spread_count++;
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

        // T-6' (Ô_w): capture the band-resolved Wilson phase θ_n(k) of the
        // INPUT gdir for the H_ow operator.  gamma_unwrapped is the
        // unwrapped per-band loop phase tracked by the Hungarian matching;
        // the per-band pairing with the SMO weights is the same ordering
        // used by the γ weight channel.  R2/R3 (2026-08-12): Ô_w is a
        // k-local operator, so every physical k of this string stores the
        // string's θ_n (each k belongs to exactly one string per gdir; the
        // wrapped PBC slot duplicates k_0 and is skipped).
        if (alpha == gdir_orig - 1)
        {
            const int ik0 = k_index_[istring][0];
            std::vector<double> theta = gamma_unwrapped;
            const bool have_prev = (static_cast<size_t>(ik0) < ow_theta_prev_k_.size()
                                    && ow_theta_prev_k_[ik0].size() == theta.size());
            if (have_prev)
            {
                // D2 jump freeze (V-H8): a 2π branch discontinuity in θ_n
                // would kick the Hamiltonian discontinuously (λ·θ_n·P̂
                // changes by λ·2π·P̂).  When |Δθ_n| > π/2 relative to the
                // previous measurement of this string, keep the previous θ_n
                // and flag the jump instead of applying it — the operator
                // stays on the continuous branch.
                bool jump = false;
                for (size_t n = 0; n < theta.size(); ++n)
                {
                    const double d = std::abs(theta[n] - ow_theta_prev_k_[ik0][n]);
                    if (d > 0.5 * ModuleBase::PI)
                    {
                        jump = true;
                        break;
                    }
                }
                if (jump)
                {
                    // R4 recovery (2026-08-12): a legitimate evolution past
                    // the branch point must eventually be accepted; after
                    // ow_theta_freeze_max_ consecutive frozen steps the new θ
                    // is adopted and the counter resets (otherwise the
                    // operator would be frozen on the stale branch forever).
                    if (static_cast<size_t>(ik0) >= ow_theta_freeze_count_.size())
                        ow_theta_freeze_count_.resize(ik0 + 1, 0);
                    const int cnt = ow_theta_freeze_count_[ik0] + 1;
                    ow_theta_freeze_count_[ik0] = cnt;
                    if (cnt >= ow_theta_freeze_max_)
                    {
                        std::cout << "   [DeltaP Ô_w] D2 branch jump: θ_n "
                                  << "re-accepted after " << cnt << " frozen steps "
                                  << "(R4 recovery)" << std::endl;
                        ow_theta_freeze_count_[ik0] = 0;
                    }
                    else
                    {
                        std::cout << "   [DeltaP Ô_w] D2 branch jump: θ_n frozen at "
                                  << "previous values (|Δθ|>π/2, V-H8; frozen "
                                  << cnt << "/" << ow_theta_freeze_max_ << ")"
                                  << std::endl;
                        theta = ow_theta_prev_k_[ik0];
                    }
                }
                else if (static_cast<size_t>(ik0) < ow_theta_freeze_count_.size())
                {
                    ow_theta_freeze_count_[ik0] = 0;
                }
            }
            else
            {
                // First measurement for this k (new run / re-anchored ref):
                // R4 — anchor to the frozen continuity reference when it
                // exists, so the operator starts on the same branch as the
                // previous converged run instead of a raw (branch-ambiguous)
                // sheet (D2 initialization anchor).
                theta = anchor_ow_theta_to_ref(theta, w_In_matrix, gdir_);
            }
            if (ow_theta_k_.size() < static_cast<size_t>(nks))
                ow_theta_k_.resize(nks);
            if (ow_theta_prev_k_.size() < static_cast<size_t>(nks))
                ow_theta_prev_k_.resize(nks);
            for (int j = 0; j < nppstr_ - 1; ++j)
            {
                const int ik = k_index_[istring][j];
                if (ik < 0 || ik >= nks) continue;
                ow_theta_k_[ik] = theta;
                ow_theta_prev_k_[ik] = theta;
            }
            ow_theta_gdir_ = gdir_;
            ow_theta_valid_ = true;
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
            // Use a relative threshold: raw_sum must be meaningful (>1e-6 rad
            // per band) to avoid dividing tiny numerical noise by gamma_unw_sum.
            double tol = 1e-6 * n_dim;
            if (std::abs(gamma_raw_sum) > tol && std::abs(gamma_raw_sum - gamma_unw_sum) > 1e-10)
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
                    if (std::abs(g - prev) < M_PI)
                    {
                        // DEBUG (Phase 0.3-lite forensics): per-string Stage-A
                        // no-shift path for the gdir direction.
                        if (alpha == gdir_orig - 1 && n_strings_processed == 1)
                            std::cout << " [SAdbg] iat=" << iat << " alpha=" << alpha
                                      << " noshift g=" << std::setprecision(9) << g
                                      << " prev=" << prev << std::endl;
                        prev_gamma[iat] = g;
                        continue;
                    }

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
                    {
                        std::cout << "   DeltaP branch-set: atom " << iat
                                  << " rescaled=" << std::scientific << std::setprecision(6) << g
                                  << " selected=" << best_val
                                  << " prev=" << prev
                                  << " delta=" << best_val - g << std::endl;
                        // DEBUG (Phase 0.3-lite forensics): Stage-A shift
                        // path for the gdir direction.
                        if (alpha == gdir_orig - 1)
                            std::cout << " [SAdbg] iat=" << iat << " alpha=" << alpha
                                      << " shifted g=" << g << " -> " << best_val
                                      << " prev=" << prev << std::endl;
                    }
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
                // Frozen branch shift (Phase 0.3-lite): record the applied
                // shift per atom/direction for freeze_branch_ref().
                last_shift_[iat][alpha] = delta;

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
                // Frozen branch shift (Phase 0.3-lite): record the applied
                // shift per atom/direction for freeze_branch_ref().
                last_shift_[iat][alpha] = delta;
            }
        }
        else
        {
            for (int iat = 0; iat < nat_; ++iat)
            {
                double avg_raw = gamma_accum[iat] / n_strings_processed;
                double target = target_gamma_[iat];
                // Phase 0.3-lite (Route A+ operator mode) continuity readout:
                // once a frozen branch shift exists (last SCF convergence),
                // the Stage-B target is avg_raw + branch_shift_ — i.e. the
                // search keeps the reported γ on the SAME lattice point family
                // while the reading follows the physical raw exactly.  The
                // nearest-lattice-point-to-anchor form pins the reading to
                // the anchor (the lattice has points within ~0.005 of it), so
                // the physical response (dγ/dt_Γ ~ 0.1) is swallowed and the
                // outer loop cannot drive γ (T4a finding 2026-08-09).  Before
                // the first freeze, fall back to the FROZEN last-converged
                // anchor (ref_gamma_, seeded from branch.dat and updated only
                // at SCF convergence — NOT the per-iteration drifting
                // W_prev_): t_γ then only initializes the branch when no
                // reference exists (first call / no branch.dat).  Legacy
                // gamma mode ("target") keeps the target-aware behavior
                // unchanged (zero regression).
                if (branch_anchor_ == "continuity"
                    && has_branch_shift_
                    && static_cast<size_t>(iat) < branch_shift_.size()
                    && !std::isnan(branch_shift_[iat][alpha]))
                {
                    target = avg_raw + branch_shift_[iat][alpha];
                }
                else if (branch_anchor_ == "continuity"
                    && has_ref_gamma_
                    && static_cast<size_t>(iat) < ref_gamma_.size()
                    && !std::isnan(ref_gamma_[iat][alpha]))
                {
                    target = ref_gamma_[iat][alpha];
                }

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
            // Frozen branch shift (Phase 0.3-lite): record the applied shift
            // per atom/direction for freeze_branch_ref().
            last_shift_[iat][alpha] = delta;
            // DEBUG (Phase 0.3-lite anchor forensics, 2026-08-06): Stage-B
            // search inputs/outputs for the gdir direction so the branch
            // selection can be audited against the raw value.  The shift
            // source is labeled (frozen=avg_raw+branch_shift_, anchor=ref_).
            if (branch_anchor_ == "continuity" && alpha == gdir_orig - 1)
            {
                const bool frozen = (has_branch_shift_
                                     && static_cast<size_t>(iat) < branch_shift_.size()
                                     && !std::isnan(branch_shift_[iat][alpha]));
                std::cout << " [SBdbg] iat=" << iat
                          << " alpha=" << alpha
                          << " mode=" << (frozen ? "frozen" : "anchor")
                          << " avg_raw=" << std::setprecision(9) << avg_raw
                          << " anchor=" << std::setprecision(9) << target
                          << " best=" << std::setprecision(9) << best_val
                          << " shift=" << std::setprecision(9) << delta
                          << " bdist=" << std::setprecision(6) << best_dist;
                for (size_t q = 0; q < shift_amp.size(); ++q)
                    std::cout << " s" << q << "=" << std::setprecision(6) << shift_amp[q];
                std::cout << std::endl;
            }
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

    // L1.2/L1.3 finalize (INPUT gdir only): band/k-average ⟨η⟩ and per-atom
    // spread averaged over the k-strings of the constraint direction.
    if (alpha == gdir_orig - 1)
    {
        if (eta_count > 0)
        {
            last_eta_avg_ = eta_accum / eta_count;
            last_eta_max_ = eta_max;
        }
        if (spread_count > 0)
        {
            last_spread_I_.assign(nat_, 0.0);
            for (int iat = 0; iat < nat_; ++iat)
                last_spread_I_[iat] = spread_accum[iat] / spread_count;
        }
    }

    // Route A+ operator observable: keep only the INPUT constraint direction.
    // Γ_I^HR = τ_α(I)·⟨P̂_I⟩ with τ in Direct (fractional) coordinates
    // (B-6 convention, consistent with the H_HR operator and hhrdbg block).
    if (alpha == gdir_orig - 1)
    {
        if (static_cast<int>(gamma_op_.size()) != nat_)
            gamma_op_.assign(nat_, 0.0);
        for (int iat = 0; iat < nat_; ++iat)
        {
            int I0 = 0, T0 = 0;
            ucell.iat2iait(iat, &I0, &T0);
            gamma_op_[iat] = ucell.atoms[T0].taud[I0][alpha] * p_hat_accum[iat];
        }
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
        if (comm == MPI_COMM_NULL) { scf_mode_ = false; return; }
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

// (T-6', R4, 2026-08-12) First-measurement Ô_w branch anchor.  The raw θ_n
// from the first Wilson-loop measurement of a run carries a per-band 2π
// ambiguity.  When the frozen continuity anchor (ref_gamma_, seeded from
// branch.dat and updated at SCF convergence) exists, choose the per-band 2π
// sheet of θ_n that brings the implied per-atom γ (= Σ_n w̃_In·θ_n with the
// per-band-normalized SMO weights, the same convention as the γ observable)
// closest to ref_gamma_ — coordinate descent over bands with candidates
// {−2π, 0, +2π}.  This anchors the H_ow operator to the same branch as the
// previous converged run (D2 initialization anchor).  Without a reference the
// raw sheet is kept (first-ever run defines the anchor).
std::vector<double> DeltaP::anchor_ow_theta_to_ref(
    const std::vector<double>& theta,
    const std::vector<std::vector<double>>& w_In,
    int gdir_val) const
{
    std::vector<double> out = theta;
    if (!has_ref_gamma_ || static_cast<int>(ref_gamma_.size()) != nat_)
        return out;
    const int alpha = gdir_val - 1;
    if (alpha < 0 || alpha >= 3) return out;
    bool ref_ok = true;
    for (int iat = 0; iat < nat_; ++iat)
        if (std::isnan(ref_gamma_[iat][alpha])) { ref_ok = false; break; }
    if (!ref_ok) return out;

    const int N = static_cast<int>(theta.size());
    std::vector<double> w_tot(N, 0.0);
    for (int n = 0; n < N; ++n)
        for (int iat = 0; iat < nat_; ++iat)
            if (n < static_cast<int>(w_In.size())
                && iat < static_cast<int>(w_In[n].size()))
                w_tot[n] += w_In[n][iat];
    auto implied_gamma = [&](const std::vector<double>& th) {
        std::vector<double> g(nat_, 0.0);
        for (int iat = 0; iat < nat_; ++iat)
        {
            double gI = 0.0;
            for (int n = 0; n < N; ++n)
            {
                if (n >= static_cast<int>(w_In.size())
                    || iat >= static_cast<int>(w_In[n].size()))
                    continue;
                if (w_tot[n] > 1e-30)
                    gI += (w_In[n][iat] / w_tot[n]) * th[n];
            }
            g[iat] = gI;
        }
        return g;
    };
    auto dist2 = [&](const std::vector<double>& g) {
        double d = 0.0;
        for (int iat = 0; iat < nat_; ++iat)
            d += (g[iat] - ref_gamma_[iat][alpha]) * (g[iat] - ref_gamma_[iat][alpha]);
        return d;
    };
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 64)
    {
        improved = false;
        for (int n = 0; n < N; ++n)
        {
            const double base_d = dist2(implied_gamma(out));
            double best_shift = 0.0;
            double best_d = base_d;
            for (int s = -1; s <= 1; s += 2)
            {
                std::vector<double> cand = out;
                cand[n] += s * ModuleBase::TWO_PI;
                const double d = dist2(implied_gamma(cand));
                if (d < best_d - 1e-20) { best_d = d; best_shift = s * ModuleBase::TWO_PI; }
            }
            if (best_shift != 0.0) { out[n] += best_shift; improved = true; }
        }
    }
    const double rms = std::sqrt(dist2(implied_gamma(out)) / std::max(1, nat_));
    std::cout << "   [DeltaP Ô_w] first-measurement θ anchored to continuity ref"
              << " (RMS γ deviation " << std::setprecision(4) << rms << " rad)"
              << std::endl;
    return out;
}

// (T-6', R2/R3, 2026-08-12) Fill kstring_data_ for one k-string (S_k + D_I)
// at the given wavefunctions and mark kstring_gdir_/kstring_string_.  This
// centralizes the per-string rebuild used by compute_hk_correction /
// compute_gamma_op_hk / compute_hk_force for the k-local H_ow pass (the
// k-local operator needs S_k/D_I at every physical k, not just string 0).
void DeltaP::fill_kstring(int istring, const psi::Psi<std::complex<double>>* psi,
                          int nbands, int nrow)
{
    const int nks = psi->get_nk();
    for (int j = 0; j < nppstr_; ++j)
    {
        kstring_data_[j].S_k.clear();
        kstring_data_[j].dS_k.clear();
        kstring_data_[j].D_I.clear();
    }
    for (int j = 0; j < nppstr_; ++j)
    {
        const int ik = k_index_[istring][j];
        if (ik >= nks) continue;
        kstring_data_[j].kvec_d = kv_->kvec_d[ik];
        psi->fix_k(ik);
        compute_S_k(j);
        compute_D_I(j, psi->get_pointer(), nbands, nrow);
    }
#ifdef __MPI
    for (int j = 0; j < nppstr_; ++j)
    {
        for (int iat = 0; iat < nat_; ++iat)
        {
            const int r = nproj_per_atom_[iat];
            for (int lm = 0; lm < r; ++lm)
            {
                if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                const int sz = kstring_data_[j].D_I[iat][lm].size();
                if (sz > 0)
                {
                    MPI_Comm comm = paraV_->comm();
                    if (comm != MPI_COMM_NULL)
                        MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                      2 * sz, MPI_DOUBLE, MPI_SUM, comm);
                }
            }
        }
    }
#endif
    kstring_gdir_ = gdir_;
    kstring_string_ = istring;
}

void DeltaP::compute_hk_correction(const UnitCell& ucell,
                                   const psi::Psi<std::complex<double>>* psi,
                                   const elecstate::ElecState* pelec,
                                   const std::vector<double>& lambda,
                                   std::unordered_map<int, std::vector<std::complex<double>>>& hk_correction)
{
    ModuleBase::TITLE("DeltaP", "compute_hk_correction");
    hk_correction.clear();

    // Route A+ operator observable: Γ_I^HK depends only on the current
    // wavefunctions (not on λ); recompute it from scratch on every call.
    // Early returns below leave it zeroed (e.g. HK disabled).
    gamma_op_hk_.assign(nat_, 0.0);
    // T-6' (Ô_w): the weight-channel operator observable Γ_I^w is zeroed here
    // like Γ_I^HK; it is filled by the link loop below in "ow" mode.
    gamma_op_w_.assign(nat_, 0.0);

    if (nppstr_ < 2 || kstring_data_.empty()) return;
    // T-6' (Ô_w): this call builds the k-local H_ow (band-resolved θ_n(k)·P̂,
    // R2/R3) at every physical k of the INPUT gdir in addition to the H_HK
    // Berry-connection term when the constraint operator mode is "ow".  It
    // requires the per-k band-resolved Wilson phase captured by the preceding
    // compute_gamma_scf (same wavefunctions); without it, H_ow is skipped
    // with a warning and only H_HK is applied.
    const bool ow_mode = (operator_mode_ == "ow");
    if (ow_mode && (!ow_theta_valid_ || ow_theta_gdir_ != gdir_) && GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP Ô_w] WARNING: θ_n unavailable (compute_gamma_scf "
                  << "did not run); H_ow skipped, H_HK only" << std::endl;
    }

    const int nks = psi->get_nk();
    const int nbands = paraV_->get_wfc_global_nbands();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();

#ifdef __MPI
    int nproc = 1;
    {
        MPI_Comm comm = paraV_->comm();
        if (comm != MPI_COMM_NULL)
            MPI_Comm_size(comm, &nproc);
    }
#else
    const int nproc = 1;
#endif
    // F-6 (TODO 3.1, 2026-08-13): nproc > 1 uses the distributed-GEMM path
    // (correct local block of H_sym in any 2D block-cyclic distribution);
    // nproc == 1 keeps the original loop code byte-identical (hard serial
    // A/B constraint).  The Γ^HK observable is completed by a uniform-count
    // Allreduce over the orbital/band partial sums (A' scheme, same family
    // as compute_D_I).
    const bool mpi_path = (nproc > 1);

    // TODO 3.2/3.3 (2026-08-13): the nrow == ncol guard is now scoped to the
    // serial path only.  The MPI path is distribution-agnostic: H_sym is the
    // (nrow × ncol) local block of the exact Hermitian correction (pzgemm
    // block sizes are numroc-derived, so nrow != ncol is natural), the Γ^HK
    // observable uses the A' row/band partial-sum Allreduce, and
    // contributeHk adds the local block element-wise into hsk->get_hk().
    // Non-square local blocks (odd nwfc, e.g. h2o1/h2o_asym 529 on a 2×2
    // grid) are therefore valid under MPI; the serial path still assumes a
    // full local matrix (nrow == ncol == nlocal) by construction, so the
    // check below is defensive only.
    if (nproc == 1 && nrow != ncol)
    {
        ModuleBase::WARNING_QUIT("DeltaP::compute_hk_correction",
            "Serial LCAO assumes nrow == ncol (full local matrix).");
    }

    // Rebuild S_k/D_I if they belong to a different direction or string.
    // compute_gamma_scf leaves kstring_data_ from the last alpha (gdir=3)
    // and the last string; we need the INPUT gdir and string 0.
    if (kstring_gdir_ != gdir_ || kstring_string_ != 0)
    {
        setup_kstring(*kv_);
        kstring_data_.assign(nppstr_, KSpaceData());
        fill_kstring(0, psi, nbands, nrow);
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

    // Per-atom Γ_I^HK accumulator (complex; finalized after the link loop).
    // Serial path: T·Π trace over the full orbital sum (all rows local).
    // MPI path: per-rank row/band partials completed by an Allreduce (A'
    // scheme) so the per-atom observable is rank-independent.
    std::vector<std::complex<double>> e_hk_I(nat_, {0.0, 0.0});

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

        // Per-atom SMO weights w_In[I][n] = Σ_{lm∈I} |D_I[iat][lm][n]|² kept
        // separate from the λ-weighted w_eff so that Γ_I^HK can be split per
        // atom (Route A+): E_HK = Σ_I λ_I·Γ_I^HK with
        // Γ_I^HK = −0.5·Im[Σ_j Σ_p f_p·w_{I,p}^{(j)}·T_pp^{(j)}].
        std::vector<std::vector<double>> w_IJ(nocc_use, std::vector<double>(nat_, 0.0));
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
            for (int n = 0; n < nocc_use; ++n)
            {
                double w_In = 0.0;
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I[iat].size() > static_cast<size_t>(lm) &&
                        kstring_data_[j].D_I[iat][lm].size() > static_cast<size_t>(n))
                    {
                        w_In += std::norm(kstring_data_[j].D_I[iat][lm][n]);
                    }
                }
                w_IJ[n][iat] = w_In;
            }
        }

        // Compute effective weights: w_eff[n] = Σ_I lambda[I] * w_IJ[n][I]
        std::vector<double> w_eff(nocc_use, 0.0);
        for (int n = 0; n < nocc_use; ++n)
            for (int iat = 0; iat < nat_; ++iat)
                w_eff[n] += lambda[iat] * w_IJ[n][iat];

#ifdef __MPI
        if (mpi_path)
        {
            // F-6 (TODO 3.1, 2026-08-13): distributed H_HK correction.
            // SC = S_dk·C_R is a genuine distributed GEMM (S_dk lives in the
            // Hamiltonian 2D block-cyclic layout, C_R in the wfc layout); the
            // exact local block of the symmetrized operator is
            //   H_sym = (F·C_L† + C_L·F†)/2,   F = (i/2)·w_eff·SC,
            // i.e. the (local rows × local cols) block of the full serial
            // correction (T2 full-trace convention).  pzgemm uses the
            // codebase 'T'-with-pre-conjugation convention (cal_dm_psi).
            const int nlocal = paraV_->desc[2];
            const int nbands_g = paraV_->desc_wfc[3];
            const int ncol_b = paraV_->ncol_bands;
            const std::complex<double> one_c(1.0, 0.0), zero_c(0.0, 0.0);
            const int one_i = 1;
            const char N_ch = 'N', T_ch = 'T';

            // SC = S_dk · C_R  (nlocal × nbands_g, desc_wfc layout)
            std::vector<std::complex<double>> SC(nrow * ncol_b, {0.0, 0.0});
            ScalapackConnector::gemm(N_ch, N_ch, nlocal, nbands_g, nlocal,
                                     one_c, S_dk_.data(), one_i, one_i, paraV_->desc,
                                     c_R, one_i, one_i, paraV_->desc_wfc,
                                     zero_c, SC.data(), one_i, one_i, paraV_->desc_wfc);

            // F = (i/2)·w_eff[g]·SC (g = global band of the local column)
            std::vector<std::complex<double>> F(nrow * ncol_b, {0.0, 0.0});
            for (int n = 0; n < ncol_b; ++n)
            {
                const int g = paraV_->local2global_col(n);
                if (g < 0 || g >= nocc_use) continue;
                const std::complex<double> coeff = half_i * w_eff[g];
                for (int a = 0; a < nrow; ++a)
                    F[a + n * nrow] = coeff * SC[a + n * nrow];
            }

            // Pre-conjugated copies ('T' convention of pzgemm).
            std::vector<std::complex<double>> CL_conj(nrow * ncol_b);
            std::vector<std::complex<double>> F_conj(nrow * ncol_b);
            for (int i = 0; i < nrow * ncol_b; ++i)
            {
                CL_conj[i] = std::conj(c_L[i]);
                F_conj[i] = std::conj(F[i]);
            }

            // M1 = F·C_L† and M2 = C_L·F† in the desc layout (local block).
            std::vector<std::complex<double>> M1(nrow * ncol, {0.0, 0.0});
            std::vector<std::complex<double>> M2(nrow * ncol, {0.0, 0.0});
            ScalapackConnector::gemm(N_ch, T_ch, nlocal, nlocal, nbands_g,
                                     one_c, F.data(), one_i, one_i, paraV_->desc_wfc,
                                     CL_conj.data(), one_i, one_i, paraV_->desc_wfc,
                                     zero_c, M1.data(), one_i, one_i, paraV_->desc);
            ScalapackConnector::gemm(N_ch, T_ch, nlocal, nlocal, nbands_g,
                                     one_c, c_L, one_i, one_i, paraV_->desc_wfc,
                                     F_conj.data(), one_i, one_i, paraV_->desc_wfc,
                                     zero_c, M2.data(), one_i, one_i, paraV_->desc);

            // H_sym = 0.5·(M1 + M2): the exact Hermitian local block.
            std::vector<std::complex<double>> H_sym(nrow * ncol, {0.0, 0.0});
            for (int i = 0; i < nrow * ncol; ++i)
                H_sym[i] = 0.5 * (M1[i] + M2[i]);
            hk_correction[ik_L] = std::move(H_sym);

            // Γ_I^HK per atom: T = C_L†·SC and Π = C_L†·C_L are band-space
            // (nocc_use×nocc_use) matrices whose orbital sums span all ranks.
            // Band-pair completeness (TODO 3.2/3.3, 2026-08-13): with dim1 > 1
            // the plain A' local-band loop never fills pairs spanning two
            // process columns — the full band set for this rank's rows is
            // gathered within the process-row group first (one contribution
            // per row group: ranks with coord[1] != 0 zero their partials),
            // then a single Allreduce completes the row sums.  Pi uses the
            // serial storage convention Pi[i][j] = Π_{j,i} so the full-trace
            // kernel below (Pi[pp][p] = Π_{p,pp}) is bit-compatible with the
            // serial path.
            std::vector<std::complex<double>> T_part(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> Pi_part(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> cL_all, SC_all;
            std::vector<int> gpos;
            gather_band_columns(paraV_, c_L, cL_all, gpos);
            gather_band_columns(paraV_, SC.data(), SC_all, gpos);
            for (int g1 = 0; g1 < nocc_use; ++g1)
            {
                const int p1 = gpos[g1];
                if (p1 < 0) continue;
                for (int g2 = 0; g2 < nocc_use; ++g2)
                {
                    const int p2 = gpos[g2];
                    if (p2 < 0) continue;
                    std::complex<double> tsum(0.0, 0.0), psum(0.0, 0.0);
                    for (int a = 0; a < nrow; ++a)
                    {
                        tsum += std::conj(cL_all[a + p1 * nrow]) * SC_all[a + p2 * nrow];
                        psum += std::conj(cL_all[a + p2 * nrow]) * cL_all[a + p1 * nrow];
                    }
                    T_part[g1 * nocc_use + g2] += tsum;
                    Pi_part[g1 * nocc_use + g2] += psum;
                }
            }
            {
                MPI_Comm comm = paraV_->comm();
                if (comm != MPI_COMM_NULL)
                {
                    if (paraV_->coord[1] != 0)
                    {
                        T_part.assign(T_part.size(), {0.0, 0.0});
                        Pi_part.assign(Pi_part.size(), {0.0, 0.0});
                    }
                    MPI_Allreduce(MPI_IN_PLACE, T_part.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                    MPI_Allreduce(MPI_IN_PLACE, Pi_part.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                }
            }
            for (int p = 0; p < nocc_use; ++p)
            {
                const double fp = pelec->wg(ik_L, p);
                if (fp == 0.0) continue;
                for (int iat = 0; iat < nat_; ++iat)
                {
                    std::complex<double> acc(0.0, 0.0);
                    for (int pp = 0; pp < nocc_use; ++pp)
                        acc += w_IJ[pp][iat] * T_part[p * nocc_use + pp] * Pi_part[pp * nocc_use + p];
                    e_hk_I[iat] += fp * acc;
                }
            }
        }
        else
#endif
        {
            // Serial path (nproc == 1): byte-identical to the pre-F-6 code
            // (hard serial A/B constraint) — the full orbital/band space is
            // local, so the loops below ARE the full H_sym.

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

        // Route A+ operator observable: Γ_I^HK per atom = per-atom split of
        // the ACTUAL applied-operator expectation Tr[ρ·H_sym] (T2 verdict
        // 2026-08-04).  H_sym is linear in λ (w_eff[n] = Σ_I λ_I·w_IJ[n][I]),
        // so per atom:
        //   Γ_I^HK = −0.5·Σ_p wg(ik_L,p)·Im[Σ_{p'} w_IJ[p'][I]·T_{pp'}·Π_{p'p}]
        // with T_{pp'} = (C_L†·S_dk·C_R)_{pp'} and Π_{p'p} = (C_L†·C_L)_{p'p}.
        // The pre-T2 diagonal convention (T_pp only) assumed Π = I and
        // overstated the coupling by ~18% in the non-orthogonal LCAO basis
        // (E'(λ) slope −13.3 eV/Ry instead of ≲1; fixed by the full trace).
        std::vector<std::complex<double>> T_full(nocc_use * nocc_use, {0.0, 0.0});
        std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int pp = 0; pp < nocc_use; ++pp)
            {
                for (int a = 0; a < nrow; ++a)
                {
                    T_full[p * nocc_use + pp] += std::conj(c_L[a + p * nrow]) * SC[a + pp * nrow];
                    Pi[p * nocc_use + pp] += std::conj(c_L[a + pp * nrow]) * c_L[a + p * nrow];
                }
            }
        }
        for (int p = 0; p < nocc_use; ++p)
        {
            const double fp = pelec->wg(ik_L, p);
            if (fp == 0.0) continue;
            for (int iat = 0; iat < nat_; ++iat)
            {
                std::complex<double> acc(0.0, 0.0);
                for (int pp = 0; pp < nocc_use; ++pp)
                {
                    acc += w_IJ[pp][iat] * T_full[p * nocc_use + pp] * Pi[pp * nocc_use + p];
                }
                e_hk_I[iat] += fp * acc;
            }
        }

        // (H_HK link loop continues; the k-local H_ow operator is built in
        // the per-string pass below — R2/R3, 2026-08-12)

        hk_correction[ik_L] = H_sym;
        }
    }

    // T-6' (Ô_w), R2/R3 (2026-08-12): H_ow is a k-local operator (Route A++
    // §2.1: applied to ψ_n(k) at each k), so it is built here for EVERY
    // physical k of the INPUT gdir (each k belongs to exactly one string),
    // NOT only on string-0's links like the H_HK Berry term.
    //   H_ow(k) = sym( Σ_n θ_n(k)·A_C(k)[n]·C_k† )   (nrow × nrow)
    //   A_C(k)[μ][n] = Σ_I λ_I·Σ_{lm∈I} S_k[I][lm][μ]·D_I[I][lm][n]
    // (the SMO-weighted projector A = Σ_I λ_I·P̂_I applied to ψ_n(k));
    // θ_n(k) is the Wilson-loop phase of the string containing k.  The
    // per-atom expectation split Γ_I^w uses the full Gram trace (T2
    // convention): Γ_I^w += wg(m)·θ_n·Σ_{lm∈I} D*[lm,m]·D[lm,n]·Π[n,m].
    // No per-band weight normalization is applied: Ô_w = θ_n·P̂_I is the
    // exact weight-channel operator of Route A++ §1.3 (the S^{-1/2} Löwdin
    // rotation of the projector is deferred; V-H3' quantifies its residual).
    // F-6 (TODO 3.1): H_ow under MPI is deferred (TODO 3.2/3.3) — the Ô_w
    // branches build nrow×nrow local blocks with serial-only index semantics
    // (local rows for both the row and the column of the local H block).  The
    // square-block guard would only let them run on even-nwfc systems, where
    // they would silently mis-place the local block; skip with a warning.
    if (ow_mode && mpi_path && GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP] WARNING: operator_mode=ow under MPI is deferred "
                  << "(TODO 3.2/3.3); only the H_HK correction is applied."
                  << std::endl;
    }
    if (ow_mode && !mpi_path && ow_theta_valid_ && ow_theta_gdir_ == gdir_ && ow_kernel_valid_)
    {
        // T-17 (V-H8, S1): frozen-kernel path — H_ow is rebuilt from the
        // kernel captured at the last edge (frozen K_I, C, θ) scaled by the
        // current λ, NOT from the live C.  Between edges the SCF sees a
        // fixed H_ow(λ) so the period-2 limit cycle (state-dependent live
        // operator between two near-degenerate self-consistent solutions,
        // S0 2026-08-12) cannot form.  No fill_kstring / psi access needed.
        for (int istring = 0; istring < total_string_; ++istring)
        {
            for (int j = 0; j < nppstr_ - 1; ++j)
            {
                const int ik = k_index_[istring][j];
                if (ik < 0 || ik >= nks) continue;
                if (ow_K_I_k_.size() <= static_cast<size_t>(ik) || ow_K_I_k_[ik].empty()) continue;
                if (ow_C_k_.size() <= static_cast<size_t>(ik)) continue;
                const std::vector<double>& theta = ow_theta_frozen_k_[ik];
                if (theta.size() < static_cast<size_t>(nocc_use)) continue;

                // A_C(λ) = Σ_I λ_I·K_I  (nrow × nocc_use) from the frozen
                // per-atom kernel (exact in λ; the λ-dependence of H_ow is
                // carried only here).
                std::vector<std::complex<double>> A_C(nrow * nocc_use, {0.0, 0.0});
                for (int mu = 0; mu < nrow; ++mu)
                    for (int n = 0; n < nocc_use; ++n)
                        for (int iat = 0; iat < nat_; ++iat)
                        {
                            const double lam = lambda[iat];
                            if (lam == 0.0) continue;
                            A_C[mu * nocc_use + n]
                                += lam * ow_K_I_k_[ik][iat][mu * nocc_use + n];
                        }

                // M_ow = A_C · diag(θ_frozen) · C_frozen†  (nrow × nrow)
                std::vector<std::complex<double>> M_ow(nrow * nrow, {0.0, 0.0});
                for (int beta = 0; beta < nrow; ++beta)
                    for (int alpha = 0; alpha < nrow; ++alpha)
                    {
                        std::complex<double> sum(0.0, 0.0);
                        for (int n = 0; n < nocc_use; ++n)
                            sum += theta[n] * A_C[alpha * nocc_use + n]
                                 * std::conj(ow_C_k_[ik][beta * nocc_use + n]);
                        M_ow[alpha + beta * nrow] = sum;
                    }
                // hk_correction[ik] += sym(M_ow)
                std::vector<std::complex<double>>& H_ow = hk_correction[ik];
                if (H_ow.empty())
                    H_ow.assign(nrow * nrow, {0.0, 0.0});
                for (int beta = 0; beta < nrow; ++beta)
                    for (int alpha = 0; alpha < nrow; ++alpha)
                        H_ow[alpha + beta * nrow]
                            += 0.5 * (M_ow[alpha + beta * nrow]
                                      + std::conj(M_ow[beta + alpha * nrow]));
            }
        }
    }
    else if (ow_mode && !mpi_path && ow_theta_valid_ && ow_theta_gdir_ == gdir_)
    {
        // Live path (pre-S1 behavior): kernel invalid (defensive fallback,
        // e.g. θ unavailable at the kernel build) — rebuild H_ow from the
        // live C exactly as before.
        for (int istring = 0; istring < total_string_; ++istring)
        {
            if (kstring_gdir_ != gdir_ || kstring_string_ != istring)
                fill_kstring(istring, psi, nbands, nrow);
            for (int j = 0; j < nppstr_ - 1; ++j)
            {
                const int ik = k_index_[istring][j];
                if (ik < 0 || ik >= nks) continue;
                if (ow_theta_k_.size() <= static_cast<size_t>(ik)
                    || ow_theta_k_[ik].size() < static_cast<size_t>(nocc_use))
                    continue;
                psi->fix_k(ik);
                const std::complex<double>* c_k = psi->get_pointer();

                // A_C = P̂_λ·C_k  (nrow × nocc_use)
                std::vector<std::complex<double>> A_C(nrow * nocc_use, {0.0, 0.0});
                for (int mu = 0; mu < nrow; ++mu)
                {
                    for (int n = 0; n < nocc_use; ++n)
                    {
                        std::complex<double> sum(0.0, 0.0);
                        for (int iat = 0; iat < nat_; ++iat)
                        {
                            const int r = nproj_per_atom_[iat];
                            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                            const double lam = lambda[iat];
                            if (lam == 0.0) continue;
                            if (kstring_data_[j].S_k.size() <= static_cast<size_t>(iat)) continue;
                            for (int lm = 0; lm < r; ++lm)
                            {
                                if (kstring_data_[j].S_k[iat].size() <= static_cast<size_t>(lm)) continue;
                                if (static_cast<size_t>(mu) >= kstring_data_[j].S_k[iat][lm].size()) continue;
                                if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                                if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                                sum += lam * kstring_data_[j].S_k[iat][lm][mu]
                                     * kstring_data_[j].D_I[iat][lm][n];
                            }
                        }
                        A_C[mu + n * nrow] = sum;
                    }
                }
                // M_ow = A_C · diag(θ) · C_k†  (nrow × nrow)
                std::vector<std::complex<double>> M_ow(nrow * nrow, {0.0, 0.0});
                for (int beta = 0; beta < nrow; ++beta)
                {
                    for (int alpha = 0; alpha < nrow; ++alpha)
                    {
                        std::complex<double> sum(0.0, 0.0);
                        for (int n = 0; n < nocc_use; ++n)
                            sum += ow_theta_k_[ik][n] * A_C[alpha + n * nrow]
                                 * std::conj(c_k[beta + n * nrow]);
                        M_ow[alpha + beta * nrow] = sum;
                    }
                }
                // hk_correction[ik] += sym(M_ow)
                std::vector<std::complex<double>>& H_ow = hk_correction[ik];
                if (H_ow.empty())
                    H_ow.assign(nrow * nrow, {0.0, 0.0});
                for (int beta = 0; beta < nrow; ++beta)
                    for (int alpha = 0; alpha < nrow; ++alpha)
                        H_ow[alpha + beta * nrow]
                            += 0.5 * (M_ow[alpha + beta * nrow]
                                      + std::conj(M_ow[beta + alpha * nrow]));
            }
        }
    }

    // Γ_I^HK = −0.5·Im[Σ_j Σ_p f_p·Σ_{p'} w_{I,p'}^{(j)}·T_{pp'}^{(j)}·Π_{p'p}^{(j)}]
    // (per-atom split of the ACTUAL applied-operator expectation; differs
    // from compute_hk_force's E_HK diagonal convention in the non-orthogonal
    // basis — see the T2 dated document).
    for (int iat = 0; iat < nat_; ++iat)
    {
        gamma_op_hk_[iat] = -0.5 * e_hk_I[iat].imag();
    }
    // T-17 (V-H8, S1): Γ_I^w is NOT finalized here — gamma_op_w_ is owned
    // exclusively by compute_gamma_op_hk (which runs first in the same
    // measurement round): the frozen path caches the applied-operator value
    // (ow_gamma_w_frozen_, HG-2) and the live path measures it from the
    // same wavefunctions.  A second finalize here would either zero the
    // frozen accounting (empty local accumulator) or duplicate the live
    // measurement with a stale ψ — both wrong.
}

// (T-17, V-H8, S1, 2026-08-12) Build the frozen Ô_w operator kernel from the
// CURRENT wavefunctions.  The H_ow operator's ψ-dependence enters through the
// SMO projector overlaps D_I = S_k†·C and the wavefunction C itself; S1
// freezes both (plus the band-resolved Wilson phase θ) at the last "edge"
// (first measurement, P2 λ update via on_phase2, freeze_branch_ref) so the
// SCF between edges sees a FIXED H_ow instead of
// the state-dependent live operator whose two near-degenerate self-consistent
// solutions produced the period-2 limit cycle (S0, 2026-08-12).  The kernel
// keeps the λ-dependence exact and cheap:
//   A_C(λ)[μ][n] = Σ_I λ_I·K_I[μ][n],  K_I[μ][n] = Σ_{lm∈I} S_k[I][lm][μ]·D_I[I][lm][n]
// so λ changes (inner-loop BFGS trials, P2 update) scale the frozen kernel
// without re-measuring ψ.  The applied-operator Γ_I^w (per-atom split of
// Tr[ρ·H_ow], full Gram trace with the snapshot θ and C) is cached in
// ow_gamma_w_frozen_ — the accounting (escon = −λ·Γ^w) always refers to the
// operator that was actually applied (HG-2).  Sets ow_kernel_valid_; when no
// k carried a usable θ the kernel stays invalid and the caller falls back to
// the live path (same behavior as the pre-S1 code).
void DeltaP::compute_ow_kernel(const psi::Psi<std::complex<double>>* psi,
                               const elecstate::ElecState* pelec,
                               int nbands, int nocc_use, int nrow)
{
    if (!ow_theta_valid_ || ow_theta_gdir_ != gdir_) return;
    const int nks = psi->get_nk();
    const int nat = nat_;
    if (static_cast<int>(ow_theta_k_.size()) < nks) return;

    // Pre-size the per-k storage; only k's with a usable θ snapshot get filled.
    ow_K_I_k_.assign(nks, std::vector<std::vector<std::complex<double>>>());
    ow_T_I_k_.assign(nks, std::vector<std::vector<std::complex<double>>>());
    ow_C_k_.assign(nks, std::vector<std::complex<double>>());
    ow_theta_frozen_k_.assign(nks, std::vector<double>());
    for (int ik = 0; ik < nks; ++ik)
    {
        if (ow_theta_k_[ik].size() < static_cast<size_t>(nocc_use)) continue;
        ow_K_I_k_[ik].assign(nat, std::vector<std::complex<double>>(nrow * nocc_use, {0.0, 0.0}));
        ow_T_I_k_[ik].assign(nat, std::vector<std::complex<double>>(nocc_use * nocc_use, {0.0, 0.0}));
        ow_C_k_[ik].assign(nrow * nocc_use, {0.0, 0.0});
        ow_theta_frozen_k_[ik] = ow_theta_k_[ik];
    }

    std::vector<std::complex<double>> e_w(nat, {0.0, 0.0});
    bool any = false;
    for (int istring = 0; istring < total_string_; ++istring)
    {
        if (kstring_gdir_ != gdir_ || kstring_string_ != istring)
            fill_kstring(istring, psi, nbands, nrow);
        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            const int ik = k_index_[istring][j];
            if (ik < 0 || ik >= nks) continue;
            if (ow_K_I_k_[ik].empty()) continue;
            psi->fix_k(ik);
            const std::complex<double>* c_k = psi->get_pointer();

            // Frozen C snapshot: ow_C_k_[ik][μ*nocc_use + n] = C_k[μ + n*nrow].
            for (int n = 0; n < nocc_use; ++n)
                for (int mu = 0; mu < nrow; ++mu)
                    ow_C_k_[ik][mu * nocc_use + n] = c_k[mu + n * nrow];

            // Per-atom λ-independent kernel K_I and band-space T_I.
            for (int iat = 0; iat < nat; ++iat)
            {
                const int r = nproj_per_atom_[iat];
                if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                if (kstring_data_[j].S_k.size() <= static_cast<size_t>(iat)) continue;
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].S_k[iat].size() <= static_cast<size_t>(lm)) continue;
                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                    const auto& Sk = kstring_data_[j].S_k[iat][lm];
                    const auto& DI = kstring_data_[j].D_I[iat][lm];
                    if (Sk.size() < static_cast<size_t>(nrow)) continue;
                    if (DI.size() < static_cast<size_t>(nocc_use)) continue;
                    for (int mu = 0; mu < nrow; ++mu)
                        for (int n = 0; n < nocc_use; ++n)
                            ow_K_I_k_[ik][iat][mu * nocc_use + n] += Sk[mu] * DI[n];
                    for (int m = 0; m < nocc_use; ++m)
                        for (int n = 0; n < nocc_use; ++n)
                            ow_T_I_k_[ik][iat][m * nocc_use + n]
                                += std::conj(DI[m]) * DI[n];
                }
            }

            // Applied-operator Γ_I^w with the snapshot θ and C (full Gram
            // trace; Π[n*N+m] = (C_frozen†·C_frozen)[m,n]).
            const std::vector<double>& theta = ow_theta_frozen_k_[ik];
            std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
            for (int n = 0; n < nocc_use; ++n)
                for (int m = 0; m < nocc_use; ++m)
                    for (int a = 0; a < nrow; ++a)
                        Pi[n * nocc_use + m]
                            += std::conj(ow_C_k_[ik][a * nocc_use + m])
                             * ow_C_k_[ik][a * nocc_use + n];
            for (int m = 0; m < nocc_use; ++m)
            {
                const double fm = pelec->wg(ik, m);
                if (fm == 0.0) continue;
                for (int n = 0; n < nocc_use; ++n)
                {
                    const double thn = theta[n];
                    if (thn == 0.0) continue;
                    for (int iat = 0; iat < nat; ++iat)
                        e_w[iat] += fm * thn * ow_T_I_k_[ik][iat][m * nocc_use + n]
                                  * Pi[n * nocc_use + m];
                }
            }
            any = true;
        }
    }

    // Finalize: real per-atom split with the R8 imaginary-residue warning
    // (same convention as the live finalize in compute_gamma_op_hk).
    ow_gamma_w_frozen_.assign(nat, 0.0);
    for (int iat = 0; iat < nat; ++iat)
    {
        const double re = e_w[iat].real();
        const double im = e_w[iat].imag();
        if (GlobalV::MY_RANK == 0 && std::abs(im) > 1e-8 * std::max(1.0, std::abs(re)))
            std::cout << " [DeltaP Ô_w] WARNING: Im(Γ_I^w)=" << std::scientific
                      << std::setprecision(3) << im << " at iat=" << iat
                      << " (rel to Re " << re << "; R8 indexing check)"
                      << std::endl;
        ow_gamma_w_frozen_[iat] = re;
    }
    ow_kernel_valid_ = any;
    ow_kernel_stale_ = false;
}

void DeltaP::compute_gamma_op_hk(const UnitCell& ucell,
                                 const psi::Psi<std::complex<double>>* psi,
                                 const elecstate::ElecState* pelec)
{
    // Route A+ operator observable: per-atom Γ_I^HK at the CURRENT
    // wavefunctions (λ-independent), as the per-atom split of the ACTUAL
    // applied-operator expectation (full T·Π trace, same as the escon/SCF
    // observable in compute_hk_correction — T2 verdict 2026-08-04).
    // Same serial-only semantics as compute_hk_correction (nrow == ncol).
    gamma_op_hk_.assign(nat_, 0.0);
    // T-6' (Ô_w): Γ_I^w (weight-channel operator observable) is λ-independent
    // per atom, so it is filled here too (compute_gamma_op runs before
    // apply_hk_correction in iter_finish).
    gamma_op_w_.assign(nat_, 0.0);

    if (nppstr_ < 2 || kstring_data_.empty()) return;
    const bool ow_mode = (operator_mode_ == "ow");

    const int nks = psi->get_nk();
    const int nbands = paraV_->get_wfc_global_nbands();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();
#ifdef __MPI
    int nproc = 1;
    {
        MPI_Comm comm = paraV_->comm();
        if (comm != MPI_COMM_NULL)
            MPI_Comm_size(comm, &nproc);
    }
#else
    const int nproc = 1;
#endif
    // F-6 (TODO 3.1, 2026-08-13): Γ_I^HK is a global observable — the
    // band-space T·Π trace spans every orbital and band.  Under MPI each
    // rank contributes its local rows × local bands and a uniform-count
    // Allreduce completes the trace (A' scheme, same family as
    // compute_D_I), making Γ^HK rank-independent; the serial path keeps the
    // original loops byte-identical (hard serial A/B constraint).
    const bool mpi_path = (nproc > 1);
    // TODO 3.2/3.3 (2026-08-13): non-square local blocks are valid on the
    // MPI path (the A' row/band partial-sum Allreduce is
    // distribution-agnostic — for a fixed band pair the owning process
    // column's ranks together cover every orbital row); the serial path
    // keeps the full-local-matrix assumption (nrow == ncol == nlocal), so
    // the check below is defensive only.
    if (nproc == 1 && nrow != ncol) return;

    // Rebuild S_k/D_I if they belong to a different direction or string.
    // compute_gamma_scf leaves kstring_data_ from the last alpha (gdir=3)
    // and the last string; we need the INPUT gdir and string 0.
    if (kstring_gdir_ != gdir_ || kstring_string_ != 0)
    {
        setup_kstring(*kv_);
        kstring_data_.assign(nppstr_, KSpaceData());
        fill_kstring(0, psi, nbands, nrow);
    }

    if (S_dk_.empty())
    {
        compute_S_dk(ucell);
    }

    double occ_bands_d = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands_d - std::floor(occ_bands_d)) > 0.0)
        occ_bands_d = std::floor(occ_bands_d) + 1.0;
    const int nocc = static_cast<int>(occ_bands_d);
    const int nocc_use = std::min(nocc, nbands);
    if (nocc_use <= 0) return;

    std::vector<std::complex<double>> e_hk_I(nat_, {0.0, 0.0});
    // T-6' (Ô_w): per-atom Γ_I^w accumulator (only filled in "ow" mode).
    std::vector<std::complex<double>> e_w_I(nat_, {0.0, 0.0});
    for (int j = 0; j < nppstr_ - 1; ++j)
    {
        const int ik_L = k_index_[0][j];
        const int ik_R = k_index_[0][j + 1];
        if (ik_L >= nks || ik_R >= nks) continue;

        psi->fix_k(ik_L);
        const std::complex<double>* c_L = psi->get_pointer();
        psi->fix_k(ik_R);
        const std::complex<double>* c_R = psi->get_pointer();

        // Per-atom weights w_In[I][n] (same link-j / k_L convention as
        // compute_hk_correction and compute_hk_force).
        std::vector<std::vector<double>> w_IJ(nocc_use, std::vector<double>(nat_, 0.0));
        for (int iat = 0; iat < nat_; ++iat)
        {
            const int r = nproj_per_atom_[iat];
            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
            for (int n = 0; n < nocc_use; ++n)
            {
                double w_In = 0.0;
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I[iat].size() > static_cast<size_t>(lm)
                        && kstring_data_[j].D_I[iat][lm].size() > static_cast<size_t>(n))
                    {
                        w_In += std::norm(kstring_data_[j].D_I[iat][lm][n]);
                    }
                }
                w_IJ[n][iat] = w_In;
            }
        }

#ifdef __MPI
        if (mpi_path)
        {
            // F-6 MPI path: SC = S_dk·C_R is a genuine distributed GEMM
            // (pzgemm, 'T'-with-pre-conjugation convention); the band-space
            // T·Π traces are completed by the row-group gather + Allreduce
            // (band-pair completeness, TODO 3.2/3.3, 2026-08-13): with
            // dim1 > 1 the plain A' local-band loop never fills pairs
            // spanning two process columns.  Pi uses the serial storage
            // convention Pi[i][j] = Π_{j,i} (bit-compatible with the serial
            // path below), so Γ^HK is rank-independent and matches serial.
            const int nlocal = paraV_->desc[2];
            const int nbands_g = paraV_->desc_wfc[3];
            const int ncol_b = paraV_->ncol_bands;
            const std::complex<double> one_c(1.0, 0.0), zero_c(0.0, 0.0);
            const int one_i = 1;
            const char N_ch = 'N';

            std::vector<std::complex<double>> SC(nrow * ncol_b, {0.0, 0.0});
            ScalapackConnector::gemm(N_ch, N_ch, nlocal, nbands_g, nlocal,
                                     one_c, S_dk_.data(), one_i, one_i, paraV_->desc,
                                     c_R, one_i, one_i, paraV_->desc_wfc,
                                     zero_c, SC.data(), one_i, one_i, paraV_->desc_wfc);

            std::vector<std::complex<double>> T_part(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> Pi_part(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> cL_all, SC_all;
            std::vector<int> gpos;
            gather_band_columns(paraV_, c_L, cL_all, gpos);
            gather_band_columns(paraV_, SC.data(), SC_all, gpos);
            for (int g1 = 0; g1 < nocc_use; ++g1)
            {
                const int p1 = gpos[g1];
                if (p1 < 0) continue;
                for (int g2 = 0; g2 < nocc_use; ++g2)
                {
                    const int p2 = gpos[g2];
                    if (p2 < 0) continue;
                    std::complex<double> tsum(0.0, 0.0), psum(0.0, 0.0);
                    for (int a = 0; a < nrow; ++a)
                    {
                        tsum += std::conj(cL_all[a + p1 * nrow]) * SC_all[a + p2 * nrow];
                        psum += std::conj(cL_all[a + p2 * nrow]) * cL_all[a + p1 * nrow];
                    }
                    T_part[g1 * nocc_use + g2] += tsum;
                    Pi_part[g1 * nocc_use + g2] += psum;
                }
            }
            {
                MPI_Comm comm = paraV_->comm();
                if (comm != MPI_COMM_NULL)
                {
                    if (paraV_->coord[1] != 0)
                    {
                        T_part.assign(T_part.size(), {0.0, 0.0});
                        Pi_part.assign(Pi_part.size(), {0.0, 0.0});
                    }
                    MPI_Allreduce(MPI_IN_PLACE, T_part.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                    MPI_Allreduce(MPI_IN_PLACE, Pi_part.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                }
            }
            for (int p = 0; p < nocc_use; ++p)
            {
                const double fp = pelec->wg(ik_L, p);
                if (fp == 0.0) continue;
                for (int iat = 0; iat < nat_; ++iat)
                {
                    std::complex<double> acc(0.0, 0.0);
                    for (int pp = 0; pp < nocc_use; ++pp)
                        acc += w_IJ[pp][iat] * T_part[p * nocc_use + pp] * Pi_part[pp * nocc_use + p];
                    e_hk_I[iat] += fp * acc;
                }
            }
        }
        else
#endif
        {
        // SC = S_dk * C_R (serial path, byte-identical to pre-F-6)
        std::vector<std::complex<double>> SC(nrow * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int a = 0; a < nrow; ++a)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int b = 0; b < ncol; ++b)
                {
                    sum += S_dk_[a + b * nrow] * c_R[b + p * nrow];
                }
                SC[a + p * nrow] = sum;
            }
        }
        // Route A+ operator observable: Γ_I^HK per atom = per-atom split of
        // the ACTUAL applied-operator expectation Tr[ρ·H_sym] (T2 verdict
        // 2026-08-04).  H_sym is linear in λ (w_eff[n] = Σ_I λ_I·w_IJ[n][I]),
        // so per atom:
        //   Γ_I^HK = −0.5·Σ_p wg(ik_L,p)·Im[Σ_{p'} w_IJ[p'][I]·T_{pp'}·Π_{p'p}]
        // with T_{pp'} = (C_L†·S_dk·C_R)_{pp'} and Π_{p'p} = (C_L†·C_L)_{p'p}.
        // The pre-T2 diagonal convention (T_pp only) assumed Π = I and
        // overstated the coupling by ~18% in the non-orthogonal LCAO basis
        // (E'(λ) slope −13.3 eV/Ry instead of ≲1; fixed by the full trace).
        std::vector<std::complex<double>> T_full(nocc_use * nocc_use, {0.0, 0.0});
        std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int pp = 0; pp < nocc_use; ++pp)
            {
                for (int a = 0; a < nrow; ++a)
                {
                    T_full[p * nocc_use + pp] += std::conj(c_L[a + p * nrow]) * SC[a + pp * nrow];
                    Pi[p * nocc_use + pp] += std::conj(c_L[a + pp * nrow]) * c_L[a + p * nrow];
                }
            }
        }
        for (int p = 0; p < nocc_use; ++p)
        {
            const double fp = pelec->wg(ik_L, p);
            if (fp == 0.0) continue;
            for (int iat = 0; iat < nat_; ++iat)
            {
                std::complex<double> acc(0.0, 0.0);
                for (int pp = 0; pp < nocc_use; ++pp)
                {
                    acc += w_IJ[pp][iat] * T_full[p * nocc_use + pp] * Pi[pp * nocc_use + p];
                }
                e_hk_I[iat] += fp * acc;
            }
        }
        }
    }

    // T-6' (Ô_w), R2/R3 (2026-08-12): Γ_I^w is the per-atom split of
    // Tr[ρ·H_ow] and H_ow is k-local, so it must cover EVERY physical k of
    // the INPUT gdir (one string each), not only string-0's links.  Same
    // full-Gram-trace convention as compute_hk_correction's H_ow block (no
    // per-band weight normalization — Ô_w = θ_n·P̂_I, Route A++ §1.3).
    // T-17 (V-H8, S1): the Γ_I^w accounting must refer to the APPLIED
    // operator.  At an edge (first measurement, P2 λ update, D2 freeze
    // recovery, freeze_branch_ref) the frozen kernel is rebuilt from the
    // current wavefunctions and the snapshot θ (compute_ow_kernel); between
    // edges the cached applied-operator value is returned directly so the
    // reported Γ_I^w is exactly the operator that was applied in the solve
    // (HG-2), instead of a fresh measurement at a slightly different ψ.
    // F-6 (TODO 3.1): the Ô_w Γ^I^w measurement is serial-only (its Π trace
    // sums local rows only); under MPI it is deferred (TODO 3.2/3.3) and
    // gamma_op_w_ stays zeroed (H_ow itself is also skipped, see
    // compute_hk_correction).
    if (ow_mode && mpi_path && GlobalV::MY_RANK == 0)
    {
        std::cout << " [DeltaP] WARNING: operator_mode=ow Γ^w under MPI is deferred "
                  << "(TODO 3.2/3.3); Γ^w=0 on the observable." << std::endl;
    }
    bool ow_used_frozen = false;
    if (ow_mode && !mpi_path && ow_theta_valid_ && ow_theta_gdir_ == gdir_)
    {
        if (ow_kernel_stale_ || !ow_kernel_valid_)
        {
            compute_ow_kernel(psi, pelec, nbands, nocc_use, nrow);
        }
        if (ow_kernel_valid_)
        {
            gamma_op_w_ = ow_gamma_w_frozen_;
            ow_used_frozen = true;
        }
        else
        {
            // Live path (pre-S1 behavior, defensive fallback): θ valid but
            // the kernel build saw no usable k — measure Γ_I^w live.
            for (int istring = 0; istring < total_string_; ++istring)
            {
                if (kstring_gdir_ != gdir_ || kstring_string_ != istring)
                    fill_kstring(istring, psi, nbands, nrow);
                for (int j = 0; j < nppstr_ - 1; ++j)
                {
                    const int ik = k_index_[istring][j];
                    if (ik < 0 || ik >= nks) continue;
                    if (ow_theta_k_.size() <= static_cast<size_t>(ik)
                        || ow_theta_k_[ik].size() < static_cast<size_t>(nocc_use))
                        continue;
                    psi->fix_k(ik);
                    const std::complex<double>* c_k = psi->get_pointer();

                    // Π[n*N+m] = (C_k†·C_k)[m,n] (frozen-C Gram matrix).
                    std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
                    for (int n = 0; n < nocc_use; ++n)
                        for (int m = 0; m < nocc_use; ++m)
                            for (int a = 0; a < nrow; ++a)
                                Pi[n * nocc_use + m] += std::conj(c_k[a + m * nrow]) * c_k[a + n * nrow];
                    for (int m = 0; m < nocc_use; ++m)
                    {
                        const double fm = pelec->wg(ik, m);
                        if (fm == 0.0) continue;
                        for (int n = 0; n < nocc_use; ++n)
                        {
                            const double thn = ow_theta_k_[ik][n];
                            if (thn == 0.0) continue;
                            for (int iat = 0; iat < nat_; ++iat)
                            {
                                const int r = nproj_per_atom_[iat];
                                std::complex<double> t_I(0.0, 0.0);
                                for (int lm = 0; lm < r; ++lm)
                                {
                                    if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                                    if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(m)) continue;
                                    if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                                    t_I += std::conj(kstring_data_[j].D_I[iat][lm][m])
                                         * kstring_data_[j].D_I[iat][lm][n];
                                }
                                e_w_I[iat] += fm * thn * t_I * Pi[n * nocc_use + m];
                            }
                        }
                    }
                }
            }
        }
    }

    for (int iat = 0; iat < nat_; ++iat)
    {
        gamma_op_hk_[iat] = -0.5 * e_hk_I[iat].imag();
    }
    // T-6' (Ô_w): finalize Γ_I^w (real per-atom split of Tr[ρ·H_ow]).
    // T-17 (V-H8, S1): skipped on the frozen path — gamma_op_w_ already
    // holds the applied-operator value (ow_gamma_w_frozen_).
    if (ow_mode && !ow_used_frozen)
    {
        for (int iat = 0; iat < nat_; ++iat)
        {
            // R8 (2026-08-12): soft per-atom imaginary-residue warning (see
            // compute_hk_correction for the rationale).
            const double re = e_w_I[iat].real();
            const double im = e_w_I[iat].imag();
            if (GlobalV::MY_RANK == 0 && std::abs(im) > 1e-8 * std::max(1.0, std::abs(re)))
                std::cout << " [DeltaP Ô_w] WARNING: Im(Γ_I^w)=" << std::scientific
                          << std::setprecision(3) << im << " at iat=" << iat
                          << " (rel to Re " << re << "; R8 indexing check)"
                          << std::endl;
            gamma_op_w_[iat] = re;
        }
    }
}

// (T-6', R1, 2026-08-12) Ô_w geometric force (ow mode).  H_HR and its A1/A2
// forces are gated off in ow mode; the k-local H_ow = sym(Σ_n θ_n·P̂_λ|ψ_n⟩⟨ψ_n|)
// replaces them.  Its force is the frozen-C/θ derivative of
//   E_ow = Σ_I λ_I·Γ_I^w,  Γ_I^w = Σ_k Σ_m f_m·Re[Σ_n θ_n·M^I_mn·Π_nm]
// with M^I_mn = Σ_{lm∈I} D*_I,lm,m·D_I,lm,n (per-atom split of Tr[ρ·H_ow],
// full Gram trace — the applied-operator expectation, T2 convention).  With C
// frozen, Π = C†C is constant and only the SMO projector overlaps D = S_k†·C
// carry the R-dependence, so the force kernel is the same two-center ∂S_k/∂R
// family as the A1/dW kernels:
//   ∂Γ_I^w/∂R_J,a = Σ_lm Re[ Σ_m conj(∂D_m/∂R_J,a)·X_lm[m]
//                            + Σ_n conj(C_n)·θ_n·Z_lm[n] ]
//   X_lm[m] = Σ_n θ_n·D_n·Π_nm,  Z_lm[n] = Σ_m f_m·D_m·Π_mn
//   F_ow[I][a] = −λ_I·∂Γ_I^w/∂R_I,a  (bra: +∂S_k/∂R),  F_ow[μ][a] = −∂Γ/∂R_μ (ket: −)
// Returns false (force untouched) when the ow data (per-k θ) is unavailable.
bool DeltaP::compute_ow_force(const UnitCell& ucell,
                              const psi::Psi<std::complex<double>>* psi,
                              const elecstate::ElecState* pelec,
                              const std::vector<double>& lambda,
                              std::vector<double>& force_out)
{
    if (operator_mode_ != "ow" || !ow_theta_valid_ || ow_theta_gdir_ != gdir_)
        return false;

    const int nks = psi->get_nk();
    const int nbands = paraV_->get_wfc_global_nbands();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();
    if (nrow != ncol) return false;
#ifdef __MPI
    // Defense-in-depth (2026-08-12): compute_hk_force already skips the
    // whole H_HK/ow force path under nproc>1 (serial-only, full-row S_k and
    // full C_k required); keep compute_ow_force self-contained so a future
    // caller cannot silently compute a wrong local-block force (Π = C†C and
    // force_out are not Allreduced here).
    if (GlobalV::NPROC > 1) return false;
#endif

    double occ_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands - std::floor(occ_bands)) > 0.0)
        occ_bands = std::floor(occ_bands) + 1.0;
    const int nocc = static_cast<int>(occ_bands);
    const int nocc_use = std::min(nocc, nbands);
    if (nocc_use <= 0) return false;

    const int nat = nat_;
    const int npol = ucell.get_npol();
    const int* iat2iwt = paraV_->iat2iwt_;
    std::vector<double> f_ow(nat * 3, 0.0);
    double e_ow = 0.0;
    bool any = false;

    for (int istring = 0; istring < total_string_; ++istring)
    {
        if (kstring_gdir_ != gdir_ || kstring_string_ != istring)
            fill_kstring(istring, psi, nbands, nrow);
        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            const int ik = k_index_[istring][j];
            if (ik < 0 || ik >= nks) continue;
            if (ow_theta_k_.size() <= static_cast<size_t>(ik)
                || ow_theta_k_[ik].size() < static_cast<size_t>(nocc_use))
                continue;
            const std::vector<double>& theta = ow_theta_k_[ik];
            psi->fix_k(ik);
            const std::complex<double>* c_k = psi->get_pointer();
            any = true;

            // Frozen-C Gram matrix Π[n][m] = (C_k†·C_k)[m][n].
            std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
            for (int n = 0; n < nocc_use; ++n)
                for (int m = 0; m < nocc_use; ++m)
                    for (int a = 0; a < nrow; ++a)
                        Pi[n * nocc_use + m] += std::conj(c_k[a + m * nrow]) * c_k[a + n * nrow];

            // Per-projector-channel weight vectors:
            //   X[iat][lm][m] = Σ_n θ_n·D_n·Π_nm
            //   Z[iat][lm][n] = Σ_m f_m·D_m·Π_mn
            std::vector<std::vector<std::vector<std::complex<double>>>> X(nat);
            std::vector<std::vector<std::vector<std::complex<double>>>> Z(nat);
            for (int iat = 0; iat < nat; ++iat)
            {
                const int r = nproj_per_atom_[iat];
                if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                X[iat].resize(r);
                Z[iat].resize(r);
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                    const auto& DI = kstring_data_[j].D_I[iat][lm];
                    if (DI.size() < static_cast<size_t>(nocc_use)) continue;
                    X[iat][lm].assign(nocc_use, {0.0, 0.0});
                    Z[iat][lm].assign(nocc_use, {0.0, 0.0});
                    for (int m = 0; m < nocc_use; ++m)
                    {
                        std::complex<double> xm(0.0, 0.0);
                        for (int n = 0; n < nocc_use; ++n)
                            xm += theta[n] * DI[n] * Pi[n * nocc_use + m];
                        X[iat][lm][m] = xm;
                    }
                    for (int n = 0; n < nocc_use; ++n)
                    {
                        std::complex<double> zn(0.0, 0.0);
                        for (int m = 0; m < nocc_use; ++m)
                        {
                            const double fm = pelec->wg(ik, m);
                            if (fm == 0.0) continue;
                            zn += fm * DI[m] * Pi[m * nocc_use + n];
                        }
                        Z[iat][lm][n] = zn;
                    }
                }
            }

            for (int iat = 0; iat < nat; ++iat)
            {
                if (static_cast<size_t>(iat) >= lambda.size() || lambda[iat] == 0.0) continue;
                if (X[iat].empty()) continue;
                const double lam = lambda[iat];
                const ModuleBase::Vector3<double> tau0 = ucell.get_tau(iat);
                int T0 = 0, I0 = 0;
                ucell.iat2iait(iat, &I0, &T0);
                const int nw0 = ucell.atoms[T0].nw;
                const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;
                const int nproj0 = max_l_plus_1 * max_l_plus_1;

                AdjacentAtomInfo adjs;
                gd_->Find_atom(ucell, tau0, T0, I0, &adjs);
                for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
                {
                    const int T1 = adjs.ntype[ad];
                    const int I1 = adjs.natom[ad];
                    const int iat1 = ucell.itia2iat(T1, I1);
                    const ModuleBase::Vector3<int> R = adjs.box[ad];
                    const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];
                    const double dist = ucell.cal_dtau(iat, iat1, R).norm() * ucell.lat0;
                    // S_k pair cutoff (same as the dW kernel / compute_real_overlaps).
                    if (dist >= orb_cutoff_[T1] + rm_) continue;
                    if (iat1 == iat && R.x == 0 && R.y == 0 && R.z == 0) continue;

                    const ModuleBase::Vector3<double> dtau = tau0 - tau1;
                    const Atom* atom1 = &ucell.atoms[T1];
                    const int nw1 = atom1->nw;
                    // S_k phase (compute_S_k convention: 2π k·R).
                    const double argk = ModuleBase::TWO_PI * (kstring_data_[j].kvec_d.x * R.x
                                                              + kstring_data_[j].kvec_d.y * R.y
                                                              + kstring_data_[j].kvec_d.z * R.z);
                    const std::complex<double> phase_sk(std::cos(argk), std::sin(argk));

                    for (int iw1 = 0; iw1 < nw1; ++iw1)
                    {
                        const int L1 = atom1->iw2l[iw1];
                        const int N1 = atom1->iw2n[iw1];
                        const int m1 = atom1->iw2m[iw1];
                        const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                        std::vector<std::vector<double>> nlm;
                        intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 1, nlm);
                        if (nlm.empty() || nlm[0].empty()) continue;

                        // First-zeta (l,m) projector channels (same layout as
                        // compute_real_overlaps / deltap_force_stress.hpp).
                        std::vector<double> nlm_target(4 * nproj0, 0.0);
                        int target_L = 0;
                        for (int iw = 0; iw < nw0; ++iw)
                        {
                            const int L0 = ucell.atoms[T0].iw2l[iw];
                            if (L0 != target_L) continue;
                            for (int m = 0; m < 2 * L0 + 1; ++m)
                            {
                                const int idx = L0 * L0 + m;
                                nlm_target[idx] = nlm[0][iw + m];
                                nlm_target[nproj0 + idx] = nlm[1][iw + m];
                                nlm_target[2 * nproj0 + idx] = nlm[2][iw + m];
                                nlm_target[3 * nproj0 + idx] = nlm[3][iw + m];
                            }
                            target_L++;
                        }

                        for (int s = 0; s < npol; ++s)
                        {
                            const int gmu = iat2iwt[iat1] + npol * iw1 + s;
                            for (int lm = 0; lm < nproj0; ++lm)
                            {
                                if (static_cast<size_t>(lm) >= X[iat].size()) continue;
                                if (X[iat][lm].empty()) continue;
                                // inner = Σ_m conj(C[μ,m])·X[m] + Σ_n conj(C[μ,n])·θ_n·Z[n]
                                std::complex<double> inner(0.0, 0.0);
                                for (int m = 0; m < nocc_use; ++m)
                                    inner += std::conj(c_k[gmu + m * nrow]) * X[iat][lm][m];
                                for (int n = 0; n < nocc_use; ++n)
                                    inner += std::conj(c_k[gmu + n * nrow]) * theta[n] * Z[iat][lm][n];
                                if (std::abs(inner) < 1e-30) continue;
                                for (int a = 0; a < 3; ++a)
                                {
                                    const double grad = nlm_target[(1 + a) * nproj0 + lm];
                                    if (grad == 0.0) continue;
                                    // ∂S_k/∂R_bra = +phase·grad, ∂S_k/∂R_ket = −phase·grad;
                                    // F = −∂E_ow/∂R: bra gets −λ·Re[dS·inner], ket +λ·Re[dS·inner].
                                    const double contrib = (phase_sk * grad * inner).real();
                                    f_ow[iat * 3 + a] -= lam * contrib;
                                    f_ow[iat1 * 3 + a] += lam * contrib;
                                }
                            }
                        }
                    }
                }
            }

            // E_ow per-atom split (diagnostic): the same full-Gram-trace
            // contraction as compute_gamma_op_hk's Γ_I^w.
            for (int m = 0; m < nocc_use; ++m)
            {
                const double fm = pelec->wg(ik, m);
                if (fm == 0.0) continue;
                for (int n = 0; n < nocc_use; ++n)
                {
                    const double thn = theta[n];
                    if (thn == 0.0) continue;
                    for (int iat = 0; iat < nat; ++iat)
                    {
                        const int r = nproj_per_atom_[iat];
                        if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                        std::complex<double> t_I(0.0, 0.0);
                        for (int lm = 0; lm < r; ++lm)
                        {
                            if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                            if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(m)) continue;
                            if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                            t_I += std::conj(kstring_data_[j].D_I[iat][lm][m])
                                 * kstring_data_[j].D_I[iat][lm][n];
                        }
                        e_ow += lambda[iat] * fm * thn * (t_I * Pi[n * nocc_use + m]).real();
                    }
                }
            }
        }
    }

    if (!any) return false;
    for (int iat = 0; iat < nat; ++iat)
        for (int a = 0; a < 3; ++a)
            force_out[iat * 3 + a] += f_ow[iat * 3 + a];
#ifdef __MPI
    if (GlobalV::MY_RANK == 0)
#endif
    {
        std::cout << " [DeltaP ow-force] E_ow=" << std::setprecision(10) << e_ow
                  << " Ry  F_ow_max=" << std::setprecision(6);
        double fmax = 0.0;
        for (size_t i = 0; i < f_ow.size(); ++i)
            fmax = std::max(fmax, std::abs(f_ow[i]));
        std::cout << fmax << " Ry/Bohr";
        for (int iat = 0; iat < nat; ++iat)
            for (int a = 0; a < 3; ++a)
                std::cout << " " << std::setprecision(5) << f_ow[iat * 3 + a];
        std::cout << std::endl;
    }
    return true;
}

bool DeltaP::compute_hk_force(const UnitCell& ucell,
                              const psi::Psi<std::complex<double>>* psi,
                              const elecstate::ElecState* pelec,
                              const std::vector<double>& lambda,
                              std::vector<double>& force_out,
                              double& e_hk_out,
                              std::vector<double>* stress_out)
{
    ModuleBase::TITLE("DeltaP", "compute_hk_force");
    ModuleBase::timer::start("DeltaP", "compute_hk_force");

    force_out.assign(nat_ * 3, 0.0);
    e_hk_out = 0.0;
    // F-8 (2026-08-17): H_HK stress output (serial path only).  Zeroed up
    // front so a refused path (MPI) cannot leak a stale stress.
    if (stress_out != nullptr)
    {
        stress_out->assign(6, 0.0);
    }

    if (nppstr_ < 2 || kstring_data_.empty())
    {
        ModuleBase::timer::end("DeltaP", "compute_hk_force");
        return false;
    }

    const int nks = psi->get_nk();
    const int nbands = paraV_->get_wfc_global_nbands();
    const int nrow = paraV_->get_row_size();
    const int ncol = paraV_->get_col_size();

#ifdef __MPI
    int nproc = 1;
    {
        MPI_Comm comm = paraV_->comm();
        if (comm != MPI_COMM_NULL)
            MPI_Comm_size(comm, &nproc);
    }
#else
    const int nproc = 1;
#endif
    // TODO 3.2/3.3 (2026-08-13): the H_HK analytic force is now MPI-enabled.
    // nproc > 1 uses the distributed path: SC/T/Pi/U/dW are computed from
    // the local blocks with pzgemm + the A' row/band partial-sum Allreduce
    // (same family as compute_hk_correction); nproc == 1 keeps the original
    // loop code byte-identical (hard serial A/B constraint).  The serial
    // path assumes a full local matrix (nrow == ncol == nlocal), so the
    // check below is defensive only.
    const bool mpi_path = (nproc > 1);
    if (nproc == 1 && nrow != ncol)
    {
        ModuleBase::WARNING("DeltaP::compute_hk_force",
            "Serial LCAO assumes nrow == ncol (full local matrix).");
        ModuleBase::timer::end("DeltaP", "compute_hk_force");
        return false;
    }

    // ---- ensure k-string data for the INPUT gdir (mirror compute_hk_correction) ----
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
    }

    // ---- S_dk (serial full-row) ----
    if (S_dk_.empty())
    {
        compute_S_dk(ucell);
    }

    // Number of occupied bands (same convention as compute_hk_correction)
    double occ_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occ_bands - std::floor(occ_bands)) > 0.0)
        occ_bands = std::floor(occ_bands) + 1.0;
    const int nocc = static_cast<int>(occ_bands);
    const int nocc_use = std::min(nocc, nbands);
    if (nocc_use <= 0)
    {
        ModuleBase::timer::end("DeltaP", "compute_hk_force");
        return false;
    }

    const std::complex<double> half_i(0.0, 0.5);
    const int nat = nat_;
    const int npol = ucell.get_npol();
    const double dk_step = 1.0 / (nppstr_ - 1);
    double dkv[3] = {0.0, 0.0, 0.0};
    dkv[gdir_ - 1] = dk_step;
    const int* iat2iwt = paraV_->iat2iwt_;

    // Phase gradient factor for the bra-position term.  The S_dk phase is
    // 2π(dk·R − dk·τ_frac) with τ_frac = latvec⁻¹·τ (Direct coordinates) and
    // τ stored in lat0 units, so dτ_frac/dR_Bohr = latvec⁻¹/lat0.  The
    // per-axis derivative factor is therefore −2πi·(latvec⁻¹·dk)_α/lat0
    // (for an orthogonal cell this reduces to −2πi·dk_α/(L_α·lat0)).
    const ModuleBase::Matrix3 latvec_inv = ucell.latvec.Inverse();
    double dkv_grad[3] = {0.0, 0.0, 0.0};
    dkv_grad[0] = dkv[0] * latvec_inv.e11 + dkv[1] * latvec_inv.e21 + dkv[2] * latvec_inv.e31;
    dkv_grad[1] = dkv[0] * latvec_inv.e12 + dkv[1] * latvec_inv.e22 + dkv[2] * latvec_inv.e32;
    dkv_grad[2] = dkv[0] * latvec_inv.e13 + dkv[1] * latvec_inv.e23 + dkv[2] * latvec_inv.e33;

    // acc[iat][alpha*nocc+p] accumulates the full-trace kernel
    // f_p·Σ_{p'}(dW[p']·T_{pp'} + W[p']·U[p][p'])·Π_{p'p} over links;
    // the final force is F_Jα = -Re[(i/2) Σ_p acc] = (1/2) Σ_p Im(acc).
    std::vector<std::vector<std::complex<double>>> acc(
        nat, std::vector<std::complex<double>>(3 * nocc_use, {0.0, 0.0}));
    std::complex<double> e_hk(0.0, 0.0);
    // Route A+ operator observable: per-atom Γ_I^HK accumulator
    // (E_HK = Σ_I λ_I·Γ_I^HK; split before the λ sum; full-trace convention
    // matching compute_hk_correction — T2 verdict 2026-08-04).
    std::vector<std::complex<double>> e_hk_I(nat_, {0.0, 0.0});
    // F-8 (2026-08-17): H_HK stress kernel accumulator, per link
    // stress_kern[α*3+β] = Σ_p f_p Σ_pp' (stress_dW·T + W·stress_U)·Π
    // (serial path only; see the pair-loop kernels below).  Finalized to
    // σ^HK_{αβ} = −0.5·Im(Σ_j stress_kern)/Ω (sign flip vs the B-7 force
    // convention F = +0.5·Im(Σ acc): σ = +∂E/∂ε, F = −∂E/∂R).
    std::vector<std::complex<double>> stress_kern(9, {0.0, 0.0});

    for (int j = 0; j < nppstr_ - 1; ++j)
    {
        const int ik_L = k_index_[0][j];
        const int ik_R = k_index_[0][j + 1];
        if (ik_L >= nks || ik_R >= nks) continue;

        psi->fix_k(ik_L);
        const std::complex<double>* c_L = psi->get_pointer();
        psi->fix_k(ik_R);
        const std::complex<double>* c_R = psi->get_pointer();

        // W_p = Σ_I λ_I Σ_lm |D_I[lm][p]|²  (link j uses the k_L data);
        // w_IJ[n][iat] keeps the per-atom split for Γ_I^HK.
        std::vector<std::vector<double>> w_IJ(nocc_use, std::vector<double>(nat, 0.0));
        std::vector<double> W(nocc_use, 0.0);
        for (int iat = 0; iat < nat; ++iat)
        {
            const int r = nproj_per_atom_[iat];
            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
            for (int n = 0; n < nocc_use; ++n)
            {
                double w = 0.0;
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I[iat].size() > static_cast<size_t>(lm)
                        && kstring_data_[j].D_I[iat][lm].size() > static_cast<size_t>(n))
                    {
                        w += std::norm(kstring_data_[j].D_I[iat][lm][n]);
                    }
                }
                w_IJ[n][iat] = w;
                W[n] += lambda[iat] * w;
            }
        }

#ifdef __MPI
        if (mpi_path)
        {
            // F-8 (2026-08-17): the H_HK stress is serial-only (the strain
            // derivative is a per-pair lattice-vector weighted kernel that the
            // distributed Dloc path does not carry); the cal_force_stress gate
            // refuses MPI + hk-mode + cal_stress before this is reached, so a
            // stress request here is a programming error — leave it zeroed.
            if (stress_out != nullptr)
            {
                ModuleBase::WARNING("DeltaP::compute_hk_force",
                    "H_HK stress is serial-only (MPI path); stress left zero.");
            }
            // TODO 3.2/3.3 (2026-08-13): distributed H_HK force link body.
            // SC = S_dk·C_R is a genuine distributed GEMM (desc × desc_wfc);
            // T/Pi/U/dW are band-space quantities completed by the A'
            // uniform-count Allreduce over local rows × local bands, so the
            // per-atom force kernel below is rank-independent.
            const int nlocal = paraV_->desc[2];
            const int nbands_g = paraV_->desc_wfc[3];
            const int ncol_b = paraV_->ncol_bands;
            const std::complex<double> one_c(1.0, 0.0), zero_c(0.0, 0.0);
            const int one_i = 1;
            const char N_ch = 'N';

            // SC = S_dk · C_R  (nlocal×nlocal desc × nlocal×nbands desc_wfc)
            std::vector<std::complex<double>> SC(nrow * ncol_b, {0.0, 0.0});
            ScalapackConnector::gemm(N_ch, N_ch, nlocal, nbands_g, nlocal,
                                     one_c, S_dk_.data(), one_i, one_i, paraV_->desc,
                                     c_R, one_i, one_i, paraV_->desc_wfc,
                                     zero_c, SC.data(), one_i, one_i, paraV_->desc_wfc);

            // T_full / Pi: full-band partials over local rows × ALL band
            // pairs.  With dim1 > 1 the plain A' local-band loop never fills
            // pairs spanning two process columns (TODO 3.2/3.3 band-pair
            // completeness fix) — the full band set for this rank's rows is
            // gathered within the process-row group first, so every pair is
            // covered and the row sum below is completed by a single
            // Allreduce (one contribution per row group).
            std::vector<std::complex<double>> T_full(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
            std::vector<std::complex<double>> cL_all, SC_all;
            std::vector<int> gpos;
            gather_band_columns(paraV_, c_L, cL_all, gpos);
            gather_band_columns(paraV_, SC.data(), SC_all, gpos);
            for (int g1 = 0; g1 < nocc_use; ++g1)
            {
                const int p1 = gpos[g1];
                if (p1 < 0) continue;
                for (int g2 = 0; g2 < nocc_use; ++g2)
                {
                    const int p2 = gpos[g2];
                    if (p2 < 0) continue;
                    std::complex<double> tsum(0.0, 0.0), psum(0.0, 0.0);
                    for (int a = 0; a < nrow; ++a)
                    {
                        tsum += std::conj(cL_all[a + p1 * nrow]) * SC_all[a + p2 * nrow];
                        psum += std::conj(cL_all[a + p2 * nrow]) * cL_all[a + p1 * nrow];
                    }
                    T_full[g1 * nocc_use + g2] += tsum;
                    Pi[g1 * nocc_use + g2] += psum;
                }
            }

            // U_Jα = (C_L† ∂S_dk/∂R_Jα C_R): under MPI the serial path's
            // direct (row, col) accumulation is impossible — C_R exists only
            // for local ROWS while the derivative block spans local rows ×
            // local COLS — so the chain is: build the (nrow×ncol) local
            // block D_Jα of ∂S_dk/∂R_Jα (same two-center kernels as serial),
            // V_Jα = D_Jα·C_R by pzgemm, then the local-row partial of
            // C_L†·V_Jα completed by the A' Allreduce below.  dW_Jα =
            // ∂W/∂R_Jα is a local-row partial sum of the projector-
            // derivative kernel (A' as well).
            std::vector<std::complex<double>> U_part(
                static_cast<size_t>(nat) * 3 * nocc_use * nocc_use, {0.0, 0.0});
            std::vector<double> dW_part(static_cast<size_t>(nat) * 3 * nocc_use, 0.0);
            // Dloc[(J*3 + a)*nrow*ncol + lr + lc*nrow] = local block element
            // of ∂S_dk/∂R_Jα at (global row gmu, global col gnu).
            std::vector<std::complex<double>> Dloc(
                static_cast<size_t>(nat) * 3 * nrow * ncol, {0.0, 0.0});

            for (int iat = 0; iat < nat; ++iat)
            {
                const ModuleBase::Vector3<double> tau0 = ucell.get_tau(iat);
                int T0 = 0, I0 = 0;
                ucell.iat2iait(iat, &I0, &T0);
                const int nw0 = ucell.atoms[T0].nw;
                const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;
                const int nproj0 = max_l_plus_1 * max_l_plus_1;
                // Direct (fractional) bra position for the S_dk phase — see
                // compute_S_dk for the τ-unit rationale (B-6).
                const ModuleBase::Vector3<double> taud0 = ucell.atoms[T0].taud[I0];

                AdjacentAtomInfo adjs;
                gd_->Find_atom(ucell, tau0, T0, I0, &adjs);

                for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
                {
                    const int T1 = adjs.ntype[ad];
                    const int I1 = adjs.natom[ad];
                    const int iat1 = ucell.itia2iat(T1, I1);
                    const ModuleBase::Vector3<int> R = adjs.box[ad];
                    const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];
                    const double dist = ucell.cal_dtau(iat, iat1, R).norm() * ucell.lat0;

                    const bool sdk_pair = dist <= orb_cutoff_[T0] + orb_cutoff_[T1];
                    const bool sk_pair = dist < orb_cutoff_[T1] + rm_;
                    if (!sdk_pair && !sk_pair) continue;

                    const ModuleBase::Vector3<double> dtau = tau0 - tau1;
                    const Atom* atom1 = &ucell.atoms[T1];
                    const int nw1 = atom1->nw;

                    // S_dk phase (compute_S_dk convention: 2π(dk·R − dk·τ_bra))
                    double arg = ModuleBase::TWO_PI * (dkv[0] * R.x + dkv[1] * R.y + dkv[2] * R.z
                                                       - dkv[0] * taud0.x - dkv[1] * taud0.y - dkv[2] * taud0.z);
                    const std::complex<double> phase_sdk(std::cos(arg), std::sin(arg));
                    // S_k phase (compute_S_k convention: 2π k·R)
                    double argk = ModuleBase::TWO_PI * (kstring_data_[j].kvec_d.x * R.x
                                                        + kstring_data_[j].kvec_d.y * R.y
                                                        + kstring_data_[j].kvec_d.z * R.z);
                    const std::complex<double> phase_sk(std::cos(argk), std::sin(argk));

                    // ---------- ∂S_dk/∂R kernel (full orbital set) ----------
                    if (sdk_pair)
                    {
                        for (int iw1 = 0; iw1 < nw1; ++iw1)
                        {
                            const int L1 = atom1->iw2l[iw1];
                            const int N1 = atom1->iw2n[iw1];
                            const int m1 = atom1->iw2m[iw1];
                            const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                            std::vector<std::vector<double>> nlm;
                            overlap_intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 1, nlm);
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
                                    if (lr < 0 || lc < 0) continue;
                                    for (int a = 0; a < 3; ++a)
                                    {
                                        const double g = nlm[1 + a][iw0];
                                        // Orbital part: ∂ov/∂R_bra = +g,
                                        // ∂ov/∂R_ket = -g.
                                        if (g != 0.0)
                                        {
                                            Dloc[(static_cast<size_t>(iat) * 3 + a) * nrow * ncol
                                                 + lr + lc * nrow] += phase_sdk * g;
                                            Dloc[(static_cast<size_t>(iat1) * 3 + a) * nrow * ncol
                                                 + lr + lc * nrow] -= phase_sdk * g;
                                        }
                                        // Phase part: ∂phase/∂R_bra,α =
                                        // -2πi·dkv_grad[α]/lat0 (bra only,
                                        // unconditional).
                                        Dloc[(static_cast<size_t>(iat) * 3 + a) * nrow * ncol
                                             + lr + lc * nrow] -=
                                            std::complex<double>(0.0, ModuleBase::TWO_PI * dkv_grad[a] / ucell.lat0)
                                            * phase_sdk * ov;
                                    }
                                }
                            }
                        }
                    }

                    // ---------- ∂S_k/∂R kernel for W (first-zeta set) ----------
                    if (sk_pair)
                    {
                        for (int iw1 = 0; iw1 < nw1; ++iw1)
                        {
                            const int L1 = atom1->iw2l[iw1];
                            const int N1 = atom1->iw2n[iw1];
                            const int m1 = atom1->iw2m[iw1];
                            const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                            std::vector<std::vector<double>> nlm;
                            intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 1, nlm);
                            if (nlm.empty() || nlm[0].empty()) continue;

                            std::vector<double> nlm_target(4 * nproj0, 0.0);
                            int target_L = 0;
                            for (int iw = 0; iw < ucell.atoms[T0].nw; ++iw)
                            {
                                const int L0 = ucell.atoms[T0].iw2l[iw];
                                if (L0 != target_L) continue;
                                for (int m = 0; m < 2 * L0 + 1; ++m)
                                {
                                    const int idx = L0 * L0 + m;
                                    nlm_target[idx] = nlm[0][iw + m];
                                    nlm_target[nproj0 + idx] = nlm[1][iw + m];
                                    nlm_target[2 * nproj0 + idx] = nlm[2][iw + m];
                                    nlm_target[3 * nproj0 + idx] = nlm[3][iw + m];
                                }
                                target_L++;
                            }

                            for (int s = 0; s < npol; ++s)
                            {
                                const int gmu = iat2iwt[iat1] + npol * iw1 + s;
                                const int lr = paraV_->global2local_row(gmu);
                                if (lr < 0) continue;
                                for (int lm = 0; lm < nproj0; ++lm)
                                {
                                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                                    const auto& DI = kstring_data_[j].D_I[iat][lm];
                                    // Loop over LOCAL band columns: c_L columns
                                    // are local bands (desc_wfc layout); the
                                    // global band p = local2global_col(n1) is
                                    // used for the D_I / dW_part indexing.
                                    for (int n1 = 0; n1 < ncol_b; ++n1)
                                    {
                                        const int p = paraV_->local2global_col(n1);
                                        if (p < 0 || p >= nocc_use) continue;
                                        const std::complex<double> Dp = DI[p];
                                        if (std::abs(Dp) < 1e-15) continue;
                                        const std::complex<double> Cmu = c_L[lr + n1 * nrow];
                                        for (int a = 0; a < 3; ++a)
                                        {
                                            const double grad = nlm_target[(1 + a) * nproj0 + lm];
                                            if (grad == 0.0) continue;
                                            // ∂S_k/∂R_bra = phase·grad,
                                            // ∂S_k/∂R_ket = −phase·grad.
                                            const double re_term = (std::conj(Dp) * std::conj(phase_sk * grad) * Cmu).real();
                                            dW_part[(static_cast<size_t>(iat) * 3 + a) * nocc_use + p]
                                                += 2.0 * lambda[iat] * re_term;
                                            dW_part[(static_cast<size_t>(iat1) * 3 + a) * nocc_use + p]
                                                -= 2.0 * lambda[iat] * re_term;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // V_Jα = D_Jα·C_R (pzgemm), then the local-row partial of U.
            for (int J = 0; J < nat; ++J)
            {
                for (int a = 0; a < 3; ++a)
                {
                    const std::complex<double>* D = Dloc.data()
                        + (static_cast<size_t>(J) * 3 + a) * nrow * ncol;
                    std::vector<std::complex<double>> V(nrow * ncol_b, {0.0, 0.0});
                    ScalapackConnector::gemm(N_ch, N_ch, nlocal, nbands_g, nlocal,
                                             one_c, D, one_i, one_i, paraV_->desc,
                                             c_R, one_i, one_i, paraV_->desc_wfc,
                                             zero_c, V.data(), one_i, one_i, paraV_->desc_wfc);
                    // Full-band gather (same band-pair completeness fix as
                    // T/Pi above) before the local-row partial of C_L†·V.
                    std::vector<std::complex<double>> V_all;
                    gather_band_columns(paraV_, V.data(), V_all, gpos);
                    std::complex<double>* UJ = U_part.data()
                        + (static_cast<size_t>(J) * 3 + a) * nocc_use * nocc_use;
                    for (int g1 = 0; g1 < nocc_use; ++g1)
                    {
                        const int p1 = gpos[g1];
                        if (p1 < 0) continue;
                        for (int g2 = 0; g2 < nocc_use; ++g2)
                        {
                            const int p2 = gpos[g2];
                            if (p2 < 0) continue;
                            std::complex<double> usum(0.0, 0.0);
                            for (int r = 0; r < nrow; ++r)
                            {
                                usum += std::conj(cL_all[r + p1 * nrow]) * V_all[r + p2 * nrow];
                            }
                            UJ[g1 * nocc_use + g2] += usum;
                        }
                    }
                }
            }

            // Row-sum Allreduce: T/Pi/U partials are identical within each
            // process-row group (same rows, gathered bands), so exactly one
            // rank per row group contributes (coord[1] == 0); dW is a plain
            // A' partial over this rank's own rows and needs every rank.
            {
                MPI_Comm comm = paraV_->comm();
                if (comm != MPI_COMM_NULL)
                {
                    if (paraV_->coord[1] != 0)
                    {
                        T_full.assign(T_full.size(), {0.0, 0.0});
                        Pi.assign(Pi.size(), {0.0, 0.0});
                        U_part.assign(U_part.size(), {0.0, 0.0});
                    }
                    MPI_Allreduce(MPI_IN_PLACE, T_full.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                    MPI_Allreduce(MPI_IN_PLACE, Pi.data(), 2 * nocc_use * nocc_use,
                                  MPI_DOUBLE, MPI_SUM, comm);
                    MPI_Allreduce(MPI_IN_PLACE, U_part.data(),
                                  2 * static_cast<int>(U_part.size()), MPI_DOUBLE, MPI_SUM, comm);
                    MPI_Allreduce(MPI_IN_PLACE, dW_part.data(),
                                  static_cast<int>(dW_part.size()), MPI_DOUBLE, MPI_SUM, comm);
                }
            }

            // ---- accumulate this link into the per-atom force kernel ----
            // Full-trace frozen-C derivative (same kernel as the serial
            // path, now with rank-independent band-space quantities):
            //   acc[iat][a*nocc+p] = f_p·Σ_{p'}(dW[p']·T_{pp'} + W[p']·U[p][p'])·Π_{p'p}
            for (int iat = 0; iat < nat; ++iat)
            {
                for (int a = 0; a < 3; ++a)
                {
                    for (int p = 0; p < nocc_use; ++p)
                    {
                        const double fp = pelec->wg(ik_L, p);
                        if (fp == 0.0) continue;
                        std::complex<double> kern(0.0, 0.0);
                        for (int pp = 0; pp < nocc_use; ++pp)
                        {
                            kern += (dW_part[(static_cast<size_t>(iat) * 3 + a) * nocc_use + pp]
                                         * T_full[p * nocc_use + pp]
                                     + W[pp] * U_part[(static_cast<size_t>(iat) * 3 + a) * nocc_use * nocc_use
                                                      + p * nocc_use + pp])
                                    * Pi[pp * nocc_use + p];
                        }
                        acc[iat][a * nocc_use + p] += fp * kern;
                    }
                }
            }
            // Full-trace E_HK and per-atom Γ_I^HK (split before the λ sum).
            for (int p = 0; p < nocc_use; ++p)
            {
                const double fp = pelec->wg(ik_L, p);
                if (fp == 0.0) continue;
                for (int pp = 0; pp < nocc_use; ++pp)
                {
                    const std::complex<double> tpi = T_full[p * nocc_use + pp] * Pi[pp * nocc_use + p];
                    e_hk += fp * W[pp] * tpi;
                    for (int iat = 0; iat < nat; ++iat)
                    {
                        e_hk_I[iat] += fp * w_IJ[pp][iat] * tpi;
                    }
                }
            }

        }
        else
#endif
        {
        // Serial path (nproc == 1): byte-identical to the pre-Q2 code.
        // SC = S_dk * C_R (nrow x nocc_use)
        std::vector<std::complex<double>> SC(nrow * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int a = 0; a < nrow; ++a)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int b = 0; b < ncol; ++b)
                {
                    sum += S_dk_[a + b * nrow] * c_R[b + p * nrow];
                }
                SC[a + p * nrow] = sum;
            }
        }
        // Full-trace (T2 verdict 2026-08-04): the applied H_HK expectation is
        // −0.5·Im[Σ_p f_p Σ_{p'} W[p']·T_{pp'}·Π_{p'p}] with the FULL link
        // matrix T_{pp'} = (C_L† S_dk C_R)_{pp'} and the occupied Gram matrix
        // Π_{p'p} = (C_L† C_L)_{p'p} of the LEFT coefficients (frozen-C
        // constant — the R-derivative chain is unchanged).  The pre-T2
        // diagonal form (T_pp, Π = I) overstated the coupling by ~18% in the
        // non-orthogonal LCAO basis.
        std::vector<std::complex<double>> T_full(nocc_use * nocc_use, {0.0, 0.0});
        std::vector<std::complex<double>> Pi(nocc_use * nocc_use, {0.0, 0.0});
        for (int p = 0; p < nocc_use; ++p)
        {
            for (int pp = 0; pp < nocc_use; ++pp)
            {
                for (int a = 0; a < nrow; ++a)
                {
                    T_full[p * nocc_use + pp] += std::conj(c_L[a + p * nrow]) * SC[a + pp * nrow];
                    Pi[p * nocc_use + pp] += std::conj(c_L[a + pp * nrow]) * c_L[a + p * nrow];
                }
            }
        }

        // ---- Per-atom derivative kernels for this link ----
        // U[iat][(alpha*nocc+p)*nocc+pp] = (C_L† ∂S_dk/∂R_Jα C_R)_{p,pp}
        // dW[iat][alpha*nocc+p] = ∂W_p/∂R_Jα  (real, W is real)
        std::vector<std::vector<std::complex<double>>> U(
            nat, std::vector<std::complex<double>>(3 * nocc_use * nocc_use, {0.0, 0.0}));
        std::vector<std::vector<double>> dW(nat, std::vector<double>(3 * nocc_use, 0.0));

        // F-8 (2026-08-17): H_HK stress kernels for this link, strain
        // derivative of the same E_HK = −0.5·Im[Σ f W T Π].  Strain
        // convention (RouteA++ §4.1): the cell stretches, fractional
        // coordinates fixed ⇒ the S_dk/S_k phases are strain-invariant
        // (∂phase/∂ε = 0 theorem, reduced-k fixed) and every two-center
        // orbital derivative enters with the full pair separation weight
        // d_β = (tau0 − tau1)_β·lat0 (the bra−ket Cartesian vector
        // including the periodic image):
        //   stress_U[α][β][p][pp] = Σ_pairs ½d_β·(C_L† ∂S_dk/∂R_bra,α C_R)_{p,pp}
        //   stress_dW[α][β][p']    = Σ_pairs ½d_β·∂W_p'/∂R_bra,α
        // (bra-only, mirroring the per-atom force kernels; the ½ is the
        // double-visit factor — each unordered pair is visited twice with
        // equal-sign contributions (weight and derivative flip together)).
        // The final σ^HK_{αβ} = +0.5·Im(Σ_j kern)/Ω (σ = −(1/Ω)·dE_HK/dε,
        // dE_HK/dε = −0.5·Im(Σ kern) ⇒ σ = +0.5·Im(Σ kern)/Ω).
        std::vector<std::complex<double>> stress_U(
            9 * nocc_use * nocc_use, {0.0, 0.0});
        std::vector<double> stress_dW(9 * nocc_use, 0.0);

        for (int iat = 0; iat < nat; ++iat)
        {
            const ModuleBase::Vector3<double> tau0 = ucell.get_tau(iat);
            int T0 = 0, I0 = 0;
            ucell.iat2iait(iat, &I0, &T0);
            const int nw0 = ucell.atoms[T0].nw;
            const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;
            const int nproj0 = max_l_plus_1 * max_l_plus_1;
            // Direct (fractional) bra position for the S_dk phase — see
            // compute_S_dk for the τ-unit rationale (B-6).  Keep taud here;
            // do not substitute get_tau() (lat0-unit) into this phase.
            const ModuleBase::Vector3<double> taud0 = ucell.atoms[T0].taud[I0];

            AdjacentAtomInfo adjs;
            gd_->Find_atom(ucell, tau0, T0, I0, &adjs);

            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = ucell.itia2iat(T1, I1);
                const ModuleBase::Vector3<int> R = adjs.box[ad];
                const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];
                const double dist = ucell.cal_dtau(iat, iat1, R).norm() * ucell.lat0;

                // Pair is used by the S_dk path (compute_S_dk cutoff) and/or the
                // S_k path (compute_real_overlaps cutoff: rm_).
                const bool sdk_pair = dist <= orb_cutoff_[T0] + orb_cutoff_[T1];
                const bool sk_pair = dist < orb_cutoff_[T1] + rm_;
                if (!sdk_pair && !sk_pair) continue;

                const ModuleBase::Vector3<double> dtau = tau0 - tau1;
                // F-8: pair strain weight in Cartesian (Bohr) — under the
                // fixed-fractional strain convention (RouteA++ §4.1) the
                // full bra−ket separation d = tau0 − tau1 stretches by
                // (1+ε), so ∂d_γ/∂ε_{αβ} = δ_{γα}·d_β and every two-center
                // orbital derivative enters the stress with the weight d_β
                // (NOT just the lattice part R·A — the τ-difference scales
                // with the cell as well).  dtau is already the Cartesian
                // separation including the periodic image (adjacent_tau).
                const ModuleBase::Vector3<double> d_bohr = dtau * ucell.lat0;
                const Atom* atom1 = &ucell.atoms[T1];
                const int nw1 = atom1->nw;

                // S_dk phase (compute_S_dk convention: 2π(dk·R − dk·τ_bra))
                double arg = ModuleBase::TWO_PI * (dkv[0] * R.x + dkv[1] * R.y + dkv[2] * R.z
                                                   - dkv[0] * taud0.x - dkv[1] * taud0.y - dkv[2] * taud0.z);
                const std::complex<double> phase_sdk(std::cos(arg), std::sin(arg));
                // S_k phase (compute_S_k convention: 2π k·R)
                double argk = ModuleBase::TWO_PI * (kstring_data_[j].kvec_d.x * R.x
                                                    + kstring_data_[j].kvec_d.y * R.y
                                                    + kstring_data_[j].kvec_d.z * R.z);
                const std::complex<double> phase_sk(std::cos(argk), std::sin(argk));

                // ---------- ∂S_dk/∂R kernel (full orbital set, overlap_intor_) ----------
                if (sdk_pair)
                {
                    // DEBUG: per-pair accumulators to verify U phase consistency with S_dk/T
                    std::vector<std::complex<double>> tacc(nocc_use, {0.0, 0.0});
                    std::vector<std::complex<double>> phaseacc(nocc_use, {0.0, 0.0});
                    for (int iw1 = 0; iw1 < nw1; ++iw1)
                    {
                        const int L1 = atom1->iw2l[iw1];
                        const int N1 = atom1->iw2n[iw1];
                        const int m1 = atom1->iw2m[iw1];
                        const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                        std::vector<std::vector<double>> nlm;
                        overlap_intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 1, nlm);
                        if (nlm.empty() || nlm[0].empty()) continue;
                        for (int iw0 = 0; iw0 < nw0; ++iw0)
                        {
                            const double ov = nlm[0][iw0];
                            if (std::abs(ov) < 1e-15) continue;
                            for (int s = 0; s < npol; ++s)
                            {
                                const int gmu = iat2iwt[iat] + npol * iw0 + s;
                                const int gnu = iat2iwt[iat1] + npol * iw1 + s;
                                // Full-trace (T2 verdict 2026-08-04): U is the
                                // FULL (nocc×nocc) link-derivative matrix
                                // U[p][pp] = (C_L† ∂S_dk/∂R C_R)_{p,pp}; the
                                // diagonal-only form assumed Π = I.
                                for (int p = 0; p < nocc_use; ++p)
                                {
                                    for (int pp = 0; pp < nocc_use; ++pp)
                                    {
                                        const std::complex<double> cb = std::conj(c_L[gmu + p * nrow]);
                                        const std::complex<double> ck = c_R[gnu + pp * nrow];
                                        for (int a = 0; a < 3; ++a)
                                        {
                                            const double g = nlm[1 + a][iw0];
                                            // Orbital part: ∂ov/∂R_bra = +g, ∂ov/∂R_ket = -g.
                                            // Summed over all atoms these cancel pair-by-pair.
                                            if (g != 0.0)
                                            {
                                                U[iat][(a * nocc_use + p) * nocc_use + pp] += cb * (phase_sdk * g) * ck;
                                                U[iat1][(a * nocc_use + p) * nocc_use + pp] -= cb * (phase_sdk * g) * ck;
                                                // F-8 stress: bra-only × the
                                                // full pair separation d_β
                                                // (orbital part; the phase part
                                                // below is strain-invariant —
                                                // §4.2(b)).  Gated: force-only
                                                // runs keep the pre-F-8 hot path.
                                                if (stress_out != nullptr)
                                                {
                                                    // F-8 (2026-08-17): the pair loop visits every
                                                    // unordered pair TWICE (bra=A and bra=B); the
                                                    // reversed visit flips BOTH the weight d→−d and
                                                    // the bra derivative g→−g, so both visits
                                                    // accumulate the same sign.  The pair-form
                                                    // Σ_pairs d_β·∂E/∂R_bra,α counts each pair once
                                                    // ⇒ halve the weight (0.5·d_β).
                                                    for (int b = 0; b < 3; ++b)
                                                    {
                                                        stress_U[(a * 3 + b) * nocc_use * nocc_use
                                                                 + p * nocc_use + pp]
                                                            += 0.5 * d_bohr[b] * cb * (phase_sdk * g) * ck;
                                                    }
                                                }
                                            }
                                            // Phase part: ∂phase/∂R_bra,α = -2πi·dkv_grad[α]/lat0
                                            // where dkv_grad = latvec⁻¹·dk (fractional-τ derivative).
                                            // The S_dk phase (2π(dk·R - dk·τ_bra)) depends on the
                                            // bra position for EVERY pair (including self-pairs and
                                            // s-channels where the orbital derivative g vanishes),
                                            // so this term must be accumulated unconditionally.
                                            U[iat][(a * nocc_use + p) * nocc_use + pp] -= cb
                                                * (std::complex<double>(0.0, ModuleBase::TWO_PI * dkv_grad[a] / ucell.lat0)
                                                   * phase_sdk * ov)
                                                * ck;
                                        }
                                        // DEBUG accumulators (diagonal band p, axis z)
                                        if (pp == p)
                                        {
                                            tacc[p] += phase_sdk * ov * cb * ck;
                                            phaseacc[p] += -std::complex<double>(0.0, ModuleBase::TWO_PI * dkv_grad[2] / ucell.lat0)
                                                           * phase_sdk * ov * cb * ck;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ---------- ∂S_k/∂R kernel for W (first-zeta projector set, intor_) ----------
                if (sk_pair)
                {
                    for (int iw1 = 0; iw1 < nw1; ++iw1)
                    {
                        const int L1 = atom1->iw2l[iw1];
                        const int N1 = atom1->iw2n[iw1];
                        const int m1 = atom1->iw2m[iw1];
                        const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                        std::vector<std::vector<double>> nlm;
                        intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 1, nlm);
                        if (nlm.empty() || nlm[0].empty()) continue;

                        // Extract the first-zeta (l,m) channels with the four
                        // derivative blocks, same channel layout as
                        // compute_real_overlaps / deltap_force_stress.hpp.
                        std::vector<double> nlm_target(4 * nproj0, 0.0);
                        int target_L = 0;
                        for (int iw = 0; iw < ucell.atoms[T0].nw; ++iw)
                        {
                            const int L0 = ucell.atoms[T0].iw2l[iw];
                            if (L0 != target_L) continue;
                            for (int m = 0; m < 2 * L0 + 1; ++m)
                            {
                                const int idx = L0 * L0 + m;
                                nlm_target[idx] = nlm[0][iw + m];
                                nlm_target[nproj0 + idx] = nlm[1][iw + m];
                                nlm_target[2 * nproj0 + idx] = nlm[2][iw + m];
                                nlm_target[3 * nproj0 + idx] = nlm[3][iw + m];
                            }
                            target_L++;
                        }

                        for (int s = 0; s < npol; ++s)
                        {
                            const int gmu = iat2iwt[iat1] + npol * iw1 + s;
                            for (int lm = 0; lm < nproj0; ++lm)
                            {
                                if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                                const auto& DI = kstring_data_[j].D_I[iat][lm];
                                for (int p = 0; p < nocc_use; ++p)
                                {
                                    const std::complex<double> Dp = DI[p];
                                    if (std::abs(Dp) < 1e-15) continue;
                                    const std::complex<double> Cmu = c_L[gmu + p * nrow];
                                    for (int a = 0; a < 3; ++a)
                                    {
                                        const double grad = nlm_target[(1 + a) * nproj0 + lm];
                                        if (grad == 0.0) continue;
                                        // ∂S_k/∂R_bra = phase·grad, ∂S_k/∂R_ket = −phase·grad;
                                        // ∂D/∂R = conj(∂S_k/∂R)·C, so the W derivative is
                                        // 2 Re[D* · conj(∂S_k/∂R) · C].
                                        const double re_term = (std::conj(Dp) * std::conj(phase_sk * grad) * Cmu).real();
                                        dW[iat][a * nocc_use + p] += 2.0 * lambda[iat] * re_term;
                                        dW[iat1][a * nocc_use + p] -= 2.0 * lambda[iat] * re_term;
                                        // F-8 stress: bra-only × the full
                                        // pair separation d_β (∂W/∂ε_{αβ} =
                                        // d_β·∂W/∂R_bra,α; σ = −0.5·Im(Σkern)/Ω
                                        // mirrors F = +0.5·Im(Σ acc) with the
                                        // sign flip from σ = +∂E/∂ε vs
                                        // F = −∂E/∂R).  Gated like the
                                        // stress_U kernel above.
                                        if (stress_out != nullptr)
                                        {
                                            // F-8 (2026-08-17): same double-visit halving as
                                            // stress_U (0.5·d_β) — the reversed visit contributes
                                            // d′·(∂W/∂R_bra) = (−d)·(−∂W/∂R_bra) = +d·∂W/∂R_bra.
                                            for (int b = 0; b < 3; ++b)
                                            {
                                                stress_dW[(a * 3 + b) * nocc_use + p]
                                                    += 0.5 * d_bohr[b] * (2.0 * lambda[iat] * re_term);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- accumulate this link into the per-atom force kernel ----
        // Full-trace frozen-C derivative of
        // E_HK = −0.5·Im[Σ_p f_p Σ_{p'} W[p']·T_{pp'}·Π_{p'p}]:
        //   acc[iat][a*nocc+p] = f_p·Σ_{p'}(dW[p']·T_{pp'} + W[p']·U[p][p'])·Π_{p'p}
        for (int iat = 0; iat < nat; ++iat)
        {
            for (int a = 0; a < 3; ++a)
            {
                for (int p = 0; p < nocc_use; ++p)
                {
                    const double fp = pelec->wg(ik_L, p);
                    if (fp == 0.0) continue;
                    std::complex<double> kern(0.0, 0.0);
                    for (int pp = 0; pp < nocc_use; ++pp)
                    {
                        kern += (dW[iat][a * nocc_use + pp] * T_full[p * nocc_use + pp]
                                 + W[pp] * U[iat][(a * nocc_use + p) * nocc_use + pp])
                                * Pi[pp * nocc_use + p];
                    }
                    acc[iat][a * nocc_use + p] += fp * kern;
                }
            }
        }
        // F-8: per-link H_HK stress kernel (mirrors the acc kernel above):
        //   stress_kern[α*3+β] = Σ_p f_p Σ_pp' (stress_dW[αβ][pp']·T[p][pp']
        //                       + W[pp']·stress_U[αβ][p][pp'])·Π[pp'][p]
        // with the strain-derivative kernels accumulated in the pair loops
        // (bra-only × pair lattice vector; phase part strain-invariant).
        if (stress_out != nullptr)
        {
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    for (int p = 0; p < nocc_use; ++p)
                    {
                        const double fp = pelec->wg(ik_L, p);
                        if (fp == 0.0) continue;
                        std::complex<double> kern(0.0, 0.0);
                        std::complex<double> kern_u(0.0, 0.0);
                        std::complex<double> kern_dw(0.0, 0.0);
                        for (int pp = 0; pp < nocc_use; ++pp)
                        {
                            kern_u += W[pp] * stress_U[(a * 3 + b) * nocc_use * nocc_use
                                                       + p * nocc_use + pp]
                                      * Pi[pp * nocc_use + p];
                            kern_dw += stress_dW[(a * 3 + b) * nocc_use + pp] * T_full[p * nocc_use + pp]
                                       * Pi[pp * nocc_use + p];
                            kern += (stress_dW[(a * 3 + b) * nocc_use + pp] * T_full[p * nocc_use + pp]
                                     + W[pp] * stress_U[(a * 3 + b) * nocc_use * nocc_use
                                                        + p * nocc_use + pp])
                                    * Pi[pp * nocc_use + p];
                        }
                        stress_kern[a * 3 + b] += fp * kern;
                    }
                }
            }
        }
        // Full-trace E_HK and per-atom Γ_I^HK (split before the λ sum).
        for (int p = 0; p < nocc_use; ++p)
        {
            const double fp = pelec->wg(ik_L, p);
            if (fp == 0.0) continue;
            for (int pp = 0; pp < nocc_use; ++pp)
            {
                const std::complex<double> tpi = T_full[p * nocc_use + pp] * Pi[pp * nocc_use + p];
                e_hk += fp * W[pp] * tpi;
                for (int iat = 0; iat < nat; ++iat)
                {
                    e_hk_I[iat] += fp * w_IJ[pp][iat] * tpi;
                }
            }
        }

        }
    }

    // F_Jα = -Re[(i/2) Σ_p acc] = (1/2) Σ_p Im(acc);  E_HK = Re[(i/2) e_hk]
    for (int iat = 0; iat < nat; ++iat)
    {
        for (int a = 0; a < 3; ++a)
        {
            double f = 0.0;
            for (int p = 0; p < nocc_use; ++p)
            {
                f += acc[iat][a * nocc_use + p].imag();
            }
            force_out[iat * 3 + a] = 0.5 * f;
        }
    }
    e_hk_out = -0.5 * e_hk.imag();
    // F-8 (2026-08-17): finalize the H_HK stress (serial path only).
    // σ^HK_{αβ} = −0.5·Im(Σ_j stress_kern_{αβ})/Ω.  The −0.5·Im mirrors the
    // B-7 force convention F = +0.5·Im(Σ acc) with the sign flip because
    // σ = +∂E/∂ε while F = −∂E/∂R (ε is a symmetric strain; the kernels
    // carry the +d_β×bra derivative weight).  The Voigt order
    // [xx,xy,xz,yy,yz,zz] matches cal_force_IJR / FORCE_STRESS.  The MPI
    // path is refused (stress left zero) and gated earlier by
    // cal_force_stress.
    if (stress_out != nullptr)
    {
        static const int voigt[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                // σ^HK_{αβ} = +0.5·Im(Σ_j stress_kern_{αβ})/Ω: σ = −(1/Ω)·dE_HK/dε
                // and dE_HK/dε = −0.5·Im(Σ kern) ⇒ σ = +0.5·Im(Σ kern)/Ω.
                (*stress_out)[voigt[a][b]] = 0.5 * stress_kern[a * 3 + b].imag() / ucell.omega;
            }
        }
    }
    for (int iat = 0; iat < nat; ++iat)
    {
        gamma_op_hk_[iat] = -0.5 * e_hk_I[iat].imag();
    }
    // T-6' (R1, 2026-08-12): in the exact weight-channel operator mode the
    // H_HR projector (and its A1/A2 forces) is replaced by the k-local H_ow;
    // its geometric force is added here (frozen C/θ, same two-center ∂S_k/∂R
    // kernel family as A1/dW).  The force_stress.hpp ow gate zeroes the H_HR
    // contribution so the printed total includes B (H_HK) + F_ow.
    if (operator_mode_ == "ow")
    {
        compute_ow_force(ucell, psi, pelec, lambda, force_out);
    }

    ModuleBase::timer::end("DeltaP", "compute_hk_force");
    return true;
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
    // Mirror the loaded branch into the frozen continuity anchor (Phase
    // 0.3-lite): a single-shot (non-SCF) run anchors the global branch
    // selection to the persisted reference as well.
    ref_gamma_ = W_prev_;
    has_ref_gamma_ = true;
    std::cout << " DeltaP: loaded branch state from " << fname << std::endl;
}

// Seed the frozen continuity anchor (ref_gamma_) from branch.dat.  Unlike
// load_branch(), this does NOT touch W_prev_ (the legacy per-SCF branch
// state), so operator-mode SCF runs keep the old probe -> W_prev_ lifecycle
// while the Stage-B anchor stays frozen (T4a wrong-branch lock-in fix).
void DeltaP::load_branch_ref()
{
    const std::string fname = "deltap_branch.dat";
    std::ifstream ifs(fname);
    if (!ifs.is_open()) return;

    int nat_file = 0;
    ifs >> nat_file;
    if (nat_file != nat_)
    {
        std::cerr << "DeltaP: branch file nat=" << nat_file
                  << " != current nat=" << nat_ << ", ignoring (branch ref)"
                  << std::endl;
        return;
    }

    ref_gamma_.resize(nat_);
    for (int iat = 0; iat < nat_; ++iat)
    {
        double gx = 0.0, gy = 0.0, gz = 0.0;
        ifs >> gx >> gy >> gz;
        ref_gamma_[iat] = ModuleBase::Vector3<double>(gx, gy, gz);
    }
    has_ref_gamma_ = true;
    std::cout << " DeltaP: loaded frozen branch reference from " << fname << std::endl;
}

// Freeze the continuity anchor to the most recent gamma reading.  W_prev_
// holds the last computation's gamma_I (updated per alpha in
// compute_wannier_polarization); the esolver calls this at SCF convergence so
// the next outer-loop SCF run anchors to this CONVERGED value instead of the
// per-iteration drifting W_prev_ (T4a wrong-branch lock-in, 2026-08-06).
// Also freezes the applied Stage-B branch shift (last_shift_) into
// branch_shift_: the next SCF run then reports avg_raw + branch_shift_
// (continuity readout that follows the physical raw, 2026-08-09) instead of
// the nearest lattice point to the anchor, which pinned the reading to the
// anchor and swallowed the physical response.
void DeltaP::freeze_branch_ref()
{
    ref_gamma_ = W_prev_;
    has_ref_gamma_ = true;
    branch_shift_ = last_shift_;
    has_branch_shift_ = true;
    // T-6' (R4, 2026-08-12): re-anchor the Ô_w θ reference with the
    // continuity anchor — the next SCF run's first measurement re-establishes
    // the per-k θ from the frozen ref_gamma_ (anchor_ow_theta_to_ref) instead
    // of continuing from a possibly stale sheet, and the D2 freeze counter
    // restarts.  The current θ (ow_theta_k_) stays until re-measured.
    ow_theta_prev_k_.clear();
    ow_theta_freeze_count_.clear();
    // T-17 (V-H8, S1): the frozen Ô_w operator kernel must re-anchor with
    // the next measurement — mark it stale so the next compute_gamma_op
    // rebuilds H_ow from the (re-anchored) θ and current wavefunctions
    // instead of continuing with the previous run's frozen operator.
    ow_kernel_stale_ = true;
}

// T-7'' (2026-08-13): freeze only the Stage-B branch shift.  Called by the
// γ-drive inner loop at its entry (the λ=0 natural point), where
// last_shift_ is the shift applied by the entry measurement (≈ 0 at the
// natural reference).  With has_branch_shift_ the Stage-B target becomes
// avg_raw + branch_shift_, so the report follows the raw smoothly instead of
// being pinned to the nearest anchor lattice point — the inner-loop residual
// then responds to λ and can be driven to the target.  Unlike
// freeze_branch_ref() this does NOT touch ref_gamma_ or the Ô_w θ state:
// the reference and the D2 freeze must stay valid mid-SCF.
void DeltaP::freeze_branch_shift()
{
    branch_shift_ = last_shift_;
    has_branch_shift_ = true;
}

// Persist per-atom Wilson-loop products W^I for the next SCF/run.
void DeltaP::save_branch() const
{
    // T-9' write guard: multi-geometry / FD runs disable branch.dat writes so
    // every geometry anchors to the same calibrated reference (silent state
    // overwrites broke A/B comparability twice: 07-30, T4a).
    if (!branch_write_) return;
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
    const int nbands = paraV_->get_wfc_global_nbands();
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
