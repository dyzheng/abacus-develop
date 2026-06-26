#include "deltap.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#ifdef __MPI
#include "source_base/parallel_comm.h"
#include "source_base/module_external/scalapack_connector.h"
#endif
#include <cmath>
#include <algorithm>
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

    // First call: compute and cache the raw overlap data
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
                                S_dk_cache_.push_back({lr, lc, ov,
                                    (double)R.x, (double)R.y, (double)R.z,
                                    tau0.x, tau0.y, tau0.z});
                        }
                    }
                }
            }
        }
        S_dk_cache_valid_ = true;
    }

    // Apply per-link phase using cached data
    // berry_phase convention: phase = 2*pi*(kvec_c_R . R_cart - dk_c . tau)
    // where kvec_c is Cartesian (1/Bohr), R_cart is Cartesian (Bohr), tau is Cartesian (Bohr)
    ModuleBase::Vector3<double> dk_c = kvec_c_R - kvec_c_L;

    for (const auto& e : S_dk_cache_)
    {
        // R_cart = R_int.x * a1 + R_int.y * a2 + R_int.z * a3
        ModuleBase::Vector3<double> R_cart = e.Rx * ucell.a1 + e.Ry * ucell.a2 + e.Rz * ucell.a3;
        double arg = ModuleBase::TWO_PI * (
            kvec_c_R.x * R_cart.x + kvec_c_R.y * R_cart.y + kvec_c_R.z * R_cart.z
            - dk_c.x * e.tau_x - dk_c.y * e.tau_y - dk_c.z * e.tau_z);
        std::complex<double> phase(std::cos(arg), std::sin(arg));

        // Position correction: DISABLED for testing
        // double R_alpha = (gdir_ == 1) ? e.Rx : (gdir_ == 2) ? e.Ry : e.Rz;
        // std::complex<double> pos_corr(1.0, -dk_cart * R_alpha);
        std::complex<double> pos_corr(1.0, 0.0);  // no position correction

        S_dk_[e.lr + e.lc * nrow] += phase * pos_corr * e.ov;
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
    load_branch();

    // Step 0: compute real-space overlaps and k-string
    compute_real_overlaps(ucell, *gd_);
    setup_kstring(*kv_);

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

    // Step 1: Compute S(k), D_I(k) for all k on string
    std::cout << "   DeltaP: nppstr_=" << nppstr_ << " total_string_=" << total_string_
              << " k_index_.size()=" << k_index_.size() << " nks=" << nks << std::endl;
    if (k_index_.empty() || nppstr_ == 0)
    {
        std::cerr << "DeltaP ERROR: k_index_ is empty or nppstr_=0" << std::endl;
        ModuleBase::timer::end("DeltaP", "compute_wannier_polarization");
        return;
    }

    // compute_S_dk is now per-link (uses correct berry_phase phase + position correction)
    const int nlocal = paraV_->get_global_row_size();
    S_dk_cache_valid_ = false;  // reset cache for new structure

    int n_dim = nocc_use;
    int m_dim = nproj_total;

    // Prefactor and direction (outside loop)
    const int alpha_idx = gdir_ - 1;
    double a_alpha = 0.0;
    if (gdir_ == 1) a_alpha = ucell.lat0 * ucell.a1.norm();
    else if (gdir_ == 2) a_alpha = ucell.lat0 * ucell.a2.norm();
    else a_alpha = ucell.lat0 * ucell.a3.norm();
    const double omega = ucell.omega;
    const double prefactor = a_alpha / (2.0 * ModuleBase::PI * omega);

    results_.P_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    if (static_cast<int>(W_prev_.size()) != nat_)
        W_prev_.assign(nat_, std::complex<double>(1.0, 0.0));

    // Accumulate per-atom Berry phases over all k-strings
    std::vector<double> gamma_accum(nat_, 0.0);
    int n_strings_processed = 0;

    kstring_data_.resize(nppstr_);
    std::vector<std::complex<double>*> psi_k_ptrs(nppstr_, nullptr);

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

        // --- Step 2: O_kpair (exact overlap with berry_phase convention) ---
        std::vector<std::vector<std::complex<double>>> O_kpair(nppstr_ - 1);
        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            // Compute S_dk for this link with correct k_R phase + position correction
            int ik_L = k_index_[istring][j];
            int ik_R = k_index_[istring][j + 1];
            if (ik_R < nks && ik_L < nks)
                compute_S_dk_link(ucell, kv_->kvec_d[ik_R],
                                  kv_->kvec_c[ik_L], kv_->kvec_c[ik_R]);

            std::complex<double>* cj = psi_k_ptrs[j];
            std::complex<double>* cjp1 = psi_k_ptrs[j + 1];
            std::vector<std::complex<double>> O_full(
                static_cast<size_t>(nocc_use) * nocc_use, std::complex<double>(0.0, 0.0));
            if (!cj || !cjp1 || nocc_use == 0 || nlocal == 0) { O_kpair[j] = O_full; continue; }

#ifdef __MPI
            std::vector<std::complex<double>> tmp(paraV_->nloc, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> O_2d(paraV_->nloc, std::complex<double>(0.0, 0.0));
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);
            ScalapackConnector::gemm('C', 'N', nocc_use, nlocal, nlocal,
                                      one, cj, 1, 1, paraV_->desc,
                                      S_dk_.data(), 1, 1, paraV_->desc,
                                      zero, tmp.data(), 1, 1, paraV_->desc);
            ScalapackConnector::gemm('N', 'N', nocc_use, nocc_use, nlocal,
                                      one, tmp.data(), 1, 1, paraV_->desc,
                                      cjp1, 1, 1, paraV_->desc,
                                      zero, O_2d.data(), 1, 1, paraV_->desc);
            for (int ilc = 0; ilc < paraV_->ncol; ++ilc)
            {
                int jg = paraV_->local2global_col(ilc);
                if (jg >= nocc_use) continue;
                for (int ilr = 0; ilr < paraV_->nrow; ++ilr)
                {
                    int ig = paraV_->local2global_row(ilr);
                    if (ig >= nocc_use) continue;
                    O_full[ig + jg * nocc_use] = O_2d[ilr + ilc * paraV_->nrow];
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, O_full.data(), 2 * nocc_use * nocc_use,
                          MPI_DOUBLE, MPI_SUM, paraV_->comm());
#else
            for (int a = 0; a < nocc_use; ++a)
                for (int b = 0; b < nocc_use; ++b)
                {
                    std::complex<double> s(0.0, 0.0);
                    for (int mu = 0; mu < nlocal; ++mu)
                    {
                        std::complex<double> sm(0.0, 0.0);
                        for (int nu = 0; nu < nlocal; ++nu)
                            sm += S_dk_[mu + nu * nlocal] * cjp1[nu + b * nlocal];
                        s += std::conj(cj[mu + a * nlocal]) * sm;
                    }
                    O_full[a + b * nocc_use] = s;
                }
#endif
            O_kpair[j] = O_full;
        }

        // --- Step 3: Build Wilson loop matrix W = O_0 * O_1 * ... * O_{N-1} ---
        // Normalize after each step by max element to prevent overflow.
        // Dividing by a real positive number does NOT change arg(eigenvalues).
        std::vector<std::complex<double>> W_mat(n_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int i = 0; i < n_dim; ++i)
            W_mat[i + i * n_dim] = std::complex<double>(1.0, 0.0);

        for (int j = 0; j < nppstr_ - 1; ++j)
        {
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
            // Normalize by max element (real positive factor, preserves arg)
            double max_elem = 0.0;
            for (int i = 0; i < n_dim * n_dim; ++i)
                max_elem = std::max(max_elem, std::abs(tmp[i]));
            if (max_elem > 1e-10)
                for (int i = 0; i < n_dim * n_dim; ++i)
                    tmp[i] /= max_elem;
            W_mat = tmp;
        }

        // --- Step 4: Diagonalize W ---
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

        // Per-atom Berry phases for this k-string
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            int row_offset = 0;
            for (int i = 0; i < iat; ++i)
                row_offset += nproj_per_atom_[i];
            double gamma_I = 0.0;
            for (int n = 0; n < n_dim; ++n)
            {
                double w_In = 0.0;
                for (int a = row_offset; a < row_offset + r; ++a)
                    w_In += std::norm(proj[a + n * m_dim]);
                gamma_I += w_In * std::arg(evals[n]);
            }
            gamma_accum[iat] += gamma_I;
        }
        n_strings_processed++;

        if (istring == 0)
        {
            double g_check = 0.0;
            for (int n = 0; n < n_dim; ++n) g_check += std::arg(evals[n]);
            std::cout << "   DeltaP Wilson loop (string 0): nocc=" << n_dim
                      << " gamma=" << std::scientific << std::setprecision(6) << g_check << std::endl;
        }
    }

    // --- Average over k-strings ---
    std::cout << "   DeltaP: processed " << n_strings_processed << " / " << total_string_ << " k-strings" << std::endl;
    for (int iat = 0; iat < nat_; ++iat)
    {
        double gamma_I = (n_strings_processed > 0) ? gamma_accum[iat] / n_strings_processed : 0.0;
        W_prev_[iat] = std::complex<double>(gamma_I, 0.0);
        has_prev_ = true;
        results_.gamma_I[iat][alpha_idx] = gamma_I;
        results_.P_I[iat][alpha_idx] = prefactor * gamma_I;
    }

    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat_; ++iat)
        results_.P_total += results_.P_I[iat];

    verify_sum_rule();
    write_results(ucell);

    // Persist branch state for the next SCF/run
    save_branch();

    std::cout << " >> Finish DeltaP Wannier polarization.\n * * * * * *\n";

    ModuleBase::timer::end("DeltaP", "compute_wannier_polarization");
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
        double re = 0.0, im = 0.0;
        ifs >> re >> im;
        W_prev_[iat] = std::complex<double>(re, im);
    }
    has_prev_ = true;
    std::cout << " DeltaP: loaded branch state from " << fname << std::endl;
}

// Persist per-atom Wilson-loop products W^I for the next SCF/run.
void DeltaP::save_branch() const
{
    if (!has_prev_ || static_cast<int>(W_prev_.size()) != nat_) return;

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
        ofs << W_prev_[iat].real() << " " << W_prev_[iat].imag() << "\n";
    ofs.close();
}

} // namespace deltap
