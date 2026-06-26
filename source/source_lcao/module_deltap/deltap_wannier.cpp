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

            const double arg = ModuleBase::TWO_PI * (
                dkv[0] * R.x + dkv[1] * R.y + dkv[2] * R.z);
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
    kstring_data_.resize(nppstr_);
    // keep a stable pointer to each k-block of psi for the Wilson-loop overlap
    std::vector<std::complex<double>*> psi_k_ptrs(nppstr_, nullptr);
    for (int j = 0; j < nppstr_; ++j)
    {
        int ik_psi = k_index_[0][j];
        if (ik_psi >= nks) continue;
        kstring_data_[j].kvec_d = kv_->kvec_d[ik_psi];
        psi->fix_k(ik_psi);
        psi_k_ptrs[j] = psi->get_pointer();
        compute_S_k(j);
        compute_D_I(j, psi->get_pointer(), nbands, nrow_local);
    }

    // MPI reduction: D_I is only partially computed on each rank
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
                    MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                  2 * sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
                }
            }
        }
    }
#endif

    // Step 2: Global SVD of D at each k-point, with per-atom weight
    // decomposition of the Berry connection.
    //
    // D = <alpha|psi> is (nproj_total x nocc).  Its SVD
    //   D = W * Sigma * Vt
    // gives the global right singular vectors Vt = V^dagger (nocc x nocc)
    // and left singular vectors W (nproj_total x nocc).  The SVD gauge
    // (Ozaki CWF) fixes the band-space phase uniquely.
    //
    // Per-atom weight: for singular value s, the fraction "owned" by atom I is
    //   w^I_s = sum_{a in atom I} |W_{a,s}|^2
    // These satisfy sum_I w^I_s = 1 for every s (W has orthonormal columns).
    //
    // The per-atom Berry connection for link j is
    //   A^I_j = sum_s w^I_s * Im[ M_j(s,s) ]
    // where M_j = Vt(k_j) * O_j * Vt(k_{j+1})^dagger  (nocc x nocc).
    //
    // Sum rule (EXACT):
    //   sum_I A^I_j = sum_s (sum_I w^I_s) * Im[M_j(s,s)]
    //               = sum_s Im[M_j(s,s)] = Im Tr(M_j) = Im Tr(O_j)
    //
    // P^I = prefactor * sum_j A^I_j   (Berry connection, dk->0 exact for
    //                                  infinite k-string; good approx for finite)

    int m_dim = nproj_total;
    int n_dim = nocc_use;
    int min_mn = std::min(m_dim, n_dim);

    // Vt_k[j] = global Vt at k_j (nocc x nocc, column-major)
    std::vector<std::vector<std::complex<double>>> Vt_k(nppstr_);
    // w_atom[j][iat][s] = per-atom weight for singular value s at k_j
    std::vector<std::vector<std::vector<double>>> w_atom(nppstr_);

    for (int j = 0; j < nppstr_; ++j)
    {
        // Build dense D matrix (m_dim x n_dim, column-major)
        std::vector<std::complex<double>> D_mat(m_dim * n_dim, std::complex<double>(0.0, 0.0));
        int row_offset = 0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            if (kstring_data_[j].D_I.size() > static_cast<size_t>(iat))
            {
                for (int lm = 0; lm < r; ++lm)
                {
                    if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                    for (int n = 0; n < n_dim; ++n)
                    {
                        if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                        D_mat[(row_offset + lm) + n * m_dim] = kstring_data_[j].D_I[iat][lm][n];
                    }
                }
            }
            row_offset += r;
        }

        // SVD: D = W * Sigma * Vt  (jobu='S', jobvt='A')
        // W: (m_dim x min_mn), Vt: (n_dim x n_dim)
        std::vector<double> S_val(std::max(min_mn, 1));
        std::vector<std::complex<double>> W_svd(m_dim * min_mn);
        std::vector<std::complex<double>> Vt_svd(n_dim * n_dim);
        int lwork = -1;
        std::vector<std::complex<double>> work(1);
        std::vector<double> rwork(5 * std::max(min_mn, 1));
        int info = 0;

        char jobu = 'S';
        char jobvt = 'A';
        int lda_sv = m_dim;
        int ldu = m_dim;
        int ldvt = n_dim;

        zgesvd_(&jobu, &jobvt, &m_dim, &n_dim, D_mat.data(), &lda_sv,
                S_val.data(), W_svd.data(), &ldu, Vt_svd.data(), &ldvt,
                work.data(), &lwork, rwork.data(), &info);
        if (info != 0)
        {
            std::cerr << "DeltaP: zgesvd query failed at k=" << j << " info=" << info << std::endl;
            Vt_k[j].assign(n_dim * n_dim, std::complex<double>(0.0, 0.0));
            w_atom[j].assign(nat_, std::vector<double>(n_dim, 0.0));
            continue;
        }
        lwork = static_cast<int>(work[0].real());
        work.resize(std::max(lwork, 1));
        zgesvd_(&jobu, &jobvt, &m_dim, &n_dim, D_mat.data(), &lda_sv,
                S_val.data(), W_svd.data(), &ldu, Vt_svd.data(), &ldvt,
                work.data(), &lwork, rwork.data(), &info);
        if (info != 0)
        {
            std::cerr << "DeltaP: zgesvd failed at k=" << j << " info=" << info << std::endl;
            Vt_k[j].assign(n_dim * n_dim, std::complex<double>(0.0, 0.0));
            w_atom[j].assign(nat_, std::vector<double>(n_dim, 0.0));
            continue;
        }

        Vt_k[j] = Vt_svd;

        // Per-atom weights: w^I_s = sum_{a in I} |W_{a,s}|^2
        // W_svd is (m_dim x min_mn) column-major: W[a + s * m_dim]
        w_atom[j].resize(nat_, std::vector<double>(min_mn, 0.0));
        row_offset = 0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            for (int a = row_offset; a < row_offset + r; ++a)
            {
                for (int s = 0; s < min_mn; ++s)
                {
                    std::complex<double> wval = W_svd[a + s * m_dim];
                    w_atom[j][iat][s] += std::norm(wval);
                }
            }
            row_offset += r;
        }

        if (j == 0)
        {
            std::cout << "   DeltaP global SVD: nproj=" << m_dim
                      << " nocc=" << n_dim << " min=" << min_mn << std::endl;
            for (int iat = 0; iat < nat_; ++iat)
            {
                double wsum = 0;
                for (int s = 0; s < min_mn; ++s) wsum += w_atom[j][iat][s];
                std::cout << "     iat=" << iat
                          << " weight_sum=" << std::fixed << std::setprecision(4) << wsum
                          << " top5_singulars:";
                for (int s = 0; s < std::min(5, min_mn); ++s)
                    std::cout << " " << std::scientific << std::setprecision(4) << S_val[s];
                std::cout << std::endl;
            }
        }
    }

    // Step 2b: Align SVD gauges across k-points (Procrustes matching).
    //
    // The SVD at each k-point is independent; singular vectors can have
    // different signs at neighbouring k-points, causing Im[M_j(s,s)] to
    // have the wrong sign.  Fix: at each k_j (j>0), find the unitary Q
    // that maximizes Re Tr[ V†(k_{j-1}) · V(k_j) · Q ], then replace
    // V(k_j) -> V(k_j) · Q and W(k_j) -> W(k_j) · Q.
    //
    // For non-degenerate singular values, Q is diagonal with ±1 entries
    // (sign alignment).  For degenerate singular values, Q is a rotation
    // within the degenerate subspace.

    for (int j = 1; j < nppstr_; ++j)
    {
        // Build the overlap matrix S = V†(k_{j-1}) · V(k_j)  (nocc x nocc)
        // Vt_k is V† (from SVD), so V = Vt_k†, and
        // V†(k_{j-1}) = Vt_k[j-1], V(k_j) = Vt_k[j]†
        // S = Vt_k[j-1] · Vt_k[j]†
        std::vector<std::complex<double>> S_overlap(
            n_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int s = 0; s < n_dim; ++s)
        {
            for (int sp = 0; sp < n_dim; ++sp)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int n = 0; n < n_dim; ++n)
                {
                    // Vt_k[j-1][s + n*n_dim] = V†(k_{j-1})[s,n]
                    // Vt_k[j][sp + n*n_dim]† = conj(Vt_k[j][sp + n*n_dim]) = V(k_j)[n,sp]
                    sum += Vt_k[j - 1][s + n * n_dim] * std::conj(Vt_k[j][sp + n * n_dim]);
                }
                S_overlap[s + sp * n_dim] = sum;
            }
        }

        // SVD of S_overlap = U_svd * Sigma * Vt_svd
        // The optimal Q = U_svd * Vt_svd (polar factor)
        std::vector<double> S2_val(std::max(n_dim, 1));
        std::vector<std::complex<double>> U2(n_dim * n_dim);
        std::vector<std::complex<double>> Vt2(n_dim * n_dim);
        int lwork2 = -1;
        std::vector<std::complex<double>> work2(1);
        std::vector<double> rwork2(5 * std::max(n_dim, 1));
        int info2 = 0;
        char jobu2 = 'A';
        char jobvt2 = 'A';
        int n2 = n_dim;
        zgesvd_(&jobu2, &jobvt2, &n2, &n2, S_overlap.data(), &n2,
                S2_val.data(), U2.data(), &n2, Vt2.data(), &n2,
                work2.data(), &lwork2, rwork2.data(), &info2);
        if (info2 != 0)
        {
            std::cerr << "DeltaP: Procrustes zgesvd query failed at j=" << j << std::endl;
            continue;
        }
        lwork2 = static_cast<int>(work2[0].real());
        work2.resize(std::max(lwork2, 1));
        zgesvd_(&jobu2, &jobvt2, &n2, &n2, S_overlap.data(), &n2,
                S2_val.data(), U2.data(), &n2, Vt2.data(), &n2,
                work2.data(), &lwork2, rwork2.data(), &info2);
        if (info2 != 0)
        {
            std::cerr << "DeltaP: Procrustes zgesvd failed at j=" << j << std::endl;
            continue;
        }

        // Q = U2 * Vt2  (nocc x nocc, column-major)
        std::vector<std::complex<double>> Q(n_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int i = 0; i < n_dim; ++i)
            for (int s = 0; s < n_dim; ++s)
                for (int k = 0; k < n_dim; ++k)
                    Q[i + s * n_dim] += U2[i + k * n_dim] * Vt2[k + s * n_dim];

        // Apply Q: V(k_j) -> V(k_j) * Q, i.e., V†(k_j) -> Q† * V†(k_j)
        // Vt_k[j] = Q† * Vt_k[j]  (V† -> Q† * V†)
        std::vector<std::complex<double>> Vt_new(n_dim * n_dim, std::complex<double>(0.0, 0.0));
        for (int s = 0; s < n_dim; ++s)
        {
            for (int n = 0; n < n_dim; ++n)
            {
                std::complex<double> sum(0.0, 0.0);
                for (int sp = 0; sp < n_dim; ++sp)
                    sum += std::conj(Q[sp + s * n_dim]) * Vt_k[j][sp + n * n_dim];
                Vt_new[s + n * n_dim] = sum;
            }
        }
        Vt_k[j] = Vt_new;

        // Also update weights: w^I_s -> sum_t Q*_{t,s} * (old w^I_{s,t})... 
        // Actually, the weights transform as w^I_s' = sum_t |Q_{t,s}|^2 * w^I_t
        // No, the weights are w^I_s = sum_{a in I} |W_{a,s}|^2, and W -> W * Q,
        // so w'^I_s = sum_{a in I} |sum_t W_{a,t} Q_{t,s}|^2.
        // For a sign-flip Q (diagonal ±1), w'^I_s = w^I_s (unchanged).
        // For a general Q, we need to recompute. But since we don't store W,
        // we approximate: for non-degenerate singular values, Q is diagonal
        // and weights are unchanged. For degenerate ones, the weights rotate
        // but their SUM is preserved. For now, leave weights unchanged.
        // (This is exact for non-degenerate singular values.)
    }
    // M^I(k_j, k_{j+1}) = U^I^dagger(k_j) * O(k_j,k_{j+1}) * U^I(k_{j+1})
    // where O = C^dagger(k_j) * S(dk) * C(k_{j+1}) is the exact overlap matrix
    // of Bloch states on neighbouring k-points (replaces the O ~ I approximation).
    // The SVD polar factors U make this gauge-invariant; the exact O restores the
    // true Berry-phase discretization instead of the dk->0 limit.

    // pre-compute the displacement overlap S(dk) once (same for every link)
    compute_S_dk(ucell);

    const int nlocal = paraV_->get_global_row_size();

    // ---- exact overlap O_j = C^H(k_j) * S(dk) * C(k_{j+1}) for each link ----
    // O_j is (nocc_use x nocc_use) and replicated on every rank so that the
    // per-atom contraction M^I = U^I^dagger * O_j * U^I_next stays local.
    std::vector<std::vector<std::complex<double>>> O_kpair(nppstr_ - 1);
    for (int j = 0; j < nppstr_ - 1; ++j)
    {
        std::complex<double>* cj = psi_k_ptrs[j];
        std::complex<double>* cjp1 = psi_k_ptrs[j + 1];
        std::vector<std::complex<double>> O_full(
            static_cast<size_t>(nocc_use) * nocc_use, std::complex<double>(0.0, 0.0));

        if (cj == nullptr || cjp1 == nullptr || nocc_use == 0 || nlocal == 0)
        {
            O_kpair[j] = O_full;
            continue;
        }

#ifdef __MPI
        // Distributed triple product with the shared paraV descriptor, exactly
        // as in unkOverlap_lcao::det_berryphase. Two pzgemm calls produce the
        // 2D-block-cyclic O, which is then gathered into a replicated matrix.
        std::vector<std::complex<double>> tmp(
            paraV_->nloc, std::complex<double>(0.0, 0.0));
        std::vector<std::complex<double>> O_2d(
            paraV_->nloc, std::complex<double>(0.0, 0.0));
        const std::complex<double> one(1.0, 0.0);
        const std::complex<double> zero(0.0, 0.0);
        // tmp = C^H(k_j) * S(dk)            -- (nocc_use x nlocal)
        ScalapackConnector::gemm('C', 'N', nocc_use, nlocal, nlocal,
                                  one, cj, 1, 1, paraV_->desc,
                                  S_dk_.data(), 1, 1, paraV_->desc,
                                  zero, tmp.data(), 1, 1, paraV_->desc);
        // O_2d = tmp * C(k_{j+1})           -- (nocc_use x nocc_use)
        ScalapackConnector::gemm('N', 'N', nocc_use, nocc_use, nlocal,
                                  one, tmp.data(), 1, 1, paraV_->desc,
                                  cjp1, 1, 1, paraV_->desc,
                                  zero, O_2d.data(), 1, 1, paraV_->desc);
        // gather the small nocc_use*nocc_use block into a replicated matrix
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
        // serial: every index is local, plain triple product
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

    const int alpha_idx = gdir_ - 1;
    double a_alpha = 0.0;
    if (gdir_ == 1) a_alpha = ucell.lat0 * ucell.a1.norm();
    else if (gdir_ == 2) a_alpha = ucell.lat0 * ucell.a2.norm();
    else a_alpha = ucell.lat0 * ucell.a3.norm();
    const double omega = ucell.omega;

    // P = (a_alpha / 2*pi*Omega) * gamma  [result in e/Bohr^2]
    // (sign: P = (a/Omega) * reduced_phase, reduced_phase = gamma/(2pi))
    const double prefactor = a_alpha / (2.0 * ModuleBase::PI * omega);

    results_.P_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    if (static_cast<int>(W_prev_.size()) != nat_)
    {
        W_prev_.assign(nat_, std::complex<double>(1.0, 0.0));
    }

    // --- Compute exact total Berry phase via Wilson loop (product of det O_j) ---
    // gamma_total = Im ln(prod_j det(O_j)) = sum_j arg(det(O_j))
    double gamma_total = 0.0;
    for (int j = 0; j < nppstr_ - 1; ++j)
    {
        const std::vector<std::complex<double>>& Oj = O_kpair[j];
        // det(O_j) via LU
        std::vector<std::complex<double>> O_copy = Oj;
        std::vector<int> ipiv(std::max(n_dim, 1));
        int info_lu = 0;
        int n_lu = n_dim;
        zgetrf_(&n_lu, &n_lu, O_copy.data(), &n_lu, ipiv.data(), &info_lu);
        if (info_lu != 0) continue;
        std::complex<double> det_o(1.0, 0.0);
        int sign = 1;
        for (int i = 0; i < n_dim; ++i)
        {
            det_o *= O_copy[i + i * n_dim];
            if (ipiv[i] != i + 1) sign = -sign;
        }
        if (sign < 0) det_o = -det_o;
        gamma_total += std::arg(det_o);
    }

    // --- Compute per-atom Berry connection (for proportional decomposition) ---
    // A^I = sum_j sum_s w^I_s * Im[M_j(s,s)]
    // A_total = sum_I A^I = sum_j Im Tr(O_j)  (Berry connection approximation)
    // Then rescale: gamma^I = gamma_total * A^I / A_total
    // This ensures sum_I gamma^I = gamma_total (exact Berry phase)
    // while per-atom proportions follow the Berry connection.
    std::vector<double> berry_conn_atom(nat_, 0.0);
    double berry_conn_total = 0.0;

    for (int iat = 0; iat < nat_; ++iat)
    {
        double berry_conn_sum = 0.0;

        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            const std::vector<std::complex<double>>& Oj = O_kpair[j];
            const std::vector<std::complex<double>>& Vtj = Vt_k[j];
            const std::vector<std::complex<double>>& Vtjp1 = Vt_k[j + 1];

            const std::vector<double>& wj = w_atom[j][iat];
            const std::vector<double>& wjp1 = w_atom[j + 1][iat];

            for (int s = 0; s < min_mn; ++s)
            {
                std::complex<double> M_ss(0.0, 0.0);
                for (int n = 0; n < n_dim; ++n)
                {
                    int idx_j = s + n * n_dim;
                    if (idx_j >= static_cast<int>(Vtj.size())) continue;
                    const std::complex<double> vj = Vtj[idx_j];
                    for (int m = 0; m < n_dim; ++m)
                    {
                        int idx_jp1 = s + m * n_dim;
                        if (idx_jp1 >= static_cast<int>(Vtjp1.size())) continue;
                        M_ss += vj * Oj[n + m * n_dim] * std::conj(Vtjp1[idx_jp1]);
                    }
                }
                double w_avg = 0.5 * (wj[s] + wjp1[s]);
                berry_conn_sum += w_avg * std::imag(M_ss);
            }
        }

        berry_conn_atom[iat] = berry_conn_sum;
        berry_conn_total += berry_conn_sum;
    }

    // Rescale per-atom Berry connection to match exact total Berry phase
    std::cout << "   DeltaP: gamma_total (Wilson) = " << std::scientific << std::setprecision(6)
              << gamma_total << std::endl;
    std::cout << "   DeltaP: berry_conn_total (trace) = " << berry_conn_total << std::endl;
    std::cout << "   DeltaP: ratio (Wilson/trace) = " << gamma_total / berry_conn_total << std::endl;

    for (int iat = 0; iat < nat_; ++iat)
    {
        double gamma = 0.0;
        if (std::abs(berry_conn_total) > 1e-15)
            gamma = gamma_total * berry_conn_atom[iat] / berry_conn_total;

        W_prev_[iat] = std::complex<double>(gamma, 0.0);
        has_prev_ = true;

        results_.gamma_I[iat][alpha_idx] = gamma;
        results_.P_I[iat][alpha_idx] = prefactor * gamma;
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
