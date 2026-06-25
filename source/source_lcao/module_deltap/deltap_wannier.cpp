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

    // Step 2: SVD of D_I at each k -> U(k) = W * V^dagger (polar decomposition)
    // D_I is (nproj_total x nocc_use) matrix
    // SVD: D = W * Sigma * V^dagger, then U = W * V^dagger (nproj_total x nocc_use)
    std::vector<std::vector<std::complex<double>>> U_k(nppstr_);

    for (int j = 0; j < nppstr_; ++j)
    {
        int m_dim = nproj_total;
        int n_dim = nocc_use;

        // Build dense D matrix (m_dim x n_dim, column-major)
        std::vector<std::complex<double>> D_mat(m_dim * n_dim, std::complex<double>(0.0, 0.0));

        int row_offset = 0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat))
            {
                row_offset += r;
                continue;
            }
            for (int lm = 0; lm < r; ++lm)
            {
                if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                for (int n = 0; n < n_dim; ++n)
                {
                    if (kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                    D_mat[(row_offset + lm) + n * m_dim] = kstring_data_[j].D_I[iat][lm][n];
                }
            }
            row_offset += r;
        }

        // SVD
        int min_mn = std::min(m_dim, n_dim);
        std::vector<double> S_val(std::max(min_mn, 1));
        std::vector<std::complex<double>> W_svd(m_dim * m_dim);
        std::vector<std::complex<double>> Vt_svd(n_dim * n_dim);
        int lwork = -1;
        std::vector<std::complex<double>> work(1);
        std::vector<double> rwork(5 * std::max(min_mn, 1));
        int info = 0;

        char jobu = 'A';
        char jobvt = 'A';
        int lda_sv = m_dim;
        int ldu = m_dim;
        int ldvt = n_dim;

        // Query
        zgesvd_(&jobu, &jobvt, &m_dim, &n_dim, D_mat.data(), &lda_sv,
                S_val.data(), W_svd.data(), &ldu, Vt_svd.data(), &ldvt,
                work.data(), &lwork, rwork.data(), &info);

        if (info != 0)
        {
            std::cerr << "DeltaP Wannier: zgesvd query failed at k=" << j << " info=" << info << std::endl;
            U_k[j].assign(m_dim * n_dim, std::complex<double>(0.0, 0.0));
            continue;
        }

        lwork = static_cast<int>(work[0].real());
        work.resize(std::max(lwork, 1));

        // Actual SVD
        zgesvd_(&jobu, &jobvt, &m_dim, &n_dim, D_mat.data(), &lda_sv,
                S_val.data(), W_svd.data(), &ldu, Vt_svd.data(), &ldvt,
                work.data(), &lwork, rwork.data(), &info);

        if (info != 0)
        {
            std::cerr << "DeltaP Wannier: zgesvd failed at k=" << j << " info=" << info << std::endl;
            U_k[j].assign(m_dim * n_dim, std::complex<double>(0.0, 0.0));
            continue;
        }

        // Polar decomposition: U = W[:, :n_dim] * V^dagger
        // W is (m_dim x m_dim), V^dagger is (n_dim x n_dim)
        // U_polar = W[:, 0:n_dim] * Vt -> (m_dim x n_dim), column-major
        std::vector<std::complex<double>> U_polar(m_dim * n_dim, std::complex<double>(0.0, 0.0));

        // U_polar[i + j * m_dim] = sum_k W_svd[i + k * m_dim] * Vt_svd[k + j * n_dim]
        // (manual loop avoids dependence on blas_connector.h / GATHER_INFO macro)
        for (int i = 0; i < m_dim; ++i)
            for (int j = 0; j < n_dim; ++j)
                for (int k = 0; k < n_dim; ++k)
                    U_polar[i + j * m_dim] += W_svd[i + k * m_dim] * Vt_svd[k + j * n_dim];

        U_k[j] = U_polar;
    }

    // Step 3: Per-atom Wilson loop using U matrices
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

    // P = -(a_alpha / 2*pi*Omega) * gamma  [result in e/Bohr^2]
    const double prefactor = -a_alpha / (2.0 * ModuleBase::PI * omega);

    results_.P_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat_, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    if (static_cast<int>(W_prev_.size()) != nat_)
    {
        W_prev_.assign(nat_, std::complex<double>(1.0, 0.0));
    }

    for (int iat = 0; iat < nat_; ++iat)
    {
        int r = nproj_per_atom_[iat];
        if (r == 0) continue;

        int row_offset = 0;
        for (int i = 0; i < iat; ++i)
            row_offset += nproj_per_atom_[i];

        std::complex<double> wilson_product(1.0, 0.0);

        for (int j = 0; j < nppstr_ - 1; ++j)
        {
            int m_dim = nproj_total;
            int n_dim = nocc_use;

            // M^I = U^I^dagger(k_j) * O_j * U^I(k_{j+1}) -- (r x r) matrix
            const std::vector<std::complex<double>>& Oj = O_kpair[j];
            std::vector<std::complex<double>> M(r * r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                for (int b = 0; b < r; ++b)
                {
                    std::complex<double> sum(0.0, 0.0);
                    for (int n = 0; n < n_dim; ++n)
                    {
                        int idx_j = (row_offset + a) + n * m_dim;
                        if (idx_j >= static_cast<int>(U_k[j].size())) continue;
                        const std::complex<double> uj = std::conj(U_k[j][idx_j]);
                        for (int m = 0; m < n_dim; ++m)
                        {
                            int idx_jp1 = (row_offset + b) + m * m_dim;
                            if (idx_jp1 >= static_cast<int>(U_k[j + 1].size())) continue;
                            sum += uj * Oj[n + m * n_dim] * U_k[j + 1][idx_jp1];
                        }
                    }
                    M[a + b * r] = sum;
                }
            }

            // det(M) via LU
            std::vector<std::complex<double>> M_copy = M;
            std::vector<int> ipiv(std::max(r, 1));
            int info_lu = 0;
            int r_int = r;
            zgetrf_(&r_int, &r_int, M_copy.data(), &r_int, ipiv.data(), &info_lu);

            if (info_lu != 0) continue;

            std::complex<double> det_m(1.0, 0.0);
            int sign = 1;
            for (int i = 0; i < r; ++i)
            {
                det_m *= M_copy[i + i * r];
                if (ipiv[i] != i + 1) sign = -sign;
            }
            if (sign < 0) det_m = -det_m;

            wilson_product *= det_m;
        }

        double gamma = std::arg(wilson_product);

        // branch tracking: unwrap gamma so it stays smooth across SCF steps
        if (has_prev_ && iat < static_cast<int>(W_prev_.size()))
        {
            double prev_gamma = std::arg(W_prev_[iat]);
            double diff = gamma - prev_gamma;
            while (diff > M_PI) { gamma -= 2.0 * M_PI; diff -= 2.0 * M_PI; }
            while (diff < -M_PI) { gamma += 2.0 * M_PI; diff += 2.0 * M_PI; }
        }
        W_prev_[iat] = wilson_product;
        has_prev_ = true;

        results_.gamma_I[iat][alpha_idx] = gamma;
        results_.P_I[iat][alpha_idx] = prefactor * gamma;
    }

    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat_; ++iat)
        results_.P_total += results_.P_I[iat];

    verify_sum_rule();
    write_results(ucell);

    std::cout << " >> Finish DeltaP Wannier polarization.\n * * * * * *\n";

    ModuleBase::timer::end("DeltaP", "compute_wannier_polarization");
}

} // namespace deltap
