#include "deltap.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#ifdef __MPI
#include "source_base/parallel_comm.h"
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
    for (int j = 0; j < nppstr_; ++j)
    {
        int ik_psi = k_index_[0][j];
        if (ik_psi >= nks) continue;
        kstring_data_[j].kvec_d = kv_->kvec_d[ik_psi];
        psi->fix_k(ik_psi);
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
    // M^I(k_j, k_{j+1}) = U^I^dagger(k_j) * U^I(k_{j+1})  [approximate: assumes <psi_kj|psi_kj+1> ~ I]
    // This is gauge-invariant (SVD eliminates phases) and exact in dk->0 limit

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

            // M^I = U^I^dagger(k_j) * U^I(k_{j+1}) -- (r x r) matrix
            std::vector<std::complex<double>> M(r * r, std::complex<double>(0.0, 0.0));
            for (int a = 0; a < r; ++a)
            {
                for (int b = 0; b < r; ++b)
                {
                    std::complex<double> sum(0.0, 0.0);
                    for (int n = 0; n < n_dim; ++n)
                    {
                        // U_polar is column-major: U[row + col * m_dim]
                        int idx_j = (row_offset + a) + n * m_dim;
                        int idx_jp1 = (row_offset + b) + n * m_dim;
                        if (idx_j < static_cast<int>(U_k[j].size()) &&
                            idx_jp1 < static_cast<int>(U_k[j + 1].size()))
                        {
                            sum += std::conj(U_k[j][idx_j]) * U_k[j + 1][idx_jp1];
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
