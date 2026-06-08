#include "hsolver_lcao_subspace.h"

#include "diago_iter_assist.h"
#include "diago_lapack.h"

#ifdef __MPI
#include "diago_scalapack.h"
#include "source_base/module_external/scalapack_connector.h"
#endif

#ifdef __ELPA
#include "diago_elpa.h"
#include "diago_elpa_native.h"
#endif

#ifdef __CUDA
#include "diago_cusolver.h"
#endif

#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_base/global_variable.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_container/base/third_party/lapack.h"
#include "source_base/module_external/blas_connector.h"
#include "source_estate/elecstate_tools.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_io/module_parameter/parameter.h"

#ifdef __LCAO
#include "source_lcao/module_operator_lcao/dspin_lcao.h"
#include "source_lcao/rho_tau_lcao.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <complex>
#include <iostream>

namespace hsolver
{

// ===================================================================
// LCAOSubspaceCache implementation
// ===================================================================

LCAOSubspaceCache::LCAOSubspaceCache() = default;

void LCAOSubspaceCache::build(
    int nk,
    int nbands,
    int nat,
    const std::complex<double>* H0_sub_raw,
    const std::complex<double>* S_sub_raw,
    std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_all,
    const std::vector<double>& ekb_ref_all,
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref)
{
    nk_ = nk;
    nbands_ = nbands;
    nat_ = nat;

    const int nn = nbands * nbands;
    H0_sub_.assign(H0_sub_raw, H0_sub_raw + nk * nn);
    S_sub_.assign(S_sub_raw, S_sub_raw + nk * nn);
    P_I_sub_ = std::move(P_I_sub_all);
    ekb_ref_ = ekb_ref_all;
    lambda_ref_ = lambda_ref;
    valid_ = true;
}

void LCAOSubspaceCache::clear()
{
    nk_ = 0;
    nbands_ = 0;
    nat_ = 0;
    valid_ = false;
    H0_sub_.clear();
    S_sub_.clear();
    P_I_sub_.clear();
    ekb_ref_.clear();
    lambda_ref_.clear();
}

const std::complex<double>* LCAOSubspaceCache::H0_sub(int ik) const
{
    if (!valid_ || ik < 0 || ik >= nk_) return nullptr;
    return H0_sub_.data() + ik * nbands_ * nbands_;
}

const std::complex<double>* LCAOSubspaceCache::S_sub(int ik) const
{
    if (!valid_ || ik < 0 || ik >= nk_) return nullptr;
    return S_sub_.data() + ik * nbands_ * nbands_;
}

const std::complex<double>* LCAOSubspaceCache::P_I_sub(int ik, int iat) const
{
    if (!valid_ || ik < 0 || ik >= nk_ || iat < 0 || iat >= nat_) return nullptr;
    if (P_I_sub_[ik][iat].empty()) return nullptr;
    return P_I_sub_[ik][iat].data();
}

const double* LCAOSubspaceCache::ekb_ref(int ik) const
{
    if (!valid_ || ik < 0 || ik >= nk_) return nullptr;
    return ekb_ref_.data() + ik * nbands_;
}

const std::vector<ModuleBase::Vector3<double>>& LCAOSubspaceCache::lambda_ref() const
{
    return lambda_ref_;
}

// ===================================================================
// HSolverLCAOSubspace implementation
// ===================================================================

HSolverLCAOSubspace::HSolverLCAOSubspace(const Parallel_Orbitals* ParaV_in,
                                          std::string method_in,
                                          SubspaceMode mode_in,
                                          SubspacePrecision precision_in)
    : ParaV_(ParaV_in), method_(method_in), mode_(mode_in), precision_(precision_in)
{
}

void HSolverLCAOSubspace::solve(hamilt::Hamilt<std::complex<double>>* pHamilt,
                                 psi::Psi<std::complex<double>>& psi,
                                 elecstate::ElecState* pes,
                                 elecstate::DensityMatrix<std::complex<double>, double>& dm,
                                 Charge& chr,
                                 const int nspin,
                                 const bool skip_charge,
                                 const std::vector<ModuleBase::Vector3<double>>* lambda)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "solve");
    ModuleBase::timer::start("HSolverLCAOSubspace", "solve");

    if (!cache_.is_valid() || mode_ == SubspaceMode::FullSpace)
    {
        solve_fullspace(pHamilt, psi, pes, dm, chr, nspin, skip_charge);
        ModuleBase::timer::end("HSolverLCAOSubspace", "solve");
        return;
    }

    SubspaceSolverResult result;
    if (mode_ == SubspaceMode::Subspace)
    {
        result = solve_subspace(pHamilt, psi, pes, dm, nspin, skip_charge, lambda);
    }
    else if (mode_ == SubspaceMode::FirstOrder)
    {
        result = solve_first_order(pHamilt, psi, pes, dm, nspin, skip_charge, lambda);
    }

    if (!result.success)
    {
        solve_fullspace(pHamilt, psi, pes, dm, chr, nspin, skip_charge);
    }

    ModuleBase::timer::end("HSolverLCAOSubspace", "solve");
}

void HSolverLCAOSubspace::solve_fullspace(hamilt::Hamilt<std::complex<double>>* pHamilt,
                                           psi::Psi<std::complex<double>>& psi,
                                           elecstate::ElecState* pes,
                                           elecstate::DensityMatrix<std::complex<double>, double>& dm,
                                           Charge& chr,
                                           const int nspin,
                                           const bool skip_charge)
{
    // Standard HSolverLCAO solve
    for (int ik = 0; ik < psi.get_nk(); ++ik)
    {
        pHamilt->updateHk(ik);
        psi.fix_k(ik);

        if (method_ == "scalapack_gvx")
        {
#ifdef __MPI
            DiagoScalapack<std::complex<double>> sa;
            sa.diag(pHamilt, psi, &(pes->ekb(ik, 0)));
#endif
        }
#ifdef __ELPA
        else if (method_ == "genelpa")
        {
            DiagoElpa<std::complex<double>> el;
            el.diag(pHamilt, psi, &(pes->ekb(ik, 0)));
        }
        else if (method_ == "elpa")
        {
            DiagoElpaNative<std::complex<double>> el;
            el.diag(pHamilt, psi, &(pes->ekb(ik, 0)));
        }
#endif
#ifdef __CUDA
        else if (method_ == "cusolver")
        {
            DiagoCusolver<std::complex<double>> cu;
            hamilt::MatrixBlock<std::complex<double>> hk, sk;
            pHamilt->matrix(hk, sk);
            cu.diag(hk, sk, psi, &(pes->ekb(ik, 0)));
        }
#endif
        else if (method_ == "lapack")
        {
            DiagoLapack<std::complex<double>> la;
            la.diag(pHamilt, psi, &(pes->ekb(ik, 0)));
        }
        else
        {
            ModuleBase::WARNING_QUIT("HSolverLCAOSubspace::solve_fullspace",
                                      "This method is not supported for lcao basis in ABACUS!");
        }
    }

    elecstate::calculate_weights(pes->ekb, pes->wg, pes->klist,
                                  pes->eferm, pes->f_en,
                                  pes->nelec_spin, pes->skip_weights);
    elecstate::calEBand(pes->ekb, pes->wg, pes->f_en);
    elecstate::cal_dm_psi(dm.get_paraV_pointer(), pes->wg, psi, dm);
    dm.cal_DMR();

    if (!skip_charge)
    {
        LCAO_domain::dm2rho(dm.get_DMR_vector(), nspin, &chr);
    }
}

SubspaceSolverResult HSolverLCAOSubspace::solve_subspace(
    hamilt::Hamilt<std::complex<double>>* pHamilt,
    psi::Psi<std::complex<double>>& psi,
    elecstate::ElecState* pes,
    elecstate::DensityMatrix<std::complex<double>, double>& dm,
    const int nspin,
    const bool skip_charge,
    const std::vector<ModuleBase::Vector3<double>>* lambda)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "solve_subspace");
    ModuleBase::timer::start("HSolverLCAOSubspace", "solve_subspace");

    SubspaceSolverResult result{};

    if (!cache_.is_valid())
    {
        ModuleBase::timer::end("HSolverLCAOSubspace", "solve_subspace");
        return result;
    }

    const int nk = cache_.nk();
    const int nbands = cache_.nbands();
    const int nn = nbands * nbands;
    const int nat = cache_.nat();

    // Resize temporary buffers if needed
    if (static_cast<int>(h_tmp_.size()) < nn)
    {
        h_tmp_.resize(nn);
        s_tmp_.resize(nn);
        s_copy_.resize(nn);
        eigenvalues_.resize(nbands);
        eigenvectors_.resize(nn);
    }

    // Storage for eigenvectors per k-point
    std::vector<std::vector<std::complex<double>>> vcc_all(nk);
    double max_eig_change = 0.0;

    for (int ik = 0; ik < nk; ik++)
    {
        // Copy H0_sub from cache to temporary
        std::memcpy(h_tmp_.data(), cache_.H0_sub(ik), sizeof(std::complex<double>) * nn);

        // Apply lambda correction: H_sub = H0_sub + sum_I (lambda_I - lambda_ref_I) * P_I_sub
        apply_lambda_correction(h_tmp_.data(), ik, nbands, lambda);

        // Copy S_sub for diag_hegvd (it modifies S in place)
        std::memcpy(s_copy_.data(), cache_.S_sub(ik), sizeof(std::complex<double>) * nn);

        // Diagonalize: H_sub * V = S_sub * V * eps
        DiagoIterAssist<std::complex<double>>::diag_hegvd(
            nbands, nbands, h_tmp_.data(), s_copy_.data(), nbands,
            eigenvalues_.data(), eigenvectors_.data());

        // Store eigenvectors
        vcc_all[ik].assign(eigenvectors_.data(), eigenvectors_.data() + nn);

        // Update eigenvalues in pelec
        for (int ib = 0; ib < nbands; ib++)
        {
            double old_eig = cache_.ekb_ref(ik)[ib];
            pes->ekb(ik, ib) = eigenvalues_[ib];
            double change = std::abs(eigenvalues_[ib] - old_eig);
            if (change > max_eig_change) max_eig_change = change;
        }
    }

    // Compute weights from new eigenvalues
    elecstate::calculate_weights(pes->ekb, pes->wg, pes->klist,
                                  pes->eferm, pes->f_en,
                                  pes->nelec_spin, pes->skip_weights);
    elecstate::calEBand(pes->ekb, pes->wg, pes->f_en);

    // Rotate psi, compute full-space DMR, then restore psi
    const int nlocal = ParaV_->get_global_row_size();
    rotate_psi_subspace(psi, vcc_all, nbands, nk);

    elecstate::cal_dm_psi(dm.get_paraV_pointer(), pes->wg, psi, dm);
    dm.cal_DMR();

    result.success = true;
    result.used_subspace_approximation = true;
    result.max_eigenvalue_change = max_eig_change;
    result.nbands = nbands;
    result.nk = nk;

    ModuleBase::timer::end("HSolverLCAOSubspace", "solve_subspace");
    return result;
}

SubspaceSolverResult HSolverLCAOSubspace::solve_first_order(
    hamilt::Hamilt<std::complex<double>>* pHamilt,
    psi::Psi<std::complex<double>>& psi,
    elecstate::ElecState* pes,
    elecstate::DensityMatrix<std::complex<double>, double>& dm,
    const int nspin,
    const bool skip_charge,
    const std::vector<ModuleBase::Vector3<double>>* lambda)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "solve_first_order");
    ModuleBase::timer::start("HSolverLCAOSubspace", "solve_first_order");

    SubspaceSolverResult result{};

    if (!cache_.is_valid())
    {
        ModuleBase::timer::end("HSolverLCAOSubspace", "solve_first_order");
        return result;
    }

    const int nk = cache_.nk();
    const int nbands = cache_.nbands();
    const int nat = cache_.nat();
    const auto& lambda_ref = cache_.lambda_ref();

    double max_eig_change = 0.0;

    // First-order eigenvalue shift: delta_eps = sum_I dl_I * diag(P_I_sub)_ib
    for (int ik = 0; ik < nk; ik++)
    {
        int spin_sign = (pes->klist->isk[ik] == 0) ? 1 : -1;

        for (int ib = 0; ib < nbands; ib++)
        {
            double delta_epsilon = 0.0;

            for (int iat = 0; iat < nat; iat++)
            {
                const std::complex<double>* p = cache_.P_I_sub(ik, iat);
                if (p == nullptr) continue;

                double p_diag = p[ib + ib * nbands].real();
                double dl = (*lambda)[iat][2] - lambda_ref[iat][2];
                delta_epsilon += dl * p_diag;
            }

            double old_eig = cache_.ekb_ref(ik)[ib];
            double new_eig = old_eig - spin_sign * delta_epsilon;
            pes->ekb(ik, ib) = new_eig;

            double change = std::abs(new_eig - old_eig);
            if (change > max_eig_change) max_eig_change = change;
        }
    }

    // Compute weights from shifted eigenvalues
    elecstate::calculate_weights(pes->ekb, pes->wg, pes->klist,
                                  pes->eferm, pes->f_en,
                                  pes->nelec_spin, pes->skip_weights);
    elecstate::calEBand(pes->ekb, pes->wg, pes->f_en);

    // Reuse current wavefunctions (unchanged in first-order approximation)
    elecstate::cal_dm_psi(dm.get_paraV_pointer(), pes->wg, psi, dm);
    dm.cal_DMR();

    result.success = true;
    result.used_subspace_approximation = true;
    result.max_eigenvalue_change = max_eig_change;
    result.nbands = nbands;
    result.nk = nk;

    ModuleBase::timer::end("HSolverLCAOSubspace", "solve_first_order");
    return result;
}

bool HSolverLCAOSubspace::build_subspace(
    hamilt::Hamilt<std::complex<double>>* pHamilt,
    psi::Psi<std::complex<double>>& psi,
    elecstate::ElecState* pes,
    elecstate::DensityMatrix<std::complex<double>, double>& dm,
    const int nspin,
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "build_subspace");
    ModuleBase::timer::start("HSolverLCAOSubspace", "build_subspace");

    const int nk = psi.get_nk();
    const int nbands = ParaV_->get_nbands();
    const int nat = lambda_ref.size();
    const int nlocal = ParaV_->get_global_row_size();
    const int nn = nbands * nbands;

    // Full diagonalization to get correct psi at lambda_ref
    solve_fullspace(pHamilt, psi, pes, dm, *pes->charge, nspin, true);

    // Clear existing cache
    cache_.clear();

    // Allocate arrays
    std::vector<std::complex<double>> H0_sub_raw(nk * nn);
    std::vector<std::complex<double>> S_sub_raw(nk * nn);
    std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_all(nk);

    // Compute H0_sub, S_sub, P_I_sub for each k-point
    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);

        // H0_sub = C^dag H C, S_sub = C^dag S C
        DiagoIterAssist<std::complex<double>>::cal_hs_subspace(pHamilt, psi,
                                                                H0_sub_raw.data() + ik * nn,
                                                                S_sub_raw.data() + ik * nn);

        // P_I_sub = C^dag D_I C for each constrained atom
        auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(pHamilt);
        if (dspin_op != nullptr)
        {
            dspin_op->cal_PI_sub(
                pes->klist->kvec_d[ik],
                psi.get_pointer(),
                nbands,
                P_I_sub_all[ik]);
        }
    }

    // Collect eigenvalues
    std::vector<double> ekb_ref_all(nk * nbands);
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nbands; ib++)
        {
            ekb_ref_all[ik * nbands + ib] = pes->ekb(ik, ib);
        }
    }

    // Build the cache
    cache_.build(nk, nbands, nat,
                 H0_sub_raw.data(), S_sub_raw.data(),
                 std::move(P_I_sub_all),
                 ekb_ref_all, lambda_ref);

    ModuleBase::timer::end("HSolverLCAOSubspace", "build_subspace");
    return true;
}

void HSolverLCAOSubspace::clear_subspace()
{
    if (!persistent_)
    {
        cache_.clear();
    }
}

bool HSolverLCAOSubspace::build_subspace_lcao(
    hamilt::Hamilt<std::complex<double>>* pHamilt,
    psi::Psi<std::complex<double>>& psi,
    elecstate::ElecState* pes,
    elecstate::DensityMatrix<std::complex<double>, double>& dm,
    Charge& chr,
    const int nspin,
    const bool skip_charge,
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "build_subspace_lcao");
    ModuleBase::timer::start("HSolverLCAOSubspace", "build_subspace_lcao");

    const int nk = psi.get_nk();
    const int nbands = ParaV_->get_nbands();
    const int nat = lambda_ref.size();
    const int nlocal = ParaV_->get_global_row_size();
    const int nn = nbands * nbands;

    // Full diagonalization to get correct psi at lambda_ref
    solve_fullspace(pHamilt, psi, pes, dm, chr, nspin, skip_charge);

    // Clear existing cache
    cache_.clear();

    // Allocate arrays
    std::vector<std::complex<double>> H0_sub_raw(nk * nn);
    std::vector<std::complex<double>> S_sub_raw(nk * nn);
    std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_all(nk);

    // For LCAO: compute H_sub = C^† H C and S_sub = C^† S C using correct dimensions
    // C is stored as psi[nk][nbands][nlocal] with leading dimension = nlocal
    const int lda = nlocal;  // leading dimension of psi matrix

    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);
        std::complex<double>* psi_ptr = psi.get_pointer();

        // Get H(k) and S(k) matrices from Hamiltonian
        pHamilt->updateHk(ik);
        hamilt::MatrixBlock<std::complex<double>> h_mat, s_mat;
        pHamilt->matrix(h_mat, s_mat);

        // H_sub = C^† H C: (nbands x nlocal) * (nlocal x nlocal) * (nlocal x nbands)
        // First compute temp = H * C: (nlocal x nlocal) * (nlocal x nbands) -> (nlocal x nbands)
        std::vector<std::complex<double>> h_temp(lda * nbands, {0.0, 0.0});
        const std::complex<double> one = {1.0, 0.0};
        const std::complex<double> zero = {0.0, 0.0};

        zgemm_("N", "N", &lda, &nbands, &lda,
               &one, h_mat.p, &lda,
               psi_ptr, &lda,
               &zero, h_temp.data(), &lda);

        // Then H_sub = C^† * temp: (nbands x nlocal) * (nlocal x nbands) -> (nbands x nbands)
        zgemm_("C", "N", &nbands, &nbands, &lda,
               &one, psi_ptr, &lda,
               h_temp.data(), &lda,
               &zero, H0_sub_raw.data() + ik * nn, &nbands);

        // S_sub = C^† S C
        std::vector<std::complex<double>> s_temp(lda * nbands, {0.0, 0.0});
        zgemm_("N", "N", &lda, &nbands, &lda,
               &one, s_mat.p, &lda,
               psi_ptr, &lda,
               &zero, s_temp.data(), &lda);

        zgemm_("C", "N", &nbands, &nbands, &lda,
               &one, psi_ptr, &lda,
               s_temp.data(), &lda,
               &zero, S_sub_raw.data() + ik * nn, &nbands);

        // P_I_sub = C^† D_I C for each constrained atom (if DeltaSpin is active)
        auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(pHamilt);
        if (dspin_op != nullptr)
        {
            dspin_op->cal_PI_sub(
                pes->klist->kvec_d[ik],
                psi.get_pointer(),
                nbands,
                P_I_sub_all[ik]);
        }
    }

    // Collect eigenvalues
    std::vector<double> ekb_ref_all(nk * nbands);
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nbands; ib++)
        {
            ekb_ref_all[ik * nbands + ib] = pes->ekb(ik, ib);
        }
    }

    // Build the cache
    cache_.build(nk, nbands, nat,
                 H0_sub_raw.data(), S_sub_raw.data(),
                 std::move(P_I_sub_all),
                 ekb_ref_all, lambda_ref);

    ModuleBase::timer::end("HSolverLCAOSubspace", "build_subspace_lcao");
    return true;
}

void HSolverLCAOSubspace::update_subspace_cache(
    hamilt::Hamilt<std::complex<double>>* pHamilt,
    psi::Psi<std::complex<double>>& psi,
    elecstate::ElecState* pes,
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref)
{
    ModuleBase::TITLE("HSolverLCAOSubspace", "update_subspace_cache");
    ModuleBase::timer::start("HSolverLCAOSubspace", "update_subspace_cache");

    const int nk = psi.get_nk();
    const int nbands = ParaV_->get_nbands();
    const int nat = lambda_ref.size();
    const int nn = nbands * nbands;

    // Allocate arrays
    std::vector<std::complex<double>> H0_sub_raw(nk * nn);
    std::vector<std::complex<double>> S_sub_raw(nk * nn);
    std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_all(nk);

    // Compute H0_sub, S_sub for each k-point using current psi
    // Note: psi should contain the wavefunctions from the previous SCF step
    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);

        // H0_sub = C^dag H C, S_sub = C^dag S C
        DiagoIterAssist<std::complex<double>>::cal_hs_subspace(pHamilt, psi,
                                                                H0_sub_raw.data() + ik * nn,
                                                                S_sub_raw.data() + ik * nn);

        // P_I_sub = C^dag D_I C for each constrained atom (if DeltaSpin is active)
        auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(pHamilt);
        if (dspin_op != nullptr)
        {
            dspin_op->cal_PI_sub(
                pes->klist->kvec_d[ik],
                psi.get_pointer(),
                nbands,
                P_I_sub_all[ik]);
        }
    }

    // Collect eigenvalues from pes
    std::vector<double> ekb_ref_all(nk * nbands);
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nbands; ib++)
        {
            ekb_ref_all[ik * nbands + ib] = pes->ekb(ik, ib);
        }
    }

    // Update the cache
    cache_.build(nk, nbands, nat,
                 H0_sub_raw.data(), S_sub_raw.data(),
                 std::move(P_I_sub_all),
                 ekb_ref_all, lambda_ref);

    ModuleBase::timer::end("HSolverLCAOSubspace", "update_subspace_cache");
}

void HSolverLCAOSubspace::apply_lambda_correction(
    std::complex<double>* h_sub,
    int ik,
    int nbands,
    const std::vector<ModuleBase::Vector3<double>>* lambda)
{
    if (lambda == nullptr || !cache_.is_valid()) return;

    const int nat = cache_.nat();
    const int nn = nbands * nbands;
    const auto& lambda_ref = cache_.lambda_ref();

    for (int iat = 0; iat < nat; iat++)
    {
        const std::complex<double>* p = cache_.P_I_sub(ik, iat);
        if (p == nullptr) continue;

        double dl = (*lambda)[iat][2] - lambda_ref[iat][2];
        const std::complex<double> coeff(dl, 0.0);

        for (int ij = 0; ij < nn; ij++)
        {
            h_sub[ij] += coeff * p[ij];
        }
    }
}

void HSolverLCAOSubspace::rotate_psi_subspace(
    psi::Psi<std::complex<double>>& psi,
    const std::vector<std::vector<std::complex<double>>>& vcc_all,
    int nbands,
    int nk)
{
    const int nlocal = ParaV_->get_global_row_size();

    // Save original psi for all k-points
    std::vector<std::vector<std::complex<double>>> psi_save(nk);
    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);
        std::complex<double>* psi_ptr = psi.get_pointer();
        psi_save[ik].assign(psi_ptr, psi_ptr + nlocal * nbands);
    }

    // Rotate psi for each k-point and compute DMR
    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);
        std::complex<double>* psi_ptr = psi.get_pointer();

        // C_new = C_old * V
        std::vector<std::complex<double>> temp(nlocal * nbands, {0.0, 0.0});
        const std::complex<double> one = {1.0, 0.0};
        const std::complex<double> zero = {0.0, 0.0};

        zgemm_("N", "N", &nlocal, &nbands, &nbands,
               &one, psi_ptr, &nlocal,
               vcc_all[ik].data(), &nbands,
               &zero, temp.data(), &nlocal);

        std::memcpy(psi_ptr, temp.data(), sizeof(std::complex<double>) * nlocal * nbands);
    }

    // Restore original psi
    for (int ik = 0; ik < nk; ik++)
    {
        psi.fix_k(ik);
        std::complex<double>* psi_ptr = psi.get_pointer();
        std::memcpy(psi_ptr, psi_save[ik].data(),
                     sizeof(std::complex<double>) * nlocal * nbands);
    }
}

// ===================================================================
// Type conversion helpers
// ===================================================================

SubspaceMode subspace_mode_from_string(const std::string& s)
{
    if (s == "full" || s == "fullspace" || s == "full_space")
        return SubspaceMode::FullSpace;
    if (s == "subspace" || s == "subspace_diag")
        return SubspaceMode::Subspace;
    if (s == "first_order" || s == "firstorder" || s == "linear_response")
        return SubspaceMode::FirstOrder;
    return SubspaceMode::FullSpace;
}

std::string subspace_mode_to_string(SubspaceMode m)
{
    switch (m)
    {
        case SubspaceMode::FullSpace:
            return "FullSpace";
        case SubspaceMode::Subspace:
            return "Subspace";
        case SubspaceMode::FirstOrder:
            return "FirstOrder";
        default:
            return "Unknown";
    }
}

// Explicit template instantiations
// (Not needed for this non-template class)

} // namespace hsolver
