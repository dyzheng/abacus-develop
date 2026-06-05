#ifndef HSOLVER_LCAO_SUBSPACE_H
#define HSOLVER_LCAO_SUBSPACE_H

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "source_base/vector3.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_estate/elecstate.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_hamilt/hamilt.h"
#include "source_psi/psi.h"

namespace hsolver
{

/// @brief Precision mode for subspace operations
enum class SubspacePrecision {
    fp32,  ///< Use single precision for GEMM operations
    fp64   ///< Use double precision (default)
};

/// @brief LCAO subspace diagonalization cache
///
/// Stores the subspace Hamiltonian, overlap, and projector matrices
/// computed at a reference lambda value. Used to accelerate subsequent
/// diagonalizations by solving in the reduced nbands×nbands subspace
/// instead of the full NLOCAL×NLOCAL space.
///
/// @par Memory layout
/// - H0_sub: [nk * nbands * nbands], flattened 1D array (column-major)
/// - S_sub:  [nk * nbands * nbands], flattened 1D array (column-major)
/// - P_I_sub: nested vector [nk][nat][nbands*nbands] for per-atom projector matrices
/// - ekb_ref: reference eigenvalues [nk * nbands]
/// - lambda_ref: lambda values at which the subspace was built [nat]
class LCAOSubspaceCache
{
public:
    LCAOSubspaceCache();
    ~LCAOSubspaceCache() = default;

    // Non-copyable, movable
    LCAOSubspaceCache(const LCAOSubspaceCache&) = delete;
    LCAOSubspaceCache& operator=(const LCAOSubspaceCache&) = delete;
    LCAOSubspaceCache(LCAOSubspaceCache&&) = default;
    LCAOSubspaceCache& operator=(LCAOSubspaceCache&&) = default;

    /// @brief Build the subspace cache from computed data
    void build(int nk,
               int nbands,
               int nat,
               const std::complex<double>* H0_sub_raw,
               const std::complex<double>* S_sub_raw,
               std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_all,
               const std::vector<double>& ekb_ref_all,
               const std::vector<ModuleBase::Vector3<double>>& lambda_ref);

    /// @brief Clear all cached data and reset to empty state
    void clear();

    /// @brief Check if the cache has been built and is valid
    bool is_valid() const { return valid_; }

    int nk() const { return nk_; }
    int nbands() const { return nbands_; }
    int nat() const { return nat_; }

    const std::complex<double>* H0_sub(int ik) const;
    const std::complex<double>* S_sub(int ik) const;
    const std::complex<double>* P_I_sub(int ik, int iat) const;
    const double* ekb_ref(int ik) const;
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref() const;
    const std::vector<std::vector<std::vector<std::complex<double>>>>& P_I_sub_all() const
    {
        return P_I_sub_;
    }

private:
    int nk_ = 0;
    int nbands_ = 0;
    int nat_ = 0;
    bool valid_ = false;

    std::vector<std::complex<double>> H0_sub_;
    std::vector<std::complex<double>> S_sub_;
    std::vector<std::vector<std::vector<std::complex<double>>>> P_I_sub_;
    std::vector<double> ekb_ref_;
    std::vector<ModuleBase::Vector3<double>> lambda_ref_;
};

/// @brief Result of a subspace diagonalization operation
struct SubspaceSolverResult
{
    bool success = false;
    bool used_subspace_approximation = false;
    double max_eigenvalue_change = 0.0;
    int nbands = 0;
    int nk = 0;
};

/// @brief Subspace diagonalization mode
enum class SubspaceMode {
    FullSpace,       ///< Full HSolverLCAO diagonalization (ground truth)
    Subspace,        ///< Subspace diagonalization with wavefunction rotation
    FirstOrder       ///< First-order eigenvalue response (no wavefunction change)
};

/// @brief LCAO subspace diagonalization solver
///
/// Provides an optional fast diagonalization path for LCAO calculations
/// within the DeltaSpin lambda loop and MD ionic steps. When the subspace
/// cache is valid, solves in the reduced nbands×nbands subspace instead
/// of the full NLOCAL×NLOCAL space.
///
/// @par Algorithm
///   Phase 1 (build_subspace):
///     1. Full diagonalization to get correct psi at lambda_ref
///     2. Compute H0_sub = C^dag H C, S_sub = C^dag S C
///     3. Compute P_I_sub = C^dag D_I C for each constrained atom
///
///   Phase 2 (solve):
///     1. H_sub(lambda) = H0_sub + sum_I (lambda_I - lambda_ref_I) * P_I_sub
///     2. diag_hegvd: H_sub V = S_sub V eps
///     3. [Optional] Rotate psi: C_new = C_original * V
///     4. Build full-space DM from rotated psi
///
/// @par MD ionic step reuse
/// The subspace cache can persist across MD ionic steps. Call
/// `set_persistent(true)` to prevent automatic cache clearing.
/// The cache should be manually cleared when the atomic positions
/// change significantly (e.g., after a geometry update).
class HSolverLCAOSubspace
{
public:
    HSolverLCAOSubspace(const Parallel_Orbitals* ParaV_in,
                        std::string method_in,
                        SubspaceMode mode_in = SubspaceMode::FullSpace,
                        SubspacePrecision precision_in = SubspacePrecision::fp64);

    /// @brief Solve using subspace diagonalization (if cache is valid)
    ///
    /// If the subspace cache is not built or mode is FullSpace,
    /// falls back to standard HSolverLCAO diagonalization.
    void solve(hamilt::Hamilt<std::complex<double>>* pHamilt,
               psi::Psi<std::complex<double>>& psi,
               elecstate::ElecState* pes,
               elecstate::DensityMatrix<std::complex<double>, double>& dm,
               Charge& chr,
               const int nspin,
               const bool skip_charge,
               const std::vector<ModuleBase::Vector3<double>>* lambda = nullptr);

    /// @brief Build the subspace cache at current lambda reference point
    ///
    /// Performs a full diagonalization and extracts subspace matrices.
    /// Call this when the system is near convergence or at the start
    /// of an MD ionic step.
    bool build_subspace(hamilt::Hamilt<std::complex<double>>* pHamilt,
                        psi::Psi<std::complex<double>>& psi,
                        elecstate::ElecState* pes,
                        elecstate::DensityMatrix<std::complex<double>, double>& dm,
                        const int nspin,
                        const std::vector<ModuleBase::Vector3<double>>& lambda_ref);

    /// @brief Update subspace cache using current wavefunctions (without full diagonalization)
    ///
    /// Uses the current psi to rebuild H_sub and S_sub. Assumes psi contains
    /// the wavefunctions from the previous SCF step. This is cheaper than
    /// build_subspace() because it skips the full diagonalization.
    void update_subspace_cache(hamilt::Hamilt<std::complex<double>>* pHamilt,
                               psi::Psi<std::complex<double>>& psi,
                               elecstate::ElecState* pes,
                               const std::vector<ModuleBase::Vector3<double>>& lambda_ref);

    /// @brief Clear the subspace cache
    void clear_subspace();

    /// @brief Check if subspace cache is valid
    bool has_subspace() const { return cache_.is_valid(); }

    /// @brief Set subspace diagonalization mode
    void set_mode(SubspaceMode mode) { mode_ = mode; }
    SubspaceMode mode() const { return mode_; }

    /// @brief Set execution precision
    void set_precision(SubspacePrecision prec) { precision_ = prec; }
    SubspacePrecision precision() const { return precision_; }

    /// @brief Enable/disable persistent cache (for MD ionic step reuse)
    void set_persistent(bool persistent) { persistent_ = persistent; }
    bool is_persistent() const { return persistent_; }

    /// @brief Get the reference lambda values
    const std::vector<ModuleBase::Vector3<double>>& lambda_ref() const
    {
        return cache_.lambda_ref();
    }

private:
    /// @brief Solve in subspace (SubspaceMode::Subspace)
    SubspaceSolverResult solve_subspace(hamilt::Hamilt<std::complex<double>>* pHamilt,
                                         psi::Psi<std::complex<double>>& psi,
                                         elecstate::ElecState* pes,
                                         elecstate::DensityMatrix<std::complex<double>, double>& dm,
                                         const int nspin,
                                         const bool skip_charge,
                                         const std::vector<ModuleBase::Vector3<double>>* lambda);

    /// @brief Solve with first-order response (SubspaceMode::FirstOrder)
    SubspaceSolverResult solve_first_order(hamilt::Hamilt<std::complex<double>>* pHamilt,
                                            psi::Psi<std::complex<double>>& psi,
                                            elecstate::ElecState* pes,
                                            elecstate::DensityMatrix<std::complex<double>, double>& dm,
                                            const int nspin,
                                            const bool skip_charge,
                                            const std::vector<ModuleBase::Vector3<double>>* lambda);

    /// @brief Full space solve fallback
    void solve_fullspace(hamilt::Hamilt<std::complex<double>>* pHamilt,
                         psi::Psi<std::complex<double>>& psi,
                         elecstate::ElecState* pes,
                         elecstate::DensityMatrix<std::complex<double>, double>& dm,
                         Charge& chr,
                         const int nspin,
                         const bool skip_charge);

    /// @brief Apply DeltaSpin correction to subspace Hamiltonian
    void apply_lambda_correction(std::complex<double>* h_sub,
                                  int ik,
                                  int nbands,
                                  const std::vector<ModuleBase::Vector3<double>>* lambda);

    /// @brief Rotate wavefunctions in subspace: C_new = C_old * V
    void rotate_psi_subspace(psi::Psi<std::complex<double>>& psi,
                              const std::vector<std::vector<std::complex<double>>>& vcc_all,
                              int nbands,
                              int nk);

    const Parallel_Orbitals* ParaV_ = nullptr;
    const std::string method_;
    SubspaceMode mode_ = SubspaceMode::FullSpace;
    SubspacePrecision precision_ = SubspacePrecision::fp64;
    bool persistent_ = false;

    LCAOSubspaceCache cache_;

    // Temporary buffers (reused across solve calls)
    std::vector<std::complex<double>> h_tmp_;
    std::vector<std::complex<double>> s_tmp_;
    std::vector<std::complex<double>> s_copy_;
    std::vector<double> eigenvalues_;
    std::vector<std::complex<double>> eigenvectors_;
};

/// @brief Convert string to SubspaceMode
SubspaceMode subspace_mode_from_string(const std::string& s);

/// @brief Convert SubspaceMode to string
std::string subspace_mode_to_string(SubspaceMode m);

} // namespace hsolver

#endif // HSOLVER_LCAO_SUBSPACE_H
