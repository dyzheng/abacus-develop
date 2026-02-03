#ifndef PY_ESOLVER_LCAO_HPP
#define PY_ESOLVER_LCAO_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <fstream>

#include "../utils/pybind_utils.h"
#include "interfaces/i_scf_controller.hpp"
#include "interfaces/i_hamiltonian_builder.hpp"
#include "interfaces/i_charge_mixer.hpp"
#include "interfaces/i_diagonalizer.hpp"
#include "components/scf_controller_lcao.hpp"

// Forward declarations for ABACUS types
class UnitCell;
class Charge;
class Parallel_Orbitals;
namespace elecstate {
    struct fenergy;
    class ElecState;
    template <typename TK, typename TR> class DensityMatrix;
}
namespace hamilt {
    template <typename T> class HContainer;
    template <typename T> class AtomPair;
    template <typename T> class BaseMatrix;
    template <typename TK, typename TR> class HamiltLCAO;
}
namespace ModuleESolver {
    template <typename TK, typename TR> class ESolver_KS_LCAO;
}

namespace py = pybind11;

namespace py_esolver
{

/**
 * @brief Accessor class for charge density data
 *
 * Provides Python access to charge density (rho) in real and reciprocal space
 */
class PyChargeAccessor
{
public:
    PyChargeAccessor() = default;

    /// Set internal pointers from Charge object
    void set_from_charge(const Charge* chr);

    /// Set data directly (for compatibility with existing code)
    void set_data(const double* rho_ptr, int nspin, int nrxx);

    /// Get real-space charge density as numpy array (nspin, nrxx)
    py::array_t<double> get_rho() const;

    /// Get reciprocal-space charge density as numpy array (nspin, ngmc)
    py::array_t<std::complex<double>> get_rhog() const;

    /// Get core charge density
    py::array_t<double> get_rho_core() const;

    /// Get number of spin channels
    int get_nspin() const { return nspin_; }

    /// Get number of real-space grid points
    int get_nrxx() const { return nrxx_; }

    /// Get number of G-vectors
    int get_ngmc() const { return ngmc_; }

    /// Check if data is valid
    bool is_valid() const { return (chr_ptr_ != nullptr || rho_ptr_ != nullptr) && nspin_ > 0; }

private:
    const Charge* chr_ptr_ = nullptr;
    const double* rho_ptr_ = nullptr;  // Direct pointer for compatibility
    int nspin_ = 0;
    int nrxx_ = 0;
    int ngmc_ = 0;
};

/**
 * @brief Accessor class for force data
 *
 * Provides Python access to atomic forces
 */
class PyForceAccessor
{
public:
    PyForceAccessor() = default;

    /// Set force data from raw pointer
    void set_from_matrix(const double* force_ptr, int nat);

    /// Get forces as numpy array with shape (nat, 3)
    py::array_t<double> get_forces() const;

    /// Get number of atoms
    int get_nat() const { return nat_; }

    /// Check if data is valid
    bool is_valid() const { return nat_ > 0 && !forces_.empty(); }

private:
    std::vector<double> forces_;
    int nat_ = 0;
};

/**
 * @brief Accessor class for stress tensor data
 *
 * Provides Python access to stress tensor
 */
class PyStressAccessor
{
public:
    PyStressAccessor() = default;

    /// Set stress data from raw pointer (3x3 matrix)
    void set_from_matrix(const double* stress_ptr);

    /// Get stress tensor as numpy array with shape (3, 3)
    py::array_t<double> get_stress() const;

    /// Get stress in Voigt notation (6,): xx, yy, zz, yz, xz, xy
    py::array_t<double> get_stress_voigt() const;

    /// Check if data is valid
    bool is_valid() const { return valid_; }

private:
    std::array<double, 9> stress_;
    bool valid_ = false;
};

/**
 * @brief Accessor class for energy data
 *
 * Provides Python access to various energy components
 */
class PyEnergyAccessor
{
public:
    PyEnergyAccessor() = default;

    /// Set from fenergy structure
    void set_from_fenergy(const elecstate::fenergy* f_en);

    /// Set energies directly (for compatibility)
    void set_energies(double etot, double eband, double hartree,
                      double etxc, double ewald, double demet,
                      double exx, double evdw);

    /// Get total energy (Ry)
    double get_etot() const { return etot_; }

    /// Get band energy (Ry)
    double get_eband() const { return eband_; }

    /// Get Hartree energy (Ry)
    double get_hartree_energy() const { return hartree_energy_; }

    /// Get exchange-correlation energy (Ry)
    double get_etxc() const { return etxc_; }

    /// Get Ewald energy (Ry)
    double get_ewald_energy() const { return ewald_energy_; }

    /// Get -TS term for metals (Ry)
    double get_demet() const { return demet_; }

    /// Get exact exchange energy (Ry)
    double get_exx() const { return exx_; }

    /// Get van der Waals energy (Ry)
    double get_evdw() const { return evdw_; }

    /// Get all energies as a dictionary
    py::dict get_all_energies() const;

private:
    double etot_ = 0.0;
    double eband_ = 0.0;
    double hartree_energy_ = 0.0;
    double etxc_ = 0.0;
    double ewald_energy_ = 0.0;
    double demet_ = 0.0;
    double exx_ = 0.0;
    double evdw_ = 0.0;
};

/**
 * @brief Accessor class for Hamiltonian matrix data
 *
 * Provides Python access to H(R), S(R), H(k), S(k) matrices
 */
template <typename TK, typename TR = double>
class PyHamiltonianAccessor
{
public:
    PyHamiltonianAccessor() = default;

    /// Set from HamiltLCAO object
    void set_from_hamilt(hamilt::HamiltLCAO<TK, TR>* hamilt_lcao, int nks, const Parallel_Orbitals* pv);

    /// Set dimensions directly (for compatibility)
    void set_dimensions(int nbasis, int nks);

    /// Set H(k) data for a specific k-point
    void set_Hk_data(int ik, const TK* data, int nrow, int ncol);

    /// Set S(k) data for a specific k-point
    void set_Sk_data(int ik, const TK* data, int nrow, int ncol);

    /// Get number of basis functions
    int get_nbasis() const { return nbasis_; }

    /// Get number of k-points
    int get_nks() const { return nks_; }

    /// Get local matrix size (for 2D distribution)
    int get_nloc() const { return nloc_; }

    /// Get H(k) matrix for specific k-point (local part in 2D distribution)
    py::array_t<TK> get_Hk(int ik) const;

    /// Get S(k) matrix for specific k-point (local part in 2D distribution)
    py::array_t<TK> get_Sk(int ik) const;

    /// Get H(R) in sparse COO format: returns (row_indices, col_indices, R_vectors, values)
    py::tuple get_HR_sparse() const;

    /// Get S(R) in sparse COO format: returns (row_indices, col_indices, R_vectors, values)
    py::tuple get_SR_sparse() const;

    /// Get H(R) as dictionary: {(iat1, iat2, R): matrix}
    py::dict get_HR() const;

    /// Get S(R) as dictionary: {(iat1, iat2, R): matrix}
    py::dict get_SR() const;

    /// Check if data is valid
    bool is_valid() const { return (hamilt_ptr_ != nullptr || nbasis_ > 0) && nks_ > 0; }

private:
    hamilt::HamiltLCAO<TK, TR>* hamilt_ptr_ = nullptr;
    const Parallel_Orbitals* pv_ = nullptr;
    int nbasis_ = 0;
    int nks_ = 0;
    int nloc_ = 0;
    int nrow_ = 0;
    int ncol_ = 0;

    // For direct data access (compatibility mode)
    std::vector<const TK*> hk_ptrs_;
    std::vector<const TK*> sk_ptrs_;
    std::vector<std::pair<int, int>> matrix_dims_;
};

/**
 * @brief Accessor class for density matrix data
 *
 * Provides Python access to DM(k) and DM(R)
 */
template <typename TK, typename TR = double>
class PyDensityMatrixAccessor
{
public:
    PyDensityMatrixAccessor() = default;

    /// Set from DensityMatrix object
    void set_from_dm(elecstate::DensityMatrix<TK, TR>* dm);

    /// Set dimensions directly (for compatibility)
    void set_dimensions(int nks, int nrow, int ncol);

    /// Set DM(k) data for a specific k-point
    void set_DMK_data(int ik, const TK* data);

    /// Get DM(k) for specific k-point
    py::array_t<TK> get_DMK(int ik) const;

    /// Get all DM(k) matrices
    std::vector<py::array_t<TK>> get_DMK_all() const;

    /// Get DM(R) in sparse format as dictionary
    py::dict get_DMR() const;

    /// Get number of k-points
    int get_nks() const { return nks_; }

    /// Get matrix row dimension
    int get_nrow() const { return nrow_; }

    /// Get matrix column dimension
    int get_ncol() const { return ncol_; }

    /// Check if data is valid
    bool is_valid() const { return (dm_ptr_ != nullptr || nks_ > 0); }

private:
    elecstate::DensityMatrix<TK, TR>* dm_ptr_ = nullptr;
    int nks_ = 0;
    int nrow_ = 0;
    int ncol_ = 0;

    // For direct data access (compatibility mode)
    std::vector<const TK*> dmk_ptrs_;
};

/**
 * @brief Main wrapper class for ESolver_KS_LCAO
 *
 * Provides Python interface for LCAO calculations with breakpoint support.
 * Now uses the component-based architecture with ISCFController.
 *
 * Template parameters:
 *   TK: Type for k-space quantities (double for gamma-only, complex<double> for multi-k)
 *   TR: Type for real-space quantities (typically double)
 */
template <typename TK, typename TR = double>
class PyESolverLCAO
{
public:
    PyESolverLCAO();
    ~PyESolverLCAO();

    // ==================== Initialization ====================

    /// Initialize from INPUT file directory
    void initialize(const std::string& input_dir);

    /// Call before_all_runners
    void before_all_runners();

    // ==================== SCF Control ====================

    /// Prepare for SCF calculation
    void before_scf(int istep = 0);

    /// Run a single SCF iteration
    void run_scf_iteration(int iter);

    /// Run complete SCF loop
    void run_scf(int max_iter = 100);

    /// Finalize SCF calculation
    void after_scf(int istep = 0);

    // ==================== Status Queries ====================

    /// Check if SCF is converged
    bool is_converged() const { return conv_esolver_; }

    /// Get current iteration number
    int get_niter() const { return niter_; }

    /// Get charge density difference (drho)
    double get_drho() const { return drho_; }

    /// Get current SCF step
    int get_istep() const { return istep_; }

    // ==================== Data Accessors ====================

    /// Get charge density accessor
    PyChargeAccessor get_charge() const;

    /// Get energy accessor
    PyEnergyAccessor get_energy() const;

    /// Get Hamiltonian accessor
    PyHamiltonianAccessor<TK, TR> get_hamiltonian() const;

    /// Get density matrix accessor
    PyDensityMatrixAccessor<TK, TR> get_density_matrix() const;

    // ==================== Wave Function Access ====================

    /// Get wave function coefficients for k-point ik
    py::array_t<TK> get_psi(int ik) const;

    /// Get eigenvalues for k-point ik
    py::array_t<double> get_eigenvalues(int ik) const;

    /// Get occupation numbers for k-point ik
    py::array_t<double> get_occupations(int ik) const;

    // ==================== K-point Information ====================

    /// Get number of k-points
    int get_nks() const;

    /// Get k-vector in direct coordinates for k-point ik
    py::array_t<double> get_kvec_d(int ik) const;

    /// Get k-point weights
    py::array_t<double> get_wk() const;

    // ==================== System Information ====================

    /// Get number of basis functions
    int get_nbasis() const;

    /// Get number of bands
    int get_nbands() const;

    /// Get number of spin channels
    int get_nspin() const;

    /// Get number of atoms
    int get_nat() const;

    // ==================== Force and Stress ====================

    /// Calculate forces on atoms
    void cal_force();

    /// Calculate stress tensor
    void cal_stress();

    /// Get force accessor (call cal_force first)
    PyForceAccessor get_force() const;

    /// Get stress accessor (call cal_stress first)
    PyStressAccessor get_stress() const;

    // ==================== Position and Cell Update ====================

    /// Update atomic positions (Angstrom, Cartesian)
    void update_positions(py::array_t<double> positions);

    /// Update cell vectors (Angstrom)
    void update_cell(py::array_t<double> cell);

    /// Get atomic positions (Angstrom, Cartesian)
    py::array_t<double> get_positions() const;

    /// Get cell vectors (Angstrom)
    py::array_t<double> get_cell() const;

    // ==================== Component Access (New API) ====================

    /// Get SCF controller component
    pyabacus::esolver::ISCFController* get_scf_controller()
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
        return nullptr;
    }

    /// Get Hamiltonian builder component
    pyabacus::esolver::IHamiltonianBuilder<TK, TR>* get_hamiltonian_builder()
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
        return nullptr;
    }

    /// Get charge mixer component
    pyabacus::esolver::IChargeMixer* get_charge_mixer()
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
        return nullptr;
    }

    /// Get diagonalizer component
    pyabacus::esolver::IDiagonalizer<TK>* get_diagonalizer()
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
        return nullptr;
    }

    // ==================== Configuration (New API) ====================

    /// Set SCF convergence criteria
    void set_convergence_criteria(double drho_threshold, double energy_threshold, int max_iter)
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
    }

    /// Set mixing parameters
    void set_mixing_beta(double beta)
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
    }

    /// Set mixing method
    void set_mixing_method(const std::string& method)
    {
        // Phase 3 placeholder: requires full ABACUS library linkage
    }

private:
    // Internal state
    bool initialized_ = false;
    bool scf_started_ = false;
    bool conv_esolver_ = false;
    int istep_ = 0;
    int niter_ = 0;
    double drho_ = 0.0;
    double diag_ethr_ = 1e-2;

    // Note: ABACUS objects removed - Phase 3 requires full library linkage
    // For now, ESolver mode uses placeholder implementations
    // Use Driver mode for actual calculations

    // Output stream management
    std::ofstream ofs_running_;
    std::ofstream ofs_warning_;

    // Cached system dimensions
    int nat_ = 0;
    int ntype_ = 0;
    int nks_ = 0;
    int nbasis_ = 0;
    int nbands_ = 0;
    int nspin_ = 1;

    // Force and stress accessors
    PyForceAccessor force_accessor_;
    PyStressAccessor stress_accessor_;
    bool force_calculated_ = false;
    bool stress_calculated_ = false;

    // Helper methods
    void setup_output_streams(const std::string& output_dir);
    void cleanup_output_streams();
    void cache_system_info();
    void update_accessors();
};

// Type aliases for common use cases
using PyESolverLCAO_Gamma = PyESolverLCAO<double, double>;
using PyESolverLCAO_MultiK = PyESolverLCAO<std::complex<double>, double>;

} // namespace py_esolver

#endif // PY_ESOLVER_LCAO_HPP
