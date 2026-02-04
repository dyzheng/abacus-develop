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
#include <fstream>

#include "../utils/pybind_utils.h"
#include "interfaces/i_scf_controller.hpp"
#include "interfaces/i_hamiltonian_builder.hpp"
#include "interfaces/i_charge_mixer.hpp"
#include "interfaces/i_diagonalizer.hpp"
#include "components/scf_controller_lcao.hpp"

// Include accessor headers
#include "accessors/py_charge_accessor.hpp"
#include "accessors/py_force_stress_accessor.hpp"
#include "accessors/py_energy_accessor.hpp"
#include "accessors/py_hamiltonian_accessor.hpp"
#include "accessors/py_density_matrix_accessor.hpp"

// Forward declarations for ABACUS types
class UnitCell;
namespace ModuleESolver {
    template <typename TK, typename TR> class ESolver_KS_LCAO;
}

// Unit conversion constants
namespace py_esolver_constants {
    constexpr double BOHR_TO_ANG = 0.529177249;
    constexpr double ANG_TO_BOHR = 1.0 / BOHR_TO_ANG;
}

namespace py = pybind11;

namespace py_esolver
{

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

    // ABACUS objects
    std::unique_ptr<UnitCell> ucell_;
    ModuleESolver::ESolver_KS_LCAO<TK, TR>* esolver_ = nullptr;
    std::string input_dir_;

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
