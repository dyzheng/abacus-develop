#ifndef PY_ESOLVER_PW_HPP
#define PY_ESOLVER_PW_HPP

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

// Include accessor headers
#include "accessors/py_charge_accessor.hpp"
#include "accessors/py_force_stress_accessor.hpp"
#include "accessors/py_energy_accessor.hpp"

// Forward declarations for ABACUS types
class UnitCell;
namespace ModuleESolver {
    template <typename T, typename Device> class ESolver_KS_PW;
}
namespace base_device {
    class DEVICE_CPU;
}

// Unit conversion constants are defined in py_esolver_lcao.hpp
// Use those definitions to avoid redefinition errors

namespace py = pybind11;

namespace py_esolver
{

/**
 * @brief Main wrapper class for ESolver_KS_PW
 *
 * Provides Python interface for plane wave calculations with breakpoint support.
 *
 * Template parameters:
 *   T: Type for wave function coefficients (complex<float> or complex<double>)
 */
template <typename T>
class PyESolverPW
{
public:
    PyESolverPW();
    ~PyESolverPW();

    // ==================== Initialization ====================

    /// Initialize from INPUT file directory
    void initialize(const std::string& input_dir);

    /// Call before_all_runners
    void before_all_runners();

    /// Cleanup resources (calls after_all_runners)
    void cleanup();

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

    // ==================== Wave Function Access ====================

    /// Get wave function coefficients for k-point ik
    py::array_t<T> get_psi(int ik) const;

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

    /// Get number of plane waves for k-point ik
    int get_npw(int ik) const;

    /// Get maximum number of plane waves
    int get_npwx() const;

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

private:
    // Internal state
    bool initialized_ = false;
    bool scf_started_ = false;
    bool conv_esolver_ = false;
    int istep_ = 0;
    int niter_ = 0;
    double drho_ = 0.0;

    // ABACUS objects
    std::unique_ptr<UnitCell> ucell_;
    ModuleESolver::ESolver_KS_PW<T, base_device::DEVICE_CPU>* esolver_ = nullptr;
    std::string input_dir_;
    std::string output_dir_;  // absolute path to output directory

    // Output stream management
    std::ofstream ofs_running_;
    std::ofstream ofs_warning_;

    // Cached system dimensions
    int nat_ = 0;
    int ntype_ = 0;
    int nks_ = 0;
    int npwx_ = 0;
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
    void init_hardware();
    void finalize_hardware();
};

// Type aliases for common use cases
using PyESolverPW_CF = PyESolverPW<std::complex<float>>;
using PyESolverPW_CD = PyESolverPW<std::complex<double>>;

} // namespace py_esolver

#endif // PY_ESOLVER_PW_HPP
