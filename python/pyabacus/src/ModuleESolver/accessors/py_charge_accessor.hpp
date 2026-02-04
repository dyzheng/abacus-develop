#ifndef PY_CHARGE_ACCESSOR_HPP
#define PY_CHARGE_ACCESSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

// Forward declarations
class Charge;

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

} // namespace py_esolver

#endif // PY_CHARGE_ACCESSOR_HPP
