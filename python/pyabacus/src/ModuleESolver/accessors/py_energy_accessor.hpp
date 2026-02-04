#ifndef PY_ENERGY_ACCESSOR_HPP
#define PY_ENERGY_ACCESSOR_HPP

#include <pybind11/pybind11.h>

// Forward declarations
namespace elecstate {
    struct fenergy;
}

namespace py = pybind11;

namespace py_esolver
{

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

} // namespace py_esolver

#endif // PY_ENERGY_ACCESSOR_HPP
