#ifndef PY_FORCE_STRESS_ACCESSOR_HPP
#define PY_FORCE_STRESS_ACCESSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>

namespace py = pybind11;

namespace py_esolver
{

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

} // namespace py_esolver

#endif // PY_FORCE_STRESS_ACCESSOR_HPP
