#ifndef PY_DENSITY_MATRIX_ACCESSOR_HPP
#define PY_DENSITY_MATRIX_ACCESSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

// Forward declarations
namespace elecstate {
    template <typename TK, typename TR> class DensityMatrix;
}

namespace py = pybind11;

namespace py_esolver
{

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

} // namespace py_esolver

#endif // PY_DENSITY_MATRIX_ACCESSOR_HPP
