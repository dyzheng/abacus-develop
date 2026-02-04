#ifndef PY_HAMILTONIAN_ACCESSOR_HPP
#define PY_HAMILTONIAN_ACCESSOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <utility>

// Forward declarations
class Parallel_Orbitals;
namespace hamilt {
    template <typename TK, typename TR> class HamiltLCAO;
}

namespace py = pybind11;

namespace py_esolver
{

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

} // namespace py_esolver

#endif // PY_HAMILTONIAN_ACCESSOR_HPP
