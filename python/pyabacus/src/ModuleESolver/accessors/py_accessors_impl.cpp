/**
 * @file py_accessors_impl.cpp
 * @brief Implementation of accessor classes for Python bindings
 *
 * This file contains implementations for:
 * - PyForceAccessor
 * - PyStressAccessor
 * - PyChargeAccessor
 * - PyEnergyAccessor
 * - PyHamiltonianAccessor
 * - PyDensityMatrixAccessor
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "py_charge_accessor.hpp"
#include "py_force_stress_accessor.hpp"
#include "py_energy_accessor.hpp"
#include "py_hamiltonian_accessor.hpp"
#include "py_density_matrix_accessor.hpp"

// ABACUS headers
#include "source_estate/module_charge/charge.h"
#include "source_estate/fp_energy.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/hamilt_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include <complex>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

namespace py_esolver
{

// ============================================================================
// PyForceAccessor Implementation
// ============================================================================

void PyForceAccessor::set_from_matrix(const double* force_ptr, int nat)
{
    nat_ = nat;
    if (force_ptr == nullptr || nat <= 0)
    {
        forces_.clear();
        nat_ = 0;
        return;
    }

    forces_.resize(nat * 3);
    std::copy(force_ptr, force_ptr + nat * 3, forces_.begin());
}

py::array_t<double> PyForceAccessor::get_forces() const
{
    if (!is_valid())
    {
        throw std::runtime_error("Force data not available.");
    }

    std::vector<ssize_t> shape = {static_cast<ssize_t>(nat_), 3};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    std::copy(forces_.begin(), forces_.end(), ptr);

    return result;
}

// ============================================================================
// PyStressAccessor Implementation
// ============================================================================

void PyStressAccessor::set_from_matrix(const double* stress_ptr)
{
    if (stress_ptr == nullptr)
    {
        valid_ = false;
        std::fill(stress_.begin(), stress_.end(), 0.0);
        return;
    }

    std::copy(stress_ptr, stress_ptr + 9, stress_.begin());
    valid_ = true;
}

py::array_t<double> PyStressAccessor::get_stress() const
{
    if (!is_valid())
    {
        throw std::runtime_error("Stress data not available.");
    }

    std::vector<ssize_t> shape = {3, 3};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    std::copy(stress_.begin(), stress_.end(), ptr);

    return result;
}

py::array_t<double> PyStressAccessor::get_stress_voigt() const
{
    if (!is_valid())
    {
        throw std::runtime_error("Stress data not available.");
    }

    // Voigt notation: xx, yy, zz, yz, xz, xy
    std::vector<ssize_t> shape = {6};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // stress_ is stored as row-major 3x3: [0,1,2], [3,4,5], [6,7,8]
    // which corresponds to: [xx,xy,xz], [yx,yy,yz], [zx,zy,zz]
    ptr[0] = stress_[0];  // xx
    ptr[1] = stress_[4];  // yy
    ptr[2] = stress_[8];  // zz
    ptr[3] = stress_[5];  // yz
    ptr[4] = stress_[2];  // xz
    ptr[5] = stress_[1];  // xy

    return result;
}

// ============================================================================
// PyChargeAccessor Implementation
// ============================================================================

void PyChargeAccessor::set_from_charge(const Charge* chr)
{
    if (chr == nullptr)
    {
        chr_ptr_ = nullptr;
        rho_ptr_ = nullptr;
        nspin_ = 0;
        nrxx_ = 0;
        ngmc_ = 0;
        return;
    }

    chr_ptr_ = chr;
    rho_ptr_ = nullptr;  // Use chr_ptr_ instead
    nspin_ = chr->nspin;
    nrxx_ = chr->nrxx;
    ngmc_ = chr->ngmc;
}

void PyChargeAccessor::set_data(const double* rho_ptr, int nspin, int nrxx)
{
    chr_ptr_ = nullptr;  // Not using Charge object
    rho_ptr_ = rho_ptr;
    nspin_ = nspin;
    nrxx_ = nrxx;
    ngmc_ = 0;
}

py::array_t<double> PyChargeAccessor::get_rho() const
{
    if (!is_valid())
    {
        throw std::runtime_error("Charge data not available. Run SCF first.");
    }

    // Create numpy array with shape (nspin, nrxx)
    std::vector<ssize_t> shape = {static_cast<ssize_t>(nspin_), static_cast<ssize_t>(nrxx_)};

    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Copy data from either chr_ptr_ or rho_ptr_
    if (chr_ptr_ != nullptr && chr_ptr_->rho != nullptr)
    {
        // Copy from Charge object (rho is double** with shape [nspin][nrxx])
        for (int is = 0; is < nspin_; ++is)
        {
            if (chr_ptr_->rho[is] != nullptr)
            {
                std::copy(chr_ptr_->rho[is], chr_ptr_->rho[is] + nrxx_, ptr + is * nrxx_);
            }
        }
    }
    else if (rho_ptr_ != nullptr)
    {
        // Copy from flat array (legacy mode)
        std::copy(rho_ptr_, rho_ptr_ + nspin_ * nrxx_, ptr);
    }
    else
    {
        throw std::runtime_error("No valid charge data source.");
    }

    return result;
}

py::array_t<std::complex<double>> PyChargeAccessor::get_rhog() const
{
    if (chr_ptr_ == nullptr || chr_ptr_->rhog == nullptr)
    {
        throw std::runtime_error("Reciprocal-space charge density not available.");
    }

    // Create numpy array with shape (nspin, ngmc)
    std::vector<ssize_t> shape = {static_cast<ssize_t>(nspin_), static_cast<ssize_t>(ngmc_)};

    auto result = py::array_t<std::complex<double>>(shape);
    auto buf = result.request();
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);

    // Copy from Charge object (rhog is complex** with shape [nspin][ngmc])
    for (int is = 0; is < nspin_; ++is)
    {
        if (chr_ptr_->rhog[is] != nullptr)
        {
            std::copy(chr_ptr_->rhog[is], chr_ptr_->rhog[is] + ngmc_, ptr + is * ngmc_);
        }
    }

    return result;
}

py::array_t<double> PyChargeAccessor::get_rho_core() const
{
    if (chr_ptr_ == nullptr || chr_ptr_->rho_core == nullptr)
    {
        throw std::runtime_error("Core charge density not available.");
    }

    // Create numpy array with shape (nrxx,)
    std::vector<ssize_t> shape = {static_cast<ssize_t>(nrxx_)};

    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    std::copy(chr_ptr_->rho_core, chr_ptr_->rho_core + nrxx_, ptr);

    return result;
}

// ============================================================================
// PyEnergyAccessor Implementation
// ============================================================================

void PyEnergyAccessor::set_from_fenergy(const elecstate::fenergy* f_en)
{
    if (f_en == nullptr)
    {
        etot_ = 0.0;
        eband_ = 0.0;
        hartree_energy_ = 0.0;
        etxc_ = 0.0;
        ewald_energy_ = 0.0;
        demet_ = 0.0;
        exx_ = 0.0;
        evdw_ = 0.0;
        return;
    }

    etot_ = f_en->etot;
    eband_ = f_en->eband;
    hartree_energy_ = f_en->hartree_energy;
    etxc_ = f_en->etxc;
    ewald_energy_ = f_en->ewald_energy;
    demet_ = f_en->demet;
    exx_ = f_en->exx;
    evdw_ = f_en->evdw;
}

void PyEnergyAccessor::set_energies(double etot, double eband, double hartree,
                                     double etxc, double ewald, double demet,
                                     double exx, double evdw)
{
    etot_ = etot;
    eband_ = eband;
    hartree_energy_ = hartree;
    etxc_ = etxc;
    ewald_energy_ = ewald;
    demet_ = demet;
    exx_ = exx;
    evdw_ = evdw;
}

py::dict PyEnergyAccessor::get_all_energies() const
{
    py::dict result;
    result["etot"] = etot_;
    result["eband"] = eband_;
    result["hartree_energy"] = hartree_energy_;
    result["etxc"] = etxc_;
    result["ewald_energy"] = ewald_energy_;
    result["demet"] = demet_;
    result["exx"] = exx_;
    result["evdw"] = evdw_;
    return result;
}

// ============================================================================
// PyHamiltonianAccessor Implementation (template)
// ============================================================================

template <typename TK, typename TR>
void PyHamiltonianAccessor<TK, TR>::set_from_hamilt(hamilt::HamiltLCAO<TK, TR>* hamilt_lcao, int nks, const Parallel_Orbitals* pv)
{
    hamilt_ptr_ = hamilt_lcao;
    pv_ = pv;
    nks_ = nks;

    if (hamilt_lcao == nullptr)
    {
        nbasis_ = 0;
        nloc_ = 0;
        nrow_ = 0;
        ncol_ = 0;
        return;
    }

    // Get dimensions from Parallel_Orbitals if available
    if (pv != nullptr)
    {
        nrow_ = pv->get_row_size();
        ncol_ = pv->get_col_size();
        nloc_ = nrow_ * ncol_;
        nbasis_ = pv->get_global_row_size();
    }

    // Initialize pointer arrays for compatibility mode
    hk_ptrs_.resize(nks, nullptr);
    sk_ptrs_.resize(nks, nullptr);
    matrix_dims_.resize(nks, {nrow_, ncol_});
}

template <typename TK, typename TR>
void PyHamiltonianAccessor<TK, TR>::set_dimensions(int nbasis, int nks)
{
    nbasis_ = nbasis;
    nks_ = nks;
    hk_ptrs_.resize(nks, nullptr);
    sk_ptrs_.resize(nks, nullptr);
    matrix_dims_.resize(nks, {0, 0});
}

template <typename TK, typename TR>
void PyHamiltonianAccessor<TK, TR>::set_Hk_data(int ik, const TK* data, int nrow, int ncol)
{
    if (ik >= 0 && ik < nks_)
    {
        hk_ptrs_[ik] = data;
        matrix_dims_[ik] = {nrow, ncol};
    }
}

template <typename TK, typename TR>
void PyHamiltonianAccessor<TK, TR>::set_Sk_data(int ik, const TK* data, int nrow, int ncol)
{
    if (ik >= 0 && ik < nks_)
    {
        sk_ptrs_[ik] = data;
        matrix_dims_[ik] = {nrow, ncol};
    }
}

template <typename TK, typename TR>
py::array_t<TK> PyHamiltonianAccessor<TK, TR>::get_Hk(int ik) const
{
    if (!is_valid() || ik < 0 || ik >= nks_)
    {
        throw std::runtime_error("Invalid k-point index or Hamiltonian not available.");
    }

    if (hk_ptrs_[ik] == nullptr)
    {
        throw std::runtime_error("H(k) data not set for this k-point.");
    }

    auto [nrow, ncol] = matrix_dims_[ik];
    std::vector<ssize_t> shape = {nrow, ncol};

    auto result = py::array_t<TK>(shape);
    auto buf = result.request();
    TK* ptr = static_cast<TK*>(buf.ptr);

    std::copy(hk_ptrs_[ik], hk_ptrs_[ik] + nrow * ncol, ptr);

    return result;
}

template <typename TK, typename TR>
py::array_t<TK> PyHamiltonianAccessor<TK, TR>::get_Sk(int ik) const
{
    if (!is_valid() || ik < 0 || ik >= nks_)
    {
        throw std::runtime_error("Invalid k-point index or overlap matrix not available.");
    }

    if (sk_ptrs_[ik] == nullptr)
    {
        throw std::runtime_error("S(k) data not set for this k-point.");
    }

    auto [nrow, ncol] = matrix_dims_[ik];
    std::vector<ssize_t> shape = {nrow, ncol};

    auto result = py::array_t<TK>(shape);
    auto buf = result.request();
    TK* ptr = static_cast<TK*>(buf.ptr);

    std::copy(sk_ptrs_[ik], sk_ptrs_[ik] + nrow * ncol, ptr);

    return result;
}

template <typename TK, typename TR>
py::dict PyHamiltonianAccessor<TK, TR>::get_HR() const
{
    py::dict result;
    if (hamilt_ptr_ != nullptr)
    {
        // Get HR from HamiltLCAO
        auto* hR = hamilt_ptr_->getHR();
        if (hR != nullptr)
        {
            // Iterate over all atom pairs and R vectors
            int num_pairs = hR->size_atom_pairs();
            for (int ipair = 0; ipair < num_pairs; ++ipair)
            {
                auto& atom_pair = hR->get_atom_pair(ipair);
                int iat1 = atom_pair.get_atom_i();
                int iat2 = atom_pair.get_atom_j();

                // Iterate over R vectors for this atom pair
                for (int iR = 0; iR < atom_pair.get_R_size(); ++iR)
                {
                    auto R_index = atom_pair.get_R_index(iR);
                    auto matrix = atom_pair.get_HR_values(R_index.x, R_index.y, R_index.z);
                    int nrow = matrix.get_row_size();
                    int ncol = matrix.get_col_size();

                    if (nrow > 0 && ncol > 0)
                    {
                        // Create key tuple (iat1, iat2, Rx, Ry, Rz)
                        py::tuple key = py::make_tuple(iat1, iat2, R_index.x, R_index.y, R_index.z);

                        // Create numpy array for matrix data
                        std::vector<ssize_t> shape = {static_cast<ssize_t>(nrow), static_cast<ssize_t>(ncol)};
                        auto arr = py::array_t<TR>(shape);
                        auto buf = arr.request();
                        TR* ptr = static_cast<TR*>(buf.ptr);
                        std::copy(matrix.get_pointer(), matrix.get_pointer() + nrow * ncol, ptr);

                        result[key] = arr;
                    }
                }
            }
        }
    }
    return result;
}

template <typename TK, typename TR>
py::dict PyHamiltonianAccessor<TK, TR>::get_SR() const
{
    py::dict result;
    if (hamilt_ptr_ != nullptr)
    {
        // Get SR from HamiltLCAO
        auto* sR = hamilt_ptr_->getSR();
        if (sR != nullptr)
        {
            // Iterate over all atom pairs and R vectors
            int num_pairs = sR->size_atom_pairs();
            for (int ipair = 0; ipair < num_pairs; ++ipair)
            {
                auto& atom_pair = sR->get_atom_pair(ipair);
                int iat1 = atom_pair.get_atom_i();
                int iat2 = atom_pair.get_atom_j();

                // Iterate over R vectors for this atom pair
                for (int iR = 0; iR < atom_pair.get_R_size(); ++iR)
                {
                    auto R_index = atom_pair.get_R_index(iR);
                    auto matrix = atom_pair.get_HR_values(R_index.x, R_index.y, R_index.z);
                    int nrow = matrix.get_row_size();
                    int ncol = matrix.get_col_size();

                    if (nrow > 0 && ncol > 0)
                    {
                        // Create key tuple (iat1, iat2, Rx, Ry, Rz)
                        py::tuple key = py::make_tuple(iat1, iat2, R_index.x, R_index.y, R_index.z);

                        // Create numpy array for matrix data
                        std::vector<ssize_t> shape = {static_cast<ssize_t>(nrow), static_cast<ssize_t>(ncol)};
                        auto arr = py::array_t<TR>(shape);
                        auto buf = arr.request();
                        TR* ptr = static_cast<TR*>(buf.ptr);
                        std::copy(matrix.get_pointer(), matrix.get_pointer() + nrow * ncol, ptr);

                        result[key] = arr;
                    }
                }
            }
        }
    }
    return result;
}

// Explicit template instantiations
template class PyHamiltonianAccessor<double, double>;
template class PyHamiltonianAccessor<std::complex<double>, double>;

// ============================================================================
// PyDensityMatrixAccessor Implementation (template)
// ============================================================================

template <typename TK, typename TR>
void PyDensityMatrixAccessor<TK, TR>::set_from_dm(elecstate::DensityMatrix<TK, TR>* dm)
{
    dm_ptr_ = dm;

    if (dm == nullptr)
    {
        nks_ = 0;
        nrow_ = 0;
        ncol_ = 0;
        return;
    }

    nks_ = dm->get_DMK_nks();
    nrow_ = dm->get_DMK_nrow();
    ncol_ = dm->get_DMK_ncol();

    // Initialize pointer arrays for compatibility mode
    dmk_ptrs_.resize(nks_, nullptr);
}

template <typename TK, typename TR>
void PyDensityMatrixAccessor<TK, TR>::set_dimensions(int nks, int nrow, int ncol)
{
    nks_ = nks;
    nrow_ = nrow;
    ncol_ = ncol;
    dmk_ptrs_.resize(nks, nullptr);
}

template <typename TK, typename TR>
void PyDensityMatrixAccessor<TK, TR>::set_DMK_data(int ik, const TK* data)
{
    if (ik >= 0 && ik < nks_)
    {
        dmk_ptrs_[ik] = data;
    }
}

template <typename TK, typename TR>
py::array_t<TK> PyDensityMatrixAccessor<TK, TR>::get_DMK(int ik) const
{
    if (!is_valid() || ik < 0 || ik >= nks_)
    {
        throw std::runtime_error("Invalid k-point index or density matrix not available.");
    }

    if (dmk_ptrs_[ik] == nullptr)
    {
        throw std::runtime_error("DM(k) data not set for this k-point.");
    }

    std::vector<ssize_t> shape = {static_cast<ssize_t>(nrow_), static_cast<ssize_t>(ncol_)};

    auto result = py::array_t<TK>(shape);
    auto buf = result.request();
    TK* ptr = static_cast<TK*>(buf.ptr);

    std::copy(dmk_ptrs_[ik], dmk_ptrs_[ik] + nrow_ * ncol_, ptr);

    return result;
}

template <typename TK, typename TR>
std::vector<py::array_t<TK>> PyDensityMatrixAccessor<TK, TR>::get_DMK_all() const
{
    std::vector<py::array_t<TK>> result;
    for (int ik = 0; ik < nks_; ++ik)
    {
        result.push_back(get_DMK(ik));
    }
    return result;
}

template <typename TK, typename TR>
py::dict PyDensityMatrixAccessor<TK, TR>::get_DMR() const
{
    py::dict result;
    if (dm_ptr_ != nullptr)
    {
        // Get DMR from DensityMatrix
        auto* dmR = dm_ptr_->get_DMR_pointer(1);  // ispin = 1
        if (dmR != nullptr)
        {
            // Iterate over all atom pairs and R vectors
            int num_pairs = dmR->size_atom_pairs();
            for (int ipair = 0; ipair < num_pairs; ++ipair)
            {
                auto& atom_pair = dmR->get_atom_pair(ipair);
                int iat1 = atom_pair.get_atom_i();
                int iat2 = atom_pair.get_atom_j();

                // Iterate over R vectors for this atom pair
                for (int iR = 0; iR < atom_pair.get_R_size(); ++iR)
                {
                    auto R_index = atom_pair.get_R_index(iR);
                    auto matrix = atom_pair.get_HR_values(R_index.x, R_index.y, R_index.z);
                    int nrow = matrix.get_row_size();
                    int ncol = matrix.get_col_size();

                    if (nrow > 0 && ncol > 0)
                    {
                        // Create key tuple (iat1, iat2, Rx, Ry, Rz)
                        py::tuple key = py::make_tuple(iat1, iat2, R_index.x, R_index.y, R_index.z);

                        // Create numpy array for matrix data
                        std::vector<ssize_t> shape = {static_cast<ssize_t>(nrow), static_cast<ssize_t>(ncol)};
                        auto arr = py::array_t<TR>(shape);
                        auto buf = arr.request();
                        TR* ptr = static_cast<TR*>(buf.ptr);
                        std::copy(matrix.get_pointer(), matrix.get_pointer() + nrow * ncol, ptr);

                        result[key] = arr;
                    }
                }
            }
        }
    }
    return result;
}

// Explicit template instantiations
template class PyDensityMatrixAccessor<double, double>;
template class PyDensityMatrixAccessor<std::complex<double>, double>;

} // namespace py_esolver
