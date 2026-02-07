/**
 * @file py_esolver_bindings.cpp
 * @brief Pybind11 bindings for ESolver classes
 *
 * This file contains all pybind11 binding definitions for:
 * - Accessor classes (Charge, Energy, Force, Stress, Hamiltonian, DensityMatrix)
 * - ESolverLCAO classes (gamma and multi-k variants)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include "py_esolver_lcao.hpp"
#include "py_esolver_pw.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// ============================================================================
// Accessor Bindings
// ============================================================================

void bind_charge_accessor(py::module& m)
{
    py::class_<py_esolver::PyChargeAccessor>(m, "ChargeAccessor",
        R"pbdoc(
        Accessor for charge density data.

        Provides access to real-space charge density (rho) and related quantities.
        )pbdoc")
        .def(py::init<>())
        .def("get_rho", &py_esolver::PyChargeAccessor::get_rho,
            R"pbdoc(
            Get real-space charge density as numpy array.

            Returns
            -------
            numpy.ndarray
                Charge density with shape (nspin, nrxx)
            )pbdoc")
        .def("get_rhog", &py_esolver::PyChargeAccessor::get_rhog,
            R"pbdoc(
            Get reciprocal-space charge density as numpy array.

            Returns
            -------
            numpy.ndarray
                Charge density in G-space with shape (nspin, ngmc)
            )pbdoc")
        .def("get_rho_core", &py_esolver::PyChargeAccessor::get_rho_core,
            R"pbdoc(
            Get core charge density as numpy array.

            Returns
            -------
            numpy.ndarray
                Core charge density with shape (nrxx,)
            )pbdoc")
        .def_property_readonly("nspin", &py_esolver::PyChargeAccessor::get_nspin,
            "Number of spin channels")
        .def_property_readonly("nrxx", &py_esolver::PyChargeAccessor::get_nrxx,
            "Number of real-space grid points")
        .def_property_readonly("ngmc", &py_esolver::PyChargeAccessor::get_ngmc,
            "Number of G-vectors for charge density")
        .def("is_valid", &py_esolver::PyChargeAccessor::is_valid,
            "Check if charge data is available");
}

void bind_energy_accessor(py::module& m)
{
    py::class_<py_esolver::PyEnergyAccessor>(m, "EnergyAccessor",
        R"pbdoc(
        Accessor for energy data.

        Provides access to various energy components from the calculation.
        All energies are in Rydberg units.
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("etot", &py_esolver::PyEnergyAccessor::get_etot,
            "Total energy (Ry)")
        .def_property_readonly("eband", &py_esolver::PyEnergyAccessor::get_eband,
            "Band energy (Ry)")
        .def_property_readonly("hartree_energy", &py_esolver::PyEnergyAccessor::get_hartree_energy,
            "Hartree energy (Ry)")
        .def_property_readonly("etxc", &py_esolver::PyEnergyAccessor::get_etxc,
            "Exchange-correlation energy (Ry)")
        .def_property_readonly("ewald_energy", &py_esolver::PyEnergyAccessor::get_ewald_energy,
            "Ewald energy (Ry)")
        .def_property_readonly("demet", &py_esolver::PyEnergyAccessor::get_demet,
            "-TS term for metals (Ry)")
        .def_property_readonly("exx", &py_esolver::PyEnergyAccessor::get_exx,
            "Exact exchange energy (Ry)")
        .def_property_readonly("evdw", &py_esolver::PyEnergyAccessor::get_evdw,
            "van der Waals energy (Ry)")
        .def("get_all_energies", &py_esolver::PyEnergyAccessor::get_all_energies,
            "Get all energies as a dictionary");
}

void bind_force_accessor(py::module& m)
{
    py::class_<py_esolver::PyForceAccessor>(m, "ForceAccessor",
        R"pbdoc(
        Accessor for force data.

        Provides access to atomic forces in Ry/Bohr units.
        )pbdoc")
        .def(py::init<>())
        .def("get_forces", &py_esolver::PyForceAccessor::get_forces,
            R"pbdoc(
            Get forces as numpy array.

            Returns
            -------
            numpy.ndarray
                Forces with shape (nat, 3) in Ry/Bohr
            )pbdoc")
        .def_property_readonly("nat", &py_esolver::PyForceAccessor::get_nat,
            "Number of atoms")
        .def("is_valid", &py_esolver::PyForceAccessor::is_valid,
            "Check if force data is available");
}

void bind_stress_accessor(py::module& m)
{
    py::class_<py_esolver::PyStressAccessor>(m, "StressAccessor",
        R"pbdoc(
        Accessor for stress tensor data.

        Provides access to stress tensor in kbar units.
        )pbdoc")
        .def(py::init<>())
        .def("get_stress", &py_esolver::PyStressAccessor::get_stress,
            R"pbdoc(
            Get stress tensor as numpy array.

            Returns
            -------
            numpy.ndarray
                Stress tensor with shape (3, 3) in kbar
            )pbdoc")
        .def("get_stress_voigt", &py_esolver::PyStressAccessor::get_stress_voigt,
            R"pbdoc(
            Get stress in Voigt notation.

            Returns
            -------
            numpy.ndarray
                Stress in Voigt notation (6,): xx, yy, zz, yz, xz, xy
            )pbdoc")
        .def("is_valid", &py_esolver::PyStressAccessor::is_valid,
            "Check if stress data is available");
}

template <typename TK>
void bind_hamiltonian_accessor(py::module& m, const std::string& suffix)
{
    using HamiltAccessor = py_esolver::PyHamiltonianAccessor<TK>;

    std::string class_name = "HamiltonianAccessor" + suffix;

    py::class_<HamiltAccessor>(m, class_name.c_str(),
        R"pbdoc(
        Accessor for Hamiltonian matrix data.

        Provides access to H(k), S(k), H(R), and S(R) matrices.
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("nbasis", &HamiltAccessor::get_nbasis,
            "Number of basis functions")
        .def_property_readonly("nks", &HamiltAccessor::get_nks,
            "Number of k-points")
        .def("get_Hk", &HamiltAccessor::get_Hk,
            R"pbdoc(
            Get H(k) matrix for specific k-point.

            Parameters
            ----------
            ik : int
                K-point index

            Returns
            -------
            numpy.ndarray
                Hamiltonian matrix at k-point ik
            )pbdoc", "ik"_a)
        .def("get_Sk", &HamiltAccessor::get_Sk,
            R"pbdoc(
            Get S(k) overlap matrix for specific k-point.

            Parameters
            ----------
            ik : int
                K-point index

            Returns
            -------
            numpy.ndarray
                Overlap matrix at k-point ik
            )pbdoc", "ik"_a)
        .def("get_HR", &HamiltAccessor::get_HR,
            "Get H(R) in sparse format")
        .def("get_SR", &HamiltAccessor::get_SR,
            "Get S(R) in sparse format")
        .def("is_valid", &HamiltAccessor::is_valid,
            "Check if Hamiltonian data is available");
}

template <typename TK>
void bind_density_matrix_accessor(py::module& m, const std::string& suffix)
{
    using DMAccessor = py_esolver::PyDensityMatrixAccessor<TK>;

    std::string class_name = "DensityMatrixAccessor" + suffix;

    py::class_<DMAccessor>(m, class_name.c_str(),
        R"pbdoc(
        Accessor for density matrix data.

        Provides access to DM(k) and DM(R) matrices.
        )pbdoc")
        .def(py::init<>())
        .def_property_readonly("nks", &DMAccessor::get_nks,
            "Number of k-points")
        .def_property_readonly("nrow", &DMAccessor::get_nrow,
            "Number of rows in density matrix")
        .def_property_readonly("ncol", &DMAccessor::get_ncol,
            "Number of columns in density matrix")
        .def("get_DMK", &DMAccessor::get_DMK,
            R"pbdoc(
            Get DM(k) for specific k-point.

            Parameters
            ----------
            ik : int
                K-point index

            Returns
            -------
            numpy.ndarray
                Density matrix at k-point ik
            )pbdoc", "ik"_a)
        .def("get_DMK_all", &DMAccessor::get_DMK_all,
            "Get all DM(k) matrices as a list")
        .def("get_DMR", &DMAccessor::get_DMR,
            "Get DM(R) in sparse format")
        .def("is_valid", &DMAccessor::is_valid,
            "Check if density matrix data is available");
}

// ============================================================================
// ESolver Bindings
// ============================================================================

template <typename TK, typename TR>
void bind_esolver_lcao(py::module& m, const std::string& suffix)
{
    using ESolver = py_esolver::PyESolverLCAO<TK, TR>;

    std::string class_name = "ESolverLCAO" + suffix;

    py::class_<ESolver>(m, class_name.c_str(),
        R"pbdoc(
        Python wrapper for ESolver_KS_LCAO.

        This class provides a Python interface for LCAO calculations
        with support for breakpoints and state inspection during SCF.

        Example
        -------
        >>> esolver = ESolverLCAO_gamma()
        >>> esolver.initialize("./")
        >>> esolver.before_all_runners()
        >>> esolver.before_scf(0)
        >>> for iter in range(1, 101):
        ...     esolver.run_scf_iteration(iter)
        ...     energy = esolver.get_energy()
        ...     print(f"Iter {iter}: E = {energy.etot}")
        ...     if esolver.is_converged():
        ...         break
        >>> # Breakpoint before after_scf - inspect state here
        >>> charge = esolver.get_charge()
        >>> hamiltonian = esolver.get_hamiltonian()
        >>> esolver.after_scf(0)
        )pbdoc")
        .def(py::init<>())

        // Initialization
        .def("initialize", &ESolver::initialize,
            R"pbdoc(
            Initialize ESolver from INPUT file.

            Parameters
            ----------
            input_dir : str
                Directory containing INPUT, STRU, and other input files
            )pbdoc", "input_dir"_a)
        .def("before_all_runners", &ESolver::before_all_runners,
            "Initialize calculation environment")

        // SCF Control
        .def("before_scf", &ESolver::before_scf,
            R"pbdoc(
            Prepare for SCF calculation.

            Parameters
            ----------
            istep : int, optional
                Ion step index (default: 0)
            )pbdoc", "istep"_a = 0)
        .def("run_scf_iteration", &ESolver::run_scf_iteration,
            R"pbdoc(
            Run a single SCF iteration.

            Parameters
            ----------
            iter : int
                Iteration number (1-based)
            )pbdoc", "iter"_a)
        .def("run_scf", &ESolver::run_scf,
            R"pbdoc(
            Run complete SCF loop.

            Parameters
            ----------
            max_iter : int, optional
                Maximum number of iterations (default: 100)
            )pbdoc", "max_iter"_a = 100)
        .def("after_scf", &ESolver::after_scf,
            R"pbdoc(
            Finalize SCF calculation.

            Parameters
            ----------
            istep : int, optional
                Ion step index (default: 0)
            )pbdoc", "istep"_a = 0)

        // Status
        .def("is_converged", &ESolver::is_converged,
            "Check if SCF is converged")
        .def("is_oscillating", &ESolver::is_oscillating,
            "Check if density oscillation is detected")
        .def_property_readonly("niter", &ESolver::get_niter,
            "Current iteration number")
        .def_property_readonly("drho", &ESolver::get_drho,
            "Charge density difference")
        .def_property_readonly("istep", &ESolver::get_istep,
            "Current ion step")

        // Data Accessors
        .def("get_charge", &ESolver::get_charge,
            "Get charge density accessor")
        .def("get_energy", &ESolver::get_energy,
            "Get energy accessor")
        .def("get_hamiltonian", &ESolver::get_hamiltonian,
            "Get Hamiltonian accessor")
        .def("get_density_matrix", &ESolver::get_density_matrix,
            "Get density matrix accessor")

        // Wave functions
        .def("get_psi", &ESolver::get_psi,
            "Get wave function coefficients for k-point ik", "ik"_a)
        .def("get_eigenvalues", &ESolver::get_eigenvalues,
            "Get eigenvalues for k-point ik", "ik"_a)
        .def("get_occupations", &ESolver::get_occupations,
            "Get occupation numbers for k-point ik", "ik"_a)

        // K-points
        .def_property_readonly("nks", &ESolver::get_nks,
            "Number of k-points")
        .def("get_kvec_d", &ESolver::get_kvec_d,
            "Get k-vector in direct coordinates", "ik"_a)
        .def("get_wk", &ESolver::get_wk,
            "Get k-point weights")

        // System info
        .def_property_readonly("nbasis", &ESolver::get_nbasis,
            "Number of basis functions")
        .def_property_readonly("nbands", &ESolver::get_nbands,
            "Number of bands")
        .def_property_readonly("nspin", &ESolver::get_nspin,
            "Number of spin channels")
        .def_property_readonly("nat", &ESolver::get_nat,
            "Number of atoms")

        // Force and stress
        .def("cal_force", &ESolver::cal_force,
            "Calculate forces on atoms")
        .def("cal_stress", &ESolver::cal_stress,
            "Calculate stress tensor")
        .def("get_force", &ESolver::get_force,
            "Get force accessor (call cal_force first)")
        .def("get_stress", &ESolver::get_stress,
            "Get stress accessor (call cal_stress first)")

        // Position and cell update
        .def("update_positions", &ESolver::update_positions,
            R"pbdoc(
            Update atomic positions.

            Parameters
            ----------
            positions : numpy.ndarray
                Atomic positions with shape (nat, 3) in Angstrom
            )pbdoc", "positions"_a)
        .def("update_cell", &ESolver::update_cell,
            R"pbdoc(
            Update cell vectors.

            Parameters
            ----------
            cell : numpy.ndarray
                Cell vectors with shape (3, 3) in Angstrom
            )pbdoc", "cell"_a)
        .def("get_positions", &ESolver::get_positions,
            "Get atomic positions in Angstrom")
        .def("get_cell", &ESolver::get_cell,
            "Get cell vectors in Angstrom")
        .def("cleanup", &ESolver::cleanup,
            "Cleanup resources (calls after_all_runners)");
}

// ============================================================================
// ESolver PW Bindings
// ============================================================================

template <typename T>
void bind_esolver_pw(py::module& m, const std::string& suffix)
{
    using ESolver = py_esolver::PyESolverPW<T>;

    std::string class_name = "ESolverPW" + suffix;

    py::class_<ESolver>(m, class_name.c_str(),
        R"pbdoc(
        Python wrapper for ESolver_KS_PW.

        This class provides a Python interface for plane wave calculations
        with support for breakpoints and state inspection during SCF.

        Example
        -------
        >>> esolver = ESolverPW_cd()
        >>> esolver.initialize("./")
        >>> esolver.before_all_runners()
        >>> esolver.run_scf(100)
        >>> energy = esolver.get_energy()
        >>> print(f"Total energy: {energy.etot}")
        )pbdoc")
        .def(py::init<>())

        // Initialization
        .def("initialize", &ESolver::initialize,
            R"pbdoc(
            Initialize ESolver from INPUT file.

            Parameters
            ----------
            input_dir : str
                Directory containing INPUT, STRU, and other input files
            )pbdoc", "input_dir"_a)
        .def("before_all_runners", &ESolver::before_all_runners,
            "Initialize calculation environment")
        .def("cleanup", &ESolver::cleanup,
            "Cleanup resources (calls after_all_runners)")

        // SCF Control
        .def("before_scf", &ESolver::before_scf,
            R"pbdoc(
            Prepare for SCF calculation.

            Parameters
            ----------
            istep : int, optional
                Ion step index (default: 0)
            )pbdoc", "istep"_a = 0)
        .def("run_scf_iteration", &ESolver::run_scf_iteration,
            R"pbdoc(
            Run a single SCF iteration.

            Parameters
            ----------
            iter : int
                Iteration number (1-based)
            )pbdoc", "iter"_a)
        .def("run_scf", &ESolver::run_scf,
            R"pbdoc(
            Run complete SCF loop.

            Parameters
            ----------
            max_iter : int, optional
                Maximum number of iterations (default: 100)
            )pbdoc", "max_iter"_a = 100)
        .def("after_scf", &ESolver::after_scf,
            R"pbdoc(
            Finalize SCF calculation.

            Parameters
            ----------
            istep : int, optional
                Ion step index (default: 0)
            )pbdoc", "istep"_a = 0)

        // Status
        .def("is_converged", &ESolver::is_converged,
            "Check if SCF is converged")
        .def_property_readonly("niter", &ESolver::get_niter,
            "Current iteration number")
        .def_property_readonly("drho", &ESolver::get_drho,
            "Charge density difference")
        .def_property_readonly("istep", &ESolver::get_istep,
            "Current ion step")

        // Data Accessors
        .def("get_charge", &ESolver::get_charge,
            "Get charge density accessor")
        .def("get_energy", &ESolver::get_energy,
            "Get energy accessor")

        // Wave functions
        .def("get_psi", &ESolver::get_psi,
            "Get wave function coefficients for k-point ik", "ik"_a)
        .def("get_eigenvalues", &ESolver::get_eigenvalues,
            "Get eigenvalues for k-point ik", "ik"_a)
        .def("get_occupations", &ESolver::get_occupations,
            "Get occupation numbers for k-point ik", "ik"_a)

        // K-points
        .def_property_readonly("nks", &ESolver::get_nks,
            "Number of k-points")
        .def("get_kvec_d", &ESolver::get_kvec_d,
            "Get k-vector in direct coordinates", "ik"_a)
        .def("get_wk", &ESolver::get_wk,
            "Get k-point weights")

        // System info
        .def("get_npw", &ESolver::get_npw,
            "Get number of plane waves for k-point ik", "ik"_a)
        .def_property_readonly("npwx", &ESolver::get_npwx,
            "Maximum number of plane waves")
        .def_property_readonly("nbands", &ESolver::get_nbands,
            "Number of bands")
        .def_property_readonly("nspin", &ESolver::get_nspin,
            "Number of spin channels")
        .def_property_readonly("nat", &ESolver::get_nat,
            "Number of atoms")

        // Force and stress
        .def("cal_force", &ESolver::cal_force,
            "Calculate forces on atoms")
        .def("cal_stress", &ESolver::cal_stress,
            "Calculate stress tensor")
        .def("get_force", &ESolver::get_force,
            "Get force accessor (call cal_force first)")
        .def("get_stress", &ESolver::get_stress,
            "Get stress accessor (call cal_stress first)")

        // Position and cell update
        .def("update_positions", &ESolver::update_positions,
            R"pbdoc(
            Update atomic positions.

            Parameters
            ----------
            positions : numpy.ndarray
                Atomic positions with shape (nat, 3) in Angstrom
            )pbdoc", "positions"_a)
        .def("update_cell", &ESolver::update_cell,
            R"pbdoc(
            Update cell vectors.

            Parameters
            ----------
            cell : numpy.ndarray
                Cell vectors with shape (3, 3) in Angstrom
            )pbdoc", "cell"_a)
        .def("get_positions", &ESolver::get_positions,
            "Get atomic positions in Angstrom")
        .def("get_cell", &ESolver::get_cell,
            "Get cell vectors in Angstrom");
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_esolver_pack, m)
{
    m.doc() = R"pbdoc(
        PyABACUS ESolver Module
        -----------------------

        This module provides Python bindings for ABACUS ESolver classes,
        enabling Python-controlled SCF workflows with breakpoint support.

        Main Classes
        ------------
        ESolverLCAO_gamma : ESolver for gamma-only LCAO calculations
        ESolverLCAO_multi_k : ESolver for multi-k LCAO calculations
        ESolverPW_cf : ESolver for plane wave calculations (single precision)
        ESolverPW_cd : ESolver for plane wave calculations (double precision)

        Accessor Classes
        ----------------
        ChargeAccessor : Access charge density data
        EnergyAccessor : Access energy components
        HamiltonianAccessor_gamma/multi_k : Access Hamiltonian matrices (LCAO only)
        DensityMatrixAccessor_gamma/multi_k : Access density matrices (LCAO only)

        Example (LCAO)
        --------------
        >>> from pyabacus.esolver import ESolverLCAO_gamma
        >>> esolver = ESolverLCAO_gamma()
        >>> esolver.initialize("./")
        >>> esolver.before_all_runners()
        >>> esolver.run_scf(100)
        >>> energy = esolver.get_energy()
        >>> esolver.cleanup()

        Example (PW)
        ------------
        >>> from pyabacus.esolver import ESolverPW_cd
        >>> esolver = ESolverPW_cd()
        >>> esolver.initialize("./")
        >>> esolver.before_all_runners()
        >>> esolver.run_scf(100)
        >>> energy = esolver.get_energy()
        >>> esolver.cleanup()
    )pbdoc";

    // Bind accessor classes
    bind_charge_accessor(m);
    bind_energy_accessor(m);
    bind_force_accessor(m);
    bind_stress_accessor(m);
    bind_hamiltonian_accessor<double>(m, "_gamma");
    bind_hamiltonian_accessor<std::complex<double>>(m, "_multi_k");
    bind_density_matrix_accessor<double>(m, "_gamma");
    bind_density_matrix_accessor<std::complex<double>>(m, "_multi_k");

    // Bind LCAO ESolver classes
    bind_esolver_lcao<double, double>(m, "_gamma");
    bind_esolver_lcao<std::complex<double>, double>(m, "_multi_k");

    // Bind PW ESolver classes
    bind_esolver_pw<std::complex<float>>(m, "_cf");
    bind_esolver_pw<std::complex<double>>(m, "_cd");
}
