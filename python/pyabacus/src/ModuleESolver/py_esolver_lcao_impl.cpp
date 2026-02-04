/**
 * @file py_esolver_lcao_impl.cpp
 * @brief Implementation of PyESolverLCAO class methods
 *
 * This file contains the implementation of the main PyESolverLCAO wrapper class.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "py_esolver_lcao.hpp"

// ABACUS headers for actual implementation
#include "source_estate/module_charge/charge.h"
#include "source_estate/fp_energy.h"
#include "source_estate/elecstate.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/hamilt_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_basis/module_ao/parallel_orbitals.h"

// Additional ABACUS headers for Phase 3 implementation
#include "source_esolver/esolver_ks_lcao.h"
#include "source_esolver/esolver.h"
#include "source_cell/unitcell.h"
#include "source_cell/check_atomic_stru.h"
#include "source_io/read_input.h"
#include "source_io/input_conv.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/global_variable.h"
#include "source_base/global_file.h"
#include "source_base/parallel_global.h"
#include "source_base/timer.h"
#include "source_base/memory.h"
#include "source_base/matrix.h"

#include <complex>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace py_esolver
{

// ============================================================================
// PyESolverLCAO Implementation (template)
// ============================================================================

template <typename TK, typename TR>
PyESolverLCAO<TK, TR>::PyESolverLCAO()
{
    // Constructor - initialization deferred to initialize()
}

template <typename TK, typename TR>
PyESolverLCAO<TK, TR>::~PyESolverLCAO()
{
    // Clean up ESolver (UnitCell is managed by unique_ptr)
    if (esolver_ != nullptr)
    {
        delete esolver_;
        esolver_ = nullptr;
    }
    cleanup_output_streams();
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::setup_output_streams(const std::string& output_dir)
{
    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir))
    {
        fs::create_directories(output_dir);
    }

    // Open running log file
    std::string running_log = output_dir + "/running_scf.log";
    ofs_running_.open(running_log, std::ios::out);

    // Open warning log file
    std::string warning_log = output_dir + "/warning.log";
    ofs_warning_.open(warning_log, std::ios::out);
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::cleanup_output_streams()
{
    // Close files
    if (ofs_running_.is_open())
    {
        ofs_running_.close();
    }
    if (ofs_warning_.is_open())
    {
        ofs_warning_.close();
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::cache_system_info()
{
    if (ucell_ != nullptr)
    {
        nat_ = ucell_->nat;
        ntype_ = ucell_->ntype;
        nspin_ = PARAM.inp.nspin;
    }
    if (esolver_ != nullptr)
    {
        // Get dimensions from ESolver's internal state
        nks_ = esolver_->get_kv().get_nks();
        nbands_ = PARAM.inp.nbands;
        // nbasis_ will be set after orbital setup
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::update_accessors()
{
    // Update accessors with current data from ESolver
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        // Energy accessor will be updated when get_energy() is called
        // Charge accessor will be updated when get_charge() is called
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::initialize(const std::string& input_dir)
{
    // Store input directory for later use
    input_dir_ = input_dir;

    // Change to input directory
    std::string original_dir = fs::current_path().string();
    if (!input_dir.empty() && input_dir != ".")
    {
        fs::current_path(input_dir);
    }

    // Initialize MPI parameters (MPI should already be initialized by mpi4py)
#ifdef __MPI
    int nproc = 1;
    int my_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PARAM.set_pal_param(my_rank, nproc, 1);
    GlobalV::MY_RANK = my_rank;
    GlobalV::NPROC = nproc;
#endif

    // Setup output streams
    std::string output_dir = input_dir.empty() ? "OUT.PYABACUS" : input_dir + "/OUT.PYABACUS";
    setup_output_streams(output_dir);

    // Read INPUT file
    ModuleIO::ReadInput ri(GlobalV::MY_RANK);
    ri.read_parameters(PARAM, "INPUT");

    // Initialize MPI pools (required for LCAO calculations)
#ifdef __MPI
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                PARAM.inp.bndpar,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);

    // Initialize DIAG_WORLD and GRID_WORLD (required for LCAO basis initialization)
    Parallel_Global::split_diag_world(PARAM.inp.diago_proc,
                                      GlobalV::NPROC,
                                      GlobalV::MY_RANK,
                                      GlobalV::DRANK,
                                      GlobalV::DSIZE,
                                      GlobalV::DCOLOR);
    Parallel_Global::split_grid_world(PARAM.inp.diago_proc,
                                      GlobalV::NPROC,
                                      GlobalV::MY_RANK,
                                      GlobalV::GRANK,
                                      GlobalV::GSIZE);
#endif

    // Create UnitCell
    ucell_ = std::make_unique<UnitCell>();

    // Setup UnitCell
    ucell_->setup(PARAM.inp.latname,
                  PARAM.inp.ntype,
                  PARAM.inp.lmaxmax,
                  PARAM.inp.init_vel,
                  PARAM.inp.fixed_axes);

    // Read structure file
    ucell_->setup_cell(PARAM.globalv.global_in_stru, GlobalV::ofs_running);

    // Create ESolver based on gamma_only flag
    esolver_ = new ModuleESolver::ESolver_KS_LCAO<TK, TR>();

    // Cache system information
    cache_system_info();

    // Restore original directory
    if (!input_dir.empty() && input_dir != ".")
    {
        fs::current_path(original_dir);
    }

    initialized_ = true;
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::before_all_runners()
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        esolver_->before_all_runners(*ucell_, PARAM.inp);
        cache_system_info();
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::before_scf(int istep)
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }
    istep_ = istep;
    scf_started_ = true;
    conv_esolver_ = false;
    niter_ = 0;
    drho_ = 0.0;
    diag_ethr_ = 1e-2;
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<TK>*>(esolver_);
        base_esolver->before_scf(*ucell_, istep);
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::run_scf_iteration(int iter)
{
    if (!scf_started_)
    {
        throw std::runtime_error("SCF not started. Call before_scf() first.");
    }
    niter_ = iter;
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<TK>*>(esolver_);

        // Run iter_init
        base_esolver->iter_init(*ucell_, istep_, iter);

        // Run hamilt2rho
        double ethr = base_esolver->get_diag_ethr();
        base_esolver->hamilt2rho(*ucell_, istep_, iter, ethr);

        // Run iter_finish
        base_esolver->iter_finish(*ucell_, istep_, niter_, conv_esolver_);

        // Update drho
        drho_ = base_esolver->get_drho();
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::run_scf(int max_iter)
{
    before_scf(istep_);

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        run_scf_iteration(iter);
        if (conv_esolver_)
        {
            break;
        }
    }
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::after_scf(int istep)
{
    if (!scf_started_)
    {
        throw std::runtime_error("SCF not started. Call before_scf() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<TK>*>(esolver_);
        base_esolver->after_scf(*ucell_, istep, conv_esolver_);
    }
    scf_started_ = false;
}

template <typename TK, typename TR>
PyChargeAccessor PyESolverLCAO<TK, TR>::get_charge() const
{
    PyChargeAccessor accessor;
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const Charge* chr = esolver_->get_pelec()->charge;
        accessor.set_from_charge(chr);
    }
    return accessor;
}

template <typename TK, typename TR>
PyEnergyAccessor PyESolverLCAO<TK, TR>::get_energy() const
{
    PyEnergyAccessor accessor;
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const elecstate::fenergy* f_en = &(esolver_->get_pelec()->f_en);
        accessor.set_from_fenergy(f_en);
    }
    return accessor;
}

template <typename TK, typename TR>
PyHamiltonianAccessor<TK, TR> PyESolverLCAO<TK, TR>::get_hamiltonian() const
{
    PyHamiltonianAccessor<TK, TR> accessor;
    if (esolver_ != nullptr)
    {
        // Get Hamiltonian from ESolver
        auto* p_hamilt = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(esolver_->get_p_hamilt());
        if (p_hamilt != nullptr)
        {
            accessor.set_from_hamilt(p_hamilt, nks_, &(esolver_->get_pv()));
        }
    }
    return accessor;
}

template <typename TK, typename TR>
PyDensityMatrixAccessor<TK, TR> PyESolverLCAO<TK, TR>::get_density_matrix() const
{
    PyDensityMatrixAccessor<TK, TR> accessor;
    if (esolver_ != nullptr)
    {
        // Get density matrix from ESolver's dmat
        const auto& dmat = esolver_->get_dmat();
        // Access the dm pointer directly from Setup_DM
        auto* dm = dmat.dm;
        if (dm != nullptr)
        {
            accessor.set_from_dm(dm);
        }
    }
    return accessor;
}

template <typename TK, typename TR>
py::array_t<TK> PyESolverLCAO<TK, TR>::get_psi(int ik) const
{
    if (esolver_ != nullptr)
    {
        psi::Psi<TK>* psi_ptr = esolver_->get_psi();
        if (psi_ptr != nullptr && ik >= 0 && ik < nks_)
        {
            psi_ptr->fix_k(ik);
            int nbands = psi_ptr->get_nbands();
            int nbasis = psi_ptr->get_current_nbas();

            std::vector<ssize_t> shape = {static_cast<ssize_t>(nbands), static_cast<ssize_t>(nbasis)};
            auto result = py::array_t<TK>(shape);
            auto buf = result.request();
            TK* ptr = static_cast<TK*>(buf.ptr);

            // Copy wave function coefficients
            const TK* psi_data = psi_ptr->get_pointer();
            std::copy(psi_data, psi_data + nbands * nbasis, ptr);

            return result;
        }
    }
    return py::array_t<TK>();
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_eigenvalues(int ik) const
{
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const auto& ekb = esolver_->get_pelec()->ekb;
        if (ik >= 0 && ik < nks_ && ekb.nr > 0)
        {
            int nbands = ekb.nc;
            std::vector<ssize_t> shape = {static_cast<ssize_t>(nbands)};
            auto result = py::array_t<double>(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);

            // Copy eigenvalues for k-point ik
            for (int ib = 0; ib < nbands; ++ib)
            {
                ptr[ib] = ekb(ik, ib);
            }

            return result;
        }
    }
    return py::array_t<double>();
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_occupations(int ik) const
{
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const auto& wg = esolver_->get_pelec()->wg;
        if (ik >= 0 && ik < nks_ && wg.nr > 0)
        {
            int nbands = wg.nc;
            std::vector<ssize_t> shape = {static_cast<ssize_t>(nbands)};
            auto result = py::array_t<double>(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);

            // Copy occupation numbers for k-point ik
            for (int ib = 0; ib < nbands; ++ib)
            {
                ptr[ib] = wg(ik, ib);
            }

            return result;
        }
    }
    return py::array_t<double>();
}

template <typename TK, typename TR>
int PyESolverLCAO<TK, TR>::get_nks() const
{
    if (esolver_ != nullptr)
    {
        return esolver_->get_kv().get_nks();
    }
    return nks_;
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_kvec_d(int ik) const
{
    std::vector<ssize_t> shape = {3};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    if (esolver_ != nullptr)
    {
        const auto& kv = esolver_->get_kv();
        if (ik >= 0 && ik < kv.get_nks() && static_cast<size_t>(ik) < kv.kvec_d.size())
        {
            ptr[0] = kv.kvec_d[ik].x;
            ptr[1] = kv.kvec_d[ik].y;
            ptr[2] = kv.kvec_d[ik].z;
            return result;
        }
    }

    ptr[0] = ptr[1] = ptr[2] = 0.0;
    return result;
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_wk() const
{
    if (esolver_ != nullptr)
    {
        const auto& kv = esolver_->get_kv();
        int nks = kv.get_nks();
        if (nks > 0 && kv.wk.size() >= static_cast<size_t>(nks))
        {
            std::vector<ssize_t> shape = {static_cast<ssize_t>(nks)};
            auto result = py::array_t<double>(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);

            for (int ik = 0; ik < nks; ++ik)
            {
                ptr[ik] = kv.wk[ik];
            }

            return result;
        }
    }
    return py::array_t<double>();
}

template <typename TK, typename TR>
int PyESolverLCAO<TK, TR>::get_nbasis() const
{
    if (esolver_ != nullptr)
    {
        const auto& pv = esolver_->get_pv();
        return pv.get_global_row_size();
    }
    return nbasis_;
}

template <typename TK, typename TR>
int PyESolverLCAO<TK, TR>::get_nbands() const
{
    if (esolver_ != nullptr && esolver_->get_psi() != nullptr)
    {
        return esolver_->get_psi()->get_nbands();
    }
    return nbands_;
}

template <typename TK, typename TR>
int PyESolverLCAO<TK, TR>::get_nspin() const
{
    if (esolver_ != nullptr)
    {
        return esolver_->get_kv().get_nspin();
    }
    return nspin_;
}

template <typename TK, typename TR>
int PyESolverLCAO<TK, TR>::get_nat() const
{
    if (ucell_ != nullptr)
    {
        return ucell_->nat;
    }
    return nat_;
}

// ============================================================================
// Force and Stress Methods
// ============================================================================

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::cal_force()
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        ModuleBase::matrix force;
        esolver_->cal_force(*ucell_, force);
        force_accessor_.set_from_matrix(force.c, nat_);
    }
    force_calculated_ = true;
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::cal_stress()
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        ModuleBase::matrix stress;
        esolver_->cal_stress(*ucell_, stress);
        stress_accessor_.set_from_matrix(stress.c);
    }
    stress_calculated_ = true;
}

template <typename TK, typename TR>
PyForceAccessor PyESolverLCAO<TK, TR>::get_force() const
{
    if (!force_calculated_)
    {
        throw std::runtime_error("Forces not calculated. Call cal_force() first.");
    }
    return force_accessor_;
}

template <typename TK, typename TR>
PyStressAccessor PyESolverLCAO<TK, TR>::get_stress() const
{
    if (!stress_calculated_)
    {
        throw std::runtime_error("Stress not calculated. Call cal_stress() first.");
    }
    return stress_accessor_;
}

// ============================================================================
// Position and Cell Update Methods
// ============================================================================

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::update_positions(py::array_t<double> positions)
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }

    auto buf = positions.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
    {
        throw std::runtime_error("Positions must have shape (nat, 3)");
    }

    if (ucell_ != nullptr)
    {
        double* ptr = static_cast<double*>(buf.ptr);
        int nat_input = static_cast<int>(buf.shape[0]);

        if (nat_input != nat_)
        {
            throw std::runtime_error("Number of atoms mismatch: expected " +
                                     std::to_string(nat_) + ", got " + std::to_string(nat_input));
        }

        // Convert Cartesian (Angstrom) to fractional coordinates
        for (int iat = 0; iat < nat_; ++iat)
        {
            int it = ucell_->iat2it[iat];
            int ia = ucell_->iat2ia[iat];

            // Get Cartesian position in Angstrom
            double x_ang = ptr[iat * 3 + 0];
            double y_ang = ptr[iat * 3 + 1];
            double z_ang = ptr[iat * 3 + 2];

            // Convert to Bohr
            double x_bohr = x_ang * py_esolver_constants::ANG_TO_BOHR;
            double y_bohr = y_ang * py_esolver_constants::ANG_TO_BOHR;
            double z_bohr = z_ang * py_esolver_constants::ANG_TO_BOHR;

            // Convert to fractional coordinates using inverse lattice vectors
            ModuleBase::Vector3<double> cart(x_bohr, y_bohr, z_bohr);
            ModuleBase::Vector3<double> frac = cart * ucell_->G;

            ucell_->atoms[it].tau[ia].x = frac.x;
            ucell_->atoms[it].tau[ia].y = frac.y;
            ucell_->atoms[it].tau[ia].z = frac.z;

            // Also update taud (direct coordinates)
            ucell_->atoms[it].taud[ia] = frac;
        }

        // Mark that ionic positions have been updated
        ucell_->ionic_position_updated = true;
    }

    force_calculated_ = false;
    stress_calculated_ = false;
}

template <typename TK, typename TR>
void PyESolverLCAO<TK, TR>::update_cell(py::array_t<double> cell)
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }

    auto buf = cell.request();
    if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3)
    {
        throw std::runtime_error("Cell must have shape (3, 3)");
    }

    if (ucell_ != nullptr)
    {
        double* ptr = static_cast<double*>(buf.ptr);

        // Update lattice vectors (convert from Angstrom to Bohr)
        // Matrix3 uses e11, e12, etc. instead of e[i][j]
        double* latvec_ptr = &(ucell_->latvec.e11);
        for (int i = 0; i < 9; ++i)
        {
            latvec_ptr[i] = ptr[i] * py_esolver_constants::ANG_TO_BOHR;
        }

        // Update a1, a2, a3 vectors
        ucell_->a1.x = ucell_->latvec.e11;
        ucell_->a1.y = ucell_->latvec.e12;
        ucell_->a1.z = ucell_->latvec.e13;
        ucell_->a2.x = ucell_->latvec.e21;
        ucell_->a2.y = ucell_->latvec.e22;
        ucell_->a2.z = ucell_->latvec.e23;
        ucell_->a3.x = ucell_->latvec.e31;
        ucell_->a3.y = ucell_->latvec.e32;
        ucell_->a3.z = ucell_->latvec.e33;

        // Recalculate cell volume and reciprocal lattice
        ucell_->omega = std::abs(ucell_->latvec.Det()) * ucell_->lat0 * ucell_->lat0 * ucell_->lat0;
        ucell_->GT = ucell_->latvec.Inverse();
        ucell_->G = ucell_->GT.Transpose();
        ucell_->GGT = ucell_->G * ucell_->GT;
        ucell_->invGGT = ucell_->GGT.Inverse();

        // Mark that cell parameters have been updated
        ucell_->cell_parameter_updated = true;
    }

    force_calculated_ = false;
    stress_calculated_ = false;
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_positions() const
{
    if (ucell_ != nullptr && nat_ > 0)
    {
        std::vector<ssize_t> shape = {static_cast<ssize_t>(nat_), 3};
        auto result = py::array_t<double>(shape);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (int iat = 0; iat < nat_; ++iat)
        {
            // Get fractional coordinates
            const auto& tau = ucell_->get_tau(iat);

            // Convert fractional to Cartesian (Bohr)
            ModuleBase::Vector3<double> cart = tau * ucell_->latvec;

            // Convert Bohr to Angstrom
            ptr[iat * 3 + 0] = cart.x * ucell_->lat0 * py_esolver_constants::BOHR_TO_ANG;
            ptr[iat * 3 + 1] = cart.y * ucell_->lat0 * py_esolver_constants::BOHR_TO_ANG;
            ptr[iat * 3 + 2] = cart.z * ucell_->lat0 * py_esolver_constants::BOHR_TO_ANG;
        }

        return result;
    }

    // Return empty array if not initialized
    int nat = get_nat();
    std::vector<ssize_t> shape = {static_cast<ssize_t>(nat > 0 ? nat : 1), 3};
    auto result = py::array_t<double>(shape);
    return result;
}

template <typename TK, typename TR>
py::array_t<double> PyESolverLCAO<TK, TR>::get_cell() const
{
    if (ucell_ != nullptr)
    {
        std::vector<ssize_t> shape = {3, 3};
        auto result = py::array_t<double>(shape);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        // Convert lattice vectors from Bohr to Angstrom
        // Matrix3 uses e11, e12, etc. instead of e[i][j]
        const double* latvec_ptr = &(ucell_->latvec.e11);
        for (int i = 0; i < 9; ++i)
        {
            ptr[i] = latvec_ptr[i] * ucell_->lat0 * py_esolver_constants::BOHR_TO_ANG;
        }

        return result;
    }

    // Return identity matrix if not initialized
    std::vector<ssize_t> shape = {3, 3};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::fill(ptr, ptr + 9, 0.0);
    ptr[0] = ptr[4] = ptr[8] = 1.0;  // Identity matrix
    return result;
}

// Explicit template instantiations
template class PyESolverLCAO<double, double>;
template class PyESolverLCAO<std::complex<double>, double>;

} // namespace py_esolver
