/**
 * @file py_esolver_pw_impl.cpp
 * @brief Implementation of PyESolverPW class methods
 *
 * This file contains the implementation of the main PyESolverPW wrapper class
 * for plane wave calculations.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "py_esolver_pw.hpp"
#include "py_esolver_lcao.hpp"  // For py_esolver_constants

// ABACUS headers for actual implementation
#include "source_estate/module_charge/charge.h"
#include "source_estate/fp_energy.h"
#include "source_estate/elecstate.h"

// Additional ABACUS headers for PW implementation
#include "source_esolver/esolver_ks_pw.h"
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
#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_hsolver/kernels/hegvd_op.h"
#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>
#ifdef __DSP
#include "source_base/kernels/dsp/dsp_connector.h"
#endif

#include <complex>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace py_esolver
{

// ============================================================================
// PyESolverPW Implementation (template)
// ============================================================================

template <typename T>
PyESolverPW<T>::PyESolverPW()
{
    // Constructor - initialization deferred to initialize()
}

template <typename T>
PyESolverPW<T>::~PyESolverPW()
{
    // Clean up ESolver if not already cleaned by cleanup()
    if (esolver_ != nullptr)
    {
        delete esolver_;
        esolver_ = nullptr;
    }
    // UnitCell is managed by unique_ptr, auto-destroyed
    cleanup_output_streams();
}

template <typename T>
void PyESolverPW<T>::setup_output_streams(const std::string& output_dir)
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

template <typename T>
void PyESolverPW<T>::cleanup_output_streams()
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

template <typename T>
void PyESolverPW<T>::cache_system_info()
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
    }
}

template <typename T>
void PyESolverPW<T>::initialize(const std::string& input_dir)
{
    // Store input directory for later use
    input_dir_ = input_dir;

    // Change to input directory
    std::string original_dir = fs::current_path().string();
    if (!input_dir.empty() && input_dir != ".")
    {
        fs::current_path(input_dir);
    }

    std::cout << "[PyABACUS] Initializing PW ESolver..." << std::endl;
    std::cout << "[PyABACUS] Input directory: " << fs::current_path().string() << std::endl;

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

    // Read INPUT file
    std::cout << "[PyABACUS] Resetting PARAM to defaults and reading INPUT..." << std::endl;
    ModuleIO::ReadInput ri(GlobalV::MY_RANK);
    ri.read_parameters(PARAM, "INPUT");

    // Create output directory and open GlobalV::ofs_running / ofs_warning
    // (matching Driver::reading() step 2)
    ri.create_directory(PARAM);

    // Write INPUT.info (matching Driver::reading() step 3)
    std::string info_file = PARAM.globalv.global_out_dir + "INPUT.info";
    ri.write_parameters(PARAM, info_file);

    // CRITICAL: Call Input_Conv::Convert() to set global variables
    // (matching Driver::reading() step 4)
    Input_Conv::Convert();

    // Initialize timer (matching Driver::init())
    ModuleBase::timer::start();

    // Compute absolute output directory path for diagnostic output
    std::string suffix = PARAM.inp.suffix;
    output_dir_ = fs::current_path().string() + "/OUT." + suffix;

    std::cout << "[PyABACUS] Configuration: basis_type=" << PARAM.inp.basis_type
              << ", calculation=" << PARAM.inp.calculation
              << ", nspin=" << PARAM.inp.nspin
              << ", gamma_only=" << PARAM.inp.gamma_only << std::endl;
    std::cout << "[PyABACUS] STRU file: " << PARAM.globalv.global_in_stru << std::endl;
    std::cout << "[PyABACUS] Output directory: " << output_dir_ << std::endl;

    // Initialize MPI pools (required for PW calculations)
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

    // Initialize DIAG_WORLD and GRID_WORLD
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
    unitcell::check_atomic_stru(*ucell_, PARAM.inp.min_dist_coef);

    // Initialize hardware (GPU/DSP)
    init_hardware();

    // Create ESolver for plane wave calculations
    esolver_ = new ModuleESolver::ESolver_KS_PW<T, base_device::DEVICE_CPU>();

    // Cache system information
    cache_system_info();

    // Restore original directory
    if (!input_dir.empty() && input_dir != ".")
    {
        fs::current_path(original_dir);
    }

    initialized_ = true;
}

template <typename T>
void PyESolverPW<T>::before_all_runners()
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Change to input directory so ABACUS can find KPT and other files
        std::string original_dir = fs::current_path().string();
        if (!input_dir_.empty() && input_dir_ != ".")
        {
            fs::current_path(input_dir_);
        }

        esolver_->before_all_runners(*ucell_, PARAM.inp);
        cache_system_info();

        // Restore original directory
        if (!input_dir_.empty() && input_dir_ != ".")
        {
            fs::current_path(original_dir);
        }
    }
}

template <typename T>
void PyESolverPW<T>::cleanup()
{
    std::cout << "[PyABACUS] Cleaning up PW ESolver..." << std::endl;

    // 1. Finalize ESolver computation (writes output files, etc.)
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        esolver_->after_all_runners(*ucell_);
    }

    // 2. Delete ESolver and UnitCell to release ALL owned resources
    //    (K_Vectors, Hamiltonian, Charge, psi, PW_Basis, etc.)
    //    This must happen while timer is still active (destructors may call timer::tick)
    if (esolver_ != nullptr)
    {
        delete esolver_;
        esolver_ = nullptr;
    }
    ucell_.reset();

    // 3. Finalize hardware (GPU/DSP handles)
    finalize_hardware();

    // 4. Clear timer pool for next run (skip timer::finish / Memory::finish
    //    which produce NaN output when ofs_running is already closed)
    ModuleBase::timer::timer_pool.clear();

    // 5. Close GlobalV log files
    ModuleBase::Global_File::close_all_log(GlobalV::MY_RANK,
                                           PARAM.inp.out_alllog,
                                           PARAM.inp.calculation);

    // 6. Free MPI communicators created by init_pools/split_diag_world/split_grid_world
    //    They will be recreated in the next initialize() call.
#ifdef __MPI
    if (initialized_)
    {
        if (POOL_WORLD != MPI_COMM_NULL)  { MPI_Comm_free(&POOL_WORLD); }
        if (KP_WORLD != MPI_COMM_NULL)    { MPI_Comm_free(&KP_WORLD); }
        if (INT_BGROUP != MPI_COMM_NULL)  { MPI_Comm_free(&INT_BGROUP); }
        if (BP_WORLD != MPI_COMM_NULL)    { MPI_Comm_free(&BP_WORLD); }
        if (GRID_WORLD != MPI_COMM_NULL)  { MPI_Comm_free(&GRID_WORLD); }
        if (DIAG_WORLD != MPI_COMM_NULL)  { MPI_Comm_free(&DIAG_WORLD); }
    }
#endif

    // 7. Reset member state for potential reuse or clean destruction
    initialized_ = false;
    scf_started_ = false;
    conv_esolver_ = false;
    force_calculated_ = false;
    stress_calculated_ = false;
    nat_ = 0;
    ntype_ = 0;
    nks_ = 0;
    npwx_ = 0;
    nbands_ = 0;
    nspin_ = 1;

    std::cout << "[PyABACUS] Cleanup complete. Output log: "
              << output_dir_ << "/running_scf.log" << std::endl;
}

template <typename T>
void PyESolverPW<T>::init_hardware()
{
#if ((defined __CUDA) || (defined __ROCM))
    if (PARAM.inp.device == "gpu")
    {
        ModuleBase::createGpuBlasHandle();
        hsolver::createGpuSolverHandle();
        container::kernels::createGpuBlasHandle();
        container::kernels::createGpuSolverHandle();
    }
#endif

#ifdef __DSP
    if (GlobalV::NPROC > PARAM.inp.kpar)
    {
        ModuleBase::WARNING_QUIT(
            "PyESolverPW::init_hardware",
            "Number of processors must be equal to KPAR for DSP hardware initialization.");
    }
    std::cout << " ** Initializing DSP Hardware..." << std::endl;
    mtfunc::dspInitHandle(GlobalV::MY_RANK % PARAM.inp.dsp_count);
#endif
}

template <typename T>
void PyESolverPW<T>::finalize_hardware()
{
#if defined(__CUDA) || defined(__ROCM)
    if (PARAM.inp.device == "gpu")
    {
        ModuleBase::destoryBLAShandle();
        hsolver::destroyGpuSolverHandle();
        container::kernels::destroyGpuBlasHandle();
        container::kernels::destroyGpuSolverHandle();
    }
#endif

#ifdef __DSP
    std::cout << " ** Closing DSP Hardware..." << std::endl;
    mtfunc::dspDestoryHandle(GlobalV::MY_RANK);
#endif
}

template <typename T>
void PyESolverPW<T>::before_scf(int istep)
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
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<T>*>(esolver_);

        // Initialize diag_ethr in ESolver (same as runner() does before SCF loop)
        base_esolver->set_diag_ethr(PARAM.inp.pw_diag_thr);

        base_esolver->before_scf(*ucell_, istep);
    }
}

template <typename T>
void PyESolverPW<T>::run_scf_iteration(int iter)
{
    if (!scf_started_)
    {
        throw std::runtime_error("SCF not started. Call before_scf() first.");
    }
    niter_ = iter;
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<T>*>(esolver_);

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

template <typename T>
void PyESolverPW<T>::run_scf(int max_iter)
{
    if (!initialized_)
    {
        throw std::runtime_error("ESolver not initialized. Call initialize() first.");
    }

    std::cout << "[PyABACUS] Starting SCF (PW, max_iter=" << max_iter << ")" << std::endl;

    // Use base class pointer to access public methods
    auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<T>*>(esolver_);

    // 1) before_scf
    before_scf(istep_);

    // 2) SCF loop with oscillation check (matching ESolver_KS::runner())
    niter_ = max_iter;
    base_esolver->set_scf_nmax_flag(false);

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Set scf_nmax_flag on last iteration
        if (iter == max_iter)
        {
            base_esolver->set_scf_nmax_flag(true);
        }

        run_scf_iteration(iter);

        // Check convergence OR oscillation (matching ESolver_KS::runner())
        if (conv_esolver_ || base_esolver->get_oscillate_esolver())
        {
            niter_ = iter;
            if (base_esolver->get_oscillate_esolver())
            {
                std::cout << " !! Density oscillation is found, STOP HERE !!" << std::endl;
            }
            break;
        }
    }

    // 3) Always call after_scf (matching ESolver_KS::runner())
    after_scf(istep_);

    // 4) Print correct total energy via cal_energy() (in Ry, same as ESolver_KS_PW::cal_energy)
    double etot_ry = esolver_->cal_energy();
    std::cout << "[PyABACUS] SCF finished: converged=" << (conv_esolver_ ? "yes" : "no")
              << ", niter=" << niter_
              << ", etot=" << std::fixed << std::setprecision(10) << etot_ry << " Ry"
              << " (" << etot_ry * py_esolver_constants::RY_TO_EV << " eV)" << std::endl;
}

template <typename T>
void PyESolverPW<T>::after_scf(int istep)
{
    if (!scf_started_)
    {
        throw std::runtime_error("SCF not started. Call before_scf() first.");
    }
    if (esolver_ != nullptr && ucell_ != nullptr)
    {
        // Use base class pointer to access public methods
        auto* base_esolver = static_cast<ModuleESolver::ESolver_KS<T>*>(esolver_);
        base_esolver->after_scf(*ucell_, istep, conv_esolver_);
    }
    scf_started_ = false;
}

template <typename T>
PyChargeAccessor PyESolverPW<T>::get_charge() const
{
    PyChargeAccessor accessor;
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const Charge* chr = esolver_->get_pelec()->charge;
        accessor.set_from_charge(chr);
    }
    return accessor;
}

template <typename T>
PyEnergyAccessor PyESolverPW<T>::get_energy() const
{
    PyEnergyAccessor accessor;
    if (esolver_ != nullptr && esolver_->get_pelec() != nullptr)
    {
        const elecstate::fenergy* f_en = &(esolver_->get_pelec()->f_en);
        accessor.set_from_fenergy(f_en);
    }
    return accessor;
}

template <typename T>
py::array_t<T> PyESolverPW<T>::get_psi(int ik) const
{
    if (esolver_ != nullptr)
    {
        psi::Psi<T>* psi_ptr = esolver_->get_psi();
        if (psi_ptr != nullptr && ik >= 0 && ik < nks_)
        {
            psi_ptr->fix_k(ik);
            int nbands = psi_ptr->get_nbands();
            int nbasis = psi_ptr->get_current_nbas();

            std::vector<ssize_t> shape = {static_cast<ssize_t>(nbands), static_cast<ssize_t>(nbasis)};
            auto result = py::array_t<T>(shape);
            auto buf = result.request();
            T* ptr = static_cast<T*>(buf.ptr);

            // Copy wave function coefficients
            const T* psi_data = psi_ptr->get_pointer();
            std::copy(psi_data, psi_data + nbands * nbasis, ptr);

            return result;
        }
    }
    return py::array_t<T>();
}

template <typename T>
py::array_t<double> PyESolverPW<T>::get_eigenvalues(int ik) const
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

template <typename T>
py::array_t<double> PyESolverPW<T>::get_occupations(int ik) const
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

template <typename T>
int PyESolverPW<T>::get_nks() const
{
    if (esolver_ != nullptr)
    {
        return esolver_->get_kv().get_nks();
    }
    return nks_;
}

template <typename T>
py::array_t<double> PyESolverPW<T>::get_kvec_d(int ik) const
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

template <typename T>
py::array_t<double> PyESolverPW<T>::get_wk() const
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

template <typename T>
int PyESolverPW<T>::get_npw(int ik) const
{
    if (esolver_ != nullptr)
    {
        psi::Psi<T>* psi_ptr = esolver_->get_psi();
        if (psi_ptr != nullptr && ik >= 0 && ik < nks_)
        {
            psi_ptr->fix_k(ik);
            return psi_ptr->get_current_nbas();
        }
    }
    return 0;
}

template <typename T>
int PyESolverPW<T>::get_npwx() const
{
    if (esolver_ != nullptr)
    {
        psi::Psi<T>* psi_ptr = esolver_->get_psi();
        if (psi_ptr != nullptr)
        {
            return psi_ptr->get_nbasis();
        }
    }
    return npwx_;
}

template <typename T>
int PyESolverPW<T>::get_nbands() const
{
    if (esolver_ != nullptr && esolver_->get_psi() != nullptr)
    {
        return esolver_->get_psi()->get_nbands();
    }
    return nbands_;
}

template <typename T>
int PyESolverPW<T>::get_nspin() const
{
    if (esolver_ != nullptr)
    {
        return esolver_->get_kv().get_nspin();
    }
    return nspin_;
}

template <typename T>
int PyESolverPW<T>::get_nat() const
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

template <typename T>
void PyESolverPW<T>::cal_force()
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

template <typename T>
void PyESolverPW<T>::cal_stress()
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

template <typename T>
PyForceAccessor PyESolverPW<T>::get_force() const
{
    if (!force_calculated_)
    {
        throw std::runtime_error("Forces not calculated. Call cal_force() first.");
    }
    return force_accessor_;
}

template <typename T>
PyStressAccessor PyESolverPW<T>::get_stress() const
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

template <typename T>
void PyESolverPW<T>::update_positions(py::array_t<double> positions)
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

template <typename T>
void PyESolverPW<T>::update_cell(py::array_t<double> cell)
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

template <typename T>
py::array_t<double> PyESolverPW<T>::get_positions() const
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

template <typename T>
py::array_t<double> PyESolverPW<T>::get_cell() const
{
    if (ucell_ != nullptr)
    {
        std::vector<ssize_t> shape = {3, 3};
        auto result = py::array_t<double>(shape);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        // Convert lattice vectors from Bohr to Angstrom
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

// Explicit template instantiations for PW (complex types only)
template class PyESolverPW<std::complex<float>>;
template class PyESolverPW<std::complex<double>>;

} // namespace py_esolver
