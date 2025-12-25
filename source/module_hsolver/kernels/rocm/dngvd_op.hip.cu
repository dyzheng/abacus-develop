#include "module_hsolver/kernels/dngvd_op.h"

#include <hip/hip_runtime.h>
#include <base/macros/macros.h>

namespace hsolver {

    // Initialize GPU solver handle (stub implementation)
    void createGpuSolverHandle() {
        return;
    }

    // Destroy GPU solver handle (stub implementation)
    void destroyGpuSolverHandle() {
        return;
    }

#ifdef __LCAO
    // GPU implementation of generalized eigenvalue solver for double precision
    template <>
    void dngvd_op<double, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nstart,
        const int ldh,
        const double* _hcc,
        const double* _scc,
        double* _eigenvalue,
        double* _vcc,
        int* fail_info)
    {
        // Allocate host memory for matrices
        std::vector<double> hcc(ldh * nstart, 0.0);
        std::vector<double> scc(ldh * nstart, 0.0);
        std::vector<double> vcc(ldh * nstart, 0.0);
        std::vector<double> eigenvalue(nstart, 0);
        
        // Copy Hamiltonian matrix from device to host
        hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(double) * hcc.size(), hipMemcpyDeviceToHost));
        
        // Copy overlap matrix from device to host
        hipErrcheck(hipMemcpy(scc.data(), _scc, sizeof(double) * scc.size(), hipMemcpyDeviceToHost));
        
        base_device::DEVICE_CPU* cpu_ctx = {};
        
        // Call CPU version to solve generalized eigenvalue problem
        dngvd_op<double, base_device::DEVICE_CPU>()(
            cpu_ctx,
            nstart,
            ldh,
            hcc.data(),
            scc.data(),
            eigenvalue.data(),
            vcc.data(),
            fail_info);
        
        // Copy eigenvectors back to device
        hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(double) * vcc.size(), hipMemcpyHostToDevice));
        
        // Copy eigenvalues back to device
<<<<<<< HEAD
        hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(doubl int* fail_info)
=======
        hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
    }
#endif // __LCAO

    // GPU implementation of generalized eigenvalue solver for single precision complex
    template <>
    void dngvd_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nstart,
        const int ldh,
        const std::complex<float>* _hcc,
        const std::complex<float>* _scc,
        float* _eigenvalue,
        std::complex<float>* _vcc,
        int* fail_info)
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
    {
        // Allocate host memory for complex matrices
        std::vector<std::complex<float>> hcc(ldh * nstart, {0, 0});
        std::vector<std::complex<float>> scc(ldh * nstart, {0, 0});
        std::vector<std::complex<float>> vcc(ldh * nstart, {0, 0});
        std::vector<float> eigenvalue(nstart, 0);
        
        // Copy complex Hamiltonian matrix from device to host
<<<<<<< HEAD
        hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<float>) * hcc.size(), hipMemcpyDevice     eigenvalue.data(),
=======
        hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<float>) * hcc.size(), hipMemcpyDeviceToHost));
        
        // Copy complex overlap matrix from device to host
        hipErrcheck(hipMemcpy(scc.data(), _scc, sizeof(std::complex<float>) * scc.size(), hipMemcpyDeviceToHost));
        
        base_device::DEVICE_CPU* cpu_ctx = {};
        
        // Call CPU solver for complex single precision
        dngvd_op<std::complex<float>, base_device::DEVICE_CPU>()(
            cpu_ctx,
            nstart,
            ldh,
            hcc.data(),
            scc.data(),
            eigenvalue.data(),
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
            vcc.data(),
            fail_info);
        
        // Copy complex eigenvectors back to device
        hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<float>) * vcc.size(), hipMemcpyHostToDevice));
        
        // Copy eigenvalues back to device
        hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(float) * eigenvalue.size(), hipMemcpyHostToDevice));
    }

<<<<<<< HEAD
    // GPU implementation of generalized eigenvalue solver for double precision compt, {0, 0});
=======
    // GPU implementation of generalized eigenvalue solver for double precision complex
    template <>
    void dngvd_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nstart,
        const int ldh,
        const std::complex<double>* _hcc,
        const std::complex<double>* _scc,
        double* _eigenvalue,
        std::complex<double>* _vcc,
        int* fail_info)
    {
        // Allocate host memory for double complex matrices
        std::vector<std::complex<double>> hcc(ldh * nstart, {0, 0});
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        std::vector<std::complex<double>> scc(ldh * nstart, {0, 0});
        std::vector<std::complex<double>> vcc(ldh * nstart, {0, 0});
        std::vector<double> eigenvalue(nstart, 0);
        
        // Copy double complex Hamiltonian matrix from device to host
        hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<double>) * hcc.size(), hipMemcpyDeviceToHost));
        
        // Copy double complex overlap matrix from device to host
<<<<<<< HEAD
        hipErrcheck(hipMemcpy(scc.data(), _scc,  eigenvectors back to device
=======
        hipErrcheck(hipMemcpy( eigenvectors back to device
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<double>) * vcc.size(), hipMemcpyHostToDevice));
        
        // Copy eigenvalues back to device
        hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
    }

#ifdef __LCAO
    // GPU implementation of standard eigenvalue solver for double precision
    template <>
    void dnevx_op<double, base_device::DEVICE_GPU>::operator()(
<<<<<<< HEAD
        const base_device:: _hcc, sizeof(double) * hcc.size(), hipMemcpyDeviceToHost));
=======
        c _hcc, sizeof(double) * hcc.size(), hipMemcpyDeviceToHost));
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        
        base_device::DEVICE_CPU* cpu_ctx = {};
        
        // Call CPU standard eigenvalue solver
        dnevx_op<double, base_device::DEVICE_CPU>()(cpu_ctx, nstart, ldh, hcc.data(), m, eigenvalue.data(), vcc.data());
        
        // Copy eigenvectors back to device
        hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(double) * vcc.size(), hipMemcpyHostToDevice));
        
        // Copy eigenvalues back to device
<<<<<<< HEAD
        hipErrcheck(hipMemcpy(envalue,
=======
        hipEenvalue,
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        std::complex<float>* _vcc)
    {
        // Allocate host memory for complex matrices
        std::vector<std::complex<float>> hcc(ldh * ldh, {0, 0});
        std::vector<std::complex<float>> vcc(ldh * ldh, {0, 0});
        std::vector<float> eigenvalue(ldh, 0);
        
        // Copy complex Hamiltonian matrix from device to host
        hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<float>) * hcc.size(), hipMemcpyDeviceToHost));
        
<<<<<<< HEAD
        base_device::DEVICE_CPU* cpu_ctx = {};
        
     // Copy eigenvalues back to device
=======
        base_device::DEVICE_CPU* cpu_ctx = // Copy eigenvalues back to device
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(float) * eigenvalue.size(), hipMemcpyHostToDevice));
    }

    // GPU implementation of standard eigenvalue solver for double precision complex
    template <>
    void dnevx_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nstart,
        const int ldh,
        const std::complex<double>* _hcc,
        const int m,
<<<<<<< HEAD
        double* _eigenvalue,
        std::cot));
=======
        double* _eigenvalut));
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        
        base_device::DEVICE_CPU* cpu_ctx = {};
        
        // Call CPU solver for complex double precision
        dnevx_op<std::complex<double>, base_device::DEVICE_CPU>()(
            cpu_ctx,
            nstart,
            ldh,
            hcc.data(),
            m,
            eigenvalue.data(),
            vcc.data());
        
        // Copy double complex eigenvectors back to device
        hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<double>) * vcc.size(), hipMemcpyHostToDevice));
<<<<<<< HEAD
        
        //lex<float>* hcc,
=======
 lex<float>* hcc,
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        std::complex<float>* scc,
        const int m,
        float* eigenvalue,
        std::complex<float>* vcc,
        int* fail_info)
    {
    }

    // Stub implementation for generalized eigenvalue solver (double precision complex)
    template <>
    void dngvx_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* d,
        const int nbase,
        const int ldh,
        std::complex<double>* hcc,
        std::complex<double>* scc,
        const int m,
<<<<<<< HEAD
        double* eigenvle* vcc,
=======
    le* vcc,
>>>>>>> 8c430373269e0a744ed2fda32b97d3b3df6a0188
        int* fail_info)
    {
    }
#endif // __LCAO

} // namespace hsolver
