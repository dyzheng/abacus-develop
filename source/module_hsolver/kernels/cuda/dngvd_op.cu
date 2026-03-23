#include "module_hsolver/kernels/dngvd_op.h"
#include "helper_cuda.h"
#include "module_base/memory.h"

#include <base/macros/macros.h>

#include <cusolverDn.h>
#include <vector>
#include <iostream>

namespace hsolver
{

static cusolverDnHandle_t cusolver_H = nullptr;

void createGpuSolverHandle()
{
    if (cusolver_H == nullptr)
    {
        cusolverErrcheck(cusolverDnCreate(&cusolver_H));
    }
}

void destroyGpuSolverHandle()
{
    if (cusolver_H != nullptr)
    {
        cusolverErrcheck(cusolverDnDestroy(cusolver_H));
        cusolver_H = nullptr;
    }
}

static inline
void xhegvd_wrapper(
    const cublasFillMode_t& uplo,
    const int& n,
    double* A, const int& lda,
    double* B, const int& ldb,
    double* W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int* devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double* work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Dsygvd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnDsygvd_bufferSize(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, B, ldb, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(double) * lwork, "Dsygvd::workspace");
    ModuleBase::Memory::record_gpu("Dsygvd::workspace", sizeof(double) * lwork);

    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnDsygvd(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, B, ldb, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    // free the buffer before checking info_gpu to avoid memory leak
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Dsygvd", info_gpu); }
}

static inline
void xhegvd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        std::complex<float> * B, const int& ldb,
        float * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Chegvd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnChegvd_bufferSize(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda,
                                                 reinterpret_cast<const float2 *>(B), ldb, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(float2) * lwork, "Chegvd::workspace");
    ModuleBase::Memory::record_gpu("Chegvd::workspace", sizeof(float2) * lwork);

    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnChegvd(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<float2 *>(A), lda, reinterpret_cast<float2 *>(B), ldb, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Chegvd", info_gpu); }
}

static inline
void xhegvd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        std::complex<double> * B, const int& ldb,
        double * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Zhegvd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZhegvd_bufferSize(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda,
                                                 reinterpret_cast<const double2 *>(B), ldb, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(double2) * lwork, "Zhegvd::workspace");
    ModuleBase::Memory::record_gpu("Zhegvd::workspace", sizeof(double2) * lwork);

    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZhegvd(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, reinterpret_cast<double2 *>(B), ldb, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Zhegvd", info_gpu); }
}

static inline
void xheevd_wrapper(
    const cublasFillMode_t& uplo,
    const int& n,
    double* A, const int& lda,
    double* W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int* devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double* work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Dsyevd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnDsyevd_bufferSize(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(double) * lwork, "Dsyevd::workspace");
    ModuleBase::Memory::record_gpu("Dsyevd::workspace", sizeof(double) * lwork);
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnDsyevd(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n, A, lda, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Dsyevd", info_gpu); }
}

static inline
void xheevd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        float * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Cheevd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnCheevd_bufferSize(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(float2) * lwork, "Cheevd::workspace");
    ModuleBase::Memory::record_gpu("Cheevd::workspace", sizeof(float2) * lwork);
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnCheevd(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n, reinterpret_cast<float2 *>(A), lda, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Cheevd", info_gpu); }
}

static inline
void xheevd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        double * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    cudaMallocCheck((void**)&devInfo, sizeof(int), "Zheevd::devInfo");

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZheevd_bufferSize(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda, W, &lwork));
    // allocate memery
    cudaMallocCheck((void**)&work, sizeof(double2) * lwork, "Zheevd::workspace");
    ModuleBase::Memory::record_gpu("Zheevd::workspace", sizeof(double2) * lwork);
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZheevd(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, W, work, lwork, devInfo));

    cudaErrcheck(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrcheck(cudaFree(work));
    cudaErrcheck(cudaFree(devInfo));
    if (0 != info_gpu) { throw DiagoCudaException("Zheevd", info_gpu); }
}

template <typename T>
struct dngvd_op<T, base_device::DEVICE_GPU>
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const base_device::DEVICE_GPU* d,
                    const int nstart,
                    const int ldh,
                    const T* A, // hcc
                    const T* B, // scc
                    Real* W,    // eigenvalue
                    T* V)
    {
        assert(nstart == ldh);
        // A to V
        cudaErrcheck(cudaMemcpy(V, A, sizeof(T) * ldh * nstart, cudaMemcpyDeviceToDevice));
        try
        {
            xhegvd_wrapper(CUBLAS_FILL_MODE_UPPER, nstart, V, ldh,
                (T*)B, ldh, W);
        }
        catch (const DiagoCudaException& e)
        {
            // GPU cusolver failed, try CPU LAPACK fallback
            std::cerr << "WARNING: " << e.what()
                      << ", attempting CPU LAPACK fallback for this subspace diagonalization (n="
                      << nstart << ")" << std::endl;

            const int mat_size = nstart * ldh;
            std::vector<T> h_A(mat_size);
            std::vector<T> h_B(mat_size);
            std::vector<T> h_V(mat_size);
            std::vector<Real> h_W(nstart);

            // D2H: copy original A and B from GPU to CPU
            cudaErrcheck(cudaMemcpy(h_A.data(), A, sizeof(T) * mat_size, cudaMemcpyDeviceToHost));
            cudaErrcheck(cudaMemcpy(h_B.data(), B, sizeof(T) * mat_size, cudaMemcpyDeviceToHost));

            try {
                // Call CPU LAPACK version
                base_device::DEVICE_CPU* cpu_ctx = {};
                dngvd_op<T, base_device::DEVICE_CPU>()(cpu_ctx, nstart, ldh,
                                                        h_A.data(), h_B.data(),
                                                        h_W.data(), h_V.data());

                // H2D: copy results back to GPU
                cudaErrcheck(cudaMemcpy(V, h_V.data(), sizeof(T) * mat_size, cudaMemcpyHostToDevice));
                cudaErrcheck(cudaMemcpy(W, h_W.data(), sizeof(Real) * nstart, cudaMemcpyHostToDevice));

                std::cerr << "CPU LAPACK fallback succeeded for n=" << nstart << std::endl;
            }
            catch (const std::exception& cpu_error) {
                // CPU fallback also failed, re-throw original GPU exception to trigger CG fallback
                std::cerr << "ERROR: CPU LAPACK fallback also failed: " << cpu_error.what() << std::endl;
                std::cerr << "Re-throwing original GPU exception to trigger CG fallback" << std::endl;
                throw e;  // Re-throw original GPU exception
            }
        }
    }
};

template <typename T>
struct dnevx_op<T, base_device::DEVICE_GPU>
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const base_device::DEVICE_GPU* d,
                    const int nstart,
                    const int ldh,
                    const T* A, // hcc
                    const int m,
                    Real* W, // eigenvalue
                    T* V)
    {
        assert(nstart <= ldh);
        // A to V
        cudaErrcheck(cudaMemcpy(V, A, sizeof(T) * nstart * ldh, cudaMemcpyDeviceToDevice));
        try
        {
            xheevd_wrapper(CUBLAS_FILL_MODE_LOWER, nstart, V, ldh, W);
        }
        catch (const DiagoCudaException& e)
        {
            // GPU cusolver failed, try CPU LAPACK fallback
            std::cerr << "WARNING: " << e.what()
                      << ", attempting CPU LAPACK fallback for this standard diagonalization (n="
                      << nstart << ")" << std::endl;

            const int mat_size = nstart * ldh;
            std::vector<T> h_A(mat_size);
            std::vector<T> h_V(mat_size);
            std::vector<Real> h_W(nstart);

            // D2H: copy original A from GPU to CPU
            cudaErrcheck(cudaMemcpy(h_A.data(), A, sizeof(T) * mat_size, cudaMemcpyDeviceToHost));

            try {
                // Call CPU LAPACK version
                base_device::DEVICE_CPU* cpu_ctx = {};
                dnevx_op<T, base_device::DEVICE_CPU>()(cpu_ctx, nstart, ldh,
                                                        h_A.data(), m,
                                                        h_W.data(), h_V.data());

                // H2D: copy results back to GPU
                cudaErrcheck(cudaMemcpy(V, h_V.data(), sizeof(T) * mat_size, cudaMemcpyHostToDevice));
                cudaErrcheck(cudaMemcpy(W, h_W.data(), sizeof(Real) * nstart, cudaMemcpyHostToDevice));

                std::cerr << "CPU LAPACK fallback succeeded for n=" << nstart << std::endl;
            }
            catch (const std::exception& cpu_error) {
                // CPU fallback also failed, re-throw original GPU exception to trigger CG fallback
                std::cerr << "ERROR: CPU LAPACK fallback also failed: " << cpu_error.what() << std::endl;
                std::cerr << "Re-throwing original GPU exception to trigger CG fallback" << std::endl;
                throw e;  // Re-throw original GPU exception
            }
        }
    }
};

template <typename T>
struct dngvx_op<T, base_device::DEVICE_GPU>
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const base_device::DEVICE_GPU* d,
                    const int nbase,
                    const int ldh,
                    T* hcc,
                    T* scc,
                    const int m,
                    Real* eigenvalue,
                    T* vcc)
    {

    }
};

template struct dngvd_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct dnevx_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct dngvx_op<std::complex<float>, base_device::DEVICE_GPU>;

template struct dngvd_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct dnevx_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct dngvx_op<std::complex<double>, base_device::DEVICE_GPU>;

#ifdef __LCAO
template struct dngvd_op<double, base_device::DEVICE_GPU>;
template struct dnevx_op<double, base_device::DEVICE_GPU>;
template struct dngvx_op<double, base_device::DEVICE_GPU>;
#endif

} // namespace hsolver