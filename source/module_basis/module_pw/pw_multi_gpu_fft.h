#ifndef PW_MULTI_GPU_FFT_H
#define PW_MULTI_GPU_FFT_H

/**
 * @file pw_multi_gpu_fft.h
 * @brief Asynchronous GPU-parallel 3D FFT via CPU-staging MPI.
 *
 * Data layout convention (matches CPU gatherp_scatters/gathers_scatterp):
 *
 *   Before MPI (real space, per-process):
 *     in: (nplane, ny, nx)   — z-slabs owned by this rank
 *
 *   After fftxy + gather_sticks:
 *     h_mpi_send: (nplane, nstot) — indexed [iz_local * nstot + istot]
 *
 *   MPI_Alltoallv send/recv counts (matching PW_Basis convention):
 *     send: numr[ip] = numz[rank] * nst_per[ip]   startr[ip]
 *     recv: numg[ip] = nst       * numz[ip]        startg[ip]
 *
 *   After Alltoallv raw recv layout: concatenation of blocks
 *     block[ip] = (nst, numz[ip])  values from process ip
 *     i.e. h_mpi_recv[startg[ip] + is * numz[ip] + izip]
 *          = stick is, local z-index izip on process ip
 *
 *   After rearrange (mirrors CPU gatherp_scatters loop):
 *     h_fftz_in: (nst, nz)  — indexed [is * nz + iz_global]
 *     where iz_global = startz[ip] + izip
 *
 *   This is the layout consumed by cufftPlanMany (batch=nst, dist=nz).
 */

// ---- Debug control macros ----
// Define FFT_DBG_NUMERICAL to enable per-phase numerical checksums.
// Prints L2-norm and max|val| at each pipeline stage so you can compare
// GPU vs CPU and pinpoint where values diverge.
//
// Define FFT_DBG_SEGFAULT to re-enable the old entry/exit trace prints.
//
// Neither is defined by default; enable via -DFFT_DBG_NUMERICAL in cmake.
// #define FFT_DBG_NUMERICAL
// #define FFT_DBG_SEGFAULT

#if defined(__CUDA) || defined(__ROCM)
#ifdef __CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#define gpuStream_t         cudaStream_t
#define gpuEvent_t          cudaEvent_t
#define gpuMalloc           cudaMalloc
#define gpuFree             cudaFree
#define gpuMallocHost       cudaMallocHost
#define gpuFreeHost         cudaFreeHost
#define gpuMemcpyAsync      cudaMemcpyAsync
#define gpuMemcpyD2H        cudaMemcpyDeviceToHost
#define gpuMemcpyH2D        cudaMemcpyHostToDevice
#define gpuMemcpyD2D        cudaMemcpyDeviceToDevice
#define gpuStreamCreate     cudaStreamCreate
#define gpuStreamDestroy    cudaStreamDestroy
#define gpuEventCreate(e)   cudaEventCreate(e)
#define gpuEventDestroy(e)  cudaEventDestroy(e)
#define gpuEventRecord(e,s) cudaEventRecord(e,s)
#define gpuStreamWaitEvent(s,e,f) cudaStreamWaitEvent(s,e,f)
#define gpuEventSynchronize(e)    cudaEventSynchronize(e)
#define gpuStreamSynchronize(s)   cudaStreamSynchronize(s)
#else // __ROCM
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#define gpuStream_t         hipStream_t
#define gpuEvent_t          hipEvent_t
#define gpuMalloc           hipMalloc
#define gpuFree             hipFree
#define gpuMallocHost       hipMallocHost
#define gpuFreeHost         hipFreeHost
#define gpuMemcpyAsync      hipMemcpyAsync
#define gpuMemcpyD2H        hipMemcpyDeviceToHost
#define gpuMemcpyH2D        hipMemcpyHostToDevice
#define gpuMemcpyD2D        hipMemcpyDeviceToDevice
#define gpuStreamCreate     hipStreamCreate
#define gpuStreamDestroy    hipStreamDestroy
#define gpuEventCreate(e)   hipEventCreate(e)
#define gpuEventDestroy(e)  hipEventDestroy(e)
#define gpuEventRecord(e,s) hipEventRecord(e,s)
#define gpuStreamWaitEvent(s,e,f) hipStreamWaitEvent(s,e,f)
#define gpuEventSynchronize(e)    hipEventSynchronize(e)
#define gpuStreamSynchronize(s)   hipStreamSynchronize(s)
#endif // __CUDA / __ROCM

#ifdef __MPI
#include <mpi.h>
#endif

#include <complex>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>

namespace ModulePW
{

// ---- Numerical debug helpers (host-side, header-only) ----
#ifdef FFT_DBG_NUMERICAL
template <typename T>
static void fft_dbg_host_summary(const char* label, int rank,
                                 const std::complex<T>* buf, size_t n)
{
    double sum2 = 0.0, maxv = 0.0;
    size_t nz_count = 0;
    for (size_t i = 0; i < n; ++i)
    {
        double re = static_cast<double>(buf[i].real());
        double im = static_cast<double>(buf[i].imag());
        double a2 = re * re + im * im;
        sum2 += a2;
        if (a2 > maxv) maxv = a2;
        if (a2 == 0.0) ++nz_count;
    }
    printf("[FFT_NUM rank=%d] %s: n=%zu  L2=%.8e  max|z|=%.8e  zeros=%zu\n",
           rank, label, n, std::sqrt(sum2), std::sqrt(maxv), nz_count);
}

template <typename T>
static void fft_dbg_dev_summary(const char* label, int rank,
                                const std::complex<T>* d_buf, size_t n)
{
    std::vector<std::complex<T>> h(n);
    gpuMemcpyAsync(h.data(), d_buf, n * sizeof(std::complex<T>),
                   gpuMemcpyD2H, nullptr);
    gpuStreamSynchronize(nullptr);
    fft_dbg_host_summary(label, rank, h.data(), n);
}
#define FFT_DBG_HOST(label, rank, buf, n) fft_dbg_host_summary(label, rank, buf, n)
#define FFT_DBG_DEV(label, rank, buf, n)  fft_dbg_dev_summary(label, rank, buf, n)
#else
#define FFT_DBG_HOST(label, rank, buf, n) ((void)0)
#define FFT_DBG_DEV(label, rank, buf, n)  ((void)0)
#endif

/**
 * @brief RAII context for multi-GPU FFT pipeline resources.
 */
template <typename FPTYPE>
class MultiGpuFftContext
{
public:
    MultiGpuFftContext() = default;
    ~MultiGpuFftContext() { release(); }

    MultiGpuFftContext(const MultiGpuFftContext&) = delete;
    MultiGpuFftContext& operator=(const MultiGpuFftContext&) = delete;

    void init(int nx, int ny, int nz,
              int nplane, int nst, int nstot, int poolnproc,
              int chunk_sz = 16)
    {
        nx_       = nx;    ny_  = ny;    nz_  = nz;
        nplane_   = nplane; nst_ = nst;  nstot_ = nstot;
        nproc_    = poolnproc;
        chunk_    = std::min(chunk_sz, nplane);
        int stick_chunk = std::min(chunk_sz, nst);

        const int xy_chunk_elems = chunk_  * nx * ny;
        const int  z_chunk_elems = stick_chunk * nz;

        for (int b = 0; b < 2; ++b)
        {
            gpuMalloc(reinterpret_cast<void**>(&d_xy_chunk[b]),
                      xy_chunk_elems * sizeof(std::complex<FPTYPE>));
        }
        for (int b = 0; b < 2; ++b)
        {
            gpuMalloc(reinterpret_cast<void**>(&d_z_chunk[b]),
                      z_chunk_elems * sizeof(std::complex<FPTYPE>));
        }
        gpuMalloc(reinterpret_cast<void**>(&d_sticks),
                  (size_t)nst * nz * sizeof(std::complex<FPTYPE>));
        gpuMalloc(reinterpret_cast<void**>(&d_transpose_buf),
                  (size_t)nplane * nx * ny * sizeof(std::complex<FPTYPE>));

        for (int b = 0; b < 2; ++b)
        {
            gpuMallocHost(reinterpret_cast<void**>(&h_d2h_buf[b]),
                          xy_chunk_elems * sizeof(std::complex<FPTYPE>));
            gpuMallocHost(reinterpret_cast<void**>(&h_h2d_buf[b]),
                          z_chunk_elems  * sizeof(std::complex<FPTYPE>));
        }

        gpuMallocHost(reinterpret_cast<void**>(&h_mpi_send),
                      (size_t)nplane * nstot * sizeof(std::complex<FPTYPE>));
        gpuMallocHost(reinterpret_cast<void**>(&h_mpi_recv),
                      (size_t)nst * nz * sizeof(std::complex<FPTYPE>));
        gpuMallocHost(reinterpret_cast<void**>(&h_fftz_in),
                      (size_t)nst * nz * sizeof(std::complex<FPTYPE>));
        gpuMallocHost(reinterpret_cast<void**>(&h_mpi_send_bac),
                      (size_t)nst * nz * sizeof(std::complex<FPTYPE>));
        gpuMallocHost(reinterpret_cast<void**>(&h_mpi_recv_bac),
                      (size_t)nplane * nstot * sizeof(std::complex<FPTYPE>));

        for (int b = 0; b < 2; ++b)
        {
            gpuStreamCreate(&compute_stream[b]);
            gpuStreamCreate(&d2h_stream[b]);
            gpuStreamCreate(&h2d_stream[b]);
            gpuEventCreate(&fft_done[b]);
            gpuEventCreate(&d2h_done[b]);
            gpuEventCreate(&h2d_done[b]);
            gpuEventCreate(&fftz_done[b]);
        }

        initialised_ = true;
    }

    // ------------------------------------------------------------------
    // Forward: real-space → reciprocal-space
    // ------------------------------------------------------------------
    template <typename FftXY, typename FftZ>
    void fft_forward(const std::complex<FPTYPE>* in,
                     FftXY   fft_xy,
                     FftZ    fft_z,
                     const int* istot2ixy,
                     const int* startz,
                     const int* numz,
                     const int* numr,
                     const int* numg,
                     const int* startr,
                     const int* startg,
                     MPI_Comm   comm,
                     std::complex<FPTYPE>* d_result)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
#ifdef FFT_DBG_SEGFAULT
        if (rank == 0) printf("DEBUG: fft_forward entry, nplane=%d, nst=%d, nstot=%d, chunk=%d\n",
                              nplane_, nst_, nstot_, chunk_);
#endif
        assert(initialised_);
        const int nxy = nx_ * ny_;

        FFT_DBG_DEV("fwd:input(nplane*nxy)", rank, in, (size_t)nplane_ * nxy);

        // =============================================================
        // Phase 1: GPU fftxy + async D2H, double-buffered over z-chunks
        // =============================================================
        const int num_chunks = (nplane_ + chunk_ - 1) / chunk_;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            const int b       = chunk % 2;
            const int z_off   = chunk * chunk_;
            const int z_count = std::min(chunk_, nplane_ - z_off);
            const int elems   = z_count * nxy;

            if (chunk >= 2)
                gpuEventSynchronize(d2h_done[(chunk - 2) % 2]);

            gpuMemcpyAsync(d_xy_chunk[b],
                           in + (size_t)z_off * nxy,
                           elems * sizeof(std::complex<FPTYPE>),
                           gpuMemcpyD2D, compute_stream[b]);

            fft_xy(d_xy_chunk[b], d_xy_chunk[b], compute_stream[b]);
            gpuEventRecord(fft_done[b], compute_stream[b]);

            gpuStreamWaitEvent(d2h_stream[b], fft_done[b], 0);
            gpuMemcpyAsync(h_d2h_buf[b], d_xy_chunk[b],
                           elems * sizeof(std::complex<FPTYPE>),
                           gpuMemcpyD2H, d2h_stream[b]);
            gpuEventRecord(d2h_done[b], d2h_stream[b]);
        }

        for (int b = 0; b < 2; ++b)
            gpuStreamSynchronize(d2h_stream[b]);

        // Assemble h_mpi_send in (nstot, nplane) layout to match CPU
        // gatherp_scatters convention: h_mpi_send[istot * nplane + iz]
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            const int b       = chunk % 2;
            const int z_off   = chunk * chunk_;
            const int z_count = std::min(chunk_, nplane_ - z_off);

            for (int iz = 0; iz < z_count; ++iz)
            {
                const std::complex<FPTYPE>* src =
                    h_d2h_buf[b] + (size_t)iz * nxy;
                for (int istot = 0; istot < nstot_; ++istot)
                    h_mpi_send[istot * nplane_ + (z_off + iz)]
                        = src[istot2ixy[istot]];
            }
        }

        FFT_DBG_HOST("fwd:after_fftxy+gather(h_mpi_send)", rank,
                      h_mpi_send, (size_t)nplane_ * nstot_);

#ifdef FFT_DBG_NUMERICAL
        // Verify send buffer per-process segments
        for (int ip = 0; ip < nproc_; ++ip) {
            fft_dbg_host_summary(
                (std::string("fwd:send_seg[") + std::to_string(ip) + "]").c_str(),
                rank, h_mpi_send + startr[ip], numr[ip]);
        }
#endif

        // =============================================================
        // Phase 2: MPI_Alltoallv
        // =============================================================
#ifdef FFT_DBG_NUMERICAL
        {
            printf("[FFT_NUM rank=%d] fwd:alltoallv params: nst=%d nstot=%d nz=%d nplane=%d\n",
                   rank, nst_, nstot_, nz_, nplane_);
            for (int ip = 0; ip < nproc_; ++ip)
                printf("[FFT_NUM rank=%d]   ip=%d numr=%d startr=%d numg=%d startg=%d\n",
                       rank, ip, numr[ip], startr[ip], numg[ip], startg[ip]);
        }
#endif
        {
#ifdef FFT_DBG_SEGFAULT
            if (rank == 0) printf("DEBUG: before MPI_Alltoallv\n");
#endif
            const MPI_Datatype mpi_type = (sizeof(FPTYPE) == 4)
                                              ? MPI_COMPLEX
                                              : MPI_DOUBLE_COMPLEX;
            MPI_Alltoallv(h_mpi_send, numr, startr, mpi_type,
                          h_mpi_recv, numg, startg, mpi_type,
                          comm);
#ifdef FFT_DBG_SEGFAULT
            if (rank == 0) printf("DEBUG: after MPI_Alltoallv\n");
#endif
        }

        FFT_DBG_HOST("fwd:after_alltoallv(h_mpi_recv)", rank,
                      h_mpi_recv, (size_t)nst_ * nz_);

        // ---------------------------------------------------------------
        // Rearrange recv buffer → h_fftz_in (nst, nz)
        // ---------------------------------------------------------------
        std::memset(h_fftz_in, 0, (size_t)nst_ * nz_ * sizeof(std::complex<FPTYPE>));
        for (int ip = 0; ip < nproc_; ++ip)
        {
            const int nzip = numz[ip];
            for (int is = 0; is < nst_; ++is)
            {
                const std::complex<FPTYPE>* src =
                    h_mpi_recv + startg[ip] + (size_t)is * nzip;
                std::complex<FPTYPE>* dst =
                    h_fftz_in + (size_t)is * nz_ + startz[ip];
                std::memcpy(dst, src, nzip * sizeof(std::complex<FPTYPE>));
            }
        }

        FFT_DBG_HOST("fwd:after_rearrange(h_fftz_in)", rank,
                      h_fftz_in, (size_t)nst_ * nz_);

        // =============================================================
        // Phase 3: H2D + GPU fftz (single batch)
        // =============================================================
#ifdef FFT_DBG_SEGFAULT
        if (rank == 0) printf("DEBUG: Phase 3 fftz start, nst=%d\n", nst_);
#endif
        {
            const size_t total_elems = (size_t)nst_ * nz_;
            gpuMemcpyAsync(d_sticks, h_fftz_in,
                      total_elems * sizeof(std::complex<FPTYPE>),
                      gpuMemcpyH2D, compute_stream[0]);
            gpuStreamSynchronize(compute_stream[0]);
            fft_z(d_sticks, d_sticks, compute_stream[0]);
            gpuStreamSynchronize(compute_stream[0]);
        }

        FFT_DBG_DEV("fwd:after_fftz(d_sticks)", rank,
                     d_sticks, (size_t)nst_ * nz_);

        // Copy d_sticks → d_result
        if (d_result != d_sticks)
        {
            gpuMemcpyAsync(d_result, d_sticks,
                      (size_t)nst_ * nz_ * sizeof(std::complex<FPTYPE>),
                      gpuMemcpyD2D, compute_stream[0]);
            gpuStreamSynchronize(compute_stream[0]);
        }
#ifdef FFT_DBG_SEGFAULT
        if (rank == 0) printf("DEBUG: fft_forward complete\n");
#endif
    }

    // ------------------------------------------------------------------
    // Backward: reciprocal-space → real-space
    // ------------------------------------------------------------------
    template <typename FftXY, typename FftZ>
    void fft_backward(const std::complex<FPTYPE>* in_sticks,
                      std::complex<FPTYPE>*       d_out,
                      FftXY   fft_xy_bac,
                      FftZ    fft_z_bac,
                      const int* istot2ixy,
                      const int* startz,
                      const int* numz,
                      const int* numr,
                      const int* numg,
                      const int* startr,
                      const int* startg,
                      MPI_Comm   comm)
    {
        assert(initialised_);
        const int nxy = nx_ * ny_;
        int rank;
        MPI_Comm_rank(comm, &rank);

        FFT_DBG_DEV("bac:input(in_sticks)", rank, in_sticks, (size_t)nst_ * nz_);

        // =============================================================
        // Phase 1: GPU fftz_bac (single batch) + D2H
        // =============================================================
        {
            const size_t total_elems = (size_t)nst_ * nz_;
            gpuMemcpyAsync(d_sticks, in_sticks,
                      total_elems * sizeof(std::complex<FPTYPE>),
                      gpuMemcpyD2D, compute_stream[0]);
            fft_z_bac(d_sticks, d_sticks, compute_stream[0]);
            gpuStreamSynchronize(compute_stream[0]);

            FFT_DBG_DEV("bac:after_fftz(d_sticks)", rank,
                         d_sticks, total_elems);

            gpuMemcpyAsync(h_mpi_send_bac, d_sticks,
                      total_elems * sizeof(std::complex<FPTYPE>),
                      gpuMemcpyD2H, compute_stream[0]);
            gpuStreamSynchronize(compute_stream[0]);
        }

        FFT_DBG_HOST("bac:after_fftz_d2h(h_mpi_send_bac)", rank,
                      h_mpi_send_bac, (size_t)nst_ * nz_);

        // ---------------------------------------------------------------
        // Inverse rearrange: (nst, nz) → Alltoallv send layout
        // ---------------------------------------------------------------
        std::memcpy(h_fftz_in, h_mpi_send_bac,
                    (size_t)nst_ * nz_ * sizeof(std::complex<FPTYPE>));
        std::memset(h_mpi_send_bac, 0,
                    (size_t)nst_ * nz_ * sizeof(std::complex<FPTYPE>));

        for (int ip = 0; ip < nproc_; ++ip)
        {
            const int nzip = numz[ip];
            for (int is = 0; is < nst_; ++is)
            {
                const std::complex<FPTYPE>* src =
                    h_fftz_in + (size_t)is * nz_ + startz[ip];
                std::complex<FPTYPE>* dst =
                    h_mpi_send_bac + startg[ip] + (size_t)is * nzip;
                std::memcpy(dst, src, nzip * sizeof(std::complex<FPTYPE>));
            }
        }

        FFT_DBG_HOST("bac:after_inv_rearrange(h_mpi_send_bac)", rank,
                      h_mpi_send_bac, (size_t)nst_ * nz_);

#ifdef FFT_DBG_NUMERICAL
        // Verify send buffer per-process segments
        for (int ip = 0; ip < nproc_; ++ip) {
            fft_dbg_host_summary(
                (std::string("bac:send_seg[") + std::to_string(ip) + "]").c_str(),
                rank, h_mpi_send_bac + startg[ip], numg[ip]);
        }
#endif

        // =============================================================
        // Phase 2: MPI_Alltoallv (inverse)
        // =============================================================
#ifdef FFT_DBG_NUMERICAL
        {
            printf("[FFT_NUM rank=%d] bac:alltoallv params: nst=%d nstot=%d nz=%d nplane=%d\n",
                   rank, nst_, nstot_, nz_, nplane_);
            for (int ip = 0; ip < nproc_; ++ip)
                printf("[FFT_NUM rank=%d]   ip=%d numg=%d startg=%d numr=%d startr=%d numz=%d startz=%d\n",
                       rank, ip, numg[ip], startg[ip], numr[ip], startr[ip], numz[ip], startz[ip]);
        }
#endif
        {
            const MPI_Datatype mpi_type = (sizeof(FPTYPE) == 4)
                                              ? MPI_COMPLEX
                                              : MPI_DOUBLE_COMPLEX;
            MPI_Alltoallv(h_mpi_send_bac, numg, startg, mpi_type,
                          h_mpi_recv_bac, numr, startr, mpi_type,
                          comm);
        }

        FFT_DBG_HOST("bac:after_alltoallv(h_mpi_recv_bac)", rank,
                      h_mpi_recv_bac, (size_t)nplane_ * nstot_);

        // =============================================================
        // Phase 3: scatter + H2D + fftxy_bac
        // =============================================================
        const int num_chunks = (nplane_ + chunk_ - 1) / chunk_;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            const int b       = chunk % 2;
            const int z_off   = chunk * chunk_;
            const int z_count = std::min(chunk_, nplane_ - z_off);
            const int elems   = z_count * nxy;

            if (chunk >= 2)
                gpuEventSynchronize(fftz_done[(chunk - 2) % 2]);

            // Scatter sticks from (nstot, nplane) layout to xy-plane
            std::memset(h_d2h_buf[b], 0,
                        elems * sizeof(std::complex<FPTYPE>));
            for (int iz = 0; iz < z_count; ++iz)
            {
                std::complex<FPTYPE>* dst = h_d2h_buf[b] + (size_t)iz * nxy;
                for (int istot = 0; istot < nstot_; ++istot)
                    dst[istot2ixy[istot]]
                        = h_mpi_recv_bac[istot * nplane_ + (z_off + iz)];
            }

            gpuMemcpyAsync(d_xy_chunk[b], h_d2h_buf[b],
                           elems * sizeof(std::complex<FPTYPE>),
                           gpuMemcpyH2D, h2d_stream[b]);
            gpuEventRecord(h2d_done[b], h2d_stream[b]);

            gpuStreamWaitEvent(compute_stream[b], h2d_done[b], 0);
            fft_xy_bac(d_xy_chunk[b], d_xy_chunk[b], compute_stream[b]);
            gpuEventRecord(fftz_done[b], compute_stream[b]);

            gpuMemcpyAsync(d_out + (size_t)z_off * nxy,
                           d_xy_chunk[b],
                           elems * sizeof(std::complex<FPTYPE>),
                           gpuMemcpyD2D, compute_stream[b]);
        }

        for (int b = 0; b < 2; ++b)
            gpuStreamSynchronize(compute_stream[b]);

        FFT_DBG_DEV("bac:output(d_out)", rank, d_out, (size_t)nplane_ * nxy);
    }

    bool is_initialised() const { return initialised_; }

    std::complex<FPTYPE>* d_sticks = nullptr;
    std::complex<FPTYPE>* d_transpose_buf = nullptr;
    int get_nxy() const { return nx_ * ny_; }
    int get_nplane() const { return nplane_; }

private:
    void release()
    {
        if (!initialised_) return;
        for (int b = 0; b < 2; ++b)
        {
            if (d_xy_chunk[b]) { gpuFree(d_xy_chunk[b]); d_xy_chunk[b] = nullptr; }
            if (d_z_chunk[b])  { gpuFree(d_z_chunk[b]);  d_z_chunk[b]  = nullptr; }
            if (h_d2h_buf[b])  { gpuFreeHost(h_d2h_buf[b]); h_d2h_buf[b] = nullptr; }
            if (h_h2d_buf[b])  { gpuFreeHost(h_h2d_buf[b]); h_h2d_buf[b] = nullptr; }
            gpuEventDestroy(fft_done[b]);
            gpuEventDestroy(d2h_done[b]);
            gpuEventDestroy(h2d_done[b]);
            gpuEventDestroy(fftz_done[b]);
            gpuStreamDestroy(compute_stream[b]);
            gpuStreamDestroy(d2h_stream[b]);
            gpuStreamDestroy(h2d_stream[b]);
        }
        if (d_sticks)        { gpuFree(d_sticks);        d_sticks        = nullptr; }
        if (d_transpose_buf) { gpuFree(d_transpose_buf); d_transpose_buf = nullptr; }
        if (h_mpi_send)      { gpuFreeHost(h_mpi_send);  h_mpi_send      = nullptr; }
        if (h_mpi_recv)      { gpuFreeHost(h_mpi_recv);  h_mpi_recv      = nullptr; }
        if (h_fftz_in)       { gpuFreeHost(h_fftz_in);   h_fftz_in       = nullptr; }
        if (h_mpi_send_bac)  { gpuFreeHost(h_mpi_send_bac); h_mpi_send_bac = nullptr; }
        if (h_mpi_recv_bac)  { gpuFreeHost(h_mpi_recv_bac); h_mpi_recv_bac = nullptr; }
        initialised_ = false;
    }

    int nx_ = 0, ny_ = 0, nz_ = 0;
    int nplane_ = 0, nst_ = 0, nstot_ = 0, nproc_ = 0;
    int chunk_  = 16;
    bool initialised_ = false;

    std::complex<FPTYPE>* d_xy_chunk[2] = {nullptr, nullptr};
    std::complex<FPTYPE>* d_z_chunk[2]  = {nullptr, nullptr};

    std::complex<FPTYPE>* h_d2h_buf[2]  = {nullptr, nullptr};
    std::complex<FPTYPE>* h_h2d_buf[2]  = {nullptr, nullptr};

    std::complex<FPTYPE>* h_mpi_send     = nullptr;
    std::complex<FPTYPE>* h_mpi_recv     = nullptr;
    std::complex<FPTYPE>* h_fftz_in      = nullptr;
    std::complex<FPTYPE>* h_mpi_send_bac = nullptr;
    std::complex<FPTYPE>* h_mpi_recv_bac = nullptr;

    gpuStream_t compute_stream[2];
    gpuStream_t d2h_stream[2];
    gpuStream_t h2d_stream[2];
    gpuEvent_t  fft_done[2];
    gpuEvent_t  d2h_done[2];
    gpuEvent_t  h2d_done[2];
    gpuEvent_t  fftz_done[2];
};

} // namespace ModulePW

#endif // defined(__CUDA) || defined(__ROCM)
#endif // PW_MULTI_GPU_FFT_H
