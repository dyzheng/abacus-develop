#include "reciprocal_projector.h"

#include "module_base/parallel_comm.h"
#include "module_base/parallel_device.h"
#include "module_base/parallel_reduce.h"

#include <algorithm>
#include <cstring>

namespace hamilt
{

template <typename T, typename Device>
ReciprocalProjector<T, Device>::ReciprocalProjector(T* vkb,
                                                     const T* vkb_cpu,
                                                     int nkb,
                                                     int npwx,
                                                     const Real* deeq,
                                                     const T* deeq_nc,
                                                     int ntype,
                                                     const int* na_per_type,
                                                     const int* nh_per_type,
                                                     const int* isk,
                                                     int ik,
                                                     const int* deeq_bounds,
                                                     const VKBBatchManager<T>* batch_manager)
    : vkb_(vkb),
      vkb_cpu_(vkb_cpu),
      nkb_(nkb),
      npwx_(npwx),
      deeq_(deeq),
      deeq_nc_(deeq_nc),
      ntype_(ntype),
      isk_(isk),
      ik_(ik),
      batch_manager_(batch_manager)
{
    // Copy per-type arrays
    na_per_type_.assign(na_per_type, na_per_type + ntype);
    nh_per_type_.assign(nh_per_type, nh_per_type + ntype);

    // Copy deeq bounds
    deeq_bounds_[0] = deeq_bounds[0];
    deeq_bounds_[1] = deeq_bounds[1];
    deeq_bounds_[2] = deeq_bounds[2];

    // Allocate batch VKB buffer on device if batching is enabled with >1 batch
    if (batch_manager_ != nullptr && batch_manager_->get_nbatch() > 1)
    {
        const size_t batch_elems = batch_manager_->get_max_batch_elements();
        resmem_op()(this->ctx_, this->vkb_batch_gpu_, batch_elems, "ReciprocalProjector::vkb_batch");
        gpu_memory_used_ += batch_elems * sizeof(T);
    }
}

template <typename T, typename Device>
ReciprocalProjector<T, Device>::~ReciprocalProjector()
{
    if (vkb_batch_gpu_ != nullptr)
    {
        delmem_op()(this->ctx_, this->vkb_batch_gpu_);
        vkb_batch_gpu_ = nullptr;
    }
    if (ps_ != nullptr)
    {
        delmem_op()(this->ctx_, this->ps_);
        ps_ = nullptr;
    }
    if (becp_batch_ != nullptr)
    {
        delmem_op()(this->ctx_, this->becp_batch_);
        becp_batch_ = nullptr;
    }
}

template <typename T, typename Device>
void ReciprocalProjector<T, Device>::compute_becp(const T* psi,
                                                   T* becp,
                                                   int nbands,
                                                   int npw,
                                                   int max_npw,
                                                   int npol)
{
    if (nkb_ <= 0)
    {
        return;
    }

    // Check if we are using batched mode
    if (batch_manager_ != nullptr && batch_manager_->get_nbatch() > 1)
    {
        // ===== BATCHED MODE =====
        const int max_nkb_batch = batch_manager_->get_max_nkb_batch();

        // Ensure becp_batch_ is large enough
        const size_t needed = static_cast<size_t>(max_nkb_batch) * nbands;
        if (needed > becp_batch_alloc_)
        {
            if (becp_batch_ != nullptr)
            {
                delmem_op()(this->ctx_, this->becp_batch_);
                gpu_memory_used_ -= becp_batch_alloc_ * sizeof(T);
            }
            resmem_op()(this->ctx_, this->becp_batch_, needed, "ReciprocalProjector::becp_batch");
            becp_batch_alloc_ = needed;
            gpu_memory_used_ += needed * sizeof(T);
        }

        for (int ibatch = 0; ibatch < batch_manager_->get_nbatch(); ibatch++)
        {
            int atom_start = 0, atom_end = 0, nkb_batch = 0;
            batch_manager_->get_batch_info(ibatch, atom_start, atom_end, nkb_batch);
            const int jkb_offset = batch_manager_->get_jkb_offset(ibatch);

            if (nkb_batch == 0)
            {
                continue;
            }

            // 1. Copy batch rows from CPU VKB to GPU buffer
            syncmem_h2d_op()(this->ctx_, this->cpu_ctx_,
                             vkb_batch_gpu_,
                             vkb_cpu_ + static_cast<size_t>(jkb_offset) * npwx_,
                             static_cast<size_t>(nkb_batch) * npwx_);

            // 2. becp_batch = vkb_batch^H * psi
            //    Write to contiguous becp_batch_ with stride nkb_batch (not nkb_).
            char transa = 'C';
            char transb = 'N';
            if (nbands == 1)
            {
                int inc = 1;
                gemv_op()(this->ctx_, transa, npw, nkb_batch,
                          &this->one_, vkb_batch_gpu_, npwx_,
                          psi, inc,
                          &this->zero_, becp_batch_, inc);
            }
            else
            {
                gemm_op()(this->ctx_, transa, transb,
                          nkb_batch, nbands, npw,
                          &this->one_, vkb_batch_gpu_, npwx_,
                          psi, max_npw,
                          &this->zero_, becp_batch_, nkb_batch);
            }

            // 3. MPI reduction across pool
            Parallel_Common::reduce_dev(this->ctx_, becp_batch_, nkb_batch * nbands, POOL_WORLD);

            // 4. Scatter becp_batch_ (stride nkb_batch) into full becp (stride nkb_)
            //    becp layout: [nbands][nkb_], batch portion at column jkb_offset.
            for (int ib = 0; ib < nbands; ib++)
            {
                const T* src = becp_batch_ + static_cast<size_t>(ib) * nkb_batch;
                T* dst = becp + static_cast<size_t>(ib) * nkb_ + jkb_offset;
                syncmem_d2d_op()(this->ctx_, this->ctx_,
                                 dst, src, nkb_batch);
            }
        }
    }
    else
    {
        // ===== FULL MODE =====
        char transa = 'C';
        char transb = 'N';
        if (nbands == 1)
        {
            int inc = 1;
            gemv_op()(this->ctx_, transa, npw, nkb_,
                      &this->one_, vkb_, npwx_,
                      psi, inc,
                      &this->zero_, becp, inc);
        }
        else
        {
            gemm_op()(this->ctx_, transa, transb,
                      nkb_, nbands, npw,
                      &this->one_, vkb_, npwx_,
                      psi, max_npw,
                      &this->zero_, becp, nkb_);
        }

        // MPI reduction across pool
        Parallel_Common::reduce_dev(this->ctx_, becp, nkb_ * nbands, POOL_WORLD);
    }
}

template <typename T, typename Device>
void ReciprocalProjector<T, Device>::apply_deeq_and_accumulate(const T* becp,
                                                                T* hpsi,
                                                                int nbands,
                                                                int npw,
                                                                int max_npw,
                                                                int npol)
{
    if (nkb_ <= 0)
    {
        return;
    }

    // Check if we are using batched mode
    if (batch_manager_ != nullptr && batch_manager_->get_nbatch() > 1)
    {
        // ===== BATCHED MODE =====
        const int max_nkb_batch = batch_manager_->get_max_nkb_batch();

        // Ensure ps_ is large enough for the largest batch
        const size_t ps_needed = static_cast<size_t>(max_nkb_batch) * nbands;
        if (ps_needed > ps_alloc_)
        {
            if (ps_ != nullptr)
            {
                delmem_op()(this->ctx_, this->ps_);
                gpu_memory_used_ -= ps_alloc_ * sizeof(T);
            }
            resmem_op()(this->ctx_, this->ps_, ps_needed, "ReciprocalProjector::ps_batch");
            ps_alloc_ = ps_needed;
            gpu_memory_used_ += ps_needed * sizeof(T);
        }

        // Ensure becp_batch_ is large enough (may already be allocated by compute_becp)
        const size_t becp_needed = static_cast<size_t>(max_nkb_batch) * nbands;
        if (becp_needed > becp_batch_alloc_)
        {
            if (becp_batch_ != nullptr)
            {
                delmem_op()(this->ctx_, this->becp_batch_);
                gpu_memory_used_ -= becp_batch_alloc_ * sizeof(T);
            }
            resmem_op()(this->ctx_, this->becp_batch_, becp_needed, "ReciprocalProjector::becp_batch");
            becp_batch_alloc_ = becp_needed;
            gpu_memory_used_ += becp_needed * sizeof(T);
        }

        for (int ibatch = 0; ibatch < batch_manager_->get_nbatch(); ibatch++)
        {
            int atom_start = 0, atom_end = 0, nkb_batch = 0;
            batch_manager_->get_batch_info(ibatch, atom_start, atom_end, nkb_batch);
            const int jkb_offset = batch_manager_->get_jkb_offset(ibatch);

            if (nkb_batch == 0)
            {
                continue;
            }

            // 1. Copy batch VKB rows from CPU to GPU
            syncmem_h2d_op()(this->ctx_, this->cpu_ctx_,
                             vkb_batch_gpu_,
                             vkb_cpu_ + static_cast<size_t>(jkb_offset) * npwx_,
                             static_cast<size_t>(nkb_batch) * npwx_);

            // 2. Gather batch becp from full becp (stride nkb_) into becp_batch_ (stride nkb_batch)
            //    becp layout: [nbands][nkb_], batch portion at column jkb_offset.
            //    becp_batch_ layout: [nbands][nkb_batch], contiguous.
            for (int ib = 0; ib < nbands; ib++)
            {
                const T* src = becp + static_cast<size_t>(ib) * nkb_ + jkb_offset;
                T* dst = becp_batch_ + static_cast<size_t>(ib) * nkb_batch;
                syncmem_d2d_op()(this->ctx_, this->ctx_,
                                 dst, src, nkb_batch);
            }

            // 3. Zero ps workspace
            setmem_op()(this->ctx_, this->ps_, 0, nkb_batch * nbands);

            // 4. Apply D matrix: ps_batch = D * becp_batch
            //    Use becp_batch_ which has contiguous nkb_batch stride, matching
            //    the nonlocal_op kernel's expectation: becp[ib * nkb + ...].
            int sum_batch = 0;
            int iat_scan = 0;
            nonlocal_op_t nonlocal_op;

            if (npol == 1)
            {
                const int current_spin = isk_[ik_];
                for (int it = 0; it < ntype_; it++)
                {
                    const int na_type = na_per_type_[it];
                    const int nproj = nh_per_type_[it];
                    const int type_start = iat_scan;
                    const int type_end = iat_scan + na_type;
                    const int batch_type_start = std::max(type_start, atom_start);
                    const int batch_type_end = std::min(type_end, atom_end);
                    if (batch_type_start < batch_type_end)
                    {
                        const int na_in_batch = batch_type_end - batch_type_start;
                        int iat_local = batch_type_start;
                        nonlocal_op(this->ctx_,
                                    na_in_batch, nbands, nproj,
                                    sum_batch, iat_local, current_spin, nkb_batch,
                                    deeq_bounds_[0], deeq_bounds_[1], deeq_bounds_[2],
                                    deeq_,
                                    ps_, becp_batch_);
                    }
                    iat_scan += na_type;
                }
            }
            else
            {
                // Non-collinear case (npol == 2)
                for (int it = 0; it < ntype_; it++)
                {
                    const int na_type = na_per_type_[it];
                    const int nproj = nh_per_type_[it];
                    const int type_start = iat_scan;
                    const int type_end = iat_scan + na_type;
                    const int batch_type_start = std::max(type_start, atom_start);
                    const int batch_type_end = std::min(type_end, atom_end);
                    if (batch_type_start < batch_type_end)
                    {
                        const int na_in_batch = batch_type_end - batch_type_start;
                        int iat_local = batch_type_start;
                        nonlocal_op(this->ctx_,
                                    na_in_batch, nbands, nproj,
                                    sum_batch, iat_local, nkb_batch,
                                    deeq_bounds_[0], deeq_bounds_[1], deeq_bounds_[2],
                                    deeq_nc_,
                                    ps_, becp_batch_);
                    }
                    iat_scan += na_type;
                }
            }

            // 5. hpsi += vkb_batch * ps_batch
            if (nbands == 1)
            {
                int inc = 1;
                gemv_op()(this->ctx_, 'N', npw, nkb_batch,
                          &this->one_, vkb_batch_gpu_, npwx_,
                          ps_, inc,
                          &this->one_, hpsi, inc);
            }
            else
            {
                gemm_op()(this->ctx_, 'N', 'T',
                          npw, nbands, nkb_batch,
                          &this->one_, vkb_batch_gpu_, npwx_,
                          ps_, nbands,
                          &this->one_, hpsi, max_npw);
            }
        }
    }
    else
    {
        // ===== FULL MODE =====
        // 1. Allocate/resize ps if needed
        const size_t needed = static_cast<size_t>(nkb_) * nbands;
        if (needed > ps_alloc_)
        {
            if (ps_ != nullptr)
            {
                delmem_op()(this->ctx_, this->ps_);
                gpu_memory_used_ -= ps_alloc_ * sizeof(T);
            }
            resmem_op()(this->ctx_, this->ps_, needed, "ReciprocalProjector::ps");
            ps_alloc_ = needed;
            gpu_memory_used_ += needed * sizeof(T);
        }

        // 2. Zero-fill ps
        setmem_op()(this->ctx_, this->ps_, 0, nkb_ * nbands);

        // 3. Apply D matrix for each atom type
        int sum = 0;
        int iat = 0;
        nonlocal_op_t nonlocal_op;

        if (npol == 1)
        {
            const int current_spin = isk_[ik_];
            for (int it = 0; it < ntype_; it++)
            {
                const int nproj = nh_per_type_[it];
                nonlocal_op(this->ctx_,
                            na_per_type_[it], nbands, nproj,
                            sum, iat, current_spin, nkb_,
                            deeq_bounds_[0], deeq_bounds_[1], deeq_bounds_[2],
                            deeq_,
                            ps_, becp);
            }
        }
        else
        {
            // Non-collinear case (npol == 2)
            for (int it = 0; it < ntype_; it++)
            {
                const int nproj = nh_per_type_[it];
                nonlocal_op(this->ctx_,
                            na_per_type_[it], nbands, nproj,
                            sum, iat, nkb_,
                            deeq_bounds_[0], deeq_bounds_[1], deeq_bounds_[2],
                            deeq_nc_,
                            ps_, becp);
            }
        }

        // 4. Accumulate: hpsi += vkb * ps
        if (nbands == 1)
        {
            int inc = 1;
            gemv_op()(this->ctx_, 'N', npw, nkb_,
                      &this->one_, vkb_, npwx_,
                      ps_, inc,
                      &this->one_, hpsi, inc);
        }
        else
        {
            gemm_op()(this->ctx_, 'N', 'T',
                      npw, nbands, nkb_,
                      &this->one_, vkb_, npwx_,
                      ps_, nbands,
                      &this->one_, hpsi, max_npw);
        }
    }
}

template <typename T, typename Device>
size_t ReciprocalProjector<T, Device>::get_memory_bytes() const
{
    return gpu_memory_used_;
}

template class ReciprocalProjector<std::complex<float>, base_device::DEVICE_CPU>;
template class ReciprocalProjector<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ReciprocalProjector<std::complex<float>, base_device::DEVICE_GPU>;
template class ReciprocalProjector<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace hamilt
