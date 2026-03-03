#include "psi.h"
#include "module_base/tool_quit.h"

namespace psi
{

template <typename T, typename Device>
void Psi<T, Device>::set_storage_mode(PsiStorageMode mode)
{
    if (mode == storage_mode_)
    {
        return; // Already in requested mode
    }

    // For now, only support setting mode before allocation
    if (this->psi != nullptr)
    {
        ModuleBase::WARNING_QUIT("Psi::set_storage_mode",
                                 "Cannot change storage mode after allocation");
    }

    storage_mode_ = mode;
}

template <typename T, typename Device>
T* Psi<T, Device>::get_cpu_pointer(int ik)
{
    if (storage_mode_ == PsiStorageMode::PAGED_GPU)
    {
        if (ik < 0 || ik >= this->nk)
        {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer", "Invalid k-point index");
        }
        if (psi_cpu_ == nullptr)
        {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer",
                                     "CPU buffer not allocated in PAGED_GPU mode");
        }
        return psi_cpu_ + static_cast<size_t>(ik) * this->nbands * this->nbasis;
    }
    else
    {
        return this->psi + static_cast<size_t>(ik) * this->nbands * this->nbasis;
    }
}

template <typename T, typename Device>
const T* Psi<T, Device>::get_cpu_pointer(int ik) const
{
    if (storage_mode_ == PsiStorageMode::PAGED_GPU)
    {
        if (ik < 0 || ik >= this->nk)
        {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer", "Invalid k-point index");
        }
        if (psi_cpu_ == nullptr)
        {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer",
                                     "CPU buffer not allocated in PAGED_GPU mode");
        }
        return psi_cpu_ + static_cast<size_t>(ik) * this->nbands * this->nbasis;
    }
    else
    {
        return this->psi + static_cast<size_t>(ik) * this->nbands * this->nbasis;
    }
}

// Load k-point data from CPU to GPU device buffer
template <typename T, typename Device>
void Psi<T, Device>::load_k_to_gpu(int ik)
{
    if (storage_mode_ != PsiStorageMode::PAGED_GPU)
    {
        return; // No-op if not in paged mode
    }

    if (ik < 0 || ik >= this->nk)
    {
        ModuleBase::WARNING_QUIT("Psi::load_k_to_gpu", "Invalid k-point index");
    }

    if (psi_cpu_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("Psi::load_k_to_gpu", "CPU buffer not allocated");
    }

    const size_t k_size = static_cast<size_t>(this->nbands) * this->nbasis;
    const T* src = psi_cpu_ + static_cast<size_t>(ik) * this->nbands * this->nbasis;

    // Transfer from CPU to Device
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>()(
        this->ctx, cpu_ctx, this->psi, src, k_size);

    current_k_gpu_ = ik;
    this->psi_current = this->psi;
}

// Store k-point data from GPU device buffer back to CPU
template <typename T, typename Device>
void Psi<T, Device>::store_k_from_gpu(int ik)
{
    if (storage_mode_ != PsiStorageMode::PAGED_GPU)
    {
        return; // No-op if not in paged mode
    }

    if (ik < 0 || ik >= this->nk)
    {
        ModuleBase::WARNING_QUIT("Psi::store_k_from_gpu", "Invalid k-point index");
    }

    if (psi_cpu_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("Psi::store_k_from_gpu", "CPU buffer not allocated");
    }

    const size_t k_size = static_cast<size_t>(this->nbands) * this->nbasis;
    T* dst = psi_cpu_ + static_cast<size_t>(ik) * this->nbands * this->nbasis;

    // Transfer from Device to CPU
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, Device>()(
        cpu_ctx, this->ctx, dst, this->psi, k_size);
}

// Ensure the specified k-point is loaded on GPU, loading if necessary.
// NOTE: Callers must call store_k_from_gpu(current_k_gpu_) before switching
// k-points if the GPU buffer has been modified (e.g., by a diagonalization solver),
// otherwise modifications will be lost.
template <typename T, typename Device>
void Psi<T, Device>::ensure_k_on_gpu(int ik)
{
    if (storage_mode_ != PsiStorageMode::PAGED_GPU)
    {
        return; // No-op if not in paged mode
    }

    if (current_k_gpu_ != ik)
    {
        load_k_to_gpu(ik);
    }
}

// Explicit instantiations for paging methods only (full class instantiated in psi.cpp)
#define INSTANTIATE_PAGING_METHODS(T, Device)                                                                          \
    template void Psi<T, Device>::set_storage_mode(PsiStorageMode);                                                    \
    template T* Psi<T, Device>::get_cpu_pointer(int);                                                                  \
    template const T* Psi<T, Device>::get_cpu_pointer(int) const;                                                      \
    template void Psi<T, Device>::load_k_to_gpu(int);                                                                  \
    template void Psi<T, Device>::store_k_from_gpu(int);                                                               \
    template void Psi<T, Device>::ensure_k_on_gpu(int);

INSTANTIATE_PAGING_METHODS(float, base_device::DEVICE_CPU)
INSTANTIATE_PAGING_METHODS(std::complex<float>, base_device::DEVICE_CPU)
INSTANTIATE_PAGING_METHODS(double, base_device::DEVICE_CPU)
INSTANTIATE_PAGING_METHODS(std::complex<double>, base_device::DEVICE_CPU)
#if ((defined __CUDA) || (defined __ROCM))
INSTANTIATE_PAGING_METHODS(float, base_device::DEVICE_GPU)
INSTANTIATE_PAGING_METHODS(std::complex<float>, base_device::DEVICE_GPU)
INSTANTIATE_PAGING_METHODS(double, base_device::DEVICE_GPU)
INSTANTIATE_PAGING_METHODS(std::complex<double>, base_device::DEVICE_GPU)
#endif

#undef INSTANTIATE_PAGING_METHODS

} // namespace psi
