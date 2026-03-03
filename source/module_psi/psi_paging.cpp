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

// Stub implementations for GPU methods (full implementation in Task batch 2)
template <typename T, typename Device>
void Psi<T, Device>::load_k_to_gpu(int ik)
{
    // Will be implemented in Tasks 6-10
}

template <typename T, typename Device>
void Psi<T, Device>::store_k_from_gpu(int ik)
{
    // Will be implemented in Tasks 6-10
}

template <typename T, typename Device>
void Psi<T, Device>::ensure_k_on_gpu(int ik)
{
    // Will be implemented in Tasks 6-10
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
