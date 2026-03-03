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

// Explicit instantiations
template class Psi<float, base_device::DEVICE_CPU>;
template class Psi<std::complex<float>, base_device::DEVICE_CPU>;
template class Psi<double, base_device::DEVICE_CPU>;
template class Psi<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Psi<float, base_device::DEVICE_GPU>;
template class Psi<std::complex<float>, base_device::DEVICE_GPU>;
template class Psi<double, base_device::DEVICE_GPU>;
template class Psi<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace psi
