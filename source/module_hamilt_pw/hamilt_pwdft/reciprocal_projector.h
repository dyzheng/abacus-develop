#ifndef RECIPROCAL_PROJECTOR_H
#define RECIPROCAL_PROJECTOR_H

#include "projector_base.h"
#include "vkb_batch_manager.h"
#include "module_base/macros.h"
#include "module_base/module_device/memory_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/nonlocal_op.h"

#include <vector>

namespace hamilt
{

/**
 * @brief Reciprocal-space VKB projector with optional atom batching.
 *
 * Implements the ProjectorBase interface using reciprocal-space VKB projectors.
 * Supports two modes:
 * - Full mode: entire VKB matrix (nkb x npwx) on device (original behavior)
 * - Batched mode: VKB computed on CPU, transferred batch-by-batch to GPU
 *
 * @tparam T Element type (std::complex<float> or std::complex<double>)
 * @tparam Device Device type (DEVICE_CPU or DEVICE_GPU)
 */
template <typename T, typename Device>
class ReciprocalProjector : public ProjectorBase<T>
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    /**
     * @brief Construct a ReciprocalProjector.
     *
     * @param[in] vkb Pointer to the full VKB matrix on device (nkb x npwx).
     *                 For batched mode, this should be nullptr (CPU vkb used instead).
     * @param[in] vkb_cpu Pointer to the full VKB matrix on CPU (nkb x npwx).
     *                     Only used in batched mode. nullptr for full mode.
     * @param[in] nkb Total number of beta projectors (KB projectors).
     * @param[in] npwx Leading dimension (max plane waves across k-points).
     * @param[in] deeq Pointer to the D matrix on device.
     * @param[in] deeq_nc Pointer to the noncollinear D matrix on device (or nullptr).
     * @param[in] ntype Number of atom types.
     * @param[in] na_per_type Array of number of atoms per type (length ntype).
     * @param[in] nh_per_type Array of number of projectors per type (length ntype).
     * @param[in] isk Pointer to k-point to spin mapping array.
     * @param[in] ik Current k-point index.
     * @param[in] deeq_bounds Array of 3 bounds: {bound2, bound3, bound4} for deeq indexing.
     * @param[in] batch_manager Optional pointer to initialized VKBBatchManager.
     *                          nullptr means no batching (full VKB on GPU).
     */
    ReciprocalProjector(T* vkb,
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
                        const VKBBatchManager<T>* batch_manager = nullptr);

    ~ReciprocalProjector() override;

    /**
     * @brief Compute projection coefficients becp = <beta|psi>.
     *
     * In full mode, uses a single GEMM/GEMV on the full VKB matrix.
     * In batched mode, iterates over batches, copying VKB rows batch-by-batch.
     *
     * @param[in] psi Pointer to the wavefunction array.
     * @param[out] becp Pointer to the output projection coefficients.
     * @param[in] nbands Number of bands.
     * @param[in] npw Number of plane waves for current k-point.
     * @param[in] max_npw Leading dimension for plane-wave index.
     * @param[in] npol Number of spinor components (1 or 2).
     */
    void compute_becp(const T* psi, T* becp, int nbands, int npw, int max_npw, int npol) override;

    /**
     * @brief Apply the D matrix to becp and accumulate into hpsi.
     *
     * Computes hpsi += |beta> * D * becp in full or batched mode.
     *
     * @param[in] becp Pointer to the projection coefficients from compute_becp.
     * @param[in,out] hpsi Pointer to the Hamiltonian-psi product to accumulate into.
     * @param[in] nbands Number of bands.
     * @param[in] npw Number of plane waves for current k-point.
     * @param[in] max_npw Leading dimension for plane-wave index.
     * @param[in] npol Number of spinor components (1 or 2).
     */
    void apply_deeq_and_accumulate(const T* becp, T* hpsi, int nbands, int npw, int max_npw, int npol) override;

    /**
     * @brief Get the current GPU memory usage of this projector in bytes.
     *
     * @return The number of bytes of device memory in use.
     */
    size_t get_memory_bytes() const override;

  private:
    // Non-owning pointers (lifetime managed externally)
    T* vkb_ = nullptr;                                  ///< VKB on device (full mode)
    const T* vkb_cpu_ = nullptr;                        ///< VKB on CPU (batched mode)
    const Real* deeq_ = nullptr;                        ///< D matrix on device
    const T* deeq_nc_ = nullptr;                        ///< Noncollinear D matrix on device
    const int* isk_ = nullptr;                          ///< k-point to spin mapping
    const VKBBatchManager<T>* batch_manager_ = nullptr; ///< nullptr = full mode

    int nkb_ = 0;   ///< Total number of beta projectors
    int npwx_ = 0;  ///< Leading dimension (max plane waves)
    int ntype_ = 0;  ///< Number of atom types
    int ik_ = 0;     ///< Current k-point index

    std::vector<int> na_per_type_; ///< Number of atoms per type
    std::vector<int> nh_per_type_; ///< Number of projectors per type
    int deeq_bounds_[3] = {};      ///< deeq indexing bounds {bound2, bound3, bound4}

    // Owned GPU buffers for batched mode
    T* vkb_batch_gpu_ = nullptr;      ///< Batch VKB buffer on GPU
    mutable T* ps_ = nullptr;         ///< D*becp workspace on device
    mutable size_t ps_alloc_ = 0;     ///< Current ps allocation size (in elements)
    mutable T* becp_batch_ = nullptr; ///< Contiguous becp buffer for batched mode (nkb_batch stride)
    mutable size_t becp_batch_alloc_ = 0; ///< Current becp_batch allocation size (in elements)

    Device* ctx_ = {};
    base_device::DEVICE_CPU* cpu_ctx_ = {};

    using gemv_op = hsolver::gemv_op<T, Device>;
    using gemm_op = hsolver::gemm_op<T, Device>;
    using nonlocal_op_t = nonlocal_pw_op<Real, Device>;
    using setmem_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;
    using syncmem_d2d_op = base_device::memory::synchronize_memory_op<T, Device, Device>;

    T one_{1, 0};
    T zero_{0, 0};

    size_t gpu_memory_used_ = 0; ///< Track device memory usage in bytes
};

} // namespace hamilt

#endif // RECIPROCAL_PROJECTOR_H
