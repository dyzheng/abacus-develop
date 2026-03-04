#ifndef VKB_BATCH_MANAGER_H
#define VKB_BATCH_MANAGER_H

#include <vector>
#include <cstddef>

namespace hamilt {

/// Compute optimal batch size based on GPU memory budget.
/// Uses 25% of available GPU memory for VKB projector storage.
/// Returns at least 1 even when memory is very limited.
int compute_optimal_batch_size(int total_atoms, int avg_nproj_per_atom,
                               int npwx, size_t available_gpu_mem);

/**
 * @brief Manages batching of atoms for VKB projector computation on GPU.
 *
 * For large systems, the full VKB matrix (nkb x npwx) may exceed GPU memory.
 * This class splits atoms into batches that fit within a memory budget,
 * enabling batch-by-batch computation of VKB projectors.
 *
 * @tparam T Element type, typically std::complex<float> or std::complex<double>
 */
template <typename T>
class VKBBatchManager {
public:
    VKBBatchManager() = default;
    ~VKBBatchManager() = default;

    /**
     * @brief Initialize batch layout with auto-detected batch size.
     *
     * @param[in] total_atoms Total number of atoms
     * @param[in] nproj_per_atom Array of projector counts per atom (length = total_atoms)
     * @param[in] npwx Maximum number of plane waves
     * @param[in] gpu_mem_budget Available GPU memory in bytes
     */
    void init(int total_atoms, const int* nproj_per_atom, int npwx, size_t gpu_mem_budget);

    /**
     * @brief Initialize batch layout with optional explicit batch size.
     *
     * @param[in] total_atoms Total number of atoms
     * @param[in] nproj_per_atom Array of projector counts per atom (length = total_atoms)
     * @param[in] npwx Maximum number of plane waves
     * @param[in] gpu_mem_budget Available GPU memory in bytes
     * @param[in] explicit_batch_atoms Explicit number of atoms per batch (0 = auto-detect)
     */
    void init(int total_atoms, const int* nproj_per_atom, int npwx,
              size_t gpu_mem_budget, int explicit_batch_atoms);

    /**
     * @brief Get batch information for a given batch index.
     *
     * @param[in] ibatch Batch index (0 <= ibatch < get_nbatch())
     * @param[out] atom_start Starting atom index (inclusive)
     * @param[out] atom_end Ending atom index (exclusive)
     * @param[out] nkb_batch Number of projectors in this batch
     */
    void get_batch_info(int ibatch, int& atom_start, int& atom_end, int& nkb_batch) const;

    int get_nbatch() const { return nbatch_; }
    int get_max_nkb_batch() const { return max_nkb_batch_; }
    size_t get_max_batch_elements() const { return max_batch_elements_; }
    bool is_initialized() const { return nbatch_ > 0; }

    /**
     * @brief Get the jkb (projector) offset for a given batch.
     *
     * This is the cumulative sum of nkb for all batches before this one,
     * i.e., the starting row index into the full VKB matrix for this batch.
     *
     * @param[in] ibatch Batch index (0 <= ibatch < get_nbatch())
     * @return Starting jkb index for this batch
     */
    int get_jkb_offset(int ibatch) const { return batch_jkb_offset_[ibatch]; }

private:
    int nbatch_ = 0;
    int max_nkb_batch_ = 0;              ///< Max nkb across all batches
    size_t max_batch_elements_ = 0;       ///< Max nkb_batch * npwx
    int npwx_ = 0;
    std::vector<int> batch_atom_start_;   ///< Starting atom index for each batch
    std::vector<int> batch_atom_count_;   ///< Number of atoms in each batch
    std::vector<int> batch_nkb_;          ///< Number of projectors in each batch
    std::vector<int> batch_jkb_offset_;   ///< Cumulative jkb offset for each batch
};

} // namespace hamilt

#endif // VKB_BATCH_MANAGER_H
