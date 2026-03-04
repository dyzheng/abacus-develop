#include "vkb_batch_manager.h"

#include <algorithm>
#include <complex>
#include <numeric>
#include <stdexcept>

namespace hamilt {

int compute_optimal_batch_size(int total_atoms, int avg_nproj_per_atom,
                               int npwx, size_t available_gpu_mem)
{
    // Use 25% of available GPU memory for VKB storage
    const size_t vkb_budget = available_gpu_mem / 4;

    // Memory per atom = avg_nproj * npwx * sizeof(complex<double>)
    const size_t mem_per_atom =
        static_cast<size_t>(avg_nproj_per_atom) * npwx * sizeof(std::complex<double>);

    if (mem_per_atom == 0)
    {
        return total_atoms;
    }

    const int batch_from_budget = static_cast<int>(vkb_budget / mem_per_atom);

    // At least 1 atom per batch, at most total_atoms
    return std::max(1, std::min(batch_from_budget, total_atoms));
}

template <typename T>
void VKBBatchManager<T>::init(int total_atoms, const int* nproj_per_atom,
                              int npwx, size_t gpu_mem_budget)
{
    init(total_atoms, nproj_per_atom, npwx, gpu_mem_budget, 0);
}

template <typename T>
void VKBBatchManager<T>::init(int total_atoms, const int* nproj_per_atom,
                              int npwx, size_t gpu_mem_budget,
                              int explicit_batch_atoms)
{
    npwx_ = npwx;

    // Determine batch size
    int batch_size = 0;
    if (explicit_batch_atoms > 0)
    {
        batch_size = explicit_batch_atoms;
    }
    else
    {
        // Compute average projectors per atom for auto-detection
        int total_nproj = 0;
        for (int i = 0; i < total_atoms; i++)
        {
            total_nproj += nproj_per_atom[i];
        }
        const int avg_nproj = (total_atoms > 0) ? (total_nproj / total_atoms) : 0;
        // Use sizeof(T) for the actual element type in the budget computation
        // but compute_optimal_batch_size uses sizeof(complex<double>) as worst case
        batch_size = compute_optimal_batch_size(total_atoms, avg_nproj, npwx, gpu_mem_budget);
    }

    // Ensure batch_size is valid
    batch_size = std::max(1, std::min(batch_size, total_atoms));

    // Build batch layout
    batch_atom_start_.clear();
    batch_atom_count_.clear();
    batch_nkb_.clear();
    batch_jkb_offset_.clear();
    max_nkb_batch_ = 0;
    max_batch_elements_ = 0;

    int atom_idx = 0;
    int cumulative_jkb = 0;
    while (atom_idx < total_atoms)
    {
        const int atoms_in_batch = std::min(batch_size, total_atoms - atom_idx);
        int nkb = 0;
        for (int i = atom_idx; i < atom_idx + atoms_in_batch; i++)
        {
            nkb += nproj_per_atom[i];
        }

        batch_atom_start_.push_back(atom_idx);
        batch_atom_count_.push_back(atoms_in_batch);
        batch_nkb_.push_back(nkb);
        batch_jkb_offset_.push_back(cumulative_jkb);

        cumulative_jkb += nkb;

        if (nkb > max_nkb_batch_)
        {
            max_nkb_batch_ = nkb;
        }

        const size_t batch_elements = static_cast<size_t>(nkb) * npwx;
        if (batch_elements > max_batch_elements_)
        {
            max_batch_elements_ = batch_elements;
        }

        atom_idx += atoms_in_batch;
    }

    nbatch_ = static_cast<int>(batch_atom_start_.size());
}

template <typename T>
void VKBBatchManager<T>::get_batch_info(int ibatch, int& atom_start, int& atom_end, int& nkb_batch) const
{
    if (ibatch < 0 || ibatch >= nbatch_)
    {
        throw std::out_of_range("VKBBatchManager::get_batch_info: ibatch out of range");
    }
    atom_start = batch_atom_start_[ibatch];
    atom_end = batch_atom_start_[ibatch] + batch_atom_count_[ibatch];
    nkb_batch = batch_nkb_[ibatch];
}

// Explicit template instantiations
template class VKBBatchManager<std::complex<float>>;
template class VKBBatchManager<std::complex<double>>;

} // namespace hamilt
