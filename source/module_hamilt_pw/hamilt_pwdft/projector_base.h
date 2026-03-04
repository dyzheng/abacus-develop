#ifndef PROJECTOR_BASE_H
#define PROJECTOR_BASE_H

#include <cstddef>

namespace hamilt
{

/**
 * @brief Abstract base class for nonlocal pseudopotential projector operations.
 *
 * This class defines the interface for computing nonlocal pseudopotential
 * contributions to the Hamiltonian-wavefunction product (H|psi>). The nonlocal
 * contribution is computed in two steps:
 *   1. Compute projection coefficients: becp = <beta|psi>
 *   2. Apply the D matrix and accumulate: hpsi += |beta> * D * becp
 *
 * The default implementation uses reciprocal-space VKB projectors via GEMM
 * operations. This base class allows alternative implementations (e.g.,
 * real-space projectors) to be substituted without modifying the Nonlocal
 * operator.
 *
 * @tparam T The element type, typically std::complex<float> or std::complex<double>.
 */
template <typename T>
class ProjectorBase
{
  public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~ProjectorBase() = default;

    /**
     * @brief Compute projection coefficients becp = <beta|psi>.
     *
     * Projects the wavefunctions onto the nonlocal beta projectors to obtain
     * the overlap coefficients. In reciprocal space, this is typically
     * performed as a matrix multiply: becp = vkb^H * psi.
     *
     * @param[in] psi Pointer to the wavefunction array, dimensions [nbands][npol*npw].
     * @param[out] becp Pointer to the output projection coefficient array.
     * @param[in] nbands Number of bands (wavefunctions) to process.
     * @param[in] npw Number of plane waves for the current k-point.
     * @param[in] max_npw Leading dimension for the plane-wave index (max over k-points).
     * @param[in] npol Number of spinor components (1 for collinear, 2 for noncollinear).
     */
    virtual void compute_becp(const T* psi, T* becp, int nbands, int npw, int max_npw, int npol) = 0;

    /**
     * @brief Apply the D matrix to becp and accumulate into hpsi.
     *
     * Computes the nonlocal contribution: hpsi += |beta> * D * becp.
     * In reciprocal space, this is typically: hpsi += vkb * (D * becp).
     * The D matrix (deeq) encodes the pseudopotential strength and may
     * include spin-orbit coupling terms.
     *
     * @param[in] becp Pointer to the projection coefficient array from compute_becp.
     * @param[in,out] hpsi Pointer to the Hamiltonian-psi product array to accumulate into.
     * @param[in] nbands Number of bands (wavefunctions) to process.
     * @param[in] npw Number of plane waves for the current k-point.
     * @param[in] max_npw Leading dimension for the plane-wave index (max over k-points).
     * @param[in] npol Number of spinor components (1 for collinear, 2 for noncollinear).
     */
    virtual void apply_deeq_and_accumulate(const T* becp, T* hpsi, int nbands, int npw, int max_npw, int npol) = 0;

    /**
     * @brief Get the current GPU memory usage of the projector in bytes.
     *
     * Returns the total number of bytes of GPU device memory currently
     * allocated by this projector instance (e.g., for VKB arrays, D matrices,
     * workspace buffers).
     *
     * @return The number of bytes of GPU memory in use.
     */
    virtual size_t get_memory_bytes() const = 0;
};

} // namespace hamilt

#endif // PROJECTOR_BASE_H
