#ifndef OP_PW_ALLOC_H
#define OP_PW_ALLOC_H

#include "op_pw.h"

#include "source_base/matrix.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_psi/psi.h"

#include <complex>
#include <memory>
#include <string>
#include <vector>

class UnitCell;
class K_Vectors;

namespace constraint
{
class WeightGrid;
struct ChannelProfile;
}

namespace alloc
{

/**
 * @brief Read the PW allocation observables n_i/m_i = Tr(D K_i).
 *
 * The PW analogue of AllocConstraintLcao::project_allocations() evaluates
 * <psi|K_i|psi> for every occupied band.  K_i is the same low-rank operator
 * injected by AllocConstraintPW: its local weight plus the target-space
 * complement correction and the owned target projector.  Results are summed
 * over the k-points in each pool and then over pools.
 */
bool compute_pw_alloc_observables(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<double>& n_alloc,
    std::vector<double>& m_alloc,
    std::string& error);

/**
 * @brief Form the frozen-target complement density Q_c D Q_c on the PW grid.
 *
 * The returned rho_c[spin][ir] uses the same electron-density normalization as
 * ElecStatePW::psiToRho (occupation / omega).  It is rank-local in the same
 * way as the weight grid, so the shared constraint_force() kernel performs the
 * pool reduction.  This is the S3 frozen-target force approximation for PW:
 * the target projectors are treated as geometry-fixed and only the weight
 * derivative enters the force.
 */
bool compute_pw_alloc_complement_density(
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<std::vector<double>>& rho_c,
    std::string& error);

/**
 * @brief Complete geometric derivative of the allocation target projectors.
 *
 * This is the missing dQ/dP half of the PW allocation force.  It contracts
 * the onsite-projector displacement derivative with the wavefunction and
 * with W_i Q_c|psi>, adds the Fréchet derivative of ST^{-1/2}, and returns
 * the force contribution not covered by compute_pw_alloc_complement_density()
 * (which owns only Q_c dW_i Q_c).
 *
 * force is accumulated in Ry/Bohr and reduced over every k/band pool.  The
 * implementation is the collinear scalar-CPU closure and rejects spinors.
 */
bool compute_pw_alloc_target_force(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    ModuleBase::matrix& force,
    std::string& error);

/**
 * @brief Complete geometric strain derivative of the PW target projectors.
 *
 * The strain counterpart of compute_pw_alloc_target_force(): it contracts
 * the onsite-projector strain derivative with the occupied bands and the
 * W_i Q_c vectors, adds the ST^{-1/2} Fréchet strain derivative, and returns
 * the missing dQ/deps target contribution in Ry/Bohr^3.  The explicit
 * Q_c dW_i/deps Q_c part is supplied by ConstraintLoop::compute_stress().
 */
bool compute_pw_alloc_target_stress(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    const std::vector<double>& mu,
    const std::vector<constraint::ChannelProfile>& channels,
    ModuleBase::matrix& stress,
    std::string& error);

} // namespace alloc

namespace hamilt
{

class AllocPwCache;

#ifndef ALLOCPW_TEMPLATE_H
#define ALLOCPW_TEMPLATE_H

template <class T> class AllocConstraintPW : public T {};

#endif

/**
 * @brief PW operator for the unified allocation constraint.
 *
 * In one band application this operator forms
 *
 *   V|psi> - P V|psi> - V P|psi> + P V P|psi> + sum_i lambda_i Q_i|psi>,
 *
 * with V = sum_i lambda_i w_i and P the orthonormal target-shell projector.
 * The operator is deliberately PW-native: no full-basis matrix is built.
 * The first production closure supports collinear complex<double> CPU bands.
 */
template <typename T, typename Device>
class AllocConstraintPW<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  public:
    AllocConstraintPW(const int* isk_in,
                      const UnitCell* ucell_in,
                      const ModulePW::PW_Basis_K* wfcpw_in);

    template <typename T_in, typename Device_in = Device>
    explicit AllocConstraintPW(
        const AllocConstraintPW<OperatorPW<T_in, Device_in>>* op);

    virtual ~AllocConstraintPW();

    virtual void init(const int ik_in) override;

    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T* tmpsi_in,
                     T* tmhpsi,
                     const int ngk_ik = 0,
                     const bool is_first_node = false) const override;

    const int* get_isk() const { return isk; }
    const UnitCell* get_ucell() const { return ucell; }
    const ModulePW::PW_Basis_K* get_wfcpw() const { return wfcpw; }

    /**
     * @brief Enable fusion of the allocation local potential into Veff.
     *
     * grid_size is the real-space length of the Veff potential array.  The
     * fast path is accepted only for the double-precision CPU build; the
     * allocation operator then omits its own V|psi> transform and Veff adds
     * V|psi> while it already has the wavefunction in real space.
     */
    void enable_fold_potential(const int grid_size);

    /**
     * @brief Return the active allocation potential for one spin channel.
     *
     * can_fold is true only when Veff can safely consume this array in its
     * existing real-space multiply.  The pointer is owned by the operator
     * cache and remains valid until the next multiplier or geometry change.
     */
    const std::vector<double>* get_fold_potential(const int spin,
                                                  bool& can_fold) const;

  private:
    const int* isk = nullptr;
    const UnitCell* ucell = nullptr;
    const ModulePW::PW_Basis_K* wfcpw = nullptr;
    // The cache is mutable because OperatorPW::act is const but must retain
    // geometry- and multiplier-keyed target products across band applications.
    mutable std::unique_ptr<AllocPwCache> cache;
    // Set by Veff during Hamiltonian construction.  A zero grid means no
    // compatible Veff consumer was registered, so act() keeps the standalone
    // local-potential transform.
    int fold_potential_grid = 0;
};

} // namespace hamilt

#endif
