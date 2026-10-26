#ifndef OP_PW_ALLOC_H
#define OP_PW_ALLOC_H

#include "op_pw.h"

#include "source_base/matrix.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_psi/psi.h"

#include <complex>
#include <string>
#include <vector>

class UnitCell;
class K_Vectors;

namespace constraint
{
class WeightGrid;
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

} // namespace alloc

namespace hamilt
{

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

    virtual ~AllocConstraintPW() = default;

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

  private:
    const int* isk = nullptr;
    const UnitCell* ucell = nullptr;
    const ModulePW::PW_Basis_K* wfcpw = nullptr;
};

} // namespace hamilt

#endif
