#ifndef ALLOC_CONSTRAINT_LCAO_OPERATOR_H
#define ALLOC_CONSTRAINT_LCAO_OPERATOR_H

#include <string>
#include <vector>

#include "source_base/matrix.h"
#include "source_lcao/module_operator_lcao/operator_lcao.h"

class UnitCell;
class Grid_Driver;
class LCAO_Orbitals;
class TwoCenterBundle;

namespace hamilt
{

/**
 * @brief Add the S3 unified-allocation constraint to H(R).
 *
 * The kernels K_i are geometry-fixed and built once by
 * alloc::AllocConstraintLcao::build().  Each SCF iteration only applies
 *   H_sigma(R) += sum_i mu_i * sign_{i,sigma} * K_i(R).
 */
template <typename TK, typename TR>
class AllocConstraintLcaoOperator : public OperatorLCAO<TK, TR>
{
  public:
    AllocConstraintLcaoOperator(
        HS_Matrix_K<TK>* hsk_in,
        const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
        HContainer<TR>* hR_in)
        : OperatorLCAO<TK, TR>(hsk_in, kvec_d_in, hR_in)
    {
        this->cal_type = calculation_type::lcao_alloc_lambda;
    }

    void contributeHR() override;
};

/**
 * @brief Add the explicit target-orbital derivative to the S3 force.
 *
 * The S3 force in ConstraintLoop is the frozen-weight contribution
 * -lambda Tr(D Q_c dW Q_c).  This routine adds the missing
 * -lambda Tr(D dK/dS dS) part, equivalently -Tr(E dS), through the existing
 * two-center overlap derivative path.  It only acts on the Gamma-only real
 * allocation state and accumulates into `force` (Ry/Bohr).
 */
bool cal_alloc_constraint_dS_force(
    const UnitCell& ucell,
    const Grid_Driver& gd,
    const LCAO_Orbitals& orb,
    const TwoCenterBundle& two_center_bundle,
    const std::vector<const HContainer<double>*>& dmr,
    const int nspin,
    const std::vector<double>& mu,
    ModuleBase::matrix& force,
    std::string& error);

} // namespace hamilt

#endif
