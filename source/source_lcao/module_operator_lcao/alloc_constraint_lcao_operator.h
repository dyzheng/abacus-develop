#ifndef ALLOC_CONSTRAINT_LCAO_OPERATOR_H
#define ALLOC_CONSTRAINT_LCAO_OPERATOR_H

#include "source_lcao/module_operator_lcao/operator_lcao.h"

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

} // namespace hamilt

#endif
