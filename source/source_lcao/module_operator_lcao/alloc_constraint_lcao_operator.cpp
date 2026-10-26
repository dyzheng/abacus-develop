#include "source_lcao/module_operator_lcao/alloc_constraint_lcao_operator.h"

#include "source_base/tool_quit.h"
#include "source_estate/module_constraint/alloc_constraint_lcao.h"
#include "source_io/module_parameter/parameter.h"

#include <type_traits>

namespace hamilt
{

template <typename TK, typename TR>
void AllocConstraintLcaoOperator<TK, TR>::contributeHR()
{
    // Branch: S3 supports only the real Gamma-only HContainer storage.  The
    // input guard refuses other combinations, so complex instantiations are
    // compiled as explicit no-ops rather than carrying a silent alternative.
    if constexpr (!std::is_same<TR, double>::value)
    {
        return;
    }
    else
    {
        // Branch: real Gamma-only storage applies the sparse kernels directly
        // to the flat H(R) wrapper.
        alloc::AllocConstraintLcao& state
            = alloc::AllocConstraintLcao::instance();
        // Branch: the first reference SCF runs before the adapter has seen a
        // converged overlap matrix; no constraint is applied yet.
        // Branch: before the first reference SCF the target space and kernels
        // do not exist yet, so the unconstrained Hamiltonian is used.
        if (!state.active() || state.kernels().empty())
        {
            return;
        }
        // Guard: the operator must see the same multiplier count as kernels.
        // Guard: a multiplier/kernel count mismatch would silently omit or
        // mis-pair a constraint.
        if (state.mu_eff().size() != state.kernels().size())
        {
            ModuleBase::WARNING_QUIT(
                "AllocConstraintLcaoOperator::contributeHR",
                "allocation multiplier/kernel count mismatch");
        }
        const int spin = (PARAM.inp.nspin == 2) ? this->current_spin : 0;
        double* h = this->hR->get_wrapper();
        for (size_t alpha = 0; alpha < state.kernels().size(); ++alpha)
        {
            const double coefficient
                = state.mu_eff()[alpha] * state.injection_sign(
                      static_cast<int>(alpha), spin);
            // Branch: a zero multiplier contributes nothing.
            if (coefficient == 0.0)
            {
                continue;
            }
            const HContainer<double>& kernel = state.kernels()[alpha];
            // Guard: a layout mismatch would pair unrelated AO blocks.
            if (kernel.get_nnr() != this->hR->get_nnr())
            {
                ModuleBase::WARNING_QUIT(
                    "AllocConstraintLcaoOperator::contributeHR",
                    "allocation kernel/H(R) layout mismatch");
            }
            const double* k = kernel.get_wrapper();
            for (size_t i = 0; i < this->hR->get_nnr(); ++i)
            {
                h[i] += coefficient * k[i];
            }
        }
    }
}

template class AllocConstraintLcaoOperator<double, double>;
template class AllocConstraintLcaoOperator<std::complex<double>, double>;
template class AllocConstraintLcaoOperator<std::complex<double>, std::complex<double>>;

} // namespace hamilt
