#include "constraint_inject_lcao.h"

#include "source_lcao/module_gint/gint_interface.h"

namespace constraint
{

std::vector<hamilt::HContainer<double>> ConstraintInjectLCAO::build(
    const std::vector<std::vector<double>>& cw,
    ModuleGint::GintInfo* gint_info)
{
    // The vlocal kernel reads the shared GintInfo; make sure it points at the
    // esolver's active instance (normally already set by the LCAO esolver).
    ModuleGint::Gint::set_gint_info(gint_info);

    std::vector<hamilt::HContainer<double>> W;
    W.reserve(cw.size());
    for (size_t alpha = 0; alpha < cw.size(); ++alpha)
    {
        W.push_back(gint_info->get_hr<double>());
        ModuleGint::cal_gint_vl(cw[alpha].data(), &W.back());
    }
    return W;
}

void ConstraintInjectLCAO::add_weighted(
    const std::vector<double>& mu,
    const std::vector<hamilt::HContainer<double>>& W,
    hamilt::HContainer<double>* H)
{
    // Branch A: no active constraints — nothing to add.
    if (W.empty() || mu.empty())
    {
        return;
    }
    for (size_t alpha = 0; alpha < W.size(); ++alpha)
    {
        // Branch B: zero multiplier — the contribution is a no-op by value.
        if (mu[alpha] == 0.0)
        {
            continue;
        }
        const double* w = W[alpha].get_wrapper();
        double* h = H->get_wrapper();
        for (size_t i = 0; i < H->get_nnr(); ++i)
        {
            h[i] += mu[alpha] * w[i];
        }
    }
}

} // namespace constraint
