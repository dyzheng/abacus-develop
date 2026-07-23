#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"

namespace pw_deltap {

namespace {
    bool s_active = false;
    std::vector<double> s_lambda;
    std::vector<int> s_constrain;
}

void set_deltap_pw_lambda(const std::vector<double>& lambda,
                          const std::vector<int>& constrain)
{
    s_lambda = lambda;
    s_constrain = constrain;
}

const std::vector<double>& get_deltap_pw_lambda()
{
    return s_lambda;
}

const std::vector<int>& get_deltap_pw_constrain()
{
    return s_constrain;
}

void set_deltap_pw_active(bool active)
{
    s_active = active;
}

bool is_deltap_pw_active()
{
    return s_active;
}

bool run_deltap_lambda_loop(const int iter,
                            const double drho,
                            const Input_para& inp)
{
    if (!inp.deltap_switch)
        return false;

    // Phase A: read lambda from STRU (no gamma computation yet).
    // The lambda values come from dp_target in atom_spec, parsed in
    // ESolver_KS_PW::before_all_runners and stored via set_deltap_pw_lambda().
    // For now, just propagate the STRU-specified lambda and activate the operator.
    set_deltap_pw_active(true);
    return false; // don't skip solver — no inner loop yet
}

} // namespace pw_deltap
