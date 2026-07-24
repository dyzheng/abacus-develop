#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_unk/berryphase.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"

namespace pw_deltap {

namespace {
    bool s_active = false;
    std::vector<double> s_lambda;
    std::vector<int> s_constrain;
    double s_gamma_total = 0.0;   // cached total gamma from last computation
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

    set_deltap_pw_active(true);
    return false;
}

double compute_total_gamma_pw(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi_in,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    int nbands)
{
    if (gdir < 1 || gdir > 3) return 0.0;
    if (wfcpw == nullptr || psi_in == nullptr || rhopw == nullptr) return 0.0;

    berryphase bp;
    bp.direction = gdir;
    bp.GDIR = gdir;
    bp.set_kpoints(kv, gdir);

    if (bp.total_string == 0 || bp.nppstr < 2)
        return 0.0;

    double gamma_total = 0.0;
    for (int istr = 0; istr < bp.total_string; istr++)
    {
        gamma_total += bp.stringPhase(ucell, istr, nbands,
                                       wfcpw->npwk_max, psi_in, rhopw, wfcpw, kv);
    }

    s_gamma_total = gamma_total;
    return gamma_total;
}

} // namespace pw_deltap
