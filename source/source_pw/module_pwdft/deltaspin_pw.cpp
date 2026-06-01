#include "source_pw/module_pwdft/deltaspin_pw.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_estate/module_charge/charge_mixing.h"
#include "source_io/module_parameter/parameter.h"

namespace pw
{

bool run_deltaspin_lambda_loop(const int iter,
                               const double drho,
                               const Input_para& inp,
                               Charge_Mixing* p_chgmix)
{
    if (!inp.sc_mag_switch)
    {
        return false;
    }

    spinconstrain::SpinConstrain<std::complex<double>>& sc
        = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();

    if (inp.sc_lambda_strategy == "linear_scan")
    {
        sc.set_drho(drho);
        sc.run_lambda_linear_scan(iter);
        return true;
    }

    // Helper lambda for init_lambda_mixing
    auto try_init_lambda_mixing = [&]() {
        if (inp.sc_mixing_lambda_beta != 0.0
            && !sc.is_lambda_mixing_enabled()
            && sc.get_nat() > 0)
        {
            double beta = inp.sc_mixing_lambda_beta;
            if (beta < 0.0)
            {
                beta = p_chgmix->get_mixing_beta();
            }
            if (beta > 0.0)
            {
                sc.init_lambda_mixing(beta);
            }
        }
    };

    bool ran_loop = false;

    if (inp.sc_scf_thr_mode == "immediate")
    {
        if (iter >= 1)
        {
            // immediate mode: init mixing only when drho < mixing_restart
            if (drho > 0 && drho < inp.mixing_restart)
            {
                try_init_lambda_mixing();
            }
            sc.set_drho(drho);
            sc.run_lambda_loop(iter - 1);
            if (!sc.mag_converged()) { sc.set_mag_converged(true); }
            ran_loop = true;
        }
    }
    else if (inp.sc_scf_thr_mode == "off")
    {
        return false;
    }
    else // "threshold"
    {
        // threshold mode: never init/mix lambda mixing
        if (!sc.mag_converged() && drho > 0 && drho < inp.sc_scf_thr)
        {
            sc.set_drho(drho);
            sc.run_lambda_loop(iter - 1);
            sc.set_mag_converged(true);
            ran_loop = true;
        }
        else if (sc.mag_converged())
        {
            sc.set_drho(drho);
            sc.run_lambda_loop(iter - 1);
            ran_loop = true;
        }
    }

    if (ran_loop && sc.is_lambda_mixing_enabled()
        && inp.sc_lambda_strategy != "linear_scan"
        && inp.sc_scf_thr_mode == "immediate")
    {
        sc.mix_lambda();
    }

    return ran_loop;
}

void check_deltaspin_oscillation(const int iter,
                                 const double drho,
                                 Charge_Mixing* p_chgmix,
                                 const Input_para& inp)
{
    /// Return if DeltaSpin is not enabled
    if (!inp.sc_mag_switch)
    {
        return;
    }

    /// Get the singleton instance of SpinConstrain
    spinconstrain::SpinConstrain<std::complex<double>>& sc
        = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();

    /// Check if higher magnetization precision is needed
    if (!sc.higher_mag_prec)
    {
        /// Detect SCF oscillation
        sc.higher_mag_prec = p_chgmix->if_scf_oscillate(iter, drho, inp.sc_os_ndim, inp.scf_os_thr);

        /// If oscillation detected, set mixing restart step for next iteration
        if (sc.higher_mag_prec)
        {
            p_chgmix->mixing_restart_step = iter + 1;
        }
    }
}

}
