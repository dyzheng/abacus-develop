#include "source_pw/module_pwdft/deltaspin_pw.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_estate/module_charge/charge_mixing.h"

namespace pw
{

bool run_deltaspin_lambda_loop(const int iter,
                               const double drho,
                               const Input_para& inp)
{
    fprintf(stderr, "[DS-LAMBDA] iter=%d drho=%.6e sc_mag=%d\n", iter, drho, inp.sc_mag_switch);
    fflush(stderr);
    /// Return false if DeltaSpin is not enabled
    if (!inp.sc_mag_switch)
    {
        return false;
    }

    /// Get the singleton instance of SpinConstrain
    spinconstrain::SpinConstrain<std::complex<double>>& sc
        = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();

    fprintf(stderr, "[DS-LAMBDA] mag_converged=%d\n", sc.mag_converged());
    fflush(stderr);
    /// Case 1: Magnetic moments not yet converged and SCF is close to convergence.
    /// This is the first time we enter the lambda loop after SCF is nearly converged.
    if (!sc.mag_converged() && drho > 0 && drho < inp.sc_scf_thr)
    {
        fprintf(stderr, "[DS-LAMBDA] Case 1: entering lambda loop\n");
        fflush(stderr);
        /// Optimize lambda to get target magnetic moments
        sc.run_lambda_loop(iter);
        sc.set_mag_converged(true);
        fprintf(stderr, "[DS-LAMBDA] Case 1: lambda loop done\n");
        fflush(stderr);
        return true;
    }
    /// Case 2: Magnetic moments already converged in previous iteration.
    /// The lambda values and charge density were already updated in Case 1.
    /// Skip the solver so the SCF can converge with the existing charge density.
    /// Re-running the lambda loop would re-update the charge density and disrupt SCF mixing.
    else if (sc.mag_converged())
    {
        fprintf(stderr, "[DS-LAMBDA] Case 2: mag already converged, skip solver\n");
        fflush(stderr);
        return true;
    }

    /// Default: run the normal solver
    return false;
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
