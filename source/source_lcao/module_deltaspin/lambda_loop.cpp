#include "spin_constrain.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

#include "basic_funcs.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_gint/gint_interface.h"

// lambda = initial_lambda + delta_lambda/(spin2 - spin1) * (target_spin - spin1)
/*inline void next_lambda(std::vector<ModuleBase::Vector3<double>>& initial_lambda,
                        std::vector<ModuleBase::Vector3<double>>& delta_lambda,
                        std::vector<ModuleBase::Vector3<double>>& lambda,
                        std::vector<ModuleBase::Vector3<double>>& spin1,
                        std::vector<ModuleBase::Vector3<double>>& spin2,
                        std::vector<ModuleBase::Vector3<double>>& target_spin)
{
    for (int ia = 0; ia < lambda.size(); ia++)
    {
        for (int ic = 0; ic < 3; ic++)
        {
            lambda[ia][ic] = initial_lambda[ia][ic] + delta_lambda[ia][ic] / (spin2[ia][ic] - spin1[ia][ic]) * (target_spin[ia][ic] - spin1[ia][ic]);
        }
    }
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_lambda_loop(int outer_step)
{
    // init parameters
    int nat = this->get_nat();
    std::vector<ModuleBase::Vector3<double>> initial_lambda(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> delta_lambda(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> spin1(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> spin2(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> delta_spin(nat, 0.0);
    // current lambda is this->lambda_
    // current spin is this->Mi_
    // target spin is this->target_mag_
    // loop to optimize lambda to get target spin
    int step = -1;
    do
    {
        // set initial lambda
        where_fill_scalar_else_2d(this->constrain_, 0, 0.0, this->lambda_, initial_lambda);
        // save current spin to spin1 if step > 0
        if (step > 0)
        {
            spin1 = this->Mi_;
        }
        // calculate current spin
        this->cal_mw_from_lambda(step);
        // save current spin to spin2
        spin2 = this->Mi_;
        // calculate delta_spin = target_spin - spin
        subtract_2d(this->target_mag_, spin2, delta_spin);
        // check RMS error and stop if needed
        // calculate RMS error
        double sum = 0.0;
        for (int ia = 0; ia < nat; ia++)
        {
            for (int ic = 0; ic < 3; ic++)
            {
                sum += std::pow(delta_spin[ia][ic],2);
            }
        }
        double rms_error = std::sqrt(sum/nat);
        std::cout << "RMS error = " << rms_error <<" in step:" <<step << std::endl;
        // check RMS error and stop if needed
        if(rms_error < 1e-5)
        {
            std::cout<<"success"<<std::endl;
            break;
        }
        // calculate delta_lambda
        if(1)//step == 0)
        {
            for(int ia = 0; ia < nat; ia++)
            {
                for(int ic = 2; ic < 3; ic++)
                {
                    delta_lambda[ia][ic] = 0.01;//- delta_spin[ia][ic] / 10.0;
                    this->lambda_[ia][ic] = initial_lambda[ia][ic] + delta_lambda[ia][ic];
                    std::cout<<__LINE__<<"lambda["<<ia<<"] = "<<this->lambda_[ia][ic]<<std::endl;
                }
            }
        }
        else
        {
            //calculate next lambda
            next_lambda(initial_lambda, delta_lambda, this->lambda_, spin1, spin2, this->target_mag_);
            // calculate delta_lambda = this->lambda - initial_lambda
            subtract_2d(this->lambda_, initial_lambda, delta_lambda);
        }
        step++;
    } while (step < this->nsc_);
    
}*/


template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_lambda_loop(
        int outer_step,
		bool rerun)
{
    // init controlling parameters
    int nat = this->get_nat();
    int ntype = this->get_ntype();
    std::vector<ModuleBase::Vector3<double>> initial_lambda(nat,0.0);
    std::vector<ModuleBase::Vector3<double>> delta_lambda(nat,0.0);
    // set nu, dnu and dnu_last_step
    std::vector<ModuleBase::Vector3<double>> dnu(nat, 0.0), dnu_last_step(nat, 0.0);
    // two controlling temp variables
    std::vector<ModuleBase::Vector3<double>> temp_1(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> spin(nat, 0.0), delta_spin(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> search(nat, 0.0), search_old(nat, 0.0);
    std::vector<ModuleBase::Vector3<double>> new_spin(nat, 0.0), spin_plus(nat, 0.0);

    double alpha_opt, alpha_plus;
    double beta = 0.0, g = 0.0, mean_error = 0.0, mean_error_old = 0.0, rms_error = 0.0;

    double alpha_trial = this->alpha_trial_;

    const double zero = 0.0;
    const double one = 1.0;

#ifdef __MPI
	auto iterstart = MPI_Wtime();
#else
	auto iterstart = std::chrono::system_clock::now();
#endif

    double inner_loop_duration = 0.0;

    this->print_header();
    // lambda loop
    for (int i_step = -1; i_step < this->nsc_; i_step++)
    {
        double duration = 0.0;
        if (i_step == -1)
        {

            this->cal_mw_from_lambda(i_step);
            spin = this->Mi_;
            where_fill_scalar_else_2d(this->constrain_, 0, zero, this->lambda_, initial_lambda);
            print_2d("initial lambda (eV/uB): ", initial_lambda, this->nspin_, ModuleBase::Ry_to_eV);
            print_2d("initial spin (uB): ", spin, this->nspin_);
            print_2d("target spin (uB): ", this->target_mag_, this->nspin_);
            i_step++;
        }
        else
        {
            where_fill_scalar_else_2d(this->constrain_, 0, zero, delta_lambda, delta_lambda);
            add_scalar_multiply_2d(initial_lambda, delta_lambda, one, this->lambda_);

            this->cal_mw_from_lambda(i_step);

            new_spin = this->Mi_;
            bool GradLessThanBound = this->check_gradient_decay(new_spin, spin, delta_lambda, dnu_last_step);
            if (i_step >= this->nsc_min_ && GradLessThanBound)
            {
                add_scalar_multiply_2d(initial_lambda, dnu_last_step, one, this->lambda_);
                this->update_psi_charge(dnu_last_step.data());
#ifdef __MPI
		        duration = (double)(MPI_Wtime() - iterstart);
#else
			    duration =
                    (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now()
                    - iterstart)).count() / static_cast<double>(1e6);
#endif
                inner_loop_duration += duration;
                std::cout << "Total TIME(s) = " << inner_loop_duration << std::endl;
                this->print_termination();
                break;
            }
            spin = new_spin;
        }
        // continue the lambda loop
        subtract_2d(spin, this->target_mag_, delta_spin);
        where_fill_scalar_2d(this->constrain_, 0, zero, delta_spin);
        search = delta_spin;
        for (int ia = 0; ia < nat; ia++)
        {
            for (int ic = 0; ic < 3; ic++)
            {
                temp_1[ia][ic] = std::pow(delta_spin[ia][ic],2);
            }
        }
        mean_error = sum_2d(temp_1) / nat;
        rms_error = std::sqrt(mean_error);
        if(i_step == 0)
        {
            // set current_sc_thr_ to max(rms_error * sc_drop_thr, this->sc_thr_)
            this->current_sc_thr_ = std::max(rms_error * this->sc_drop_thr_, this->sc_thr_);
        }
#ifdef __MPI
			duration = (double)(MPI_Wtime() - iterstart);
#else
			duration =
               (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now()
                - iterstart)).count() / static_cast<double>(1e6);
#endif
        inner_loop_duration += duration;
        if (this->check_rms_stop(outer_step, i_step, rms_error, duration, inner_loop_duration))
        {
            //add_scalar_multiply_2d(initial_lambda, dnu_last_step, 1.0, this->lambda_);
            this->update_psi_charge(dnu_last_step.data(), rerun);
            if(PARAM.inp.basis_type == "pw")
            {
                //double check Atomic spin moment
                this->cal_Mi_pw();
                subtract_2d(this->Mi_, this->target_mag_, delta_spin);
                where_fill_scalar_2d(this->constrain_, 0, zero, delta_spin);
                search = delta_spin;
                for (int ia = 0; ia < nat; ia++)
                {
                    for (int ic = 0; ic < 3; ic++)
                    {
                        temp_1[ia][ic] = std::pow(delta_spin[ia][ic],2);
                    }
                }
                mean_error = sum_2d(temp_1) / nat;
                rms_error = std::sqrt(mean_error);
                std::cout<<"Current RMS: "<<rms_error<<std::endl;
                if(rms_error > this->current_sc_thr_ * 10 && rerun == true && this->higher_mag_prec == true)
                {
                    std::cout<<"Error: RMS error is too large, rerun the loop"<<std::endl;
                    this->run_lambda_loop(outer_step, false);
                }
            }
            break;
        }
        // If a non-BFGS strategy is active, use it instead of the traditional BFGS update
        if (this->strategy_)
        {
            auto result = this->strategy_->update_lambda(
                this->lambda_, this->Mi_, this->target_mag_, this->constrain_,
                this->sc_thr_, i_step, nat);
            if (result.status == "converged")
            {
                this->update_psi_charge(nullptr, rerun);
                if(PARAM.inp.basis_type == "pw")
                {
                    this->cal_Mi_pw();
                    subtract_2d(this->Mi_, this->target_mag_, delta_spin);
                    where_fill_scalar_2d(this->constrain_, 0, zero, delta_spin);
                    for (int ia = 0; ia < nat; ia++)
                    {
                        for (int ic = 0; ic < 3; ic++)
                        {
                            temp_1[ia][ic] = std::pow(delta_spin[ia][ic],2);
                        }
                    }
                    rms_error = std::sqrt(sum_2d(temp_1) / nat);
                }
                break;
            }
            if (result.status.find("fallback") != std::string::npos)
            {
                // fallback: let the outer SCF continue and retry lambda loop
                this->update_psi_charge(nullptr, rerun);
                break;
            }
            // Strategy updated lambda, continue to next iteration for a fresh SCF solve
            continue;
        }
#ifdef __MPI
		iterstart = MPI_Wtime();
#else
		iterstart = std::chrono::system_clock::now();
#endif
        if (i_step >= 2)
        {
            beta = mean_error / mean_error_old;
            add_scalar_multiply_2d(search, search_old, beta, search);
        }
        /// check if restriction is needed
        this->check_restriction(search, alpha_trial);

        dnu_last_step = dnu;
        add_scalar_multiply_2d(dnu, search, alpha_trial, dnu);
        delta_lambda = dnu;

        where_fill_scalar_else_2d(this->constrain_, 0, zero, delta_lambda, delta_lambda);
        add_scalar_multiply_2d(initial_lambda, delta_lambda, one, this->lambda_);

        this->cal_mw_from_lambda(i_step, delta_lambda.data());

        spin_plus = this->Mi_;

        alpha_opt = this->cal_alpha_opt(spin, spin_plus, alpha_trial);
        /// check if restriction is needed
        this->check_restriction(search, alpha_opt);

        alpha_plus = alpha_opt - alpha_trial;
        scalar_multiply_2d(search, alpha_plus, temp_1);
        add_scalar_multiply_2d(dnu, temp_1, one, dnu);
        delta_lambda = dnu;

        search_old = search;
        mean_error_old = mean_error;

        g = 1.5 * std::abs(alpha_opt) / alpha_trial;
        if (g > 2.0)
        {
            g = 2;
        }
        else if (g < 0.5)
        {
            g = 0.5;
        }
        alpha_trial = alpha_trial * pow(g, 0.7);
    }

    return;
}

#ifdef __LCAO
#include "source_lcao/module_operator_lcao/dspin_lcao.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_estate/elecstate_tools.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/module_external/blas_connector.h"
#include "source_base/module_external/scalapack_connector.h"

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_lambda_loop_lcao(int outer_step)
{
    const int nat = this->get_nat();
    const int nks = this->kv_.get_nks();
    const int nk = nks / 2;
    psi::Psi<std::complex<double>>* psi_t = static_cast<psi::Psi<std::complex<double>>*>(this->psi);
    const int nbands = this->ParaV->get_nbands();
    const double alpha_damp = 0.8;
    const int max_inner_iter = this->nsc_;

    this->print_header();

    // ── Phase 1: Full diagonalization to get C_k, e_k, Mi ──
    this->cal_mw_from_lambda(-1);
    std::vector<ModuleBase::Vector3<double>> spin(nat);
    spin = this->Mi_;

    std::vector<ModuleBase::Vector3<double>> initial_lambda(nat, 0.0);
    const double zero = 0.0;
    where_fill_scalar_else_2d(this->constrain_, 0, zero, this->lambda_, initial_lambda);

    print_2d("initial lambda (eV/uB): ", initial_lambda, this->nspin_, ModuleBase::Ry_to_eV);
    print_2d("initial spin (uB): ", spin, this->nspin_);
    print_2d("target spin (uB): ", this->target_mag_, this->nspin_);

    // Check initial convergence
    std::vector<ModuleBase::Vector3<double>> delta_spin(nat, 0.0);
    subtract_2d(spin, this->target_mag_, delta_spin);
    where_fill_scalar_2d(this->constrain_, 0, zero, delta_spin);
    double rms_error = 0.0;
    {
        double sum = 0.0;
        for (int ia = 0; ia < nat; ia++)
            for (int ic = 0; ic < 3; ic++)
                sum += std::pow(delta_spin[ia][ic], 2);
        rms_error = std::sqrt(sum / nat);
    }
    this->current_sc_thr_ = std::max(rms_error * this->sc_drop_thr_, this->sc_thr_);

    if (rms_error < this->current_sc_thr_)
    {
        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- 0"
                  << "       RMS = " << rms_error << std::endl;
        std::cout << "Meet convergence criterion ( < " << this->current_sc_thr_ << " ), exit." << std::endl;
        this->print_termination();
        this->pelec->psiToRho(*psi_t);
        return;
    }

    // ── Phase 2: Compute analytical chi (for first step only) ──
    // P_I_sub needed for analytical chi calculation
    auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(this->p_operator);
    std::vector<std::vector<std::vector<std::complex<double>>>> PI_sub(nks);
    for (int ik = 0; ik < nks; ik++)
    {
        psi_t->fix_k(ik);
        dspin_op->cal_PI_sub(this->kv_.kvec_d[ik], psi_t->get_pointer(), nbands, PI_sub[ik]);
    }

    // Analytical Jacobian: chi_I = dM_I^z / dlambda_I
    std::vector<double> chi(nat, 0.0);
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_[iat].z == 0) { continue; }
        double chi_val = 0.0;
        for (int ik = 0; ik < nks; ik++)
        {
            if (PI_sub[ik][iat].empty()) { continue; }
            const auto& P = PI_sub[ik][iat];
            const double wk = this->pelec->klist->wk[ik];
            for (int n = 0; n < nbands; n++)
            {
                const double fn = this->pelec->wg(ik, n) / wk;
                for (int m = n + 1; m < nbands; m++)
                {
                    const double fm = this->pelec->wg(ik, m) / wk;
                    const double de = this->pelec->ekb(ik, n) - this->pelec->ekb(ik, m);
                    if (std::abs(de) < 1e-10) { continue; }
                    const double P_nm_sq = std::norm(P[n * nbands + m]);
                    chi_val += 2.0 * wk * (fn - fm) * P_nm_sq / de;
                }
            }
        }
        chi[iat] = chi_val;
    }

    // ── Phase 3-6: Newton iteration with full diagonalization + secant chi update ──
    std::vector<ModuleBase::Vector3<double>> lambda_old(nat);
    std::vector<ModuleBase::Vector3<double>> Mi_old(nat);

    for (int inner = 0; inner < max_inner_iter; inner++)
    {
        // Save old state for secant update
        lambda_old = this->lambda_;
        Mi_old = spin;

        // Newton step: delta_lambda = alpha_damp * (target - current) / chi
        const double lambda_max = this->restrict_current_;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_[iat].z == 0) { continue; }
            double chi_clamped = chi[iat];
            if (std::abs(chi_clamped) < 0.1)
            {
                chi_clamped = (chi_clamped >= 0) ? 0.1 : -0.1;
            }
            double delta_lambda_z = alpha_damp * (this->target_mag_[iat].z - spin[iat].z) / chi_clamped;
            // Clamp delta_lambda to sccut
            if (std::abs(delta_lambda_z) > lambda_max)
            {
                delta_lambda_z = (delta_lambda_z > 0) ? lambda_max : -lambda_max;
            }
            this->lambda_[iat].z = initial_lambda[iat].z + delta_lambda_z;
        }

        // Full diagonalization to get real Mi (SCF self-consistent)
        this->cal_mw_from_lambda(inner);
        spin = this->Mi_;

        // Secant chi update: chi = (Mi_new - Mi_old) / (lambda_new - lambda_old)
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_[iat].z == 0) { continue; }
            double dlambda = this->lambda_[iat].z - lambda_old[iat].z;
            double dMi = spin[iat].z - Mi_old[iat].z;
            if (std::abs(dlambda) > 1e-10)
            {
                // Blend: use secant if stable, otherwise keep analytical
                double chi_secant = dMi / dlambda;
                // If secant chi has same sign and reasonable magnitude, use it
                if (chi_secant * chi[iat] > 0 && std::abs(chi_secant) > 0.01 && std::abs(chi_secant) < 100.0)
                {
                    chi[iat] = chi_secant;
                }
            }
        }

        // Check convergence
        subtract_2d(spin, this->target_mag_, delta_spin);
        where_fill_scalar_2d(this->constrain_, 0, zero, delta_spin);
        {
            double sum = 0.0;
            for (int ia = 0; ia < nat; ia++)
                for (int ic = 0; ic < 3; ic++)
                    sum += std::pow(delta_spin[ia][ic], 2);
            rms_error = std::sqrt(sum / nat);
        }

        std::cout << "Step (Outer -- Inner) =  " << outer_step << " -- " << std::left << std::setw(5) << inner + 1
                  << "       RMS = " << rms_error << std::endl;

        if (rms_error < this->current_sc_thr_)
        {
            std::cout << "Meet convergence criterion ( < " << this->current_sc_thr_ << " ), exit." << std::endl;
            break;
        }
    }

    this->print_termination();

    // ── Phase 7: Update DM/charge from current psi ──
    elecstate::cal_dm_psi(this->ParaV, this->pelec->wg, *psi_t, *this->dm_);
    this->dm_->cal_DMR();

    int nspin = PARAM.inp.nspin;
    if (PARAM.inp.nspin == 4) { nspin = 1; }
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        ModuleBase::GlobalFunc::ZEROS(this->pelec->charge->rho[is], this->pelec->charge->nrxx);
    }
    ModuleGint::cal_gint_rho(this->dm_->get_DMR_vector(), nspin, this->pelec->charge->rho);
    this->pelec->charge->renormalize_rho();
}
#endif // __LCAO
