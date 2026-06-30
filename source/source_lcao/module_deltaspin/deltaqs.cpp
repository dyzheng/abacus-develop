#include "spin_constrain.h"

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "basic_funcs.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/constants.h"

#ifdef __LCAO
#include "source_lcao/module_operator_lcao/dspin_lcao.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_hsolver/hsolver_lcao.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_estate/elecstate_tools.h"
#include "source_base/parallel_reduce.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_lcao/module_deltaqs/upf_valence_parser.h"
#endif

template <typename TK>
void spinconstrain::SpinConstrain<TK>::init_deltaqs(
    const UnitCell& ucell,
    bool charge_switch,
    const std::string& qs_mode,
    const std::string& charge_mode,
    double sc_charge_thr,
    double charge_alpha_trial,
    double charge_sccut,
    bool ground_state_search,
    int outer_max_iter,
    double outer_thr,
    bool gradient_output)
{
    int nat = this->get_nat();

    this->charge_constraint_enabled_ = charge_switch;
    this->qs_mode_ = qs_mode;
    this->charge_mode_ = charge_mode;
    this->sc_charge_thr_ = sc_charge_thr;
    this->charge_alpha_trial_ = charge_alpha_trial / ModuleBase::Ry_to_eV;
    this->charge_restrict_current_ = charge_sccut / ModuleBase::Ry_to_eV;
    this->ground_state_search_ = ground_state_search;
    this->outer_max_iter_ = outer_max_iter;
    this->outer_thr_ = outer_thr;
    this->gradient_output_ = gradient_output;

    this->mu_.resize(nat, 0.0);
    this->target_charge_.resize(nat, 0.0);
    this->Ni_.resize(nat, 0.0);
    this->z_val_.resize(nat, 0.0);
    this->constrain_charge_.resize(nat, 0);

    auto tc_tmp = ucell.get_target_charge();
    auto mu_tmp = ucell.get_mu();
    auto cc_tmp = ucell.get_constrain_charge();

    if ((int)tc_tmp.size() >= nat) this->target_charge_.assign(tc_tmp.begin(), tc_tmp.begin() + nat);
    if ((int)mu_tmp.size() >= nat) this->mu_.assign(mu_tmp.begin(), mu_tmp.begin() + nat);
    if ((int)cc_tmp.size() >= nat) this->constrain_charge_.assign(cc_tmp.begin(), cc_tmp.begin() + nat);

    // Read Z_val from pseudopotential for valence mode
    if (charge_mode_ == "valence") {
        int atom_idx = 0;
        for (int it = 0; it < ucell.ntype; it++) {
            for (int ia = 0; ia < ucell.atoms[it].na; ia++) {
                this->z_val_[atom_idx] = ucell.atoms[it].ncpp.zv;
                atom_idx++;
            }
        }
        // Convert target charge (valence) to absolute projected charge
        // valence = N_projected - Z_val => N_projected = valence + Z_val
        for (int iat = 0; iat < nat; iat++) {
            if (this->constrain_charge_[iat] != 0) {
                this->target_charge_[iat] = this->target_charge_[iat] + this->z_val_[iat];
            }
        }
    }

    // Determine CSZ projection basis
#ifdef __LCAO
    if (charge_switch) {
        auto csz_configs = deltaqs::determine_csz_basis(ucell);
        deltaqs::print_csz_configs(csz_configs, ucell);
        bool csz_valid = deltaqs::validate_csz_orbitals(ucell, csz_configs);
        if (!csz_valid) {
            ModuleBase::WARNING_QUIT("DeltaQS::init_deltaqs",
                "CSZ basis validation failed. Check pseudopotential and orbital files.");
        }
    }
#endif

    // Expand atomLabels_ to have one label per atom (not per element type)
    auto type_labels = ucell.get_atomLabels();
    this->atomLabels_.clear();
    int atom_idx = 0;
    for (int it = 0; it < ucell.ntype; it++) {
        for (int ia = 0; ia < ucell.atoms[it].na; ia++) {
            std::string label = type_labels[it] + "_" + std::to_string(atom_idx);
            this->atomLabels_.push_back(label);
            atom_idx++;
        }
    }

    if (this->qs_mode_ == "auto")
    {
        bool has_spin = false;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_[iat].x != 0 || this->constrain_[iat].y != 0 || this->constrain_[iat].z != 0)
            {
                has_spin = true;
                break;
            }
        }
        if (charge_switch && has_spin) this->qs_mode_ = "deltaqs";
        else if (charge_switch) this->qs_mode_ = "deltaq";
        else this->qs_mode_ = "deltaspin";
    }

    std::cout << "[DeltaQS] Mode: " << this->qs_mode_ << std::endl;
    std::cout << "[DeltaQS] Charge constraint enabled: " << (charge_switch ? "yes" : "no") << std::endl;
    if (charge_switch)
    {
        int n_charge_constrained = 0;
        int label_size = (int)this->atomLabels_.size();
        int tc_size = (int)this->target_charge_.size();
        int mu_size = (int)this->mu_.size();
        int cc_size = (int)this->constrain_charge_.size();
        int safe_nat = nat;
        if (safe_nat > label_size) safe_nat = label_size;
        if (safe_nat > tc_size) safe_nat = tc_size;
        if (safe_nat > mu_size) safe_nat = mu_size;
        if (safe_nat > cc_size) safe_nat = cc_size;
        for (int iat = 0; iat < safe_nat; iat++)
        {
            if (iat < cc_size && this->constrain_charge_[iat] != 0)
            {
                n_charge_constrained++;
                std::string label = (iat < label_size) ? this->atomLabels_[iat] : "?";
                double tc = (iat < tc_size) ? this->target_charge_[iat] : 0.0;
                double mu = (iat < mu_size) ? this->mu_[iat] * ModuleBase::Ry_to_eV : 0.0;
                std::cout << "[DeltaQS]   Atom " << iat << " (" << label
                          << "): target_N=" << tc
                          << " mu=" << mu << " eV/e" << std::endl;
                std::cout.flush();
            }
        }
        std::cout << "[DeltaQS] Charge-constrained atoms: " << n_charge_constrained << "/" << nat << std::endl;
        std::cout.flush();
    }
    if (ground_state_search)
    {
        std::cout << "[DeltaQS] Ground state search: enabled (max_iter=" << outer_max_iter
                  << ", thr=" << outer_thr << " eV)" << std::endl;
    }
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::zero_Ni()
{
    for (auto& ni : this->Ni_) ni = 0.0;
}

template <>
void spinconstrain::SpinConstrain<double>::zero_Ni()
{
    for (auto& ni : this->Ni_) ni = 0.0;
}

#ifdef __LCAO
template <>
void spinconstrain::SpinConstrain<std::complex<double>>::cal_ni_lcao(const int& step, bool print)
{
    if (!this->charge_constraint_enabled_) return;

    this->zero_Ni();
    int nat = this->get_nat();
    this->Ni_.resize(nat, 0.0);

    std::vector<ModuleBase::Vector3<int>> constrain_all(nat, ModuleBase::Vector3<int>(1, 1, 1));

    if (this->nspin_ == 2)
    {
        this->dm_->switch_dmr(1);
        const hamilt::HContainer<double>* dmr = this->dm_->get_DMR_pointer(1);
        auto moments = static_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(
            this->p_operator)->cal_moment(dmr, constrain_all);
        this->dm_->switch_dmr(0);
        for (int iat = 0; iat < nat; iat++)
        {
            this->Ni_[iat] = moments[iat];
        }
    }
    else if (this->nspin_ == 4)
    {
        const hamilt::HContainer<double>* dmr = this->dm_->get_DMR_pointer(1);
        auto* dspin_op = static_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>*>(
            this->p_operator);
        for (int iat = 0; iat < nat; iat++)
        {
            this->Ni_[iat] = 0.0;
            if (constrain_all[iat].x + constrain_all[iat].y + constrain_all[iat].z == 0) continue;
            const hamilt::HContainer<std::complex<double>>* pre_hr_iat = dspin_op->get_pre_hr(iat);
            if (!pre_hr_iat) continue;
            for (int iap = 0; iap < pre_hr_iat->size_atom_pairs(); iap++)
            {
                hamilt::AtomPair<std::complex<double>>& tmp = pre_hr_iat->get_atom_pair(iap);
                int iat1 = tmp.get_atom_i();
                int iat2 = tmp.get_atom_j();
                int row_size = tmp.get_row_size();
                int col_size = tmp.get_col_size();
                for (int ir = 0; ir < tmp.get_R_size(); ir++)
                {
                    const ModuleBase::Vector3<int> r_index = tmp.get_R_index(ir);
                    const double* dmr_data = dmr->find_matrix(iat1, iat2, r_index[0], r_index[1], r_index[2])->get_pointer();
                    const std::complex<double>* hr_data = tmp.get_pointer(ir);
                    int index = 0;
                    double charge = 0.0;
                    for (int irow = 0; irow < row_size; irow += 2)
                    {
                        for (int icol = 0; icol < col_size; icol += 2)
                        {
                            charge += (dmr_data[index] + dmr_data[index + col_size + 1]) * hr_data[index].real();
                            index += 2;
                        }
                        index += col_size;
                    }
                    this->Ni_[iat] += charge;
                }
            }
        }
#ifdef __MPI
        Parallel_Reduce::reduce_all(this->Ni_.data(), nat);
#endif
    }

    if (print)
    {
        std::cout << "[DeltaQS] Ni at step " << step << ":";
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] != 0)
                std::cout << " " << this->atomLabels_[iat] << "=" << this->Ni_[iat];
        }
        std::cout << std::endl;
    }
}
#endif

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::print_Ni(std::ofstream& ofs_running)
{
    int nat = this->get_nat();
    
    if (this->charge_mode_ == "valence") {
        ofs_running << "\n VALENCE STATE (valence = N_projected - Z_val):" << std::endl;
        ofs_running << std::setw(10) << "Atom" << std::setw(10) << "Z_val" 
                    << std::setw(15) << "Ni" << std::setw(15) << "Valence"
                    << std::setw(15) << "Target" << std::setw(15) << "Delta" << std::endl;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] == 0) continue;
            double valence = this->Ni_[iat] - this->z_val_[iat];
            double target_valence = this->target_charge_[iat] - this->z_val_[iat];
            double delta = valence - target_valence;
            ofs_running << std::setw(10) << this->atomLabels_[iat]
                        << std::setw(10) << std::fixed << std::setprecision(2) << this->z_val_[iat]
                        << std::setw(15) << std::setprecision(6) << this->Ni_[iat]
                        << std::setw(15) << valence
                        << std::setw(15) << target_valence
                        << std::setw(15) << delta << std::endl;
        }
    } else {
        ofs_running << "\n CHARGE PROJECTION Ni (electrons):" << std::endl;
        ofs_running << std::setw(10) << "Atom" << std::setw(15) << "Ni"
                    << std::setw(15) << "Target" << std::setw(15) << "Delta" << std::endl;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] == 0) continue;
            double delta = this->Ni_[iat] - this->target_charge_[iat];
            ofs_running << std::setw(10) << this->atomLabels_[iat]
                        << std::setw(15) << std::fixed << std::setprecision(6) << this->Ni_[iat]
                        << std::setw(15) << this->target_charge_[iat]
                        << std::setw(15) << delta << std::endl;
        }
    }
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::print_Charge_Force(std::ofstream& ofs_running)
{
    int nat = this->get_nat();
    ofs_running << "\n CHARGE FORCE (mu, Lagrange multiplier for charge constraint):" << std::endl;
    ofs_running << std::setw(10) << "Atom" << std::setw(15) << "mu(Ry)" << std::setw(15) << "mu(eV)" << std::endl;
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] == 0) continue;
        ofs_running << std::setw(10) << this->atomLabels_[iat]
                    << std::setw(15) << std::fixed << std::setprecision(6) << this->mu_[iat]
                    << std::setw(15) << this->mu_[iat] * ModuleBase::Ry_to_eV << std::endl;
    }
}

template <>
double spinconstrain::SpinConstrain<std::complex<double>>::cal_charge_escon()
{
    if (!this->charge_constraint_enabled_) return 0.0;
    double escon_q = 0.0;
    int nat = this->get_nat();
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] == 0) continue;
        escon_q -= this->mu_[iat] * (this->Ni_[iat] - this->target_charge_[iat]);
    }
    return escon_q;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::write_gradient_file(int step)
{
    if (!this->gradient_output_) return;

    int nat = this->get_nat();
    std::string fname = "deltaqs_gradient_" + std::to_string(step) + ".dat";
    std::ofstream ofs(fname);
    ofs << "# DeltaQS Gradient Output (step " << step << ")" << std::endl;
    ofs << "# Atom  Ni  Mi_z  target_N  target_M  mu(Ry)  lambda_z(Ry)  mu(eV)  lambda_z(eV)" << std::endl;
    ofs << std::scientific << std::setprecision(10);
    for (int iat = 0; iat < nat; iat++)
    {
        double ni = (iat < (int)this->Ni_.size()) ? this->Ni_[iat] : 0.0;
        double tc = (iat < (int)this->target_charge_.size()) ? this->target_charge_[iat] : 0.0;
        double mu = (iat < (int)this->mu_.size()) ? this->mu_[iat] : 0.0;
        ofs << this->atomLabels_[iat] << "  "
            << ni << "  "
            << this->Mi_[iat].z << "  "
            << tc << "  "
            << this->target_mag_[iat].z << "  "
            << mu << "  "
            << this->lambda_[iat].z << "  "
            << mu * ModuleBase::Ry_to_eV << "  "
            << this->lambda_[iat].z * ModuleBase::Ry_to_eV << std::endl;
    }
    ofs.close();
    std::cout << "[DeltaQS] Gradient written to: " << fname << std::endl;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_outer_loop(int outer_step)
{
    int nat = this->get_nat();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[DeltaQS] === OUTER OPTIMIZATION STEP " << outer_step << " ===" << std::endl;

    double max_grad_mu = 0.0;
    double max_grad_lambda = 0.0;
    double mu_ref = 0.0;
    bool has_ref = false;

    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] == 0) continue;
        if (!has_ref) { mu_ref = this->mu_[iat]; has_ref = true; continue; }
        double grad_mu = std::abs(this->mu_[iat] - mu_ref);
        if (grad_mu > max_grad_mu) max_grad_mu = grad_mu;
    }

    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_[iat].z == 0 && this->constrain_[iat].x == 0 && this->constrain_[iat].y == 0) continue;
        double grad_lambda = std::abs(this->lambda_[iat].z);
        if (grad_lambda > max_grad_lambda) max_grad_lambda = grad_lambda;
    }

    double max_grad = std::max(max_grad_mu, max_grad_lambda) * ModuleBase::Ry_to_eV;
    std::cout << "[DeltaQS] max|dE/dN| = " << max_grad_mu * ModuleBase::Ry_to_eV << " eV" << std::endl;
    std::cout << "[DeltaQS] max|dE/dM| = " << max_grad_lambda * ModuleBase::Ry_to_eV << " eV" << std::endl;
    std::cout << "[DeltaQS] max|gradient| = " << max_grad << " eV" << std::endl;

    if (max_grad < this->outer_thr_)
    {
        std::cout << "[DeltaQS] CONVERGED: gradient " << max_grad << " eV < threshold " << this->outer_thr_ << " eV" << std::endl;
        std::cout << std::string(60, '=') << "\n" << std::endl;
        return;
    }

    double step_size = 0.1;
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] != 0)
        {
            this->target_charge_[iat] += step_size * this->mu_[iat];
        }
        if (this->constrain_[iat].z != 0)
        {
            this->target_mag_[iat].z += step_size * this->lambda_[iat].z;
        }
    }

    std::cout << "[DeltaQS] Updated targets:" << std::endl;
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] != 0 || this->constrain_[iat].z != 0)
        {
            std::cout << "  " << this->atomLabels_[iat]
                      << "  N_target=" << this->target_charge_[iat]
                      << "  M_target=" << this->target_mag_[iat].z << std::endl;
        }
    }
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::update_mu_simple(double step_factor)
{
    if (!this->charge_constraint_enabled_) return;
    int nat = this->get_nat();
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] == 0) continue;
        double delta_N = this->Ni_[iat] - this->target_charge_[iat];
        double mu_step = this->charge_alpha_trial_ * delta_N * step_factor;
        if (std::abs(mu_step) > this->charge_restrict_current_)
        {
            mu_step = (mu_step > 0 ? 1.0 : -1.0) * this->charge_restrict_current_;
        }
        this->mu_[iat] += mu_step;
    }
}

#ifdef __LCAO
template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_lambda_loop(int outer_step, bool rerun)
{
    bool has_spin_constraint = false;
    int nat = this->get_nat();
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_[iat].x != 0 || this->constrain_[iat].y != 0 || this->constrain_[iat].z != 0)
        {
            has_spin_constraint = true;
            break;
        }
    }

    if (has_spin_constraint)
    {
        this->run_lambda_loop(outer_step, rerun);
    }

    if (this->charge_constraint_enabled_)
    {
        this->cal_ni_lcao(outer_step, false);

        double rms_charge = 0.0;
        int n_charge = 0;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] == 0) continue;
            double dN = this->Ni_[iat] - this->target_charge_[iat];
            rms_charge += dN * dN;
            n_charge++;
        }
        if (n_charge > 0) rms_charge = std::sqrt(rms_charge / n_charge);

        std::cout << "[DeltaQS] Charge RMS: " << rms_charge << " e (threshold: " << this->sc_charge_thr_ << ")" << std::endl;

        for (int mu_step = 0; mu_step < this->nsc_; mu_step++)
        {
            this->update_mu_simple(1.0);

            if (has_spin_constraint)
            {
                auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(
                    this->p_operator);
                if (dspin_op) dspin_op->update_lambda();
            }

            psi::Psi<std::complex<double>>* psi_t = static_cast<psi::Psi<std::complex<double>>*>(this->psi);
            hamilt::Hamilt<std::complex<double>>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>>*>(this->p_hamilt);
            hsolver::HSolverLCAO<std::complex<double>> hsolver_t(this->ParaV, PARAM.inp.ks_solver);
            hsolver_t.solve(hamilt_t, psi_t[0], this->pelec, *this->dm_, *this->pelec->charge, this->nspin_, true);
            elecstate::calculate_weights(this->pelec->ekb, this->pelec->wg, this->pelec->klist,
                                         this->pelec->eferm, this->pelec->f_en, this->pelec->nelec_spin,
                                         this->pelec->skip_weights);
            elecstate::calEBand(this->pelec->ekb, this->pelec->wg, this->pelec->f_en);

            this->cal_ni_lcao(mu_step, false);
            if (has_spin_constraint) this->cal_mi_lcao(mu_step);

            rms_charge = 0.0;
            n_charge = 0;
            for (int iat = 0; iat < nat; iat++)
            {
                if (this->constrain_charge_[iat] == 0) continue;
                double dN = this->Ni_[iat] - this->target_charge_[iat];
                rms_charge += dN * dN;
                n_charge++;
            }
            if (n_charge > 0) rms_charge = std::sqrt(rms_charge / n_charge);
            std::cout << "[DeltaQS] mu step " << mu_step << ": charge RMS = " << rms_charge << std::endl;

            if (rms_charge < this->sc_charge_thr_)
            {
                std::cout << "[DeltaQS] Charge constraint converged." << std::endl;
                break;
            }
        }
    }

    if (this->gradient_output_)
    {
        this->write_gradient_file(outer_step);
    }
}
#endif

#ifdef __LCAO
template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_grid_scan(
    int scan_atom, double N_min, double N_max, double N_step,
    double M_min, double M_max, double M_step)
{
    int nat = this->get_nat();
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[DeltaQS] === GRID SCAN ===" << std::endl;
    std::cout << "[DeltaQS] Scan atom: " << scan_atom << std::endl;
    std::cout << "[DeltaQS] N range: [" << N_min << ", " << N_max << "] step " << N_step << std::endl;
    std::cout << "[DeltaQS] M range: [" << M_min << ", " << M_max << "] step " << M_step << std::endl;

    std::string fname = "deltaqs_grid_scan.dat";
    std::ofstream ofs(fname);
    ofs << "# DeltaQS Grid Scan: atom " << scan_atom << std::endl;
    ofs << "# N  M  E(Ry)  mu(Ry)  lambda_z(Ry)  Ni  Mi_z" << std::endl;
    ofs << std::scientific << std::setprecision(10);

    int total_points = 0;
    double E_min_found = 1e10;
    double N_opt = 0, M_opt = 0;

    for (double N_val = N_min; N_val <= N_max + 1e-8; N_val += N_step)
    {
        for (double M_val = M_min; M_val <= M_max + 1e-8; M_val += M_step)
        {
            if (scan_atom >= 0 && scan_atom < nat)
            {
                this->target_charge_[scan_atom] = N_val;
                this->target_mag_[scan_atom].z = M_val;
            }

            this->run_qs_lambda_loop(total_points, true);

            double E = this->pelec->f_en.etot;
            double mu_val = (scan_atom < nat) ? this->mu_[scan_atom] : 0.0;
            double lam_val = (scan_atom < nat) ? this->lambda_[scan_atom].z : 0.0;
            double Ni_val = (scan_atom < nat) ? this->Ni_[scan_atom] : 0.0;
            double Mi_val = (scan_atom < nat) ? this->Mi_[scan_atom].z : 0.0;

            ofs << N_val << "  " << M_val << "  " << E << "  "
                << mu_val << "  " << lam_val << "  " << Ni_val << "  " << Mi_val << std::endl;

            if (E < E_min_found)
            {
                E_min_found = E;
                N_opt = N_val;
                M_opt = M_val;
            }

            total_points++;
            std::cout << "[DeltaQS] Grid point " << total_points << ": N=" << N_val
                      << " M=" << M_val << " E=" << E << " Ry" << std::endl;
        }
    }

    ofs.close();
    std::cout << "[DeltaQS] Grid scan complete: " << total_points << " points" << std::endl;
    std::cout << "[DeltaQS] Minimum energy: " << E_min_found << " Ry at N=" << N_opt << " M=" << M_opt << std::endl;
    std::cout << "[DeltaQS] Results written to: " << fname << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_gradient_descent(
    int max_steps, double step_size, double conv_thr)
{
    int nat = this->get_nat();
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[DeltaQS] === GRADIENT DESCENT ===" << std::endl;
    std::cout << "[DeltaQS] max_steps=" << max_steps << " step_size=" << step_size
              << " conv_thr=" << conv_thr << std::endl;

    std::string fname = "deltaqs_gradient_descent.dat";
    std::ofstream ofs(fname);
    ofs << "# DeltaQS Gradient Descent" << std::endl;
    ofs << "# step  E(Ry)  max|grad|  target_updates..." << std::endl;
    ofs << std::scientific << std::setprecision(10);

    for (int step = 0; step < max_steps; step++)
    {
        this->run_qs_lambda_loop(step, true);

        double max_grad = 0.0;
        double mu_ref = 0.0;
        bool has_ref = false;

        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] == 0) continue;
            if (!has_ref) { mu_ref = this->mu_[iat]; has_ref = true; continue; }
            double g = std::abs(-this->mu_[iat] + mu_ref);
            if (g > max_grad) max_grad = g;
        }
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_[iat].z == 0) continue;
            double g = std::abs(-this->lambda_[iat].z);
            if (g > max_grad) max_grad = g;
        }
        max_grad *= ModuleBase::Ry_to_eV;

        double E = this->pelec->f_en.etot;
        ofs << step << "  " << E << "  " << max_grad;
        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] != 0)
                ofs << "  " << this->target_charge_[iat];
            if (this->constrain_[iat].z != 0)
                ofs << "  " << this->target_mag_[iat].z;
        }
        ofs << std::endl;

        std::cout << "[DeltaQS] Step " << step << ": E=" << E << " Ry, max|grad|=" << max_grad << " eV" << std::endl;

        if (max_grad < conv_thr)
        {
            std::cout << "[DeltaQS] CONVERGED at step " << step << std::endl;
            break;
        }

        for (int iat = 0; iat < nat; iat++)
        {
            if (this->constrain_charge_[iat] != 0)
                this->target_charge_[iat] += step_size * this->mu_[iat];
            if (this->constrain_[iat].z != 0)
                this->target_mag_[iat].z += step_size * this->lambda_[iat].z;
        }
    }

    ofs.close();
    std::cout << "[DeltaQS] Results written to: " << fname << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_lbfgs(
    int max_steps, double conv_thr, int history_size)
{
    int nat = this->get_nat();
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[DeltaQS] === L-BFGS OPTIMIZATION ===" << std::endl;

    std::vector<int> active_charge, active_spin;
    for (int iat = 0; iat < nat; iat++)
    {
        if (this->constrain_charge_[iat] != 0) active_charge.push_back(iat);
        if (this->constrain_[iat].z != 0) active_spin.push_back(iat);
    }

    int dim = active_charge.size() + active_spin.size();
    if (active_charge.size() > 1) dim -= 1;

    std::cout << "[DeltaQS] Optimization dimension: " << dim << std::endl;
    std::cout << "[DeltaQS] Active charge atoms: " << active_charge.size() << std::endl;
    std::cout << "[DeltaQS] Active spin atoms: " << active_spin.size() << std::endl;

    std::string fname = "deltaqs_lbfgs.dat";
    std::ofstream ofs(fname);
    ofs << "# DeltaQS L-BFGS Optimization (dim=" << dim << ")" << std::endl;
    ofs << "# step  E(Ry)  max|grad|(eV)" << std::endl;
    ofs << std::scientific << std::setprecision(10);

    std::vector<std::vector<double>> s_history, y_history;
    std::vector<double> x_prev(dim, 0.0), g_prev(dim, 0.0);

    for (int step = 0; step < max_steps; step++)
    {
        this->run_qs_lambda_loop(step, true);

        std::vector<double> gradient(dim, 0.0);
        int idx = 0;
        double mu_ref = (active_charge.size() > 0) ? this->mu_[active_charge.back()] : 0.0;
        for (int i = 0; i < (int)active_charge.size() - 1; i++)
        {
            gradient[idx++] = (-this->mu_[active_charge[i]] + mu_ref) * ModuleBase::Ry_to_eV;
        }
        for (int iat : active_spin)
        {
            gradient[idx++] = (-this->lambda_[iat].z) * ModuleBase::Ry_to_eV;
        }

        double max_grad = 0.0;
        for (double g : gradient) if (std::abs(g) > max_grad) max_grad = std::abs(g);

        double E = this->pelec->f_en.etot;
        ofs << step << "  " << E << "  " << max_grad << std::endl;
        std::cout << "[DeltaQS] L-BFGS step " << step << ": E=" << E << " Ry, |grad|=" << max_grad << " eV" << std::endl;

        if (max_grad < conv_thr)
        {
            std::cout << "[DeltaQS] L-BFGS CONVERGED at step " << step << std::endl;
            break;
        }

        std::vector<double> x_curr(dim, 0.0);
        idx = 0;
        for (int i = 0; i < (int)active_charge.size() - 1; i++)
            x_curr[idx++] = this->target_charge_[active_charge[i]];
        for (int iat : active_spin)
            x_curr[idx++] = this->target_mag_[iat].z;

        std::vector<double> direction = gradient;
        for (int i = 0; i < dim; i++) direction[i] = -direction[i];

        if (step > 0 && !s_history.empty())
        {
            std::vector<double> q = gradient;
            int m = s_history.size();
            std::vector<double> alpha(m);
            for (int i = m - 1; i >= 0; i--)
            {
                double rho = 1.0;
                double dot_sy = 0.0;
                for (int j = 0; j < dim; j++) dot_sy += s_history[i][j] * y_history[i][j];
                if (std::abs(dot_sy) > 1e-15) rho = 1.0 / dot_sy;

                double dot_sq = 0.0;
                for (int j = 0; j < dim; j++) dot_sq += s_history[i][j] * q[j];
                alpha[i] = rho * dot_sq;

                for (int j = 0; j < dim; j++) q[j] -= alpha[i] * y_history[i][j];
            }

            double dot_yy = 0.0, dot_sy = 0.0;
            int last = m - 1;
            for (int j = 0; j < dim; j++)
            {
                dot_yy += y_history[last][j] * y_history[last][j];
                dot_sy += s_history[last][j] * y_history[last][j];
            }
            double gamma = (std::abs(dot_yy) > 1e-15) ? dot_sy / dot_yy : 1.0;

            std::vector<double> r(dim);
            for (int j = 0; j < dim; j++) r[j] = gamma * q[j];

            for (int i = 0; i < m; i++)
            {
                double rho = 1.0;
                double dot_sy = 0.0;
                for (int j = 0; j < dim; j++) dot_sy += s_history[i][j] * y_history[i][j];
                if (std::abs(dot_sy) > 1e-15) rho = 1.0 / dot_sy;

                double dot_yr = 0.0;
                for (int j = 0; j < dim; j++) dot_yr += y_history[i][j] * r[j];
                double beta = rho * dot_yr;

                for (int j = 0; j < dim; j++) r[j] += s_history[i][j] * (alpha[i] - beta);
            }
            direction = r;
            for (int i = 0; i < dim; i++) direction[i] = -direction[i];
        }

        double step_size = 0.1;
        idx = 0;
        for (int i = 0; i < (int)active_charge.size() - 1; i++)
            this->target_charge_[active_charge[i]] += step_size * direction[idx++];
        for (int iat : active_spin)
            this->target_mag_[iat].z += step_size * direction[idx++];

        if (step > 0)
        {
            std::vector<double> s_vec(dim), y_vec(dim);
            for (int j = 0; j < dim; j++)
            {
                s_vec[j] = x_curr[j] - x_prev[j];
                y_vec[j] = gradient[j] - g_prev[j];
            }
            s_history.push_back(s_vec);
            y_history.push_back(y_vec);
            if ((int)s_history.size() > history_size)
            {
                s_history.erase(s_history.begin());
                y_history.erase(y_history.begin());
            }
        }
        x_prev = x_curr;
        g_prev = gradient;
    }

    ofs.close();
    std::cout << "[DeltaQS] L-BFGS results written to: " << fname << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

template <>
void spinconstrain::SpinConstrain<std::complex<double>>::run_qs_attribution(const std::string& ref_label)
{
    int nat = this->get_nat();
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[DeltaQS] === ATTRIBUTION ANALYSIS ===" << std::endl;
    std::cout << "[DeltaQS] Reference: " << ref_label << std::endl;

    double total_M = 0.0;
    for (int iat = 0; iat < nat; iat++)
    {
        total_M += this->Mi_[iat].z;
    }

    double E_qs = this->pelec->f_en.etot;

    std::cout << "[DeltaQS] DeltaQS energy: " << E_qs << " Ry" << std::endl;
    std::cout << "[DeltaQS] Total magnetization S* = " << total_M << " uB" << std::endl;

    std::string fname = "deltaqs_attribution.dat";
    std::ofstream ofs(fname);
    ofs << "# DeltaQS Attribution Analysis" << std::endl;
    ofs << "# Reference: " << ref_label << std::endl;
    ofs << "# E_DeltaQS = " << E_qs << " Ry" << std::endl;
    ofs << "# Total_M = " << total_M << " uB" << std::endl;
    ofs << "#" << std::endl;
    ofs << "# Atom  Ni  Mi_z  target_N  target_M  mu(eV)  lambda_z(eV)" << std::endl;

    for (int iat = 0; iat < nat; iat++)
    {
        double ni = (iat < (int)this->Ni_.size()) ? this->Ni_[iat] : 0.0;
        double tc = (iat < (int)this->target_charge_.size()) ? this->target_charge_[iat] : 0.0;
        ofs << this->atomLabels_[iat] << "  "
            << ni << "  " << this->Mi_[iat].z << "  "
            << tc << "  " << this->target_mag_[iat].z << "  "
            << this->mu_[iat] * ModuleBase::Ry_to_eV << "  "
            << this->lambda_[iat].z * ModuleBase::Ry_to_eV << std::endl;
    }

    std::cout << "\n[DeltaQS] Attribution categories:" << std::endl;
    std::cout << "  A: Same M, different magnetic configuration" << std::endl;
    std::cout << "  B: M not in scan range (S*=" << total_M << " uB)" << std::endl;
    std::cout << "  C: Many-body effect (fractional spin)" << std::endl;

    bool is_integer_M = (std::abs(total_M - std::round(total_M)) < 0.01);
    if (!is_integer_M)
    {
        std::cout << "\n[DeltaQS] NOTE: S* = " << total_M << " is non-integer!" << std::endl;
        std::cout << "[DeltaQS] This may indicate category C (fractional spin state)." << std::endl;
    }
    else
    {
        std::cout << "\n[DeltaQS] S* = " << total_M << " is integer. Likely category A or B." << std::endl;
    }

    ofs.close();
    std::cout << "[DeltaQS] Attribution written to: " << fname << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}
#endif

template class spinconstrain::SpinConstrain<std::complex<double>>;
template class spinconstrain::SpinConstrain<double>;
