#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/dspin_lcao.h"
#include "module_hamilt_pw/hamilt_pwdft/onsite_projector.h"
#include "module_hsolver/diago_iter_assist.h"
#include "spin_constrain.h"
#include "module_parameter/parameter.h"

template <>
void SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_mw_from_lambda(int i_step)
{
    ModuleBase::TITLE("SpinConstrain","cal_mw_from_lambda");
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
    // lambda has been updated in the lambda loop
    if(PARAM.inp.basis_type == "lcao")
    {
        if(GlobalV::NSPIN==2)
        {
            dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(this->p_operator)->update_lambda();
        }
        else if(GlobalV::NSPIN==4)
        {
            dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>*>(this->p_operator)->update_lambda();
        }
        // diagonalization without update charge
        this->phsol->solve(this->p_hamilt, this->psi[0], this->pelec, this->KS_SOLVER, true);
        this->pelec->calculate_weights();
        this->pelec->calEBand();
        elecstate::ElecStateLCAO<std::complex<double>>* pelec_lcao
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec);
        if (this->KS_SOLVER == "genelpa" || this->KS_SOLVER == "scalapack_gvx" || this->KS_SOLVER == "lapack" || this->KS_SOLVER == "cg_in_lcao")
        {
            elecstate::cal_dm_psi(this->ParaV, pelec_lcao->wg, *(this->psi), *(pelec_lcao->get_DM()));
        }
        pelec_lcao->get_DM()->cal_DMR();
        this->cal_MW(i_step);
    }
    else
    {
        if(i_step == -1)
        {
            //std::cout<<__FILE__<<__LINE__<<"istep == 0"<<std::endl;
            this->phsol->solve(this->p_hamilt, this->psi[0], this->pelec, this->KS_SOLVER, true);
            this->pelec->calculate_weights();
            this->cal_Mi_pw();
        }
        else
        {
            this->zero_Mi();
            auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
            const int nbands = this->psi->get_nbands();
            const int npol = this->psi->npol;
            const int nkb = onsite_p->get_tot_nproj();
            const int nk = this->psi->get_nk();
            std::vector<std::complex<double>> becp_tmp(nk * nbands * nkb * npol, 0.0);
            for (int ik = 0; ik < nk; ++ik)
            {
                /// update H(k) for each k point
                this->p_hamilt->updateHk(ik);

                this->psi->fix_k(ik);

                const std::complex<double>* becp_new = onsite_p->get_becp();
                hsolver::DiagoIterAssist<std::complex<double>>::diag_responce(
                    this->p_hamilt,
                    this->psi[0],
                    becp_new,
                    &becp_tmp[ik * nbands * nkb * npol],
                    nkb * npol,
                    &this->pelec->ekb(ik, 0)
                );
            }
            // calculate weights from ekb to update wg
            this->pelec->calculate_weights();
            // calculate Mi from existed becp
            for(int ik = 0; ik < nk; ik++)
            {
                const std::complex<double>* becp = &becp_tmp[ik * nbands * nkb * npol];
                // becp(nbands*npol , nkb)
                // mag = wg * \sum_{nh}becp * becp
                const int nh = nkb / this->Mi_.size();
                for(int ib = 0;ib<nbands;ib++)
                {
                    const double weight = this->pelec->wg(ik, ib);
                    int begin_ih = 0;
                    for(int iat = 0; iat < this->Mi_.size(); iat++)
                    {
                        std::complex<double> occ[4] = {ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO};
                        for(int ih = 0; ih < nh; ih++)
                        {
                            const int index = ib*npol*nkb + begin_ih + ih;
                            occ[0] += conj(becp[index]) * becp[index];
                            occ[1] += conj(becp[index]) * becp[index + nkb];
                            occ[2] += conj(becp[index + nkb]) * becp[index];
                            occ[3] += conj(becp[index + nkb]) * becp[index + nkb];
                        }
                        // occ has been reduced and calculate mag
                        this->Mi_[iat].x += weight * (occ[1] + occ[2]).real();
                        this->Mi_[iat].y += weight * (occ[1] - occ[2]).imag();
                        this->Mi_[iat].z += weight * (occ[0] - occ[3]).real();
                        begin_ih += nh;
                    }
                }
            }
            //for(int i = 0; i < this->Mi_.size(); i++)
            //{
            //    std::cout<<"atom"<<i<<": "<<" mag: "<<this->Mi_[i].x<<" "<<this->Mi_[i].y<<" "<<this->Mi_[i].z<<" "<<this->lambda_[i].x<<" "<<this->lambda_[i].y<<" "<<this->lambda_[i].z<<std::endl;
            //    GlobalV::ofs_running<<std::setprecision(12)<<"atom"<<i<<": "<<" mag: "<<this->Mi_[i].x<<" "<<this->Mi_[i].y<<" "<<this->Mi_[i].z<<" "<<this->lambda_[i].x<<" "<<this->lambda_[i].y<<" "<<this->lambda_[i].z<<std::endl;
            //}
        }
    }
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
}