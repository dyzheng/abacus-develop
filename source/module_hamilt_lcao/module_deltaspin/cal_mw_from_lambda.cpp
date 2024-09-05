#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/dspin_lcao.h"
#include "module_hamilt_pw/hamilt_pwdft/onsite_projector.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_parameter/parameter.h"
#include "spin_constrain.h"

template <>
void SpinConstrain<std::complex<double>>::cal_mw_from_lambda(int i_step)
{
    ModuleBase::TITLE("SpinConstrain", "cal_mw_from_lambda");
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
    // lambda has been updated in the lambda loop
    if (PARAM.inp.basis_type == "lcao")
    {
        psi::Psi<std::complex<double>>* psi_t = static_cast<psi::Psi<std::complex<double>>*>(this->psi);
        hamilt::Hamilt<std::complex<double>>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>>*>(this->p_hamilt);
        hsolver::HSolver<std::complex<double>, base_device::DEVICE_CPU>* hsolver_t = static_cast<hsolver::HSolver<std::complex<double>, base_device::DEVICE_CPU>*>(this->phsol);
        if (GlobalV::NSPIN == 2)
        {
            dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(this->p_operator)
                ->update_lambda();
        }
        else if (GlobalV::NSPIN == 4)
        {
            dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>*>(
                this->p_operator)
                ->update_lambda();
        }
        // diagonalization without update charge
        hsolver_t->solve(hamilt_t, psi_t[0], this->pelec, this->KS_SOLVER, true);
        this->pelec->calculate_weights();
        this->pelec->calEBand();
        elecstate::ElecStateLCAO<std::complex<double>>* pelec_lcao
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec);
        if (this->KS_SOLVER == "genelpa" || this->KS_SOLVER == "scalapack_gvx" || this->KS_SOLVER == "lapack"
            || this->KS_SOLVER == "cg_in_lcao")
        {
            elecstate::cal_dm_psi(this->ParaV, pelec_lcao->wg, *psi_t, *(pelec_lcao->get_DM()));
        }
        pelec_lcao->get_DM()->cal_DMR();
        this->cal_MW(i_step);
    }
    else
    {
        if (i_step == -1)
        {
            // std::cout<<__FILE__<<__LINE__<<"istep == 0"<<std::endl;
            if (PARAM.inp.device == "cpu")
            {
                psi::Psi<std::complex<double>>* psi_t = static_cast<psi::Psi<std::complex<double>>*>(this->psi);
                hamilt::Hamilt<std::complex<double>>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>>*>(this->p_hamilt);
                hsolver::HSolver<std::complex<double>, base_device::DEVICE_CPU>* hsolver_t = static_cast<hsolver::HSolver<std::complex<double>, base_device::DEVICE_CPU>*>(this->phsol);
                hsolver_t->solve(hamilt_t, psi_t[0], this->pelec, this->KS_SOLVER, true);
            }
            else
            {
                psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi_t = static_cast<psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>(this->psi);
                hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>*>(this->p_hamilt);
                hsolver::HSolver<std::complex<double>, base_device::DEVICE_GPU>* hsolver_t = static_cast<hsolver::HSolver<std::complex<double>, base_device::DEVICE_GPU>*>(this->phsol);
                hsolver_t->solve(hamilt_t, psi_t[0], this->pelec, this->KS_SOLVER, true);
            }
            this->pelec->calculate_weights();
            this->cal_Mi_pw();
        }
        else
        {
            this->zero_Mi();
            int size_becp = 0;
            std::vector<std::complex<double>> becp_tmp;
            int nk = 0;
            int nkb = 0;
            int nbands = 0;
            int npol = 0;
            const int* nh_iat = nullptr;
            if (PARAM.inp.device == "cpu")
            {
                psi::Psi<std::complex<double>>* psi_t = static_cast<psi::Psi<std::complex<double>>*>(this->psi);
                hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>*>(this->p_hamilt);
                auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
                nbands = psi_t->get_nbands();
                npol = psi_t->npol;
                nkb = onsite_p->get_tot_nproj();
                nk = psi_t->get_nk();
                nh_iat = &onsite_p->get_nh(0);
                size_becp = nbands * nkb * npol;
                becp_tmp.resize(size_becp * nk);
                for (int ik = 0; ik < nk; ++ik)
                {
                    /// update H(k) for each k point
                    hamilt_t->updateHk(ik);

                    psi_t->fix_k(ik);

                    const std::complex<double>* becp_new = onsite_p->get_becp();
                    hsolver::DiagoIterAssist<std::complex<double>>::diag_responce(hamilt_t,
                                                                                  psi_t[0],
                                                                                  becp_new,
                                                                                  &becp_tmp[ik * nbands * nkb * npol],
                                                                                  nkb * npol,
                                                                                  &this->pelec->ekb(ik, 0));
                }
            }
#if ((defined __CUDA) || (defined __ROCM))
            else
            {
                base_device::DEVICE_GPU* ctx = {};
                base_device::DEVICE_CPU* cpu_ctx = {};
                psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi_t = static_cast<psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>(this->psi);
                hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>* hamilt_t = static_cast<hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>*>(this->p_hamilt);
                auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_GPU>::get_instance();
                nbands = psi_t->get_nbands();
                npol = psi_t->npol;
                nkb = onsite_p->get_tot_nproj();
                nk = psi_t->get_nk();
                nh_iat = &onsite_p->get_nh(0);
                size_becp = nbands * nkb * npol;
                becp_tmp.resize(size_becp * nk);
                std::complex<double>* becp_pointer = nullptr;
                // allocate memory for becp_pointer in GPU device
                base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(ctx, becp_pointer, size_becp);
                for (int ik = 0; ik < nk; ++ik)
                {
                    /// update H(k) for each k point
                    hamilt_t->updateHk(ik);

                    psi_t->fix_k(ik);

                    const std::complex<double>* becp_new = onsite_p->get_becp();
                    hsolver::DiagoIterAssist<std::complex<double>, base_device::DEVICE_GPU>::diag_responce(hamilt_t,
                                                                                  psi_t[0],
                                                                                  becp_new,
                                                                                  becp_pointer,
                                                                                  nkb * npol,
                                                                                  &this->pelec->ekb(ik, 0));
                    // copy becp_pointer from GPU to CPU
                    base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(cpu_ctx, ctx, &becp_tmp[ik * size_becp], becp_pointer, size_becp);   
                }

                // free memory for becp_pointer in GPU device
                base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(ctx, becp_pointer);
            }
#endif
            // calculate weights from ekb to update wg
            this->pelec->calculate_weights();
            // calculate Mi from existed becp
            for (int ik = 0; ik < nk; ik++)
            {
                const std::complex<double>* becp = &becp_tmp[ik * size_becp];
                // becp(nbands*npol , nkb)
                // mag = wg * \sum_{nh}becp * becp
                for (int ib = 0; ib < nbands; ib++)
                {
                    const double weight = this->pelec->wg(ik, ib);
                    int begin_ih = 0;
                    for (int iat = 0; iat < this->Mi_.size(); iat++)
                    {
                        const int nh = nh_iat[iat];
                        std::complex<double> occ[4]
                            = {ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO};
                        for (int ih = 0; ih < nh; ih++)
                        {
                            const int index = ib * npol * nkb + begin_ih + ih;
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
            Parallel_Reduce::reduce_double_allpool(GlobalV::KPAR,
                                                   GlobalV::NPROC_IN_POOL,
                                                   &(this->Mi_[0][0]),
                                                   3 * this->Mi_.size());
            // for(int i = 0; i < this->Mi_.size(); i++)
            //{
            //     std::cout<<"atom"<<i<<": "<<" mag: "<<this->Mi_[i].x<<" "<<this->Mi_[i].y<<" "<<this->Mi_[i].z<<"
            //     "<<this->lambda_[i].x<<" "<<this->lambda_[i].y<<" "<<this->lambda_[i].z<<std::endl;
            // }
        }
    }
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
}