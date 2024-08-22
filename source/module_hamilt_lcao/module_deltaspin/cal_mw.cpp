#include <iostream>

#include "module_base/matrix.h"
#include "module_base/name_angular.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/dspin_lcao.h"
#include "module_hamilt_pw/hamilt_pwdft/onsite_projector.h"
#include "spin_constrain.h"

template <>
ModuleBase::matrix SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_MW_k(
    const std::vector<std::vector<std::complex<double>>>& dm)
{
    ModuleBase::TITLE("module_deltaspin", "cal_MW_k");
    int nw = this->get_nw();
    const int nlocal = (this->nspin_ == 4) ? nw / 2 : nw;
    ModuleBase::matrix MecMulP(this->nspin_, nlocal, true), orbMulP(this->nspin_, nlocal, true);
    for(size_t ik = 0; ik != this->kv_.get_nks(); ++ik)
    {
        std::complex<double> *sk = nullptr;
        if (this->nspin_ == 4)
        {
            dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(this->p_hamilt)->updateSk(ik, 1);
            sk = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(this->p_hamilt)->getSk();
        }
        else
        {
            dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(this->p_hamilt)->updateSk(ik, 1);
            sk = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(this->p_hamilt)->getSk();
        }
        ModuleBase::ComplexMatrix mud(this->ParaV->ncol, this->ParaV->nrow, true);
#ifdef __MPI
        const char T_char = 'T';
        const char N_char = 'N';
        const int one_int = 1;
        const std::complex<double> one_float = {1.0, 0.0}, zero_float = {0.0, 0.0};        
        pzgemm_(&N_char,
                &T_char,
                &nw,
                &nw,
                &nw,
                &one_float,
                dm[ik].data(),
                &one_int,
                &one_int,
                this->ParaV->desc,
                sk,
                &one_int,
                &one_int,
                this->ParaV->desc,
                &zero_float,
                mud.c,
                &one_int,
                &one_int,
                this->ParaV->desc);
        this->collect_MW(MecMulP, mud, nw, this->kv_.isk[ik]);
#endif
    }
#ifdef __MPI
    MPI_Allreduce(MecMulP.c, orbMulP.c, this->nspin_*nlocal, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif 

    return orbMulP;
}

template <>
void SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_MW(const int& step, bool print)
{
    ModuleBase::TITLE("module_deltaspin", "cal_MW");
    ModuleBase::timer::tick("SpinConstrain", "cal_MW");
    // calculate MW from lambda in real space projection method
    {
        this->zero_Mi();
        const hamilt::HContainer<double>* dmr
            = dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_DMR_pointer(1);
        std::vector<double> moments;
        if(GlobalV::NSPIN==2)
        {
            dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->switch_dmr(2);
            moments = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(this->p_operator)->cal_moment(dmr, this->get_constrain());
            dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->switch_dmr(0);
            for(int iat=0;iat<this->Mi_.size();iat++)
            {
                this->Mi_[iat].x = 0.0;
                this->Mi_[iat].y = 0.0;
                this->Mi_[iat].z = moments[iat];
            }
        }
        else if(GlobalV::NSPIN==4)
        {
            moments = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>*>(this->p_operator)->cal_moment(dmr, this->get_constrain());
            for(int iat=0;iat<this->Mi_.size();iat++)
            {
                this->Mi_[iat].x = moments[iat*3];
                this->Mi_[iat].y = moments[iat*3+1];
                this->Mi_[iat].z = moments[iat*3+2];
            }
        }
    }
    /*else
    {
        const std::vector<std::vector<std::complex<double>>>& dm
            = dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_DMK_vector();
        this->calculate_MW(this->convert(this->cal_MW_k(dm)));
    }*/
    //this->print_Mi(print);

    ModuleBase::timer::tick("SpinConstrain", "cal_MW");
}

template <>
void SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_Mi_pw()
{
    this->zero_Mi();
    auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
    // loop over k-points to calculate Mi of \sum_{k,i,l,m}<Psi_{k,i}|alpha_{l,m}><alpha_{l,m}|Psi_{k,i}>
    const int nbands = this->psi->get_nbands();
    std::vector<double> charge(this->Mi_.size(), 0.0);
    for(int ik = 0; ik < this->psi->get_nk(); ik++)
    {
        this->psi->fix_k(ik);
        onsite_p->init_k(ik);
        onsite_p->overlap_proj_psi(
                        nbands * this->psi->npol,
                        this->psi->get_pointer());
        const std::complex<double>* becp = onsite_p->get_becp();
        // becp(nbands*npol , nkb)
        // mag = wg * \sum_{nh}becp * becp
        int nkb = onsite_p->get_size_becp() / nbands / this->psi->npol;
        for(int ib = 0;ib<nbands;ib++)
        {
            const double weight = this->pelec->wg(ik, ib);
            int begin_ih = 0;
            for(int iat = 0; iat < this->Mi_.size(); iat++)
            {
                std::complex<double> occ[4] = {ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO, ModuleBase::ZERO};
                const int nh = onsite_p->get_nh(iat);
                for(int ih = 0; ih < nh; ih++)
                {
                    const int index = ib*2*nkb + begin_ih + ih;
                    occ[0] += conj(becp[index]) * becp[index];
                    occ[1] += conj(becp[index]) * becp[index + nkb];
                    occ[2] += conj(becp[index + nkb]) * becp[index];
                    occ[3] += conj(becp[index + nkb]) * becp[index + nkb];
                }
                // occ has been reduced and calculate mag
                this->Mi_[iat].z += weight * (occ[0] - occ[3]).real();
                this->Mi_[iat].x += weight * (occ[1] + occ[2]).real();
                this->Mi_[iat].y += weight * (occ[1] - occ[2]).imag();
                charge[iat] += weight * (occ[0] + occ[3]).real();
                begin_ih += nh;
            }
        }
    }
    // reduce mag from all k-pools
    Parallel_Reduce::reduce_double_allpool(GlobalV::KPAR, GlobalV::NPROC_IN_POOL, &(this->Mi_[0][0]), 3 * this->Mi_.size());
    Parallel_Reduce::reduce_double_allpool(GlobalV::KPAR, GlobalV::NPROC_IN_POOL, &(charge[0]), charge.size());
    // print charge
    for(int i = 0; i < this->Mi_.size(); i++)
    {
        std::cout<<"atom"<<i<<": "<<" charge: "<<charge[i]<<" mag: "<<this->Mi_[i].x<<" "<<this->Mi_[i].y<<" "<<this->Mi_[i].z<<std::endl;
    }
    
}

template <>
void SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::set_operator(
    hamilt::Operator<std::complex<double>>* op_in)
{
    this->p_operator = op_in;
}

template <>
void SpinConstrain<double, base_device::DEVICE_CPU>::set_operator(
    hamilt::Operator<double>* op_in)
{
    this->p_operator = op_in;
}