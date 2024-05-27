#include <iostream>

#include "module_base/matrix.h"
#include "module_base/name_angular.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/dspin_lcao.h"
#include "spin_constrain.h"

template <>
ModuleBase::matrix SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_MW_k(
    LCAO_Matrix* LM,
    const std::vector<std::vector<std::complex<double>>>& dm)
{
    ModuleBase::TITLE("module_deltaspin", "cal_MW_k");
    int nw = this->get_nw();
    const int nlocal = (this->nspin_ == 4) ? nw / 2 : nw;
    ModuleBase::matrix MecMulP(this->nspin_, nlocal, true), orbMulP(this->nspin_, nlocal, true);
    for(size_t ik = 0; ik != this->kv_.nks; ++ik)
    {
        if (this->nspin_ == 4)
        {
            dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(this->p_hamilt)->updateSk(ik, LM, 1);
        }
        else
        {
            dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(this->p_hamilt)->updateSk(ik, LM, 1);
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
                LM->Sloc2.data(),
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
void SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::cal_MW(const int& step, LCAO_Matrix* LM, bool print)
{
    ModuleBase::TITLE("module_deltaspin", "cal_MW");
    if(1)
    {
        this->zero_Mi();
        const hamilt::HContainer<double>* dmr
            = dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_DMR_pointer(1);
        std::vector<double> moments;
        if(GlobalV::NSPIN==2)
        {
            dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->switch_dmr(2);
            moments = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(this->p_operator)->cal_moment(dmr);
            dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->switch_dmr(0);
            for(int iat=0;iat<this->Mi_.size();iat++)
            {
                this->Mi_[iat].z = moments[iat];
            }
        }
        else if(GlobalV::NSPIN==4)
        {
            moments = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>*>(this->p_operator)->cal_moment(dmr);
            for(int iat=0;iat<this->Mi_.size();iat++)
            {
                this->Mi_[iat].x = moments[iat*3];
                this->Mi_[iat].y = moments[iat*3+1];
                this->Mi_[iat].z = moments[iat*3+2];
            }
        }
    }
    else
    {
        const std::vector<std::vector<std::complex<double>>>& dm
            = dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_DMK_vector();
        this->calculate_MW(this->convert(this->cal_MW_k(LM, dm)));
    }
    this->print_Mi(print);
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