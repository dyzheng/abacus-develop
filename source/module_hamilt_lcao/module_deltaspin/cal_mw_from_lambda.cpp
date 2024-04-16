#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/dspin_lcao.h"
#include "spin_constrain.h"

template <>
void SpinConstrain<std::complex<double>, psi::DEVICE_CPU>::cal_mw_from_lambda(int i_step)
{
    ModuleBase::TITLE("SpinConstrain","cal_mw_from_lambda");
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
    // lambda has been updated in the lambda loop
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
    elecstate::ElecStateLCAO<std::complex<double>>* pelec_lcao
        = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec);
    this->pelec->calculate_weights();
    this->pelec->calEBand();
    if (this->KS_SOLVER == "genelpa" || this->KS_SOLVER == "scalapack_gvx" || this->KS_SOLVER == "lapack" || this->KS_SOLVER == "cg_in_lcao")
    {
        elecstate::cal_dm_psi(this->ParaV, pelec_lcao->wg, *(this->psi), *(pelec_lcao->get_DM()));
    }
    pelec_lcao->get_DM()->cal_DMR();
    this->cal_MW(i_step, this->LM);
    ModuleBase::timer::tick("SpinConstrain", "cal_mw_from_lambda");
}