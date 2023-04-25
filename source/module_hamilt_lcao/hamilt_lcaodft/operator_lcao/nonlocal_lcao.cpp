#include "nonlocal_lcao.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"

#ifdef __DEBUG
#include "module_base/global_variable.h"
#include "module_base/export.h"
#endif

namespace hamilt
{

template class Nonlocal<OperatorLCAO<double>>;

template class Nonlocal<OperatorLCAO<std::complex<double>>>;

template<typename T>
void Nonlocal<OperatorLCAO<T>>::contributeHR()
{
    ModuleBase::TITLE("Nonlocal<OperatorLCAO>", "contributeHR");
    if(!this->HR_fixed_done)
    {
        ModuleBase::timer::tick("Nonlocal<LCAO>", "contributeHR");
        this->genH->calculate_NL_no(this->HR_pointer->data());
#ifdef __DEBUG
        if(GlobalV::NSPIN !=4 ) ModuleBase::dump_array(this->HR_pointer->data(), this->LM->Hloc_fixedR.size(), "HR_Nonlocal.txt");
        else ModuleBase::dump_array(this->LM->Hloc_fixedR_soc.data(), this->LM->Hloc_fixedR_soc.size(), "HR_Nonlocal.txt");
#endif
        ModuleBase::timer::tick("Nonlocal<LCAO>", "contributeHR");
        this->HR_fixed_done = true;
    }
    return;
}

}