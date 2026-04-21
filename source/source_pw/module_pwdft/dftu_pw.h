#ifndef DFTU_PW_H
#define DFTU_PW_H

#include "source_io/module_parameter/parameter.h"
#include "source_cell/unitcell.h"
#include "source_base/matrix.h"

class Plus_U;
class Charge_Mixing;

namespace pw
{

void iter_init_dftu_pw(const int iter,
                       const int istep,
                       Plus_U& dftu,
                       const void* psi,
                       const ModuleBase::matrix& wg,
                       const UnitCell& ucell,
                       const Input_para& inp,
                       Charge_Mixing* p_chgmix);

}

#endif
