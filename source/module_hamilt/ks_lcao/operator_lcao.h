#ifndef __OPERATORLCAO
#define __OPERATORLCAO
#include"module_hamilt/operator.h"
#include "module_base/timer.h"

//headers for Lcao_tools
#include "src_lcao/LCAO_gen_fixedH.h"
#include "src_lcao/LCAO_matrix.h"
#include "src_lcao/LCAO_hamilt.h"
#include "module_gint/gint_gamma.h"
#include "module_gint/gint_k.h"
#include "src_lcao/local_orbital_charge.h"
#include "src_lcao/local_orbital_wfc.h"

namespace hamilt
{

// warpper for pointers of classes used in LCAO base 
template<typename T>
class Lcao_tools;

template<std::complex<double>>
class Lcao_tools
{
    public:
    // used for k-dependent grid integration.
    Gint_k* GK = nullptr;

    // use overlap matrix to generate fixed Hamiltonian
    LCAO_gen_fixedH* genH = nullptr;

    LCAO_Matrix* LM = nullptr;

    LCAO_Hamilt* uhm = nullptr;

    Local_Orbital_wfc* lowf = nullptr;

    Local_Orbital_Charge* loc = nullptr;
};

template<double>
class Lcao_tools
{
    public:
    // temporary class members
    // used for gamma only algorithms.
    Gint_Gamma* GG = nullptr;

    // use overlap matrix to generate fixed Hamiltonian
    LCAO_gen_fixedH* genH = nullptr;

    LCAO_Matrix* LM = nullptr;

    LCAO_Hamilt* uhm = nullptr;

    Local_Orbital_wfc* lowf = nullptr;

    Local_Orbital_Charge* loc = nullptr;
};

template<typename T>
class OperatorLCAO : public Operator<T>
{
    public:
    virtual ~OperatorLCAO(){};
    
    void updateHR(T* hr_pointer);

    virtual void contributeHR()
    {
        return;
    }

    void matrixHk(ModuleBase::Vector3<double> kvec_in = ModuleBase::Vector3<double>(0, 0, 0));

    virtual void contributeHk()
    {
        return;
    }

    protected:
    Lcao_tools<T> lcao_tools;  


};

}//end namespace hamilt

#endif