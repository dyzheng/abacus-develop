#ifndef __EKINETICLCAO
#define __EKINETICLCAO
#include "operator_lcao.h"

namespace hamilt
{

#ifndef __EKINETICTEMPLATE
#define __EKINETICTEMPLATE

template<class T> class Ekinetic : public T 
{};

#endif

template<typename T>
class Ekinetic<OperatorLCAO<T>> : OperatorLCAO<T> 
{
    public:

    virtual void contributeHR() override;

    virtual void contributeHk() override;

}

}
#endif