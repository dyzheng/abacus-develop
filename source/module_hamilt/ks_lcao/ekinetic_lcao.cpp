#include "ekinetic_lcao.h"

namespace hamilt
{

template class Ekinetic<OperatorLCAO<double>>;

template class Ekinetic<OperatorLCAO<std::complex<double>>>;

template<>
void Ekinetic<OperatorLCAO<T>>::contributeHR()
{

}

}