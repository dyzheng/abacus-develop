#ifndef OPDFTULCAO_H
#define OPDFTULCAO_H
#include "module_base/timer.h"
#include "operator_lcao.h"

namespace hamilt
{

#ifndef __OPDFTUTEMPLATE
#define __OPDFTUTEMPLATE

template <class T>
class OperatorDFTU : public T
{
};

#endif

template <typename TK, typename TR>
class OperatorDFTU<OperatorLCAO<TK, TR>> : public OperatorLCAO<TK, TR>
{
  public:
    OperatorDFTU<OperatorLCAO<TK, TR>>(LCAO_Matrix* LM_in,
                                  const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                  hamilt::HContainer<TR>* hR_in,
                                  hamilt::HContainer<TR>* sR_in,
                                  std::vector<TK>* hK_in,
                                  const UnitCell* ucell_in,
                                  const std::vector<int>& isk_in);
    ~OperatorDFTU();

    virtual void contributeHR() override;

    virtual void contributeHk(int ik) override;

  private:

    void initialize(const UnitCell* ucell_in);

    hamilt::HContainer<TR>* VU_ = nullptr;
    const hamilt::HContainer<TR>* sR_ = nullptr;

    const std::vector<int>& isk;
};
} // namespace hamilt
#endif