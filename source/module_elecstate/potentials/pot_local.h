#ifndef POTLOCAL_H
#define POTLOCAL_H

#include "module_base/matrix.h"
#include "pot_base.h"

namespace elecstate
{

class PotLocal : public PotBase
{
  public:
    PotLocal(const ModuleBase::matrix* vloc_in, // local pseduopotentials
             const ModuleBase::ComplexMatrix* sf_in,
             const ModulePW::PW_Basis* rho_basis_in)
        : vloc_(vloc_in), sf_(sf_in)
    {
        assert(this->vloc_->nr == this->sf_->nr);
        this->rho_basis_ = rho_basis_in;
        this->ntype_ = this->vloc_->nr;
        this->fixed_mode = true;
        this->dynamic_mode = false;
    }

    void cal_fixed_v(double* vl_pseudo) override;

    /// @brief get the value of vloc at G=0;
    /// @return vl(0)
    static double get_vl_of_0() { return vl_of_0; }

    private:

    /// @brief save the value of vloc at G=0; this is a static member because there is only one vl(0) for all instances
    static double vl_of_0;

    // std::vector<double> vltot;

    const ModuleBase::matrix* vloc_ = nullptr; // local pseduopotentials
    const ModuleBase::ComplexMatrix* sf_ = nullptr; // structure factors
    int ntype_ = 0;
};

} // namespace elecstate

#endif