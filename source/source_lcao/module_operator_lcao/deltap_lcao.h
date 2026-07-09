#ifndef DELTAP_LCAO_H
#define DELTAP_LCAO_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
#include "source_lcao/module_operator_lcao/operator_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include <unordered_map>
#include <complex>
#include <vector>

namespace hamilt
{

template <typename TK, typename TR>
class DeltaPOperator : public OperatorLCAO<TK, TR>
{
  public:
    DeltaPOperator(HS_Matrix_K<TK>* hsk_in,
                   const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                   hamilt::HContainer<TR>* hR_in,
                   const UnitCell& ucell_in,
                   const Grid_Driver* gridD_in,
                   const TwoCenterIntegrator* intor,
                   const std::vector<double>& orb_cutoff,
                   double rm);

    ~DeltaPOperator();

    virtual void contributeHR() override;

    virtual void contributeHk(int ik) override;

    void update_lambda() { this->dp_hr_done = false; }

    void set_lambda(const std::vector<double>& lambda_in)
    {
        this->lambda_ = lambda_in;
        this->dp_hr_done = false;
    }

    const std::vector<double>& get_lambda() const { return this->lambda_; }
    const std::vector<double>& get_lambda_save() const { return this->lambda_save_; }

    void set_gdir(int gdir) { this->gdir_ = gdir; }

    void set_hk_correction(const std::unordered_map<int, std::vector<std::complex<double>>>& correction)
    {
        hk_correction_ = correction;
    }

  private:
    const UnitCell* ucell = nullptr;
    const Grid_Driver* gridD = nullptr;
    const Parallel_Orbitals* paraV = nullptr;
    hamilt::HContainer<TR>* HR = nullptr;
    const TwoCenterIntegrator* intor_ = nullptr;
    std::vector<double> orb_cutoff_;
    double rm_ = 3.0;
    int gdir_ = 3;

    std::vector<hamilt::HContainer<TR>*> pre_hr;

    std::vector<double> lambda_;
    std::vector<double> lambda_save_;
    bool initialized = false;
    bool dp_hr_done = false;

    // k-dependent HK correction for Berry connection operator
    std::unordered_map<int, std::vector<std::complex<double>>> hk_correction_;

    void cal_pre_HR();
    void cal_HR_IJR(const int& iat1,
                    const int& iat2,
                    const std::unordered_map<int, std::vector<double>>& nlm1_all,
                    const std::unordered_map<int, std::vector<double>>& nlm2_all,
                    TR* data_pointer);
};

} // namespace hamilt

#endif
