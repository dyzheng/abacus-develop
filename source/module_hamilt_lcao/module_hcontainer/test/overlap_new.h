#ifndef OVERLAPNEW_H
#define OVERLAPNEW_H
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/operator_lcao.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"

namespace hamilt
{

#ifndef __OVERLAPNEWTEMPLATE
#define __OVERLAPNEWTEMPLATE

template <class T>
class OverlapNew : public T
{
};

#endif

template <typename T>
class OverlapNew<OperatorLCAO<T>> : public OperatorLCAO<T>
{
  public:
    OverlapNew<OperatorLCAO<T>>(LCAO_Matrix* LM_in,
                                const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                hamilt::HContainer<T>* SR_in,
                                std::vector<T>* SK_pointer_in,
                                const UnitCell* ucell_in,
                                Grid_Driver* GridD_in,
                                const Parallel_Orbitals* paraV);

    virtual void contributeHR() override;

    virtual void contributeHk(int ik) override;

  private:
    const UnitCell* ucell = nullptr;

    hamilt::HContainer<T>* SR = nullptr;

    std::vector<T>* SK_pointer = nullptr;

    bool SR_fixed_done = false;

    /**
     * @brief initialize SR, search the nearest neighbor atoms
     * HContainer is used to store the overlap matrix with specific <I,J,R> atom-pairs
     * the size of SR will be fixed after initialization
     */
    void initialize_SR(Grid_Driver* GridD_in, const Parallel_Orbitals* paraV);

    /**
     * @brief calculate the overlap matrix with specific <I,J,R> atom-pairs
     * nearest neighbor atoms don't need to be calculated again
     * loop the atom-pairs in SR and calculate the overlap matrix
     */
    void calculate_SR();

    /**
     * @brief calculate the SR local matrix of <I,J,R> atom pair
    */
    void cal_SR_IJR(
        const int& iat1, 
        const int& iat2, 
        const Parallel_Orbitals* paraV,
        const ModuleBase::Vector3<double>& dtau,
        T* data_pointer);
};

} // namespace hamilt
#endif