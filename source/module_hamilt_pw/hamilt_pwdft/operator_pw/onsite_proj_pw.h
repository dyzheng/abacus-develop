#ifndef MODULEHAMILTPW_ONSITE_PROJ_PW_H
#define MODULEHAMILTPW_ONSITE_PROJ_PW_H

#include "operator_pw.h"

#include "module_cell/unitcell.h"
#include "module_hsolver/kernels/math_kernel_op.h"

namespace hamilt {

#ifndef ONSITETEMPLATE_H
#define ONSITETEMPLATE_H

template<class T> class OnsiteProj : public T {};
// template<typename Real, typename Device = base_device::DEVICE_CPU>
// class OnsiteProj : public OperatorPW<T, Device> {};

#endif

template<typename T, typename Device>
class OnsiteProj<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
  public:
    OnsiteProj(const int* isk_in,
             const UnitCell* ucell_in);

    template<typename T_in, typename Device_in = Device>
    explicit OnsiteProj(const OnsiteProj<OperatorPW<T_in, Device_in>>* onsite_proj);

    virtual ~OnsiteProj();

    virtual void init(const int ik_in)override;

    virtual void act(const int nbands,
        const int nbasis,
        const int npol,
        const T* tmpsi_in,
        T* tmhpsi,
        const int ngk = 0)const override;

    const int *get_isk() const {return this->isk;}
    const UnitCell *get_ucell() const {return this->ucell;}

  private:
    void add_onsite_proj(T *hpsi_in, const T *psi_in, const int npol, const int m) const;

    const int* isk = nullptr;

    const UnitCell* ucell = nullptr;

    mutable int nkb_m = 0;

    mutable T *ps = nullptr;
    int tnp = 0;
    Device* ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};

    using gemv_op = hsolver::gemv_op<T, Device>;
    using gemm_op = hsolver::gemm_op<T, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt

#endif