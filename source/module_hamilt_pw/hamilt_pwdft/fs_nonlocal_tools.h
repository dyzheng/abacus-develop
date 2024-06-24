#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_HAMILT_PW_HAMILT_PWDFT_FS_NONLOCAL_TOOLS_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_HAMILT_PW_HAMILT_PWDFT_FS_NONLOCAL_TOOLS_H

#include "module_base/module_device/device.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_cell/klist.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_pw/hamilt_pwdft/VNL_in_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/stress_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_psi/psi.h"

#include <complex>

namespace hamilt
{

template <typename FPTYPE, typename Device>
class FS_Nonlocal_tools
{
  public:
    FS_Nonlocal_tools(const pseudopot_cell_vnl* nlpp_in,
                      const UnitCell* ucell_in,
                      const psi::Psi<std::complex<FPTYPE>, Device>* psi_in,
                      const K_Vectors* kv_in,
                      const ModulePW::PW_Basis_K* wfc_basis_in,
                      const Structure_Factor* sf_in,
                      const ModuleBase::matrix& wg,
                      const ModuleBase::matrix& ekb);
    ~FS_Nonlocal_tools();

    void cal_becp(int ik, int npm);
    void cal_dbecp_s(int ik, int npm, int ipol, int jpol, FPTYPE* stress);
    void cal_dbecp_f(int ik, int npm, int ipol);
    void cal_force(int ik, int npm, FPTYPE* force);

  private:
    void allocate_memory(const ModuleBase::matrix& wg, const ModuleBase::matrix& ekb);
    void delete_memory();

  private:
    const Structure_Factor* sf_;
    const pseudopot_cell_vnl* nlpp_;
    const UnitCell* ucell_;
    const psi::Psi<std::complex<FPTYPE>, Device>* psi_;
    const K_Vectors* kv_;
    const ModulePW::PW_Basis_K* wfc_basis_;

    Device* ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = {};
    int nkb;
    int nbands;

    int max_nh = 0;
    int max_npw = 0;
    int ntype;
    bool nondiagonal;
    int pre_ik_s = -1;
    int pre_ik_f = -1;

    int* atom_nh = nullptr;
    int* atom_na = nullptr;
    int* h_atom_nh = nullptr;
    int* h_atom_na = nullptr;

    std::vector<FPTYPE> g_plus_k;

    int* gcar_zero_indexes = nullptr;
    int gcar_zero_counts[3] = {0, 0, 0};
    std::complex<FPTYPE>* vkb_save = nullptr;
    void transfer_gcar(int npw, int npw_max, const FPTYPE* gcar_in);
    void save_vkb(int npw, int ipol);
    void revert_vkb(int npw, int ipol);

    FPTYPE* d_wg = nullptr;
    FPTYPE* d_ekb = nullptr;
    FPTYPE* gcar = nullptr;
    FPTYPE* deeq = nullptr;
    FPTYPE* kvec_c = nullptr;
    FPTYPE* qq_nt = nullptr;
    // for operators
    FPTYPE* hd_ylm = nullptr;      // (lmax + 1) * (lmax + 1) * npw
    FPTYPE* hd_ylm_deri = nullptr; // 3 * (lmax + 1) * (lmax + 1) * npw
    FPTYPE* hd_vq = nullptr;
    FPTYPE* hd_vq_deri = nullptr; // this->ucell->atoms[it].ncpp.nbeta * npw
    FPTYPE* d_g_plus_k = nullptr; // npw * 5
    FPTYPE* d_pref = nullptr;     // this->ucell->atoms[it].ncpp.nh
    FPTYPE* d_gk = nullptr;
    FPTYPE* d_vq_tab = nullptr;
    // allocate the memory for vkb and vkb_deri.
    FPTYPE** vq_ptrs = nullptr;
    FPTYPE** d_vq_ptrs = nullptr;
    FPTYPE** ylm_ptrs = nullptr;
    FPTYPE** d_ylm_ptrs = nullptr;
    FPTYPE** vq_deri_ptrs = nullptr;
    FPTYPE** d_vq_deri_ptrs = nullptr;
    FPTYPE** ylm_deri_ptrs1 = nullptr;
    FPTYPE** d_ylm_deri_ptrs1 = nullptr;
    FPTYPE** ylm_deri_ptrs2 = nullptr;
    FPTYPE** d_ylm_deri_ptrs2 = nullptr;

    std::complex<FPTYPE>* ppcell_vkb = nullptr;
    std::complex<FPTYPE>** vkb_ptrs = nullptr;
    std::complex<FPTYPE>** d_vkb_ptrs = nullptr;
    std::complex<FPTYPE>* d_sk = nullptr;
    std::complex<FPTYPE>* d_pref_in = nullptr;

    // becp and dbecp:
    std::complex<FPTYPE>* dbecp = nullptr;
    std::complex<FPTYPE>* becp = nullptr;

    using gemm_op = hsolver::gemm_op<std::complex<FPTYPE>, Device>;
    using cal_stress_nl_op = hamilt::cal_stress_nl_op<FPTYPE, Device>;
    using cal_dbecp_noevc_nl_op = hamilt::cal_dbecp_noevc_nl_op<FPTYPE, Device>;

    using resmem_complex_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    using resmem_complex_h_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>;
    using setmem_complex_op = base_device::memory::set_memory_op<std::complex<FPTYPE>, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<std::complex<FPTYPE>, Device>;
    using delmem_complex_h_op = base_device::memory::delete_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>;
    using syncmem_complex_h2d_op
        = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;
    using syncmem_complex_d2h_op
        = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>;

    using resmem_var_op = base_device::memory::resize_memory_op<FPTYPE, Device>;
    using resmem_var_h_op = base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_CPU>;
    using setmem_var_op = base_device::memory::set_memory_op<FPTYPE, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<FPTYPE, Device>;
    using delmem_var_h_op = base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_CPU>;
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    using syncmem_var_d2h_op = base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, Device>;

    using resmem_int_op = base_device::memory::resize_memory_op<int, Device>;
    using delmem_int_op = base_device::memory::delete_memory_op<int, Device>;
    using syncmem_int_h2d_op = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;

    using cal_vq_op = hamilt::cal_vq_op<FPTYPE, Device>;
    using cal_vq_deri_op = hamilt::cal_vq_deri_op<FPTYPE, Device>;

    using cal_vkb_op = hamilt::cal_vkb_op<FPTYPE, Device>;
    using cal_vkb_deri_op = hamilt::cal_vkb_deri_op<FPTYPE, Device>;
};

} // namespace hamilt

#endif