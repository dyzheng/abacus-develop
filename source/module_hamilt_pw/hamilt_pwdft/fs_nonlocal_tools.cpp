#include "fs_nonlocal_tools.h"

#include "module_base/math_polyint.h"
#include "module_base/math_ylmreal.h"
#include "module_base/memory.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/force_op.h"

namespace hamilt
{

template <typename FPTYPE, typename Device>
FS_Nonlocal_tools<FPTYPE, Device>::FS_Nonlocal_tools(pseudopot_cell_vnl* nlpp_in,
                                                     const UnitCell* ucell_in,
                                                     const psi::Psi<std::complex<FPTYPE>, Device>* psi_in,
                                                     const K_Vectors* kv_in,
                                                     const ModulePW::PW_Basis_K* wfc_basis_in,
                                                     const Structure_Factor* sf_in,
                                                     const ModuleBase::matrix& wg,
                                                     const ModuleBase::matrix& ekb)
    : nlpp_(nlpp_in), ucell_(ucell_in), psi_(psi_in), kv_(kv_in), wfc_basis_(wfc_basis_in), sf_(sf_in)
{
    // get the device context
    this->device = base_device::get_device_type<Device>(this->ctx);
    this->nkb = nlpp_->nkb;
    this->nbands = psi_->get_nbands();
    this->max_npw = wfc_basis_->npwk_max;
    this->ntype = ucell_->ntype;

    // There is a contribution for jh<>ih in US case or multi projectors case
    // Actually, the judge of nondiagonal should be done on every atom type
    this->nondiagonal = (GlobalV::use_uspp || this->nlpp_->multi_proj) ? true : false;

    // allocate memory
    this->allocate_memory(wg, ekb);
}

template <typename FPTYPE, typename Device>
FS_Nonlocal_tools<FPTYPE, Device>::~FS_Nonlocal_tools()
{
    // delete memory
    delete_memory();
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::allocate_memory(const ModuleBase::matrix& wg, const ModuleBase::matrix& ekb)
{
    // allocate memory

    // prepare the memory of stress and init some variables:
    this->h_atom_nh = new int[this->ntype], this->h_atom_na = new int[this->ntype];
    for (int ii = 0; ii < this->ntype; ii++)
    {
        h_atom_nh[ii] = this->ucell_->atoms[ii].ncpp.nh;
        h_atom_na[ii] = this->ucell_->atoms[ii].na;
    }

    this->deeq = this->nlpp_->template get_deeq_data<FPTYPE>();
    this->kvec_c = this->wfc_basis_->template get_kvec_c_data<FPTYPE>();
    this->qq_nt = this->nlpp_->template get_qq_nt_data<FPTYPE>();

    int max_nbeta = 0;
    for (int it = 0; it < this->ntype; it++) // loop all elements
    {
        max_nbeta = std::max(this->ucell_->atoms[it].ncpp.nbeta, max_nbeta);
        this->max_nh = std::max(this->ucell_->atoms[it].ncpp.nh, max_nh);
    }

    // allocate the memory on CPU.

    // allocate the memory for vkb and vkb_deri.
    this->vq_ptrs = new FPTYPE*[max_nh];
    this->ylm_ptrs = new FPTYPE*[max_nh];
    this->vq_deri_ptrs = new FPTYPE*[max_nh];
    this->ylm_deri_ptrs1 = new FPTYPE*[max_nh];
    this->ylm_deri_ptrs2 = new FPTYPE*[max_nh];
    this->vkb_ptrs = new std::complex<FPTYPE>*[max_nh];
    if (this->device == base_device::GpuDevice)
    {
        hamilt::pointer_array_malloc<Device>()((void**)&d_ylm_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void**)&d_vq_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void**)&d_vkb_ptrs, max_nh);

        hamilt::pointer_array_malloc<Device>()((void**)&d_vq_deri_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void**)&d_ylm_deri_ptrs1, max_nh);
        hamilt::pointer_array_malloc<Device>()((void**)&d_ylm_deri_ptrs2, max_nh);
    }

    resmem_var_op()(this->ctx, this->hd_vq, max_nbeta * max_npw);
    resmem_var_op()(this->ctx, this->hd_vq_deri, max_nbeta * max_npw);
    const int _lmax = this->nlpp_->lmaxkb;
    resmem_var_op()(this->ctx, this->hd_ylm, (_lmax + 1) * (_lmax + 1) * max_npw);
    resmem_var_op()(this->ctx, this->hd_ylm_deri, 3 * (_lmax + 1) * (_lmax + 1) * max_npw);

    if (this->device == base_device::GpuDevice)
    {
        resmem_var_op()(this->ctx, d_wg, wg.nr * wg.nc);
        resmem_var_op()(this->ctx, d_ekb, ekb.nr * ekb.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_wg, wg.c, wg.nr * wg.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_ekb, ekb.c, ekb.nr * ekb.nc);
        resmem_int_op()(this->ctx, atom_nh, this->ntype);
        resmem_int_op()(this->ctx, atom_na, this->ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_nh, h_atom_nh, this->ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_na, h_atom_na, this->ntype);

        resmem_var_op()(this->ctx, d_g_plus_k, max_npw * 5);
        resmem_var_op()(this->ctx, d_pref, max_nh);
        resmem_var_op()(this->ctx, d_vq_tab, this->nlpp_->tab.getSize());

        resmem_complex_op()(this->ctx, d_sk, max_npw);
        resmem_complex_op()(this->ctx, d_pref_in, max_nh);

        this->ppcell_vkb = this->nlpp_->template get_vkb_data<FPTYPE>();
    }
    else
    {
        this->d_wg = wg.c;
        this->d_ekb = ekb.c;
        this->atom_nh = h_atom_nh;
        this->atom_na = h_atom_na;
        this->ppcell_vkb = this->nlpp_->vkb.c;
    }

    // prepare the memory of the becp and dbecp:
    // becp: <Beta(nkb,npw)|psi(nbnd,npw)>
    // dbecp: <dBeta(nkb,npw)/dG|psi(nbnd,npw)>
    resmem_complex_op()(this->ctx, becp, this->nbands * nkb, "Stress::becp");
    resmem_complex_op()(this->ctx, dbecp, 6 * this->nbands * nkb, "Stress::dbecp");
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::delete_memory()
{
    // delete memory
    delete[] this->h_atom_nh;
    delete[] this->h_atom_na;

    delete[] this->vkb_ptrs;
    delete[] this->vq_ptrs;
    delete[] this->ylm_ptrs;
    delete[] this->vq_deri_ptrs;
    delete[] this->ylm_deri_ptrs1;
    delete[] this->ylm_deri_ptrs2;

    delmem_var_op()(this->ctx, hd_vq);
    delmem_var_op()(this->ctx, hd_vq_deri);
    delmem_var_op()(this->ctx, hd_ylm);
    delmem_var_op()(this->ctx, hd_ylm_deri);

    // delete memory on GPU
    if (this->device == base_device::GpuDevice)
    {
        delmem_var_op()(this->ctx, d_wg);
        delmem_var_op()(this->ctx, d_ekb);
        delmem_int_op()(this->ctx, atom_nh);
        delmem_int_op()(this->ctx, atom_na);
        delmem_var_op()(this->ctx, d_g_plus_k);
        delmem_var_op()(this->ctx, d_pref);
        delmem_var_op()(this->ctx, d_vq_tab);
        delmem_complex_op()(this->ctx, d_sk);
        delmem_complex_op()(this->ctx, this->d_pref_in);
        base_device::memory::delete_memory_op<FPTYPE*, Device>()(this->ctx, d_ylm_ptrs);
        base_device::memory::delete_memory_op<FPTYPE*, Device>()(this->ctx, d_vq_ptrs);
        base_device::memory::delete_memory_op<std::complex<FPTYPE>*, Device>()(this->ctx, d_vkb_ptrs);
        base_device::memory::delete_memory_op<FPTYPE*, Device>()(this->ctx, d_vq_deri_ptrs);
        base_device::memory::delete_memory_op<FPTYPE*, Device>()(this->ctx, d_ylm_deri_ptrs1);
        base_device::memory::delete_memory_op<FPTYPE*, Device>()(this->ctx, d_ylm_deri_ptrs2);
    }

    if (becp != nullptr)
    {
        delmem_complex_op()(this->ctx, becp);
    }
    if (dbecp != nullptr)
    {
        delmem_complex_op()(this->ctx, dbecp);
    }
    if (this->pre_ik_f != -1)
    {
        delmem_int_op()(this->ctx, gcar_zero_indexes);
        delmem_complex_op()(this->ctx, vkb_save);
        delmem_var_op()(this->ctx, gcar);
    }
}

// cal_becp
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_becp(int ik, int npm)
{
    if (this->becp == nullptr)
    {
        resmem_complex_op()(this->ctx, becp, this->nbands * this->nkb);
    }

    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    const int npw = this->wfc_basis_->npwk[ik];

    std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;

    std::vector<FPTYPE> g_plus_k = cal_gk(ik, this->wfc_basis_);
    int lmax_ = this->nlpp_->lmaxkb;
    // prepare ylm，size: (lmax+1)^2 * this->max_npw
    cal_ylm(lmax_, npw, g_plus_k.data(), hd_ylm);
    for (int it = 0; it < this->ucell_->ntype; it++) // loop all elements
    {
        int lenth_vq = this->ucell_->atoms[it].ncpp.nbeta * npw;
        // prepare inputs for calculating vkb，vkb1，vkb2
        // prepare vq and vq', size: nq * this->max_npw
        std::vector<double> vq(lenth_vq); // cal_vq(it, g_plus_k.data(), npw);
        // std::vector<double> vq2(vq.size());

        FPTYPE *gk = g_plus_k.data(), *vq_tb = this->nlpp_->tab.ptr;

        if (this->device == base_device::GpuDevice)
        {
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_g_plus_k, g_plus_k.data(), g_plus_k.size());
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_vq_tab, this->nlpp_->tab.ptr, this->nlpp_->tab.getSize());
            gk = d_g_plus_k;
            vq_tb = d_vq_tab;
        }
        cal_vq_op()(this->ctx,
                    vq_tb,
                    it,
                    gk,
                    npw,
                    this->nlpp_->tab.getBound2(),
                    this->nlpp_->tab.getBound3(),
                    GlobalV::DQ,
                    this->ucell_->atoms[it].ncpp.nbeta,
                    hd_vq);
        cal_vq_deri_op()(this->ctx,
                         vq_tb,
                         it,
                         gk,
                         npw,
                         this->nlpp_->tab.getBound2(),
                         this->nlpp_->tab.getBound3(),
                         GlobalV::DQ,
                         this->ucell_->atoms[it].ncpp.nbeta,
                         hd_vq_deri);

        // prepare（-i）^l, size: nh
        std::vector<std::complex<double>> pref = cal_pref(it);
        int nh = pref.size();

        for (int ia = 0; ia < h_atom_na[it]; ia++)
        {
            // prepare SK
            std::complex<FPTYPE>* sk = this->sf_->get_sk(ik, it, ia, this->wfc_basis_);
            // 1. calculate becp
            // 1.a calculate vkb

            if (this->device == base_device::GpuDevice)
            {
                syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_sk, sk, npw);
                syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_pref_in, pref.data(), nh);

                prepare_vkb_ptr(this->ucell_->atoms[it].ncpp.nbeta,
                                this->nlpp_->nhtol.c,
                                this->nlpp_->nhtol.nc,
                                npw,
                                it,
                                vkb_ptr,
                                vkb_ptrs,
                                hd_ylm,
                                ylm_ptrs,
                                hd_vq,
                                vq_ptrs);

                // transfer the pointers from CPU to GPU
                hamilt::synchronize_ptrs<Device>()((void**)d_vq_ptrs, (const void**)vq_ptrs, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_ylm_ptrs, (const void**)ylm_ptrs, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_vkb_ptrs, (const void**)vkb_ptrs, nh);

                cal_vkb_op()(this->ctx, nh, npw, d_vq_ptrs, d_ylm_ptrs, d_sk, d_pref_in, d_vkb_ptrs);

                // syncmem_complex_d2h_op()(this->cpu_ctx, this->ctx, ppcell_vkb, ppcell_vkb_d, nh * npw);
            }
            else
            {
                prepare_vkb_ptr(this->ucell_->atoms[it].ncpp.nbeta,
                                this->nlpp_->nhtol.c,
                                this->nlpp_->nhtol.nc,
                                npw,
                                it,
                                vkb_ptr,
                                vkb_ptrs,
                                hd_ylm,
                                ylm_ptrs,
                                hd_vq,
                                vq_ptrs);

                cal_vkb_op()(this->ctx, nh, npw, vq_ptrs, ylm_ptrs, sk, pref.data(), vkb_ptrs);
            }
            delete[] sk;

            // 2.b calculate becp = vkb * psi
            vkb_ptr += nh * npw;
        }
    }
    const char transa = 'C';
    const char transb = 'N';
    gemm_op()(this->ctx,
              transa,
              transb,
              nkb,
              npm,
              npw,
              &ModuleBase::ONE,
              ppcell_vkb,
              npw,
              ppsi,
              this->max_npw,
              &ModuleBase::ZERO,
              becp,
              nkb);

    // becp calculate is over , now we should broadcast this data.
    if (this->device == base_device::GpuDevice)
    {
        std::complex<FPTYPE>* h_becp = nullptr;
        resmem_complex_h_op()(this->cpu_ctx, h_becp, this->nbands * nkb);
        syncmem_complex_d2h_op()(this->cpu_ctx, this->ctx, h_becp, becp, this->nbands * nkb);
        Parallel_Reduce::reduce_pool(h_becp, this->nbands * nkb);
        syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, becp, h_becp, this->nbands * nkb);
        delmem_complex_h_op()(this->cpu_ctx, h_becp);
    }
    else
    {
        Parallel_Reduce::reduce_pool(becp, this->nbands * this->nkb);
    }
}

// cal_dbecp
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_dbecp_s(int ik, int npm, int ipol, int jpol, FPTYPE* stress)
{
    if (this->dbecp == nullptr)
    {
        resmem_complex_op()(this->ctx, dbecp, this->nbands * this->nkb);
    }

    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    const int npw = this->wfc_basis_->npwk[ik];
    std::complex<FPTYPE>* vkb_deri_ptr = this->ppcell_vkb;

    if (this->pre_ik_s != ik)
    { // k point has changed, we need to recalculate the g_plus_k
        this->g_plus_k = cal_gk(ik, this->wfc_basis_);

        int lmax_ = this->nlpp_->lmaxkb;
        // prepare ylm，size: (lmax+1)^2 * this->max_npw
        cal_ylm(lmax_, npw, g_plus_k.data(), hd_ylm);
        cal_ylm_deri(lmax_, npw, g_plus_k.data(), hd_ylm_deri);
        this->pre_ik_s = ik;
    }
    for (int it = 0; it < this->ucell_->ntype; it++) // loop all elements
    {
        int lenth_vq = this->ucell_->atoms[it].ncpp.nbeta * npw;
        // prepare inputs for calculating vkb，vkb1，vkb2
        // prepare vq and vq', size: nq * this->max_npw
        std::vector<double> vq(lenth_vq); // cal_vq(it, g_plus_k.data(), npw);
        // std::vector<double> vq2(vq.size());

        FPTYPE *gk = g_plus_k.data(), *vq_tb = this->nlpp_->tab.ptr;

        if (this->device == base_device::GpuDevice)
        {
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_g_plus_k, g_plus_k.data(), g_plus_k.size());
            syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_vq_tab, this->nlpp_->tab.ptr, this->nlpp_->tab.getSize());
            gk = d_g_plus_k;
            vq_tb = d_vq_tab;
        }
        cal_vq_op()(this->ctx,
                    vq_tb,
                    it,
                    gk,
                    npw,
                    this->nlpp_->tab.getBound2(),
                    this->nlpp_->tab.getBound3(),
                    GlobalV::DQ,
                    this->ucell_->atoms[it].ncpp.nbeta,
                    hd_vq);
        cal_vq_deri_op()(this->ctx,
                         vq_tb,
                         it,
                         gk,
                         npw,
                         this->nlpp_->tab.getBound2(),
                         this->nlpp_->tab.getBound3(),
                         GlobalV::DQ,
                         this->ucell_->atoms[it].ncpp.nbeta,
                         hd_vq_deri);
        // prepare（-i）^l, size: nh
        std::vector<std::complex<double>> pref = cal_pref(it);
        int nh = pref.size();
        for (int ia = 0; ia < h_atom_na[it]; ia++)
        {
            std::complex<FPTYPE>* sk = this->sf_->get_sk(ik, it, ia, this->wfc_basis_);
            int index = 0;
            // 2. calculate dbecp：
            // 2.a. calculate dbecp_noevc, repeat use the memory of ppcell.vkb

            if (this->device == base_device::GpuDevice)
            {
                syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_sk, sk, npw);
                syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_pref_in, pref.data(), nh);

                prepare_vkb_deri_ptr(this->ucell_->atoms[it].ncpp.nbeta,
                                     this->nlpp_->nhtol.c,
                                     this->nlpp_->nhtol.nc,
                                     npw,
                                     it,
                                     ipol,
                                     jpol,
                                     vkb_deri_ptr,
                                     vkb_ptrs,
                                     hd_ylm,
                                     ylm_ptrs,
                                     hd_ylm_deri,
                                     ylm_deri_ptrs1,
                                     ylm_deri_ptrs2,
                                     hd_vq,
                                     vq_ptrs,
                                     hd_vq_deri,
                                     vq_deri_ptrs);

                // transfer the pointers from CPU to GPU
                hamilt::synchronize_ptrs<Device>()((void**)d_vq_ptrs, (const void**)vq_ptrs, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_vq_deri_ptrs, (const void**)vq_deri_ptrs, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_ylm_ptrs, (const void**)ylm_ptrs, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_ylm_deri_ptrs1, (const void**)ylm_deri_ptrs1, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_ylm_deri_ptrs2, (const void**)ylm_deri_ptrs2, nh);
                hamilt::synchronize_ptrs<Device>()((void**)d_vkb_ptrs, (const void**)vkb_ptrs, nh);
                cal_vkb_deri_op()(this->ctx,
                                  nh,
                                  npw,
                                  ipol,
                                  jpol,
                                  d_vq_ptrs,
                                  d_vq_deri_ptrs,
                                  d_ylm_ptrs,
                                  d_ylm_deri_ptrs1,
                                  d_ylm_deri_ptrs2,
                                  d_sk,
                                  d_pref_in,
                                  d_g_plus_k,
                                  d_vkb_ptrs);
            }
            else
            {

                prepare_vkb_deri_ptr(this->ucell_->atoms[it].ncpp.nbeta,
                                     this->nlpp_->nhtol.c,
                                     this->nlpp_->nhtol.nc,
                                     npw,
                                     it,
                                     ipol,
                                     jpol,
                                     vkb_deri_ptr,
                                     vkb_ptrs,
                                     hd_ylm,
                                     ylm_ptrs,
                                     hd_ylm_deri,
                                     ylm_deri_ptrs1,
                                     ylm_deri_ptrs2,
                                     hd_vq,
                                     vq_ptrs,
                                     hd_vq_deri,
                                     vq_deri_ptrs);
                cal_vkb_deri_op()(this->ctx,
                                  nh,
                                  npw,
                                  ipol,
                                  jpol,
                                  vq_ptrs,
                                  vq_deri_ptrs,
                                  ylm_ptrs,
                                  ylm_deri_ptrs1,
                                  ylm_deri_ptrs2,
                                  sk,
                                  pref.data(),
                                  g_plus_k.data(),
                                  vkb_ptrs);
            }
            delete[] sk;

            vkb_deri_ptr += nh * npw;
        }
    }
    // 2.b calculate dbecp = dbecp_noevc * psi
    const char transa = 'C';
    const char transb = 'N';

    gemm_op()(this->ctx,
              transa,
              transb,
              nkb,
              npm,
              npw,
              &ModuleBase::ONE,
              ppcell_vkb,
              npw,
              ppsi,
              this->max_npw,
              &ModuleBase::ZERO,
              dbecp,
              nkb);
    // calculate stress for target (ipol, jpol)
    const int current_spin = this->kv_->isk[ik];
    cal_stress_nl_op()(this->ctx,
                       nondiagonal,
                       ipol,
                       jpol,
                       nkb,
                       npm,
                       this->ntype,
                       current_spin, // uspp only
                       this->nbands,
                       ik,
                       this->nlpp_->deeq.getBound2(),
                       this->nlpp_->deeq.getBound3(),
                       this->nlpp_->deeq.getBound4(),
                       atom_nh,
                       atom_na,
                       d_wg,
                       d_ekb,
                       qq_nt,
                       deeq,
                       becp,
                       dbecp,
                       stress);
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_dbecp_f(int ik, int npm, int ipol)
{
    if (this->dbecp == nullptr)
    {
        resmem_complex_op()(this->ctx, dbecp, 3 * this->nbands * this->nkb);
    }
    std::complex<FPTYPE>* dbecp_ptr = this->dbecp + ipol * this->nbands * this->nkb;
    const std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
    std::complex<FPTYPE>* vkb_deri_ptr = this->ppcell_vkb;

    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    const int npw = this->wfc_basis_->npwk[ik];
    if (this->pre_ik_f == -1)
    {
        resmem_var_op()(this->ctx, gcar, 3 * this->wfc_basis_->npwk_max);
        resmem_int_op()(this->ctx, gcar_zero_indexes, 3 * this->wfc_basis_->npwk_max);
    }

    if (this->pre_ik_f != ik)
    {
        this->transfer_gcar(npw,
                            this->wfc_basis_->npwk_max,
                            &(this->wfc_basis_->gcar[ik * this->wfc_basis_->npwk_max].x));
    }

    this->save_vkb(npw, ipol);

    const std::complex<double> coeff = ipol == 0 ? ModuleBase::NEG_IMAG_UNIT : ModuleBase::ONE;

    // calculate the vkb_deri for ipol with the memory of ppcell_vkb
    cal_vkb1_nl_op<FPTYPE, Device>()(this->ctx, nkb, npw, npw, npw, ipol, coeff, vkb_ptr, gcar, vkb_deri_ptr);

    // do gemm to get dbecp and revert the ppcell_vkb for next ipol
    const char transa = 'C';
    const char transb = 'N';
    gemm_op()(this->ctx,
              transa,
              transb,
              this->nkb,
              npm,
              npw,
              &ModuleBase::ONE,
              vkb_deri_ptr,
              npw,
              ppsi,
              this->max_npw,
              &ModuleBase::ZERO,
              dbecp_ptr,
              nkb);

    this->revert_vkb(npw, ipol);
    this->pre_ik_f = ik;
}

// save_vkb
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::save_vkb(int npw, int ipol)
{
    if(this->device == base_device::CpuDevice)
    {
        const int gcar_zero_count = this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max];
        const int* gcar_zero_ptrs = &this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max+1];
        const std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
        std::complex<FPTYPE>* vkb_save_ptr = this->vkb_save;
        //find the zero indexes to save the vkb values to vkb_save
        for(int ikb = 0;ikb < this->nkb;++ikb)
        {
            for (int icount = 0; icount < gcar_zero_count; ++icount)
            {
                *vkb_save_ptr = vkb_ptr[gcar_zero_ptrs[icount]];
                ++vkb_save_ptr;
            }
            vkb_ptr += npw;
        }
    }
    else
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
	    saveVkbValues<FPTYPE>(
                this->gcar_zero_indexes, 
                this->ppcell_vkb, 
                this->vkb_save, 
                nkb, 
                this->gcar_zero_counts[ipol],
                npw,
                ipol,
                this->wfc_basis_->npwk_max);
#endif
    }
}

// revert_vkb
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::revert_vkb(int npw, int ipol)
{
    const std::complex<FPTYPE> coeff = ipol==0?ModuleBase::NEG_IMAG_UNIT:ModuleBase::ONE;
    if(this->device == base_device::CpuDevice)
    {
        const int gcar_zero_count = this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max];
        const int* gcar_zero_ptrs = &this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max+1];
        std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
        const std::complex<FPTYPE>* vkb_save_ptr = this->vkb_save;
        //find the zero indexes to save the vkb values to vkb_save
        for(int ikb = 0;ikb < this->nkb;++ikb)
        {
            for (int icount = 0; icount < gcar_zero_count; ++icount)
            {
                vkb_ptr[gcar_zero_ptrs[icount]] = *vkb_save_ptr * coeff;
                ++vkb_save_ptr;
            }
            vkb_ptr += npw;
        }
    }
    else
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        revertVkbValues<FPTYPE>(
            this->gcar_zero_indexes, 
            this->ppcell_vkb, 
            this->vkb_save, 
            nkb, 
            this->gcar_zero_counts[ipol],
            npw, 
            ipol,
            this->wfc_basis_->npwk_max,
            coeff);
#endif
    }
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::transfer_gcar(int npw, int npw_max, const FPTYPE* gcar_in)
{
    std::vector<FPTYPE> gcar_tmp(3 * npw_max);
    gcar_tmp.assign(gcar_in, gcar_in + 3 * npw_max);
    std::vector<int> gcar_zero_indexes_tmp(3 * npw_max);

    int* gcar_zero_ptrs[3];
    for (int i = 0; i < 3; i++)
    {
        gcar_zero_ptrs[i] = &gcar_zero_indexes_tmp[i * npw_max];
        gcar_zero_ptrs[i][0] = -1;
        this->gcar_zero_counts[i] = 0;
    }
    for (int ig = 0; ig < npw; ig++)
    {
        // calculate gcar.x , gcar.y/gcar.x, gcar.z/gcar.y
        // if individual gcar is less than 1e-15, we will record the index
        for (int i = 0; i < 3; ++i)
        {
            if (std::abs(gcar_tmp[ig * 3 + i]) < 1e-15)
            {
                ++gcar_zero_counts[i];
                gcar_zero_ptrs[i][gcar_zero_counts[i]] = ig;
            }
        }
        // four cases for the gcar of y and z
        if (gcar_zero_ptrs[0][gcar_zero_counts[0]] == ig && gcar_zero_ptrs[1][gcar_zero_counts[1]] == ig)
        { // x == y == 0, z = z
        }
        else if (gcar_zero_ptrs[0][gcar_zero_counts[0]] != ig && gcar_zero_ptrs[1][gcar_zero_counts[1]] == ig)
        { // x != 0, y == 0, z = z/x
            gcar_tmp[ig * 3 + 2] /= gcar_tmp[ig * 3];
        }
        else if (gcar_zero_ptrs[0][gcar_zero_counts[0]] == ig && gcar_zero_ptrs[1][gcar_zero_counts[1]] != ig)
        { // x == 0, y != 0, y = y, z = z/y
            gcar_tmp[ig * 3 + 2] /= gcar_tmp[ig * 3 + 1];
        }
        else
        { // x != 0, y != 0, y = y/x, z = z/y
            gcar_tmp[ig * 3 + 2] /= gcar_tmp[ig * 3 + 1];
            gcar_tmp[ig * 3 + 1] /= gcar_tmp[ig * 3];
        }
    }
    for (int i = 0; i < 3; ++i)
    { // record the counts to the first element
        gcar_zero_ptrs[i][0] = gcar_zero_counts[i];
    }
    // prepare the memory for vkb_save
    const int max_count = std::max(gcar_zero_counts[0], std::max(gcar_zero_counts[1], gcar_zero_counts[2]));
    resmem_complex_op()(this->ctx, this->vkb_save, this->nkb * max_count);
    // transfer the gcar and gcar_zero_indexes to the device
    syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, gcar, gcar_tmp.data(), 3 * npw_max);
    syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, gcar_zero_indexes, gcar_zero_indexes_tmp.data(), 3 * npw_max);
}

// cal_force
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_force(int ik, int npm, FPTYPE* force)
{
    const int current_spin = this->kv_->isk[ik];
    const int force_nc = 3;
    // calculate the force
    cal_force_nl_op<FPTYPE, Device>()(this->ctx,
                                      nondiagonal,
                                      npm,
                                      this->nbands,
                                      this->ntype,
                                      current_spin,
                                      this->nlpp_->deeq.getBound2(),
                                      this->nlpp_->deeq.getBound3(),
                                      this->nlpp_->deeq.getBound4(),
                                      force_nc,
                                      this->nbands,
                                      ik,
                                      nkb,
                                      atom_nh,
                                      atom_na,
                                      this->ucell_->tpiba,
                                      d_wg,
                                      d_ekb,
                                      qq_nt,
                                      deeq,
                                      becp,
                                      dbecp,
                                      force);
}

// cal_gk
template <typename FPTYPE, typename Device>
std::vector<FPTYPE> FS_Nonlocal_tools<FPTYPE, Device>::cal_gk(int ik, const ModulePW::PW_Basis_K* wfc_basis)
{
    int npw = wfc_basis->npwk[ik];
    std::vector<FPTYPE> gk(npw * 5);
    ModuleBase::Memory::record("stress_nl::gk", 5 * npw * sizeof(FPTYPE));
    ModuleBase::Vector3<FPTYPE> tmp;
    for (int ig = 0; ig < npw; ++ig)
    {
        tmp = wfc_basis->getgpluskcar(ik, ig);
        gk[ig * 3] = tmp.x;
        gk[ig * 3 + 1] = tmp.y;
        gk[ig * 3 + 2] = tmp.z;
        FPTYPE norm = sqrt(tmp.norm2());
        gk[3 * npw + ig] = norm * this->ucell_->tpiba;
        gk[4 * npw + ig] = norm < 1e-8 ? 0.0 : 1.0 / norm * this->ucell_->tpiba;
    }
    return gk;
}

// cal_vq
template <typename FPTYPE, typename Device>
std::vector<FPTYPE> FS_Nonlocal_tools<FPTYPE, Device>::cal_vq(int it, const FPTYPE* gk, int npw)
{
    // calculate beta in G-space using an interpolation table
    const int nbeta = this->ucell_->atoms[it].ncpp.nbeta;

    std::vector<FPTYPE> vq(nbeta * npw);
    ModuleBase::Memory::record("stress_nl::vq", nbeta * npw * sizeof(FPTYPE));

    for (int nb = 0; nb < nbeta; nb++)
    {
        FPTYPE* vq_ptr = &vq[nb * npw];
        const FPTYPE* gnorm = &gk[3 * npw];
        for (int ig = 0; ig < npw; ig++)
        {
            vq_ptr[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(this->nlpp_->tab,
                                                                       it,
                                                                       nb,
                                                                       GlobalV::NQX,
                                                                       GlobalV::DQ,
                                                                       gnorm[ig]);
        }
    }
    return vq;
}

// cal_vq_deri
template <typename FPTYPE, typename Device>
std::vector<FPTYPE> FS_Nonlocal_tools<FPTYPE, Device>::cal_vq_deri(int it, const FPTYPE* gk, int npw)
{
    // calculate beta in G-space using an interpolation table
    const int nbeta = this->ucell_->atoms[it].ncpp.nbeta;

    std::vector<FPTYPE> vq(nbeta * npw);
    ModuleBase::Memory::record("stress_nl::vq_deri", nbeta * npw * sizeof(FPTYPE));

    for (int nb = 0; nb < nbeta; nb++)
    {
        const FPTYPE* gnorm = &gk[3 * npw];
        FPTYPE* vq_ptr = &vq[nb * npw];
        for (int ig = 0; ig < npw; ig++)
        {
            vq_ptr[ig] = this->Polynomial_Interpolation_nl(this->nlpp_->tab, it, nb, GlobalV::DQ, gnorm[ig]);
        }
    }
    return vq;
}

// cal_ylm
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_ylm(int lmax, int npw, const FPTYPE* gk_in, FPTYPE* ylm)
{

    int x1 = (lmax + 1) * (lmax + 1);
    ModuleBase::Memory::record("stress_nl::ylm", x1 * npw * sizeof(FPTYPE));

    if (this->device == base_device::GpuDevice)
    {
        std::vector<FPTYPE> ylm_cpu(x1 * npw);
        ModuleBase::YlmReal::Ylm_Real(cpu_ctx, x1, npw, gk_in, ylm_cpu.data());
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, ylm, ylm_cpu.data(), ylm_cpu.size());
    }
    else
    {
        ModuleBase::YlmReal::Ylm_Real(cpu_ctx, x1, npw, gk_in, ylm);
    }

    return;
}
// cal_ylm_deri
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_ylm_deri(int lmax, int npw, const FPTYPE* gk_in, FPTYPE* ylm_deri)
{
    int x1 = (lmax + 1) * (lmax + 1);
    ModuleBase::Memory::record("stress_nl::dylm", 3 * x1 * npw * sizeof(FPTYPE));

    if (this->device == base_device::GpuDevice)
    {
        std::vector<FPTYPE> dylm(3 * x1 * npw);
        for (int ipol = 0; ipol < 3; ipol++)
        {
            FS_Nonlocal_tools<FPTYPE, Device>::dylmr2(x1, npw, gk_in, &dylm[ipol * x1 * npw], ipol);
        }
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, ylm_deri, dylm.data(), dylm.size());
    }
    else
    {
        for (int ipol = 0; ipol < 3; ipol++)
        {
            FS_Nonlocal_tools<FPTYPE, Device>::dylmr2(x1, npw, gk_in, &ylm_deri[ipol * x1 * npw], ipol);
        }
    }

    return;
}
// cal_pref
template <typename FPTYPE, typename Device>
std::vector<std::complex<FPTYPE>> FS_Nonlocal_tools<FPTYPE, Device>::cal_pref(int it)
{
    const int nh = this->ucell_->atoms[it].ncpp.nh;
    std::vector<std::complex<FPTYPE>> pref(nh);
    for (int ih = 0; ih < nh; ih++)
    {
        pref[ih] = std::pow(std::complex<FPTYPE>(0.0, -1.0), this->nlpp_->nhtol(it, ih));
    }
    return pref;
}

// cal_vkb
// cpu version first, gpu version later
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_vkb(int it,
                                                int ia,
                                                int npw,
                                                const FPTYPE* vq_in,
                                                const FPTYPE* ylm_in,
                                                const std::complex<FPTYPE>* sk_in,
                                                const std::complex<FPTYPE>* pref_in,
                                                std::complex<FPTYPE>* vkb_out)
{
    int ih = 0;
    // loop over all beta functions
    for (int nb = 0; nb < this->ucell_->atoms[it].ncpp.nbeta; nb++)
    {
        int l = this->nlpp_->nhtol(it, ih);
        // loop over all m angular momentum
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            std::complex<FPTYPE>* vkb_ptr = &vkb_out[ih * npw];
            const FPTYPE* ylm_ptr = &ylm_in[lm * npw];
            const FPTYPE* vq_ptr = &vq_in[nb * npw];
            // loop over all G-vectors
            for (int ig = 0; ig < npw; ig++)
            {
                vkb_ptr[ig] = ylm_ptr[ig] * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
            }
            ih++;
        }
    }
}

// cal_vkb
// cpu version first, gpu version later
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_vkb_deri(int it,
                                                     int ia,
                                                     int npw,
                                                     int ipol,
                                                     int jpol,
                                                     const FPTYPE* vq_in,
                                                     const FPTYPE* vq_deri_in,
                                                     const FPTYPE* ylm_in,
                                                     const FPTYPE* ylm_deri_in,
                                                     const std::complex<FPTYPE>* sk_in,
                                                     const std::complex<FPTYPE>* pref_in,
                                                     const FPTYPE* gk_in,
                                                     std::complex<FPTYPE>* vkb_out)
{
    int x1 = (this->nlpp_->lmaxkb + 1) * (this->nlpp_->lmaxkb + 1);
    int ih = 0;
    // loop over all beta functions
    for (int nb = 0; nb < this->ucell_->atoms[it].ncpp.nbeta; nb++)
    {
        int l = this->nlpp_->nhtol(it, ih);
        // loop over all m angular momentum
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            std::complex<FPTYPE>* vkb_ptr = &vkb_out[ih * npw];
            const FPTYPE* ylm_ptr = &ylm_in[lm * npw];
            const FPTYPE* vq_ptr = &vq_in[nb * npw];
            // set vkb to zero
            for (int ig = 0; ig < npw; ig++)
            {
                vkb_ptr[ig] = std::complex<FPTYPE>(0.0, 0.0);
            }
            // first term: ylm * vq * sk * pref
            // loop over all G-vectors
            if (ipol == jpol)
            {
                for (int ig = 0; ig < npw; ig++)
                {
                    vkb_ptr[ig] -= ylm_ptr[ig] * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
                }
            }
            // second term: ylm_deri * vq_deri * sk * pref
            //  loop over all G-vectors
            const FPTYPE* ylm_deri_ptr1 = &ylm_deri_in[(ipol * x1 + lm) * npw];
            const FPTYPE* ylm_deri_ptr2 = &ylm_deri_in[(jpol * x1 + lm) * npw];
            const FPTYPE* vq_deri_ptr = &vq_deri_in[nb * npw];
            const FPTYPE* gkn = &gk_in[4 * npw];
            for (int ig = 0; ig < npw; ig++)
            {
                vkb_ptr[ig] -= (gk_in[ig * 3 + ipol] * ylm_deri_ptr2[ig] + gk_in[ig * 3 + jpol] * ylm_deri_ptr1[ig])
                               * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
            }
            // third term: ylm * vq_deri * sk * pref
            //  loop over all G-vectors
            for (int ig = 0; ig < npw; ig++)
            {
                vkb_ptr[ig] -= 2.0 * ylm_ptr[ig] * vq_deri_ptr[ig] * sk_in[ig] * pref_in[ih] * gk_in[ig * 3 + ipol]
                               * gk_in[ig * 3 + jpol] * gkn[ig];
            }
            ih++;
        }
    }
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::prepare_vkb_ptr(int nbeta,
                                                        double* nhtol,
                                                        int nhtol_nc,
                                                        int npw,
                                                        int it,
                                                        std::complex<FPTYPE>* vkb_out,
                                                        std::complex<FPTYPE>** vkb_ptrs,
                                                        FPTYPE* ylm_in,
                                                        FPTYPE** ylm_ptrs,
                                                        FPTYPE* vq_in,
                                                        FPTYPE** vq_ptrs)
{
    // std::complex<FPTYPE>** vkb_ptrs[nh];
    // const FPTYPE** ylm_ptrs[nh];
    // const FPTYPE** vq_ptrs[nh];
    int ih = 0;
    for (int nb = 0; nb < nbeta; nb++)
    {
        int l = nhtol[it * nhtol_nc + ih];
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            vkb_ptrs[ih] = &vkb_out[ih * npw];
            ylm_ptrs[ih] = &ylm_in[lm * npw];
            vq_ptrs[ih] = &vq_in[nb * npw];
            ih++;
        }
    }
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::prepare_vkb_deri_ptr(int nbeta,
                                                             double* nhtol,
                                                             int nhtol_nc,
                                                             int npw,
                                                             int it,
                                                             int ipol,
                                                             int jpol,
                                                             std::complex<FPTYPE>* vkb_out,
                                                             std::complex<FPTYPE>** vkb_ptrs,
                                                             FPTYPE* ylm_in,
                                                             FPTYPE** ylm_ptrs,
                                                             FPTYPE* ylm_deri_in,
                                                             FPTYPE** ylm_deri_ptr1s,
                                                             FPTYPE** ylm_deri_ptr2s,
                                                             FPTYPE* vq_in,
                                                             FPTYPE** vq_ptrs,
                                                             FPTYPE* vq_deri_in,
                                                             FPTYPE** vq_deri_ptrs

)
{
    int ih = 0;
    int x1 = (this->nlpp_->lmaxkb + 1) * (this->nlpp_->lmaxkb + 1);
    for (int nb = 0; nb < nbeta; nb++)
    {
        int l = nhtol[it * nhtol_nc + ih];
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            vkb_ptrs[ih] = &vkb_out[ih * npw];
            ylm_ptrs[ih] = &ylm_in[lm * npw];
            vq_ptrs[ih] = &vq_in[nb * npw];

            ylm_deri_ptr1s[ih] = &ylm_deri_in[(ipol * x1 + lm) * npw];
            ylm_deri_ptr2s[ih] = &ylm_deri_in[(jpol * x1 + lm) * npw];
            vq_deri_ptrs[ih] = &vq_deri_in[nb * npw];

            ih++;
        }
    }
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::dylmr2(const int nylm,
                                               const int ngy,
                                               const FPTYPE* gk,
                                               FPTYPE* dylm,
                                               const int ipol)
{
    //-----------------------------------------------------------------------
    //
    //     compute \partial Y_lm(G) \over \partial (G)_ipol
    //     using simple numerical derivation (SdG)
    //     The spherical harmonics are calculated in ylmr2
    //
    // int nylm, ngy, ipol;
    // number of spherical harmonics
    // the number of g vectors to compute
    // desired polarization
    // FPTYPE g (3, ngy), gg (ngy), dylm (ngy, nylm)
    // the coordinates of g vectors
    // the moduli of g vectors
    // the spherical harmonics derivatives
    //
    const FPTYPE delta = 1e-6;
    const FPTYPE small = 1e-15;

    ModuleBase::matrix ylmaux;
    // dg is the finite increment for numerical derivation:
    // dg = delta |G| = delta * sqrt(gg)
    // dgi= 1 /(delta * sqrt(gg))
    // gx = g +/- dg

    std::vector<FPTYPE> gx(ngy * 3);

    std::vector<FPTYPE> dg(ngy);
    std::vector<FPTYPE> dgi(ngy);

    ylmaux.create(nylm, ngy);

    ModuleBase::GlobalFunc::ZEROS(dylm, nylm * ngy);
    ylmaux.zero_out();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ig = 0; ig < 3 * ngy; ig++)
    {
        gx[ig] = gk[ig];
    }
    //$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ig = 0; ig < ngy; ig++)
    {
        const int igx = ig * 3, igy = ig * 3 + 1, igz = ig * 3 + 2;
        FPTYPE norm2 = gx[igx] * gx[igx] + gx[igy] * gx[igy] + gx[igz] * gx[igz];
        dg[ig] = delta * sqrt(norm2);
        if (dg[ig] > small)
        {
            dgi[ig] = 1.0 / dg[ig];
        }
        else
        {
            dgi[ig] = 0.0;
        }
    }
    //$OMP END PARALLEL DO

    //$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ig = 0; ig < ngy; ig++)
    {
        const int index = ig * 3 + ipol;
        gx[index] = gk[index] + dg[ig];
    }
    //$OMP END PARALLEL DO

    base_device::DEVICE_CPU* cpu = {};
    ModuleBase::YlmReal::Ylm_Real(cpu, nylm, ngy, gx.data(), dylm);
    //$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int ig = 0; ig < ngy; ig++)
    {
        const int index = ig * 3 + ipol;
        gx[index] = gk[index] - dg[ig];
    }
    //$OMP END PARALLEL DO

    ModuleBase::YlmReal::Ylm_Real(cpu, nylm, ngy, gx.data(), ylmaux.c);

    //  zaxpy ( - 1.0, ylmaux, 1, dylm, 1);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int lm = 0; lm < nylm; lm++)
    {
        for (int ig = 0; ig < ngy; ig++)
        {
            dylm[lm * ngy + ig] -= ylmaux(lm, ig);
            dylm[lm * ngy + ig] *= 0.5 * dgi[ig];
        }
    }

    return;
}

template <typename FPTYPE, typename Device>
FPTYPE FS_Nonlocal_tools<FPTYPE, Device>::Polynomial_Interpolation_nl(const ModuleBase::realArray& table,
                                                                      const int& dim1,
                                                                      const int& dim2,
                                                                      const FPTYPE& table_interval,
                                                                      const FPTYPE& x // input value
)
{

    assert(table_interval > 0.0);
    const FPTYPE position = x / table_interval;
    const int iq = static_cast<int>(position);

    const FPTYPE x0 = position - static_cast<FPTYPE>(iq);
    const FPTYPE x1 = 1.0 - x0;
    const FPTYPE x2 = 2.0 - x0;
    const FPTYPE x3 = 3.0 - x0;
    const FPTYPE y = (table(dim1, dim2, iq) * (-x2 * x3 - x1 * x3 - x1 * x2) / 6.0
                      + table(dim1, dim2, iq + 1) * (+x2 * x3 - x0 * x3 - x0 * x2) / 2.0
                      - table(dim1, dim2, iq + 2) * (+x1 * x3 - x0 * x3 - x0 * x1) / 2.0
                      + table(dim1, dim2, iq + 3) * (+x1 * x2 - x0 * x2 - x0 * x1) / 6.0)
                     / table_interval;

    return y;
}

// template instantiation
template class FS_Nonlocal_tools<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class FS_Nonlocal_tools<double, base_device::DEVICE_GPU>;
#endif

} // namespace hamilt
