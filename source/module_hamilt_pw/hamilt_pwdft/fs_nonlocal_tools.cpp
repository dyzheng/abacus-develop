#include "fs_nonlocal_tools.h"

#include "module_base/math_polyint.h"
#include "module_base/math_ylmreal.h"
#include "module_base/memory.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/force_op.h"
#include "nonlocal_maths.hpp"

namespace hamilt
{
// all stories start from the following function. This function retrieves the interpolation table
// from instance of pseudopot_cell_vnl, the tab member variable. It will be send to gpu if the device_id
// is GpuDevice.
template<typename FPTYPE, typename Device>
void retrieve_interptab(FPTYPE* tab, 
                        const int size, 
                        FPTYPE* tab_host, 
                        FPTYPE* tab_device, 
                        Device* device_to,  
                        Device* device_from,
                        const base_device::AbacusDevice_t device_id)
{
    // ----------------------------------------------------------------------------------->8
    // fetch the ppvnl.tab here, but...why??? it will never change...
    // explicit interpolation table, involving GlobalV::NQX and GlobalV::DQ
    // 4*pi/sqrt(omega) * Jl(qr) in which q goes in linspace(0, GlobalV::NQX, GlobalV::DQ)
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    tab_host = tab; // this is what cpu knows
    if (device_id == base_device::GpuDevice) // also let gpu know
    {
        // send the vq table "h2d"
        syncmem_var_h2d_op()(device_to, device_from, tab_device, tab, size);
        tab_host = tab_device;
    }
    // ----------------------------------------------------------------------------------->8
}

template <typename FPTYPE, typename Device>
void tabulate_gk(const int ik, 
                 const ModulePW::PW_Basis_K* pw_basis,
                 const double& tpiba,
                 FPTYPE* out,
                 std::vector<FPTYPE>& out_host,
                 FPTYPE* out_device,
                 Device* device_to,  
                 Device* device_from,
                 const base_device::AbacusDevice_t device_id)
{
    // ----------------------------------------------------------------------------------->8
    // the ik will always refreshed. Once ik changes, should re-calculate the Ylm(q).
    // and actually the derivative of Ylm(q) is also needed.
    // There should a function individually (relatively) calculate the ylm and dylm instead
    // of calculating them inside the cal_becp and cal_dbecp_s.

    // Should refresh the g_plus_k first, then calculate the ylm and dylm.
    // Another important thing is the function cal_gk. It tabulates the q = G+k, 
    // however, the g_plus_k it is not simply the G+k, but also with their norm and 
    // reciprocal of the norm, size: npw * 5. First memory block is npw * 3, the second and 
    // the third memory block are 1 * npw.
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;

    int npw = pw_basis->npwk[ik];
    out_host.resize(npw * 5);
    ModuleBase::Vector3<FPTYPE> q;
    for (int ig = 0; ig < npw; ++ig)
    {
        // written in memory block from 0 to 3*npw. This is like a matrix with npw rows and 3 columns
        q = pw_basis->getgpluskcar(ik, ig);
        out_host[ig * 3]     = q.x;
        out_host[ig * 3 + 1] = q.y;
        out_host[ig * 3 + 2] = q.z;
        // the following written in memory block from 3*npw to 5*npw, the excess 2*npw is for norm and 1/norm
        // for memory consecutive consideration, there are blocks storing the norm and 1/norm.
        FPTYPE norm = sqrt(q.norm2());
        out_host[3 * npw + ig] = norm * tpiba; // one line with length npw, storing the norm
        out_host[4 * npw + ig] = norm < 1e-8 ? 0.0 : 1.0 / norm * tpiba; // one line with length npw, storing 1/norm
    }
    
    out = out_host.data(); // this is what cpu knows
    if (device_id == base_device::GpuDevice) // also let gpu know
    {
        syncmem_var_h2d_op()(device_to, device_from, out_device, out_host.data(), out_host.size());
        out = out_device;
    }
}

template <typename FPTYPE>
void dylmr2(const int nylm,
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
void tabulate_ylm(FPTYPE* q, 
                  const int nq, 
                  const int lmax,
                  FPTYPE* out_device,
                  FPTYPE* out_grad_device,
                  Device* device_to,  
                  Device* device_from,
                  const base_device::AbacusDevice_t device_id)
{
    using syncmem_var_h2d_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    const int ntot_ylm = (lmax + 1) * (lmax + 1);
    if (device_id == base_device::GpuDevice)
    {
        // allocate
        std::vector<FPTYPE> out_host(ntot_ylm * nq);
        // calculate
        ModuleBase::YlmReal::Ylm_Real(device_from, ntot_ylm, nq, q, out_host.data());
        // send from cpu to gpu
        syncmem_var_h2d_op()(device_to, device_from, out_device, out_host.data(), out_host.size());
    }
    else
    {
        // calculate. Why not implement this logic branch inside some function???
        ModuleBase::YlmReal::Ylm_Real(device_from, ntot_ylm, nq, q, out_device);
    }
    // the function cal_ylm proceeds like, calculate on cpu, if gpu is available, copy to gpu.
    if (out_grad_device == nullptr) {return;}
    if (device_id == base_device::GpuDevice)
    {
        // allocate
        std::vector<FPTYPE> out_grad_host(3 * ntot_ylm * nq);
        // calculate
        for (int ipol = 0; ipol < 3; ipol++)
        {
            dylmr2(ntot_ylm, nq, q, &out_grad_host[ipol * ntot_ylm * nq], ipol);
        }
        // send from cpu to gpu
        syncmem_var_h2d_op()(device_to, device_from, out_grad_device, out_grad_host.data(), out_grad_host.size());
    }
    else
    {
        for (int ipol = 0; ipol < 3; ipol++)
        {
            dylmr2(ntot_ylm, nq, q, &out_grad_device[ipol * ntot_ylm * nq], ipol);
        }
    }
}


template <typename FPTYPE, typename Device>
void tabulate_sk(const int ik,
                 const int nat,
                 const int nq,
                 const Structure_Factor* sf,
                 const ModulePW::PW_Basis_K* pw_basis,
                 std::complex<FPTYPE>* out_device,
                 std::complex<FPTYPE>* out_host,
                 Device* device_to)
{
    // calculate sk
    // the "h" and "d" appears at the first place is either "host" or "device"
    // the "d" and "s" appears at the second place is either "double (64-bit)" or "single (32-bit)"
    // therefore the following hd_sk is the double precision, sk on the host device, this is done
    // concurrently by cpu and gpu (if there is).
    using resmem_complex_op = base_device::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    resmem_complex_op()(device_to, out_host, nat * nq);
    sf->get_sk(device_to, ik, pw_basis, out_host);
    out_device = out_host; // copy the starting memory address of hd_sk to d_sk
    // ----------------------------------------------------------------------------------->8
}

void cal_dvkb_index(const int nproj_it,
                    const int lmax,
                    const int* itiproj2l,
                    const int nproj_it_max,
                    const int npw,
                    const int it,
                    const int ipol,
                    const int jpol,
                    int* out)
{
    int iproj = 0;
    const int total_lm = (lmax + 1) * (lmax + 1);
    for (int iproj_it = 0; iproj_it < nproj_it; iproj_it++)
    {
        int l = itiproj2l[it * nproj_it_max + iproj];
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            out[iproj * 4] = lm;
            out[iproj * 4 + 1] = iproj_it;
            out[iproj * 4 + 2] = ipol * total_lm + lm;
            out[iproj * 4 + 3] = jpol * total_lm + lm;
            iproj++;
        }
    }
}

void tabulate_pref(const int it,
                   const int nproj_it,
                   const int* itich2l,
                   const int nch_it_max,
                   std::vector<std::complex<double>>& pref)
{
    int nch = 0;
    for (int iproj = 0; iproj < nproj_it; iproj++)
    {
        nch += 2 * itich2l[it * nch_it_max + iproj] + 1;
    }
    pref.resize(nch);
    for (int ich = 0; ich < nch; ich++)
    {
        pref[ich] = std::pow(std::complex<double>(0.0, -1.0), itich2l[it * nch_it_max + ich]);
        // it is actually nh2l, which means to get the angular momentum...
    }
}

template <typename FPTYPE, typename Device>
void cal_vkb(const int ntype,
             const std::vector<int>& natom,
             const int lmax,
             const int* nproj,
             const double* itich2l,
             const FPTYPE* interp_tab,
             const FPTYPE* q,
             const int nq,
             const double& dx,                              // interplation step
             const int nx,                                  // number of interpolation points
             const FPTYPE* ylm,
             const std::complex<FPTYPE>* sk,
             FPTYPE* vq_buf,
             std::complex<FPTYPE>* pref_buf,
             std::vector<int>& vkb_map,
             int* vkb_map_device,
             std::complex<FPTYPE>* out,
             Device* device_to,
             Device* device_from,
             const base_device::AbacusDevice_t device_id)
{
    using syncmem_complex_h2d_op
        = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;
    using syncmem_int_h2d_op 
        = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;

    int nproj_it_max = 0;
    for (int it = 0; it < ntype; it++)
    {
        nproj_it_max = std::max(nproj_it_max, nproj[it]);
    }

    int nch_max = 0;
    for (int it = 0; it < ntype; it++)
    {
        int nch = 0;
        for (int iproj = 0; iproj < nproj[it]; iproj++)
        {
            nch += 2 * itich2l[it * nproj_it_max + iproj] + 1;
        }
        nch_max = std::max(nch_max, nch);
    }

    // no one will be happy with a index specified as double...
    int* itich2l_ = new int[nch_max * ntype];
    for (int i = 0; i < nch_max * ntype; i++)
    {
        itich2l_[i] = itich2l[i]; // IMPLICIT DATATYPE CONVERSION
    }

    for (int it = 0; it < ntype; it++) // loop all elements
    {
        // simply interpolate the vq table
        hamilt::cal_vq_op<FPTYPE, Device>()(device_to,
                                            interp_tab,    // gpu knows it!
                                            it,
                                            q,             // gpu knows it!
                                            nq,
                                            nproj_it_max,  // interp_tab dim2: maximal number of radials among all atom types
                                            nx,            // interp_tab dim3: number of interpolation points
                                            dx,            // interpolation step
                                            nproj[it],
                                            vq_buf);
        // vq_buf is only for one atom type, has dimension nbeta * nq
        // this design saves memory but binds the calculation of vq and vkb together...

        // prepare（-i）^l, size: nh (total number of m-chennels of beta)
        std::vector<std::complex<double>> pref;
        tabulate_pref(it, nproj[it], itich2l_, nch_max, pref);

        const int nch = pref.size();
        // indexing...
        vkb_map.resize(nch * 4);
        cal_dvkb_index(nproj[it],     // nproj
                       lmax,
                       itich2l_,      // lproj
                       nch_max,       // maximal total number of CHANNELS of radials among all atom types
                       nq,            // nq
                       it,                                 
                       0,             // ipol (index of component, x, y or z)
                       0,             // jpol (index of component, x, y or z)
                       vkb_map.data());

        // device_pref_in (pref_buf) and device_dvkb_indexes (vkb_map_device) are those
        // really feed in the functor cal_vkb_op, now give them the memory address.
        if (device_id == base_device::GpuDevice)
        {
            syncmem_complex_h2d_op()(device_to, device_from, pref_buf, pref.data(), nch);
            syncmem_int_h2d_op()(device_to, device_from, vkb_map_device, vkb_map.data(), nch * 4);
        }
        else
        {
            pref_buf = pref.data();
            vkb_map_device = vkb_map.data();
        }
        // calculate vkb = vq * sk * pref * ylm
        for (int ia = 0; ia < natom[it]; ia++)
        {
            hamilt::cal_vkb_op<FPTYPE, Device>()(device_to, nch, nq, vkb_map_device, vq_buf, ylm, sk, pref_buf, out);
            // move the pointer to the next atom
            out += nch * nq;
            sk += nq;
        }
    }
    delete[] itich2l_;
}

template<typename FPTYPE, typename Device>
void cal_dvkb_stress(const int ntype,
                     const std::vector<int>& natom,
                     const int lmax,
                     const int* nproj,
                     const double* itich2l,
                     const FPTYPE* interp_tab,
                     const FPTYPE* gk,
                     const int ngk,
                     const double& interp_dx,                              // interplation step
                     const int interp_nx,                                  // number of interpolation points
                     const FPTYPE* ylm,
                     const FPTYPE* ylm_deri,
                     const std::complex<FPTYPE>* sk,
                     FPTYPE* vq_buf,
                     FPTYPE* vq_deri_buf,
                     std::complex<FPTYPE>* pref_buf,
                     std::vector<int>& vkb_map,
                     int* vkb_map_device,
                     const int ipol,
                     const int jpol,
                     std::complex<FPTYPE>* out,
                     Device* device_to,
                     Device* device_from,
                     const base_device::AbacusDevice_t device_id)
{
    using syncmem_complex_h2d_op
        = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;
    using syncmem_int_h2d_op
        = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;

    int nproj_it_max = 0;
    for (int it = 0; it < ntype; it++)
    {
        nproj_it_max = std::max(nproj_it_max, nproj[it]);
    }

    int nch_max = 0;
    for (int it = 0; it < ntype; it++)
    {
        int nch = 0;
        for (int iproj = 0; iproj < nproj[it]; iproj++)
        {
            nch += 2 * itich2l[it * nproj_it_max + iproj] + 1;
        }
        nch_max = std::max(nch_max, nch);
    }

    // no one will be happy with a index specified as double...
    int* itich2l_ = new int[nch_max * ntype];
    for (int i = 0; i < nch_max * ntype; i++)
    {
        itich2l_[i] = itich2l[i]; // IMPLICIT DATATYPE CONVERSION
    }

    for (int it = 0; it < ntype; it++) // loop all elements
    {
        // simply interpolation from tab to vkb, this has been appeared in function cal_becp
        // but is calculated again...
        hamilt::cal_vq_op<FPTYPE, Device>()(device_to,
                                            interp_tab, // in, the table with linspace(0, GlobalV::NQX, GlobalV::DQ)
                                            it,
                                            gk,
                                            ngk,
                                            nproj_it_max,
                                            interp_nx,
                                            interp_dx,
                                            nproj[it],
                                            vq_buf); // out
        // simply an interpolation but the interpolation function is... modified...
        // but this quantity, the dvkb is only used when calculating the dbecp_s
        hamilt::cal_vq_deri_op<FPTYPE, Device>()(device_to,
                                                 interp_tab, // in, the table with linspace(0, GlobalV::NQX, GlobalV::DQ)
                                                 it,
                                                 gk,
                                                 ngk,
                                                 nproj_it_max,
                                                 interp_nx,
                                                 interp_dx,
                                                 nproj[it],
                                                 vq_deri_buf); // out

        // prepare（-i）^l, size: nh (total number of m-chennels of beta)
        std::vector<std::complex<double>> pref;
        tabulate_pref(it, nproj[it], itich2l_, nch_max, pref);

        const int nch = pref.size();
        // indexing...
        vkb_map.resize(nch * 4);
        cal_dvkb_index(nproj[it],
                       lmax,
                       itich2l_,
                       nch_max,
                       ngk,
                       it,
                       ipol,
                       jpol,
                       vkb_map.data());

        // if it is gpu, let cpu tell gpu
        if (device_id == base_device::GpuDevice)
        {
            syncmem_complex_h2d_op()(device_to, device_from, pref_buf, pref.data(), nch);
            syncmem_int_h2d_op()(device_to, device_from, vkb_map_device, vkb_map.data(), nch * 4);
        }
        else // if it is cpu, cpu knows it
        {
            vkb_map_device = vkb_map.data();
            pref_buf = pref.data();
        }
        for (int ia = 0; ia < natom[it]; ia++)
        {
            hamilt::cal_vkb_deri_op<FPTYPE, Device>()(device_to,     // supports run both on cpu and gpu
                                                      nch,
                                                      ngk,
                                                      ipol,
                                                      jpol,
                                                      vkb_map_device, // gpu knows it!
                                                      vq_buf,         // what we just calculated
                                                      vq_deri_buf,    // what we just calculated
                                                      ylm,            // gpu knows it after calling cal_becp...
                                                      ylm_deri,       // what we just calculated
                                                      sk,             // gpu knows it after calling cal_becp...
                                                      pref_buf,       // gpu knows it!
                                                      gk,             // gpu knows it!
                                                      out);           // gpu knows it!

            // move to the next atom
            sk += ngk; 
            out += nch * ngk;
        }
    }
    delete[] itich2l_;
}

template <typename FPTYPE, typename Device>
FS_Nonlocal_tools<FPTYPE, Device>::FS_Nonlocal_tools(const pseudopot_cell_vnl* nlpp_in,
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

    this->tabtpr = &(nlpp_->tab);
    this->nhtol = &(nlpp_->nhtol);
    this->lprojmax = nlpp_->lmaxkb;
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
    this->h_atom_nh.resize(this->ntype);
    this->h_atom_na.resize(this->ntype);
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

    // allocate the memory for vkb and vkb_deri.
    if (this->device == base_device::GpuDevice)
    {
        resmem_int_op()(this->ctx, this->d_dvkb_indexes, max_nh * 4);
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
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_nh, h_atom_nh.data(), this->ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_na, h_atom_na.data(), this->ntype);

        resmem_var_op()(this->ctx, d_g_plus_k, max_npw * 5);
        resmem_var_op()(this->ctx, d_pref, max_nh);
        resmem_var_op()(this->ctx, d_vq_tab, this->nlpp_->tab.getSize());
        resmem_complex_op()(this->ctx, d_pref_in, max_nh);

        this->ppcell_vkb = this->nlpp_->template get_vkb_data<FPTYPE>();
    }
    else
    {
        this->d_wg = wg.c;
        this->d_ekb = ekb.c;
        this->atom_nh = h_atom_nh.data();
        this->atom_na = h_atom_na.data();
        this->ppcell_vkb = this->nlpp_->vkb.c;
    }
}

template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::delete_memory()
{
    // delete memory

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
        delmem_complex_op()(this->ctx, this->d_pref_in);
        delmem_int_op()(this->ctx, d_dvkb_indexes);
    }

    if (becp != nullptr)
    {
        delmem_complex_op()(this->ctx, becp);
        delmem_complex_op()(this->ctx, hd_sk);
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
// starts from vkb (nkb, ng) table
// it should be merely the multiplication of matrix (vkb, ng) * (ng, nbands) -> (vkb, nbands)
// should be irrelevant with what the matrix is.
// the vkb index generation should be maintained elsewhere.
// vkb already has atomic position information, calculated from the vq and sk
// . the multiplication with sk should be within specific operator
// because the atom selection task is operator-specific.
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_becp(int ik, int npm)
{
    ModuleBase::TITLE("FS_Nonlocal_tools", "cal_becp");
    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_becp");

    FPTYPE* vq_tb = nullptr;
    retrieve_interptab(this->tabtpr->ptr,           // [in] the table (ntype, nprojmax, GlobalV::NQX) 
                       this->tabtpr->getSize(),     // [in] ntype*nprojmax*GlobalV::NQX
                       vq_tb,                       // [out]
                       this->d_vq_tab,              // [out]
                       this->ctx, 
                       this->cpu_ctx, 
                       this->device);

    FPTYPE* gk = nullptr;
    tabulate_gk(ik,                     // [in]
                this->wfc_basis_,       // [in]
                this->ucell_->tpiba,    // [in]
                gk,                     // [out]
                this->g_plus_k,         // [out]
                this->d_g_plus_k,       // [out]
                this->ctx, 
                this->cpu_ctx, 
                this->device);

    tabulate_ylm(this->g_plus_k.data(),         // [in]
                 this->wfc_basis_->npwk[ik],    // [in]
                 this->lprojmax,                // [in] lmax over all radials
                 this->hd_ylm,                  // [out]
                 this->hd_ylm_deri,             // [out]
                 this->ctx, 
                 this->cpu_ctx, 
                 this->device);

    std::complex<FPTYPE>* d_sk = nullptr;
    tabulate_sk(ik,
                this->ucell_->nat, 
                this->wfc_basis_->npwk[ik], 
                this->sf_, 
                this->wfc_basis_,
                d_sk, 
                this->hd_sk, 
                this->ctx);
    
    const int npw = this->wfc_basis_->npwk[ik];

    // in the following the vkb, which is the fourier transform of beta function involving
    // radial parts the tab, and angular parts the ylm, is calculated.
    // for radial parts, should do interpolation according to gk to get the vq. New codes
    // uses cubic spline, while the old uses polynomial interpolation, not the best choice.

    std::vector<int> nproj(this->ucell_->ntype);
    for (int it = 0; it < this->ucell_->ntype; it++)
    {
        nproj[it] = this->ucell_->atoms[it].ncpp.nbeta;
    }

    std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
    cal_vkb(this->ucell_->ntype,        // ntype, from ucell
            this->h_atom_na,            // # of atom of each type, from ucell
            this->lprojmax,             // lmax over all radials
            nproj.data(),               // # of radials of each atom type
            this->nhtol->c,             // itiproj2l mapping, size (ntype, nprojmax)
            vq_tb,                      // tabulated vq
            gk,                         // g+k x/y/z (3*npw) with norm (npw) and 1/norm (npw)
            npw,                        // # of plane waves, one dimension of gk
            GlobalV::DQ,                // step of interpolate table
            GlobalV::NQX,               // # of interpolation points
            this->hd_ylm,               // ylm, on "host"
            d_sk,                       // sk, on "device"
            this->hd_vq,                // cached vq, on "host"
            this->d_pref_in,            // prefactor (i)^l, saved on "device"
            this->dvkb_indexes,         // vkb needed indexes
            this->d_dvkb_indexes,       // vkb needed indexes, saved on "device"
            vkb_ptr,                    // output, the vkb
            this->ctx,
            this->cpu_ctx,
            this->device);

    // ----------------------------------------------------------------------------------->8
    // what cal_becp function should do is the following, what it should not do is above.
    const char transa = 'C';
    const char transb = 'N';
    // allocate memory for becp on gpu
    const int npol = this->ucell_->get_npol();
    const int size_becp = this->nbands * npol * this->nkb;
    if (this->becp == nullptr)
    {
        resmem_complex_op()(this->ctx, becp, size_becp);
    }
    // the starting memory address of psi of present k-point
    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    int npm_npol = npm * npol;
    gemm_op()(this->ctx,
              transa,  
              transb,
              nkb,
              npm_npol,
              npw,
              &ModuleBase::ONE,
              vkb_ptr,
              npw,
              ppsi,
              this->max_npw,
              &ModuleBase::ZERO,
              becp,
              nkb);
    // becp calculate is over , now we should broadcast this data.
    const int size_becp_act = npm * npol * this->nkb;
    if (this->device == base_device::GpuDevice)
    {
        std::complex<FPTYPE>* h_becp = nullptr;
        // allocate memory for h_becp on cpu
        resmem_complex_h_op()(this->cpu_ctx, h_becp, size_becp_act);
        // copy data from becp to h_becp on cpu
        syncmem_complex_d2h_op()(this->cpu_ctx, this->ctx, h_becp, becp, size_becp_act);
        // MPI reduce, get the merged becp data on cpu
        Parallel_Reduce::reduce_pool(h_becp, size_becp_act);
        // send merged becp data to gpu to the array becp
        syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, becp, h_becp, size_becp_act);
        // release the memory of h_becp on cpu (a temporary memory)
        delmem_complex_h_op()(this->cpu_ctx, h_becp);
    }
    else
    {
        Parallel_Reduce::reduce_pool(becp, size_becp_act);
    }

    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_becp");
}

// cal_dbecp_s
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_dbecp_s(int ik, int npm, int ipol, int jpol, FPTYPE* stress)
{
    ModuleBase::TITLE("FS_Nonlocal_tools", "cal_dbecp_s");
    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_dbecp_s");

    const int npw = this->wfc_basis_->npwk[ik];
    // calculate dvkb, the radial part (seems the angular part is calculated on cpu?)
    std::complex<FPTYPE>* vkb_deri_buf = this->ppcell_vkb; // reuse the memory of vkb
    // on different device, use different variable
    FPTYPE *vq_tb = (this->device == base_device::GpuDevice)? this->d_vq_tab: this->tabtpr->ptr; 
    FPTYPE *gk = (this->device == base_device::GpuDevice)? this->d_g_plus_k: g_plus_k.data();
    std::complex<FPTYPE>* d_sk = this->hd_sk;

    std::vector<int> nproj(this->ucell_->ntype);
    for (int it = 0; it < this->ucell_->ntype; it++)
    {
        nproj[it] = this->ucell_->atoms[it].ncpp.nbeta;
    }
    // because the vq and vq_deri are calculated on-the-fly to save memory, the following
    // function param list is a bit long.
    cal_dvkb_stress(this->ucell_->ntype,    // ntype
                    this->h_atom_na,        // # of atom of each type
                    this->lprojmax,         // lmax over all radials
                    nproj.data(),           // # of radials of each atom type
                    this->nlpp_->nhtol.c,   // itiproj2l mapping, size (ntype, nproj_max)
                    vq_tb,                  // tabulated vq
                    gk,                     // g+k x/y/z (3*npw) with norm (npw) and 1/norm (npw)
                    npw,                    // # of plane waves, one dimension of gk
                    GlobalV::DQ,            // step of interpolate table
                    GlobalV::NQX,           // # of interpolation points
                    this->hd_ylm,           // ylm, on "host"
                    this->hd_ylm_deri,      // dylm/dq, on "host"
                    d_sk,                   // structure factor, on "device"
                    this->hd_vq,            // cached vq, on "host"
                    this->hd_vq_deri,       // cached vq_deri, on "host"
                    this->d_pref_in,        // prefactor (i)^l, saved on "device"
                    this->dvkb_indexes,     // vkb needed indexes
                    this->d_dvkb_indexes,   // vkb needed indexes, saved on "device"
                    ipol,                   // the first index of direction of strain/stress
                    jpol,                   // the second index of direction of strain/stress
                    vkb_deri_buf,           // output, the vkb_deri
                    this->ctx,
                    this->cpu_ctx,
                    this->device);

    // calculate dbecp_s
    const int npol = this->ucell_->get_npol();
    const int size_becp = this->nbands * npol * this->nkb;
    const int npm_npol = npm * npol;
    if (this->dbecp == nullptr) // allocate memory for dbecp
    {
        resmem_complex_op()(this->ctx, dbecp, size_becp);
    }
    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    const char transa = 'C';
    const char transb = 'N';

    gemm_op()(this->ctx,
              transa,
              transb,  
              nkb,
              npm_npol,
              npw,
              &ModuleBase::ONE,
              ppcell_vkb,
              npw,
              ppsi,
              this->max_npw,
              &ModuleBase::ZERO,
              dbecp,
              nkb);

    // ----------------------------------------------------------------------------------->8
    // the following calculates the stress, but why it is here?
    // the correct place for calling these functions below are in stress_func_nl.cpp
    // calculate stress for target (ipol, jpol)
    if(this->ucell_->get_npol() == 1)
    {
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
    else
    {
        cal_stress_nl_op()(this->ctx,
                           ipol,
                           jpol,
                           nkb,
                           npm,
                           this->ntype,
                           this->nbands,
                           ik,
                           this->nlpp_->deeq_nc.getBound2(),
                           this->nlpp_->deeq_nc.getBound3(),
                           this->nlpp_->deeq_nc.getBound4(),
                           atom_nh,
                           atom_na,
                           d_wg,
                           d_ekb,
                           qq_nt,
                           this->nlpp_->template get_deeq_nc_data<FPTYPE>(),
                           becp,
                           dbecp,
                           stress);
    }
    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_dbecp_s");
}

// cal_dbecp_f
// starts from vkb (nkb, ng) table
// it should be again merely the multiplication of matrix (vkb, ng) * (ng, nbands) -> (vkb, nbands)
// the vkb is backed-up, and the memory space is reused for calculate ONE COMPONENT of dbecp
// . the direction of force is indexed by ipol (for stress, there are two, ipol and jpol).
// the dbecp_f is simply the becp multiplied with -i(G+k)_i
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::cal_dbecp_f(int ik, int npm, int ipol)
{
    ModuleBase::TITLE("FS_Nonlocal_tools", "cal_dbecp_f");
    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_dbecp_f");

    const int npw = this->wfc_basis_->npwk[ik];

    // STAGE1: calculate dvkb_f
    // calculate gcarx, gcary/gcarx and gcarz/gcary, overwrite gcar
    if (this->pre_ik_f == -1) // if it is the very first run, we allocate
    {
        resmem_var_op()(this->ctx, gcar, 3 * this->wfc_basis_->npwk_max);
        resmem_int_op()(this->ctx, gcar_zero_indexes, 3 * this->wfc_basis_->npwk_max);
    }
    // first refresh the value of gcar_zero_indexes, gcar_zero_counts
    if (this->pre_ik_f != ik)
    { // the following lines will cause UNDEFINED BEHAVIOR because memory layout of vector3 instance
      // is assumed to be always contiguous but it is not guaranteed.
        this->transfer_gcar(npw,
                            this->wfc_basis_->npwk_max,
                            &(this->wfc_basis_->gcar[ik * this->wfc_basis_->npwk_max].x));
    }

    // backup vkb values to vkb_save
    this->save_vkb(npw, ipol);
    // for x, the coef is -i, for y and z it is 1
    const std::complex<double> coeff = ipol == 0 ? ModuleBase::NEG_IMAG_UNIT : ModuleBase::ONE;

    const std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
    std::complex<FPTYPE>* vkb_deri_ptr = this->ppcell_vkb;
    // calculate the vkb_deri for ipol with the memory of ppcell_vkb
    cal_vkb1_nl_op<FPTYPE, Device>()(this->ctx, nkb, npw, npw, npw, ipol, coeff, vkb_ptr, gcar, vkb_deri_ptr);

    // ------------------------------------------------------------------------------->8

    // STAGE2: calculate dbecp_f
    // NPOL
    // either 1 or 2, for NSPIN 1, 2 or 4 calculation
    // once NSPIN 4, there are doubled number of pw in each "row" of psi
    // on the other hand, for NSPIN 4 calculation, the number of bands is also doubled
    const int npol = this->ucell_->get_npol();
    const int npm_npol = npm * npol;
    const int size_becp = this->nbands * npol * this->nkb;
    if (this->dbecp == nullptr) // if it is the very first run, we allocate
    { // why not judging whether dbecp == nullptr inside resmem_complex_op?
        resmem_complex_op()(this->ctx, dbecp, 3 * size_becp);
    }
    // do gemm to get dbecp and revert the ppcell_vkb for next ipol
    const std::complex<FPTYPE>* ppsi = &(this->psi_[0](ik, 0, 0));
    // move the pointer to corresponding read&write position, according to ipol
    std::complex<FPTYPE>* dbecp_ptr = this->dbecp + ipol * size_becp; // [out]
    const char transa = 'C';
    const char transb = 'N';
    gemm_op()(this->ctx,
              transa,
              transb,
              this->nkb,
              npm_npol,
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
    ModuleBase::timer::tick("FS_Nonlocal_tools", "cal_dbecp_f");
}

// save_vkb
template <typename FPTYPE, typename Device>
void FS_Nonlocal_tools<FPTYPE, Device>::save_vkb(int npw, int ipol)
{
    if (this->device == base_device::CpuDevice)
    {
        const int gcar_zero_count = this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max];
        const int* gcar_zero_ptrs = &this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max + 1];
        const std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
        std::complex<FPTYPE>* vkb_save_ptr = this->vkb_save;
        // find the zero indexes to save the vkb values to vkb_save
        for (int ikb = 0; ikb < this->nkb; ++ikb)
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
        saveVkbValues<FPTYPE>(this->gcar_zero_indexes,
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
    const std::complex<FPTYPE> coeff = ipol == 0 ? ModuleBase::NEG_IMAG_UNIT : ModuleBase::ONE;
    if (this->device == base_device::CpuDevice)
    {
        const int gcar_zero_count = this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max];
        const int* gcar_zero_ptrs = &this->gcar_zero_indexes[ipol * this->wfc_basis_->npwk_max + 1];
        std::complex<FPTYPE>* vkb_ptr = this->ppcell_vkb;
        const std::complex<FPTYPE>* vkb_save_ptr = this->vkb_save;
        // find the zero indexes to save the vkb values to vkb_save
        for (int ikb = 0; ikb < this->nkb; ++ikb)
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
        revertVkbValues<FPTYPE>(this->gcar_zero_indexes,
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
    std::vector<FPTYPE> gcar_tmp(3 * npw_max); // [out], will overwritten this->gcar
    gcar_tmp.assign(gcar_in, gcar_in + 3 * npw_max); // UNDEFINED BEHAVIOR!!! nobody always knows the memory layout of vector3
    std::vector<int> gcar_zero_indexes_tmp(3 * npw_max); // a "checklist"

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
                ++gcar_zero_counts[i]; // num of zeros on each direction
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
    if(this->ucell_->get_npol() == 1)
    {
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
    else
    {
        cal_force_nl_op<FPTYPE, Device>()(this->ctx,
                                          npm,
                                          this->nbands,
                                          this->ntype,
                                          this->nlpp_->deeq_nc.getBound2(),
                                          this->nlpp_->deeq_nc.getBound3(),
                                          this->nlpp_->deeq_nc.getBound4(),
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
                                          this->nlpp_->template get_deeq_nc_data<FPTYPE>(),
                                          becp,
                                          dbecp,
                                          force);
    }
}

// template instantiation
template class FS_Nonlocal_tools<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class FS_Nonlocal_tools<double, base_device::DEVICE_GPU>;
#endif




} // namespace hamilt
