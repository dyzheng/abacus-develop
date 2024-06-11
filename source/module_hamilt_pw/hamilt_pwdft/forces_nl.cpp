#include "forces.h"
#include "stress_func.h"
#include "module_base/math_ylmreal.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/output_log.h"
// new
#include "module_base/complexmatrix.h"
#include "module_base/libm/libm.h"
#include "module_base/math_integral.h"
#include "module_base/mathzone.h"
#include "module_base/timer.h"
#include "module_base/tool_threading.h"
#include "module_elecstate/potentials/efield.h"
#include "module_elecstate/potentials/gatefield.h"
#include "module_hamilt_general/module_ewald/H_Ewald_pw.h"
#include "module_hamilt_general/module_surchem/surchem.h"
#include "module_hamilt_general/module_vdw/vdw.h"
#include "module_psi/kernels/device.h"
#include "module_base/memory.h"
#include "module_base/math_polyint.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

#include <chrono>

// cal_vq
template <typename FPTYPE>
std::vector<FPTYPE> cal_vq(int it, const FPTYPE* gk, int npw)
{
    // calculate beta in G-space using an interpolation table
    const int nbeta = GlobalC::ucell.atoms[it].ncpp.nbeta;

    std::vector<FPTYPE> vq(nbeta * npw);
    ModuleBase::Memory::record("stress_nl::vq", nbeta * npw * sizeof(FPTYPE));

    for (int nb = 0; nb < nbeta; nb++)
    {
        FPTYPE* vq_ptr = &vq[nb * npw];
        const FPTYPE* gnorm = &gk[3 * npw];
        for (int ig = 0; ig < npw; ig++)
        {
            vq_ptr[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(GlobalC::ppcell.tab,
                                                                    it,
                                                                    nb,
                                                                    GlobalV::NQX,
                                                                    GlobalV::DQ,
                                                                    gnorm[ig]);
        }
    }
    return vq;
}




// cal_ylm
template <typename FPTYPE, typename Device>
void Forces<FPTYPE, Device>::cal_ylm(int lmax, int npw, const FPTYPE* gk_in, FPTYPE* ylm)
{
    int x1 = (lmax + 1) * (lmax + 1);
    ModuleBase::Memory::record("stress_nl::ylm", x1 * npw * sizeof(FPTYPE));

    if (this->device == base_device::GpuDevice)
    {
        std::vector<FPTYPE> ylm_cpu(x1 * npw);
        ModuleBase::YlmReal::Ylm_Real(cpu_ctx, x1, npw, gk_in, ylm_cpu.data());
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, ylm, ylm_cpu.data(), ylm_cpu.size());
    } else {
        ModuleBase::YlmReal::Ylm_Real(cpu_ctx, x1, npw, gk_in, ylm);
    }
    
    return ;
}

// cal_pref
template <typename FPTYPE>
std::vector<std::complex<FPTYPE>> cal_pref(int it)
{
    const int nh = GlobalC::ucell.atoms[it].ncpp.nh;
    std::vector<std::complex<FPTYPE>> pref(nh);
    for(int ih=0;ih<nh;ih++)
    {
        pref[ih] = std::pow(std::complex<FPTYPE>(0.0, -1.0), GlobalC::ppcell.nhtol(it, ih));
    }
    return pref;
}



// cal_vkb
// cpu version first, gpu version later
template <typename FPTYPE>
void cal_vkb(
    int it,
    int ia,
    int npw,
    const FPTYPE* vq_in,
    const FPTYPE* ylm_in,
    const std::complex<FPTYPE>* sk_in,
    const std::complex<FPTYPE>* pref_in,
    std::complex<FPTYPE>* vkb_out)
{
    int ih=0;
    // loop over all beta functions
    for(int nb=0;nb<GlobalC::ucell.atoms[it].ncpp.nbeta;nb++)
    {
        int l = GlobalC::ppcell.nhtol(it, ih);
        // loop over all m angular momentum
        for(int m=0;m<2*l+1;m++)
        {
            int lm = l*l + m;
            std::complex<FPTYPE>* vkb_ptr = &vkb_out[ih * npw];
            const FPTYPE* ylm_ptr = &ylm_in[lm * npw];
            const FPTYPE* vq_ptr = &vq_in[nb * npw];
            // loop over all G-vectors
            for(int ig=0;ig<npw;ig++)
            {
                vkb_ptr[ig] = ylm_ptr[ig] * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
            }
            ih++;
        }
    }
}

// cal_vkb
// cpu version first, gpu version later
template <typename FPTYPE>
void cal_vkb_deri(
    int it,
    int ia,
    int npw,
    int ipol,
    int jpol,
    const FPTYPE* vq_in, const FPTYPE* vq_deri_in,
    const FPTYPE* ylm_in, const FPTYPE* ylm_deri_in,
    const std::complex<FPTYPE>* sk_in,
    const std::complex<FPTYPE>* pref_in,
    const FPTYPE* gk_in,
    std::complex<FPTYPE>* vkb_out)
{
    int x1 = (GlobalC::ppcell.lmaxkb + 1) * (GlobalC::ppcell.lmaxkb + 1);
    int ih=0;
    // loop over all beta functions
    for(int nb=0;nb<GlobalC::ucell.atoms[it].ncpp.nbeta;nb++)
    {
        int l = GlobalC::ppcell.nhtol(it, ih);
        // loop over all m angular momentum
        for(int m=0;m<2*l+1;m++)
        {
            int lm = l*l + m;
            std::complex<FPTYPE>* vkb_ptr = &vkb_out[ih * npw];
            const FPTYPE* ylm_ptr = &ylm_in[lm * npw];
            const FPTYPE* vq_ptr = &vq_in[nb * npw];
            // set vkb to zero
            for(int ig=0;ig<npw;ig++)
            {
                vkb_ptr[ig] = std::complex<FPTYPE>(0.0, 0.0);
            }
            // first term: ylm * vq * sk * pref
            // loop over all G-vectors
            if(ipol == jpol)
            {
                for(int ig=0;ig<npw;ig++)
                {
                    vkb_ptr[ig] -= ylm_ptr[ig] * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
                }
            }
            //second term: ylm_deri * vq_deri * sk * pref
            // loop over all G-vectors
            const FPTYPE* ylm_deri_ptr1 = &ylm_deri_in[(ipol * x1 + lm) * npw];
            const FPTYPE* ylm_deri_ptr2 = &ylm_deri_in[(jpol * x1 + lm) * npw];
            const FPTYPE* vq_deri_ptr = &vq_deri_in[nb * npw];
            const FPTYPE* gkn = &gk_in[4 * npw];
            for(int ig=0;ig<npw;ig++)
            {
                vkb_ptr[ig] -= (gk_in[ig*3+ipol] * ylm_deri_ptr2[ig] + gk_in[ig*3+jpol] * ylm_deri_ptr1[ig]) 
                                * vq_ptr[ig] * sk_in[ig] * pref_in[ih];
            }
            //third term: ylm * vq_deri * sk * pref
            // loop over all G-vectors
            for(int ig=0;ig<npw;ig++)
            {
                vkb_ptr[ig] -= 2.0 * ylm_ptr[ig] * vq_deri_ptr[ig] * sk_in[ig] * pref_in[ih]
                               * gk_in[ig*3+ipol] * gk_in[ig*3+jpol] * gkn[ig];
            }
            ih++;
        }
    }
}


// cal_gk
template <typename FPTYPE>
std::vector<FPTYPE> cal_gk(int ik, ModulePW::PW_Basis_K* wfc_basis)
{
    int npw = wfc_basis->npwk[ik];
    std::vector<FPTYPE> gk(npw * 5);
    ModuleBase::Memory::record("stress_nl::gk", 5 * npw * sizeof(FPTYPE));
    ModuleBase::Vector3<FPTYPE> tmp;
    for(int ig=0;ig<npw;++ig)
    {
        tmp = wfc_basis->getgpluskcar(ik, ig);
        gk[ig*3] = tmp.x;
        gk[ig*3+1] = tmp.y;
        gk[ig*3+2] = tmp.z;
        FPTYPE norm = sqrt(tmp.norm2());
        gk[3 * npw + ig] = norm * GlobalC::ucell.tpiba;
        gk[4 * npw + ig] = norm<1e-8?0.0:1.0/norm*GlobalC::ucell.tpiba;
    }
    return gk;
}

template <typename FPTYPE, typename Device>
void prepare_vkb_ptr(
    int nbeta, double* nhtol, int nhtol_nc, int npw, int it,
    std::complex<FPTYPE>*vkb_out, std::complex<FPTYPE>** vkb_ptrs,
    FPTYPE* ylm_in, FPTYPE** ylm_ptrs,
    FPTYPE* vq_in, FPTYPE** vq_ptrs
) {   
    // std::complex<FPTYPE>** vkb_ptrs[nh];
    // const FPTYPE** ylm_ptrs[nh];
    // const FPTYPE** vq_ptrs[nh];
    int ih=0;
    for(int nb=0;nb<nbeta;nb++)
    {
        int l = nhtol[it*nhtol_nc+ih];
        for(int m=0;m<2*l+1;m++)
        {
            int lm = l*l + m;
            vkb_ptrs[ih] = &vkb_out[ih * npw];
            ylm_ptrs[ih] = &ylm_in[lm * npw];
            vq_ptrs[ih] = &vq_in[nb * npw];
            ih++;
        }
    }
}


template <typename FPTYPE, typename Device>
void Forces<FPTYPE, Device>::cal_force_nl_new(ModuleBase::matrix& forcenl,
                                          const ModuleBase::matrix& wg,
                                          const ModuleBase::matrix& ekb,
                                          K_Vectors* p_kv,
                                          ModulePW::PW_Basis_K* wfc_basis,
                                          const psi::Psi<complex<FPTYPE>, Device>* psi_in)
{
    ModuleBase::TITLE("Forces", "cal_force_nl");
    ModuleBase::timer::tick("Forces", "cal_force_nl");

    const int nkb = GlobalC::ppcell.nkb;
    int wg_nc = wg.nc;
    if (nkb == 0 || psi_in == nullptr || wfc_basis == nullptr)
    {
        return; // mohan add 2010-07-25
    }
    // There is a contribution for jh<>ih in US case or multi projectors case
    // Actually, the judge of nondiagonal should be done on every atom type
    const bool nondiagonal = (GlobalV::use_uspp || GlobalC::ppcell.multi_proj) ? true : false;
    this->device = base_device::get_device_type<Device>(this->ctx);

    int max_nbeta=0, _lmax = GlobalC::ppcell.lmaxkb,max_nh=0;
    for(int it=0;it<GlobalC::ucell.ntype;it++)//loop all elements 
    {
        max_nh = std::max(GlobalC::ucell.atoms[it].ncpp.nh,max_nh);
        max_nbeta = std::max(GlobalC::ucell.atoms[it].ncpp.nbeta,max_nbeta);
    }

    // prepare the memory of the becp and dbecp:
    // becp: <Beta(nkb,npw)|psi(nbnd,npw)>
    // dbecp: <dBeta(nkb,npw)/dG|psi(nbnd,npw)>
    std::complex<FPTYPE> *dbecp = nullptr, *becp = nullptr,  *vkb1 = nullptr;
    resmem_complex_op()(this->ctx, becp, wg_nc * nkb, "Force::becp");
    resmem_complex_op()(this->ctx, dbecp, 3 * wg_nc * nkb, "Force::dbecp");
    resmem_complex_op()(this->ctx, vkb1, this->npwx * max_nh, "Force::vkb1");


    // prepare the memory of stress and init some variables:
    int *atom_nh = nullptr, *atom_na = nullptr, *h_atom_nh = new int[GlobalC::ucell.ntype],
        *h_atom_na = new int[GlobalC::ucell.ntype];
    for (int ii = 0; ii < GlobalC::ucell.ntype; ii++)
    {
        h_atom_nh[ii] = GlobalC::ucell.atoms[ii].ncpp.nh;
        h_atom_na[ii] = GlobalC::ucell.atoms[ii].na;
    }
    FPTYPE *force = nullptr,  *d_wg = nullptr, *d_ekb = nullptr, *gcar = nullptr,
           *deeq = GlobalC::ppcell.get_deeq_data<FPTYPE>(), *kvec_c = wfc_basis->get_kvec_c_data<FPTYPE>(),
           *qq_nt = GlobalC::ppcell.get_qq_nt_data<FPTYPE>();



    // allocate the memory for ops.
    FPTYPE *h_ylm = new FPTYPE[(_lmax+1)*(_lmax+1)*this->npwx], *hd_ylm = nullptr, // (lmax + 1) * (lmax + 1) * npw
            *hd_vq = nullptr, *hd_vq_deri = nullptr, //this->ucell->atoms[it].ncpp.nbeta * npw
            *h_g_plus_k = new FPTYPE[this->npwx * 5], *d_g_plus_k = nullptr, // npw * 5
            *h_pref = new FPTYPE[max_nh], *d_pref = nullptr, // this->ucell->atoms[it].ncpp.nh
            *d_gk = nullptr, *d_vq_tab = nullptr; // npw


    
    FPTYPE  **vq_ptrs= new FPTYPE*[max_nh], **d_vq_ptrs = nullptr,
            **ylm_ptrs= new FPTYPE*[max_nh], **d_ylm_ptrs = nullptr;
    std::complex<FPTYPE>** vkb_ptrs = new std::complex<FPTYPE>*[max_nh];
    std::complex<FPTYPE>** d_vkb_ptrs = nullptr;
    std::complex<FPTYPE>*d_sk = nullptr, *d_pref_in = nullptr ;
    std::complex<FPTYPE>* ppcell_vkb = nullptr;

    resmem_var_op()(this->ctx, hd_vq, max_nbeta*this->npwx);
    resmem_var_op()(this->ctx, hd_ylm, (_lmax+1)*(_lmax+1)*this->npwx);
    if (this->device == base_device::GpuDevice)
    {
        resmem_var_op()(this->ctx, d_wg, wg.nr * wg.nc);
        resmem_var_op()(this->ctx, d_ekb, ekb.nr * ekb.nc);
        resmem_var_op()(this->ctx, gcar, 3 * wfc_basis->nks * wfc_basis->npwk_max);
        resmem_var_op()(this->ctx, force, forcenl.nr * forcenl.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_wg, wg.c, wg.nr * wg.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_ekb, ekb.c, ekb.nr * ekb.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, force, forcenl.c, forcenl.nr * forcenl.nc);
        syncmem_var_h2d_op()(this->ctx,
                             this->cpu_ctx,
                             gcar,
                             &wfc_basis->gcar[0][0],
                             3 * wfc_basis->nks * wfc_basis->npwk_max);
        resmem_int_op()(this->ctx, atom_nh, GlobalC::ucell.ntype);
        resmem_int_op()(this->ctx, atom_na, GlobalC::ucell.ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_nh, h_atom_nh, GlobalC::ucell.ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_na, h_atom_na, GlobalC::ucell.ntype);
    
        resmem_var_op()(this->ctx, d_g_plus_k, this->npwx * 5);
        resmem_var_op()(this->ctx, d_pref, max_nh);
        resmem_var_op()(this->ctx, d_vq_tab, GlobalC::ppcell.tab.getSize());
        hamilt::pointer_array_malloc<Device>()((void **)&d_ylm_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_vq_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_vkb_ptrs, max_nh);

        resmem_complex_op()(this->ctx, d_sk,  this->npwx, "Stress::d_sk");
        resmem_complex_op()(this->ctx, d_pref_in,  max_nh, "Stress::pref_in");    

        ppcell_vkb= GlobalC::ppcell.get_vkb_data<FPTYPE>();
    }
    else
    {
        d_wg = wg.c;
        d_ekb = ekb.c;
        force = forcenl.c;
        gcar = &wfc_basis->gcar[0][0];
        atom_nh = h_atom_nh;
        atom_na = h_atom_na;
        ppcell_vkb = GlobalC::ppcell.vkb.c;
    }


    int nks = wfc_basis->nks;
    for(int ik=0;ik<nks;ik++)//loop k points
    {
        // only for uspp: move the spin index in deeq
        int current_spin = p_kv->isk[ik];
        const int nbasis = wfc_basis->npwk[ik];



        psi_in[0].fix_k(ik);
        std::complex<FPTYPE>* ppsi = &(psi_in[0](ik, 0, 0));
        char transa = 'C';
        char transb = 'N';

        ///
        /// only occupied band should be calculated.
        ///
        int nbands_occ = GlobalV::NBANDS;
        while (wg(ik, nbands_occ - 1) == 0.0)
        {
            nbands_occ--;
            if (nbands_occ == 0)
            {
                break;
            }
        }
        const int npm = GlobalV::NPOL * nbands_occ;
        const int npw = p_kv->ngk[ik];


        std::vector<FPTYPE> g_plus_k = cal_gk<FPTYPE>(ik, wfc_basis);
        std::complex<FPTYPE>* becp_ptr = becp;


        std::complex<FPTYPE>* dbecp_ptr[3];
        for(int i=0;i<3;i++)
        {
            dbecp_ptr[i] = &dbecp[i * wg_nc * nkb];
        }


        int lmax = GlobalC::ppcell.lmaxkb;
        //prepare ylm，size: (lmax+1)^2 * npwx
        // std::vector<double> ylm =  cal_ylm(lmax, npw, g_plus_k.data());
        // if (this->device == base_device::GpuDevice)
        // {
        //     syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_ylm, ylm.data(), ylm.size());
        // }
        cal_ylm(lmax, npw, g_plus_k.data(),hd_ylm);

        for(int it=0;it<GlobalC::ucell.ntype;it++)//loop all elements 
        {

            // prepare inputs for calculating vkb，vkb1，vkb2 
            // prepare vq and vq', size: nq * npwx 
            int lenth_vq = GlobalC::ucell.atoms[it].ncpp.nbeta*npw;
            std::vector<double> vq(lenth_vq);
            
            FPTYPE *gk = g_plus_k.data(), *vq_tb = GlobalC::ppcell.tab.ptr;

            if (this->device == base_device::GpuDevice)
            {
                syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_g_plus_k, g_plus_k.data(), g_plus_k.size());
                syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_vq_tab, GlobalC::ppcell.tab.ptr, GlobalC::ppcell.tab.getSize());
                gk = d_g_plus_k;
                vq_tb = d_vq_tab;
            }
            hamilt::cal_vq_op<FPTYPE, Device>()(
                this->ctx, vq_tb, it, gk,
                npw, GlobalC::ppcell.tab.getBound2(),GlobalC::ppcell.tab.getBound3(),
                GlobalV::DQ, GlobalC::ucell.atoms[it].ncpp.nbeta, hd_vq
            );   
            //prepare（-i）^l, size: nh


            std::vector<complex<double>> pref =  cal_pref<double>(it);
            int nh = pref.size();

            for(int ia=0;ia<h_atom_na[it];ia++)
            {
                // prepare SK
                std::complex<FPTYPE>* sk=GlobalC::ppcell.psf->get_sk( ik,it,ia, wfc_basis);;
                // 1. calculate becp
                // 1.a calculate vkb
                // 2.b calculate becp = vkb * psi
                if (this->device == base_device::GpuDevice)
                {

                    prepare_vkb_ptr<FPTYPE, Device>(
                            GlobalC::ucell.atoms[it].ncpp.nbeta, GlobalC::ppcell.nhtol.c, 
                            GlobalC::ppcell.nhtol.nc, npw, it,
                            ppcell_vkb, vkb_ptrs,
                            hd_ylm, ylm_ptrs,
                            hd_vq, vq_ptrs
                        );

                    // transfer the pointers from CPU to GPU
                    hamilt::synchronize_ptrs<Device>()((void**)d_vq_ptrs, (const void**)vq_ptrs, nh);
                    hamilt::synchronize_ptrs<Device>()((void**)d_ylm_ptrs, (const void**)ylm_ptrs, nh);
                    hamilt::synchronize_ptrs<Device>()((void**)d_vkb_ptrs, (const void**)vkb_ptrs, nh);  

                    syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_sk, sk, npw);
                    syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_pref_in, pref.data(), nh);

                    hamilt::cal_vkb_op<FPTYPE, Device>()(
                        this->ctx, nh, npw, d_vq_ptrs, d_ylm_ptrs,
                        d_sk, d_pref_in, d_vkb_ptrs
                    );
                        
                } else {
                    prepare_vkb_ptr<FPTYPE, Device>(
                            GlobalC::ucell.atoms[it].ncpp.nbeta, GlobalC::ppcell.nhtol.c, 
                            GlobalC::ppcell.nhtol.nc, npw, it,
                            ppcell_vkb, vkb_ptrs,
                            hd_ylm, ylm_ptrs,
                            hd_vq, vq_ptrs
                        );

                    hamilt::cal_vkb_op<FPTYPE, Device>()(
                        this->ctx, nh, npw, vq_ptrs, ylm_ptrs,
                        sk, pref.data(), vkb_ptrs
                    );
                }

   
                //syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, ppcell_vkb_d, ppcell_vkb, nh * npw);
                gemm_op()(this->ctx,
                        transa,
                        transb,
                        nh,
                        npm,
                        npw,
                        &ModuleBase::ONE,
                        ppcell_vkb,
                        npw,
                        &(psi_in[0](ik, 0, 0)),
                        npwx,
                        &ModuleBase::ZERO,
                        becp_ptr,
                        nkb);
                    
  

                becp_ptr += nh;

                for (int ipol = 0; ipol < 3; ipol++)
                {

                    cal_vkb1_nl_op()(this->ctx,
                                    nh,
                                    nbasis,
                                    wfc_basis->npwk_max,
                                    npw,
                                    nbasis,
                                    ik,
                                    ipol,
                                    ModuleBase::NEG_IMAG_UNIT,
                                    ppcell_vkb,
                                    gcar,
                                    vkb1);

                    gemm_op()(this->ctx,
                            transa,
                            transb,
                            nh,
                            npm,
                            nbasis,
                            &ModuleBase::ONE,
                            vkb1,
                            nbasis,
                            &(psi_in[0](ik, 0, 0)),
                            this->npwx,
                            &ModuleBase::ZERO,
                            dbecp_ptr[ipol],
                            nkb);
                    dbecp_ptr[ipol] += nh;
                } // end ipol
                delete [] sk;
            }//end ia


        }//end it
        // becp calculate is over , now we should broadcast this data.
        if (this->device == base_device::GpuDevice)
        {
            std::complex<FPTYPE> *h_becp = nullptr;
            resmem_complex_h_op()(this->cpu_ctx, h_becp, GlobalV::NBANDS * nkb);
            syncmem_complex_d2h_op()(this->cpu_ctx, this->ctx, h_becp, becp, GlobalV::NBANDS * nkb);
            Parallel_Reduce::reduce_pool(h_becp, GlobalV::NBANDS * nkb);
            syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, becp, h_becp, GlobalV::NBANDS * nkb);
            delmem_complex_h_op()(this->cpu_ctx, h_becp);
        }
        else
        {
            Parallel_Reduce::reduce_pool(becp, GlobalV::NBANDS * nkb);
        }
        int index = 0;        

        //		don't need to reduce here, keep dbecp different in each processor,
        //		and at last sum up all the forces.
        //		Parallel_Reduce::reduce_pool( dbecp.ptr, dbecp.ndata);
        
        cal_force_nl_op()(this->ctx,
                          nondiagonal,
                          nbands_occ,
                          wg_nc,
                          GlobalC::ucell.ntype,
                          current_spin,
                          GlobalC::ppcell.deeq.getBound2(),
                          GlobalC::ppcell.deeq.getBound3(),
                          GlobalC::ppcell.deeq.getBound4(),
                          forcenl.nc,
                          GlobalV::NBANDS,
                          ik,
                          nkb,
                          atom_nh,
                          atom_na,
                          GlobalC::ucell.tpiba,
                          d_wg,
                          d_ekb,
                          qq_nt,
                          deeq,
                          becp,
                          dbecp,
                          force);
    } // end ik

    if (this->device == base_device::GpuDevice)
    {
        syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, forcenl.c, force, forcenl.nr * forcenl.nc);
    }
    // sum up forcenl from all processors
    Parallel_Reduce::reduce_all(forcenl.c, forcenl.nr* forcenl.nc);

    delete [] h_atom_nh;
    delete [] h_atom_na;
    delmem_complex_op()(this->ctx, becp);
    delmem_complex_op()(this->ctx, dbecp);
    delmem_complex_op()(this->ctx, vkb1);
    if (this->device == base_device::GpuDevice) {
        delmem_var_op()(this->ctx, d_wg);
        delmem_var_op()(this->ctx, d_ekb);
        delmem_var_op()(this->ctx, gcar);
        delmem_int_op()(this->ctx, atom_nh);
        delmem_int_op()(this->ctx, atom_na);
    }
	ModuleBase::timer::tick("Forces","cal_force_nl");
}

template class Forces<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Forces<double, base_device::DEVICE_GPU>;
#endif