#include "stress_func.h"
#include "module_base/math_polyint.h"
#include "module_base/math_ylmreal.h"
#include "module_base/timer.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/module_device/device.h"
#include "module_base/memory.h"
//calculate the nonlocal pseudopotential stress in PW
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::stress_nl(ModuleBase::matrix& sigma,
                                            const ModuleBase::matrix& wg,
                                            const ModuleBase::matrix& ekb,
                                            Structure_Factor* p_sf,
                                            K_Vectors* p_kv,
                                            ModuleSymmetry::Symmetry* p_symm,
                                            ModulePW::PW_Basis_K* wfc_basis,
                                            const psi::Psi<complex<FPTYPE>, Device>* psi_in,
                                            pseudopot_cell_vnl* nlpp_in,
                                            const UnitCell& ucell_in)
{
    ModuleBase::TITLE("Stress_Func", "stress_nl");
    ModuleBase::timer::tick("Stress_Func", "stress_nl");


    const int npwx = wfc_basis->npwk_max;
    this->nlpp = nlpp_in;
    this->ucell = &ucell_in;
    const int nkb = nlpp->nkb;
    int wg_nc = wg.nc;
    if (nkb == 0)
    {
        ModuleBase::timer::tick("Stress_Func", "stress_nl");
        return;
    }

    this->device = base_device::get_device_type<Device>(this->ctx);

    // There is a contribution for jh<>ih in US case or multi projectors case
    // Actually, the judge of nondiagonal should be done on every atom type
    const bool nondiagonal = (GlobalV::use_uspp || nlpp->multi_proj) ? true : false;
    // prepare the memory of stress and init some variables:
    int *atom_nh = nullptr, *atom_na = nullptr, *h_atom_nh = new int[this->ucell->ntype],
        *h_atom_na = new int[this->ucell->ntype];
    for (int ii = 0; ii < this->ucell->ntype; ii++)
    {
        h_atom_nh[ii] = this->ucell->atoms[ii].ncpp.nh;
        h_atom_na[ii] = this->ucell->atoms[ii].na;
    }
    FPTYPE *stress = nullptr, *sigmanlc = nullptr, *d_wg = nullptr, *d_ekb = nullptr, *gcar = nullptr,
           *deeq = nlpp->get_deeq_data<FPTYPE>(), *kvec_c = wfc_basis->get_kvec_c_data<FPTYPE>(),
           *qq_nt = nlpp->get_qq_nt_data<FPTYPE>();

    int max_nbeta=0,max_nh=0,max_npw=0 ,_lmax = nlpp->lmaxkb;
    for(int it=0;it<this->ucell->ntype;it++)//loop all elements 
    {
        max_nbeta = std::max(this->ucell->atoms[it].ncpp.nbeta,max_nbeta);
        max_nh = std::max(this->ucell->atoms[it].ncpp.nh,max_nh);
    }
    for(int ik=0;ik<p_kv->get_nks();ik++)//loop k points
    {
        max_npw = std::max(p_kv->ngk[ik],max_npw);
    }
    
    // allocate the memory for ops.
    FPTYPE *h_ylm = new FPTYPE[(_lmax+1)*(_lmax+1)*max_npw], *hd_ylm = nullptr, // (lmax + 1) * (lmax + 1) * npw
            *h_ylm_deri = new FPTYPE[3*(_lmax+1)*(_lmax+1)*max_npw], *hd_ylm_deri = nullptr, //3 * (lmax + 1) * (lmax + 1) * npw
            *hd_vq = nullptr, *hd_vq_deri = nullptr, //this->ucell->atoms[it].ncpp.nbeta * npw
            *h_g_plus_k = new FPTYPE[max_npw * 5], *d_g_plus_k = nullptr, // npw * 5
            *h_pref = new FPTYPE[max_nh], *d_pref = nullptr, // this->ucell->atoms[it].ncpp.nh
            *d_gk = nullptr, *d_vq_tab = nullptr; // npw

    // allocate the memory for vkb and vkb_deri.
    FPTYPE  **vq_ptrs= new FPTYPE*[max_nh], **d_vq_ptrs = nullptr,
            **ylm_ptrs= new FPTYPE*[max_nh], **d_ylm_ptrs = nullptr, 
            **vq_deri_ptrs= new FPTYPE*[max_nh], **d_vq_deri_ptrs = nullptr,
            **ylm_deri_ptrs1= new FPTYPE*[max_nh], **d_ylm_deri_ptrs1 = nullptr,
            **ylm_deri_ptrs2= new FPTYPE*[max_nh], **d_ylm_deri_ptrs2 = nullptr;  

    std::complex<FPTYPE>** vkb_ptrs = new std::complex<FPTYPE>*[max_nh];
    std::complex<FPTYPE>** d_vkb_ptrs = nullptr;
    std::complex<FPTYPE>*d_sk = nullptr, *d_pref_in = nullptr ;
    std::complex<FPTYPE>* ppcell_vkb = nullptr;

    resmem_var_op()(this->ctx, hd_vq, max_nbeta*max_npw);
    resmem_var_op()(this->ctx, hd_vq_deri, max_nbeta*max_npw);
    resmem_var_op()(this->ctx, hd_ylm, (_lmax+1)*(_lmax+1)*max_npw);
    resmem_var_op()(this->ctx, hd_ylm_deri, 3*(_lmax+1)*(_lmax+1)*max_npw);

    resmem_var_op()(this->ctx, stress, 9);
    setmem_var_op()(this->ctx, stress, 0, 9);
    resmem_var_h_op()(this->cpu_ctx, sigmanlc, 9);

    if (this->device == base_device::GpuDevice)
    {
        resmem_var_op()(this->ctx, d_wg, wg.nr * wg.nc);
        resmem_var_op()(this->ctx, d_ekb, ekb.nr * ekb.nc);
        resmem_var_op()(this->ctx, gcar, 3 * p_kv->get_nks() * wfc_basis->npwk_max);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_wg, wg.c, wg.nr * wg.nc);
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_ekb, ekb.c, ekb.nr * ekb.nc);
        syncmem_var_h2d_op()(this->ctx,
                             this->cpu_ctx,
                             gcar,
                             &wfc_basis->gcar[0][0],
                             3 * p_kv->get_nks() * wfc_basis->npwk_max);
        resmem_int_op()(this->ctx, atom_nh, this->ucell->ntype);
        resmem_int_op()(this->ctx, atom_na, this->ucell->ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_nh, h_atom_nh, this->ucell->ntype);
        syncmem_int_h2d_op()(this->ctx, this->cpu_ctx, atom_na, h_atom_na, this->ucell->ntype);
    
        
        
        resmem_var_op()(this->ctx, d_g_plus_k, max_npw * 5);
        resmem_var_op()(this->ctx, d_pref, max_nh);
        resmem_var_op()(this->ctx, d_vq_tab, nlpp->tab.getSize());
    
        hamilt::pointer_array_malloc<Device>()((void **)&d_ylm_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_vq_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_vkb_ptrs, max_nh);

        hamilt::pointer_array_malloc<Device>()((void **)&d_vq_deri_ptrs, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_ylm_deri_ptrs1, max_nh);
        hamilt::pointer_array_malloc<Device>()((void **)&d_ylm_deri_ptrs2, max_nh);

        resmem_complex_op()(this->ctx, d_sk,  max_npw, "Stress::d_sk");
        resmem_complex_op()(this->ctx, d_pref_in,  max_nh, "Stress::pref_in");

        ppcell_vkb= nlpp->get_vkb_data<FPTYPE>();
    }
    else
    {
        d_wg = wg.c;
        d_ekb = ekb.c;
        gcar = &wfc_basis->gcar[0][0];
        atom_nh = h_atom_nh;
        atom_na = h_atom_na;
        ppcell_vkb = nlpp->vkb.c;
    }

    // prepare the memory of the becp and dbecp:
    // becp: <Beta(nkb,npw)|psi(nbnd,npw)>
    // dbecp: <dBeta(nkb,npw)/dG|psi(nbnd,npw)>
    std::complex<FPTYPE> *dbecp = nullptr, *becp = nullptr;
    resmem_complex_op()(this->ctx, becp, wg_nc * nkb, "Stress::becp");
    resmem_complex_op()(this->ctx, dbecp, 6 * wg_nc * nkb, "Stress::dbecp");

    int nks = p_kv->get_nks();
    for(int ik=0;ik<nks;ik++)//loop k points
    {
        // only for uspp: move the spin index in deeq
        const int current_spin = p_kv->isk[ik];
        // prepare parameters to calculate gemm
        const std::complex<FPTYPE> *ppsi = nullptr;
        ppsi = &(psi_in[0](ik, 0, 0));
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

        std::vector<FPTYPE> g_plus_k = cal_gk(ik, wfc_basis);
        std::complex<FPTYPE>* becp_ptr = becp;
        std::complex<FPTYPE>* dbecp_ptr[6];
        for(int i=0;i<6;i++)
        {
            dbecp_ptr[i] = &dbecp[i * wg_nc * nkb];
        }

        int lmax = nlpp->lmaxkb;
        //prepare ylm，size: (lmax+1)^2 * npwx
        cal_ylm(lmax, npw, g_plus_k.data(),hd_ylm);
        //prepare ylm'，size: 3 * (lmax+1)^2 * npwx，contain x,y,z 3 axis
        cal_ylm_deri(lmax, npw, g_plus_k.data(),hd_ylm_deri);

        /////////////////////////////
        //TODO: ylm,ylm_deri pass to GPU
        /////////////////////////////

        for(int it=0;it<this->ucell->ntype;it++)//loop all elements 
        {
            int lenth_vq = this->ucell->atoms[it].ncpp.nbeta*npw;
            // prepare inputs for calculating vkb，vkb1，vkb2 
            // prepare vq and vq', size: nq * npwx 
            std::vector<double> vq(lenth_vq);//cal_vq(it, g_plus_k.data(), npw);
            //std::vector<double> vq2(vq.size());

            FPTYPE *gk = g_plus_k.data(), *vq_tb = nlpp->tab.ptr;

            if (this->device == base_device::GpuDevice)
            {
                syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_g_plus_k, g_plus_k.data(), g_plus_k.size());
                syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, d_vq_tab, nlpp->tab.ptr, nlpp->tab.getSize());
                gk = d_g_plus_k;
                vq_tb = d_vq_tab;
                
            }
            cal_vq_op()(
                this->ctx, vq_tb, it, gk,
                npw, nlpp->tab.getBound2(),nlpp->tab.getBound3(),
                GlobalV::DQ, this->ucell->atoms[it].ncpp.nbeta, hd_vq
            );
            cal_vq_deri_op()(
                this->ctx, vq_tb, it, gk,
                npw, nlpp->tab.getBound2(),nlpp->tab.getBound3(),
                GlobalV::DQ, this->ucell->atoms[it].ncpp.nbeta, hd_vq_deri
            );      
            
            // prepare（-i）^l, size: nh
            std::vector<complex<double>> pref = cal_pref(it);
            int nh = pref.size();


            for(int ia=0;ia<h_atom_na[it];ia++)
            {
                // prepare SK
                std::complex<FPTYPE>* sk = p_sf->get_sk(ik, it, ia, wfc_basis);
                // 1. calculate becp
                // 1.a calculate vkb


                if (this->device == base_device::GpuDevice)
                {
                    syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_sk, sk, npw);
                    syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx, d_pref_in, pref.data(), nh);


                    prepare_vkb_ptr(
                            this->ucell->atoms[it].ncpp.nbeta, nlpp->nhtol.c, 
                            nlpp->nhtol.nc, npw, it,
                            ppcell_vkb, vkb_ptrs,
                            hd_ylm, ylm_ptrs,
                            hd_vq, vq_ptrs
                        );

                    // transfer the pointers from CPU to GPU
                    hamilt::synchronize_ptrs<Device>()((void**)d_vq_ptrs, (const void**)vq_ptrs, nh);
                    hamilt::synchronize_ptrs<Device>()((void**)d_ylm_ptrs, (const void**)ylm_ptrs, nh);
                    hamilt::synchronize_ptrs<Device>()((void**)d_vkb_ptrs, (const void**)vkb_ptrs, nh);  


                    cal_vkb_op()(
                        this->ctx, nh, npw, d_vq_ptrs, d_ylm_ptrs,
                        d_sk, d_pref_in, d_vkb_ptrs
                    );

                    //syncmem_complex_d2h_op()(this->cpu_ctx, this->ctx, ppcell_vkb, ppcell_vkb_d, nh * npw);

                        
                } else {
                    prepare_vkb_ptr(
                            this->ucell->atoms[it].ncpp.nbeta, nlpp->nhtol.c, 
                            nlpp->nhtol.nc, npw, it,
                            ppcell_vkb, vkb_ptrs,
                            hd_ylm, ylm_ptrs,
                            hd_vq, vq_ptrs
                        );

                    cal_vkb_op()(
                        this->ctx, nh, npw, vq_ptrs, ylm_ptrs,
                        sk, pref.data(), vkb_ptrs
                    );
                }

                // 2.b calculate becp = vkb * psi
                int npm = GlobalV::NPOL * nbands_occ;


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
                        ppsi,
                        npwx,
                        &ModuleBase::ZERO,
                        becp_ptr,
                        nkb);
                    



                becp_ptr += nh;
                //calculate stress（00，01，02，11，12，22）
                int index = 0;
                for(int ipol=0;ipol<3;ipol++)
                {
                    for(int jpol=0;jpol<ipol+1;jpol++)
                    {
                        // 2. calculate dbecp：
                        // 2.a. calculate dbecp_noevc, repeat use the memory of ppcell.vkb
                        
                        if (this->device == base_device::GpuDevice)
                        {

                            prepare_vkb_deri_ptr(
                                this->ucell->atoms[it].ncpp.nbeta, nlpp->nhtol.c, 
                                nlpp->nhtol.nc, npw, it, ipol, jpol,
                                ppcell_vkb, vkb_ptrs,
                                hd_ylm, ylm_ptrs,
                                hd_ylm_deri, ylm_deri_ptrs1, ylm_deri_ptrs2,
                                hd_vq, vq_ptrs,
                                hd_vq_deri, vq_deri_ptrs
                            );

                            // transfer the pointers from CPU to GPU
                            hamilt::synchronize_ptrs<Device>()((void**)d_vq_deri_ptrs, (const void**)vq_deri_ptrs, nh);
                            hamilt::synchronize_ptrs<Device>()((void**)d_ylm_deri_ptrs1, (const void**)ylm_deri_ptrs1, nh);
                            hamilt::synchronize_ptrs<Device>()((void**)d_ylm_deri_ptrs2, (const void**)ylm_deri_ptrs2, nh);
                            cal_vkb_deri_op()(
                                this->ctx, nh, npw, ipol, jpol,
                                d_vq_ptrs, d_vq_deri_ptrs,
                                d_ylm_ptrs, d_ylm_deri_ptrs1, d_ylm_deri_ptrs2,
                                d_sk, d_pref_in, d_g_plus_k, d_vkb_ptrs
                            );   
                        } 
                        else 
                        {

                            prepare_vkb_deri_ptr(
                                this->ucell->atoms[it].ncpp.nbeta, nlpp->nhtol.c, 
                                nlpp->nhtol.nc, npw, it, ipol, jpol,
                                ppcell_vkb, vkb_ptrs,
                                hd_ylm, ylm_ptrs,
                                hd_ylm_deri, ylm_deri_ptrs1, ylm_deri_ptrs2,
                                hd_vq, vq_ptrs,
                                hd_vq_deri, vq_deri_ptrs
                            );
                            cal_vkb_deri_op()(
                                this->ctx, nh, npw, ipol, jpol,
                                vq_ptrs, vq_deri_ptrs,
                                ylm_ptrs, ylm_deri_ptrs1, ylm_deri_ptrs2,
                                sk, pref.data(), g_plus_k.data(), vkb_ptrs
                            );
                        }
                 
                        // 2.b calculate dbecp = dbecp_noevc * psi

                        gemm_op()(this->ctx,
                        transa,
                        transb,
                        nh,
                        npm,
                        npw,
                        &ModuleBase::ONE,
                        ppcell_vkb,
                        npw,
                        ppsi,
                        npwx,
                        &ModuleBase::ZERO,
                        dbecp_ptr[index],
                        nkb);


                        dbecp_ptr[index++] += nh;
                    }//jpol
                }//ipol
                delete [] sk;
            }//ia
        }//it
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
        // 3. stress(ipol, jpol) += \sum becp * dbecp * D_pp'
        int index = 0;

        for(int ipol=0;ipol<3;ipol++)
        {
            for(int jpol=0;jpol<=ipol;jpol++)
            {
                cal_stress_nl_op()(this->ctx,
                                   nondiagonal,
                                   ipol,
                                   jpol,
                                   nkb,
                                   nbands_occ,
                                   this->ucell->ntype,
                                   current_spin,//uspp only
                                   wg_nc,
                                   ik,
                                   nlpp->deeq.getBound2(),
                                   nlpp->deeq.getBound3(),
                                   nlpp->deeq.getBound4(),
                                   atom_nh,
                                   atom_na,
                                   d_wg,
                                   d_ekb,
                                   qq_nt,
                                   deeq,
                                   becp,
                                   &dbecp[index * wg_nc * nkb],
                                   stress);
                index++;
            }//jpol
        }//ipol
    }//ik
                
    syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, sigmanlc, stress, 9);
	// sum up forcenl from all processors
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			if(m>l) 
			{
				sigmanlc[l * 3 + m] = sigmanlc[m * 3 + l];
			}
            Parallel_Reduce::reduce_all(sigmanlc[l * 3 + m]); //qianrui fix a bug for kpar > 1
		}
	}




    //        Parallel_Reduce::reduce_all(sigmanl.c, sigmanl.nr * sigmanl.nc);
        
	for (int ipol = 0; ipol<3; ipol++)
	{
		for(int jpol = 0; jpol < 3; jpol++)
		{
			sigmanlc[ipol * 3 + jpol] *= 1.0 / this->ucell->omega;
		}
	}
	
	for (int ipol = 0; ipol<3; ipol++)
	{
		for(int jpol = 0; jpol < 3; jpol++)
		{
			sigma(ipol,jpol) = sigmanlc[ipol * 3 + jpol] ;
		}
	}
	//do symmetry
    if (ModuleSymmetry::Symmetry::symm_flag == 1)
    {
        p_symm->symmetrize_mat3(sigma, this->ucell->lat);
    } // end symmetry

    delete [] h_atom_nh;
    delete [] h_atom_na;
    delmem_var_op()(this->ctx, stress);
    delmem_complex_op()(this->ctx, becp);
    delmem_complex_op()(this->ctx, dbecp);
	delmem_var_h_op()(this->cpu_ctx, sigmanlc);
    if (this->device == base_device::GpuDevice)
    {
        delmem_var_op()(this->ctx, d_wg);
        delmem_var_op()(this->ctx, d_ekb);
        delmem_int_op()(this->ctx, atom_nh);
        delmem_int_op()(this->ctx, atom_na);
    }
	ModuleBase::timer::tick("Stress_Func","stress_nl");
}
// cal_gk
template <typename FPTYPE, typename Device>
std::vector<FPTYPE> Stress_Func<FPTYPE, Device>::cal_gk(int ik, ModulePW::PW_Basis_K* wfc_basis)
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
        gk[3 * npw + ig] = norm * this->ucell->tpiba;
        gk[4 * npw + ig] = norm<1e-8?0.0:1.0/norm*this->ucell->tpiba;
    }
    return gk;
}

// cal_vq
template <typename FPTYPE, typename Device>
std::vector<FPTYPE> Stress_Func<FPTYPE, Device>::cal_vq(int it, const FPTYPE* gk, int npw)
{
    // calculate beta in G-space using an interpolation table
    const int nbeta = this->ucell->atoms[it].ncpp.nbeta;

    std::vector<FPTYPE> vq(nbeta * npw);
    ModuleBase::Memory::record("stress_nl::vq", nbeta * npw * sizeof(FPTYPE));

    for (int nb = 0; nb < nbeta; nb++)
    {
        FPTYPE* vq_ptr = &vq[nb * npw];
        const FPTYPE* gnorm = &gk[3 * npw];
        for (int ig = 0; ig < npw; ig++)
        {
            vq_ptr[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(nlpp->tab,
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
std::vector<FPTYPE> Stress_Func<FPTYPE, Device>::cal_vq_deri(int it, const FPTYPE* gk, int npw)
{
    // calculate beta in G-space using an interpolation table
    const int nbeta = this->ucell->atoms[it].ncpp.nbeta;

    std::vector<FPTYPE> vq(nbeta * npw);
    ModuleBase::Memory::record("stress_nl::vq_deri", nbeta * npw * sizeof(FPTYPE));

    for (int nb = 0; nb < nbeta; nb++)
    {
        const FPTYPE* gnorm = &gk[3 * npw];
        FPTYPE* vq_ptr = &vq[nb * npw];
        for (int ig = 0; ig < npw; ig++)
        {
            vq_ptr[ig] = this->Polynomial_Interpolation_nl(
						nlpp->tab, 
                        it, 
                        nb, 
                        GlobalV::DQ, 
                        gnorm[ig] );
        }
    }
    return vq;
}


// cal_ylm
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::cal_ylm(int lmax, int npw, const FPTYPE* gk_in, FPTYPE* ylm)
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
// cal_ylm_deri
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::cal_ylm_deri(int lmax, int npw, const FPTYPE* gk_in,FPTYPE* ylm_deri)
{
    int x1 = (lmax + 1) * (lmax + 1);
    ModuleBase::Memory::record("stress_nl::dylm", 3* x1 * npw * sizeof(FPTYPE));

    if (this->device == base_device::GpuDevice)
    {
        std::vector<FPTYPE> dylm(3 * x1 * npw);
        for(int ipol = 0;ipol<3;ipol++)
        {
            dylmr2(x1, npw, gk_in, &dylm[ipol * x1 * npw], ipol);
        }
        syncmem_var_h2d_op()(this->ctx, this->cpu_ctx, ylm_deri, dylm.data(), dylm.size());
    } else {
        for(int ipol = 0;ipol<3;ipol++)
        {
            dylmr2(x1, npw, gk_in, &ylm_deri[ipol * x1 * npw], ipol);
        }
    }


    
    return ;
}
// cal_pref
template <typename FPTYPE, typename Device>
std::vector<std::complex<FPTYPE>> Stress_Func<FPTYPE, Device>::cal_pref(int it)
{
    const int nh = this->ucell->atoms[it].ncpp.nh;
    std::vector<std::complex<FPTYPE>> pref(nh);
    for(int ih=0;ih<nh;ih++)
    {
        pref[ih] = std::pow(std::complex<FPTYPE>(0.0, -1.0), nlpp->nhtol(it, ih));
    }
    return pref;
}


// cal_vkb
// cpu version first, gpu version later
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::cal_vkb(
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
    for(int nb=0;nb<this->ucell->atoms[it].ncpp.nbeta;nb++)
    {
        int l = nlpp->nhtol(it, ih);
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
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::cal_vkb_deri(
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
    int x1 = (nlpp->lmaxkb + 1) * (nlpp->lmaxkb + 1);
    int ih=0;
    // loop over all beta functions
    for(int nb=0;nb<this->ucell->atoms[it].ncpp.nbeta;nb++)
    {
        int l = nlpp->nhtol(it, ih);
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

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::prepare_vkb_ptr(
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
void Stress_Func<FPTYPE, Device>::prepare_vkb_deri_ptr(
    int nbeta, double* nhtol, int nhtol_nc, int npw, int it,
    int ipol, int jpol,
    std::complex<FPTYPE>*vkb_out, std::complex<FPTYPE>** vkb_ptrs,
    FPTYPE* ylm_in, FPTYPE** ylm_ptrs,
    FPTYPE* ylm_deri_in, FPTYPE** ylm_deri_ptr1s, FPTYPE** ylm_deri_ptr2s,
    FPTYPE* vq_in, FPTYPE** vq_ptrs,
    FPTYPE* vq_deri_in, FPTYPE** vq_deri_ptrs

) {   
    int ih=0;
    int x1 = (nlpp->lmaxkb + 1) * (nlpp->lmaxkb + 1);
    for(int nb=0;nb<nbeta;nb++)
    {
        int l = nhtol[it*nhtol_nc+ih];
        for(int m=0;m<2*l+1;m++)
        {
            int lm = l*l + m;
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
void Stress_Func<FPTYPE, Device>::get_dvnl1(ModuleBase::ComplexMatrix &vkb,
                                            const int ik,
                                            const int ipol,
                                            Structure_Factor *p_sf,
                                            ModulePW::PW_Basis_K *wfc_basis)
{
    if (GlobalV::test_pp)
        ModuleBase::TITLE("Stress_Func", "get_dvnl1");

    const int npw = wfc_basis->npwk[ik];
    const int lmaxkb = nlpp->lmaxkb;
    if (lmaxkb < 0)
    {
		return;
	}

	const int nhm = nlpp->nhm;
	ModuleBase::matrix vkb1(nhm, npw);
	vkb1.zero_out();
	FPTYPE *vq = new FPTYPE[npw];
	const int x1= (lmaxkb + 1)*(lmaxkb + 1);

	ModuleBase::matrix dylm(x1, npw);
	ModuleBase::Vector3<FPTYPE> *gk = new ModuleBase::Vector3<FPTYPE>[npw];
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int ig = 0;ig < npw;ig++)
	{
		gk[ig] = wfc_basis->getgpluskcar(ik, ig);
	}
			   
	dylmr2(x1, npw, reinterpret_cast<double*>(gk), dylm.c, ipol);

	const int imag_pow_period = 4;
    // result table of pow(0-1i, int)
    static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
	int jkb = 0;
	for(int it = 0;it < this->ucell->ntype;it++)
	{
		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("it",it);
		// calculate beta in G-space using an interpolation table
		const int nbeta = this->ucell->atoms[it].ncpp.nbeta;
		const int nh = this->ucell->atoms[it].ncpp.nh;

		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("nbeta",nbeta);

		for (int nb = 0;nb < nbeta;nb++)
		{
			if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("ib",nb);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int ig = 0;ig < npw;ig++)
			{
				const FPTYPE gnorm = gk[ig].norm() * this->ucell->tpiba;

				//cout << "\n gk[ig] = " << gk[ig].x << " " << gk[ig].y << " " << gk[ig].z;
				//cout << "\n gk.norm = " << gnorm;

				vq [ig] = ModuleBase::PolyInt::Polynomial_Interpolation(
						nlpp->tab, it, nb, GlobalV::NQX, GlobalV::DQ, gnorm );

			} // enddo

			// add spherical harmonic part
			for (int ih = 0;ih < nh;ih++)
			{
				if (nb == nlpp->indv(it, ih))
				{
					const int lm = static_cast<int>( nlpp->nhtolm(it, ih) );
#ifdef _OPENMP
#pragma omp parallel for
#endif
					for (int ig = 0;ig < npw;ig++)
					{
						vkb1(ih, ig) = dylm(lm, ig) * vq [ig];
					}

				}

			} // end ih

		} // end nbeta

		// vkb1 contains all betas including angular part for type nt
		// now add the structure factor and factor (-i)^l
		for (int ia=0; ia<this->ucell->atoms[it].na; ia++)
		{
            std::complex<FPTYPE> *sk = p_sf->get_sk(ik, it, ia, wfc_basis);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
			for (int ih = 0;ih < nh;ih++)
			{
				for (int ig = 0;ig < npw;ig++)
				{	
					std::complex<FPTYPE> pref = pref_tab[int(nlpp->nhtol(it, ih)) % imag_pow_period];      //?
					vkb(jkb + ih, ig) = vkb1(ih, ig) * sk [ig] * pref;
				}
				
			} // end ih
			jkb += nh;
		delete [] sk;
		} // end ia
	} // enddo
	delete [] gk;
	delete [] vq;
	return;
}//end get_dvnl1

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::get_dvnl2(ModuleBase::ComplexMatrix &vkb,
                                            const int ik,
                                            Structure_Factor *p_sf,
                                            ModulePW::PW_Basis_K *wfc_basis)
{
    if (GlobalV::test_pp)
        ModuleBase::TITLE("Stress", "get_dvnl2");
    //	ModuleBase::timer::tick("Stress","get_dvnl2");
    const int npw = wfc_basis->npwk[ik];
    const int lmaxkb = nlpp->lmaxkb;
	if(lmaxkb < 0)
	{
		return;
	}

	const int nhm = nlpp->nhm;
	ModuleBase::matrix vkb1(nhm, npw);
	FPTYPE *vq = new FPTYPE[npw];
	const int x1= (lmaxkb + 1)*(lmaxkb + 1);

	ModuleBase::matrix ylm(x1, npw);
	ModuleBase::Vector3<FPTYPE> *gk = new ModuleBase::Vector3<FPTYPE>[npw];
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int ig = 0;ig < npw;ig++)
	{
		gk[ig] = wfc_basis->getgpluskcar(ik, ig);
	}
	ModuleBase::YlmReal::Ylm_Real(x1, npw, gk, ylm);

	const int imag_pow_period = 4;
    // result table of pow(0-1i, int)
    static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
	int jkb = 0;
	for(int it = 0;it < this->ucell->ntype;it++)
	{
		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("it",it);
		// calculate beta in G-space using an interpolation table
		const int nbeta = this->ucell->atoms[it].ncpp.nbeta;
		const int nh = this->ucell->atoms[it].ncpp.nh;

		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("nbeta",nbeta);

		for (int nb = 0;nb < nbeta;nb++)
		{
			if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("ib",nb);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int ig = 0;ig < npw;ig++)
			{
				const FPTYPE gnorm = gk[ig].norm() * this->ucell->tpiba;
	//cout << "\n gk[ig] = " << gk[ig].x << " " << gk[ig].y << " " << gk[ig].z;
	//cout << "\n gk.norm = " << gnorm;
				vq [ig] = Polynomial_Interpolation_nl(
						nlpp->tab, it, nb, GlobalV::DQ, gnorm );

			} // enddo

							// add spherical harmonic part
			for (int ih = 0;ih < nh;ih++)
			{
				if (nb == nlpp->indv(it, ih))
				{
					const int lm = static_cast<int>( nlpp->nhtolm(it, ih) );
#ifdef _OPENMP
#pragma omp parallel for
#endif
					for (int ig = 0;ig < npw;ig++)
					{
						vkb1(ih, ig) = ylm(lm, ig) * vq [ig];
					}
				}
			} // end ih
		} // end nbeta

		// vkb1 contains all betas including angular part for type nt
		// now add the structure factor and factor (-i)^l
		for (int ia=0; ia<this->ucell->atoms[it].na; ia++)
		{
            std::complex<FPTYPE> *sk = p_sf->get_sk(ik, it, ia, wfc_basis);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
			for (int ih = 0;ih < nh;ih++)
			{
				for (int ig = 0;ig < npw;ig++)
				{
					std::complex<FPTYPE> pref = pref_tab[int(nlpp->nhtol(it, ih)) % imag_pow_period];      //?
					vkb(jkb + ih, ig) = vkb1(ih, ig) * sk [ig] * pref;
				}
			
			} // end ih
			jkb += nh;
			delete [] sk;
		} // end ia
	}	 // enddo

	delete [] gk;
	delete [] vq;
//	ModuleBase::timer::tick("Stress","get_dvnl2");

	return;
}


template <typename FPTYPE, typename Device>
FPTYPE Stress_Func<FPTYPE, Device>::Polynomial_Interpolation_nl
(
    const ModuleBase::realArray &table,
    const int &dim1,
    const int &dim2,
    const FPTYPE &table_interval,
    const FPTYPE &x                             // input value
)
{

	assert(table_interval>0.0);
	const FPTYPE position = x  / table_interval;
	const int iq = static_cast<int>(position);

	const FPTYPE x0 = position - static_cast<FPTYPE>(iq);
	const FPTYPE x1 = 1.0 - x0;
	const FPTYPE x2 = 2.0 - x0;
	const FPTYPE x3 = 3.0 - x0;
	const FPTYPE y=
			( table(dim1, dim2, iq)   * (-x2*x3-x1*x3-x1*x2) / 6.0 +
			table(dim1, dim2, iq+1) * (+x2*x3-x0*x3-x0*x2) / 2.0 -
			table(dim1, dim2, iq+2) * (+x1*x3-x0*x3-x0*x1) / 2.0 +
			table(dim1, dim2, iq+3) * (+x1*x2-x0*x2-x0*x1) / 6.0 )/table_interval ;


	return y;
}

template <typename FPTYPE, typename Device>
FPTYPE Stress_Func<FPTYPE, Device>::Polynomial_Interpolation_nl(const ModuleBase::realArray& table,
                                                                const int& dim1,
                                                                const int& dim2,
                                                                const int& dim3,
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
    const FPTYPE y = (table(dim1, dim2, dim3, iq) * (-x2 * x3 - x1 * x3 - x1 * x2) / 6.0
                      + table(dim1, dim2, dim3, iq + 1) * (+x2 * x3 - x0 * x3 - x0 * x2) / 2.0
                      - table(dim1, dim2, dim3, iq + 2) * (+x1 * x3 - x0 * x3 - x0 * x1) / 2.0
                      + table(dim1, dim2, dim3, iq + 3) * (+x1 * x2 - x0 * x2 - x0 * x1) / 6.0)
                     / table_interval;

    return y;
}

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::dylmr2 (
	const int nylm,
	const int ngy,
	const FPTYPE *gk,
	FPTYPE* dylm,
	const int ipol)
{
  //-----------------------------------------------------------------------
  //
  //     compute \partial Y_lm(G) \over \partial (G)_ipol
  //     using simple numerical derivation (SdG)
  //     The spherical harmonics are calculated in ylmr2
  //
  //int nylm, ngy, ipol;
  // number of spherical harmonics
  // the number of g vectors to compute
  // desired polarization
  //FPTYPE g (3, ngy), gg (ngy), dylm (ngy, nylm)
  // the coordinates of g vectors
  // the moduli of g vectors
  // the spherical harmonics derivatives
  //
	const FPTYPE delta = 1e-6;
	FPTYPE *dg, *dgi;

	ModuleBase::matrix ylmaux;
	// dg is the finite increment for numerical derivation:
	// dg = delta |G| = delta * sqrt(gg)
	// dgi= 1 /(delta * sqrt(gg))
	// gx = g +/- dg


	FPTYPE *gx = new FPTYPE[3 * ngy];
	 

	dg = new FPTYPE [ngy];
	dgi = new FPTYPE [ngy];

	ylmaux.create (nylm, ngy);

	ModuleBase::GlobalFunc::ZEROS(dylm, nylm * ngy);
	ylmaux.zero_out();

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int ig = 0;ig< 3 * ngy;ig++){
		gx[ig] = gk[ig];
	}
	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int ig = 0;ig< ngy;ig++)
    {
        FPTYPE norm2 = gx[ig * 3] * gx[ig * 3] + gx[ig * 3 + 1] * gx[ig * 3 + 1] + gx[ig * 3 + 2] * gx[ig * 3 + 2];
		dg [ig] = delta * sqrt(norm2) ;
		if (norm2 > 1e-9) {
			dgi [ig] = 1.0 / dg [ig];
		}
		else{
			dgi [ig] = 0.0;
		}
	}
	//$OMP END PARALLEL DO

	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int ig = 0;ig< ngy;ig++)
	{
		gx [ig*3 + ipol] = gk[ ig*3 + ipol] + dg [ig];
	}
	//$OMP END PARALLEL DO

    ModuleBase::YlmReal::Ylm_Real(cpu_ctx, nylm, ngy, gx, dylm);
	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int ig = 0;ig< ngy;ig++)
	{
		gx [ig * 3 + ipol] = gk [ ig * 3 + ipol] - dg [ig];
	}
	//$OMP END PARALLEL DO

    ModuleBase::YlmReal::Ylm_Real(cpu_ctx, nylm, ngy, gx, ylmaux.c);


	//  zaxpy ( - 1.0, ylmaux, 1, dylm, 1);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
	for(int lm = 0;lm< nylm;lm++)
	{
		for(int ig = 0;ig< ngy;ig++)
		{
			dylm[lm * ngy + ig] -= ylmaux(lm,ig);
			dylm[lm * ngy + ig] *= 0.5 * dgi [ig];
		}
	}

	delete[] gx;
	delete[] dg;
	delete[] dgi;

	return;
}

template class Stress_Func<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_Func<double, base_device::DEVICE_GPU>;
#endif