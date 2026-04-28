#include "op_pw_proj.h"

#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_pw/module_pwdft/kernels/onsite_op.h"


namespace hamilt {

template<typename T, typename Device>
OnsiteProj<OperatorPW<T, Device>>::OnsiteProj(const int* isk_in,
		const UnitCell* ucell_in,
		Plus_U *p_dftu, // mohan add 2025-11-06 
		const bool cal_delta_spin,
		const bool cal_dftu)
{
    this->classname = "OnsiteProj";
    this->cal_type = calculation_type::pw_onsite;
    this->isk = isk_in;
    this->ucell = ucell_in;
    this->has_delta_spin = cal_delta_spin;
    this->has_dftu = cal_dftu;
    this->dftu = p_dftu; // mohan add 2025-11-08
}

template<typename T, typename Device>
OnsiteProj<OperatorPW<T, Device>>::~OnsiteProj() {
    delmem_complex_op()(this->ps);
    if(this->init_delta_spin)
    {
        delmem_int_op()(this->ip_iat);
        delmem_complex_op()(this->lambda_coeff);
    }
    if(this->has_dftu)
    {
        if(!init_delta_spin)
        {
            delmem_int_op()(this->ip_iat);
        }
        delmem_int_op()(this->orb_l_iat);
        delmem_int_op()(this->ip_m);
        delmem_int_op()(this->vu_begin_iat);
        delmem_complex_op()(this->vu_device);
    }
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::init(const int ik_in)
{
    ModuleBase::timer::start("OnsiteProj", "getvnl");
    this->ik = ik_in;

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    onsite_p->tabulate_atomic(ik_in);
    this->tnp = onsite_p->get_tot_nproj();

    if(this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }

    ModuleBase::timer::end("OnsiteProj", "getvnl");
}

//--------------------------------------------------------------------------
// this function sum up each non-local pseudopotential located on each atom,
//--------------------------------------------------------------------------
template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::add_onsite_proj(T *hpsi_in, const int npol, const int m, const int npwx) const
{
    ModuleBase::timer::start("OnsiteProj", "add_onsite_proj");

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    // apply the operator to the wavefunction
    //std::cout << "use of tab_atomic at " << __FILE__ << ": " << __LINE__ << std::endl;
    const std::complex<double>* tab_atomic = onsite_p->get_tab_atomic();
    const int npw = onsite_p->get_npw();
    // npwx passed as parameter
    char transa = 'N';
    char transb = 'T';
    int npm = m;
    gemm_op()(
        transa,
        transb,
        npw,
        npm,
        this->tnp,
        &this->one,
        tab_atomic,
        npw,
        this->ps,
        npm,
        &this->one,
        hpsi_in,
        npwx
    );
    ModuleBase::timer::end("OnsiteProj", "add_onsite_proj");
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::update_becp(const T *psi_in, const int npol, const int m, const int npwx) const
{
    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    // calculate <alpha|psi> 
    // std::cout << __FILE__ << ":" << __LINE__ << " nbands = " << m << std::endl;
    onsite_p->overlap_proj_psi(m, psi_in);
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::cal_ps_delta_spin(const int npol, const int m) const
{
    if(!this->has_delta_spin) return;

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    const std::complex<double>* becp = onsite_p->get_becp();

    spinconstrain::SpinConstrain<std::complex<double>>& sc = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();
    auto& constrain = sc.get_constrain();
    auto& lambda = sc.get_sc_lambda();

    // T *ps = new T[tnp * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * tnp);
    if (this->nkb_m < m * tnp) {
        resmem_complex_op()(this->ps, tnp * m, "OnsiteProj<PW>::ps");
        this->nkb_m = m * tnp;
    }
    setmem_complex_op()(this->ps, 0, tnp * m);

    if(!this->init_delta_spin)
    {
        this->init_delta_spin = true;
        //prepare ip_iat and lambda_coeff
        resmem_int_op()(this->ip_iat, onsite_p->get_tot_nproj());
        resmem_complex_op()(this->lambda_coeff, this->ucell->nat * 4);
        std::vector<int> ip_iat0(onsite_p->get_tot_nproj());
        int ip0 = 0;
        for(int iat=0;iat<this->ucell->nat;iat++)
        {
            for(int ip=0;ip<onsite_p->get_nh(iat);ip++)
            {
                ip_iat0[ip0++] = iat;
            }
        }
        syncmem_int_h2d_op()(this->ip_iat, ip_iat0.data(), onsite_p->get_tot_nproj());
    }

    // prepare array of nh_iat and lambda_array to pass to the onsite_ps_op operator
    std::vector<std::complex<double>> tmp_lambda_coeff(this->ucell->nat * 4);
    for(int iat=0;iat<this->ucell->nat;iat++)
    {
        tmp_lambda_coeff[iat * 4] = std::complex<double>(lambda[iat][2], 0.0);
        tmp_lambda_coeff[iat * 4 + 1] = std::complex<double>(lambda[iat][0], lambda[iat][1]);
        tmp_lambda_coeff[iat * 4 + 2] = std::complex<double>(lambda[iat][0], -1 * lambda[iat][1]);
        tmp_lambda_coeff[iat * 4 + 3] = std::complex<double>(-1 * lambda[iat][2], 0.0);
    }
    syncmem_complex_h2d_op()(this->lambda_coeff, tmp_lambda_coeff.data(), this->ucell->nat * 4);
    // TODO: code block above should be moved to the init function

    hamilt::onsite_ps_op<Real, Device>()(
        this->ctx,   // device context
        m, 
        npol,
        this->ip_iat, 
        tnp,  
        this->lambda_coeff,
        this->ps, becp);
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::cal_ps_dftu(
		const int npol, 
		const int m) const
{
	if(!this->has_dftu) 
	{
		return;
	}

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    const std::complex<double>* becp = onsite_p->get_becp();

    // T *ps = new T[tnp * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * tnp);
    if (this->nkb_m < m * tnp) {
        resmem_complex_op()(this->ps, tnp * m, "OnsiteProj<PW>::ps");
        this->nkb_m = m * tnp;
    }
    if(!this->has_delta_spin) 
    {
        setmem_complex_op()(this->ps, 0, tnp * m);
    }

    if(!this->init_dftu)
    {
        this->init_dftu = true;
        //prepare orb_l_iat, ip_m, vu_begin_iat and vu_device
        resmem_int_op()(this->orb_l_iat, this->ucell->nat);
        resmem_int_op()(this->ip_m, onsite_p->get_tot_nproj());
        resmem_int_op()(this->vu_begin_iat, this->ucell->nat);
        // recal the ip_iat
        resmem_int_op()(this->ip_iat, onsite_p->get_tot_nproj());
        std::vector<int> ip_iat0(onsite_p->get_tot_nproj());
        std::vector<int> ip_m0(onsite_p->get_tot_nproj());
        std::vector<int> vu_begin_iat0(this->ucell->nat);
        std::vector<int> orb_l_iat0(this->ucell->nat);
        int ip0 = 0;
        int vu_begin = 0;
        for(int iat=0;iat<this->ucell->nat;iat++)
        {
            const int it = this->ucell->iat2it[iat];
            const int target_l = this->dftu->orbital_corr[it];
            orb_l_iat0[iat] = target_l;
            const int nproj = onsite_p->get_nh(iat);
            if(target_l == -1)
            {
                for(int ip=0;ip<nproj;ip++)
                {
                    ip_iat0[ip0] = iat;
                    ip_m0[ip0++] = -1;
                }
                vu_begin_iat0[iat] = 0;
                continue;
            }
            else
            {
                const int tlp1 = 2 * target_l + 1;
                vu_begin_iat0[iat] = vu_begin;
                vu_begin += tlp1 * tlp1 * npol * npol;
                const int m_begin = target_l * target_l;
                const int m_end  = (target_l + 1) * (target_l + 1);
                for(int ip=0;ip<nproj;ip++)
                {
                    ip_iat0[ip0] = iat;
                    if(ip >= m_begin && ip < m_end)
                    {
                        ip_m0[ip0++] = ip - m_begin;
                    }
                    else
                    {
                        ip_m0[ip0++] = -1;
                    }
                }
            }
        }
        syncmem_int_h2d_op()(this->orb_l_iat, orb_l_iat0.data(), this->ucell->nat);
        syncmem_int_h2d_op()(this->ip_iat, ip_iat0.data(), onsite_p->get_tot_nproj());
        syncmem_int_h2d_op()(this->ip_m, ip_m0.data(), onsite_p->get_tot_nproj());
        syncmem_int_h2d_op()(this->vu_begin_iat, vu_begin_iat0.data(), this->ucell->nat);

        resmem_complex_op()(this->vu_device, dftu->get_size_eff_pot_pw());
    }

    // For nspin=2, need to select spin-up or spin-down potential
    if(PARAM.inp.nspin == 2 && this->isk[this->ik] == 1)
    {
        const int size_eff_pot_pw = dftu->get_size_eff_pot_pw() / 2;
        syncmem_complex_h2d_op()(this->vu_device, 
            dftu->get_eff_pot_pw(0) + size_eff_pot_pw, 
            size_eff_pot_pw);
        hamilt::onsite_ps_op<Real, Device>()(
            this->ctx,   // device context
            m, 
            npol,
            this->orb_l_iat,
            this->ip_iat,
            this->ip_m,
            this->vu_begin_iat, 
            tnp,  
            this->vu_device,
            this->ps, becp);
    }
    else
    {
        syncmem_complex_h2d_op()(this->vu_device, dftu->get_eff_pot_pw(0), dftu->get_size_eff_pot_pw());
        hamilt::onsite_ps_op<Real, Device>()(
            this->ctx,   // device context
            m, 
            npol,
            this->orb_l_iat,
            this->ip_iat,
            this->ip_m,
            this->vu_begin_iat, 
            tnp,  
            this->vu_device,
            this->ps, becp);
    }
}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::add_onsite_proj(
		std::complex<float> *hpsi_in, 
		const int npol, 
		const int m,
		const int npwx) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::update_becp(
		const std::complex<float> *psi_in, 
		const int npol, 
		const int m,
		const int npwx) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::cal_ps_delta_spin(
		const int npol, 
		const int m) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::cal_ps_dftu(
		const int npol, 
		const int m) const
{}

#if ((defined __CUDA) || (defined __ROCM))
template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>::add_onsite_proj(
		std::complex<float> *hpsi_in, 
		const int npol, 
		const int m,
		const int npwx) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>::update_becp(
		const std::complex<float> *psi_in, 
		const int npol, 
		const int m,
		const int npwx) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>::cal_ps_delta_spin(
		const int npol, 
		const int m) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>::cal_ps_dftu(
		const int npol, 
		const int m) const
{}
#endif

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node)const
{
    ModuleBase::timer::start("Operator", "OnsiteProjPW");
    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    this->update_becp(tmpsi_in, npol, nbands, onsite_p->get_npwx());
    this->cal_ps_delta_spin(npol, nbands);
    this->cal_ps_dftu(npol, nbands);
    this->add_onsite_proj(tmhpsi, npol, nbands, onsite_p->get_npwx());
    ModuleBase::timer::end("Operator", "OnsiteProjPW");
}

template<typename T, typename Device>
template<typename T_in, typename Device_in>
hamilt::OnsiteProj<OperatorPW<T, Device>>::OnsiteProj(const OnsiteProj<OperatorPW<T_in, Device_in>> *nonlocal)
{
    this->classname = "OnsiteProj";
    this->cal_type = calculation_type::pw_nonlocal;
}

template class OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class OnsiteProj<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;

#if ((defined __CUDA) || (defined __ROCM))
template class OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class OnsiteProj<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
#endif
} // namespace hamilt
