#include "op_pw_proj.h"

#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/onsite_projector.h"
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
    ModuleBase::timer::tick("OnsiteProj", "getvnl");
    this->ik = ik_in;

    std::cout << "[DIAG-INIT] OnsiteProj::init ik=" << ik_in << std::endl;

    // DEBUG: dump first 5 elements of psi for this ik
    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    onsite_p->tabulate_atomic(ik_in);
    this->tnp = onsite_p->get_tot_nproj();

    if(this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }

    ModuleBase::timer::tick("OnsiteProj", "getvnl");
}

//--------------------------------------------------------------------------
// this function sum up each non-local pseudopotential located on each atom,
//--------------------------------------------------------------------------
template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::add_onsite_proj(T *hpsi_in, const int npol, const int m) const
{
    ModuleBase::timer::tick("OnsiteProj", "add_onsite_proj");

    // DIAGNOSTIC: output first 3 hpsi values before and after
    if(m == 28 && (this->ik == 0 || this->ik == 1))
    {
        std::cout << "[HPSI-PW] add_onsite_proj BEFORE ik=" << this->ik << " m=" << m << " hpsi[0..2]=";
        for(int i=0;i<3;i++) std::cout << " (" << hpsi_in[i].real() << "," << hpsi_in[i].imag() << ")";
        std::cout << std::endl;
    }

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    // apply the operator to the wavefunction
    //std::cout << "use of tab_atomic at " << __FILE__ << ": " << __LINE__ << std::endl;
    const std::complex<double>* tab_atomic = onsite_p->get_tab_atomic();
    const int npw = onsite_p->get_npw();
    const int npwx = onsite_p->get_npwx();

    // DIAG: print hpsi norms for first 5 bands
    if(m == 28 && (this->ik == 0 || this->ik == 1))
    {
        std::cout << "[HPSI-NORM-PW] ik=" << this->ik << " m=" << m << " bands_norm[0..4]=";
        for(int b=0; b<5 && b<m; ++b)
        {
            double norm_sq = 0.0;
            for(int i=0; i<npwx; ++i)
            {
                T val = hpsi_in[b * npwx + i];
                norm_sq += std::norm(val);
            }
            std::cout << " " << norm_sq;
        }
        std::cout << std::endl;
    }
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

    // DIAGNOSTIC: output first 3 hpsi values after
    if(m == 28 && (this->ik == 0 || this->ik == 1))
    {
        std::cout << "[HPSI-PW] add_onsite_proj AFTER ik=" << this->ik << " m=" << m << " hpsi[0..2]=";
        for(int i=0;i<3;i++) std::cout << " (" << hpsi_in[i].real() << "," << hpsi_in[i].imag() << ")";
        std::cout << std::endl;
    }

    ModuleBase::timer::tick("OnsiteProj", "add_onsite_proj");
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::update_becp(const T *psi_in, const int npol, const int m) const
{
    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    // calculate <alpha|psi> 
    // DIAGNOSTIC: print psi_in pointer address and first 3 values
    std::cout << "[DIAG-UB] update_becp ik=" << this->ik << " psi_in=" << (const void*)psi_in 
              << " nbands=" << m << " psi[0..2]=";
    for(int i=0;i<3;i++) std::cout << " (" << psi_in[i].real() << "," << psi_in[i].imag() << ")";
    std::cout << std::endl;
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

    if(npol == 2)
    {
        // npol==2: 4-element lambda per atom (2x2 spin matrix)
        std::vector<std::complex<double>> tmp_lambda_coeff(this->ucell->nat * 4);
        for(int iat=0;iat<this->ucell->nat;iat++)
        {
            tmp_lambda_coeff[iat * 4] = std::complex<double>(lambda[iat][2], 0.0);
            tmp_lambda_coeff[iat * 4 + 1] = std::complex<double>(lambda[iat][0], lambda[iat][1]);
            tmp_lambda_coeff[iat * 4 + 2] = std::complex<double>(lambda[iat][0], -1 * lambda[iat][1]);
            tmp_lambda_coeff[iat * 4 + 3] = std::complex<double>(-1 * lambda[iat][2], 0.0);
        }
        syncmem_complex_h2d_op()(this->lambda_coeff, tmp_lambda_coeff.data(), this->ucell->nat * 4);

        hamilt::onsite_ps_op<Real, Device>()(
            this->ctx,
            m,
            npol,
            this->ip_iat,
            tnp,
            this->lambda_coeff,
            this->ps, becp);
    }
    else // npol == 1, nspin=1 or nspin=2
    {
        // npol==1: 1-element lambda per atom (z-component scaled by spin sign)
        // For nspin=1: sign=1 for all k-points
        // For nspin=2: sign=1 for spin-up (isk=0), sign=-1 for spin-down (isk=1)
        const int sign = this->isk[this->ik] == 0 ? 1 : -1;
        std::vector<std::complex<double>> tmp_lambda_coeff(this->ucell->nat);
        for(int iat=0;iat<this->ucell->nat;iat++)
        {
            tmp_lambda_coeff[iat] = std::complex<double>(lambda[iat][2] * sign, 0.0);
        }
        syncmem_complex_h2d_op()(this->lambda_coeff, tmp_lambda_coeff.data(), this->ucell->nat);

        hamilt::onsite_ps_op<Real, Device>()(
            this->ctx,
            m,
            npol,
            this->ip_iat,
            tnp,
            this->lambda_coeff,
            this->ps, becp);
    }
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
                const int vu_fold = (npol == 2) ? 4 : 1;
                vu_begin += tlp1 * tlp1 * vu_fold;
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

    // For nspin=2 spin-down, select the second half of eff_pot_pw
    // For nspin=1 or nspin=2 spin-up, use the full array (vu_begin_iat offsets are relative to full array)
    if(PARAM.inp.nspin == 2 && this->isk[this->ik] == 1)
    {
        const int half_size = dftu->get_size_eff_pot_pw() / 2;
        syncmem_complex_h2d_op()(this->vu_device, dftu->get_eff_pot_pw(0) + half_size, half_size);
    }
    else
    {
        syncmem_complex_h2d_op()(this->vu_device, dftu->get_eff_pot_pw(0), dftu->get_size_eff_pot_pw());
    }

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

    /*
    int sum = 0;
    if (npol == 1)
    {
        const int current_spin = this->isk[this->ik];
    }
    else
    {
        for (int iat = 0; iat < this->ucell->nat; iat++)
        {
            const int it = this->ucell->iat2it[iat];
            const int target_l = dftu->orbital_corr[it];
            const int nproj = onsite_p->get_nh(iat);
            if(target_l == -1)
            {
                sum += nproj;
                continue;
            }
            const int ip_begin = target_l * target_l;
            const int ip_end = (target_l + 1) * (target_l + 1);
            const int tlp1 = 2 * target_l + 1;
            const int tlp1_2 = tlp1 * tlp1;
            const std::complex<double>* vu = dftu->get_eff_pot_pw(iat);
            // each projector (each atom) must multiply coefficient
            // with all the other projectors.
            for (int ib = 0; ib < m; ib+=2)
            {
                for (int ip2 = ip_begin; ip2 < ip_end; ip2++)
                {
                    const int psind = (sum + ip2) * m + ib;
                    const int m2 = ip2 - ip_begin;
                    for (int ip1 = ip_begin; ip1 < ip_end; ip1++)
                    {
                        const int becpind1 = ib * tnp + sum + ip1;
                        const int m1 = ip1 - ip_begin;
                        const int index_mm = m1 * tlp1 + m2;
                        const std::complex<double> becp1 = becp[becpind1];
                        const std::complex<double> becp2 = becp[becpind1 + tnp];
                        ps[psind] += vu[index_mm] * becp1
                                    + vu[index_mm + tlp1_2 * 2] * becp2;
                        ps[psind + 1] += vu[index_mm + tlp1_2 * 1] * becp1
                                    + vu[index_mm + tlp1_2 * 3] * becp2;
                    } // end ip1
                } // end ip2
            } // end ib
            sum += nproj;
        } // end iat
    }*/
}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::add_onsite_proj(
		std::complex<float> *hpsi_in, 
		const int npol, 
		const int m) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::update_becp(
		const std::complex<float> *psi_in, 
		const int npol, 
		const int m) const
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
		const int m) const
{}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>::update_becp(
		const std::complex<float> *psi_in, 
		const int npol, 
		const int m) const
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
    ModuleBase::timer::tick("Operator", "OnsiteProjPW");
    this->update_becp(tmpsi_in, npol, nbands);
    this->cal_ps_delta_spin(npol, nbands);

    // DIAGNOSTIC: dump becp and ps before cal_ps_dftu
    if(this->has_dftu)
    {
        auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
        const std::complex<double>* becp = onsite_p->get_h_becp();
        int nkb = onsite_p->get_tot_nproj();
        std::cout << "[DIAG-OP] OnsiteProj::act ik=" << this->ik << " npol=" << npol 
                  << " nbands=" << nbands << " tnp=" << this->tnp << std::endl;
        std::cout << "[DIAG-OP]   becp[0..4]=";
        for(int i=0;i<5;i++) std::cout << " (" << becp[i].real() << "," << becp[i].imag() << ")";
        std::cout << " | sum|becp|^2=";
        double sum2 = 0;
        for(int i=0;i<nkb;i++) sum2 += std::norm(becp[i]);
        std::cout << sum2 << std::endl;
    }

    this->cal_ps_dftu(npol, nbands);

    // DIAGNOSTIC: dump ps after cal_ps_dftu
    if(this->has_dftu)
    {
        std::cout << "[DIAG-OP]   ps[0..9]=";
        for(int i=0;i<10;i++) std::cout << " (" << this->ps[i].real() << "," << this->ps[i].imag() << ")";
        std::cout << std::endl;
    }

    this->add_onsite_proj(tmhpsi, npol, nbands);
    ModuleBase::timer::tick("Operator", "OnsiteProjPW");
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
