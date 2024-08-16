#include "onsite_proj_pw.h"

#include "module_base/blas_connector.h"
#include "module_base/timer.h"
#include "module_base/parallel_reduce.h"
#include "module_base/tool_quit.h"
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"
#include "module_hamilt_pw/hamilt_pwdft/onsite_projector.h"
#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

namespace hamilt {

template<typename T, typename Device>
OnsiteProj<OperatorPW<T, Device>>::OnsiteProj(const int* isk_in,
                                               const UnitCell* ucell_in)
{
    this->classname = "OnsiteProj";
    this->cal_type = calculation_type::pw_onsite;
    this->isk = isk_in;
    this->ucell = ucell_in;
}

template<typename T, typename Device>
OnsiteProj<OperatorPW<T, Device>>::~OnsiteProj() {
    delmem_complex_op()(this->ctx, this->ps);
}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::init(const int ik_in)
{
    ModuleBase::timer::tick("OnsiteProj", "getvnl");
    this->ik = ik_in;

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    onsite_p->init_k(GlobalV::NQX, 
                           GlobalV::DQ, 
                           ik_in);
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
void OnsiteProj<OperatorPW<T, Device>>::add_onsite_proj(T *hpsi_in, const T *psi_in, const int npol, const int m) const
{
    ModuleBase::timer::tick("OnsiteProj", "add_onsite_proj");

    auto* onsite_p = projectors::OnsiteProjector<double, Device>::get_instance();
    // calculate <alpha|psi> 
    onsite_p->overlap_proj_psi(m, psi_in);
    const std::complex<double>* becp = onsite_p->get_becp();
    const std::complex<double>* tab_atomic = onsite_p->get_tab_atomic();
    const int npw = onsite_p->get_npw();
    const int npwx = onsite_p->get_npwx();

    SpinConstrain<T, base_device::DEVICE_CPU>& sc = SpinConstrain<T, base_device::DEVICE_CPU>::getScInstance();
    auto& constrain = sc.get_constrain();
    auto& lambda = sc.get_sc_lambda();

    // T *ps = new T[tnp * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * tnp);
    if (this->nkb_m < m * tnp) {
        resmem_complex_op()(this->ctx, this->ps, tnp * m, "OnsiteProj<PW>::ps");
        this->nkb_m = m * tnp;
    }
    setmem_complex_op()(this->ctx, this->ps, 0, tnp * m);

    int sum = 0;
    int iat = 0;
    if (npol == 1)
    {
        const int current_spin = this->isk[this->ik];
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            // for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            // {
            //     // each atom has nproj, means this is with structure factor;
            //     // each projector (each atom) must multiply coefficient
            //     // with all the other projectors.
            //     for (int ib = 0; ib < m; ++ib)
            //     {
            //         for (int ip2 = 0; ip2 < nproj; ip2++)
            //         {
            //             for (int ip = 0; ip < nproj; ip++)
            //             {
            //                 this->ps[(sum + ip2) * m + ib]
            //                     += this->ppcell->deeq(current_spin, iat, ip, ip2) * this->becp[ib * tnp + sum + ip];
            //             } // end ib
            //         } // end ih
            //     } // end jh
            //     sum += nproj;
            //     ++iat;
            // } // end na
        } // end nt
    }
    else
    {
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->tnp / this->ucell->atoms[it].na;
            for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            {
                if(constrain[iat].x == 0 && constrain[iat].y == 0 && constrain[iat].z == 0)
                {
                    sum += nproj;
                    ++iat;
                    continue;
                }
                const std::complex<double> coefficients0(lambda[iat][2], 0.0);
                const std::complex<double> coefficients1(lambda[iat][0] , lambda[iat][1]);
                const std::complex<double> coefficients2(lambda[iat][0] , -1 * lambda[iat][1]);
                const std::complex<double> coefficients3(-1 * lambda[iat][2], 0.0);
                // each atom has nproj, means this is with structure factor;
                // each projector (each atom) must multiply coefficient
                // with all the other projectors.
                for (int ib = 0; ib < m; ib+=2)
                {
                    for (int ip = 0; ip < nproj; ip++)
                    {
                        const int psind = (sum + ip) * m + ib;
                        const int becpind = ib * tnp + sum + ip;
                        const std::complex<double> becp1 = becp[becpind];
                        const std::complex<double> becp2 = becp[becpind + tnp];
                        ps[psind] += coefficients0 * becp1
                                        + coefficients2 * becp2;
                        ps[psind + 1] += coefficients1 * becp1
                                            + coefficients3 * becp2;
                    } // end ip
                } // end ib
                sum += nproj;
                ++iat;
            } // end na
        } // end nt
    }
    // qianrui optimize 2021-3-31
    char transa = 'N';
    char transb = 'T';
    int npm = m;
    gemm_op()(
        this->ctx,
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
    ModuleBase::timer::tick("OnsiteProj", "add_onsite_proj");
}

template<>
void OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>::add_onsite_proj(std::complex<float> *hpsi_in, const std::complex<float> *psi_in, const int npol, const int m) const
{}

template<typename T, typename Device>
void OnsiteProj<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik)const
{
    ModuleBase::timer::tick("Operator", "OnsiteProjPW");

    this->add_onsite_proj(tmhpsi, tmpsi_in, npol, nbands);

    ModuleBase::timer::tick("Operator", "OnsiteProjPW");
}

template<typename T, typename Device>
template<typename T_in, typename Device_in>
hamilt::OnsiteProj<OperatorPW<T, Device>>::OnsiteProj(const OnsiteProj<OperatorPW<T_in, Device_in>> *nonlocal)
{
    this->classname = "OnsiteProj";
    this->cal_type = calculation_type::pw_nonlocal;
    // FIXME: 
}

template class OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class OnsiteProj<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;

#if ((defined __CUDA) || (defined __ROCM))
template class OnsiteProj<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class OnsiteProj<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
#endif
} // namespace hamilt