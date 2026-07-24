#include "forces.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_pw/module_pwdft/kernels/force_op.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_pw/module_pwdft/deltap_pw.h"
#include "source_base/constants.h"
#include <iomanip>
#include <iostream>

template <typename FPTYPE, typename Device>
void Forces<FPTYPE, Device>::cal_force_onsite(ModuleBase::matrix& force_onsite,
                                          const ModuleBase::matrix& wg,
                                          const ModulePW::PW_Basis_K* wfc_basis,
										  const UnitCell& ucell_in,
										  const Plus_U &dftu,
										  const psi::Psi <std::complex<FPTYPE>, Device>* psi_in)
{
    ModuleBase::TITLE("Forces", "cal_force_onsite");
    if(psi_in == nullptr || wfc_basis == nullptr)
    {
        return;
    }
    ModuleBase::timer::start("Forces", "cal_force_onsite");

    FPTYPE* force = nullptr;
    resmem_var_op()(force, ucell_in.nat * 3);
    base_device::memory::set_memory_op<FPTYPE, Device>()(force, 0.0, ucell_in.nat * 3);

    auto* onsite_p = projectors::OnsiteProjector<FPTYPE, Device>::get_instance();

    const int nks = wfc_basis->nks;
    for (int ik = 0; ik < nks; ik++)
    {
        int nbands_occ = wg.nc;
        while (wg(ik, nbands_occ - 1) == 0.0)
        {
            nbands_occ--;
            if (nbands_occ == 0)
            {
                break;
            }
        }
        const int npm = nbands_occ;
        onsite_p->get_fs_tools()->cal_becp(ik, npm);
        for (int ipol = 0; ipol < 3; ipol++)
        {
            onsite_p->get_fs_tools()->cal_dbecp_f(ik, npm, ipol);
        }
        if(PARAM.inp.dft_plus_u)
        {
            onsite_p->cal_force_onsite_dftu(ik, npm, force, dftu, nks, wg.c);
        }
        if(PARAM.inp.sc_mag_switch)
        {
            spinconstrain::SpinConstrain<std::complex<double>>& sc = 
              spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();
            onsite_p->cal_force_onsite_dspin(ik, npm, force, sc.get_sc_lambda().data(), wg.c);
        }
        if(PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
        {
            const auto& dp_lambda = pw_deltap::get_deltap_pw_lambda();
            const auto& dp_constrain = pw_deltap::get_deltap_pw_constrain();
            int nat = ucell_in.nat;
            std::vector<ModuleBase::Vector3<double>> lam(nat, ModuleBase::Vector3<double>(0,0,0));
            for (int iat = 0; iat < nat; iat++)
            {
                bool ok = (dp_constrain.empty() || static_cast<size_t>(iat) >= dp_constrain.size() || dp_constrain[iat] != 0);
                lam[iat].z = ok ? dp_lambda[iat] : 0.0;
            }

            // Allocate temp force buffer for DeltaP contribution only
            FPTYPE* f_deltap = nullptr;
            resmem_var_op()(f_deltap, nat * 3);
            base_device::memory::set_memory_op<FPTYPE, Device>()(f_deltap, 0.0, nat * 3);
            onsite_p->cal_force_onsite_dspin(ik, npm, f_deltap, lam.data(), wg.c);

            // Print DeltaP force contribution separately for verification
            std::cout << " [DeltaP-F] λ=(";
            for (int iat = 0; iat < nat; iat++)
            {
                double val = (dp_constrain.empty() || static_cast<size_t>(iat) >= dp_constrain.size() || dp_constrain[iat] != 0) ? dp_lambda[iat] : 0.0;
                if (iat > 0) std::cout << ", ";
                std::cout << std::scientific << std::setprecision(3) << val;
            }
            std::cout << ") Ry";
            for (int iat = 0; iat < nat; iat++)
            {
                std::cout << " F" << iat+1 << "=(";
                for (int ipol = 0; ipol < 3; ipol++)
                {
                    if (ipol > 0) std::cout << ", ";
                    double f_eV_A = f_deltap[iat * 3 + ipol] * ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A;
                    std::cout << std::fixed << std::setprecision(6) << f_eV_A;
                }
                std::cout << ")";
            }
            std::cout << " eV/A" << std::endl;

            // Accumulate into total force
            for (int iat = 0; iat < nat; iat++)
                for (int ipol = 0; ipol < 3; ipol++)
                    force[iat * 3 + ipol] += f_deltap[iat * 3 + ipol];
            delmem_var_op()(f_deltap);
        }
        
    }

    syncmem_var_d2h_op()(force_onsite.c, force, force_onsite.nr * force_onsite.nc);
    delmem_var_op()(force);
    Parallel_Reduce::reduce_all(force_onsite.c, force_onsite.nr * force_onsite.nc);

    ModuleBase::timer::end("Forces", "cal_force_onsite");
}

template class Forces<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Forces<double, base_device::DEVICE_GPU>;
#endif
