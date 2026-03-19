// GPU implementation of v_xc: LDA + GGA XC potential on device
// LDA part computed by GPU kernels, GGA gradcorr also on GPU

#include "xc_functional.h"
#include "xc_functional_gpu.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_parameter/parameter.h"
#include "module_base/module_device/memory_op.h"

#if defined(__CUDA) || defined(__ROCM)

std::tuple<double,double,ModuleBase::matrix> XC_Functional::v_xc_gpu(
    const int &nrxx,
    const Charge* const chr,
    const UnitCell *ucell)
{
    ModuleBase::TITLE("XC_Functional", "v_xc_gpu");
    ModuleBase::timer::tick("XC_Functional", "v_xc_gpu");

    // If libxc is requested, fall back to CPU
    if (use_libxc)
    {
        ModuleBase::timer::tick("XC_Functional", "v_xc_gpu");
        return v_xc(nrxx, chr, ucell);
    }

    double etxc = 0.0;
    double vtxc = 0.0;
    const int nspin = PARAM.inp.nspin;
    ModuleBase::matrix v(nspin, nrxx);

    using resmem_d = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_d = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_d2h = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
    using syncmem_h2d = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;

    // Upload rho and rho_core to GPU
    double* d_rho[4] = {nullptr};
    double* d_rho_core = nullptr;
    double* d_vxc[4] = {nullptr};

    int nspin_rho = (nspin == 4) ? 4 : nspin;
    for (int is = 0; is < nspin_rho; ++is)
    {
        resmem_d()(gpu_ctx, d_rho[is], nrxx);
        syncmem_h2d()(gpu_ctx, cpu_ctx, d_rho[is], chr->rho[is], nrxx);
        resmem_d()(gpu_ctx, d_vxc[is], nrxx);
    }
    resmem_d()(gpu_ctx, d_rho_core, nrxx);
    syncmem_h2d()(gpu_ctx, cpu_ctx, d_rho_core, chr->rho_core, nrxx);

    const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
    const std::vector<int> fids = XC_Functional::get_func_id();

    // Step 1: LDA part on GPU
    if (nspin == 1 || (nspin == 4 && !PARAM.globalv.domag && !PARAM.globalv.domag_z))
    {
        XC_GPU::v_xc_lda_gpu_nspin1(nrxx, d_rho[0], d_rho_core, d_vxc[0],
                                      etxc, vtxc, fids, hybrid_alpha);
    }
    else if (nspin == 2)
    {
        XC_GPU::v_xc_lda_gpu_nspin2(nrxx, d_rho[0], d_rho[1], d_rho_core,
                                      d_vxc[0], d_vxc[1],
                                      etxc, vtxc, fids, hybrid_alpha);
    }
    else if (nspin == 4)
    {
        XC_GPU::v_xc_lda_gpu_nspin4(nrxx, d_rho[0], d_rho[1], d_rho[2], d_rho[3],
                                      d_rho_core,
                                      d_vxc[0], d_vxc[1], d_vxc[2], d_vxc[3],
                                      etxc, vtxc, fids, hybrid_alpha);
    }

    // Step 2: GGA gradient correction on GPU
    // gradcorr_gpu accumulates into d_vxc, etxc, vtxc
    const int ft = XC_Functional::get_func_type();
    if (ft >= 2) // GGA or hybrid GGA
    {
        int nspin0 = nspin;
        if (nspin == 4) nspin0 = 1;
        if (nspin == 4 && (PARAM.globalv.domag || PARAM.globalv.domag_z)) nspin0 = 2;

        // For nspin=4 with magnetism, gradcorr_gpu needs the noncolin_rho treatment
        // which is complex. For now, only support nspin=1 and nspin=2 on GPU.
        // nspin=4 with domag falls back to CPU gradcorr.
        if (nspin != 4 || (!PARAM.globalv.domag && !PARAM.globalv.domag_z))
        {
            XC_GPU::gradcorr_gpu(nrxx, chr->rhopw->npw,
                                  nspin0, nspin,
                                  d_rho, d_rho_core,
                                  d_vxc,
                                  etxc, vtxc,
                                  fids, hybrid_alpha, ft,
                                  chr->rhopw, ucell->tpiba,
                                  PARAM.globalv.domag, PARAM.globalv.domag_z,
                                  nullptr, false);
        }
        else
        {
            // nspin=4 with magnetism: copy vxc back, run CPU gradcorr, copy back
            for (int is = 0; is < nspin_rho; ++is)
                syncmem_d2h()(cpu_ctx, gpu_ctx, &v(is, 0), d_vxc[is], nrxx);

            std::vector<double> dum;
            gradcorr(etxc, vtxc, v, chr, chr->rhopw, ucell, dum);

            // Copy updated v back to GPU vxc arrays
            for (int is = 0; is < nspin_rho; ++is)
                syncmem_h2d()(gpu_ctx, cpu_ctx, d_vxc[is], &v(is, 0), nrxx);
        }
    }

    // Step 3: Copy results back to CPU matrix
    for (int is = 0; is < nspin_rho; ++is)
    {
        syncmem_d2h()(cpu_ctx, gpu_ctx, &v(is, 0), d_vxc[is], nrxx);
    }

    // Free GPU memory
    for (int is = 0; is < nspin_rho; ++is)
    {
        delmem_d()(gpu_ctx, d_rho[is]);
        delmem_d()(gpu_ctx, d_vxc[is]);
    }
    delmem_d()(gpu_ctx, d_rho_core);

    // MPI reduction
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= ucell->omega / chr->rhopw->nxyz;
    vtxc *= ucell->omega / chr->rhopw->nxyz;

    ModuleBase::timer::tick("XC_Functional", "v_xc_gpu");
    return std::make_tuple(etxc, vtxc, std::move(v));
}

#endif // __CUDA || __ROCM
