#include "nonlocal_pw.h"

#include "module_parameter/parameter.h"
#include "module_base/blas_connector.h"
#include "module_base/timer.h"
#include "module_base/parallel_reduce.h"
#include "module_base/tool_quit.h"
#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

#include <algorithm>

namespace hamilt {

template<typename T, typename Device>
Nonlocal<OperatorPW<T, Device>>::Nonlocal(const int* isk_in,
                                               const pseudopot_cell_vnl* ppcell_in,
                                               const UnitCell* ucell_in,
                                               const ModulePW::PW_Basis_K* wfc_basis)
{
    if( isk_in == nullptr || ppcell_in == nullptr || ucell_in == nullptr)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "Constuctor of Operator::NonlocalPW is failed, please check your code!");
    }
    this->classname = "Nonlocal";
    this->cal_type = calculation_type::pw_nonlocal;
    this->wfcpw = wfc_basis;
    this->isk = isk_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->deeq = this->ppcell->template get_deeq_data<Real>();
    this->deeq_nc = this->ppcell->template get_deeq_nc_data<Real>();
    this->vkb = this->ppcell->template get_vkb_data<Real>();

    // Initialize VKB batching if requested
    const int vkb_batch_atoms = PARAM.inp.vkb_batch_atoms;
    if (vkb_batch_atoms > 0
        && this->ppcell->nkb > 0 && ucell_in->nat > 0)
    {
        // Build nproj per atom array
        std::vector<int> nproj_per_atom(ucell_in->nat);
        for (int iat = 0; iat < ucell_in->nat; iat++)
        {
            const int it = ucell_in->iat2it[iat];
            nproj_per_atom[iat] = ucell_in->atoms[it].ncpp.nh;
        }

        const int npwx = this->ppcell->npwx;

        // Use 25% of a conservative 4GB estimate for auto-detection
        const size_t gpu_mem_budget = 4ULL * 1024 * 1024 * 1024;

        vkb_manager_.init(ucell_in->nat, nproj_per_atom.data(), npwx,
                          gpu_mem_budget, vkb_batch_atoms);

        // Only enable batching if it actually creates multiple batches
        if (vkb_manager_.get_nbatch() > 1)
        {
            use_vkb_batching_ = true;

            // Allocate CPU buffer for full VKB
            const int nkb = this->ppcell->nkb;
            vkb_cpu_ = new T[static_cast<size_t>(nkb) * npwx]();

            // Note: this->vkb points to ppcell's VKB buffer (allocated by ppcell)
            // We keep it on GPU for force/stress calculations

            // Allocate GPU buffer for largest batch
            resmem_complex_op()(this->ctx, vkb_batch_gpu_,
                               vkb_manager_.get_max_batch_elements(), "Nonlocal::vkb_batch");
        }
    }
}

template<typename T, typename Device>
Nonlocal<OperatorPW<T, Device>>::~Nonlocal() {
    delmem_complex_op()(this->ctx, this->ps);
    delmem_complex_op()(this->ctx, this->becp);

    // Cleanup VKB batching resources
    if (becp_batch_ != nullptr)
    {
        delmem_complex_op()(this->ctx, becp_batch_);
        becp_batch_ = nullptr;
    }
    if (ps_batch_ != nullptr)
    {
        delmem_complex_op()(this->ctx, ps_batch_);
        ps_batch_ = nullptr;
    }
    if (vkb_cpu_ != nullptr)
    {
        delete[] vkb_cpu_;
        vkb_cpu_ = nullptr;
    }
    if (vkb_batch_gpu_ != nullptr)
    {
        delmem_complex_op()(this->ctx, vkb_batch_gpu_);
        vkb_batch_gpu_ = nullptr;
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::init(const int ik_in)
{
    ModuleBase::timer::tick("Nonlocal", "getvnl");
    this->ik = ik_in;
    // Calculate nonlocal pseudopotential vkb
    if(this->ppcell->nkb > 0)
    {
        if (use_vkb_batching_)
        {
            // Compute full VKB on GPU and keep it for force/stress calculations
            // Also copy to CPU for batched transfer in act()
            const int nkb = this->ppcell->nkb;
            const int npwx = this->ppcell->npwx;

            // Compute on GPU (this->vkb already points to ppcell's VKB buffer)
            this->ppcell->getvnl(this->ctx, *this->ucell, this->ik, this->vkb);

            // Copy from GPU to CPU for batched operations
            using syncmem_d2h_op = base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, Device>;
            syncmem_d2h_op()(this->cpu_ctx, this->ctx,
                            this->vkb_cpu_, this->vkb,
                            static_cast<size_t>(nkb) * npwx);

            // Keep GPU VKB for force/stress calculations (don't free it)
        }
        else
        {
            // Original: compute full VKB on device (GPU or CPU)
            this->ppcell->getvnl(this->ctx, *this->ucell, this->ik, this->vkb);
        }
    }

    if(this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }

    ModuleBase::timer::tick("Nonlocal", "getvnl");
}

//--------------------------------------------------------------------------
// this function sum up each non-local pseudopotential located on each atom,
//--------------------------------------------------------------------------
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp(T *hpsi_in, const T *becp, const int m) const
{
    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp");

    // number of projectors
    int nkb = this->ppcell->nkb;

    // T *ps = new T[nkb * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * nkb);
    if (this->nkb_m < m * nkb) {
        resmem_complex_op()(this->ctx, this->ps, nkb * m, "Nonlocal<PW>::ps");
        this->nkb_m = m * nkb;
    }
    setmem_complex_op()(this->ctx, this->ps, 0, nkb * m);

    int sum = 0;
    int iat = 0;
    if (this->npol == 1)
    {
        const int current_spin = this->isk[this->ik];
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            // denghui replace 2022-10-20
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, current_spin, nkb,   // additional index params
                this->ppcell->deeq.getBound2(), this->ppcell->deeq.getBound3(), this->ppcell->deeq.getBound4(), // realArray operator()
                this->deeq, // array of data
                this->ps, this->becp); //  array of data
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            //                     += this->ppcell->deeq(current_spin, iat, ip, ip2) * this->becp[ib * nkb + sum + ip];
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
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            // added by denghui at 20221109
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, nkb,   // additional index params
                this->ppcell->deeq_nc.getBound2(), this->ppcell->deeq_nc.getBound3(), this->ppcell->deeq_nc.getBound4(), // realArray operator()
                this->deeq_nc, // array of data
                this->ps, this->becp); //  array of data
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            // {
            //     // each atom has nproj, means this is with structure factor;
            //     // each projector (each atom) must multiply coefficient
            //     // with all the other projectors.
            //     for (int ib = 0; ib < m; ib+=2)
            //     {
            //         for (int ip2 = 0; ip2 < nproj; ip2++)
            //         {
            //             for (int ip = 0; ip < nproj; ip++)
            //             {
            //                 psind = (sum + ip2) * m + ib;
            //                 becpind = ib * nkb + sum + ip;
            //                 becp1 = becp[becpind];
            //                 becp2 = becp[becpind + nkb];
            //                 ps[psind] += this->ppcell->deeq_nc(0, iat, ip2, ip) * becp1
            //                              + this->ppcell->deeq_nc(1, iat, ip2, ip) * becp2;
            //                 ps[psind + 1] += this->ppcell->deeq_nc(2, iat, ip2, ip) * becp1
            //                                  + this->ppcell->deeq_nc(3, iat, ip2, ip) * becp2;
            //             } // end ib
            //         } // end ih
            //     } // end jh
            //     sum += nproj;
            //     ++iat;
            // } // end na
        } // end nt
    }

    // use simple method.
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // qianrui optimize 2021-3-31
    char transa = 'N';
    char transb = 'T';
    if (m == 1)
    {
        int inc = 1;
        // denghui replace 2022-10-20
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        gemv_op()(
            this->ctx,
            transa,
            this->npw,
            this->ppcell->nkb,
            &this->one,
            this->vkb,
            this->ppcell->npwx,
            this->ps,
            inc,
            &this->one,
            hpsi_in,
            inc);
    }
    else
    {
        int npm = m;
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // denghui replace 2022-10-20
        gemm_op()(
            this->ctx,
            transa,
            transb,
            this->npw,
            npm,
            this->ppcell->nkb,
            &this->one,
            this->vkb,
            this->ppcell->npwx,
            this->ps,
            npm,
            &this->one,
            hpsi_in,
            this->max_npw
        );
    }
    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node)const
{
    ModuleBase::timer::tick("Operator", "NonlocalPW");
    if(is_first_node)
    {
        setmem_complex_op()(this->ctx, tmhpsi, 0, nbasis*nbands/npol);
    }
    if(!PARAM.inp.use_paw)
    {
        this->npw = ngk_ik;
        this->max_npw = nbasis / npol;
        this->npol = npol;

        if (this->ppcell->nkb > 0)
        {
            if (use_vkb_batching_)
            {
                // ===== BATCHED PATH =====
                this->act_batched(nbands, nbasis, npol, tmpsi_in, tmhpsi);
            }
            else
            {
                // ===== ORIGINAL PATH =====
                //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                // qianrui optimize 2021-3-31
                int nkb = this->ppcell->nkb;
                if (this->nkb_m < nbands * nkb) {
                    resmem_complex_op()(this->ctx, this->becp, nbands * nkb, "Nonlocal<PW>::becp");
                }
                // ModuleBase::ComplexMatrix becp(nbands, nkb, false);
                char transa = 'C';
                char transb = 'N';
                if (nbands == 1)
                {
                    int inc = 1;
                    gemv_op()(
                        this->ctx,
                        transa,
                        this->npw,
                        nkb,
                        &this->one,
                        this->vkb,
                        this->ppcell->npwx,
                        tmpsi_in,
                        inc,
                        &this->zero,
                        this->becp,
                        inc);
                }
                else
                {
                    int npm = nbands;
                    gemm_op()(
                        this->ctx,
                        transa,
                        transb,
                        nkb,
                        npm,
                        this->npw,
                        &this->one,
                        this->vkb,
                        this->ppcell->npwx,
                        tmpsi_in,
                        max_npw,
                        &this->zero,
                        this->becp,
                        nkb
                    );
                }

                Parallel_Reduce::reduce_pool(becp, nkb * nbands);

                this->add_nonlocal_pp(tmhpsi, becp, nbands);
            }
        }
    }
    else
    {
#ifdef USE_PAW
        this->npw = ngk_ik;
        this->max_npw = nbasis / npol;
        this->npol = npol;
        std::complex<double> *vnlpsi;
        vnlpsi = new std::complex<double> [npw];
        for(int ibands = 0; ibands < nbands; ibands++)
        {
            GlobalC::paw_cell.paw_nl_psi(0,reinterpret_cast<const std::complex<double>*> (&tmpsi_in[ibands*max_npw]),vnlpsi);
            for(int i = 0; i < npw; i++)
            {
                tmhpsi[ibands*max_npw+i] += vnlpsi[i];
            }
        }
        delete[] vnlpsi;
#endif
    }
    ModuleBase::timer::tick("Operator", "NonlocalPW");
}

//--------------------------------------------------------------------------
// Batched VKB path: compute VKB projectors batch-by-batch on GPU
// to reduce GPU memory from nkb*npwx to max_nkb_batch*npwx
//--------------------------------------------------------------------------
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act_batched(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi) const
{
    ModuleBase::timer::tick("Nonlocal", "act_batched");

    const int npwx = this->ppcell->npwx;
    const int max_nkb_batch = vkb_manager_.get_max_nkb_batch();

    // Lazy pre-allocation: allocate once at max_nkb_batch * nbands, reuse across batches and calls.
    // Only reallocate if nbands changes (unlikely but handled).
    const size_t needed = static_cast<size_t>(max_nkb_batch) * nbands;
    if (needed > this->batch_alloc_size_)
    {
        if (this->becp_batch_ != nullptr)
        {
            delmem_complex_op()(this->ctx, this->becp_batch_);
        }
        if (this->ps_batch_ != nullptr)
        {
            delmem_complex_op()(this->ctx, this->ps_batch_);
        }
        resmem_complex_op()(this->ctx, this->becp_batch_, needed, "Nonlocal::becp_batch");
        resmem_complex_op()(this->ctx, this->ps_batch_, needed, "Nonlocal::ps_batch");
        this->batch_alloc_size_ = needed;
    }

    for (int ibatch = 0; ibatch < vkb_manager_.get_nbatch(); ibatch++)
    {
        int atom_start = 0, atom_end = 0, nkb_batch = 0;
        vkb_manager_.get_batch_info(ibatch, atom_start, atom_end, nkb_batch);
        const int jkb_offset = vkb_manager_.get_jkb_offset(ibatch);

        if (nkb_batch == 0)
        {
            continue;
        }

        // 1. Copy batch rows from CPU VKB to GPU buffer
        syncmem_complex_h2d_op()(this->ctx, this->cpu_ctx,
            vkb_batch_gpu_,
            vkb_cpu_ + static_cast<size_t>(jkb_offset) * npwx,
            static_cast<size_t>(nkb_batch) * npwx);

        // 2. Zero-fill batch temporaries (pre-allocated above the loop)
        setmem_complex_op()(this->ctx, this->ps_batch_, 0, nkb_batch * nbands);

        // 3. becp_batch = vkb_batch^H * psi
        char transa = 'C';
        char transb = 'N';
        if (nbands == 1)
        {
            int inc = 1;
            gemv_op()(this->ctx, transa, this->npw, nkb_batch,
                     &this->one, vkb_batch_gpu_, npwx,
                     tmpsi_in, inc,
                     &this->zero, this->becp_batch_, inc);
        }
        else
        {
            gemm_op()(this->ctx, transa, transb,
                     nkb_batch, nbands, this->npw,
                     &this->one, vkb_batch_gpu_, npwx,
                     tmpsi_in, this->max_npw,
                     &this->zero, this->becp_batch_, nkb_batch);
        }

        Parallel_Reduce::reduce_pool(this->becp_batch_, nkb_batch * nbands);

        // 4. ps_batch = D * becp_batch (apply nonlocal D coefficients)
        //    Iterate over atom types, but only process atoms in [atom_start, atom_end)
        int sum_batch = 0;
        int iat_scan = 0; // global atom index scanner
        if (this->npol == 1)
        {
            const int current_spin = this->isk[this->ik];
            for (int it = 0; it < this->ucell->ntype; it++)
            {
                const int na_type = this->ucell->atoms[it].na;
                const int nproj = this->ucell->atoms[it].ncpp.nh;
                // Determine overlap of this type's atoms with the batch range
                const int type_start = iat_scan;
                const int type_end = iat_scan + na_type;
                const int batch_type_start = std::max(type_start, atom_start);
                const int batch_type_end = std::min(type_end, atom_end);
                if (batch_type_start < batch_type_end)
                {
                    const int na_in_batch = batch_type_end - batch_type_start;
                    // iat_batch must start at batch_type_start for correct deeq indexing
                    int iat_batch = batch_type_start;
                    nonlocal_op()(
                        this->ctx,
                        na_in_batch, nbands, nproj,
                        sum_batch, iat_batch, current_spin, nkb_batch,
                        this->ppcell->deeq.getBound2(),
                        this->ppcell->deeq.getBound3(),
                        this->ppcell->deeq.getBound4(),
                        this->deeq,
                        this->ps_batch_, this->becp_batch_);
                    // sum_batch is updated by nonlocal_op() via reference
                }
                iat_scan += na_type;
            }
        }
        else
        {
            // Non-collinear case (npol == 2)
            for (int it = 0; it < this->ucell->ntype; it++)
            {
                const int na_type = this->ucell->atoms[it].na;
                const int nproj = this->ucell->atoms[it].ncpp.nh;
                const int type_start = iat_scan;
                const int type_end = iat_scan + na_type;
                const int batch_type_start = std::max(type_start, atom_start);
                const int batch_type_end = std::min(type_end, atom_end);
                if (batch_type_start < batch_type_end)
                {
                    const int na_in_batch = batch_type_end - batch_type_start;
                    // iat_batch must start at batch_type_start for correct deeq indexing
                    int iat_batch = batch_type_start;
                    nonlocal_op()(
                        this->ctx,
                        na_in_batch, nbands, nproj,
                        sum_batch, iat_batch, nkb_batch,
                        this->ppcell->deeq_nc.getBound2(),
                        this->ppcell->deeq_nc.getBound3(),
                        this->ppcell->deeq_nc.getBound4(),
                        this->deeq_nc,
                        this->ps_batch_, this->becp_batch_);
                    // sum_batch is updated by nonlocal_op() via reference
                }
                iat_scan += na_type;
            }
        }

        // 5. hpsi += vkb_batch * ps_batch (accumulate to output)
        if (nbands == 1)
        {
            int inc = 1;
            gemv_op()(this->ctx, 'N', this->npw, nkb_batch,
                     &this->one, vkb_batch_gpu_, npwx,
                     this->ps_batch_, inc,
                     &this->one, tmhpsi, inc);
        }
        else
        {
            gemm_op()(this->ctx, 'N', 'T',
                     this->npw, nbands, nkb_batch,
                     &this->one, vkb_batch_gpu_, npwx,
                     this->ps_batch_, nbands,
                     &this->one, tmhpsi, this->max_npw);
        }
    }

    ModuleBase::timer::tick("Nonlocal", "act_batched");
}

template<typename T, typename Device>
template<typename T_in, typename Device_in>
hamilt::Nonlocal<OperatorPW<T, Device>>::Nonlocal(const Nonlocal<OperatorPW<T_in, Device_in>> *nonlocal)
{
    this->classname = "Nonlocal";
    this->cal_type = calculation_type::pw_nonlocal;
    this->ik = nonlocal->get_ik();
    this->isk = nonlocal->get_isk();
    this->ppcell = nonlocal->get_ppcell();
    this->ucell = nonlocal->get_ucell();
    this->deeq = this->ppcell->d_deeq;
    this->deeq_nc = this->ppcell->template get_deeq_nc_data<Real>();
    this->vkb = this->ppcell->template get_vkb_data<Real>();
    if( this->isk == nullptr || this->ppcell == nullptr || this->ucell == nullptr)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "Constuctor of Operator::NonlocalPW is failed, please check your code!");
    }

    // Initialize VKB batching (same logic as primary constructor)
    const int vkb_batch_atoms = PARAM.inp.vkb_batch_atoms;
    if (vkb_batch_atoms > 0
        && this->ppcell->nkb > 0 && this->ucell->nat > 0)
    {
        // Build nproj per atom array
        std::vector<int> nproj_per_atom(this->ucell->nat);
        for (int iat = 0; iat < this->ucell->nat; iat++)
        {
            const int it = this->ucell->iat2it[iat];
            nproj_per_atom[iat] = this->ucell->atoms[it].ncpp.nh;
        }

        const int npwx = this->ppcell->npwx;

        // Use 25% of a conservative 4GB estimate for auto-detection
        const size_t gpu_mem_budget = 4ULL * 1024 * 1024 * 1024;

        vkb_manager_.init(this->ucell->nat, nproj_per_atom.data(), npwx,
                          gpu_mem_budget, vkb_batch_atoms);

        // Only enable batching if it actually creates multiple batches
        if (vkb_manager_.get_nbatch() > 1)
        {
            use_vkb_batching_ = true;

            // Allocate CPU buffer for full VKB
            const int nkb = this->ppcell->nkb;
            vkb_cpu_ = new T[static_cast<size_t>(nkb) * npwx]();

            // Note: this->vkb will be allocated temporarily in init() for getvnl,
            // then freed immediately after copying to CPU

            // Allocate GPU buffer for largest batch
            resmem_complex_op()(this->ctx, vkb_batch_gpu_,
                               vkb_manager_.get_max_batch_elements(), "Nonlocal::vkb_batch");
        }
    }
}

template class Nonlocal<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;
// template Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *nonlocal);
#if ((defined __CUDA) || (defined __ROCM))
template class Nonlocal<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
// template Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *nonlocal); template
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *nonlocal); template
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *nonlocal);
#endif
} // namespace hamilt