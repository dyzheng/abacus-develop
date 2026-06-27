#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#ifdef __MPI
#include "source_base/parallel_comm.h"
#endif
#include <cmath>
#include <set>

namespace deltap {

void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv,
                  const TwoCenterIntegrator* intor, const TwoCenterIntegrator* overlap_intor,
                  const std::vector<double>& orb_cutoff,
                  double rm, int gdir, const Parallel_Orbitals* paraV,
                  cal_r_overlap_R* r_overlap)
{
    intor_ = intor;
    overlap_intor_ = overlap_intor;
    r_overlap_ = r_overlap;
    orb_cutoff_ = orb_cutoff;
    rm_ = rm;
    gdir_ = gdir;
    nat_ = ucell.nat;
    gd_ = &gd;
    kv_ = &kv;
    paraV_ = paraV;
    ModuleBase::TITLE("DeltaP", "init");
}

void DeltaP::compute_atomic_polarization(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi, const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DeltaP", "compute_atomic_polarization");
    ModuleBase::timer::start("DeltaP", "compute_atomic_polarization");

    if (PARAM.inp.deltap_method == "wannier")
    {
        compute_wannier_polarization(ucell, psi, pelec);
        ModuleBase::timer::end("DeltaP", "compute_atomic_polarization");
        return;
    }

    std::cout << "\n * * * * * *\n << Start DeltaP atomic polarization decomposition\n";

    // Step 1: compute real-space overlaps
    compute_real_overlaps(ucell, *gd_);

    // Step 2: setup k-string
    setup_kstring(*kv_);

    // Step 3: Allocate kstring_data_ and populate kvec_d + S/dS + D_I for all k on string
    const int nks = psi->get_nk();
    const int nbands = psi->get_nbands();
    const int nrow_local = paraV_->get_row_size();

    kstring_data_.resize(nppstr_);

    // First pass: compute S, dS, and D_I for all k-points on the first string
    for (int j = 0; j < nppstr_; j++)
    {
        int ik_psi = k_index_[0][j];  // first string only for now
        if (ik_psi >= nks) continue;

        // Set kvec_d from K_Vectors
        kstring_data_[j].kvec_d = kv_->kvec_d[ik_psi];

        psi->fix_k(ik_psi);
        const std::complex<double>* psi_k = psi->get_pointer();

        compute_S_k(j);
        compute_D_I(j, psi_k, nbands, nrow_local);
    }

    // MPI reduction: D_I is only partially computed on each rank (local rows only)
    // Must Allreduce to get the full sum across all processes
#ifdef __MPI
    for (int j = 0; j < nppstr_; j++)
    {
        for (int iat = 0; iat < nat_; iat++)
        {
            int r = nproj_per_atom_[iat];
            for (int lm = 0; lm < r; lm++)
            {
                if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat)) continue;
                if (kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm)) continue;
                int sz = kstring_data_[j].D_I[iat][lm].size();
                if (sz > 0)
                {
                    MPI_Allreduce(MPI_IN_PLACE, kstring_data_[j].D_I[iat][lm].data(),
                                  2 * sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
                }
            }
        }
    }
#endif

    // Step 3.5: Gauge fixing (SMO-anchored, Method 5)
    gauge_enabled_ = (PARAM.inp.deltap_gauge_mode == "smo_anchored");
    anchor_thr_ = PARAM.inp.deltap_anchor_thr;
    if (gauge_enabled_)
    {
        gauge_fix_smo_anchored(nbands);
    }

    // Second pass: compute Berry connection (needs all D_I for finite difference)
    for (int j = 0; j < nppstr_; j++)
    {
        int ik_psi = k_index_[0][j];
        if (ik_psi >= nks) continue;
        psi->fix_k(ik_psi);
        const std::complex<double>* psi_k = psi->get_pointer();
        const double* wg = &(pelec->wg(ik_psi, 0));
        compute_berry_connection(j, psi_k, nbands, nrow_local, wg);
    }

    // Step 4: Integrate to polarization
    integrate_polarization(ucell, nbands);

    // Step 5: Verify and output
    verify_sum_rule();
    write_results(ucell);

    std::cout << " >> Finish DeltaP atomic polarization decomposition.\n * * * * * *\n";

    ModuleBase::timer::end("DeltaP", "compute_atomic_polarization");
}

void DeltaP::setup_kstring(const K_Vectors& kv)
{
    ModuleBase::TITLE("DeltaP", "setup_kstring");
    ModuleBase::timer::start("DeltaP", "setup_kstring");

    const int mp_x = kv.nmp[0];
    const int mp_y = kv.nmp[1];
    const int mp_z = kv.nmp[2];
    const int direction = gdir_;

    // Fallback: with symmetry=-1, nmp may be [0,0,0] even though
    // the full Monkhorst-Pack grid was generated.  Infer from k-points.
    int mp_x_use = mp_x, mp_y_use = mp_y, mp_z_use = mp_z;
    if (mp_x == 0 || mp_y == 0 || mp_z == 0)
    {
        std::set<double> kx_set, ky_set, kz_set;
        for (int ik = 0; ik < kv.get_nkstot(); ++ik)
        {
            kx_set.insert(std::round(kv.kvec_d[ik].x * 1e6) / 1e6);
            ky_set.insert(std::round(kv.kvec_d[ik].y * 1e6) / 1e6);
            kz_set.insert(std::round(kv.kvec_d[ik].z * 1e6) / 1e6);
        }
        mp_x_use = kx_set.size();
        mp_y_use = ky_set.size();
        mp_z_use = kz_set.size();
        std::cout << "   DeltaP: nmp was [0,0,0], inferred ["
                  << mp_x_use << "," << mp_y_use << "," << mp_z_use
                  << "] from " << kv.get_nkstot() << " k-points" << std::endl;
    }

    int mp_dir = 0;
    int num_string = 0;
    if (direction == 1) { mp_dir = mp_x_use; num_string = mp_y_use * mp_z_use; }
    else if (direction == 2) { mp_dir = mp_y_use; num_string = mp_x_use * mp_z_use; }
    else { mp_dir = mp_z_use; num_string = mp_x_use * mp_y_use; }

    total_string_ = num_string;
    k_index_.resize(total_string_);
    for (int istring = 0; istring < total_string_; istring++)
    {
        k_index_[istring].resize(mp_dir + 1);
    }

    int string_index = -1;
    if (direction == 1)
    {
        for (int iz = 0; iz < mp_z; iz++)
        {
            for (int iy = 0; iy < mp_y; iy++)
            {
                string_index++;
                for (int ix = 0; ix < mp_x; ix++)
                {
                    k_index_[string_index][ix] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (ix == mp_x - 1)
                        k_index_[string_index][ix + 1] = k_index_[string_index][0];
                }
            }
        }
    }
    else if (direction == 2)
    {
        for (int iz = 0; iz < mp_z; iz++)
        {
            for (int ix = 0; ix < mp_x; ix++)
            {
                string_index++;
                for (int iy = 0; iy < mp_y; iy++)
                {
                    k_index_[string_index][iy] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (iy == mp_y - 1)
                        k_index_[string_index][iy + 1] = k_index_[string_index][0];
                }
            }
        }
    }
    else
    {
        for (int iy = 0; iy < mp_y; iy++)
        {
            for (int ix = 0; ix < mp_x; ix++)
            {
                string_index++;
                for (int iz = 0; iz < mp_z; iz++)
                {
                    k_index_[string_index][iz] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (iz == mp_z - 1)
                        k_index_[string_index][iz + 1] = k_index_[string_index][0];
                }
            }
        }
    }

    nppstr_ = mp_dir + 1;

    ModuleBase::timer::end("DeltaP", "setup_kstring");
}

} // namespace deltap
