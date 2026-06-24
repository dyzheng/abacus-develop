#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"

namespace deltap {

void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv,
                  const TwoCenterIntegrator* intor, const std::vector<double>& orb_cutoff,
                  double rm, int gdir)
{
    intor_ = intor;
    orb_cutoff_ = orb_cutoff;
    rm_ = rm;
    gdir_ = gdir;
    nat_ = ucell.nat;
    gd_ = &gd;
    kv_ = &kv;
    ModuleBase::TITLE("DeltaP", "init");
}

void DeltaP::compute_atomic_polarization(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi, const elecstate::ElecState* pelec) {}

void DeltaP::setup_kstring(const K_Vectors& kv)
{
    ModuleBase::TITLE("DeltaP", "setup_kstring");
    ModuleBase::timer::start("DeltaP", "setup_kstring");

    const int mp_x = kv.nmp[0];
    const int mp_y = kv.nmp[1];
    const int mp_z = kv.nmp[2];
    const int direction = gdir_;

    int mp_dir = 0;
    int num_string = 0;
    if (direction == 1) { mp_dir = mp_x; num_string = mp_y * mp_z; }
    else if (direction == 2) { mp_dir = mp_y; num_string = mp_x * mp_z; }
    else { mp_dir = mp_z; num_string = mp_x * mp_y; }

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
void DeltaP::verify_sum_rule() {}
void DeltaP::write_results(const UnitCell& ucell) const {}

} // namespace deltap
