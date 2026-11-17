#include "mbis_alm_gpu.h"

namespace mbis_alm
{

// CPU-only build: no device backend is compiled in.  compute_mbis() falls back
// to its host fixed point when this returns false.
bool mbis_gpu_available()
{
    return false;
}

bool compute_mbis_gpu(const std::vector<double>& /*rho*/,
                      const Grid& /*grid*/,
                      const std::vector<Atom>& /*atoms*/,
                      const MbisConfig& /*cfg*/,
                      MbisState& /*out*/)
{
    return false;
}

} // namespace mbis_alm
