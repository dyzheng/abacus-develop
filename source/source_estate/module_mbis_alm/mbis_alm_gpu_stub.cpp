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

bool compute_mbis_gpu_flat(const std::vector<double>& /*rho*/,
                           const Grid& /*grid*/,
                           const std::vector<Atom>& /*atoms*/,
                           const MbisConfig& /*cfg*/,
                           std::vector<double>& /*w_flat*/,
                           std::vector<double>& /*Nout*/,
                           std::vector<double>& /*zeta_out*/,
                           double& /*Dout*/)
{
    return false;
}

bool mbis_gpu_local_step(const std::vector<double>& /*rho_local*/,
                         const Grid& /*grid*/,
                         const int /*nplane*/,
                         const int /*startz*/,
                         const std::vector<Atom>& /*atoms*/,
                         const MbisConfig& /*cfg*/,
                         const std::vector<double>& /*N*/,
                         const std::vector<double>& /*zeta*/,
                         const bool /*want_weights*/,
                         std::vector<std::vector<double>>& /*w*/,
                         std::vector<double>& /*Nout*/,
                         std::vector<double>& /*MRout*/)
{
    return false;
}

MbisGpuLocalWorkspace::MbisGpuLocalWorkspace() = default;
MbisGpuLocalWorkspace::~MbisGpuLocalWorkspace() = default;

bool MbisGpuLocalWorkspace::ready() const
{
    return false;
}

bool MbisGpuLocalWorkspace::init(const std::vector<double>& /*rho_local*/,
                                 const Grid& /*grid*/,
                                 const int /*nplane*/,
                                 const int /*startz*/,
                                 const std::vector<Atom>& /*atoms*/,
                                 const MbisConfig& /*cfg*/)
{
    return false;
}

bool MbisGpuLocalWorkspace::step(const std::vector<double>& /*N*/,
                                 const std::vector<double>& /*zeta*/,
                                 const bool /*want_weights*/,
                                 std::vector<std::vector<double>>& /*w*/,
                                 std::vector<double>& /*Nout*/,
                                 std::vector<double>& /*MRout*/)
{
    return false;
}

} // namespace mbis_alm
