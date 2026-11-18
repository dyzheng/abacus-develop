#include "constraint_mbis_gpu.h"

namespace constraint
{

// CPU-only build: no device contraction is available.  WeightGrid keeps the
// generic host derivative grids in that case.
bool mbis_force_gpu(const MbisDerivGrid& /*grid*/, std::vector<double>& /*force*/)
{
    return false;
}

bool mbis_stress_gpu(const MbisDerivGrid& /*grid*/, std::vector<double>& /*stress*/)
{
    return false;
}

} // namespace constraint
