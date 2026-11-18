#ifndef CONSTRAINT_MBIS_GPU_H
#define CONSTRAINT_MBIS_GPU_H

/**
 * @file constraint_mbis_gpu.h
 * @brief Direct device force/stress contraction for frozen-partition MBIS.
 *
 * The generic force/stress kernels materialise the huge derivative grids
 *   dw_ [nalpha][nat*3*nrxx],  dwe_ [nalpha][9*nrxx]
 * and then contract them with the channel densities.  For MBIS the frozen
 * partition gives the derivative in closed form at each grid point:
 *   dw_alpha/dR_J = [1_{J in alpha} - w_alpha] zeta_J w_J (r-R_J)/d,
 * so the force/stress contraction can be evaluated directly on the device
 * without ever forming dw_/dwe_.  Only the small nat*3 / 3*3 results return
 * to the host.
 *
 * The device backend (cuda|rocm/constraint_mbis_gpu.*) implements this; the
 * CPU stub returns false so callers keep the generic host path.
 */

#include <vector>

namespace constraint
{

/// Plain-pointer description of the local (rank) grid and MBIS state.
struct MbisDerivGrid
{
    int nat = 0;
    int nalpha = 0;
    int nrxx = 0;

    // Real-space index mapping (see WeightGrid::grid_fraction).
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nplane = 0;
    int startz_current = 0;

    // Row-major LATTICE_CONSTANT * LATTICE_VECTORS (Bohr).
    double cell[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const double* taud = nullptr;  // nat*3 fractional atom positions
    const double* w = nullptr;     // nat*nrxx per-atom weights
    const double* cw = nullptr;    // nalpha*nrxx per-constraint weights
    const double* dens = nullptr;  // nalpha*nrxx channel density per alpha
    const double* zeta = nullptr;  // nat Slater exponents
    const int* in_alpha = nullptr; // nat*nalpha membership
    const double* mu = nullptr;    // nalpha multipliers
    double dV = 0.0;               // cell volume / nxyz
};

/// Local force contribution (nat*3, PW layout iat*3+ipol).  Already carries
/// the -mu*dV factors; the caller only adds it to the force buffer.
bool mbis_force_gpu(const MbisDerivGrid& grid, std::vector<double>& force);

/// Local stress contribution (9, row-major a*3+b).  Carries the -mu*dV
/// factors of the strain-derivative term; the caller still adds the
/// volume/Jacobian diagonal term before the pool reduction.
bool mbis_stress_gpu(const MbisDerivGrid& grid, std::vector<double>& stress);

} // namespace constraint

#endif
