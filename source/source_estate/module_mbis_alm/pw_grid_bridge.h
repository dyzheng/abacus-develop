#ifndef MODULE_MBIS_ALM_PW_GRID_BRIDGE_H
#define MODULE_MBIS_ALM_PW_GRID_BRIDGE_H

/**
 * @file pw_grid_bridge.h
 * @brief Conversion from an ABACUS PW real-space FFT grid to the standalone
 *        mbis_alm::Grid / atoms.
 *
 * Convention (verified against ABACUS cubes for LiCl and Fe2):
 *   a_i   = LATTICE_CONSTANT * LATTICE_VECTORS_i        (Bohr)
 *   dv_i  = a_i / n_i ,   origin = 0
 *   point(ix,iy,iz) = sum_i i * dv_i
 *   density index: ir = (ix*n1 + iy)*n2 + iz          (z fastest)
 *
 * MPI note: ABACUS stores chr.rho as a rank-local array of length nrxx.  The
 * MBIS core is serial, so the caller must gather the full grid density to rank
 * 0 (or pass an all-reduce-aware wrapper) before calling compute_mbis; the
 * weights can then be scattered back to the rank-local layout.
 */

#include "mbis_alm.h"

namespace mbis_alm
{

/// Build the real-space grid from FFT dimensions and lattice vectors (Bohr).
inline Grid make_grid(const int n[3], const Vec3 a[3])
{
    Grid g;
    g.n = {{n[0], n[1], n[2]}};
    g.origin = Vec3{0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
    {
        g.dv[i] = Vec3{a[i].x / n[i], a[i].y / n[i], a[i].z / n[i]};
    }
    return g;
}

/// Build atoms from Cartesian positions (Bohr) and valence charges.
inline std::vector<Atom> make_atoms(const std::vector<Vec3>& R,
                                    const std::vector<double>& Zval)
{
    std::vector<Atom> atoms(R.size());
    for (std::size_t i = 0; i < R.size(); ++i)
    {
        atoms[i].R = R[i];
        atoms[i].Zval = Zval[i];
    }
    return atoms;
}

} // namespace mbis_alm

#endif
