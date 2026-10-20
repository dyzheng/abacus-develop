#ifndef CONSTRAINT_DIPOLE_H
#define CONSTRAINT_DIPOLE_H

#include <vector>

#include "source_base/vector3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

namespace constraint
{

/**
 * @brief Passive dipole observables (module A / M1, plan 2026-10-20).
 *
 * Pure observation on the charge-density grid: nothing here feeds back into
 * the constraint loop.  Two frames:
 *  - molecular: p = sum_I Z*_I (R_I - o) - int rho (r - o) dr  [e*Bohr];
 *    origin-independent for neutral cells, printed with its origin for
 *    charged ones.
 *  - saw: p_z referenced to the Efield-style saw function for periodic
 *    slabs; the derived quantity is the potential drop dV = 4 pi p_z / A.
 *
 * The dipole-derived charge transfer q_dip (charge_transfer) is the
 * dipole-consistent counterpart of the Becke fragment charge (property-1
 * sec. 5.2): for a two-fragment system q_dip = (p - p_ref).d / |d|^2 with d
 * the fragment separation.
 */
class DipoleObserver
{
  public:
    // Dimensionless saw, periodic on x in [0,1); mirrors
    // Efield::saw_function (a = pos_max, b = pos_dec) so the two modules can
    // never disagree on the frame shape.  The decay (jump) region is
    // [pos_max, pos_max + pos_dec].
    static double saw(const double pos_max, const double pos_dec,
                      const double x);

    // Molecular-frame dipole about 'origin' [e*Bohr].  rho[is][ir] is the
    // per-spin density on rho_basis (nspin = 1/2; nspin = 4 is guarded out
    // at configure time).  MPI: pool-reduced.
    static ModuleBase::Vector3<double> molecular(const UnitCell& ucell,
            const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin,
            const ModuleBase::Vector3<double>& origin);

    // Saw-frame dipole along the third lattice vector [e*Bohr]:
    //   p_z = L_z * [ sum_I Z*_I saw(x_I) - int rho(r) saw(x(r)) dr ]
    // with x the fractional third coordinate and L_z the cell height.
    // Guarded (configure time) to cells with a z-aligned third lattice
    // vector orthogonal to a,b.  MPI: pool-reduced.
    static double saw_dipole(const UnitCell& ucell,
            const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin,
            const double pos_max, const double pos_dec);

    // Fractional third-lattice coordinate of the lowest plane-averaged
    // electron density plane (auto placement of the saw decay plane).
    static double lowest_density_plane(const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin);

    // Potential drop across the cell [Hartree]: dV = 4 pi p_z / A_perp.
    // Sign convention (locked by the unit test): positive sheet above
    // negative sheet (larger z) -> dV > 0.  Convert with
    // ModuleBase::Hartree_to_eV (1 Hartree = 27.211396 V for a potential).
    static double potential_drop(const double p_saw, const UnitCell& ucell);

    // Dipole-derived charge transfer [e]:
    //   q_dip = (p - p_ref) . (R_a - R_b) / |R_a - R_b|^2
    // = fragment a's excess positive charge relative to the reference.
    // The caller supplies the centers projected onto the physically
    // meaningful axis (the saw frame projects them onto z in init).
    static double charge_transfer(const ModuleBase::Vector3<double>& p,
            const ModuleBase::Vector3<double>& p_ref,
            const ModuleBase::Vector3<double>& center_a,
            const ModuleBase::Vector3<double>& center_b);

    // Z*-weighted ionic center of a fragment [Bohr, Cartesian].
    static ModuleBase::Vector3<double> ionic_center(const UnitCell& ucell,
            const std::vector<int>& fragment);

    // Smallest periodic distance [Bohr] from any atom to the saw decay
    // window [pos_max, pos_max + pos_dec] (fractional third coordinate);
    // 0 when an atom sits inside, otherwise the gap to the nearer edge.
    // Caller warns below ~1 Bohr.
    static double min_atom_decay_distance(const UnitCell& ucell,
            const double pos_max, const double pos_dec);
};

} // namespace constraint

#endif
