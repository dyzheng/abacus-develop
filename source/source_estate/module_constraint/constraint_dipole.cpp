#include "constraint_dipole.h"

#include <algorithm>
#include <cmath>

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"

namespace constraint
{

double DipoleObserver::saw(const double pos_max, const double pos_dec,
                           const double x)
{
    // Shape mirror of Efield::saw_function (efield.cpp): ramp up with slope
    // 1 on [0, pos_max], decay on [pos_max, pos_max+pos_dec], periodic.
    const double fac = 1.0 - pos_dec;
    // Branch A: rising ramp (left of the maximum).
    if (x <= pos_max)
    {
        return x - pos_max + 0.5 * fac;
    }
    // Branch B: right of the decay window (ramp continues periodically).
    else if (x > pos_max + pos_dec)
    {
        return x - pos_max - 1.0 + 0.5 * fac;
    }
    // Branch C: inside the decay window (steep descent).
    else
    {
        return 0.5 * fac - fac * (x - pos_max) / pos_dec;
    }
}

double DipoleObserver::saw_slope(const double pos_max, const double pos_dec,
                                 const double x)
{
    const double fac = 1.0 - pos_dec;
    // Branch A: rising ramp.
    if (x <= pos_max)
    {
        return 1.0;
    }
    // Branch B: right of the decay window -> periodic rising ramp.
    else if (x > pos_max + pos_dec)
    {
        return 1.0;
    }
    // Branch C: decay window (steep negative slope).
    else
    {
        return -fac / pos_dec;
    }
}

ModuleBase::Vector3<double> DipoleObserver::molecular(
    const UnitCell& ucell, const ModulePW::PW_Basis* rho_basis,
    const double* const* rho, const int nspin,
    const ModuleBase::Vector3<double>& origin)
{
    // Ionic part: sum_I Z*_I (R_I - origin).
    ModuleBase::Vector3<double> p(0.0, 0.0, 0.0);
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double> r
                = ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
            p.x += zv * (r.x - origin.x);
            p.y += zv * (r.y - origin.y);
            p.z += zv * (r.z - origin.z);
        }
    }
    // Electronic part: int rho(r) (r - origin) dr.  Grid indexing mirrors
    // Efield::cal_elec_dipole (efield.cpp:179-198) so the two modules read
    // the same grid points; startz_current keeps the MPI plane split exact.
    const int nspin0 = (nspin == 2) ? 2 : 1;
    ModuleBase::Vector3<double> pe(0.0, 0.0, 0.0);
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int i = ir / (rho_basis->ny * rho_basis->nplane);
        const int j = ir / rho_basis->nplane - i * rho_basis->ny;
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        const ModuleBase::Vector3<double> rfrac(
            static_cast<double>(i) / rho_basis->nx,
            static_cast<double>(j) / rho_basis->ny,
            static_cast<double>(k) / rho_basis->nz);
        const ModuleBase::Vector3<double> r
            = rfrac * ucell.latvec * ucell.lat0;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        pe.x += rho_tot * (r.x - origin.x);
        pe.y += rho_tot * (r.y - origin.y);
        pe.z += rho_tot * (r.z - origin.z);
    }
    // Sum reduction over the PW pool (no-op in serial builds).
#ifdef __MPI
    Parallel_Reduce::reduce_pool(pe.x);
    Parallel_Reduce::reduce_pool(pe.y);
    Parallel_Reduce::reduce_pool(pe.z);
#endif
    const double dV = ucell.omega / static_cast<double>(rho_basis->nxyz);
    p.x -= pe.x * dV;
    p.y -= pe.y * dV;
    p.z -= pe.z * dV;
    return p;
}

ModuleBase::Vector3<double> DipoleObserver::ionic_center(
    const UnitCell& ucell, const std::vector<int>& fragment)
{
    ModuleBase::Vector3<double> c(0.0, 0.0, 0.0);
    double zsum = 0.0;
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia, ++iat)
        {
            // Branch: atom in the fragment accumulates into the center.
            if (std::find(fragment.begin(), fragment.end(), iat)
                != fragment.end())
            {
                const ModuleBase::Vector3<double> r
                    = ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
                c.x += zv * r.x;
                c.y += zv * r.y;
                c.z += zv * r.z;
                zsum += zv;
            }
        }
    }
    // Guard: an empty fragment can not reach here (configure rejects empty
    // fragments), but never divide by zero.
    if (zsum > 0.0)
    {
        c.x /= zsum;
        c.y /= zsum;
        c.z /= zsum;
    }
    return c;
}

double DipoleObserver::charge_transfer(
    const ModuleBase::Vector3<double>& p,
    const ModuleBase::Vector3<double>& p_ref,
    const ModuleBase::Vector3<double>& center_a,
    const ModuleBase::Vector3<double>& center_b)
{
    const ModuleBase::Vector3<double> d = center_a - center_b;
    const double r2 = d * d; // |d|^2
    // Guard: coincident centers carry no charge-separation direction.
    if (r2 < 1e-20)
    {
        return 0.0;
    }
    return ((p - p_ref) * d) / r2;
}

double DipoleObserver::saw_dipole(const UnitCell& ucell,
        const ModulePW::PW_Basis* rho_basis,
        const double* const* rho, const int nspin,
        const double pos_max, const double pos_dec)
{
    // Effective cell height along the third lattice vector.  Configure-time
    // guard restricts this frame to z-aligned third vectors orthogonal to
    // a,b, so latvec.e33 * lat0 is the exact height.
    const double lz = ucell.latvec.e33 * ucell.lat0;
    // Ionic part: sum_I Z*_I saw(x_I), x_I = fractional third coordinate.
    double p_ion = 0.0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            double x = ucell.atoms[it].taud[ia].z;
            // Wrap into [0, 1): taud can carry small out-of-range values.
            x -= std::floor(x);
            p_ion += zv * saw(pos_max, pos_dec, x);
        }
    }
    // Electronic part: int rho(r) saw(x(r)) dr (same indexing as molecular).
    const int nspin0 = (nspin == 2) ? 2 : 1;
    double p_elec = 0.0;
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        const double x = static_cast<double>(k) / rho_basis->nz;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        p_elec += rho_tot * saw(pos_max, pos_dec, x);
    }
    // Sum reduction over the PW pool (no-op in serial builds).
#ifdef __MPI
    Parallel_Reduce::reduce_pool(p_elec);
#endif
    const double dV = ucell.omega / static_cast<double>(rho_basis->nxyz);
    return lz * (p_ion - p_elec * dV);
}

double DipoleObserver::lowest_density_plane(
    const ModulePW::PW_Basis* rho_basis,
    const double* const* rho, const int nspin)
{
    const int nspin0 = (nspin == 2) ? 2 : 1;
    // Plane-averaged profile over the global z index; the pool reduction
    // hands every rank the complete profile.
    std::vector<double> profile(rho_basis->nz, 0.0);
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        profile[k] += rho_tot;
    }
    // Sum reduction over the PW pool (no-op in serial builds).
#ifdef __MPI
    Parallel_Reduce::reduce_pool(profile.data(),
                                 static_cast<int>(profile.size()));
#endif
    const int kmin = static_cast<int>(std::min_element(
                          profile.begin(), profile.end())
                      - profile.begin());
    return static_cast<double>(kmin) / rho_basis->nz;
}

double DipoleObserver::potential_drop(const double p_saw,
                                      const UnitCell& ucell)
{
    // A_perp = |a x b| * lat0^2 (z-aligned third vector, configure guard).
    const double axb = ucell.latvec.e11 * ucell.latvec.e22
                       - ucell.latvec.e12 * ucell.latvec.e21;
    const double area = std::abs(axb) * ucell.lat0 * ucell.lat0;
    // dV = 4 pi p_z / A [Hartree]; sign locked by SawDipoleAndPotentialDrop.
    return ModuleBase::FOUR_PI * p_saw / area;
}

double DipoleObserver::min_atom_decay_distance(const UnitCell& ucell,
        const double pos_max, const double pos_dec)
{
    const double lz = ucell.latvec.e33 * ucell.lat0;
    double dmin = 1e30;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            double x = ucell.atoms[it].taud[ia].z;
            x -= std::floor(x);
            // Distance into the decay window [pos_max, pos_max+pos_dec]
            // (periodic): 0 inside, else the gap to the nearer edge,
            // converted to Bohr.
            double d = 0.0;
            // Branch A: atom inside the decay window.
            if (x >= pos_max && x <= pos_max + pos_dec)
            {
                d = 0.0;
            }
            // Branch B: outside — gap to the nearer window edge, periodic.
            else
            {
                const double d_left
                    = std::fmod(x - (pos_max + pos_dec) + 1.0, 1.0);
                const double d_right
                    = std::fmod(pos_max - x + 1.0, 1.0);
                d = std::min(d_left, d_right);
            }
            dmin = std::min(dmin, d * lz);
        }
    }
    return dmin;
}

} // namespace constraint
