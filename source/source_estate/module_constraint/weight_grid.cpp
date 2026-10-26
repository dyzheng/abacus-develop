#include "weight_grid.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "constraint_dipole.h"
#include "source_base/global_function.h"
#include "source_base/parallel_reduce.h"
#include "source_base/module_grid/partition.h"

namespace constraint
{
namespace
{

// C2 smootherstep on [0,1] with zero first and second derivatives at both
// ends.  It is used for the finite Hirshfeld image window and for the smooth
// hand-over to the explicit remainder channel.
double smoothstep5(const double x, double* deriv)
{
    // Branch: clamp outside the transition interval so the window is exactly
    // constant in the fully covered and fully empty regions.
    if (x <= 0.0)
    {
        if (deriv != nullptr)
        {
            *deriv = 0.0;
        }
        return 0.0;
    }
    if (x >= 1.0)
    {
        if (deriv != nullptr)
        {
            *deriv = 0.0;
        }
        return 1.0;
    }
    const double x2 = x * x;
    const double x3 = x2 * x;
    if (deriv != nullptr)
    {
        *deriv = 30.0 * x2 * (1.0 - x) * (1.0 - x);
    }
    return x3 * (10.0 + x * (-15.0 + 6.0 * x));
}

// Invert a row-major 3x3 matrix.  The Hirshfeld image bounds need the
// reciprocal lattice vectors; a singular cell has no periodic image list.
bool invert3(const double a[9], double inv[9])
{
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7])
                       - a[1] * (a[3] * a[8] - a[5] * a[6])
                       + a[2] * (a[3] * a[7] - a[4] * a[6]);
    // Branch: a singular lattice cannot define reciprocal bounds.
    if (std::abs(det) < 1e-30)
    {
        return false;
    }
    const double id = 1.0 / det;
    inv[0] = (a[4] * a[8] - a[5] * a[7]) * id;
    inv[1] = (a[2] * a[7] - a[1] * a[8]) * id;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) * id;
    inv[3] = (a[5] * a[6] - a[3] * a[8]) * id;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) * id;
    inv[5] = (a[2] * a[3] - a[0] * a[5]) * id;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) * id;
    inv[7] = (a[1] * a[6] - a[0] * a[7]) * id;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) * id;
    return true;
}

} // namespace

WeightGrid::WeightGrid(const UnitCell& ucell,
                       const ModulePW::PW_Basis* rho_basis,
                       const std::vector<double>& radii,
                       const WeightType type,
                       const double screening_radius,
                       const int dipole_axis,
                       const double dipole_pos_max,
                       const double dipole_pos_dec)
    : ucell_(ucell), rho_basis_(rho_basis), radii_(radii), type_(type),
      screening_radius_(screening_radius), dipole_axis_(dipole_axis),
      dipole_pos_max_(dipole_pos_max), dipole_pos_dec_(dipole_pos_dec)
{
    nat_ = ucell_.nat;
    nrxx_ = rho_basis_->nrxx;
    // Branch A: dipole coordinates are a single global observable; there is
    // exactly one constraint and it spans the whole cell.
    if (is_dipole())
    {
        constraint_atoms_.resize(1);
        constraint_atoms_[0].resize(nat_);
        for (int iat = 0; iat < nat_; ++iat)
        {
            constraint_atoms_[0][iat] = iat;
        }
    }
    // Branch B: Becke fragments keep the one-constraint-per-atom default.
    else
    {
        constraint_atoms_.resize(nat_);
        for (int iat = 0; iat < nat_; ++iat)
        {
            constraint_atoms_[iat] = {iat};
        }
    }
}

void WeightGrid::prepare_hirshfeld_geometry()
{
    hirshfeld_atom_frac_ = atom_fractional_positions();
    hirshfeld_sigma_.assign(nat_, 0.0);
    hirshfeld_cutoff_.assign(nat_, 0.0);
    hirshfeld_charge_.assign(nat_, 0.0);

    double max_cutoff = 0.0;
    int iat = 0;
    for (int it = 0; it < ucell_.ntype; ++it)
    {
        for (int ia = 0; ia < ucell_.atoms[it].na; ++ia)
        {
            // The partition radius supplies the reference-density width.  A
            // floor avoids a zero-width Gaussian for malformed radius input.
            const double sigma = std::max(radii_[iat], 0.25);
            hirshfeld_sigma_[iat] = sigma;
            // Branch A: the common screening radius doubles as a user cutoff
            // for every atom species.
            if (screening_radius_ > 0.0)
            {
                hirshfeld_cutoff_[iat] = screening_radius_;
            }
            // Branch B: default cutoff is three Gaussian widths per atom.
            else
            {
                hirshfeld_cutoff_[iat] = 3.0 * sigma;
            }
            // The pseudo/atomic charge only weights the relative promolecule
            // density; a missing zv falls back to one positive unit.
            hirshfeld_charge_[iat] = std::max(ucell_.atoms[it].ncpp.zv, 1.0);
            max_cutoff = std::max(max_cutoff, hirshfeld_cutoff_[iat]);
            ++iat;
        }
    }

    // Image bounds from the reciprocal lattice: for any translation L,
    // n_i = (L.b_i)/(2 pi), so |n_i| is bounded by
    // (cell span + cutoff) |b_i|/(2 pi).  The extra layer is harmless and
    // makes the sum safe for skewed cells.
    double a[9] = {ucell_.latvec.e11, ucell_.latvec.e12,
                   ucell_.latvec.e13, ucell_.latvec.e21,
                   ucell_.latvec.e22, ucell_.latvec.e23,
                   ucell_.latvec.e31, ucell_.latvec.e32,
                   ucell_.latvec.e33};
    for (double& value : a)
    {
        value *= ucell_.lat0;
    }
    double inv[9];
    int nbound[3] = {1, 1, 1};
    double cell_span = 0.0;
    // Branch: a valid cell gives reciprocal-derived image bounds; a singular
    // cell can only use the nearest image plus a small safety shell.
    if (invert3(a, inv))
    {
        const double twopi = 2.0 * std::acos(-1.0);
        for (int i = 0; i < 8; ++i)
        {
            const int sx = (i & 1) ? 1 : 0;
            const int sy = (i & 2) ? 1 : 0;
            const int sz = (i & 4) ? 1 : 0;
            const double x = sx * a[0] + sy * a[3] + sz * a[6];
            const double y = sx * a[1] + sy * a[4] + sz * a[7];
            const double z = sx * a[2] + sy * a[5] + sz * a[8];
            cell_span = std::max(cell_span, std::sqrt(x * x + y * y + z * z));
        }
        for (int i = 0; i < 3; ++i)
        {
            // B = 2 pi (A^{-1})^T, hence row i of B is column i of inv.
            const double bx = inv[i];
            const double by = inv[3 + i];
            const double bz = inv[6 + i];
            const double bnorm = std::sqrt(bx * bx + by * by + bz * bz);
            nbound[i] = static_cast<int>(
                            std::ceil((2.0 * cell_span + max_cutoff)
                                      * bnorm / twopi))
                        + 2;
        }
    }

    hirshfeld_images_.clear();
    for (int n1 = -nbound[0]; n1 <= nbound[0]; ++n1)
    {
        for (int n2 = -nbound[1]; n2 <= nbound[1]; ++n2)
        {
            for (int n3 = -nbound[2]; n3 <= nbound[2]; ++n3)
            {
                const double x = n1 * a[0] + n2 * a[3] + n3 * a[6];
                const double y = n1 * a[1] + n2 * a[4] + n3 * a[7];
                const double z = n1 * a[2] + n2 * a[5] + n3 * a[8];
                // Keep the translation table across all atomic cutoffs.
                if (std::sqrt(x * x + y * y + z * z)
                    <= 2.0 * cell_span + max_cutoff + 1e-12)
                {
                    hirshfeld_images_.push_back(
                        ModuleBase::Vector3<double>(x, y, z));
                }
            }
        }
    }
}

double WeightGrid::hirshfeld_atom_density(
    const int iat,
    const ModuleBase::Vector3<double>& rfrac,
    ModuleBase::Vector3<double>* grad) const
{
    const ModuleBase::Vector3<double> rcart
        = rfrac * ucell_.latvec * ucell_.lat0;
    const ModuleBase::Vector3<double> ratom
        = hirshfeld_atom_frac_[iat] * ucell_.latvec * ucell_.lat0;
    const double alpha = 1.0 / (2.0 * hirshfeld_sigma_[iat]
                               * hirshfeld_sigma_[iat]);
    const double rc = hirshfeld_cutoff_[iat];
    const double r0 = 0.5 * rc;
    const double width = rc - r0;

    double q = 0.0;
    // Branch: the caller requested the position derivative cache.
    if (grad != nullptr)
    {
        grad->x = 0.0;
        grad->y = 0.0;
        grad->z = 0.0;
    }
    for (const ModuleBase::Vector3<double>& image : hirshfeld_images_)
    {
        const ModuleBase::Vector3<double> d
            = rcart - ratom - image;
        const double r2 = d.norm2();
        // Branch: this periodic image lies outside the compact window.
        if (r2 >= rc * rc)
        {
            continue;
        }
        const double r = std::sqrt(r2);
        double ds = 0.0;
        // Window = 1 - smootherstep((r-r0)/(rc-r0)); its sign is chosen so
        // the density is exactly one inside r0 and exactly zero at rc.
        const double step = smoothstep5((r - r0) / width, &ds);
        const double window = 1.0 - step;
        const double dwindow_dr = -ds / width;
        const double g = std::exp(-alpha * r2);
        q += hirshfeld_charge_[iat] * g * window;
        // Branch: derivative accumulation is requested and the image is not
        // coincident with the grid point (the vector derivative vanishes at
        // r=0 by symmetry).
        if (grad != nullptr && r > 1e-14)
        {
            const double radial = g * (2.0 * alpha * window
                                       - dwindow_dr / r);
            grad->x += hirshfeld_charge_[iat] * radial * d.x;
            grad->y += hirshfeld_charge_[iat] * radial * d.y;
            grad->z += hirshfeld_charge_[iat] * radial * d.z;
        }
    }
    return q;
}

void WeightGrid::build_hirshfeld()
{
    prepare_hirshfeld_geometry();
    w_.assign(nat_, std::vector<double>(nrxx_, 0.0));
    rem_.assign(nrxx_, 0.0);

    double local_maxdev = 0.0;
    double local_maxrem = 0.0;
    std::vector<double> q(nat_, 0.0);
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        double denom = 0.0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            q[iat] = hirshfeld_atom_density(iat, grid_fraction(ir), nullptr);
            denom += q[iat];
        }
        // Explicit remainder: q_rem/qfloor = 1 - smootherstep(denom/qfloor).
        // It is one in vacuum and exactly zero once the atomic promolecule
        // density reaches the floor, so covered regions keep an atom-only
        // partition of unity.
        const double t = denom / hirshfeld_qfloor_;
        const double rem_fraction = 1.0 - smoothstep5(t, nullptr);
        const double qrem = hirshfeld_qfloor_ * rem_fraction;
        const double total = denom + qrem;
        for (int iat = 0; iat < nat_; ++iat)
        {
            w_[iat][ir] = q[iat] / total;
        }
        rem_[ir] = qrem / total;
        local_maxrem = std::max(local_maxrem, rem_[ir]);
        local_maxdev = std::max(local_maxdev,
                                std::abs(denom / total + rem_[ir] - 1.0));
    }

    max_rem_ = local_maxrem;
    maxdev_ = local_maxdev;
#ifdef __MPI
    Parallel_Reduce::reduce_max_pool(rho_basis_->poolnproc, max_rem_);
    Parallel_Reduce::reduce_max_pool(rho_basis_->poolnproc, maxdev_);
#endif

    // Derive fragment-constraint weights exactly as in the Becke backend.
    cw_.assign(constraint_atoms_.size(), std::vector<double>(nrxx_, 0.0));
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                cw_[alpha][ir] += w_[iat][ir];
            }
        }
    }
}

ModuleBase::Vector3<double> WeightGrid::min_image_displacement(
    const ModuleBase::Vector3<double>& from_frac,
    const ModuleBase::Vector3<double>& to_frac) const
{
    // Wrap the fractional displacement into [-0.5, 0.5) so that the
    // Cartesian displacement is the nearest periodic image.
    double dx = to_frac.x - from_frac.x;
    double dy = to_frac.y - from_frac.y;
    double dz = to_frac.z - from_frac.z;
    auto wrap = [](double v) {
        v -= std::floor(v + 0.5);
        return v;
    };
    const ModuleBase::Vector3<double> disp_frac(wrap(dx), wrap(dy), wrap(dz));
    // Cartesian displacement (Bohr): frac * latvec * lat0.
    return disp_frac * ucell_.latvec * ucell_.lat0;
}

ModuleBase::Vector3<double> WeightGrid::grid_fraction(const int ir) const
{
    // ABACUS real-space index mapping: ir = ixy * nplane + iz_local.
    const int i = ir / (rho_basis_->ny * rho_basis_->nplane);
    const int j = ir / rho_basis_->nplane - i * rho_basis_->ny;
    const int k = ir % rho_basis_->nplane + rho_basis_->startz_current;
    return ModuleBase::Vector3<double>(
        static_cast<double>(i) / rho_basis_->nx,
        static_cast<double>(j) / rho_basis_->ny,
        static_cast<double>(k) / rho_basis_->nz);
}

ModuleBase::Vector3<double> WeightGrid::grid_position(const int ir) const
{
    return grid_fraction(ir) * ucell_.latvec * ucell_.lat0;
}

std::vector<ModuleBase::Vector3<double>>
WeightGrid::atom_fractional_positions() const
{
    // UnitCell stores taud per species; flatten it into the global atom order
    // used by build(), build_derivatives(), and the stress contraction.
    std::vector<ModuleBase::Vector3<double>> taud(nat_);
    int iat_glob = 0;
    for (int it = 0; it < ucell_.ntype; ++it)
    {
        for (int ia = 0; ia < ucell_.atoms[it].na; ++ia)
        {
            taud[iat_glob] = ucell_.atoms[it].taud[ia];
            ++iat_glob;
        }
    }
    return taud;
}

double WeightGrid::weight_strain_derivative(const int alpha,
                                            const int a,
                                            const int b,
                                            const int ir) const
{
    // Guard: the strain derivative needs its derivative cache and two valid
    // Cartesian components.
    if (dwe_.empty() || alpha < 0
        || alpha >= static_cast<int>(dwe_.size()) || a < 0 || a > 2 || b < 0
        || b > 2 || ir < 0 || ir >= nrxx_)
    {
        return 0.0;
    }
    return dwe_[alpha][(a * 3 + b) * nrxx_ + ir];
}

void WeightGrid::build()
{
    // Branch: a dipole coordinate has its own analytic weight source; it
    // does not partition the density and must not run the Becke loop.
    if (is_dipole())
    {
        build_dipole();
        return;
    }
    // Branch: periodic Hirshfeld has its own promolecule-density source and
    // an explicit remainder channel; it must not enter the Becke kernel.
    if (is_hirshfeld())
    {
        build_hirshfeld();
        return;
    }
    // Per-atom fractional (direct) coordinates.
    const std::vector<ModuleBase::Vector3<double>> taud
        = atom_fractional_positions();

    std::vector<double> dRR(nat_ * nat_, 0.0);
    for (int I = 0; I < nat_; ++I)
    {
        for (int J = I + 1; J < nat_; ++J)
        {
            const double d = min_image_displacement(taud[I], taud[J]).norm();
            dRR[I * nat_ + J] = d;
            dRR[J * nat_ + I] = d;
        }
    }

    // Involved-center set (neighbor list).  screening_radius_ <= 0 means all
    // atoms participate at every grid point (exact partition of unity).
    const bool screen = screening_radius_ > 0.0;
    std::vector<int> iR(nat_);
    std::iota(iR.begin(), iR.end(), 0);

    w_.assign(nat_, std::vector<double>(nrxx_, 0.0));
    std::vector<double> drR(nat_);
    std::vector<int> neigh;

    double local_maxdev = 0.0;
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        const ModuleBase::Vector3<double> rpos = grid_position(ir);

        // Distances to every atom (nearest periodic image).
        for (int I = 0; I < nat_; ++I)
        {
            drR[I] = min_image_displacement(taud[I], grid_fraction(ir)).norm();
        }

        if (screen)
        {
            // Neighbor screening: keep only atoms within the cutoff.  An
            // empty neighbor set falls back to all atoms so the partition of
            // unity (and hence charge conservation) is never violated.
            neigh.clear();
            for (int I = 0; I < nat_; ++I)
            {
                if (drR[I] <= screening_radius_)
                {
                    neigh.push_back(I);
                }
            }
            if (neigh.empty())
            {
                neigh = iR;
            }
        }
        else
        {
            neigh = iR;
        }

        double sum = 0.0;
        for (size_t ic = 0; ic < neigh.size(); ++ic)
        {
            const double w = Grid::Partition::w_becke_adjusted(
                nat_, drR.data(), dRR.data(), radii_.data(),
                static_cast<int>(neigh.size()), neigh.data(),
                static_cast<int>(ic));
            w_[neigh[ic]][ir] = w;
            sum += w;
        }
        local_maxdev = std::max(local_maxdev, std::abs(sum - 1.0));
    }

    // Global partition-of-unity audit (no-op in serial builds).
    maxdev_ = local_maxdev;
#ifdef __MPI
    Parallel_Reduce::reduce_max_pool(rho_basis_->poolnproc, maxdev_);
#endif

    // Derive per-constraint weights from per-atom weights.
    cw_.assign(constraint_atoms_.size(), std::vector<double>(nrxx_, 0.0));
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                cw_[alpha][ir] += w_[iat][ir];
            }
        }
    }
}

void WeightGrid::build_hirshfeld_derivatives()
{
    prepare_hirshfeld_geometry();
    wat_deriv_.assign(nat_, std::vector<double>(nat_ * 3 * nrxx_, 0.0));
    drem_.assign(nat_ * 3 * nrxx_, 0.0);
    // The Hirshfeld strain derivative is not part of S5.2.  Clear the cache so
    // the stress caller cannot mistake an absent derivative for a zero one.
    dwe_.clear();

    std::vector<double> q(nat_, 0.0);
    std::vector<ModuleBase::Vector3<double>> dq(nat_);
    for (int ir = 0; ir < nrxx_; ++ir)
    {
        const ModuleBase::Vector3<double> rfrac = grid_fraction(ir);
        double denom = 0.0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            q[iat] = hirshfeld_atom_density(iat, rfrac, &dq[iat]);
            denom += q[iat];
        }

        const double t = denom / hirshfeld_qfloor_;
        double dstep = 0.0;
        const double rem_fraction = 1.0 - smoothstep5(t, &dstep);
        const double qrem = hirshfeld_qfloor_ * rem_fraction;
        // d q_rem / d denom = -smoothstep'(t); outside the transition this is
        // exactly zero, so the remainder leaves the atom derivative intact.
        const double dqrem_ddenom = -dstep;
        const double total = denom + qrem;
        const double inv_total = 1.0 / total;
        const double inv_total2 = inv_total * inv_total;

        for (int J = 0; J < nat_; ++J)
        {
            // Only atom J's own promolecule density responds directly to
            // R_J; all other atoms respond through the shared denominator.
            const ModuleBase::Vector3<double>& ddenom = dq[J];
            const ModuleBase::Vector3<double> dD
                = (1.0 + dqrem_ddenom) * ddenom;
            for (int iat = 0; iat < nat_; ++iat)
            {
                ModuleBase::Vector3<double> dw(0.0, 0.0, 0.0);
                // Branch: only atom J's own promolecule density responds
                // directly to moving R_J; every atom still responds through
                // the shared denominator.
                if (iat == J)
                {
                    dw = dq[iat] * inv_total;
                }
                dw -= dD * (q[iat] * inv_total2);
                for (int d = 0; d < 3; ++d)
                {
                    wat_deriv_[iat][(J * 3 + d) * nrxx_ + ir] = dw[d];
                }
            }
            ModuleBase::Vector3<double> drem
                = ddenom * (dqrem_ddenom * inv_total)
                  - dD * (qrem * inv_total2);
            for (int d = 0; d < 3; ++d)
            {
                drem_[(J * 3 + d) * nrxx_ + ir] = drem[d];
            }
        }
    }

    // Derive per-constraint derivative grids from the per-atom cache.
    dw_.assign(constraint_atoms_.size(),
               std::vector<double>(nat_ * 3 * nrxx_, 0.0));
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            const std::vector<double>& wd = wat_deriv_[iat];
            std::vector<double>& dalpha = dw_[alpha];
            for (size_t k = 0; k < dalpha.size(); ++k)
            {
                dalpha[k] += wd[k];
            }
        }
    }
}

void WeightGrid::build_dipole()
{
    cw_.assign(1, std::vector<double>(nrxx_, 0.0));
    observable_offset_.assign(1, 0.0);
    const double lz = ucell_.latvec.e33 * ucell_.lat0;
    // Branch A: periodic saw frame.  The density weight is -Lz*s(x), and
    // the ionic part of p_z is Lz*sum Z*s(x_I).
    if (type_ == WeightType::DipoleSaw)
    {
        for (int ir = 0; ir < nrxx_; ++ir)
        {
            const double x = grid_fraction(ir).z;
            cw_[0][ir] = -lz * DipoleObserver::saw(dipole_pos_max_,
                                                   dipole_pos_dec_, x);
        }
        int iat = 0;
        for (int it = 0; it < ucell_.ntype; ++it)
        {
            for (int ia = 0; ia < ucell_.atoms[it].na; ++ia)
            {
                double x = ucell_.atoms[it].taud[ia].z;
                x -= std::floor(x);
                observable_offset_[0]
                    += ucell_.atoms[it].ncpp.zv * lz
                       * DipoleObserver::saw(dipole_pos_max_,
                                             dipole_pos_dec_, x);
                ++iat;
            }
        }
    }
    // Branch B: molecular frame about the Cartesian origin.  Delta semantics
    // subtract the reference dipole, so the charged-cell origin dependence
    // cancels; the raw absolute value keeps this convention in the report.
    else
    {
        for (int ir = 0; ir < nrxx_; ++ir)
        {
            cw_[0][ir] = -grid_position(ir)[dipole_axis_];
        }
        int iat = 0;
        for (int it = 0; it < ucell_.ntype; ++it)
        {
            for (int ia = 0; ia < ucell_.atoms[it].na; ++ia)
            {
                const ModuleBase::Vector3<double> r
                    = ucell_.atoms[it].taud[ia] * ucell_.latvec * ucell_.lat0;
                observable_offset_[0]
                    += ucell_.atoms[it].ncpp.zv * r[dipole_axis_];
                ++iat;
            }
        }
    }
    // A dipole does not form a partition of unity; maxdev has no meaning.
    maxdev_ = 0.0;
}

namespace
{
// Grid point / atom coincidence threshold (Bohr): within this distance the
// Becke weight of the coinciding atom is exactly 1 and every other weight
// exactly 0 (the cell function s(+-1) = {1, 0} with s'(+-1) = 0), so all
// position derivatives vanish identically and the raw kernel would hit a
// 0/0 at mu = +-1.  Treating the point as zero is exact in the limit.
const double k_deriv_eps = 1e-8;
} // namespace

void WeightGrid::build_derivatives()
{
    // Branch: the dipole density weight is a fixed function of the grid, not
    // of atomic positions.  Its force comes from the ionic offset derivative
    // instead, so an all-zero derivative grid is exact (and avoids the
    // expensive Becke derivative build).
    if (is_dipole())
    {
        dw_.assign(1, std::vector<double>(nat_ * 3 * nrxx_, 0.0));
        dwe_.clear();
        return;
    }
    // Branch: the periodic Hirshfeld backend has its own smooth quotient
    // derivative; using the Becke minimum-image kernel here would be wrong.
    if (is_hirshfeld())
    {
        build_hirshfeld_derivatives();
        return;
    }
    // Per-atom fractional (direct) coordinates — same geometry setup as
    // build(): the derivative grid is a pure function of the geometry and
    // must be recomputed together with the weights.
    const std::vector<ModuleBase::Vector3<double>> taud
        = atom_fractional_positions();

    std::vector<double> dRR(nat_ * nat_, 0.0);
    std::vector<double> pair_dir(nat_ * nat_ * 3, 0.0);
    for (int I = 0; I < nat_; ++I)
    {
        for (int J = I + 1; J < nat_; ++J)
        {
            // pair_dir is R_I - R_K for the minimum-image pair, matching the
            // dRR scalar.  It cannot be reconstructed from the grid-centered
            // drR/eR at a periodic boundary.
            const ModuleBase::Vector3<double> disp
                = min_image_displacement(taud[J], taud[I]);
            const double d = disp.norm();
            dRR[I * nat_ + J] = d;
            dRR[J * nat_ + I] = d;
            for (int dd = 0; dd < 3; ++dd)
            {
                const double dir = disp[dd] / d;
                pair_dir[(I * nat_ + J) * 3 + dd] = dir;
                pair_dir[(J * nat_ + I) * 3 + dd] = -dir;
            }
        }
    }

    // Involved-center set (same neighbor rule as build()).
    const bool screen = screening_radius_ > 0.0;
    std::vector<int> iR(nat_);
    std::iota(iR.begin(), iR.end(), 0);

    // Reverse map atom -> constraints, used to distribute each atom center's
    // strain derivative to every fragment constraint that contains it.
    std::vector<std::vector<int>> atom_to_constraint(nat_);
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            atom_to_constraint[iat].push_back(static_cast<int>(alpha));
        }
    }

    // Per-atom derivative cache (zero-initialized; coincident grid points
    // stay zero by the exact argument above).
    wat_deriv_.assign(nat_, std::vector<double>(nat_ * 3 * nrxx_, 0.0));
    dwe_.assign(constraint_atoms_.size(),
                std::vector<double>(9 * nrxx_, 0.0));
    std::vector<double> drR(nat_);
    std::vector<double> eR(nat_ * 3, 0.0);
    std::vector<int> neigh;

    for (int ir = 0; ir < nrxx_; ++ir)
    {
        const ModuleBase::Vector3<double> rfrac = grid_fraction(ir);

        // Distances and direction cosines to every atom (nearest periodic
        // image; eR points from the grid point toward the atom, the M0
        // kernel convention).
        bool on_atom = false;
        for (int I = 0; I < nat_; ++I)
        {
            const ModuleBase::Vector3<double> disp
                = min_image_displacement(taud[I], rfrac);
            drR[I] = disp.norm();
            on_atom = on_atom || drR[I] < k_deriv_eps;
            if (drR[I] > k_deriv_eps)
            {
                // eR points from the grid point toward the atom (the M0
                // kernel convention); min_image_displacement above returns
                // the atom -> grid-point vector, hence the sign flip.
                for (int d = 0; d < 3; ++d)
                {
                    eR[3 * I + d] = -disp[d] / drR[I];
                }
            }
            else
            {
                // Coincident atom: leave eR zero; the point is skipped below.
                for (int d = 0; d < 3; ++d)
                {
                    eR[3 * I + d] = 0.0;
                }
            }
        }

        if (on_atom)
        {
            // Branch A: grid point coincides with an atom.  Every weight
            // derivative vanishes identically (see k_deriv_eps comment);
            // the cached grid is already zero-initialized, so skip the
            // kernel (which would divide 0 by 0 at mu = +-1).
            continue;
        }

        // Branch B: regular grid point — evaluate the M0 kernel for every
        // involved weight center against every atom position.
        if (screen)
        {
            neigh.clear();
            for (int I = 0; I < nat_; ++I)
            {
                if (drR[I] <= screening_radius_)
                {
                    neigh.push_back(I);
                }
            }
            if (neigh.empty())
            {
                neigh = iR;
            }
        }
        else
        {
            neigh = iR;
        }

        for (size_t ic = 0; ic < neigh.size(); ++ic)
        {
            const int iat = neigh[ic];
            for (int J = 0; J < nat_; ++J)
            {
                double dw[3] = {0.0, 0.0, 0.0};
                Grid::Partition::w_becke_adjusted_deriv(
                    nat_, drR.data(), dRR.data(), radii_.data(), eR.data(),
                    pair_dir.data(),
                    static_cast<int>(neigh.size()), neigh.data(),
                    static_cast<int>(ic), J, dw);
                for (int d = 0; d < 3; ++d)
                {
                    wat_deriv_[iat][(J * 3 + d) * nrxx_ + ir] = dw[d];
                }
            }
            double dwe[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0};
            Grid::Partition::w_becke_adjusted_strain_deriv(
                nat_, drR.data(), dRR.data(), radii_.data(), eR.data(),
                pair_dir.data(), static_cast<int>(neigh.size()), neigh.data(),
                static_cast<int>(ic), dwe);
            for (const int alpha : atom_to_constraint[iat])
            {
                for (int k = 0; k < 9; ++k)
                {
                    dwe_[alpha][k * nrxx_ + ir] += dwe[k];
                }
            }
        }
    }

    // Derive per-constraint derivative grids from the per-atom cache
    // (fragment constraints: sum the fragment atoms' derivatives).
    dw_.assign(constraint_atoms_.size(), std::vector<double>(nat_ * 3 * nrxx_, 0.0));
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            const std::vector<double>& wd = wat_deriv_[iat];
            std::vector<double>& dalpha = dw_[alpha];
            for (size_t k = 0; k < dalpha.size(); ++k)
            {
                dalpha[k] += wd[k];
            }
        }
    }
}

double WeightGrid::dipole_atom_offset_derivative(const int alpha,
                                                 const int J,
                                                 const int d) const
{
    // Guard: only the single dipole constraint carries an offset.
    if (alpha != 0 || !is_dipole() || J < 0 || J >= nat_ || d < 0 || d > 2)
    {
        return 0.0;
    }
    // Locate the atom in the UnitCell (global iat order).
    int iat = 0;
    for (int it = 0; it < ucell_.ntype; ++it)
    {
        for (int ia = 0; ia < ucell_.atoms[it].na; ++ia)
        {
            // Branch: this is the requested global atom.
            if (iat == J)
            {
                const double zv = ucell_.atoms[it].ncpp.zv;
                // Branch A: molecular offset Z*R_axis -> derivative Z on
                // the selected axis only.
                if (type_ == WeightType::DipoleMolecular)
                {
                    return (d == dipole_axis_) ? zv : 0.0;
                }
                // Branch B: saw offset Lz*Z*s(x) -> Z*ds/dx for z only
                // (d/dz = ds/dx * dx/dz, and Lz*dx/dz = 1).
                if (d == 2)
                {
                    double x = ucell_.atoms[it].taud[ia].z;
                    x -= std::floor(x);
                    return zv * DipoleObserver::saw_slope(
                                    dipole_pos_max_, dipole_pos_dec_, x);
                }
                return 0.0;
            }
            ++iat;
        }
    }
    return 0.0;
}

void WeightGrid::set_dipole_saw_pos(const double pos_max)
{
    // Guard: only the saw source has a movable decay plane.
    if (type_ != WeightType::DipoleSaw)
    {
        return;
    }
    dipole_pos_max_ = pos_max;
    // The old derivative grid (all zeros) is still valid in value, but drop
    // it so a later build_derivatives() cannot be mistaken for the new
    // geometry's cache.
    dw_.clear();
    build();
}

void WeightGrid::set_constraint_atoms(
    const std::vector<std::vector<int>>& atoms)
{
    constraint_atoms_ = atoms;
    // Branch A: per-atom weights not built yet (set before build()).
    // build() derives cw_ from constraint_atoms_ itself, so nothing else to do.
    if (w_.empty())
    {
        return;
    }
    // Branch B: per-atom weights cached (set after build()).
    // Re-derive per-constraint weights from the cached per-atom weights.
    cw_.assign(constraint_atoms_.size(), std::vector<double>(nrxx_, 0.0));
    for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
    {
        for (const int iat : constraint_atoms_[alpha])
        {
            for (int ir = 0; ir < nrxx_; ++ir)
            {
                cw_[alpha][ir] += w_[iat][ir];
            }
        }
    }
    // Branch B (derivatives): the per-atom derivative cache is also
    // geometry-derived, so a post-build constraint-map change must re-derive
    // the per-constraint derivative grid the same way.
    if (!wat_deriv_.empty())
    {
        dw_.assign(constraint_atoms_.size(),
                   std::vector<double>(nat_ * 3 * nrxx_, 0.0));
        for (size_t alpha = 0; alpha < constraint_atoms_.size(); ++alpha)
        {
            for (const int iat : constraint_atoms_[alpha])
            {
                const std::vector<double>& wd = wat_deriv_[iat];
                std::vector<double>& dalpha = dw_[alpha];
                for (size_t k = 0; k < dalpha.size(); ++k)
                {
                    dalpha[k] += wd[k];
                }
            }
        }
    }
}

} // namespace constraint
