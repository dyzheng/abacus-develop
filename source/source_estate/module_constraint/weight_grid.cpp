#include "weight_grid.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "source_base/global_function.h"
#include "source_base/parallel_reduce.h"
#include "source_base/module_grid/partition.h"

namespace constraint
{

WeightGrid::WeightGrid(const UnitCell& ucell,
                       const ModulePW::PW_Basis* rho_basis,
                       const std::vector<double>& radii,
                       const WeightType type,
                       const double screening_radius)
    : ucell_(ucell), rho_basis_(rho_basis), radii_(radii), type_(type),
      screening_radius_(screening_radius)
{
    nat_ = ucell_.nat;
    nrxx_ = rho_basis_->nrxx;
    // Default constraint set: one constraint per atom.
    constraint_atoms_.resize(nat_);
    for (int iat = 0; iat < nat_; ++iat)
    {
        constraint_atoms_[iat] = {iat};
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

void WeightGrid::build()
{
    // Per-atom fractional (direct) coordinates.
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
    // Per-atom fractional (direct) coordinates — same geometry setup as
    // build(): the derivative grid is a pure function of the geometry and
    // must be recomputed together with the weights.
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

    // Involved-center set (same neighbor rule as build()).
    const bool screen = screening_radius_ > 0.0;
    std::vector<int> iR(nat_);
    std::iota(iR.begin(), iR.end(), 0);

    // Per-atom derivative cache (zero-initialized; coincident grid points
    // stay zero by the exact argument above).
    wat_deriv_.assign(nat_, std::vector<double>(nat_ * 3 * nrxx_, 0.0));
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
                    static_cast<int>(neigh.size()), neigh.data(),
                    static_cast<int>(ic), J, dw);
                for (int d = 0; d < 3; ++d)
                {
                    wat_deriv_[iat][(J * 3 + d) * nrxx_ + ir] = dw[d];
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
