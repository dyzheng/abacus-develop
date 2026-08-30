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

void WeightGrid::set_constraint_atoms(
    const std::vector<std::vector<int>>& atoms)
{
    constraint_atoms_ = atoms;
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
}

} // namespace constraint
