#ifndef CONSTRAINT_WEIGHT_GRID_H
#define CONSTRAINT_WEIGHT_GRID_H

#include <vector>

#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

namespace constraint
{

enum class WeightType
{
    Becke, // Becke partition with heteronuclear size adjustment (M0)
};

/**
 * @brief Real-space weight field on the PW density grid (architecture layer M1).
 *
 * For every constraint alpha the weight w_alpha(r_g) is stored as a per-grid
 * point array (contiguous in alpha).  The default constraint set is one
 * constraint per atom (w_alpha = w_i); fragment constraints (sum over a
 * subset of atoms) are supported through constraint_atoms.
 *
 * The weight is a pure function of the geometry: build() is deterministic,
 * depends only on atomic positions / radii, and must be re-run once per
 * geometry (MD step).  Density never enters the weight construction.
 */
class WeightGrid
{
  public:
    /**
     * @param ucell        Unit cell with atomic positions (taud).
     * @param rho_basis    PW density grid (rank-local domain decomposition).
     * @param radii        Partition radius (Bohr) of every atom, indexed by
     *                     global atom index iat.
     * @param type         Weight recipe; only Becke is implemented in phase 1.
     * @param screening_radius  Involve only atoms within this distance (Bohr)
     *                     of a grid point.  <= 0 involves all atoms at every
     *                     grid point (exact partition of unity).
     */
    WeightGrid(const UnitCell& ucell,
               const ModulePW::PW_Basis* rho_basis,
               const std::vector<double>& radii,
               WeightType type = WeightType::Becke,
               double screening_radius = 0.0);

    // (Re)compute all weights and the partition-of-unity audit.  Calling this
    // twice with the same geometry yields bit-identical results.
    void build();

    int nat() const { return nat_; }
    int nrxx() const { return nrxx_; }
    int nconstraint() const { return constraint_atoms_.size(); }
    double screening_radius() const { return screening_radius_; }
    // max_g | sum_alpha w_alpha(g) - 1 | over the whole (MPI-reduced) grid.
    double max_partition_deviation() const { return maxdev_; }

    // Per-constraint weights: cw_[alpha][ir_local].
    const std::vector<std::vector<double>>& constraint_weights() const
    {
        return cw_;
    }
    const std::vector<double>& constraint_weight(const int alpha) const
    {
        return cw_[alpha];
    }
    // Constraint -> atom mapping (default: one atom per constraint).
    const std::vector<std::vector<int>>& constraint_atoms() const
    {
        return constraint_atoms_;
    }

    // Cartesian position (Bohr) of local grid point ir.
    ModuleBase::Vector3<double> grid_position(const int ir) const;

  private:
    // Fractional (direct) coordinates of local grid point ir.
    ModuleBase::Vector3<double> grid_fraction(const int ir) const;
    // Minimum-image displacement vector (Bohr) from 'from' to 'to', where
    // both are fractional coordinates inside the cell.
    ModuleBase::Vector3<double> min_image_displacement(
        const ModuleBase::Vector3<double>& from_frac,
        const ModuleBase::Vector3<double>& to_frac) const;

    const UnitCell& ucell_;
    const ModulePW::PW_Basis* rho_basis_;
    std::vector<double> radii_;
    WeightType type_;
    double screening_radius_;
    int nat_ = 0;
    int nrxx_ = 0;

    // Per-atom weights w_[iat][ir_local] and derived per-constraint weights.
    std::vector<std::vector<double>> w_;
    std::vector<std::vector<double>> cw_;
    std::vector<std::vector<int>> constraint_atoms_;

    // Partition-of-unity audit result (global max over ranks).
    double maxdev_ = 0.0;

  public:
    // Replace the constraint -> atom mapping (default: one atom per
    // constraint).  Only the derived per-constraint weights change; the
    // per-atom weights are untouched.
    void set_constraint_atoms(const std::vector<std::vector<int>>& atoms);
};

} // namespace constraint

#endif
