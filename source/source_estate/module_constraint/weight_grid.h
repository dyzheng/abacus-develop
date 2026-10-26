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
    DipoleSaw,      // M5: dipole weight -Lz*saw(z/Lz), global one-constraint
    DipoleMolecular // M5: dipole weight -r_axis, global one-constraint
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
               double screening_radius = 0.0,
               int dipole_axis = 2,
               double dipole_pos_max = 0.0,
               double dipole_pos_dec = 0.1);

    // (Re)compute all weights and the partition-of-unity audit.  Calling this
    // twice with the same geometry yields bit-identical results.
    void build();

    // (Re)compute the position-derivative grid d w_alpha / d R_J (M6 force
    // kernel input).  Pure function of the geometry like build(); must be
    // re-run once per geometry.  The grid layout is
    //   dw_[alpha][(J * 3 + d) * nrxx_ + ir]
    // with J the global atom index and d the Cartesian component (Bohr^-1).
    // Cost is O(N_g * N_at^3) once per geometry; storage is
    // 3 * N_at * N_alpha * N_g doubles (negligible next to the density).
    void build_derivatives();

    bool derivatives_built() const { return !dw_.empty(); }

    // d w_alpha / d R_J, Cartesian component d, at local grid point ir.
    double weight_derivative(const int alpha,
                             const int J,
                             const int d,
                             const int ir) const
    {
        return dw_[alpha][(J * 3 + d) * nrxx_ + ir];
    }
    // Raw per-constraint derivative grid (same layout as weight_derivative).
    const std::vector<std::vector<double>>& weight_derivatives() const
    {
        return dw_;
    }

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
    // Per-atom Becke weight w_iat(g), independent of the fragment map.  Used
    // by the S2 allocation read-out to split a grid density among physical
    // atoms even when the active constraint maps are fragments or dipoles.
    const std::vector<double>& atom_weight(const int iat) const
    {
        return w_[iat];
    }
    // Constraint -> atom mapping (default: one atom per constraint).
    const std::vector<std::vector<int>>& constraint_atoms() const
    {
        return constraint_atoms_;
    }

    // M5 dipole support.  The observable offset is the ionic part of the
    // dipole (saw: Lz*sum Z*s; molecular: sum Z*R_axis); the observer adds it
    // to int rho*w.  Non-dipole kinds return 0 and are not dipole.
    bool is_dipole() const
    {
        return type_ == WeightType::DipoleSaw
               || type_ == WeightType::DipoleMolecular;
    }
    double observable_offset(const int alpha) const
    {
        return observable_offset_.empty() ? 0.0
                                          : observable_offset_[alpha];
    }
    // d(offset)/dR_{J,d}: the only explicit-coordinate force term of a
    // dipole constraint (the density weight itself is geometry-independent).
    double dipole_atom_offset_derivative(const int alpha, const int J,
                                         const int d) const;
    void set_dipole_saw_pos(const double pos_max);
    double dipole_saw_pos() const { return dipole_pos_max_; }
    double dipole_saw_pos_dec() const { return dipole_pos_dec_; }

    // Cartesian position (Bohr) of local grid point ir.
    ModuleBase::Vector3<double> grid_position(const int ir) const;

    // Strain derivative d w_alpha / d eps_{a b} at a local grid point for a
    // homogeneous deformation with fixed fractional grid and atomic
    // coordinates.  This is a pure geometry derivative: it needs
    // build_derivatives() but no density.  It is accumulated directly by the
    // Becke strain kernel because the grid-nearest atom image (used by drR)
    // and the minimum-image atom pair (used by dRR) can differ in a periodic
    // cell, so a single atom-position contraction is not sufficient.
    // Dipole weights are coordinate functions rather than Becke partitions;
    // their strain derivative is not represented by this formula and the
    // constraint stress kernel rejects that combination explicitly.
    double weight_strain_derivative(const int alpha,
                                    const int a,
                                    const int b,
                                    const int ir) const;

    // The density grid this weight field lives on.
    const ModulePW::PW_Basis* rho_basis() const { return rho_basis_; }

  private:
    // Fractional (direct) coordinates of local grid point ir.
    ModuleBase::Vector3<double> grid_fraction(const int ir) const;
    // Global atom order fractional (direct) coordinates used by every weight
    // and derivative loop.
    std::vector<ModuleBase::Vector3<double>> atom_fractional_positions() const;
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
    int dipole_axis_ = 2;
    double dipole_pos_max_ = 0.0;
    double dipole_pos_dec_ = 0.1;
    int nat_ = 0;
    int nrxx_ = 0;

    // Per-atom weights w_[iat][ir_local] and derived per-constraint weights.
    std::vector<std::vector<double>> w_;
    std::vector<std::vector<double>> cw_;
    std::vector<double> observable_offset_;
    // Per-atom position-derivative cache wat_deriv_[iat][(J*3+d)*nrxx_+ir]
    // and the derived per-constraint derivative grid dw_ (same layout).
    std::vector<std::vector<double>> wat_deriv_;
    std::vector<std::vector<double>> dw_;
    // Per-constraint strain-derivative cache
    // dwe_[alpha][(a*3+b)*nrxx_+ir].
    std::vector<std::vector<double>> dwe_;
    std::vector<std::vector<int>> constraint_atoms_;

    // M5: build the single global dipole weight and its ionic offset.
    void build_dipole();

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
