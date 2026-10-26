#ifndef ALLOC_OPERATOR_H
#define ALLOC_OPERATOR_H

#include <vector>

#include "source_estate/module_constraint/weight_grid.h"

namespace alloc
{

/**
 * @brief Unified allocation operator A_i = Q_i + Qc W_i Qc on the grid.
 *
 * The target orbitals are real-space grid functions.  They are orthonormalized
 * in their R x R overlap subspace before use, so the target projector P and
 * its complement Qc are well defined.  This grid backend is the S1 read-out
 * prototype; the production LCAO/PW backends will share the same algebra.
 */
class AllocOperator
{
  public:
    /**
     * @brief Build from shared weights and R target grid functions.
     *
     * target_atom[a] is the global atom label of target orbital a.  The
     * WeightGrid constraint map must contain exactly one singleton per atom;
     * otherwise per-atom allocation weights are ambiguous and this function
     * throws std::invalid_argument.
     */
    void build_grid(const constraint::WeightGrid& wg,
                    const std::vector<int>& target_atom,
                    const std::vector<std::vector<double>>& phi0);

    int ntarget() const { return static_cast<int>(atom_of_.size()); }
    int nat() const { return static_cast<int>(w_.size()); }
    int nrxx() const { return nrxx_; }
    double dV() const { return dV_; }
    double cond() const { return cond_; }
    int atom_of(const int target) const { return atom_of_[target]; }
    const std::vector<double>& weight(const int atom) const
    {
        return w_[atom];
    }

    // o_a = <phi_a|x> for the orthonormalized targets.
    void overlaps(const std::vector<double>& x, std::vector<double>& o) const;
    void apply_P(const std::vector<double>& x, std::vector<double>& out) const;
    void apply_Qc(const std::vector<double>& x, std::vector<double>& out) const;
    void apply_Qi(const int atom, const std::vector<double>& x,
                  std::vector<double>& out) const;
    void apply_A(const int atom, const std::vector<double>& x,
                 std::vector<double>& out) const;

  private:
    std::vector<std::vector<double>> phi_; // orthonormalized, [R][nrxx]
    std::vector<int> atom_of_;             // [R] global atom index
    std::vector<std::vector<double>> w_;   // [nat][nrxx]
    int nrxx_ = 0;
    double dV_ = 0.0;
    double cond_ = 1.0;

    void check_vector(const std::vector<double>& x, const char* name) const;
};

} // namespace alloc

#endif
