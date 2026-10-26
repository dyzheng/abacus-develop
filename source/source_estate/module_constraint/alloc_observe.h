#ifndef ALLOC_OBSERVE_H
#define ALLOC_OBSERVE_H

#include <vector>

#include "source_estate/module_constraint/alloc_operator.h"

namespace alloc
{

/**
 * @brief Read-only decomposition of charge and spin into target/complement.
 *
 * All n/m values use the same real-space grid as the WeightGrid and the
 * supplied band functions.  Energies are not evaluated here.
 */
struct AllocResult
{
    std::vector<double> n_T; // per atom, target contribution
    std::vector<double> m_T; // per atom, target spin moment
    std::vector<double> n_c; // per atom, complement charge
    std::vector<double> m_c; // per atom, complement spin moment
    std::vector<double> n_tot;
    std::vector<double> m_tot;
    std::vector<double> n_c_grid; // local complementary charge density
    double cond = 1.0;
};

class AllocObserve
{
  public:
    explicit AllocObserve(const AllocOperator& op) : op_(op) {}

    AllocResult observe(const std::vector<std::vector<double>>& up,
                        const std::vector<double>& occ_up,
                        const std::vector<std::vector<double>>& dn,
                        const std::vector<double>& occ_dn) const;

  private:
    void accumulate_spin(const std::vector<std::vector<double>>& bands,
                         const std::vector<double>& occ,
                         const double spin_sign,
                         std::vector<double>& n_T,
                         std::vector<double>& m_T,
                         std::vector<double>& n_c_grid) const;

    const AllocOperator& op_;
};

} // namespace alloc

#endif
