#ifndef DELTAP_PW_H
#define DELTAP_PW_H

#include <vector>

#include "source_estate/module_charge/charge_mixing.h"
#include "source_io/module_parameter/parameter.h"

/**
 * @file deltap_pw.h
 * @brief DeltaP entry points and persistent data for PW basis.
 *
 * This file mirrors deltaspin_pw.h for DeltaP.  It holds the persistent
 * per-atom lambda and constrain arrays (function-local statics) so the
 * OnsiteProj operator (recreated every SCF step) can read them.
 *
 * TODO: In a follow-up, migrate to a DeltapConstrain singleton analogous
 * to SpinConstrain for cleaner separation.
 */

namespace pw_deltap {

void set_deltap_pw_lambda(const std::vector<double>& lambda,
                          const std::vector<int>& constrain);

const std::vector<double>& get_deltap_pw_lambda();
const std::vector<int>& get_deltap_pw_constrain();

void set_deltap_pw_active(bool active);
bool is_deltap_pw_active();

/**
 * @brief Run the inner lambda loop for DeltaP in PW basis.
 *
 * This function is the PW counterpart of the LCAO lambda-update logic
 * in esolver_ks_lcao.cpp::deltap_update_lambda().  It is called from
 * hamilt2rho_single().
 *
 * @param iter  Current SCF iteration index (0-based).
 * @param drho  Charge density difference.
 * @param inp   Input parameters.
 * @return      true if solver should be skipped (inner loop executed),
 *              false to proceed with normal SCF step.
 */
bool run_deltap_lambda_loop(const int iter,
                            const double drho,
                            const Input_para& inp);

} // namespace pw_deltap

#endif // DELTAP_PW_H
