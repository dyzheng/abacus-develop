#ifndef DELTAP_PW_H
#define DELTAP_PW_H

#include <vector>
#include <complex>

#include "source_estate/module_charge/charge_mixing.h"
#include "source_io/module_parameter/parameter.h"
#include "source_psi/psi.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"

class UnitCell;

namespace pw_deltap {

/**
 * @brief Initialize DeltaP PW from INPUT + STRU into the shared
 * DeltapScfSolver SCF state machine (owned by this module).
 *
 * Called once from ESolver_KS_PW::before_all_runners (driver_run.cpp, outside
 * the ionic-step loop; per-run setup, not per ionic step).  The backend captures
 * the stable psi / kv / wfcpw / rhopw objects (psi is allocated by
 * Setup_Psi_pw::before_runner before this call), so subsequent per-SCF-step
 * updates run through deltap_iter_finish without re-passing the basis.
 *
 * @param ucell   Unit cell.
 * @param inp     Input parameters.
 * @param psi_cpu Host-side wavefunctions (complex<double>).
 * @param kv      K-point vectors.
 * @param wfcpw   PW basis for wavefunctions.
 * @param rhopw   PW basis for charge density.
 */
void deltap_init(const UnitCell& ucell,
                 const Input_para& inp,
                 const psi::Psi<std::complex<double>>* psi_cpu,
                 const K_Vectors* kv,
                 const ModulePW::PW_Basis_K* wfcpw,
                 const ModulePW::PW_Basis* rhopw);

/// Current per-atom lambda (Ry), read by on-site force / stress / H operator.
const std::vector<double>& get_deltap_pw_lambda();

/// Per-atom constrain flags, read by on-site force / stress / H operator.
const std::vector<int>& get_deltap_pw_constrain();

/// Constraint energy correction −Σ λ·γ (Ry), applied to f_en.dp_escon.
double get_deltap_pw_escon();

/**
 * @brief Reset per-SCF-cycle DeltaP state (lambda-set flag, branch tracking).
 *
 * Called once per SCF cycle (before_scf) so that lambda is allowed to update
 * again for the changed geometry and cross-iteration 2pi branch tracking
 * restarts from an empty history.
 */
void reset_deltap_pw_scf_cycle();

/**
 * @brief Per-iteration logic for DeltaP PW (synchronous two-phase mode).
 *
 * Called from ESolver_KS_PW::iter_finish() after each SCF iteration.  When
 * the charge density is converged enough (drho < deltap_inner_thr), it:
 *   1. Computes total Berry phase gamma from the current wavefunctions
 *   2. Delegates the per-atom gamma measurement, gradient-descent lambda
 *      update and escon to the shared DeltapScfSolver state machine
 *   3. Reports the [DeltaP-PW] diagnostic line
 *
 * The PW inner loop (deltap_inner_nmax > 0) is rejected with WARNING_QUIT.
 *
 * @param ucell     Unit cell.
 * @param drho      Current charge density deviation.
 * @param psi_cpu   Host-side wavefunctions.
 * @param kv        K-point vectors.
 * @param wfcpw     PW basis for wavefunctions.
 * @param rhopw     PW basis for charge density.
 * @param inp       Input parameters.
 */
void deltap_iter_finish(
    const UnitCell& ucell,
    double drho,
    const psi::Psi<std::complex<double>>* psi_cpu,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    const Input_para& inp);

} // namespace pw_deltap

#endif // DELTAP_PW_H
