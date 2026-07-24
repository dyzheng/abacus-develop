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

namespace pw_deltap {

void set_deltap_pw_lambda(const std::vector<double>& lambda,
                          const std::vector<int>& constrain);

const std::vector<double>& get_deltap_pw_lambda();
const std::vector<int>& get_deltap_pw_constrain();

void set_deltap_pw_active(bool active);
bool is_deltap_pw_active();

/**
 * @brief Run the inner lambda loop for DeltaP in PW basis.
 */
bool run_deltap_lambda_loop(const int iter,
                            const double drho,
                            const Input_para& inp);

/**
 * @brief Compute total Berry phase gamma along gdir from PW wavefunctions.
 *
 * Uses G-space overlaps <u_{m,k}|u_{n,k+dk}> along k-strings in the
 * specified direction.  Returns the unwrapped Berry phase in radians,
 * summed over occupied bands and all k-strings.
 *
 * @param psi_in  Wavefunctions (host-side, complex<double>).
 * @param kv      K-point vectors.
 * @param wfcpw   PW basis for wavefunctions.
 * @param rhopw   PW basis for charge density (for the G-phase link).
 * @param gdir    Direction index (1=x, 2=y, 3=z).
 * @param nbands  Number of occupied bands to include.
 * @return        Total Berry phase gamma in radians, or 0.0 if no k-strings.
 */
double compute_total_gamma_pw(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi_in,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    int nbands);

} // namespace pw_deltap

#endif // DELTAP_PW_H
