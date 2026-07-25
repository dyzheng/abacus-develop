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

void set_deltap_pw_lambda(const std::vector<double>& lambda,
                          const std::vector<int>& constrain);

const std::vector<double>& get_deltap_pw_lambda();
const std::vector<int>& get_deltap_pw_constrain();
const std::vector<double>& get_deltap_pw_targets();
double get_deltap_pw_escon();

void set_deltap_pw_active(bool active);
bool is_deltap_pw_active();

/**
 * @brief Store the PW Hamiltonian pointer for inner loop use.
 *
 * Must be called during before_scf where the template type is known.
 * Follows the DeltaSpin pattern (SpinConstrain stores hamilt as void*).
 */
void set_deltap_pw_hamilt(void* hamilt);

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
 * @param ucell   Unit cell.
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

/**
 * @brief Per-iteration logic for DeltaP PW: compute gamma and update lambda.
 *
 * This is called from ESolver_KS_PW::iter_finish() after each SCF iteration.
 * When charge density is converged enough (drho < deltap_inner_thr), it:
 *   1. Computes total Berry phase gamma from current wavefunctions
 *   2. Updates per-atom lambda via gradient descent (gamma - target)
 *   3. Stores updated lambda so the OnsiteProj picks it up next iteration
 *
 * Phase A behavior (no multi-k, Gamma-only): falls through without update.
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

void compute_per_atom_gamma_from_becp(
    const UnitCell& ucell,
    int nocc,
    double gamma_total,
    std::vector<double>& gamma_per_atom);

/**
 * @brief Compute per-atom gamma via k-string Wilson loop eigenvalue decomposition.
 *
 * For each k-string along gdir:
 *   1. Build overlap matrices M_j = <u_n(k_j)|u_m(k_{j+1})> (unkdotp_G)
 *      with G-phase for the boundary link (unkdotp_G0).
 *   2. Compute Wilson loop product M_total = Π_j M_j.
 *   3. Diagonalize M_total via zgeev → eigenvalues e^{iθ_n}, eigenvectors V.
 *   4. Project becp at k₀ onto V: proj[α,n] = Σ_m V_{mn}* × becp[α,m,k₀].
 *   5. Weights: w[I,n] = Σ_{α∈I} |proj[α,n]|².
 *   6. gamma[I] += w[I,n] × θ_n / Σ_J w[J,n].
 *
 * For multi-string meshes (e.g. 3x3x3), eigenvalues across strings are
 * matched via Hungarian algorithm to resolve 2π branch ambiguity.
 * For single-string meshes (1x1xN), direct diagonalization is sufficient.
 *
 * @param ucell      Unit cell
 * @param nocc       Number of occupied bands
 * @param psi_cpu    Wavefunctions (host-side)
 * @param kv         K-point vectors
 * @param wfcpw      PW basis for wavefunctions
 * @param rhopw      PW basis for charge density (G-phase FFT)
 * @param gdir       Direction (1=x, 2=y, 3=z)
 * @param gamma_per_atom Output: per-atom gamma [nat]
 */
void compute_per_atom_gamma_kstring(
    const UnitCell& ucell,
    int nocc,
    const psi::Psi<std::complex<double>>* psi_cpu,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis* rhopw,
    int gdir,
    std::vector<double>& gamma_per_atom);

} // namespace pw_deltap

#endif // DELTAP_PW_H
