# DeltaQS Bug Fix Log - 2026-07-01

## Bugs Found and Fixed

### Bug 1: Pure Charge Mode Segfault (P0)
**Status**: FIXED

**Symptom**: Segfault at address 0x30 when `sc_mag_switch=0, sc_charge_switch=1`

**Root Cause**: Two issues:
1. `onsite_radius` autoset in `read_input_item_exx_dftu.cpp:849` only checked `dft_plus_u || sc_mag_switch`, not `sc_charge_switch`. When both `dft_plus_u=0` and `sc_mag_switch=0`, `onsite_radius` stayed at 0.0, causing `overlap_orb_onsite` to never be created. The `intor_` pointer in DeltaSpin operator was null, causing crash in `cal_pre_HR()` → `intor_->snap()`.

2. Lambda loop activation gate in `esolver_ks_lcao.cpp:460` only checked `sc_mag_switch`, not `sc_charge_switch`. For pure charge mode, `run_constraint_loop()` was never called, so the charge constraint was never enforced.

**Fixes**:
1. `read_input_item_exx_dftu.cpp:849`: Added `|| para.input.sc_charge_switch` to the autoset condition
2. `esolver_ks_lcao.cpp:460`: Changed `if (PARAM.inp.sc_mag_switch)` to `if (PARAM.inp.sc_mag_switch || PARAM.inp.sc_charge_switch)`
3. `esolver_ks_lcao.cpp:464`: Added `&& PARAM.inp.sc_mag_switch` to `linear_scan` branch (spin-only feature)
4. `esolver_ks_lcao.cpp:477`: Added `&& PARAM.inp.sc_mag_switch` to `direction_only nspin=2` branch
5. `esolver_ks_lcao.cpp:526`: Added `&& PARAM.inp.sc_mag_switch` to `direction_only nspin=4` branch

**Verification**: T2 (pure charge ±0.3e perturbation) converges with Ni matching targets within 0.003e

### Bug 2: CG Oscillation in Combined Q+S Mode (P0)
**Status**: PARTIALLY FIXED

**Symptom**: RMS oscillates wildly (12→17→12) in combined charge+spin CG loop

**Root Causes**:
1. **alpha_factor clamping bug**: When `rms_plus > rms_error` (step made things worse), `alpha_factor` was negative but clamped to 1.0. This kept the bad step AND increased alpha for next iteration (`g = 1.5 * 1.0 = 1.5`).

2. **No CG restart**: When Polak-Ribiere beta became negative or very large, no restart to steepest descent.

3. **Subspace acceleration incompatibility**: The subspace approximation becomes inaccurate for large mu changes in the unified CG loop.

**Fixes**:
1. `deltaqs.cpp:646-672`: Changed `alpha_factor` clamping:
   - `alpha_factor < 0` → set to 0.0 (reject bad step entirely)
   - Added `g = 0.3` for `alpha_factor <= 0.1` (significant step size reduction)
   
2. `deltaqs.cpp:605-614`: Added CG restart: `if (beta < 0 || beta > 10) beta = 0`

3. `deltaspin_lcao.cpp:93-97`: Automatic subspace disable when `sc_charge_switch=1`:
   ```cpp
   if (inp.sc_charge_switch) {
       accel_mode = "off";
       accel_rms_thr = -1.0;
   }
   ```

**Verification**:
- T7 (single atom charge perturbation): Converges
- T3_full (Q+S combined, `sc_strategy accuracy`): Converges with Ni and Mi matching targets
- T3 (Q+S combined, `sc_strategy fast` + auto-override): Still slow to converge (needs more outer iterations)
- T8 (Q+S with target=natural charge): Slow convergence (needs more iterations)

### Bug 3: Subspace Acceleration Incompatibility (P1)
**Status**: WORKAROUND IN PLACE

**Issue**: Subspace acceleration approximates H_sub = H0_sub + Δλ·P_I_sub. This is accurate for small perturbations around a reference lambda. But in the unified CG loop, mu changes can be large, making the subspace approximation inaccurate.

**Workaround**: Automatic disable of subspace acceleration when charge constraint is enabled (see Bug 2 fix #3).

**Future Work**: Could rebuild subspace cache more frequently or use adaptive step sizes to keep perturbations small.

## Test Results Summary

| Test | Description | Status | Notes |
|------|-------------|--------|-------|
| T0 | Natural charges | PASS | N₀=N₁=13.7563 e, p=86% |
| T1 | Pure spin | PASS | Mi=±2.0 μB, E=-6773.088 eV |
| T2 | Pure charge ±0.3e | PASS | Ni=14.059/13.461, targets=14.056/13.456 |
| T3 | Q+S combined | SLOW | Converges with accuracy mode, slow with auto-override |
| T5 | Thermodynamic scan | PENDING | Waiting for T3 convergence |
| T7 | Single atom charge | PASS | Ni=14.257, target=14.256 |
| T8 | Q+S target=natural | SLOW | Very slow convergence, needs investigation |

## Files Modified

1. `source/source_io/module_parameter/read_input_item_exx_dftu.cpp` - onsite_radius autoset
2. `source/source_esolver/esolver_ks_lcao.cpp` - lambda loop activation gate
3. `source/source_lcao/module_deltaspin/deltaqs.cpp` - CG line search and restart
4. `source/source_lcao/module_deltaspin/deltaspin_lcao.cpp` - subspace auto-disable

## Remaining Issues

1. **Slow convergence for combined Q+S**: The unified CG loop converges but slowly. May need better preconditioning or step size adaptation.
2. **Temporary debug prints**: `cal_ni_lcao` prints all atoms (not just constrained), `cal_mi_lcao_wrapper` has `print=true`. Should revert after testing.
3. **4-process MPI segfault**: Pre-existing issue, not related to DeltaQS changes.
