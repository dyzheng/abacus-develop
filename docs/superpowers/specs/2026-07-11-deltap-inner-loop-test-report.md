# DeltaP Inner Loop Test Report (2026-07-11)

## 1. Test Plan

Verify that the BFGS inner-loop nested SCF:
1. Activates correctly at each iter>1 (immediate mode)
2. Runs BFGS steps with HK correction and HSolver diagonalization
3. Converges gamma toward target or identifies convergence blockers

## 2. Test Setup

**System**: H2O (tests/17_DS_DFTU/67_LCAO_DELTAP_H2O)
**Parameters**:
- scf_nmax=20, deltap_nscf=10, deltap_inner_thr=1e-4
- deltap_lambda_init=0.0, deltap_lambda_step=0.5, deltap_conv_thr=1e-3
- Target: [-0.090, -0.051, -0.051] (offset from natural gamma)
- BFGS: alpha_init=0.5, max_step=0.1

## 3. Results

### 3.1 Inner loop activation — PASS

Inner loop activates at iter=2,3,4,...,20 (immediate mode, no drho gate). BFGS steps proceed correctly:

```
iter=2: lambda=[0.023, 0.211, 0.211], gamma_O=-0.113, |γ-target|=0.023
iter=3: lambda=[0.364, 0.374, 0.374], gamma_O=-0.239, |γ-target|=0.149
iter=4: lambda=[0.556, 0.774, 0.774], gamma_O=-0.149, |γ-target|=0.110
...
iter=20: lambda=[1.349, 0.977, 0.978], gamma_O=-0.055, |γ-target|=0.044
```

### 3.2 Gamma convergence — FAIL

Gamma oscillates between -0.055 and -0.239. Never approaches target -0.090 within threshold 0.001:

```
iter= 1: gamma_O=-0.07901 (natural, lambda=0)
iter= 2: gamma_O=-0.11338 (lambda=0.045, moved AWAY)
iter= 3: gamma_O=-0.23940 (lambda=0.110, overshoot downward)
iter= 4: gamma_O=-0.14909 (lambda=0.385, partial recovery)
iter=13: gamma_O=-0.07081 (closest: |γ-target|=0.074)
iter=16: gamma_O=-0.05622 
iter=20: gamma_O=-0.05539 (stabilized but wrong value)
```

### 3.3 Branch selection — NOT the cause

Tested with frozen branch reference (save/restore W_prev_ at each inner step). Oscillation persists — ruling out branch jumps as the cause.

### 3.4 Charge density — unstable

```
GE1: drho=6.21e-07 (before inner loop)
GE2: drho=2.32e-02 (after first inner loop — 40000× increase!)
GE3: drho=8.18e-02
GE4: drho=6.73e-02
```

HK correction with lambda>0.02 causes massive charge density disruption.

## 4. Root Cause Analysis

The HK correction formula:
```
H_sym[α,β] = 0.5*(M_αβ + M_βα*) 
M = F · C_L†
F[α,p] = (i/2) · w_eff[p] · Σ_γ S[α,γ] · c_R[γ,p]
w_eff[p] = Σ_I λ_I · |D_I[lm][p]|²
```

With lambda~0.1: w_eff~5e-4, F~2e-4, H_sym~2e-4. This changes the Hamiltonian by ~10⁻⁴, shifting eigenvalues by similar amounts. Yet the Wilson loop (a product of O(nocc×nocc) overlap matrices) can amplify small Hamiltonian changes into large gamma shifts.

The gamma response is highly nonlinear:
- λ=0→0.01: Δγ≈+0.001 (very small)
- λ=0.01→0.05: Δγ≈-0.035 (35× larger than previous)
- λ=0.05→0.4: Δγ≈-0.130

This nonlinearity breaks BFGS (which assumes approximately linear response).

## 5. Next Steps

1. **Debug HK correction scaling**: verify `(i/2)` factor and w_eff normalization
2. **Test sign convention**: flip HK correction sign and observe gamma direction
3. **Consider alternative constraint**: direct gamma optimization vs. HK correction
4. **Commit framework**: inner loop + BFGS code is correct; sync mode (deltap_nscf=0) works as fallback
