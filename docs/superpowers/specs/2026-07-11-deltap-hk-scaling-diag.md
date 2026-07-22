# DeltaP HK Correction Sign/Scale Diag (2026-07-11)

## 1. Test Plan

Diagnose why BFGS inner loop cannot converge gamma to target.
Hypothesis: HK correction uses wrong lambda (trial vs adjusted sign mismatch).

## 2. Test Setup

H2O, scf_nmax=3, deltap_nscf=2, target=[-0.090, -0.051, -0.051]
BFGS: alpha_init=0.5, max_step=0.1

## 3. Results

### 3.1 Before fix (trial lambda for HK correction)

```
step=0: lambda(trial)=-0.0055, |HK|=0.042, gamma=-0.0793
step=1: lambda(trial)=-0.0042, |HK|=0.013, gamma=-0.0791
```
|HK|/|λ| ratio: step0=7.6, step1=3.1 — 2.5× discrepancy. Lambda sign differs from operator's stored lambda (BFGS adjust reverses sign).

Gamma oscillates: iter2=-0.113, iter3=-0.239, iter4=-0.149.

### 3.2 After fix (recompute HK with adjusted lambda after accept_trial)

```
step=0: lambda(adj)=-0.0013, |HK|=0.042, gamma=-0.0805
step=1: lambda(adj)=-0.0041, |HK|=0.035, gamma=-0.0790
```
|HK|/|λ| ratio: step0=31.5, step1=8.7. Improved but not fully linear.

**Critical result:**
```
iter=2: gamma_O=-0.09095, |gamma-target|=0.0016  ← converged within 2σ!
iter=3: gamma_O=-0.05369 (oscillates back due to charge density update)
```

### 3.3 Longer test (scf_nmax=10, nscf=5)

iter=2 is best (|γ-target|=0.0016), then lambda overshoots as charge density re-converges:
```
iter=2: gamma=-0.091, lambda=-0.11, |γ-t|=0.002
iter=4: gamma=-0.087, lambda=-0.67, |γ-t|=0.009
iter=8: gamma=-0.068, lambda=-0.77, |γ-t|=0.022
```

## 4. Root Cause

**B10 resolved**: HK correction was computed with BFGS trial lambda (negative), but operator stored BFGS-adjusted lambda (positive). The trial uses `bfgs.step()` output; the adjusted uses `bfgs.get_lambda()` after `accept_trial()`. Sign mismatch caused correction applied with wrong magnitude.

**Fix**: Added `compute_hk_correction(lambda_adjusted)` + `set_hk_correction` after `get_lambda()` at step (h).

## 5. Remaining Issue

After inner loop converges gamma at iter=2, charge density update at iter=3 disrupts convergence. This is the charge-lambda coupling that requires more SCF iterations to relax. Solution: more outer steps + smaller max_step for gradual lambda evolution.

## 6. Next Steps

- [x] Fix HK correction lambda mismatch (B10)
- [ ] Fix sync mode gradient descent sign (use target - gamma)
- [ ] Reduce max_step to 0.05 for gradual lambda convergence
- [ ] Test with scf_nmax=50 to allow charge re-convergence
