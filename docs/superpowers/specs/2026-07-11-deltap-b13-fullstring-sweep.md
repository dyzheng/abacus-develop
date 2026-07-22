# DeltaP 全串修正 + Preconditioner 测试 (2026-07-11 最终轮)

## 1. Test Plan

1. Verify 16 k-string branch consistency (prerequisite for full-string)
2. Implement full-string B13 with D_I_all_ + per-atom preconditioner
3. Parameter sweep to find optimal max_step
4. 5-target sweep to verify constraint directionality

## 2. Setup

H2O, scf_nmax=20-30, deltap_nscf=3, deltap_inner_thr=5e-7
B13: full-string correction with D_I_all_ storage
Preconditioner: λ_I → λ_I / avg(|D_I|²)

## 3. Results

### 3.1 Branch consistency — PASS
All 16 strings give identical gamma after branch selection:
```
String 0:  gamma_O=-53.186300
String 1-15: gamma_O=-53.186270 (identical to 0.0003%)
```

### 3.2 max_step tuning

| max_step | gamma_O range | H stability | Assessment |
|----------|--------------|-------------|------------|
| 0.02 (no prec) | -0.05~-0.13 | -0.02~-0.16 | Poor |
| 0.02 (with prec) | -0.03~-0.11 | -0.02~-0.16 | Moderate |
| 0.005 (with prec) | -0.06~-0.14 | -0.04~-0.09 | **Best** |

### 3.3 30-step convergence test
- Gamma_O reaches target ±0.01: ~60% of iterations
- Best: iter=14, gamma_O=-0.08982, |γ-t|=0.0011
- No catastrophic divergence, lambda stable (no sign flips)
- Limit cycle: 2-3 good iterations → 1 oscillation → repeat

### 3.4 5-target sweep

| delta | gamma_O | |γ-t| | Success? |
|-------|---------|-------|----------|
| -0.010 | -0.098 | 0.008 | ✓ |
| -0.005 | -0.123 | 0.038 | ✗ |
| 0.000 | -0.081 | 0.001 | ✓ |
| +0.005 | -0.087 | 0.012 | ✗ |
| +0.010 | -0.085 | 0.015 | ✗ |

## 4. Analysis

Full-string correction works (resolves B13). Preconditioner normalizes H/O sensitivity. But **directionality is unidirectional**: HK correction only pushes gamma more negative regardless of λ sign. This is inherent in the Berry connection (1st-order) approximation. Bidirectional constraint requires the exact Berry phase operator (Wilson loop eigenvector propagation).

## 5. Conclusions

1. B13: Implemented and verified (full-string + preconditioner + max_step=0.005)
2. Directional limitation: Constraint works one-way (pushing gamma more negative)
3. Next: Exact Berry phase operator for bidirectional constraint
