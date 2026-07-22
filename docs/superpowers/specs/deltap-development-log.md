# DeltaP Development Log

> **Compressed state index for long-context development.**

---

## Key Techniques & Conclusions (anti-repetition)

1. **HK correction sign**: `F = -(i/2)*w_eff*SC` not `+(i/2)`. Negative drives gamma toward more negative target. **Do not flip.**

2. **SMO + Löwdin**: `S^{-1/2}·S·S^{-1/2} = I` confirms S^{-1/2} numerically perfect. Do not remove Löwdin — raw weights amplify gamma noise 100×.

3. **Wilson loop unitary projection**: Newton-Schulz iteration (deterministic) with SVD fallback. Scaling MUST use `1/sqrt(frob2)` (not `1/sqrt(frob2/n_dim)`) to guarantee ‖X‖₂ ≤ 1 < √3 convergence.

4. **Eigenvalue matching**: Hungarian global optimal (Kuhn-Munkres O(n³)) replaces greedy nearest-neighbor. Deterministic even with near-degenerate eigenphases.

5. **Branch spacing**: Per-atom branch jump = ±2π·w_In (per-band weight), NOT uniform 2π·scale.

6. **Zeta rescaling**: Reference should be Σγ_unwrapped (branch-free), NOT arg(det W) (bounded to (-π,π]).

7. **Gauge anchor switch**: phase_corrections_ must be applied retroactively to all previous gauge_phase_[jj][n].

8. **BFGS class**: Actually implements Fletcher-Reeves CG, not BFGS. Renamed to FletcherReevesCG.

9. **SMO overlap**: max_asym=6.7e-17 → symmetric to machine precision. D_I conj convention harmless (double-conj in gauge fixing cancels).

10. **unkOverlap_lcao bottleneck**: `berryphase_overlap` takes ~30s per k-pair. Not in modified code. Blocks SCF integration test.

---

## Active Bugs

| ID | Bug | Status |
|----|-----|--------|
| B16 | Branch unwrapping inconsistent across λ | Fixed (P0 Hungarian matching) |
| Z01 | unkOverlap_lcao performance bottleneck (~30s/k-pair) | Open — blocks SCF test |

## Closed Bugs

| ID | Bug | Fix | Round |
|----|-----|-----|-------|
| B14 | k-string indices out of range | symmetry=-1 workaround | R-1 |
| B15 | Wilson loop eigenvalues collapse | SVD U·V† → NS projection | R-1 |
| B17 | n_dim=4 too small | mitigated by SVD/NS | R-1 |
| C2 | Zeta rescaling wrong reference | γ_unw_sum / γ_raw_sum | R-2 |
| H3 | Branch selection wrong spacing | per-band w_In spacing | R-2 |
| H4 | phase_corrections_ never applied | retroactive loop on anchor switch | R-2 |
| H1 | SMO overlap asymmetry | verified max_asym=6.7e-17 | R-3 |
| H2 | D_I conj convention suspect | verified harmless (double-conj cancels) | R-3 |
| M5 | Wrong S^{-1/2} diagnostic checks | replaced with correct identity | R-3 |
| C3 | psi-lambda inconsistency | downgraded to Medium | R-0 |
| H5 | fmod loses accumulated phase | downgraded to Low | R-0 |

---

## File Map

| File | Purpose | Last modified |
|------|---------|---------------|
| `deltap_wannier.cpp` | Wilson loop + NS + Hungarian + HK correction | 2026-07-13 |
| `deltap_gauge.cpp` | Gauge fixing + H4 anchor correction | 2026-07-13 |
| `deltap_overlap.cpp` | SMO overlap S^{-1/2} computation | 2026-07-09 |
| `deltap.h` | DeltaP class, FletcherReevesCG, branch state | 2026-07-13 |
| `deltap_lcao.cpp` | DeltaPOperator (HR + HK) | 2026-07-09 |
| `esolver_ks_lcao.cpp` | Inner loop + iter_finish | 2026-07-13 |
| `bfgs.h` | FletcherReevesCG optimizer | 2026-07-13 |

---

## Spec Index

| Date | Document | Key finding |
|------|----------|-------------|
| 2026-07-13 | `2026-07-13-deltap-progress-and-plan.md` | **总览: P0/P1 完成, P3 基本完成, SCF 测试受阻** |
| 2026-07-13 | `2026-07-13-deltap-risk-review-evaluation.md` | Rebuttal 评估: 共识 + 分歧 + 盲区 |
| 2026-07-13 | `2026-07-13-deltap-risk-review-rebuttal.md` | C1/C3 降级, B16 为实际阻滞 |
| 2026-07-13 | `2026-07-13-deltap-ppt-content-v2.json` | 领导层 PPT (13页) |
| 2026-07-13 | `2026-07-13-deltap-presentation-script.md` | 配套讲稿 (含知识点+领导提问) |
| 2026-07-13 | `2026-07-13-deltap-test-data.md` | 真实测试数据汇总 (7 表) |
| 2026-07-12 | `2026-07-12-deltap-risk-points-and-solutions.md` | 18 项风险点 + 解决方案 |
| 2026-07-12 | `2026-07-12-deltap-root-cause-analysis.md` | B14+B15 found+fixed |
| 2026-07-12 | `2026-07-12-deltap-algorithm-technical-review.md` | 完整算法推导 + 21 项风险 |
| 2026-07-10/11 | earlier specs | HK sign, parallel, B13, Löwdin |

---

## Next Steps (Priority)

1. **Fix Z01** — profile/fix `unkOverlap_lcao` performance (~30s/k-pair) to unblock SCF integration
2. **B16 diagnostics** — add runtime diagnostic output to verify Hungarian algorithm eliminates non-determinism
3. **Run regression** — SCF smoke test (λ=0 baseline, λ=0.05 constraint, 3-run determinism, inner loop)
4. **C1** — `compute_S_dk_link` for O(dk²) precision (deferred P3)
5. **BTO W90** — full Wannier90 validation (after Stage 3 passes)

---

## 2026-07-21: BN PES Sampling (γ Directional Stiffness)

### What was done
Ran 9-point PES scan of BN in (γ_B, γ_N) constraint space with Δ=0.02 and Δ=0.10:
- 4 directions: ±axis, ±diagonal, ±anti-diagonal + center
- 2 independent runs (Δ=0.02, Δ=0.10)

### Key Findings
1. **Electronic polarization stiffness**: < 1e-6 Ry/rad² (= 3e-5 eV/rad²) — essentially zero. BN's Berry phase is an almost free degree of freedom at the electronic level.
2. **Energy at noise floor**: All E_tot values collapse to -338.713359 or -338.713360. Hessian fitting gives negative eigenvalues (numerical artifact).
3. **Branch selection bug**: diag_minus (3.90, 3.40) diverged at iter 44 after 33 stable iterations. Per-string branch flipped due to small SCF density fluctuation.
4. **λ quantization**: Only 3-4 discrete λ levels (two-phase mode sets λ once). Center point has non-zero λ (-1.15e-5, -8.13e-6).

### Bug: Branch Divergence in P3
At diag_minus iter 44, γ jumped from (3.90, 3.40) → (-4.59, 15.12) → (-0.81, 8.95). Root cause: small SCF density change caused one Wilson loop string to cross ±π boundary, flipping its branch selection. The multi-band exhaustive search (K=3) evaluates each string independently — no cross-iteration continuity check.

**Proposed fix**: Per-string branch continuity tracking (compare raw per-string gamma across iterations), outlier rejection if total gamma jumps > threshold.

### Files
- `tests/deltap_bn_sampling/` — 9-point sampling dirs + run_all.sh
- `tests/deltap_bn_sampling/results.csv` — extracted data
- `docs/superpowers/specs/2026-07-21-deltap-bn-pes-sampling.md` — full analysis
