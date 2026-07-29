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
| C-01~C-20 | 07-29 评审新发现（dp_escon rank0、力不自洽、PW λ 混淆、gdir≠3 错配、MPI 越界、PW 过冲、分支晶格口径等） | **Open — 见 `2026-07-29-deltap-risk-assessment-review.md` §3/§4** |
| B16 | Branch unwrapping inconsistent across λ | Fixed (P0 Hungarian matching)（注：match 文件 MPI 写竞争 C-10 削弱其跨 run 可靠性） |

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
| Z01 | unkOverlap_lcao 性能瓶颈 | 07-20 快速 O_kpair 路径上线（~30000×），07-29 核实 | R-6 |

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

> 2026-07-29 起以 `2026-07-29-deltap-risk-assessment-review.md` §9 的 P0/P1/P2 清单为准。以下旧条目保留备查。

1. ~~**Fix Z01**~~ — 07-20 快速 O_kpair 路径已上线（本文档此前状态滞后，07-29 核实）
2. **B16 diagnostics** — add runtime diagnostic output to verify Hungarian algorithm eliminates non-determinism
3. **Run regression** — SCF smoke test (λ=0 baseline, λ=0.05 constraint, 3-run determinism, inner loop)
4. **C1** — `compute_S_dk_link` for O(dk²) precision (deferred P3)（07-29 确认：函数已写但从未被调用，为死代码）
5. **BTO W90** — full Wannier90 validation (after Stage 3 passes)

---

## 2026-07-29 (round 2): 第二阶段修复 — C-05, C-11, C-02 Step1, C-07

### What was done
Executed TODO-1 through TODO-4 from the Phase-2 review/todo document:
- **C-05 + S-01**: Fixed gdir≠3 direction mismatch in compute_hk_correction.
  Added kstring_gdir_/kstring_string_ tracking members; on stale detection,
  rebuilds kstring_data_ (setup_kstring + compute_S_k/D_I + MPI Allreduce) for
  string 0 and input gdir.  gdir=3 single-string meshes incur no extra work.
- **C-11**: Unified branch shift lattice to per-band normalized amplitudes.
  Added current_zeta_scale tracking; Step-3 uses per-band normalized shifts ×
  zeta scale; all three global search locations (constraint matrix, total,
  per_atom) now use `current_zeta_scale * 2π·w_In(n,I) / Σ_i w_In(n,i)` from
  `w_In_first_string_` instead of the inconsistent per-atom normalization.
- **C-02 Step 1**: Force/stress corrections:
  1. lambda multiplied by tau_alpha (atomic fractional coordinate)
  2. Removed unjustified `force = force * 2.0`
  3. Fixed stress: integer r_vector → Cartesian coordinates via a1/a2/a3
  4. Added documentation header noting H_HK + dtau/dR terms not yet implemented
- **C-07**: compute_resta_z now skips with WARNING under MPI (2D block-cyclic
  indexing bug; function is experimental).

### Files modified
- `deltap.h`: +2 members (kstring_gdir_, kstring_string_)
- `deltap_wannier.cpp`: C-05 rebuild logic, C-11 lattice unification (3 sites),
  C-07 MPI guard
- `deltap_force_stress.hpp`: tau_alpha factor, removed ×2, Cartesian stress,
  limitations header

### Test plan (TODO-3b, deferred)
FD force validation requires coordinated ABACUS runs with frozen λ — planned
as a separate round.  See `2026-07-29-deltap-phase2-review-todo.md` §TODO-3b.

### Validation status (2026-07-29 end-of-day)
- [x] Main binary compiles and links (`abacus_basic_para`)
- [x] FD validation test script prepared (`tests/deltap_fd_force/run_fd.sh`)
- [x] H2O test inputs prepared (`tests/deltap_fd_force/h2o/`)
- [ ] **TODO-3b FD力验证**: 需在 HPC 集群上运行（8次 SCF × 2位移 × 12原子）
- [ ] **C-11 回归 BN 9-point PES**: 运行 `tests/deltap_bn_sampling/run_all.sh` 确认 9/9 收敛
- [ ] **现有 CI 测试重新生成 reference**: deltap_results.dat 会因口径修正而变化
- [ ] **gdir=1,2 测试**: 三个方向各跑一次 BN 约束 SCF

Full validation status report: `docs/superpowers/specs/2026-07-29-deltap-phase2-validation.md`

---

## 2026-07-29: 全面风险评审（理论+实现）

### What was done
对 feat/deltap 分支（merge-base b9669d37..HEAD 93a0d782）做系统静态风险评审：19 份设计文档 + 30 余源文件（deltap_wannier.cpp 2205 行全量、deltap_lcao、deltap_force_stress、esolver_ks_lcao/PW、deltap_pw、op_pw_proj、bfgs.h、deltap_common 及共享文件 diff）。产出 `2026-07-29-deltap-risk-assessment-review.md`，登记 61 项风险。

### Key new findings (不在 07-12/13 评审覆盖内)
- **C-01 (Critical)**：`dp_escon` 在 rank0 守卫内赋值（esolver_ks_lcao.cpp:793）→ MPI etot 跨 rank 不一致。
- **C-02 (Critical)**：力/应力与 H 不自洽——HR 算符含 τ_α（deltap_lcao.cpp:81）但力无 τ_α（deltap_force_stress.hpp:255）且比 dspin 模板多 ×2（:180）；HK 部分无力贡献；应力用整数晶格矢量量纲错。→ relax/MD 结果无效。
- **C-03/04 (Critical, PW)**：PW 初始 λ=目标 γ 值（deltap_pw.cpp:33）；has_deltap 误传 deltap_switch（hamilt_pw.cpp:127）→ corr=0 也被微扰。
- **C-05 (Critical)**：gdir≠3 时 hk_correction 的 k-string 方向（恒 z）与 S_dk 方向（输入 gdir）错配（deltap_wannier.cpp:1627 + 340-344）。
- **C-06 (Critical, MPI)**：hk_correction 假设 nrow==ncol，非方形 2D 网格越界写——np>1 crash 系列提交未根治。
- **C-08 (Critical, PW)**：PW 内循环不重解 psi，λ 线性过冲 inner_nmax 倍（deltap_pw.cpp:191-208）。
- **C-11 (Critical)**：分支平移晶格三种口径不一致（γ 定义 per-band 归一 vs 搜索 per-atom 归一 vs Step-3 未归一），与 dev log 第 5 条自相矛盾。
- **C-12/13 (relax/MD 失效)**：deltap_lambda_set_ 跨离子步不重置；hR 重建时不补加已有 λ（dspin 有处理，dp 没有）。
- **T-13 确认**：E_eff 换算漏 Ry→Ha 因子 2（esolver_ks_lcao.cpp:804），07-27 R8 怀疑属实。

### Bug list updates
| ID | 内容 | 状态 |
|----|------|------|
| C-01~C-20 | 见评审报告 §3/§4 | **Open（全部新建）** |
| S-01~S-17 | 疑似/脆弱点 | Open（需测试裁决） |
| Z01 | unkOverlap_lcao 性能 | **Closed**（07-20 快速路径已上线，本文档状态滞后已更正） |

### Files modified this round
- 新增 `docs/superpowers/specs/2026-07-29-deltap-risk-assessment-review.md`（本报告）；更新本文档。源码未动。

### Next steps
按评审报告 §9：P0 = C-01/C-03/C-04/C-06/C-02(+声明不支持 relax/MD)/C-12/C-13/R-2 回归；每项配可复现测试（MPI np=2 smoke、FD 力验证、gdir=1 约束、PW STRU-target 启动）。

---

## 2026-07-21: BN PES Sampling (γ Directional Stiffness)

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
