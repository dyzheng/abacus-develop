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
| C-21 | LCAO `deltap_init` 三个 raw `new` 无 delete（每次运行泄漏 DeltaP/unkOverlap_lcao/cal_r_overlap_R） | **Fixed (R1 07-31：改 unique_ptr + 前向声明)** |
| C-22 | `deltap_constraint_lambda_` 回退路径维度错（应为 m 却 assign nat，约束矩阵加载失败时触发） | **Fixed (R2 08-01：状态机 `lambda_cstr` 按矩阵模式 m 维初始化，连带修复)** |
| C-23 | target/约束矩阵每 MPI rank 各自读文件，无 rank0 读 + Bcast 收敛路径 | **Fixed (R4 08-01：`DeltapScfSolver::init` rank0 解析 + `Parallel_Common` Bcast；尺寸不匹配改 WARNING_QUIT)** |
| C-24 | PW 死代码：`run_deltap_lambda_loop` no-op、`s_hamilt`/`set_deltap_pw_hamilt`、`inner_nmax>0` 死分支、`compute_per_atom_gamma_from_becp` | **Fixed (R1 07-31：全部删除；SCF 周期重置改名 `reset_deltap_pw_scf_cycle` 保留)** |
| C-25 | `deltap_solver.h` 全仓库 0 引用；`deltap_common.h` 4 函数仅 1 个被用 | **Fixed (R1 07-31：删文件 + 删 3 个未用函数)** |
| C-26 | real 实例每 SCF 迭代重复 WARNING；`iter_finish` DeltaP 块 ~140 行职责混杂 | **Partial（R2 08-01：块已缩至 22 行，状态机接管；WARNING 收口留 R4）** |
| C-27 | PW 无 target 时 `targets[iat]` 对空 vector 越界读（UB，实际按 0 约束 γ→0） | **Fixed (R3 08-01：`deltap_init` 显式把空 target 填 0 向量，行为不变)** |
| C-28 | PW 无 MPI λ 同步 + 打印无 rank 守卫（S-09：多 rank 时 λ/escon 不一致、重复输出） | **Fixed (R4 08-01：backend `sync_lambda` Bcast + `s_lambda` 全 rank 刷新；init/report 打印 rank0 守卫)** |
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
| `esolver_ks_lcao.cpp` | Inner loop + iter_finish（R2 已瘦身为接线层：init + backend 绑定） | 2026-08-01 |
| `bfgs.h` | FletcherReevesCG optimizer | 2026-07-13 |
| `deltap_common.h` | 纯函数库（R2 起 6 个函数全部被 DeltapScfSolver 使用） | 2026-08-01 |
| `deltap_solver.h` | deprecated 死文件，全仓库 0 引用，计划删除 | 2026-07-31 |
| `deltap_pw.cpp` | PW 数值 + 状态机实例（R3 起：7 全局 → 2 算子状态 + DeltapScfSolver 实例 + PW backend） | 2026-08-01 |
| `esolver_ks_pw.cpp` | PW esolver 接线（R3 起：init 20 行 → `deltap_init` 6 行） | 2026-08-01 |
| `deltap_scf.h/.cpp`（新增） | DeltapScfSolver 状态机（basis-independent 控制流，LCAO + PW 均已接入；含 unwrap_branch_2pi / gamma_report） | 2026-08-01 |
| `source_esolver/test/deltap_common_test.cpp`（新增） | deltap_common 纯函数单测（10 用例） | 2026-08-01 |

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
| 2026-07-31 | `2026-07-31-deltap-esolver-refactor-design.md` | esolver 侧重构设计：死代码/重复实现/双状态机清理，DeltapScfSolver 方案，4 轮迁移 |
| 2026-08-01 | `2026-08-01-deltap-esolver-refactor-r2.md` | R2 实施：DeltapScfSolver 状态机上线，同步路径 A/B 逐字节全等，内循环冒烟通过 |
| 2026-08-01 | `2026-08-01-deltap-esolver-refactor-r3.md` | R3 实施：PW 接入同一状态机（7 全局 → 状态机实例），PW A/B 逐字节全等 + LCAO/内循环回归 |
| 2026-08-01 | `2026-08-01-deltap-esolver-refactor-r4.md` | R4 实施：MPI rank0+Bcast（C-23）、PW λ 同步/rank 守卫（S-09/C-28）、打印/WARNING 收口、deltap_common 单测 10 例 |
| 2026-07-12 | `2026-07-12-deltap-risk-points-and-solutions.md` | 18 项风险点 + 解决方案 |
| 2026-07-12 | `2026-07-12-deltap-root-cause-analysis.md` | B14+B15 found+fixed |
| 2026-07-12 | `2026-07-12-deltap-algorithm-technical-review.md` | 完整算法推导 + 21 项风险 |
| 2026-07-10/11 | earlier specs | HK sign, parallel, B13, Löwdin |

---

## Next Steps (Priority)

> 2026-07-29 起以 `2026-07-29-deltap-risk-assessment-review.md` §9 的 P0/P1/P2 清单为准。以下旧条目保留备查。

0. **R0 esolver 重构（R1/R2/R3/R4 全部完成）** — 按 `2026-07-31-deltap-esolver-refactor-design.md` 4 轮迁移：~~R1 删死代码~~ → ~~R2 LCAO 抽 `DeltapScfSolver`~~ → ~~R3 PW 接入同一状态机~~ → ~~R4 MPI rank0+Bcast（C-23）、PW λ 同步（C-28）、打印/WARNING 收口、deltap_common 单测~~；设计文档状态已改"已实施"
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
- [x] **Serial smoke test**: Si 2×2×2 SCF + nscf DeltaP — converges, produces deltap_results.dat
- [x] **MPI smoke test**: np=2 Si 2×2×2 nscf — no crash, produces results (exit 0)
- [x] **Unit tests (gauge)**: 4/4 passed
- [x] **Unit tests (math)**: 3/3 passed
- [x] **Unit tests (BFGS)**: 21/21 passed
- [x] **Unit tests (smoothness)**: 4/8 passed (same 4 pre-existing failures as original HEAD)
- [x] **BN 收敛性诊断**: SCF 在 deltap_inner_nmax=20 时收敛，确认内循环解决 BN 不收敛问题
- [ ] **TODO-3b FD力验证**: 需在 HPC 集群上运行
- [ ] **MPI np=2 segfault（报告发现）**: 此环境未复现，需在报告测试环境中获取 stack trace
- [ ] **现有 CI 测试重新生成 reference**: deltap_results.dat 会因口径修正而变化

### BN convergence findings (2026-07-29)
- BN (2×2×2) with scf_nmax=50/deltap_inner_nmax=0 → SCF NOT CONVERGED (physical: γ(λ) nearly flat)
- BN with scf_nmax=100/deltap_inner_nmax=0 → SCF NOT CONVERGED (two-phase mode insufficient)
- BN with scf_nmax=100/deltap_inner_nmax=20 → **SCF CONVERGED** (inner loop works)
- Constraint residual |γ−t| large (~5.8 rad) even after convergence → BN electronic stiffness < 1e-6 Ry/rad² (confirmed zero, matches 07-21 doc)
- Recommendation for BN: use `deltap_inner_nmax >= 10` and accept that |γ−t| will be large due to physics

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

---

## 2026-07-30: P 系列算例构建 A 组（P01/P02/P08/P09）

### What was done
按 `2026-07-30-pseries-test-cases-design.md` 为 4 个 P0 旗舰测试构建自包含算例目录（README/run.sh/cases），未运行 ABACUS。`bash -n` 4/4 通过。提取键（rawG/E_KohnSham/TOTAL-FORCE/DeltaP-PW γ_total）与既有测试输出样例逐一比对一致。

### Files modified this round
- 新增 `可靠性测试设计集/P01-H2O极化率三步裁决/{README.md,run.sh,cases/h2o/{STRU,KPT}}`
- 新增 `可靠性测试设计集/P02-H2O平衡偶极四方对标/{README.md,run.sh,cases/h2o/{STRU,KPT}}`
- 新增 `可靠性测试设计集/P08-约束线性与对称性/{README.md,run.sh,cases/h2o/{STRU,KPT}}`
- 新增 `可靠性测试设计集/P09-无缓存确定性B16复测/{README.md,run.sh,cases/h2o/{STRU,KPT},cases/hbn/{STRU,KPT}}`
- 新增 `docs/superpowers/specs/2026-07-30-pseries-groupA-cases-built.md`（含假设/偏差清单 7 条）

### Key decisions
- P01 α_ref 按任务字面公式实现后取绝对值进判据（dip_cor 记账符号属 F1/F2 阻塞）
- P02 PW 通道 ks_solver=cg（genelpa 仅 LCAO）
- P09 仅 LCAO 通道（PW wrapped γ 比较口径待定），E 判据按 Ry→Ha 换算后 1e-8 Ha

### Next steps
冒烟运行 P02 LCAO 通道与 P09 H₂O 单次；构建 B/C/D 组其余 14 个测试。

---

## 2026-07-30: P 系列算例构建 B 组（P03/P04/P05/P15）

### What was done
按 `2026-07-30-pseries-test-cases-design.md` 为 4 个分子响应测试构建自包含算例目录（README/run.sh/cases），未运行 ABACUS。`bash -n` 4/4 通过；awk 拟合/提取/位移生成逻辑用样例数据冒烟通过；提取键与真实输出样例比对一致。

### Files modified this round
- 新增 `可靠性测试设计集/P03-H2O-Born有效电荷/{README.md,run.sh,cases/h2o/{STRU,KPT},cases/INPUT_lcao_force.tmpl,cases/INPUT_pw_berry.tmpl}`
- 新增 `可靠性测试设计集/P04-小分子偶极组/{README.md,run.sh,cases/{ch4,co,nh3,hf,h2s}/{STRU,KPT},cases/INPUT_lcao.tmpl,cases/INPUT_pw.tmpl}`
- 新增 `可靠性测试设计集/P05-小分子极化率组/{README.md,run.sh,cases/{tzdp,dzp}/{h2o,nh3,ch4,co,hf}/{STRU,KPT},cases/INPUT_efield.tmpl,cases/INPUT_deltap.tmpl}`
- 新增 `可靠性测试设计集/P15-极化率张量各向异性/{README.md,run.sh,cases/{h2o,nh3,co}/{STRU,KPT},cases/INPUT_efield.tmpl,cases/INPUT_deltap.tmpl}`
- 新增 `docs/superpowers/specs/2026-07-30-pseries-groupB-cases-built.md`（含假设/偏差清单 8 条）

### Key decisions
- P15 DeltaP 窗口加 λ=0 点（R² 判据需 ≥3 点）
- P04 CO 符号按任务书"O 端为负"约定（CO_EXPECT_SIGN=-1，README 说明与文献 C⁻O⁺ 冲突）
- P05 dzp 层级轨道缺失 → 运行时自动 SKIP；α 换算链系数集中常量区待 F1/F2 定稿
- awk 数值重建 STRU 需同时设 OFMT/CONVFMT 保 7 位精度

### Next steps
冒烟运行 P04 验证端到端；F1/F2 定稿后回填 P03/P05/P15 常量区；构建 C/D 组。

---

## 2026-07-30: P 系列算例构建 D 组（P11/P12/P13/P14 固体）

### What was done
按 `2026-07-30-pseries-test-cases-design.md` 为 4 个固体测试构建自包含算例目录（README/run.sh/cases），未运行真实 ABACUS。`bash -n` 4/4 通过；用 stub 二进制端到端干跑 4/4（提取/判据/计数/退出码/阻塞 WARNING 全部验证）；P14 的 `wannier90.x -pp` 用真实程序验证 seed.win 模板合法（seed.nnkp 正常生成）。

### Files modified this round
- 新增 `可靠性测试设计集/P11-hBN介电常数/{README.md,run.sh,cases/{STRU,KPT_664,KPT_996,INPUT_lcao.tmpl,INPUT_pw.tmpl}}`
- 新增 `可靠性测试设计集/P12-NaCl-Si-Born有效电荷/{README.md,run.sh,cases/{STRU_NaCl,STRU_Si,KPT,INPUT_lcao.tmpl,INPUT_pw.tmpl}}`
- 新增 `可靠性测试设计集/P13-BaTiO3自发极化/{README.md,run.sh,cases/{STRU.tmpl,KPT,INPUT_lcao.tmpl,INPUT_lcao_target.tmpl,INPUT_pw.tmpl}}`
- 新增 `可靠性测试设计集/P14-wannier90交叉验证/{README.md,run.sh,cases/{STRU_H2O,STRU_NH3,STRU_CH4,STRU_Si,KPT_GAMMA,KPT_SI,INPUT_lcao.tmpl,INPUT_pw.tmpl,INPUT_nscf.tmpl}}`
- 新增 `docs/superpowers/specs/2026-07-30-pseries-groupD-cases-built.md`（含假设/偏差清单 10 条）

### Key decisions
- P11 通道② 选 Z* 路径（周期 PW 无宏观场）；ε 两通道判据随 F1/F2 挂起
- P12 fcc 初基胞 F1 有效盒长取 a/√3（常量区标注待备忘录）；阳离子位移分数化 (δ/a,δ/a,−δ/a)
- P13 PW wrapped γ 用 LCAO Δγ 做 2π 预测-校正；w90 通道仅检测+模板
- P14 w90 管线 scf→-pp→nscf(towannier90)→wannier90 全脚手架，mmn 用 find 通配收集；Si 不进 w90 循环

### Next steps
首次实跑 D 组核对提取正则；F1/F2 定稿后回填 P11/P12 常量区；C 组（P06/P07/P10/P16/P17/P18）构建。

---

## 2026-07-30: P 系列算例构建 C 组（P06/P07/P10/P16/P17/P18）

### What was done
按 `2026-07-30-pseries-test-cases-design.md` 为 6 个收敛/等效测试构建自包含算例目录（README/run.sh/cases），未运行 ABACUS。`bash -n` 6/6 通过；内嵌 awk 全部编译通过；P18 cube 分析双路径（python3/awk）、P17 STRU_ION_D 几何解析、P06 拟合 awk 用合成数据冒烟通过；提取键与真实输出样例逐一比对一致。

### Files modified this round
- 新增 `可靠性测试设计集/P06-盒尺寸收敛/{README.md,run.sh,.gitignore,cases/{KPT,STRU_L12,STRU_L15,STRU_L18,STRU_L21,STRU_L24,INPUT.deltap.tmpl,INPUT.efield.tmpl}}`
- 新增 `可靠性测试设计集/P07-基组与截断双收敛/{README.md,run.sh,.gitignore,cases/{KPT,basis_dzp/STRU,basis_tzdp/STRU,basis_qzdp/STRU,INPUT.deltap.tmpl,INPUT.efield.tmpl,INPUT.pw.tmpl,INPUT.pw_efield.tmpl}}`
- 新增 `可靠性测试设计集/P10-ED曲线约束vs外场/{README.md,run.sh,.gitignore,cases/{KPT,STRU,INPUT.deltap.tmpl,INPUT.efield.tmpl}}`
- 新增 `可靠性测试设计集/P16-PW约束核整改验证/{README.md,run.sh,.gitignore,cases/{KPT,STRU_15BOHR,STRU_30BOHR,INPUT.pw.tmpl,INPUT.lcao.tmpl}}`
- 新增 `可靠性测试设计集/P17-场致几何弛豫对照/{README.md,run.sh,.gitignore,cases/{KPT,STRU,INPUT.relax.deltap.tmpl,INPUT.relax.efield.tmpl}}`
- 新增 `可靠性测试设计集/P18-实空间密度对照/{README.md,run.sh,.gitignore,cases/{KPT,STRU,INPUT.deltap.tmpl,INPUT.efield.tmpl,INPUT.pw.tmpl}}`
- 新增 `docs/superpowers/specs/2026-07-30-pseries-groupC-cases-built.md`（含假设/偏差清单 10 条）

### Key decisions
- P17 弛豫末帧解析 OUT.*/STRU_ION_D（STRU.cif 经源码核实为初始结构不可用）
- P17 的 E 对应值按任务书 F1=−πλ/(2a)（与设计文档 ±0.002/±0.004 不一致，属 F1 阻塞项，常量区单点可改）
- P10 κ 三方量纲不同，输出各自 α 等价量比对；P18 判据在 z 平面平均剖面上计算
- P07 J3 跨通道 μ 比较在 γ 域 mod 2π 最小差后换算；PW 通道显式 nbands 8
- P16 覆盖率无直接输出键（留 TODO），判据只依赖 dγ/dλ 响应

### Next steps
冒烟运行 P07 验证端到端；F1/F2 定稿后回填 P06/P10/P17/P18 常量区；P07 dzp/qzdp 层级轨道待用户补充。

---

## 2026-07-30: P 系列整体验证 + 冒烟修复（KPT/gamma_only/F2）

### What was done
18/18 run.sh `bash -n` 复核通过；真实 ABACUS 冒烟（H₂O 30 Bohr LCAO）发现三个系统性问题并全部修复：
1. **KPT 沿 gdir 需 ≥2 k 点**（Wilson 环 k-string）：分子算例 KPT 全部 `1 1 1`→`1 1 2`；P02/P05/P15 多方向测试 run.sh 改按 gdir 动态生成 KPT。
2. **gamma_only 必须为 0**（源码 `esolver_ks_lcao.cpp:832`：deltap_corr 仅支持 multi-k）：确认全部 39 处 INPUT 模板已为 0。
3. **F2 自旋因子实测 = 2**（nspin=1）：Σγ_raw=−12.718 rad → unwrap(−π,π]=−0.1514 → ÷2 → μ=1.838 D vs 实验 1.855 D（差 0.017 D，判据 0.05 内）。F2 写入 P01/P02/P04/P05/P06/P10/P11/P14/P15 常量区并附证据注释；raw γ→μ 一律先 unwrap。F1 公式同步经源码（`esolver_ks_lcao.cpp:823` E=−πλ/(2a) Ha）确认。
另加 `可靠性测试设计集/.gitignore`（runs/）。

### Files modified this round
- `可靠性测试设计集/P*/cases/**/KPT*`（30 个分子 KPT 改 1 1 2）
- `可靠性测试设计集/{P02,P05,P15}*/run.sh`（per-gdir KPT 生成）
- `可靠性测试设计集/{P01,P02,P04,P05,P06,P10,P11,P14,P15}*/run.sh`（F2 常量区 + unwrap）
- 新增 `可靠性测试设计集/.gitignore`、`docs/superpowers/specs/2026-07-30-pseries-cases-build-round.md`

### Key decisions
- F1/F2 常量区工作值从"占位"升级为"实测依据值"（F2=2、F1=−πλ/(2a)），保留"待备忘录定稿"WARNING 与单点修改机制
- 本机算力不足以跑完整测试矩阵（ecut100+thr 1e-8 单点 >15 min），冒烟用降级设置（ecut50/thr 1e-6，50 s 收敛），生产设置留给 HPC

### Bug/fix list update
- 修复：分子算例 KPT 缺 gdir 方向 k-string（致命，所有分子 DeltaP 算例跑不出 γ）
- 修复：偶极换算缺 unwrap + 自旋因子（不修复则 P02/P04 数值必 FAIL）

### Next steps
1. HPC 完整跑 P02（四通道）与 P09（确定性）两个无前置 P0
2. P08 九点扫描定 λ 窗口 → 喂 P01
3. F3（frozen-λ 能量记账）核实后备忘录定稿
4. PW 通道小成本冒烟（验证 γ_total 提取端到端）
5. groupA–D 文档假设/偏差清单逐条复核关闭

---

## 2026-07-31: esolver 侧重构设计评审（只审不改）

### What was done
完整审查 DeltaP 的 esolver 相关文件（LCAO/PW 两路径 + 公共头 + operator），输出重构设计方案 `2026-07-31-deltap-esolver-refactor-design.md`。**本轮未改任何运行时代码**。

审查结论（详见设计文档 §1）：
- 死代码：`deltap_solver.h` 0 引用；`deltap_common.h` 4 函数仅 1 个被用；PW `run_deltap_lambda_loop` no-op、`s_hamilt`/`set_deltap_pw_hamilt` 整组死代码、`inner_nmax>0` 后死分支（WARNING_QUIT 使其不可达）、`compute_per_atom_gamma_from_becp` 无调用者；LCAO `deltap_init` 三个 raw `new` 无 delete。
- 重复实现：残差（4+ 处）、λ_eff=Cᵀλ（3+ 处）、梯度下降+mixing（LCAO/PW 各一）、2π 分支跟踪（LCAO/PW 各一）、escon（两处）。
- 双状态机：LCAO 类成员 + flags vs PW 文件级全局单例，P1→P2→P3 门控细节不一致（drho>0 vs drho<=0 边界），离子步重置时机不一致。
- 结构：`iter_finish`（esolver_ks_lcao.cpp:695-835）~140 行混 6 种职责；`if constexpr` 的 else 分支对 real 实例每迭代重复 WARNING。
- 潜伏 bug：`deltap_constraint_lambda_` 回退维度 nat→m（C-22）；约束矩阵尺寸不匹配仅 cerr；target/矩阵每 rank 各读文件（C-23）。

设计方案核心：抽 `deltap_scf::DeltapScfSolver` 状态机（basis-independent 控制流 + Backend 回调注入基组操作），`deltap_common.h` 收拢为纯函数唯一实现，PW 全局单例改实例，ESolver 只留 ≤10 行接线。迁移分 4 轮（R1 删死代码 → R2 LCAO 抽状态机 → R3 PW 接入 → R4 MPI 收敛 + 修 bug），每轮独立可编译可回归，stdout 逐字符保真（测试解析依赖）。

### Files modified this round
- 新增 `docs/superpowers/specs/2026-07-31-deltap-esolver-refactor-design.md`（审查 + 目标架构 + 4 轮迁移 + 风险验收）
- 更新 `docs/superpowers/specs/deltap-development-log.md`（File Map / Spec Index / Active Bugs C-21~C-26 / Next Steps）

### Key decisions
- 数值核心 `module_deltap`（Wilson loop/gauge/branch/HK correction）不进入本次重构范围，降低回归风险
- `DeltapScfSolver` 以回调注入基组差异（set_lambda/set_hk_correction/compute_gamma/solve_frozen/sync_rho_from_dm/reset_charge_mixing），控制流一份，LCAO/PW 共用
- `select_branch_set`（多带权重版）保留在 module_deltap，只把"跨 SCF 步最近分支"统一为 `unwrap_2pi`
- 输出格式是测试解析锚点，重构全程逐字符保真迁移

### Bug/fix list update
- 新增 Active Bugs：C-21（LCAO raw new 泄漏）、C-22（constraint_lambda 回退维度错）、C-23（target/矩阵每 rank 各自读文件）、C-24（PW 死代码组）、C-25（deltap_solver.h/deltap_common.h 死代码）、C-26（iter_finish 巨型块 + 重复 WARNING）

### Next steps
1. R1：删死代码 + common 唯一化（零行为变化，P01 冒烟 + stdout diff 验收）
2. R2：LCAO 抽取 `DeltapScfSolver`（esolver_ks_lcao.cpp 净删 ≥350 行目标）
3. R3：PW 接入同一状态机（deltap_pw.cpp 净删 ≥150 行目标）
4. R4：MPI rank0+Bcast 读文件、修 C-22、real 实例 WARNING 收口

---

## 2026-07-31: 重构 R1 实施完成（删死代码 + 修泄漏，零行为变化）

### What was done
按设计文档 Round 1 实施 esolver 侧 DeltaP 清理，7 个文件 **+53/−388**：

1. 删 `source/source_esolver/deltap_solver.h`（全仓库 0 引用）。
2. `deltap_common.h` 155→19 行：删未用函数 `update_lambda`/`to_effective_lambda`/`compute_max_residual`，仅保留 `compute_dp_escon`。
3. PW `deltap_pw.cpp/.h` −169 行：删 `run_deltap_lambda_loop`（no-op）、`s_active`/`set_deltap_pw_active`/`is_deltap_pw_active`（只写不读）、`s_hamilt`/`set_deltap_pw_hamilt`、`inner_nmax>0` 后不可达 inner-loop 死分支、`compute_per_atom_gamma_from_becp`（无调用者）。
4. **行为保留**：`set_deltap_pw_hamilt` 内含的"每 SCF 周期重置"重命名为 `reset_deltap_pw_scf_cycle()` 并在原调用点保留；`inner_nmax>0` 的 `WARNING_QUIT` 拒绝语义保留。
5. LCAO 修 C-21 泄漏：`void* dp_scf_/berry_ovl_scf_/r_overlap_scf_` + raw `new` → `std::unique_ptr` + 头文件前向声明，所有 `static_cast` 改 `.get()`。

### Test setup
本机 Release + MPI1 进程；LCAO 冒烟（H₂O 30 Bohr，ecut50/thr1e-6，deltap 开，λ=0）与 PW 冒烟（ecut30，berry_phase=1，total 模式）各跑新旧两版二进制采集 A/B 日志。

### Results
- 编译：esolver / module_pwdft / abacus_basic_para 全 PASS（仅既有 if constexpr 警告）。
- LCAO 冒烟 48 s 收敛，PW 冒烟 40 s 收敛，exit=0。
- A/B diff：DeltaP 行（`[DeltaP P1]`/`[rawG]`/`[E-field]`/`[DeltaPOp]`/`[DeltaP-PW]`）逐字节 **IDENTICAL**；全日志仅日期/计时列差异（非确定项）。

### Files modified this round
- `source/source_esolver/deltap_solver.h`（删除）
- `source/source_esolver/deltap_common.h`（精简 155→19）
- `source/source_pw/module_pwdft/deltap_pw.{h,cpp}`（删死代码）
- `source/source_esolver/esolver_ks_pw.cpp`（删 2 个调用 + 改名 reset）
- `source/source_esolver/esolver_ks_lcao.{h,cpp}`（unique_ptr 化）
- 新增 `docs/superpowers/specs/2026-07-31-deltap-esolver-refactor-r1.md`

### Bug/fix list update
- Fixed：C-21（泄漏）、C-24（PW 死代码组）、C-25（deltap_solver.h + common 未用函数）
- Open 保留：C-22（constraint_lambda 维度）、C-23（MPI 读文件）、C-26（iter_finish 巨型块，R2 处理）

### Next steps
1. R2：新增 `deltap_scf.{h,cpp}`（DeltapScfSolver 状态机），挂 CMakeLists/Makefile.Objects；迁入 `deltap_init`/`deltap_compute_gamma`/`deltap_inner_loop`/`deltap_update_lambda`；iter_finish 收缩
2. R3：PW 接入同一状态机
3. R4：MPI rank0+Bcast、修 C-22、WARNING 收口

---

## 2026-08-01: 重构 R2 实施完成（DeltapScfSolver 状态机上线）

### What was done
把 LCAO esolver 的 DeltaP SCF 控制流抽到新组件 `source/source_esolver/deltap_scf.{h,cpp}`：

1. **`DeltapScfSolver` 状态机**（~470 行）：持有 `DeltapParams`（INPUT 快照 + target/C/t）与
   `DeltapState`（λ_eff/λ_cstr/γ/flags/escon），实现 `init` / `reset_ionic_step` /
   `inner_loop`（BFGS 冻结密度）/ `iter_finish`（γ 测量 → P2 梯度下降 → escon/HK → 报告）。
  基组差异经 `Backend` 回调注入：set_lambda / get_lambda / apply_hk_correction /
  compute_gamma / solve_frozen / sync_lambda / on_phase2 / get_optimizer /
  compute_gamma_raw / lattice_period。
2. **`deltap_common.h` 重建为纯函数库**：`compute_residual` / `max_norm` / `gd_update` /
   `gd_update_total` / `to_effective_lambda` / `compute_dp_escon`，全部有调用者。
3. **ESolver 瘦身**：8 成员 + 3 flags → 1 个 `unique_ptr<DeltapScfSolver>`；4 个 helper →
   `deltap_init`（基建 + 参数快照 + backend 绑定）+ `deltap_make_backend`（10 个一行回调）；
   `iter_finish` DeltaP 块 ~140 行 → 22 行接线；`hamilt2rho_single` 改
   `skip_solve = deltap_scf_solver_->inner_loop(drho)`。
4. 挂载：`source/source_esolver/CMakeLists.txt` 加 `deltap_scf.cpp`（Makefile.Objects 目录 glob 自动覆盖）。

### Test setup
本机 Release + MPI1；同步路径（h2o_lcao，nscf=0，λ=0）R2 vs R1 二进制 A/B diff；
内循环路径（h2o_inner，nscf=4 + target.dat）R2 功能冒烟。

### Results
- 编译 PASS（含 `if constexpr` 保护 complex-only 回调，double 实例正常实例化）。
- 同步路径：DeltaP 行（[DeltaP P1]/[rawG]/[E-field]/[DeltaPOp]）**IDENTICAL**；
  全日志仅日期/计时列差异。
- 内循环：`inner loop start: nscf=4` → `inner loop done: final l0=-1.30e-2 ...`，SCF 收敛。
- `iter_finish` 块 140→22 行；`esolver_ks_lcao.cpp` 净删 ~430 行。

### Key decisions
- 复刻悬空 else 打印语义（has_any_target=false 不打印；true 且非 rank0 打印 "No targets"），
  保 A/B 全等，R4 清理。
- BFGS 对象仍归 `DeltaP` 持有，状态机经 `get_optimizer` 回调驱动；`solve_frozen` 每 inner
  迭代新建 HSolverLCAO（无状态依赖，等价）。
- `lambda_cstr` 初始化按矩阵模式 m 维（连带修 C-22）；内循环 per-atom 残差空 target 越界
  UB 修复（非 UB 路径不变）。

### Files modified this round
- 新增 `source/source_esolver/deltap_scf.{h,cpp}`、`source/source_esolver/CMakeLists.txt`（+1 行）
- `source/source_esolver/deltap_common.h`（纯函数库重建）
- `source/source_esolver/esolver_ks_lcao.{h,cpp}`（成员收拢 + 状态机接线）
- 新增 `docs/superpowers/specs/2026-08-01-deltap-esolver-refactor-r2.md`

### Bug/fix list update
- Fixed：C-22（lambda_cstr 维度）；C-26 Partial（iter_finish 已收缩，WARNING 收口留 R4）
- Open 保留：C-23（MPI 读文件）

### Next steps
1. R3：PW 接入 `DeltapScfSolver`（删全局单例 `s_lambda_set` 等，backend = k-string Wilson loop，`deltap_common` 增 `unwrap_2pi`）
2. R4：MPI rank0+Bcast（C-23）、悬空 else 打印清理、real 实例 WARNING 收口
3. `deltap_common` 单测（source/source_esolver/test/）

---

## 2026-08-01: 重构 R3 实施完成（PW 接入同一 DeltapScfSolver 状态机）

### What was done
把 PW 侧 DeltaP 的 SCF 控制流从文件级全局单例迁移到 R2 的 `DeltapScfSolver` 状态机：

1. **状态机泛化两点**（对 LCAO 零影响，A/B 已证）：
   - `DeltapParams::unwrap_branch_2pi` + `DeltapState::gamma_prev`：PW 的跨 SCF 最近分支
     2π unwrap 移入状态机（`deltap_common::unwrap_2pi` 纯函数），`reset_ionic_step()` 一并清空。
   - `DeltapState::gamma_report`：分支选择后的 γ 用于 max_res/escon/report；LCAO 的
     branch selection 在 `compute_gamma` 内已完成 → gamma_report == gamma_I。
2. **`deltap_pw.cpp` 重写**：7 个文件级状态（`s_lambda_set`/`s_gamma_total`/`s_dp_escon`/
   `s_gamma_prev`/`s_targets`/`s_lambda`/`s_constrain`）→ 2 个算子状态（`s_lambda`/`s_constrain`，
   forces/stress/op_pw_proj 消费）+ 匿名 namespace 唯一 `DeltapScfSolver` 实例。
   `make_backend`：set_lambda 写回 `s_lambda`、compute_gamma = `compute_per_atom_gamma_kstring`
   （折叠 gdir 的 1D per-atom γ）；`compute_total_gamma_pw`/`compute_per_atom_gamma_kstring`
   移入匿名 namespace（对外 0 引用）。
3. **`esolver_ks_pw.cpp`**：init 块 20 行 → `pw_deltap::deltap_init(ucell, inp, psi_cpu, kv, wfcpw, rhopw)` 6 行；
   `deltap_iter_finish`/`reset_deltap_pw_scf_cycle`/`get_deltap_pw_escon` 调用点原样保留。
   `deltap_pw.h` 公开面收敛：删 `set_deltap_pw_lambda/targets`/`get_deltap_pw_targets`，加 `deltap_init`。

### Test setup
本机 Release + MPI1；PW 冒烟（h2o_pw：ecut30、berry_phase=1、gdir=3、total 模式 INPUT、无 STRU target）
R3 vs R1 基线 A/B；LCAO 同步（h2o_lcao）R3 vs R2 A/B；LCAO 内循环（h2o_inner，nscf=4）功能冒烟。

### Results
- 编译 PASS（仅既有 `if constexpr` 警告）。
- PW A/B：`[DeltaP-PW]` 两行 + CG1–CG13 能量/EDiff/DRHO 逐字节 **IDENTICAL**；
  仅墙钟列差异；耗时 38.60s → 35.75s（删重复第二次 γ 测量）。
- LCAO 同步：1909 行 DeltaP/[rawG]/[E-field] **IDENTICAL**。
- LCAO 内循环：`inner loop done: final l0=-1.3039e-02 ...`（与 R2 一致），GE39 DRHO=6.4e-07 收敛。
- 运行日志：`/tmp/r3_pw.log`、`/tmp/r3_lcao.log`、`/tmp/r3_inner.log`。

### Key decisions
- PW 无 target = 约束 γ→0：历史 `targets[iat]` 空 vector 越界读（UB，实际按 0）；R3 显式填 0 向量（修 C-27，行为不变）。
- `[DeltaP-PW]` 报告逐字节保真：掩码 max_res/λ_avg/γ/atom 由 `report_pw` 按历史语义计算；
  状态机内 max_res 对 PW 不打印（target 空 → 0）。
- `inner_nmax>0` WARNING_QUIT 消息逐字保留，触发点仍在 `deltap_iter_finish`（switch&&corr 门控后），
  比历史略早（不再先算 γ）。
- PW 保持 per-atom λ 更新（`p.total_mode=false`）；INPUT total 模式被忽略的历史不一致保留，
  标注留后续统一（R4+ 可选）。
- PW 仍无 MPI λ Bcast/rank 守卫（S-09），R4 处理；状态机对 PW 不启用 sync_lambda/on_phase2/HK。

### Files modified this round
- `source/source_esolver/deltap_common.h`（+`unwrap_2pi`，含 constants.h）
- `source/source_esolver/deltap_scf.{h,cpp}`（+`unwrap_branch_2pi`/`gamma_report`/`gamma_prev`；iter_finish 顺序微调）
- `source/source_pw/module_pwdft/deltap_pw.{h,cpp}`（全局单例 → 状态机实例 + backend；公开面收敛）
- `source/source_esolver/esolver_ks_pw.cpp`（init 接线 20→6 行）
- 新增 `docs/superpowers/specs/2026-08-01-deltap-esolver-refactor-r3.md`

### Bug/fix list update
- Fixed：C-27（PW 空 target 越界 UB）；`s_gamma_total` 死状态删除；NaN-γ_total 路径不再烧 `lambda_set`（清理）
- Open 保留：C-23（MPI rank0+Bcast 读文件）；S-09（PW 无 λ Bcast/rank 守卫）；LCAO 悬空 else 打印；real WARNING 收口

### Next steps
1. R4：MPI rank0+Bcast（C-23）、PW λ Bcast/rank 守卫（S-09）、悬空 else 清理、real WARNING 收口、
   `set_deltap_pw_*` 残余接口收口（setter 已删，getter 保留给算子消费者）
2. 可选：PW 接 `deltap_target_file`/`deltap_constraint_matrix`；PW total 模式统一；`deltap_common` 单测
3. 总设计文档（`2026-07-31-deltap-esolver-refactor-design.md`）状态改为"已实施"

---

## 2026-08-01: 重构 R4 实施完成（MPI 收敛 + 收口 + 单测）

### What was done
1. **C-23：`DeltapScfSolver::init` rank0 读 + Bcast**。target 文件与约束矩阵文件改为
   rank0 解析后经 `Parallel_Common::bcast_*`（MPI_COMM_WORLD）广播：target 广播 loaded/total/向量；
   矩阵广播 file_open/loaded/m/n + C/t 数据，各 rank 重建同一 `params_`。文件缺失仍静默保留 STRU 目标。
2. **约束矩阵尺寸不匹配 → `WARNING_QUIT`**（rank0，文案保留；历史为 cerr 后静默丢弃约束）。
3. **S-09/C-28：PW λ Bcast + rank 守卫**。backend 增 `sync_lambda`（`NPROC>1` 时 Bcast rank0 值，
   随后 `s_lambda = lam` 刷新全 rank 算子存储 → escon 一致）；`deltap_init` 消息与 `report_pw` 打印
   加 `MY_RANK==0` 守卫（消除多 rank 重复输出）。
4. **LCAO 打印/WARNING 收口**：悬空 else 清理（STRU target 打印改 `has_any_target && MY_RANK==0` 一条）；
   real 实例 WARNING 从 `iter_finish` 每迭代重复 → `before_all_runners` 一次性（rank0）。
5. **`deltap_common` 单测**：新增 `source/source_esolver/test/deltap_common_test.cpp`（10 用例），
   挂 `MODULE_ESOLVER_deltap_common_test`（LIBS parameter/math_libs/base/device），全部 PASSED。

### Test setup
本机 Release + MPI1；三条冒烟（h2o_pw / h2o_lcao / h2o_inner）R4 二进制 vs 既有基线；
单测 `MODULE_ESOLVER_deltap_common_test`。

### Results
- 编译 PASS（仅既有 `if constexpr` 警告）。
- PW A/B：`[DeltaP-PW]` 两行 + CG1–13 能量/EDiff/DRHO 一致（仅墙钟列不同）。
- LCAO 同步：DeltaP/[rawG]/[E-field] IDENTICAL；内循环：`final l0=-1.3039e-02` 一致，GE39 收敛。
- 单测：10/10 PASSED。
- 日志：`/tmp/r4_pw.log`、`/tmp/r4_lcao.log`、`/tmp/r4_inner.log`。

### Key decisions
- 文件 Bcast 用 `Parallel_Common`（MPI_COMM_WORLD），与 LCAO λ Bcast 同通信域；
  无 MPI 构建走纯 rank0 路径（单进程）。
- PW `sync_lambda` 在 Bcast 后写回 `s_lambda`，使 `get_lambda`/escon 全 rank 一致
  （LCAO 的 sync_lambda 只改局部副本，属既有模式，未动）。
- γ 测量（Wilson loop）跨 rank 一致性未在本轮核对（单进程环境），文档标注留后续。

### Files modified this round
- `source/source_esolver/deltap_scf.cpp`（init rank0+Bcast；矩阵尺寸错 WARNING_QUIT）
- `source/source_pw/module_pwdft/deltap_pw.cpp`（sync_lambda + rank 守卫）
- `source/source_esolver/esolver_ks_lcao.cpp`（悬空 else 清理；real WARNING 收口到 before_all_runners）
- 新增 `source/source_esolver/test/deltap_common_test.cpp` + `test/CMakeLists.txt`（+1 AddTest）
- 新增 `docs/superpowers/specs/2026-08-01-deltap-esolver-refactor-r4.md`
- `2026-07-31-deltap-esolver-refactor-design.md` 状态 → 已实施

### Bug/fix list update
- Fixed：C-23（rank0+Bcast）、C-28（PW λ 同步/rank 守卫，即 S-09）、约束矩阵尺寸错→WARNING_QUIT、
  悬空 else 打印、real WARNING 每迭代重复
- Open 保留：PW γ 测量跨 rank 一致性核对（新）；PW 接 target/约束矩阵文件、PW total 模式统一（可选增强）

### Next steps
1. 4 轮重构全部完成；设计文档已标"已实施"。总验收核对：esolver_ks_lcao DeltaP 净删 ~430 行、
   deltap_pw 净删 ~150 行、deltap_solver.h 消失、deltap_common 全函数被调用且有 10 例单测 — 满足。
2. 可选：2-rank MPI 冒烟（C-23/S-09 实跑验证）；PW 文件 target/矩阵接线；PW total 模式统一。

---

## 2026-08-01: 重构 R5 补算例 + 补文档 + 全量测试验证

### What was done
1. **复验既有算例**（新二进制）：test_C_I / test_C_total / test_stru_target / center /
   deltap_pw_h2o / deltap_bn_test（MPI 4-rank）全部跑通；test_C_I 在约束矩阵模式下
   GE14–16 能量/DRHO 与旧日志逐字节一致（重构未改变 SCF 物理）。
2. **新增算例（入库，验证通过）**：
   - `tests/deltap_bn_sampling/test_mask/`：STRU `dp_constrain 0/1` 混合掩码
     → P3 起 λ=(2.58e-03, 0.0e+00)，N 原子 λ 恒 0（掩码端到端生效）。
   - `tests/deltap_bn_sampling/test_gdir2/`：`deltap_gdir 2` → γ 收敛到 target
     （|γ-t|≤2.8e-4），GE50 DRHO=3.4e-08。
   - `tests/deltap_relax/`：relax 多离子步复现器（force 路径既有 double-free）。
3. **修复既有空指针 bug**：`DeltaPOperator` 构造函数对 `hR=nullptr` 无守卫
   （`FORCE_STRESS.cpp` 力计算路径必崩）→ 加 `hR ? hR->get_paraV() : nullptr`。
4. **MPI 实跑**：LCAO 4-rank（方阵）通过（C-23 rank0+Bcast）；PW 2-rank 通过
   （C-28 rank 守卫：`[DeltaP-PW]` 仅打印一次，λ Bcast 后各 rank escon 一致）。
5. **单测回归**：`deltap_common` 10/10、`esolver_dp` 6/6。
6. 文档：新增 R5 dated 文档（含完整功能矩阵 21 项）。

### Test setup
本机 Release + MPI；`build/abacus_basic_para`（含 R5 空指针守卫）。
运行目录 `/tmp/deltap_r5/`；BN 用例 ecutwfc=100、KPT 2×2×2、scf_thr 1e-8、scf_nmax 50。

### Results
- 通过：test_C_I（GE16 收敛）、test_C_total（GE16）、test_stru_target（内循环跑通）、
  center（P1→P3 完整）、test_mask、test_gdir2、deltap_bn_test（MPI4）、
  deltap_pw_h2o（1/2-rank，FINAL_ETOT -442.02440453 eV）、单测 16/16。
- total 模式（total_0.0）补跑通过：λ 更新数学与 6beb70bc3 逐行一致（非回归）；
  旧 total_0.0.log 基线早于 total 提交 `8e73e0f82`（目标均分语义变更），无法 A/B。
- 失败（既有 bug，非回归）：deltap_relax 在 force 路径 double-free；
  LCAO 2-rank 被既有方阵网格限制拦截（改用 4-rank）。
- 旧日志 A/B 结论：仓库旧 `.log` 与当前 INPUT 参数不一致（旧 run 用 step=0.5/
  mixing=0.5/inner_nmax=20）+ `deltap_branch.dat` 状态漂移 → 不可逐字节对比；
  以物理判据 + 能量锚点验收（test_C_I GE14–16 逐字节一致作为最强证据）。

### Files modified this round
- `source/source_lcao/module_operator_lcao/deltap_lcao.cpp`（+空指针守卫，7 行）
- 新增 `tests/deltap_bn_sampling/test_mask/{INPUT,KPT,STRU,target.dat,README.md}`
- 新增 `tests/deltap_bn_sampling/test_gdir2/{INPUT,KPT,STRU,target.dat,README.md}`
- 新增 `tests/deltap_relax/{INPUT,KPT,STRU,README.md}`
- 新增 `docs/superpowers/specs/2026-08-01-deltap-esolver-refactor-r5-test-coverage.md`
- 刷新各用例目录 `.log` 基线（gitignore，不入库）

### Bug/fix list update
- Fixed（R5）：DeltaPOperator 构造空指针守卫（force/relax 路径首个必崩点）。
- Open：relax force 路径 double-free（`cal_force_stress` OMP 区，混合基组 nlm 越界读，
  重构前 `85b2af322` 引入）；LCAO 2-rank 方阵网格限制（`cbb37b7ae` 引入）；
  PW 未接 target/约束矩阵文件；PW total 模式假 total（均历史已知）。

### Next steps
1. 修 `cal_force_stress` 越界读 → 复跑 deltap_relax → run_fd.sh FD 验收（C-02）。
2. 补跑 gdir=1（deltap_compare）新二进制基线（total_0.0 已在 R5 完成）。
3. PW 接 target/约束矩阵文件 + total 模式统一（F13/F14）。
4. P01–P18 用例（6beb70bc3）纳入 CI 脚本。

---

## 2026-08-01: 最新两 commit 评审 + 后续 TODO 汇总

### What was done
评审 `b825fed52`（R1-R4 重构主体）与 `139380f64`（R5 测试+修崩），输出完整评审意见与
分级 TODO 清单（P0 修 bug / P1 功能补齐 / P2 测试验证 / P3 增强），见
`2026-08-01-deltap-commit-review-and-todo.md`。源码未动。

### 评审结论
- 两 commit 均可合入：A/B 逐字节保真证据充分（每轮 diff + test_C_I 能量锚点），
  既有 bug 与回归划分清晰，文档符合规范。
- 遗留观察项（非阻塞）：total 模式无旧基线可 A/B（仅靠逐行比对）；PW γ 跨 rank
  一致性未核对。

### TODO 汇总（详见评审文档 §2）
- **P0**：① `cal_force_stress` nlm 越界读 → relax → FD 力验收（C-02）；
  ② C-01~C-20 未关闭项逐条核对（C-01 dp_escon rank0、C-12/13 relax λ 重置等）
- **P1**：③ PW 接 target/约束矩阵文件；④ PW total 模式统一；⑤ PW γ 跨 rank
  一致性核对；⑥ LCAO 2-rank 方阵网格限制（C-06）
- **P2**：⑦ P01–P18 纳入 CI；⑧ gdir=1 / total 新基线；⑨ B16 Hungarian 确定性
  运行时诊断；⑩ HPC 跑 P02/P09 + F3 备忘录定稿
- **P3**：⑪ `compute_S_dk_link` 接线或删除；⑫ BTO W90 全量验证

### Files modified this round
- 新增 `docs/superpowers/specs/2026-08-01-deltap-commit-review-and-todo.md`；更新本文档。

---

## 2026-08-01: 提交评审（b825fed52 + 139380f64）与 TODO 分级

### What was done
对 R1–R4 重构 commit（`b825fed52`）与 R5 commit（`139380f64`）做完整评审：
逐文件 diff + 与重构前基线（`6beb70bc3`/`aeb9a0f9c`）行为比对 + MPI/边界路径推演。
完整评审意见见 `2026-08-01-deltap-commit-review-and-todo.md`。

### 评审结论
- 总体：方向正确、实现质量良好，**无 P0**；净删 ~850 行、单测 16/16、A/B 冒烟一致。
- 主要发现（均非回归，多为既有行为或遗留）：
  1. LCAO `sync_lambda` Bcast 后未写回 operator（与 PW R4 修复不一致）→ 非 rank0 λ/escon 可能不一致。
  2. PW `deltap_init` 每离子步在 `before_all_runners` 重复执行 → 状态重置、λ 归 `lambda_init`，
     与 LCAO（惰性 init + reset）语义不一致。
  3. PW γ 测量跨 rank 一致性未核对（R4 遗留）。
  4. force 路径双释放（既有，R5 已记录）为 P1 专项。
  5. `DeltapState::lambda_eff` 死字段；target 文件 EOF 不校验；PW 无 on_phase2（均 P2/P3）。

### TODO 汇总（详见评审文档 §3）
- P0：无。
- P1（4 项）：T1 LCAO sync_lambda 写回；T2 PW init 惰性化；T3 PW γ 跨 rank 核对；
  T7 force 路径双释放修复（C-02 专项）。
- P2（4 项）：T4 lambda_eff 死字段；T5 PW on_phase2 统一；T6 target EOF 校验；T8 PW 文件/约束/total 接线。
- P3（5 项）：T9 注释修正；T10/T11 空 vector 防御；T12 注释规则；T13 用例断言/CI。

### Files modified this round
- 新增 `docs/superpowers/specs/2026-08-01-deltap-commit-review-and-todo.md`
- 本 dev-log 追加本节

### Next steps
1. T1（LCAO sync_lambda 写回）+ T3（PW γ 跨 rank 核对）可合成一次 MPI 一致性小迭代，
   2-rank/4-rank 实跑验收。
2. T2（PW init 惰性化）随 relax/多离子步支持一起做。
3. T7（force 路径）为最大专项：nlm 布局统一 + 长度守卫 + H_HK/∂τ/∂R 力项 + run_fd.sh 验收。
4. T4–T6、T9–T13 为清理项，可随日常迭代顺手合入。

---

## 2026-08-01: MPI 一致性迭代（T1 LCAO λ 写回 + T3 PW γ 同步）

### What was done
执行评审 TODO 第一轮：T1（LCAO `sync_lambda` Bcast 后写回 `dp_op` λ）与 T3
（PW `compute_gamma` 后按 rank0 Bcast γ），合成一次 MPI 一致性小迭代。源码改动
2 文件，详见 `2026-08-01-deltap-mpi-consistency-t1-t3.md`。

### 验收结果
- 单测 16/16；PW 1-rank 与 R5 基线逐字节一致（T3 对 NPROC=1 no-op）。
- PW 2-rank：per-rank 诊断证明两 rank escon/γ 完全一致；rank0 输出与 1-rank 基线
  浮点噪声内一致；最终无诊断复跑与带诊断复跑逐字节一致（确定性）。
- LCAO 4-rank（deltap_bn_test 方阵网格）：per-rank 诊断证明四 rank λ 完全一致
  `(3.262e-06, -3.501e-06)`；P2 iter=11 drho=6.92e-06 与 R5 记录一致；P1 轨迹比
  旧日志更平滑（消除旧 rank 间 λ 不一致导致的 iter=2 扰动）。
- h2o_lcao 4-rank 崩溃排查：stash 掉改动后旧二进制同样在进程 1 静默 exit=1 →
  **既有问题**（非本次引入），与 LCAO 2-rank 方阵限制（`cbb37b7ae`）同族。

### TODO 状态更新
- **T1 ✅ 完成**（LCAO sync_lambda 写回）；**T3 ✅ 完成**（PW γ rank0 Bcast）。
- T2（PW init 惰性化）、T7（force 路径专项）待做；T4–T6、T9–T13 为清理项。

### Files modified this round
- `source/source_esolver/esolver_ks_lcao.cpp`（T1）
- `source/source_pw/module_pwdft/deltap_pw.cpp`（T3）
- 新增 `docs/superpowers/specs/2026-08-01-deltap-mpi-consistency-t1-t3.md`；本 dev-log 追加本节

### Next steps
1. T2（PW init 惰性化 + 每离子步 reset）随 relax/多离子步支持推进。
2. T7（force 路径）最大专项：nlm 布局统一 + 长度守卫 + H_HK/∂τ/∂R 力项 + run_fd.sh 验收。
3. 观察项：PW γ 测量 nproc 敏感性（~0.02%）如影响数值基线再评估。
4. 清理项 T4–T6、T9–T13 随日常迭代合入。

---

## 2026-08-02: T1+T3 迭代（d983206a1）严格评审 + TODO 修订

### What was done
对 T1+T3 MPI 一致性迭代做批判性评审，输出 `2026-08-02-deltap-mpi-consistency-review.md`。
源码未动。

### 评审结论（五条批评）
1. **T3 验收同义反复**：Bcast 后各 rank γ 必然一致，缺 pre-fix 不一致证据；
   `unkdotp_G` 本有 POOL_WORLD Allreduce，T3 可能修的是不存在的 bug。
2. **KPAR>1 未分析（真风险）**：`bcast_double` 硬编码 MPI_COMM_WORLD root=0；
   pool 分 k 时 pool0 的 γ 完整性未验证，若无完整 γ 则 T3 传播错值；无守卫。
3. **T1 因果归因弱**：旧日志 iter=2 扰动跨重构，变量不唯一；缺 pre-fix LAMDBG。
4. **nproc 敏感性被回避**：并行基线=rank0 值后不复现串行基线，CI 策略未定。
5. **h2o_lcao 4-rank 崩溃分类不充分**：无 backtrace 即归入 `cbb37b7ae` 同族。
正面：stash 对照规范、单测 16/16、1-rank 逐字节保真、文档合规。

### TODO 修订（D 系列，替代 T2/T7 二选一）
| # | 事项 | 优先级 |
|---|------|--------|
| D1 | KPAR>1 守卫或 γ 完整性分析 | P0（最优先） |
| D2 | h2o_lcao 4-rank 崩溃 backtrace 分类 | P0 |
| D3 | T1/T3 补 pre-fix 证据或改措辞为"防御性加固" | P1 |
| D4 | nproc 基线策略定稿（CI 固定 nproc；escon 差异定性） | P1 |
| D5 | T7 force 专项三步：(a) nlm 修复→relax 不崩；(b) FD 验收；(c) H_HK/∂τ/∂R 力项（C-02 收尾） | P0（欠账后启动） |
| D6 | T2 PW 惰性 init 并入 D5(a)（无 T7 无消费方） | — |

### Files modified this round
- 新增 `docs/superpowers/specs/2026-08-02-deltap-mpi-consistency-review.md`；本文档追加本节。

---

## 2026-08-02: MPI 一致性 commit（d983206a1）评审 + D1–D6 修订

### What was done
评审 `d983206a1`（T1 LCAO λ 写回 + T3 PW γ 同步）：逐行 diff + 调用链推演 +
与上轮验证记录交叉核对。完整评审意见见
`2026-08-02-deltap-mpi-consistency-review.md`。源码未动。

### 评审结论
- 总体：可保留，无回归（PW 1-rank 逐字节一致、单测 16/16、跨 rank 一致经实跑证明）。
- 正面：T1 写回幂等语义正确；rank0-wins 事实源契约两端统一；验证方法学扎实
  （临时诊断→证明→移除、确定性复跑、stash 对照归因）；文档规范。
- 五条批评：
  1. C1 PW γ Bcast 用 MPI_COMM_WORLD、LCAO 用 POOL，通信域不统一；`unkdotp`
     的 KPAR=1 注释与 2-rank(KPAR=2) 实跑矛盾，约束未固化。
  2. C2 `sync_lambda` 只覆盖同步模式；内循环 `inner_loop` 的 `set_lambda`
     （deltap_scf.cpp:210,235）不走同步，多 rank 内循环从未验证。
  3. C3 LCAO 侧只验 λ 未验 escon/γ_report（依赖 module_deltap 内部 Bcast，
     缺实跑断言）。
  4. C4 2-rank PW 数值基线未固化（γ 差 0.02%、escon 差 7e-6 Ry），nproc 敏感
     性未立项追根因。
  5. C5 临时诊断已删，无持久化跨 rank 一致性校验，MPI 冒烟未入 CI。

### D1–D6 修订 TODO
| ID | 级别 | 事项 | 状态 |
|---|---|---|---|
| D1 | P1 | 统一 PW/LCAO 同步通信域（POOL）+ 显式化 KPAR/psi 布局约束（修正 `unkdotp` 陈旧注释） | 待做 |
| D2 | P1 | 内循环路径 λ 同步下沉（`inner_loop` 的 `set_lambda` 后接 `sync_lambda`） | 待做 |
| D3 | P2 | LCAO 4-rank 补 escon/γ_report per-rank 断言（闭环 T1 验收） | 待做 |
| D4 | P2 | 固化 2-rank PW 参考值 + nproc 敏感性根因专项 | 待做 |
| D5 | P2 | 跨 rank 一致性校验沉淀为可复用诊断 + MPI 冒烟入 CI | 待做 |
| D6 | P3 | `f_en.dp_escon` rank 守卫/注释 + 空 λ vector 防御统一 | 待做 |

### Files modified this round
- 新增 `docs/superpowers/specs/2026-08-02-deltap-mpi-consistency-review.md`；本 dev-log 追加本节

### Next steps
1. D1+D2+D3 合成一次 MPI 迭代：通信域统一 + 内循环同步 + LCAO escon/γ 断言。
2. D4/D5 随测试轮推进（2-rank 参考值、一致性诊断、CI 冒烟）。
3. D6 清理项随日常迭代合入。

---

## 2026-08-02: D1–D6 修订实施（按 2026-08-02 评审 TODO）

### What was done
按 `2026-08-02-deltap-mpi-consistency-review.md` 的 D1–D6 全部落地：
- D1 PW KPAR=1 守卫（`GlobalV::KPAR>1` → WARNING_QUIT）+ 通信域注释统一
  （KPAR=1 时 POOL_WORLD==MPI_COMM_WORLD）+ `unk_overlap_pw.cpp`/`deltap_pw.h`
  注释修正（顺带闭合 T9）。
- D2 `DeltapScfSolver::apply_lambda` 统一同步出口，覆盖内循环 trial/final λ。
- D3 LCAO 4-rank escon/γ per-rank 断言（临时诊断实跑后移除）。
- D4 固化 2-rank PW 参考值 + nproc 敏感性定性（浮点噪声，非缺陷）。
- D5 PW Bcast 前一致性回归守卫 + `tests/deltap_mpi_smoke/run.sh` 冒烟。
- D6 两处 `sync_lambda` 空 λ 防御 + 两处 `f_en.dp_escon` 契约注释。
详见 `2026-08-02-deltap-d1-d6-revision.md`。

### 验收结果
- 单测 16/16；PW 1-rank 与 R5 基线逐字节一致；PW 2-rank 0 divergence 告警、
  与上轮输出逐字节一致；LCAO 4-rank P2 iter=11 锚点不变、40 个 escon 值每
  个恰出现 4 次（4 rank 每迭代全一致）。
- KPAR 负向测试：`kpar 2` → WARNING_QUIT（exit=1）。
- MPI 冒烟脚本端到端 PASS（PW 2-rank + LCAO 4-rank）。

### TODO 状态更新
- D1 ✅ / D2 ✅ / D3 ✅ / D4 ✅（定性 + 参考值）/ D5 ✅（守卫 + 脚本，CI 接线
  待 runner 环境）/ D6 ✅。
- 后续：T2（PW init 惰性化）、T7（force 路径专项）、D4 根因（可选）、冒烟入 CI。

### Files modified this round
- `source/source_pw/module_pwdft/deltap_pw.{cpp,h}`、`source/source_esolver/deltap_scf.{cpp,h}`、
  `source/source_esolver/esolver_ks_lcao.cpp`、`source/source_esolver/esolver_ks_pw.cpp`、
  `source/source_io/module_unk/unk_overlap_pw.cpp`
- 新增 `tests/deltap_mpi_smoke/run.sh`
- 新增 `docs/superpowers/specs/2026-08-02-deltap-d1-d6-revision.md`；本 dev-log 追加本节

### Next steps
1. 提交本轮改动；`center/INPUT` 与杂散 `STRU.cif` 不提交。
2. 冒烟脚本接 CI（需 MPI+赝势 runner）。
3. T2（PW init 惰性化，随 relax 支持）、T7（force 路径）继续推进。

---

## 2026-08-02: D1–D6 提交前评审反馈处理

### What was done
按提交前评审修正 2 个阻塞项 + 闭合 1 个覆盖缺口：
- 阻塞项 1：`deltap_pw.h` 注释恢复 "Called once"（核实 `before_all_runners`
  在 `driver_run.cpp:67`、离子步循环外，每 run 一次；此前按 T9 改为 per
  ionic step 属错误推断）。
- 阻塞项 2：`tests/deltap_mpi_smoke/run.sh` 的 `set -e` 缺陷——`run_case`
  失败时提前终止脚本，改调用处 `|| true` 累计失败；负向测试验证两个用例与
  FAILED 摘要均执行。
- 缺口 3：D2 内循环路径补 4-rank 实跑（`test_stru_target`，inner_nmax=3，
  临时副本运行），`inner loop done` 出现、无 divergence 告警；该用例已入
  冒烟脚本（三用例全 PASS，repo 无污染）。
- 小观察记录：D5 守卫每次 γ 测量多一次 world Allreduce，被守护路径实测
  不发散——可接受现状，降级 debug-only 列为 TODO。

### 验收
- 冒烟脚本三用例 PASS：PW 2-rank、BN 4-rank、内循环 4-rank；负向（错 marker）
  FAIL 摘要与 exit=1 正确。
- 单测 16/16、PW 1-rank A/B 逐字节一致（此前已验证，注释/脚本改动不影响）。

### Files modified this round
- `source/source_pw/module_pwdft/deltap_pw.h`（注释纠正）
- `tests/deltap_mpi_smoke/run.sh`（set -e 修复 + 内循环用例）
- `docs/superpowers/specs/2026-08-02-deltap-d1-d6-revision.md`（评审反馈处理记录）
- 本 dev-log 追加本节

### Next steps
1. 提交：7 源码文件 + 两份 2026-08-02 文档 + `tests/deltap_mpi_smoke/` +
   dev-log；排除 `center/INPUT` 与杂散 `STRU.cif`。
2. D5 守卫降级 debug-only（可选）；冒烟脚本接 CI（需 MPI+赝势 runner）。
3. T2（PW init 惰性化）、T7（force 路径）继续推进。
