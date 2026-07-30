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
