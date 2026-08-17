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

11. **非正交基算符记账必须走全迹**：H_HK 期望在非正交基下是 T·Π 全迹而非对角
    T_pp 迹（h2o1 差 18%）；T2 修正后 E'(λ) 斜率 224 → −13.3 → −0.013 eV/Ry
    （三数量级压平、±0.001 八位对称、λ² 抛物、变分下界恢复）。
12. **target-aware 分支选择的 γ 报告是靶点跟随的**，不能作为任何收敛判据或 FD
    可观测量；operator 模式下 γ 只做外循环读数，且必须连续性锚定（branch.dat
    缺失时 Stage-B 回退 target-aware = 自证循环，T4a/T3' 两案同款）。
13. **力一致性是记账性质，残差大小是 λ*(R) 路径性质**：escon=−λΓ 恒等式与驱动
    信号无关（一行定理成立），但 FD 残差 ∝ λ*² 且 λ* 依赖驱动信号——弱耦合
    驱动（γ：dγ/dλ≈−0.3）强制 λ* 大 → 泄漏大；强耦合驱动（Γ：dΓ/dλ≈−4.2）
    保持 λ*~1e-3。驻点 FD 选驱动可观测量的标准 = 耦合强度，不是"直测直钉"。
14. **锚定三形态 + 冻结纪律（T-7'/T-7''，2026-08-13）**——"分支移位只在
    建立物理参考的时刻评估一次；任何响应测量期间（内循环 trial、FD 位移、
    secant 步）shift 必须冻结"：
    | 锚定方式 | 失败形态 | 出处 |
    |---|---|---|
    | target-aware | 读数钉在靶点上 → 假收敛 | T4a |
    | 连续性锚（跨测量重评估） | 读数钉在前值上 → 吞掉 <半量子的物理响应 | T-7'（内循环 trial） |
    | 冻结 shift | 正确：raw 平滑移动、report 跟随 | T-1 / T-7'' 入口冻结 |
    T-4a 与 T-7' 是同一规则的两处违反；T-7'' 验证入口冻结后 report=raw
    精确跟随（shift0=0，检查点 1/2 全过）。
15. **γ-drive 冻密度内循环不可行——响应为负且符号跨几何不稳定（T-7''，
    2026-08-13）**：解锚后内循环残差 = 真实靶点差，但冻密度 dγ_O/dλ_O
    = −1.28 rad/Ry（负，两臂 trial-0 孤立测量一致），且非对角耦合≈对角
    （λ_O1 驱动同时推开 H 的 γ）——scalar-CG 翻号、对角 Jacobian 2-循环
    极限环，结构性不可收敛。SCF 级响应本几何 −0.62、T-18 几何 +0.059：
    符号不稳定。T-6' 开放问题（测量通道 vs 算符控制权限）裁定为**算符
    控制权限问题**；γ-hold 若要成立须改自洽驱动（iter_finish 每密度
    重驱动），生产路径维持 proxy 驱动。
16. **算符力通道与 μ 通道的隐含场不必一致（F-2b，2026-08-13）**：链接算符
    （H_HK）是合格*极化算符*（μ/能量通道已闭合），但其*力通道*（B 项
    −λ∂Γ^HK/∂R）不是真实场力——实测力/μ 隐含场杠杆比 **17.7×（hk）/
    15.7×（proxy）**；**移除 H_HR 后总力变化 ≤0.6%**，力失配是 H_HK 内禀
    而非 τ·P̂ 代理的"荷错位"（F-2 归因证伪）。正确场算符要求 L_F=L_μ；
    力级 EFC 修正量 = λ·(L_μ·Z*−L_F)（Z* 取 efield 通道实测）。
17. **T3 与 F-2b 双命题并存（评审 2026-08-13 定稿）**："B 项 = E_HK(R) 的
    内部精确梯度"（T3，记账自洽）与"E_HK(R) 的 R 依赖不服从 −E·P 场物理"
    （F-2b，力/极化 Maxwell 失配 17.7×）是两个不同命题，都成立。物理本质：
    H_HK 能量的 R 依赖走极化通道（∂γ/∂R=Born 电荷）+ 基组几何通道
    （S_dk/SMO 投影位置导数）双通道，真实场力只允许前者——**"有力无极化"**。
    Ô_θ/EFC 的算符期望即 γ，其 R 导数构造上纯极化通道（Maxwell 一致性白来）
    → EFC 是场模式力/应力的唯一第一性解法。应力版同族（压电 Maxwell：
    ∂σ/∂λ vs ∂P/∂ε）是 F-8 验收判据，不是可选项。
18. **分布式算符修正的本地块必须走 GEMM，观测可走 Allreduce（F-6，2026-08-13）**：
    LCAO 2D 块循环下，H 的本地块 (nrow×ncol) 的行/列是不同轨道集合
    （`nrow==ncol` 只保证尺寸巧合，不保证索引语义）——任何"用本地行当列
    索引"的算符修正（H_sym、H_ow）在 MPI 下都是错的。修正矩阵本地块的正确
    求法是分布式 GEMM（`H_sym = 0.5·(F·C_L† + C_L·F†)`，pzgemm 'T'+预共轭）；
    全局标量观测（Γ^HK = Tr[ρ·H_sym] 的 T·Π 迹）则只需"本地行×本地带部分和
    + 均匀计数 Allreduce"（A' 族）。同一函数的两条路径可以共存：串行分支
    逐字节保留（硬约束），MPI 分支走 GEMM。默认参数陷阱：`deltap_observable`
    默认 operator——"legacy gamma 模式"测试可能实际走 Γ 路径，验收设计必须
    先核对默认值。
19. **跨进程列的带对必须行组 gather，A' 本地带归约在 dim1>1 下不完整；
    Allgatherv 搬 complex 计数要 ×2（2026-08-13，Q2/3.2）**：带空间矩阵
    （T/Π/U）的"本地行×本地带部分和 + 全 comm Allreduce"只在每个带对都落在
    某个 rank 的本地带集内时才完整——进程列把带集切成不相交块后，跨列带对
    无人填充（F-6 的 A' 诊断即此；h2o 系 14 带 / 2 列网格下旧代码只差 ~1e-3
    小项，F_HK 留 ~1e-5 痕迹）。正确做法：行组（同 coord[0]）内 Allgatherv
    全带列 → 本地行×全带对循环 → 一行组贡献一次（coord[1]!=0 清空）→ 单次
    Allreduce。**配套陷阱**：gather 用 MPI_DOUBLE 搬运 complex<double>，
    发送/接收计数必须是 `2·nrow·ncol_b`——按复数元素数填会每 rank 只发一半，
    缓冲区列间交叠+后半零（h2o_asym Γ^HK 6.671→5.180，22% 偏差；h2o1 恰好
    占据带全在进程列 0 的干净区而不触发）。F_HK 力 MPI（compute_hk_force）
    与守卫收窄（非方本地块 nproc==1 外合法）同轮落地；h2o1/h2o_asym 4-rank
    与 2-rank 的 Γ/γ/λ/F_HK vs 串行逐位一致，co（NBANDS=15）4-rank F_HK 与
    串行一致，mpi_smoke 4/4。F-6 文档"h2o1/h2o_asym=529 轨道/265×264"系笔误
    ——实为 23 轨道，2×2 网格下本地块 12×12/12×11/11×12/11×11。

---

## 流程教训（2026-08-04 复盘锐评，来自提交记录的事实）

> 背景：力代码 07-09 提交，首次 FD 验证 08-02 才做（隔 3 周），当天暴露
> B-6（τ 单位 16 倍）与 B-7（H_HK 力缺失）。以下为流程约束，后续开发必须遵守：

1. **最便宜的判决性实验最先跑**。1 小时的单分子 FD 优先于 1 周的测试基建。
   反例：FD 脚本 07-29 建好（e629523a6/5d7b9867b），08-02 才首跑。
2. **物理未验证的子系统不做保真重构**。反例：R1-R5 对 τ 单位错 16 倍的对象做
   逐字节保真，锚点在 B-6 修复后全部返工（1b2625fdd 重建）。验证物理 → 再重构。
3. **"跑通/不崩/逐字节一致"不算验证**。每个 PASS 必须注明独立参照物
   （FD / 实验值 / 解析解）。六周内唯一与独立参照物对比的验证
   （H2O 偶极 1.838 vs 1.855 D）恰好是最有说服力的结果。
4. **验收用例必须包含"恰好会失败"的设计**：奇数 NBANDS、非对称分子、
   非整除进程网格。反例：MPI 四轮修复（07-28 五连补丁 → 590ff7861 → T1/T3 →
   D1-D6 → D_I 混带）每轮都用恰好不暴露问题的用例验收（BN NBANDS 整除、
   Σγ 巧合守恒），四轮都漏。
5. **"声明不支持"不是修复的替代品**。07-29 文档写下"relax/MD 无效"后，
   修复优先级反而被压低两周。写 limitation 的同时必须挂 TODO 和 owner。
6. **单体系/单方向证据不作数**。力的全部 FD 证据曾长期只有 h2o1 的 O1-z；
   x/y 的 PASS 多为对称性零力。最少三体系：1 非对称分子 + 1 对称分子 + 1 固体。

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

---

## 2026-08-02: Force/Stress 开发指导文档（公式推导 + 代码审查）

### What was done
源码只读审查（deltap_lcao、deltap_force_stress、FORCE_STRESS、forces_onsite、
compute_hk_correction、deltap_common）+ 约束力/应力公式推导，输出
`2026-08-02-deltap-force-stress-dev-guide.md`。未改运行时代码。

### 核心结论
1. **实现矩阵**：LCAO 有 A1（SMO Pulay）力 + S1 应力（未验证、必崩）；
   A2（∂τ/∂R HF 项）、B（H_HK 力/应力）、C（escon λdγ/dR 记账）均未实现；
   PW 只有 onsite Pulay 力、无应力。
2. **公式分解**：F = F_KS − ρ·∂H_HR/∂R − ρ·∂H_HK/∂R + λ·dγ/dR；B/C 响应项
   仅在 H_c=λδγ/δρ 时相消（H_HK 近似满足，H_HR 不满足）→ FD 为最终裁判。
3. **B-1 根因推断**：`cal_force_stress` nlm_target 按 (nwl+1)² 分配但提取循环
   按全部 nw（多 ζ 逐一计数）→ 越界写，即 relax 崩溃/double-free 根因
   （静态分析，待 T7-a ASAN 实证）。`cal_pre_HR` 同模式多 ζ 覆盖语义丢轨道（O3）。
4. **T7 三步计划**：a) 修 B-1 + 核 C-12/13 → b) FD 验证（微分对象=E_tot+escon，
   残差即 A2 独立测量）→ c) 补 A2 + 定量 B/C → 关闭 C-02。

### Files modified this round
- 新增 `docs/superpowers/specs/2026-08-02-deltap-force-stress-dev-guide.md`；本文档追加本节。

### 2026-08-02（v2 修订）: dspin 比对修正
- 依据 DeltaSpin 源码比对（spin_constrain/cal_mw/dspin_force_stress）修订
  force/stress 指导文档为 v2：
  1. §2.3 重写：dspin Pulay-only 精确性三条件（观测量=算符 expectation/能量相消/
     约束激活）→ DeltaP ①②不满足、③可满足；
  2. A2（∂τ/∂R）提为最高优先：Born 电荷型主导力（电场焓 −E·P 类比）；
  3. FD 改双组协议：组① frozen λ（残差≈λ·dγ/dR+A2）+ 组②每位移点 λ 重收敛
     （relax 可用性判决）；
  4. 新增 §7 公式集（F1-F10）、§8 check-list、§9 TODO（T7-a→b→c→d）；
  5. 附带：建议仿 dspin magnetic-force 打印 ∂E'/∂γ=λ 诊断；dspin 自身 FD 文档
     本仓库未查到，"与 dspin 同构"不构成 FD 豁免。

---

## 2026-08-02: T7-a 实证 — B-1 nlm 越界 / B-5 跨步 UAF / B-3 λ 轨迹

### What was done
按 force-stress 指导文档 §9-T7-a 执行首轮实证修复；详见
`2026-08-02-deltap-force-stress-t7a.md`。三条独立缺陷一次修完。

### 关键结论
1. **B-1**：`snap()` ket 平铺索引 == 原子全局 `iw`（L→N→m 同构，m 为
   0,1,−1,2,−2… 序）；旧提取循环把每个 L 的非首 iw 写入块外，最后一个 L 越界
   （单 ζ 也崩，channel=3 必然 `index+3·length ≥ 4·length`）。修复 = 提取端
   对齐消费端 l²+m 协议，保持 first-ζ 通道集。
2. **O3 裁定**：`deltap_overlap.cpp:91` "Select first zeta of each l, same as
   DeltaSpin" + `nproj_per_atom_=(nwl+1)²` ⇒ γ/SMO/H_HR 全链都是 first-ζ 投影基，
   多 ζ 的 N>0 轨道不进约束投影是**设计语义**，从 bug 列表移除（记 LIMITATION）。
3. **B-5（新）**：`before_scf` 每离子步重建 p_hamilt，backend 捕获的 dp_op 悬垂
   ⇒ step2 `get_lambda()` UAF（bad_alloc/bad_array_new_length，gdb 实证）。
   修复：lambda 动态解析 + `state_.lambda_eff` 持久化（`apply_lambda`）+ 重建后播种。
4. **B-3 关闭**：λ 跨步连续（−5.5e-3 → −1.10e-2 → −1.65e-2，每步一次 GD），
   无重复/无丢失。
5. **顺带**：`cal_stress_IJR` 9→6 元越界写修复；消费端补 npol=2 DM 行尾步进。

### 验证
- relax 3 步 exit=0；ASAN relax 3 步 + cell-relax 1 步 **0 错误**；
- 单测 16/16；MPI 冒烟三用例 PASS；test_C_I 锚点能量逐字节一致。

### Files modified
- `deltap_force_stress.hpp`（B-1/stress/npol）
- `esolver_ks_lcao.cpp`（B-5/B-3）
- `deltap_scf.cpp`（lambda_eff 持久化）
- 新增 `2026-08-02-deltap-force-stress-t7a.md`；本文档追加本节。

### Next steps
- T7-b：FD 双组协议（frozen-λ + λ 重收敛；判据 5e-4 Ry/Bohr）；run_fd.sh 补 escon。
- T7-c：实现 A2（∂τ/∂R HF 项）+ λ 诊断打印 + 应力 FD。
- T7-d：P17 relax vs efield 对照。
- 记录：O3 first-ζ 为设计语义；A1 符号待 FD 实证（未验证状态维持）。

---

## 2026-08-03: T7-b 收尾 + T7-c B-7（H_HK 解析力）实现与排查

### What was done
按 `2026-08-02-deltap-force-stress-dev-guide.md` 继续 force/stress 开发。
详见 `2026-08-02-deltap-force-stress-t7b.md`（收尾）与
`2026-08-03-deltap-force-stress-t7c-b7.md`（B-7 实现与排查）。

### T7-b 收尾
1. 组② 完整 3×3 位移矩阵跑通（`run_fd.sh h2o1 0.005 1 2`）：5/9 FAIL，
   失败模式与组① 一致（x/z 大残差、y 对称零力 PASS），`script_exit=1`；
2. `tests/deltap_fd_force/README.md` 重写为双组协议文档；
3. 已补 escon 记账验证、`deltap_lambda_init_file`、h2o1 算例（T7-b 交付项）。

### T7-c B-7：H_HK 解析力（进行中，核心发现）
1. **公式**：E_HK = Re[(i/2)Σ_j Σ_n f_n W_n (C_L† S_dk C_R)_{nn}]，冻结 C 的
   HF 力 = −Re[(i/2)Σ f(W·∂T + ∂W·T)]；∂S_dk/∂R（含 bra 侧相位导数
   −2πi dk/lat0）、∂S_k/∂R（first-ζ 投影集导数块）均用 snap(cal_deri=1)；
2. **实现**：`DeltaP::compute_hk_force`（串行守卫 nproc==1+nrow==ncol）+
   静态存储（`s_stored_hk_force`）+ FORCE_STRESS fcs 汇总块并入（使
   TOTAL-FORCE 打印含 F_HK）；
3. **E_HK 交叉验证通过**：base λ* 下 E_HK=+2.5838e-3 Ry；组① O1-z ±δ
   FD 得 ∂E_HK/∂R ≈ +2.86 eV/Å，与 T7-b 隔离推断 B≈2.64 吻合；
4. **接入排错两处**：① F_HK 不进打印（getForceStress 内部打印）→ 静态存储
   + FORCE_STRESS 并入；② store 时机太晚（static 空）→ compute+store 移
   到 getForceStress 之前；
5. **未决（排查中）**：z 方向 uniform −2.24e-3 Ry/Bohr 偏移（x/y 精确吻合
   f_hk，z 不吻合；∑F_z 不守恒）——需对比 with/disabled 运行的
   force_deltap 与标准力分量定位。

### Files modified
- `deltap.h`/`deltap_wannier.cpp`（compute_hk_force 实现，含临时 hkdbg 打印）
- `deltap_lcao.h/.cpp`（s_stored_hk_force/e_hk 静态）
- `esolver_ks_lcao.cpp`（cal_force 接入 + [DeltaP HK-force] 诊断）
- `FORCE_STRESS.cpp`（fcs 并入 f_hk；临时 fsdbg 打印与 `if (false)` 待清理）
- `tests/deltap_fd_force/README.md`（T7-b 双组协议文档）
- 新增 `2026-08-03-deltap-force-stress-t7c-b7.md`；`t7b.md` 补 §3.5

### Next steps
- 定位并修复 z uniform 偏移（见 t7c-b7 文档 §3.3/§6）；
- 清理临时调试代码；组① O1-z FD 复验（期望残差降 B 项量级）；
- A2（∂τ/∂R）+ τ 单位决策（B-6 BLOCKER）；应力 S1；MPI 串行-only 记录。

### 2026-08-03（续）：z-shift 根因定位 —— ABACUS 净力修正 vs f_hk 不守恒
- **定位**：`FORCE_STRESS.cpp` 力汇总末尾 `fcs(iat,i) -= sum/nat`（gate/
  efield 关闭时）对总力做均匀修正；Σ_z f_hk = +6.728e-3 → sum/nat =
  +2.2427e-3 恰为观察到的 uniform z 偏移；x/y Σ=0 故精确吻合。
- **证据链**：启用/禁用两版 f_hk 与 force_deltap 逐位一致、同二进制两次
  运行逐位一致、禁用 compute_hk_force 后回基线、ASAN 0 错误。
- **真问题**：f_hk z 分量不满足平移不变性（Σ_J F_Jz ≠ 0）。候选：
  (a) S_dk 相位项 ∂phase/∂R_bra 缺补偿；(b) E_HK 定义本身非平移不变。
  需整体平移数值实验区分。详见 `2026-08-03-deltap-force-stress-t7c-b7.md`
  §3.3/§6。

### 2026-08-03（续2）：判据实验①② + 相位链 g==0 缺陷修复（T7-c B-7 闭环）
- **判据①（整体平移 z，gdir=3）**：∂E_HK/∂Δ_z = −0.115 Ry/Bohr vs
  −∂escon/∂Δ_z = +0.0074 Ry/Bohr —— 差 15.5× 且异号 → **解读 (b)（C 项
  补偿）证伪 → (a)（实现漏项）成立**。E_HK uniform 敏感性被精确量化 =
  π·dk/lat0·Re(Σ f W T)（两处实验均精确吻合）。
- **判据②（gdir=1，KPT 2×1×1，平移 x）**：∂E_HK/∂Δ_x = −0.100 Ry/Bohr，
  ΣF_HK,x ≈ +0.103 —— 机制沿 dk 方向跟随，无 z 专属 bug。
- **缺陷定位**：`compute_hk_force` 的 `if (g == 0.0) continue;` 把 g=0
  配对（自配对、s 通道等）的相位导数整体跳过（逐对 tacc/phaseacc/实际 U
  累积打印证实：phaseacc 总和 = −2πi·dk/lat0·T，实际 ΣU 缺失一大块）。
- **修复**：相位项与轨道导数解耦、无条件累积。修复后 ΣF_HK,z = +0.1147
  ≈ −∂E_HK/∂Δ（FD 0.1150）✓；O1-z FD 残差 5.33 → 3.35 eV/Å（SCF 侧逐位
  不变，残差变化全来自 F_HK）。
- **残差分解**：3.35 = 均值扣除摊平项（0.99，ΣF_HK,z=+2.96 ÷ 3）+ 缺项
  A2/C（≈2.37 eV/Å，O1-z）。
- **设计决策点**（待用户确认）：D-A E_HK loop 约定（+dk 位置型 vs signed
  闭合环 E_HK≡0）；D-B FD 对比协议用未扣除力；D-C 实现 A2（∂τ/∂R）与 C
  （λ·dγ/dR）；D-D 高精度验收（ecutwfc=100/ecutrho≥400）。
- 详见 `2026-08-03-deltap-force-stress-t7c-b7-continued.md`。

### Files modified（本轮）
- `deltap_wannier.cpp`：g==0 相位项修复（`compute_hk_force`）+ 临时
  hkchk/hkpair 打印（hkpair 已 `#if 0`）
- 新增 `2026-08-03-deltap-force-stress-t7c-b7-continued.md`

### Next steps
- 用户确认 D-A/D-B 后：run_fd.sh 对比协议改未扣除力；实现 A2（∂τ/∂R，
  含 B-6 τ 单位决策）与 C（λ·dγ/dR）；
- 清理临时调试打印；高精度重跑组① O1-z；MPI/ASAN 回归。

### 2026-08-03（续3）：相位 τ 单位修复（B-6 确认）+ 判据①复跑（用户评审复核）
- **用户评审**：《评审：T7-c continued 实现思路》—— 15.5× 失配 = 晶胞
  尺寸 15.87（2.4% 内吻合）= B-6 τ 单位 bug 在 `compute_S_dk` 相位里的
  体现（`tau0=get_tau()` 为 lat0 单位 ≈Å 数值，与分数步长 dkv 混用，
  相位放大 L 倍）；解读 (b) 应平反；§7-4 "trace 结构性属性"与 D-A
  （signed-dk）撤回；先做相位单位验证，再谈 A2/C。
- **实施**（`deltap_wannier.cpp`）：`compute_S_dk` / `compute_hk_force`
  相位 τ 改用 Direct 分数坐标 `taud`；相位导数因子改
  `−2πi·(latvec⁻¹·dkv)_α/lat0`（正交晶胞 = −2πi·dkv_α/(L_α·lat0)，与
  处方一致）；g==0 修复保留。与 ABACUS 参考 `unk_overlap_lcao.cpp:529`
  （`kRn = 2π(kvec_c·R − dk·tau1)`）数值等价 ✓。
- **判据①复跑（gdir=3）**：E_HK 由 +2.58e-3 → −8.63e-2 Ry（S_dk 虚数
  主导，Im(T)→−0.999）；ΣF_HK,z = +0.1147 → **−1.4e-4 ≈ 0**（冻结 C 下
  E_HK 平移不变）；∂E_HK/∂Δ_z（FD，弛豫）= −5.75e-3，∂escon/∂Δ_z =
  −7.4e-3 —— 同号、比值 0.78，"E' 原点不变"补偿未复现。
- **15.87× 预言不成立的机理**：单位修复改变 S_dk 相位本身（O 自相位
  ≈+1 → −i），Re(T_00) 塌缩 +0.743 → −0.022；相位导数 ∝ dkv_eff·Re(T)
  收缩 ~530×（含 Re(T) 额外 33×）并翻转符号。"15.5×≈15.87"是 O 恰在
  τ_lat0≈8 处的几何巧合。
- **撤回**：解读 (b) 证伪结论、§7-4 trace 结构性属性、D-A（signed-dk）。
  **保留**：g==0 修复（仍自洽）、D-B、D-D、残差分解方法论。
- **残差预算作废重排**：§6.3 "0.99 摊平 + 2.37 缺项"前提（ΣF_HK,z=
  +2.958）已不存在；O1-z 单原子在 ecutwfc=50 下 FD 仍 ~4.3 eV/Å，
  噪声+弛豫+缺项混合，不作验收依据。
- 详见 `2026-08-03-deltap-force-stress-t7c-b7-continued.md` §10。

### Files modified（本轮）
- `deltap_wannier.cpp`：`compute_S_dk` / `compute_hk_force` 相位 τ 单位
  修复（taud + latvec⁻¹ 导数因子）+ 注释更新
- `2026-08-03-deltap-force-stress-t7c-b7-continued.md`：追加 §10 复核 +
  撤回标记

### Next steps（更新）
- ecutwfc=100 + ecutrho≥400 + scf_thr 1e-8 重跑判据①与 O1-z（D-D），
  重建 A2/C 残差预算（摊平项已消失）；
- 实现 C（λ·dγ/dR，escon FD −7.4e-3 同号响应 → E' 原点敏感性主要候选）
  与 A2（∂τ/∂R）；
- 回归面重建：test_C_I 类锚点（H_c/SCF/E_HK 随相位修复全面变化）；
- 审查 compute_S_dk_link（未使用）与参考实现位置修正项是否需要补；
- 清理调试打印；MPI/ASAN 回归。

### 2026-08-03（续4）：相位修复轮评审处理 + 正确版判据①（冻结 λ 总 E' FD）
- **用户评审**：修复正确、ΣF_HK,z≈0 是核心结果；两处结论性错误需回改：
  (1) "15.87× 预言不成立"→ 应表述为"数值巧合叠加在真实单位 bug 上"
  （dkv_eff 恰按 L=15.87 收缩已确认；530× = 15.87×（单位）× 33×
  （Re(T) 塌缩），与预言机制完全一致）；(2) "E' 补偿未复现"比较对象错
  （弛豫 FD 混入 H_HR 的 ψ 响应），正确检验 = 冻结 λ 总 E' 均匀平移 FD，
  分离 ∂(TrρH_HR)/∂Δ（A2）与 ∂escon/∂Δ（C）；"C 是主要候选"是错误
  比较的推论，优先级改为先做正确版判据①，A2/C 成对定量。
- **实施**：(a) §10.2/§10.3 结论措辞回改；(b) `compute_S_dk` 相位处加
  B-6 族防护注释（τ 必须 Direct 坐标，勿混入 lat0 单位 H_HR τ）；
  (c) `deltap_force_stress.hpp` 加临时 `[hhrdbg] E_H_HR =
  Σ_I λ_I·τ_α(I)·⟨P̂_I⟩` 打印（cal_force_IJR 值块累积，与 SCF 的
  H_HR=λτP̂ 定义一致）。
- **冻结 λ 判据①结果**（组①，λ=base 收敛值，step 0.0，gdir=3）：
  ∂E'/∂Δ = **−3.254e-2 Ry/Bohr** = ∂E_H_HR/∂Δ（−2.926e-2，A2 内容）
  + ∂escon/∂Δ（−3.70e-3，C 内容），1.5% 内恒等；∂E0/∂Δ + ∂E_HK/∂Δ ≈
  +2.9e-4 ≈ 0；冻结 λ 下 E_HK FD = +1.22e-4 ≈ 解析 −ΣF_HK,z=+1.36e-4
  （10% 闭合）—— 上轮组②的 −5.75e-3 确认是 λ 漂移污染。
- **A2/C 预算**：A2 均匀力 = +2.93e-2（lat0 τ）→ B-6 分数 τ 后 +1.84e-3；
  C 均匀力 = +3.70e-3（冻结 λ，组② +7.4e-3 是 λ 漂移虚高约 2×）。
  **H_HR 与 escon 同号相加，"−E·P 对偶补偿"未在数据中复现**（用户预言
  与实测不符，如实记录）；E' 均匀分量由均值扣除消去，relax 不受影响，
  真正要验证的是逐原子 A2/C（留 ecutwfc=100 高精度轮）。
- 详见 `2026-08-03-deltap-force-stress-t7c-b7-continued.md` §10.8。

### Files modified（本轮）
- `deltap_wannier.cpp`：B-6 族防护注释（compute_S_dk / compute_hk_force
  相位处）
- `deltap_force_stress.hpp` + `deltap_lcao.h`：临时 hhrdbg E_H_HR/⟨P̂⟩
  打印（cal_force_IJR 增加可选 p_hat 参数）
- `2026-08-03-deltap-force-stress-t7c-b7-continued.md`：§10.2/§10.3 回改
  + §10.7 优先级重排 + §10.8 冻结 λ 正确版判据①

### Next steps（更新）
- A2（∂τ/∂R）与 C（λ·dγ/dR）成对实现/成对验证（§10.8 已定量均匀预算；
  逐原子预算留 ecutwfc=100 + ecutrho≥400 + scf_thr 1e-8 高精度轮，
  先重建 test_C_I 锚点再做 D-D 验收）；
- 清理调试打印（hkdbg/hkchk/fsdbg/hhrdbg）；MPI/ASAN 回归。

### 2026-08-03（续5）：B-6 修复轮（H_HR τ 单位 → 分数坐标）+ A/B 验证
- **用户评审**：A2/C 都不要先做，插入 B-6（H_HR τ → 分数坐标），顺序
  B-6 → 锚点 → A2/C → D-D；连带项（E-field 等效、bn_sampling 标定）为
  B-6 包一部分；§10.8"补偿未复现"改写为 O5 代理差距定量化。
- **实施**：三处 λτ 改 `taud`（deltap_force_stress.hpp:63/:226 +
  deltap_lcao.cpp:125，均加 B-6 防护注释）；PW 无 τ 不受影响。
- **A/B 验证**（h2o base L=15.8753、BN center L=3.615，1-rank）：
  E_H_HR **精确缩小 L 倍**（−0.4471→−0.0281，15.90×，0.15% 吻合）；
  E' 位移 = TrρH_HR 位移（99.8% 闭合）；**同步模式 λ 单步冻结 → λ 不重
  收敛**（update_lambda_gd 只做一次 GD），escon/γ 不变，E_HK 0.2% 漂移。
- **机制澄清**：λ 只依赖 λ=0 的 γ（B-6 前后相同）；内循环 BFGS 模式
  预期 λ ×L 重收敛（算符不变），未验证 = D2 缺口实测内容。
- **E-field 重推导**：F1 公式 E=−πλ/(2a)（λ Ry、a Bohr）**只在分数 τ
  下成立**；旧代码等效场强 L 倍（h2o 2L/π≈10.1×）；公式与工作值不变，
  π/2 因子仍归 F1 备忘录。
- **锚点影响**：test_C_I（λ≡0）不受影响；λ≠0 用例 E' 位移 ≈ TrρH_HR
  位移（BN center +2.7e-3 Ry、h2o +0.418 Ry），γ 轨迹可能分支翻转。
- 详见 `2026-08-03-deltap-b6-tau-unit-fix.md`；§10.8 均匀预算已回改为
  A2=+1.85e-3 / C=+3.70e-3（B-6 后冻结 λ）。

### Files modified（本轮）
- `deltap_lcao.cpp`、`deltap_force_stress.hpp`：B-6 τ → taud + 注释
- `2026-08-03-deltap-force-stress-t7c-b7-continued.md`：§10.8 回改
  （O5 定量 + B-6 后预算）
- `2026-08-03-deltap-b6-tau-unit-fix.md`：新增 B-6 轮文档
- `deltap-development-log.md`：本段

### Next steps（更新）
- 重建 λ≠0 锚点：bn_sampling 9-label results.csv、deltap_bn_test、
  relax（ecutwfc=100；test_C_I 跳过）；
- 补 D2：4-rank inner_nmax>0 冒烟，实测内循环 λ ×L 重收敛；
- A2/C 成对实现 + 成对验证（冻结 λ FD，新预算 A2=1.85e-3/C=3.70e-3）；
- D-D 高精度轮；清理调试打印；MPI/ASAN 回归。

### 2026-08-03（续6）：锚点重建轮（B-6 后生产设置）+ D2 内循环 λ×L 核实
- **执行**：commit `53f94042d`（相位修复 + B-6 + 调试打印 #if 0 包裹）；
  全部 λ≠0 锚点用例 INPUT 显式加 ecutrho 400（relax 还 ecutwfc 50→100），
  重建 bn_sampling 9-label + deltap_bn_test + deltap_relax +
  test_stru_target（4-rank 内循环）。
- **锚点内容**：E'（FINAL_ETOT_IS）+ 逐 iter λ/γ 轨迹（工件
  deltap_lambda_gamma.dat，*.dat 被 .gitignore 排除不提交）+ 分支选择
  deltap_branch.dat；results.csv 已更新（9-label 新值）。
- **轨迹要点**：λ 冻结于 iter≈11；γ 分支翻转 18–41 次/50 iter（5/9 末态
  |γ−t|≈2.6–5.8）——B-6 后同步模式约束驱动弱 L 倍，γ 目标常不可达
  （预期，非回归 bug；λ_step ×L 重标定归 F1 备忘录）。
- **D2 核实（center+inner_nmax=5）**：BFGS 内循环 λ 按 L=3.615 重收敛
  （N 原子 3.66×，1.2% 吻合；τ=0 的 B 原子不受影响）——算符不变性在
  收敛 λ 模式确认；同步模式 λ 不重收敛（B-6 轮）。test_stru_target 的
  λ 符号对 SCF 噪声敏感（±5e-3 皆可），不作 λ 量级判据。
- **relax 锚点**：3 离子步未收敛（grad 0.67 eV/Å），λ 逐步增长
  （−5.5e-3→−1.66e-2），E_HK 随 λ 增大；无崩溃（B-1 回归面 OK）。
- 详见 `2026-08-03-deltap-b6-anchor-rebuild.md`。

### Files modified（本轮）
- bn_sampling 9-label / deltap_bn_test / test_stru_target / deltap_relax
  INPUT：显式 ecutrho 400（relax 另 ecutwfc 100）
- `tests/deltap_bn_sampling/results.csv`：9-label 新锚点
- `2026-08-03-deltap-b6-anchor-rebuild.md`：新增锚点轮文档
- `deltap-development-log.md`：本段

### Next steps（更新）
- A2/C 成对实现 + 成对验证（hhrdbg 翻回 #if 1；冻结 λ FD，
  预算 A2=1.85e-3 / C=3.70e-3）；
- D-D 高精度验收（run_fd.sh 支持 ECUTWFC/ECUTRHO/SCF_THR 覆盖）；
- F1 备忘录 λ_step ×L 决策；A2/C 后清理调试打印 + MPI/ASAN。

### 2026-08-03（续7）：A2 实现验证 + 组②激活约束 FD 判决轮
- **A2 实现**（`deltap_force_stress.hpp`）：p_hat 累积转正（cal_force_IJR
  值块×DM 对角收缩），逐原子对角力 `F_Jβ −= λ_J⟨P̂_J⟩(L⁻¹)_{αβ}/lat0`
  reduce_all 前加入；应力无对应项（固定分数坐标 ∂τ/∂ε=0，F8）。
- **A2 验证（PASS，4 ppm）**：h2o1 base（λ*≠0）A2 前后 SCF 逐位不变，
  z 力差 == 公式（含均值扣除）；x/y 逐位一致。
- **组② FD（target=γ*(base)，ecutwfc=100/ecutrho=400/scf_thr=1e-8，
  全 3×3 矩阵）**：残差 0.02–0.32 eV/Å（判据 0.0129）。完整归因 =
  **O·dλ/dR（λ 响应项）**：ΔE_deltap=E_HK+E_H_HR+escon ∝ λ(R)，
  λ(R)=0.001(γ(R)−t) 随几何重导出，±δ 间 λ 变 ~1.5e-5；E' 对 λ 极敏感
  （dE'/dλ ≈ 405 eV/Ry 含 ψ 响应）。**dspin 定理前提（λ 于约束驻点，
  γ≡t）未被同步单步 GD 满足** → 残差是"代理差距×协议 dλ/dR"，非干净
  O5 定量，不能判决 A1+A2+B。
- **纯 DFT 对照（PASS）**：H1-x/O1-z FD vs 解析力残差 3e-4 eV/Å——
  标准 LCAO 力与 E 曲面一致，残差无标准力成分。
- **组① 参考（冻结 λ*，O1-z）**：残差 +0.615 eV/Å = C + ψ 响应
  （O1 自身 C≈+0.12，余为跨原子+响应）——与"组① 不严格闭合"理论一致。
- **内循环（deltap_inner_nmax>0）可用性**：bn 测试 SCF 呈极限环
  （drho 6e-4–1e-3 振荡）、位移点分支翻转、1-rank vs 4-rank 结果不同
  （E' 差 0.85 eV）——内循环判决 FD 前必须先修稳定性。
- **判决**：A1+A2+B 无直接错误证据（各分量已单独验证）；组② 现协议
  无法作 relax 判决；下一步修内循环稳定性 → 内循环组② 判决 FD。
- 详见 `2026-08-03-deltap-force-stress-t7c-a2-group2-fd.md`。

### Files modified（本轮）
- `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`：
  A2 实现（未提交，待清理 hhrdbg 后提交）
- `2026-08-03-deltap-force-stress-t7c-a2-group2-fd.md`：新增本轮文档
- `deltap-development-log.md`：本段
- `tests/deltap_fd_force/h2o1/target.dat`：临时改为 γ*(base)（工作区，
  提交时还原或说明）

### Next steps（更新）
- **修内循环 SCF 稳定性**（bn 极限环）：λ 更新与密度混合解耦/λ 阻尼，
  使 deltap_inner_nmax>0 在 ±δ 位移点可复现收敛（新阻塞，优先级最高）；
- 内循环组② 判决 FD（bn 或 h2o1+非平凡 target）：残差≈0（定理）则 C
  降级为可选增强；显著则为 O5 定量 → 决定补 C 或回算符形式；
- 组① 残差重测（C+响应预算）；清理调试打印；MPI/ASAN 回归；提交。
- **回归面**：MPI 冒烟 3/3 PASS；单测 math/gauge/common PASS；
  **smoothness 4/8 FAIL（B-6 相位约定改动致单测参考约定过期，预先存在，
  非 A2 引入，P1 TODO 待更新测试参考）**。

---

## 2026-08-03: 力求解知识更新总结文档

### What was done
沉淀 T7 专项全部认知，输出 `2026-08-03-deltap-force-knowledge-update.md`：
力求解严格分解（A1/A2/B/C/R 六项，前三项已实现验证）、九大算法难点
（D-1 观测量/算符二元性、D-2 τ 单位、D-3 H_HK 构造、D-4 dspin 定理前提、
D-5 分支离散、D-6 均值扣除、D-7 FD 噪声预算、D-8 内循环极限环、D-9 响应项）、
验证状态总表、修正后公式集 F1-F10、认知更新 7 条、剩余路线。
### 核心结论索引
- relax 正确性只需 A1+A2+B（dspin 定理，约束激活时 C+R≡0）；判决需驻点 λ 协议。
- E' 原点敏感性 = 代理差距定量（H_HR 比真 γ 大 ~8×，O5）。
- FD 判据级验收必须 ecutwfc=100/ecutrho≥400/scf_thr=1e-8。

### 2026-08-03（v2 重写）: 知识文档可读性修订
- 应反馈重写 `2026-08-03-deltap-force-knowledge-update.md` 为 v2 详解版：
  补全部名词解释（γ/Wilson loop/SMO/λ/escon/H_HR/H_HK/原点敏感性/平移不变性/
  均值扣除/驻点/组①②/egg-box/A1-R 代号/锚点）；
  状态改 ✅/❌/⏳ 三档醒目表（第 0 部分一图看懂）；
  九大难点每条按"问题/影响/对策/状态"四要素重写。无代码改动。

---

## 2026-08-03: 组② 驻点协议判决 FD —— O5 代理差距定量（结构性 FAIL）

### What was done
执行知识文档路线第 1 项（脚本级驻点 λ 组② 判决 FD，最小判决矩阵 h2o1 O1-z）：
- 验证 λ 注入路径：`deltap_lambda_init_file` + `deltap_lambda_step 0.0` 可冻结 λ（PASS，无代码改动）；
- 实测 E'(λ) 响应：∂E'/∂λ_O1 ≈ 222–224 eV/Ry（λ∈[−0.01,+0.01] 线性非零）；
  base 全分量弦斜率 404–518 eV/Ry；
- 实测 γ(λ) 响应：[rawG] 原始 γ 单调（∂γ_O1/∂λ_O1≈+0.30 base、+0.6–0.95 disp_plus），
  P3 报告 γ（target-aware 分支选择）为阶梯非单调（移位步长 >1e-3 容差）；
- 组② 驻点 FD：base/minus/plus 三几何均 |γ_report−t|<1e-3 后
  F_FD(O1-z)=−85.56 eV/Å（分支 A，λ_st(plus)=+0.002）或 −419.8（分支 B，λ_st=+0.01）
  vs F_ana=−0.7473 → **残差 84.8–419 eV/Å，判据 0.0129，FAIL 6600–32000 倍**；
- λ-leakage 闭合：残差 84.8164 = ∂E'/∂λ_O1(224.4 eV/Ry) × Δλ_st/2δ(0.3780 Ry/Å)，
  **5 位有效数字闭合**。
### 核心结论索引
- **dspin 定理 DeltaP 版前提被证伪**：λ 在约束驻点时 ∂E'/∂λ ≈ 222–224 eV/Ry ≠ 0
  （escon 观测量 γ_Wilson ≠ H_c 算符期望 ⟨Ô⟩_proxy），λ-leakage 完全主导组② 残差；
- **任何"约束激活（γ≡t）"的 relax 力都不一致**（84.8–419 eV/Å），补 C 不解决
  （约束路径 C+R≡0），出路是 O5（算符/记账形式）；
- 附带：|γ_report−t|<1e-3 停止判据不适定（分支阶梯 → 多解，E' 差 1.77 eV），
  对 γ_raw 判据结论不变（仍 FAIL）。
### 修改文件
- 新增 `2026-08-03-deltap-force-stationary-group2-fd.md`（本轮 dated 文档，177 行）。
- `deltap-development-log.md` 追加本段。
- 无源码改动（纯实验轮）；排除 `STRU.cif`。
### 回归面
- HEAD=4c446b950 不变；MPI 冒烟 3/3、单测 PASS、smoothness 4/8 FAIL（B-6 参考过期，
  P1 TODO 维持）。
### Next steps（更新）
- 路线第 1 项状态：**已执行，结构性 FAIL，O5 触发**（不是"残差≈0 则 A1+A2+B 精确"）。
- O5 决策（用户）：(a) 算符重构使 ⟨Ô⟩→γ（加强 H_HK/改 H_HR）；(b) 记账重构 escon 用
  ⟨Ô⟩（约束变为 ⟨Ô⟩=t）。下一步低配实验：临时开 hhrdbg 打印逐原子 ⟨P̂⟩/E_H_HR/λ、
  E_HK/λ vs γ_report，分解差距来源（H_HR vs H_HK）。
- D-D 正式验收暂停（约束激活场景力结构性不一致，验收无意义）；smoothness 参考更新
  等 O5 决策后与锚点重建一并处理。

---

## 2026-08-03: Tier-1 判据体系算例构建（hf / co / h2o_asym）

### What was done
按"正确性归非对称小分子"的决策，在 `tests/deltap_fd_force/` 新建三个 Tier-1
算例骨架（仅构建，未运行）：
- `hf/`：HF 沿 z 轴（键长 0.9168 Å，15.873 Å 盒中心）——非对称双原子，无分支简并；
- `co/`：CO 沿 z 轴（键长 1.128 Å）——第二候选（π 空间更重）；
- `h2o_asym/`：h2o1 的 H2 预畸变（+0.03 x, +0.05 z Å）破 C2v——3 原子覆盖的备选。
三者与 h2o1 同盒同 KPT（Gamma 1×1×2，gdir=3）；不入库 INPUT（run_fd.sh 生成，
生产设置走 ECUTWFC=100/ECUTRHO=400/SCF_THR=1e-8 环境变量覆盖）。
`tests/deltap_fd_force/README.md` 追加 Tier-1 一节（体系表 + 三项筛选协议：
γ(λ) 平滑性 / ±δ 分支零翻转 / 同步模式收敛）。

### Files modified this round
- 新增 `tests/deltap_fd_force/{hf,co,h2o_asym}/{STRU,KPT,target.dat}`
- `tests/deltap_fd_force/README.md` 追加 Tier-1 节

### Next steps
1. 运行三项筛选协议，选定判决体系（预期 hf 首选）
2. 用选定体系跑分支分解方案 B（连续性-only 重跑）终局确认

---

## 2026-08-04: Tier-1 筛选执行状态 + MPI D_I 混带 bug 定位（中断恢复快照）

### What was done
- 执行 Tier-1 三体系（hf/co/h2o_asym）筛选 3（同步收敛 base）——串行全部 PASS
  （hf 37 iter / co / h2o_asym，均 scf_nmax 内收敛，λ*≈−γ(base)）。
- 定位 **MPI 阻塞级 bug**：co/h2o_asym 4-rank 崩溃于 D_I Allreduce（MPI_ERR_TRUNCATE）。
  根因：MPI 下 `psi->get_nbands()` 返回本地带数（ncol_bands，nb=1 交错分布），
  D_I 尺寸 rank 相关（NBANDS=15 → 8/7）；shim/gdb/插桩三重证据闭合。
- 定位 **MPI 数值错误**：D_I Allreduce 交错混带，逐原子 γ 被污染但 Σγ 守恒——
  hf 4-rank γ=(−5.107,−6.831) vs 串行 (−5.909,−6.031)，Σγ −11.938 vs −11.940；
  hf MPI base 还因混带 γ 使 SCF 跑到 iter=100 未收敛。
- 定位 **串行 FFTW-OMP 卡死**（环境）：atomic_rho→recip2real 全线程屏障自旋，
  `OMP_NUM_THREADS=1` 绕过。
- **数据损失**：两次系统崩溃清空 /tmp/dp_screen（串行 base 三体系数据 +
  9 个冻结 λ 扫描运行 + shim/gdb 分析产物全丢）；仓库侧 hf MPI base（对照用）幸存。
- 状态快照落 `2026-08-04-deltap-tier1-screening-status.md`（含 P0–P3 TODO）。

### 修改文件
- 新增 `docs/superpowers/specs/2026-08-04-deltap-tier1-screening-status.md`
- `deltap-development-log.md` 追加本段
- 无源码改动（临时插桩已回退，工作树干净）

### 下一步（摘要）
1. **P0**：修 D_I MPI 路径（全局带数 + 列复制/行通信子归约），回归 = hf 4-rank γ 与串行逐原子一致；
   MPI smoke 补 NBANDS 非均匀整除用例。
2. **P1**：重跑筛选 1/2/3（串行，OMP_NUM_THREADS=1，工作区用仓库侧 gitignored 目录）。
3. **P2**：hf MPI-vs-串行 γ 对照正式化为回归锚点。
4. **P3**：run_fd.sh/README 注明 OMP_NUM_THREADS=1；结果正式落 dated 文档。

---

## 2026-08-04: 总览文档更新（取代 07-13 版）

### What was done
新建 `2026-08-04-deltap-progress-and-plan.md` 作为最新总览入口：
状态仪表盘（✅/❌/⏳ 三档）、功能可用性矩阵（标注 LCAO 多 rank 在 D_I 修复前
不可信）、07-13 后进展时间线、关键技术结论（dspin 定理/三件套归因/FD 处方/
均值扣除陷阱/B-6）、P0-P3 TODO（P0=D_I 方案 A' 修复 + 同族审计）、文档索引。
无代码改动。

---

## 2026-08-04: 力测试结果可画图数据包

### What was done
汇总 T7 专项全部力测试数据为 `2026-08-04-deltap-force-test-results-plotting.md`：
总结论表（8 项测试 PASS/FAIL 标注）+ 12 个 CSV 数据集（D1-D12，含测试体系
结构描述、两档精度标注、判据 0.01286 eV/Å）+ 8 张建议图的图型与图注 +
数据使用注意事项（精度档位/均值扣除/协议伪差 vs 结构性 FAIL 的区分）。
核心图：D9 生产场景对比（log）、D7 E'(λ) 线性非零（∂E'/∂λ≈222-224 eV/Ry）、
D8 驻点判决（84.8-419 eV/Å，λ-leakage 5 位闭合）。无代码改动。

### 2026-08-04（v2）: 可画图数据包按读者分层重写
- `2026-08-04-deltap-force-test-results-plotting.md` v2：第 1 部分面向材料应用
  用户（图 U1 场景力误差/U2 H2O 偶极 vs 实验 1.838 vs 1.855 D/U3 BN 零刚度
  物理发现/U4 可用性矩阵 + 3 条 FAQ：能量可用力不可用、旧 λ 重标定、KPAR=1 限制）；
  原 12 数据集移入第 2 部分开发者附录（D1-D11 + 图清单 + 注意事项）。

---

## 2026-08-04: 六周开发复盘锐评 → 流程教训入库

### What was done
基于提交记录（~85 commit，06-26→08-04）做优先级倒挂复盘，六条流程教训
写入 dev log 顶部"流程教训"区（技术结论区之后）：判决性实验最先跑 /
物理未验证不保真重构 / PASS 必须注明独立参照物 / 验收用例要"恰好会失败" /
limitation 声明必须挂 TODO / 单体系证据不作数。无代码改动。

---

## 2026-08-04: 力问题解决方案与 TODO 重规划

### What was done
输出 `2026-08-04-deltap-force-resolution-plan.md`（唯一权威路线，取代分散计划）：
- 理论收拢：F_path = F_Pulay + (Γ_op−t)·(∂γ/∂R)/(∂γ/∂λ)；Γ_op=γ 时修正项
  恒零（dspin 条件正确表述）；84.8 eV/Å 残差与该式兼容。
- 三路线：A（escon 改 ⟨Ô⟩ 记账，1-2 天，预言残差 ~0.05 eV/Å）；
  B（解析 ∂γ/∂R，~1 周，严格解，副产品 C/Born 电荷）；C（弱约束产品化，兜底）。
- TODO 三阶段：Phase 0 可信面（D_I MPI / 分支判别+连续性 / smoothness /
  内循环极限环）→ Phase 1 Route A 原型 + 决策点 → Phase 2 实施验收。
- 可证伪的残差路线图：84.8 →(分支) 0.4 →(A) 0.05 →(B) <0.0129。
- 不做清单：单独补 C / 分支修复前驻点判决 / D_I 修复前用 LCAO 多 rank 数据 /
  路线定案前 D-D 验收。

---

## 2026-08-04: Route A+ 设计审阅稿

### What was done
输出 `2026-08-04-deltap-route-a-plus-design.md`（待审阅，未实施）：
三组件同改（escon=−ΣλΓ / SCF 约束变量改 Γ / 外循环 secant 校准 t_Γ 使 γ→t_γ）；
完整公式（Γ 逐原子定义、E'≡E_KS 恒等式、Pulay 代表恢复、λ-leakage 变 O(λ)）；
代码清单（module_deltap Γ 计算、deltap_scf 模式开关、外循环挂钩）；
测试计划 T0–T7 带可证伪预言（驻点残差 84.8 → ≤0.02 eV/Å）；
附带收益：分支阶梯与 SCF 解耦（分支连续性降级为外循环读数需求）。

---

## 2026-08-04: Route A+ 联动重推导（8 项）

### What was done
输出 `2026-08-04-deltap-route-a-plus-derivations.md`（待审阅）：
D1 E_eff 重推导（算符斜坡定义 E_eff=λ/(2a)，**π 消失**——旧公式的 π/2 悬案
疑似错配共轭对象所致；焓斜率口径作交叉验证）；D2 力公式全集（残差改写为
λ·dΓ/dR，驻点预言 ≤0.02 eV/Å）；D3 应力同构推导（S1 地位确认，H_HK 应力
仍缺）；D4 PES 口径修复（E'=E_KS(ψ*) 干净约束能量面）；D5 极化率链两口径
α 公式 + P10 改 E₀(μ) 曲线重合判据；D6 P17 改极化匹配协议；D7 secant 外循环
公式（含保护/兜底）；D8 不变项确认（F2/total/BFGS/力实现/FD 对象）。
验证锚点 V1-V4 + T0-T7。无代码改动。

---

## 2026-08-04: 执行 TODO 指导文档（当前唯一权威执行清单）

### What was done
输出 `2026-08-04-deltap-execution-todo.md`：Stage 0 收尾（commit D_I/S_k 修复、
co f0 归因、锚点重建#2）→ Stage 1 Route A+ 串行实现（Γ 计算/状态机切换/外循环，
带文件锚点）→ Stage 2 串行判决 T1-T5（T3 驻点残差 ≤0.02 eV/Å 为判决点）
→ Stage 3 hk_correction MPI 修复（插在 T3 后，串行逐字节为硬约束）
→ Stage 4 锚点#3 + Tier-1 全矩阵 + D-D → Stage 5 文档清理。
含依赖图、阻塞规则、不做清单。无代码改动。

---

## 2026-08-04: D_I/S_k A' 修复验证 + co/hf MPI 逐原子 γ 对照 + 相位配对 sort bug 修复

### What was done
- **D_I/S_k A' 修复**（此前未提交的工作区改动）验证通过：
  `deltap_berry.cpp` D_I 改全局带槽位（local2global_col 映射，Allreduce 计数统一）、
  `deltap_overlap.cpp` nlm 键改全局轨道索引（iat2iwt 偏移）、5 处
  `psi->get_nbands()` → `paraV_->get_wfc_global_nbands()`。
  回归：单测 16/16（10 deltap_common + 6 esolver_dp）；MPI smoke 4/4
  （PW 2-rank、BN 4-rank、**co 4-rank 奇数 NBANDS=15**、inner-loop 4-rank）。
- **co smoke 路径修复**：`tests/deltap_mpi_smoke/run.sh` 中 co 用例路径
  `deltap_co_lcao` → `deltap_mpi_smoke/deltap_co_lcao`（原路径不存在导致 4 例全 FAIL）。
- **Stage 0.2 对照（λ=0，ecutwfc=100/ecutrho=400/scf_thr=1e-8）**：
  - hf：串行 γ=(-9.173,-2.767) == 4-rank 逐原子一致（六位）。
  - co：修复前串行 (-6.711,-9.209) vs 4-rank (-6.731,-9.189)，Δ=±0.020 rad 且
    Σγ 守恒（-15.920）——"总和守恒+逐原子漂移"模式复发，量级比 D_I 混带小 40 倍。
- **定位并修复残余 bug：相位配对贪心 sort 的负距离缺陷**（deltap_wannier.cpp
  "Reorder evals to match gamma_unwrapped"）：
  `min(|a-g|, 2π-|a-g|)` 在 |a-g|>2π 时给出**负**距离，贪心匹配优先选错带。
  co 的 gamma 跟踪在近简并带（-3.2573 两带差 1e-12）处 Hungarian 平局，
  串行/MPI 以 ~1e-13 输入噪声落入不同槽位 → sort 把权重配到错误的 γ 上
  （两运行 perm 相同但 γ 槽位交换）→ 逐原子 γ 差 0.02 rad。
  修复：`diff = |remainder(arg(eval) − fmod(γ,2π), 2π)|`（正确圆周距离）。
  **注意**：该 bug 在串行也存在（串行配对同样错），co 串行 γ 从
  (-6.711,-9.209) 修正为 (-6.702,-9.219) → 锚点需重建（Stage 0.3 已计划）。
- **修复后 Stage 0.2 对照**：co 串行 (-6.702,-9.219) == 4-rank 完全一致；
  hf 仍一致 (-9.173,-2.767)。逐轮 γ 轨迹也逐字节一致。

### Files modified this round
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（sort 相位距离修复）
- `tests/deltap_mpi_smoke/run.sh`（co 路径 + 用例）
- 提交内容含此前 D_I/S_k A' 修复与 Tier-1 骨架/文档

### Next steps
1. Stage 0.3：锚点重建 #2（S_k 修复 + sort 修复驱动，12 用例 rc=0）。
2. Stage 1：Route A+ 串行实现（Γ 计算 / 状态机切换 / 外循环 secant）。

---

## 2026-08-04: Stage 0.3 锚点重建 #2 完成（S_k 键修复 + 相位 sort 修复驱动）

### What was done
- 12 用例全部 rc=0：bn_sampling 9-label（1-rank）+ deltap_bn_test（4-rank）+
  test_stru_target（4-rank, inner loop 3 步）+ deltap_relax（1-rank, 3 离子步）。
- 归档 `/tmp/deltap_anchor2/`：run.log + deltap_lambda_gamma.dat +
  deltap_branch.dat（12 用例）。`deltap_lambda_gamma.dat` 为 B-6 轮临时补丁
  产物（当前二进制不写），本轮从 run.log `[DeltaP P*]` 行重新提取轨迹替换。
- `tests/deltap_bn_sampling/results.csv` 重新生成（9-label 新锚点）。
- **关键差异（vs B-6 时代锚点）**：修复后 9-label 末态 |γ−t| 全部 <0.13 rad
  （B-6 时代 5/9 为 2.6–5.8 rad），λ 衰减到 ~1e-6（B-6 时代 ±2.2–2.7e-3），
  E_tot label 间差 <1e-3 eV（B-6 时代 ~0.21 eV）。约束自洽恢复——相位配对
  sort 负距离 bug 使近简并带权重配错带、残差方向错，修复后 λ→0、γ→target。
- bn_test 4-rank γ=(4.016,3.515)/λ=(8.4e-7,7.2e-6)；test_stru_target
  内循环末 λ=(1.50e-2,1.31e-2)、|γ−t|=6.7e-3；relax 3 离子步 E 单调降
  −483.57→−485.40→−487.31 eV，λ 增长到 −2.38e-2，γ 恒 ~(−7.95,−2.38)。

### Files modified this round
- `tests/deltap_bn_sampling/results.csv`（9-label 新锚点）
- `docs/superpowers/specs/2026-08-04-deltap-anchor2-s-k-rebuild.md`（本轮 dated 文档）
- 仓库内 9-label `deltap_lambda_gamma.dat` 陈旧轨迹已用新鲜提取替换（gitignored）

### Next steps
1. Stage 1.1a/b：`deltap.h` 加 `gamma_op_` + `compute_operator_observable()`；
   `compute_gamma_scf` 顺带累加 Γ_I^HR。
2. Stage 1.1c/d：`compute_hk_correction` 按原子拆 Γ_I^HK；T0 对照
   （per-k vs 实空间 p_hat <1e-10，不一致即停）。
3. Stage 1.2/1.3：observable 状态机切换 + 外循环 secant。

---

## 2026-08-04: Stage 1 Route A+ 串行实现完成（Γ 计算 + 状态机 + 外循环 secant）

### What was done
- **1.1 Γ 计算（module_deltap）**：
  - `deltap.h`：成员 `gamma_op_`/`gamma_op_hk_`，`compute_operator_observable()`
    （返回 Γ_HR+Γ_HK 组合向量）、`gamma_op_hk()`、`compute_gamma_op_hk()`；
    `compute_hk_correction` 签名加 `const elecstate::ElecState* pelec`。
  - `deltap_wannier.cpp`：Γ_I^HR = τ_α(I)·Σ_kw Σ_n f_n·w_In（物理 k 点，
    `j < nppstr_-1` 排除包裹副本）；Γ_I^HK 按原子拆出（−0.5·Im[Σ_j Σ_p
    f_p·w_IJ[p][iat]·T_diag[p]]）。
  - **T0 通过**：per-k ⟨P̂⟩ == 实空间 hhrdbg，12 位全同
    （7.199716499951 2.140887285468 2.140887285237）；Γ_I^HK 与 E_HK 自洽。
- **1.2 状态机（deltap_scf）**：`observable_mode`（operator/gamma，默认 operator）；
  `scf_observable()/scf_target()` 选择 Γ/t_Γ 或 γ/t_γ；INPUT `deltap_observable`
  + 非法值检查；PW 固定 gamma 模式（无 Γ 实现）。
- **1.3 外循环 secant**：`t_Γ^(k+1)=t_Γ^(k)+κ·(t_γ−γ)`；κ 首轮 1、
  后续 Δt_Γ/Δγ clamp [0.3,3]；|Δt_Γ| 限幅 0.5 rad；连续 2 步发散 κ 减半、
  3 步 WARNING 保持。relax 走 `reset_ionic_step`，单点走 `iter_finish`
  （conv_esolver 由 LCAO 侧 `drho < scf_thr` 本地判定 + 每 SCF 一次守卫）。
- 打印：P3 加 Γ 列；E-field 换 `E_eff=λ/(2a)`（operator-ramp，无 π，V1 钉符号）；
  init 标注 t_Γ 初始化。

### Results
- **gamma 模式零回归**：relax 3 离子步 3044 行 DeltaP 输出逐字节一致（0 diff）；
  bn_test 4-rank P 行轨迹逐字节一致（19 行 diff 全为分支文件加载状态 + 计时）；
  center 两次独立运行逐字节一致（确定性）。
- **operator 冒烟**（h2o1/base，1-rank）：rc=0；Γ 列=Γ_HR+Γ_HK
  （7.195=3.600+3.596）；单点收敛 secant 触发一次 κ=1；E-field operator-ramp；
  首轮 |Γ−t_Γ|=12.7（t_Γ=t_γ 粗假设，预期，T4 判决）。
- 单测：deltap_common 11/11（新增 operator escon 用例）、esolver_dp 6/6。

### 关键 bug 修复（本轮）
- 单点 secant 不触发：DeltaP 块先于 `ESolver_KS::iter_finish` 执行，conv_esolver
  是过期值 → LCAO 侧本地 `drho < scf_thr` 判定。
- `compute_operator_observable()` 最初只返回 HR 部分，与注释（combined Γ）不符
  → 改为按位相加返回。

### Files modified this round（未提交）
- `source/source_lcao/module_deltap/deltap.h`、`deltap_wannier.cpp`
- `source/source_esolver/deltap_scf.h`、`deltap_scf.cpp`、`esolver_ks_lcao.cpp`
- `source/source_io/module_parameter/input_parameter.h`、`read_input_item_other.cpp`
- `source/source_pw/module_pwdft/deltap_pw.cpp`
- `source/source_esolver/test/deltap_common_test.cpp`
- `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`（hhrdbg 已 #if 0）
- 文档：`2026-08-04-deltap-stage1-route-a-plus.md`

### Next steps
1. Stage 2 判决：T1（E' 恒等式）→ T2（∂E'/∂λ）→ **T3（驻点组② 复判，判决点）**。
2. Stage 1 改动保持未提交，T3 通过后统一提交。

---

## 2026-08-04/05（凌晨）：Stage 2 T2 判定 + Γ^HK 记账修复（Route A+ 硬信号 PASS）

### 做了什么
- T2 首测：h2o1/base λ 扫描（±0.01/±0.001/0 Ry，1-rank 串行），E'(λ) 斜率
  **−13.3 eV/Ry**（O(1)）→ 按 TODO 偏离动作回 1.1。
- 归因：escon 的 Γ^HK 用 E_HK-split 对角约定，非正交 LCAO 基下实际施加的
  H_HK 算符期望需全 T·Π Gram 迹；对角约定高估 ~18%
  （探针：E_HK_conv=0.0574 vs E_HK_actual=0.0470 Ry @ λ=+0.01）。
- 修复：`compute_hk_correction`/`compute_gamma_op_hk` 的 Γ_I^HK 改为
  −0.5·Im[Σ f_p·Σ_{p'} w_IJ[p'][I]·T_{pp'}·Π_{p'p}]（逐原子实际期望）；
  escon ≡ −⟨H_c⟩ 精确，E' ≡ E_KS(ψ*) 恒等式恢复。
- T2 复测：斜率 **−0.013 eV/Ry**（判据 ≲1 的 1/80；±0.001 两点 8 位对称；
  E'(±0.01) 抛物 +1.5/+1.3 meV；变分下界恢复）。**PASS（硬信号）**。
- T1 判定：构造性恒等式（T0 12 位 + 探针），PASS。
- 回归：deltap_common 11/11、esolver_dp 6/6。

### Files modified this round（未提交）
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（Γ^HK T·Π 全迹修复）
- 文档：`2026-08-04-deltap-t2-escon-hk-trace-fix.md`、TODO Stage 2 进展

### 关键 bug/修复
- **Γ^HK 记账 vs H_c 不一致（T2 首轮 FAIL 根因）**：对角 T_pp 约定 ≠ 实际
  Tr[ρ·H_sym]；非正交基 Gram 因子 Π=C_L†C_L ≠ I。修复后 E'(λ) 斜率
  −13.3 → −0.013 eV/Ry。

### Next steps
- T3（判决点）：三几何驻点 FD（冻结 t_Γ* 协议已接线：`deltap_proxy_target_file`
  + `deltap_secant off`）；预言残差 84.8 → ≤0.02 eV/Å。
- T4（外循环）：Q1 数据 dγ/dt_Γ≈0.07 → 首轮 κ=1 不足，实测 Δγ/Δt_Γ 更新 κ。

---

## 2026-08-05：F_HK 全迹对齐 + 双闭合 + T3 判决（Stage 2 判决点 PASS）

### 做了什么
- **F_HK 全迹对齐（按用户评审：先对齐再跑 T3）**：`compute_hk_force` 的
  E_HK/F_HK/U 累加从对角约定（T_pp、Π=I）扩到全迹（T_full·Π 双循环，
  Π=C_L†C_L 冻结 C 下为常数，导数链不动）。E_HK 0.05732 → **0.0471550682 Ry**
  （−17.75%，与 Gram 修正 ~18% 吻合）；FINAL_ETOT_IS 逐位不变；Γ 末态与
  T2 能量侧表逐位一致 → **力侧≡能量侧全迹**。
- **双闭合（B-7 范式，λ=+0.01 冻结）**：
  - 均匀平移 z：E_HK-FD −9.0e-5 ↔ −ΣF_HK,z −9e-5 Ry/Bohr ✓；
  - escon 总量：F_FD +0.432 ↔ −(force_deltap+f_hk) +0.4252 eV/Å（1.6%）✓；
  - 单原子 E_HK 单独 FD ✗ 4.7×——**C-响应项**（重收敛 FD = 冻结-C 解析力 +
    ∂E_HK/∂C·∂C/∂R）；B-7 的 4% 闭合是 B-6 前 τ 单位 bug（16×）掩盖的假象。
  - 总残差 ∝λ 严格线性（0.213@λ=0.01 → 0.0420@λ=0.002，比值 0.197≈0.2）。
- **修复 T3 接线断点（新发现）**：`DeltapScfSolver::inner_loop` 的 BFGS 残差
  用了 `params_.t`（per-atom 模式为空 → r=Γ−0，把 Γ 驱动到 0 而非 t_Γ*）；
  改为 operator 模式用 `scf_target`（=t_proxy=冻结 t_Γ*），gamma 模式用
  `params_.target`，矩阵模式保留 `params_.t`。
- **T3 判决 PASS**：t_Γ*=(6.698,1.977,1.977) 冻结，三几何内循环 BFGS 重收敛
  λ 使 |Γ−t_Γ*|<1e-3：F_FD(O1z)=−0.76102 ↔ F_ana=−0.7472648965 eV/Å →
  **残差 −0.0138 eV/Å ≤ 0.02 判据（84.8 → 0.0138，6100×）**。
  λ-leakage 结构 F_FD,KS=−92.92 + F_FD,escon=+92.16 相消至 −0.76（T2 平直性
  收益实证）；disp± E'(λ) 平直性复测 ✓（≤5.4e-5 eV）；λ*(R) 无分支阶梯 ✓。
- 回归：deltap_common 11/11、esolver_dp 6/6、deltap_math 3/3 PASS；
  smoothness 4/8 FAIL 为**预先存在**（B-6 参考过期，2026-08-03 已记录）。

### Files modified this round（未提交）
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（compute_hk_force 全迹；
  debug 块已翻回 #if 0）
- `source/source_esolver/deltap_scf.cpp`（inner_loop 残差改 scf_target；
  FORCE_STRESS debug 块已翻回 #if 0）
- 文档：`2026-08-05-deltap-fhk-fulltrace-t3.md`（含全部数据）、TODO Stage 2 进展

### 关键 bug/修复
- **F_HK 力侧仍用对角约定（T2 修复只覆盖能量侧）**：力侧与能量侧约定错位
  ~18% ≈ 0.5 eV/Å ≈ T3 判据的 40 倍，会让 T3 残差被已知项污染 → 改为
  T·Π 全迹后双闭合与 T3 残差同时干净。
- **inner_loop 残差用 params_.t 而非 scf_target**：per-atom 模式 t 为空，
  BFGS 把 Γ 驱动到 0，T3 冻结协议下 base 几何根本不收敛到 t_Γ* → 改用
  scf_target 后 base 收敛 λ=(0,0,0)、Γ=t_Γ* 到 4.5e-4。
- **非正交基记账通用教训**：算符期望必须走 Tr[ρ·H_sym]=T·Π 全迹，对角
  T_pp 高估 ~18%（T2 能量侧 + 本轮力侧同一教训）。

### Next steps
1. **T4（外循环）**：t_γ=0.9γ_natural，secant ≤5 步 |γ−t_γ|<1e-2；
   Q1 dγ/dt_Γ≈0.07 → 首轮 κ=1 不足，须实测 Δγ/Δt_Γ 更新 κ；Q2 翻号逻辑已在实现中。
2. smoothness 单测参考值更新（B-6 约定，~0.5h）。
3. Stage 1+2 改动保持未提交，T4 通过后统一评审提交。

---

## 2026-08-05（晚）：T3 评审闭合计算 + 统一提交 + T4a/T4b 计划

### 评审结论与落实
- **T3 判决成立但需诚实标注**：残差 0.0138 eV/Å > 严格闭合判据 0.0129（超 7%）。
  判决成立（预言带 ≤0.02 内、6100× 改善），但 PASS 必须注明参照物：
  残差 = O(λ) 驻点泄漏（λ*≠0 下 F_FD 与 F_ana(λ=0) 的系统差）。
- **闭合计算（§3.7，评审公式 + 实测数据）**：
  `leak = −λ*·(dΓ/dλ)·Δλ*/(2δ) = −(−1.09e-3)(−4.15)(−1.09e-3)/0.01 = +0.0127 eV/Å`
  （λ*=1.1e-3 取整 → 0.0129 = 严格判据）。闭合率 0.0138/0.0129 ≈ 1.07 →
  **残差 ~100% 归因为 O(λ) 驻点泄漏，无未识别项**；7% 缺口 = λ* 三位小数精度
  + ΣΓ 斜率代单分量斜率的估计误差。与"λ-leakage 84.8"轮归因同族。
- **生产验收判据形式（顺带产出）**：残差 ∝ |λ*|（§3.3 实证 ∝λ 严格线性）→
  **"在 |λ*|≤λ₀ 工作窗内残差 ≤X"**，λ→0（t_Γ*=自然 Γ）残差→0 线性。
- **评审确认的优点记录**：λ-leakage 结构实证（KS −92.92/escon +92.16 相消）；
  BFGS 接线 bug 抓得准（空 target 模式经典坑）；单原子 E_HK 4.7× 失配的 C-响应
  定性 + λ 线性验证（0.213→0.042，ratio 5.13）；FINAL_ETOT_IS 逐位不变证明
  F_HK 对齐纯力侧。

### T4 修正（评审）
- **修正 1**：不直接上 t_γ=0.9γ_natural（Δλ≈1.9 Ry 暴力微扰区）。分两步：
  T4a 机制验证（0.98·γ_natural，Δγ≈0.11/ΔΓ≈1.6/Δλ≈0.4 Ry，固定几何 scf
  外循环，验证 secant 步数预言 2–4 步、κ 更新、翻号逻辑）；
  T4b 窗口测绘（0.95→0.9 逐步，记录 (t_Γ, λ*, γ, E')，终点冻结 λ FD，
  画"力残差 vs |λ*|"曲线）。
- **修正 2**：t_Γ 单步限幅 0.5 → 1.0 rad；κ 实测后改用割线预测步长
  （实测 κ≈14 超出原 clamp 上限 3，否则 ≤5 步结构性不可能）；
  需新增 scf 模式固定几何外循环驱动（secant 更新后未达 tol 则继续 SCF）。

### 行政项
- **统一提交（本条目执行）**：Stage 1 + Γ^HK Gram 修正 + F_HK 全迹 + BFGS
  接线修复 + T2/T3 文档。commit 消息："约束激活力从 84.8 → 0.0138 eV/Å，机制闭合"。

### Files modified this round
- 文档：`2026-08-05-deltap-fhk-fulltrace-t3.md`（§3.7 诚实标注+闭合计算、§5 计划）
- TODO：Stage 2 T3 诚实标注、T4 行更新（T4a/T4b + 限幅修正）

### Next steps
1. T4a（评审修正后）：secant 限幅 1.0 + κ 割线预测 + 固定几何外循环驱动 → 跑 0.98 靶点。
2. T4b：窗口测绘 + 力残差 vs |λ*| 曲线。
3. smoothness 参考值更新。

---

## 2026-08-05（深夜）：T4a 执行——外循环被"分支锚定"阻塞（判决性发现）

### 做了什么
- **T4a 前置代码**（评审修正 2，未提交）：secant 单步限幅 0.5→1.0 rad；κ 实测后
  割线预测（clamp [0.3,3]→[0.3,20]，实测 κ≈14 必须放行，否则"≤5 步"结构性
  不可能）；新增固定几何外循环驱动：`deltap_outer_nmax`/`deltap_outer_thr`
  INPUT（scf 模式首轮 λ=0 自由测量自然 (Γ_nat,γ_nat)，secant 从自然点起步；
  |γ−t_γ| 未达标则 `outer_redrive` 请求 SCF 循环继续而非终止，重武装内循环
  λ BFGS）。编译干净，单测 11/11、6/6 PASS。
- **T4a 运行**（h2o1，t_γ=0.98×γ_natural，2-k 点，串行）：**协议无效**。
  λ=0 自由跑已报告 γ≈(−5.415,−3.529,−3.529)=t_γ 分支，|γ−t_γ|=5.09e-3<1e-2，
  外循环 1 步"收敛"且 λ 全程为 0——测量伪迹，非物理响应。
- **对照实验**：同 build 同 INPUT，仅 target 换回自然值 → γ 收敛 (−5.521,−3.601,
  −3.601)（与 T2/T3/冒烟一致）。两运行波函数相同（λ=0），报告 γ 差 0.105≈靶点差
  0.110（1:1 跟随）→ **per-atom γ 报告值被靶点分支锚定**（global target-aware
  branch selection 每次重锚到离 t_γ 最近的分支，量子 ~0.07–0.105 rad @ 2-k 网格）。
  原始 Wilson 总量不变（6.169556）。

### 根因与结论
- **分支锚定伪迹吞没了 0.98 靶点**（Δγ=0.11 ≈ 分支量子 0.105，低于可观测分辨率）；
  0.9 靶点在量子之上但 λ*≈1.9 Ry 暴力区（评审已排除）。**设计文档 Phase 0.3
  （分支连续性）从"T4 之后收尾"升格为"外循环可验证性的硬前置"**——读数连续性
  是 Route A+ 外循环的前提，不是后处理。

### 建议修复（Phase 0.3-lite，待评审）
1. 分支参考与 t_γ 解耦：报告 γ 锚定到上次测量值（跨步连续，W_prev_ 机制已有），
   t_γ 仅在无参考分支时初始化分支。
2. 自然参考分支：外循环前 λ=0 自由跑/加载 branch.dat 钉住自然分支；
   γ(t_Γ)=自然 γ+物理响应（斜率 ~0.07），0.98 靶点需 0.11 rad 物理移动（预言 2–4 步）。
3. 回归安全性：t_γ=自然 的运行（T2/T3/gamma 锚点）两锚一致 → 改动应为 no-op，
   需 natural-target 对照 + gamma 零回归验证。

### Files modified this round
- 代码（未提交）：`deltap_scf.h/.cpp`（outer_nmax/outer_thr/outer_redrive/
  first_pass_done/κ 割线/限幅）、`esolver_ks_lcao.cpp`（conv_esolver 重驱动）、
  `input_parameter.h`、`read_input_item_other.cpp`
- 文档：`2026-08-05-deltap-t4a-branch-anchor.md`（T4a 发现全记录）

### Next steps（待评审）
1. 批准/修改 Phase 0.3-lite 方案 → 实现 → 回归（natural-target 对照 + gamma 零回归）。
2. 重跑 T4a（0.98）→ T4b（0.95→0.9 窗口测绘，力残差 vs |λ*| 曲线）。
3. smoothness 单测参考值更新。

---

## 2026-08-05: Route A+ 版算法完整推导合并文档

### What was done
输出 `2026-08-05-deltap-algorithm-derivation-route-a-plus.md`——当前开发版本
（escon=−λΓ + Gram 全迹 + 完整力求解）的唯一完整算法文档：γ 定义与测量链、
H_c=H_HR+H_HK 定义、Γ 逐原子定义（含 Π 全迹）、E'≡E_KS(ψ*) 恒等式、
力公式全集（A1/A2/B + O(λ) 残余）、应力同构、三级 λ 调节、E_eff=λ/(2a) 新公式、
测量纪律（target-aware 不可作判据/连续性锚/未扣除力/高精度处方）、
验证状态表与 LIMITATION 5 条、历史版本数值换算。
07-12 主文档自此标记为"历史骨架"。无代码改动。

---

## 2026-08-07: Route A++ 文档评审与执行方案三层调整

### What was done
评审 `DeltaP-RouteA++混合规范严格推导与改进设计.md`，输出
`2026-08-07-deltap-route-a-plus-plus-review-and-plan.md`：
- 采纳：§1.4 定理（H_HR 是 θ_n→τ_α 代理，误差 ∝ λ·spread_I——A+ 适用域
  可计算判据）、H_HK 应力推导、PW 记账统一、⟨η⟩ 审计；
- 修正：η→1/λ*≥5× 降调为实测记录（Cauchy–Schwarz 仅给弱下界）；
  Tr[ρ·Ô_θ]=0 齐次性需数值验证（新增 V-H0）；
  **新增风险：θ_n 进哈密顿量使分支跳变改变势本身**（D2 设计约束：初始化
  锚定连续值 + 跳变冻结，新增 V-H8 SCF 稳定性测试）；
- 执行三层：基线 Route A+（T4b 并入 V-H3）/ L1 本周（PW 记账、⟨η⟩、spread）
  / L2（应力）/ L3 研究轨道（Ô_w 先行，编译开关可回退）；
- 锚点第 4 次重建推迟到 L3 全绿后一次；执行纪律 D1-D7 + 测试点 V-H0~V-H8。

### 2026-08-07（补）: 执行 TODO 落地为 T-1~T-10
- `2026-08-04-deltap-execution-todo.md` 顶部插入"当前 TODO（2026-08-07 调整后）"
  权威节：T-1 Phase 0.3-lite 分支锚定门控（先做）→ T-2 T4a' → T-3 commit →
  T-4 PW Γ 记账 → T-5 ⟨η⟩ 输出 → T-6 spread_I 输出 → T-7 V-H2/V-H5+A+ 适用域
  文档 → T-8 hk MPI → T-9 应力+V-H7 → T-10 Ô_w（开关隔离+D2 约束）。

---

## 2026-08-09: Phase 0.3-lite 冻结位移实现 + T4a' 重跑（外循环机制判决失败，根因钉死）

### What was done
- **Phase 0.3-lite 补完（frozen branch shift）**：continuity 模式 Stage B 读数从
  "最近格点→锚"改为 `avg_raw + frozen_shift`（`branch_shift_` 在 SCF 收敛时由
  `freeze_branch_ref()` 冻结自 `last_shift_`）——修复"最近格点把读数钉在锚
  ~0.005 内、吞掉物理响应"的缺陷。三 Stage-B 路径（约束/总量/逐原子）均记录
  `last_shift_`；首轮无 shift 时 fallback 到 ref_gamma_ 锚；Stage A 继续用冻结
  ref_gamma_（不随计算漂移）。gamma 模式锁定 target 锚（零回归门控不变）。
- **T4a' 重跑两轮（单任务 MPI，均到 nmax=8 自然结束）**：
  - 实验 A（内循环 max_step=0.005 原值）：λ=0 自由首测报告自然 γ ✓；frozen
    读数跟随 raw ✓（bdist 1e-3–1e-2）；物理斜率恢复 κ_O=12.3（dγ/dt_Γ=0.081）
    ✓；但外循环发散（|γ−t|∞ 0.110→0.225，WARNING×3）。
  - 实验 B（max_step=0.1 加速实验）：内循环失控（Γ_O 到 8.6、λ 到 Ry 量级、
    两个 H 报告分裂），**改动已回退**。
- 文档：`2026-08-09-deltap-t4a-frozen-shift.md`（完整数据表 + 三层根因分析）。

### Root cause（T4a' 外循环发散，三层钉死）
1. 内循环 BFGS `max_step=0.005 Ry` 限制 λ 预算 = 0.1 Ry/外循环步，0.98 靶点需
   Δλ≈0.2–1.0 Ry → Γ 追不上 t_Γ（|Γ−t_Γ| 卡 0.8–0.9）→ secant 在非松弛点
   测量，κ 翻号振荡（+12.3→−20→+10）。
2. 加速实验：单 α 线搜索无法同时驱动反号分量（O 需 λ 负、H 需 λ 正）；
   H 的 Γ_H(λ_H) 非单调（U 形，λ<−0.1 悬崖）。
3. **H 原子 Γ-proxy 退化（物理层，关键发现）**：dΓ_H/dλ_H≈0（U 形底）但
   dγ_H/dλ_H≈+0.27 强——Γ 旋钮对 H 的 γ 几乎无控制力，secant 的 t_Γ_H 代理
   结构性失效。Route A+ 外驱动合法工作区 |Δγ|≲0.03–0.05/原子（线性窗），
   0.98 靶点（Δγ≈0.11）超出。

### 验证状态
- 单测：math 3/3、gauge 4/4 PASS（smoothness 4 FAIL 既有，与本改动无关）。
- 参考 ref2（target=自然）：31 迭代收敛 γ=(−5.521,−3.601,−3.601)、λ=0，
  外循环 |γ−t_γ|∞=2.12e-3 < 1e-2 正常终止——冻结位移路径零回归。
- T-1 验收：③ λ=0 自由首测自然 γ ✓；gamma 零回归门控保持（未跑 BN 对照，
  基线存档无效，见交接记录）；② natural-target no-op 由 ref2 覆盖 ✓。

### File list（本轮新增/修改）
- `source/source_lcao/module_deltap/deltap.h`：branch_shift_/last_shift_/
  has_branch_shift_ 成员 + load_branch_ref/freeze_branch_ref/set_branch_anchor
- `source/source_lcao/module_deltap/deltap_wannier.cpp`：Stage A/B continuity
  门控、Stage B 冻结位移读数 + last_shift_ 记录、freeze_branch_ref 冻结 shift、
  SAdbg/SBdbg 取证打印
- `source/source_esolver/deltap_scf.{h,cpp}`、`esolver_ks_lcao.cpp`、
  `input_parameter.h`、`read_input_item_other.cpp`：T4a 外循环驱动
  （outer_nmax/thr、λ=0 自由首测、secant 从自然起点、κ 上限 20、限幅 1.0）
- 文档：`2026-08-09-deltap-t4a-frozen-shift.md`

### Next steps
1. **T-3 commit**：Phase 0.3-lite + T4a 前置代码 + 本轮文档（max_step=0.1 已回退）。
2. T4a' 重设计（评审 3 方案）：小靶点线性窗验证 / per-atom 内循环解耦 /
   外循环直接 γ 残差驱动。
3. 每轮 T4a 重跑后从 ref2 恢复 branch.dat（save_branch 覆盖锚）。

## 2026-08-11: T-4' γ 直驱实现 + T3' 复判（FAIL，结构性）+ T-9' branch 守卫

### What was done
- **deltap_drive 开关（T-4'）**：INPUT `deltap_drive=proxy|gamma`（operator 模式）。
  gamma 直驱 = λ 残差直接驱动报告 γ→t_γ，secant/t_Γ 翻译层退役
  （`secant_update_proxy` 短路、first-pass 门控排除、proxy_target_file 跳过）；
  **escon 记账永远 −λ·Γ**（与驱动信号无关，一行定理落点）。legacy gamma 模式
  锁定 gamma（与 branch_anchor 锁定同模式，零回归）。
- **T-9' branch.dat 写入守卫**：INPUT `deltap_branch_write`（默认 true 保持跨
  运行行为；FD/多几何设 false 防参考静默覆盖——07-30 A/B、T4a 参考被毁两案）。
- **T3' 复判（γ 驱动驻点 FD）**：三几何跑完，残差 −0.502 eV/Å = 判据 25× /
  严格闭合 39× / T3 代理版 36× 差 → **FAIL（结构性）**。
- 回归：单测 11/11 + 3/3 + 4/4 PASS（smoothness 4 FAIL 既有）；MPI smoke 4/4。
- 文档：`2026-08-11-deltap-gamma-drive-t3prime.md`。

### T3' FAIL 根因（三层，定量）
1. **γ↔λ 弱耦合（评审 Q1 数据自证）**：dγ/dλ≈−0.3 vs dΓ/dλ≈−4.2 → γ 驱动 FD
   腿需 λ* ~14× 于代理驱动 → O(λ)² 泄漏 ~200×（leak∝λ*²·dΓdλ/δ：
   代理版 0.0127 eV/Å@λ*=1.1e-3 vs γ 驱动 ~1.3 eV/Å@λ*=0.011）。
   "T3' 更干净"的前提（λ* 小）被弱耦合破坏——力一致性是记账性质，但残差
   大小是 λ*(R) 路径性质。
2. **冻结-自洽响应符号分裂**：+δ 腿内循环在冻结密度下 |γ−t|→6e-4（λ=−0.021），
   密度再平衡后自洽残差回 4.9e-3（λ 推反了）。冻结 λ 验证：dγ0/dλ0=+0.5@λ0=−0.01
   vs −0.38@λ0=−0.021——γ(λ) 非单调，与 Γ 的线性 −4.2 对比鲜明。
3. **装置教训**：首轮缺 branch.dat → 连续性锚无参考 → Stage-B 回退 target-aware
   → 报告 γ 钉在靶点（自证循环，T4a 同款）→ 伪残差驱动 λ=6.4e-3。补参考后
   base 干净（|γ−t_γ*|→9e-16、λ*=0、E' 与 T3 base 逐位一致）。
   规则：**用 γ 报告做任何判据前必须确认连续性锚已激活**。

### 历史印证
08-03 组② 驻点实验（旧记账 γ 驱动）残差 84.8 eV/Å（λ-leakage 5 位闭合）；
Route A+ 换 Γ 代理记账才到 0.0138。T3' = 08-03 换新记账重跑——新记账修 escon
恒等式，没修 γ↔λ 弱耦合本身，失败继承。驻点 FD 唯一合法协议 = Γ 代理驱动
（T3 PASS 成立）。

### File list（本轮新增/修改）
- `source/source_esolver/deltap_scf.{h,cpp}`：drive 门控（observable/target/
  init/inner_loop/update_lambda_gd/escon/secant 全路径）
- `source/source_esolver/esolver_ks_lcao.cpp`：drive plumbing + gamma 锁定 +
  branch_write plumbing
- `source/source_io/module_parameter/input_parameter.h`、
  `read_input_item_other.cpp`：`deltap_drive`、`deltap_branch_write` INPUT
- `source/source_lcao/module_deltap/deltap.{h,cpp}`：`set_branch_write` +
  `save_branch()` 守卫
- 文档：`2026-08-11-deltap-gamma-drive-t3prime.md`

### Next steps
1. **commit 本轮**（T-4' 实现 + T3' 判决文档 + T-9' 守卫）。
2. T3' 不入生产协议；T-7'（per-atom Jacobian / 自洽 λ 重触发）获新前置证据。
3. T-5' 窗口测绘改用 proxy 驱动；T-6' Ô_w 提前（评审已裁，H γ 可达性同源）。

---

## 2026-08-12: 公式重推导 + 代码评审 + 风险提示（T-6' 工作区）

### What was done
重推导当前算法完整公式链（测量/H_c 两模式/Γ 记账恒等式/驱动/力/应力/E_eff），
评审 HEAD + 未提交 T-6'（Ô_w）实现，输出
`2026-08-12-deltap-formula-review-risks.md`。核心发现：
- **R1（阻塞）**：ow 模式 F_ow 完全缺失，且 force_stress.hpp 注释谎称存在
  "compute_hk_force 'ow' branch"（实际没有）——ow 模式 H 含 H_ow 但 Pulay 力缺席；
- **R2/R3（高）**：H_ow/θ 只覆盖 string-0——多 string 网格（BN 2×2×2）仅 2/8 k
  点拿到算符；Ô_w 是 k 局域算符，应移出 link 循环按 nks 逐 k 构建
  （顺带解除方阵限制）；
- **R4（中）**：D2 跳变冻结无恢复路径（prev 不更新则永久冻结）、与 gamma
  驱动存在失配窗口、首轮裸 θ 未定锚；
- **R5（中）**：ow 模式 T2 类 E'(λ) 平直性验证未做（地基，~10 min）；
- R6（ow+应力静默缺失，建议 WARNING_QUIT）/ R8（e_w_I 取实部建议加断言）。
处置顺序：R5 → R1 → R2/R3 → R4 → V-H8/V-H3'。R1+R2 修复前 ow 数据不作证据。

### 2026-08-12（补）: T-6' 修复 TODO 落地（T-11~T-18）
- 执行 TODO 文档新增 T-6' 修复节：T-11 ow 模式 E'(λ) 平直性（R5）→ T-12
  F_ow 实现（R1 阻塞，含 force_stress.hpp 虚假注释修正）→ T-13 H_ow/θ k 局域化
  （R2/R3，顺带解除方阵限制）→ T-14 D2 冻结恢复+首轮定锚（R4）→ T-15/T-16
  （应力 WARNING_QUIT、虚部断言，并入 T-12 commit）→ T-17 V-H8 → T-18 V-H3'
  Ô_w 判决（dγ/dλ≥3 rad/Ry + γ-hold FD < 0.02 eV/Å，判决点）。

### 2026-08-12（补 2）: T-6' 实现轮收尾——R5 平直性首测 PASS 信号 + V-H8 极限环定位

#### What was done
- 落地 R1–R6/R8（见 `2026-08-12-deltap-ow-t6mode.md` §4.3）：F_ow 实现接线、
  H_ow/θ 全 string k 局域化、D2 冻结恢复 + 首轮锚定、ow+应力 WARNING_QUIT、
  Im(Γ^w) rank-0 软告警；提交前补 `compute_ow_force` MPI 防御守卫。
- **R5 首测（ow λ 扫描 ±0.01，冻结 λ）**：±0.001 收敛窗斜率
  [−0.001,0]=−0.361、[0,+0.001]=+0.395 eV/Ry，对称 λ² 抛物，
  线性系数 a≈+0.017 eV/Ry（T2 参照 −0.013 同量级，判据 ≲1 的 ~1/60）→
  **ow 记账恒等式在收敛窗成立（R5 PASS 信号）**。诚实标注：±0.003/±0.01
  四点在 50 iter 内 SCF 未收敛，只作稳定性判据不作记账判据。
- **V-H8（ow λ=+0.01 长跑 150 iter）**：未收敛（极限环，Γ/escon 稳定但
  drho 不降），**D2 跳变冻结零触发** → 不稳定性是状态依赖 H_ow 自身
  （非 2π 分支跳变）；同参数 proxy 27 iter 收敛 → ow 特有实锤。
- **λ=0 零回归**：ow vs proxy E 逐位同（−481.6973727147502 eV）、γ 自然值
  同、escon=0 同；Γ 不同为预期（不同算符自然值）。
- 回归：编译 ✅；`ctest -R deltap` 3/4 ✅（smoothness 4/8 = R9 预先存在）。

#### Key conclusions
1. **ow 记账恒等式**在收敛窗内成立（Γ^w 全 Gram 迹 + Γ^HK 记账与施加的
   H_ow/H_HK 一致）——"非正交基记账走全迹"在 Ô_w 上第二次独立复现。
2. **V-H8 判定**：H_ow 极限环不是分支跳变问题（R4 已排除），是状态依赖
   哈密顿量刚度问题——T-6' 下一主线 = H_ow SCF 稳定化（内循环冻 θ /
   混合 / θ 限幅），否则 V-H3' 在 λ*~2e-3 之外无法执行。
3. ow 的 Γ 自然值 (−19.1,−4.6,−4.6) 与 proxy (4.4,1.4,1.4) 不同是算符语义
   差异，λ=0 时 H_c=0 不受影响（E/γ/escon 全同）。

#### File list
- `source/source_lcao/module_deltap/deltap.h`、`deltap_wannier.cpp`：
  Ô_w θ 捕获（per-k）/fill_kstring/H_ow 构建/Γ^w/anchoring/D2 冻结恢复/
  compute_ow_force/R8 告警/MPI 防御守卫
- `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`：
  ow 门控（cal_stress→QUIT、力清零）+ 虚假注释修正
- `source/source_lcao/module_operator_lcao/deltap_lcao.cpp`：ow 模式
  contributeHR 门控（H_HR 不重复计入）
- `source/source_esolver/esolver_ks_lcao.cpp`、`deltap_scf.h`：
  operator_mode plumbing
- `source/source_io/module_parameter/input_parameter.h`、
  `read_input_item_other.cpp`：`deltap_operator_mode` INPUT
- 文档：`2026-08-12-deltap-formula-review-risks.md`（评审）、
  `2026-08-12-deltap-ow-t6mode.md`（本轮）

#### Next steps
1. **commit 本轮**（R1–R6/R8 + R5 首测 + V-H8 定位 + 评审文档入树）。
2. **H_ow SCF 稳定化**（V-H8 主线，T-6' 继续）：内循环冻 θ / 混合 / θ 限幅
   → λ=±0.01 收敛 → 重跑 R5 全窗做满"斜率 vs |λ|"。
3. **V-H3' 判决**（γ-hold FD < 0.02 eV/Å @ 0.98 靶点，须先过 2）。
4. R1 F_ow 闭合计算（单原子 E_ow FD ↔ 解析 F_ow，进 T-7' 前）。

---

## 2026-08-12（晚）：T-17 V-H8 SCF 稳定化（S1 冻结核）— h2o1 极限环消除

> 轮文档：`2026-08-12-deltap-ow-scf-stab-t17.md`。S0 诊断（前轮）：周期 2 极限环
> = 活算符（每 iter_finish 用活 C 重建 H_ow）在双近邻自洽解间往返（Γ_O1
> −20.304↔−20.359 @ λ=+0.01；β 只选盆）。

### 修改（T-17，S1）

- **`deltap_wannier.cpp`/`deltap.h`**：ow 算符核拆分 + 冻结。
  - 新 `compute_ow_kernel()`：渡边处从当前 ψ 构建冻结核（per-atom
    K_I[μ][n]=Σ_lm S_k·D_I 无 λ、C 快照、T_I[m][n]=Σ_lm D*·D、θ 快照），
    并缓存已施加算符 Γ^w（`ow_gamma_w_frozen_`，全 Gram 迹）；
  - `compute_hk_correction` ow 块：kernel 有效 → 冻结路径（A_C(λ)=Σλ_I·K_I
    精确缩放，无 fill_kstring/ψ 访问）；kernel 无效 → 原 live 路径（防御）；
  - `compute_gamma_op_hk` ow 块：渡边（stale||invalid）→ 重建核；否则返回
    缓存 Γ^w（记账 = 已施加算符，HG-2）；γ 报告两种 drive 均 live；
  - 渡边触发：首次测量、P2 `on_phase2`（esolver `mark_ow_kernel_stale`）、
    `freeze_branch_ref`；**D2 恢复不再重建核**（冻结 λ 扫描下算符应固定；
    否则近简并体系每 ~50 冻结步被踢 0.004→0.08）。

### 结果（h2o1，R5 全窗全部收敛）

| λ (Ry) | E' (eV) | Γ_O1 | iter |
|---|---|---|---|
| −0.01 | −481.6753785535242 | −18.534 | 39 ✓ |
| −0.003 | −481.6951438739899 | −18.926 | 36 ✓ |
| −0.001 | −481.6971248710081 | −19.036 | 36 ✓ |
| 0 | −481.6973727147502（逐位同旧） | −19.111 | 26 ✓ |
| +0.001 | −481.6971250754233 | −19.175 | 34 ✓ |
| +0.003 | −481.6951654681053 | −19.307 | 36 ✓ |
| +0.01 | −481.6752569479165 | −19.772 | 41 ✓ |

- R5 斜率：±0.001 两弦 +0.2478/−0.2476 eV/Ry → 线性系数 **a=0.000 eV/Ry**
  （旧 a=0.017；判据 ≲1，对照 T2 −0.013），λ² 曲率 b≈247.6 eV/Ry²；
- **极限环消除**：41 iter 收敛、Γ 单解逐位稳定、无 150 iter 翻跳；β=0.1 vs
  0.4 收敛到冻结解（ΔE=0.19 meV，旧盆选择 3.2 meV）；
- 零回归：ow λ=0 与 proxy λ=+0.01 均逐位一致；ctest 3/4（smoothness R9 预
  先存在）。

### BN 定位（T-17 两体系验收的诚实标注）

- BN ow λ=+0.003：S1 后 P1 达 drho<1e-3、P2 渡边正常、D2 踢消除，但 P3 密度
  在 ~0.003–0.013 小幅振荡不达 1e-8；
- **BN proxy（pre-S1 等价路径，S1 零介入）同型不收敛**（~0.003–0.007）→
  BN 不收敛 = 活 H_HK + 近简并能带（E_gap≈0.02 eV）的既有刚性，非 Ô_w/S1
  回归（gamma 模式无 H_c 16 iter 收敛，定位在 H_c）。残余问题归 S2 矩阵阻尼。

### Key conclusions
1. **S1 冻结核 = 消除 H_ow C 依赖极限环的充分条件**：渡边之间 SCF 见固定
   H_ow(λ)，λ 依赖经 per-atom 核精确缩放（内循环 BFGS 可复用）；
2. **记账一致性 HG-2 由构造满足**：Γ^w 报告 = 已施加算符（Γ^w_applied ≡
   Γ^w_measured），escon = −λ·Γ 逐点精确；
3. **γ 报告保持 live**：S1 只冻结算符构造，不触碰 drive 机制（λ 更新信号）；
4. 实测偏差（相对 S1 手记 point 5）：扫描 INPUT 全部 γ-drive，冻结不 gate
   drive（否则 T-17 验收对象不受任何影响）；γ-drive 下冻结已由 h2o1 全窗证实。
5. 通用教训：**"状态文件/算子冻结的恢复阈值若在固定约束协议下触发，会周期性
   踢回系统"**——D2 恢复只更新读数、不重建冻结算符（近简并体系实测证据）。

### File list
- `source/source_lcao/module_deltap/deltap.h`：`mark_ow_kernel_stale`、
  `compute_ow_kernel` 声明 + 冻结核成员（ow_K_I_k_/ow_C_k_/ow_T_I_k_/
  ow_theta_frozen_k_/ow_gamma_w_frozen_/valid/stale）
- `source/source_lcao/module_deltap/deltap_wannier.cpp`：`compute_ow_kernel`
  实现；`compute_hk_correction`/`compute_gamma_op_hk` ow 块冻结路径；
  `freeze_branch_ref` 标 stale
- `source/source_esolver/esolver_ks_lcao.cpp`：`on_phase2` 标 stale（P2 渡边）
- 文档：`2026-08-12-deltap-ow-scf-stab-t17.md`（本轮）

### Next steps
1. **commit 本轮**（T-17 S1 + R5 全窗 + BN 定位 + 轮文档）。
2. **T-18（V-H3' Ô_w 判决）**：h2o1 ow 全窗收敛已解锁——dγ/dλ 实测（预言
   ≥3 rad/Ry，对照 proxy 0.3）+ γ 直驱驻点 FD @ 0.98 靶点，残差 <0.02 eV/Å。
3. **BN**：S2 矩阵阻尼 / H_HK 冻结（近简并体系，ow/proxy 共同受益，独立立项）。
4. 内循环（nscf>0）+ ow 的 BFGS 残差读数冻结问题记录在案（当前无用例）。

---

## 2026-08-12 深夜——T-18（V-H3' Ô_w 判决）FAIL：dγ/dλ 实测未达预言，γ-hold 路径仍不可用

### Round summary
T-17（S1 冻结核）commit 后立即进入 T-18 判决点。本轮**零代码改动**（纯测量）：
（1）从 T-17 R5 全窗扫描提取高精度收敛态 γ（`[SBdbg]` best，9 位小数）；
（2）两次 γ-drive 驱动探针（自然靶点验证 + O1 靶点 +0.02 rad 负结果）。
结论：**预言 dγ/dλ ≥ 3 rad/Ry 未达（实测 ~0.26 大尺度、~1 量级局部且符号
翻转）→ T-18 FAIL by prediction → 0.98 靶点 FD 不执行（已知会偏离），
按 TODO 纪律回来评审，不评估 Ô_θ（L3.2）、T-7' 不提前。**

### Key results（全部收敛态、S1 冻结核）
- R5 全窗 γ(λ)：O 原子大尺度 dγ₀/dλ ≈ +0.26 rad/Ry（[−0.01,+0.01] 窗）；
  局部弦斜率 ±0.8~±1.4 且符号多次翻转 → 非单调（V-H8 必查项 FAIL）；
  H 原子 ≈ +0.17。与 proxy 的 0.3–0.5 同量级 → Ô_w 耦合增强至多 ~2×，
  远非所需 10×。
- 驱动探针 a2（O1 靶点 +0.02 rad）：内循环 BFGS（冻密度）20 步 rms 平台
  9.8e-3~1.47e-2、α 翻号，**不收敛**；λ* 停在 −0.0156（末次试验点）；
  密度弛豫后自洽 γ0=−6.400316 vs 靶 −6.379392，|γ−t|∞=2.09e-2（21× conv_thr）
  ——γ-hold 连 0.02 rad 靶点都钉不住。冻结-自洽响应分裂在 ow 模式重现且更极端
  （自洽 dγ₀/dλ ≈ +0.06，方向与靶点相反）。
- a1（自然靶点）验证通过：λ*=(0,0,0)、|γ−t|→5e-11、E' 与 t17e2 十三位一致。

### Analysis
- 预测条件未实现：θ_n·P̂_I 直接微分相位通道的 10× 耦合增强被实测否定；
  γ(Wilson 相位)对 H_ow 的响应被密度弛豫抵消，Ô_w 未改变 γ↔λ 弱耦合值域。
- 强跑 0.98 FD 的可预言残差：λ*≥0.1 Ry（乐观外推）→ leak ≥5× T3' 的 0.502
  ≥ 2.5 eV/Å（判据 125×）——已知会偏离，不跑。
- 非 S1 伪影：R5 是 S1 后唯一可收敛 ow 数据；a2 驱动探针（独立协议）交叉确认。
- 给评审的开放问题：γ 弱耦合+非单调是测量通道问题还是算符控制权限问题；
  Ô_θ 变体若仍基于 γ 报告则需先在"密度-相位响应"层面重新设计。

### File list
- `docs/superpowers/specs/2026-08-12-deltap-ow-vh3-t18.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（T-18 状态更新）

### Next steps
1. T-18 FAIL 归档，等评审裁定（Ô_w 之后的路：换算符设计 / γ 报告通道重构）。
2. T-5'（窗口测绘，proxy 驱动）与 BN S2（矩阵阻尼）不受影响，可并行。
3. T-7'（per-atom Jacobian）维持原序，不因本轮提前。

---

## 2026-08-12: "物理刚度"假说与 D1–D3 判决实验设计

### What was done
针对 T-18 FAIL 的开放问题（测量通道 vs 算符权限），提出第三假说并成文
`2026-08-12-deltap-physical-stiffness-hypothesis.md`：
- **假说**：h2o1 的 γ 刚度 κ≈20-30 Ry/rad² 由极化率决定（κ_phys=(eL/π)²/α≈18.6），
  响应 0.26 rad/Ry 已贴近物理上限，proxy 与 Ô_w 同量级即证据；
- 定量链：G↔μ（F2 校准）→ κ_phys 估计 → T2 抛物线+T-18 响应联合估 κ≈30；
  三条 E_eff 候选公式张力表（(a) λ/2a 低估、(b) πλ/L、实测值更靠近 (b)）；
- 判决实验：D1（E(G) 直接拟合 κ，复用+4 个补充 λ 点）、D2（锯齿场交叉）、
  D3（γ 噪声底——T-18 非单调可能只是噪声主导的前置检查）；
- 决策树：κ∈[10,40] → 叙事修正+EFC 换锚"路径泄漏消除"+A+ 窗口产品化；
  κ≲2 → 算符权限成立，Ô_θ 重评。
无代码改动。

---

## 2026-08-13——T-5' 窗口测绘完成：0.995/1.005 双向不可达，可达域量化

### Round summary
零代码改动（纯测量 + 文档）。原协议（proxy 驱动 + secant 外循环）两向均发散
（t0995/t1005：κ=1 首轮方向错误、修正 κ 超调、λ 顶预算 0.1、SCF 破坏），
改走 T3 式冻结 t_Γ 协议（`deltap_proxy_target_file` + secant off + 内循环），
12 点全部 SCF 收敛，测得干净的可达域曲线。

### Key results（全部收敛态、连续性锚、proxy 模式）
- O 通道可达域：Δγ_O ∈ [−0.004, +0.013] rad（λ∈[−0.1,+0.1]，预算封顶）；
  0.995 侧（+0.013）比 1.005 侧（−0.004）宽 ~3×，但都达不到 0.995 靶（+0.0276）。
- H 通道死：λ_H=±0.1 下 γ_H 仅 ±0.001 rad（Γ_H U 形退化 dΓ_H/dλ≈−0.35；
  dγ_H/dλ≈±0.01）——"单侧可达域"不存在，双侧均不可达。
- 约束松弛失效：|t_Γ−Γ_nat|≳0.1 时 |Γ−t_Γ| 残差 0.08–0.55（判据 100–500×）。
- secant 外循环（proxy）对任何非自然 t_γ 发散（与 T-2 0.98 靶同族）。
- 交叉印证：ow dγ/dλ≈0.26（T-18）与 proxy 可达域 ±0.01（T-5'）一致——
  γ↔算符耦合 ~0.1 rad/Ry 量级是物理强度；换算符未改变值域。

### Analysis
γ-hold 约束的可达域问题是"算符-观测量耦合物理强度"，不是协议选择：
两条独立路径（secant 外循环 / Ô_w 换算符）均已实证关闭。LIMITATION 更新：
合法工作窗 |Δγ|≲0.01 rad/原子（O），H 通道不可用。

### File list
- `docs/superpowers/specs/2026-08-13-deltap-t5-window-sweep.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-05-deltap-algorithm-derivation-route-a-plus.md`
  §8 LIMITATION 项 1 更新（量化可达域 + Ô_w 未扩窗）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（T-5' 状态）

### Next steps
1. T-5' 完成归档；等评审裁定 T-6''/T-7' 序（开放问题偏向"算符控制权限"，
   T-5' 数据支持）。
2. T-8'（L1 三件：PW Γ 记账 / ⟨η⟩ / spread_I）可穿插。
3. Stage 3 hk MPI、L2 应力生产面顺序不变。

---

## 2026-08-13——D1+D2：κ 拟合病态判定 + 锯齿场交叉验证（物理刚度假说执行完毕）

### Round summary
纯测量轮（9 次串行 SCF ~25 min，零代码改动）：D1 冻结 λ 五点（−0.05/−0.02/0/
+0.02/+0.05，proxy）+ D2 锯齿场四点（±0.001/±0.002 a.u.，λ=0 冻结、dip_cor=1）。
结论：**D1 的 E'(G) κ 拟合病态（判据作废）；D2 直接测量 dG/dE=0.316 rad/a.u.
（低预言 3.3×，超 2× 判据）→ 物理刚度坐实且更强（α_LCAO=3.02 Bohr³、
κ_phys=60.4 Ry/rad²）；E_eff 公式 (b) πλ/L 胜出（低估 1.6–3.1×），
代码 (a) 低估 5–10×。**

### Key results
- D1：E'(λ) 曲率 b≈15–30 eV/Ry² 被 escon/Γ 通道主导（γ 通道 ~2%），E'(G) 非单值
  抛物线 → κ 拟合判据不可执行；dΣγ/dλ=+0.056 rad/Ry（线性良好）→ dG/dλ=+0.028
  rad/Ry（proxy）。
- D2：dΣγ/dE=+0.632 rad/a.u.（5 点 LSQ 残差 σ=9.3e-5 rad，线性干净）→
  dG/dE=+0.316；逐原子 O−1.58/H+1.11 rad/a.u.（反号电荷重排）；
  E'(E) 线性项 μ₀=1.94 D（实验 1.85 D，5% 吻合——偶极记账自洽）；
  E'(E) 二次项正曲率（+3.6 meV @|E|=0.001）——约束态污染，α 只能从 γ(E) 取。
- E_eff 裁决：实测 0.177 a.u./Ha（proxy ΣG）/ 0.329（ow O）；(a) 0.0333 出局，
  (b) 0.1047 胜出。代码 `E_eff` 打印待换 (b)。
- 对 T-18 的重新解读：ow 0.26 rad/Ry 相对 D2 物理上限 1/κ≈0.017 rad/Ry 已放大
  ~15×——弱耦合=体系物理（基组极化率），无算符提升空间；EFC 价值重锚"路径泄漏
  消除"（假说 §4.1 路线 2）。

### Analysis
物理刚度假说在数值体系层面成立且更强：真实场也只能以 0.32 rad/a.u. 移动 G，
推 0.1 rad 需 E≈15 V/Å（暴力）。假说与实验 α 的差距（3.0 vs 9.8）是 LCAO 紧缩
基局限，非算符失配。γ-hold 工作窗按泄漏预算反解（评审修正）：|λ*|_max≈0.4–1.6e-3
  Ry → |ΔG|_max≈1e-5–5e-5 rad——生产只允许靶点≈自然值（T-5' ±0.013 rad 仅是
  SCF 可达域）。

### File list
- `docs/superpowers/specs/2026-08-13-deltap-d1-d2-kappa-field.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-12-deltap-physical-stiffness-hypothesis.md`
  （§6 执行结果 + §2/§4 数字修正）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（状态更新）

### Next steps
1. ✅ 代码 `E_eff` 打印已换 (b)（`deltap_scf.cpp` report()，operator 模式
   `π·λ_avg/(2·a_alpha)`，gamma legacy 原样）——lamm002 重跑 E_eff
   −1.714e-2 → −5.386e-2 V/Å，FINAL_ETOT_IS 逐位一致（纯打印零回归），未提交；
2. ✅ LIMITATION §8：已按泄漏预算重写工作窗（|λ*|_max≈0.4–1.6e-3 Ry、|ΔG|_max≈
   1e-5–5e-5 rad；可达域 ±0.013 rad 仅 SCF 口径）+ E_eff 换算写死（print 已带
   响应校准注释）；
3. T-9'（branch 写守卫）随下个 commit 落地；然后 T-7'（per-atom Jacobian）；
4. D3 不再单独跑（D2 残差 σ=9.3e-5 rad ≪ 1e-3 已覆盖噪声判据）。

---

## 2026-08-13: 新旧方案通俗对比文档

### What was done
输出 `2026-08-13-deltap-scheme-comparison-plain.md`：面向材料背景读者的
通俗解说——共同测量链（Wilson 环）→ 旧方案（约束 γ、记账 −λγ，账算不符
导致能量斜/力错 84.8/靶点跟随假收敛）→ 新方案 Route A+（约束 Γ、记账 −λΓ、
E'≡E_KS 恒等式、力残差 0.0138）→ 诚实代价清单（翻译层、弱耦合窗口=物理
刚度 κ≈60、H 死通道、E_eff 换公式）→ 后续方向（Ô_w 已否、EFC 换锚泄漏消除）。
含关键数字速查表。无代码改动。

---

## 2026-08-13（补）——评审修正轮：口径对照表 + E_eff 校准注释 + 泄漏预算工作窗

### Round summary
响应评审四项要求（D1/D2 提交前）：Q1 口径对照表、Q2 E_eff 响应校准、
工作窗改按泄漏预算反解、P 系列基组重定标警示。零物理改动（E_eff 打印注释
一行，已重跑逐位零回归）。

### Key results
- **Q1 口径对照表**（写入 `2026-08-12-deltap-ow-vh3-t18.md` §1.5 与主算法文档
  §8 LIMITATION #1）：五个响应口径（ow-R5-大尺度 0.26 / ow-a2-自洽 0.06 /
  ow-D2反推 0.052 / proxy-D1-自洽 0.028 / 物理上限 1/κ≈0.017，rad/Ry）。
  "T-18 与 proxy 同量级" 仅成立在 R5-大尺度 vs 旧 proxy 混合口径；按自洽总 G
  口径 ow(0.052)=1.9× proxy(0.028)，proxy 已贴 1/κ 上限 1.6×——"无算符提升
  空间"结论明确建立在自洽总 G 口径上。密度弛豫屏蔽 80–95%（ow 冻结 0.8–1.4
  vs 自洽 0.06–0.26）。
- **Q2 E_eff 打印**：operator 模式行尾加 `[formula (b) πλ/(2a); D2
  response-calibrated ×~1.6 (proxy) / ×~3.1 (ow O), see ...]`；用户手册 §8
  输出说明同步。lamm002 重跑 E_eff −5.386e-2 V/Å（同前）+ 注释，FINAL_ETOT_IS
  逐位一致（−481.6909131438185）。
- **工作窗（泄漏预算反解，取代拍脑袋的 0.005 rad）**：T3 锚 0.0138 eV/Å @
  λ*=1.09e-3 → 斜率 12.7 eV/Å/Ry；T3' 0.502 @ 1.05e-2 → 斜率 ~48。按 0.02
  eV/Å 预算：**|λ*|_max ≈ 0.4–1.6e-3 Ry → |ΔG|_max ≈ 1e-5–5e-5 rad
  （0.01–0.1 mrad）**；评审 dλ*/dR 模型更严（~2e-6 rad）。ΔG=0.005 rad →
  λ*=0.18 Ry → leak≈2.3（T3 线性）–8.6（T3' 线性）eV/Å ≫ 0.02。**生产 γ-hold 只允许靶点≈自然值**；
  T-5' 的 ±0.013 rad 仅是 SCF 可达域。EFC（E_phys=E_tot+λ(G−t)）是大 λ 场景
  恢复力精度的唯一路径。
- **P 系列**：`2026-07-30-pseries-test-cases-design.md` §3.3 加 LCAO 极化率
  基组重定标警示（α_LCAO=3.02 vs 实验 9.8，3.3×；P01/P05 判据须用同基组参考）。

### File list
- `docs/superpowers/specs/2026-08-12-deltap-ow-vh3-t18.md`（§1.5 口径表 + §1/§5 重标）
- `docs/superpowers/specs/2026-08-05-deltap-algorithm-derivation-route-a-plus.md`
  （§8 LIMITATION #1：双口径窗口 + 口径表）
- `source/source_esolver/deltap_scf.cpp`（E_eff 打印注释，~15 行）
- `docs/deltap_user_manual.md`（§8 E_eff 公式/校准 + LCAO α 警示）
- `docs/superpowers/specs/2026-07-30-pseries-test-cases-design.md`（§3.3 警示）
- `docs/superpowers/specs/2026-08-13-deltap-d1-d2-kappa-field.md`（§5.4 窗口修正）
- `docs/superpowers/specs/2026-08-12-deltap-physical-stiffness-hypothesis.md`
  （§4.1/§6 窗口与 E_eff 状态）

### Next steps
1. 本轮（D1/D2 + E_eff + 评审修正）统一提交，commit 消息含"物理刚度假说执行
   完毕：α_LCAO=3.02、κ=60.4 Ry/rad²、E_eff 换 (b)"。
2. T-9'（branch 写守卫）随下个 commit；然后 T-7'（per-atom Jacobian，
   顺带实测 γ-hold 的 dλ*/dR 验证泄漏模型）。
3. EFC（路径泄漏消除）若立项：以本窗为基线做"大 λ 恢复力精度"判决实验。

## 2026-08-13——T-7' per-atom Jacobian：单元层 PASS，集成层 FAIL（报告 γ 锚定量化钉死内循环）

### Round summary
实现 componentwise secant（对角 Jacobian）内循环更新 + INPUT
`deltap_inner_scheme=cg|jacobi` 开关；ow a2 集成对照（cg/jacobi 两臂）。
单元层 8/8 PASS；集成层两臂内循环均不收敛——jacobi 消除 α 翻号但驱动
可观测量（连续性锚定的报告 γ）被钉在自然值附近，残差退化为常数。
顺带修复：空字符串 INPUT 值（如 `deltap_target_file  `）导致的解析期
段错误（read_sync_string 空 str_values UB）。

### Key results
- **单元层**：反号对角系统 r₁=2λ₁−1、r₂=−3λ₂+1，jacobi 1 步收敛 vs
  scalar-α 60 步不收敛（cw_ok=1 steps=1 vs sc_ok=0 steps=60）。bfgs 8/8。
- **集成层（ow γ-drive a2，O1 靶=自然+0.02 rad，inner_nmax=20）**：
  cg 复现 T-18 a2 失败签名（α 翻号 −2.35e-2→+1.17e-2→…，rms 1.13↔1.55e-2
  平台，|γ−t|=1.80e-2）；jacobi α 稳定 +0.25（max_step 钳位）**不翻号**，
  但 rms 恒 ~1.13e-2、λ_O1 爬至 +9.7e-3、**|γ−t|=2.14e-2≈靶点全量（零进展）**。
- **机制（[SBdbg] O1 轨迹）**：内循环 trial 中 raw 相位对 λ 有响应
  （−7.958…−7.977 振荡，Δraw 达 ±0.019 rad），但报告 γ 被连续性锚
  （ref_gamma_=branch.dat 自然值）钉死 −7.958±0.001——branch shift 逐
  trial 吸收 raw 移动（0→4.9e-3→1.26e-2→1.89e-2→…）。残差 r=γ_report−t
  恒 ≈−0.020±1e-3 → 任何光滑优化器不可收敛。这是 T4a 教训（锚定 γ 报告
  不可作收敛判据）在内循环的新形态，也是 T3' 评审"γ(λ) 非光滑不可逆"
  预言在冻密度层的定量确认。
- **T-7' 判定**：优化器基建 ✅（两套驱动通用）；ow γ-drive 内循环 ❌
  （可观测量解锚前不可收敛）。按"偏离即停"落档，不扩大使命。
- **空值 INPUT 守卫**：`read_sync_string` 空 str_values 时保留默认值
  （原为 UB/段错误）。验证 `deltap_target_file  ` 不再崩。

### File list
- `source/module_optimizer/bfgs.h`（componentwise 模式：per-component
  secant/max_step/自适应 γ，~100 行）
- `source/module_optimizer/test/bfgs_test.cpp`（+2 测试）
- `source/source_esolver/deltap_scf.{h,cpp}`（DeltapParams::inner_scheme + 接线）
- `source/source_esolver/esolver_ks_lcao.cpp`（p.inner_scheme 填充）
- `source/source_io/module_parameter/input_parameter.h`、`read_input_item_other.cpp`
  （`deltap_inner_scheme` INPUT）
- `source/source_io/module_parameter/read_input_tool.h`（空值守卫）
- `docs/superpowers/specs/2026-08-13-deltap-t7p-per-atom-jacobi.md`（本轮轮文档）

### Next steps
1. commit T-7'（代码 + 守卫 + 轮文档）。
2. **T-7'' 立项（机制级）**：ow γ-drive 内循环驱动可观测量改 raw γ
   （compute_gamma_raw 已接线）或冻结 branch-shift 读数；落地后重跑
   ow a2 两臂验证内循环收敛。T-18 0.98 判决仍受 |λ*|_max≈0.4–1.6e-3 Ry
   工作窗限制。
3. T-8'（L1 三件）可穿插。

## 2026-08-13——T-7'' ow γ-drive 内循环解锚：机制 PASS，收敛 FAIL（冻密度响应为负 + 强交叉耦合）

### Round summary
按评审重设验收（0.005 rad 靶点 + 检查点 1/2）实现 T-7''：内循环入口冻结
Stage-B branch shift（`freeze_branch_shift()`，不动 ref_gamma_/θ），报告从
锚钉切换为 raw 精确跟随。解锚机制验证通过；两臂内循环均不收敛，根因为
物理层——冻密度 γ(λ) 响应为负且交叉耦合强。

### Key results
- **检查点 1（raw↔shift 记账）✅**：入口 shift0 = report0 − raw0 = 0；
  全部 trial mode=frozen、best=raw（shift=0）；残差 = 真实靶点差
  （rms 初值 2.9024e-3 = 0.005/√3 精确）；退出报告 = raw + frozen_shift
  一致无跳变。
- **检查点 2（θ/Π 冻结）✅**：两臂 0 次 D2 jump；trial 只调 compute_gamma，
  不扰 compute_gamma_op。
- **收敛 ❌**：cg 翻号 rms 1.8–5.4e-3 振荡；jacobi 2-循环极限环
  3.7↔6.4e-3。SCF 级：cg |γ−t|=2.69e-3（半达成，E=−481.72649）、
  jacobi 5.93e-3（未达成，E=−481.69087）；ow_nat 自然 E=−481.69777。
- **决定性测量**：trial 0（仅 O1 λ≠0）两臂一致 dγ_O/dλ_O(frozen) =
  −1.28 rad/Ry（负）；非对角耦合≈对角（trial 0 后 H1 残差 −0.0018≈O1）。
  SCF 级本几何 −0.62 vs T-18 几何 +0.059——符号跨几何不稳定。
- **裁定**：T-6' 开放问题 → 算符控制权限问题（非测量通道）。γ-hold 冻
  密度内循环不可行；下一步 a（自洽驱动，iter_finish 每密度重驱动）或
  b（生产维持 proxy 驱动 + LIMITATION）。

### File list
- `source/source_lcao/module_deltap/deltap.h`、`deltap_wannier.cpp`
  （`freeze_branch_shift()`：仅冻 branch_shift_，~10 行）
- `source/source_esolver/deltap_scf.h`（Backend 钩子）、`deltap_scf.cpp`
  （内循环入口冻结 + shift0 记账打印，门控 operator+gamma）、
  `esolver_ks_lcao.cpp`（钩子接线）
- `docs/superpowers/specs/2026-08-13-deltap-t7pp-deanchor.md`（本轮轮文档）
- dev log Key Conclusions 条目 14 升级为"锚定三形态表 + 冻结纪律"，新增条目 15

### Next steps
1. commit T-7''（解锚 + 轮文档 + dev log）。
2. γ-hold 自洽驱动（iter_finish λ 更新，用 SCF 级残差）立项评估；验收
   T-18 同款 γ-hold FD < 0.02 eV/Å @ 小靶点。
3. 生产路径维持 proxy 驱动；LIMITATION 补 γ-drive 不可达域（T-7'/T-7''）。

## 2026-08-13——F-1 均匀 λ 冒烟：E'(λ) 平直性复跑（场模式基线，零代码）

### Round summary
场模式 P0 三件套第一件（纯测量）：h2o1 全原子 λ_I≡λ 均匀
（`deltap_lambda_init <λ>` 标量冻结），λ ∈ ±0.01/±0.003/±0.001/0 扫描，
proxy 模式复跑 T2 平直性判据。F-1 PASS，为 F-2（锯齿场对拍）建立能量基线。

### Key results
- E'(λ) 端点弦斜率 **−0.0127 eV/Ry**（判据 ≲1 的 1/80；T2 修复后 −0.013
  逐点复现）；分解线性 a=−0.0127 eV/Ry + 曲率 b=+14.3 eV/Ry²（escon 通道
  主导，D1 的 15–30 同量级）。
- 变分下界恢复：λ=0 严格极小，E'(±0.01)−E'(0)=+1.55/+1.30 meV（T2 逐位
  一致）；±0.001 八位对称（−481.6977557 = −481.6977557；13 位差 5.7e-8 eV）。
- 记账恒等式逐点精确：escon=−ΣλΓ；dΣΓ/dλ=−4.15 Ry/Ry（T2 的 −4.2 ✓）；
  dΣγ_raw/dλ=+0.053 rad/Ry（D1 proxy 自洽 +0.056 ✓）。
- E_eff 打印（formula (b)）线性：±2.693e-2 V/Å @ |λ|=0.01 ✓。
- 锚定约定：无 branch.dat 种子自由跑自然 γ=(−7.958,−2.380,−2.380)，与
  base_clean（−5.515,−3.601,−3.601）差分支量子约定，E' 12 位一致
  （7e-13 eV），不影响 F-1 判据。

### File list
- `docs/superpowers/specs/2026-08-13-deltap-f1-uniform-lambda.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（F-1 状态 → ✅）
- dev log 本条

### Next steps
1. commit F-1（纯文档，轮文档 + dev log + TODO 状态）。
2. **F-2（判决点）**：h2o1 DeltaP 均匀 λ vs 锯齿场 efield 双通道 E₀(μ) 对拍；
   DeltaP 通道改用 base_clean 分支锚（与 efield 通道同锚），判据曲线偏差
   <1e-3 eV、力偏差 <1–2%。
3. F-2 通过 → 场模式进用户文档主推；不通过 → 残差分解闭合后评审。

## 2026-08-13——F-2（场模式判决点）对拍：严格判据不通过，残差 100% 闭合（零代码）

### Round summary
h2o1 双通道 E₀(μ) 对拍：DeltaP 均匀 λ（F-1 批）vs 锯齿场 efield（D2 大场点 +
本轮 ±0.0003/±0.0005/±0.0007 小场点 + amp=0 基线 + no-dip_cor 力对照，11+3 点
串行）。§5.3 P10 协议（DeltaP 直接 E'(λ)=E₀(μ)；efield Legendre 变换；matched-μ）。

### Key results
- **能量 E₀(μ)**：重叠窗 max|dev|=1.55 meV（@|λ|=0.01）；|λ|≤0.007 Ry 内 <1e-3 eV ✓。
  dev 全部来自 DeltaP 侧 escon/Γ 曲率（b=14.3 eV/Ry²，D1 已知）；efield 侧
  Legendre 变换后平坦到 ±0.003 meV（μ(E) 与 F(E) 自洽，零污染）。
- **力（matched-μ）**：0.00%@λ=0 → 2.6%@|λ|=0.001 → 21%@|λ|=0.01。根因 =
  约束反作用项（∂E₀/∂μ=−λ·杠杆≠0 → F_DP 含 F_PES 之外的项）。
  **力-隐含杠杆 1.31 a.u./Ry vs μ-杠杆 0.084 a.u./Ry：15.6× 不一致** —— 算符
  力/极化杠杆不匹配，量化解释了 D2 的 1.6–3.1× 校准为何不能同时用于能量与力。
- **设置层发现**：dip_cor 开起常数 +3.70 meV（=½·4πμ₀²/Ω 镜像自能，μ₀=0.76316
  精确吻合）；efield 基线必须是 efield-on amp=0（不能用 base_clean）。力对比用
  no-dip_cor 系列（与 DeltaP 同静电口径，E=0 锚点 0.003% 一致）。
- ΣF=0：DeltaP 全 λ 精确零 ✓；efield ~6e-5（已知同族）。

### 裁定
F-2 严格判据不通过 → 按用户规定"残差分解闭合后回评审"。场模式定位：能量/响应
用途在 |λ|≤0.007 Ry 窗口内可用（<1e-3 eV）；力侧 relax 不可用（约束反作用项，
对齐需 EFC——P3 Ô_θ/EFC 立项动机补 F-2 力杠杆 15.6× 实测）。

### File list
- `docs/superpowers/specs/2026-08-13-deltap-f2-field-comparison.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（F-2 → 已执行/判决）
- dev log 本条

### Next steps
1. commit F-2（轮文档 + dev log + TODO）。
2. 评审点：F-2 判决的窗口表述（能量 |λ|≤0.007 可用、力侧 EFC 依赖）是否接受；
   EFC 优先级是否上调（P3 动机已实测）。
3. 若评审通过：F-3（ow 处置）→ F-5（LIMITATION 补 F-2 窗口/杠杆表）→ F-4/F-6 按序。

## 2026-08-13——F-2b（HK-only 场模式对拍）：可证伪预言 FAIL，根因从 H_HR 重定位到 H_HK 力通道

### Round summary
评审"EFC 之前先做 HK-only 场模式"：新 INPUT 值 `deltap_operator_mode hk`
（H_HR off + A1/A2 力 off + escon=−λΓ^HK，5 文件门控），h2o1 λ 均匀冻结
扫描 7 点 vs efield no-dip 系列。目标：分离"链接算符力通道"与"H_HR 代理力
通道"对 F-2 力失配的贡献。串行 7 点全 rc=0。

### Key results
- **可证伪预言 FAIL**：matched-μ 力偏差 @|λ|=0.01 仍 −26.1/+26.2%（hk），
  未塌缩到百分之几（同批 proxy −26.2/+26.7；F-2 文档 −35.6/+21.0 为
  dev/F_DP 口径）。杠杆比 L_F/L_μ = 17.7×（hk）/15.7×（proxy）。
- **H_HR 对力失配可忽略（F-2 根因证伪）**：hk vs proxy 总力差 ≤0.6%
  （@+0.01 为 6.1 meV/Å，偏差 200 meV/Å 的 3%）——"τ·P̂ 打在原始布居
  （7.2/2.14）→ 15.6×"不成立。
- **失配内禀于 H_HK ∂Γ^HK/∂R 力通道**：B 项力杠杆 −30.1 eV/Å/Ry
  （∂Γ^HK(O1)/∂R_z≈+1.17 rad/Bohr），F_std 密度响应 +9.4 抵消 ~31%。
- **E'(λ) 曲率未随 H_HR 移除缩小**：b(hk)=14.66 ≈ b(proxy)=14.26 eV/Ry²
  （max|res| 34 μeV）；hk 与 proxy E'(λ) 差 ≤34 μeV——能量侧窗口结论不变
  （|λ|≤0.007 Ry <1e-3 eV）。
- **γ 响应 hk(0.046) < proxy(0.053) rad/Ry**（批内 Σγ 口径）——τ·P̂ 对 γ 是
  加性正贡献，移除后略降；与 T-18/D2 冻结口径的差异是协议差（冻结 vs
  自洽 5–10×，同族已知）。
- 记账自洽：escon=−λΣΓ^HK 逐点精确、E_HK=λΣΓ、λ=0 与 base_clean 逐位
  一致、ΣF=0 全点。

### 裁定
F-2b FAIL（判决性，闭合）：场模式力侧不可交付，需**力级 EFC**（目标
L_F=L_μ；修正量 λ·(L_μ·Z*−L_F)）；能量侧可交付（窗口不变）。EFC 立项
保留，目标函数从"荷错位修正"更新为"∂Γ/∂R 力杠杆修正"；能量侧 EFC 动机
不再成立。开放项：b 与朴素 −½Γ₁ 预测差因子 2（29.3 vs 14.7 eV/Ry²），
escon/总能抵消的精确二次结构待查（不影响判决）。

### File list
- `source/source_io/module_parameter/read_input_item_other.cpp`（hk 校验+注解+experimental）
- `source/source_io/module_parameter/input_parameter.h`（hk 注释）
- `source/source_lcao/module_operator_lcao/deltap_lcao.cpp`（contributeHR 门控 ow||hk）
- `source/source_lcao/module_operator_lcao/deltap_force_stress.hpp`（A1/A2 门控 ow||hk + 应力 WARNING_QUIT + 虚假注释修正）
- `source/source_lcao/module_deltap/deltap.h`（operator 模式返回 Γ^HK 单独）
- `docs/superpowers/specs/2026-08-13-deltap-f2b-hk-field-mode.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-05-deltap-algorithm-derivation-route-a-plus.md`（§8 LIMITATION：场模式双行 + α_LCAO 声明 + E_eff 校准）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（F-2b 判决 + F-3 完成标记）
- dev log 本条 + 关键结论区 #16

### Next steps
1. commit（F-2b + F-3 + F-5 一并）。
2. 力级 EFC 设计（目标函数：L_F=L_μ；输入：hk 批杠杆表 + efield Z* 通道）。
3. F-4（T-7'' 内循环解锚，约束模式最后机制件）→ F-6/F-7/F-8 按序。

## 2026-08-13——F-2b 评审裁定落地（文档轮）：双命题并存定稿 + F-4 封口 + F-8 Maxwell 判据

### Round summary
评审接受 F-2b 负结果（并认领其"H_HR 荷错位"根因被证伪），裁定：
① T3/F-2b 双命题并存（记账自洽 vs 场物理失配）须写明防误读，物理本质 =
"H_HK 能量 R 依赖走极化+基组几何双通道，有力无极化"；② EFC 定位为场模式
力/应力的唯一第一性解法（Ô_θ 的 R 导数纯极化通道，Maxwell 白来）；
③ L_F=L_μ 公式降级为半经验补丁（empirical, not ab-initio；Z* 需 2–3 几何
可迁移验证）；④ F-4 不做（三理由：冻密度 γ 响应符号反转死亡 / γ-hold 力
泄漏 0.5 eV/Å / 场模式无泄漏有消费者），约束模式封口；⑤ F-8 验收加
"应力-极化 Maxwell 一致性"判据（∂σ/∂λ vs ∂P/∂ε）。

### Key results
- F-2b 文档新增 §5.6 双命题并存 + §6 EFC 定位/半经验标注。
- TODO：F-4 → ❌ 封口（三理由）；F-8 验收加 Maxwell 判据；序 = F-6 → F-7 → F-8。
- 08-05 主算法文档 §8 LIMITATION：力侧条目补双通道物理 + 半经验补丁纪律 +
  约束模式封口声明。
- 注：评审引用的物理 ∂γ/∂R≈0.9 rad/Bohr 为估计——T3 时代 disp± 目录
  （tests/deltap_fd_force/h2o1/）的 λ 设置与现协议不同（γ 靶点非自然值），
  不可直接提取；量化留给 F-8 Maxwell 哨兵 / 专门位移实测。

### 裁定
纯文档轮（零代码），已按评审四条裁定全部落地。场模式能量/响应侧具备写
用户文档条件（力侧挂起 + 半经验 Z* 补丁可选标注）。

### File list
- `docs/superpowers/specs/2026-08-13-deltap-f2b-hk-field-mode.md`（§5.6 + §6 更新）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（F-4 封口 + F-8 判据）
- `docs/superpowers/specs/2026-08-05-deltap-algorithm-derivation-route-a-plus.md`（§8 LIMITATION）
- dev log 本条 + 关键结论区 #17

### Next steps
1. commit（文档轮）。
2. F-6（hk MPI）→ F-7（L1 三件）→ F-8（应力，含 Maxwell 判据）按序。
3. 场模式用户文档（能量/响应侧可交付；力侧挂起 + 半经验标注）可与 F-6 并行起草。

## 2026-08-13——F-6（TODO 3.1）：hk_correction MPI 修复 PASS

### Round summary
TODO 3.1（F-6）：`compute_hk_correction` 的 MPI 语义修复。根因两层——
① psi 带索引按全局 nocc_use 遍历本地带（部分和/越读）；② H_sym 的
（α,β）两个下标都用本地行轨道，而 H 本地块 (nrow×ncol) 的列应是本地列轨道
（nrow==ncol 守卫只保证尺寸巧合相等，不保证索引语义）。2×2 网格下 nwfc
偶数（hf=18/co=10）跑通但 Γ/E' 错（hf E' 差 2.33 eV）；nwfc 奇数
（h2o1/h2o_asym=529）非对角 rank 265×264 触发 WARNING_QUIT（gdb catch
exit_group 实证）。附带发现 `deltap_observable` 默认 operator——hf/co 走
Γ 路径，3.1 验收必须连带修 `compute_gamma_op_hk`（同族 A' 方案直接适用）。

### Key results
- **修复方案**：nproc==1 原循环逐字节保留（硬约束）；nproc>1 走 pzgemm
  分布式 GEMM——`SC = S_dk·C_R`（desc×desc_wfc→desc_wfc）、
  `F=(i/2)·w_eff[g]·SC`、`H_sym = 0.5·(F·C_L† + C_L·F†)`（'T'+预共轭约定，
  同 cal_dm_psi，精确 Hermitian 本地块）；Γ^HK 的 T/Π 走本地行×本地带部分和
  + 均匀计数 Allreduce（A' 族）。
- **串行 A/B 硬约束 PASS**：h2o1 FINAL_ETOT_IS −481.6964709588435 eV 逐位
  一致；全部 DeltaP 诊断行/GE 能量列(md5)/TOTAL-FORCE 逐位一致（仅 TIME 变）。
- **hf/co corr=1 4-rank PASS**：hf E' Δ=2.7e-6 eV、co Δ=2.5e-5 eV（串行
  参考 −687.094383147832 / −612.0598717620463）；rawG 逐原子 γ 全精度一致；
  P3 行（γ/λ/Γ/escon 打印精度）一致。co 的 λ[0] 第七位漂移（4.644467 vs
  4.644466）直接解释 escon Δ。
- **h2o_asym 4-rank 仍被方阵守卫拦**：rc=1，gdb 确认 WARNING_QUIT
  （compute_hk_correction nrow≠ncol 守卫）✓。
- **deltap_mpi_smoke 4/4 PASS**：PW 2-rank / BN 4-rank / CO(NBANDS=15) 4-rank
  / BN inner-loop 4-rank。
- **ΔE' 归因（诚实标注）**：pzgemm 块循环求和顺序 vs 串行稠密循环的固有 FP
  差异（~1e-14/元素），经 30+ 迭代 SCF 反馈与 λ 累积放大——远低于一切判据
  （fd_force 0.0129 eV/Å、T 系列 1e-3 eV），非可消除偏差。
- **范围裁定**：h2o1/h2o_asym（529 轨道奇数）4-rank 仍被拦（3.2/3.3 前置：
  非方本地块 + compute_hk_force MPI）；F_HK 力保持既有 MPI 跳过（LIMITATION）。
  TODO 3.1 原文验收（hf/co γ+E' + h2o_asym 拦）全部满足。

### 裁定
3.1 PASS。H_sym 本地块必须走分布式 GEMM（A' 带映射对观测充分、对本地块不
充分——行/列块语义是硬需求）；Γ^HK 观测走 A' Allreduce。ow 分支 MPI 门控
跳过（TODO 3.2/3.3）。

### File list
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（compute_hk_correction
  + compute_gamma_op_hk 的 MPI 路径；守卫消息更新；ow MPI 门控）
- `docs/superpowers/specs/2026-08-13-deltap-f6-hk-mpi.md`（本轮轮文档）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（3.1 → ✅）
- dev log 本条 + 关键结论区 #18

### Next steps
1. commit（F-6 全部文件，消息按 TODO 纪律）。
2. 3.2（operator 4-rank Γ/γ 跨 rank 一致性专项）→ 3.3（非方本地块 + F_HK
   力 MPI 后 co/h2o_asym 4-rank 收敛复测）。
3. F-7（L1 三件）→ F-8（应力，含压电 Maxwell 判据）。

## 2026-08-13——Q1/Q2 + TODO 3.2/3.3：F_HK 力 MPI + 非方本地块解锁 + F-6 带对计数修复

### Round summary
评审裁定执行：Q1（nrow≠ncol 守卫收窄到 nproc==1，非方本地块 MPI 合法）+
Q2（compute_hk_force 的 F_HK 力 MPI 化，场模式生产化前置）。过程中抓到并修掉
一个 F-6 遗留的隐藏 bug：`gather_band_columns` 的 `MPI_Allgatherv` 用
MPI_DOUBLE 搬运 complex 时计数没 ×2，每 rank 只发一半数据。3.2（跨 rank
Γ/γ 一致性）与 3.3（co/h2o_asym 4-rank 收敛复测）随同验收 PASS。

### Key results
- **Q1 守卫收窄**：`compute_hk_correction` / `compute_gamma_op_hk` /
  `compute_hk_force` 的 `nrow != ncol` 检查只对 `nproc==1` 生效；h2o_asym
  4-rank（12×11 非方本地块）rc=0 解锁。
- **Q2 F_HK 力 MPI**：SC 走 pzgemm；T/Pi/U 走行组 gather + 全带对循环 +
  单次 Allreduce（coord[1]!=0 清空）；dW 走 A' 全 rank 归约；归约后与串行
  相同力度 kernel 累加。串行块逐字节保留。
- **隐藏 bug（F-6 带对计数的实际形式）**：Allgatherv 计数应为
  `2·nrow·ncol_b`（complex = 2 double）。症状：h2o_asym 4-rank Γ^HK 6.671→
  5.180（~22%，奇数占据带整列归零）；h2o1 恰好不受影响（占据带全在进程列 0
  干净区）。修复后 h2o_asym F_HK 与串行逐分量一致。
- **串行 A/B 硬约束 PASS**：h2o1 5271 行中仅墙钟 + 单条 profile 计时 4 行差。
- **3.2 PASS**：h2o_asym 4-rank / 2-rank（1×2 网格）vs 串行——λ、
  Γ=(6.660,1.987,1.857)、escon=−0.051842、E_HK=0.0237007424、F_HK 9 分量
  逐位一致；h2o1 4-rank 大分量逐位、湮没分量 ~1e-12 FP。
- **3.3 PASS**：co（NBANDS=15 奇数）4-rank 收敛，λ=(4.644466,6.623731)e-3、
  escon=−0.065207、E_HK=0.0289416363、F_HK 与串行一致。
- **deltap_mpi_smoke 4/4 PASS**：PW 2-rank / BN 4-rank / CO 4-rank /
  BN inner-loop 4-rank。

### 裁定
Q1+Q2+3.2+3.3 PASS。F_HK 力从"L2 遗留"升级为场模式多 rank 可用。遗留：
Ô_w（H_ow/Γ^w）MPI 门控跳过（L3.1 立项时做）。

### File list
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（gather_band_columns
  辅助 + ×2 计数；三函数行组 gather 全带对 T/Pi/U；守卫收窄）
- `source/source_lcao/module_deltap/deltap.h`（compute_hk_force 注释）
- `docs/superpowers/specs/2026-08-13-deltap-q1-q2-hk-force-mpi.md`（本轮）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（3.2/3.3 → ✅）
- dev log 本条 + 关键结论区 #19

### Next steps
1. commit（消息含"F_HK 力 MPI + 非方本地块解锁 + F-6 带对计数修复"）。
2. F-7（L1 三件：PW Γ 记账 / ⟨η⟩ / spread_I）→ F-8（应力，含应力-极化
   Maxwell 判据 ∂σ/∂λ ↔ ∂P/∂ε）。
3. Ô_w MPI（H_ow 本地块 + Γ^w 行组 gather）待 L3.1 立项。

---

## 2026-08-13: 适用范围与能力边界权威文档

### What was done
新建 `2026-08-13-deltap-capability-boundaries.md`（取代主算法文档 §8，后者已加
重定向）：一句话定位（小窗口精密约束 + 场模式能量/响应；relax/MD、大靶点、
LCAO 多 rank 力路径不可用）；三模式能力矩阵（约束/场/实验性）；定量边界
（Γ 路径 0.0138、γ-hold 0.502、可达域 O[−0.004,+0.013]/H 双侧死、场能量窗
|λ|≤0.007、E_eff 校准 ×1.6、κ≈60、FD 高精度处方）；LIMITATION 全表 L1-L10；
失效模式诊断表 F1-F8；物理适用域与生态位（bulk 周期场 = EFC 前置）。
无代码改动。

## 2026-08-13 (2): 能力边界文档同步（3.2/3.3 落地后）

L2（方阵网格守卫）、L3（F_HK 力串行-only）标记已解除（656a2a59d：非方本地块
pzgemm 解锁 + F_HK 力行组 gather Allreduce）；F2 诊断行改为"复现=回归"；
能力矩阵 LCAO 多 rank 行刷新。注：3.2/3.3 顺带修掉 Allgatherv complex 计数
×2 bug（h2o_asym Γ 22% 偏差根因）。

---

## 2026-08-17: F-7（L1 三件）— PW Γ 记账 + ⟨η⟩ + spread_I

### What was done
- **L1.1 PW Γ 记账**（RouteA++ §5 最小补丁）：`deltap_pw.cpp` 新增
  `compute_gamma_op_pw`（Γ^PW_I = ⟨P̂_I^onsite⟩ = Σ_k Σ_ib wg(k,ib)·Σ_{ih∈I}|becp|²，
  becp 布局 ib·npol·nkb+ispin·nkb+ih，pool-reduce 后 rank 一致）；
  backend 接 `compute_gamma_op`；`deltap_init` 传 `pelec->wg`；
  `observable_mode="operator"` + `drive="gamma"`（PW λ 驱动保持 γ 残差不变，
  只换 escon 记账，与 LCAO γ-drive 同构）。`report_pw` 行尾加 Γ/atom 诊断。
- **L1.2 ⟨η⟩ 输出**：`deltap_wannier.cpp` p_hat_accum 循环（per-k per-band raw
  D_I 投影）累加 η_n(k)=max(0,1−Σ_I w_In)，INPUT gdir 单次计数；
  SCF 收敛时 esolver 打印 `[DeltaP L1] <eta>=... (max ...)`。
- **L1.3 spread_I 输出**：per-string w_In_matrix（Löwdin tilde）+ gamma_unwrapped
  加权 std-dev；公式提取为 `deltap_common::compute_theta_spread` /
  `compute_smo_leakage`（可单测）。新增 `deltap_l1_test.cpp` 单测。
- 文件：deltap_pw.{h,cpp}、esolver_ks_pw.cpp、deltap_wannier.cpp、deltap.h、
  deltap_common.h、esolver_ks_lcao.cpp、test/CMakeLists.txt + deltap_l1_test.cpp。

### Results
- L1.1：escon≡−λΓ 逐点恒等（7/7）；E'(λ) slope −0.187 eV/Ry（ecut=20）、
  −0.178（ecut=40）；旧 γ 记账 ~109 eV/Ry → 改善 ~600×，≪F-1 判据 1 eV/Ry。
- L1.2：H2O ⟨η⟩=0.000（<1% ✓，SMO 完备）；CO 1.67%（单带 max 8.4%，机制非恒零）。
- L1.3：H2O spread_I=(0.112,0.121,0.121)；CO=(0.257,0.197)；单测 4/4
  （均匀 θ spread=0、加权 std-dev 核对、退化零权重、η clamp）。
- 回归：MPI smoke 4/4（PW 2-rank + LCAO 4-rank×3）；h2o 重构前后逐位一致。

### Next steps
- F-8（L2 应力）开工前写 Maxwell 判据协议（应变 FD + matched-μ + ∂σ/∂λ vs
  ∂P/∂ε 双口径判据）。
- Stage 4：4.1 锚点重建在 F-7/F-8 之后（PW Γ 锚点用例纳入套件）。
- 能力边界文档 L5 已更新（PW 记账统一；target 文件/约束矩阵接线未立项）。

## 2026-08-17 (2): F-7b 立项（评审裁定：先复测后开发）

用户指出 ecutwfc=20 验收不足（LCAO 100 / PW ≥80）。TODO 新增 F-7b：L1.1 生产档
复测（ecutwfc=80/ecutrho=320/scf_thr=1e-8 七点恒等 + E'(λ) 平直性）+ ⟨H_c⟩/λ
vs Γ^PW 逐点对比归因（(a) 噪声销案 / (b) 结构性记 LIMITATION）。~0.5 天机时。
F-8 顺延至 F-7b 与 Maxwell 协议评审后开工。

## 2026-08-17 (3): F-7 评审交接文档

`2026-08-17-deltap-f7-review-handoff.md`：F-7 评审定稿（L1.1 有条件通过→F-7b；
L1.2/L1.3 通过；CO ⟨η⟩ 关联检查入 4.2 验收项）+ F-7b/F-8 完整任务规格 +
Stage 4 衔接（4.1 含 PW Γ 锚点）。执行顺序：F-7b ∥ F-8 协议 → F-8 → Stage 4。

## 2026-08-17 (4): F-7b 生产档复测 + 归因（结构性失配假说被证伪）

`2026-08-17-deltap-f7b-l1-retest.md`（五段式，含原始数据表）。按 F-7 评审
handoff 的 F-7b 规格执行（真 ecut=80 七点恒等 + E'(λ) 平直性 + 判决性归因）。

### 发现 0（过程层）：原"F-7b ecut=80"实为 ecut=40
scan.py 的 `ecutwfc` 替换嵌套在 `if re.search(deltap_lambda_init)` 分支内，
而 base INPUT 无该行 → 替换永不执行（npwx=97373=ecut=40 档，真 ecut=80
npwx=275340）。原 1.784× 数据全部是 ecut=40 + inner_thr=1e-3 的产物。
扫描脚本已修正（run_scan3/4.py）。

### 测量 1（真 ecut=80 + inner_thr=1e-6，七点）
- escon≡−λΣΓ 恒等 7/7（代数恒等）；ΣΓ 漂移 dΓ/dλ ≈ −2.0 Ry/Ry。
- **E'(λ) 线性项（中心差）：±0.001→+0.00014、±0.003→+0.00007、
  ±0.01→+0.00018 Ry/Ry（全部 ≤0.0002 Ry/Ry ≈ 0.0025 eV/Ry）——平坦。**

### 测量 2（1.784× 归因：判据选错，非算符失配）
dE_band/dλ = +14.17 Ry/Ry vs ΣΓ = 7.94：差 6.23 = 密度响应。能量分量闭合
（ecut=40，λ=±0.001）：E_band +193.00 + deband −158.06 + E_H +89.37 +
E_xc −16.06 + dp_escon −108.06 = **+0.19 ≈ 实测 +0.179 eV/Ry ✓**。
E_band 对 λ 非变分（HF 只适用于总能量），1.784× 是判据选错的伪影。

### 测量 3（stale-escon 机制判决）
| 配置 | escon 测量态 | 线性项 |
|------|--------------|--------|
| ecut=40, inner_thr=1e-3（原数据） | drho≈7.4e-4 | +0.0131 Ry/Ry |
| ecut=80, inner_thr=1e-3 | drho≈5.7e-4 | −0.0214 Ry/Ry |
| ecut=80, inner_thr=1e-6 | drho≈3.7e-7 | +0.00014 Ry/Ry |

⇒ 线性项 = escon/Γ 测量态伪影：PW 同步两相模式每周期只测一次 Γ（首个
drho<inner_thr 迭代，默认 1e-3 时 drho≈7e-4 未收敛态），该陈旧 Γ 进入
FINAL_ETOT 的 escon，而本征值用收敛 ψ。符号随 ecut 反转（+0.16%@40、
−0.27%@80）进一步排除算符失配。

### 结论与动作
- **L1.1 条件解除，F-7 完全 PASS**：真生产档 E'(λ) 线性项 0.0025 eV/Ry
  ≪ F-1 判据 1 eV/Ry（400×），优于 ecut=40 旧数据 ~100×。
- 能力边界：L5 行更新（escon 恒等精确 + F-7b 归因）；新增 L11（escon
  一次性测量伪影，escon 精度判据须 inner_thr≤1e-6）。
- F-8 协议文档补 §7 评审精化（A/B 侧 H_c 构成一致性、∂E'/∂λ 口径注释）。
- 可选代码级修复（收敛末迭代刷新 escon）留 Stage 4，本轮不扩使命。

## 2026-08-17 (4): 开发进展总结文档

`2026-08-17-deltap-progress-summary.md`：全项目进展定稿——Route A+ 算法核心
（记账恒等式/力分解/双路径分野/锚定纪律）、六个关键技术点子标题（弱耦合窗口、
场模式、MPI 修复链、PW 记账与伪影、内循环/算符轨道否定结论、诊断输出）、
十二个测试点子标题（T0/T2/T3/T3'/T-5'/D1/D2/T-18/T-7'/F-1/F-2/F-2b/MPI 矩阵/
F-7/F-7b/锚点套件/FD 协议纪律）、当前状态与 F-8→Stage 4→Stage 5 路线、文档索引。

## 2026-08-17 (5): F-7b 闭环 — Q1 escon 代码修复 + Q2 scan.py 审计 + F-8 协议签核

`2026-08-17-deltap-f7b-escon-refresh.md`：
- **Q1（修代码，非留文档）**：PW escon 测量时机从"首个 drho<inner_thr 迭代
  一次性测量"改为"每 SCF iter_finish 在当前 ψ 上刷新"——新增
  `pw_deltap::refresh_pw_escon`（deltap_pw.h/.cpp，复用 compute_gamma_op_pw +
  compute_dp_escon）、`DeltapScfSolver::set_escon`（deltap_scf.h）、
  `esolver_ks_pw.cpp::iter_finish` 开头调用并同步 f_en.dp_escon。
  验证：ecut=80 + 默认 inner_thr=1e-3 下 E'(λ) 中心差 **−0.0214 → −2.5e-5
  Ry/Ry**（改善 ~850×）；MPI smoke 4/4。L11 降级为历史注记。
- **Q2**：scan.py 嵌套 if bug 影响面 = 仅 F-7b 复测（/tmp 全盘 find 唯一
  scan.py 在 /tmp/deltap_l1_1_pw/）；F-1/F-2/F-2b/F-6 不经该脚本，无重跑。
- **F-8 协议签核修正已并入**：§5 PASS 行只证明内部自洽 + 记账一致（对
  17.7× 零信息量）；efield 压电对照升级必测项。

## 2026-08-17 (5): 进展总结文档重写（物理导向版）

`2026-08-17-deltap-progress-summary.md` 按评审意见重写：从工程日志风格转为
物理能力导向——§1 DeltaP 是什么（约束/场模式物理图像）、§2 已能做什么
（SCF 约束/小窗口力/场模式能量响应/BN 零刚度发现）、§3 定量边界的物理归因、
§4 在研项（F-8 应力/Stage 4/场模式力挂起原因）、§5 使用注意事项（设置/解读/
明确不要做的）、§6 正确性保障方法论（恒等式探针/独立参照物/逐字节硬约束/
否定结论落档/文档纪律）、§7 路线一瞥。

## 2026-08-17 (6): F-8 H_HK 应力实现 + V-H7/Maxwell + efield 压电必测项

- **实现**：`compute_hk_force` 增 H_HK 应变导数核（串行；bra-only × 全分离
  矢量 d_β，相位应变不变）；σ^HK = +0.5·Im(Σkern)/Ω。两个修正：σ 符号
  （力侧 F=+0.5·Im 的 ε 版镜像，σ=+0.5 非 −0.5）与 ÷2 double-visit 因子
  （pair 双向访问 d/g 同翻 ⇒ 每对同号双计）。
- **V-H7**（λ=−0.005，ε=±0.5%/±1% 四点）：dE_HK/dε=+0.0054507 Ry →
  σ_FD=−2.020e-7 vs σ_ana=−2.156e-7 → **6.8%**；内部 Maxwell
  V·∂σ/∂λ=+1.1664 vs −∂Γ/∂ε=+1.0901 → **7.0%**。严格 5% 线上未过，
  残差 100% 归因 C 响应（(∂E_HK/∂C)(dC/dε)，E_HK 非变分；与 T3 力侧
  7% 同源同量级），机制一致性成立。
- **efield 必测项**（dip_cor=0、amp=0 基线、matched-γ）：dσ/dE=−67.79
  kbar/a.u.（严格线性）、dΣγ/dE=+0.484 rad/a.u.。**同体系杠杆**：应力侧
  L_σ/L_μ=1.78×（总应力）/2.68×（σ^HK 分量）；力侧 L_F/L_μ=52.5×
  （h2o 17.7× 同族）⇒ **17.7× 类失配是力路径特有，非几何导数普适**——
  F-8 前瞻警告（压电 Maxwell 同族问题）未兑现，EFC 不需要为压电路径立项。
- **回归**：单测 11/11（math/gauge/l1；smoothness 4/8 为既有 L9 参考过期）、
  MPI smoke 4/4、lam_m005 σ 清理后逐位一致；E'(λ) λ² 抛物零线性项。
- 文件：`2026-08-17-deltap-f8-stress.md`；INPUT 文档 hk 应力注释同步。

## 2026-08-17 (7): Stage 4.1 锚点重建 #3 + PW Γ 锚点 + L9 smoothness 修复

- **LCAO 锚点 #3**（12 用例，全串行 OMP=1）：9 标签位移系 + bn_test（4-rank）+
  test_stru_target（4-rank inner loop）+ relax（1-rank）全部 rc=0；λ 收敛、
  E' 带内平滑；bn_center 重跑与 P3 逐位一致（确定性验证）。operator 模式
  下 B 原子 τ=0 ⇒ Γ_B 结构性 ≈0.01（约束作用在 Γ 非 γ），|γ−t|≈4 rad 为
  预期，锚点 #2（gamma 模式）不可比——非回归。结果入
  `tests/deltap_bn_sampling/results.csv`（新表头 Gamma_B,Gamma_N）。
- **PW Γ 锚点生产档** `tests/deltap_pw_h2o_anchor/`：λ=0 Γ/atom=
  (5.236100,1.348236,1.348093) FINAL=−466.94573912497 eV；λ=−0.01 冻结
  escon=+0.079529=λΣΓ 恒等 ✓。KPT 须写 `Monkhorst-Pack`。
- **L9 修复（smoothness 8/8）**：根因三层——① 生成器独立随机相位使 Wilson
  乘积 |prod|≈0（arg 无定义，任何 seed 都过不了 dP∝eps 判据；"seed 188
  良态"系相邻比≈1 的错误判据）；② T5 用逐元素随机相位冒充规范变换（非
  波函数规范，破坏带结构）；③ T8 用均匀全局相位旋转冒充结构位移（与
  C/dS_k 不一致）。修复：D_I 正交归一 2×2 块 + T5 周期逐带规范 +
  T8 perturb_D_I 结构扰动。重建后单测 5/5（math/gauge/smoothness/l1/common）
  全 PASS，复现器 seed 1–200 全 PASS（T4 最小裕度 1.67）。
- 文件：`2026-08-17-deltap-anchor3-rebuild.md`；TODO 4.1 ✅；
  progress-summary Stage 4 行更新。
