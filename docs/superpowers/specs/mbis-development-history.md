# MBIS 局域电荷/自旋约束：开发与验证变迁总览（权威索引）

> 本页是 MBIS 系列的**索引 + 更正台账 + 当前规范**。阅读任一早期 spec 前先看本页。
> 凡标注「⚠️ 已被取代/更正」的文档，以本页与 `deltap-development-log.md`（S6.34–S6.72）
> 指定文档为准。分支 `mbis`（原 `feat/unified-local-moment-allocation`），远程 `zdy`，
> PR: https://github.com/dyzheng/abacus-develop/pull/new/mbis

> **横向对比综述（公式+机制+实测）**：`local-moment-constraint-algorithms-review.md`。
> **完整演进综述（2026-05-17→今，DeltaSpin→DeltaP→Allocation→MBIS）**：`constraint-algorithms-complete-review.md`。
> **优势论证 + 开发/测试路线**：`mbis-why-better-and-roadmap.md`。


---

## 0. 一句话现状

MBIS（迭代 Slater Stockholder，**带布居约束的不动点**）已作为
`constraint_weight_type mbis` 接入 ABACUS 主约束循环，支持**电荷 + 自旋同时约束**、
**MPI 并行**（pool gather）、**Picard 求解**（稳定性 `ρ(∂F/∂p)<1`）+ **快速 mu**（secant）：

- **LiCl（电荷）**：3 个外步 `CONVERGED`（`constraint_kappa_min≈0.01`,`constraint_step_max≈1.5`）。
- **bcc Fe2（体心 0.50，`constraint_mbis_images 1`）**：**对称稳定** N=16/16、M=±2.85、ρ≈0.31。
- **FeO（charge+spin，无 U）**：能量已自洽（`−ΣμQ` 修复），但 PBE 金属磁性不稳，
  需 **DFT+U** 做受控的 charge+spin 验证。

---

## 1. 时间线与阶段（文档 → 要点）

| 阶段 | 文档 | 要点 |
|---|---|---|
| allocation 设计 | `2026-10-21-unified-local-moment-allocation-design.md`、`2026-10-21-local-moment-algorithms-comparison.md`、`2026-10-29-allocation-development-summary.md` | 正交目标轨道 + 互补子空间分配 |
| allocation→MBIS | `2026-10-30-allocation-vs-mbis-formula-analysis.md`、`mbis-alignment-plan.md`、`mbis-prototype-licl-comparison.md`、`mbis-charge-spin-constraint-algorithm.md`、`mbis-feasibility-fe-and-orbital-separation.md` | 公式对齐、LiCl 三方案、Fe/磁性/轨道分离 |
| 模块三阶段 | `mbis-grid-bridge.md`(S1)、`mbis-weightgrid-backend.md`(S2)、`mbis-constraint-loop-integration.md`(S3) | Grid 桥接 → WeightGrid 后端 → 主循环接入 |
| ALM | `mbis-alm-penalty-plus-lagrange.md`、`-calibration`、`-comparison`、`-convergence-steps`、`-adapter-e2e`、`-e2e-real-cube`、`-real-cube-validation`、`-integration-plan`、`-development-summary` | 增广拉格朗日、k 标定、真实密度 |
| 收敛（100 步） | `mbis-convergence-attempts-log.md`、`mbis-50-steps-inexact-inner.md`、`2026-10-30-100-step-budget-analysis.md`、`mbis-solver-correction-and-alm.md`、`mbis-constrained-solver-math-and-plan.md`、`2026-10-30-alm-in-scf-result.md`、`mbis-faster-mu-controller.md` | FD 探针/in-SCF ALM/mixing 尝试、非精确内层、secant 快速 mu |
| 对称性/多解 | `mbis-correctness-and-symmetry.md`、`mbis-kl-multiplicity-hessian.md`、`mbis-unique-solver.md`、`mbis-hessian-certification.md`、`mbis-site-equivalence-fix.md`、`mbis-symmetrization-clarification.md`、`mbis-constrained-objective-correction.md`、`mbis-fixedpoint-jacobian-newton.md`、`mbis-fixedpoint-spectral-radius.md` | 破缺根因、多解、选解、稳定性判据（**含多次更正**，见 §2） |
| 真实 cube/根因 | `mbis-real-cube-stability.md`、`mbis-minimum-image-symmetry-fix.md`、`mbis-smooth-force-pro.md`、`mbis-d-degeneracy-fallback.md`、`mbis-fe2-test.md`、`mbis-in-basin-l-projection-test.md` | **min-image 为真正根因修复**；力光滑；D 退化 |
| 主循环 e2e | `mbis-abacus-e2e-validation.md`、`mbis-mpi-gather.md` | 约束激活、Q 与独立一致；pool gather np=1/2/4 一致 |
| FeO/能量 | `mbis-feo-charge-spin-validation.md`、`mbis-feo-branch-flip-analysis.md`、`mbis-energy-audit-deltaspin-comparison.md` | charge+spin 同时生效；能量记账审计与修复 |

`lcao-mbis-saturation.md`、`2026-10-30-why-licl-ok-fe2-breaks.md` 等为**被后续更正的历史诊断**，见 §2。

---

## 2. 更正 / 撤回台账（**阅读时以"取代/依据"为准**）

| 旧结论（文档） | 状态 | 取代 / 依据 |
|---|---|---|
| MBIS = **自由最小化** KL `D(p)`（`correctness-and-symmetry`、`unique-solver`、`kl-multiplicity-hessian`） | ❌ 撤回 | MBIS 是**带约束不动点**：`constrained-objective-correction`(S6.59) |
| 多初值 + **最低 D** 选"正确解"（`unique-solver`） | ❌ 撤回 | `D` 在 `ζ→0`（均匀分割）**=0 全局最小**，退化：`d-degeneracy-fallback`(S6.65) |
| **自由 D 的 Hessian** 认证/鞍点分类（`hessian-certification`、`kl-multiplicity-hessian`） | ❌ 对象错 | 稳定性判据是 **`ρ(∂F/∂p)`**：`fixedpoint-spectral-radius`(S6.61) |
| `‖J‖₂=32.7 ⇒ Picard 可能不稳`（`fixedpoint-jacobian-newton`） | ⚠️ 更正 | 真实谱半径 **0.81<1 稳定**：`fixedpoint-spectral-radius` |
| **同种原子 `symmetrize`** 修复 Fe2 破缺（`site-equivalence-fix`、`symmetrization-clarification`） | ❌ 移除 | 根因是 **direct-pro corner 低估** → **min-image**：`minimum-image-symmetry-fix`(S6.63) |
| "Fe2 在任何 `images` 下都 16/16 对称"（早期） | ⚠️ 条件 | `images=0`(direct) 会破缺；0.51 胞两 Fe 本不等价：`real-cube-stability`(S6.62) |
| legacy `cc_escon = Σμ(Q−t)` 正确 | ❌ 错 | 应 **`−ΣμQ`**（DeltaSpin `−ΣλM`）：`energy-audit-deltaspin-comparison` / S6.72 |
| FeO `BRANCH_FLIP` = 参考非全局极小 / MBIS 权重 bug | ❌ 否 | 是**能量记账（大 Q）致假 flip**；权重/观测正确：`feo-branch-flip-analysis` + `energy-audit` |
| in-SCF ALM（`alm` 调度）可加速（`solver-correction-and-alm`） | ⚠️ 负结果 | 罚项过刚、dampen 砍 k 过慢：`convergence-attempts-log`、`alm-in-scf-result` |

> `lcao-mbis-saturation.md`（"饱和"）、`why-licl-ok-fe2-breaks.md` 的结论均已被上述根因
> （min-image）与更正取代。

---

## 3. 当前规范算法（canonical）

**定义（每原子 A，Slater-s）**
```
pro_A(x) = N_A · ζ_A³/(8π) · exp(−ζ_A · d_min(x, R_A))
w_A(x)   = pro_A / Σ_B pro_B ,   N_A = ∫ w_A ρ ,   ζ_A = 3 N_A / ∫ r w_A ρ
```
`d_min` 为**周期最小镜像距离**（见 `images`）。`Σ_A N_A = ∫ρ`（求和规则）自动成立。

**求解**
- **Picard 不动点**（`compute_mbis`，初值 `ζ~1` 或 warm）。**规范解 = 该不动点**（稳定）。
- **不要用 D 选解**：`D` 仅诊断（`ζ→0` 时 `D=0` 退化）。
- **稳定性判据**：`ρ(∂F/∂p) < 1`（`mbis_fixedpoint_spectral_radius`）。
- `compute_mbis_newton_fp` 为诊断；退化时**回退 Picard**。

**`constraint_mbis_images`（0/1/…）**
- `k` → 取**晶格平移壳层半径 R=k**：`(2k+1)³` 个平移。
- `0` → `{0}`（**direct**，仅 home cell；**只适合大盒**）。
- `≥1` → 含最近/更远周期像（≈ 最小镜像；**小胞必须 ≥1**，否则 corner 低估 → 破缺/乱值）。

**pro 与均径 r**
- 均径 `r`（用于 ζ）与 pro 使用**同一 `images` 集**（`images=0`→direct；`≥1`→min-image）。
- **力**用**光滑晶格和** pro（`Σ_T exp`，对原子位置可微），解析导数与 FD 一致
  （`smooth-force-pro`）；观测/迭代用上面的 pro/r。

**能量表达式（关键，`energy-audit`/S6.72）**
- legacy：`cc_escon = −Σ_alpha μ_alpha · Q_alpha^{channel}`（电荷用 `Q`，collinear spin 用 `M`；
  **DeltaSpin `−ΣλM` 形式，显式扣除约束势 `∫V_con ρ`**）。
- audit 的 `e_con = Σμ(Q−t)`（物理泛函，仅用于报告/收敛残差）。
- allocation：`cc_escon = −Σμt`（eband 已含 μQ）。
- 分支守卫 `constraint_branch_tol`：约束相 `etot` 比 mu=0 参考低超 tol ⇒ `BRANCH_FLIP`（拒绝验证）。

**mu 控制器**
- secant/trust-region；`kappa = clamp(|dQ/dmu|, constraint_kappa_min, kappa_max)`，`|dmu|≤constraint_step_max`。
- **stiff 通道**（如 MBIS `|dQ/dmu|≈0.04`）：`constraint_kappa_min≈0.01` + `constraint_step_max≈1.5`
  → 数步收敛（LiCl 3 步）；默认 `0.3/0.05` 保守但慢。

**主循环**
- `WeightGrid::build_mbis`：主循环为 **MPI pool gather**（z-fastest 重排），全局 `compute_mbis`，
  再 scatter 权重回本地（`mpi-gather`）。
- `cutoff=10` bohr（主循环默认）；`constraint_mbis_images` 由用户设。

---

## 4. 验证状态

| 体系 | 设置 | 结果 |
|---|---|---|
| LiCl（电荷） | LCAO nspin1, MBIS, delta | 3 外步 `CONVERGED`（q=1.9945，|res|<1e-4） |
| bcc Fe2（0.50） | `constraint_mbis_images 1` | 对称稳定：N=16/16、M=±2.849、ρ≈0.31、Picard=Newton |
| LiCl/Fe2 独立 cube | C++ vs Python | N 一致（Li 2.0447；Fe 16/16、ζ1.078 等） |
| 主循环 e2e | LiCl `constraint_weight_type mbis` | 约束激活、Q 与独立一致；`np=1/2/4` 一致（MPI gather 正确） |
| FeO（无 U）charge+spin | `nspin 2`, 无 DFT+U | 两约束同时生效（参考 Fe 14.27/2.85 与独立一致）；**能量 −ΣμQ 修复后自洽**；但 PBE 磁性不稳→spin 翻态；需 DFT+U 受控 |
| 力 | 模块 gtest | 12/12（`ForceFiniteDifference` 在光滑 pro 下恢复） |

---

## 5. 已知限制 / TODO

1. **FeO charge+spin 受控验证需 DFT+U**（PBE 无 U 金属、磁态近简并，SCF 会换态）。
2. 大 `Q` 约束能量修复（`−ΣμQ`）应在 gapped（DFT+U）体系复核自旋翻转对称性。
3. **主循环默认 `constraint_mbis_images=0` 对小胞不安全**（应在 `build_mbis` 里自动提示/默认
   ≥1，或按盒子尺寸自适应）。
4. `compute_mbis_newton_fp` 在真实大体系可能退化（主用 Picard）。
5. 应力/力主路径走 `module_constraint` 通用 `WeightGrid` 力核；MBIS 模块 force 仅单测（已光滑）。

---

## 6. 阅读指引（防重复踩坑）

- **先读本页 §2（更正台账）+ 本页 §3（当前规范）**，再看 dev log `S6.59`、`S6.61`、`S6.63`、
  `S6.65`、`S6.71`、`S6.72`。
- 标注「⚠️ 已被取代」的早期 spec（`unique-solver`、`hessian-certification`、`kl-multiplicity-hessian`、
  `correctness-and-symmetry`、`site-equivalence-fix`、`symmetrization-clarification`、
  `alm-development-summary`、`lcao-mbis-saturation`、`why-licl-ok-fe2-breaks`）**只作历史记录**，
  **结论以下文为准**。
