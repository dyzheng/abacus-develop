# 2026-10-30 对齐 ORCA MBIS 的方案（Allocation / Hirshfeld 后端）

## 0. 结论（先给判断）

**值得做，且代码已有约 70% 的脚手架**：现有 `WeightType::HirshfeldPeriodic`
后端已经是"每原子 s 型 pro-density + 周期镜像求和 + C2 窗口 + 导数"的
stockholder 分区，只差三件事：

1. pro-density 的**布居与展宽改为自洽迭代**（按 KL 散度最小化），而不是取
   `σ=划分半径、q=zv` 的固定值；
2. pro-density 形状由 **Gaussian 换成 Slater s 函数**（与 ORCA 同族）；
3. 增加**原子多极**（偶极/四极）积分输出。

**但不要替换现有 allocation**：现有算符 `A_i=Q_i+Q_c W_i Q_c` 的
`Q_i`（目标轨道投影）是 MBIS 没有的（XMCD/壳层口径），且其**固定线性权重**
对约束/力更稳健。MBIS 应作为**新增的纯空间后端/观测量**，与投影族互补。

## 1. 现状 vs MBIS（逐项对照）

| 维度 | 当前 `HirshfeldPeriodic`（代码事实） | ORCA MBIS |
|---|---|---|
| pro-density 形状 | **Gaussian** `q·exp(−r²/2σ²)` | **Slater s** `q·e^{−ζr}` |
| pro-density 参数 | `σ = max(radii,0.25)`、`q = zv`，**固定** | `q_A、ζ_A` **迭代优化** |
| 参数确定 | 几何/赝势给定，**不迭代、无 KL** | 最小化 `D_KL(ρ_A^pro‖ρ_A)` |
| 权重来源 | `w_A=ρ_A^pro/Σρ_B^pro`，**几何-only**（代码注释明确"密度不进入权重"） | 依赖 ρ，**自洽** |
| 分区 | 轨道投影 + 空间互补（混合） | **纯实空间** stockholder |
| 观测量 | 单极（电荷/自旋） | 单极 + **偶极/四极/八极** |
| 角色 | 约束（线性、精确求和、有力核） | 后处理分析（非线性） |
| 周期性 | 已支持镜像求和 + C2 窗口 | 分子（需 PBC 适配） |

关键代码位置：
- `source/source_estate/module_constraint/weight_grid.h:17`（`WeightType`）、
  `:209-227`（Hirshfeld 字段）；
- `weight_grid.cpp:112`（`prepare_hirshfeld_geometry`：固定 σ/q + 镜像界）、
  `:215`（`hirshfeld_atom_density`：Gaussian 参考密度）、`build_hirshfeld*`；
- 输入 `constraint_weight_type = becke|hirshfeld`（`input_parameter.h:757`）；
- 选择后端 `constraint_loop.cpp:494`。

**重要语义差异**：`weight_grid.h` 明确"权重是几何的纯函数，密度不进入"。
MBIS 的权重**依赖密度**——这是对齐时最本质的改动，决定了必须引入**外层迭代**
（pro-density 在 SCF/μ 步之间更新）与**力的分区响应项**。

## 2. "对齐 ORCA"的具体含义

要能与 ORCA 数值对照，需要：

1. **同族 pro-density**：s 型 Slater，参数 `(q_A, ζ_A)` 每原子；
2. **同目标函数**：逐原子最小化 `D_KL(ρ_A^pro‖ρ_A)`（s-Slater 下可解析：
   `q_A=∫ρ_A`，`ζ_A` 由 `∫r ρ_A` 的比值定）；
3. **同一多极定义**：
   - `q_A = Z_A − ∫ρ_A`
   - `μ_A = −∫(r−R_A)ρ_A`（以核为原点）
   - `Q_A^{ab} = −∫(r−R_A)_a(r−R_A)_b ρ_A`
   - 分子偶极重建：`μ_mol = Σ_A [q_A R_A + μ_A]`；
4. **同单位**（ORCA 用 a.u.；电荷 e、偶极 a.u./D）；
5. **验证集**：H2O、LiCl 等 ORCA 可跑的分子（ABACUS 用大盒子/团簇近似，
   或对 PBC 适配的 MBIS 与 ORCA 同分子对照）。

## 3. 分阶段方案

### Phase A（只读观察量，低风险，建议先做）

目标：在 ABACUS 输出可与 ORCA 对照的 MBIS 电荷 + 原子偶极/四极，**不动约束/力**。

- 新增 `WeightType::MBIS`（或 `hirshfeld` 的 `iterative=true` 分支）：
  1. 初始化：Slater s，`q_A=Z_A`、`ζ_A` 由共价/划分半径给初值；
  2. 迭代：`w_A=ρ_A^pro/Σρ_B^pro` → `ρ_A=w_Aρ` → 对每原子拟合 `q_A,ζ_A`
     （KL 最小，s-Slater 解析解）→ 更新，至 `|Δ| < 1e-6`；
  3. 复用现有周期镜像 `hirshfeld_images_/hirshfeld_cutoff_` 与 C2 窗口；
  4. 网格积分 `q_A`、`μ_A`、`Q_A`（`ρ_A` 在 FFT 网格上）；
- 输出块 `MBIS ATOMIC CHARGES / DIPOLE / QUADRUPOLE`（对齐 ORCA 的
  `MBIS_LARGEPRINT`）；
- 验证：H2O/LiCl 的 `q_A`、`μ_A`、`Σ(q_A R_A+μ_A)` 与 ORCA 对照；
  求和 `Σq_A=N`、`Σ(q_A R_A+μ_A)=μ_mol`。
- 成本：复用现有镜像/窗口/网格，新增一个迭代循环（O(N_at·N_grid)），
  相对 SCF 很小。

### Phase B（MBIS 作为约束，中风险）

目标：用 MBIS 的 `w_A` 作为 `A_i` 的权重，约束 MBIS 电荷（可选自旋、偶极）。

- **冻结分区近似**：每个外层/μ 步用上一步收敛的 `w_A^MBIS(ρ_n)` 作为**固定**
  权重，步内仍是线性算符（沿用现有 ALM/外循环）；步间更新 pro-density →
  嵌套固定点。这是把非线性降到"外层迭代"的务实做法。
- 约束类型：
  - `kind=Charge`：`q_A^MBIS = Z_A − ∫w_Aρ`；
  - `kind=Spin`（spin-MBIS）：同一 `w_A` 分 `ρ↑−ρ↓`；
  - **新增** `kind=Dipole`：约束 `μ_A^{local}` 分量（现有框架没有的能力，
    对可极化力场/铁电有意义）。
- 力：冻结分区 Hellmann–Feynman + 显式 `∂w/∂R`（现有 Becke/Hirshfeld 导数核）
  + 分区响应修正；在 KL 收敛点可用包络定理简化隐式项。
- 验证：`Σq_A=N`、力 FD、与 Phase A 的观测量逐位一致。

### Phase C（完全自洽 + 多极约束，高风险/按需）

- 把 `∂w/∂ρ` 纳入约束 Jacobian（真正的自洽 MBIS 约束）；
- 四极约束、以及 cMBIS（强制原子多极之和精确匹配分子总偶极/四极；
  ORCA 原生也没有 cMBIS）；
- 仅在 Phase B 的冻结近似不能满足精度时再投入。

## 4. 周期性与网格的注意点

1. **PBC**：电荷/多极是周期安全的（权重随原子密度衰减）；但"分子偶极重建"
   在 PBC 下需用 Berry 相位/sawtooth 或有限团簇，不能直接 `Σ q_A R_A`。
2. **网格质量**：MBIS 原子偶极对网格敏感；ABACUS 的 FFT 密度网格若偏粗，
   需提高 `ecutrho` 或做插值；ORCA 建议 GRID5/6 同源。
3. **镜像求和**：现有 `hirshfeld_images_` 已按倒格矢给界；Slater 尾部比
   Gaussian 长，cutoff/窗口需相应放大并做求和审计（`Σ_A w_A=1`）。
4. **Slater vs Gaussian**：要与 ORCA 数值对齐必须用 Slater；保留 Gaussian
   作为快速选项。

## 5. 已知 MBIS 局限（要在文档/结果中标注）

- 原始 MBIS 分子偶极常**高估 ~10%**；需 cMBIS 才严格匹配（ORCA 无）。
- AIM 分区本就不唯一；与 Becke/DDEC/DeltaSpin 数值不可混用。
- 高斯基组 vs Slater 参考密度的差异会带来小偏差（需对照表）。

## 6. 与现有方法的关系（重要）

- **DeltaSpin**：约束 onsite 投影矩（XMCD 口径）；MBIS 是**空间 AIM**，
  其 spin-MBIS 更接近 Becke 空间矩（此前测得 projector ≈3.6–3.9 vs
  Becke ≈2.8）——**不能把 MBIS spin 当作 DeltaSpin 投影矩**。
- **当前 allocation**：`q=m_T+m_c` 是"壳层投影+空间互补"的混合坐标，有
  固定线性算符（约束/力友好）；MBIS 是**纯空间自洽**坐标。两者应并存：
  - 需 XMCD/壳层 → projector（DeltaSpin）；
  - 需可极化力场/ESP/原子多极 → **MBIS**；
  - 需"壳层+空间"折中 → 现有 allocation。
- **同时约束电荷+自旋**：MBIS 天然支持（同一 `w_A` 读 `ρ↑±ρ↓`），
  但需注意电荷冗余（R8：全原子电荷约束+固定电子数→线性相关，
  用"1 全局 + N−1 差异"）。

## 7. 建议的下一步（最小可验证）

1. **Phase A**：在 `weight_grid` 加 `MBIS` 迭代后端 + 多极输出；
   用 H2O/LiCl 与 ORCA 对照（q、μ_A、分子偶极重建），确认数值口径。
2. 对照通过后，再评估 **Phase B**（冻结分区约束）的成本/精度；
   若目标只是分析/力场参数化，Phase A 已足够。
3. 不建议直接跳到 Phase C（自洽 Jacobian/多极约束），除非有明确需求。

**一句话**：对齐 MBIS 是"把现有 fixed-Gaussian Hirshfeld 后端升级为
iterative-Slater-KL 分区 + 多极输出"；脚手架已在，先做只读观察量最划算，
约束化按需推进，绝不替换投影/分配主路径。
