# 实空间权重约束（constraint）用户使用说明

> 版本基线：feat/deltap 分支 `fc93bac1e`（2026-09-09）。功能状态：**SCF 级电荷/自旋约束可用，且同一 run 可混合（v2 JSON 列表）；约束力已在限定包络内经 FD 验收（H₂O 类小分子、|μ|≲0.5 Ry、残差<1e-4），relax/MD 尚未接线；应力不支持。**
>
> 2026-09-11 增补：§2 新增 `constraint_step_max` / `constraint_step_probe`
> 两个外环步长上限参数（默认值与此前硬编码行为一致，老输入无需改动）。

## 1. 功能概述

在实空间网格上定义原子/片段权重 w_I(r)（逐点 Σ_I w_I ≡ 1），对约束观测量
Q_α = ∫ w_α(r) d_α(r) dr 施加 Lagrange 约束，约束势 V_con = Σ_α μ_α w_α(r)
注入有效势。乘子 μ 由外环逐分量 secant 自动收敛（带 κ 限幅、翻号检测、
μ_max 熔断）。

- **观测量 = 注入算符**：读数与注入共用同一权重场，记账恒等式自洽；
- **基组无关口径**：PW 与 LCAO 在同一密度网格上读数，同密度下逐位一致；
- **每 SCF 打印审计行**：`total_charge`（Σ_α Q_α，全片段覆盖时=nelec）与单位分解
  偏差 maxdev（应恒为 ~1e-16）永久自检。

## 2. INPUT 参数

| 参数 | 缺省 | 说明 |
|---|---|---|
| `constraint` | false | 总开关 |
| `constraint_type` | charge | run 级通道：`charge`（通道 ρ↑+ρ↓）或 `spin`（通道 m=ρ↑−ρ↓，要求 nspin=2）。**只对旧 v1 靶文件生效**；v2 靶文件逐约束自带 `type` 并取代 run 级值（run 级 ≠ charge 时打 WARNING）。混合 run 直接用 v2 文件，不需要也不支持 run 级表达混合 |
| `constraint_weight_type` | becke | 权重类型，当前仅 `becke`（Becke 模糊 Voronoi 分区，3 阶迭代多项式 + 共价半径异核修正） |
| `constraint_target_file` | （必填） | 靶点 JSON 文件路径；**缺失或为空 → WARNING_QUIT**（不允许无靶点隐式约束） |
| `constraint_target_mode` | delta | `delta`：靶点 = 参考态读数 + 偏移（推荐）；`absolute`：绝对靶点（会打印口径 WARNING——Becke 电荷与 SZV/ Mulliken 量级不同） |
| `constraint_mu_max` | 5.0 | 乘子熔断上限（Ry）；顶限且残差平台 → 判 UNREACHABLE 并报告 Q(μ) 端点 |
| `constraint_thr` | 1e-4 | 约束收敛容差（每分量 \|Q−t\|，单位 e 或 μB） |
| `constraint_step_max` | 0.05 | 外环**每步** \|Δμ\| 上限（Ry）。外步数 ≈ \|μ\*\|/该值，故调小时 `scf_nmax` 要相应放大 |
| `constraint_step_probe` | 0.0 | **仅首步**（该分量尚无历史读数、还没有割线斜率时）的 \|Δμ\| 上限（Ry）。`0` = 沿用 `constraint_step_max`（旧行为）。必须满足 `0 ≤ probe ≤ step_max`，越界 → WARNING_QUIT |

**为什么有"首步"专用上限**：外环用割线法推进 μ，需要两次读数才能得到斜率
`dQ/dμ`；首步没有历史，只能回落到保守斜率 `kappa_min`，于是**首步永远顶到
`constraint_step_max`**——一个与靶点远近无关的固定过冲。电荷通道一般无碍
（MgO 实测线性区 ≥±0.8 e），但响应刚度大的通道（例如 FeO 的自旋通道，
瞬态响应 ~20 μB/Ry）一个 0.05 Ry 的首步就能把 SCF 推离参考分支。
把 `constraint_step_probe` 设成小值（实测 0.004 Ry 够用）即可先量一个**局部**
斜率，之后由 `constraint_step_max` 接管。**该参数只影响首步**：步数增加有限，
但收敛的 μ\* 与步长选择无关（FeO 实测 `step_max` 0.05 vs 0.01，μ\* 只差 0.1%）。
注意这两个参数是**缓解**手段，不替代分支守卫（在线分支守卫尚未落地，
见 §5）。

## 3. 靶点文件（JSON）

### 3.1 v2 格式（推荐，阶段 A 起支持逐约束类型/混合）

```json
{"constraints": [
  {"type": "charge", "target": 0.1, "atoms": [0]},
  {"type": "spin",   "target": 0.1, "atoms": [0]},
  {"type": "charge", "target": -0.05, "atoms": [1, 2], "mu_max": 3.0}]}
```

每条约束对象（键任意顺序；解析器为仓库内手写 JSON 子集，仅支持下列结构）：

- `type`（必填）：`charge`（通道 ρ↑+ρ↓）或 `spin`（通道 m=ρ↑−ρ↓，**要求 nspin=2**）；
  其余值（含 `dipole`）→ ERROR（偶极阶段 A 不实现，守卫拒绝而非猜测）；
- `target`（必填）：目标值——delta 模式为偏移量（charge 单位 e、spin 单位 μB）；
- `atoms`（可选）：该约束的原子片段（0 起全局下标，**扁平数组**，多原子=求和；
  嵌套片段是 v1 写法，在 v2 中 → ERROR）；缺省 = 单原子 [列表序号]；
- `mu_max`（可选）：该约束独立的乘子熔断上限（Ry），覆盖 run 级值；缺省 = run 级。

**混合**：`charge` 与 `spin` 可在同一列表并存（列表顺序即审计行 c[0..N-1] 顺序）；
spin 分量使整个 run 必须是 nspin=2。重复 (type, atoms) 组合打 WARNING（近共线预警）。
同原子双类型混合的真实例见 213 用例（charge +0.1 e + spin +0.1 μB on O）。

### 3.2 v1 格式（旧，已 deprecate）

```json
{"targets": [0.1, -0.1], "atoms": [[0], [1, 2]]}
```

- `targets[i]`：第 i 个约束目标（delta 模式为偏移量）；
- `atoms[i]`：第 i 个约束的原子片段（**可嵌套**：多原子=片段求和）；缺省 → 逐原子；
- 整表共享 run 级 `constraint_type` 单通道；自动转换到内部模型继续跑，并打印
  deprecation WARNING；三个已注册旧用例（211/212/212_NAO）**零修改**保持逐位复现。
  新用户请迁移到 3.1。

### 3.3 文件级规则

- 同一文件混用 `constraints` 与 `targets` → ERROR（不猜测语义）；
- v2 文件存在时，run 级 `constraint_type` 被逐约束 `type` 取代（run 级 ≠ charge 打
  WARNING "supersede"，不静默忽略）；
- 缺失或空靶文件 → WARNING_QUIT（不允许无靶点隐式约束）。

## 4. 输出解读

```
CONSTRAINT_AUDIT nconstraint=2 e_con=... max_residual=... total_charge=... nelec=8 maxdev=2.2e-16
CONSTRAINT_AUDIT c[0] kind=charge q=6.3554 t=6.3554 mu=-0.1812 res=-1.4e-05
CONSTRAINT_AUDIT c[1] kind=spin q=0.09993 t=0.10001 mu=-0.0815 res=-7.9e-05
[constraint] outer step 15 after SCF iteration 117 (phase=constrained)
[constraint] final status: CONVERGED (targets reached within 0.0001 e)
```

- `c[i] kind=...`：逐约束通道标签（charge/spin；A4 起生产审计恒带）；
- `q/t/mu/res`：逐约束读数/靶点/乘子（Ry）/残差；
- `total_charge`：Σ_α Q_α（混合 run 含 spin 分量、按 e 记账——为信息性总和，不声称
  恒等于 nelec；单约束/全片段覆盖时才具备 sum-rule 语义）；
- 终态：`CONVERGED`（全部达标）或 `UNREACHABLE`（熔断，附 Q(μ) 端点——目标物理不可达，非数值故障）。

**符号约定**：两通道均为负响应——正 μ 排斥该区域电荷/自旋上。delta>0（增电荷/增磁矩）对应 μ*<0。与 DeltaSpin 的 λ 换算：**μ = −λ**。

## 5. 当前边界（务必阅读）

1. **力**：已接线（PW/LCAO 同一网格核）并通过 FD 判决，能力声明**限定包络**：
   H₂O 类小分子、|μ|≲0.5 Ry、约束残差 <1e-4。包络外或逐轴未覆盖处不可默认外推；
   relax/MD/几何优化**尚未接线**（MD 接线前需逐几何复核 §3.1 势存活与 reset 语义）；
2. **应力**：不支持；
3. **混合约束**：v2 列表支持同 run charge+spin 混合（需 nspin=2）；**偶极/多极未实现**
   （`type:"dipole"` → ERROR，守卫拒绝）；run 级 `constraint_type` 仍是单值（只表达 v1）；
4. **平台**：仅 double/CPU/double_grid（其余 WARNING_QUIT 拒绝）；KPAR=1；
5. **多自旋约束**：近共线靶点（如同时对 O 和 H 约束磁矩）收敛显著变慢，属预期行为；
   混合双约束实测也会拉长外环（H₂O 同原子 case 3→15 外步）——阶段 B 将上 Broyden；
6. 金属/近简并体系未验证。
7. **在线分支守卫尚未落地**：外环只判 `|Q−t| < thr`，**无法区分"达标"与"SCF
   换了自洽解"**。多解体系（DFT+U 氧化物、强关联绝缘体等）上必须逐点事后核对：
   约束态能量不得低于锚点（`E(δ) ≥ E_anchor`），且能升与线性响应预言
   `½·δ·|μ*|` 一致。缓解手段是 §2 的两个步长上限；在线守卫（能量式
   `BRANCH_FLIP`，一旦触发即熔断并报告，不静默接受）计划落地，落地前请把
   上述事后核对作为多解体系上的强制检查项。

## 6. 示例用例（已注册测试套件）

- `tests/01_PW/211_PW_constraint_h2o/`：PW 电荷约束（delta=+0.1 e on O）
- `tests/01_PW/212_PW_constraint_h2o_spin/`：PW 自旋约束（delta=+0.1 μB on O）
- `tests/01_PW/213_PW_constraint_h2o_mixed/`：PW **混合**约束（v2 列表，charge +0.1 e + spin +0.1 μB 同 on O；μ_c=−0.1812、μ_s=−0.0815，含耦合偏移实测）
- `tests/02_NAO_Gamma/212_NAO_constraint_h2o/`：LCAO 电荷约束

各目录 README 含期望值；FD 验证脚本（开发用）：`tests/constraint_fd_force/tools/`。

## 7. 设计/验证文档

架构：`plan-architecture.md`；进展与测试总览：
`docs/superpowers/specs/2026-08-31-constraint-framework-progress-summary.md`
（一期闭合+二期 2.1–2.5）、`docs/superpowers/specs/2026-09-09-stageA-progress-summary.md`
（阶段 A：混合 charge+spin 数据流/测试/μ 耦合实测）；
术语表：`docs/superpowers/specs/deltap-constraint-glossary.md`。
