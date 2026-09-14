# 实空间权重约束（constraint）用户使用说明

> 版本基线：feat/deltap 分支 `fc93bac1e`（2026-09-09）。功能状态：**SCF 级电荷/自旋约束可用，且同一 run 可混合（v2 JSON 列表）；约束力已在限定包络内经 FD 验收（H₂O 类小分子、|μ|≲0.5 Ry、残差<1e-4），relax/MD 尚未接线；应力不支持。**
>
> 2026-09-11 增补：§2 新增 `constraint_step_max` / `constraint_step_probe`
> 两个外环步长上限参数（默认值与此前硬编码行为一致，老输入无需改动）。
>
> 2026-09-11 增补（第二件）：**在线能量分支守卫**落地——§2 新增
> `constraint_branch_tol`（默认 0 = 关），触发时熔断并报 `BRANCH_FLIP`；
> §4 增补状态 token，§5.7 由"尚未落地"改写为使用与限制说明。
>
> 2026-09-11 增补（第三件）：**审计行 on-site 投影矩**（最小 II-1b）——DFT+U
> 运行下每条约束的审计行新增 `onsite=`（该片段原子的 DFT+U 关联轨道占据迹差，
> 即日志 `atomic mag` 的同一个量），与 Becke 加权 `q` 并排可读；无 DFT+U 时
> 该 token 不出现，历史输出逐位不变。
>
> 2026-09-14 增补（第四件）：**双迭代调度**——§2 新增
> `constraint_mu_schedule`（`outer` 默认 / `inner`）、`constraint_inner_thr`、
> `constraint_inner_nmax` 三个 INPUT。`outer` 是此前唯一且默认的行为，
> 老输入**逐位不变**（已用对照二进制在 211 PW 算例上验证）；
> `inner` 是新增的 SCF 内环 μ 更新（DeltaSpin `lambda_loop` 血统）+ settle 检查。
> §4 增补 `inner step`/`settle` 审计行与 `MIX_RESET` 日志行，§5.9 给出使用与
> 边界说明；**"何时用哪个"的定量决策表在 §5.9 标注为待实测填表**
> （对照研究 Q1–Q3 完成后回填）。设计/验证：见 §7。

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
| `constraint_mu_schedule` | outer | μ 更新的调度：`outer`（现状，**默认**）= SCF 完整收敛 → 一次 M4 秒差步；`inner`（新增）= SCF **迭代内** drho 越过 `constraint_inner_thr` 即更新 μ → 清 mixing 历史（`mix_reset`）→ 继续 SCF，并在判决前做 settle 检查。取值只有这两个，其他 → WARNING_QUIT（不静默回退到 outer）。**`outer` 为默认即零回归**（211 算例逐位对照通过） |
| `constraint_inner_thr` | 1e-3 | **仅 `inner` 生效**的 drho 门控：只有 SCF 迭代的密度残差 drho 小于该值才更新 μ（密度还在漂时读出的是 mixing 噪声，不是 Q(μ)）。`inner` 模式下必须 `> 0`，否则 WARNING_QUIT；`outer` 模式下该值被完全忽略。**默认值三档标定见 §5 第 9 条** |
| `constraint_inner_nmax` | 20 | **仅 `inner` 生效**的安全预算：一次 run 内最多做多少次 SCF 内 μ 更新；用尽即打印告警并**降级回 `outer`**（绝不无限空转）。需要 N 个外步的扫描通常也需 ~N 次内更新，故扫描应调大。`inner` 模式下必须 `> 0`，否则 WARNING_QUIT |
| `constraint_branch_tol` | 0.0 | **在线能量分支守卫**容差（Ry，`0` = 关）。>0 时：每个 SCF 收敛点比较约束态总能量 `E_tot = E_KS + Σ_α μ_α(Q_α−t_α)` 与同一 run 的参考能量 `E_ref`（μ=0 参考相）；若 `E_tot` 比 `E_ref` 低超过该容差，判定 SCF 换了自洽解 → **熔断并报 `BRANCH_FLIP`**（不再静默当 CONVERGED）。必须 `≥ 0`，负数 → WARNING_QUIT |

**为什么有"首步"专用上限**：外环用割线法推进 μ，需要两次读数才能得到斜率
`dQ/dμ`；首步没有历史，只能回落到保守斜率 `kappa_min`，于是**首步永远顶到
`constraint_step_max`**——一个与靶点远近无关的固定过冲。电荷通道一般无碍
（MgO 实测线性区 ≥±0.8 e），但响应刚度大的通道（例如 FeO 的自旋通道，
瞬态响应 ~20 μB/Ry）一个 0.05 Ry 的首步就能把 SCF 推离参考分支。
把 `constraint_step_probe` 设成小值（实测 0.004 Ry 够用）即可先量一个**局部**
斜率，之后由 `constraint_step_max` 接管。**该参数只影响首步**：步数增加有限，
但收敛的 μ\* 与步长选择无关（FeO 实测 `step_max` 0.05 vs 0.01，μ\* 只差 0.1%）。
注意这两个参数是**缓解**手段，不替代分支守卫（在线分支守卫尚未落地，
见 §5；在线能量守卫已按 §2 的 `constraint_branch_tol` 落地）。

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

`inner` 调度下，同一审计发射器把标签换成 `inner` / `settle`（`outer step` 的
文本在 `outer` 模式逐位不变）：

```
[constraint] inner step 3 after SCF iteration 27 (phase=constrained)   # SCF 迭代内的 μ 更新
[constraint] MIX_RESET at SCF iteration 27 (mu update)                 # 与上一次更新一一配对
[constraint] mixing recovered after 4 SCF iteration(s) (reset cost)    # 复位代价（间距）
[constraint] settle check PASSED: residual 2.1e-05 < 0.0001 on the settled density (INNER schedule CONVERGED)
[constraint] settle check FAILED (x1): residual 3.4e-04 >= 0.0001 on the settled density (...)
[constraint] INNER schedule degraded to OUTER (settle check failed twice): ...
```

- `inner step N`：第 N 次 **SCF 迭代内** μ 更新（对应一次 `mix_reset`）；
- `MIX_RESET`：Broyden/DIIS 历史被清空（不动点映射随 μ 改变），与 μ 更新一一对应；
- `mixing recovered after K SCF iteration(s)`：两次复位之间隔了多少迭代——这就是
  `mix_reset` 的**复位代价**（mixing 破坏的定量读数，供 §5.9 决策表使用）；
- `settle check PASSED/FAILED`：内环宣告 CONVERGED 后的**临时**判决核验
  （冻结 μ、等密度松弛后重读 `|Q−t|`）；FAILED 计一次 `settle_fail`，连续两次
  → 降级 `outer`；
- `degraded to OUTER (<reason>)`：`inner` 因预算用尽（`constraint_inner_nmax`）
  或 settle 连续反弹而降级——**降级一定打印原因**，绝不静默。

- `c[i] kind=...`：逐约束通道标签（charge/spin；A4 起生产审计恒带）；
- `q/t/mu/res`：逐约束读数/靶点/乘子（Ry）/残差；
- `onsite=...`（**仅 DFT+U 运行时出现**）：该约束片段内原子的 **on-site 投影矩**
  之和——每个原子取 DFT+U 关联轨道占据矩阵的自旋迹差
  `Tr[M↑] − Tr[M↓]`（即 running log 里 `atomic mag: iat <m>` 的同一个量，投影
  球半径由 `onsite_radius` 控制）。它与 `q` 是两个**不同的观测量**：`q` 是
  Becke 加权盆地矩，`onsite` 是关联轨道的局域投影矩。TM 磁性体系上二者会
  **脱钩**（II-1b：FeO 换态点 Becke 掉 1.28 μB 而 on-site 只动 0.20 μB），
  `onsite=` 就是为此提供的同点对照读数；
- `total_charge`：Σ_α Q_α（混合 run 含 spin 分量、按 e 记账——为信息性总和，不声称
  恒等于 nelec；单约束/全片段覆盖时才具备 sum-rule 语义）；
- 终态三态：
  - `CONVERGED`：全部达标（残差 < `constraint_thr`），且分支守卫未触发；
  - `UNREACHABLE`：熔断（顶到 μ 上限且残差平台——目标物理不可达，非数值故障），附 Q(μ) 端点；
  - `BRANCH_FLIP`：**在线能量分支守卫熔断**——某个 SCF 收敛点的约束态能量低于参考能量
    超过 `constraint_branch_tol`，说明 SCF 换了自洽解（换态），**靶点未被验证**，
    结果不得当作 CONVERGED 使用（见 §5.7）。

开启分支守卫后，每个 SCF 收敛点还会打印一行（机器可读，可 grep `branch guard` / `BRANCH_FLIP`）：

```
[constraint] branch guard armed: constraint_branch_tol=0.001 Ry, e_ref=-562.440523 Ry (mu = 0 reference energy)
[constraint] branch guard: e_tot=-562.427201 Ry, de=e_tot-e_ref=0.013322 Ry (tol=0.001000 Ry)
[constraint] BRANCH_FLIP: constrained energy is 0.047280 Ry (0.643282 eV) BELOW the reference, ...
[constraint] final status: BRANCH_FLIP (fused: ... the targets are NOT validated)
```

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
7. **在线分支守卫（`constraint_branch_tol > 0`）**：外环单凭 `|Q−t| < thr`
   无法区分"达标"与"SCF 换了自洽解"；开启守卫后，每个 SCF 收敛点用能量判据
   `E_tot < E_ref − tol` 判断换态，触发即熔断报 `BRANCH_FLIP`（不静默接受）。
   适用要点与**残余盲区**：
   - **只抓"向下的换态"**：换态后的解若能量仍在 `E_ref` 之上，守卫不触发
     （这是判据边界，不是可调项）；因此多解体系仍建议保留事后核对
     `E(δ) ≥ E_anchor`、能升 ≈ 线性响应 `½·δ·|μ*|`（FeO II-1 实测 1.4%/2.4% 吻合）；
   - **只在 SCF 收敛点判**：固定 μ 下 SCF 自身双稳（不收敛）时守卫不触发，
     这类点表现为 `RUNNING`（`scf_nmax` 耗尽）——仍按失败处理；
   - **正确量级**：容差要高于 SCF 能量噪声、低于预期的约束能升。参考 FeO II-1：
     δ=+0.1 μB 时能升 +0.0439 eV（≈3.2e-3 Ry），故 `1e-3 Ry` 合适；
     H₂O 集成用例 δ=+0.1 e 能升 +0.0087 Ry，`1e-3 Ry` 同样安全（无假触发）；
   - **实验开关**：与 `ABA_CONSTRAINT_FIXED_MU`（无参考相）不兼容，同时给出 →
     WARNING_QUIT；守卫需要 `E_KS` 输入，未接线（缺能量）也 WARNING_QUIT；
   - 熔断只停止 SCF 并打印终态，进程正常退出、照常写能量与密度——**脚本/使用者必须
     检查 `final status` 是否是 `CONVERGED`**，不得只看 `!FINAL_ETOT_IS`。
8. **`q`（Becke 加权矩）≠ on-site 局域矩**：约束作用在 Becke 加权观测量的共轭
   量上（这一点是 CDFT 意义上严格的），但"约束 Fe 的自旋"**不等于**"控制 Fe 的
   d 局域矩"——Becke 盆地含尾部与间隙磁化，最软的响应通道是尾部重排。TM 磁性
   体系上请**同时读 `q` 与 `onsite`**（§4）：II-1b 实测在 FeO 换态点两者反向
   脱钩（`q` −1.28 μB vs `onsite` −0.20 μB），而良态的 δ=+0.1 μB 点两者同向
   （+0.100 vs +0.074 μB）。多解体系上还需配合 §5.7 的事后能量核对。

9. **双迭代调度（`constraint_mu_schedule`）**：`outer`（默认）与 `inner`
   两种 μ 更新调度的**实现**已落地并验证（OUTER 逐位零回归；INNER 由 6 个单测
   + 3 发门控 sabotage 覆盖，见 §7 的 spec）。使用边界：
   - **`outer` 是默认且零回归**：不设该参数 = 旧行为；单测里把 drho 置 0
     （任意门控都满足）也**绝不**触发内环（`OuterLegacyBitIdentical`）；
   - **`inner` 的三道守卫**：`inner_thr`（drho 门控，严格 `<`）、`inner_nmax`
     （预算，用尽降级）、settle check（内环 CONVERGED 需经松弛复核）；
     任一守卫触发都会**打印原因**，不会静默；
   - **`inner` 的已知代价**：每次 μ 更新都清 mixing 历史（`mix_reset`），
     重启一段 Broyden 收敛；代价大小可在日志里用
     `mixing recovered after K SCF iteration(s)` 读到；
   - **INNER 仍保留 OUTER 秒差步作安全网**：只有当门控配置自相矛盾
     （`constraint_inner_thr ≤ scf_thr`，收敛时门控反而关闭）或已降级时，
     外步才接管——因此不会"因调度而停在未达标的点"；
   - ✅ **`inner_thr` 已完成三档标定（2026-09-14，证据 `tests/deltap_inner_thr/`）**：
     默认 **1e-3 保留**。三体系（212/213 PW + MgO LCAO）在 1e-3/1e-4/1e-5 下
     μ* 相对差 ≤ 0.34%、`E_tot` 差 ≤ 4.2e-7 eV ⇒ 门控是**纯成本旋钮**；但成本
     效应的**符号随体系翻转**（212 +4.8%→**−9.5%**→−7.1%、213 −46%/−47%/−45% 基本平、
     MgO **−75%**→−72%→−66% 收紧变差），故把它当**逐体系旋钮**：INNER 打不过
     `outer` 时可试 1e-4，但必须实测确认。仍建议 `inner_thr ≫ scf_thr`
     （否则门控在 SCF 收敛时形同关闭）；
   - ✅ **"何时用哪个"决策表（2026-09-14 实测回填）**：

| 场景 | 推荐 | 实测依据（SCF 迭代数，证据 `tests/deltap_dual_iteration/`） |
|---|---|---|
| OUTER 外步 **≲7** 即收敛（易体系、小扰动） | **`outer`（默认）** | H₂O 电荷 +14%（63→72，且 settle 反弹 2 次后降级）、H₂O 自旋 +4.8%（42→44） |
| OUTER 外步 **≳15**（多约束/大扰动/体相） | **`inner`** | H₂O 混合 **−46%**（117→63）、MgO 体相电荷 **−75%**（381→94） |
| 任何 `inner` 使用 | **保留 `mix_reset`**（默认即开） | 关掉复位：211 迭代 72→161、213 用满 `scf_nmax` 仍不收敛、MgO 内更新 145 次后 settle 连败降级 ⇒ mixing 历史腐败是真实代价 |
| 任何 `inner` 使用 | **保留 settle 检查**（默认即开） | 实测抓到 **3 次**"内环宣告 CONVERGED、密度一松弛靶点即破"（否则直接误报 CONVERGED） |
| `inner` 的门控取值 | 默认 **1e-3**；INNER 打不过 `outer` 时试 **1e-4** | 门控是纯成本旋钮（μ* 差 ≤0.34%、`E_tot` 差 ≤4.2e-7 eV），但方向体系相关：212 +4.8%→−9.5%、213 平、MgO −75%→−72%（收紧变差）⇒ 逐体系实测，不可盲调（证据 `tests/deltap_inner_thr/`） |

   - **正确性等价（实测 4 体系全部命中）**：两策略收敛后的 μ* 相对差 ≤ 0.17%
     （判据 <1%）、`E_tot` 差 ≤ 8.5e-7 eV（判据 <1e-6 eV）；即策略只改**代价**，
     不改**答案**。
   - 一句话经验：**外步越少越该用 `outer`；外步很多时 `inner` 把"每外步一次 SCF
     重收敛"省掉，收益 = 省掉的收敛次数**。
   - ⚠️ MgO 体相长跑目前会在收尾阶段触发**预存在堆破坏 C-29**（OUTER/INNER 都中，
     父提交二进制复现），崩溃前的 μ*/`E_tot`/审计行仍可信；未修前请用 `timeout`
     包裹并只信已落盘输出。

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
双迭代调度：`docs/superpowers/plans/2026-09-11-dual-iteration-strategy.md`（设计）、
`docs/superpowers/specs/2026-09-14-dual-iteration-inner-schedule.md`（Q0 落地与
逐位回归证据；Q1–Q3 对照研究待续）。
