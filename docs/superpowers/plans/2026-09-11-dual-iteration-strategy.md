# 双迭代收敛策略（λ 外环-SCF 内环 vs SCF 外环-λ 内环）实施与定量对比计划

> 背景：用户要求统一约束框架同时支持两种 μ 更新调度，并做定量对比研究。
> 用户假设：内环 λ（DeltaSpin 式）可能改善"SCF 本身不易收敛"的场景；同时承认内环 λ 可能造成 charge–λ 耦合破坏 mixing（DIIS/Broyden）的数学前提、引入新的收敛困难。两种机制都必须在设计中被显式测量，而不是只比快慢。

---

## 0. 两种策略的精确定义（调度图）

**策略 OUTER（现状，λ 外环-SCF 内环）**：
```
repeat:  SCF 完整收敛 → 读 Q → M4.step 更新 μ（每外步一次）
until:   |Q−t|<thr 且 SCF 收敛
```
每个外步付出一次完整 SCF 收敛的代价；μ 更新总是作用在收敛密度上（响应读数干净）。

**策略 INNER（新增，SCF 外环-λ 内环，DeltaSpin lambda_loop 式）**：
```
SCF 迭代中：每迭代算 drho；
  若 drho < inner_thr（门控）→ 读 Q → M4.step 更新 μ（可能每迭代一次）→
    mix_reset（清 mixing 历史，DeltaP 先例 +5 行）→ 继续 SCF
收敛条件：同一迭代同时满足 drho < scf_thr 且 |Q−t| < thr
再加 Settle check（见 §3.4，防"内环收敛而弛豫反弹"陷阱）
```

## 1. 历史证据与本案的差异（设计前提）

| 证据 | 内容 | 对本案的含义 |
|---|---|---|
| DeltaSpin lambda_loop（生产） | nspin=2 磁约束内环 BFGS/PR-CG 生产可用，门控 sc_scf_thr_mode=threshold/immediate | **内环 λ 对本约束家族（局域势×密度观测量）有直接成功先例**——用户假设有据 |
| DeltaP 内环失败（T-7″，F-4 封口） | 冻密度响应符号反转、结构性不收敛 | 那是 **γ（Wilson 相位）观测量特有**（响应非单调）；电荷/自旋的冻密度响应是独立粒子磁化率 χ₀（负、单调），不适用该否决 |
| "内环收敛而外环弛豫反弹"（2026-07-20） | 冻密度优化 μ 后，电荷一弛豫约束即破 | **必须加 Settle check**：INNER 宣告收敛后，固定 μ 跑到 drho<scf_thr 再验 \|Q−t\| |
| mixing 破坏（用户担心点） | μ 在 SCF 中途变化 ⇒ Broyden/DIIS 的不动点映射 G(ρ) 改变，历史缓存失效 | DeltaP 已验证的对策：**μ 更新后 mix_reset()**；本设计把"更新后 mixing 行为"列为**测量对象**而非假设 |
| 两阶段门控（deltaspin/DeltaP） | drho 大时不碰 μ，近收敛才更新 | 沿用为 inner_thr 门控（默认 1e-3，可 INPUT） |

## 2. 实施设计（TDD，改动面集中在 M8/M4/M7）

### 2.1 输入与守卫（M7）

- 新增 INPUT：`constraint_mu_schedule = outer|inner`（默认 outer——现状逐位不变）；
  `constraint_inner_thr = 1e-3`（内环门控）；`constraint_inner_nmax = 20`（每次 SCF 内最多内环步数，防内环不收敛挂死）；
- 守卫：inner 模式 + `constraint_inner_nmax≤0` → ERROR；inner + 旧 v1 文件 → 允许（schedule 与 schema 无关）。

### 2.2 状态机扩展（M8 ConstraintLoop）

CONSTRAINED 相内新增 inner 分支：
```
on_iteration(iter, drho, rho):   # 新增钩子（esolver 每 SCF 迭代末调用）
    if schedule==inner and phase==CONSTRAINED and drho < inner_thr:
        observe(); st = mu_solver.step(Q, targets, mu)
        if st==CONVERGED: armed_settle = true   # 进入 settle 检查
        else: inject()  # 下一迭代生效
        request_mix_reset()
```
收敛裁决改为：`(drho < scf_thr) and (|Q−t| < thr) and settle_passed`。

### 2.3 mixing 复位（esolver 侧）

`request_mix_reset()` 在 PW/LCAO 两路径调各自的 mixing 重置（DeltaP 先例：`mix_reset()` 清 Broyden 历史）。**同时记录日志行**（`MIX_RESET at iter N (mu update)`）——这是 §4 测量 mixing 破坏效应的数据源。

### 2.4 Settle check（防历史陷阱的硬设计）

内环宣告收敛 → 冻结 μ → 继续 SCF 至 drho<scf_thr → 重读 Q：
- `|Q−t| < thr` → 真 CONVERGED；
- 否则 → 回到内环继续（计一次 settle_fail；settle_fail≥2 → 降级建议/熔断报告）。

### 2.5 单测（TDD，全部 mock 密度序列，不跑真 SCF）

| 测试 | 断言 |
|---|---|
| `InnerScheduleGating` | drho>inner_thr 时不更新 μ；跨过门控后每迭代更新；inner_nmax 用尽即停并报告 |
| `InnerMixResetOnUpdate` | 每次 μ 更新恰触发一次 mix_reset 请求；不更新不触发 |
| `InnerSettleCheck` | settle 通过→CONVERGED；settle 反弹→回内环；settle_fail×2→报告 |
| `InnerAntiFakeConvergence` | μ=0 自由跑不得收敛到非自然靶点（T4a 纪律，inner 模式同考） |
| `OuterLegacyBitIdentical` | schedule=outer 默认行为与现版本逐位一致（回归） |
| `InnerGuards` | 非法参数组合 ERROR；sabotage（门控置 0）恰中 FAIL |

## 3. 定量对比研究设计

### 3.1 对照矩阵

固定：同一二进制、同一网格、同一靶点、同一 scf_thr、同一初猜。

| 体系 | 通道 | SCF 难度 | 目的 |
|---|---|---|---|
| H₂O charge（211） | charge | 易 | 基线对照（inner 不应显著更差） |
| H₂O spin（212） | spin | 易 | 基线 |
| H₂O 混合（213） | charge+spin | 易-中 | 多约束内环行为 |
| MgO charge δ=+0.5 | charge | 中 | bulk 对照 |
| **FeO spin（低解锚，δ=+0.1）** | spin | **难（双稳地貌）** | 用户假设的主考场：inner 是否更稳/更快 |
| Mg 远侧 δ=−1.5（Q 硬化区） | charge | **难** | 极端 μ 区（注意与 C-29/不收敛熔断联动，mpirun 必裹 timeout） |

### 3.2 度量（逐 run 落盘）

| 指标 | 定义 | 解读 |
|---|---|---|
| **总对角化次数** | Σ diag_once 调用（含 settle） | 真实成本主指标 |
| 墙钟/SCF 迭代数 | total + 每外步分解 | 辅指标 |
| μ*、res、E_tot | 两策略终态 | **正确性等价性**：两策略收敛后 μ* 差 <1%、E_tot 差 <1e-6 eV |
| mix_reset 次数与"复位代价" | 复位后到 drho<scf_thr 的迭代数 | **mixing 破坏的定量证据**（用户担心点的直接测量） |
| 失败事件 | 不收敛/settle_fail/熔断/挂死 | 健壮性对比 |
| Q(μ) 轨迹 | 全轨迹落盘 | 振荡/极限环可视化证据 |

### 3.3 判决性问题（研究要回答的）

1. 在 SCF 易收敛体系上，inner 是否至少不劣于 outer（总对角化次数 ±20% 内）？
2. 在 SCF 难收敛体系（FeO、Mg 远侧）上，inner 是否显著更优（次数或成功率）？
3. mixing 破坏是否可测：复位代价 vs 不复位对照（若复位代价 ≈ 新启动 SCF，则 mixing 破坏不是主因；若复位后加速，则历史缓存腐败是真实机制）；
4. settle 反弹率：内环宣告收敛中有多少比例被弛豫反弹（若高 → inner 的收敛声明信用低，需收紧 inner_thr）。

### 3.4 执行步骤

| 步 | 内容 | 判据 |
|---|---|---|
| Q0 | 单测全绿 + legacy 逐位回归（§2.5） | ctest 11/11 |
| Q1 | H₂O 三用例双策略对照 | 正确性等价（μ* 1%、E 1e-6）；记录成本差 |
| Q2 | MgO + FeO 双策略对照 | 主判决数据（§3.3-1/2/4） |
| Q3 | mixing 机制专项（FeO：inner+reset vs inner 无 reset） | §3.3-3 证据 |
| Q4 | 汇总 spec：何时用哪个策略的**决策表**入用户手册 §5 | 用户可见结论 |

## 4. 风险登记

| 风险 | 应对 |
|---|---|
| inner 模式引入新的不收敛形态（mixing 腐败） | mix_reset 默认开；Q3 专项测量；留 outer 回退（默认就是 outer） |
| settle 反弹率高导致 inner 名不副实 | settle_fail≥2 降级；inner_thr 收紧；如实报告 |
| FeO 双稳地貌下两策略都失败 | 与分支守卫（BRANCH_FLIP）联动记录，不算策略失败算体系性质 |
| C-29/SCF 不收敛路径 | 所有远侧扫描 mpirun 裹 timeout；"SCF 不收敛→熔断"缺口（已登记）在 inner 模式同样适用，若先落地则两策略共享 |
| 与阶段 B（Broyden）的混淆 | 本任务只改**调度**不改求解器（M4 仍是逐分量 secant）；Broyden 是独立后续 |

## 5. 工期估计

实现（§2）1.5–2 天（含单测）；对照研究（§3）1–2 天（含机时，H₂O 便宜、FeO/Mg 远侧贵）。合计 ~3–4 天。
