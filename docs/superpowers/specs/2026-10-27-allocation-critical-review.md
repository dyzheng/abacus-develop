# Unified allocation 与 DeltaSpin 的批判性对比评估

- 日期：2026-10-27
- 分支：`feat/unified-local-moment-allocation`
- 方法：`A_i = Q_i + Q_c W_i Q_c`
- 对比对象：native DeltaSpin、Becke 空间权重、PW allocation
- 关联文档：
  `docs/superpowers/specs/2026-10-21-unified-local-moment-allocation-design.md`、
  `docs/superpowers/specs/2026-10-27-fe2-licl-comparison.md`

## 0. 结论摘要

**求和一致性：通过。** 在当前目标集合与全原子分配算符下，
`Σ_i A_i = I` 是代数恒等式；网格单测的 `max_sum_error=3.109e-15`，
Fe2 的两个 spin 约束满足 `Σq=M_tot`（差 `1.9e-10 μB`），LiCl/Fe2 的
实空间总电子数审计也分别达到 `dsum_n=-2.19e-12` 和 `1.33e-11 e`。

**效率：基础优化后部分改善，但 Fe2 仍不合格。** 优化前 Fe2-bcc 同靶
`±3.8 μB` 下，allocation 需要 81 次电子迭代、84 s，native DeltaSpin 只需
13 次、4 s；总墙钟约慢 24 倍。S5.5 后墙钟为 59–63 s，S5.6 的 8-band
归约批处理和 workspace 复用进一步降到 56–61 s；单次 hPsi 由 `20.82` 降到
`13.13–13.83 ms`，约为 DeltaSpin 的 `1.86–1.96x`。但 Fe2 总墙钟仍约
`16–17x`，核心
问题已经从“每 band 常数”转为“85 对 13 次电子迭代 × 额外的 `V|psi>`
FFT”。LiCl 在两边同为 19 次电子迭代时，已由优化前的 `2.28x` 降到
`1.08x`，满足只读场景门槛。

**语义：当前不能与 DeltaSpin 当作同一可观测量。** 同靶 `±3.8 μB` 时，
allocation 的 `q=Tr[ρA_i]` 达到 `-3.80005/+3.80007 μB`，但传统
orbital-sum 只有约 `-2.927/+2.926 μB`；DeltaSpin 的 orbital-sum 则与
`±3.8001 μB` 目标一致。两者约束的是不同算符，`0.625 eV` 的总能差不能被
解释为“同一物理态的能量误差”。

**决策建议：有条件继续优化，不建议直接放弃；但立即停止功能扩张。**
当一个限时性能/语义收口任务处理；若达不到第 8 节的门槛，则放弃生产约束
路径，只保留只读 allocation 诊断。若项目目标仅为 Fe 的自旋约束，当前应
直接使用 DeltaSpin，而不是 allocation。

## 1. Test plan

1. 核对 `A_i` 的单位分解、正性和 `n_i/m_i` 的可加性，区分代数恒等式、
   单测和真实 PW 审计三层证据。
2. 用 Fe2-bcc 比较同靶 allocation 与 native DeltaSpin 的收敛步数、墙钟、
   hPsi 成本和最终读数。
3. 用 LiCl 固定同一密度，比较 Becke 与 allocation 的局域电荷，确认差异
   来自分片定义而不是 SCF 数值噪声。
4. 对慢的根因做分层拆解：外层迭代数、每次 hPsi 成本、算符重复计算、
   集合通信和缺失的快路径。
5. 给出继续优化/放弃的可执行判定门槛。

## 2. Test setup

完整输入见 `tests/constraint_s5_parity/{fe2_bcc_alloc,fe2_bcc_ds,licl_free,licl_alloc}`。

- 运行环境：4 个 MPI rank；OpenMP、OpenBLAS、MKL 均固定单线程。
- Fe2-bcc：2×2×2 k 网格，`kpar=2`，`ecutwfc=20 Ry`，`nspin=2`，
  `nbands=27`，`scf_thr=1e-8`，共同目标 `-3.8/+3.8 μB`。
- Fe2 allocation：`constraint_alloc_target_l=2`、`outer` 调度、
  `constraint_trust_region=1`、`constraint_thr=1e-4`。
- Fe2 DeltaSpin：`sc_mag_switch=1`、`sc_thr=1e-4`。
- LiCl：15 Å 盒，Γ 点，`ecutwfc=100 Ry`，`nelec=10`，
  `onsite_radius=6.0`；Becke 与 allocation 仅差 `constraint_alloc_switch`。

## 3. Results

### 3.1 Fe2-bcc

| 指标 | allocation | native DeltaSpin |
|---|---:|---:|
| 总能 / eV | -6370.6101496879373 | -6369.985258222084 |
| 约束读数 / μB | `-3.800054394 / +3.800065059` | `-3.8000818583 / +3.8000808496` |
| 约束残差 / μB | `6.51e-5` | `8.60e-5` |
| 电子迭代 | 81 | 13 |
| hPsi 调用次数 | 2723 | 289 |
| hPsi 总时间 / s | 56.68 | 2.04 |
| 单次 hPsi / ms | 20.82 | 7.06 |
| 墙钟 / s | 84.47 | 3.51 |
| 传统 orbital-sum 磁矩 / μB | `-2.9268 / +2.9257` | `-3.8001 / +3.8001` |

派生比值：

```text
总墙钟比           = 24.07x
电子迭代比         =  6.23x
hPsi 调用次数比    =  9.42x
hPsi 单次成本比    =  2.95x
hPsi 总时间比      = 27.78x
```

### 3.2 LiCl

| 指标 | Becke | allocation（S5.5 前） |
|---|---:|---:|
| 总能 / eV | -643.97565238003517 | -643.97565238003517 |
| 电子迭代 | 19 | 19 |
| 墙钟 / s | 169 | 385 |
| hPsi 总时间 / s | 99.23 | 291.78 |
| 单次 hPsi / ms | 122.7 | 360.7 |
| Li 局域电荷 / e | 3.095906027 | 2.947887372 |

LiCl 中 allocation 的关键分解：

```text
目标壳层占据 n_T          = 1.99987921 e
空间总电荷 n              = 3.09590603 e
空间互补电荷 n_c=n-n_T    = 1.09602682 e
allocation q              = 2.947887372 e
allocation 互补部分 q-n_T = 0.948008162 e
q 与空间分片的差          = -0.148018658 e
```

这个差值恰好等于 Becke 与 allocation 的 Li 电荷差，说明差异来自
“目标子空间先扣除、剩余部分再做空间分配”的定义，而不是 SCF 误差。

## 4. 求和一致性：满足，但审计仍不完整

### 4.1 代数层：满足

在 `Σ_i W_i = I`、`Q_c = I - P` 且所有原子都纳入分配时：

```text
Σ_i A_i = P + Q_c (Σ_i W_i) Q_c = P + Q_c = I
```

因此 `Σ_i n_i=N`、`Σ_i m_i=M_tot` 是构造性质，不依赖约束收敛“凑出来”。
这也意味着单原子/子集约束的 `q` 不应被误认为全空间分区；LiCl 只约束 Li，
所以 Li 的 `q=2.947887 e` 不是电子总数，也不应与 Becke Li 电荷直接等同。
更重要的是，求和一致只保证“分片加起来等于总量”，**不保证分片本身等于
DeltaSpin、Becke 或某个实验定义**。它是守恒性/可实现性的必要条件，不是
物理正确性的充分条件。

### 4.2 单测层：满足

- `MODULE_ESTATE_constraint_alloc_operator`：
  `max_sum_error=3.109e-15`，正性检查通过。
- `MODULE_ESTATE_constraint_alloc_observe`：
  `sum_n=4.489690342809e-02/4.489690342809e-02`，
  `sum_m=5.826800748409e-08/5.826800746066e-08`。

### 4.3 真实 PW 审计：部分满足

- Fe2 两个 spin 约束：`q0+q1=1.0665e-5`，`tot_mag=1.0665e-5`，
  差 `1.9e-10 μB`；同时 `sum_n=32`、`dsum_n=1.33e-11`。
- LiCl：`sum_n=10`、`dsum_n=-2.19e-12`。
- Fe2 allocation 的 `sum_m` 与 `tot_mag` 也一致到 `dsum_m=4.30e-15`。

**审计缺口：** `ALLOC-PW audit sum_n/sum_m` 主要对实空间权重分片求和，
而不是显式对 `Σ_i CONSTRAINT_AUDIT q_i` 求和。目前只有 Fe2 spin 这一例
直接验证了 `Σq=M_tot`；还没看到“所有原子都加 charge 约束”的 `Σq=N`
集成测试，也没有非共线、`bndpar>1`、不同 `target_l` 的 q 求和回归。
因此“求和性质成立”有充分依据，但“所有生产路径的 q 审计闭合”尚未完全证明。

## 5. 优点

1. **精确可加性强于普通投影**：不依赖约束残差；`ΣA_i=I` 是可以单测钉死的
   代数性质。对需要严格碎片电荷/自旋闭合的问题，这是实质优势。
2. **目标轨道与空间分片统一**：`(n_T,m_T,n_c,m_c)` 同时保留 d/f 目标壳层
   占据和外部空间分配，比纯 Becke 或纯 onsite 投影提供更多诊断信息。
3. **可插值**：空目标集合退化为 Becke，完备目标集合退化为正交投影；同一套
   公式可覆盖 charge、collinear spin，并已有 force/stress 路径。
4. **可解释的互补子空间**：`Q_c W_i Q_c` 明确说明“目标轨道之外还剩多少
   密度、分给谁”，对配体极化、空穴和电荷转移问题有潜在价值。
5. **正性**：`A_i≥0`，电荷分配不会出现负的局域电子数；这是相比某些
   差分投影的直接优势。

## 6. 问题与慢的根源

### 6.1 语义问题：q 不是 DeltaSpin 的磁矩

Fe2 是最直接的警告：同一 `±3.8 μB` 目标下，allocation 的约束观测量满足，
但传统 orbital-sum 只有约 `±2.93 μB`。说明 `q` 是一个“目标子空间 + 互补
空间”的新定义，而不是 DeltaSpin 的 onsite 原子轨道磁矩。若用户把它当作
DeltaSpin 的替代品，会得到错误解释；若把它当作新可观测量，则必须先定义
其物理意义、目标集依赖和与实验量的映射。

### 6.2 慢的第一层：约束外层需要反复重建电子态

Fe2 allocation 只有 8 个外层步，却消耗 81 次电子迭代；DeltaSpin 有 13 个
DS 外层步，但只消耗 13 次电子迭代。差别在于 native DeltaSpin 的内层 lambda
优化不需要每次重新收敛整个电子密度，而 allocation 的 `Q(mu)` 更新会改变
固定点映射，通常要再次跑若干 SCF 迭代。trust-region 的拒绝和半径收缩还会
进一步增加重启。

### 6.3 慢的第二层：每次 hPsi 本身更贵

Fe2 单次 hPsi 从 `7.06 ms` 升到 `20.82 ms`（`2.95x`）；LiCl 从
`122.7 ms` 升到 `360.7 ms`（同样约 `2.94x`）。allocation 算符对每个 band
仍要执行：局部势作用、目标投影/重叠、低秩分配代数和 pool reduction。
目标空间和 `V|φ>` 虽有缓存，但 per-band 的 `build_raw_overlap`、
`apply_local_potential`、`apply_pw_alloc_delta_h_orthogonal` 和集合通信没有
消失。Fe2 的 hPsi 总时间因此达到 DeltaSpin 的 `27.8x`。

两个放大因子近似相互独立：若调用次数仍为 `289`，仅单次成本就会产生
`289×20.82 ms≈6.0 s`，仍是 DeltaSpin 的 `2.95x`；若单次成本相同，
`2723×7.06 ms≈19.2 s`，仍是 `9.42x`。因此不能只靠“减少外层迭代”把
allocation 追平，必须同时降低**每 band、每约束**的算符成本。

### 6.4 慢的第三层：零乘子和只读场景没有快路径（S5.5 已修复）

LiCl 的 `mu=0`、目标为 `delta=0`，理论上 allocation 修正恒为零。优化前
`AllocConstraintPW::act()` 没有零乘子早退，仍然构造势场、目标乘积并进入
per-band 分配代数；同样 19 次电子迭代、809 次 hPsi 下比 Becke 多花
`216 s`。S5.5 在构造势场前检查 `mu_eff()`，该额外墙钟已压到 `14 s`
（`183` 对 `169 s`），说明这部分确实是纯实现浪费。

### 6.5 慢的第四层：集合通信和逐 band trace（S5.5/S5.6 已批处理）

优化前观测路径在 `alpha × band × k` 上逐项做 `reduce_pool(trace)`；S5.5
已改为每个 band 一次 alpha 合并归约。Hamiltonian 路径的目标乘积
`m_raw`、`vphi` 和 R×R 矩阵已合并成一次归约。S5.6 再把每 band 的
`<phi|psi>` 与 `<Vphi|psi>` 合并成每 8-band block 一次打包归约，并复用
FFT/向量 workspace；每 band 的四个 `nbasis` 临时向量也合并成一个
`delta`。剩余成本主要是每 band 必须做的一次额外 `V|psi>` FFT，以及
eigensolver 本身，而不再是集合通信或临时分配。目标秩自适应压缩仍未做。

### 6.6 鲁棒性和数值问题

- 普通 outer secant 在 Fe2 的陡峭 `Q(mu)` 分支上振荡；必须启用 trust-region。
- `±3.5/±3.9/±3.6` 目标附近出现平台、分支跳变或无法到达的区间，说明
  `Q(mu)` 不一定单调，外层求根不能只靠简单 secant。
- `S^{-1/2}` 正交化会让目标轨道泄漏到邻原子；当前 Fe2/LiCl 条件数尚可
  （`cond=1.455`、`1.072`），但没有系统的近线性依赖和半径敏感性扫描。
- 只支持 collinear `nspin=1/2`、CPU double；`nspin=4`、GPU、`bndpar>1`
  的力/应力路径仍未闭环。

## 7. 与 DeltaSpin/Becke 的定位

| 维度 | Becke/Hirshfeld | native DeltaSpin | unified allocation |
|---|---|---|---|
| 观测量 | 空间权重分片 | onsite 原子轨道磁矩 | `Tr[ρA_i]`，目标+互补 |
| 求和一致性 | 权重归一化时成立 | 通常不是显式全空间分区 | 代数上精确成立 |
| 轨道含义 | 弱 | 强 | 目标壳层强，互补部分依赖空间权重 |
| 每次 SCF 成本 | 低 | 中 | 高（当前约 3x hPsi） |
| 迭代效率 | 基准 | 高 | 低（当前 Fe2 约 6x 迭代） |
| 生产成熟度 | 高 | 高 | 原型，多个路径受限 |
| 适用场景 | 电荷/空间碎片 | 自旋约束 | 需要精确可加 + 目标/互补联合诊断 |

## 8. 继续优化还是放弃

### 8.1 不建议直接放弃的理由

求和一致性和正性确实成立；这不是一个“数学上错误、只能靠调参凑结果”的方案。
它提供了 DeltaSpin 和 Becke 都不具备的能力：目标壳层与互补空间同时可加、
同一算符覆盖 charge/spin、已有解析力/应力框架。对于电荷转移、配体空穴和
多片段约束，这个能力可能有独立价值。

### 8.2 但也不应无条件继续

如果项目只需要“约束 Fe 的磁矩”，当前答案很清楚：**用 DeltaSpin，不要用
allocation**。allocation 目前慢约 24 倍，且传统磁矩读数不等于 DeltaSpin；
继续扩功能会把一个尚未定义清楚的观测量推进到更多物理场景。

### 8.3 建议的止损门槛

只做一个限时优化冲刺，先不增加新物理功能；满足以下条件才进入生产化：

1. **零乘子快路径**：`lambda=0` 的只读/参考 SCF 不进入 allocation 算符，
   LiCl 这类场景相对 Becke 的额外墙钟降到 `<1.3x`。
2. **每次 hPsi 成本**：同体系相对 DeltaSpin/Becke 的单次成本降到 `<1.5x`；
   归约批量化，目标空间和局域势作用复用。
3. **总收敛成本**：Fe2 这类 multi-k spin 案例的总墙钟降到 DeltaSpin 的
   `<3x`，电子迭代数降到 `<2x`；否则说明外层求解器仍未解决。
4. **语义闭环**：明确 `q` 与 onsite orbital moment、Becke、实验量的关系；
   在文档中固定 `target_l`、目标集和半径口径，并做敏感性扫描。
5. **求和审计闭环**：加入全原子 charge 约束的 `Σq=N`、spin 的 `Σq=M_tot`、
   非共线、`kpar>1` 和不同 `target_l` 的集成测试，`dsum<1e-8`。
6. **鲁棒性**：在 Fe2/FeO 类多分支体系上，不依赖手工挑选目标；普通
   outer 或自动 trust 策略都能在预算内收敛，或明确报告不可达而不是静默振荡。

如果 2–3 周内只能改善代码常数、仍达不到第 2、3 条，建议停止 allocation
生产约束路径；保留只读 `ALLOC-PW` 作为诊断工具，约束主路径回到 DeltaSpin
或 Becke。若连第 4、5 条都无法定义清楚，则不是“优化问题”，应直接放弃
该方案作为通用约束框架。

## 9. 性能还有多少空间：分层判断

### 9.1 三个乘数，而不是一个“慢”字

Fe2 的总时间可以近似拆成：

```text
T ≈ N_iter × N_hpsi_per_iter × t_hpsi + T_observable + T_setup
```

当前 allocation 对 DeltaSpin 的 `24x` 墙钟，大致来自：

```text
电子迭代数        81 / 13       = 6.23x
每次迭代 hPsi 数  33.6 / 22.2   ≈ 1.51x
单次 hPsi 成本    20.82 / 7.06  = 2.95x
```

因此不存在单一开关能把 24x 直接降到 1x。至少需要同时处理
“外层迭代数”和“每 band 算子成本”两个乘数；只优化其中一个，收益会被
另一个乘数放大回去。

### 9.2 不反复重建电子态：方向正确，但 `inner/ALM` 当前不是即插即用

代码已经提供 `constraint_mu_schedule=inner` 和 `alm`。`inner` 在
`drho` 低于阈值后于 SCF 内更新 mu；`alm` 用罚函数反馈让密度和乘子一起
演化。它们理论上都能减少“每个 mu 都重新收敛到严格残差”的浪费。

这里必须区分两个“inner”：native DeltaSpin PW 的内循环是
`SpinConstrain::run_lambda_loop()`，在 `cal_mw_from_lambda()` 中保存每个 k
点的 \(nbands\times nbands\) 子空间 Hamiltonian/重叠矩阵，用
`diag_response()` 或 `diag_subspace_psi()` 旋转波函数；必要时再调用
`HSolverPW` 做全空间精修。allocation 的 `inner` 则是
`ConstraintLoop::take_inner_step()`，只按 `drho` 门控更新 mu，既不调用
`SpinConstrain` 的 lambda 循环，也不复用那套子空间矩阵。Fe2 allocation
输入当前仍是 `outer`，所以连 allocation 自己的内循环也没有启用。

因此前面的 `24x` 是“当前 allocation 路径”对“native DeltaSpin 路径”的
比较，不是 allocation 算法的最优下界；native 路径已经享受子空间旋转，
而 allocation 没有。复用 DeltaSpin 的子空间机制应单独列为高优先级候选，
不能用下面的 `inner/ALM` 负结果否定它。

用同一 Fe2 输入只切换调度（`scf_nmax=300`，4 MPI，单线程）得到：

| 调度 | 电子迭代 | hPsi 次数 | hPsi 时间 / s | 最终残差 / μB | 结果 |
|---|---:|---:|---:|---:|---|
| `outer` | 81 | 2723 | 56.68 | `6.51e-5` | 收敛，墙钟 84 s |
| `inner` | 198 | 5261 | 95.04 | `1.15e-3` | 未收敛；trust radius 降到 `1e-8`，27 次拒绝 |
| `alm` | 300 | 6820 | 151.44 | `6.54e-2` | 触及 `scf_nmax`，仍为 RUNNING，墙钟 223 s |

这组负结果说明：**只取消“每次 mu 都完整重建”的调度并不能自动加速**。
`inner` 在固定密度上更新 mu 后仍要重置 mixing，而 allocation 的 `Q(mu)`
分支陡峭、可能非单调；观测噪声会把 trust radius 推到下限。`alm` 的自动
`k` 被限制在 `0.1 Ry/μB` 的下界，罚项太弱，300 步仍未把残差压到阈值。

可行的方向不是简单“关掉外层”，而是把 \((\rho,\mu)\) 作为联合固定点求解：

- 用上一个 mu 的密度/波函数做 warm start，并用上一轮 \(d\rho/d\mu\) 外推
  初始密度，而不是把 mixing 历史全部丢弃；
- 在联合变量上做 Anderson/Broyden 或带线搜索的 trust-region，而不是只对
  mu 做一维 secant；
- ALM 的罚参数按约束残差和响应归一化，并保留最终一次严格无罚 SCF 作为
  验收；否则“未收敛的罚函数解”不能用于能量、力和应力。

这些改动仍以同一约束方程为终点，属于算法加速；不是把约束换成另一个
近似物理量。

### 9.3 哪些量可以预存，哪些不能

| 量 | 依赖 | 能否跨 SCF 复用 | 说明 |
|---|---|---|---|
| \(w_i(r)\), \(W_i\) | 几何、权重后端 | 可以 | 同一几何内固定；已是低成本项 |
| \(\phi_a\), \(S^{-1/2}\), \(Q_i\) | 几何、k、target_l | 可以 | 目标空间与 lambda 无关，现有 target cache 已覆盖 |
| \(W_i\phi_a\), \(\Phi^\dagger W_i\Phi\) | 几何、k、target_l | 可以，且应按约束预存 | 因 \(V=\sum\lambda_iW_i\) 对 lambda 线性，换 mu 时可重组，无需重做 FFT |
| \(V\phi_a\), \(M=\Phi^\dagger V\Phi\) | lambda | 可由上一行线性组装 | 当前 product cache 以 lambda 为键，换 mu 会重建；这是可消除的重复 |
| \(V\psi\), \(\Phi^\dagger\psi\), \(\Phi^\dagger V\psi\) | 波函数 \(\psi\) | 不能跨 band/迭代通用预存 | 只能在同一 \(\psi\) 的观测、力、应力阶段复用 |
| \(A_i\psi\) | \(\psi\) | 不能全局预存 | 每个新 trial vector 都必须重新作用低秩算符 |
| 完整 \(A_i\) 矩阵 | 几何、目标集 | 不应形成 | 它是“局域 + 秩 R”的算子；稠密矩阵会从 \(O(RN)\) 退化为 \(O(N^2)\) |

结论是：**能预存的是“几何/目标/权重侧”的小量，不能预存的是“对任意新
波函数的算子作用”。** 当前 target cache 已经做了前半部分；后半部分只能
通过批处理、BLAS-3、减少临时向量和更好的 eigensolver 来摊薄。

观测路径的独立浪费已在 S5.5 消除：`compute_pw_alloc_observables()` 现在
每个 k 只构造一次 `phi_orth`，每个 band 一次算出所有约束的 \(q_\alpha\)，
并把约束维归约合并为一次。改写时必须区分已池归约的 raw overlap 与
rank-local 的 \(Q_cW_iQ_c\)，否则会重复计算 \(Q_i\)。

### 9.4 “把投影修正当成固定量”到底能做什么

需要区分三种“固定”：

1. **固定 \(A_i\) 本身**：在固定几何、目标集和 k 点下，\(A_i\) 本来就不依赖
   \(\rho\) 或 lambda；这一步已经是精确的，并不会减少每 band 的
   \(A_i\psi\) 作用成本。
2. **固定 \(V=\sum\lambda_iW_i\)**：这会把不同 mu 的约束势混为一谈。V 对
   lambda 是线性的，正确做法是预存 \(W_i\phi_a\) 后快速重组，而不是把 V
   整体冻结。
3. **固定 \(A_i\psi\) 或整个 ΔH**：这是 lagged operator/准牛顿近似，不是
   原约束方程的精确解。它可以作为预热、预条件器或前几步的近似，但最终
   能量、力和应力必须用当前密度下的精确算符重新计算；否则收敛判据和
   Hellmann-Feynman 导数不再对应同一个泛函。

如果把 \(Q_cW_iQ_c\) 进一步替换成局域 \(W_i\)，方法就退化为 Becke 型
空间分片，目标轨道含义随之消失；这不是 allocation 的加速，而是换方法。

### 9.5 激进方案排序

| 方案 | 是否保持原方程 | 潜在收益 | 工作量/风险 | 建议 |
|---|---|---|---|---|
| 零 lambda / 只读快路径 | 是 | LiCl 的 `2.28x` 额外开销可接近消除 | 低 | 先做 |
| 按约束预存 \(W_i\phi\), \(\Phi^\dagger W_i\Phi\)，lambda 线性组装 | 是 | 减少每次 mu 变化的 FFT/R×R 重建 | 低-中 | 先做 |
| 多 band BLAS-3、workspace 复用、批量 trace | 是 | 降低每 hPsi 常数和集合通信 | 中 | 先做 |
| warm start + 密度外推 + 联合 Broyden/Anderson | 是（最终严格验收） | 可能把 81 步压到 30–50 步 | 中 | 第二阶段 |
| 改进 ALM/inner 的全局化、线搜索和罚参数 | 是 | 减少外层重启，接近 DeltaSpin 的迭代量级 | 中-高 | 试点 |
| 复用 DeltaSpin 的 \(nbands\times nbands\) 子空间旋转/对角化 | 近收敛时近似，需全空间复核 | 可显著减少每次 mu 变化的 hPsi | 中-高 | 优先研究 |
| 直接约束最小化 / 波函数层 PCG | 是 | 可能消除“外层 SCF + 内层 lambda”双层结构 | 高 | 研究型 |
| 低秩 Woodbury / 子空间 eigensolver | 是 | 把秩 R 修正并入 Davidson/子空间求解 | 高 | 研究型 |
| 局域 Wannier/实空间截断目标 | 否（近似） | 大体系可显著降低 O(RN) | 中 | 仅在物理口径可接受时 |
| GPU（同精度） | 是 | 硬件常数 2–5x | 中 | 与算法优化并行 |
| 混合精度投影代数 | 否（需误差验证） | 降低内存带宽和代数成本 | 中 | 先做数值验证 |
| 目标秩截断 / 自适应 active space | 否（近似） | 降低 R 和 hPsi 成本 | 中 | 需误差上界 |

其中前四项是“不改变物理定义”的工程优化，应优先于任何新功能。真正可能
把 allocation 拉进 DeltaSpin 同一数量级的，是联合固定点求解或波函数层
直接约束最小化；仅靠缓存和冻结，最多把 24x 降到大约 5–8x，仍不足以让
它成为默认自旋约束后端。

### 9.6 一个现实的性能目标

- 对 LiCl 这类 `mu=0`、只读或参考 SCF：目标是相对 Becke `<1.3x`，靠
  零 lambda 快路径和观测快路径即可做到。
- 对 Fe2 这类多 k 自旋约束：先把单次 hPsi 从 `2.95x` 降到约 `1.5–2x`，
  再把外层迭代从 81 降到 30–50；总墙钟有机会从 `24x` 降到约 `5–8x`。
- 要稳定达到 DeltaSpin 的 `<3x`，必须再解决联合收敛，不能只靠微优化。
  这是由 allocation 每 band 必须施加秩 R 非局域修正这一事实决定的；它
  不可能像纯局域 Becke 那样便宜，也不应把“和 Becke 同成本”作为目标。

### 9.7 S5.5/S5.6 实测更新

| 体系/指标 | 优化前 | S5.5 | S5.6 | 对照 |
|---|---:|---:|---:|---:|
| LiCl 墙钟 / s | 385 | 183 | 未重跑（同一快路径） | Becke 169 |
| LiCl hPsi 总时间 / s | 291.78 | 102.59 | 未重跑 | Becke 99.23 |
| LiCl 单次 hPsi / ms | 360.66 | 126.80 | 未重跑 | Becke 122.66 |
| Fe2 电子迭代 | 81 | 84 | 85 | DeltaSpin 13 |
| Fe2 hPsi 总时间 / s | 56.68 | 44.42–47.35 | 40.55–42.73 | DeltaSpin 2.04 |
| Fe2 单次 hPsi / ms | 20.82 | 15.19–16.19 | 13.13–13.83 | DeltaSpin 7.06 |
| Fe2 墙钟 / s | 84 | 59–63 | 56–61 | DeltaSpin 3.51 |

- LiCl 的零乘子早退、观测批量化已满足 `<1.3x` 门槛；数值与优化前逐位一致。
- Fe2 单次 hPsi 已从基线的 `2.95x` DeltaSpin 降到 `1.86–1.96x`，墙钟从
  `84 s` 降到 `56–61 s`。但 85 对 13 次电子迭代仍使总墙钟约为 `16–17x`；
  下一阶段必须
  处理 DeltaSpin 式子空间旋转与 warm start。
- S5.7 进一步删除 FFT 前的冗余清零/拷贝，并把观测改为每个 band 一次实空间
  变换后按约束权重积分。Fe2 为 91 次迭代、hPsi `42.47 s`、单次
  `12.71 ms`、墙钟 `56.45 s`；LiCl 为 hPsi `92.21 s`、墙钟 `160 s`，
  `q` 与总能逐位不变。详见
  `docs/superpowers/specs/2026-10-28-allocation-fft-buffer-trim.md`。

## 10. Next steps

1. 已完成：operator 的 raw overlap 与 `<Vphi|psi>` 按 8-band block 合并，
   观测侧和零 lambda 快路径也已完成。
2. 已完成 FFT/向量 workspace 复用；继续测量 R=5N/7N 和大 nbands 的
   block-size/内存曲线。
3. 复用 DeltaSpin 的 lambda 内层/子空间求解和 warm-start，避免每次 mu 变化
   都把整个密度重新收敛。
4. 定义 `q` 的物理口径和与 DeltaSpin/Becke 的映射，补全所有原子 charge
   求和的集成回归。
5. 零 lambda 快路径、势场缓存、观测/operator 批处理和 workspace 复用已
   完成；继续做按约束的 \(W_i\phi\) 预存，再决定是否投入联合固定点求解。
6. 以第 8.3 节门槛做一次正式 go/no-go 评审；未通过则按“只读保留、生产
   放弃”收口。

## 11. Verification

- `ctest -R MODULE_ESTATE_constraint_alloc`：10/10 PASS。
- `ctest -R constraint`：23/23 PASS。
- 本次补跑并记录：
  `alloc_operator max_sum_error=3.109e-15`；
  `alloc_observe` 的 `sum_n/sum_m` 一致。
- 本次额外在 `/tmp` 复制 Fe2 输入、只切换 `constraint_mu_schedule`：
  `inner` 在 198 步后未收敛并退出，`alm` 在 300 步后仍为 RUNNING；
  结果已列入第 9.2 节，仓库输入和代码未改动。
- S5.5 优化后：`ctest -R MODULE_ESTATE_constraint` 23/23 PASS；Fe2 数值
  与优化前一致到 `1e-4 μB`，LiCl 数值逐位一致。
- 详细前后数据与 bug/fix 见
  `docs/superpowers/specs/2026-10-27-allocation-performance-optimization.md`。
- S5.6：`ctest -R MODULE_ESTATE_constraint` 23/23 PASS；Fe2 `85` 次迭代、
  hPsi `40.55–42.73 s`、墙钟 `56–61 s`，最终
  `q=-3.799950937/+3.799961892 μB`。8-band 分块与不限内存批处理逐位一致。
  详见 `docs/superpowers/specs/2026-10-27-allocation-band-batching.md`。

## 12. 2026-10-28 update: warm start 与 A/B 结论

- 外层参数 A/B 证明 `Q(mu)` 的分支边界不是步长常数问题：`trust=0`、
  `step_max=0.005`、放宽 `inner_thr` 都失稳；`step_max=0.01` 只在特定
  Fe2 输入上避开磁支跳变，不能作为默认值。
- 已实现单次固定密度子空间旋转（`constraint_alloc_subspace_rotation=1`）：
  Fe2 从 91 次迭代/3342 次 hPsi/56 s 降到 82 次/1983 次/39–42 s，
  ETOT 差 `2.5e-8 eV`。这让“复用 DeltaSpin 子空间旋转”从设想变成有效
  快路径，但外层迭代仍为 DeltaSpin 的约 6 倍。
- 固定密度 mu 更新超过 1 次在 Fe2 上过冲并触发磁支翻转；默认预算为 1，
  >1 仅保留为实验开关。下一门槛是构造 \(A_i\) 子空间矩阵并做真正的受约束
  子空间最小化，而不是继续调外层步长。
- 详见 `docs/superpowers/specs/2026-10-28-allocation-subspace-warm-start.md`。

