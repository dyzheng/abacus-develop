# 实空间权重约束（module_constraint）开发者文档

> 版本基线：feat/deltap `fc93bac1e`（2026-09-09）。读者：后续开发者。
> 阶段 A（A0–A6）已完成：混合 charge+spin 数据流（M2/M3a/M6/M8 逐约束通道）落地，
> 集成用例 213 + 三旧用例回归全绿，μ 耦合实测归档（见 `2026-09-09-stageA-progress-summary.md`）。
> 配套：用户文档 `docs/constraint_user_manual.md`；架构 `plan-architecture.md`；术语 `docs/superpowers/specs/deltap-constraint-glossary.md`。

---

## 1. 模块地图（M 编号 → 文件 → 职责）

| 层 | 模块 | 文件（均在 `source/source_estate/module_constraint/`，除注明） | 职责 |
|---|---|---|---|
| L0 | M0 | `source/source_base/module_grid/partition.{h,cpp}` | Becke 权重数学核：`w_becke_adjusted`（含异核 χ_ij）、`w_becke_adjusted_deriv`（解析位置导数） |
| L1 | M1 | `weight_grid.{h,cpp}` | 网格权重场 `WeightGrid`：构造/缓存/单位分解审计/导数网格 |
| L1 | M2 | `constraint_observe.{h,cpp}` | 读数核 `ConstraintObserver::observe`（静态无状态；逐约束 channels 重载为 A2 唯一实现） |
| L2 | M3a | `constraint_inject_pw.{h,cpp}` | PW veff 注入核 `ConstraintInjectPW::inject`（静态无状态；逐约束 channels 重载为 A3 唯一实现） |
| L2 | M3b | `constraint_inject_lcao.{h,cpp}` | LCAO W^α HContainer（**运行时审计仪器**；生产注入走 M3a 同核） |
| L3 | M4 | `mu_solver.{h,cpp}` | 外环乘子求解器 `MuSolver`（逐分量 secant+护栏+熔断） |
| L3 | M5 | `constraint_accounting.{h,cpp}` | 记账 `ConstraintAccounting`：E_con + key=value 审计行（A4 起逐约束 kind= 标签） |
| L4 | M6 | `constraint_deriv.{h,cpp}` | 力核 `constraint_force`（静态自由函数，PW/LCAO 共用；A5 起 per-α channels 折叠为唯一实现） |
| L5 | M7 | `constraint_io.{h,cpp}` | 配置/JSON 解析/语义守卫 `configure_from_inputs`（A1：`ConstraintSpec` 数据模型 + v1/v2 双格式） |
| 编排 | M8 | `constraint_loop.{h,cpp}` | 外环状态机 `ConstraintLoop`（单例，唯一持有状态的类） |

**结构性事实：读/注/记/力四核均保持“单一实现 + 薄适配”**——逐约束 channels 版本是唯一实现，
旧 DensityChannel/单通道入口为 homogeneous-profile 薄适配（A2/A3/A5 后无第二路径残留）。

**结构性事实：本模块从不直接接触 `psi` 或哈密顿算符对象。** 它与 KS 循环的唯一耦合点是**有效势网格**（v_eff）与**密度网格**（rho）——这是与 DeltaSpin（投影算符进 H）/DeltaP（Wilson loop 进 ψ）的本质区别，也是双基组同口径的根源。

## 2. 数据格式

### 2.1 输入：INPUT 参数 → `ConstraintConfig`（M7，constraint_io.h:21）

| INPUT | 字段 | 类型/约束 |
|---|---|---|
| constraint | `enabled` | bool |
| constraint_type | `type` | "charge"\|"spin"（其余 ERROR） |
| constraint_weight_type | `weight_type` | 仅 "becke" |
| constraint_target_mode | `target_mode` | "delta"（默认）\|"absolute"（WARNING） |
| constraint_mu_max | `mu_max` | Ry，>0 |
| constraint_thr | `thr` | e，>0 |
| constraint_target_file | （文件内容） | 见 2.2（v1 旧格式 deprecated / v2 新格式）；空 → ERROR |

### 2.2 靶点 JSON → 数据模型（M7，constraint_io.h）

**v2（新，推荐）**：逐约束对象列表 → `std::vector<ConstraintSpec>`（A1，生产入口权威）：

```json
{"constraints": [
  {"type": "charge", "target": 0.1, "atoms": [0]},
  {"type": "spin",   "target": 0.1, "atoms": [0], "mu_max": 3.0}]}
```

`ConstraintSpec{kind, atoms, chan, target, mu_max}`（constraint_io.h:114）：
`kind ∈ {Charge, Spin}`；`chan = build_channel_profile(kind)`（**工厂唯一构建**，消费者禁手填）；
`atoms` 为扁平全局原子片段（缺省 = [列表序号]）；`mu_max` 逐约束覆盖 run 级（缺省 = run 级）。
守卫：`type` 非 {charge,spin}（含 dipole）→ ERROR；spin 且 nspin≠2 → ERROR；空/越界/嵌套
atoms → ERROR；混用 v1+v2 键 → ERROR；重复 (kind,atoms) → WARNING（近共线预警）。

**v1（旧，deprecated）** → 按 run 级 `constraint_type` 转换（整表单通道）：

```json
{"targets": [0.1, -0.1], "atoms": [[0], [1, 2]]}
```
`targets[i].value`（delta 偏移）+ `targets[i].atoms`（可嵌套片段；缺省逐原子）。
自动转换继续跑并打印 deprecation WARNING；三个注册旧用例（211/212/212_NAO）零修改逐位复现。

解析器是**手写 JSON 子集**（非通用 JSON）；v1/v2 分派由 `detect_format` 完成。

### 2.3 权重网格内存布局（M1，weight_grid.h）

| 数据 | 布局 | 说明 |
|---|---|---|
| `w_[iat][ir]` | 原子 × 局域网格点 | 逐原子 Becke 权重 |
| `cw_[alpha][ir]` | 约束 × 局域网格点 | 片段 = 原子权重求和（`set_constraint_atoms` 派生） |
| `dw_[alpha][(J*3+d)*nrxx + ir]` | 约束 ×（原子×3）× 网格点 | ∂w_α/∂R_J^d，M6 力核输入 |
| `maxdev_` | 标量 | 单位分解审计 max_g\|Σw−1\|（MPI 归约） |

网格 = `rho_basis`（PW 密度网格，**rank 局域域分解**；LCAO 用 pw_rhod 密集网格）。所有跨 rank 求和走 `Parallel_Reduce::reduce_pool`。

### 2.4 输出：审计行（M5，机器可读 key=value）

```
CONSTRAINT_AUDIT nconstraint=2 e_con=.. max_residual=.. total_charge=.. nelec=8 maxdev=2.2e-16
CONSTRAINT_AUDIT c[0] kind=charge q=.. t=.. mu=.. res=..
CONSTRAINT_AUDIT c[1] kind=spin q=.. t=.. mu=.. res=..
```
逐约束 detail 行的 `kind=` 标签（M5，A4）：仅当调用方传入逐约束 kinds 时出现
（legacy 5 参 audit 空 kinds → 历史输出逐位不变）。
`ConstraintAudit`（constraint_accounting.h:13）同时以结构体形式供 esolver 取 `e_con` 汇入 `fenergy::cc_escon` → `calculate_etot()`。

> **run 级 `constraint_type` 的语义（A1 起）**：只对 v1 靶文件（整表单通道）生效；v2 文件逐约束
> `type` 取代 run 级值（run 级 ≠ charge 时打 WARNING，不静默忽略）。混合 run 由 v2 列表表达。

## 3. 数据流：一步 SCF 内 ψ/Hamilton 与约束模块的完整链路

```
                 ┌─────────────────────────── 每 SCF 迭代 ───────────────────────────┐
                 │                                                                   │
 v_eff(ρ_in) ──► │ inject_potential / _lcao (M8→M3a)   v_eff += Σ_α μ_α·w_α          │
 （HSolver 前）  │      │                                                          │
                 │      ▼                                                          │
                 │ H[v_eff]：PW 直接用网格势；LCAO 经 Veff::contributeHR              │
                 │          （cal_gint_vl 网格积分）→ H(R) → H(k)                    │
                 │      ▼                                                          │
                 │ HSolver 对角化 H ψ = ε (S) ψ  →  ψ_n                              │
                 │      ▼                                                          │
                 │ elecstate：ψ → ρ_out（charge.rho，同一 pw_rhod 网格）              │
                 │      ▼                                                          │
                 │ observe (M8→M2)：Q_α = Σ_g w_α·d_α·ΔV（reduce_pool）              │
                 │      ▼                                                          │
                 │ iter_finish：cc_escon = Σμ(Q−t) 汇入 etot；                       │
                 │   SCF 收敛? → on_scf_converged → M4.step 更新 μ → 继续/结束        │
                 └───────────────────────────────────────────────────────────────────┘

 外环收敛后（SCF 结束）：
   compute_force (M8→M6)：F_J = −Σ_α μ_α Σ_g d_α·∂w_α/∂R_J·ΔV → Forces / FORCE_STRESS
   M3b 运行时审计：Tr[W^α·DM] vs ∫w_α·ρ（LCAO，done() 时一次）
```

**关键数据对应关系**：
- **ψ 从不进约束模块**。ψ 的唯一角色是经由 elecstate 产生 ρ；约束读数在 ρ 上做（这正是 PW≡LCAO 同口径的原因——两条基组路径在 ρ 网格处汇合）。
- **Hamiltonian 从不被约束模块修改**。LCAO 侧 H 含 Σμ_αW^α 是**隐式**的：μw 在 v_eff 网格里，生产 `cal_gint_vl` 的积分线性性保证 H += Σμ_α∫φ_μ w_α φ_ν（Task 2.3 设计决策，spec: 2026-08-31-lcao-esolver-wiring §4）。
- **μ 的生命周期**：M4 每外步更新一次；注入只读当前 μ_（constraint_loop.h:133）。
- **逐约束数据流（A4 起）**：`ConstraintLoop::init` 从 specs 派生平行数组
  `kinds_/channels_/mu_caps_`（与权重网格约束顺序一一对应）；observe/inject/
  audit/add_back/force 全部消费这组平行数组——混合 charge+spin 列表无第二路径。
  外环收敛后 `compute_force` 以**单趟 per-α 调用**（A5）完成混合力；A4 的两趟掩码
  组合已收回。

### 3.1 生命周期陷阱（已两次踩中，永久检查项）

μw 注入 v_eff 后，**势对象会在 SCF 生命周期末端被无 μw 重建**：
1. PW：`Potential::get_vnew()` 快照 vnew = v_phys(out) − [v_phys(in)+μw] → SCC 力污染（修复：`forces_scc.cpp` 修正副本回加 μw）；
2. LCAO：收敛轮 `update_from_charge` 重建 v_eff → `cal_pulay_fs` 的 μw Pulay 缺失（修复：`FORCE_STRESS.cpp` 的 `ConstraintPulayPotGuard` RAII）。
**任何新的势注入点（应力、偶极、未来的新通道）必须回答："力求值/能量决算时刻，注入的 μw 还在势里吗？"** 修复统一用 `ConstraintLoop::add_back_constraint_potential()` 的**修正副本**模式，禁止改写共享势。

## 4. 关键函数 ↔ 公式 ↔ 操作 ↔ 单测覆盖

| 函数 | 公式/操作 | 单测（目标::用例） |
|---|---|---|
| `Grid::Partition::w_becke_adjusted` | s(μ)=½[1−f_3(μ)]，P_i=∏s，w_i=P_i/ΣP；χ_ij 异核修正 | test_partition：解析对拍/单位分解/对称性 |
| `…::w_becke_adjusted_deriv` | ∂w/∂R 链式（f_3′ 多项式闭式 × 几何导数） | test_partition：FD <1e-6 |
| `WeightGrid::build` | 逐点调 M0 核 + 近邻筛选 + maxdev 审计 | weight_grid::PartitionOfUnity / MPI ×2 |
| `WeightGrid::build_derivatives` | 逐点调 M0 导数核组装 dw_ | DerivGridAnalytic / **TranslationInvariance** / MPI |
| `ConstraintObserver::observe` | Q_α=Σ_g w_α·d_α·ΔV + reduce_pool | observe::AtomicSuperposition / IndependentReferenceBecke / 偶极对拍 |
| `ConstraintInjectPW::inject` | v_eff += Σμw（charge 全自旋 / spin ±） | inject_pw::PointwiseInjection / **ObservableEqualsInjectionOperator** / SplitInjectionSpin |
| `MuSolver::step` | κ_i=sgn·clamp(\|ΔQ/Δμ\|)，Δμ=−(Q−t)/κ | mu_solver 7 用例（含翻号/熔断/反假收敛） |
| `ConstraintAccounting::audit` | E_con=Σμ(Q−t) + 审计行 | accounting 4 用例 |
| `constraint_force` | F_J=−Σ_α μ_α Σ_g d_α·∂w_α/∂R_J ΔV（双基组同核；per-α d_α=read_up·ρ↑+read_dn·ρ↓，A5） | deriv::ForceOnSyntheticDensity / NewtonThirdLaw / ForceLinearInMu / **MixedChannelForce / MixedForceNewtonThirdLaw** |
| `ConstraintLoop::on_scf_converged` | 两阶段门控状态机 + conv_esolver 门控 | loop::IgnoresUnconvergedScf / InjectMatchesObserver / SpinChannelConvergesOnLinearResponse |
| `configure_from_inputs` | 全部语义守卫（PW/LCAO 共享；v1/v2 解析+逐约束 specs 出参） | io::Guards / ConfigureFromInputsShared / **MixedConstraintListParsing / MixedGuards** |
| `ConstraintInjectLCAO::build/trace` | W^α_μν Gint 积分；Tr[W·DM] 审计 | inject_lcao::PartitionSumRuleEqualsOverlap（ΣW≡S，2e-15） |

## 5. 单测覆盖总账（11 个注册 ctest 目标全绿，2026-09-09 实测）与未覆盖点

已覆盖：全部数学核、守卫分支、双通道、MPI 一致性、记账、力核、**混合 charge+spin 列表的读/注/编排/力**。
**反向破坏验证**：每轮 sabotage 恰中对应新测试（A2/A3/A4/A5/A6 共 5 轮，含 staging guard 恢复、
通道误读、cap 退化、混合列表 spin 误读 charge 等判别），守卫非摆设。
**未覆盖/弱覆盖**（后续开发注意）：
1. `response_sign=+1` 路径——参数保留但无正响应通道实例，测试只验证它在 spin 上 FAIL（sabotage）；
2. `ConstraintInjectLCAO` 的 k 点（complex）变体——生产未接线、测试仅 gamma 实数；
3. nspin=4（非共线）——显式不支持（守卫拒绝），无测试需求；
4. `fixed_mu_` 实验开关——无单测（属诊断工具，默认关）；
5. FD 级力/力矩验收不在单测（在 `tests/constraint_fd_force/tools/` 脚本 + 集成用例）；
6. **混合力的 stationary4 FD 未跑**（按计划留 A6 后验收轮，不抢跑）：阶段 A 的力能力
   声明基于单通道 18 轴 FD + 混合单元级自洽/牛三（G5），包络限定 H₂O 类小分子、
   |μ|≲0.5 Ry、残差<1e-4。

## 6. 设计结构评估（是否得当 / 歧义 / 兼容风险）

### 6.1 设计得当之处
- **状态集中**：只有 `ConstraintLoop` 持有状态（单例）；M2/M3a/M5/M6 全是静态无状态核——可测试性极好（37+ 单测全部内存合成、无 IO 依赖）；
- **观测量==注入算符**：由"注入与读数共享同一 WeightGrid 实例"**构造性**保证，不靠纪律（constraint_loop.h:44 注释）；
- **守卫前置且双基组共享**：`configure_from_inputs` 单点守卫，PW/LCAO 不会跑出口径分歧；
- **通道契约集中**（A1）：`ConstraintSpec.kind` + `build_channel_profile` 工厂（±1 读/注符号）是
  观察器/注入器/力核三端唯一共享契约；旧 DensityChannel 入口为薄适配，无字符串散落；
- **RAII 修正副本**处理势生命周期（Pulay 守卫），不污染共享状态。

### 6.2 歧义与兼容风险（按严重度）

| 风险 | 级别 | 说明与缓解 |
|---|---|---|
| **时序契约隐式**：inject 必须在 HSolver 前、observe 在 iter_finish、v_eff 重建时机会抹掉 μw（§3.1）。esolver 重构（ABACUS 正在进行的 esolver 重构线）最易破坏此契约，且**无静态检查** | 高 | 缓解：新增注入点时强制走 §3.1 检查；在 esolver 钩子处保留集成冒烟（211/212/213/212_NAO 用例）作为回归网；考虑在 inject 处加"本迭代 v_eff 已被重建？"的断言式守卫 |
| **ConstraintLoop 单例全局态**：relax/MD 多几何步、或未来多实例（如嵌套循环）场景，单例是隐患（deltaspin 同款先例，已有 reset() 纪律） | 中 | 缓解：每几何 init() 重置全部状态（已实现）；MD 接线时必须复核 reset 语义 |
| **v1/v2 双格式并存（v1 deprecated）**：旧靶文件仍自动转换运行，但语义（整表单通道）弱于 v2；长期双格式是维护面 | 低 | v1 打 deprecation WARNING；三旧用例（211/212/212_NAO）锁兼容；新功能只进 v2 |
| **网格布局假设**：权重建在 pw_rhod（LCAO dense grid），由 double_grid 守卫保护；若未来 LCAO 网格策略变化（single grid 解禁等）会**静默错** | 中 | 缓解：守卫列表即兼容性清单——解禁任何守卫前必须重跑 V1+力 FD |
| **`fixed_mu_` 环境变量开关**：隐式接口（无 INPUT 参数、无单测），生产头文件可见 | 低 | 定位是诊断工具；若长期保留建议转为正式 INPUT 或移入测试工具 |
| **手写 JSON 子集解析器**：靶文件格式演进（加 types 字段等）需同步改解析器；无第三方依赖是有意决策（M7 spec 在案） | 低 | 格式变更须同步更新用户手册 §3 与 schema 测试 |
| **`response_sign` 公开参数**：+1 路径无真实通道、无正向测试；误设 +1 会驱动发散（有熔断兜底） | 低 | 保持默认 −1；若新通道确需 +1，先补正响应 mock 测试 |

### 6.3 扩展点指南（后续开发对照）

| 要加什么 | 动哪里 | 必过检查 |
|---|---|---|
| 新约束类型（偶极等） | ConstraintKind/Profile 工厂 + M7 守卫 + 逐约束 channels（阶段 A 模板已就绪） | mock 收敛 + 反假收敛 + §3.1 势存活检查；dipole 当前被守卫拒绝 |
| 新权重（Hirshfeld） | WeightType 枚举 + M0 核 + M1 构造 | 单位分解 + 导数 FD + 平移不变性 |
| 应力 | M6 新核（推导先行，R6 在案） | 应力 FD + Maxwell 闭合 + 势存活检查 |
| Broyden 多约束 | M4（保留逐分量作 fallback） | mock 病态映射 + 近共线对（47 步案例）+ **H₂O 混合耦合实测（μ 偏移 −0.0048/−0.0092、外环 3→15 步，213 用例）作基准** |
| MD/relax 接线 | 每几何 init + build + build_derivatives | Γ-path 先例（DeltaP 4.3）+ 力 FD 全绿前提 |

## 7. 文档与验证资产索引

- 测试：11 个注册 ctest 目标（`module_constraint/test/`，2026-09-09 实测 11/11）+ 4 集成用例
  （tests/01_PW/211、212、213，tests/02_NAO_Gamma/212_NAO）+ FD 脚本（tests/constraint_fd_force/tools/）；
- spec 链：每模块 dated spec（docs/superpowers/specs/2026-08-3*、2026-09-0*）；
- 评审链：2026-08-26 五份方案评审、2026-08-31 进展总结、2026-09-08 力 FD 根因两案（SCC/Pulay）、
  2026-09-08/09 阶段 A 逐 Task 评审（A1–A6，含 2.6/2.7 判决门闭合）；
  汇总：`2026-09-09-stageA-progress-summary.md`。
