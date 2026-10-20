# 电荷约束方法改进设计：偶极口径 + 高效求解

**日期**：2026-10-20
**分支**：feat/deltap（基线 commit `2ecee61`）
**依据**：性质1-4 测试报告（2026-10-15/19）
**状态**：设计已获用户逐节确认（brainstorming 流程），待实施计划

---

## 0. 需求与决策记录

用户两条核心诉求：

1. **电荷估计要有物理意义，和偶极矩对应**——性质1 已实证 Becke 片段电荷是记账数
   （Li 片段随分区半径在 2.50–3.26 e 间变化，展宽 0.77 e；LiCl 的 Becke 记账 +0.096 e
   vs 偶极口径电荷转移 0.71 e vs 实验反推 0.82 e），绝对片段电荷不能直接读成离子性。
2. **两层循环效率太低**——性质4 实测约束 MD 106 s/fs（2 原子体系！）、52.5 SCF/步、
   10.5 次外层 μ 更新/步，外推 10 ps ≈ 295 h / 1356 元，恒电势 MD 训练集生产不实用。

用户确认的决策（按顺序）：

| 决策点 | 选择 |
|---|---|
| 目标系综 | **两者都要**：定 Q（扫描/标定）+ 定 μ（巨正则生产 MD） |
| 偶极对应落法 | **分两步走**：偶极报告口径先行（低风险），偶极约束坐标第二阶段 |
| 效率攻势 | **保守路线 + 单层循环深水区**（全组合） |
| 偶极约束体系形态 | **周期界面优先**（saw 通道），分子通道随行 |
| 推进顺序 | **方案一**：A → C → D → B（风险递增，前期模块为后期铺路） |

## 1. 问题清单（四份报告的实证证据）

| # | 问题 | 证据 |
|---|---|---|
| P1 | Becke 片段电荷分区敏感，绝对值无物理意义 | 性质1 §5.3：同一密度换半径 Li 2.50–3.26 e；归一化/sum rule 与分区无关（机器精度） |
| P2 | 物理电荷转移的权威口径是偶极矩 | 性质1 §5.2：μ_dipole = 6.87 D（实验 7.96 D），q = 0.71 e（实验 0.82 e） |
| P3 | 外层循环可推翻磁分支致崩溃 | 性质2 §5.2：APNS LiNiO₂ 自旋三点 `psi_norm <= 0`，残差在 1e-5~1e-4 振荡 125 次未收敛，TMAG 3.00→4.66→−2.06 翻转 |
| P4 | 默认 step_max=0.05 Ry 太小 | 性质3 §2.3 批次 C：提到 0.5 Ry 后 6 点全部 270–850 s 收敛且解不变；默认下大 \|δ\| 点被墙钟截断 |
| P5 | 陡段局部外推失效 | 性质3 §6.4：δ=−0.8 外推低估 53%；但陡段外 [−0.9,−2.0] 严格线性（dμ/dδ=−2.339 Ry/e，外推偏差 ≤0.18%） |
| P6 | MD 每离子步重跑自由相+从零割线 | 性质4 §6.4：`init()` 每步重置全部状态（esolver_ks_pw.cpp:210）；52.5 SCF/步、10.5 次 μ 更新 |
| P7 | inner schedule 在 MD 中有害 | 性质4：74 vs 52 SCF/步、μ 欠收敛 1.7%，已弃用 |
| P8 | UNREACHABLE 语义易误读为物理极限 | 性质3 §6.2：探针 C 实测是 `mu_max=5.0` 协议边界（δ_min=−1.058），扫描范围内无物理墙 |

## 2. 总体架构

现有分层不动，新功能沿同一分层插入、**默认关闭**；Becke 电荷/自旋通道行为逐位不变
（性质1-4 的已测点构成回归保护网）：

```
ConstraintLoop（单例，外层控制）
 ├─ WeightGrid        Becke 权重（几何纯函数）          [现有，不动]
 ├─ Observer          q_a = ∫ρ·w_a                      [现有，不动]
 ├─ DipoleObserver    p（分子直积 / 周期 saw 口径）      [新增·模块A：纯观测]
 │                    └─ 同一观测量升级为可约束坐标      [新增·模块B：stage C]
 ├─ MuSolver          secant → 信赖域+Broyden           [加固·模块C2/E]
 ├─ Injector PW/LCAO  v += μ·w（新增 v += −μ·saw/r）    [扩展·模块B]
 ├─ ALM 单层循环      v += (μ+k(Q−t))·w，SCF 内更新     [新增·模块D]
 └─ 持久态            μ/κ/phase 跨离子步保留            [新增·模块C1]
```

涉及文件（现有）：`source/source_estate/module_constraint/{constraint_loop,constraint_io,
constraint_observe,constraint_inject_pw,constraint_inject_lcao,constraint_deriv,
constraint_accounting,mu_solver,weight_grid}.{h,cpp}`，注入点 `esolver_ks_pw.cpp:210/253/350/419`
与 `esolver_ks_lcao.cpp:273/770/930/1092`，力 `forces.cpp:751`、`FORCE_STRESS.cpp:526`。

新增文件预计：`module_constraint/constraint_dipole.{h,cpp}`（观测+saw/分子两种口径）、
`module_constraint/constraint_alm.{h,cpp}`（单层循环），其余在现有文件内扩展。

## 3. 模块 A：偶极报告口径（第一步，纯观测零风险）

### 3.1 观测量

- **分子通道**（真空盒，Γ 点）：`p_α = Σ_i Z*_i·R_iα − ∫ρ(r)·r_α dr`。
  体系带电（ΣZ*≠nelec）时偶极依赖原点：实现取**全胞离子电荷中心**
  `ΣZ*_i R_i / ΣZ*_i` 为参考点，报告中显式打印；`q_dip` 用 `p − p_ref`
  消去该原点依赖，带电体系的绝对 `p` 只作带原点的记录。
- **周期 saw 通道**（优先）：复用 `efield.cpp` 的 saw 口径
  `p_z = Σ_i Z*_i·saw(z_i) − ∫ρ·saw(z) dr`，同时报 **ΔV = 4π·p_z/A**（转 Volt）——
  即界面电势降，与恒电势语义直接对应。saw 跳变面默认放在电子密度最低的 z 平面
  （可由参数覆盖），运行时守卫：任何原子距跳变面 < 1 Bohr 时报警。
- **偶极导出电荷**（双片段）：`q_dip = (p − p_ref)·d̂ / |R_f1 − R_f2|`，
  d̂ = 两约束片段质心连线单位矢量，p_ref = 自由相偶极（delta 口径，与
  `constraint_target_mode=delta` 对齐）。与 Becke q 并列打印，两者之差即分区口径偏差的
  实时量化。saw 通道只有 z 分量有物理意义：实现把两片段中心投影到 z 轴后计算
  `q_dip = (p_z − p_ref,z)/(R_f1,z − R_f2,z)`，并在 `|Δz| < 1e-6` 时判为不可用。

### 3.2 输出与开关

- `final_report()` 每离子步输出一次（O(nrxx) 积分，成本可忽略）；CONSTRAINT_AUDIT
  块增加 `dipole=` / `q_dip=` / `deltaV=`（周期通道）字段。
- INPUT：`constraint_dipole_report`（bool，**默认 false**——不改现有审计行格式，
  保护下游解析脚本）。

### 3.3 验证

- LiCl 同一份密度复现性质1 §5.2 的 6.8665 D（数值积分精度内）；q_dip(δ=0)=0、
  q_dip(δ) 沿性质3 扫描单调；与 Becke q 的偏差量级复现 ~0.6 e。
- 单元测试：玩具密度（点电荷+高斯）的解析偶极对拍；带电体系参考点依赖检验。

## 4. 模块 B：偶极约束坐标（stage C，周期界面优先）

### 4.1 物理核心

偶极对密度的导数是坐标本身 → 约束势是**线性斜坡 = 均匀电场**，乘子 μ 直接是物理场，
无任何分区依赖（性质1 的 0.77 e 展宽在此坐标下不存在）：

| | 分子通道 | 周期 saw 通道（优先） |
|---|---|---|
| 可观测量 | p_α = ΣZ*·R_α − ∫ρ·r_α dr | p_z = ΣZ*·saw(z) − ∫ρ·saw(z)dr（⇔ ΔV=4πp_z/A） |
| 约束势 | v(r) = −μ·r_α | v(z) = −μ·saw(z)，复用 Efield 加场机器 |
| 约束力 | F_i += −μ·Z*_i·α̂（常数项；PW 无 Pulay；LCAO Pulay 自动包含 v_con） | F_iz += −μ·Z*_i·saw'(z_i)；saw' 斜坡段为常数 |
| μ 的物理身份 | 均匀电场（Ry/(e·Bohr)），dE/dp = −μ | 电场；μ×斜坡宽度 = 电势降 |

### 4.2 关键设计决定

1. 新 `ConstraintKind::Dipole`，解除 constraint_io.cpp:420 的 "dipole" 守卫拒绝；
   JSON v2 扩展：
   `{"type":"dipole","axis":"z","frame":"saw"|"molecular","saw_pos":<可选>,"target":<e·Bohr 或 V>}`。
   与 charge/spin 共用同一 `ConstraintLoop`/`MuSolver`（ChannelProfile 增加权重来源维：
   Becke 权重 vs saw/ramp 函数——观测量与注入算子同源的原则保持不变）。
2. **互斥守卫**：dipole 约束与 `efield_amp≠0` 或偶极校正同开 → WARNING_QUIT（双重计数）。
3. **delta 语义**：δp = p − p_ref（自由相偶极为参考），复用现有 target_mode 机制。
4. **范围限制**：只支持全局偶极单约束（电极-电解液界面天然是二体电荷分离）；
   多片段偶极分解欠定，明确不做。
5. 带电体系的参考点规则与模块 A 一致。

### 4.3 验证

- LiCl 偶极约束扫描 vs 电荷约束扫描交叉对拍：给定 q_dip 目标收敛后 p 应命中；
  两坐标乘子满足链式法则 μ_q = μ_p·(dp/dq)（用性质3 的 30 点数据独立拟合 dp/dq 校验）。
- FD 力检验复用性质2 框架（固定 μ 位移腿）。
- saw 通道：LiCl 长盒（z 方向拉长）+ 带电体系验证 ΔV 口径与收敛性。

## 5. 模块 C：效率保守路线

### 5.1 C1 μ 跨离子步热启动（最大单点收益）

- 现状：`ESolver_KS_PW::before_scf`（esolver_ks_pw.cpp:210）每个离子步调
  `ConstraintLoop::init()`，头文件明示 "Re-initialization replaces all state"
  → phase 回 REFERENCE、μ=0、割线历史清零（P6 根源）。
- 设计：`init()` 拆为"几何重建"（WeightGrid 重建，几何纯函数必须每步做）与
  "求解器状态"（μ、κ 历史、phase）两条路径；MD/relax 下后者保留。
  首离子步走完整 REFERENCE 相；后续步直接以 μ_prev 进入 CONSTRAINED。
- **delta 模式 Q_ref 免自由相反推**：`Q_ref(new) = Q_obs(μ_prev, new) − κ·μ_prev`
  （第一迭代后即有 Q_obs，κ 为在线割线斜率）。
  **MD 推荐 absolute 模式**（t 固定、天然免自由相、语义更干净——性质4 已观察到
  delta 模式下 t 随几何漂移 3.19→3.44 e）；delta+MD 组合保留反推路径并打印提示。
- 预期：外层更新 ~10 → 1–2 次/步；SCF 52.5 → 10–15 次/步（**3.5–5×**）。
- 开关：`constraint_warm_start`（bool，默认 true 且仅对 MD/relax 生效；单点 scf 不变）。

### 5.2 C2 自适应信赖域步长

- `MuSolver` 增加信赖域：每步计算 ρ = 实际残差下降 / 预测残差下降；
  ρ>0.5 → 半径×2（上限 1.0 Ry）；ρ<0.1 → 半径×0.5 且**回退该步**（μ 恢复原值重走）。
- 多约束时分量 secant 升级为 Broyden 秩一 Jacobian（处理 charge+spin 混合列表的耦合）。
- 同时缓解 P3（大步翻分支 → 信赖域拒绝回退）与 P4（0.05 硬帽废除）。
- 兼容：`constraint_step_max` 语义改为"初始信赖半径"，默认值与旧行为衔接。

### 5.3 C3 渐进 SCF 收紧

- 外层探测期 drho 门槛放宽至 max(1e-5, 100×scf_thr)；max|res| < 10×constraint_thr
  后切回 scf_thr 跑终步。割线只需要 Q 的相对精度，不需要每步 1e-8。
- hook 点不变（iter_finish 的 observe/on_scf_converged），只调整内部收敛判据传递。

### 5.4 C4 定 μ 生产 MD 产品化

- `ABA_CONSTRAINT_FIXED_MU` 环境变量 → 正式 INPUT/JSON：v2 约束条目支持
  `"fixed_mu": <Ry>`。固定 μ 时外层循环不启动，**一次 SCF 即一帧**
  （性质2 的 12 条 FD 位移腿已实测此路径稳定）；审计行照常输出 Q 供数据标注。
- 用法闭环：定 Q 扫描（性质3）得 μ-Q 曲线 → 选 μ → 定 μ 生产 MD（恒电势系综）。

### 5.5 模块 C 验证

- LiCl δ 扫描 30 点：外层步数降 ≥3×，μ 与性质3 §3.1 表逐点偏差 <0.1%。
- LiCl 约束 MD 100 步：成本对比（目标 ≥3×），⟨μ⟩ 与性质4 一致（<0.5%）。
- APNS LiNiO₂ 自旋三点（性质2 待重跑项）在信赖域+分支锚定下收敛。

## 6. 模块 D：单层循环——增广拉格朗日（深水区）

### 6.1 公式与机制

```
E_ALM[ρ] = E_KS[ρ] + Σ_a [ μ_a·(Q_a[ρ]−t_a) + (k/2)·(Q_a[ρ]−t_a)² ]
v_con(r) = Σ_a [ μ_a + k·(Q_a[ρ]−t_a) ]·w_a(r)      ← 每个 SCF 迭代更新
```

Q_a[ρ] 每迭代已知（O(nrxx)），势随之平滑更新。**与现有 inner schedule 的本质区别**
（性质4 实测 inner 有害：74 vs 52 SCF/步、μ 欠收敛 1.7%）：inner 用割线大步 +
mixing reset；ALM 用固定增益 k 的小步连续更新，不动点映射不被撕裂，**不需要
mixing reset**。

### 6.2 误差结构与控制

- 不动点处约束残差 (Q−t) = (μ*−μ)/k，由 k 控制；报告物理乘子
  **μ_phys = μ + k(Q−t)**（即恒电势标签，进入审计行）。
- 可选每离子步 0–1 次外层 ALM 更新 `μ ← μ + k(Q−t)` 把残差压到任意小（通常不需要）。
- **k 自动设定**：k·χ 目标 5–10。χ=|dQ/dμ| 的实测值：LiCl 电荷通道
  0.075–0.65 e/Ry（性质3：近中性 1/1.54、远段 1/2.339、陡段 1/13.3 Ry/e 的倒数）、
  LiNiO₂ 电荷 ~2.5 e/Ry、自旋 ~12 μB/Ry（性质2 中心差分）。χ 由首步 probe
  （小 μ 扰动）在线测量，k 允许 INPUT 覆盖（`constraint_alm_k`）。

### 6.3 风险与对策

- k 过大 → 势-密度反馈变刚（相当于给 mixing 加额外通道）→ k 上限守卫 +
  与 mixing_beta 联动的阻尼建议；振荡时自动降 k。
- 深水区纪律：**重新过性质2（FD 力/力矩）与性质4（⟨μ⟩ 平均力）全套判据后**才准
  进入生产路径；此前以 `constraint_mu_schedule=alm` 实验开关存在（现有枚举
  "outer"/"inner" 扩展第三值）。

## 7. 模块 E：鲁棒性（随 C 落地）

- **E1 磁分支锚定**：nspin=2 时外层循环监测 TMAG，相对参考相变化超阈值（默认 0.5 μB）
  触发信赖域回退——直接针对性质2 APNS 崩溃签名（TMAG 3.00→4.66→−2.06）。
- **E2** spin 通道默认开启 `branch_tol` 能量分支守卫（现默认 0=关；charge 通道不变）。
- **E3** UNREACHABLE 日志措辞改为 "protocol cap (constraint_mu_max) reached,
  increase constraint_mu_max to extend"（P8：协议边界 ≠ 物理极限）。

## 8. 输入参数与文件格式汇总（全部向后兼容）

| 参数 | 类型 | 默认 | 模块 |
|---|---|---|---|
| `constraint_dipole_report` | bool | false | A |
| `constraint_warm_start` | bool | true（仅 MD/relax 生效） | C1 |
| `constraint_mu_schedule` | + "alm" 枚举 | outer（不变） | D |
| `constraint_alm_k` | double | 0=自动（k·χ≈7.5） | D |
| JSON v2 `type:"dipole"` + `axis`/`frame`/`saw_pos` | 扩展 | — | B |
| JSON v2 条目 `"fixed_mu": <Ry>` | 扩展 | — | C4 |
| `constraint_step_max` 语义 | 初始信赖半径 | 0.05（兼容） | C2 |

## 9. 测试策略与验收标准

- **单元测试**：每新模块配 gtest（沿 `module_constraint/test/` 现有 8 个测试文件模式）。
- **代码规范**（AGENTS.md）：每个 if-else 分支前必须有注释；禁 goto/setjmp；
  函数 >300 行拆分；优先 RAII/std::vector。
- **回归网**：性质1-4 已测点构成 golden——Becke 通道输出逐位或判据内不变。
- **每轮改动产出 dated 测试文档**（docs/superpowers/specs/YYYY-MM-DD-<topic>.md：
  测试计划/设置/结果/分析/下一步），并更新 `deltap-development-log.md`。

| 模块 | 验收标准 |
|---|---|
| A | LiCl 偶极复现 6.8665 D；q_dip 与 Becke q 并列输出正确；带电体系参考点守卫生效 |
| C | 扫描外层步数 ≥3× 下降且 μ 逐点 <0.1% 吻合性质3；MD ≥3× 加速且 ⟨μ⟩ <0.5% 吻合性质4；APNS 自旋三点收敛 |
| D | LiCl 扫描 μ_phys 与性质3 偏差 <0.5%；性质2 FD 判据复测通过；性质4 ⟨μ⟩ 复测 <5% 判据 |
| B | q_dip 目标收敛后 p 命中；链式法则 μ_q=μ_p·dp/dq 在性质3 数据上闭合 <1%；FD 力通过 |
| E | 构造的磁分支翻转用例触发回退而非崩溃 |

## 10. 里程碑

| # | 内容 | 依赖 | 风险 |
|---|---|---|---|
| M1 | 模块 A（偶极报告）+ 单元测试 + LiCl 复算文档 | 无 | 低 |
| M2 | 模块 C1+C4（热启动+定 μ 产品化） | M1 | 中（delta 模式 Q_ref 反推精度） |
| M3 | 模块 C2+C3+E（信赖域+渐进 SCF+鲁棒性） | M2 | 中 |
| M4 | 模块 D（ALM 单层循环）+ 性质2/4 重验 | M3 | 高（深水区，k 稳定性） |
| M5 | 模块 B（偶极约束坐标，saw 优先） | M3（加固后的求解器）、M1（观测代码） | 中高（周期边界/Pulay） |

## 11. 风险登记

| 风险 | 缓解 |
|---|---|
| delta+MD 的 Q_ref 反推不准导致目标漂移 | absolute 模式推荐 + 反推值与周期性强校验（每 N 步真自由相）可选 |
| ALM 的 k 与 mixing 耦合震荡 | k 上限守卫 + 自动降 k + 实验开关隔离生产路径 |
| saw 跳变面被原子穿越（MD） | 运行时守卫 + 跳变面密度最低自动选位 |
| LCAO 下偶极约束的 Pulay 项口径错误 | FD 力检验先行（性质2 框架），不过不合入 |
| 信赖域回退与 mixing 历史交互 | 回退时 mix_reset（已有机制），记录回退次数入审计 |
