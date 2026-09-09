# 实空间权重约束：阶段 A（混合 charge+spin）进展总结

> 截至 HEAD `fc93bac1e`（2026-09-09）。前置汇总：`2026-08-31-constraint-framework-progress-summary.md`
> （一期闭合 + 二期 2.1–2.5）。本总结覆盖：Task 2.6/2.7 判决门闭合 + 阶段 A（A0–A6）全部交付与实测。
> 范围收缩（用户决定）：阶段 A 只做 charge+spin 混合；偶极不开发（守卫拒绝）；Broyden/松紧 SCF 属阶段 B（测量驱动）。

---

## 1. 一句话现状

同一 run 可对任意原子/片段施加 **charge + spin 混合约束**（v2 JSON 列表，逐约束
type/target/atoms/mu_max），PW 与 LCAO 共口径；约束力经 FD 验收并声明**限定包络**
可用；同原子双类型混合的 μ 耦合已实测（两分量同向刚化）——阶段 B Broyden 立项数据到手。

## 2. 交付清单（阶段 A，A0–A6 + 判决门）

| Task | 内容 | 提交 | 门 |
|---|---|---|---|
| 2.6/2.7 | 力 FD/力矩/对拍判决门收口 + 闭合文档（覆盖表/能力包络/风险清单） | bb814be57（含 A0 决策） | 闭合 |
| A0 | schema 决策（run 级 target_mode / v2+v1 自动转换+deprecation / mu_max 逐约束） | 纸面入 A1 spec | ✅ |
| A1 | `ConstraintSpec`/`ChannelProfile` 数据模型 + v1/v2 解析 + 守卫 | 14fabba2c | G1 |
| A2 | M2 读数逐约束 channel | 2ba46981e | G2 |
| A3 | M3a 注入逐约束 channel | 9c540f104 | G3 |
| A4 | M8 总控接线 + M5 kind= 审计 | f268a2cfc | G4 |
| A5 | M6 力核 per-α channel（两趟掩码收回） | 3bf9f4c7d | G5 |
| A6 | H₂O 混合集成 213 + 三旧回归 + μ 耦合实测 | fc93bac1e | G6 |

## 3. 测试总览（2026-09-09 实测）

### 3.1 单元测试（模块 constraint，11 个注册 ctest 目标全绿 11/11，~49 s）

逐 Task 新增判别测试全部在案：io `MixedConstraintListParsing`/`MixedGuards`；observe/inject
混合通道对拍（含恒等式）；loop `MixedConvergesOnLinearResponse`（收敛+反假收敛+kind= 审计）、
`MixedFuseHonorsPerComponentCap`（异构 mu_max 熔断）；deriv `MixedChannelForce`（分量短路
1e-12 / legacy 叠加 1e-12 / 逐分量 quadrature 锚 1e-8 / 零 μ 精确 0）、`MixedForceNewtonThirdLaw`
（混合牛三 1e-9，RHS=observer 5 点 FD 独立折叠）。

### 3.2 sabotage 判别链（守卫↔测试一一对应，全部恰中）

| 轮 | 破坏 | 恰中 |
|---|---|---|
| A1 | spin∩nspin=1 守卫置 false 等 3 处 | `MixedGuards` |
| A2/A3 | 逐约束选择恒用 channels[0] | 各混合通道新测试 |
| A4 | 恢复 core staging mixed 守卫 / cap 退化为标量首 cap | loop+io 3 测试 / 1 测试 |
| A5 | 混合列表 spin 误读 charge buffer | 恰 2 新 deriv 测试，legacy 4 绿 |
| A6 | 恢复 A1 staging mixed 守卫（复验） | io `MixedGuards` + loop 2 测试 |

### 3.3 集成测试（4 用例注册 CASES_CPU.txt，result.ref 入库）

| 用例 | 场景 | 关键值（实测 np1） | 能量偏差 vs ref |
|---|---|---|---|
| 211_PW_constraint_h2o | PW charge +0.1 e on O | μ_c=−0.176548，CONVERGED | 4.1e-9 a.u. |
| 212_PW_constraint_h2o_spin | PW spin +0.1 μB on O | μ_s=−0.072339，CONVERGED | 9.5e-11 a.u. |
| 213_PW_constraint_h2o_mixed | PW **charge+spin 同原子 O**（v2 列表） | μ_c=−0.181173、μ_s=−0.081544，外步 15 CONVERGED，res<1e-4，maxdev=2.2e-16 | （新 ref，本批入库） |
| 212_NAO_constraint_h2o | LCAO charge +0.1 e on O | μ_c=−0.219304，CONVERGED | 2.4e-11 a.u. |

三旧用例走 v1 deprecated 格式兼容路径，**零修改逐位复现**（A6 回归实测）。

### 3.4 μ 耦合实测（P3/阶段 B 立项数据）

| 场景 | μ_c (Ry) | μ_s (Ry) |
|---|---|---|
| 211（nspin1 charge-only） | −0.176548 | — |
| 212（nspin2 spin-only） | — | −0.072339 |
| nspin2 charge-only 基线（spin target 0 休眠） | −0.176354 | 0 |
| **213 混合（同原子）** | **−0.181173** | **−0.081544** |
| 耦合偏移 | −0.004819 | −0.009206 |

结论：nspin 1→2 对 μ_c 仅 +1.9e-4（可忽略）；**跨类型耦合不可忽略且同向刚化**
（对方通道存在 → 本通道需更负 μ）；外环收敛 3→15 步退化。→ 阶段 B Broyden/Jacobian
立项成立（对角 secant 不再充分）。混合力 FD（stationary4）留作阶段 A 后的验收轮，未抢跑。

## 4. 力口径与能力声明（2.6/2.7 闭合后）

- **单通道力**：PW 18 腿全轴 PASS；LCAO 根因定位（μw Pulay 生命周期缺失）修复并验证
  最坏轴 35× 富余；覆盖表诚实（未覆盖轴列明+外推理由）。净力/补偿前力入标准检查项；
  F_ana 补偿前后双值入档。
- **能力包络（限定声明，不作普适外推）**：H₂O 类小分子、|μ|≲0.5 Ry、约束残差<1e-4。
- **尚未接线**：relax/MD 几何优化；应力；混合力 stationary4 FD。

## 5. 风险清单新增（2.7 起永久项）

- **"势生命周期末端丢 μw"同族模式**：PW 丢在 vnew 快照（SCC 力污染）、LCAO 丢在 v_eff
  重建（Pulay 缺失）——任何新注入点（应力/偶极/未来通道）必须回答"力求值/能量决算时刻
  μw 还在势里吗"，修复统一用 `ConstraintLoop::add_back_constraint_potential()` 修正副本。
- **力包络外推禁令**：能力声明限定包络；逐轴未过 FD 处不得默认外推（FD 归因标准检查项：
  净力/补偿前力 + F_ana 双值入档）。

## 6. 下一步

- **Task A7**（收尾文档，本批）：用户手册/开发者指南/进展总结更新 → 评审申请。
- 阶段 A 后验收轮：混合力 stationary4 FD（不抢跑项落地）。
- 阶段 B（测量已齐，立项依据在 §3.4）：Broyden/Jacobian 多约束；DeltaSpin 量级对等。
- 开放项跟踪（不得遗失）：① 严格 PW≡LCAO 需只读观测口（A 工具补件）；② torque 脚本
  E' 口径修复；③ DeltaSpin 量级对等（B）。

---

> 文档链：方案 → 架构（plan-architecture.md）→ 逐 Task spec（2026-09-08/09 taskA0–A6 + 评审
> A1–A6）→ 判决门文档（2026-08-31 总结、2026-09-08 力 FD 两案、task26 closure）→ 本总结。
> 全部运行记录见 `deltap-development-log.md`（2026-09-08/09 各节，编号 (1)–(23)）。
