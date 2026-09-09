# Task A5 严格评审（M6 力核逐约束 channel，G5）

> 评审对象：commit 3bf9f4c7d + spec `2026-09-09-taskA5-deriv-channel-force.md`。本轮无代码改动。

## 裁定：✅ 通过（G5 确认），批准启动 Task A6（阶段 A 最后一个工程 Task）

## 独立核实

| 项 | 核实结果 |
|---|---|
| 提交与推送 | 3bf9f4c7d 在 HEAD；zdy/feat/deltap = 3bf9f4c7d5（推送属实）；工作树干净 |
| 测试 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11 PASS**；deriv 靶两新测试在案（MixedChannelForce:597、MixedForceNewtonThirdLaw:679） |
| 力核签名 | per-α `channels` 数组入核（constraint_deriv.h:72），错配 abort（:58）；compute_force 单趟 per-α（constraint_loop.cpp:390-413），A4 两趟掩码确已收回，驻点守卫原样保留 |
| 纪律 | 混合力 FD 未抢跑（留 A6 后验收轮）；sabotage（spin 误读 charge）恰中 2 新测试 |

## 物理审查（重点）

**混合牛顿第三定律的表述正确**：本轮测试用的是非平凡恒等式
Σ_J F_J = −Σ_α μ_α dQ_α/dδ（δ=均匀平移），而非朴素的 Σ_J F_J=0——这是对的：
约束块单独不必求和为零（LCAO Pulay 案的 net_z=Σforcecon=7.08 eV/Å 正是这个
性质的体现），平移不变性只对含 KS 补偿项的总力成立。RHS 用独立 5 点 FD 折叠
路径计算，构成"折叠可证伪器"。该表述与我评审链的推导一致，且把 A2 曾发现的
净力现象变成了永久性回归锚。

## 结论

阶段 A 代码层最后一块（读数 A2 / 注入 A3 / 编排 A4 / 力 A5）全部以单一实现 +
薄适配落地，无第二路径残留。**批准启动 A6**：H₂O 混合集成（charge+spin 同原子）
+ 三旧用例逐位回归 + μ 耦合偏移测量（阶段 B 立项依据）。

---

## 本轮记录

- 评审轮，无代码改动。推送/11-11/测试名/力核签名/牛三恒等式表述均经亲验。
