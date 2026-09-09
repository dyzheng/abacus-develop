# Task A2 严格评审（M2 逐约束 channel 读数，G2）

> 评审对象：commit 2ba46981e + spec `2026-09-09-taskA2-observe-channel-read.md`。本轮无代码改动。

## 裁定：✅ 通过（G2 确认），批准启动 Task A3

## 独立核实

| 项 | 核实结果 |
|---|---|
| 提交与推送 | 2ba46981e 在 HEAD；zdy/feat/deltap = 2ba46981e2（推送属实）；工作树干净 |
| 测试 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11 PASS**；observe 靶 7 tests（5 旧 + 2 新：MixedChannelReading:509、MixedChannelConservation:563） |
| 守卫 | `channels.size()!=nconstraint` → WARNING_QUIT（observe.cpp:51）；nspin=1 下 spin profile → WARNING_QUIT（:67）——parse 层之外的纵深防御在位 |
| 架构 | 单一求积实现 + 旧 DensityChannel 入口薄适配（commit stat 与 diff 一致：observe.h 仅 +13 行适配层）——无第二套读数路径的声称成立 |
| 诚实边界 | "spin 片段 ≈ 原子磁矩和"因 Becke 盆地漏磁放弃恒等式改用三条干净断言——正确取舍（盆地分划的矩不守恒是已知物理，硬凑恒等式会埋假验收） |
| sabotage | 逐 α 恒用 channels[0] → 恰中 2 新测试、5 旧绿（守卫判别力实证） |

## 给 A3 的提醒（承前）

- G3 的核心是**逐通道恒等式 + 交叉零项逐点断言**（charge μ 不进自旋差势、spin μ 不进总势）——这是混合场景下"观测量==注入算符"的最后一块构造性保证；
- A2 已把 (w,chan) 错配守卫做进读数侧，A3 注入侧应有**同型守卫**（channels.size()!=nconstraint → 返回 false 而非 WARNING_QUIT 的契约选择需与现有 inject 的 false-return 契约一致）。

---

## 本轮记录

- 评审轮，无代码改动。推送状态、11/11、守卫行号、测试名均经亲验。
