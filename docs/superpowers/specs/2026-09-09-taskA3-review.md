# Task A3 严格评审（M3a 逐约束注入，G3）

> 评审对象：commit 9c540f104 + spec `2026-09-09-taskA3-inject-channel.md`。本轮无代码改动。

## 裁定：✅ 通过（G3 确认），批准启动 Task A4

## 独立核实

| 项 | 核实结果 |
|---|---|
| 提交与推送 | 9c540f104 在 HEAD；zdy/feat/deltap = 9c540f104a（推送属实）；工作树干净 |
| 测试 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11 PASS**；inject_pw 靶 7 tests（5 旧 + 2 新：MixedChannelCrossZeroPointwise:340、MixedObservableEqualsInjection:385） |
| 守卫契约 | 注入侧 4 条 false-return 路径在案（constraint_inject_pw.cpp:20/41/48/62），buffer 不动——与 A2 评审提醒的同型守卫要求一致且遵守既有 false-return 契约 |
| 核心验证强度 | MixedObservableEqualsInjection 用的是**完整混合恒等式** E=∫(ρ↑V↑+ρ↓V↓)≡Σμ_αQ_α（δ 探针 1e-12 + 平滑自旋密度 1e-10），且与 A2 读数共用同 (w,chan)——混合场景"观测量==注入算符"的构造性保证闭环 |
| sabotage | 逐 α 恒用 channels[0] → 恰中 2 新测试、5 旧绿（判别力实证） |
| 范围纪律 | 无偶极成分、无 Broyden/松紧 SCF（阶段 B 内容未抢跑） |

## A4 开工条件确认

A4（M8 接线 + M5 kind 标签，G4）可以启动。已入档的两条纪律再次强调为 A4 验收项：
1. staging guard（constraint_io.cpp:956-977）移除必须可证伪：G4 混合收敛测试先失败→移除后通过→恢复守卫 sabotage 恰中 FAIL；
2. `configure_from_inputs` 签名改造保持 PW/LCAO 双调用点同步，单测之外须有双基组编译验证。

---

## 本轮记录

- 评审轮，无代码改动。推送状态、11/11、测试名、守卫行号均经亲验。
