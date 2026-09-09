# Task A4 严格评审（M8 接线 + M5 kind 标签，G4）

> 评审对象：commit f268a2cfc + spec `2026-09-09-taskA4-loop-wiring.md`。本轮无代码改动。

## 裁定：✅ 通过（G4 确认），批准启动 Task A5

## 独立核实

| 项 | 核实结果 |
|---|---|
| 提交与推送 | f268a2cfc 在 HEAD；zdy/feat/deltap = f268a2cfc7（推送属实）；工作树干净 |
| 测试 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11 PASS** |
| G4 核心测试 | MixedConvergesOnLinearResponse（constraint_loop_test.cpp:386）在案；审计行 kind 标签断言（`c[0] kind=charge`/`c[1] kind=spin`，:461-462）真断言 |
| 逐约束 cap | `MuSolverParams::mu_max_per_component`（mu_solver.h:17）落位；MixedFuseHonorsPerComponentCap（:465）覆盖逐分量熔断 |
| staging guard 移除 | 可证伪链完整：编译红→守卫在位红→移除绿→sabotage A（恢复 core 守卫恰中 2 测试）/sabotage B2（cap 退化恰中 1 测试）——评审纪律项①兑现 |
| 双调用点同步 | `configure_from_inputs` 在 esolver_ks_pw.cpp:204 与 esolver_ks_lcao.cpp:267 同签名同步改造——评审纪律项②兑现 |
| 诚实中间态 | compute_force 两趟掩码组合明确标注为 A5 前的过渡态（计划 A5 由 per-α 力核签名收回）——技术债显式登记而非隐藏 |
| 兼容性 | legacy 5 参 audit 空 kinds 输出逐位不变；legacy 入口承接混合拒绝（该入口丢弃 specs，语义正确） |

## 给 A5 的验收提醒（承前 + 新增）

1. G5 判据：混合力解析对拍 + 牛顿第三定律（1e-10）+ μ=0 分量短路；
2. per-α 力核签名收回两趟掩码时，须保留**同构场景与历史逐位一致**的回归断言（A4 已锁定的行为不得漂移）；
3. 混合力 FD 不在本 Task 抢跑（留在 A6 后的验收轮，stationary4 处方）。

---

## 本轮记录

- 评审轮，无代码改动。推送/11/11/测试名/行号/双调用点均经亲验。
