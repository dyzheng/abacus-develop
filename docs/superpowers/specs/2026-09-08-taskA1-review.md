# Task A1 严格评审（M7 混合 schema IO，G1）

> 评审对象：commit 14fabba2c + spec `2026-09-08-taskA1-mixed-schema-io.md`。本轮无代码改动。

## 裁定：✅ 通过（G1 确认），批准启动 Task A2

## 独立核实

| 项 | 核实结果 |
|---|---|
| 提交与推送 | 14fabba2c 在本地 HEAD；`zdy/feat/deltap` = 14fabba2c47e…（推送属实）；工作树干净 |
| 测试 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11 PASS**（含 10 旧 + 2 新，旧 v1 语义兼容） |
| 数据模型 | `build_channel_profile` 工厂在位（分支注释齐全），spin+nspin 守卫错误信息精确（constraint_io.cpp:436-442） |
| 偏差 1（混合阶段性 ERROR） | 代码核实：staging guard 明确标注 "removed at A4"（constraint_io.cpp:956-977），legacy 单通道链不受影响（:984）——**这是防御性设计而非偷懒**：防止单通道 loop 把 spin 靶当 charge 静默错跑，且 S3 sabotage 证明该守卫真实可触发 |
| 偏差 2（configure_from_inputs 签名推迟 A4） | 合理——避免引入无人消费的形参；specs 消费方（loop）本来就在 A4 接线 |
| 测试存在性 | MixedConstraintListParsing（:302）、MixedGuards（:395）在案，全文件 11 tests |
| spec 质量 | TDD 环完整（编译红→stub 红→全绿）、sabotage 三发恰中目标、v1 兼容有 T3 回归锁定、警告走 ofs_warning 不污染 result.ref（A0 前瞻条款兑现） |

## 给 A2/A3 的两条提醒

1. **staging guard 的移除必须可证伪**：A4 移除 constraint_io.cpp:956-977 的混合拒绝时，
   G4 的 loop 级混合收敛测试必须先于移除失败、移除后通过，并配一次 sabotage
   （恢复守卫 → 恰中 FAIL）证明移除是受控的；
2. A2/A3 单测从 parse 层取 specs（偏差 2 的路径）可接受，但 A4 改
   `configure_from_inputs` 签名时须保持 PW/LCAO 两 esolver 调用点同步（双调用点
   曾是一期技术债来源，勿再分叉）。

## 结论

A1 交付与计划/A0 契约逐字一致，G1 判定成立。**批准启动 Task A2**（M2 逐约束
channel 读数；G2：混合 mock 对拍 + 守恒和断言）。

---

## 本轮记录

- 评审轮，无代码改动。推送状态、11/11 测试、staging guard 标记均经亲验。
