# Task A1：M7 混合约束数据模型 + schema 解析与守卫（TDD 轮）

> 批复：`docs/superpowers/specs/2026-09-08-task26-closure-review.md`（2.6/2.7 门闭合，
> 批准启动 Task A1：按 A0 决策记录 G1 断言清单写失败测试）。
> 契约源：`docs/superpowers/specs/2026-09-08-taskA0-schema-decisions.md` §1（G1 清单）与 §3（D1–D4 定稿）。
> 计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A1。
> 本 Task 只动 `constraint_io.{h,cpp}` + `test/constraint_io_test.cpp`（轻量单测，无重算）。

---

## 1. 测试方案（Test plan）

TDD 环：失败测试先行 → 实现 → 既有 11 个 MODULE_ESTATE_constraint 测试全绿 →
sabotage（守卫移除须恰中 FAIL）→ spec 回填 → 日志条目 → commit。

- **T1 `MixedConstraintListParsing`**（G1 断言，parse 层契约）：
  - v2 `{"constraints":[charge, spin]}` → `specs` 两条，各自
    `{kind, atoms, chan, target, mu_max}` 正确；`chan` 逐字段等于
    `build_channel_profile(kind)` 工厂值（禁手填验证）；charge=(+1,+1,+1,+1)、
    spin=(+1,-1,+1,-1)；
  - v1 `{"targets","atoms"}` + run 级 type → 自动转换：`format==V1`、逐条
    `kind=channel_from_type(run_type)`、`mu_max=run` 级、deprecation WARNING 置位；
  - v2 atoms 嵌套写法按契约拒绝（v2 逐约束 `atoms` 为扁平数组；嵌套是 v1 语义），
    扁平 atoms 与缺省 atoms（= 列表内序 [i]）正确；
  - 逐约束 `mu_max` 缺省回退 run 级、显式值生效。
- **T2 `MixedGuards`**（G1 断言，configure 层守卫）：
  - v2 含 spin 约束 + nspin=1 → ERROR（含 "nspin"）；
  - v2 `{"type":"dipole"}` → ERROR "not implemented"（偶极 A 阶段不开发）；
  - v2 空 `constraints` → ERROR；
  - v2 atoms 越界 [0,nat) → ERROR；
  - 重复 (kind, atoms) 组合 → WARNING（近共线预警，warnings 含标记）；
  - 同现 `constraints`+`targets` → ERROR（不猜测语义）；
  - v2 + run 级 `constraint_type != "charge"` → WARNING "supersede"（不静默忽略）；
  - v2 混合 kind（charge+spin 并存）在 configure 层 → 阶段性 ERROR
    （**A1 登记偏差 1**：单通道 legacy loop 尚不能表达混合通道，静默错跑违纪律；
    A4 loop 吃 specs 后移除该守卫——见 §3.3）。
- **T3 既有回归**：现有 `ConstraintIOTest.*` 全部经 legacy 包装保持绿
  （v1 语义逐位不变；v2 单 kind 单 cap 可经 cfg 表达 → configure 放行）。

## 2. 测试设置（Setup）

- 平台：容器 gcc 单测（`build/`，feat/deltap HEAD bb814be57 + 2 笔评审轮未提交改动）。
- 轻量命令：`cd build && make MODULE_ESTATE_constraint_io -j8 && ctest -R MODULE_ESTATE_constraint_io`；
  全量回归 `ctest -R MODULE_ESTATE_constraint`（基线 11/11 PASS，~40 s）。无重型计算。
- 调用面（已核实）：`constraint_io` 仅被 `esolver_ks_{pw,lcao}.cpp`
  （`configure_from_inputs`，A1 不改签名——见偏差 2）与 `constraint_loop`
  （持有 cfg，单通道 legacy 链，A2–A4 改造）消费。
- 本机 gtest 靶名 `MODULE_ESTATE_constraint_io`（11 tests 基线全绿）。

## 3. 结果（Results）

- 基线核对（改造前）：`ctest -R MODULE_ESTATE_constraint` = **11/11 PASS**（~40 s）。
- TDD：新 API 缺失时编译红 → stub（parse 恒 false / configure 恒 ERROR）红 → 全实现绿。
- 实现后：`MODULE_ESTATE_constraint_io` 11 tests **全绿**（10 旧 + 2 新）；全套
  `MODULE_ESTATE_constraint` **11/11 PASS**（旧用例 v1 语义逐位不变）。
- **sabotage（守卫↔测试一一对应，均恰中 `MixedGuards`，其余 10 tests 绿）**：

| # | sabotage（移除守卫） | 恰中 |
|---|---|---|
| S1 | v2 per-constraint `spin && nspin!=2` 守卫置 false | `MixedGuards` 1 FAIL |
| S2 | 未知 type（含 dipole）不再拒绝、静默当 charge | `MixedGuards` 1 FAIL |
| S3 | 混合 kind 的 configure 阶段性 ERROR 置 false | `MixedGuards` 1 FAIL |

- 新增数据模型（constraint_io.h）：`ConstraintKind{Charge,Spin}`、`ChannelProfile`
  `{read_up,read_dn,inj_up,inj_dn}`、`ConstraintSpec{kind,atoms,chan,target,mu_max}`、
  `ConstraintFileFormat{V1,V2}`、`build_channel_profile(kind)` 工厂、`parse_constraint_file`
  （v1+v2 统一解析入口）、13 参 `configure_constraint(cfg,specs,warnings,...)` 扩展核。

## 4. 分析（Analysis）

- 工厂契约（A0 D4）已单测锁定：charge=(+1,+1,+1,+1)、spin=(+1,-1,+1,-1)；
  `spec.chan` 恒等于工厂值（若实现手填/漏推导即红——T1 `same_profile` 断言）。
- v2 解析是自研 JSON 子集解析器首次碰"对象数组"（A0 风险登记项）：按白名单 schema
  实现——对象内字段任意顺序（每键独立扫描）、嵌套对象/嵌套 atoms 拒绝、
  引号内 `}` 不误截（对象切片扫描跳过字符串）。空 constraints / 空 atoms /
  越界 / mu_max≤0 / 缺 type/target 全部结构化 ERROR，失败测试锁定。
- **偏差登记 1**（configure 层混合拒绝，A4 移除）：legacy 单通道 loop 无法表达
  混合通道 run（会把 spin 靶当 charge 静默错跑）→ 阶段性 ERROR；解析层照常出
  specs（T1 断言 charge+spin 两条 specs 正确），A2–A4 loop 吃 specs 时移除该守卫
  （S3 sabotage 已证明其存在且可一击移除）。
- **偏差登记 2**（configure_from_inputs 签名不变）：计划 A1 曾写"configure_from_inputs
  输出 specs"。A1 实际只产出 parse 层 API + configure 扩展核（13 参），esolver 的
  configure_from_inputs 签名保持原样（调用核、打印 warnings）；specs 的真实消费方
  （loop）在 A4 才接线，届时再加 out-param 一次性到位，避免 A1 引入无人消费的形参
  与 esolver 空改。A2/A3 单测直接从 parse_constraint_file 取 specs。
- 警告输出纪律：deprecation/supersede/absolute/duplicate 等非致命警告由
  configure_from_inputs 逐条 `ModuleBase::WARNING`（ofs_warning 流）每 run 一次，
  不触主输出 → 不污染 result.ref 口径（G6 三旧用例逐位回归届时把关）。
- v2 单 kind + 同 cap 文件可经 legacy cfg 放行（T2 断言 cfg.type/targets/mu_max
  正确填充）——即 v2 用户迁移立即可跑（单通道场景），混合场景待 A4。

## 5. 下一步（Next Steps）

- 提交本 Task（含评审轮 2 笔文档改动）后推送 zdy；向用户汇报请求裁决。
- 用户批准后启动 **Task A2**（M2 逐约束 channel 读数）：observe 签名接
  `std::vector<ConstraintSpec>`/profiles，混合 mock 对拍 + 守恒和断言（G2）。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口（A 阶段工具补件）；
  ② torque 脚本 E' 口径修复；③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`constraint_io.{h,cpp}`（数据模型 + v2 解析 + 守卫 + configure 核重构）；
  `test/constraint_io_test.cpp`（+2 tests：MixedConstraintListParsing / MixedGuards）。
  轻量单测，无重算。更新：本 spec、开发日志条目 (13)、计划 A1 勾选。
- 判定：**G1 通过**（io 单测全绿 + 旧用例兼容 + 3 sabotage 恰中）。
