# 一期开发进展实证 Review（M0–M8 + V1/V3 判决性验证）

> 评审对象：一期完成报告（M7/M3a/M5/M8 + 判决性验证，提交 e43b1d5d5/b78d55553/4e15b4775/dff13618b）。
> 评审方式：**不采信报告文本**，两路独立核实——(a) 实跑测试与集成用例复现；(b) 8 个 commit 的代码评审。
> 本轮无代码改动，属评审轮。

---

## 一、总体裁定：✅ 通过（claims 基本属实，可开二期；1 处失信须修正 + 3 处建议修复）

## 二、实证核实结果（报告声称 vs 实际）

### 2.1 测试与复现 — 全部属实

| 报告声称 | 独立核实结果 | 结论 |
|---|---|---|
| ctest constraint/partition/read_input/elecstate_pw/weight_grid_mpi 全绿 | 实测：constraint 9/9、partition 1/1、read_input 2/2、elecstate_pw 1/1、weight_grid_mpi 2/2 | ✅ |
| H₂O 冒烟：delta=+0.1 e，6 外步 CONVERGED，res=3.06e-5 | 实跑复现：μ=−0.1765 Ry、outer step 6、CONVERGED，逐项吻合 | ✅ |
| V1：total_charge=8==nelec，O=6.2555/H=0.87227，maxdev=2.2e-16 | 实跑复现，逐位一致 | ✅ |
| unitcell_test_pw 失败为既有 CWD 问题 | 属实：support/ 未拷入构建树，换 CWD 后 3/3 PASS | ✅ |
| V3 ±0.05–0.3 e 全可达、熔断、反假收敛 | spec 文档（2026-08-31-v1-v3-validation.md）数据链完整自洽（μ*≈−1.7·delta，7/7 可达） | ✅ |

### 2.2 代码评审 — 护栏与修复全部真实

- M4 七项护栏（逐分量 secant/κ clamp/翻号检测/单步限幅/μ 硬顶/联合熔断/反假收敛）全部如实实现且测试真覆盖（37/37 串行单测实测 PASS）；
- "观测量==注入算符"非口头声明：ConstraintLoop 单一权重实例构造保证 + 数值恒等式测试（∫ρ·μw dr == μ·Q，1e-10）；
- 三个 bug 修复（越界防御/conv_esolver 门控/嵌套 fragments 空白）均真实且有回归测试锁定；
- AGENTS.md 纪律：分支前置注释合规、无 goto/setjmp、RAII 合规、无 >300 行函数、spec 文档链完整。

## 三、发现的问题（按优先级）

### ❌ P1：计划 Task 10 (V2) 勾选失信
`2026-08-30-realspace-weight-constraint-phase1.md` 第 352 行 Task 10 已勾 `[x]`，但 V2 实际阻塞（无 Multiwfn，spec §3 自认）。**必须改回 `[ ]` 或标注"阻塞"**——勾选失信会破坏计划文档作为验收依据的可信度（本仓库纪律：PASS 必须有独立参照物）。

### ⚠️ P2：inject() 返回值被丢弃，违反自身头文件契约
`constraint_inject_pw.h:27-29` 承诺 "return false ... outer loop must WARNING_QUIT"，但 `constraint_loop.cpp:88-89` 丢弃返回值。当前因 mu_ 长度派生自权重对象而不会触发，但契约与行为不一致是潜伏 bug 温床。建议：检查返回值并 WARNING_QUIT，或改契约。

### ⚠️ P3：报告中 deltaspin "从未生成"表述不准确
实测 `ctest -N -R deltaspin`：4 个目标已注册，2 个已构建且 PASS，2 个 Not Run。应表述为"2/4 未构建"。虽是回归说明的措辞问题，但评审文化要求精确。

### ⚠️ P4：集成用例未注册 ctest，CI 不可重现
`tests/constraint_pw_h2o/` 无 ctest 注册，README 引用绝对路径。建议二期开头补注册（tests/ 集成测试体系）。

### ⚠️ 轻微（记录不阻断）
- `constraint_loop.cpp:113` 注释（"delta mode"）与紧随的 `if (absolute)` 分支错位；
- esolver 钩子实际 +90 行/4 挂载点 vs 计划口径"~10 行/单钩子"——功能正确，属计划口径偏差，before_scf 的 ~50 行配置块建议下移至模块内；
- M7 自研 JSON 子集解析器（398 行）未按计划复用 sc_parse_json——spec 有理由陈述但未正面回应复用指令；功能无问题，技术债登记。

## 四、对 T12"开二期"判决的评审意见

判决成立，但附带一项债务声明：一期验收门原计划为 V1+V2+V3+反假收敛四项，**V2 未闭环**（外部工具阻塞，非代码问题）。M2 高斯基准（1e-8）+ 单点钉（1e-12）提供了内部覆盖，作为开二期的依据可接受；但 V2 必须作为**未清债务**跟踪——任何"基组无关/口径正确"的对外声明（尤其增量 1 的可发表主张）在 V2 补拍完成前不得作出。建议在二期计划中将"V2 补拍"列为阻塞性前置项。

## 五、结论

一期开发质量高、报告基本诚实、判决性验证数据链完整可复现。**裁定：通过，可开二期。** 须立即修正 P1（勾选失信），建议二期开头一并处理 P2/P4；V2 债务转入二期前置项。

---

## 本轮记录

- 评审轮，无代码改动。两路独立核实（测试实跑+代码评审）均完成；
- 待跟进清单：P1 计划勾选修正（立即）、P2 inject 返回值契约、P4 集成用例 ctest 注册、V2 Multiwfn 补拍（二期前置）。

---

## 修复回执（2026-08-31，`2026-08-31-phase1-review-fix.md`）

- P1 ✅：计划 Task 10 (V2) 改回 `[ ]` + 阻塞标注，V2 债务转二期前置。
- P2 ✅：inject() 两路返回值检查 + WARNING_QUIT（契约兑现）。
- P3 ✅：实测确认 4 注册/2 PASS/2 Not Run；文档按精确措辞表述。
- P4 ✅：用例迁入 `tests/01_PW/211_PW_constraint_h2o/` 并注册
  CASES_CPU.txt，result.ref 生成且 np=4 比对 PASS；README 相对路径。
- 轻微 ✅ 已登记：注释错位已修；esolver 钩子口径偏差 + M7 JSON 复用
  债务转入二期（不阻断）。
- V2 债务跟踪：Multiwfn 补拍（或 V2a/V2b 内部闭环）为二期阻塞性前置项；
  完成前不作"基组无关/口径正确"声明。
