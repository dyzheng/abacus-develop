# 阶段 A：电荷+自旋同时约束——分步开发与验证计划

> 依据：《统一多约束框架架构重设计》评审（2026-09-08-unified-multi-constraint-redesign-review.md）修订版阶段 A。
> **范围收缩（用户决定）：本期只做 charge+spin 混合；偶极（z 权重/Dipole kind）不开发，守卫拒绝。Broyden/松紧 SCF 不开发（测量驱动，属阶段 B）。**
> **硬前置：Task 2.6 判决门闭合。** 闭合前只允许 Task A0（纸面）。
> **硬前置状态（2026-09-08 更新）**：Task 2.6 判决门经修订版路径 1 收口闭合
> （见 `docs/superpowers/specs/2026-09-08-task26-closure.md`），A1 可起跑。
> 原则：每个 Task = 失败测试先行 → 实现 → 验证 → commit；前一 Task 全绿才开下一 Task；任何守卫配 sabotage 验证。

---

## Task A0：数据模型与兼容决策（纸面，可立即开始，半天）

> 已执行：`docs/superpowers/specs/2026-09-08-taskA0-schema-decisions.md`（2026-09-08 (9)）。
> 决策定稿与其理由、schema 契约、G1 失败测试清单、2.6 门状态均在该记录中；A1 spec 直接并入引用。

- [x] 决策 1（简化，偏离重设计文档）：`target_mode` 保持 **run 级**（charge/spin 混合的常态是双 delta）；逐约束 mode 待有真实需求再加。理由：YAGNI + 减少状态面。
- [x] 决策 2：JSON 新格式与向后兼容方案：

```json
{"constraints": [
  {"type": "charge", "target": 0.1, "atoms": [0]},
  {"type": "spin",   "target": 0.1, "atoms": [0]}
]}
```
   旧格式 `{"targets":[...],"atoms":[...]}` 自动按 run 级 `constraint_type` 转换 + deprecation WARNING（3 个注册用例零修改继续跑）。
- [x] 决策 3：mu_max 逐约束（保留重设计这一条——软/硬通道确实需要不同熔断上限，且 M4 已按分量处理）。
- [x] 产出：schema 决策记录（并入 A1 的 spec）。

## Task A1：M7 数据模型 + 解析 + 守卫（1 天）

> **状态（2026-09-08）**：已完成并通过 G1（spec `2026-09-08-taskA1-mixed-schema-io.md`）。
> 两处偏差登记：configure 层混合 run 阶段性 ERROR（A4 移除，防单通道 loop 静默错跑）；
> configure_from_inputs 签名推迟到 A4（specs 的真实消费方 loop 届时接线）。
> Step 4 sabotage 实测：每处守卫移除恰中 `MixedGuards` 1 个测试 FAIL（计划"恰 2 FAIL"
> 是断言数估计，实际该守卫在 T2 有独立断言且 v1 路径守卫分置，如实记录于 spec §3）。

**Files:** `constraint_io.{h,cpp}`、`test/constraint_io_test.cpp`

- [x] **Step 1 失败测试——新 schema 解析**：
```cpp
TEST(ConstraintIOTest, MixedConstraintListParsing)
{
    // 新格式：charge+spin 两条 → specs 正确（kind/atoms/target/chan 推导）；
    // 旧格式 + run 级 type → 自动转换 + deprecation 标记；
    // 嵌套/平铺 atoms、缺省 atoms、逐约束 mu_max 缺省回退 run 级。
}
```
- [x] **Step 2 失败测试——守卫**：
```cpp
TEST(ConstraintIOTest, MixedGuards)
{
    // spin 约束 + nspin=1 → ERROR；{"type":"dipole"} → ERROR "not implemented"；
    // 空 constraints → ERROR；atoms 越界 → ERROR；重复 kind+atoms 组合 → WARNING（近共线预警）
}
```
- [x] **Step 3 实现**：`ConstraintSpec { kind, atoms, chan, target, mu_max }`（ChannelProfile 由 kind 工厂推导：charge=(+1,+1,+1,+1)、spin=(+1,−1,+1,−1)）；`configure_from_inputs` 输出 `std::vector<ConstraintSpec>`。
- [x] **Step 4 测试通过 + sabotage**：移除 spin/nspin 守卫 → 恰 2 FAIL。
- [x] **Step 5 spec + 日志 + commit**。

**验证门 G1**：io 单测全绿 + 旧用例兼容测试通过。

## Task A2：M2 逐约束 channel 读数（0.5 天）

**Files:** `constraint_observe.{h,cpp}`、测试

- [ ] Step 1 失败测试——混合 mock：
```cpp
TEST(ConstraintObserveTest, MixedChannelReading)
{
    // 合成密度 ρ↑/ρ↓ 已知：charge 分量读 ρ↑+ρ↓、spin 分量读 ρ↑−ρ↓，
    // 同一 observe 调用内两分量各对拍解析期望（1e-12）；守恒和：charge 分量 Σ=N_el
}
```
- [ ] Step 2-3 实现：`observe` 签名改为接收 `std::vector<ChannelProfile>`（或 specs），按分量组合密度。
- [ ] Step 4 通过 + spec + commit。**G2**。

## Task A3：M3a 逐约束 channel 注入（0.5 天）

**Files:** `constraint_inject_pw.{h,cpp}`、测试

- [ ] Step 1 失败测试——**逐通道恒等式（核心不变量）**：
```cpp
TEST(ConstraintInjectPWTest, MixedObservableEqualsInjection)
{
    // 同一 specs 下：charge 分量 ∫ρ·(μ_c w_c) == μ_c·Q_c；
    // spin 分量 v_eff↑−v_eff↓ 含 2μ_s w_s、v_eff↑+v_eff↓ 不含 μ_s w_s；
    // 交叉零项：charge μ 不进自旋差势、spin μ 不进总势（逐点断言）
}
```
- [ ] Step 2-3 实现：按分量 `inj_up/inj_dn` 注入两个自旋势。
- [ ] Step 4 通过 + sabotage（翻转某分量 inj 符号 → 恰中 FAIL）+ spec + commit。**G3**。

## Task A4：M8 接线 + M5 审计标签（1 天）

**Files:** `constraint_loop.{h,cpp}`、`constraint_accounting.{h,cpp}`、测试

- [ ] Step 1 失败测试——loop 级混合收敛：
```cpp
TEST(ConstraintLoopTest, MixedConvergesOnLinearResponse)
{
    // 合成线性响应 Q_c(μ_c)、Q_s(μ_s) 独立通道：外环分别收敛；
    // 反假收敛：μ=0 自由跑不得收敛到非自然靶点（charge 与 spin 分量各自断言）；
    // 审计行含 kind 标签：c[0] kind=charge、c[1] kind=spin
}
```
- [ ] Step 2-3 实现：ConstraintLoop 持 `std::vector<ConstraintSpec>`，observe/inject 传 specs；审计行加 `kind=`。
- [ ] Step 4 通过 + spec + commit。**G4**。

## Task A5：M6 力核逐约束 channel（0.5 天）

**Files:** `constraint_deriv.{h,cpp}`、测试

- [ ] Step 1 失败测试：
```cpp
TEST(ConstraintDerivTest, MixedChannelForce)
{
    // 合成密度下 charge 分量用 ρ↑+ρ↓、spin 分量用 m，逐分量对拍解析期望；
    // 牛顿第三定律 Σ_J F_J ≡ 0（1e-10）；μ=0 分量短路
}
```
- [ ] Step 2-3 实现：力核签名加 channel 数组（d_α 按 read_up/read_dn 组合）。
- [ ] Step 4 通过 + spec + commit。**G5**。
- 注：混合力 FD 属 A6 后的验收轮（stationary4 协议，不抢跑）。

## Task A6：H₂O 混合集成用例（0.5 天）

**Files:** `tests/01_PW/213_PW_constraint_h2o_mixed/`（注册 CASES_CPU.txt）

- [ ] Step 1 用例：nspin=2、charge +0.1 e on O + spin +0.1 μB on O（同原子双类型——最直接的用户场景）；
- [ ] Step 2 验收：两分量均 CONVERGED（res<1e-4）；μ_c、μ_s 与单约束值（−0.1765 / −0.07234）同量级（耦合偏移如实记录——这是 P3 的实测数据）；maxdev=2.2e-16；result.ref 入库；
- [ ] Step 3 回归：211（charge）/212（spin）/212_NAO 三旧用例（旧格式兼容路径）逐位复现；
- [ ] Step 4 sabotage 复验 + `ctest -R constraint` 全绿；
- [ ] Step 5 spec（含 μ 偏移测量表——阶段 B 立项依据）+ 日志 + commit。**G6**。

## Task A7：文档与收尾（0.5 天）

- [ ] 用户手册 §2/§3：新 JSON 格式 + 旧格式 deprecation + 混合约束边界（偶极未实现、力 FD 前提不变）；
- [ ] 开发者指南：ConstraintSpec 数据模型 + 逐约束 channel 数据流更新 + 风险表刷新（channel run 级全局一项移除）；
- [ ] 进展总结文档更新；评审申请。

---

## 验证设计总表（稳步推进的检查点）

| 门 | 验证内容 | 失败处理 |
|---|---|---|
| G1 | schema/守卫/sabotage/旧格式兼容 | 不进 A2 |
| G2/G3 | 混合 mock 对拍、逐通道恒等式 | 不进下一步 |
| G4 | loop 混合收敛 + 反假收敛 + 审计标签 | 不进 A5 |
| G5 | 混合力解析对拍 + 牛三 | 不进 A6 |
| G6 | 集成收敛 + 三旧用例逐位回归 + 全量 ctest | 不闭合 |

**纪律**：μ 更新后密度不重置（热启动 §3.1 显式保证）；每 Task 的 spec 五段式；sabotage 与守卫一一对应；混合场景的 (μ,Q) 历史落盘（阶段 B 的 Jacobian 测量原料）。

**工期估计**：A1–A7 合计约 4 天（含验证），前置 2.6 闭合不计。
