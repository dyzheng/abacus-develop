# Task A4：M8 总控接线 + M5 审计 kind 标签（G4，TDD 轮）

> 批复：`docs/superpowers/specs/2026-09-09-taskA3-review.md`（A3 G3 通过，A4 解锁；
> 两条验收纪律重申为 A4 验收项）。计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A4。
> 范围：`constraint_loop.{h,cpp}`、`constraint_io.{h,cpp}`、`constraint_accounting.{h,cpp}`、
> `mu_solver.{h,cpp}`、PW/LCAO 双 esolver 调用点、相关测试（轻量单测 + 编译验证，无重算）。

---

## 1. 测试方案（Test plan）

TDD 环：失败测试先行（新 API 编译红）→ 实现 → 守卫在位红（可证伪第 1 步）→
守卫移除绿 → 2×sabotage 恰中 → spec 回填 → 日志 → commit。

- **T1 `ConstraintLoopTest.MixedConvergesOnLinearResponse`**（G4 门，经扩展 configure
  核取 specs —— A1 staging guard 所在地）：
  - v2 JSON `[charge 0.01 @{0}, spin 0.02 @{1,2}]` → `configure_constraint`（13 参，
    specs 出参）必须 **OK**（守卫在位时此 ASSERT 红）→ specs 驱动 `loop.init`；
  - **反假收敛**：μ=0 自由跑不得命中任一靶点（charge Q_ref≠t、spin m_ref=0≠t），
    首步 secant 两分量 μ 均 <0；
  - **独立线性响应收敛**：charge 斜率 −1 → μ_c*=−0.01；spin 拆分注入斜率 −2 →
    μ_s*=−0.01；终态 CONVERGED / DONE / conv=true；两分量 Q 落在各自 target；
  - **M5 kind= 标签**：审计行含 `c[0] kind=charge`、`c[1] kind=spin`。
- **T2 `ConstraintLoopTest.MixedFuseHonorsPerComponentCap`**（A0 D3 逐约束 cap 落位）：
  - 异构 cap JSON `[charge 0.01@{0} mu_max 5.0, spin 0.5@{1,2} mu_max 0.05]`；
    spin 根 μ*=−0.25 超出自身 0.05 cap → 钉在 **−0.05** 并 plateau 熔断
    UNREACHABLE；charge 独立收敛 −0.01 不受 spin cap 影响。
  - 判别力：若实现退化为"标量取首约束 cap"（5.0），spin 会跑到 −0.25 收敛 → 测试 FAIL。
- **T3 io `MixedGuards` 扩展断言**（守卫移除的 io 侧观测口）：
  - 扩展核（13 参 + specs）对 mixed / 异构 cap 均 **OK** 且 specs 完整；
  - legacy 11 参入口对 mixed / 异构 cap 仍 **ERROR**（expressibility guard 移到此处）。
- **T4 回归**：既有 10 个 io 测试 + 全部 legacy loop 测试（含 spin、fuse、force 对拍）
  保持绿；双基组编译验证（PW+LCAO esolver 调用点同步改）。

## 2. 测试设置（Setup）

- 平台：容器 gcc 单测（`build/`，feat/deltap HEAD 9c540f104 + 评审轮 2 笔未提交）。
- 轻量命令：`make MODULE_ESTATE_constraint_{loop,io,accounting,mu_solver} -j8` →
  `ctest -R MODULE_ESTATE_constraint`（11 靶，~40 s）；生产库编译
  `make esolver elecstate -j8`（ENABLE_LCAO=ON → esolver_ks_pw.cpp 与
  esolver_ks_lcao.cpp 双调用点同批编译）。
- 复用基建：H₂O 3 原子、rhopw 40³、`make_cfg` legacy 入口（8 个旧 loop 测试原样）；
  混合 mock 用 loop **自带** WeightGrid（fragments {{0},{1,2}}）构造
  Σ=ρ_ref−μ_c·w0/S_c、m=−2μ_s·w12/S_m（S=∫w²dV），ρ↑=(Σ+m)/2、ρ↓=(Σ−m)/2
  ——Q_c 只见 μ_c、Q_s 只见 μ_s，解耦由构造保证。

## 3. 结果（Results）

- 基线（A3 后）：`ctest -R MODULE_ESTATE_constraint` 11/11 PASS。
- **守卫移除可证伪链**（本次提交实测顺序）：
  1. **红①（新 API 缺失）**：G4 测试引用 specs 版 `loop.init` / 6 参
     `configure_from_inputs` 前不存在 → 编译红（A1/A2/A3 同款 TDD 起始态）。
  2. **红②（守卫在位）**：实现完成但保留 A1 staging guard（mixed 拒绝在扩展核内）
     → `MixedConvergesOnLinearResponse` 于 configure ASSERT 失败
     （error="...loop wiring (Tasks A2-A4)..."）；io `MixedGuards` 扩展 OK 断言同红。
  3. **绿（守卫移除）**：扩展核不再拒绝 mixed/异构 cap（expressibility guard 移入
     legacy 11 参入口）→ T1/T2/T3 全绿。
  4. **sabotage A（恢复 core 守卫）** → 恰中 **2 测试**：`MixedConvergesOnLinearResponse`
     （loop_test.cpp:402 断言 OK 失败）+ `MixedGuards`（io_test.cpp:527），其余全绿。
  5. **sabotage B2（逐分量 cap 退化为标量首 cap）** → 恰中 **1 测试**：
     `MixedFuseHonorsPerComponentCap`（mu[1] 收敛到 −0.25 而非钉 −0.05）；
     legacy `FuseUnreachable` 保持绿（判别力实证：同 cap 场景经 legacy cfg 路径不受影响）。
- 实现后测试：全套 `MODULE_ESTATE_constraint` **11/11 PASS**（loop 靶内 +2 新测试、
  io 靶扩展断言）；`esolver` + `elecstate` 生产库编译通过（ENABLE_LCAO=ON）。

## 4. 分析（Analysis）

- **M8 接线后单路径原则**：`ConstraintLoop` 两条 init（legacy 从 cfg 推导 homogeneous
  specs / A4 specs 直入）收敛到同一份 `specs_`，observe/inject/add_back/audit/force
  全部只读 `specs_` 派生的平行数组 `kinds_/channels_/mu_caps_` —— 无第二读数/注入路径，
  与 A2/A3 "单一实现 + 薄适配" 同构。
- **守卫处置决策**：mixed 与异构 cap 的"表达力守卫"从扩展核移到 **legacy 11 参入口**
  （该入口丢弃 specs、只能产出单类型 cfg，混合会静默错跑）；specs 出参的
  `configure_from_inputs`（esolver 唯一生产入口）不受限。io 测试对 legacy 路径的原
  mixed-ERROR 断言因此保持绿（语义未丢，只是落点随消费方移动）。
- **MuSolver 逐分量 cap**（D3 落位）：`MuSolverParams` 增 `mu_max_per_component`
  （平行于分量列表；空 → 标量回退），pin/fuse 判定按 cap_i。loop 每次 init 从 specs
  取 cap 向量；legacy 同 cap 场景退化为逐分量同值，位级不变（`FuseUnreachable` 绿）。
- **compute_force 混合组合（A5 前的诚实中间态）**：M6 核仍单 channel 签名；loop 按
  kind 拆两趟（charge 掩码 / spin 掩码），核线性于 μ 且零 μ 短路 → 同构场景与历史
  单调用逐位一致（`ComputeForceConvergedMatchesKernel` EXACT 通过）。A5 再落
  "力核签名加 channel 数组"。
- **M5 kind= 审计**：`ConstraintAccounting::audit` 增 kinds 重载（legacy 5 参入口保持
  空 kinds、输出逐位不变）；审计行 c[i] 后加 `kind=charge|spin`；`kind_to_type_string`
  从 io.cpp 匿名空间提为公开（审计与 cfg.type 词汇同源，防漂移）。
- **configure_from_inputs 签名**（A1 偏差 2 兑现）：`(cfg, specs, ucell, radii, error)`
  specs 出参；PW/LCAO 双 esolver 调用点同步修改 + 双基组编译验证（ENABLE_LCAO=ON）。
- 范围纪律：未触偶极/Broyden/松紧 SCF；未抢跑 A5 力核签名；测试驱动 mock 含 A6
  同原子双类型片段形状（{0} 与 {1,2} 重叠）但 cap 语义测试独立。

## 5. 下一步（Next Steps）

- 提交本 Task（含评审轮 2 笔文档：A3-review spec + dev-log 编号顺延/重排）后推送 zdy；
  向用户汇报请求裁决。
- 批准后启动 **Task A5**（M6 力核逐约束 channel，G5：混合力对拍 + 牛三 + μ=0 短路；
  力核签名加 channel 数组后收回 A4 的两趟掩码组合）。A4 遗留说明：compute_force
  混合组合仅经 loop 层（无直接单测——A5 换签名后由 G5 直接覆盖）。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口；② torque 脚本 E' 修复；
  ③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`constraint_loop.{h,cpp}`（specs 接线 + 逐约束 channels/kinds/caps +
  kind= 审计 + force 两趟组合）、`constraint_io.{h,cpp}`（configure_from_inputs specs
  出参 + 守卫移入 legacy 入口 + kind_to_type_string 公开）、`constraint_accounting.{h,cpp}`
  （kinds 重载 + kind= 标签）、`mu_solver.{h,cpp}`（逐分量 mu_max）、
  `esolver_ks_{pw,lcao}.cpp`（双调用点同步）、`test/constraint_loop_test.cpp`（+2 tests）、
  `test/constraint_io_test.cpp`（MixedGuards 扩展 + ConfigureFromInputsShared 新签名）、
  `test/CMakeLists.txt`（accounting 靶补 constraint_io.cpp 源）。轻量单测，无重算。
- 判定：**G4 通过**（混合收敛 + 反假收敛 + kind= 审计标签 + 逐约束 cap 落位；
  守卫移除经 红②→绿→sabotage A/B2 恰中 可证伪；双基组编译通过）。
