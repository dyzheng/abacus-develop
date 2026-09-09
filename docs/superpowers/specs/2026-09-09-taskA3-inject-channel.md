# Task A3：M3a 逐约束 channel 注入（G3，TDD 轮）

> 批复：`docs/superpowers/specs/2026-09-09-taskA2-observe-channel-read.md`（A2 G2 通过，
> 批准启动 A3）。计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A3。
> 范围：`constraint_inject_pw.{h,cpp}` + `test/constraint_inject_pw_test.cpp`（轻量单测，无重算）。

---

## 1. 测试方案（Test plan）

TDD 环：失败测试先行（新 API 编译红）→ 实现 → 既有注入/约束全套测试绿 →
sabotage（逐 α channel 选择破坏 → 恰中混合注入测试）→ spec 回填 → 日志 → commit。

- **T1 `MixedChannelCrossZeroPointwise`**（交叉零项逐点断言，1e-12）：
  两约束片段 {{0},{1,2}}，profiles [charge, spin]，μ_c/μ_s 异号；
  注入后逐点：
  - 自旋差势 Δ=veff(0)−veff(1) 只含 spin 分量：Δ = Δ0 − 2μ_s·w_{1,2}（charge 两行同号互消）；
  - 总势 Σ=veff(0)+veff(1) 只含 charge 分量：Σ = Σ0 + 2μ_c·w_0（spin ± 互消）；
  - 即 **charge μ 不进自旋差势、spin μ 不进总势**。
- **T2 `MixedObservableEqualsInjection`**（逐通道恒等式，observable==injection
  operator）：
  - 5 约束片段 {{0},{1},{2},{1,2},{0}}、profiles
    [charge,charge,charge,spin,spin]、逐 α μ；
  - δ 密度探针（ρ↑=δ / ρ↑=ρ↓=δ，1e-12）：
    E=∫(ρ↑·veff_up + ρ↓·veff_dn) dV == Σ_α μ_α·Q_α(mixed observe)——单次混合调用内
    全局恒等式逐点成立；
  - 自旋分辨高斯密度（每原子 up/dn 已知、总量 10）：同一恒等式在平滑密度上
    成立（1e-10 相对）。
- **T3 回归**：既有 5 个注入测试（DensityChannel 旧入口，含 split spin）与全套
  `MODULE_ESTATE_constraint_*` 保持绿（legacy 入口改造为对逐约束核心的薄适配，
  单一注入实现）。

## 2. 测试设置（Setup）

- 平台：容器 gcc 单测（`build/`，feat/deltap HEAD 2ba46981e）。
- 轻量命令：`make MODULE_ESTATE_constraint_inject_pw -j8` →
  `ctest -R MODULE_ESTATE_constraint_inject_pw`；全量 `ctest -R MODULE_ESTATE_constraint`。
- 复用基建：H₂O 3 原子、rhopw 40³、Becke WeightGrid + `set_constraint_atoms`、
  δ-密度探针模式（PointwiseInjectionNspin1 / SplitInjectionSpin 同款）。

## 3. 结果（Results）

- 基线（A2 后）：`ctest -R MODULE_ESTATE_constraint` 11/11 PASS。
- 实现：`constraint_inject_pw.{h,cpp}`——新增逐约束入口
  `inject(wg, mu, vector<ChannelProfile>&, veff)` 为**唯一注入实现**（nspin=2：
  逐 α `veff(0)+=μ·inj_up·w`、`veff(1)+=μ·inj_dn·w`；nspin=1：仅 charge profile
  放行、只写行 0）；旧 `DensityChannel` 入口改为薄适配（homogeneous profile 列表 →
  委托），mu 长度与 spin+nspin=1 守卫保留在适配层与核心层。
- 测试：`MODULE_ESTATE_constraint_inject_pw` **7/7 PASS**（5 旧 + 2 新）；全套
  `MODULE_ESTATE_constraint` **11/11 PASS**；estate 生产库 `elecstate` 编译通过。
- **sabotage（S1）**：逐 α channel 选择破坏为恒用 `channels[0]` →
  恰中 **2 个新混合测试** FAIL（MixedChannelCrossZeroPointwise +
  MixedObservableEqualsInjection），5 个旧测试全绿。

## 4. 分析（Analysis）

- 单实现原则与 A2 同构：混合注入与 legacy 单通道共用同一注入循环，
  channel/μ 只是逐 α 参数——spin 的 ±λ 拆分与 charge 的总势耦合均出自
  `build_channel_profile` 一套符号（A1 工厂 + A2 读数同源），无第二注入路径。
- G3 断言的两层：① 交叉零项逐点（charge μ 不进差势、spin μ 不进总势，
  Σ/Δ 结构 1e-12）直接锁定"不串道"；② observable==injection operator 全局
  恒等式 `E=∫(ρ↑V↑+ρ↓V↓) ≡ Σμ_αQ_α`（δ 探针 1e-12 + 平滑自旋密度 1e-10 相对）——
  与 A2 混合读数共用同一 (w, chan)，闭环验证注入=观测同一算子。
- 守卫保持 return-false 契约（不动 buffer）：mu/channels 长度与权重网格
  nconstraint 不匹配、spin profile 落入 nspin=1 buffer，均拒绝；loop 侧
  调用点语义不变（A4 换 specs 时接线）。
- 与 A2 的接口对称：observe/inject 都收 `vector<ChannelProfile>`（非整
  ConstraintSpec），specs→profiles 映射留在调用方（A4 loop 一次性做）。

## 5. 下一步（Next Steps）

- 提交本 Task 后推送 zdy；向用户汇报请求裁决。
- 批准后启动 **Task A4**（M8 接线 + M5 审计标签，G4：loop 混合收敛 + 反假收敛 +
  kind= 审计标签）。**评审提醒（A1-review 入档）届时执行**：
  ① 移除 `constraint_io.cpp` staging guard（混合拒绝）必须先写 G4 混合收敛失败
  测试（移除前红）→ 移除后绿 → sabotage（恢复守卫恰中 FAIL）证明移除受控；
  ② 改 `configure_from_inputs` 签名时 PW/LCAO 双 esolver 调用点同步修改。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口（A 阶段工具补件）；
  ② torque 脚本 E' 修复；③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`constraint_inject_pw.{h,cpp}`（逐约束 channel 注入核心 + legacy 适配）；
  `test/constraint_inject_pw_test.cpp`（+2 tests）。轻量单测，无重算。
- 判定：**G3 通过**（逐通道恒等式 + 交叉零项逐点断言 + S1 sabotage 恰中）。
