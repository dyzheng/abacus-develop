# Task A2：M2 逐约束 channel 读数（G2，TDD 轮）

> 批复：`docs/superpowers/specs/2026-09-08-taskA1-review.md`（A1 G1 通过，批准启动 A2）。
> 契约源：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A2；A0 D4 数据模型。
> 范围：`constraint_observe.{h,cpp}` + `test/constraint_observe_test.cpp`（轻量单测，无重算）。

---

## 1. 测试方案（Test plan）

TDD 环：失败测试先行（新 API 编译红）→ 实现 → 既有 observe/约束全套测试绿 →
sabotage（逐分量读数守卫/符号破坏须恰中 FAIL）→ spec 回填 → 日志 → commit。

- **T1 `MixedChannelReading`**（delta 密度钉，1e-12 对拍，**同一 observe 调用内两分量**）：
  - 权重网格片段 `[[0],[1],[2],[1,2],[0]]`（5 条约束），逐约束 profiles
    `[charge,charge,charge,spin,spin]`——α3 spin 与 α1/α2 charge 片段重叠、
    α4 spin 与 α0 charge 同原子（A6 场景形状）；
  - `ρ↑=δ(g*), ρ↓=0` → 每条 Q_α = w_α(g*)（charge 读 ρ↑+ρ↓、spin 读 ρ↑−ρ↓，
    此处相同）；`ρ↑=ρ↓=δ` → charge 读 2w_α、spin 读 0；多探针逐点 1e-12；
  - 意义：一次性锁定逐约束 channel 选择、索引映射、dV 因子与 dV 权重。
- **T2 `MixedChannelConservation`**（合成平滑密度）：
  - 自旋分辨高斯 ρ↑/ρ↓（每原子分量幅值已知，总 N_el=10）：同一混合调用内
    **charge 分量 Σ= N_el（1e-8）**（charge 片段构成 partition-of-unity）；
    spin 片段 α3 磁矩 ≈ m_H1+m_H2、α4 ≈ m_O（网格求积容差，量级/符号对拍）；
  - 常数密度 ρ↑=ρ↓=1：charge 分量 Σ=2·Ω（1e-8）、spin 分量逐条 =0（1e-12）。
- **T3 回归**：既有 observe 测试（DensityChannel 旧入口）与全套
  `MODULE_ESTATE_constraint_*` 保持绿（legacy 入口改造为对逐约束核心的薄适配，
  单一求积实现，不引入第二套读数）。

## 2. 测试设置（Setup）

- 平台：容器 gcc 单测（`build/`，feat/deltap HEAD 14fabba2c）。
- 轻量命令：`make MODULE_ESTATE_constraint_observe -j8` →
  `ctest -R MODULE_ESTATE_constraint_observe`；全量 `ctest -R MODULE_ESTATE_constraint`。
- 复用的既有基建：`make_h2o_ucell`（3 原子）、rhopw 40³（h=0.5 Bohr）、
  Becke WeightGrid + `set_constraint_atoms`（build 前设置片段映射）、
  delta-密度探针模式（PointwiseDeltaReading / SpinChannelMagnetizationReading）。

## 3. 结果（Results）

- 基线（A1 后）：`ctest -R MODULE_ESTATE_constraint` 11/11 PASS。
- 实现：`constraint_observe.{h,cpp}`——新增逐约束入口
  `observe(wg, rho, nspin, const std::vector<ChannelProfile>&, Q)` 为**唯一读数
  实现**；旧 `DensityChannel` 入口改为薄适配（homogeneous profile 列表 →
  委托逐约束核心），保留 spin+nspin=1 前置守卫。头部 include constraint_io.h
  （ChannelProfile 契约来源）。
- 测试：`MODULE_ESTATE_constraint_observe` **7/7 PASS**（5 旧 + 2 新）；全套
  `MODULE_ESTATE_constraint` **11/11 PASS**；estate 生产库 `elecstate` 编译通过。
- CMake：observe/deriv/inject_pw 三个测试靶补 `../constraint_io.cpp`（observe 现
  依赖 `build_channel_profile`；loop 靶本已含）。
- **sabotage（S1）**：逐约束 channel 选择破坏为恒用 `channels[0]` →
  恰中 **2 个新混合测试** FAIL（MixedChannelReading + MixedChannelConservation），
  5 个旧测试全绿（homogeneous 适配路径不受影响）——证明逐分量读数是新测试
  专属的断言面，旧读数语义无回归。

## 4. 分析（Analysis）

- 单实现原则：混合读数与 legacy 单通道共用同一个求积循环，channel 只是
  逐 α 参数——不存在第二套读数路径可供漂移（评审"勿两源"精神落实在实现层）。
- observe 只收 `ChannelProfile` 列表而非整 `ConstraintSpec`：M2 只关心 (w, chan)
  对，target/mu_max/atoms 与读数解耦；specs→profiles 映射留给调用方（A4 loop
  接线时一次到位）。
- 硬守卫两条（防 (w, chan) 静默错配）：`channels.size() != nconstraint` → 立即
  WARNING_QUIT；nspin=1 下出现 spin 型 profile（`read_dn != read_up`）→ 立即
  WARNING_QUIT（生产路径已有 parse 级守卫，此为纵深防御）。
- G2 守恒和的设计修正（诚实边界）：曾拟"高斯自旋密度下 spin 片段 ≈ 原子磁矩
  和"，但 Becke 盆地跨原子漏磁使该式**不是干净恒等式**——弃用；改为三个干净
  断言：① δ 探针逐 α 精确钉读数算子（1e-12）；② 自旋分辨平滑密度（每原子
  up/dn 已知、总量 10）下 charge 分量（fragments {0},{1},{2} 构成 partition-of-
  unity）Σ=N_el（1e-8）+ 逐 charge 分量对拍独立高阶求积 reference_charges
  （<0.05，与既有 AtomicSuperposition 同容差）；③ 常数密度 ρ↑=ρ↓=1 下 spin 分量
  精确为零（1e-12）、charge Σ=2Ω（1e-8）。
- 测试片段含"同原子双类型"形状（α0 charge on {0} 与 α4 spin on {0} 重叠；
  α3 spin {1,2} 与 α1/α2 charge 重叠）——A6 集成场景的读数面提前锁定。

## 5. 下一步（Next Steps）

- 提交本 Task（含评审轮 2 笔文档：A1-review spec + dev-log 条目，dev-log 中
  评审条目编号由重复 (13) 顺延为 (14)）后推送 zdy；向用户汇报请求裁决。
- 批准后启动 **Task A3**（M3a 逐约束注入，G3：逐通道恒等式 + 交叉零项逐点断言；
  注入器签名接 channel 数组，PW 侧先行）。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口（A 阶段工具补件）；
  ② torque 脚本 E' 修复；③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`constraint_observe.{h,cpp}`（逐约束 channel 读数核心 + legacy 适配）；
  `test/constraint_observe_test.cpp`（+2 tests）；`test/CMakeLists.txt`
  （observe/deriv/inject_pw 补 constraint_io.cpp 源）。轻量单测，无重算。
- 判定：**G2 通过**（混合 mock 对拍 1e-12 + 守恒和断言 + S1 sabotage 恰中）。
