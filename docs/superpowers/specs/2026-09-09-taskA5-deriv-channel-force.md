# Task A5：M6 力核逐约束 channel（G5，TDD 轮）

> 批复：`docs/superpowers/specs/2026-09-09-taskA4-review.md`（A4 G4 通过，A5 解锁；
> G5 判据=解析对拍+牛三+零乘子短路，同构回归不漂移，混合力 FD 不抢跑）。
> 计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A5。
> 范围：`constraint_deriv.{h,cpp}`（per-α 力核签名）、`constraint_loop.cpp`（compute_force
> 收回 A4 两趟掩码）、`constraint_deriv_test.cpp`（+2 测试）。轻量单测，无重算。

---

## 1. 测试方案（Test plan）

A5 把 A4 loop 侧的"两趟掩码组合"收回内核：新增 per-α 重载
`constraint_force(wg, rho, nspin, vector<ChannelProfile>&, mu, force)`，每个 α 用自身
`read_up/read_dn` 折叠密度（charge=ρ↑+ρ↓、spin=ρ↑−ρ↓），一条调用服务混合列表。
实现由上一轮 A4 交接带入（本批先审查完整性再补测试），红绿可证伪性以 **sabotage 恰中**
落实（见 §3）。

- **T1 `ConstraintDerivTest.MixedChannelForce`**（G5 核心：逐分量解析对拍 + 零乘子短路）：
  - 合成 spin-resolved 密度（up={6.0,0.8,0.9}、dn={2.0,0.2,0.1}，up+dn={8,1,1}=nelec_atom，
    磁化 m={4,0.6,0.8} **不与总电荷成比例** —— spin 折叠非平凡）；
  - fragment 映射 `{{0},{1,2}}`，channels `{charge, spin}`，μ={0.1,−0.05}；
  - **分量短路**：F(μ_c,μ_s) ≡ F(μ_c,0)+F(0,μ_s)（1e-12，核线性于 μ、零乘子逐 α 短路）；
  - **同构不漂移**：混合调用 ≡ legacy 单通道叠加（Charge{0.1,0} + Spin{0,−0.05}，1e-12）；
  - **逐分量独立解析锚**：M0 积分 quadrature（本批给 `f_quadrature` 加 per-atom
    `dens_coef` 参数）——charge 分量对总密度 {8,1,1}、spin 分量对磁化 {4,0.6,0.8} 且
    乘子撒到 {1,2}（fragment 导数网格=逐原子导数求和，weight_grid.cpp 既有保证），1e-8；
  - 全零 μ 一次混合调用 → 逐元素精确 0（`EXPECT_DOUBLE_EQ`）。
- **T2 `ConstraintDerivTest.MixedForceNewtonThirdLaw`**（G5 牛三，混合原生形态）：
  - 64³ 本地网格，同 spin-resolved 密度，fragment 映射同上；
  - 全局刚性平移下网格恒等式 Σ_J F_J = −Σ_α μ_α dQ_α/dt；RHS 用 **混合 observer**
    （`ConstraintObserver::observe(wgs, rho, 2, channels, q)`，observe.cpp 独立折叠路径）
    5-point FD（h=1e-3，同 NewtonThirdLaw 先例）；
  - 断言 1e-9；该配置网格自洽量级由单通道先例（同网格/同 h 实测 ~7e-13）支撑。
  - 设计作用：observer 是独立于力核的第二折叠实现 → 力核折叠 bug（spin 误读总电荷）
    必然破坏恒等式 —— T2 是 per-α 折叠的**可证伪器**。
- **T3 回归**：legacy 单通道 deriv 测试 4 个（合成密度解析对拍/牛三/线性/自旋读数）保持绿；
  loop 全套（G4 的 loop→力核对拍 EXACT 路径改走 per-α 内核）+ 全模块 `ctest` 11 靶；
  生产库 `elecstate`/`esolver`（ENABLE_LCAO=ON，PW+LCAO）增量编译通过。

## 2. 测试设置（Setup）

- 平台：容器 gcc 单测（`build/`，feat/deltap HEAD f268a2cfc + A4 评审 2 笔未提交）。
- 轻量命令：`make MODULE_ESTATE_constraint_deriv -j8` →
  `./.../MODULE_ESTATE_constraint_deriv`（6 tests，~43 s）；
  回归 `ctest -R MODULE_ESTATE_constraint`（11 靶，~50 s）；
  生产库 `make elecstate esolver -j8`（增量，改到 constraint_deriv/loop 两 TU + 依赖重编）。
- 复用基建：H₂O-offboundary 3 原子（9.4/12.6/6.2 网格安全几何）、120³ 网格（fixture，
  1/6 Bohr）、64³ 本地网格（FD 腿，同 NewtonThirdLaw）；`fill_rho_spin_resolved` 与
  `fill_mixed_channels` 为本批 helper（A5 交接已加，未验证前闲置）；`f_quadrature` 增
  `dens_coef` 参数（本批，既有调用点同参数化）。

## 3. 结果（Results）

- **审查**：A5 交接半成品（per-α 内核 + loop 单趟调用 + helper）完整：
  legacy 单通道入口→薄适配 homogeneous profiles→同内核；守卫（mu/channels 错配、
  spin 落 nspin=1、buffer 尺寸、derivatives_built、零 μ 短路、`#ifdef __MPI` 归约）
  无重复、无遗漏；`build_channel_profile` 出厂符号恰为 ±1 ⇒ `ρ↑±ρ↓` canonical buffer
  折叠精确（与 observer/injector 同一契约）。`f_quadrature` 通用化后 compile clean。
- **实现后测试（本批实测）**：`MODULE_ESTATE_constraint_deriv` **6/6 PASS**
  （ForceOnSyntheticDensity 11.4 s / NewtonThirdLaw 3.1 s / ForceLinearInMu 8.1 s /
  SpinChannelReadsMagnetization 8.0 s / **MixedChannelForce 8.9 s** /
  **MixedForceNewtonThirdLaw 3.2 s**）。
- **sabotage（可证伪）**：临时把"混合列表的 spinlike α 一律读 charge buffer"
  （仅 `need_charge && need_spin` 触发，同构列表路径不受影响）→ 恰中 **2 个新测试**
  FAIL（MixedChannelForce 分解断言、MixedForceNewtonThirdLaw 恒等式），legacy 4 测试
  全绿；还原后 6/6 绿。
- **回归**：`ctest -R MODULE_ESTATE_constraint` **11/11 PASS**（~50 s）；生产库
  `elecstate` + `esolver`（ENABLE_LCAO=ON，PW/LCAO 双基组）增量编译通过（esolver_ks_pw
  与 esolver_ks_lcao 因 header 依赖重编，无错误）。

## 4. 分析（Analysis）

- **单一实现原则**（与 A2/A3/A4 同构）：per-α 重载是唯一力核实现；legacy `DensityChannel`
  入口薄适配为 homogeneous profile 列表委托之——homogeneous 列表逐位等于历史单调用
  （`ComputeForceConvergedMatchesKernel` 与 deriv 旧 4 测试共同钉住），混合列表新增折叠
  路径无第二套积分代码可漂移。
- **折叠实现**：出厂 profile 只可能是 charge(+1,+1)/spin(+1,−1)，故按 `read_dn==read_up`
  区分并至多缓存两个 canonical 密度 buffer（总电荷/磁化），内层仍是单连续数组逐 α 直读，
  无 O(nalpha×nrxx) 中间量。nspin=1 时仅 charge-like 通过守卫，spin 语义与
  observer/injector 三端同契约（同一 `build_channel_profile` 工厂，防错配）。
- **守卫覆盖**：mu/channels 尺寸错配、force buffer 非 nat×3、导数网格未建、spin 落
  nspin=1 —— 全部 loud-abort；零乘子逐 α 短路保留（线性于 μ 的累加语义不破坏）。
- **解析锚的可信度**：quadrature 逐分量参考（1e-8）复用既有 ForceOnSyntheticDensity 的
  固定点 M0 导数核；charge 分量密度恰为 nelec_atom 高斯叠加（up+dn={8,1,1}），与
  quadrature 的既有验证对象一致；spin 分量经 fragment 导数=逐原子导数求和的性质把乘子
  撒到 {1,2}，1e-8 断言实测通过（与单通道解析锚同量级）。
- **牛三判据的口径说明**：A4 评审文本写"Σ_J F_J ≡ 0（1e-10）"。字面 ΣF≡0 只对常数密度
  成立（非均匀密度下约束还推密度场）；本批采用与既有 NewtonThirdLaw 相同的**网格恒等式**
  形态 Σ_J F_J = −Σ_α μ_α dQ_α/dt（RHS=同网格混合 observer 的 5-point FD），断言 1e-9，
  该配置实测自洽量级由单通道先例 ~7e-13 支撑——比字面形式更强（检验求和方向而非只验零）。
  判据变更与理由如实登记。
- **范围纪律**：未触偶极/Broyden/松紧 SCF；混合力 FD（stationary4 协议）未抢跑；
  A6 需要的力核语义（逐分量折叠）已在 G5 层面钉住，A6 集成只验整机路径。

## 5. 下一步（Next Steps）

- 提交本 Task（含 A4 评审归档：review spec 入库、dev-log (3)→(20) 重编号并移至 (19) 后）
  并推送 zdy；向用户汇报请求裁决。
- 批准后启动 **Task A6**（H₂O 混合集成用例 `tests/01_PW/213_PW_constraint_h2o_mixed`，
  nspin=2，charge+spin 同原子；验收：两分量 CONVERGED res<1e-4、μ 偏移
  vs 单约束 μ_c=−0.1765/μ_s=−0.07234 如实记录、(μ,Q) 历史落盘——阶段 B Broyden 立项
  实测；回归 211/212/212_NAO 三旧用例逐位复现；sabotage 复验；`ctest -R constraint` 全绿）。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口（A 阶段工具补件）；
  ② torque 脚本 E' 口径修复；③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`constraint_deriv.h`（per-α 重载声明）、`constraint_deriv.cpp`（per-α 折叠
  内核 + legacy 薄适配）、`constraint_loop.cpp`（compute_force 单趟 per-α 调用，A4 两趟
  掩码收回）、`constraint_loop.h`（compute_force 注释）、`test/constraint_deriv_test.cpp`
  （+2 tests：MixedChannelForce / MixedForceNewtonThirdLaw；helper
  fill_rho_spin_resolved/fill_mixed_channels；f_quadrature 加 dens_coef）。
- 文档：spec（本文件）；计划 A5 勾选；dev-log (20)=A4 评审重编号归档 + (21)=A5 完成。
- 验证命令与时长：deriv 靶 ~43 s（6/6）；模块 `ctest -R MODULE_ESTATE_constraint` ~50 s
  （11/11）；生产库增量编译（PW+LCAO）通过。
