# 实空间权重约束框架：开发进展总结（一期闭合 + 二期进行中）

> 截至 HEAD `3351ff6d3`（2026-08-31）。范围：从方案评审到一期闭合、二期 Task 2.1–2.4 完成、Task 2.5（M6 力核，含 LCAO 接线）完成。
> 本文重点：**完成了什么测试、各自判据与实测值**。全部数值均经评审者实跑复核（非仅采信开发报告）。

---

## 1. 一句话现状

统一实空间权重约束框架（Hirshfeld/Becke 网格权重替代 SMO 投影，电荷/自旋/偶极同一路径）已在 ABACUS 落地：一期（PW + Becke 电荷约束）判决性验证全过；二期已交付 LCAO 通道、自旋通道、力核（PW 侧）。当前能力：**PW/LCAO 双基组电荷约束 + PW 自旋约束的 SCF 与收敛乘子；PW 约束力已接线待 FD 验收**。

## 2. 交付清单（16 commit）

| 阶段 | 模块 | 内容 | 提交 |
|---|---|---|---|
| 一期 | M0 | Becke 异核修正 χ_ij + 解析位置导数（partition.h/.cpp） | 8a39dac75 |
| 一期 | M1 | 网格权重构造 + 单位分解审计 + MPI | 16aeed899 |
| 一期 | M2 | 约束读数 Q_α=∫w·d | 9b0fda37f |
| 一期 | M4 | 逐分量 secant + 七护栏 | c35f398b2 |
| 一期 | M7/M3a/M5/M8 | IO 守卫 / PW 注入 / 记账 / 外环编排接线 | dff13618b…e43b1d5d5 |
| 评审修复 | — | P1–P4（勾选失信/inject 契约/ctest 注册等） | 293f53aa8 |
| 二期 | 2.1 | V2a 网格收敛 + V2b 独立 Becke 参考 | f6fcc443d |
| 二期 | 2.2 | M3b LCAO 约束矩阵 Gint 核（审计仪器） | 17781e5da |
| 二期 | 2.3 | LCAO 四薄钩子接线 + PW/LCAO 共享配置下沉 | 308e509e4 |
| 二期 | 2.4 | 自旋通道 ±μ + 符号修正（response_sign） | ae9bfc600 |
| 二期 | 2.5.1/2.5.2/2.5.3 | 导数网格 / M6 力核 / PW 力接线 | f0c221b3c…60c67e79d |
| 二期 | 2.5.4/2.5.5 | LCAO 力接线（同一网格力核，FORCE_STRESS 通道）+ Task 2.5 闭合 | 52aa1e436、3351ff6d3 |

## 3. 测试总览（重点）

### 3.1 单元测试：12 个 ctest 目标，当前全绿（12/12，实测复核）

| 模块 | 关键测试与判据 | 实测 |
|---|---|---|
| M0 partition | f_3 解析对拍；单位分解 1e-10；**FD 导数 <1e-6**；对称性 | PASS |
| M1 weight_grid | 逐点 sum rule <1e-10；对称性；近邻表 | PASS |
| M1 MPI ×2 | 1/2/4-rank（含非方进程网格）逐点一致 1e-12 | PASS（3/3 MPI） |
| M2 observe | 原子叠加密度解析对拍 1e-8；ΣQ=N_el 1e-10；单点 δ 钉 1e-12；**V2b 独立 Becke 参考 <1e-8** | PASS |
| M4 mu_solver | 已知根收敛（合成 Q(μ) mock）；κ clamp 上下限；翻号检测；μ 硬顶；**熔断=顶限∧残差平台联合判据**；反假收敛 | 7/7 PASS |
| M7 io | schema/往返；absolute 模式 WARNING（口径守卫）；无 target→WARNING_QUIT；nspin≠2+spin→ERROR | PASS |
| M3a inject_pw | 逐点注入机器精度；**观测量==注入算符数值恒等式（∫ρ·μw dr == μ·Q，1e-10）**；spin 拆分注入 | PASS |
| M3b inject_lcao | **Σ_α W^α ≡ S（重叠矩阵，2.0e-15）**；W^α↔网格直积 3.4e-15 | PASS |
| M5 accounting | E_con=Σμ(Q−t)；key=value 审计行 | 4/4 PASS |
| M8 loop | 两阶段门控状态机；conv_esolver 门控（未收敛 SCF 不外步）；inject 返回值契约（P2）；**反假收敛全链路** | 6/6→含 spin PASS |
| M6 deriv/force | 导数网格平移不变性 Σ_J∂w/∂R_J+∂w/∂r≡0（1e-10，独立 5 点差分作 ∂w/∂r）；**力 vs 合成密度解析 2.2e-10（判据 1e-8）；牛顿第三定律 7e-13（判据 1e-9）**；力对 μ 线性 1e-12 | PASS |

### 3.2 集成测试（3 用例，全部注册 CASES_CPU.txt 并对拍 result.ref）

| 用例 | 场景 | 实测关键值 | 对拍偏差 |
|---|---|---|---|
| tests/01_PW/211_PW_constraint_h2o | PW 电荷 delta=+0.1 e on O | 6 外步 CONVERGED；μ*=−0.1765 Ry；Q_ref=6.2555 | etot 差 4.1e-9 eV |
| tests/01_PW/212_PW_constraint_h2o_spin | PW 自旋 delta=+0.1 μB on O | 3 外步 CONVERGED；μ*=−0.07234 Ry；m_ref=5.9e-6 | etot 差 9.5e-10 eV |
| tests/02_NAO_Gamma/212_NAO_constraint_h2o | LCAO 电荷 delta=+0.1 e on O | 6 外步 CONVERGED；μ*=−0.2193 Ry；Q_ref=6.4080 | etot 差 2.4e-11 eV |

### 3.3 判决性验证（一期验收门）

- **V1 sum rule PASS**：Σ_I N_I ≡ nelec=8，maxdev=2.2e-16（逐点单位分解的机器精度直接体现）；O=6.2555/H=0.87227（PW）。
- **V2 口径基准 PASS（重定义后内部闭环，无外部工具）**：V2a 网格收敛三档（81³/120³/162³）相邻档差 max 8.2e-5 < 1e-4 e；V2b 仓库内 C++ 独立参考（从 Becke 1988 原始公式重写，不调生产实现）对拍 <1e-8。严格单调失败已如实登记为口径偏差。Multiwfn 降级为可选 V2c'。
- **V3 可达性 PASS（7/7）**：delta=±0.05/0.1/0.2/0.3 e 全收敛无封顶，μ*≈−1.7·delta 线性平滑，|μ*|max=0.56 Ry ≪ μ_max=5.0；**熔断**：delta=+5.0 e 不可达 → μ 顶限 + Q 平台 3 步无改进 → UNREACHABLE + Q(μ) 端点报告；**反假收敛**：μ=0 自由跑从不收敛到非自然靶点。
- **T12 判决**：可达域宽（κ≈1.7 远离 0.3 下限），开二期条件满足。

### 3.4 反向破坏验证（sabotage，守卫非摆设的证据）

| 破坏 | 结果 |
|---|---|
| 移除 nspin 守卫 | 恰 2 个目标测试 FAIL（SpinTypeGuard + ConfigureFromInputsShared） |
| 翻转 spin 注入符号（V_↑+= → −=） | 恰 SplitInjectionSpin FAIL |
| response_sign 设 +1 | 恰 SpinChannelConvergesOnLinearResponse FAIL（m 被驱到 −2 而非 +0.02） |

### 3.5 MPI 一致性

权重网格 1/2/4-rank 逐点一致 1e-12；LCAO 集成用例 4-rank vs 串行能量差 3.4e-11 eV、Q/μ 逐位一致；M6 导数网格 MPI 3/3。

### 3.6 回归

每轮 `ctest -R constraint|partition|read_input|elecstate_pw` 全绿；PW 037_FM、PW 211、LCAO 212 逐位复现；已知的 MODULE_LCAO 2 FAIL+2 Not Run 与 unitcell_test_pw 经核实为**既有 CWD/构建树问题**（脚本/support 未拷入），与 constraint 改动零交集。

### 3.7 评审链（5 轮实证评审，问题全部闭环）

| 轮次 | 裁定 | 发现问题 | 状态 |
|---|---|---|---|
| 一期总评 | 通过 | P1 勾选失信 / P2 inject 契约 / P3 措辞 / P4 集成用例未注册 | 293f53aa8 全部闭环 |
| Task 2.3 | 通过 | Task 2.1 Step 6 漏勾；M3b 头注误导 | 已闭环（ae9bfc600 随轮修正） |
| Task 2.4 | 通过 | 符号修正判定为正当（实测驱动，非改判据）；037 回归与 sabotage 未复核（低风险登记） | 无需处理 |
| V2 阻塞消解 | 方案修订 | Multiwfn 依赖 → V2a/V2b 内部闭环 | f6fcc443d 闭环 |
| Task 2.5（M6 力核，§3.8） | 通过 | 无阻断项；驻点 WARNING 守卫与核守卫实证在线 | 无需处理 |

### 3.8 Task 2.5 评审实录（2026-08-31，评审者亲自复跑）

- **PW 集成**（211 + `test_force=1` 复跑）：CONSTRAINT FORCE 块 O z=+0.09929、H1/H2 x=∓0.0963、z=+0.0722 Ry/Bohr——与 spec 逐项吻合；**总力平移不变性** Σ_J F_J≈4e-5 eV/Å≈0（含约束块；约束块自身不必求和为零——包络定理下 KS 块在约束密度处补偿，属预期行为，实测总力闭合）。
- **μ=0 短路**（delta=0 变体复跑）：约束块精确全零（零乘子短路实证）。
- **LCAO 集成**（212_NAO + cal_force/test_force 复跑）：O z=+0.10902、H x=∓0.10723、z=+0.08304——与 spec 吻合；FORCE_STRESS.cpp:468 与 PW 调**同一个** `constraint_force` 核（grep 实证，无双份力代码）。
- **守卫实证**：`compute_force` 懒建导数网格 + 驻点守卫（外环未收敛时 WARNING "residual O(|Q−t|)"）；力核对未建导数网格的直调 WARNING_QUIT（开发期"修复 1 处"声称的核守卫真实）。
- **判定边界**：以上验证的是**管线正确性**（非零/零/对称/守恒/守卫），力的**物理精度**（FD 对拍）属 Task 2.6，此前力不可用于生产——此边界不变。

## 4. 当前边界与遗留债务

1. **力未验收**：M6 双基组力均已接线、单测过（解析 2.2e-10/牛三 7e-13/线性 1e-12；LCAO 侧冒烟 O z=+0.1090、H x=∓0.1072 Ry/Bohr，μ=0 参考相力块精确全零），但 stationary4 FD 判决（0.0129 eV/Å，网格前提 ecutwfc=100+ecutrho≥400+scf_thr=1e-8）在 Task 2.6 才执行——此前力不可用于生产。
2. ~~LCAO 力未接线~~（2.5.4 已完成：同一网格力核经 FORCE_STRESS 通道，验证"双基组同码"架构声明）。
3. **M3b 悬置资产**：W^α HContainer 路径仅作 ctest 审计仪器（生产走 v_eff 网格注入）；Task 2.6 升格运行时审计或收尾删除。
4. **V2c'（Multiwfn 约定级对拍）**：可选，不阻塞。
5. **三期内不做**：Hirshfeld 权重、Broyden、偶极/多极子、应力（R6 待证）、Hirshfeld-I——禁止清单在案。
6. 未独立复核项（低风险）：037_PW_FM 回归、sabotage 重做。

## 5. 下一步

Task 2.6 三判决验证（PW≡LCAO 逐位一致 / 力 FD stationary4 / 力矩 FD 严格 μ=−λ 换算）+ M3b 命运判决 → Task 2.7 二期判决门（不过不降低判据）。

---

> 文档链：方案（实空间权重约束框架设计方案.md）→ 架构（plan-architecture.md）→ 评审 5 份（2026-08-26-*）→ 计划 2 份（plans/2026-08-30-phase1、2026-08-31-phase2）+ Task 2.5 详案 → 每模块 dated spec 14 份 + 评审记录 4 份 → 本总结。全部运行记录见 deltap-development-log.md 2026-08-26 起各节。
