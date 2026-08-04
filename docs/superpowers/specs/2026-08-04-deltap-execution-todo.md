# DeltaP 执行 TODO 指导（2026-08-04，当前唯一权威执行清单）

> 路线依据：`2026-08-04-deltap-force-resolution-plan.md`（三阶段）、
> `2026-08-04-deltap-route-a-plus-design.md`（设计）、
> `2026-08-04-deltap-route-a-plus-derivations.md`（联动推导）。
> 执行纪律（dev log 流程教训）：每步带验收与可证伪预言；偏离预言即停；
> PASS 必须注明独立参照物；改 SCF 的步**一次只做一个**，锚点重建只发生两次
> （S_k 修复后一次、Route A+ operator 模式一次）。

---

## Stage 0：收尾（今天，~2 小时，全部机械操作）

| # | 任务 | 验收（参照物） | 备注 |
|---|------|----------------|------|
| 0.1 | ~~commit D_I/S_k 修复~~ **✅ 已提交**（另含相位配对 sort 修复） | CI 式回归：单测 16/16 ✅；MPI smoke 4/4（含 co 奇数 NBANDS=15）✅ | 参照物：hf/co f0 串行 vs 4-rank 逐原子 γ 六位一致 ✅（见 dev log 2026-08-04 轮） |
| 0.2 | ~~跑完 co f0（λ=0）4-rank vs 串行~~ **✅ 已完成** | 逐原子 γ 一致：co (-6.702,-9.219) == 4-rank；hf (-9.173,-2.767) == 4-rank ✅ | 修复前差 0.02 rad（Σ 守恒）→ 根因=相位配对 sort 负距离 bug，已修；结果已补进筛选状态文档 |
| 0.3 | ~~锚点重建 #2~~ **✅ 已完成** | 12 用例 rc=0 ✅；E'/λ/γ/branch 轨迹存档（/tmp/deltap_anchor2）✅；`results.csv` 更新 ✅ | 与 1b2625fdd 同脚本；本次是 S_k 键 + 相位 sort 修复驱动；见 `2026-08-04-deltap-anchor2-s-k-rebuild.md` |

**Stage 0 出口检查**：git 干净；hf 4-rank 逐原子 γ = 串行（±0.01 rad）记录在案。

---

## Stage 1：Route A+ 串行实现（主线，~2 天）

### 1.1 Γ 计算（module_deltap）

| # | 任务 | 文件锚点 | 验收 |
|---|------|----------|------|
| 1.1a | `DeltaP` 新增成员 `std::vector<double> gamma_op_`（nat）+ `compute_operator_observable()` | `deltap.h`（members 区）| 编译过 |
| 1.1b | Γ_I^HR = τ_α(I)·Σ_k w_k Σ_n f_n w_In(k)——在 `compute_gamma_scf` 的 w_In 累加点顺带累加 | `deltap_wannier.cpp:1056` 附近（w_norm 累用处）、`w_In_first_string_`（:368） | T0 见下 |
| 1.1c | Γ_I^HK：`compute_hk_correction` 的 E_HK 累加在 Σ_I λ_I 求和前按原子拆出（w_eff[n] = Σ_I λ_I w_In 处，:1755-1774） | `deltap_wannier.cpp:1755` 起 | E_HK = Σ_I λ_I·Γ_I^HK 数值自洽 |
| 1.1d | **T0（可证伪）**：Γ_I^HR 的 per-k 形式 vs 实空间 Tr[DMR·pre_hr]（hhrdbg p_hat 机制翻回）| 两口径差 <1e-10 | **不一致 → 停**：以实空间口径为准实现，记录差异原因 |

### 1.2 状态机切换（deltap_scf）

| # | 任务 | 文件锚点 | 验收 |
|---|------|----------|------|
| 1.2a | INPUT 新增 `deltap_observable`（`operator` 默认 / `gamma` 旧路径） | `input_parameter.h:625` 附近 + `read_input_item_other.cpp` | 读入打印正确 |
| 1.2b | `DeltapState` 加 `gamma_op`（nat）；backend `compute_gamma` 回调返回后同步填 | `deltap_scf.h:49`、`deltap_scf.cpp:294` 附近 | — |
| 1.2c | 残差口径切换：`compute_residual` 的输入在 operator 模式用 `gamma_op`（:306/:309/:178/:219 共 4 处） | `deltap_scf.cpp` | gamma 模式逐字节不回归 |
| 1.2d | escon 切换：operator 模式 `compute_dp_escon(lambda, gamma_op)`（:315） | `deltap_common.h:157` 不改签名 | 单测补 operator 用例 |
| 1.2e | target 语义：用户给 t_γ（不变），内部 t_Γ 初值 = t_γ（κ=1 首轮） | `DeltapParams::target` 读入处 | 打印标注 |

### 1.3 外循环 secant（最小实现）

| # | 任务 | 锚点 | 验收 |
|---|------|------|------|
| 1.3a | `DeltapState` 加 `t_proxy`、`gamma_meas_prev`、`t_proxy_prev` | `deltap_scf.h` | — |
| 1.3b | `reset_ionic_step` 挂钩 secant 更新（公式见推导文档 §7：κ clamp [0.3,3]，单步限幅 0.5 rad，发散 WARNING 不中断） | `deltap_scf.cpp` `reset_ionic_step` | 单点/relax 各触发一次正确 |
| 1.3c | 打印：`[DeltaP P3]` 加 Γ 列 + escon_new；`[E-field]` 换 E_eff=λ/(2a)（标 operator-ramp，符号待 V1 钉死） | `deltap_scf.cpp:425-456` | 输出格式文档同步 |

**Stage 1 出口检查**：gamma 模式全锚点逐字节一致（零回归证据）；operator 模式编译+冒烟跑通。

---

## Stage 2：Route A+ 串行判决（T1–T5，~1 天，决定路线成败）

按设计文档 §5 执行，全部 h2o1 串行、生产设置（ecutwfc=100/ecutrho=400/scf_thr=1e-8，
`OMP_NUM_THREADS=1` 或 4，写进 run 脚本）：

| # | 测试 | 可证伪预言 | 偏离时的动作 |
|---|------|-----------|--------------|
| T1 | E' 恒等式（E' vs E_KS(ψ*)） | 差 <1e-8 eV | 差大 → escon 接线错，回 1.2d |
| T2 | ∂E'/∂λ 重测（base λ 扫描） | 224 eV/Ry → ≲1 eV/Ry（O(λ)） | 仍是 O(1) → Γ 与 H_c 不一致，回 1.1 |
| **T3** | **驻点组② 复判**（三几何驻点 FD；约束变量=Γ，驻点判据 \|Γ−t_Γ\|<1e-3） | **残差 84.8 → ≤0.02 eV/Å** | **≫0.02 → 停**，残差分解（λ-leak 重算）找未识别项 |
| T4 | 外循环收敛（t_γ=0.9γ_natural） | ≤5 步 \|γ−t_γ\|<1e-2 | 发散 → κ 限幅/映射单调性检查 |
| T5 | 组① 冻结 λ FD | 0.615 → ~0.05 eV/Å（λ·dΓ/dR） | 显著更大 → 响应项重估 |

**T3 是判决点**：通过 → Route A+ 成立，进 Stage 3；不通过 → 停，带着残差分解回来评审（不要盲目转 Route B）。

---

## Stage 3：MPI 收口（T3 通过后，~2 天）

| # | 任务 | 验收 |
|---|------|------|
| 3.1 | **hk_correction MPI 修复**（本地列索引 → 全局带映射，与 D_I A' 同族；`deltap_wannier.cpp:1651` 起，c_L/c_R 的 p·nrow 索引） | **修复前后串行 A/B 逐字节一致**（硬约束）；hf/co corr=1 4-rank 与串行逐原子 γ、E' 一致；h2o_asym 仍被方阵守卫拦（记录） |
| 3.2 | T7：operator 模式 4-rank Γ/γ 跨 rank 一致 | 逐原子 ±0.01 rad |
| 3.3 | co/h2o_asym corr=1 4-rank 收敛性复测 | co 4-rank 收敛（此前 100 iter 不收敛归因 H_HK） |

---

## Stage 4：锚点 #3 + 验收（~1 天）

| # | 任务 | 验收 |
|---|------|------|
| 4.1 | operator 模式锚点重建（第三次，预期内） | 12 用例 + Γ 列存档；gamma 模式锚点冻结标注 legacy |
| 4.2 | Tier-1 三体系驻点 FD 全矩阵（hf/co/h2o_asym） | 各体系残差 < 判据 0.0129 或文档化偏差 |
| 4.3 | D-D 正式验收 + deltap_relax 端到端（≥3 离子步能量下降、力平滑） | 通过判据 |
| 4.4 | V1（efield 对照钉 E_eff 符号/因子）、V3（BN 新记账刚度复核） | 见推导文档 §10 |

## Stage 5：文档与清理（~0.5 天）

- dev-guide v3（三种记账推导 + dspin 恒等式 + Route A+ 结构）；
- E-field 语义更新（推导文档 §1.2 入手册）；`deltap_observable` 关键词文档；
- 清理：hhrdbg/hkdbg/fsdbg 等 #if 0 块（A2/C 验证已用毕）；
- dev log + 总览文档（2026-08-04-deltap-progress-and-plan.md）刷新。

---

## 依赖与阻塞规则

```
0.1–0.3 → 1.1 → 1.2 → 1.3 → T1→T2→T3（判决）
T3 通过 → 3.1（hk MPI）→ 3.2/3.3 → 4.x → 5
T3 不通过 → 停止，残差分解后回来评审
Phase 0.3（分支连续性）只需在 T4 投产前完成，可与 Stage 1 并行
```

**绝对不做**：单独补 C；T3 前做 hk MPI 修复（污染验收面）；operator 锚点前做
D-D 验收；用 LCAO 多 rank 数据做任何判决（直到 3.2）。

## 每轮交付物（AGENTS.md 纪律）

每个 Stage 结束：dated 文档（测试计划/设置/结果/分析/下一步）+ dev log 追加 +
本文档对应行勾选状态更新。
