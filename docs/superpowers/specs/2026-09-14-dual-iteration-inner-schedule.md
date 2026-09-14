# 双迭代收敛策略 Q0：INNER 调度（μ 的 SCF 内环更新）+ 守卫 + OUTER 逐位回归

> 计划：`docs/superpowers/plans/2026-09-11-dual-iteration-strategy.md`（§2 实施）。
> 本轮范围 = 计划 §2（实现 + 单测 + 守卫）+ §3 的 **Q0 判据**
> （"单测全绿 + legacy 逐位回归"），并顺带完成 **Q1–Q3 定量对照**与
> **Q4 决策表**（§3.5）。未完成的只剩 C-29 定位（预存在 bug，见 §5）。
> 日期：2026-09-14。分支：`feat/deltap`。

---

## 1. 测试计划（要验证什么）

计划 §2 要求把 μ 更新调度做成一个显式二值开关，并给出全套守卫。本轮的
可证伪判据：

1. **INNER 调度存在且门控正确**：`constraint_mu_schedule=inner` 时，
   `drho < constraint_inner_thr` 才在 SCF 迭代内更新 μ；门控是**严格**
   不等式；`constraint_inner_nmax` 用尽即降级回 OUTER 并**大声报告**，
   绝不静默、绝不挂死。
2. **mixing 复位契约**：每次 μ 真正改变 → 恰好一次 `mix_reset()` 请求
   （Broyden/DIIS 缓存描述的不动点映射已不存在）；μ 未改变 → 不复位。
   特别地，M4 报 `CONVERGED` 的那一步**不**改变 μ（收敛判据前置于任何更新），
   因此**不**请求复位。
3. **Settle check**：内环宣告 CONVERGED 只是**临时**判决。冻结 μ、等密度
   松弛后再核 `|Q−t|`；通过才算 CONVERGED，反弹则撤回判决回内环继续；
   连续两次反弹 → 降级 OUTER。这是 2026-07-20"内环收敛而外环弛豫反弹"
   陷阱的硬设计对策。
4. **反假收敛（T4a 纪律，INNER 同考）**：门控打开（drho 很小）**不等于**
   靶点达成。μ=0 参考态上 Q≠t，运行必须保持 RUNNING 且 μ 必须离开 0。
5. **OUTER 零回归**：默认 `outer` 路径与改动前**逐位一致**（含审计行文本、
   μ 轨迹、外步计数）。
6. **守卫可证伪（sabotage）**：故意破坏门控时，恰好是门控相关测试失败，
   且 OUTER 回归测试**始终绿**。

## 2. 测试设置

| 项 | 内容 |
|---|---|
| 构建（单测） | `build/`（Debug，`BUILD_TESTING=ON`）：`make -j14 MODULE_ESTATE_constraint_loop MODULE_ESTATE_constraint_io`，`ctest -R constraint` |
| 构建（集成） | `build_rel/abacus_basic_para`（Release，`BUILD_TESTING=OFF`） |
| 运行 | `mpirun --allow-run-as-root -np 4`；**对比运行固定 `OMP_NUM_THREADS=1`**（见 §4.3） |
| 单测 | `source/source_estate/module_constraint/test/constraint_loop_test.cpp`：mock 高斯密度、归一化线性响应 `Q(μ)=Q_ref−μ`、`mock_iteration()` 按真实钩子次序驱动（observe → on_iteration → on_scf_converged） |
| 集成算例 | `tests/01_PW/211_PW_constraint_h2o`（PW 电荷）、`212_PW_constraint_h2o_spin`（PW 自旋）、`213_PW_constraint_h2o_mixed`（PW 混合）、`tests/02_NAO_Gamma/212_NAO_constraint_h2o`（LCAO 电荷） |
| 对照二进制 | 改动前源码（`git checkout` 后重建）→ 跑同一算例 → 与改动后逐位对比（仅 211，见 §3.3） |

**新增 INPUT**（计划 §2.1，默认值使老输入逐位不变）：

| 参数 | 默认 | 语义 |
|---|---|---|
| `constraint_mu_schedule` | `outer` | `outer` = SCF 完整收敛后一次 M4 步（现状）；`inner` = SCF 迭代内 drho 越门即更新 μ |
| `constraint_inner_thr` | `1e-3` | INNER 的 drho 门控；`inner` 模式下必须 > 0（否则 ERROR） |
| `constraint_inner_nmax` | `20` | INNER 每次 run 的 μ 更新预算；用尽 → 降级 OUTER；`inner` 模式下必须 > 0（否则 ERROR） |

**改动文件**：`constraint_io.{h,cpp}`（配置字段 + 3 条守卫）、`constraint_loop.{h,cpp}`
（`on_iteration()` 钩子、`take_inner_step()`、`degrade_to_outer()`、`max_residual()`、
`print_audit_line()` 重构、Settle Branch S/S1/S2）、
`esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp`（在 `on_scf_converged` **之前**接线
`on_iteration` + 条件 `p_chgmix->mix_reset()`）、
`input_parameter.h` / `read_input_item_other.cpp`（3 个 INPUT）、
`test/constraint_loop_test.cpp`（6 个新测试）。

**钩子次序（必须保持）**：`observe → set_scf_energy(etot−cc_escon) → [on_iteration
+ maybe mix_reset] → on_scf_converged`。

## 3. 结果

### 3.1 单测

`ctest -R constraint`：**11/11 全绿**（loop 测试 25/25：19 个既有 + 6 个新增）。
新增 6 例与断言：

| 测试 | 断言要点 | 结果 |
|---|---|---|
| `InnerScheduleGating` | drho≥门控不更新；越门**每迭代一次**；`inner_nmax=3` 用尽即降级（`inner_active()==false`） | PASS |
| `InnerMixResetOnUpdate` | 越门且 μ 变 → 恰一次复位；μ 不变（CONVERGED 步）→ 不复位；收敛由 INNER 路径达成（`outer_steps==1`） | PASS |
| `InnerSettleCheck` | settle 通过 → CONVERGED；反弹 → 撤回（status 回 RUNNING、`inner_active` 仍 true）；第 2 次反弹 → 降级 OUTER（`outer_steps` 增加） | PASS |
| `InnerAntiFakeConvergence` | 参考相门控打开也不收敛（Q 距靶点恰为 δ）；gated 迭代把 μ 推开；终态 μ*=−δ | PASS |
| `OuterLegacyBitIdentical` | 默认 outer；drho=0（任意门控都满足）也绝不触发内环；`inner_steps==0`；μ* 与旧逻辑一致 | PASS |
| `InnerGuards` | 非法 schedule → ERROR；`inner`+thr≤0 / nmax≤0 → ERROR；`outer`+两者为 0 → OK；门控**严格**不等式（drho==thr 不更新） | PASS |

### 3.2 sabotage（门控强制破坏）

改动 `constraint_loop.cpp` 的门控行，重建后跑 INNER+OUTER 6 例：

| sabotage | 失败测试 | OUTER 回归 |
|---|---|---|
| 门控放宽为 `≤`（边界） | 恰 `InnerGuards` | **绿** |
| 门控恒真（永不门控） | `InnerScheduleGating`、`InnerMixResetOnUpdate`、`InnerGuards` | **绿** |
| 门控恒假（从不更新） | 5 个 INNER 测试全失败 | **绿** |

结论：门控/边界测试确实可证伪；**OUTER 回归在三种破坏下都保持绿**——它不依赖内环逻辑。

### 3.3 OUTER 逐位回归（对照二进制）

方法：先以改动前源码重建 `build_rel/abacus_basic_para` 跑对照，再恢复改动重建跑
同一算例，固定 `np=4` + `OMP_NUM_THREADS=1`：

| 算例 | 改动前 | 改动后 | 判定 |
|---|---|---|---|
| **211 PW 电荷** | `-441.9708337535537 eV`，外步 7（迭代 14/30/40/50/60/61/63） | `-441.9708337535537 eV`，外步 7（同迭代号） | **逐位一致** |
| 212 PW 自旋 | —（未跑对照） | `-442.0408674948372 eV` vs `etotref -442.0408674948625` | 差 2.5e-11 eV，命中阈值 1e-7 |
| 213 PW 混合 | —（未跑对照） | `-441.9159885528531 eV` vs `etotref -441.9159885324229` | 差 2.0e-8 eV，命中阈值 1e-7 |
| 212 NAO 电荷 | —（未跑对照） | `-466.2533233603508 eV` vs `etotref -466.253323360341` | 差 9.8e-12 eV |

211 的审计行文本亦逐位一致（`[constraint] outer step N after SCF iteration M
(phase=constrained)`），证明 `print_audit → print_audit_line(label="outer")`
重构未改变输出。

### 3.5 定量对照研究（计划 §3 的 Q1–Q3）

**设置**：同一二进制、同一网格、同一靶点、同一初猜；`np=4`、`OMP_NUM_THREADS=1`。
成本主指标 = **SCF 迭代数**（每迭代一次对角化）。证据：`results/summary.txt` +
`results/*.audit`（`tests/deltap_dual_iteration/`）。

**Q1/Q2 双策略对照**（μ*、E_tot、成本）：

| 体系 | 策略 | μ* | E_tot [eV] | SCF 迭代 | 外步 | 内步 / 复位 | settle |
|---|---|---|---|---|---|---|---|
| 211 PW 电荷 δ=+0.1 e | OUTER | −0.1763644518 | −441.9708337535537 | **63** | 7 | 0 / 0 | — |
| | INNER | −0.1764676765 | −441.9708338634222 | **72** (+14%) | 5 | 28 / 26 | 反弹×2 → 降级 |
| 212 PW 自旋 δ=+0.1 μB | OUTER | −0.07233854468 | −442.0408674948372 | **42** | 3 | 0 / 0 | — |
| | INNER | −0.07245849357 | −442.0408670948165 | **44** (+4.8%) | 1 | 27 / 26 | 通过×1 |
| 213 PW 混合 | OUTER | c −0.1812324411 / s −0.08155649998 | −441.9159885528531 | **117** | 15 | 0 / 0 | — |
| | INNER | c −0.1813286704 / s −0.08165906454 | −441.915988601339 | **63** (−46%) | 1 | 28 / 26 | 反弹×1 → 通过×1 |
| MgO 体相电荷 δ=+0.5 e | OUTER | −1.049024275 | −7659.260116348592 | **381** | 23 | 0 / 0 | — |
| | INNER | −1.049044143 | −7659.26011634944 | **94** (−75%) | 1 | 31 / 30 | 通过×1 |

- **正确性等价（全部命中判据）**：μ* 相对差 0.002%–0.17%（判据 <1%）；
  `E_tot` 差 4.9e-8 – 8.5e-7 eV（判据 <1e-6 eV）。
- **成本交叉点**：OUTER 外步 ≲7 → INNER 略差（+5…+14%）；外步 ≳15 → INNER 显著更优
  （−46% / −75%）。机制直白：INNER 的收益 = 省掉的"每外步一次 SCF 重收敛"，
  而它的代价 = 每次内更新的 mixing 复位重启。
- **settle 检查的真实价值**：全 campaign 抓到 **3 次**"内环宣告 CONVERGED、密度一松弛
  靶点即破"（211 两次：松弛后残差 1.4e-3 / 1.65e-4 ≫ thr 1e-4；213 一次），
  若无 settle 检查这 3 次都会直接误报 CONVERGED。
- **"内环步 − 复位 = CONVERGED 步数"在真实运行里成立**：211 28−26=2（两次都反弹）、
  212 27−26=1（通过）、213 28−26=2（1 反弹 + 1 通过）、MgO 31−30=1（通过）
  ——与"CONVERGED 步不改 μ 故不复位"的契约一致。

**Q3 mixing 复位对照**（`ABA_CONSTRAINT_INNER_NO_RESET=1`，诊断开关，默认关）：

| 体系 | INNER + reset | INNER 不复位 |
|---|---|---|
| 211 PW 电荷 | 72 迭代，CONVERGED | **161 迭代**（2.2×），CONVERGED |
| 213 PW 混合 | 63 迭代，CONVERGED | **200 迭代（用满 scf_nmax）仍未收敛**，`final status: RUNNING` |
| MgO δ=+0.5 | 94 迭代，CONVERGED | 145 次内更新 → settle 连败×2 降级 → 600 迭代 |

⇒ **mixing 历史腐败是真实机制，`mix_reset` 不是可选项**：不复位让 SCF 慢 2-6 倍
或直接不收敛。计划 §3.3-3 的判决问题回答为"是被破坏，且复位是有效对策"。
复位**代价**本身很小（日志 `mixing recovered after K SCF iteration(s)`，
实测 K=1–2）。

**Q4 决策表**（已回填用户手册 §5.9）：

| 场景 | 推荐 | 依据 |
|---|---|---|
| OUTER 外步 ≲7 即收敛（易体系/小扰动） | **OUTER（默认）** | 211 +14%、212 +4.8%：内更新+复位开销换不回收益 |
| OUTER 外步 ≳15（多约束/大扰动/体相） | **INNER** | 213 −46%、MgO −75% |
| 任何 INNER 使用 | 必须保留 `mix_reset`（默认即开） | Q3 对照 |
| 任何 INNER 使用 | 必须保留 settle 检查 | 抓到 3 次假收敛 |

### 3.6 顺带发现（非本任务引入）

- **无 `OMP_NUM_THREADS` 时算例不可复现**：同二进制同 `np=4` 连跑两次，211 得
  `-441.9708337885152` / `-441.9708338483194` eV、外步数 7/6 不同。根因是
  OpenMP 线程数默认（本机 14）导致归约次序不确定。**固定 `OMP_NUM_THREADS=1`
  后逐位可复现**。`tests/integrate/Autotest.sh` 的阈值 1e-7 eV 与此噪声同量级，
  故不加线程固定时该算例的阈值比较本就是边缘的（改动前后皆然，非回归）。
- CI/autotest 由 `OMP_NUM_THREADS` 环境变量驱动（`nt=$OMP_NUM_THREADS`），
  建议在本机复跑约束算例时显式 `OMP_NUM_THREADS=1`。

## 4. 分析

### 4.1 实现侧

- **OUTER 是纯 no-op 路径**：`on_iteration()` 在 `!inner_active_` 时立即返回 false，
  无任何副作用；`print_audit_line()` 以 `label="outer", step=outer_steps_` 复现
  原字符串。§3.3 的逐位对照与 §3.2 的"破坏内环而 OUTER 恒绿"共同确认零回归。
- **发现并修复一个真实缺陷（本轮唯一代码 bug）**：Settle 反弹分支（Branch S2）
  原先只 `settle_armed_=false; ++settle_fail_`，**未把 `status_` 从 `CONVERGED`
  撤回**。后果：内环判决被撤回、run 继续跑，但对外 `status()` 仍报 CONVERGED——
  一个"判决与状态不一致"的静默错误面。由 `InnerSettleCheck` 的红灯暴露，修复为
  `status_ = MuStatus::RUNNING`。这正是 plan §2.5 单测要求的价值所在。
- **INNER 与 OUTER 并存语义**：`on_scf_converged()` 在 INNER 模式下仍保留 OUTER
  秒差步作为**安全网**（外步不被禁用）。正常 INNER 流里，内环步触发的
  Branch I 会在同一迭代置 `conv_esolver=false`，避免外步介入；只有门控配置
  自相矛盾（`inner_thr ≤ scf_thr`，收敛时门控反而关闭）或已降级时外步才接管。
  这保证"绝不因调度而停在一个未达标的点上"。
- **Settle 检查的触发时机**：`CONVERGED` 步不改变 μ（MuSolver 收敛判据前置），
  故当该迭代 SCF 已收敛（drho<scf_thr）时立即 settle 是**等价**于"密度已松弛"的，
  不必强等额外迭代；否则 Branch I 先置 conv=false，settle 推迟到随后某个
  收敛迭代。测试两种路径（同迭代通过、下一迭代反弹）均覆盖。

### 4.2 残余风险（登记，供 Q1–Q3 判决）

- INNER 的**真实收益**尚未测量——本轮只证"实现正确 + 不回归"，不证"更快/更稳"。
  计划 §3.3-1/2 的判决问题留给 Q1–Q3。
- `mix_reset` 的**复位代价**只有在真实体系（FeO/MgO）上才可测；单测里 mixing
  被完全 mock 掉。
- `inner_thr` 默认 1e-3 是沿用 DeltaSpin 口径的**未标定值**；Q1–Q3 需要给出
  与 `scf_thr` 的关系建议。

## 5. 下一步

| 步 | 内容 | 状态 |
|---|---|---|
| Q1 | H₂O 三用例（211/212/213）OUTER vs INNER 对照 | **完成**（§3.5）：正确性等价命中；成本 +14% / +4.8% / **−46%** |
| Q2 | MgO 体相电荷 δ=+0.5 对照 | **完成但 BLOCKED**（§3.5 + §3.6）：**−75%**，正确性命中；但两侧收尾均触发预存在 C-29 堆破坏（父提交复现），物理量在崩溃前已完整落盘 |
| Q3 | mixing 机制专项（INNER+reset vs 不复位） | **完成**（§3.5）：不复位 → 2.2× 慢 / 不收敛 / 降级 ⇒ 复位是刚需 |
| Q4 | 决策表入用户手册 §5.9 | **完成**（§3.5 决策表 + 手册 §5.9） |

**剩余（下一轮）**：

1. **C-29 定位**（ASAN/gdb）——它现在同时阻塞 MgO 长跑收尾与远侧扫描，优先级已高于
   任何新的对照算例；
2. FeO 自旋（双稳地貌）的 OUTER/INNER 对照——计划里"用户假设的主考场"，用户已令
   S4/S5 暂缓，且 FeO 基线锚定问题（II-1b）未收口，暂不做；
3. `inner_thr` 与 `scf_thr` 的定量关系（当前建议 `inner_thr ≫ scf_thr`）——
   可用 212/213 扫 1e-3 / 1e-4 / 1e-5 三档，成本低，价值中等。

**注**：本轮不做 FeO/Mg 远侧（用户已令 S4/S5 暂缓）；若 Q2/Q3 选用 MgO/FeO，
所有远侧扫描必须裹 `timeout` 且避开预存在堆破坏 **C-29**（非收敛收尾路径）。
