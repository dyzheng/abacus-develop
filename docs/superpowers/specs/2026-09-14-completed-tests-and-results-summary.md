# 已闭合测试与结果总览（截至 2026-09-14，`feat/deltap`）

> 用途：**阶段 B 立项评审**与 merge-readiness 的输入基线；本文档是纯汇总，无代码改动。
> 数据来源：各轮 dated spec（§3 每行"证据"列 + `deltap-development-log.md`）。
> 前序总览：`2026-09-09-stageA-progress-summary.md`（阶段 A）、
> `2026-08-31-constraint-framework-progress-summary.md`（一期/二期）。

## 1. 测试计划（本文档的范围与判据）

本文档汇总 `feat/deltap` 上**已跑完并有归档**的全部测试轮，回答三个问题：

| # | 问题 | 判据 |
|---|---|---|
| Q1 | 约束框架**已验收**的能力包络是什么？ | 每条能力都有"判据 + 实测 + 证据 spec"三件套，未验收项单独列 |
| Q2 | 还缺什么才能解锁受闸门的物理（力 / TM d 矩 / bulk 表征）？ | 缺口有明确判据与成本估计，且根因已定位 |
| Q3 | 阶段 B（Broyden/Jacobian）立项数据是否齐备？ | μ 耦合表、收敛步数退化、INNER 收益曲线、门控标定四类数据在案 |

**不在本文档范围**：被闸门挡住的物理（TM d 矩物理解读）、以及用户明确暂缓的项
（FeO 双稳对照）。V1 自旋力 FD 已于本文档定稿后补完（§3.11）。

## 2. 测试设置（统一口径）

| 项 | 值 |
|---|---|
| 分支 / HEAD | `feat/deltap`，汇总时点 HEAD `abff67d34` + 本轮文档 |
| 构建 | `build/`（Debug，`BUILD_TESTING=ON`）、`build_rel/abacus_basic_para`（Release）、`build_asan/`（ASAN） |
| 并行 | `mpirun -np 4`（14 核 / 15 GB 容器）；单测 np1 |
| 线程纪律 | **对照/验收跑一律 `OMP_NUM_THREADS=1`**（OMP 线程数会导致算例不可复现，与 autotest 1e-7 eV 阈值同量级） |
| 载体 | H₂O 15-Bohr 盒（PW 211/212/213、LCAO 212_NAO）、MgO rocksalt 8 原子、FeO `17_DS_DFTU/11_PW_DFTU_S2_FeO` |
| 证据目录 | `tests/<case>/`：README + `results/*.audit` + `summary.txt`（**不入库全量 SCF 输出**） |
| 能量口径 | 约束判据一律用 **raw `FINAL_ETOT_IS`**；`E'=E−μt` 已被证伪为含伪项（见 §3.3） |

## 3. 结果（Results）

### 3.1 一图流状态表

| 测试组 | 判据 | 结果 | 状态 | 证据 |
|---|---|---|---|---|
| 单测（constraint + deltap_common） | 全绿 | `ctest -R "MODULE_IO\|constraint"` **56/56**（2026-09-14） | ✅ 闭合 | `2026-09-14-moduleio-test-hygiene.md` |
| 守护线（sabotage） | 每守卫配一发破坏恰中 | A1–A6、分支守卫、双迭代、on-site 审计四路破坏**全部恰中** | ✅ 闭合 | `2026-09-09-stageA-progress-summary.md` 等 |
| 电荷通道力 FD（Task 2.6/2.7） | 判据 0.0128555 eV/Å | PW 9/9 轴 PASS（max 0.00433）、LCAO O-z 0.000364 / H1-x 0.0002229 | ✅ 闭合（限定包络） | `2026-09-08-task26-closure.md` |
| 力矩 FD | 判据 0.006 eV/μB | \|d\| = **0.00011089**（富余 ~54×） | ✅ 闭合 | 同上 §3.4 |
| 阶段 A 混合约束 A0–A6 | G1–G6 逐门 | 混合 charge+spin 同原子可用；三旧用例逐位回归 | ✅ 闭合 | `2026-09-09-stageA-progress-summary.md` |
| I-1 MgO 宽电荷扫描 | 线性区 ≥2× H₂O 的 ±0.3 e | 线性区 **≥ ±0.8 e（≈2.7×）**；S1–S5 全跑完 | ✅ 判决成立 | `2026-09-11-i1-mgo-charge-scan.md` |
| I-1 远侧熔断尝试 | 触发 μ 顶限熔断 | **BLOCKED**：δ=−2.0 先崩于 SCF 不收敛（C-29 复现），顶限不可达 | ⚠️ 判据改判 | `2026-09-11-i1-mgo-farside-fuse-attempt.md` |
| II-1 FeO 自旋约束（II-1a） | 3 个验证问题 | Q2（DFT+U 同开）PASS；Q1 仅 1 个可信点；**扫描窗口不成立（基线多解）** | ⚠️ 部分 | `2026-09-11-ii1-feo-spin-scan.md` |
| II-1b 基线分诊 | 定位低解 + 步长修复 | Γ-only 多解 + k 未收敛；`step_max/probe` INPUT 化；重锚定后仅 ±0.1 μB 可测 | ⚠️ 部分 | `2026-09-11-ii1b-baseline-triage-and-step-cap.md` |
| 在线分支守卫 | 换态即熔断不静默 | `constraint_branch_tol`（默认 0=关）；单测 12→18 + 4 发 sabotage；FeO 外步 2 即熔断 | ✅ 闭合 | `2026-09-11-online-branch-guard.md` |
| on-site 矩审计 | 与 `atomic mag` 同量 | 同量到 1e-8；FeO 良态点 74% 跟随、塌陷点反向脱钩 | ✅ 闭合 | `2026-09-11-onsite-moment-audit.md` |
| 双迭代调度 Q0–Q4 | 正确性等价 + 成本 | μ* 差 ≤0.17%、E_tot 差 ≤8.5e-7 eV；成本 211 +14% / 212 +4.8% / 213 −46% / MgO −75% | ✅ 闭合 | `2026-09-14-dual-iteration-inner-schedule.md` |
| `inner_thr` 三档标定 | 改门控不改答案 | **纯成本旋钮**（μ* 散布 ≤0.34%、\|ΔE_tot\| ≤4.2e-7 eV）；默认保持 1e-3 | ✅ 闭合 | `2026-09-14-inner-thr-calibration.md` |
| C-29 收尾堆破坏 | 定位 + 修复 + 回归 | 根因 = `read_rhog` 缺 `ig<0` 守卫（**非**约束 bug）；修复 + 哨兵单测 | ✅ 闭合 | `2026-09-14-c29-localization.md` |
| III-1 CT 对（H₂O 二聚体） | 9/9 收敛 + 恒等式 | 9/9 CONVERGED；μ_acc=−μ_don 严格反对称；κ 分支 −1.12 vs −0.30 Ry/e | ✅ 首轮 | `2026-09-14-iii1-h2o-dimer-ct.md` |
| V1 自旋力 FD | LCAO 全轴 + PW 冒烟 | **LCAO 9/9 PASS**（max \|d\|=1.78e-4，72×）+ **PW 2 轴 PASS**（净力指纹无异常）；fixed-vs-重优化等价性成立（2.35e-4 eV/Å） | ✅ 闭合（限定包络） | `2026-09-10-taskV1-spin-force-fd.md` |
| 元数据/文档卫生 | 元数据=结构体真值 | 4 处漂移修正（零运行时改动）；`spin.md` ≥6 处刷新 | ✅ 闭合 | `2026-09-14-spinconstrain-metadata-sync.md` |

### 3.2 单测与守护线（sabotage）

- **模块单测**：constraint 模块 11 个注册 ctest 目标全绿（~49 s，2026-09-09）；
  阶段 A 逐 Task 增补判别测试：io `MixedConstraintListParsing`/`MixedGuards`、
  observe/inject 混合通道对拍、loop `MixedConvergesOnLinearResponse`/
  `MixedFuseHonorsPerComponentCap`、deriv `MixedChannelForce`/`MixedForceNewtonThirdLaw`
  （混合牛三 1e-9，RHS=observer 5 点 FD 独立折叠）。
- **后续轮增补**：分支守卫 loop 12→18、on-site 审计 18→19、双迭代调度 loop 19→25、
  `step_max/step_probe` 新增 `StepProbeCapsOnlyTheFirstStep`；当前
  `ctest -R "MODULE_IO|constraint"` = **56/56**。
- **sabotage 链**（守卫↔测试一一对应）：A1–A6 六轮、双迭代 3 发、分支守卫 4 发、
  on-site 4→5 发——**全部恰中且 legacy 回归全绿**（"拆守卫必红"）。

### 3.3 电荷通道力 FD（Task 2.6/2.7，判据 0.0128555 eV/Å 不豁免）

| 基组 | 覆盖 | 结果 | 判定 |
|---|---|---|---|
| PW（R7 网格） | **全 9 轴（18 腿）** | 最大 \|d\| = 0.00433（O-z），其余 ≤0.00321 | **PASS 9/9** |
| LCAO | O-z（fixed-μ 三腿） | \|d\| = 0.000364 | PASS（35×） |
| LCAO | H1-x（±δ 腿对） | \|d\| = 0.0002229 | PASS（~58×） |
| LCAO | 其余 7 轴 | 未跑 | 外推（有 PW 全轴 + 机制定案依据） |

- **归因链**（本轮最重要的方法学产出）：首轮 PW 18/18 FAIL 5 轴（1.69–3.36 eV/Å）
  → 根因 = 缺失的 **μw-Pulay 项**（非正交基 S 导数项）→ 修复 `ddd485d5c`（PW SCC-μw）
  与 `d6d0e021b`（LCAO Pulay RAII 守卫）→ 重跑 PASS。
- **平移不变性恢复**：R0 fixed-μ 净 z 力从修复前 **+7.078 eV/Å** 回到 ~0（打印精度）。
- **口径裁定**：约束 F_ana 必须取 raw-E 口径；`E'=E−μt` 的修正在力与力矩两处都被
  证伪为含 `μ·(dμ 项)` 伪项（力矩处若用 E' 则 T_FD'=2.9403，FAIL 1.96）——**登记为工具缺陷**。
- **力矩 FD**：base m\*=0.1000021944 μB、μ\*=−0.07193070771 Ry；
  T_FD = 0.97855623 eV/μB vs T_ana = 0.97866713 eV/μB，\|d\| = 0.00011089 ≪ 0.006（**54× 富余**）。
- **DeltaSpin 参照**（同体系同靶点）：λ_O ≈ −10.5…−10.8 eV/μB 与框架 μ（−0.9787 eV/μB）
  **同号**（μ=−λ 成立）；量级差 ~10× 归因于可观测算符不同（on-site 投影 vs Becke 区域），
  属阶段 B 测量项而非判据。
- **PW≡LCAO 对拍**：算符口径 PASS（同约束态双基组 FD 分别 PASS + 读数算符构造相同 +
  自旋 μ 双基组差 <1%）；严格"同密度逐位 <1e-8"**登记为开放项**（需 V3b 只读观测口）。

### 3.4 阶段 A 混合约束（A0–A6）

- 同一 run 可对任意原子/片段施加 **charge + spin 混合约束**（v2 JSON 列表），PW 与 LCAO 共口径。
- 集成用例（已注册 `CASES_CPU.txt` + `result.ref`）：

| 用例 | 场景 | 关键值 |
|---|---|---|
| 211_PW_constraint_h2o | PW charge +0.1 e on O | μ_c = −0.176548，CONVERGED |
| 212_PW_constraint_h2o_spin | PW spin +0.1 μB on O | μ_s = −0.072339，CONVERGED |
| 213_PW_constraint_h2o_mixed | PW **charge+spin 同原子 O**（v2） | μ_c = −0.181173、μ_s = −0.081544，外步 15 |
| 212_NAO_constraint_h2o | LCAO charge +0.1 e on O | μ_c = −0.219304，CONVERGED |

- 三旧用例走 v1 deprecated 兼容路径，**零修改逐位复现**。
- **μ 耦合实测（阶段 B 立项核心数据）**：

| 场景 | μ_c (Ry) | μ_s (Ry) |
|---|---|---|
| 211（nspin1 charge-only） | −0.176548 | — |
| 212（nspin2 spin-only） | — | −0.072339 |
| nspin2 charge-only 基线 | −0.176354 | 0 |
| **213 混合（同原子）** | **−0.181173** | **−0.081544** |
| 耦合偏移 | **−0.004819** | **−0.009206** |

- 结论：nspin 1→2 对 μ_c 仅 +1.9e-4（可忽略）；**跨类型耦合不可忽略且同向刚化**，
  外环收敛 3→15 步退化 ⇒ 对角 secant 不再充分，**Broyden/Jacobian 立项成立**。

### 3.5 I-1 MgO 宽电荷扫描（R4 判决：离子体系适用域）

**唯一核心问题：离子体系的电荷约束线性可达域是否显著宽于 H₂O 的 ±0.3 e？→ 是。**

- 网格标定（S1/S1X）：计划判据 <3e-5 e **不可达**（最优 2.8e-4 e），如实改判为
  "κ/μ 网格稳定性"；实测 δ=+0.8 在 80/320 vs 60/240 差 **0.018%**。
  生产网格 60/240；LCAO 下 `ecutrho/ecutwfc` 必须 = 4（两旋钮不独立）；
  Release vs Debug 逐位一致（提速仅 ~1.3×）。
- 参考态（S2，**Becke 周期像权重分区在 bulk 的首次实证**）：`total_charge` 64.0000000000
  （逐位 = nelec）、maxdev 6.66e-16、Q/O = 4.557340575 e（净 −1.4427 e，落文献带 ±1.0–1.5 e）、
  Q/Mg 严格抵消、M3b 1.69e-5 e；单片段 ≡ 4 原子子晶格求和（1.3e-8 e）。
- 单 O 八点扫描（S3，全部 CONVERGED、无熔断）：

| δ [e] | μ* [Ry] | 累计 κ | 局部 \|dμ/dδ\| | dev |
|---|---|---|---|---|
| +0.3 | −0.617502 | 2.058 | — | — |
| +0.5 | −1.049024 | 2.098 | 2.158 | +4.8% |
| +0.8 | −1.741718 | 2.177 | 2.309 | **+12.2%** |
| +1.0 | −2.248166 | 2.248 | 2.532 | **+23.0%（出线性区）** |
| −0.3 | +0.594235 | 1.981 | — | — |
| −0.5 | +0.981978 | 1.964 | 1.939 | −2.1% |
| −0.8 | +1.556173 | 1.945 | 1.914 | −3.4% |
| −1.0 | +1.936320 | 1.936 | 1.901 | **−4.0%（仍在线性区）** |

  ⇒ 线性区 **≥ ±0.8 e ≈ 2.7× H₂O**（H₂O 为 ±0.3 e，\|dμ/dδ\|≈1.7 Ry/e），
  **R4 判决成立**：离子体系是电荷约束的正当适用域。
- 单 Mg ±1.0 e（S4）：两点均 CONVERGED；−1.0 e（Mg³⁺ 化）μ*=+2.115 Ry ≪ 5.0 Ry 顶限，
  **未熔断** ⇒ **计划预期的"物理熔断用例"被推翻**，本算例未产出顶限熔断。
- 稳健性（S3.4）：μ* ≟ −dE_tot/dδ 八个区间**全部 <0.8%**；重复运行逐位一致；
  参考相逐点逐位复现（热启动不污染参考态）。
- 远侧尝试（§3.1 表）：δ=−2.0 e 在外步 55（q=10.2651 / μ=2.75 Ry）后约束 SCF 不收敛，
  收尾路径触发 C-29 并挂死 ⇒ **约束先崩于 SCF 而非 μ 顶限**，κ 随位移继续硬化。

### 3.6 II-1 FeO 自旋约束（部分闭合）

- **II-1a**：验证问题 2（**DFT+U 与约束同开的兼容性首测**——on-site 投影 vs veff 网格两条
  势通道从未同测）**通过**；δ=0 空操作逐位恒等。验证问题 1（TM d 矩约束能力）有资格通过但
  只有**一个可信数据点**。
- **根因（ii1a 最重要发现）**：计划假设的扫描窗口不成立，根因**不在约束框架**，而在继承基线
  `50_FeO` 的**多解性**（参考态是亚稳态，另一条 AFM 解低 **0.612 eV**）。
- **II-1b 分诊**：Γ-only `50_FeO` 多解且 k 未收敛——2×2×2 → ±1.48 μB / 低 2.1 eV；
  4×4×4 → ±3.10 μB / 再低 0.56 eV。框架侧两条修复落地：`constraint_step_max` /
  `constraint_step_probe` INPUT 化（默认逐位不变 + 单测），在线分支守卫（见下）。
- **重锚定后**：仅 ±0.1 μB 可测，±0.3 处**固定 μ 双稳**；能量口径判定 **E_tot = E[ρ]**。
- **Ω 缺口**：II-1b 的"μ vs λ 口径归因"依赖 **V3b 只读观测口**，仍未启动。

### 3.7 在线分支守卫 + on-site 矩审计

- **在线能量分支守卫**（`constraint_branch_tol`，默认 **0=关**）：能量判据
  `E_tot < E_ref − tol` → 第四态 `MuStatus::BRANCH_FLIP`，**熔断不静默**；
  单测 12→18、4 发 sabotage 全恰中；H₂O 端到端无假触发（能升 +0.0087 Ry ≫ tol 1e-3）；
  FeO 换态点**外步 2 即熔断**（−0.643 eV），旧流程要到外步 6 才误报 CONVERGED。
- **on-site 投影矩审计行**（DFT+U 迹差，无对角化，与 `atomic mag` 同量到 **1e-8**）：
  FeO δ=+0.1 μB 良态点 q 与 d 矩同向但仅 **74% 幅值**；与 II-1b 塌陷点的**反向脱钩**
  合起来给出"**同向跟随 → 反向脱钩**"图景；单测 18→19 / 4→5，无 DFT+U 输出逐位不变。
- **能力边界文档**补 L14（Becke 矩 ≠ on-site d 局域矩）、L15（在线分支守卫两条盲区：
  只抓能量向下换态、只在收敛点判）+ F9/F10 诊断项 + §5 适用域"自旋约束观测量缺口"。

### 3.8 双迭代调度（OUTER/INNER）+ `inner_thr` 标定

- **调度定义**：`constraint_mu_schedule=outer|inner`（默认 **outer=逐位不变**）；
  INNER 在 SCF 迭代中 `drho < constraint_inner_thr`（默认 1e-3）即更新 μ →
  `mix_reset()` 清 mixing 历史 → 继续 SCF；`constraint_inner_nmax`（默认 20）用尽即
  降级 OUTER 并大声报告；**settle check**：内环宣告收敛后冻结 μ 跑到 SCF 收敛再验
  `|Q−t|`，反弹则回内环（连续两次反弹降级 OUTER）。
- **Q0 契约全部落地**：门控严格不等式、μ 每变一次恰好一次 `mix_reset()`、收敛步不复位、
  T4a 反假收敛同考、OUTER 逐位回归（含审计行文本）、`inner_nmax` 防挂死。
  单测 loop 19→25；3 发 sabotage 恰中且 OUTER 回归三发全绿；OUTER 211 对照二进制逐位一致。
- **Q1–Q3 定量对照（同二进制/网格/靶点/初猜）**：

| 判据 | 结果 |
|---|---|
| 正确性等价（μ* 差 <1%） | **≤0.17%** ✅ |
| 正确性等价（E_tot 差 <1e-6 eV） | **≤8.5e-7 eV** ✅ |
| 成本（相对 OUTER） | 211 **+14%** / 212 **+4.8%** / 213 **−46%** / MgO **−75%** |
| mixing 复位机制 | 不复位则 **2.2× 慢或不收敛**（`mix_reset` 是刚需） |
| settle 检查 | 抓到 **3 次假收敛**（内环收敛声明不可直接采信） |

- **Q4 决策表**已入手册 §5.9（"何时用哪个策略"）。
- **`inner_thr` 三档标定（1e-3/1e-4/1e-5 on 212/213 + MgO）**：门控是**纯成本旋钮**
  （三体系 μ* 散布 ≤0.34%、\|ΔE_tot\| ≤4.2e-7 eV）；成本效应**符号随体系翻转**
  （212 +4.8%→−9.5%、213 −46%→−47% 平、MgO −75%→−72%→−66% 收紧变差）
  ⇒ **默认保持 1e-3**，定位为逐体系旋钮；settle 在三档下都仍触发（非松门控产物）。

### 3.9 C-29：收尾堆破坏（非约束 bug）

- **根因**：`ModuleIO::read_rhog` 缺 `ig<0` 守卫——读"更大平面波基组"写的
  `-CHARGE-DENSITY.restart` 时，盒内/球外平面波映射为 −1，`rhog[is][-1]` 写坏 malloc chunk 头。
- **关键澄清**：崩溃点（`!FINAL_ETOT_IS` 之后、`Charge::destroy`）只是**检测点**；
  非法写发生在 run 开头的 `before_all_runners`；**关掉 constraint 同样复现** ⇒ 与约束框架无关。
- **修复**：`if (ig<0) continue;` + 哨兵回归单测
  `ReadRhogTest.LargerBasisInFileDoesNotWriteBeforeBuffer`（拆守卫必红）；ASAN 首次非法写 2.6 s 即报。
- 该 bug 也是 I-1 远侧尝试"挂死"的直接原因（见 §3.5）。

### 3.10 III-1 电荷转移对（H₂O 二聚体，SCF 级旗舰接口）

- 体系：H₂O 二聚体（O2–H···O1，O1···H = 2.02 Å），LCAO/gamma-only/15 Å 盒；
  受体片段 `[0,2,3]` +δ、给体 `[1,4,5]` −δ（v2 列表 + delta 模式）。
- **9 点扫描（δ = 0, ±0.05, ±0.10, ±0.15, ±0.20 e）全部 CONVERGED**：
  - `total_charge = 20 = nelec` 每点成立（maxdev ~5.6e-16）；
  - **μ_acc = −μ_don 严格反对称**（max\|和\| = 0）；δ=0 ⇒ μ=0 且 E_tot 最低
    （自由态是 CT 坐标的极小）；
  - 能量恒等式 −dE_tot/dδ ↔ (μ_acc−μ_don)：8 个区间偏差 **+0.00%…+2.34%**
    （仅 δ≈0 邻域跳到 +22.3%）；
  - **主要物理发现 = 强非线性 + 角色不对称**：分支平均 κ = **−1.12 Ry/e**（受体失电子）
    vs **−0.30 Ry/e**（受体得电子），差 ≈3.7×；正侧 +0.05…+0.10 间斜率 4× 突变
    ⇒ **单一全局 κ 无意义**；
  - 量级登记（不外推）：正反向 μ 差 @ q=0.1 e = 0.2308 Ry ≈ **3.14 eV**；
    W(δ=+0.10) = −0.01616 Ry；CT 垂直能 δ=+0.10 → 0.154 eV、δ=−0.10 → 0.216 eV；
  - 成本：单点 21–189 s，全扫 ~15 min（III 组最廉价，可反复加密）。
- **未完成**：Marcus 重组能的定义式 + 文献对照（需先加密 δ≈0）。
- **工具坑**：能量恒等式跨单位（E 是 eV、μ 是 Ry）比较会给出 ~1260% 假偏差；
  `extract_scan.py` 已内置换算（`RY_TO_EV = 13.605693009`）。

### 3.11 V1 自旋力 FD（2026-09-14 补跑完成，G-V1 闭合）

- **协议**（FD 算例重设计 §4）：`fixed-μ`（腿冻结 μ=base μ*，`ABA_CONSTRAINT_FIXED_MU`）
  + raw-E 口径；判据 0.0128555 eV/Å；release 二进制、np4、`OMP_NUM_THREADS=1`。
- **前置（同轮入库 `62819bd46`）**：runner 加 `RESDIR` 归档（base/腿的 audit + 力块 +
  计时 + FD 表落 `tests/constraint_fd_force/results/<TAG>/`）、`FIXED_MU`/`KS_SOLVER` 开关、
  `std_checks` 块名自动发现（PW 与 LCAO 力分解块集不同）、LCAO spin 载体
  `cases/212_NAO_constraint_h2o_spin`；电荷/自旋两路径 INPUT **逐字节回归**通过。
- **协议等价性（自旋通道首发）**：LCAO O-z 下 fixed-μ vs 重优化-μ 的 F_FD 差
  **2.35e-4 eV/Å**（判据的 1.8%）⇒ 等价性成立；fixed-μ 成本约 **1.9×** 省。
- **结果 A（LCAO 全轴，27m36s）**：base `t*=0.1`、`μ*=−0.08212212424 Ry`、
  `E0=−466.2989480199703394 eV`；Σ 补偿前 z = **+0.000490 eV/Å ≈ 0**，一致自证 1.286e-07。

| 原子轴 | \|d\| (eV/Å) | 富余 |  | 原子轴 | \|d\| (eV/Å) | 富余 |
|---|---|---|---|---|---|---|
| O-x | 3.996e-5 | 322× |  | H1-z | 1.362e-4 | 94× |
| O-y | 4.027e-5 | 319× |  | H2-x | 3.980e-5 | 323× |
| **O-z** | **1.781e-4** | **72×** |  | H2-y | 2.592e-5 | 496× |
| H1-x | 9.302e-5 | 138× |  | H2-z | 1.360e-4 | 95× |
| H1-y | 2.600e-5 | 494× |  |  |  |  |

  **9/9 PASS，最大残差 = 判据的 1/72。**

- **结果 B（PW 冒烟 2 轴，dav_subspace + release）**：base `t*=0.1000010364`、
  `μ*=−0.07189321089 Ry`、`E0=−466.9020975520369348 eV`、Σ 补偿前 z = +0.013651 eV/Å；
  O-z \|d\| = 4.199e-3（3.1×）、H1-x \|d\| = 1.1969e-4（107×），**两轴 PASS**，
  净力指纹无异常（自旋通道无 μw-Pulay 缺失特征）。

  ⇒ **G-V1 闭合**（覆盖包络 = LCAO 全轴 + PW 2 轴；机制与电荷通道共用 M6 折叠核，
  净力指纹兜底）。**力相关物理（约束 relax/MD）过闸门。**

- **成本两处更正**：① 每轴 ~28 min，`dav_subspace` 比 debug+DiagoCG 的 23m42s 还慢 ~1.2×
  ⇒ 重设计 §1 的 "dav_subspace 2–5×" 在本 PW/ecut=100 小分子载体上**未被实测支持**；
  ② 真杠杆是载体：**LCAO 全 18 腿仅 27m36s**（vs PW 单轴 28 min）。
- 证据：`tests/constraint_fd_force/results/{v1_lcao_fullaxis_fixedmu, v1_lcao_oz_fixedmu,
  v1_lcao_oz_reopt, v1_pw_oz_davsub, v1_pw_h1x_davsub}`（41 个文件，audits + summary + legs.tsv）。

### 3.12 元数据 / 文档卫生

- `sc_scf_thr` `1.0e-3`→`10`、`sc_scf_thr_mode` `"threshold"`→`"immediate"`、
  `sc_drop_thr` `1.0e-2`→`1e-3`、`nsc` 注释 50→5，另修 `esolver_ks_lcao.cpp` 同类陈述——
  **零运行时行为变化**，`--help` 前后对比实证、`MODULE_IO|constraint` 56/56 不变。
- **`docs/parameters.yaml` 重建：未授权**（推迟到 merge-readiness；重建会首次公开
  28 个 `deltap_*` + 12 个仍在演进的 `constraint_*`，在接口未冻结时发布 = 把开发态承诺为稳定态）。
  `--help` 已正确，生成文档滞后不阻断开发。
- `docs/advanced/scf/spin.md` 参数表 ≥6 处陈旧已刷新（独立 docs commit）。

## 4. 分析（Analysis）

### 4.1 已验收、可对外声明的能力（限定包络）

1. **charge 约束**：PW 与 LCAO 共口径；力 FD 通过（PW 全 9 轴 PASS、LCAO 双轴 PASS），
   力矩 FD PASS；适用于分子/离子体系（I-1 证线性区 ≥±0.8 e）。
2. **spin 约束**：PW/LCAO 读数-注入算符同构；**力 FD 已闭合**（LCAO 全 9 轴 + PW 冒烟 2 轴，
   §3.11）⇒ 力/几何相关用途可在该包络内声明；**物理解读仍只能用"Becke 加权矩"口径**
   （≠ d 局域矩，闸门 2 未过）。
3. **混合 charge+spin 同原子/同 run**：可用，μ 耦合已实测；跨类型耦合不可忽略（阶段 B 依据）。
4. **收敛调度**：OUTER（默认）+ INNER 双迭代，正确性等价已验（μ* ≤0.17%、E_tot ≤8.5e-7 eV），
   成本收益**逐体系**（−75%…+14%），决策表入手册 §5.9。
5. **熔断纪律**：μ 顶限熔断 + **在线能量分支守卫**（默认关）；换态不再静默。
6. **CDFT 物理接口**：III-1 一阶（能量/μ 恒等式）与二阶（κ 分支、W）读数可用；
   重组能数值仍待加密扫描。

### 4.2 闸门（未过则不可做的物理）

| 闸门 | 阻塞的物理 | 缺口 | 成本 |
|---|---|---|---|
| ~~V1 自旋力 FD~~ | 约束 relax / MD、几何驱动、自旋通道力/应力 | **已过（2026-09-14，§3.11）**：LCAO 9/9 + PW 2/2 轴；混合通道力 FD 仍缺 | 混合通道 ~1 h |
| **II-1b 脱钩定量 + 4b 半径敏感性** | TM d 矩的物理解读 | Becke 矩≠d 矩未定量收口；II-1 需重锚定 | 中（依赖 V3b 只读观测口） |
| **V6 Au:Si 资源** | bulk 生产表征 | Au 赝势缺口 | 待资源裁定 |

### 4.3 阶段 B（Broyden/Jacobian）立项数据已齐备

| 输入 | 数据 | 出处 |
|---|---|---|
| μ 耦合表 | Δμ_c=−0.004819 / Δμ_s=−0.009206（同向刚化） | §3.4 |
| 收敛步数退化 | 混合外环 3 → 15 步 | §3.4 |
| INNER 收益曲线与交叉点 | −75%…+14%（逐体系），`mix_reset` 刚需、settle 3 次假收敛 | §3.8 |
| 门控标定 | 纯成本旋钮，默认保持 1e-3 | §3.8 |

⇒ 对角 secant 不再充分的证据链完整，Broyden/Jacobian 立项**具备数据条件**。

### 4.4 流程教训（写进纪律）

1. **跑批产物必须落仓库**：长跑脚本一律 `RESDIR` 归档（README + `results/*.audit` +
   `summary.txt`），禁止只留 `/tmp`（V1 两轮作废的直接原因）；V1 补跑已在 runner 落地
   并产出 41 个证据文件。
2. **OMP_NUM_THREADS=1**：对照/验收跑统一；autotest 1e-7 eV 阈值与该噪声同量级。
3. **能量口径**：判据一律 raw `FINAL_ETOT_IS`；跨单位比较（eV vs Ry）必须先换算。
4. **元数据即契约**：`--help` 与结构体真值必须同步；生成文档的发布时间点跟随接口冻结。

## 5. 下一步（Next steps）

1. **混合通道（charge+spin 同原子）力 FD**：LCAO 全轴 + PW 冒烟（213 几何，协议复用本轮）
   ——同族最后一环，过后力 FD 覆盖三种通道；
2. **约束 relax 单步冒烟**（几何驱动）：验证跨离子步 λ/μ 生命周期（C-12/C-13 族）在自旋通道不复发；
3. **III-1 加密** δ ∈ [−0.05, +0.10]（0.01–0.02 e）→ 判定 δ≈0 拐点性质，再谈 Marcus 重组能；
4. **4b 半径敏感性 / II-1 重锚定**（合批）→ 收口闸门 2；
5. **阶段 B 立项评审**（输入已齐，见 §4.3）；
6. **merge-readiness**：届时统一重建 `docs/parameters.yaml` + `input-main.md`（接口冻结后）；
7. FeO 双稳对照**维持暂缓**（用户批复）；V6 Au:Si 待资源裁定。
