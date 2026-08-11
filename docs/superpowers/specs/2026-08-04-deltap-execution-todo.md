# DeltaP 执行 TODO 指导（2026-08-04，当前唯一权威执行清单）

> 路线依据：`2026-08-04-deltap-force-resolution-plan.md`（三阶段）、
> `2026-08-04-deltap-route-a-plus-design.md`（设计）、
> `2026-08-04-deltap-route-a-plus-derivations.md`（联动推导）。
> 执行纪律（dev log 流程教训）：每步带验收与可证伪预言；偏离预言即停；
> PASS 必须注明独立参照物；改 SCF 的步**一次只做一个**，锚点重建只发生两次
> （S_k 修复后一次、Route A+ operator 模式一次）。

---

## ★ 当前 TODO（2026-08-11 评审裁定后，以此节为准）

> 依据 2026-08-11 评审：T-2"失败"重定位 = Route A++ §2.4 首次实验确证
> （H 的 Γ 代理方向性失控，dΓ_H/dλ_H≈0 但 dγ_H/dλ_H 强）；方向 3 转正
> （γ 残差直驱 λ，secant/t_Γ 翻译层退役，留开关 A/B）；一行定理入设计：
> E'=E_KS(ψ*) 恒等式与驱动信号无关，泄漏仍 O(λ)（但残差大小 ∝ λ*²，
> 依赖驱动信号耦合强度——T3' 实测印证，见 T-4'）。
> Ô_w（L3.1）优先级提升至 hk MPI 之前；T-9' 守卫随下一个 commit。

| # | 任务 | 内容 | 验收（参照物） | 状态 |
|---|------|------|----------------|------|
| **T-1** | **Phase 0.3-lite 分支锚定门控** | INPUT `deltap_branch_anchor=continuity\|target`：operator 默认 continuity，gamma 锁定 target | ① gamma 零回归；② t_γ=自然值两锚一致；③ λ=0 自由首测报告自然 γ | ✅（③ ✓、① 门控锁定未跑 BN 对照） |
| **T-2** | **T4a' 外循环机制验证** | t_γ=0.98γ_natural 固定几何 secant | 预言 2–4 步收敛 → ❌ 发散，根因三层钉死（λ 预算饥饿/单 α 反号/H Γ-proxy 退化） | ❌ **重定位为 Route A++ §2.4 首次实验确证**（详见 2026-08-09 文档） |
| **T-3** | **commit（T-1+T-2 + 文档）** | Phase 0.3-lite + T4a 前置代码 + 根因文档 | CI 式回归：单测 17/17、MPI smoke 4/4 | ✅ **8e55ec821** |
| **T-4'** | **方向 3：γ 直驱 λ + T3' 复判**（~1 天） | `deltap_drive=proxy\|gamma` 开关；gamma 直驱下 secant/t_Γ 退役（开关保留）；escon 永远 −λ·Γ；T3' = γ 驱动驻点 FD 三几何 | 实现+回归 ✅（单测 11+3+4、smoke 4/4）；**T3' 判决 ❌ FAIL（结构性）**：残差 −0.502 eV/Å = 判据 25×/T3 代理版 36× 差——γ↔λ 弱耦合（dγ/dλ≈−0.3）强制 λ* ~14×、O(λ)² 泄漏 ~200×，叠加冻结-自洽响应符号分裂。驻点 FD 唯一合法协议 = Γ 代理驱动（T3 PASS 成立） | ✅ 实现 + ❌ T3'（详见 2026-08-11 文档） |
| **T-5'** | **窗口测绘 0.995/1.005 双向**（~0.5 天） | 正反两方向各测，目的改为"定量记录 H 的单侧可达域"，产出直接进 LIMITATION | 改用 proxy 驱动（Γ 约束）；γ 报告仅作外循环读数 | ☐ |
| **T-6'** | **L3.1 Ô_w 提前**（评审已裁） | θ_n·P̂_I 替换 τ_α（开关隔离 + D2 跳变冻结约束）→ V-H8（SCF 稳定）+ H 可达性复测 | 判决：γ_H>自然 是否变得可达 | ☐（**优先级 > hk MPI**） |
| **T-7'** | **per-atom Jacobian 解耦**（原方向 2） | 单 α 线搜索无法处理反号分量（T-2 根因② + T3' 冻结-自洽分裂两案前置证据） | 两套驱动（proxy/gamma）都受益 | ☐ |
| **T-8'** | **L1 三件**（可任意穿插，每件 ~0.5 天） | L1.1 PW Γ 记账；L1.2 ⟨η⟩ 完备性输出；L1.3 spread_I 诊断 | 各件独立验收（T2 类比 / H2O ⟨η⟩<1% / 均匀 θ spread=0） | ☐ |
| **T-9'** | **branch.dat 写入守卫** | INPUT `deltap_branch_write`（默认 true；FD/多几何 false） | 随 T-4' commit 落地 | ✅（随本轮 commit） |
| T-10' | Stage 3 hk MPI + L2 应力（生产面，顺序不变） | 本地列→全局带映射；H_HK 应力 + PW 应力 | hf/co corr=1 4-rank 与串行一致；静水压+剪切 FD <1% | ☐ |

**立即执行：commit T-4'+T-9' → T-6'（Ô_w 提前）→ T-7' → T-8'（穿插）。**
T-5' 依赖 T-4' 结论（改用 proxy 驱动）。Stage 3 hk MPI、L2 应力生产面顺序不变。

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
| 1.1a | ~~`DeltaP` 新增成员 `std::vector<double> gamma_op_`（nat）+ `compute_operator_observable()`~~ **✅** | `deltap.h`（members 区）| 编译过 ✅ |
| 1.1b | ~~Γ_I^HR 在 `compute_gamma_scf` 顺带累加~~ **✅**（物理 k 点 `j < nppstr_-1`，排除包裹副本） | `deltap_wannier.cpp` | T0 见下 ✅ |
| 1.1c | ~~Γ_I^HK 按原子拆出~~ **✅**（w_IJ[p][iat] + T_diag，`compute_hk_correction`/`compute_hk_force`/`compute_gamma_op_hk`） | `deltap_wannier.cpp` | E_HK=ΣλΓ 自洽 ✅ |
| 1.1d | ~~**T0**~~ **✅**：per-k ⟨P̂⟩ == 实空间 hhrdbg，12 位全同（7.199716499951 2.140887285468 2.140887285237） | h2o1/base 1-rank | **PASS**（见 stage1 dated 文档 §3.1） |

### 1.2 状态机切换（deltap_scf）

| # | 任务 | 文件锚点 | 验收 |
|---|------|----------|------|
| 1.2a | ~~INPUT 新增 `deltap_observable`~~ **✅**（读入回显 + 非法值 WARNING_QUIT 验证过） | `input_parameter.h` + `read_input_item_other.cpp` | 读入打印正确 ✅ |
| 1.2b | ~~`DeltapState` 加 `gamma_op` + backend `compute_gamma_op` 回调~~ **✅** | `deltap_scf.h`、`esolver_ks_lcao.cpp` | — |
| 1.2c | ~~残差口径切换~~ **✅**（iter_finish/update_lambda_gd/inner_loop 用 `scf_observable`；gamma 模式逐字节不回归 ✅） | `deltap_scf.cpp` | gamma 零回归 ✅ |
| 1.2d | ~~escon 切换~~ **✅**（operator 模式 `compute_dp_escon(lambda, gamma_op)`；单测补 `ComputeDpEsconOperatorObservable` 11/11） | `deltap_common_test.cpp` | ✅ |
| 1.2e | ~~target 语义~~ **✅**（t_Γ 初值=t_γ；init 打印标注） | `deltap_scf.cpp` init | 打印标注 ✅ |

### 1.3 外循环 secant（最小实现）

| # | 任务 | 锚点 | 验收 |
|---|------|------|------|
| 1.3a | ~~`DeltapState` 加 `t_proxy`、`gamma_meas_prev`、`t_proxy_prev`~~ **✅**（+ secant 发散计数） | `deltap_scf.h` | — |
| 1.3b | ~~secant 挂钩~~ **✅**（relax=`reset_ionic_step`；单点=`iter_finish` conv 判定 + 每 SCF 一次守卫；κ clamp [0.3,3]/限幅 0.5 rad/发散 WARNING 已实现） | `deltap_scf.cpp` | 单点触发一次 ✅（relax 由 gamma 回归覆盖） |
| 1.3c | ~~打印~~ **✅**：P3 加 Γ 列；E-field `E_eff=λ/(2a)` operator-ramp（V1 钉符号） | `deltap_scf.cpp` report | 输出格式见 dated 文档 §3.4 ✅ |

**Stage 1 出口检查**：~~gamma 模式全锚点逐字节一致~~ **✅**（relax 0 diff；bn_test P 行逐字节；
center 差异=分支文件加载状态，确定性验证）；~~operator 模式编译+冒烟跑通~~ **✅**
（h2o1/base rc=0，Γ 列/operator-ramp/secant 触发，见 `2026-08-04-deltap-stage1-route-a-plus.md`）。

---

## Stage 2：Route A+ 串行判决（T1–T5，~1 天，决定路线成败）

按设计文档 §5 执行，全部 h2o1 串行、生产设置（ecutwfc=100/ecutrho=400/scf_thr=1e-8，
`OMP_NUM_THREADS=1` 或 4，写进 run 脚本）：

| # | 测试 | 可证伪预言 | 偏离时的动作 |
|---|------|-----------|--------------|
| T1 | E' 恒等式（E' vs E_KS(ψ*)） | 差 <1e-8 eV | 差大 → escon 接线错，回 1.2d |
| T2 | ∂E'/∂λ 重测（base λ 扫描） | 224 eV/Ry → ≲1 eV/Ry（O(λ)） | 仍是 O(1) → Γ 与 H_c 不一致，回 1.1 |

**Stage 2 进展（2026-08-05 凌晨，T2 判定 + 1.1 修复回环）**：

- **T2 首轮实测 FAIL**：λ∈[−0.01,+0.01] 扫描 E'(λ) 斜率 −13.3 eV/Ry（O(1)，非 O(λ)）。
  按偏离动作"仍是 O(1) → Γ 与 H_c 不一致，回 1.1"定位：
  **Γ_I^HK 的记账用了 E_HK-split 对角约定（−0.5·Im[Σ f_p w_IJ T_pp]），
  而 H_c 里实际施加的 H_HK 算符期望是 Tr[ρ·H_sym] = 全 T·Π Gram 迹**；
  非正交 LCAO 基下 Π=C_L†C_L ≠ I，对角约定把耦合高估 ~18%
  （实测 E_HK_conv=0.0574 Ry vs E_HK_actual=0.0470 Ry @ λ=+0.01）。
- **修复（1.1 回环）**：`deltap_wannier.cpp` 的 `compute_hk_correction` 与
  `compute_gamma_op_hk` 的 Γ_I^HK 改为按原子拆分的实际算符期望
  Γ_I^HK = −0.5·Im[Σ_j Σ_p f_p·Σ_{p'} w_{I,p'}·T_{pp'}·Π_{p'p}]（T·Π 全迹）。
  escon = −ΣλΓ 现在等于 −⟨H_c⟩（精确），E' = E_Harris − ⟨H_c⟩ ≡ E_KS(ψ*) 恒等式恢复。
- **T2 复测 PASS（硬信号）**：同批 λ 扫描斜率 **−0.013 eV/Ry**（比首轮小 1000×，
  比判据 ≲1 小 ~80×）；±0.001 两点 E' 逐位对称（8 位一致），E'(±0.01) 呈 ~λ² 抛物
  （+1.55/+1.30 meV），且 E' ≥ E_KS(ρ₀) 变分下界恢复（首轮 −0.01 侧违反）。
- **T1 判定 PASS（构造性恒等式）**：T0（Γ^HR 12 位）+ T2 探针（Γ^HK=实际期望）
  使 escon ≡ −⟨H_c⟩ 精确成立 → E' ≡ E_KS(ψ*) 代数恒等（<1e-8 机器精度，无近似）。
  单测 11/11 + 6/6 回归通过。详见 `2026-08-04-deltap-t2-escon-hk-trace-fix.md`。

**T3 前检查（Q1/Q3）**：
- Q3 冻结协议已接线：`deltap_proxy_target_file`（init 加载冻结 t_Γ*，MPI bcast）
  + `deltap_secant off`（secant 短路，t_Γ 不再漂移）→ disp± 只重收敛 λ 使
  |Γ−t_Γ*|<1e-3。见 `deltap_scf.cpp` init/`secant_update_proxy`。
- Q1 λ* 数据（T2 扫描提取）：dΓ/dλ ≈ −4.2 Ry/Ry（ΣΓ: 10.697@−0.01 → 10.614@+0.01），
  dγ/dλ ≈ −0.3 rad/Ry（γ: −5.521→−5.515→−5.521）。Γ→0 需 λ*≈2.5 Ry（大），
  γ 对 t_Γ 的映射斜率 ≈ 0.07 rad/单位 → T4 首轮 κ=1 会偏小，需实测 Δγ/Δt_Γ 更新 κ
  （Q2 的翻号重启逻辑已在 secant 实现）。
| **T3** | **驻点组② 复判**（三几何驻点 FD；约束变量=Γ，驻点判据 \|Γ−t_Γ\|<1e-3） | **残差 84.8 → ≤0.02 eV/Å** | **≫0.02 → 停**，残差分解（λ-leak 重算）找未识别项 |

**Stage 2 进展（2026-08-05，T3 判决 PASS —— 判决点通过，Route A+ 成立）**：

- **前置（按用户评审）**：F_HK 力侧先做全迹对齐（`compute_hk_force` 的
  E_HK/F_HK/U 从对角 T_pp 扩到 T_full·Π 全迹，Π=C_L†C_L 冻结 C 下为常数，
  导数链不动），E_HK 0.05732 → 0.0471550682 Ry（−17.75%，与 Gram 修正 ~18% 吻合）；
  双闭合通过：均匀平移 E_HK-FD −9.0e-5 ↔ −ΣF_HK −9e-5 Ry/Bohr ✓、escon 总量
  +0.432 ↔ +0.4252 eV/Å（1.6%）✓。单原子 E_HK 单独 FD 4.7× 失配 = C-响应项
  （∂E_HK/∂C·∂C/∂R ∝ λ，B-7 的 4% 是 B-6 前 τ 单位 bug 掩盖的假象，非本轮回归）。
- **新修复（T3 接线断点）**：`inner_loop` BFGS 残差用了 `params_.t`（per-atom
  模式为空 → r=Γ−0 把 Γ 驱动到 0 而非 t_Γ*）→ 改 operator 模式用 `scf_target`
  （=t_proxy=冻结 t_Γ*）。修复后 base 收敛 λ=(0,0,0)、Γ=t_Γ* 到 4.5e-4。
- **T3 实测 PASS**：t_Γ*=(6.698,1.977,1.977) 冻结，三几何内循环 BFGS 重收敛
  λ（\|Γ−t_Γ*\|∞ ≤1.035e-3）；F_FD(O1z)=−0.76102 ↔ F_ana=−0.7472648965 eV/Å →
  **残差 −0.0138 eV/Å ≤ 0.02 判据（84.8 → 0.0138，6100×）**。
  λ-leakage：F_FD,KS=−92.92 + F_FD,escon=+92.16 相消（T2 平直性 dE'/dλ≈0
  保证相消——Route A+ 核心收益实证）；disp± 平直性复测 ≤5.4e-5 eV ✓；
  λ*(R) 无分支阶梯 ✓。回归 11/11、6/6、3/3 PASS（smoothness 4/8 FAIL 预先存在）。
  全部数据见 `2026-08-05-deltap-fhk-fulltrace-t3.md`。
- **诚实标注（评审追加，§3.7）**：严格判据线上未过——残差 0.0138 > 严格闭合判据
  0.0129 eV/Å（超 7%）；闭合计算 `leak=−λ*·(dΓ/dλ)·Δλ*/(2δ)=+0.0127~0.0129 eV/Å`
  （λ*_avg=−1.09e-3、dΓ/dλ≈−4.15、2δ=0.01 Bohr）→ **残差 ~100% 归因为 O(λ)
  驻点泄漏，无未识别项**（7% 缺口 = λ* 精度 + Σ斜率代单分量）。生产验收判据
  表述：**"在 \|λ*\|≤λ₀ 工作窗内残差 ≤X"**（残差 ∝\|λ*\|，λ→0 残差→0 线性）。
| T4a | 外循环机制验证（t_γ=0.98γ_natural，固定几何 scf） | secant ≤5 步 \|γ−t_γ\|<1e-2（预言 2–4 步）；κ 实测更新；翻号逻辑 | 发散 → κ 限幅/映射单调性检查 |
| T4b | 窗口测绘（0.95→0.9 逐步） | 每步记录 (t_Γ, λ*, γ, E')；终点冻结 λ FD → 画"力残差 vs \|λ*\|"曲线 | 残差不随 \|λ*\| 线性 → 响应项重估 |
| T5 | 组① 冻结 λ FD | 0.615 → ~0.05 eV/Å（λ·dΓ/dR） | 显著更大 → 响应项重估 |

**T4 前置修正（评审修正 2，2026-08-05）**：t_Γ 单步限幅 0.5 → 1.0 rad；
κ 实测后改用割线预测步长（实测 κ≈14 > 原 clamp 上限 3，否则"≤5 步"
结构性不可能）；新增 scf 模式固定几何外循环驱动（secant 更新 t_Γ 后
\|γ−t_γ\| 未达 tol 则继续 SCF 而非终止）。评审修正 1：先 T4a（0.98）后
T4b（0.95→0.9），不直接上 0.9（Δλ≈1.9 Ry 暴力微扰区）。

**T4a 判决性发现（2026-08-05 深夜）**：0.98 靶点**低于 per-atom γ 的分支量子
分辨率**——global target-aware branch selection 每次把报告 γ 重锚到离 t_γ 最近
的分支（量子 ~0.07–0.105 rad @ 2-k 网格），λ=0 自由跑已报告 γ≈t_γ
（\|γ−t_γ\|=5.1e-3，λ 全程 0），外循环 1 步假收敛。对照实验（target=自然 → γ=
自然）钉死 1:1 跟随。**Phase 0.3（分支连续性）升格为外循环硬前置**：分支参考
须与 t_γ 解耦（锚定上次测量值），并先用 λ=0 自由跑钉住自然参考分支。修复后再
跑 T4a（0.98）→ T4b。详见 `2026-08-05-deltap-t4a-branch-anchor.md`。

**T3 是判决点**：~~通过 → Route A+ 成立，进 Stage 3~~ **✅ 已通过（2026-08-05，
残差 84.8 → −0.0138 eV/Å）**。下一步：先 T4（外循环，唯一剩的 Stage 2 项，依赖
T3 结论——现在有结论了），T4 通过后统一提交 Stage 1+2 再进 Stage 3。

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
