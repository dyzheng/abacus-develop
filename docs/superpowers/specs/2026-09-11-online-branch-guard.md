# 在线能量分支守卫（`constraint_branch_tol` → `MuStatus::BRANCH_FLIP`）落地

> 批复顺序第 3 项（`2026-09-11-ii1-triage-review.md` §三.3）：能量式在线分支守卫，
> 要求"触发即熔断不静默、默认关、单测 + sabotage"。本轮交付实现 + 单测 + 两处真实
> 端到端证据（H₂O 正向控制、FeO 换态点应熔断）。

## 1. 测试计划（Test plan）

| # | 目标 | 判据 |
|---|---|---|
| G1 | 默认关逐位不变 | `constraint_branch_tol = 0`（缺省）时不进守卫代码块：无新输出行、结论与能量与改前一致 |
| G2 | 触发即熔断、不静默接受 | SCF 已收敛但 `E_tot < E_ref − tol` → `MuStatus::BRANCH_FLIP` + `phase_ = DONE`，**即使残差已达标也不得报 CONVERGED** |
| G3 | 良态 run 无假阳性 | 能升 = `½·|δ|·|μ*| > 0`（FeO II-1 实测形状）时全程不触发 |
| G4 | 容差边界 | 下探 `0.5·tol` 不触发，`2·tol` 触发（同一 run 的判别对） |
| G5 | 接线/组合守卫 | 开守卫但无能量输入 → WARNING_QUIT；与 `ABA_CONSTRAINT_FIXED_MU` 同开 → WARNING_QUIT |
| G6 | sabotage（守卫↔测试一一对应） | 逐条置假守卫，检查恰中对应测试、其余保持绿 |
| G7 | 真实体系端到端 | H₂O `211_PW_constraint_h2o`（正向控制：不假触发、默认关无变化）；FeO 已知换态点（应熔断） |

## 2. 测试设置（Test setup）

- 平台：容器 16 GB / 14 核（OMP 3）。
- 单测：`build/`（debug，BUILD_TESTING=ON）
  `make MODULE_ESTATE_constraint_loop MODULE_ESTATE_constraint_io` +
  `ctest -R constraint`（11 个注册目标）。`MODULE_ESTATE_constraint_loop` 用例数
  12 → **18**（新增 4 个功能用例 + 2 个死亡测试）。
- 生产构建：`build_rel/abacus_basic_para`（`input_parameter.h` 变更必须重链）。
- sabotage 方法：在 `constraint_loop.cpp` 上逐条把守卫条件置假 → 重链 loop 目标 →
  跑该目标 18 个用例记录红/绿分布 → 还原（备份 `/tmp/loop_backup.cpp`）。
- 端到端 A（正向控制）：`tests/01_PW/211_PW_constraint_h2o` 副本，`mpirun -np 4`，
  两次运行：缺省（关）与追加 `constraint_branch_tol 1e-3`；另跑两次"关"作噪声基线。
- 端到端 B（换态熔断）：FeO II-1 的已知换态点
  `δ = +0.3 μB on Fe(2)`，热启动自 `/tmp/feo_spin/S0B_warmstart/OUT.autotest`
  （亚稳锚，scf_thr 1e-7、scf_nmax 400、ecut 50、dft_plus_u、step_max 0.05、
  probe 0.0），追加 `constraint_branch_tol 1e-3`；对照量取上一轮（守卫不存在时）
  同一设置的 `H3_fe2_p03_hot` 结果。

## 3. 结果（Results）

### 3.1 单测（11/11 目标全绿，~50 s）

```
ctest -R constraint → 100% tests passed, 0 tests failed out of 11
MODULE_ESTATE_constraint_loop: 18 tests, 18 PASSED
```

新增用例：`BranchGuardDefaultOffDoesNotFuse`、`BranchGuardRisingEnergyConverges`、
`BranchGuardToleratesDipWithinTolerance`、
`BranchGuardPreemptsTargetReachedOnFlippedBranch`、
`BranchGuardRefusesMissingEnergyDeathTest`、`BranchGuardRefusesFixedMuDeathTest`；
`constraint_io_test` 的 `OuterStepCapAndBranchGuardGuards` 增补
"`branch_tol` 透传 / 负值 ERROR"两条断言。

### 3.2 sabotage（4 发，全部恰中）

| # | 破坏（`constraint_loop.cpp`） | 结果 |
|---|---|---|
| S1 | 守卫整块停用（`if (false && cfg_.branch_tol > 0.0)`） | 3 FAIL：`RisingEnergyConverges`、`PreemptsTargetReachedOnFlippedBranch`、`RefusesMissingEnergyDeathTest`；**默认关用例与其余 12 个保持绿** |
| S2 | 判据比较置假（`if (false && de < -cfg_.branch_tol)`） | **恰 1 FAIL**：`PreemptsTargetReachedOnFlippedBranch` |
| S3 | 缺能量接线守卫置假（`if (false && !scf_energy_set_)`） | **恰 1 FAIL**：`RefusesMissingEnergyDeathTest` |
| S4 | fixed-μ 兼容守卫置假 | **恰 1 FAIL**：`RefusesFixedMuDeathTest` |

S2 是本轮的核心判别：它证明"残差达标 + 能量换态"这一点确实被守卫拦下（若删掉比较，
该用例退化为 CONVERGED 而红）。

### 3.3 端到端 A：H₂O 211（正向控制，无假触发）

| run | `constraint_branch_tol` | `!FINAL_ETOT_IS` [eV] | 守卫输出 | 终态 |
|---|---|---|---|---|
| 关 #1 | 0（缺省） | −441.9708338945813 | 无 | CONVERGED |
| 关 #2 | 0（缺省） | −441.9708338353173 | 无 | CONVERGED |
| 关 #3 | 0（缺省） | −441.9708338572362 | 无 | CONVERGED |
| 开 | 1e-3 Ry | −441.9708338380483 | e_ref=−32.493 Ry；de=+0.00427509/+0.00711699/+0.00854805/+0.00874213/+0.00874217 Ry | CONVERGED（不假触发） |

- "关"三次运行的固有抖动 **5.9e-8 eV**（并行/线程非确定性），"开−关"差
  5.7e-8 eV 落在该抖动内；结构上守卫开启时 `branch_tol = 0` 直接跳过整块（S1 亦证）。
  入库 `result.ref`（−441.9708338609649 eV）同样落在该抖动带内。
- 约束能升实测量级 +0.0087 Ry ≈ 0.119 eV，比 `1e-3 Ry` 容差高 **~9×**（无假触发余量）。

### 3.4 端到端 B：FeO 换态点（应当熔断）

| 量 | 值 |
|---|---|
| e_ref（μ=0 参考能量） | −562.440523 Ry（= −7652.3959 eV，S0B 亚稳锚） |
| 外步 1（SCF iter 24）de | **+0.013322 Ry（+0.1813 eV）**——仍在参考分支 |
| 外步 2（SCF iter 44）de | **−0.047280 Ry（−0.6433 eV）** → 熔断 |
| 熔断点读数 | Q = 3.2033775，t = 3.437560545（res = −0.23418），μ = −0.1 Ry |
| 终态 | `BRANCH_FLIP`（"the targets are NOT validated"），进程正常退出 |
| 对照（上一轮、无守卫） | `H3_fe2_p03_hot`：同一设置**跑到外步 6 报 CONVERGED**，E = −7652.995586 eV（相对锚 **−0.600 eV**） |

审计痕迹入库：`tests/deltap_feo_spin_scan/results/guard/GUARD_fe2_p03_hot.audit`。

### 3.5 预存在失败（与本次改动无关，未修）

`build/` 全量 `make` 在 `MODULE_ESTATE_charge_test`、`MODULE_ESTATE_elecstate_energy`、
`MODULE_ESTATE_elecstate_print` 三处失败（缺 `InfoNonlocal` 符号 / 缺
`ElecState::get_dftu_energy` / `InfoNonlocal` 未声明）。`git stash` 去掉本轮全部改动后
`make MODULE_ESTATE_charge_test` 复现同一链接错误 ⇒ **预存在**，本轮不动（超出授权范围）。

## 4. 分析（Analysis）

1. **判据口径**：`E_tot = E_KS + Σ_α μ_α(Q_α − t_α)`，`E_ref` 取自 μ=0 参考相
   （该相 `e_con` 恒为 0，故 `E_ref = E_KS,ref`）——两侧同口径，不需要额外修正项。
   经验依据是 II-1b §4.1：良态 run 的能升 = `½·|δ|·|μ*|` 且为正（FeO δ=+0.1 吻合
   1.4%），因此"低于参考超过容差"只能来自换态，不是约束本身的代价。
2. **顺序是关键**（本轮最容易埋错的一处）：守卫必须在 `outer_step()` **之前**判。
   `outer_step` 推进 `mu_`；若在其后判，就会拿"新 μ"配"旧 Q"，既换口径、又会漏掉
   "残差已达标但已换态"的点。S2 sabotage（置假比较）恰中
   `PreemptsTargetReachedOnFlippedBranch`，正是这条顺序语义的判别测试。
3. **能量入参必须去修正项**：hook 时刻 `f_en.etot` 仍含**上一外步**的 `cc_escon`
   （`iter_finish` 的 `cal_energies(2)` 会重算 etot），故 esolver 传
   `etot − cc_escon`（纯 E_KS），约束项由 loop 自己按"本次 SCF 实际使用的 μ"加回。
   两个 esolver（PW/LCAO）同步接线，无第二路径。
4. **熔断语义对齐 UNREACHABLE**：`conv_esolver` 保持 true（SCF 本身确实收敛了），
   由外环给出判决 `phase_ = DONE` + `BRANCH_FLIP`，进程正常退出、照常写能量/密度。
   这是与 `UNREACHABLE` 同族的"熔断"：**判决在状态 token 里，不在退出码里**，因此
   用户/脚本必须看 `final status`。手册 §5.7 已把这条写成强制项。
5. **FeO 端到端证明了两点**：(a) 判据在真实 DFT+U 多解体系上有效（−0.643 eV 被抓）；
   (b) 熔断发生在外步 2，而旧流程要到外步 6 才"跟踪到靶点并 CONVERGED"——守卫不仅
   阻止了错误结论，还**阻止了把已换态轨迹继续当参考分支跟随**（那 4 个外步的 μ/Q
   全部来自错误分支，本就不该产生）。这也说明守卫不会误伤良态点：外步 1 的
   +0.181 eV 正常通过。
6. **已知盲区（设计边界，非 bug，已写入手册 §5.7 与开发者文档 §3.2）**：
   ① 只抓"能量向下"的换态（换态后能量仍在 `E_ref` 之上不触发）；
   ② 只在 SCF 收敛点判，固定 μ 下不收敛的点（II-1 δ=±0.3/±0.5 的极限环）表现为
   RUNNING，守卫无从介入；③ 容差是绝对能量（Ry），需要使用者按体系选择量级
   （建议：≫ SCF 能量噪声、≪ `½|δ||μ*|`）。
7. **默认关的纪律**：`branch_tol` 缺省 0、走老路径；S1 sabotage 显示即使把整块停用，
   "默认关"用例仍绿——说明"关时不变"这件事由缺省值而非守卫代码保证（这正是要求
   "默认关"的意义：不给存量 run 引入新判决）。

## 5. 下一步（Next steps）

1. （批复第 4 项）审计行打印同原子 **on-site 投影矩**（最小 II-1b）+ Fe 分区半径
   敏感性研究（定量 Becke 矩 ≉ d 矩的幅度）；
2. （批复第 5 项）能力边界文档补"Becke 矩/d 局域矩脱钩"条目，并把本守卫的两条盲区
   登记在案；
3. （批复第 6 项）**I-1 MgO 继续**（唯一干净体系），FeO S4/S5 暂缓；
4. 框架内待议：把 FeO 扫描的 `BRANCH_TOL` 打开重测 S3L 窗口（本轮已给 runner 加
   `BRANCH_TOL` 旋钮与审计抓取），以及"B1 事后判据是否升为框架默认"——本轮**不动**，
   按评审要求等守卫真实使用一轮后再判；
5. 顺带（非阻塞）：修 `build/` 里三个预存在的测试链接/编译失败（缺 mock 符号）。
