# T-7' per-atom Jacobian 解耦（componentwise secant）——实现 + 集成验证

日期：2026-08-13
状态：**单元层 PASS（8/8）/ 集成层 FAIL（机制定位）** —— 按纪律"偏离即停"，先落档再立项修复

## 1. 测试计划

T-7'（原方向 2）目标：消除 ow γ-drive / proxy 驱动内循环的单 α 线搜索对
反号 per-atom 残差分量的相互污染（T-2 根因②、T3' a2 的 α 翻号
−0.249→+0.197→−0.139）。验证分三层：

1. **单元层**：FletcherReevesCG 新增 componentwise 模式（每分量独立 secant
   α_opt[i]=α_trial[i]·(−r_i·Δr_i)/Δr_i²，每分量 max_step 限幅 + 自适应 γ）。
   反号对角系统 r₁=2λ₁−1、r₂=−3λ₂+1：jacobi 应显著快于 scalar。
2. **集成层**：D1 几何（h2o1，O=7.9365、H z=8.5223，KPT 1×1×2，gdir=3）
   ow γ-drive 探针 a2（O1 靶点 = 自然 + 0.02 rad，H 同自然，inner_nmax=20）：
   cg 对照臂应复现 T-18 a2 失败签名（α 翻号、rms 平台）；jacobi 臂应
   **内循环收敛（rms < 1e-3）且 α 不翻号**。
3. **回归**：bfgs 8/8、deltap_common 11/11、esolver_dp 6/6；主二进制重建。

## 2. 测试设置

- 系统：h2o1 盒 15.873 Å、O(7.9365,7.9365,7.9365)、H z=8.5223、KPT 1×1×2、
  ecutwfc 100 / ecutrho 400、smearing gauss 0.002、genelpa、scf_thr 1e-8。
- 目录：`/tmp/deltap_t7p/{ow_nat, ow_a2_cg, ow_a2_jacobi}`。
- 公共 INPUT：`deltap_operator_mode ow`、`deltap_drive gamma`、
  `deltap_conv_thr 1.0e-3`、`deltap_branch_write 0`（a2，守卫协议）。
- ow_nat：inner_nmax=0、无 target（自由跑，定自然 γ + branch.dat 参考）。
- a2：inner_nmax=20、target.dat（O1 = 自然+0.02 rad）、
  branch.dat 拷贝自 ow_nat；inner_scheme 分别为 cg / jacobi。
- 全部单任务串行（OMP_NUM_THREADS=1，禁 MPI 并行）。

## 3. 结果

### 3.1 单元层（bfgs_test 8/8 PASS）

- `ComponentwiseOppositeSignConverges`：jacobi 1 步收敛到 λ=(0.5, 1/3)。
- `ComponentwiseBeatsScalarOnOppositeSign`：`cw_ok=1 steps=1` vs
  `sc_ok=0 steps=60`——反号对角系统 scalar-α 60 步不收敛，componentwise
  1 步收敛。T-2 根因②的单元层实证。

### 3.2 顺带修复：空字符串 INPUT 值段错误（read_sync_string 守卫）

- 现象：`deltap_target_file  `（空值行）→ INPUT 解析期段错误
  （`std::string _M_assign`，failing at 0x8）。
- 根因：`read_information` 对空值行产出**空** `str_values`；宏
  `strvalue = item.str_values[0]` 对空 vector 是 UB（suffix 空值只是运气
  好没崩，`suffix  ` 实测退化为空字符串）。非 T-7' 引入的既有脆弱点。
- 修复：`read_sync_string` 增加 `if (!item.str_values.empty())` 守卫，
  空值保留参数默认值（UB → 确定行为）。验证：`deltap_target_file  ` 行
  不再崩溃，ow_nat 正常收敛（149 s）。

### 3.3 集成层：ow_nat 自由跑（λ=0，自然 γ）

- 自然 γ（ow 口径，D1 几何，branch.dat 17 位）：
  O=(0,0,−7.9584104105891766)、H1=(0,0,−2.3796617613107962)、
  H2=(0,0,−2.3796617612249267)。
- Γ=(−16.642,−4.050,−4.050)，escon=0，SCF 收敛。

### 3.4 集成层：ow_a2 对照（O1 靶点 = 自然 + 0.02 rad）

| 指标 | cg（scalar-α） | jacobi（componentwise） |
|---|---|---|
| 内循环 | 19 步不收敛，rms 1.13↔1.55e-2 振荡 | 20 步不收敛，rms 恒 ~1.13e-2 |
| α 行为 | **翻号**：−2.35e-2→+1.17e-2→−2.03e-2→+1.02e-2→… | **不翻号**：稳定 +0.25（max_step 钳位 0.005/0.02） |
| 终态 λ | (+1.216e-2, −3.6e-4, −3.6e-4) | (+9.73e-3, −1.09e-3, −1.09e-3) |
| \|γ−t\| | 1.795e-2 | 2.137e-2（≈靶点全量，**零进展**） |
| escon | 0.2045 Ry | 0.1577 Ry |
| SCF | 未收敛（iter 44+ 停） | 未收敛（iter 44+ 停） |

T-18 a2 失败签名（α 翻号、rms 平台）在 cg 臂完整复现；jacobi 臂消除了
翻号，但**内循环依然不收敛**。

### 3.5 机制证据（jacobi 臂，[SBdbg] iat=0，O1）

内循环 trial 的 raw / best（报告 γ）轨迹（单位 rad）：

```
raw        best（报告）   shift（分支吸收）
-7.9584    -7.9584        0        （λ=0 起点，r_O1=−0.0200）
-7.9648    -7.9648        0        （trial 0，rms→1.5426e-2）
-7.9584    -7.9584        0
-7.9632    -7.9583        +4.9e-3
-7.9706    -7.9580        +1.26e-2
-7.9773    -7.9584        +1.89e-2
-7.9712    -7.9595        +1.17e-2
-7.9648    -7.9583        +6.5e-3
-7.9708    -7.9583        +1.25e-2
…（raw 在 −7.965…−7.977 振荡，best 恒钉 −7.958±0.001）
```

- **raw 相位对 λ 有响应**（Δraw 达 ±0.019 rad，λ 预算内），但**报告 γ 被
  连续性锚钉死**在自然值 −7.9584 ± 0.001——branch shift 逐 trial 吸收 raw
  移动（shift 0→4.9e-3→1.26e-2→1.89e-2→…）。
- 内循环残差 r = γ_report − t ≈ **−0.020 常数 ±1e-3 抖动** → rms 恒 ~1.13e-2，
  任何光滑优化器（CG 或 Jacobi）都驱动不了；λ_O1 单调爬到 max_step 上限
  +9.7e-3，|γ−t| 停在 2.1e-2。

## 4. 分析

### 4.1 T-7' 单元层成立，集成层被可观测量钉死

- jacobi 按设计消除了 α 翻号（优化器层修复有效），但 ow γ-drive 内循环的
  **驱动可观测量（连续性锚定后的报告 γ）在首轮冻结前被锚钉在自然值附近**：
  分支选择（Stage-B，target=ref_gamma_，branch.dat 锚）把读数吸到离锚最近的
  格点族，raw 的物理响应被 shift 吸收。残差退化为常数 → 内循环结构性不可收敛。
- 这正是 T4a 教训（"target-aware 分支选择的 γ 报告是靶点跟随的，不能作为
  收敛判据"）在**内循环**的新形态：连续锚修复了外循环的靶点跟随，却让内循环
  读不到物理响应。T-18 a2 的"冻密度 BFGS 不收敛"由此获得更精确的归因——
  不是（仅）冻结-自洽响应分裂，而是**报告 γ 的锚定量化**（raw 响应存在，
  report 不动）。
- 旁证：T3' 评审预言的"γ(λ) 在该区间连光滑可逆都做不到"在冻密度层得到
  定量确认（raw 响应 ±1.3 rad/Ry 量级且符号随 trial 翻转，叠加锚定吸收）。

### 4.2 空值 INPUT 段错误是独立既有 bug，已修

见 §3.2。与 T-7' 无关，但咬住了工作流（ow_nat 首次启动即崩），按根因
修复原则落 4 行守卫，零回归（11/11、6/6、8/8）。

### 4.3 评审要点的回应

- 评审"方向 3（γ 直驱）采纳为主机制"——本轮的机制证据说明：γ 直驱的
  **内循环可观测量必须先解锚**（驱动 raw γ 或冻结 branch-shift 读数），
  否则直驱在 ow 模式没有可驱动的信号。这回答了 T-6' 节遗留的开放问题
  （"γ 弱耦合+非单调是测量通道问题还是算符控制权限问题"）：在首轮冻结前，
  **测量通道（锚定报告）本身就是阻塞**；算符控制权限（dγ/dλ 0.26 vs 3）
  是第二层限制。
- T-7' 基建（per-atom Jacobian）对两套驱动仍有效，保留；集成 FAIL 的
  根因不在优化器。

## 5. 结论与下一步

- **T-7' 判定**：优化器层 ✅（8/8 单测 + 集成无翻号）；ow γ-drive a2 内循环
  ❌（残差常数化，|γ−t|=2.1e-2 零进展）。按"偏离即停"，不在此轮扩大使命。
- **下一步立项（T-7''，机制级）**：ow γ-drive 内循环驱动可观测量改 **raw γ**
  （`compute_gamma_raw` 已接线）或**冻结 branch-shift 读数**，弃用锚定报告；
  落地后重跑 ow a2（cg/jacobi 两臂）验证内循环收敛性。T-18 的 0.98 靶点
  判决仍受 |λ*|_max≈0.4–1.6e-3 Ry 工作窗限制（口径见 T-18 §1.5）。
- 空值 INPUT 守卫已随本轮 commit。
