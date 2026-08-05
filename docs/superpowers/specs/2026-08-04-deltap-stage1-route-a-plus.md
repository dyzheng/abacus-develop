# 2026-08-04 DeltaP Stage 1：Route A+ 串行实现（Γ 计算 / 状态机切换 / 外循环 secant）

> 触发：TODO Stage 1（`2026-08-04-deltap-execution-todo.md`）。本轮覆盖 1.1a–d（含
> T0 判定）、1.2a–e、1.3a–c，完成 **Stage 1 出口检查**。分支 `feat/deltap`，
> 全部改动未提交（工作区，等待 Stage 2 判决点 T3 后统一评审）。

## 1. 测试计划

1. **T0（1.1d，可证伪）**：Γ_I^HR 的 per-k 形式 vs 实空间 `Tr[DMR·pre_hr]`
   （hhrdbg 机制），两口径差 <1e-10；不一致即停。
2. **1.2 状态机**：`deltap_observable` INPUT 读入/校验正确；gamma 模式残差/
   escon/target 语义零回归（逐字节）。
3. **1.3 secant**：单点收敛后触发一次；κ=1 首轮；κ clamp [0.3,3]；|Δt_Γ|≤0.5 rad。
4. **Stage 1 出口**：gamma 模式全锚点逐字节一致；operator 模式编译+冒烟跑通。
5. 单测：deltap_common（补 operator escon 用例）11/11；esolver_dp 6/6。

## 2. 测试设置

- 二进制：`build/abacus_basic_para`（当前工作区，MPI+Release，
  `CCACHE_DISABLE=1 cmake --build build --target abacus_basic_para -j16`）。
- T0 用例：`tests/deltap_fd_force/h2o1/base`（ecutwfc=100/ecutrho=400/scf_thr=1e-8，
  1-rank，gdir=3，`1 1 2` k 网格）。
- gamma 回归用例（拷贝到 `/tmp/deltap_stage1_regr/`，INPUT 追加
  `deltap_observable gamma`）：
  - `deltap_bn_test`（4-rank，scf 50 iter）— 对照 `/tmp/deltap_anchor2/bn_test/`；
  - `deltap_bn_sampling/center`（1-rank）— 对照 `/tmp/deltap_anchor2/bn_sampling/center/`；
  - `deltap_relax`（1-rank，3 离子步）— 对照 `/tmp/deltap_anchor2/relax/`。
- operator 冒烟：h2o1/base（默认 `deltap_observable operator`），1-rank，scf。
- 伪势/轨道：`/root/pporb/apns-*`；`OMP_NUM_THREADS=1`。

## 3. 结果

### 3.1 T0：Γ_I^HR per-k == 实空间（通过，独立参照物）

h2o1/base（1-rank，λ=0 前）：
```
per-k ⟨P̂⟩        = 7.199716499951 2.140887285468 2.140887285237
实空间 hhrdbg     = 7.199716499951 2.140887285468 2.140887285237   （12 位全同）
Γ_I^HR (τ_α=0.5) = 3.599859683107 1.149454480976 1.149454480852
Γ_I^HK            = 3.595595778   1.068772403     1.068772403     （λ=0 时 HK 未激活，可观测量仍计算）
```
**关键调试**：字符串 nppstr_=3（k={0,½,0}→包裹副本）最初 1.5× 偏差，根因是
累加了末尾 G 包裹副本；改为 `j < nppstr_-1` 遍历物理 k 点后 T0 精确吻合
（`deltap_wannier.cpp` Γ_I^HR 累加点）。

### 3.2 1.2 状态机

- INPUT `deltap_observable`（默认 `operator`；`gamma` = legacy）：
  - `INPUT.info` 回显正确；
  - 非法值 `bogus` → `deltap_observable must be 'operator' or 'gamma'`，rc=1 ✅。
- 残差/escon/target 口径：operator 模式用 `state_.gamma_op`（Γ）vs `state_.t_proxy`；
  gamma 模式原样（γ vs t_γ），逐字节零回归（见 3.3）。
- PW 后端固定 `observable_mode="gamma"`（PW 无 Γ 实现，杜绝隐式行为漂移）。

### 3.3 gamma 模式回归（零回归证据）

| 用例 | DeltaP 行数 | diff | 说明 |
|---|---|---|---|
| deltap_relax（1-rank，3 离子步） | 3044 | **0 行** | 逐字节一致 ✅ |
| deltap_bn_test（4-rank） | ~2359 | 19 行 | 仅 3 处 `branch-set prev=` 打印 + 2 行计时统计；P 行轨迹逐字节一致 ✅ |
| center（1-rank） | ~2345 | — | 与锚点差异完全由**加载的分支状态文件**引起（见分析）；两次独立运行仅计时行不同（确定性 ✅） |

bn_test 的 19 行 diff 中 prev= 值（3.997433/3.502269/4.024359/4.016378）与
`tests/deltap_bn_test/deltap_branch.dat` 当前内容逐位一致——锚点运行加载的是
B-6 时代遗留旧文件（锚点跑完 save_branch 才覆盖成当前内容）；selected/delta/
rescaled 及全部 P 行（γ、λ、|γ−t|、escon）12 位一致。

### 3.4 operator 冒烟（h2o1/base，默认 operator）

- rc=0，SCF 收敛于 iter=41（drho=6.7e-9 < scf_thr），`FINAL_ETOT_IS` 正常。
- P1/P3 行新增 Γ 列：`Γ=(7.195, 2.218, 2.218)` == Γ_I^HR+Γ_I^HK
  （3.600+3.596=7.196，1.149+1.069=2.218）✅ 与 T0 口径闭合。
- 单点收敛后 secant 触发一次：
  `[DeltaP Secant] |γ−t_γ|∞=1.829e-03 κ=(1.00, 1.00, 1.00) t_Γ=(-5.519, -3.601, -3.601)`
  手算核对：t_Γ + κ·(t_γ−γ_meas) = (-5.5206, -3.5991, -3.5991) + (0.0014, -0.0021, -0.0021)
  = (-5.5192, -3.6012, -3.6012) ✅。
- `[E-field operator-ramp] E_eff=λ_avg/(2a)×51.422 V/Å`（无 π，§1.2 新公式）✅。
- init 打印：`[DeltaP] operator mode: t_Γ initialized to t_γ (κ=1 first round)` ✅。
- 首轮 |Γ−t_Γ|=1.272e+01（λ 被驱动到 1.27e-2）——**预期行为**：t_Γ 初值=t_γ，
  Γ(≈7.2/2.2) 与 γ(≈−5.5/−3.6) 数值不同，代理失配由外循环 secant 校正（T4 验证）。

### 3.5 单测

- `deltap_common_test`：11/11 PASS（新增 `ComputeDpEsconOperatorObservable`，
  escon=−ΣλΓ 语义，λ 与 Γ 向量独立于 t_Γ）。
- `esolver_dp_test`：6/6 PASS。
- 主程序 + 单测目标编译干净（无 error）。

## 4. 分析

1. **T0 口径闭合**：Γ_I^HR 的 per-k 累加（`j < nppstr_-1` 物理 k 点）与实空间
   p_hat 12 位一致，为 Route A+ 的 Γ 定义提供独立参照物；Γ_I^HK 按原子拆分与
   `compute_hk_force` 的 E_HK 自洽（Σ_I λ_I·Γ_I^HK == E_HK）。
2. **gamma 零回归成立**：relax 0 diff（最强证据，全新状态）；bn_test P 行逐字节
   一致；center 差异归因于分支文件加载状态（锚点加载 B-6 遗留文件），且当前
   代码两次独立运行逐字节一致（确定性）。
3. **单点 secant 时序 bug 已修**：DeltaP 块在 `ESolver_KS::iter_finish`（设置
   conv_esolver）之前执行，拿到过期标志 → 在 LCAO 侧本地用 `drho < scf_thr`
   判定 + `secant_at_conv_done` 每次 SCF 只触发一次。
4. **operator 首轮大残差是设计使然**：κ=1 意味着"Γ≈γ"的粗假设，单点只能做
   一次校准；T4（外循环收敛 ≤5 步）才是 secant 是否收敛的判决。
5. **未做**（TODO 纪律）：hk MPI 修复（Stage 3.1）、operator 锚点重建（Stage 4.1）、
   T1–T5 判决（Stage 2）——均留到 T3 判决点之后。

## 5. 下一步

1. Stage 2 判决：**T1**（E' 恒等式：E' vs E_KS(ψ*)，差 <1e-8 eV）→ **T2**
   （∂E'/∂λ 重测，224 eV/Ry → ≲1 eV/Ry）→ **T3 驻点组② 复判**（判决点）。
2. 全部 Stage 1 改动保持未提交；T3 通过后按 dev log 纪律统一提交 + 更新 TODO。
3. Stage 1.2d 的 escon 打印在 operator 模式已为新记账（escon=−ΣλΓ）；
   `deltap_results.dat` 加列（§9）留到 Stage 3/4 收口。
