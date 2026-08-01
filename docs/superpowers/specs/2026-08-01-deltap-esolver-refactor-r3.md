# 2026-08-01 DeltaP 重构 R3：PW 接入同一 DeltapScfSolver 状态机

> 状态：**完成并验证**（A/B 全等 + LCAO/内循环回归）
> 前置：R1（死代码清理，07-31）、R2（LCAO 状态机，08-01）
> 总方案：`2026-07-31-deltap-esolver-refactor-design.md` Round 3

## 1. Test plan

1. 编译 `module_pwdft` + `esolver` + `abacus_basic_para`（CCACHE_DISABLE=1）。
2. PW 同步两阶段路径 A/B：R3 二进制 vs R1 基线（`/tmp/r1_backup/run_post_pw.baseline.log`），
   DeltaP 行 + SCF 轨迹（能量/EDiff/DRHO）逐字节一致。
3. LCAO 同步路径回归：R3 vs R2 日志（`run_r2_lcao.log`），DeltaP 行全等。
4. LCAO 内循环路径（nscf=4）功能冒烟：`inner loop done` 值一致、SCF 收敛。

## 2. Test setup

本机 Release + MPI1 进程；`build/abacus_basic_para`。
- PW：`/tmp/deltap_r1_smoke/h2o_pw/`（ecut30，berry_phase=1，gdir=3，total 模式 INPUT，
  deltap_switch=true，inner_thr=1e-2，无 STRU target）。
- LCAO：`/tmp/deltap_r1_smoke/h2o_lcao/`（λ=0 同步）与 `h2o_inner/`（nscf=4 + target.dat）。
- 冒烟命令：`timeout 900 mpirun -np 1 ... > run.log 2>&1`。

## 3. Results

| 检查项 | 结果 |
|---|---|
| 编译 | PASS（仅既有 `if constexpr` C++17 警告，LCAO 侧 R2 已有） |
| PW A/B DeltaP 行 | `[DeltaP-PW]` 两行（init + drho 报告）逐字节 **IDENTICAL** |
| PW A/B SCF 轨迹 | CG1–CG13 能量/EDiff/DRHO **IDENTICAL**；仅墙钟列不同 |
| PW 耗时 | 38.60s → 35.75s（删除重复第二次 γ 测量 + 死状态） |
| LCAO 同步回归 | 1909 行 DeltaP/[rawG]/[E-field] **IDENTICAL** |
| LCAO 内循环 | `inner loop done: final l0=-1.3039e-02 l1=-4.3066e-03 l2=-4.3066e-03`（与 R2 冒烟一致），GE39 DRHO=6.4e-07 收敛 |

R3 实际运行日志：`/tmp/r3_pw.log`、`/tmp/r3_lcao.log`、`/tmp/r3_inner.log`。

## 4. Analysis

### 4.1 结构变化

- **`deltap_pw.cpp` 全局单例 → `DeltapScfSolver` 实例**（匿名 namespace 持有）。
  删除：`s_lambda_set`/`s_gamma_total`/`s_dp_escon`/`s_gamma_prev`/`s_targets` 5 个文件级状态。
  保留：`s_lambda`/`s_constrain`（对外算子状态，forces/stress/op_pw_proj 读取）。
  原 7 全局 → 2 算子状态 + 1 状态机实例。
- **PW backend**（`make_backend`）：`set_lambda`→写回 `s_lambda`（算子消费者不变）；
  `get_lambda`→读 `s_lambda`；`compute_gamma`→`compute_per_atom_gamma_kstring` 折叠到
  gdir 的 1D per-atom γ（nocc 由 PARAM 现取，与历史一致）。
- **`deltap_common` 新增 `unwrap_2pi`**（最近分支 2π unwrap）；LCAO 的多带
  `select_branch_set` 仍在 module_deltap，不动。
- **`DeltapScfSolver` 泛化两点**（对 LCAO 零影响，A/B 已证）：
  1. `DeltapParams::unwrap_branch_2pi` + `DeltapState::gamma_prev`：PW 的跨 SCF 分支跟踪
     移入状态机，`reset_ionic_step()` 一并清空。
  2. `DeltapState::gamma_report`：分支选择后的 γ 用于 max_res/escon/report；
     LCAO 的 branch selection 在 `compute_gamma` 内已完成 → gamma_report == gamma_I。
- **`iter_finish` 顺序微调**：γ 测量 → P2 λ 更新 → 分支选择 → max_res/escon/HK → 报告。
  LCAO 的 max_res 值不变（同一 γ），输出逐字节一致。
- **esolver_ks_pw.cpp 接线**：init 块 20 行 → 6 行 `pw_deltap::deltap_init(...)`；
  `deltap_iter_finish`/`reset_deltap_pw_scf_cycle`/`get_deltap_pw_escon` 调用点原样保留。

### 4.2 行为保真点（复刻历史语义）

- **PW 无 target = 约束 γ→0**：历史上 `targets[iat]` 对空 vector 越界读（UB，实际为 0），
  λ 仍按 residual=γ−0 更新（冒烟 λ_avg=2.041e-03 与基线一致）。R3 在 `deltap_init` 中
  显式把空 target 填 0 向量（修 UB，行为不变）。
- **`[DeltaP-PW]` 报告逐字节保真**：drho / γ_total（独立 Wilson 环总数）/ λ_avg（全原子均值）/
  |res|（仅自由原子、缺 target 按 0）/ escon / γ/atom 格式照抄；掩码 max_res 在 `report_pw`
  内按历史语义计算（状态机内 max_res 对 PW 不打印）。
- **`inner_nmax>0` 的 `WARNING_QUIT` 拒绝**：消息逐字保留，触发点仍在 `deltap_iter_finish`
  （`switch && corr` 门控之后），触发时机比历史略早（不再先算 γ 再退出）。
- **total 模式不一致**：INPUT `deltap_constraint_mode=total` 但 PW 历史上一直按 per-atom 更新；
  R3 保持 per-atom（`p.total_mode=false`），标注为已知不一致，留后续轮统一。
- **MPI 不变量**：PW 侧仍无 λ Bcast/rank 守卫（历史 S-09），R4 处理；状态机对 PW 不启用
  `sync_lambda`/`on_phase2`/`apply_hk_correction`，与历史一致。

### 4.3 修复

- 空 target 越界读 UB（见上）。
- `s_gamma_total` 死状态删除（只写不读）。
- PW 报告不再重复第二次 `compute_per_atom_gamma_kstring`（确定性测量，与第一次同值，
  纯浪费；顺带 ~3s 提速）。
- NaN-γ_total 路径：历史在设置 `lambda_set` 之后才查 NaN（会烧掉本 SCF 周期的更新机会）；
  R3 改为先查 NaN 再进状态机（非退化配置下不可达，属清理而非行为变化）。

## 5. Next steps

1. **R4**：MPI rank0+Bcast 读 target/约束矩阵（C-23）；PW 的 λ Bcast / rank 守卫（S-09）；
   LCAO 悬空 else 打印清理；real 实例 WARNING 收口；`deltap_pw.cpp` 的
   `set_deltap_pw_lambda/targets` 残余接口收口（已无 setter，getter 保留给算子消费者）。
2. **可选**：PW 支持 `deltap_target_file` / `deltap_constraint_matrix`（状态机已具备，
   仅 PW 侧未接线）；PW total 模式统一（当前 INPUT total 但实际 per-atom）。
3. `deltap_common` 单测（`source/source_esolver/test/` 框架）。
4. 更新总设计文档状态为"已实施"。
