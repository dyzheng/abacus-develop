# 2026-08-01 DeltaP esolver 重构 R2：抽取 DeltapScfSolver 状态机

> 设计依据：`2026-07-31-deltap-esolver-refactor-design.md` §4 Round 2
> 目标：把 LCAO esolver 内的 DeltaP SCF 控制流（状态、门控、λ 更新、内循环、报告）
> 下沉到 basis-independent 的 `DeltapScfSolver`，`iter_finish` 从 ~140 行收缩到 ~22 行。

## 1. Test plan

1. 编译：`esolver`（含新 `deltap_scf.cpp`）与 `abacus_basic_para` 链接通过；`double` 实例
   下 complex-only 回调被 `if constexpr` 正确丢弃。
2. 同步两阶段路径（nscf=0）：R2 二进制 vs R1 二进制，同一 H₂O 算例 stdout 逐字节 diff，
   DeltaP 行必须 IDENTICAL（R1 已与 HEAD 原版 A/B 全等，等价于 R2 vs 原版）。
3. 内循环路径（nscf=4 + target 文件）：R2 二进制功能冒烟——`inner loop start/done`、
   λ 更新、SCF 收敛。
4. 泄漏修复回归：LCAO `deltap_init` 三个 raw `new` 已改 `unique_ptr`（R1 起），R2 继续无泄漏。

## 2. Test setup

- 系统：本机 14 核，Release，MPI 单进程，ccache 禁用。
- 二进制：`build/abacus_basic_para`。
- 同步路径算例 `/tmp/deltap_r1_smoke/h2o_lcao/`：H₂O 30 Bohr，`ecutwfc=50, scf_thr=1e-6,
  deltap_switch=1, deltap_corr=1, deltap_inner_nmax=0, deltap_lambda_init=0, deltap_gdir=3`（λ=0 基线）。
- 内循环算例 `/tmp/deltap_r1_smoke/h2o_inner/`：同上但 `deltap_inner_nmax=4,
  deltap_conv_thr=1e-3, deltap_target_file=target.dat`（target=[-0.1,-0.05,-0.05]）。
- 对比基线：R1 二进制日志 `/tmp/r1_backup/run_post_lcao.log`（该日志已与 HEAD 原版逐字节全等）。

## 3. Results

| 项 | 结果 |
|---|---|
| 编译（esolver + abacus_basic_para） | PASS |
| 同步路径 DeltaP 行 diff（R1 vs R2） | **IDENTICAL**（含 [DeltaP P1]、[rawG]、[E-field]、[DeltaPOp]） |
| 同步路径全日志 diff | 仅日期 + GE 行末计时列（非确定项） |
| 内循环冒烟（nscf=4） | PASS：`inner loop start: nscf=4` → `inner loop done: final l0=-1.30e-2 l1=-4.31e-3 l2=-4.31e-3`，SCF 收敛（E_KohnSham 稳定） |
| `iter_finish` DeltaP 块 | ~140 行 → 22 行 |
| 代码量 | 新增 `deltap_scf.{h,cpp}`（~470 行）；`esolver_ks_lcao.cpp` 净删 ~430 行 |

## 4. Analysis

- 新组件：
  - `source/source_esolver/deltap_scf.{h,cpp}`：`DeltapScfSolver` 状态机，持有
    `DeltapParams`（INPUT 快照 + target/C/t）+ `DeltapState`（λ/γ/phase flags/escon），
    通过 `Backend` 回调注入基组操作（set_lambda / get_lambda / apply_hk_correction /
    compute_gamma / solve_frozen / sync_lambda / on_phase2 / get_optimizer / 诊断钩子）。
  - `deltap_common.h` 重建为纯函数库：`compute_residual` / `max_norm` / `gd_update` /
    `gd_update_total` / `to_effective_lambda` / `compute_dp_escon`（R2 起全部有调用者）。
- ESolver 侧：8 个成员 + 3 flags → 1 个 `unique_ptr<DeltapScfSolver>`；四个 helper
  （init/compute_gamma/inner_loop/update_lambda）→ `deltap_init`（基建+绑定）+ 
  `deltap_make_backend`（10 个一行 lambda 回调）；`iter_finish` 只做：惰性 init、
  `reset_ionic_step()`（iter==1）、`iter_finish()`、回写 `f_en.dp_escon`。
- **行为保真要点**：
  1. 复刻了原代码悬空 `else` 的打印语义（has_any_target=false 时不打印；true 且非 rank0
     时打印 "No targets"）——已加注释，R4 清理。
  2. 内循环 BFGS 对象仍由 `DeltaP` 持有，状态机经 `get_optimizer` 回调驱动；
     `solve_frozen` 每 inner 迭代新建 HSolverLCAO（与原来每 inner_loop 新建一次等价，
     构造无状态依赖）。
  3. `if constexpr (TK==complex)` 保护 `compute_gamma`/`compute_gamma_raw`/
     `apply_hk_correction` 三个 complex-only 回调（显式实例化 double 版本必需）。
  4. 修 C-22（连带）：`lambda_cstr` 初始化在矩阵模式为 m 维（原回退路径误用 nat 维）。
  5. 顺带修复：内循环 per-atom 残差在 target 为空时的越界读（UB → 按 0 处理），
     非 UB 路径行为不变。

## 5. Next steps

1. R3：PW 接入同一状态机——`deltap_pw.cpp` 全局单例改 `DeltapScfSolver` 实例 +
   PW backend（compute_gamma = k-string Wilson loop），统一两阶段门控与离子步重置
   （删 `s_lambda_set`，用 `reset_ionic_step()`），`deltap_common` 增 `unwrap_2pi`。
2. R4：MPI rank0+Bcast 读 target/矩阵（C-23）、悬空 else 打印清理、real 实例 WARNING 收口。
3. 补 `deltap_common` 单测（source/source_esolver/test/ 框架）。
