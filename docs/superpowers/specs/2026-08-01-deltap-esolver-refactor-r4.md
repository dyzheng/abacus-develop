# 2026-08-01 DeltaP 重构 R4：MPI 收敛 + 打印/WARNING 收口 + 单测

> 状态：**完成并验证**（三路冒烟回归 + 单测全过）
> 前置：R1（死代码清理）、R2（LCAO 状态机）、R3（PW 接入状态机）
> 总方案：`2026-07-31-deltap-esolver-refactor-design.md` Round 4

## 1. Test plan

1. 编译 `module_pwdft` + `esolver` + `abacus_basic_para`（CCACHE_DISABLE=1）。
2. 三条冒烟回归：PW（h2o_pw）A/B vs R1 基线、LCAO 同步（h2o_lcao）vs R2、
   LCAO 内循环（h2o_inner，nscf=4）功能冒烟。
3. 新增 `deltap_common` 单测：编译并运行 `MODULE_ESOLVER_deltap_common_test`。

## 2. Test setup

本机 Release + MPI1；`build/abacus_basic_para`；`build/source/source_esolver/test/MODULE_ESOLVER_deltap_common_test`。
冒烟命令：`timeout 900 mpirun -np 1 ... > run.log 2>&1`。

## 3. Results

| 检查项 | 结果 |
|---|---|
| 编译 | PASS（仅既有 `if constexpr` C++17 警告） |
| PW A/B | `[DeltaP-PW]` 两行逐字节 **IDENTICAL**；CG1–13 能量/EDiff/DRHO 一致（仅墙钟列不同） |
| LCAO 同步回归 | DeltaP/[rawG]/[E-field] 行 **IDENTICAL** |
| LCAO 内循环 | `inner loop done: final l0=-1.3039e-02 ...` 与 R2 一致，GE39 DRHO=6.4e-07 收敛 |
| `deltap_common` 单测 | 10 个用例 **PASSED**（0 ms） |

运行日志：`/tmp/r4_pw.log`、`/tmp/r4_lcao.log`、`/tmp/r4_inner.log`。

## 4. Analysis

### 4.1 C-23：target/约束矩阵 rank0 读 + Bcast（`DeltapScfSolver::init`）

- target 文件：rank0 解析（total 模式单值 / per-atom 向量）→ `bcast_bool(loaded)` +
  `bcast_double(total_target)` + `bcast_double(file_target, nat)` → 各 rank 写同一 `params_.target`。
- 约束矩阵：rank0 解析 m×n + C/t → Bcast `file_open`/`loaded`/`m`/`n` + C/t 数据 → 各 rank 重建 `params_.C/t`。
- 复用 `Parallel_Common::bcast_*`（MPI_COMM_WORLD，与 LCAO λ Bcast 同一通信域）。
- 文件打不开/缺行：保持历史语义（静默保留 STRU 目标）。

### 4.2 约束矩阵尺寸不匹配 → WARNING_QUIT

历史：`std::cerr` 提示后静默丢弃约束（危险）；现在：rank0 `WARNING_QUIT`（消息文案保留，
"expected N columns, got m"），文件打开但尺寸错 = 配置错误直接终止。

### 4.3 S-09：PW λ Bcast + rank 守卫

- PW backend 增 `sync_lambda`：`NPROC>1` 时 `bcast_double(lam)`（rank0 胜出），随后
  `s_lambda = lam` 刷新各 rank 算子存储 → `get_lambda`/escon 全 rank 一致（历史缺陷修复）。
- `[DeltaP-PW]` init 消息与 `report_pw` 打印加 `GlobalV::MY_RANK == 0` 守卫（消除多 rank 重复输出）。
- 注：γ 测量本身（Wilson loop）的跨 rank 一致性不在本轮范围，文档标注留待后续核对。

### 4.4 LCAO 打印/WARNING 收口

- 悬空 else 清理：`deltap_init` 的 STRU target 打印改为 `has_any_target && MY_RANK==0`
  一条（历史非 rank0 打印矛盾的 "No targets" 行，R2 起标注，本轮删除）。
- real 实例 WARNING 收口：从 `iter_finish` 每 SCF 迭代重复 → `before_all_runners` 一次性
  （`if constexpr (!complex)` + rank0），文案保留，函数标签改为 before_all_runners。

### 4.5 deltap_common 单测

新增 `source/source_esolver/test/deltap_common_test.cpp`（10 用例）覆盖
`compute_residual`（per-atom 缺 target 按 0 / 矩阵模式）、`max_norm`、`gd_update`（掩码/mixing）、
`gd_update_total`、`to_effective_lambda`（恒等/矩阵）、`compute_dp_escon`、`unwrap_2pi`（最近分支/空 prev）。
挂 `AddTest(MODULE_ESOLVER_deltap_common_test, LIBS parameter ${math_libs} base device)`。

## 5. Next steps

1. 设计文档状态 → "已实施"（本轮已改）；总验收：`esolver_ks_lcao.cpp` DeltaP 净删 ≥350 行、
   `deltap_pw.cpp` 净删 ≥150 行、`deltap_solver.h` 消失、`deltap_common` 全部函数被调用且有单测 — 全部满足。
2. 可选后续：PW 接 `deltap_target_file`/`deltap_constraint_matrix`（状态机已支持，PW 侧未接线）；
   PW total 模式统一（INPUT total 但实际 per-atom）；PW γ 测量跨 rank 一致性核对；
   2-rank MPI 冒烟验证 C-23/S-09（本轮环境只有单进程可用）。
