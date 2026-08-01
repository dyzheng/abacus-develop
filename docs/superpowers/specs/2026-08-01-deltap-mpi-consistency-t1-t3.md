# 2026-08-01 DeltaP MPI 一致性迭代（T1 + T3）

> 评审 TODO 执行第一轮：T1（LCAO `sync_lambda` Bcast 后写回 operator）+ T3（PW γ 跨
> rank 一致性同步），合成一次 MPI 一致性小迭代。对应评审文档 §3 的 T1/T3。

## 1. Test plan

1. 实施 T1：LCAO backend `sync_lambda` 在 `MPI_Bcast` 后把（rank0 的）λ 写回
   `dp_op`，使非 rank0 的 `dp_op` λ / escon 与 rank0 一致（对齐 PW 侧
   `s_lambda = lam` 的既有修复）。
2. 实施 T3：PW backend `compute_gamma` 在本地 Wilson-loop 计算后，将 γ 按 rank0
   Bcast 到所有 rank（对齐历史 gamma 同步 `750ee179d` 的 rank0-wins 策略），保证
   `gamma_report` / escon 跨 rank 一致。
3. 验证矩阵：
   - 单测 16/16 不回归；
   - PW 1-rank A/B：与 R5 基线逐字节对比物理输出（CG 序列、`[DeltaP-PW]` 行）；
   - PW 2-rank：per-rank 临时诊断打印 γ/escon 完全一致 + rank0 输出与 1-rank 基线
     一致（浮点噪声内）；
   - LCAO 4-rank（方阵进程网格）：per-rank 临时诊断打印 λ 完全一致 + P2 触发点与
     R5 记录一致（iter=11）；
   - 排查 h2o_lcao 4-rank 崩溃是否为本次改动引入。

## 2. Test setup

- 系统：`/root/abacus-develop`，分支 `feat/deltap`，HEAD `8f0a1342d`；构建
  `CCACHE_DISABLE=1 cmake --build build --target abacus_basic_para -j14`。
- 算例：
  - PW：`/tmp/deltap_r1_smoke/h2o_pw`（3 原子，无 target，two-phase 阈值模式）；
  - LCAO：`/tmp/deltap_r5/deltap_bn_test`（BN 2 原子，8 k 点，2×2×2，target.dat，
    4-rank 方阵进程网格）；
  - 崩溃排查：`/tmp/deltap_r1_smoke/h2o_lcao`（2 k 点 4 rank，非方阵）。
- 临时诊断：`report_pw` 内 per-rank 打印 `RANKDBG escon/gamma`；LCAO `sync_lambda`
  内 per-rank 打印 `LAMDBG lambda`；验证后已移除，最终二进制无诊断输出。
- 基线：`/tmp/r1_backup/run_post_pw.baseline.log`（R5 1-rank PW）、
  `/tmp/deltap_r5/deltap_bn_test/run_mpi4.log`（R5 4-rank LCAO）。

## 3. Results

### 3.1 单测（最终二进制）
- `MODULE_ESOLVER_deltap_common_test`：10/10 PASS。
- `MODULE_ESOLVER_esolver_dp_test`：6/6 PASS。

### 3.2 PW 1-rank（T3 对 NPROC=1 为 no-op）
- `exit=0`；`[DeltaP-PW]` 行与基线逐字节一致：
  `drho=1.530e-03 γ_total=-0.1592 rad λ_avg=2.041e-03 |res|=4.739e+00 escon=-0.023414 Ry γ/atom=(4.738603, 0.692700, 0.692692)`；
- CG 能量/EDiff/DRHO 序列与基线逐字节一致（仅计时列与 commit/时间戳不同）。

### 3.3 PW 2-rank（T3 生效）
- `exit=0`；带诊断的复跑中 `RANKDBG` 两 rank 完全一致：
  `rank=0/1 escon=-0.023407 gamma=(4.73774, 0.69312, 0.693101)`；
- 最终（无诊断）运行 rank0 输出与带诊断运行逐字节一致（确定性）：
  `drho=1.504e-03 ... escon=-0.023407 Ry γ/atom=(4.737743, 0.693120, 0.693101)`；
- CG 收敛序列与 1-rank 基线在浮点噪声内一致（能量差 ≤ ~1e-5 Ha）。

### 3.4 LCAO 4-rank（T1 生效）
- `exit=0`；带诊断的复跑中 `LAMDBG` 四 rank 完全一致：
  `lambda=(3.262e-06, -3.501e-06)`，与 rank0 P3 打印 `λ=(3.26e-06, -3.50e-06)` 一致；
- `[DeltaP P2] iter=11 drho=6.92e-06 < 1.00e-03` —— 与 R5 记录（P2 iter=11）一致；
- P1 轨迹比旧 `run_mpi4.log` 更平滑（旧日志 iter=2 有 γ=(2.082,1.454) 大扰动，
  新日志无），λ 量级相同（~3e-6 Ry）；最终 γ→(4.0,3.5) target，物理一致。

### 3.5 h2o_lcao 4-rank 崩溃排查（非本次改动）
- 新二进制与旧二进制（stash 掉 T1/T3 后重建）**均**在进程 1 以 exit=1 静默退出，
  位置在 iter=1 Phase-1 report 之后、`sync_lambda` 执行之前（0 条 LAMDBG）——
  结论：**既有问题**，非 T1/T3 引入。与评审记录一致：LCAO 2-rank 方阵网格限制
  （`cbb37b7ae`）同族，h2o_lcao（2 k 点 4 rank，非方阵）本身不在已验证矩阵内。

## 4. Analysis

- T1 修复有效且必要：LCAO escon 直接读 `dp_op->get_lambda()`（`deltap_scf.cpp`
  `iter_finish`），旧代码先 `set_lambda`（本地 λ）再 Bcast 不写回，非 rank0 的
  `dp_op` λ 与 escon 与 rank0 不一致；4-rank 复跑中 LAMDBG 证明写回后四 rank 一致。
  副作用（预期修正）：非 rank0 的 H(k) 也改用 rank0 λ，消除了旧多 rank 轨迹中
  iter=2 的大扰动，轨迹更平滑、收敛物理不变。
- T3 修复有效：PW γ 经 `unkdotp_G`/`unkdotp_G0` 的 `MPI_Allreduce(POOL_WORLD)`
  与 `cal_becp` 的 `reduce_pool` 后本应一致，但缺少显式保证；Bcast rank0 结果后
  RANKDBG 证明两 rank 完全一致。rank0 值本身不受 Bcast 影响，1-rank 基线无回归。
- 观察项（记录非阻塞）：PW rank0 的 γ 测量值随 nproc 有 ~0.02% 差异
  （1-rank 4.738603 vs 2-rank 4.737743，escon 差 7e-6 Ry），与 CG 能量浮点并行噪声
  同量级，属既有测量 nproc 敏感性（`compute_per_atom_gamma_kstring` 内 becp/重叠
  并行求和顺序），不在本轮范围，列入后续观察。

## 5. Next steps

1. T2：PW `deltap_init` 改惰性 init + 每离子步 `reset_ionic_step`（对齐 LCAO），
   随 relax/多离子步支持一起做。
2. T7（最大专项）：`cal_force_stress` nlm 越界读 + 双释放修复（nlm 布局统一 +
   长度守卫 + H_HK/∂τ/∂R 力项 + `run_fd.sh` 验收）。
3. T4–T6、T9–T13 清理项可随日常迭代合入。
4. 观察项：PW γ 的 nproc 敏感性如后续影响数值基线，可评估统一 becp/重叠的求和
   顺序（或在文档中固定按 nproc 的参考值）。

## 6. Files modified

- `source/source_esolver/esolver_ks_lcao.cpp`（T1：sync_lambda 写回 + dp_op 捕获）
- `source/source_pw/module_pwdft/deltap_pw.cpp`（T3：compute_gamma rank0 Bcast）
- 文档：本文件 + `deltap-development-log.md` 追加本节
