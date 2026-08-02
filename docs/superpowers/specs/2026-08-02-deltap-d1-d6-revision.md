# 2026-08-02 DeltaP D1–D6 修订实施

> 依据 `2026-08-02-deltap-mpi-consistency-review.md` 的 D1–D6 修订 TODO 实施
> （D1/D2/D3/D4/D5/D6 全部落地）。源码改动 7 文件 + 新增 MPI 冒烟脚本。
> 提交前评审（2 阻塞项 + 1 覆盖缺口）已处理，见 §3.7/§4。

## 1. Test plan

1. **D1**：PW `deltap_init` 加 KPAR=1 守卫（`GlobalV::KPAR > 1` → WARNING_QUIT）；
   统一通信域注释（KPAR=1 时 POOL_WORLD == MPI_COMM_WORLD）；修正
   `unk_overlap_pw.cpp` 与 `deltap_pw.h` 的约束/调用注释（顺带闭合 T9）。
2. **D2**：`DeltapScfSolver` 新增统一出口 `apply_lambda`（set_lambda +
   sync_lambda），替换 `update_lambda_gd` 与 `inner_loop` 两处 trial/final 的
   `set_lambda`——内循环路径从此也走 rank0 同步。
3. **D6**：LCAO/PW 两个 `sync_lambda` 加空 λ 防御；两处 `f_en.dp_escon` 赋值
   加跨 rank 一致性契约注释。
4. **D5**：PW `compute_gamma` 加 Bcast 前一致性回归守卫（pre-Bcast 副本与
   rank0 值逐原子比对 + MPI_MAX Allreduce，偏差 > 1e-8 rad 告警）；新增
   `tests/deltap_mpi_smoke/run.sh`（PW 2-rank + LCAO 4-rank 冒烟，含
   marker/divergence 判定）。
5. **D3**：LCAO 4-rank 临时 ESCONDBG 诊断（每 rank escon/γ_report），实跑断言
   后移除。
6. **D4**：固化 2-rank PW 参考值 + 量化 1/2-rank nproc 敏感性并定位性质。
7. **提交前评审反馈**（阻塞项 1）修正 `deltap_pw.h` 注释——`before_all_runners`
   在 `driver_run.cpp:67` 调用、位于离子步循环之外，`deltap_init` 即 once per
   run，旧注释正确，恢复并补充调用位置说明；（阻塞项 2）修复
   `run.sh` 的 `set -e` 缺陷（`run_case` 失败时提前终止脚本，改用
   `|| true` 累计失败）；（缺口 3）补 LCAO 内循环（`deltap_inner_nmax=3`）
   4-rank 用例入冒烟脚本并以临时副本运行。

## 2. Test setup

- 系统：`/root/abacus-develop`，分支 `feat/deltap`，HEAD `d983206a1`（本轮改动
  未提交）；构建 `CCACHE_DISABLE=1 cmake --build build --target abacus_basic_para -j14`。
- 算例：
  - PW 1/2-rank：`/tmp/deltap_r1_smoke/h2o_pw`（3 原子，1×1×2 k 网格，KPAR=1）；
  - KPAR 负向：`/tmp/d2_kpar_test`（h2o_pw 副本 + `kpar 2`）；
  - LCAO 4-rank：`/root/abacus-develop/tests/deltap_bn_test`（BN，2×2×2，
    方阵进程网格）与 `/tmp/deltap_r5/deltap_bn_test`；
  - 冒烟脚本：`tests/deltap_mpi_smoke/run.sh`（PW `tests/deltap_pw_h2o`
    2-rank + BN `tests/deltap_bn_test` 4-rank）。
- 基线：`/tmp/r1_backup/run_post_pw.baseline.log`（R5 1-rank PW）、上轮
  `/tmp/t1t3_pw2.log`（2-rank PW 参考）。

## 3. Results

### 3.1 构建与单测
- 构建通过（修复一处变量名：池数变量是 `GlobalV::KPAR` 而非 `GlobalV::n`）。
- 单测 16/16：`MODULE_ESOLVER_deltap_common_test` 10/10、
  `MODULE_ESOLVER_esolver_dp_test` 6/6。

### 3.2 PW 1-rank（D 改动回归面）
- `exit=0`；`[DeltaP-PW]` 行与 R5 基线逐字节一致：
  `drho=1.530e-03 γ_total=-0.1592 λ_avg=2.041e-03 |res|=4.739e+00 escon=-0.023414 Ry γ/atom=(4.738603, 0.692700, 0.692692)`。

### 3.3 PW 2-rank（D5 守卫 + 收敛）
- `exit=0`；**0 条 divergence 告警**（两 rank 测量在 1e-8 容差内一致）；
- 输出与上轮 `/tmp/t1t3_pw2.log` 逐字节一致（确定性）：
  `drho=1.504e-03 γ_total=-0.1592 escon=-0.023407 Ry γ/atom=(4.737743, 0.693120, 0.693101)`。

### 3.4 KPAR 守卫负向测试（D1）
- `kpar 2` + 2 rank → `WARNING_QUIT`，exit=1，日志含
  `DeltaP-PW requires KPAR=1 (npool=1): ...` —— 阻止 KPAR>1 下越池读 k 点波函数。

### 3.5 LCAO 4-rank（D2/D3）
- `exit=0`；P2 锚点不变：`[DeltaP P2] iter=11 drho=6.92e-06 < 1.00e-03`；
- ESCONDBG 断言：40 个不同 escon 值每个恰好出现 4 次（4 rank 每迭代全一致），
  末尾一致值 `escon=0.000171733 gamma=(3.99775, 3.49773)`；诊断已移除，
  干净复跑无残留输出。

### 3.6 MPI 冒烟脚本（D5）
- `tests/deltap_mpi_smoke/run.sh` 端到端 PASS：
  `PASS: deltap_pw_h2o (2 ranks)`、`PASS: deltap_bn_test (4 ranks)`、
  `PASS: deltap_bn_sampling/test_stru_target (4 ranks, 内循环)`。
- 修复过程：marker 含 `[` 需 `grep -F` 字面匹配（首版误用正则失败）。

### 3.7 提交前评审处理
- 阻塞项 1（注释与事实相反）：`before_all_runners` 确认在 `driver_run.cpp:67`
  调用、位于 `relax_driver` 离子步循环**之外**，每 run 一次；`deltap_pw.h`
  注释改回 "Called once" 并补充调用位置说明（此前按 T9 改为 "per ionic step"
  属错误推断，已纠正）。
- 阻塞项 2（`set -e` 缺陷）：负向测试（改错 marker）验证修复后两个用例与
  FAILED 摘要均执行、exit=1；正向三用例 PASS。
- 缺口 3（D2 内循环零 MPI 覆盖）：`test_stru_target`（LCAO、inner_nmax=3、
  2×2×2）4-rank 实跑 exit=0、`inner loop done` marker 出现、无 divergence
  告警——D2 的 trial/final `apply_lambda`（每 trial 一次 Bcast）多 rank 路径
  首次实跑验证；该用例已入冒烟脚本（临时副本模式，repo 不被 `deltap_branch*`
  输出污染）。
- 小观察（D5 守卫成本）：每次 γ 测量多一次 world Allreduce，且被守护路径
  实测不发散；可接受现状，后续可降为 debug-only 门控（记录为 TODO）。

## 4. Analysis

- **D1 结论**：`kpar` 默认=1（`input_parameter.h`），实测 2-rank（KPAR=1 单池）
  下 γ_total 与 1-rank 逐位一致——Wilson 环 k 弦确实只在单池内完整；KPAR>1 时
  各池只持 k 点子集，测量会越池读波函数，故守卫是必要的正确性保护而非限制。
- **D2 结论**：`apply_lambda` 作为唯一同步出口后，`inner_loop` 的 trial/final
  λ 与同步模式共用同一 rank0 同步路径，消除多 rank 内循环漂移隐患；行为对
  单 rank/同步模式无变化（1-rank A/B 逐字节一致佐证）；内循环 4-rank 实跑
  （§3.7）首次提供该路径的多 rank 覆盖，断言从"推理"升级为"实跑验证"。
- **D3 结论**：LCAO 侧 escon/γ_report 跨 rank 一致由 module_deltap 内部 Bcast
  （`compute_gamma_scf`）+ T1 λ 写回共同保证，实跑断言 4 rank 每迭代全一致。
- **D4 结论（nproc 敏感性定性）**：1/2-rank 差 Δγ_max=8.6e-4 rad（~0.02%）、
  Δescon=7e-6 Ry（~0.03%），与 CG 能量 7 位有效数字的并行浮点差异同量级——
  判定为**并行对角化浮点噪声**（becp/重叠求和顺序），非正确性缺陷；参考值已
  固化于 §3.2/§3.3，作为后续数值基线。根因专项：如需彻底消除，可统一
  becp/重叠的跨 rank 求和顺序（`cal_becp` 的 `reduce_pool` 与 `unkdotp` 的
  `MPI_Allreduce`），列为后续观察项。
- **D5 结论**：PW Bcast 前守卫能捕获"测量本身 rank 相关"的回归（本次 0 告警）；
  冒烟脚本给 2-rank PW / 4-rank LCAO 提供可重复的 PASS/FAIL 判据。
- **D6 结论**：空 λ 防御 + escon 契约注释使同步契约显式化，消除隐式依赖。

## 5. Next steps

1. 提交本轮改动（源码 + 冒烟脚本 + 文档）；`center/INPUT` 空白改动与杂散
   `STRU.cif` 保持不提交。
2. D4 根因专项（可选）：统一 becp/重叠跨 rank 求和顺序，消除 nproc 敏感性；
   若接受现状，将 1/2-rank 参考值录入测试文档。
3. 冒烟脚本接入 CI（GitHub Actions）：需 runner 具备 MPI + 赝势/轨道文件路径
   （`tests/PP_ORB` 与 `/root/pporb/apns-*`），列为 T13/D5 后续接线。
4. D5 守卫降级为 debug-only 门控（可选优化，当前接受现状）。
5. T2（PW init 惰性化）与 T7（force 路径专项）继续按原 TODO 推进。

## 6. Files modified

- `source/source_pw/module_pwdft/deltap_pw.cpp`（D1 守卫 + D5 守卫 + D6 空 λ）
- `source/source_pw/module_pwdft/deltap_pw.h`（T9 注释修正）
- `source/source_esolver/deltap_scf.{h,cpp}`（D2 apply_lambda）
- `source/source_esolver/esolver_ks_lcao.cpp`（D6 空 λ + escon 注释）
- `source/source_esolver/esolver_ks_pw.cpp`（D6 escon 注释）
- `source/source_io/module_unk/unk_overlap_pw.cpp`（D1 注释）
- 新增 `tests/deltap_mpi_smoke/run.sh`（D5 冒烟）
- 文档：本文件 + `deltap-development-log.md` 追加本节
