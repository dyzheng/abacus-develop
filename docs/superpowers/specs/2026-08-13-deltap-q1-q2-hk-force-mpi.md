# 2026-08-13 — Q1/Q2（TODO 3.2/3.3 前置）：F_HK 力 MPI + 非方本地块解锁 + F-6 带对修复

## Test plan
1. **Q1 守卫改版**：`nrow != ncol` 守卫只对 `nproc==1` 生效（MPI 路径
   分布无关：pzgemm 本地块尺寸由 numroc 决定，非方是自然态）；h2o_asym
   4-rank 解锁。
2. **Q2 F_HK 力 MPI**（`compute_hk_force`）：SC 走 pzgemm；T/Pi/U/dW 走
   行组 gather + 单次 Allreduce；串行路径逐字节保留（硬约束）。
3. **跨 rank 一致性（3.2）**：h2o1/h2o_asym 4-rank（2×2 网格、12×11 非方
   本地块）与 2-rank（1×2 网格）的 Γ/γ/λ/escon/F_HK vs 串行全精度一致。
4. **收敛性复测（3.3）**：co（NBANDS=15 奇数）4-rank 收敛 + F_HK vs 串行。
5. **deltap_mpi_smoke 回归**：4/4 PASS。

## 根因与修复
### Q1：非方本地块守卫收窄（`compute_hk_correction` / `compute_gamma_op_hk` / `compute_hk_force`）
- F-6 后守卫（nrow != ncol → WARNING_QUIT/return）是为串行路径的
  "nrow×nrow 硬写进 hk" 语义设的；MPI 路径已走 pzgemm（分布式 GEMM 天然
  支持非方本地块）。守卫改为只对 `nproc==1` 生效（串行 LCAO 恒有
  nrow==ncol==nlocal，纯防御）；MPI 下 h2o_asym 4-rank 直接解锁。
- 实测系统：h2o1/h2o_asym 为 **23 轨道**（O 2s2p1d=13 + H 2s1p×2=10，
  F-6 文档中"529 轨道/265×264"系笔误——修正：2×2 网格 nb=1 下本地块为
  12×12 / 12×11 / 11×12 / 11×11，非方 rank 即 F-6 gdb 实证的 rank 2）。

### Q2：`compute_hk_force` MPI 路径
- `SC = S_dk·C_R`（desc × desc_wfc，pzgemm）；
- `T/Pi = C_L†·SC / C_L†·C_L` 走**行组 gather + 全带对循环 + 单次 Allreduce**
  （一行组贡献一次，coord[1]!=0 清空，防重复计数）；
- `U_Jα = C_L†·(∂S_dk/∂R_Jα)·C_R`：D_Jα 本地块（nrow×ncol）→ pzgemm
  V=D·C_R → 行组 gather → 本地行部分和；`dW` 是普通 A' 部分和（全部 rank
  归约）；
- 归约后与原串行相同力度 kernel 累加（acc/e_hk/e_hk_I 逐项一致）。
- 串行块逐字节保留（`if (mpi_path) ... else` 大括号包裹）。

### 隐藏 bug（本轮最重要的修复）：`gather_band_columns` 的 Allgatherv 计数
- 新辅助函数用 `MPI_Allgatherv(..., MPI_DOUBLE)` 搬运 `complex<double>`，
  发送/接收计数按**复数元素数**填了 `nrow·ncol_b`，实际应为 `2·nrow·ncol_b`
  个 double——每个 rank 只发了一半数据，缓冲区前半段列间交叠、后半段全零。
- 症状：h2o_asym 4-rank Γ^HK 从 6.671 掉到 5.180（~22% 偏差）——奇数带
  （1,3 号占据带，在另一进程列）整列归零；h2o1 4-rank 恰好占据带全在进程
  列 0 的干净区，**未触发**（所以此前"h2o1 逐位一致"未暴露）。
- 修复：cnt2/dsp2 与发送计数全部 ×2（`2·nrow·counts[q]`）。修复后
  h2o_asym 4-rank F_HK 与串行**逐分量一致**（见 Results）。

## Test setup
- `build/abacus_basic_para`，MPI=ON，OpenMPI 4.1.6，OMP_NUM_THREADS=1
  （单任务 MPI，严禁并行多测试）。
- 串行 A/B：h2o1 proxy λ=+0.01 冻结（F-6 基线 `deltap_f6_ser`）。
- h2o1/h2o_asym：`/tmp/deltap_q1q2_*` 同款 INPUT/STRU/KPT/target.dat，
  `mpirun -np 4` 与 `-np 2`。
- co：`tests/deltap_mpi_smoke/deltap_co_lcao`（NBANDS=15），`mpirun -np 4`。
- 回归：`tests/deltap_mpi_smoke/run.sh`。

## Results
| 项 | 结果 |
|---|---|
| h2o1 串行 A/B（修复后 vs F-6 基线） | 5271 行中仅 4 行差（墙钟 + 单条 profile 计时），逐字节硬约束 **PASS** |
| h2o_asym 4-rank vs 串行 | λ=(6.671241,1.990799,1.860024)e-3、Γ=(6.660,1.987,1.857)、escon=−0.051842、E_HK=0.0237007424、F_HK 9 分量**逐位一致** |
| h2o_asym 2-rank（1×2 网格）vs 串行 | 同上，**逐位一致** |
| h2o1 4-rank vs 串行 | Γ=(6.944,1.884,1.884)、escon=−0.107117、F_HK 大分量逐位一致，~1e-12 湮没分量末位 FP 噪声 |
| co 4-rank（NBANDS=15 奇数）vs 串行 | 收敛（P2 标记 + P3 冻结），λ=(4.644466,6.623731)e-3、escon=−0.065207、E_HK=0.0289416363、F_HK 分量与串行一致 |
| deltap_mpi_smoke | PW 2-rank / BN 4-rank / CO 4-rank / BN inner-loop 4-rank **4/4 PASS** |

- 修复前对照：h2o_asym 4-rank Γ=5.180（错 22%）、λ 轨迹偏移
  （5.180e-3 vs 6.671e-3）；旧 A' 本地带循环（F-6 时代）虽缺跨列带对
  （T[0,1] 等 ~1e-3 小项）但对 Γ 打印精度不可见，仅 F_HK ~1e-5 差——
  正是 Q2 交接时定位到的"h2o_asym F_HK 差 ~1e-5"。

## Analysis
- **F-6 的 A' 带对诊断修正**：F-6 轮"跨列带对缺失"诊断方向正确（旧本地带
  循环确实缺跨列对），但 h2o_asym 的具体表现被两个效应叠加：① 缺失项
  （跨列 T[0,1] 等）量级 ~1e-3，只在 F_HK 留下 ~1e-5 痕迹；② gather 修复
  自身的 Allgatherv 计数 bug 把整列奇数带清零，造成 ~22% Γ 偏差。二者必须
  同时修（本轮的 ×2 计数 + 全带对 gather），单独任何一个都不够。
- **非方本地块的验证强度**：h2o1/h2o_asym 4-rank（12×11/11×12 非方）+
  2-rank（1×2，单行组双列）共同覆盖 gather 的两种结构（dim1=2 跨列、
  dim1=1 单列），F_HK 与串行逐位一致 → 3.2 跨 rank 一致性 PASS。
- **λ 末位漂移**：h2o_asym 4-rank λ[0]=6.671241e-3 vs 串行 6.671237e-3
  （第七位）——pzgemm 求和顺序 FP 噪声经 λ 反馈的已知放大（F-6 同族），
  远低于判据。
- **串行路径零扰动**：A/B 硬约束通过，F_HK 存储路径（TOTAL-FORCE）未扰动。

## 裁定
Q1（守卫收窄）+ Q2（F_HK 力 MPI）+ 3.2（跨 rank 一致性）+ 3.3（co/h2o_asym
收敛复测）全部 PASS。F_HK 力从"L2 遗留"升级完成——场模式力
（F_std + B(F_HK)）在多 rank 下可用。遗留：Ô_w（H_ow/Γ^w）MPI 门控跳过
（L3.1 研究轨道，非本阶段目标）。

## File list
- `source/source_lcao/module_deltap/deltap_wannier.cpp`（gather_band_columns
  辅助 + 计数修复；compute_hk_correction / compute_gamma_op_hk /
  compute_hk_force 的行组 gather + 全带对 T/Pi/U；守卫收窄到 nproc==1）
- `source/source_lcao/module_deltap/deltap.h`（compute_hk_force 注释更新）
- `docs/superpowers/specs/2026-08-13-deltap-q1-q2-hk-force-mpi.md`（本轮）
- `docs/superpowers/specs/2026-08-04-deltap-execution-todo.md`（3.2/3.3 → ✅）
- dev log 本条 + 关键结论区 #19

## Next steps
1. commit（消息含"F_HK 力 MPI + 非方本地块解锁 + F-6 带对计数修复"）。
2. F-7（L1 三件：PW Γ 记账 / ⟨η⟩ / spread_I）→ F-8（应力，含应力-极化
   Maxwell 判据 ∂σ/∂λ ↔ ∂P/∂ε）。
3. Ô_w MPI（H_ow 本地块 + Γ^w 行组 gather）待 L3.1 立项时做。
