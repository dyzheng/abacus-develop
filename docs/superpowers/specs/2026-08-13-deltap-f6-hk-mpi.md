# 2026-08-13 — F-6（TODO 3.1）：hk_correction MPI 修复

## Test plan
1. **串行 A/B 逐字节硬约束**：h2o1 proxy 模式 λ=+0.01 冻结，修复前后
   FINAL_ETOT_IS / 全部 DeltaP 诊断行 / GE 能量列 / TOTAL-FORCE 逐位一致
   （仅 TIME 列允许变化）。
2. **hf/co corr=1 4-rank 与串行一致**：逐原子 γ（rawG 全精度）+ E'
   （FINAL_ETOT_IS）一致。
3. **h2o_asym 4-rank 仍被方阵守卫拦**（记录，不启用非方本地块）。
4. **deltap_mpi_smoke 回归**：PW 2-rank / BN 4-rank / CO(NBANDS=15) 4-rank /
   BN inner-loop 4-rank 全 PASS。

## 根因（两层，均串行专属语义）
- **带索引**：psi 列按 2D 块循环分布（本地带 `ncol_bands`），但原代码用全局
  占据带数 `nocc_use` 直接索引本地带 `c_R[γ + p·nrow]`——MPI 下部分和/越读。
- **行/列块语义**：H_sym 的（α,β）两个下标都用「本地行轨道」；H 本地块
  (nrow×ncol) 的列下标应是「本地列轨道」。`nrow==ncol` 守卫只保证缓冲区尺寸
  巧合相等，不保证索引语义。Γ^HK 的 T·Π 迹也只覆盖本地行。
- 2×2 网格下：nwfc 偶数（hf=18、co=10）→ 全 rank nrow==ncol，跑通但 Γ/E'
  错（hf E' 差 **2.33 eV**）；nwfc 奇数（h2o1/h2o_asym=529）→ 非对角 rank
  265×264 触发 WARNING_QUIT（gdb catch exit_group 实证，rank2）。
- 附带发现：`deltap_observable` 默认即 "operator"，hf/co 实际是 operator
  proxy 模式——Γ（compute_gamma_op_hk）进 escon/λ 更新，3.1 验收必须连带修
  gamma_op_hk（T7 同族，A' 方案直接适用）。

## 修复（`deltap_wannier.cpp`）
- **nproc==1**：原循环逐字节保留（硬约束，`if (mpi_path) ... else` 分发，
  非 MPI 构建下退化为原串行块）。
- **nproc>1（pzgemm 分布式 GEMM，'T'+预共轭约定，同 cal_dm_psi）**：
  - `SC = S_dk·C_R`（desc × desc_wfc → desc_wfc）；
  - `F = (i/2)·w_eff[g]·SC`，`g = local2global_col(n)`，`g ≥ nocc_use` 置零；
  - `H_sym = 0.5·(F·C_L† + C_L·F†)` —— 精确 Hermitian 本地块，直接线性写入
    hsk->get_hk()；
  - Γ^HK：`T = C_L†·SC`、`Π = C_L†·C_L` 的本地行×本地带部分和 → 均匀计数
    Allreduce → 逐原子收缩（A' 族，同 compute_D_I）。
- `compute_gamma_op_hk`：同族修复（SC 分布式 + T/Π Allreduce），Γ^HK 跨 rank
  一致。
- Ô_w（H_ow 分支与 Γ^w）在 MPI 下 WARNING 门控跳过（TODO 3.2/3.3，串行零
  回归——`mpi_path=false` 时行为不变）。
- **守卫保留**：`nrow != ncol` → WARNING_QUIT（消息更新）。剩余非方限制：
  compute_hk_force（既有 MPI 跳过，LIMITATION）与 Ô_w。

## Test setup
- `build/abacus_basic_para`，Release，ENABLE_MPI=ON，OpenMPI 4.1.6，
  OMP_NUM_THREADS=1（单任务 MPI，严禁并行多测试）。
- 串行 A/B：`/tmp/deltap_f6_ser*`（h2o1，proxy，λ=+0.01 冻结）。
- hf/co：`tests/deltap_fd_force/{hf,co}/base/`（默认 operator proxy，
  λ_init=0，λ_step=0.01，target.dat），串行 + `mpirun -np 4`。
- h2o_asym：`tests/deltap_fd_force/h2o_asym/base/`，`mpirun -np 4`。
- 回归：`tests/deltap_mpi_smoke/run.sh`。

## Results
| 项 | 串行参考 | 4-rank | Δ |
|---|---|---|---|
| h2o1 串行 A/B | −481.6964709588435 eV | −481.6964709588435 eV | 0（逐位） |
| hf E' | −687.094383147832 eV | −687.094380431602 eV | 2.7e-6 eV |
| co E' | −612.0598717620463 eV | −612.0598464601193 eV | 2.5e-5 eV |
| hf/co rawG 逐原子 γ | — | 与串行全精度（6 位）一致 | 0 |
| hf/co P3 行（γ/λ/Γ/escon 打印精度） | — | 与串行一致 | 0 |

- 串行 A/B：FINAL_ETOT_IS / 全部 DeltaP 行 / GE 能量列（md5）/ TOTAL-FORCE
  逐位一致 → **硬约束 PASS**。
- hf/co 4-rank：E' 差 2.7e-6 / 2.5e-5 eV，rawG 一致，P3 行一致 → **PASS**。
- h2o_asym 4-rank：rc=1，gdb 确认 WARNING_QUIT（compute_hk_correction
  nrow≠ncol 守卫）→ **仍被拦 ✓**。
- deltap_mpi_smoke：PW 2-rank / BN 4-rank / CO(NBANDS=15) 4-rank /
  BN inner-loop 4-rank **全 PASS**。

## Analysis
- **ΔE' 归因（诚实标注）**：pzgemm 块循环求和顺序 vs 串行稠密循环的固有 FP
  差异（~1e-14/元素），经 30+ 迭代 SCF 反馈与 λ 梯度累积放大至 ~1e-6~1e-5
  eV（co 的 λ[0] 第七位漂移 4.644467→4.644466 直接解释 escon Δ）。远低于
  一切 DeltaP 判据（fd_force 0.0129 eV/Å、T 系列 1e-3 eV）；串行 A/B 本身
  仍逐位一致。这不是可消除的偏差——除非在 MPI 下复刻串行求和顺序（违背分布
  式本意）。
- **范围裁定**：h2o1/h2o_asym（nwfc=529 奇数）4-rank 仍被守卫拦。F-6 交接
  摘要中"h2o1 4-rank E'/O1z 力/γ 一致"实为 3.2/3.3 前置（非方本地块启用 +
  F_HK 力 MPI），不在 TODO 3.1 验收内（TODO 原文 = hf/co γ+E' + h2o_asym
  拦）。F_HK 力保持既有 MPI 跳过（documented LIMITATION）。
- **与 A' 方案的关系**：TODO 3.1 原写"本地列→全局带映射，与 D_I A' 同族"。
  实测证明 A' 带映射对 Γ^HK 观测充分（全局标量），但对 H_sym 本地块不充分
  （行/列块语义 + 分布式 GEMM 是硬需求）——本批两条路径都落地：H_sym 走
  pzgemm，Γ^HK 走 A' Allreduce。

## Next steps
1. **3.2**：operator 模式 4-rank Γ/γ 跨 rank 一致（验收：逐原子 ±0.01 rad）——
   本批已修好 Γ^HK 观测；剩余 = ow Γ^w MPI + 非方本地块启用 + 跨 rank 一致性
   专项对拍。
2. **3.3**：co/h2o_asym corr=1 4-rank 收敛性复测（需先解 compute_hk_force MPI
   与非方本地块）。
3. **F-7**（L1 三件：PW Γ 记账/⟨η⟩/spread_I）→ **F-8**（应力，含压电
   Maxwell 判据）。
4. 用户文档：场模式力侧条目补 "MPI 下 F_HK 跳过 + hf/co 4-rank E' 一致性
   (≈1e-5 eV)" 标注。
