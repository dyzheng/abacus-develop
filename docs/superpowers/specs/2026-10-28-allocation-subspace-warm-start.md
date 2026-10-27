# PW allocation 外层策略 A/B 与固定密度子空间 warm start

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_esolver/esolver_ks_pw.cpp`、
  `source/source_estate/module_constraint/constraint_loop.{h,cpp}`、
  `source/source_io/module_parameter/input_parameter.h`、
  `source/source_io/module_parameter/read_input_item_other.cpp`、
  `docs/superpowers/specs/deltap-development-log.md`
- 前置文档：
  `docs/superpowers/specs/2026-10-28-allocation-fft-buffer-trim.md`、
  `docs/superpowers/specs/2026-10-27-allocation-critical-review.md`

## Test plan

1. 固定 Fe2-bcc 输入，只改外层 mu 策略，判断慢的主因是步长/信任域还是
   更深的固定点结构：
   - 关闭 trust region（`constraint_trust_region=0`）；
   - `constraint_step_max=0.01`、`0.005`；
   - 只改 `constraint_step_probe=0.001`；
   - `constraint_mu_schedule=inner`，`inner_thr=0.1/0.5`。
2. 实现一个可开关的 PW allocation 快路径：每次 mu 改变后，在固定的
   \(n_{\rm bands}\times n_{\rm bands}\) 子空间内对当前 Hamiltonian 做一次
   固定密度对角化，旋转 psi、更新本征值/占据数/密度，然后跳过该次 HSolver。
3. 对 Fe2 比较 S5.7 基线、单次旋转、2/4 次固定密度 mu 更新；记录电子迭代、
   hPsi 次数/总时间、墙钟、最终 q/残差/总能。
4. 用 LiCl 的 `mu=0` 路径确认快路径不改变只读结果；跑完整 constraint 单测。

## Test setup

- 4 MPI rank；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- Fe2-bcc：`2x2x2` k、`kpar=2`、`nspin=2`、`nbands=27`、
  `ecutwfc=20 Ry`、`scf_thr=1e-8`、目标 `-3.8/+3.8 μB`。
- LiCl：Γ 点、`nspin=1`、`nbands=8`、192³ FFT、电荷约束、`mu=0`。
- 关键输入：`constraint_alloc_switch=1`、`constraint_alloc_target_l=2`、
  `constraint_trust_region=1`、`constraint_step_max=0.02`（除注明外）。
- 新开关：
  `constraint_alloc_subspace_rotation=1`、
  `constraint_alloc_subspace_iterations=1`（默认）。

## Results

### 外层策略 A/B（Fe2，均为 4 rank）

| 策略 | 迭代 | hPsi 次数 | 最终最大残差/μB | 结论 |
|---|---:|---:|---:|---|
| S5.7 基线（trust，step_max=0.02） | 91 | 3342 | 4.98e-5 | 正确 |
| trust=0 | 200 | 4941 | 1.25e-2 | 未收敛 |
| step_max=0.01 | 87 | 2105 | 8.10e-5 | 正确，hPsi 少 37% |
| step_max=0.005 | 200 | 5095 | 8.26e-2 | 错误磁支 |
| step_probe=0.001 | 91 | 3342 | 4.98e-5 | 与基线逐位相同 |
| inner，thr=0.1 | 162（人为终止） | — | 1.8e-1 | 失稳 |
| inner，thr=0.5 | 142（人为终止） | — | 3.4e-1 | 失稳 |

`step_max=0.01` 是唯一正收益的参数点：Fe2 在第 2 个 outer step 避免了
`q[1]≈2.66 μB` 的磁支跳变，最终 `q=-3.79997/+3.80008 μB`、总能
`-6370.610150890 eV`。但 `0.005` 同样输入却落到错误磁支，说明该收益对
分支边界非常敏感，不能作为稳健默认值。

### 固定密度子空间旋转（Fe2）

| 版本 | 迭代 | hPsi 次数 | hPsi 总时间/s | 墙钟/s | 最终 q[0]/q[1]（μB） | ETOT/eV |
|---|---:|---:|---:|---:|---|---|
| S5.7 基线 | 91 | 3342 | 42.47 | 56.45 | -3.799950 / +3.799957 | -6370.610151222 |
| 单次旋转 | 82 | 1983 | 28.99–31.29 | 39–42 | -3.800014 / +3.799903 | -6370.610151197 |
| 2 次固定密度更新 | 72 后失稳 | — | — | — | 出现 `q[1]≈2.65` 跳变 | — |
| 4 次固定密度更新 | 55 后失稳 | — | — | — | trust 半径缩到 1e-5 | — |

单次旋转相对基线：迭代 `91→82`、hPsi `3342→1983`（-41%）、墙钟
`56.45→39–42 s`（约 -26%），总能差 `2.5e-8 eV`，最终残差 `9.73e-5 μB`
仍在 `constraint_thr=1e-4` 内。

### 回归

- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS（108.73 s）。
- allocation operator/observe：2/2 PASS。
- LiCl `mu=0`：19 次迭代、809 次 hPsi、hPsi 总时间 98.92 s、墙钟 169 s，
  `q=2.947887372 e`、`ETOT=-643.97565238003517 eV`、`dsum_n=-2.188e-12`；
  快路径未被触发，输出与 S5.7 一致（墙钟差异来自机器负载）。

## Analysis

- **外层参数不是根因。** 关闭 trust、缩小步长或放宽 inner 门限都会在 Fe2 的
  `Q(mu)` 分支边界附近失败；`step_max=0.01` 的收益也无法在 `0.005` 上复现。
  这说明慢的根源是“每次 mu 改变都重新收敛整个电子密度”，而不是某个步长常数。
- **单次子空间旋转是有效且低风险的。** 它把一次完整 HSolver 替换为一次
  `nbands×nbands` 子空间对角化；旋转后的 psi 直接吸收新 mu，密度在本迭代
  内重建。Fe2 的 hPsi 次数下降 41%，且最终态与基线在 `2.5e-8 eV` 内一致。
- **固定密度 mu 更新超过 1 次会过冲。** 第 2 次更新把 mu 推到密度尚未响应的
  分支，随后密度松弛时触发磁支翻转；trust 半径被反复 shrink。当前实现把
  `constraint_alloc_subspace_iterations` 默认设为 1，>1 仅作为实验开关，
  不应在生产输入中启用。
- **剩余瓶颈仍是外层迭代数与每次 hPsi 的额外 `V|psi>` FFT。** 单次旋转把
  Fe2 从 91 降到 82 次，但 DeltaSpin 仍是 13 次；要接近它，需要在固定密度
  子空间里直接构造每个约束的 \(A_i\) 矩阵并做受约束最小化，而不是反复调用
  完整的 allocation 算符和观测路径。

## Next steps

1. 预计算 \(A_i\) 在当前占据子空间中的矩阵元
   \(\langle\psi_m|A_i|\psi_n\rangle\)：每个 band 一次正 FFT 得到
   \(Q_c\psi_m\)，再用 `w_alpha` 做实空间积分；避免 inner 循环里的
   `compute_pw_alloc_observables` 重复 FFT。
2. 用该矩阵实现真正的固定密度受约束子空间求解（DeltaSpin
   `cal_hs_subspace`/`diag_responce` 的 allocation 版本），并只在残差足够小
   时把 mu 交给外层。
3. 在 force/stress 路径验证子空间旋转后的密度与约束力一致；本轮只验证了
   SCF 能量和约束读数。
4. 在 `nbands=100+` 体系复测旋转的内存峰值和 hPsi 收益，决定是否把
   `constraint_alloc_subspace_rotation` 设为 PW allocation 的默认值。
