# PW allocation 精确固定密度子空间

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 新增开关：`constraint_alloc_exact_subspace`（默认 false，实验性）
- 修改文件：`source/source_esolver/esolver_ks_pw.cpp`、
  `source/source_hsolver/diago_iter_assist.{h,cpp}`、
  `source/source_io/module_parameter/{input_parameter.h,read_input_item_other.cpp}`、
  `source/source_pw/module_pwdft/op_pw_alloc.{h,cpp}`、
  `docs/superpowers/specs/deltap-development-log.md`
- 前置文档：
  `docs/superpowers/specs/2026-10-28-allocation-subspace-warm-start.md`、
  `docs/superpowers/specs/2026-10-28-allocation-per-k-cache.md`

## Test plan

1. 在当前占据带子空间内构造
   \(A_i=Q_i+Q_c w_i Q_c\)，验证
   \(\langle\psi_m|A_i|\psi_n\rangle\) 与全空间 allocation 算符读数一致。
2. 验证 `diag_subspace_psi()` 返回的旋转矩阵方向，确保子空间预测的
   \(Q(\mu)\) 与旋转后的全空间波函数读数一致。
3. 在 Fe2-bcc 上比较：生产单次旋转基线、精确子空间 `inner=1`、
   `inner=8`、以及关闭 trust region 的 `inner=8`。
4. 跑 constraint 单测与 LiCl `mu=0` 回归，确认默认关闭时无行为变化。

## Test setup

- 4 MPI rank；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- Fe2-bcc：`2x2x2` k、`kpar=2`、`nspin=2`、`nbands=27`、
  `ecutwfc=20 Ry`、`scf_thr=1e-8`、目标 `-3.8/+3.8 μB`。
- LiCl：`gamma_only=1`、`nspin=1`、`nbands=8`、`ecutwfc=100 Ry`、
  电荷目标 `delta`、`mu=0`。
- Fe2 公共输入：`constraint_alloc_switch=1`、
  `constraint_alloc_target_l=2`、`constraint_alloc_fold_potential=1`、
  `constraint_alloc_subspace_rotation=1`、`constraint_mu_schedule=outer`、
  trust region 开启（除注明外）。

## Results

### 子空间代数与旋转检查

- 参考波函数处 `q_check` 为 `3.55e-15` 至 `1.20e-14`，说明
  \(A_i\) 与全空间 allocation 可观测量一致。
- 旋转后的子空间读数与全空间读数差为 `3.55e-15` 至 `1.51e-14`。
- 早期四方向诊断中只有列主序方向
  \(u_{m b}\) 的残差约 `2e-16`；错误方向约 `1.0`。因此
  `diag_subspace_psi()` 的 `rotation_out` 必须按列主序读取。该诊断已删除。

### Fe2 SCF

| 配置 | 电子迭代 | hPsi 次数 | hPsi 时间/s | 墙钟/s | 最终/停止点 q（μB） | ETOT/eV | 结论 |
|---|---:|---:|---:|---:|---|---|---|
| 生产基线，`exact=0` | 82 | 1984 | 9.64 | 15 | `-3.800014066/+3.799903209` | `-6370.610151200` | 正确，残差 `9.68e-5 μB` |
| 精确子空间，`inner=1` | 116（异常终止） | — | — | — | `-3.798530/+3.793894` | 无正常 FINAL_ETOT | outer 16 后 trust 半径 `1.53e-7`、20 次拒绝，rank exit 1 |
| 精确子空间，`inner=8`，trust on | 200 | 9381 | 49.31 | 75 | 停止在错误磁支 | `-6383.165871` | trust 半径缩到 `1e-8`，38 次拒绝 |
| 精确子空间，`inner=8`，trust off | 200 | 9145 | 41.05 | 63 | 停止在错误磁支 | `-6370.977116` | 固定密度 mu 直接过冲，不收敛 |

`inner=1` 的 116 次迭代耗时和 hPsi 次数没有完整 timer，因为其中一个
MPI rank 以非零码退出；这本身也说明该实验路径尚不具备生产健壮性。

### 回归

- `ctest --test-dir build -R MODULE_ESTATE_constraint --output-on-failure`：
  23/23 PASS（112.62 s）。
- LiCl `mu=0`，默认关闭精确子空间：19 次电子迭代，
  `q=2.947887372 e`、`ETOT=-643.97565238003517 eV`、
  `dsum_n=-2.18847163e-12`，与 S5.7/S5.8 逐位一致。
- `git diff --check`：PASS。

## Analysis

1. **低层严格性成立。** \(A_i\) 矩阵、广义本征问题、旋转后的子空间读数与
   全空间算符读数都在 `1e-14` 内一致；`kpar=2` 的跨 pool 归约也一致。
2. **固定密度 \(Q(\mu)\) 不能代表自洽 \(Q(\mu)\)。** 外层 step 3 的固定密度
   预测已接近目标：`m=(-3.7849, 3.79779) μB`；但 stage-2 全空间求解并更新
   密度后，下一个 outer 观测为 `(-3.92994, 3.84063) μB`。偏磁通道变化约
   `0.145 μB`，远大于 `constraint_thr=1e-4`。固定密度子空间缺少
   \(\rho(\mu)\) 的密度响应，不能作为外层收敛判据。
3. **trust 模型被两种观测混合污染。** 内层用固定密度 \(Q\) 更新 Jacobian，
   外层用密度响应后的全空间 \(Q\) 评分同一个 trial；预测下降量与真实下降量
   不一致，导致半径连续 shrink。增加 `inner` 次数只会放大这种不一致。
4. **关闭 trust 不是解。** 它消除了拒绝机制，却也失去对 Fe2 磁支跳变的保护；
   `inner=8` 在 200 步内落到错误分支，ETOT 与基线差约 `0.37 eV`。
5. **与 native DeltaSpin 的差别不是“少做一次子空间对角化”。** DeltaSpin
   的 stage 1 只负责给出好的初值，stage 2 仍在全空间求解，并在更新后的
   波函数上重新观测磁矩；它没有把固定密度投影值冒充最终可观测量。当前
   精确路径把固定密度值直接送入外层 mu 求解器，这是失败根因。

## Conclusion And Next Steps

- S6.2 的代数子问题已实现并验证，但**不能用固定密度子空间读数压缩外层
  SCF**。因此 `constraint_alloc_exact_subspace` 仅保留为默认关闭的实验诊断
  开关，不设为 PW allocation 默认值，也不进入生产推荐参数。
- 生产最优路径仍是 `constraint_alloc_subspace_rotation=1`、
  `constraint_alloc_subspace_iterations=1` 的单次 stage-1 warm start；
  Fe2 为 82 次迭代、1984 次 hPsi，结果与基线一致。
- 若继续追求 DeltaSpin 量级效率，下一步必须是真正的内层联合更新：
  在每个 lambda trial 后执行 stage-2 全空间求解，用**全空间** \(Q\) 更新 mu，
  并在同一 SCF 调用内重复，而不是用固定密度 \(Q\) 提前宣布约束收敛。
- 另一条可行路线是构造 \(\partial Q/\partial\mu\) 的密度响应近似
  （Broyden/线性响应预条件），再与 \(A_i\) 子空间联立；否则子空间只能作为
  warm start 或 preconditioner，不能作为外层 fixed-point 的替代。
