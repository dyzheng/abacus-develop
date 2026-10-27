# S5.1 PW allocation 力的跨 k-pool 互补密度

日期：2026-10-27

计划：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`，Task S5.1。

本轮解除 PW allocation 力路径的单 k-pool 限制：互补密度 `rho_c` 是实空间分片，
在各 k-pool 对相同本地网格片求和后，再交给共享 `constraint_force()` 做 pool 内积分。

## 1. 测试计划

1. 构建 `abacus_basic_para` 并复跑 allocation 单测。
2. 在相同 `1×1×2` k 网格、相同结构和约束目标下比较 `kpar=1/2`、均为 `np=4`。
3. 比较总能、`e_con`、约束力、总力及 `sum_n/sum_m` 审计。
4. 全程 `OMP_NUM_THREADS=1`，只用 MPI 进程并行。

## 2. 测试设置

- 体系：H2O，O-d 目标，Becke 权重，`constraint_alloc_target_l=2`
- `nspin=2`、`nbands=8`、`ecutwfc=20 Ry`、`ecutrho=80 Ry`
- `fixed_mu=0.01 Ry`、`scf_thr=1e-8`、`scf_nmax=100`
- k 网格：`1×1×2`，两个 k 点分属两个 k-pool
- MPI：`np=4`；OpenMP/OpenBLAS/MKL 单线程

用例目录：

- `tests/constraint_s5_parity/pw_alloc_kpar1_force_np4/`
- `tests/constraint_s5_parity/pw_alloc_kpar2_force_np4/`

## 3. 实现

`compute_pw_alloc_complement_density()` 在累计完所有本地 k 点后，对每个自旋通道
在 `KP_WORLD` 上执行一次 `MPI_Allreduce(MPI_SUM)`。`KP_WORLD` 只连接 k 点不同、
平面波/能带/实空间分片完全相同的进程，因此它恰好汇总各 k-pool 对同一网格片的
贡献，不能使用会除以 pool 大小的 `reduce_double_allpool()`。

`ESolver_KS_PW::cal_force()` 的守卫从 `KPAR * bndpar > 1` 收窄为
`bndpar > 1`；能带并行的数据归属尚未验证，继续显式拒绝。

## 4. 结果

### 4.1 构建与单测

```text
cmake --build build --target abacus_basic_para MODULE_ESTATE_constraint_alloc_pw \
                                          MODULE_ESTATE_constraint_alloc_linalg -j 8  PASS
ctest -R 'MODULE_ESTATE_constraint_alloc_(linalg|pw)$'                              PASS (2/2)
```

### 4.2 `kpar=1/2`、`np=4` 对照

| 量 | `kpar=1` | `kpar=2` | 差 |
|---|---:|---:|---:|
| total energy (eV) | -442.114230442330 | -442.114230442402 | 7.128e-11 |
| `e_con` (Ry) | -0.005100037926 | -0.005100037926 | 0 |
| `q` | -0.01000379263 | -0.01000379264 | 1e-14 |
| constraint force O-z (Ry/Bohr) | -0.0005615148 | -0.0005615160 | 1.2e-9 |
| total force O-z (eV/A) | -5.6821171917 | -5.6821172101 | 1.84e-8 |

两边 `sum_n=8.00000000`；`dsum_n` 分别为 `-5.17e-13/-1.06e-12`，
`dsum_m` 分别为 `1.48e-16/9.05e-16`。

日志：

- `tests/constraint_s5_parity/pw_alloc_kpar1_force_np4/run_kpar1_np4.log`
- `tests/constraint_s5_parity/pw_alloc_kpar2_force_np4/run_kpar2_np4.log`

## 5. 分析

1. 跨 pool 能量、约束力和总力均落在归约噪声量级，说明 `KP_WORLD` 网格片匹配正确。
2. 初版计划的 `reduce_double_allpool(npool, NPROC_IN_POOL, rho, nrxx)` 不适用于
   `rho_c`：各 rank 持有不同的 z 方向实空间片。该 helper 会先把不同网格片的贡献
   相加再除以 pool 大小，结果没有物理意义。
3. `KP_WORLD` 的对应 rank 拥有相同实空间片，故直接 `MPI_SUM` 即得到所有 k-pool
   的全局互补密度；随后 `constraint_force()` 的 `POOL_WORLD` 归约完成平面波并行。
4. 不整除的 k-pool 划分没有 `KP_WORLD`，函数显式报错而不是返回错误密度。

## 6. 下一步

1. 将 `PwTargetSpace::ST` 扩展为复数 Hermitian，并补非 Gamma k 点测试。
2. 补 PW allocation 应力。
3. 对 `bndpar > 1` 验证能带归属后，再决定是否解除剩余守卫。
4. 做完整 `kpar=2` 收敛回归与更密 k 网格对照。
