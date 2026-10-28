# PW allocation per-k target/product cache

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_pw/module_pwdft/op_pw_alloc.cpp`、
  `docs/superpowers/specs/deltap-development-log.md`
- 前置文档：
  `docs/superpowers/specs/2026-10-28-allocation-veff-potential-fold.md`

## Test plan

1. 验证切换 k 点不会反复重建 allocation 目标空间和 `V|alpha>` 乘积。
2. 在 Fe2-bcc folded-potential 路径上比较 S5.9 单槽缓存与本轮 per-k 缓存：
   电子迭代数、hPsi 次数/时间、FFT 次数和墙钟。
3. 确认最终磁矩、约束残差和总能不变；运行完整 constraint 单测。

## Test setup

- 4 MPI rank；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- Fe2-bcc：`2x2x2` k、`kpar=2`、`nspin=2`、`nbands=27`、
  `ecutwfc=20 Ry`、`scf_thr=1e-8`、目标 `-3.8/+3.8 μB`。
- 输入与 S5.9 A/B 的 fold-on 组相同：
  `constraint_alloc_fold_potential=1`、
  `constraint_alloc_subspace_rotation=1`。
- per-k 目标缓存不设上限；`V|alpha>` 乘积缓存最多保留 8 个 k 槽，避免大 k
  集时内存随 `nks * ntarget * npw` 无界增长。

## Results

### Fe2 性能

| 版本 | 迭代 | hPsi 次数 | hPsi 总时间/s | recip2real 次数 | real2recip 次数 | 墙钟/s | ETOT/eV |
|---|---:|---:|---:|---:|---:|---:|---:|
| S5.9 fold on 单槽缓存 | 82 / 82 | 1985 / 1985 | 18.36–18.70 | 46730 | 32464 | 28.84–29.73 | -6370.6101511999 |
| 本轮 per-k 缓存 | 82 / 82 | 1985 / 1985 | 15.34–16.12 | 44210 | 29944 | 24.22–25.56 | -6370.6101511999 |

- `recip2real` 与 `real2recip` 各减少 2520 次；目标乘积每次重建需要 10 个
  target 的 FFT 对，因此约消除 252 次重建。
- hPsi 总时间下降约 12% 至 16%，墙钟下降约 14% 至 18%。
- 最终 `q=-3.800014112/+3.799903104 μB`，最大残差 `9.689644075e-5 μB`，
  与 S5.9 fold-on 组逐位一致。
- 第二组墙钟更低主要来自机器负载差异；FFT 调用数完全一致，说明缓存命中
  行为确定。

### 回归

- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS。
- `git diff --check`：PASS。

## Analysis

- 单槽缓存的键同时包含 `ik`、spin、target generation 和 lambda。Davidson
  在 k 点之间切换时，即使 lambda 未变，也会把上一个 k 的乘积逐出；这是
  S5.9 之后仍可消除的纯重复工作。
- per-k 目标缓存让 `tabulate_atomic/overlap_proj_psi` 每个 k 只执行一次；
  per-k 乘积缓存让每个 `(ik, spin, lambda)` 的 `V|alpha>` FFT 只执行一次。
- 缓存上限 8 是内存与命中率的折中：常见小 k 集全部保留，大 k 集退化为
  有界窗口而不会增加无上限内存。
- 本轮不改变外层迭代数（仍为 82），因此它只解决常数项；下一阶段的收益
  必须来自固定密度内的约束子空间求解。

## Next steps

1. 构造 \(\langle\psi_m|A_i|\psi_n\rangle\)，并缓存 `H_sub`、`S_sub` 与
   `A_i,sub`，在固定密度内反复更新 lambda 而不再重建 Hamiltonian。
2. 用子空间矩阵直接计算每个 trial lambda 的约束读数，替代逐 band FFT 的
   `compute_pw_alloc_observables`。
3. 将新的子空间步与当前 per-k 缓存做 A/B，目标是显著降低 Fe2 的 82 次电子
   迭代，而不只是降低单次 hPsi 时间。
