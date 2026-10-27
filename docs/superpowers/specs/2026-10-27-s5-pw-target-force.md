# S5.1 PW allocation 完整目标投影几何导数

日期：2026-10-27

计划：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`，Task S5.1。

本轮补上 PW allocation 力中上一轮缺失的 `∂Q̂/∂R`：目标 projector 随原子移动、
projector overlap 的 `ST^{-1/2}` Fréchet 导数，以及正交目标向量的转动项。
冻结近似 `Q_c dW Q_c` 仍由共享网格核计算，新项在其后相加。

## 1. 测试计划

1. 用有限差分验证 rank-truncated `S^{-1/2}` 的解析导数。
2. 用有限差分验证目标向量 `φ_orth = φG` 及其导数的分解。
3. 用独立的算符导数分解测试核对 `dQ`、owner 归属与交叉项，排除漏项或共轭错误。
4. 在 `OMP_NUM_THREADS=1` 下比较 `np=2` 与 `np=4` 的中心构型能量、约束力、
   总力和观测量。
5. 对 H2O/O-d 的 O-z 方向做三点总能 FD，以总力而非单独 `e_con` 为验收门。

## 2. 测试设置

- 分支：`feat/unified-local-moment-allocation`
- 构建：`build/abacus_basic_para`，CPU，OpenMPI
- 体系：H2O，O-d 目标（`constraint_alloc_target_l=2`），Becke 权重
- `nspin=2`、`nbands=8`、`ecutwfc=20 Ry`、`ecutrho=80 Ry`
- `constraint_alloc_switch=1`、`fixed_mu=0.01 Ry`、`scf_thr=1e-8`
- O-z 位移：中心、`+0.005 Bohr`、`-0.005 Bohr`
- PW 验收只使用 MPI 并行；OpenMP、OpenBLAS、MKL 均固定为单线程：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

## 3. 实现

1. `alloc_linalg.{h,cpp}` 新增 `sym_inv_sqrt_derivative()`，对 rank-truncated
   `S^{-1/2}` 使用谱分解的 Fréchet 导数；被截断方向按同一阈值贡献零。
2. `alloc_pw.{h,cpp}` 新增 `build_pw_target_derivative()` 与
   `orthogonalize_pw_overlaps_derivative()`。对
   `raw_a=<alpha_a|psi>`、`d_raw_a=<d alpha_a|psi>`：

   ```text
   orth_a   = sum_b G_ab raw_b
   d_orth_a = sum_b (dG_ab raw_b + G_ab d_raw_b)
   ```

3. `onsite_proj_tools.{h,cpp}` 新增 `cal_dbecp_f_custom()`，用现有 VKB 几何导数
   一次计算 projector 行对任意 band 向量的 `<d alpha|vec>`。
4. `op_pw_alloc.{h,cpp}` 新增 `compute_pw_alloc_target_force()`。核心分解为：

   ```text
   dP psi      = sum_a [ orth_a dphi_a + d_orth_a phi_a ]
   dQ_owner    = 2 Re sum_{a in owner} conj(d_orth_a) orth_a
   cross       = sum_a [ conj(orth_a) d_orth_wq_a
                       + conj(d_orth_a) orth_wq_a ]
   dK_owner    = dQ_owner - 2 Re(cross)
   ```

   其中 `dP` 必须包含所有正交目标的转动，`dQ_owner` 必须汇总 owner 的全部目标，
   且交叉项第二项使用 `conj(d_orth)`。
5. `constraint_loop.{h,cpp}` 新增 `set_allocation_target_force()`；在冻结网格力
   `Q_c dW Q_c` 之后叠加完整目标导数项。
6. `esolver_ks_pw.cpp` 在 `Forces::cal_force()` 前同时构造 `rho_c` 和目标力。

## 4. 结果

### 4.1 单元测试

```text
ctest -R 'MODULE_ESTATE_constraint_alloc_(linalg|pw)$'   PASS (2/2)
AllocLinalgTest.InvSqrtDerivativeMatchesFiniteDifference PASS
AllocPwTest.TargetDerivativeMatchesFiniteDifference      PASS, max_diff=6.994e-10
AllocPwTest.AllocationOperatorDerivativeDecomposition    PASS
  alpha=0: analytic=-6.079803e+00  fd=-6.079803e+00  diff=5.971e-10
  alpha=1: analytic= 4.731688e-01  fd= 4.731688e-01  diff=-3.463e-09
```

### 4.2 `np=2` 与 `np=4` 中心构型

| 量 | `np=2` | `np=4` |
|---|---:|---:|
| total energy (eV) | -442.15878837 | -442.15878837 |
| `e_con` (Ry) | -0.005100062102 | -0.005100062102 |
| constraint force O-z (Ry/Bohr) | -0.0003401300 | -0.0003401303 |
| total force O-z (eV/A) | -5.6690414398 | -5.6690414444 |

约束力项约为 `-0.0087451 eV/A`，不再是冻结近似的 `-1.24e-4 eV/A`。

日志：

- `tests/constraint_s5_parity/pw_alloc_tgtf_final_center_np2/run_center_np2b.log`
- `tests/constraint_s5_parity/pw_alloc_tgtf_final_center_np4/run_center_np4b.log`

### 4.3 三点总力 FD

```text
E+      = -442.14366107 eV
E0      = -442.15878837 eV
E-      = -442.17365438 eV
delta   = 0.005 Bohr

F_FD        = -0.22044669 Ry/Bohr = -5.667912284 eV/A
F_analytic  = -5.6690414444 eV/A
|difference| = 0.00112916 eV/A
criterion    = 0.0129 eV/A                                  PASS
```

完整目标导数的误差为 `0.00113 eV/A`，比冻结近似的 `0.00462 eV/A` 降低约四倍。

## 5. 问题与修复

1. 算符分解测试最初发现 `dP` 只转动了本原子目标、`dQ` 只统计了位移原子上的目标，
   且交叉项符号/共轭不一致。修正为：所有目标参与 `dP` 旋转，每个约束汇总其
   owner 的全部目标，交叉项使用 `conj(d_orth)`。
2. 测试中的 `W` 原先把两个 6×6 约束块误当成一个 12×12 矩阵；改为显式块结构。
3. `cal_dbecp_f_custom()` 的 GEMM 输出按平面波分片局部存储。`np=4` 初测目标力
   错误，根因是缺少 pool 归约；在三个方向求和前对 `derivative_overlap` 执行
   `reduce_pool()` 后，2/4 rank 恢复一致。

## 6. 分析：为何不用 `e_con` 直接差分作门

`e_con = mu (Q-target)` 不是完整的约束总能量，且在电子自由度上不满足
Hellmann–Feynman 驻点条件。直接对 `e_con` 做几何差分只给出该标量项的偏导数，
不能作为约束力算符的独立参考。上一轮冻结力文档中的 `e_con` 直接差分结论由本轮
替代：验收应使用包含全部能量项的**总力 FD**。本轮总力误差
`0.00113 eV/A` 已通过闸门。

## 7. 下一步

1. 对互补密度 `rho_c` 做 all-pool 归约，解除 `KPAR * bndpar > 1` 的力路径守卫。
2. 用 `kpar=2`、`np=4` 验证跨 k-pool 的能量、约束力和总力。
3. 将 `PwTargetSpace::ST` 扩展为复数 Hermitian，并补非 Gamma k 点测试。
4. 补 PW allocation 应力与完整 `kpar=2` 收敛回归。
