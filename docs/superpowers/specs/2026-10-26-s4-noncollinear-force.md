# S4.3 非共线分配约束位置力：测试报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S4.3。

## 1. 测试计划

本轮补齐 LCAO `nspin=4` 分配约束的位置力，验证：

1. 固定 `lambda`、固定目标空间下的能量导数同时包含重叠矩阵 `dS`、
   权重网格 `dw` 和基函数（Pulay）三部分。
2. `dS` 项的符号与 ABACUS 总力口径一致。
3. 非共线 W 权重矩阵的基函数导数通过 Gint 局域势力核计算，并在 MPI
   pool 上正确归约。
4. 三点位置 FD 与解析 O-z 力之差小于 S4 验收阈值
   `<0.0129 eV/A`，且本轮改动不改变总能。

## 2. 测试设置

- 系统：H2O 分子，LCAO，8 电子，23 个标量 AO。
- 参数：`nspin=4`、`noncolin=1`、单 Γ 点、`ecutwfc=20 Ry`、
  `ecutrho=80 Ry`、`nbands=8`、`scf_thr=1e-8`、`scf_nmax=100`、
  Broyden `mixing_beta=0.7`、`constraint_alloc_switch=1`、
  `constraint_alloc_target_l=0`。
- 约束：O 原子，自旋 z 分量，`fixed_mu=[0,0,0.1]`，
  `constraint_target_mode=delta`。
- MPI：2 进程，命令
  `mpirun --allow-run-as-root -np 2 build/abacus_basic_para`。
- FD：O 原子沿 z 位移 `±0.005 Bohr`（`±0.0026458861 Angstrom`），
  固定 `lambda` 和全部约束目标，使用 `base/plus/minus` 三点能量。
- 互补密度：每个周期的 `D_R` 先做 `C_R = Q_c D_R Q_c^T`，再写入与
  Pauli DMR 同布局的标量自旋槽；传入 Gint 前取 `(C_R+C_R^T)/2`。

## 3. 结果

### 3.1 三点能量与 FD 力

```text
base  E = -467.1747301069827 eV
plus  E = -467.1707686730775 eV
minus E = -467.1782681236987 eV

F_FD(O-z) = -(E_plus - E_minus) / (2 * 0.0026458861 Angstrom)
          = -1.4171907516 eV/A
```

最终解析力：

```text
F_analytic(O-z) = -1.4205728783 eV/A
|F_analytic - F_FD| = 0.0033821267 eV/A < 0.0129 eV/A  PASS
```

总能保持 `-467.1747301069827 eV`，与本轮力修正前逐位相同；本轮只改变
`cal_force()` 中的约束力拼装，不影响 SCF 能量路径。

### 3.2 固定 D 的导数分解

用 base/plus/minus 导出的稠密 `S/W/K` 在固定 `D` 下做差分。目标轨道
选择 `constraint_alloc_target_l=0` 实际得到 6 个轨道
`[0,1,13,14,18,19]`，稠密 `K` 重构误差 `6.9e-6`。

```text
Tr(D dK/dz) total = +0.005127862638299 Ry/Bohr
S-only            = +0.002091355918969 Ry/Bohr
W-only            = +0.003035016715101 Ry/Bohr
E*dS/dz           = +0.002091371773201 Ry/Bohr
```

因此 O-z 的物理约束力分量为：

```text
dS force       = -0.00209137 Ry/Bohr
grid dw force  = +0.00027509 Ry/Bohr
Pulay force    = -0.00331023 Ry/Bohr
```

`grid + Pulay = -0.00303514 Ry/Bohr`，与固定 D 的 W-only
`-0.00303502 Ry/Bohr` 一致。

### 3.3 修正前后

修正前 O-z `-1.2335065989 eV/A`，距 FD `+0.183684 eV/A`。逐项定位为：

1. `dS` 项误用 `force += force_s`；重叠核返回 `+Tr(E dS/dR)`，约束能进入
   总能的符号要求 `force -= force_s`。修正量为 `-0.10757 eV/A`。
2. W 权重还缺基函数 Pulay 项；Gint 局域势核给出的固定权重基函数导数
   为 `-0.00331023 Ry/Bohr`。
3. 初次 Pulay 调用未做 MPI pool 归约，rank-local 值不是物理总量；归约后
   才与固定 D 分解一致。
4. `Q_c D_R Q_c^T` 的单个周期块不保证对称。标量密度对取对称部分不变，
   但 Gint `phi_dot_dphi` 使用 Hermitian DMR 的 2 倍收缩；传入前显式
   对称化，避免反对称部分产生伪 Pulay 力。

### 3.4 回归与构建

```text
cmake --build build --target abacus_basic_para -j 14                 PASS
ctest --test-dir build -R constraint --output-on-failure
19/19 tests passed (95.89 s)                                        PASS
git diff --check                                                     PASS
```

直接相关单测：

```text
MODULE_ESTATE_constraint_alloc_constraint  PASS
MODULE_ESTATE_constraint_alloc_spin        PASS
MODULE_ESTATE_constraint_deriv             PASS
MODULE_ESTATE_constraint_loop              PASS
```

## 4. 分析

LCAO 非共线分配的约束能不是单个网格积分 `int rho_c w`，而是
`Tr(D K)`，其中 `K = Q_i + Q_c W_i Q_c`。它的完整位置导数必须拆成：

- `dS`：`S/B/Q_c` 随几何变化，由 S4.1 的伴随矩阵 `E` 经两中心 overlap
  核计算；
- `dw`：权重场在固定网格点上的显式位置导数，由 `WeightGrid` 的
  `dw/dR` 与互补密度积分；
- 基函数：固定权值场时 `w_i` 的基函数导数，即局域势形式
  `int w_i d(rho_c)`，由 Gint 的 `phi_dot_dphi` 核计算。

三项中只有 `dw` 是此前已通过的 S3 冻结力。前两项分别修正符号并补齐后，
解析力与 FD 在 `0.00338 eV/A` 内闭合。

MPI 下不能直接使用 `Gint_fvl` 返回的局域矩阵：每个 rank 只覆盖自己的
biggrid 子集，标准 `FORCE_STRESS` 也是先算局域项、再
`Parallel_Reduce::reduce_pool`。本轮把 Pulay 力在所有约束累加后做一次 pool
归约，避免把 rank-local 值误当成物理总力。

## 5. 下一步

1. 将非共线 Pulay 项的应变对应量接入 `CONSTRAINT STRESS`，补齐 S4.3 的
   应力审计（当前 S4.3 验收只覆盖旋转不变性与位置力）。
2. 增加 Pauli/kernel adapter 单元测试：不同 spinor block 原点、
   spin-off-diagonal overlap 和互补密度对称化。
3. x 取向在 `scf_thr=1e-8` 仍存在密度混合极限环；单独调整磁密度混合，
   不把它与力路径混为一谈。
4. 自适应向量 multiplier/ALM 仍未实现；当前生产闭环只承诺固定
   `fixed_mu`。
