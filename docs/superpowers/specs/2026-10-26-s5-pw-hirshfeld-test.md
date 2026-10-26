# S5 核心：PW 分配后端与周期 Hirshfeld 权重测试

日期：2026-10-26

## 1. 测试计划

本轮先完成 S5 的底层核心，并分别验证以下性质：

1. PW 目标空间的 `ST = <alpha_a|alpha_b>`、`ST^{-1/2}` 与 LCAO 目标空间一致。
2. 完整目标集下，PW 读数满足 `sum n_T = N`、`sum m_T = M_tot`。
3. `apply_pw_alloc_delta_h()` 在单位 overlap、同一目标选择下与 LCAO
   `build_alloc_delta_h()` 的 `K psi` 逐分量一致。
4. `OnsiteProjector` 风格 tabulation 能正确构造 projector overlap 和
   `becp[band][target]`。
5. 上述两个基于 `reduce_pool` 的 helper 在 2/4 MPI rank 下与全局参考一致。
6. 周期 Hirshfeld 权重满足 partition of unity，并在大真空区域启用显式余量。
7. Hirshfeld 原子权重与余量权重的解析位置导数满足求和规则，并通过位置 FD。
8. 跨周期边界的权重导数连续（相对误差 `< 1e-6`）。
9. `constraint_weight_type=hirshfeld_periodic` 能进入 `ConstraintLoop` 的共享
   `WeightGrid`，不静默回退到 Becke。
10. 全量 `constraint` 回归不破。

## 2. 测试设置

- 系统：Ubuntu 24.04，`c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- MPI：Open MPI 4.1.6；`ENABLE_MPI=ON`
- 构建类型：`debug`
- 构建目录：`build/`
- 配置/构建命令：

```bash
cmake -S . -B build
cmake --build build --target abacus_basic_para -j 14
```

- 单元测试：
  - `MODULE_ESTATE_constraint_alloc_pw`
  - `MODULE_ESTATE_constraint_weight_grid`
  - `MODULE_ESTATE_constraint_io`
  - `MODULE_ESTATE_constraint_deriv`
  - `MODULE_ESTATE_constraint_loop`
- MPI 测试：
  - `MODULE_ESTATE_constraint_alloc_pw_mpi`
  - `MODULE_ESTATE_constraint_alloc_pw_mpi_2np`
  - `MODULE_ESTATE_constraint_alloc_pw_mpi_4np`
- Hirshfeld 测试盒：10 Bohr 立方 H2O，O 在中心，两个 H 关于 `x=5 Bohr`
  镜像；权重网格 `40 x 40 x 40`，半径 `{1.5, 0.5, 0.5} Bohr`。

## 3. 结果

### 3.1 PW 目标空间与读数

```text
alloc_pw parity: max_dnT=3.553e-14 max_dmT=1.066e-14
alloc_pw sum: n=5.113173468547e+02/5.113173468547e+02
              m=1.416982052033e+02/1.416982052033e+02 dsum=0.000e+00
alloc_pw apply: max_diff=2.082e-17
```

`ReadoutMatchesLcaoTargetSpace`、`CompleteTargetSumRules`、
`ApplyDeltaHMatchesLcaoKernel`、`ProjectorTabulationBuildsOverlapAndBecp`
四条测试全部 PASS。代数平价误差远低于 S5 闸门 `0.02 mu_B`；完整目标集
的电荷/磁矩求和误差为 0（打印精度内）。

### 3.2 PW MPI 归约

```text
MODULE_ESTATE_constraint_alloc_pw_mpi ...... Passed
MODULE_ESTATE_constraint_alloc_pw_mpi_2np .. Passed
MODULE_ESTATE_constraint_alloc_pw_mpi_4np .. Passed
```

每个 rank 持有互不重叠的平面波切片；`ST` 与 `becp` 都与各 rank 独立计算的
全局参考一致（`1e-12` 判据）。

### 3.3 周期 Hirshfeld 权重

```text
hirshfeld partition: maxdev=2.220e-16 max_remainder=1.000000
hirshfeld derivative sum: total=2.842e-14 covered_atoms=2.842e-14
hirshfeld position FD: max_abs=1.000e-11 max_deriv=7.493e-02
hirshfeld boundary FD: checked=1393 max_abs=6.972e-08 max_deriv=8.395e+00
```

- `HirshfeldPeriodicPartitionAndRemainder`：`maxdev < 1e-12`，大真空区域
  余量权重达到 `1.0`，原子权重与余量权重之和为 1。
- `HirshfeldDerivativeSumAndFiniteDifference`：全导数求和与全覆盖区域
  原子导数求和均 `< 1e-10`；位置 FD 最大绝对差 `1.000e-11`。
- `HirshfeldDerivativeAcrossPeriodicBoundary`：在原子分数坐标 `x=0.98`
  附近检查 1393 个内部点，FD 相对误差远低于 `1e-6`。
- `ConstraintLoopTest.HirshfeldWeightTypeReachesLoop`：配置正确到达 loop，
  `is_hirshfeld()==true` 且 `max_partition_deviation()<1e-10`。

### 3.4 构建与回归

```text
abacus_basic_para: built successfully
ctest -R constraint: 23/23 Passed (101.91 s)
git diff --check: clean
```

## 4. 分析

1. PW 后端的核心公式已经与 LCAO 算子对齐：目标 overlap 正交化、becp 读数
   和 `apply_dH` 的低秩修正均在代数层面闭合。MPI 测试确认 `reduce_pool`
   路径没有 rank-local 重复或漏归约。
2. 周期 Hirshfeld 的 C2 窗口、晶格镜像求和与商法则导数满足设计要求；
   余量通道在真空区显式接管权重，避免 Becke 最小镜像切换面处的导数跳变。
3. 当前实现仍是核心层，不是 PW SCF 生产接线：尚未把 `alloc_pw` 接到实际
   `OnsiteProjector`/波函数/`veff` 路径，因此还没有真实 PW/LCAO 同电子态
   三量对照表和 PW 侧 FD 力数据。
4. Hirshfeld 应力尚未实现；`constraint_stress()` 与
   `constraint_stress_densities()` 已显式拒绝该后端，避免静默返回零。
5. Becke/Hirshfeld 在同一实际电子态上的三量对照表尚未生成，故 S5 总闸门
   不能提前关闭。

## 5. 下一步

1. 将 `alloc_pw` 接入 PW esolver：读取 projector tabulation、占据波函数和
   `V phi`，形成实际约束势；补 PW 侧位置导数/FD 力。
2. 用同一 H2O 电子态对拍 LCAO/PW 的 `n_T`、`m_T` 与总磁矩，目标
   `|Delta m_T| < 0.02 mu_B`、`|Delta M| < 0.01 mu_B`。
3. 在同一电子态生成 Becke/Hirshfeld 两后端的三量对照表。
4. 评估并补齐 Hirshfeld 应变导数，使应力路径不再需要显式拒绝。
5. 上述完成后更新执行方案 S5 复选框与总闸门状态。
