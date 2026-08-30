# 2026-08-31 M1：网格权重构造 + 缓存 + 审计（weight_grid）

## 1. Test plan
- `PartitionOfUnity`：H2O 三原子 + 10 Bohr 立方盒、0.5 Bohr 网格间距（20³，
  合成网格不经 SCF），逐点 Σ_α w_α ≡ 1，判据 maxdev < 1e-10；打印耗时。
- `SymmetryMirror`：O 在盒心、两 H 关于 x=5 Bohr 镜像对称 → w_H1(ir) 与
  w_H2(mirror(ir)) 逐点相等（1e-12），w_O 自镜像逐点相等。
- `DeterministicRebuild`：同几何两次 build 位一致。
- `ScreeningKeepsPartition`：3 Bohr 截断筛选后仍为单位分解；筛选确实改变
  远处点权重（非空操作）。
- `FragmentConstraint`：约束 {O,H1} 的权重 == w_O + w_H1 逐点。
- MPI：`DistributedMatchesSerial`（test_mpi）——非方网格 24×16×20，1/2/4
  rank 分布式局部权重 vs rank0 全序列参考逐点一致（1e-12）；maxdev 全局
  allreduce 后 < 1e-10。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest；`MODULE_ESTATE_constraint_weight_grid`
  （串行）与 `MODULE_ESTATE_constraint_weight_grid_mpi`（mpirun -np 4）。
- 输入：`UcellTestPrepare` 构造 H2O（lat0=1 Bohr、latvec=10I → 10 Bohr 盒，
  tau 即 Cartesian 坐标）；PW_Basis `initgrids(1.0, latvec, 20,20,20)`。
- 半径表：O=1.5、H=0.5 Bohr（Bragg-Slater 量级）。
- 命令：`cmake --build build --target MODULE_ESTATE_constraint_weight_grid &&
  ./.../MODULE_ESTATE_constraint_weight_grid`；MPI 版
  `mpirun -np 4 ./MODULE_ESTATE_constraint_weight_grid_mpi`。

## 3. Results
- 串行 5/5 PASS；MPI 1/2/4 rank 全 PASS（逐点 vs 序列参考 < 1e-12）。
- 实测耗时：8000 点 × 3 原子 build = 8.85e-4 s（每点 1.1e-7 s），
  线性外推 ~1e6 点 ≈ 0.11 s，远小于一步 SCF。
- 实现要点：每几何一次 build（禁止依赖密度）；默认约束=每原子一约束，
  `set_constraint_atoms` 支持片段（原子权重求和，不重算 w）；最小镜像距离
  处理周期性；maxdev 走 `Parallel_Reduce::reduce_max_pool`（串行 no-op）。

## 4. Analysis
- 单位分解在任意网格点数学上恒成立（归一化在参与中心集上进行），审计
  maxdev 实际 ~1e-15；1e-10 判据为永久回归红线。
- 镜像对称逐点相等验证了权重是纯几何量且与网格离散化相容（反射映射
  x→L−x 在网格上自洽）。
- 截断筛选：参与中心集为空时回退全原子，避免真空点出现 Σw=0 破坏电荷
  守恒；筛选改变的是参与集而非配方。

## 5. Next steps
- Task 3 (M2)：`constraint_observe` 网格读数 Q_α=Σ_g w_α ρ ΔV，原子叠加
  密度解析对拍（Σ_I Q_I == N_el，1e-8/1e-10）。
