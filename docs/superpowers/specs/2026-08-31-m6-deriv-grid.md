# 2026-08-31: Task 2.5.1——M1 扩充：权重位置导数网格 ∂w_α/∂R_J

> 上游：`docs/superpowers/plans/2026-08-31-task25-m6-force-detail.md` 2.5.1。
> M6 力核 `F_J = −Σ_α μ_α ∫ρ ∂w_α/∂R_J dr` 的输入网格；调一期已交付的 M0
> 导数核 `Grid::Partition::w_becke_adjusted_deriv`，零新组件/算法。

## 测试计划（失败测试先行）

1. `WeightGridTest.DerivGridAnalytic`：组装导数网格 vs M0 逐点直接调用一致（1e-12）。
2. `WeightGridTest.DerivGridCoincidentPointZero`：网格点与原子重合处（μ=±1，
   裸核 0/0）导数恒零。
3. `WeightGridTest.DerivGridTranslationInvariance`：刚性平移链式自检
   `Σ_J ∂w_α/∂R_J + ∂w_α/∂r ≡ 0`（1e-10），∂w/∂r 用独立 5 点中心差分。
4. `WeightGridTest.DerivFragmentConstraint`：fragment 约束映射在
   build_derivatives 之后/之前设置两序一致。
5. `WeightGridMpiTest.DerivGridMPI`：4 rank 局部导数网格 vs 串行全网格参考
   逐点一致（1e-12）。
6. partition 单测回归（M0 核守卫改动）。
7. `ctest -R constraint` 全套回归。

## 测试设置

- 系统：H₂O / 10 Bohr 立方盒（O 在盒心 5,5,5；H1/H2 在 x=6.2/3.8 镜像）。
- 网格：串行 40³（0.25 Bohr 间距）；MPI 4 rank 非方网格 24×16×20。
- 半径：O=1.5、H=0.5 Bohr；Becke 异核修正（M0）。
- 构建：`cmake --build . --target MODULE_ESTATE_constraint_weight_grid[_mpi]`。

## 结果

| 测试 | 结果 |
|---|---|
| DerivGridAnalytic | PASS（20 探针 × 9 分量，1e-12） |
| DerivGridCoincidentPointZero | PASS |
| DerivGridTranslationInvariance | PASS（过滤后 20 探针 × 3 分量，1e-10） |
| DerivFragmentConstraint | PASS |
| DerivGridMPI（4 rank） | PASS（3/3 MPI 全绿） |
| MODULE_BASE_GRID_test_partition | 4/4 PASS |
| ctest -R constraint | 10/10 PASS |

实现后全绿；失败测试先行阶段各测试均先按预期 FAIL（未实现接口无法编译，
TDD 红线）。

## 分析

- **tie-break 边界奇异（重要）**：周期性最小镜像约定下，原子恰在 ±0.5 分数
  坐标边界（如 O 在 x=0.5、网格点在 x=0.0）时，`w(R)` 作为原子位置函数**不可微**
  ——跨边界镜像图像跳变，导数不连续。中心差分给"平均导数"（≈0），解析链式
  导数给"解析侧"值（如 `d w_O/dR_O,x grid=0.893 vs fd=0.000`）。这是单点测度
  零的固有奇异，对力体积分无影响。处置：`DerivGridTranslationInvariance`
  跳过图像不稳定探针（任一原子任一分量 `|wrap_frac| ≥ 0.49`，≈0.1 Bohr 保守
  余量），`build_derivatives` 实现不改（奇异点输出解析侧导数可接受）。
- **M0 核 0/0 NaN 修复**：网格点恰在原子连线延长线（μ=±1）时 `sp/s` 除零。
  该项因子 s(±1)=0 且 s'(±1)=0，乘积导数为 0，应跳过而非 0/0。`partition.cpp`
  加 `s != 0.0` / `1.0−s != 0.0` 守卫；partition 单测 4/4 无回归。
- **CMake 重复符号修复（既有问题）**：`partition.cpp` 自一期 M8 起已加入
  `base` OBJECT 库，而 `weight_grid_mpi` 测试目标仍直编它 → multiple
  definition（此前该目标在本构建树从未链成）。按串行版 `weight_grid` 目标
  既有模式，从测试 SOURCES 移除直编行，partition 符号由 `base` 提供。
- 调试测试 `DbgTieBreak`（诊断用临时打印）已删除。

## 下一步

- 2.5.2：M6 力核 `constraint_deriv`（F_J 网格积分 + reduce_pool 归约；
  charge/spin 通道分支注释纪律；失败测试：合成密度解析对拍 / 牛顿第三定律 /
  力对 μ 线性）。
