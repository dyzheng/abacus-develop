# S0 基线审计测试：Becke 归一化与权重导数

**日期**：2026-10-26
**阶段**：S0（统一局域磁矩分配执行方案）
**结论**：PASS。三中心单位分解 `maxdev < 1e-12`；解析导数求和最大误差
`8.882e-16`，违约点 0；对比文档公式已与 `WeightGrid` / `w_becke_adjusted`
实现一致。

## 1. 测试计划

1. 验证三中心 Becke 权重在每个格点满足单位分解，且未归一化乘积在等边三角形
   中心确实不等于 1（预期 `3/4`），证明显式归一化不是冗余步骤。
2. 验证所有原子权重对任一原子位置/笛卡尔分量的解析导数之和逐格点为零。
3. 核验对比文档中的 Becke 公式、归一化写法和单位分解推导与源码一致。

## 2. 测试设置

- 构建：`build/`（CMake debug，`ENABLE_MPI=ON`，`BUILD_TESTING=ON`）。
- 测试文件：`source/source_estate/module_constraint/test/weight_grid_test.cpp`。
- 体系：10 Bohr 立方盒、40×40×40 实空间网格（64,000 点）。
- 三中心用例：O 位于 `(5,5,5)` Bohr，两个 H 移到边长 1.5 Bohr 的等边三角形
  另外两个顶点；中心到三原子等距，所有 `mu_IJ = 0`。
- 导数用例：原始 H2O 构型，O `(5,5,5)`，H `(6.2,5,5)` / `(3.8,5,5)` Bohr；
  分区半径 `(1.5,0.5,0.5)` Bohr。
- 命令：

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_weight_grid
ctest --test-dir build -R '^MODULE_ESTATE_constraint_weight_grid$' --output-on-failure
```

## 3. 结果

```text
[ RUN      ] WeightGridTest.ThreeCentrePartitionOfUnity
[       OK ] WeightGridTest.ThreeCentrePartitionOfUnity
[ RUN      ] WeightGridTest.DerivativeSumRule
weight_grid derivative sum rule: max_dsum=8.882e-16 violations=0
[       OK ] WeightGridTest.DerivativeSumRule
[==========] 14 tests from 1 test suite ran. (1930 ms total)
[  PASSED  ] 14 tests.
```

关键审计值：

| 项目 | 判据 | 实测 |
|---|---:|---:|
| 三中心 `max_g |Σw−1|` | `< 1e-12` | 通过（测试断言） |
| 未归一化等边中心 `Σp` | `= 3/4` | `0.75`（容差 `1e-12`） |
| 导数求和 `max|Σ_a ∂w_a/∂R|` | `< 1e-10` | `8.882e-16` |
| 导数求和违约点数 | `0` | `0` |

文档验收：`2026-10-21-local-moment-algorithms-comparison.md` §1.1 已改为
`μ_IJ = (|r−R_I| − |r−R_J|)/|R_I−R_J|`、`p_I = Π s(μ_IJ)`、
`w_I = p_I/Σ_J p_J`，并删除“成对恒等式直接推出多中心单位分解”的错误表述。
原 `"r_I + r_J"` 公式已无残留。

## 4. 分析

- 首版反例误用了共线 H2O 的“几何中心”。三个共线点一般不存在到三点等距的
  位置，因此该构型无法实现 `Σp = 3/4`；首次运行得到 `0.99927` 后，用例改为
  真正的等边三角形顶点，随后严格得到 `0.75`。
- `w_becke_adjusted()` 在返回前执行 `P[c] / ΣP`，因此导数核也必须在同一
  归一化约定下满足零和。逐格点审计结果为机器精度级零和，未观察到镜像切换面
  违约，R6 的 Becke 导数基线为 0 违约点（本构型/网格）。
- `WeightGrid::build()` 的 `maxdev` 是全局 MPI 最大值；S0 新增测试直接覆盖
  三中心非共线几何，补上了原有 `PartitionOfUnity` 之外的反例保护。

## 5. 下一步

1. 进入 S1.1：实现无 Eigen 的 R×R Jacobi 对称特征分解与 `S^{-1/2}`。
2. S1.2/S1.3 复用本次审计约定：权重由 `WeightGrid` 提供，目标轨道正交化后
   必须单独审计条件数与单位分解。
3. S5.2 周期 Hirshfeld 后端沿用同一导数零和判据，并要求镜像切换面附近
   FD 连续。
