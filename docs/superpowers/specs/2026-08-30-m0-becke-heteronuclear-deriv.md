# 2026-08-30 M0：Becke 异核修正 χ_ij + 解析位置导数

## 1. Test plan
- `PartitionTest.BeckeHeteronuclearMidpoint`：两中心（radii=1.5/0.5）相距
  2 Bohr、网格点在中点，断言大原子权重 > 0.5（异核修正后切换面偏向大原子侧
  移），且 Σ_α w_α ≡ 1（1e-12）；交换半径顺序后符号反转（小原子权重 < 0.5）。
- `PartitionTest.BeckeDerivFD`：三中心随机几何，∂w_0/∂R_1 三分量解析值 vs
  δ=1e-5 Bohr 中心差分，判据 1e-6；另断言对不在 iR 中的中心 J 导数为零。
- 回归：原有 `Becke` / `Stratmann` 数值积分测试不得破坏。

## 2. Test setup
- 平台：容器内 gcc C++17，GoogleTest，`MODULE_BASE_GRID_test_partition` 目标。
- 输入：测试内合成几何（随机数种子固定）；沿用现有 test_partition.cpp 风格。
- 命令：`cmake --build build --target MODULE_BASE_GRID_test_partition &&
  ./build/source/source_base/module_grid/test/MODULE_BASE_GRID_test_partition
  --gtest_filter='PartitionTest.*'`。

## 3. Results
- TDD 顺序：先追加测试 → 编译失败（`w_becke_adjusted` 未声明，符合预期）→
  实现 → 测试通过。
- 首轮实现后 `BeckeDerivFD` FAIL：解析值 vs FD 误差 ~2.5e-5 且 comp 0 为
  NaN。独立复现程序（/tmp/deriv_debug*.cpp）定位根因：实现中 `const double d`
  （中心距）与内层 `for (int d = 0; d < 3; ++d)` 循环变量同名遮蔽，comp 0
  除以循环变量 0 → NaN，comp 1/2 除以 1/2 → 错误值。
- 修复（循环变量改名 dd）后：随机 5 几何 × 3 分量，解析 vs FD 最大偏差
  ~5.5e-11（判据 1e-6），全部 PASS。
- 最终：`PartitionTest.*` 4/4 PASS（含原有 Becke/Stratmann 不回归）。

## 4. Analysis
- 异核修正公式（Becke 1988 Eq.15）：u_ij=(χ_ij−1)/(χ_ij+1)，
  a_ij=clip(u_ij/(u_ij²−1), ±0.5)，μ'_ij=μ_ij+a_ij(1−μ_ij²)。中点
  （μ=0）时 μ'=a<0（大原子侧），s(μ')>0.5 → 大原子权重 > 0.5，与预期一致。
- 位置导数链式：∂w/∂R_J 经 dlnP（对数导数）组装，∂μ/∂R_J 由 drR/dRR 几何
  导数闭式给出；(R_I−R_K) 由方向余弦重建 drR[I]·eR_I−drR[K]·eR_K。
- 遮蔽 bug 是典型命名缺陷：距离变量与分量循环变量都叫 d。已固定命名
  （距离=d、分量循环=dd），后续模块避免同类命名。

## 5. Next steps
- Task 2 (M1)：`weight_grid.h/.cpp` 网格权重构造 + sum-rule 审计，合成网格
  测试（40 Bohr 盒、0.2 Bohr 间距）。
