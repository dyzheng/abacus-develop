# 2026-10-30 独立 MBIS 模块的真实 cube 交叉验证（C++ vs Python）

## 0. 结论

用真实 ABACUS 密度 cube 驱动独立 C++ 模块 `source/source_estate/module_mbis_alm/`，
结果与先前 Python 原型**逐位一致**；过程中对照验证抓到一个真实 bug 并修复。

| 体系 | 量 | C++ 模块 | Python 原型 |
|---|---|---|---|
| LiCl（nspin=1, 192³, 对角格） | Li N | **2.044663** | 2.04466 |
| | Li netQ | **+0.955337** | +0.95534 |
| | Li ζ | **4.59730** | 4.59730 |
| | Cl N | **7.955367** | 7.95537 |
| | Cl ζ | **1.61196** | 1.61196 |
| Fe2 Γ（nspin=2, 30³, bcc 斜胞, images=2+对称化） | Fe N | **16.00/16.00** | 16.00/16.00 |
| | Fe ζ | **1.07789** | 1.078 |
| | Fe 自旋 | **±2.647** | ±2.647 |

求和：LiCl `ΣN=10.000030`、Fe2 `ΣN=32.000000`、`ΣM≈0.0047`（=tot_mag）。

## 1. 工具

`tools/mbis_alm_cube.cpp`（独立编译，无 ABACUS 依赖）：
读 ABACUS cube（支持**非对角体素**、单/双自旋 cube），构建 `Grid/Atom`，
调用 `compute_mbis` + `observe`，打印每原子 N/净电荷/ζ/自旋与求和审计。

```bash
g++ -std=c++14 -O2 -I source tools/mbis_alm_cube.cpp \
    source/source_estate/module_mbis_alm/mbis_alm.cpp -o /tmp/mbis_alm_cube
/tmp/mbis_alm_cube OUT.*/chg.cube                       # nspin=1
/tmp/mbis_alm_cube OUT.*/chgs1.cube OUT.*/chgs2.cube --images 2 --symmetrize
```

## 2. 抓到并修复的 bug

首次 Fe2 运行给出 `N=16.525/15.475、ζ=1.253/1.766`（**非对称**），而 Python 为
`16/16、ζ=1.078`。根因：**对称化在同一个 `for(a)` 循环里原地覆盖 `Nn/zn`**，
导致后续原子的平均使用已被改写的值（二次污染）：

```
for a:
   sum over b: nm += Nn[b]; zm += zn[b]
   Nn[a] = nm/na          // 覆盖后，下一个 a 的和被污染
```
修复：先写入临时数组 `Ns/zs`，循环结束后再整体赋值。

## 3. 回归测试

新增 `MbisAlmTest.SymmetrizeRestoresEquivalence`（bcc 斜胞 + 两等价原子 +
`images=1`）：断言 raw 分区**非对称**（`|N0−N1|>1e-3`），而 `symmetrize=true`
后 `|N0−N1|<1e-8`、`|ζ0−ζ1|<1e-8`，且电荷求和守恒。
独立 gtest 现 **8/8 PASS**（约 5 s）。

## 4. 意义

- **C++ 实现正确**：与 Python 逐位一致（真实 LiCl/Fe2 密度）。
- **解耦有效**：cube 驱动/测试独立编译，不依赖 ABACUS 构建。
- **交叉验证有价值**：两个独立实现互为对照，直接暴露了 C++ 的原地覆盖 bug。

## 5. 下一步

- 在真实密度导出的**约束响应**上跑 ALM vs Lagrange vs penalty（半真实对比）；
- 写**薄接口层**（adapter）接 ESolver：定义 `get_density / solve / apply_potential /
  observe / update_multipliers`，核心模块保持独立，用 mock adapter 先做端到端测试。
