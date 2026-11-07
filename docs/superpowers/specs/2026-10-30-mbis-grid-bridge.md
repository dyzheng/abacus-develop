# 2026-10-30 阶段 1：ABACUS PW 网格 → 模块 Grid 的 bridge

## 0. 结论

用真实 LiCl/Fe2 数据**确认了网格约定**，并实现 `pw_grid_bridge.h`
（`make_grid` / `make_atoms`）+ 单测。核心 gtest 现 **9/9 PASS**。

## 1. 网格约定（已用真实数据核对）

```
a_i  = LATTICE_CONSTANT * LATTICE_VECTORS_i          (Bohr)
dv_i = a_i / n_i ,  origin = 0
ir   = (ix*n1 + iy)*n2 + iz                          (z 最快)
```
核对：
- **LiCl**：`LATTICE_CONSTANT=1.8897261254`，vector `15·I`
  ⇒ `a=28.3458919 Bohr`；cube `n=192, dv=0.147635` ⇒ `n·dv=28.34592` ✓。
- **Fe2**：`LATTICE_CONSTANT=8.190`，vectors `(1,½,½),(½,1,½),(½,½,1)`
  ⇒ `a_0=(8.19,4.095,4.095)`；cube `n=30, dv_0=(0.273,0.1365,0.1365)`
  ⇒ `30·dv_0=(8.19,4.095,4.095)` ✓（精确）。

## 2. bridge

```cpp
#include "source_estate/module_mbis_alm/pw_grid_bridge.h"
Grid g = mbis_alm::make_grid(n /*{nx,ny,nz}*/, a /*lattice vectors Bohr*/);
auto atoms = mbis_alm::make_atoms(R /*Cartesian Bohr*/, Zval /*ncpp.zv*/);
```
- `make_grid`：`dv_i=a_i/n_i`、`origin=0`；
- `make_atoms`：位置 + 价电子数。

## 3. 单测（`MbisAlmTest.GridBridgeMatchesLattice`）

- Fe2：`dV()*nxyz = |det(a)|`；`point(29,0,0)=(29/30)·a_0`；
- LiCl：`dV()*192³ = L³`；
- atoms：`Zval`/`R` 保持。
全部通过（核心 **9/9**，adapter 2/2，共 11/11）。

## 4. MPI 说明（待接真实 ESolver 时处理）

`chr.rho` 是 rank-local（`nrxx`）；MBIS 核心是串行。接入时：
- 先 **gather 全网格 → rank0 计算 MBIS → scatter 权重**（简单，先跑通）；
- 或把 `compute_mbis` 改为 MPI-aware（pro-density/求和 all-reduce）。
求和审计沿用 `Σ_A w_A(r)=1`。

## 5. 下一步

- `WeightGrid::build_mbis(rho_total)`：用 bridge + `compute_mbis` 填充 `w_`/`cw_`
  （与 Becke/Hirshfeld 同布局）；
- 自由密度一致性测试：Fe2/LiCl 的 MBIS N/M 与 cube 结果一致
  （LiCl Li 2.0447；Fe2 16/16、±2.65）；
- 之后接 `constraint_loop` 的外层重建与 `constraint_weight_type mbis`。
