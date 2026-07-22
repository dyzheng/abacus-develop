# snap vs center2_orb11 详细分析

> **日期**: 2026-06-27
> **结论**: 两者使用**相同的数学方法**（球贝塞尔变换 + Gaunt 系数），但 **k 空间网格密度不同**。

---

## 1. snap (TwoCenterIntegrator) 的实现

### 1.1 构建流程

```
TwoCenterBundle::tabulate(lcao_ecut, lcao_dk, lcao_dr, lcao_rmax)
  → orb_->set_grid(nk, kgrid, 't')  // k 空间网格
  → overlap_orb->tabulate(*orb_, *orb_, 'S', nr, cutoff)
    → TwoCenterTable::build(bra, ket, 'S', nr, cutoff)
      → _tabulate(it1, it2, l)
        → it1->radtab('S', *it2, l, tab, nr, rmax)  // 球贝塞尔变换
        → CubicSpline::build(nr, rgrid, tab, ...)     // 三次样条插值
```

### 1.2 snap 的 `calculate` 方法

```
TwoCenterIntegrator::calculate(...)
  → table_.lookup(..., R, val, dval, d2val)
    → CubicSpline::eval(nr, rgrid, tab, dtab, 1, &R, val, ...)
```

**snap 在查询时使用三次样条插值**，不是直接计算积分。

### 1.3 snap 的网格参数

- **k 空间**: `nk = sqrt(lcao_ecut) / lcao_dk + 4`（奇数）
  - BaTiO3: `nk = sqrt(100) / 0.01 + 4 = 1005`
- **r 空间**: `nr = cutoff / lcao_dr + 5`
  - `cutoff = min(lcao_rmax=30, 2 * rcut_max)`
  - `nr = cutoff / 0.01 + 5`

### 1.4 snap 的积分方法

`it1->radtab('S', *it2, l, tab, nr, rmax)`:
- 这是 `NumericalRadial::radtab` 函数
- 它做球贝塞尔变换：在 k 空间做积分，得到 r 空间的表
- 使用 Simpson 积分（`ModuleBase::Integral::simpson`）

## 2. center2_orb11 (unkOverlap_lcao) 的实现

### 2.1 构建流程

```
unkOverlap_lcao::init(ucell, nkstot, orb)
  → Center2_Orb::init_Table_Spherical_Bessel(Lmax, dr, dk, kmesh, Rmesh, psb)
  → cal_orb_overlap(ucell)
    → center2_orb11[T1][T2][L1][N1][L2].at(N2).cal_overlap(origin, distance, m1, m2)
```

### 2.2 center2 的 `cal_overlap` 方法

`Center2_Orb::Orb11::cal_overlap`:
- 在 r 空间查询预计算的表
- 使用球贝塞尔递归（`Sph_Bessel_Recursive::D2`）

### 2.3 center2 的网格参数

- **k 空间**: `kmesh = orb.get_kmesh() * 4 + 1`
  - `orb.get_kmesh() = sqrt(ecutwfc) / lcao_dk + 4 = 1005`
  - `kmesh = 1005 * 4 + 1 = 4021`
- **r 空间**: `Rmesh = orb.get_Rmax() / dr + 4 = 3005`

### 2.4 center2 的积分方法

同样是球贝塞尔变换，但 k 空间网格是 snap 的 **4 倍**。

## 3. 对比

| 参数 | snap | center2 | 比例 |
|------|------|---------|------|
| k 空间网格点 | 1005 | 4021 | 4× |
| r 空间网格点 | ~3005 | 3005 | ~1× |
| 数学方法 | 球贝塞尔变换 | 球贝塞尔变换 | 相同 |
| 查询方法 | 三次样条插值 | 球贝塞尔递归 | 不同 |
| Simpson 积分 | 是 | 是 | 相同 |

## 4. 分析

### 4.1 用户判断的验证

用户说"snap 使用的方法是精度更高的，不需要依赖网格密度"。

**部分正确**: snap 使用 `RadialCollection` + `NumericalRadial::radtab`，这是 ABACUS 较新的实现（ATen tensor），使用了三次样条插值进行查询，理论上更精确。

**但实际上**: snap 的 k 空间网格（1005）比 center2 的（4021）**少 4 倍**。即使 snap 的查询方法（三次样条）更精确，更粗的 k 空间网格会导致球贝塞尔变换的精度降低。

### 4.2 网格差异的影响

球贝塞尔变换在 k 空间做积分：
$$S_l(R) = \int_0^{k_{\max}} f_1(k) \, f_2(k) \, j_l(kR) \, k^2 \, dk$$

k 空间网格越密，积分越精确。center2 用 4021 个 k 点，snap 用 1005 个 k 点。

对于平滑的被积函数，1005 个 k 点可能足够。但对于高 l 值（如 l=3, f 轨道），$j_l(kR)$ 振荡更剧烈，需要更密的 k 网格。

### 4.3 提高lcao_ecut是否可以弥补？

**可以**。snap 的 k 网格由 `lcao_ecut` 控制：
- `nk = sqrt(lcao_ecut) / lcao_dk + 4`
- 当前: `nk = sqrt(100) / 0.01 + 4 = 1005`
- 如果 `lcao_ecut = 10000`: `nk = sqrt(10000) / 0.01 + 4 = 10004`

增大 `lcao_ecut` 会使 snap 的 k 网格更密，接近 center2 的精度。

但更简单的方法：**使 snap 使用与 center2 相同的 k 网格**。center2 的 kmesh = orb.kmesh × 4 + 1，而 snap 的 nk = orb.kmesh。如果让 snap 也用 orb.kmesh × 4 + 1，两者网格就一致了。

### 4.4 更根本的问题

实际上，snap 的 k 网格和 center2 的 k 网格都来自同一个 `orb.kmesh`：
- snap: `nk = sqrt(lcao_ecut) / lcao_dk + 4` = `orb.kmesh`（相同！）
- center2: `kmesh = orb.kmesh * 4 + 1`（额外乘 4！）

center2 的 `* 4 + 1` 是一个**额外的精度提升步骤**：它将 orb.kmesh 扩展 4 倍，用于更密的球贝塞尔变换。snap 没有做这个扩展。

**这是导致 snap 和 center2 数值差异的直接原因。**

## 5. 修复方案

### 方案 1: 增大 lcao_ecut（用户建议）

在 INPUT 中设置 `lcao_ecut` 更大值：
```
lcao_ecut    1600    # 16× larger, nk = sqrt(1600)/0.01 + 4 = 4004 ≈ center2's 4021
```

这会使 snap 的 k 网格接近 center2，数值差异应该减小。

**优点**: 无需改代码，只需调参数。
**缺点**: 增大所有二中心积分表的内存和构建时间。

### 方案 2: 使 snap 的网格匹配 center2

修改 `TwoCenterBundle::tabulate`，将 nk 设为 `orb.kmesh * 4 + 1`：
```cpp
int nk = orb_->kmesh() * 4 + 1;  // 匹配 center2_orb11
```

**优点**: 精确匹配 center2。
**缺点**: 修改 ABACUS 核心代码，影响所有二中心积分。

### 方案 3: 在 DeltaP 中直接使用 center2

不使用 snap，而是用 `unkOverlap_lcao` 的 `cal_orb_overlap` 结果（psi_psi）构建 S_dk。

**优点**: 与 berry_phase 完全一致。
**缺点**: 需要修复 berryphase_overlap 的内存问题。

### 推荐: 方案 1（最简单）

设置 `lcao_ecut = 1600`，验证 P_elec ratio 是否接近 1.0。
如果是，确认根因是 k 网格差异。
如果不是，说明还有其他因素（如查询方法差异）。

