# 2026-10-30 唯一解算法：多初值 + 最低 KL 目标 D

## 0. 背景

MBIS 的 KL 自洽解**非凸、可能多解**（见 `2026-10-30-mbis-kl-multiplicity-hessian.md`）：
定常迭代只落到初值盆地的一个解；等价原子的破缺解即此类。需一个**选解准则**：
**MBIS 定义下的"正确解"= KL 目标 `D` 最低的自洽解**（全局最小）。

## 1. 算法

```
输入: rho, grid, atoms, cfg, n_starts=4
for s in {0.5, 1.0, 2.0, 4.0} (init_zeta):
    cfg_s = cfg with init_zeta = s ; 清空 warm 初值
    state_s = compute_mbis(rho, grid, atoms, cfg_s)   # 定常迭代到自洽
    D_s = KL(state_s)                                  # 见下
return state with min D_s
```
KL 目标（每个解处计算）：
```
D = Σ_A ∫ ρ_A^pro(r) [ ln ρ_A^pro(r) − ln ρ_A(r) ] dr ,  ρ_A = w_A ρ
```
（`ρ_A^pro>ε` 且 `ρ_A>ε` 处积分；ε=1e-30 防 log 0。）

- **多初值**探索不同盆地（不同 `ζ` 起点 → 不同解）；
- **最低 `D`** 选出 MBIS 定义的全局最小（在搜索到的解中）；
- **可选对称投影** `cfg.symmetrize`：对等价原子把 `(N,ζ)` 投到对称子空间
  （仅在"对称点为极小"时合理，见 Hessian 判据）。

## 2. 与"唯一性"的关系

- 该算法给出**"所选解集内 D 最低"**的解——是 MBIS 目标意义下的**正确解**；
- **不保证**穷尽所有盆地（初值网格有限）；可增大 `n_starts`/扩大 `init_zeta` 范围；
- **Hessian 认证**（下一步）：在选定解处算 `∂²D/∂p²`，`≻0` 则是严格局部极小；
  若对称点在破缺方向为负（鞍点），则破缺解才是真最小，不应强制对称化。

## 3. 实现与测试

- `MbisState` 新增 `double D`；`compute_mbis` 在收敛态计算并返回 `D`；
- 新增 `compute_mbis_unique(rho, grid, atoms, cfg, n_starts=4)`：
  多初值 `init_zeta∈{0.5,1,2,4}`，取最低 `D`；
- 新 gtest `UniqueSolverPicksLowestDAndSymmetric`：
  - `D(unique) ≤ D(single-start)`；
  - 开启 `symmetrize` 后 `|N0−N1|<1e-8`、`|ζ0−ζ1|<1e-8`、电荷求和守恒；
  - 通过（核心测试累计）。

## 4. 局限与下一步

- 多初值只"选解"，非严格全局最优证明（非凸 NP 类）；可加更多起点/局部扰动。
- 实现 **Hessian（FD 或解析）** 认证与"逃鞍点"（沿负曲率方向）→ 更可靠的唯一解。
- 在 ABACUS `build_mbis` 中可选用 `compute_mbis_unique`（成本 ×n_starts，需权衡）。
