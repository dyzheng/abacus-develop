# 2026-10-30 约束求解器对比：ALM vs 纯 Lagrange vs 纯 penalty

## 0. 结果（独立线性响应模型）

模型 `Q_a(v)=Q0_a−χ_a v_a`，`χ={10, 2, 0.5, 0.01}`（含一个近零模），target `t=0`，
`tol=1e-10`。运行 `tools/mbis_alm_compare.cpp`（独立编译，仅依赖
`source/source_estate/module_mbis_alm`）：

| 方案 | 迭代 | max\|Q−t\| | 收敛 |
|---|---:|---:|---|
| Lagrange（全局步 α=0.1） | **23016** | 9.99e-11 | 是 |
| **ALM k=1/χ** | **34** | 5.82e-11 | 是 |
| **ALM k=10/χ** | **10** | 3.86e-11 | 是 |
| penalty-only k=1/χ | 1 | 5.00e-01 | **否（偏置）** |
| penalty-only k=10/χ | 1 | 9.09e-02 | **否（偏置）** |

## 1. 分析

- **纯 Lagrange 被近零模拖死**：全局步长必须满足 `α<2/χ_max=0.2`；对 `χ=0.01`
  的模收缩率 `1−αχ≈0.999`，需 ~2.3×10⁴ 步。这正是此前 allocation 63 步、
  3.8 态近零模/2-周期的数学根源（Schur 补近奇异）。
- **ALM 按模态自适应 k**：`k=κ/χ` ⇒ 每个模的收缩率 `1/(1+kχ)`；`κ=1`→1/2（34 步），
  `κ=10`→1/11（10 步）。**近零模不再拖累整体**。
- **纯 penalty 快但有偏**：有限 k 下 `Q*−t=(Q0−t)/(1+kχ)`（k=κ/χ 时为
  `(Q0−t)/(1+κ)`）；k 越大偏差越小，但**永不精确**，且内层势 `1+kχ` 变硬。

## 2. 结论与建议

1. **采用 ALM**（penalty 曲率 + Lagrange 乘子），这正是解决"纯 constraint 慢"
   的手段；独立模块 `mbis_alm` 已验证。
2. **k 按响应 arm**：`k=κ/χ`（κ≈1–10）；κ 越大外层越快、内层越硬，需折中。
   **不要用固定默认 k**（此前 200 步负结果即此）。
3. **damping** `1/(1+k|χ|)` 用于步内势迭代，防 2-周期。
4. 纯 penalty 只适合"先粗收敛再切 ALM"的预条件阶段，不能作为最终约束。
5. 未收敛时力/应力用 `μ_eff=μ+k(Q−t)` 并报告 `Q−t`；收敛后用物理乘子。

## 3. 复现

```bash
g++ -std=c++14 -O2 -I source tools/mbis_alm_compare.cpp \
    source/source_estate/module_mbis_alm/mbis_alm.cpp -o /tmp/mbis_alm_compare
/tmp/mbis_alm_compare
```

## 4. 下一步

- 把线性模型换成**真实/半真实密度**（用 `mbis_alm` 的网格接口 + 简化 KS 响应）做对比；
- 写薄接口层接 ESolver（核心模块保持解耦）；
- 在 Fe2/LiCl 上验证 κ 的选取范围。
