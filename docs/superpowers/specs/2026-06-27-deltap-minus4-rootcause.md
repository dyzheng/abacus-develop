# -4 因子根因确认：snap vs center2_orb11 数值差异

> **日期**: 2026-06-27
> **方法**: 逐 link 对比 det(O_j) 从 berry_phase (det_berryphase) vs DeltaP (compute_S_dk_link)

---

## 1. 测试方法

在 BaTiO3 ref (10×10×10, symmetry=-1) 上同时运行 berry_phase 和 DeltaP：
- berry_phase: 在 `stringPhase` 中输出每个 link 的 det(O_j)（LCAO 分支）
- DeltaP: 在 O_kpair 循环中输出每个 link 的 det(O_j)（zgetrf on O_full）

对比 string 0 的 10 个 link 的 det 值。

## 2. 结果

| link | berry det | dp det | berry \|det\| | dp \|det\| | arg diff |
|------|-----------|--------|-------------|-----------|----------|
| 0 | (-0.049, 1.000) | (-0.274, 0.955) | 1.002 | 0.994 | +0.231 |
| 1 | (0.987, -0.204) | (0.840, 0.459) | 1.008 | 0.957 | +0.704 |
| 2 | (-0.168, 1.000) | (0.378, -0.469) | 1.014 | 0.602 | -2.630 |
| 3 | (0.087, -1.012) | (0.974, 0.036) | 1.015 | 0.974 | +1.521 |
| 4 | (-0.694, -0.737) | (0.531, -0.834) | 1.012 | 0.989 | +1.323 |
| 5 | (-0.929, 0.400) | (-0.060, 0.971) | 1.012 | 0.973 | -1.102 |
| 6 | (-0.181, -0.991) | (-0.562, 0.216) | 1.008 | 0.602 | -1.757 |
| 7 | (-0.305, 0.952) | (0.841, 0.449) | 1.000 | 0.953 | -1.390 |
| 8 | (0.967, -0.230) | (-0.886, -0.438) | 0.994 | 0.989 | -2.449 |
| 9 | (-0.994, -0.045) | (-0.326, -0.065) | 0.995 | 0.332 | +0.152 |

**zeta (乘积)**:
- berry: arg = -1.124, |zeta| = 1.062
- dp: arg = -0.238, |zeta| = 0.100

## 3. 分析

### 3.1 核心发现

**berry |det| ≈ 1.0**（O_j 接近酉矩阵，正确）
**dp |det| = 0.33-0.99**（O_j 远非酉，错误）

DeltaP 的 O_j 矩阵不是酉的，意味着 `compute_S_dk_link` 计算的 S_dk 矩阵
与 berry_phase 的 midmatrix 不同。

### 3.2 差异来源

| 组件 | berry_phase | DeltaP | 差异 |
|------|-------------|--------|------|
| ⟨φ\|φ(R)⟩ | center2_orb11.cal_overlap | snap (TwoCenterIntegrator) | **数值不同** |
| ⟨φ\|r'\|φ(R)⟩ | center2_orb21_r.cal_overlap | get_psi_r_psi (cal_r_overlap_R) | **可能不同** |
| 相位 | 2π(k_R·R - dk·τ) | 相同 | ✅ |
| 位置修正 | -i·dk·tpiba·r_local | 相同 | ✅ |

**根因**: `snap` 和 `center2_orb11` 使用不同的数值方法计算同一个二中心积分，
给出不同的结果。这导致 S_dk ≠ midmatrix，O_j ≠ O_j^berry。

### 3.3 -4 因子解释

-4 因子不是简单的约定差异，而是**数值误差的累积**：
- 每个 link 的 det 偏差不同（arg diff 从 +0.23 到 -2.63）
- 10 个 link 的 det 乘积偏差累积
- 最终 zeta 的 arg 比例 = -0.238 / -1.124 = 0.212 ≈ 1/4.7

加上 3% 的 unwrap 差异，最终 P ratio = 0.966（通过经验 prefactor 补偿）。

### 3.4 之前 0.97 ratio 的解释

之前的 P_elec ratio = 0.97 不是"正确"的结果，而是**数值误差恰好接近 1**。
当使用不同的 k-mesh 或体系时，ratio 可能显著偏离 0.97。

**这意味着当前的 DeltaP 实现在定量上是不可靠的。**

## 4. 修复方案

### 4.1 必须修复：使 O_j 与 berry_phase 一致

**方案 1: 修复 berryphase_overlap gathering bug**

为 occBands×occBands 输出创建专门的 ScaLAPACK 描述符：
```cpp
int desc_occ[9];
// 创建描述符 for occBands × occBands matrix
```

**方案 2: 使 snap 与 center2_orb11 一致**

对比同一对轨道的 snap 和 cal_overlap 返回值，找到差异来源：
- m 量子数约定
- 球谐函数相位
- 径向函数插值

**方案 3: 直接使用 det_berryphase 的 det 作为 zeta，不做特征值分解**

放弃逐原子分解，只输出总量。Z* 从总量差分计算。

### 4.2 推荐路径

1. **立即**: 对比 snap vs cal_overlap 的逐元素值，确定差异来源
2. **短期**: 修复 gathering bug 或使 snap 一致
3. **验证**: 重跑 per-link det 对比，确认 |det| ≈ 1.0
4. **最终**: P_elec ratio 应 ≈ 1.0（无经验 prefactor）

## 5. 当前代码状态

当前 prefactor = $-a/(4\pi\Omega)$ 是**经验补偿**，不是理论推导。
- 补偿了 snap vs center2_orb11 的数值差异（约 -4 因子）
- 补偿了 unwrap 差异（约 3%）
- 在 BaTiO3 ref 上给出 ratio = 0.966
- 在其他结构/体系上**不保证正确**
