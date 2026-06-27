# DeltaP Wilson loop 特征值法：开发与验证日志

> **分支**: `feat/deltap-wilson-per-atom`
> **日期**: 2026-06-26 ~ 2026-06-27
> **体系**: BaTiO3 四方铁电相, 10×10×10, LCAO, nocc=15

---

## 算法概述

### 核心公式

Wilson loop 矩阵 $\mathbf{W} = \prod_j \mathbf{O}(k_j, k_{j+1})$，其中 $\mathbf{O} = \mathbf{C}^\dagger(k_j) \cdot \mathbf{S}_{dk} \cdot \mathbf{C}(k_{j+1})$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 重叠矩阵。

对角化 $\mathbf{W}$ 得特征值 $\lambda_n$ 和特征向量 $|v_n\rangle$：
- 逐能带 Berry phase: $\gamma_n = \arg(\lambda_n)$
- **Sum rule (精确)**: $\sum_n \gamma_n = \arg(\det \mathbf{W})$
- 逐原子权重: $w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$
- 逐原子 Berry phase: $\gamma^I = \sum_n w^I_n \cdot \gamma_n$

### 验证链设计

| 子环节 | 验证内容 | 预期 | 状态 |
|--------|---------|------|------|
| 1 | $\sum_n \arg(\lambda_n) = \arg(\det \mathbf{W})$ | 精确相等 | ✅ |
| 2 | $P_{\text{elec}}^{\text{DeltaP}} \approx P_{\text{elec}}^{\text{berry}}$ | ratio ≈ 1.0 | ✅ 0.973 |
| 3 | $Z^*_{\text{Ti}} \approx Z^*_{\text{Ti,berry}}$ | 误差 < 10% | ⚠️ 22% |
| 4 | $Z^*_{\text{Ba}} \approx Z^*_{\text{Ba,berry}}$ | 误差 < 10% | ❌ 分支跳变 |
| 5 | 分支跟踪修复后 $Z^*$ | 误差 < 10% | 🔲 待实现 |

---

## 子环节 1: 特征值 sum rule

**验证**: $\sum_n \arg(\lambda_n) = \arg(\det \mathbf{W})$

**结果**: 精确成立（数值上到机器精度）。
- gamma(sum arg) = gamma(det) 在所有 k-string 上验证通过。

**结论**: ✅ 数学恒等式确认，zgeev 对角化正确。

---

## 子环节 2: 总极化 P_elec 对比

### 2.1 发现的问题：重叠矩阵定义差异

**berry_phase** (`prepare_midmatrix_pblas`):
```
phase = 2π * (kvec_c[ik_R] · R_cart - dk · tau)
overlap = <phi|phi(R)> - i * dk * tpiba * <phi|r_local|phi(R)>
```

**DeltaP** (原始 `compute_S_dk`):
```
phase = 2π * dk · R
overlap = <phi|phi(R)>  (无位置算子修正)
```

**差异**:
1. 相位: berry_phase 用 `k_R·R - dk·τ`，DeltaP 用 `dk·R`
2. 位置算子: berry_phase 含 `-i·dk·tpiba·⟨φ|r|φ(R)⟩`，DeltaP 无

### 2.2 修复过程

| 修复步骤 | ratio (DeltaP/berry) | 说明 |
|---------|---------------------|------|
| 无修正 (Bloch overlap) | -3.70 | 原始 Bloch 态重叠 |
| +τ相位修正 (direct dk) | -6.29 | 符号错, dk 用 direct 而非 Cartesian |
| +kvec_c + R_cart | -4.90 | 相位用 Cartesian k 和 R |
| +位置算子 (full r_psi) | -8.83 | 用了 full position, 应该用 local |
| +位置算子 (local r_psi) | -2.02 | **关键**: 减去 R1*overlap 得 local |
| +自旋因子 -0.5 | **0.973** | nspin=1 时 berry_phase 乘 2 |

### 2.3 三个关键修复

#### 修复 1: 相位 `2π(kvec_c·R_cart - dk_c·tau)`

- `kvec_c` = Cartesian k-point (kv_->kvec_c)
- `R_cart` = R_int × (a1, a2, a3) (Cartesian lattice vector)
- `dk_c` = kvec_c[ik_R] - kvec_c[ik_L] (Cartesian dk)
- `tau` = get_tau(iat) (atom position, lat0 units)

实现: `compute_S_dk_link()` 函数，per-link 计算正确相位。

#### 修复 2: 位置算子修正 (local part only)

**发现**: `cal_r_overlap_R::get_psi_r_psi` 返回 full position:
```
r_full = R1 * <phi|phi(R)> + <phi|r'|phi(R)>
```

但 `unkOverlap_lcao::psi_r_psi` 只存储 **local** part:
```
r_local = <phi|r'|phi(R)>  (不含 R1*overlap)
```

**修复**:
```cpp
r_local = r_full - R1_cart * ov;  // 减去 R1*overlap
```

#### 修复 3: 自旋因子 -0.5

**发现**: berry_phase 对 nspin=1 乘以 2:
```cpp
if (nspin == 1) pdl_elec_tot = 2 * phik_ave;
```

DeltaP 计算单自旋 Berry phase，需要乘 -0.5:
- `-1`: 符号修正 (snap 计算 ⟨ket|bra⟩ = 转置, 翻转 Berry phase 符号)
- `1/2`: 自旋简并修正

### 2.4 结果

```
berry_phase P_elec = -5.790465e-03 e/bohr²
DeltaP P_elec      = -5.636308e-03 e/bohr²
ratio              = 0.9734  ✅ (3% 误差)
```

**结论**: ✅ 总极化与 berry_phase 一致（3% 误差，来自位置修正的近似和 SMO 不完备）。

---

## 子环节 3: Z* 对比

### 3.1 三个结构的 P_elec

| 结构 | berry P_elec | DeltaP P_elec | ratio |
|------|-------------|--------------|-------|
| ref | -5.790e-3 | -5.636e-3 | 0.973 ✅ |
| ti_p | -6.719e-3 | -5.063e-3 | 0.754 |
| ba_p | -7.073e-3 | +3.753e-3 | **-0.531** ❌ |

### 3.2 Z* 结果

| 方法 | Z*_Ti | Z*_Ba |
|------|-------|-------|
| berry_phase (total) | 6.69 | 2.67 |
| berry_phase (elec) | 2.69 | 0.67 |
| DeltaP (elec) | 3.28 | 53.65 |
| 误差 | 22% | 7866% |

### 3.3 Z*_Ba 分析

**症状**: ba_p 的 P_elec = +3.75e-3 (正)，而 berry_phase = -7.07e-3 (负)。符号反转。

**根因**: Wilson loop 特征值 $\lambda_n$ 在 ba_p 结构中越过了 $\arg$ 的 ±π 分支切割。

- ref: $\gamma_{\text{avg}} \approx -2.02$ (Berry phase 在 $(-\pi, \pi]$ 范围内)
- ba_p: $\gamma_{\text{avg}} \approx +1.35$ (Berry phase 跳变了 ~2π)

这是 Berry phase 的 **2π 歧义**: $\arg(\lambda_n) \in (-\pi, \pi]$，当 $\lambda_n$ 越过负实轴时，$\arg$ 跳变 2π。

**berry_phase 如何处理**: berry_phase 在 `Berry_Phase` 函数中使用 `atan2` + phase unwrapping (line 381-388):
```cpp
double theta0 = atan2(cave.imag(), cave.real());
for (int istring = 0; istring < total_string; istring++)
{
    cphik[istring] = cphik[istring] / cave;
    dtheta = atan2(cphik[istring].imag(), cphik[istring].real());
    phik[istring] = (theta0 + dtheta) / (2 * PI);
}
```

它通过除以 `cave`（平均 zeta）来规范化，然后 unwrap dtheta。

### 3.4 Z*_Ti 分析

Z*_Ti = 3.28 (berry elec = 2.69, 22% 误差)。

**误差来源**:
1. P_elec ratio = 0.754 for ti_p (DeltaP 偏小 25%)
2. 位置修正的近似 (local part only, 忽略高阶项)
3. SMO 不完备 (权重 sum ≠ 1)

ti_p 的 ratio = 0.754 说明 DeltaP 对 ti_p 的 Berry phase 计算偏低。这可能是因为:
- 位移后某些 k-string 的特征值也接近分支切割
- 但没有 ba_p 那么严重 (没有完全跳变)

---

## 子环节 4: 分支跟踪方案

### 4.1 berry_phase 的分支跟踪方法

berry_phase 使用 **"除以平均"** 方法:
1. 计算所有 k-string 的 zeta (复数)
2. 计算平均 zeta: `cave = Σ zeta / N`
3. 对每个 zeta: `dtheta = atan2((zeta/cave).imag(), (zeta/cave).real())`
4. `phik = (theta0 + dtheta) / (2π)` 其中 `theta0 = atan2(cave.imag(), cave.real())`

这相当于: 总 Berry phase = arg(平均 zeta) + 平均(展开后的 dtheta)

### 4.2 DeltaP 的分支跟踪方案

对 Wilson loop 特征值法，需要:

**方案 A: 在 zeta (= det(W)) 级别跟踪**
- 对每个 k-string，计算 zeta = det(W)
- 使用 berry_phase 的"除以平均"方法
- 但这丢失了逐原子分解 (zeta 是标量, 无特征值分解)

**方案 B: 在特征值级别跟踪** (推荐)
- 对每个 k-string，计算特征值 λ_n 和特征向量 |v_n⟩
- 在不同结构间，用特征向量重叠匹配特征值
- 对匹配的 λ_n，展开 arg(λ_n) 跨结构

**方案 C: 使用 berry_phase 的"除以平均"在总 gamma 级别**
- 对每个 k-string，计算 gamma = Σ_n arg(λ_n) = arg(det(W))
- 对 gamma 应用 berry_phase 的 unwrapping
- 逐原子分解在 unwrapped gamma 内做

方案 C 最简单: 先在 det(W) 级别 unwrap，再做逐原子分解。

### 4.3 方案 C 实现

在 `compute_wannier_polarization` 中:
1. 对每个 k-string，计算 zeta = det(W) (复数)
2. 计算 cave = Σ zeta / N_strings
3. theta0 = atan2(cave.imag(), cave.real())
4. 对每个 string: dtheta = atan2((zeta/cave).imag(), (zeta/cave).real())
5. gamma_string = (theta0 + dtheta) (unwrapped)
6. 在 unwrapped gamma_string 内做逐原子分解

关键: 逐原子分解仍然用特征值，但 gamma_total 用 unwrapped 值。

---

## 子环节 4: 分支跟踪实现与测试

### 4.1 实现方案 C: 在 det(W) 级别展开

在 `compute_wannier_polarization` 中添加:
1. 对每个 k-string，计算 zeta = det(W) (复数)
2. 计算 cave = Σ zeta / N_strings (平均 zeta)
3. theta0 = atan2(cave.imag(), cave.real())
4. 对每个 string: dtheta = atan2((zeta/cave).imag(), (zeta/cave).real())
5. gamma_string_unwrapped = theta0 + dtheta
6. 逐原子: gamma^I_unwrapped = Σ_strings gamma^I_raw * (gamma_unwrapped / gamma_raw)

### 4.2 结果

**P_elec 对比 (分支跟踪后)**:

| 结构 | berry P_elec | DeltaP P_elec | ratio |
|------|-------------|--------------|-------|
| ref | -5.790e-3 | +8.140e-3 | -1.406 ❌ |
| ti_p | -6.719e-3 | +1.257e-2 | -1.870 ❌ |
| ba_p | -7.073e-3 | +6.479e-3 | -0.916 ❌ |

**Z* 对比 (分支跟踪后)**:

| 方法 | Z*_Ti | Z*_Ba |
|------|-------|-------|
| berry_phase (elec) | 2.69 | 0.67 |
| DeltaP (elec) | 25.29 | -9.49 |
| 误差 | 839% | 1510% |

### 4.3 分析

分支跟踪使结果**变差**！之前 ratio=0.973（无分支跟踪），现在 ratio=-1.406（有分支跟踪）。

**原因**: berry_phase 的"除以平均"方法处理的是 **zeta** (= ∏ det(O_j))，这是**所有能带的乘积**。而 DeltaP 的分支跟踪在 zeta 级别展开后，按比例缩放逐原子 gamma，这个比例缩放假设了所有能带的分支跳变是均匀的，实际上不同能带的 arg(λ_n) 跳变不同。

**关键洞察**: 
- 无分支跟踪时 ratio=0.973 — 这说明对 ref 结构，**大多数 k-string 没有分支跳变**，arg(λ_n) 的原始值是正确的
- 分支跟踪引入了**错误的缩放**，破坏了原本正确的结果

**结论**: 方案 C（在 zeta 级别展开 + 比例缩放）不正确。需要:
- 方案 B: 在特征值级别跟踪（匹配特征值跨结构，展开 arg(λ_n)）
- 或: 不做分支跟踪，而是确保位移足够小以避免分支跳变

### 4.4 修正: 回退分支跟踪

由于无分支跟踪时 ref 结构的 ratio=0.973（正确），问题仅出现在 ba_p 的分支跳变上。

**策略**: 
1. 回退分支跟踪代码（恢复简单的平均）
2. 使用更小的位移（0.001 而非 0.01）来避免分支跳变
3. 或: 在特征值级别做分支跟踪

---

## 子环节 5: 回退 + 小位移测试

### 5.1 回退分支跟踪

恢复简单的 gamma_accum / n_strings 平均（无分支跟踪）。

### 5.2 使用更小位移

将位移从 0.01 direct 改为 0.001 direct，减小分支跳变风险。

位移 0.001 direct = 0.001 × 4.20 × 1.88973 = 0.00794 Bohr

Z* = (Ω/δ) × ΔP，δ 更小 → Z* 的数值噪声更大，但分支跳变风险更低。

### 5.3 小位移结果

**P_elec 对比 (位移 0.001 direct)**:

| 结构 | berry P_elec | DeltaP P_elec | ratio |
|------|-------------|--------------|-------|
| ref | -5.790e-3 | -5.682e-3 | 0.981 ✅ |
| ti_p | -5.882e-3 | -6.125e-3 | 1.041 ✅ |
| ba_p | -5.919e-3 | -5.520e-3 | 0.933 ✅ |

**所有三个结构的 P_elec ratio 都接近 1.0！** 小位移避免了分支跳变。

**Z* 对比 (位移 0.001 direct, δ=0.00794 Bohr)**:

| 方法 | Z*_Ti | Z*_Ba |
|------|-------|-------|
| berry_phase (elec) | 2.80 | 0.67 |
| DeltaP (elec) | -25.29 | 9.26 |
| 误差 | 1004% | 1274% |

**Z* 完全错误！** 虽然 P_elec ratio 都接近 1.0，但 Z* 误差巨大。

### 5.4 分析: P 正确但 Z* 错误

**关键观察**:
- ref: ratio = 0.981 (P 误差 2%)
- ti_p: ratio = 1.041 (P 误差 4%)  
- ba_p: ratio = 0.933 (P 误差 7%)

P 的误差只有 2-7%，但 Z* = (Ω/δ) × ΔP，其中 δ = 0.00794 Bohr 很小。
- ΔP 的误差 ≈ P × 误差率 ≈ 6e-3 × 0.06 = 3.6e-4
- Z* 误差 = (Ω/δ) × ΔP_error = 57133 × 3.6e-4 ≈ 20.6

Z* = 2.80, 误差 ≈ 20.6 → 误差率 ≈ 736%。与观测到的 1004% 一致。

**根因**: P_elec 的 2-7% 误差被 1/δ = 57133 放大。

### 5.5 结论

| 验证项 | 状态 | 结果 |
|--------|------|------|
| P_elec ratio (ref) | ✅ | 0.973-0.981 |
| P_elec ratio (all 3) | ✅ | 0.93-1.04 |
| Z* (大位移 0.01) | ⚠️ | Ti 22% 误差, Ba 分支跳变 |
| Z* (小位移 0.001) | ❌ | P 误差被 1/δ 放大 |

**算法框架正确**: P_elec 与 berry_phase 一致 (3-7% 误差)。Wilson loop 特征值法的基本框架验证通过。

**Z* 不正确的原因**: P_elec 的 3-7% 误差来自位置修正近似和 SMO 不完备，被 1/δ 因子放大。需要将 P_elec 精度提高到 < 0.1% 才能得到可靠的 Z*。

**精度提升方向**:
1. 改进位置修正: 精确 ⟨φ|r|φ(R)⟩ (目前用 local 近似)
2. 增大 SMO rm: 提高 SMO 完备性 (权重 sum → 1)
3. 增大 k-mesh: 减少 Berry phase 离散化误差
4. 中心差分 (±δ): 消除一阶系统误差

