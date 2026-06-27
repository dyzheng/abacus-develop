# DeltaP Wilson loop 特征值法：详细推导与问题分析

> **日期**: 2026-06-27
> **目标**: 详细推导算法的每个环节，分析 P_elec 3% 误差的来源，规划与 Wannier center 的直接对比

---

## 1. SMO 第一 zeta 选择验证

### 1.1 代码实现

在 `compute_real_overlaps` (deltap_overlap.cpp:82-96):
```cpp
// Select first zeta of each l, same as DeltaSpin
int target_L = 0, index = 0;
for (int iw = 0; iw < ucell.atoms[T0].nw; iw++)
{
    const int L0 = ucell.atoms[T0].iw2l[iw];
    if (L0 == target_L)
    {
        for (int m = 0; m < 2 * L0 + 1; m++)
        {
            nlm_target[index] = nlm[0][iw + m];
            index++;
        }
        target_L++;
    }
}
```

**分析**:
- `iw2l` 按顺序排列：先 l=0 的所有 (N,m)，再 l=1 的所有 (N,m)，...
- `target_L` 从 0 开始，遇到 `L0 == target_L` 时取第一个 N（即 N=0，第一 zeta）的所有 m
- 然后递增 `target_L`，确保每个 l 只取一次
- 结果：nproj = Σ_l (2l+1) = (nwl+1)²

**结论**: ✅ SMO 正确使用第一 zeta，与 DeltaSpin 一致。

### 1.2 SMO 用途区分

| 函数 | 使用 SMO? | 用途 |
|------|----------|------|
| `compute_real_overlaps` | ✅ 第一 zeta | SMO 投影 D_I = ⟨α|ψ⟩ (逐原子权重) |
| `compute_S_dk_link` | ❌ 全部 NAO | 重叠矩阵 S(dk) = ⟨φ_μ|φ_ν(R)⟩ (Wilson loop) |
| `compute_D_I` | ✅ 第一 zeta | 从 S_k 构建 D_I (用 SMO 的 S_k) |

**关键**: S_dk 使用全部 NAO 轨道（正确），SMO 投影只用于逐原子权重（也正确）。

### 1.3 BaTiO3 参数

| 原子 | 轨道文件 | nwl | nproj_SMO | nproj_NAO |
|------|---------|-----|-----------|-----------|
| Ba | 6s3p3d2f | 3 | 16 | 36 (6+9+9+6+6) |
| Ti | 6s3p3d2f | 3 | 16 | 36 |
| O | 3s3p2d1f | 3 | 16 | 27 (3+9+10+5) |

- nproj_SMO_total = 5 × 16 = 80
- nocc = 15 (30 价电子 / 2)
- nproj_NAO_total = 36+36+27+27+27 = 153 (但并行分布后 nlocal 可能不同)

---

## 2. 逐环节推导

### 2.1 重叠矩阵 O_j

$$\mathbf{O}(k_j, k_{j+1}) = \mathbf{C}^\dagger(k_j) \cdot \mathbf{S}_{dk} \cdot \mathbf{C}(k_{j+1})$$

其中 $\mathbf{S}_{dk}$ 是 NAO 位移重叠矩阵，使用 berry_phase 约定：

$$S_{dk,\mu\nu} = \sum_\mathbf{R} e^{2\pi i (\mathbf{k}_R \cdot \mathbf{R}_{cart} - \mathbf{dk} \cdot \boldsymbol{\tau})} \cdot \left[\langle\phi_\mu|\phi_\nu(\mathbf{R})\rangle - i \cdot \mathbf{dk} \cdot \text{tpiba} \cdot \langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle\right]$$

其中：
- $\mathbf{k}_R$ = Cartesian k-point at right side (dimensionless, = kvec_d × G)
- $\mathbf{R}_{cart}$ = R × (a1, a2, a3) (lattice vector, lat0 units)
- $\mathbf{dk}$ = kvec_c_R - kvec_c_L (dimensionless)
- $\boldsymbol{\tau}$ = atom position (lat0 units)
- tpiba = 2π/lat0 (1/Bohr)
- $\langle\phi_\mu|\mathbf{r}'|\phi_\nu(\mathbf{R})\rangle$ = **local** position matrix (Bohr), 从 `get_psi_r_psi` 减去 R1×overlap

### 2.2 Wilson loop 矩阵

$$\mathbf{W} = \prod_{j=0}^{N_k-1} \mathbf{O}(k_j, k_{j+1})$$

归一化：每步除以 max|element|（实正数，保持 arg 不变）。

### 2.3 特征值分解

$$\mathbf{W} = \mathbf{V} \cdot \text{diag}(\lambda_n) \cdot \mathbf{V}^{-1}$$

- $\lambda_n$ = 特征值（复数），**规范不变**（W 的相似变换不改变特征值）
- $\gamma_n = \arg(\lambda_n) \in (-\pi, \pi]$ — 逐能带 Berry phase
- **Sum rule**: $\sum_n \gamma_n = \arg(\det \mathbf{W})$ — **精确**（对数可加性）

### 2.4 逐原子权重

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$$

其中 $\langle v_n | \alpha_a \rangle = \sum_m D_{a,m} \cdot V_{m,n}$：
- $D_{a,m} = \langle \alpha_a | \psi_m \rangle$ — SMO 投影（第一 zeta）
- $V_{m,n}$ — 特征向量矩阵的第 n 列

**Sum rule**: $\sum_I w^I_n = \langle v_n | \hat{P}_{SMO} | v_n \rangle$，当 SMO 完备时 = 1。

### 2.5 逐原子 Berry phase

$$\gamma^I = \sum_n w^I_n \cdot \arg(\lambda_n)$$

**总 Berry phase**: $\sum_I \gamma^I = \sum_n (\sum_I w^I_n) \cdot \arg(\lambda_n)$

当 SMO 完备：$= \sum_n \arg(\lambda_n) = \arg(\det \mathbf{W})$ ✓

### 2.6 极化

$$P^I = \text{prefactor} \times \gamma^I$$

prefactor = $-0.5 \times a_\alpha / (2\pi \Omega)$ for nspin=1

- $-0.5$: 自旋简并因子 (berry_phase 乘 2) × 符号 (snap 转置)
- $a_\alpha$: 晶格向量长度 (Bohr)
- $\Omega$: 原胞体积 (Bohr³)

---

## 3. P_elec 3% 误差来源分析

### 3.1 可能的误差来源

| 来源 | 影响方式 | 估计大小 |
|------|---------|---------|
| A. k-string 平均中的 2π 跳变 | 个别 string 的 arg(zeta) 跳变 2π | ~3% (1 string / 100) |
| B. 位置修正近似 | local part vs exact <φ|r|φ(R)> | ~1-2% |
| C. SMO 不完备 | Σ_I w^I_n ≠ 1 | < 1% (只影响逐原子, 不影响总量) |
| D. snap 转置 | ⟨ket|bra⟩ vs ⟨bra|ket⟩ | 改变符号, 已用 -0.5 修正 |
| E. 归一化 | 每步除以 max|element| | 数值精度, < 0.01% |

### 3.2 来源 A: k-string 平均中的 2π 跳变（最可能）

**berry_phase 的处理**:
```cpp
cave = Σ zeta_i / N
theta0 = atan2(cave.imag(), cave.real())
for each i:
    dtheta = atan2((zeta_i/cave).imag(), (zeta_i/cave).real())
    phik_i = (theta0 + dtheta) / (2π)
```

这给出了**展开后的** arg(zeta_i)，消除了 2π 跳变。

**DeltaP 的处理**:
```cpp
gamma_avg = Σ arg(zeta_i) / N
```

直接平均 arg(zeta_i)，**不处理 2π 跳变**。

**差异**: 如果某个 string 的 zeta 接近负实轴，arg 从 +π 跳到 -π，导致平均值偏移 2π/N。

**估计**: 对于 100 个 string，1 个 string 跳变 → 误差 = 2π/100 / |γ| = 0.063/2.08 ≈ 3%。**与观测一致！**

### 3.3 来源 B: 位置修正近似

**当前**: 使用 `get_psi_r_psi` 返回的 full position，减去 R1×overlap 得到 local part。

**berry_phase 的 psi_r_psi**: 在 `cal_orb_r_overlap` 中直接计算 local part（不通过 get_psi_r_psi）。

**潜在差异**: `get_psi_r_psi` 使用 `cal_r_overlap_R::center2_orb21_r`，而 `cal_orb_r_overlap` 使用 `unkOverlap_lcao::center2_orb21_r`。如果两个类的初始化不同（不同的 kmesh、dr、Rmesh），结果可能不同。

### 3.4 来源 C: SMO 不完备

**只影响逐原子分解，不影响总量**。总量 P = prefactor × Σ_n arg(λ_n) = prefactor × arg(det(W))，与 SMO 无关。

---

## 4. 与 Wannier center 的直接对比方案

### 4.1 对比目标

| 量 | DeltaP | Wannier90 | 关系 |
|---|---|---|---|
| 逐能带 Berry phase | γ_n = arg(λ_n) | γ_n^W = Im ln ∏ [U†MU]_nn | 应相等 (模 2π) |
| Wannier center | (a/2π) × γ_n | ⟨r_n⟩ (from .wout) | 直接可比 (Bohr) |
| 总电子极化 | P = prefactor × Σ γ_n | P = -e/Ω × Σ ⟨r_n⟩ | 应相等 |

### 4.2 优势

- **无 1/δ 放大**: 直接比较 P (e/bohr²) 或 ⟨r⟩ (Bohr)，不经过 Z* 差分
- **无分支跳变问题**: 对比绝对值 P (单一结构)，不需要跨结构差分
- **Wannier90 已验证**: diamond 体系 Wannier90 正常运行，有 .wout

### 4.3 Diamond 对比计划

1. 用 diamond 体系（4×4×4, nocc=4, 简单可靠）
2. 计算 DeltaP 的逐能带 γ_n = arg(λ_n)
3. 从 Wannier90 .wout 提取 Wannier center ⟨r_n⟩
4. 对比 (a/2π) × γ_n 与 ⟨r_n⟩

**注意**: Wannier90 的 Wannier center 来自 8 个 Wannier 函数（含 disentanglement），而 DeltaP 用 4 个占据能带。需要取 Wannier90 中对应占据态的部分。

### 4.4 BaTiO3 对比计划

1. 计算 DeltaP 的总 P_elec (已验证 ratio=0.97)
2. 从 .mmn 文件计算 Berry phase (已有结果)
3. 对比 DeltaP γ_total 与 .mmn γ_total
4. 如果一致，逐能带 γ_n 也可对比

---

## 5. 修复 3% 误差的方案

### 5.1 实现 berry_phase 的 "除以平均" 展开

在**总量级别**（不做逐原子缩放）：

```cpp
// 1. 收集每个 string 的 zeta = det(W)
std::vector<complex> zeta_list;

// 2. 计算平均
complex cave = sum(zeta_list) / N;

// 3. 展开
double theta0 = atan2(cave.imag(), cave.real());
double gamma_total = 0;
for (each string i):
    complex ratio = zeta_list[i] / cave;
    double dtheta = atan2(ratio.imag(), ratio.real());
    gamma_total += theta0 + dtheta;
gamma_total /= N;

// 4. 逐原子: 用 raw gamma^I 的比例
for (each atom I):
    gamma_I = gamma_total * (gamma_I_raw_sum / gamma_raw_total_sum);
```

**关键改进**: 之前的 Method C 失败是因为**逐 string 缩放**。这里改为**总量缩放**：只在最终平均值上做一次缩放，不在每个 string 上缩放。

### 5.2 验证位置修正一致性

检查 `cal_r_overlap_R` 和 `unkOverlap_lcao` 的初始化参数是否一致：
- kmesh, dr, Rmesh
- orb_r 的构造

如果不一致，需要用 `unkOverlap_lcao` 的方法重新计算位置矩阵。

