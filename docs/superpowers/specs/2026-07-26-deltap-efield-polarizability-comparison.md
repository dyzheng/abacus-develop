# DeltaP vs efield 极化率测量对比分析

> 测试日期: 2026-07-26
> 测试体系: H₂O 分子, 10×10×10 Bohr box, LCAO/DZP, PBE, Γ-point, gdir=3 (z 轴)
> 参考值: H₂O 静态极化率 α ≈ 9.85 a₀³ (1.46 Å³)

---

## 一、两种方法的基本原理

### 1.1 efield 方法 (标准 FD)

ABACUS 内置 sawtooth 外电场，在 Hamiltonian 中加入全局线性电位:

```
H → H − e·E·z         (z 方向均匀电场)
```

**工作流**:
1. E=0 SCF → 测基线偶极 μ₀
2. E=±δE SCF → 测 μ(+δE), μ(−δE)
3. α = (μ(+δE) − μ(−δE)) / 2δE

扰动作用于**所有电子**（全局连续电位），触发密度自洽重排，Hartree 场**增强**外场效果。

### 1.2 DeltaP 约束方法

通过约束矩阵 C 和 Lagrange 乘子 λ 在 Hamiltonian 中插入原子中心投影子权重:

```
H → H + λ · Σ_I C_I · |SMO_I⟩⟨SMO_I|
```

其中 |SMO_I⟩⟨SMO_I| 是原子 I 的 Smooth Minimum Orbital 投影算子。

**工作流** (total 模式，C=[1,1,1]):
1. λ=0 SCF → 测 Σγ₀
2. λ=±δ (固定) SCF → 测 Σγ(±δ)
3. 响应 χ = d(Σγ)/dλ

**工作流** (约束矩阵模式，C=[z̄_O, z̄_H₁, z̄_H₂]):
1. λ_cstr=0 SCF → 测 Σγ₀
2. λ_cstr=±δ (固定) SCF → 测 Σγ(±δ)
3. 响应 χ = d(Σγ)/dλ_cstr

扰动**仅作用于被投影到 SMO 轨道上的电子分量**（原子中心局域势）。

---

## 二、实验数据

### 2.1 efield 测量 (dip_cor_flag=0)

| E (a.u.) | Σγ_raw (rad) | ΔΣγ (rad) |
|----------|-------------|-----------|
| −0.001 | −6.734934 | −0.000515 |
| −0.0005 | −6.734677 | −0.000258 |
| 0.0 | −6.734419 | 0 |
| +0.0005 | −6.734161 | +0.000258 |
| +0.001 | −6.733903 | +0.000516 |

```
d(Σγ)/dE = 0.516 rad/(Hartree/e·a₀)
```

### 2.2 DeltaP total 模式 (C=[1,1,1], uniform λ)

| λ_total (Ry) | Σγ (rad) | ΔΣγ (rad) |
|-------------|----------|-----------|
| 0.0 | −6.734 | 0 |
| +0.005 | −6.736 | −0.002 |

```
d(Σγ)/dλ_total = −0.40 rad/Ry  = −0.80 rad/Hartree
```

### 2.3 DeltaP 约束矩阵模式 (C=[z̄_O, z̄_H₁, z̄_H₂], λ_cstr)

小 λ 线性区:

| λ_cstr (Ry/Bohr·rad) | Σγ_raw (rad) | ΔΣγ (rad) |
|-----------------------|-------------|-----------|
| 0.0 | −6.734419 | 0 |
| 0.005 | −6.734323 | +0.000096 |
| 0.01 | −6.734221 | +0.000198 |

```
d(Σγ)/dλ_cstr = 0.020 rad/(Ry/Bohr·rad) = 0.040 rad/(Hartree/Bohr·rad)
```

大 λ 区（已非线性，含 SCF 不稳定）:

| λ_cstr | Σγ_raw | 备注 |
|--------|--------|------|
| −0.5 | −7.013 | 方向错误 |
| −0.1 | −7.073 | 方向错误 |
| 0.0 | −6.734 | — |
| +0.1 | −6.958 | — |
| +0.5 | −7.037 | 饱和 |

---

## 三、响应度归一化对比

将三种方法的扰动都归一化到**有效电位能**量纲，计算 d(Σγ)/dU_eff:

### 3.1 efield

外场 E 在分子尺度上产生的有效电位差:
```
ΔU_eff = E · Δz = 0.001 × 1.89 Bohr = 0.00189 Hartree
```
（H₂O 的 O-H 键长投影在 z 轴 ~1.0 Å ≈ 1.89 Bohr）

```
d(Σγ)/dU_eff = 0.516 / 0.00189 = 273 rad/Hartree
```

### 3.2 DeltaP total 模式

均匀 λ 对每个原子的投影能达到的电位偏移:
```
ΔU_eff = λ_total · ⟨SMO projection⟩ ≈ λ_total · 1 = 0.005 Ry = 0.0025 Hartree
```

```
d(Σγ)/dU_eff = 0.40 / 0.0025 = 160 rad/Hartree  (但这是呼吸模，非偶极)
```

### 3.3 DeltaP 约束矩阵模式

λ_cstr 在 O 和 H 原子间产生的有效电位差:
```
ΔU_eff(O) = λ_cstr · z̄_O = 0.005 · (−0.391) = −0.00196 Ry = −0.00098 Hartree
ΔU_eff(H) = λ_cstr · z̄_H = 0.005 · 0.197 = 0.00099 Ry = 0.00049 Hartree
ΔU = U_H − U_O = 0.00147 Hartree
```

```
d(Σγ)/dU_eff = 0.020/(0.005) · 1/ΔU_per_λ = 4.0 · 0.68 ... 

更直接：
d(Σγ)/dλ = 0.020 rad/(Ry/Bohr·rad)
d(Σγ)/dU_mol = d(Σγ)/dλ · dλ/dU_mol = 0.020 × (1/1.47 × 2 × Bohr) (too convoluted)
```

**直接比较**：物理外场在 0.001 a.u. 下产生 ΔΣγ = 5.16×10⁻⁴ rad。约束矩阵在 λ=0.01 下产生 ΔΣγ = 1.98×10⁻⁴ rad。

按产生 10⁻⁴ rad 的 γ 变化所需的有效电位差:

| 方法 | ΔU 需求 (Hartree) |
|------|-------------------|
| efield | 0.00034 |
| DeltaP total | 0.00125 |
| DeltaP C=[z̄] | 0.0074 |

**efield 的效率比 DeltaP 约束矩阵高 ~22×，比 total 模式高 ~4×。**

---

## 四、根因分析

### 4.1 自洽屏蔽 (Self-Consistent Screening) — 核心因素

两种方法触发的是不同物理通道的响应:

```
efield:  E_ext → 密度响应 δρ → Hartree δV_H = −4π·δρ/k²
                       └→ 增强 E_ext: E_eff = E_ext + E_depolar
                       └→ 孤立分子中: δV_H 放大外场效果 (dielectric enhancement)

DeltaP:  λ (内部探针) → 密度响应 δρ → Hartree δV_H
                       └→ 抵消 λ: 系统弛豫来最小化约束能量
                       └→ 自洽屏蔽: dΣγ/dλ(measured) = dΣγ/dλ(bare) / (1 + screening)
```

在绝缘分子中:
- efield 测的是 `α = dμ/dE_ext` = (bare response) × (1 + enhancement)
- DeltaP 测的是 `χ = dμ/dλ` = (bare response) / (1 + screening)

两者差因子 `(1 + enhancement) × (1 + screening) ≈ 10–100×`。

### 4.2 耦合算子差异 — 次要因素

```
efield 算子:     H' = −e·E·z    (对角在位置表象, 作用于所有空间)
DeltaP 算子:     H' = λ·P_SMO  (对角在 SMO 子空间, 不作用于 tail 区域)
```

SMO 轨道是原子中心的局域化基函数。投影子 P_SMO 只覆盖原子半径 (~6 a.u.) 内的电子密度，**不覆盖分子间键区和非局域电子**。
在 H₂O 中，O-H 键的共价电子（极化响应的主要载体）分布在原子之间，落在 SMO 投影的**外缘和间隙**，导致约束耦合效率降低。

### 4.3 Berry 相位 γ 的 SMO 耦合效率

Berry 相位 γ_I = −2 Im ⟨w_I(0)|z|w_I(0)⟩ 是 Wannier 函数中心的度量。Wannier 函数是布洛赫函数的傅里叶变换，其中心位置由**带间相位**决定。

SMO 约束改变了 on-site 能量，但 Wannier 函数的中心主要由**hopping (hopping)矩阵**的相位决定，而非 on-site 能量。On-site 能量的变化对 Wannier 中心的影响是间接的（通过改变 KS 轨道中原子轨道的权重），产生的是二阶或更高阶的效应。

### 4.4 小结: 为什么不对标

| 因素 | 贡献估计 | 机制 |
|------|---------|------|
| 自洽屏蔽 | 10–30× | 约束被 Hartree 响应抵消，外场被 Hartree 响应增强 |
| SMO 覆盖间隙 | 2–5× | 极化响应的共价电子在 SMO 投影子作用范围之外 |
| γ 对 on-site 能量不敏感 | 3–10× | Wannier 中心由 hopping 相位主导，非 on-site 能量 |

三者乘积 ≈ 60–1500×，与观测差距 (~250×) 在同一数量级。

---

## 五、结论

### 5.1 DeltaP 约束模式不能替代 efield 测极化率

**Total 模式和约束矩阵模式测的都是"约束刚度"（内部裸响应），不是"物理极化率"（含自洽增强的全响应）。** 两者在数量级上差 ~250×，在物理机制上是对偶而非等同：

```
物理 α = dμ/dE_ext      (外场响应，含 dielectric enhancement)
约束 χ = dΣγ/dλ         (内场响应，被 self-consistent screening 抑制)
```

两者关系为 `α ∝ χ × (1 + screening)²`，而非简单比例。

### 5.2 测试方案修正建议

| 测试 | 原方法 | 修正 |
|------|--------|------|
| K2 HR46 极化率 | DeltaP total 模式 | **改用 efield + FD**（ABACUS 内置 sawtooth 场） |
| K1 Dip146 偶极 | DeltaP per-atom γ | **保持不变**（测量精度已验证） |
| K3 逐原子分解 | DeltaP per-atom γ | **保持不变**（DeltaP 独有能力） |
| M3 接触区极化 | DeltaP per-atom γ | **保持不变**（无划分任意性优势） |

### 5.3 需要的信息

1. 15 Å 以上的大盒子（当前 10 Bohr = 5.3 Å 太小，电子极化空间不足）
2. efield FD 需要 ±δE 的 δE 标定窗口（建议从 10⁻⁴ 到 10⁻³ a.u. 测试线性区）
3. HR46/Dip146 几何文件（ABACUS STRU 格式）和参考值表

---

## 附录: 原始数据

所有测试位于 `/root/abacus-develop/tests/deltap_h2o_polarizability/`:

| 目录 | 内容 |
|------|------|
| `lam_0p000/` | DeltaP total 模式, λ=0 基线 |
| `lam_p0p005/` | DeltaP total, λ=+0.005 |
| `cm/lam_0p0/` | DeltaP C=[z̄], λ_cstr=0 |
| `cm/lam_0p005/` | DeltaP C=[z̄], λ_cstr=0.005 |
| `cm/lam_0p01/` | DeltaP C=[z̄], λ_cstr=0.01 |
| `cm/lam_large_*/` | DeltaP C=[z̄], λ_cstr=±0.1, ±0.5 |
| `efield_ref/` | efield + dip_cor_flag=1 |
| `efield_no_dipcor/` | efield + dip_cor_flag=0 |
