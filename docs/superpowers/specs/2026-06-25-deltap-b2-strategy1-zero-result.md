# B-2解析P^I实验结果与H^λ分析修正

> **日期**: 2026-06-25  
> **关键发现**: 策略1（仅用解析∂_k S，忽略∂_k C）给出P=0

---

## 1. 实验结果

用解析Berry connection（策略1：∂_k D_I ≈ Σ conj(∂_k S)·C，忽略S*·∂_k C）运行BaTiO3：

| 案例 | berry Pz | deltap Pz (解析策略1) |
|------|---------|---------------------|
| 平衡 | 5.102e-4 | **0.000** |
| Ti+ | 6.600e-4 | **0.000** |
| Ti- | 3.600e-4 | **0.000** |

**DeltaP Pz = 0 对所有结构** — 策略1完全无法计算极化。

## 2. 数学推导：为什么策略1给出P=0

Berry connection分解：

$$A^I_n = \underbrace{(\sum_\mu C^*_\mu \cdot \partial_k S_\mu)}_{a} \cdot \underbrace{D_I}_{b} + \underbrace{D^*_I}_{b^*} \cdot \underbrace{\partial_k D_I}_{c}$$

策略1的解析 ∂_k D_I：

$$c = \sum_\mu \text{conj}(\partial_k S_\mu) \cdot C_\mu = \text{conj}\left(\sum_\mu C^*_\mu \cdot \partial_k S_\mu\right) = \text{conj}(a)$$

因此：

$$A^I_n = a \cdot b + b^* \cdot \text{conj}(a) = a \cdot b + \text{conj}(a \cdot b) = 2\,\text{Re}(a \cdot b)$$

**A^I_n 是实数**，Im(A^I_n) = 0，故 P^I = 0。

## 3. 物理解释

Berry phase的虚部（决定极化的部分）完全来自**波函数的k导数** ∂_k C：

$$\text{Im}(A^I_n) = \text{Im}\left(D^*_I \cdot \sum_\mu S^*_\mu \cdot \partial_k C_\mu\right)$$

SMO几何贡献（∂_k S · C）给出A的实部（= ∂_k|D_I|²/2），对Berry phase无贡献。

**结论**：P^I的计算**必须**包含∂_k C（波函数k导数），不能仅用解析∂_k S。

## 4. H^λ分析修正

### 原分析的错误

原H^λ分析认为"Part A（预存储部分）= ∂_k Fourier(H^pre)"可以低成本实现。但Part A对应的是A的实部（= 0的P），对P^I的计算没有贡献。

### 修正的H^λ结构

P^I的实际贡献项：

$$P^I \propto \sum_{n,j} \text{Im}\left[D^*_I \cdot \sum_\mu S^*_\mu \cdot \partial_k C_{n\mu}\right]$$

H^λ = ∂P^I/∂C*：

$$\frac{\partial P^I}{\partial C^*_{n\nu}(k_j)} \propto \text{Im}\left[S_\nu \cdot \sum_{\mu'} S^*_{\mu'} \cdot \partial_k C_{n\mu'}\right] + \text{(terms from } \partial_k C \text{ at neighboring k)}$$

**关键问题**：H^λ包含∂_k C，而∂_k C用有限差分时涉及相邻k点的C，使H^λ在k空间非局域。

### 三种处理∂_k C的方案

| 方案 | ∂_k C来源 | H^λ代价 | P^I平滑性 |
|------|----------|---------|----------|
| 有限差分 | [C(k+dk)-C(k-dk)]/(2dk) | k空间非局域，需规范固定 | 非平滑（规范敏感） |
| 子空间响应 | 内循环中微扰论估计 | 中等，复用DeltaSpin子空间 | 平滑（解析响应） |
| Sternheimer | (H-ε)\|∂_k ψ⟩ = -(∂_k H)\|ψ⟩ | 高，需新实现 | 精确，平滑 |

## 5. 修正的实现路径

### 推荐方案：子空间响应（策略2）

在内循环中，∂_k C通过**子空间微扰**估计：

1. 在收敛的SCF波函数上构建子空间Hamiltonian H_sub(k)
2. λ变化时，H_sub(k) → H_sub(k) + δH^λ(k)
3. ∂_k C ≈ 子空间本征矢的k导数（通过有限差分在子空间内计算）
4. 子空间内的有限差分是平滑且规范无关的（子空间本征矢可对角化）

**关键优势**：子空间本征矢的k导数在子空间内是良定义的（通过对角化子空间Hamiltonian），不需要跨k点的规范固定。

### 实现步骤

```
B-1: H^λ = 2πi Σ R_α e^{2πikR} H^pre(R) · λ   ← 已推导，~50行
B-2: P^I用子空间∂_k C                            ← 需实现子空间k导数
B-3: 内循环（复用DeltaSpin CG框架）                ← 框架复用
B-4: Z*平滑性验证                                 ← 单元测试+积分测试
```

B-2的子空间k导数需要：
1. 构建子空间Hamiltonian H_sub(k) = C†(k)·H(k)·C(k)（已有，DeltaSpin的sub_h_save）
2. 对角化H_sub(k)得到本征值ε_n(k)和本征矢V_n(k)
3. ∂_k C_{nμ} = Σ_m V_{m,n}(k) · ∂_k U_{μ,m}(k)（U是子空间基矢）
4. ∂_k U通过有限差分在子空间内计算（子空间内规范可对角化固定）

这与DeltaSpin的 `SubspaceDiagonalizer` 和 `FirstOrderResponseEngine` 直接相关，可以复用。
