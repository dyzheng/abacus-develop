# H^λ实现代价分析：三种方案对比

> **核心问题**: 哪种P^I计算方法能让虚拟电场（Lagrange乘子λ）的哈密顿量修正H^λ以最小代价实现？

---

## 1. H^λ的统一结构

约束能量 $E_c = E_{KS} + \sum_I \lambda^I_\alpha (P^I_\alpha - P^I_{\text{target}})$ 对波函数变分：

$$\boxed{H^\lambda |\psi_{nk}\rangle = \sum_{I,\alpha} \lambda^I_\alpha \frac{\delta P^I_\alpha}{\delta \langle \psi_{nk}|}}$$

三种方法的区别在于 $P^I_\alpha$ 如何依赖 $\psi$，进而决定 $H^\lambda$ 的复杂度。

---

## 2. 方案A：精确Wilson loop

### P^I公式

$$P^I = -\frac{a_\alpha}{2\pi\Omega} \operatorname{Im}\left[\log \prod_j \det\left(\mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}^I(k_{j+1})\right)\right]$$

其中 $\mathbf{U}(k) = \mathbf{W}(k)\mathbf{V}^\dagger(k)$ 来自 $D_I(k) = \mathbf{W}\boldsymbol{\Sigma}\mathbf{V}^\dagger$ 的SVD极分解。

### H^λ依赖链

$$H^\lambda \leftarrow \frac{\partial P^I}{\partial \psi} \leftarrow \begin{cases} \frac{\partial \mathbf{U}}{\partial D_I} & \text{SVD导数} \\ \frac{\partial \mathbf{O}}{\partial \psi} & \text{重叠导数} \\ \frac{\partial \det}{\partial \mathbf{M}} & \text{Jacobi公式} \\ \frac{\partial \log}{\partial z} & \text{链式法则} \end{cases}$$

### 关键困难：SVD导数

$\mathbf{U} = \mathbf{W}\mathbf{V}^\dagger$ 对 $D_I$ 的导数（Papadopoulo & Rappaz 2018）：

$$\frac{\partial \mathbf{W}}{\partial D} = \mathbf{W} \cdot \mathbf{F}_W + \mathbf{V} \cdot \mathbf{G}_W$$

其中 $\mathbf{F}_W$ 涉及 $(\sigma_i^2 - \sigma_j^2)^{-1}$ 型项：

$$(F_W)_{ij} = \begin{cases} \frac{(\mathbf{W}^\dagger \dot{D} \mathbf{V})_{ij} \sigma_j - (\mathbf{W}^\dagger \dot{D} \mathbf{V})_{ji} \sigma_i}{\sigma_i^2 - \sigma_j^2} & \sigma_i \neq \sigma_j \\ 0 & \sigma_i = \sigma_j \end{cases}$$

**问题**：当奇异值接近时（$\sigma_i \approx \sigma_j$），$(\sigma_i^2 - \sigma_j^2)^{-1} \to \infty$，数值不稳定。

### 额外困难

- 需要 $\mathbf{O}(k_j, k_{j+1}) = \langle\psi_{k_j}|\psi_{k_{j+1}}\rangle$（跨k点重叠），在LCAO中需要 `unkOverlap_lcao`
- `arg()` 分支切割导致H^λ对结构不连续

### 代价评估

| 组件 | 难度 | 可复用性 |
|------|------|---------|
| SVD导数 ∂U/∂D | ⭐⭐⭐⭐⭐ 极高 | 无现有代码 |
| 跨k点重叠 ∂O/∂ψ | ⭐⭐⭐ 中等 | 部分复用berryphase |
| det导数 | ⭐⭐ 低 | Jacobi公式 |
| arg/log分支 | ⭐⭐⭐ 中等 | 需自行实现分支跟踪 |

**H^λ总代价：极高**。SVD导数是瓶颈，且数值不稳定。

---

## 3. 方案B：解析Berry connection（避免有限差分）

### P^I公式（解析版，无有限差分）

$$P^I = -\frac{a_\alpha}{2\pi\Omega} \Delta k \sum_{n,j} \operatorname{Im}\left[A^I_n(k_j)\right]$$

$$A^I_n = \sum_{lm} \left[\underbrace{\langle\psi|\partial_k\alpha^I_{lm}\rangle}_{\text{term1}} \cdot \underbrace{\langle\alpha^I_{lm}|\psi\rangle}_{D_I} + \underbrace{\langle\psi|\alpha^I_{lm}\rangle}_{D^*_I} \cdot \underbrace{\partial_k\langle\alpha^I_{lm}|\psi\rangle}_{\text{term2}}\right]$$

关键改进：term2用**解析导数**替代有限差分：

$$\partial_k D_I = \partial_k\left(\sum_\mu S^*_{\mu,Ilm}(k) \cdot C_{n\mu}(k)\right) = \sum_\mu \underbrace{(\partial_k S^*_{\mu})}_{\text{解析，k依赖}} \cdot C_{n\mu} + \sum_\mu S^*_{\mu} \cdot \underbrace{(\partial_k C_{n\mu})}_{\text{波函数k导数}}$$

### H^λ分解为两部分

**Part A（可预存储，与DeltaSpin同构）**：

$$H^{\lambda,\text{pre}}_{\mu\nu}(k) = \sum_{I,lm} \lambda^I_\alpha \cdot \partial_k\left[S_{\mu,Ilm}(k) \cdot S^*_{\nu,Ilm}(k)\right]$$

**关键发现**：$S_\mu(k) \cdot S^*_\nu(k)$ 是DeltaSpin HContainer $H^{\text{pre},I}_{\mu\nu}(R)$ 的Fourier变换！

$$S_\mu(k) \cdot S^*_\nu(k) = \sum_\mathbf{R} e^{2\pi i\mathbf{k}\cdot\mathbf{R}} \underbrace{\sum_{R_2} \langle\phi_\mu|\alpha(R_2+\mathbf{R})\rangle\langle\alpha(R_2)|\phi_\nu\rangle}_{H^{\text{pre},I}_{\mu\nu}(\mathbf{R}) \text{（已预存储！）}}$$

因此：

$$\boxed{H^{\lambda,\text{pre}}_{\mu\nu}(k) = 2\pi i \sum_\mathbf{R} R_\alpha \cdot e^{2\pi i\mathbf{k}\cdot\mathbf{R}} \cdot H^{\text{pre},I}_{\mu\nu}(\mathbf{R}) \cdot \lambda^I_\alpha}$$

**这只是DeltaSpin HContainer的加权重Fourier变换**——在相位求和中添加 $2\pi i R_\alpha$ 因子即可！

**Part B（不可预存储，涉及波函数k导数）**：

$$H^{\lambda,\text{dyn}}_{\mu\nu}(k) \propto \sum_{I,lm} \lambda^I_\alpha \cdot S_{\mu,Ilm}(k) \cdot S^*_{\nu,Ilm}(k) \cdot (i\partial_k C_{n\mu})$$

这涉及 $\partial_k C$（波函数k导数），即位置算符 $i\partial_k$ 在SMO子空间上的投影。

### Part B的三种处理策略

| 策略 | 方法 | 代价 | 平滑性 |
|------|------|------|--------|
| 策略1: 忽略 | 仅用Part A | ⭐ 最低 | Part A是解析的，天然平滑 |
| 策略2: 子空间响应 | 内循环中用微扰论估计∂_k C | ⭐⭐ 中等 | 平滑（解析响应） |
| 策略3: Sternheimer | $(H-\epsilon)\|\partial_k\psi\rangle = -(\partial_k H)\|\psi\rangle$ | ⭐⭐⭐ 较高 | 精确，平滑 |

**策略1（仅Part A）的物理含义**：

Part A捕获"SMO随k变化的几何贡献"——即SMO Bloch和的k导数对Berry connection的贡献。Part B捕获"波函数随k变化的贡献"——即标准Berry phase $\langle\psi|\partial_k\psi\rangle$ 中SMO未捕获的部分。

当SMO集较完备时（$\sum P^I \approx I$），Part B的贡献较小，Part A可能已足够。

### 代价评估

| 组件 | 难度 | 可复用性 |
|------|------|---------|
| Part A: ∂_k(S·S*) = 加权Fourier | ⭐ 极低 | **完全复用DeltaSpin HContainer** |
| Part B-策略1: 忽略 | ⭐ 无 | — |
| Part B-策略2: 子空间∂_k C | ⭐⭐ 中等 | 复用DeltaSpin子空间对角化 |
| Part B-策略3: Sternheimer | ⭐⭐⭐ 较高 | 需新实现 |

**H^λ总代价：低（策略1）到中等（策略2/3）**

---

## 4. 方案C：平行传输规范

### 与方案B的关系

平行传输改变**规范固定的方式**（用 $\max\text{Re}\langle\psi_{k_j}|\psi_{k_{j+1}}\rangle$ 替代SMO锚定），但不改变Berry connection公式本身。

**H^λ完全相同**——因为H^λ是 $P^I$ 对波函数的变分导数，与规范选择无关（term1规范不变，term2解析导数也规范不变）。

### 额外代价

- 需要计算 $\langle\psi_{k_j}|\psi_{k_{j+1}}\rangle$（跨k点重叠）来平行传输——但这只在P^I计算中需要，不在H^λ中
- 跨结构（Phase B约束循环）不连续——平行传输是k连续但结构不连续

### 代价评估

| 组件 | 难度 |
|------|------|
| H^λ | 同方案B |
| 平行传输P^I计算 | ⭐⭐ 中等（需跨k点重叠） |

**H^λ总代价：同方案B**

---

## 5. 综合对比

| 维度 | 方案A (Wilson) | 方案B (解析Berry) | 方案C (平行传输) |
|------|---------------|-----------------|----------------|
| **H^λ核心难度** | SVD导数（极高） | ∂_k H^pre Fourier（极低） | 同B |
| **预存储复用** | 无法复用HContainer | **完全复用DeltaSpin** | 同B |
| **k点依赖** | 需跨k点重叠O | Part A无跨k点需求 | P^I需跨k点重叠 |
| **规范依赖** | 规范不变（SVD） | 解析→规范不变 | 规范无关 |
| **结构平滑性** | arg()分支切割 | **解析→天然平滑** | k连续，结构不连续 |
| **数值稳定性** | 奇异值接近时不稳定 | 稳定 | 稳定 |
| **最小实现代价** | 极高 | **极低（策略1）** | 低 |
| **完整实现代价** | 极高 | 中等（策略2/3） | 中等 |

---

## 6. 结论

### 方案B（解析Berry connection）以最小代价实现H^λ

**核心原因**：H^λ的可预存储部分是DeltaSpin HContainer的k导数Fourier变换：

$$H^{\lambda,\text{pre}}(k) = 2\pi i \sum_\mathbf{R} R_\alpha \cdot e^{2\pi ikR} \cdot H^{\text{pre}}(R) \cdot \lambda$$

**实现工作量**：

1. **预存储**（已有）：$H^{\text{pre},I}_{\mu\nu}(R) = \sum_{lm} \langle\phi_\mu|\alpha^I_{lm}(R_2+R)\rangle\langle\alpha^I_{lm}(R_2)|\phi_\nu\rangle$ — DeltaSpin已实现

2. **H^λ(k)计算**（新增，~50行代码）：在现有 `contributeHR()` 的Fourier求和中，添加 $2\pi i R_\alpha$ 权重因子

3. **P^I计算**（改进）：用解析 $\partial_k S$ 替代有限差分 $\partial_k D_I$，消除非平滑性

4. **Part B**（可选）：先用策略1（忽略），验证精度后按需添加策略2（子空间响应）

### 与DeltaSpin的复用度

| DeltaSpin组件 | DeltaP复用方式 |
|---------------|---------------|
| `HContainer H^{pre}` | 直接复用，H^λ = ∂_k Fourier(H^pre) |
| `cal_pre_HR()` (snap重叠) | 直接复用，相同的⟨φ\|α⟩计算 |
| `contributeHR()` (HR构建) | 修改：添加2πiR_α权重 |
| `lambda_loop.cpp` (CG优化) | 框架复用：M→P，∂P/∂λ用解析响应 |
| 子空间对角化 | 复用：∂_k C通过子空间响应估计 |

### 推荐实现路径

```
Phase B-1: 实现H^λ = ∂_k Fourier(H^pre) · λ     ← 最小代价，复用DeltaSpin
Phase B-2: 实现解析P^I（用∂_k S替代有限差分）      ← 消除非平滑性
Phase B-3: 内循环（复用DeltaSpin CG框架，M→P）     ← 框架复用
Phase B-4: 验证Z*平滑性（复用本文单元测试框架）     ← 验证
Phase B-5: （可选）添加Part B = S·∂_k C           ← 提高精度
```

**方案B策略1是H^λ最小代价实现**：仅需在DeltaSpin的Fourier求和中添加 $2\pi i R_\alpha$ 因子，约50行新代码。方案A的SVD导数代价是其数百倍。
