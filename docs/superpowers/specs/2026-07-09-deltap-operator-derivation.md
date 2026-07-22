# DeltaP H^λ 算符严格推导

> **日期**: 2026-07-09
> **目标**: 从约束 DFT 变分原理出发，严格推导 H^λ 算符的矩阵表示
> **前序**: `DeltaP_Incremental_Design.md` B.1（初步公式）, `DeltaSpin_vs_DeltaP_Essential_Difficulties.md`（困难分析）

---

## 1. 约束泛函

约束 DFT 的能量泛函：

$$E[\psi, \lambda] = E_{\text{KS}}[\psi] + \sum_I \lambda^I \left(\gamma^I[\psi] - \gamma^I_{\text{target}}\right)$$

其中 $\gamma^I[\psi]$ 是逐原子 Berry phase（约束变量），$\lambda^I$ 是 Lagrange 乘子。

Kohn-Sham 方程：

$$\left(H_{\text{KS}} + H^\lambda\right) |\psi_{n,k}\rangle = \varepsilon_{n,k} |\psi_{n,k}\rangle$$

其中约束算符定义为：

$$\langle \phi_{\alpha,k} | H^\lambda | \psi_{n,k} \rangle = \sum_I \lambda^I \frac{\delta \gamma^I}{\delta c^*_{n,\alpha}(k)}$$

$c_{n,\alpha}(k)$ 是 LCAO 展开系数：$|\psi_{n,k}\rangle = \sum_\alpha c_{n,\alpha}(k) |\phi_{\alpha,k}\rangle$。

---

## 2. 约束变量的选择

### 2.1 精确 Berry phase（Wilson loop 特征值分解）

$$\gamma^I = \sum_m w^I_m \arg(\lambda_m)$$

其中 $\lambda_m$ 是 Wilson loop 矩阵 $\mathbf{W} = \prod_{j=0}^{N_k-1} \mathbf{O}(k_j, k_{j+1})$ 的特征值，$w^I_m = \sum_{a \in I} |\langle v_m | \alpha_a(k_0) \rangle|^2$ 是 SMO 权重，$v_m$ 是 $\mathbf{W}$ 的特征向量。

### 2.2 Berry connection（一阶近似）

$$\gamma^I_{\text{conn}} = \sum_j \sum_n w^I_n(k_j) \, \text{Im}\left[O_{nn}(k_j, k_{j+1})\right]$$

其中 $O_{mn}(k_j, k_{j+1}) = \langle u_{m,k_j} | u_{n,k_{j+1}} \rangle = \sum_{\alpha,\beta} c^*_{m,\alpha}(k_j) S_{\alpha\beta}(k_j, k_{j+1}) c_{n,\beta}(k_{j+1})$。

---

## 3. Berry connection 算符推导

### 3.1 泛函导数

对 $c^*_{p,\alpha}(k_j)$ 求导。$c^*_{p,\alpha}(k_j)$ 出现在 $O(k_j, k_{j+1})$ 中：

$$\frac{\partial O_{mn}(k_j, k_{j+1})}{\partial c^*_{p,\alpha}(k_j)} = \delta_{mp} \left[\mathbf{S}(k_j, k_{j+1}) \mathbf{C}(k_{j+1})\right]_{\alpha, n}$$

因此：

$$\frac{\delta \gamma^I_{\text{conn}}}{\delta c^*_{p,\alpha}(k_j)} = w^I_p(k_j) \cdot \frac{\partial}{\partial c^*_{p,\alpha}(k_j)} \text{Im}\left[O_{pp}(k_j, k_{j+1})\right]$$

利用 Wirtinger 微积分 $\frac{\partial \text{Im}(z)}{\partial z^*} = \frac{i}{2}$：

$$\frac{\delta \gamma^I_{\text{conn}}}{\delta c^*_{p,\alpha}(k_j)} = w^I_p(k_j) \cdot \frac{i}{2} \left[\mathbf{S}(k_j, k_{j+1}) \mathbf{C}(k_{j+1})\right]_{\alpha, p}$$

### 3.2 算符的 k 空间表示

$$\langle \phi_{\alpha,k_j} | H^\lambda | \psi_{p,k_j} \rangle = \sum_I \lambda^I w^I_p(k_j) \cdot \frac{i}{2} \sum_\beta S_{\alpha\beta}(k_j, k_{j+1}) c_{p,\beta}(k_{j+1})$$

即：

$$\boxed{H^\lambda |\psi_{p,k_j}\rangle = \frac{i}{2} \sum_I \lambda^I w^I_p(k_j) \cdot \mathbf{S}(k_j, k_{j+1}) |\psi_{p,k_{j+1}}\rangle}$$

**关键特征**：
- **k 依赖**：算符连接相邻 k 点 $k_j$ 和 $k_{j+1}$
- **能带依赖**：权重 $w^I_p(k_j)$ 依赖于能带指标 $p$
- **波函数依赖**：算符作用于 $|\psi_{p,k_{j+1}}\rangle$（下一个 k 点的波函数）

### 3.3 实空间表示

在 LCAO 基组中，重叠矩阵 $\mathbf{S}(k_j, k_{j+1})$ 的实空间分解：

$$S_{\alpha\beta}(k_j, k_{j+1}) = \sum_{\mathbf{R}} e^{i \mathbf{k}_{j+1} \cdot \mathbf{R}} \langle \phi_{\alpha,0} | \phi_{\beta,\mathbf{R}} \rangle \cdot e^{-i \Delta \mathbf{k} \cdot \boldsymbol{\tau}_\alpha}$$

其中 $\Delta \mathbf{k} = \mathbf{k}_{j+1} - \mathbf{k}_j$，$\boldsymbol{\tau}_\alpha$ 是轨道 $\alpha$ 的原子位置。

对于 k-string 上的相邻 k 点，$\Delta \mathbf{k} = \frac{\mathbf{b}_\alpha}{N_k}$（$\mathbf{b}_\alpha$ 是极化方向的倒格矢）。

算符的实空间矩阵元：

$$H^\lambda_{\alpha\beta}(\mathbf{R}) = \frac{i}{2} \sum_I \lambda^I w^I_p(k_j) \cdot e^{i \Delta \mathbf{k} \cdot (\boldsymbol{\tau}_\beta - \boldsymbol{\tau}_\alpha)} \langle \phi_{\alpha,0} | \phi_{\beta,\mathbf{R}} \rangle$$

**注意**：$w^I_p(k_j)$ 依赖于能带 $p$ 和 k 点 $k_j$，因此 $H^\lambda_{\alpha\beta}(\mathbf{R})$ 不是标准的 HContainer 矩阵元——它是**能带依赖**和 **k 点依赖**的。

---

## 4. 精确 Berry phase 算符推导

### 4.1 特征值导数

$$\gamma^I = \sum_m w^I_m \arg(\lambda_m) = \sum_m w^I_m \text{Im}(\ln \lambda_m)$$

$$\frac{\delta \gamma^I}{\delta c^*_{p,\alpha}(k_j)} = \sum_m w^I_m \cdot \text{Im}\left(\lambda_m^{-1} \frac{\delta \lambda_m}{\delta c^*_{p,\alpha}(k_j)}\right)$$

### 4.2 Wilson loop 特征值导数

Wilson loop：$\mathbf{W} = \mathbf{A}_j \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{B}_j$

其中 $\mathbf{A}_j = \prod_{l<j} \mathbf{O}(k_l, k_{l+1})$，$\mathbf{B}_j = \prod_{l>j} \mathbf{O}(k_l, k_{l+1})$。

特征值导数（非简并）：

$$\frac{\delta \lambda_m}{\delta c^*_{p,\alpha}(k_j)} = \langle v_m | \frac{\delta \mathbf{W}}{\delta c^*_{p,\alpha}(k_j)} | v_m \rangle$$

$$= \left[\mathbf{A}_j v_m\right]_p \cdot \left[\mathbf{S}(k_j, k_{j+1}) \mathbf{C}(k_{j+1}) \mathbf{B}_j v_m\right]_\alpha$$

定义传播向量：
- $|u_m(k_j)\rangle = \mathbf{A}_j |v_m\rangle$（特征向量从 $k_0$ 传播到 $k_j$）
- $|w_m(k_{j+1})\rangle = \mathbf{B}_j |v_m\rangle$（特征向量从 $k_0$ 传播到 $k_{j+1}$ 之后）

则：

$$\frac{\delta \lambda_m}{\delta c^*_{p,\alpha}(k_j)} = [u_m(k_j)]_p \cdot \left[\mathbf{S}(k_j, k_{j+1}) \mathbf{C}(k_{j+1}) w_m(k_{j+1})\right]_\alpha$$

### 4.3 算符的完整形式

$$\boxed{H^\lambda |\psi_{p,k_j}\rangle = \sum_I \lambda^I \sum_m w^I_m \cdot \text{Im}\left(\lambda_m^{-1} [u_m(k_j)]_p\right) \cdot \mathbf{S}(k_j, k_{j+1}) |\tilde{\psi}_{m,k_{j+1}}\rangle}$$

其中 $|\tilde{\psi}_{m,k_{j+1}}\rangle = \sum_n [w_m(k_{j+1})]_n |\psi_{n,k_{j+1}}\rangle$ 是 Wilson loop 特征向量在波函数空间的旋转。

**与 Berry connection 算符的对比**：

| 项 | Berry connection | 精确 Berry phase |
|---|---|---|
| 求和 | $\sum_I \lambda^I w^I_p$ | $\sum_I \lambda^I \sum_m w^I_m \text{Im}(\lambda_m^{-1} [u_m]_p)$ |
| k 连接 | $\mathbf{S}(k_j, k_{j+1}) |\psi_{p,k_{j+1}}\rangle$ | $\mathbf{S}(k_j, k_{j+1}) |\tilde{\psi}_{m,k_{j+1}}\rangle$ |
| 能带依赖 | 仅通过 $w^I_p$ | 通过 $w^I_m$, $[u_m]_p$, $|\tilde{\psi}_m\rangle$ |
| 波函数依赖 | 仅 $|\psi_{p,k_{j+1}}\rangle$ | $|\tilde{\psi}_{m,k_{j+1}}\rangle$（特征向量旋转） |
| 计算复杂度 | $O(N_k N_{\text{occ}} N_{\text{basis}}^2)$ | $O(N_k N_{\text{occ}}^3 + N_k N_{\text{occ}}^2 N_{\text{basis}})$ |

---

## 5. 实现策略

### 5.1 为什么不能预存储为 HContainer

DeltaSpin 的算符 $H^\lambda \propto |\alpha\rangle\langle\alpha|$ 是**零阶算符**：
- 与 k 无关
- 与能带无关
- 与波函数无关
- 可以预存储为 $H^{\text{pre},I}_{\alpha\beta}(\mathbf{R}) = \sum_{lm} \langle \phi_{\alpha,0} | \alpha^I_{lm} \rangle \langle \alpha^I_{lm} | \phi_{\beta,\mathbf{R}} \rangle$

DeltaP 的算符是**一阶算符**：
- k 依赖（连接相邻 k 点）
- 能带依赖（权重 $w^I_n$ 随能带变化）
- 波函数依赖（$|\tilde{\psi}_m\rangle$ 需要当前波函数）
- **不能**预存储为 HContainer

### 5.2 每步 SCF 的计算流程

```
每个 SCF 步:
  1. 对角化 H → {c_{n,α}(k), ε_{n,k}}
  2. 构建 Wilson loop W = ∏_j O(k_j, k_{j+1})
  3. 对角化 W → {λ_m, v_m}
  4. 计算传播向量 u_m(k_j), w_m(k_{j+1})
  5. 计算 SMO 权重 w^I_m
  6. 分支选择 → γ^I
  7. 计算 H^λ 矩阵元（k 空间）
  8. 将 H^λ 加到 HK（k 空间哈密顿量）
  9. 检查收敛：|γ^I - γ^I_target| < tol
```

### 5.3 算符的矩阵元计算

对于每个 k 点 $k_j$ 在 k-string 上：

$$H^\lambda_{\alpha\beta}(k_j) = \sum_I \lambda^I \sum_m w^I_m \cdot \text{Im}\left(\lambda_m^{-1} [u_m(k_j)]_p\right) \cdot S_{\alpha\beta}(k_j, k_{j+1}) \cdot [w_m(k_{j+1})]_n \cdot c_{n,\beta}(k_{j+1})$$

这涉及：
- Wilson loop 特征值 $\lambda_m$ 和特征向量 $v_m$（步骤 3）
- 传播向量 $u_m(k_j)$, $w_m(k_{j+1})$（步骤 4，矩阵乘法 $O(N_k N_{\text{occ}}^3)$）
- 重叠矩阵 $S_{\alpha\beta}(k_j, k_{j+1})$（已有，来自 berry_overlap）
- 波函数 $c_{n,\beta}(k_{j+1})$（步骤 1）

计算复杂度：$O(N_k N_{\text{occ}}^2 N_{\text{basis}})$，与对角化 $O(N_{\text{basis}}^3)$ 相比可忽略。

---

## 6. 与 DeltaSpin 的对比

| 属性 | DeltaSpin | DeltaP (Berry connection) | DeltaP (exact) |
|------|-----------|--------------------------|----------------|
| 算符阶数 | 零阶 $|\alpha\rangle\langle\alpha|$ | 一阶 $|\alpha\rangle\langle\alpha| \cdot S$ | 一阶 + 特征向量旋转 |
| k 依赖 | 无 | 相邻 k 点 | 相邻 k 点 |
| 能带依赖 | 无 | $w^I_n$ | $w^I_m, [u_m]_p$ |
| 波函数依赖 | 无 | $|\psi_{n,k_{j+1}}\rangle$ | $|\tilde{\psi}_{m,k_{j+1}}\rangle$ |
| 预存储 | HContainer(R) | 不可 | 不可 |
| 每步计算量 | 0（预存储） | $O(N_k N_{\text{occ}} N_{\text{basis}}^2)$ | $O(N_k N_{\text{occ}}^2 N_{\text{basis}})$ |
| 梯度方向 | 精确（算符 = 导数） | 近似（一阶） | 精确（算符 = 导数） |
| 收敛保证 | CG/Broyden 稳定 | 需验证 | 理论保证 |

---

## 7. 结论

1. **算符-变量一致性**：精确 Berry phase 算符 $H^\lambda$ 是 $\gamma^I$ 对波函数的精确泛函导数，保证了梯度下降的正确方向。Berry connection 算符是一阶近似，梯度方向可能有偏差。

2. **k 依赖是本质特征**：Berry phase 涉及 k 空间积分，其泛函导数天然连接相邻 k 点。这是与 DeltaSpin（实空间局域算符）的根本差异，不可通过预存储消除。

3. **实现路径**：每步 SCF 在 k 空间直接计算 $H^\lambda$ 矩阵元并加到 $H_K$，绕过 HContainer 框架。计算量 $O(N_k N_{\text{occ}}^2 N_{\text{basis}})$ 远小于对角化，不构成瓶颈。

4. **建议先用 Berry connection 算符验证框架**（实现简单，梯度方向近似正确），再升级到精确 Berry phase 算符（需要 Wilson loop 特征向量传播）。
