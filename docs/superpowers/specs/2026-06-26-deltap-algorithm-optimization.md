# DeltaP 算法优化方案：理论分析

> **日期**: 2026-06-26
> **目标**: 从理论上重新分析 DeltaP 逐原子极化分解的数学困难，提出可行的优化方案

---

## 1. 问题本质的精确表述

### 1.1 我们要分解什么？

总电子极化（Berry phase）：

$$\gamma = \mathrm{Im}\,\ln\prod_{j=0}^{N_k-1} \det\big[\mathbf{M}(k_j, k_{j+1})\big]$$

其中 $\mathbf{M}(k_j, k_{j+1})_{mn} = \langle u_{m,k_j} | u_{n,k_{j+1}}\rangle$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 重叠矩阵。

**目标**：找到 $\gamma^I$ 使得 $\sum_I \gamma^I = \gamma$，且 $\gamma^I$ 具有原子分辨性。

### 1.2 为什么直接分解失败？

**Berry phase 是 log-det，不可线性分解**：

$$\gamma = \mathrm{Im}\,\ln\det\mathbf{M} = \mathrm{Im}\,\mathrm{Tr}\ln\mathbf{M} \neq \mathrm{Im}\,\mathrm{Tr}\,\mathbf{M}$$

- 左边（Berry phase）：$\ln\det\mathbf{M} = \mathrm{Tr}\ln\mathbf{M}$，涉及 $\mathbf{M}$ 的**全部**矩阵元素
- 右边（Berry connection）：$\mathrm{Tr}\,\mathbf{M} = \sum_n M_{nn}$，只涉及**对角**元素

差异为高阶项 $\mathrm{Tr}(\mathbf{M}-\mathbf{I})^2/2$ 等，当 $dk$ 不小时不可忽略。

**实验验证**（BaTiO3, nocc=15, dk=0.1）：trace/det = 0.45-0.56，误差 44-56%。

### 1.3 已尝试方案的问题汇总

| 方案 | 分解对象 | sum rule | 精确性 | 失败原因 |
|------|---------|----------|--------|---------|
| 逐原子 Wilson loop (det) | $\det(\mathbf{U}^{I\dagger}\mathbf{O}\mathbf{U}^I)$ | 不成立 | 精确（若成立） | nproj>nocc → det=0 |
| 逐原子 SVD + 截断 | $\det(\mathbf{V}^{I\dagger}\mathbf{O}\mathbf{V}^I)$ | 不成立 | 精确（若成立） | V^I 不互补 → sum rule 破坏 |
| 全局 SVD + 权重 (trace) | $\sum_n w^I_n \mathrm{Im}\,M_{nn}$ | 成立 | 近似 | trace≠det |
| 混合 Wilson+trace | $\gamma_{\text{det}} \times A^I_{\text{trace}}/A_{\text{trace}}$ | 成立 | 近似 | 比例随结构变化 |

**核心矛盾**：精确的量（det）不可线性分解；可线性分解的量（trace）不精确。

---

## 2. 五种优化方案

### 方案 A：Wilson loop 特征值分解（推荐）

#### 2.A.1 核心思想

Wilson loop 矩阵 $\mathbf{W} = \prod_j \mathbf{M}(k_j, k_{j+1})$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 矩阵。其特征值 $\lambda_n$ 满足：

$$\det\mathbf{W} = \prod_n \lambda_n$$

因此：

$$\gamma = \mathrm{Im}\,\ln\det\mathbf{W} = \mathrm{Im}\,\ln\prod_n \lambda_n = \sum_n \mathrm{Im}\,\ln\lambda_n = \sum_n \arg(\lambda_n)$$

**逐能带 Berry phase**：$\gamma_n = \arg(\lambda_n)$

**关键性质**：
- $\sum_n \gamma_n = \gamma$ **精确成立**（对数可加性）
- $\lambda_n$ 是**规范不变的**（W 的特征值不依赖波函数相位）
- 无需 Wannierization（只需对角化 W）

#### 2.A.2 逐原子分配

Wilson loop 矩阵 $\mathbf{W}$ 的特征向量 $|v_n\rangle$ 定义在 k-string 起点的占据能带空间。逐原子权重：

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$$

满足 $\sum_I w^I_n = \sum_a |\langle v_n | \alpha_a \rangle|^2 = \langle v_n | \hat{P}_{\text{SMO}} | v_n \rangle$，其中 $\hat{P}_{\text{SMO}} = \sum_a |\alpha_a\rangle\langle\alpha_a|$ 是 SMO 投影算符。

**Sum rule**：

$$\sum_I \gamma^I = \sum_I \sum_n w^I_n \gamma_n = \sum_n \left(\sum_I w^I_n\right) \gamma_n = \sum_n \langle v_n | \hat{P}_{\text{SMO}} | v_n \rangle \cdot \gamma_n$$

当 SMO 集在占据能带空间完备时，$\hat{P}_{\text{SMO}} = \mathbf{I}$，$\sum_I w^I_n = 1$，sum rule **精确成立**。

#### 2.A.3 实现步骤

```
1. 对每条 k-string:
   a. 收集重叠矩阵 M(k_j, k_{j+1}) (nocc × nocc)
   b. 计算乘积 W = M_0 · M_1 · ... · M_{N_k-1}
   c. 对角化 W → 特征值 λ_n, 特征向量 |v_n⟩
   d. 逐能带 Berry phase: γ_n = arg(λ_n)
   e. 逐原子权重: w^I_n = Σ_{a∈I} |⟨v_n|α_a⟩|²
   f. 逐原子: γ^I = Σ_n w^I_n · γ_n
2. 对所有 k-string 平均
3. P^I = -(e·a_α)/(2π·Ω) · γ^I
```

#### 2.A.4 优势

- **精确**：$\sum_n \gamma_n = \gamma$ 严格成立
- **规范不变**：W 的特征值不依赖规范
- **无需 Wannierization**：只需矩阵乘法 + 对角化
- **可复用 .mmn**：如果 towannier90=1，直接从 .mmn 文件读取 M
- **计算量小**：W 是 nocc×nocc（如 15×15），对角化代价极低

#### 2.A.5 挑战

- 特征向量 $|v_n\rangle$ 在简并特征值时不唯一 → 权重 $w^I_n$ 可能不确定
- 但 $\sum_n w^I_n \gamma_n$ 在简并子空间内是不变的（因为 $\gamma_n$ 相同）
- 需要计算 $\langle v_n | \alpha_a \rangle$，即 SMO 在特征向量上的投影

---

### 方案 B：dk 外推法

#### 2.B.1 核心思想

Berry connection 是 Berry phase 的 $dk \to 0$ 极限：

$$\gamma(dk) = \gamma_{\text{conn}}(dk) + c_2 \cdot dk^2 + c_4 \cdot dk^4 + \cdots$$

在两个 $dk$ 值上计算 Berry connection，用 Richardson 外推：

$$\gamma(0) \approx \frac{4 \cdot \gamma_{\text{conn}}(dk/2) - \gamma_{\text{conn}}(dk)}{3}$$

#### 2.B.2 逐原子分解

Berry connection 可线性分解（sum rule 精确成立）：

$$\gamma^I_{\text{conn}}(dk) = \sum_n w^I_n \cdot \mathrm{Im}\,M_{nn}(dk)$$

外推也是线性的：

$$\gamma^I(0) \approx \frac{4 \cdot \gamma^I_{\text{conn}}(dk/2) - \gamma^I_{\text{conn}}(dk)}{3}$$

#### 2.B.3 优势

- 使用现有 Berry connection 代码，改动最小
- Sum rule 在每个 dk 精确成立
- 外推消除 $O(dk^2)$ 误差

#### 2.B.4 挑战

- 需要两组 k-mesh（如 10×10×10 和 20×20×20），计算量 4× 以上
- 假设 $dk$ 依赖光滑（在拓扑转变附近可能失效）
- 外推精度取决于高阶项 $c_4 \cdot dk^4$ 的大小

---

### 方案 C：SMO 基 Wannierization

#### 2.C.1 核心思想

在 SMO 投影空间中直接做 Wannierization（最大局域化），求最优规范变换 $\mathbf{U}(k)$：

$$\min_{\mathbf{U}(k)} \Omega = \sum_n \sum_{j \neq 0} |\langle w_{n,0} | w_{n,j} \rangle|^2$$

其中 $|w_{n,k}\rangle = \sum_m U_{mn}(k) |u_{m,k}\rangle$ 是规范变换后的态。

Wannierization 后，$\mathbf{U}^\dagger(k_j) \mathbf{M}(k_j) \mathbf{U}(k_{j+1})$ 近似对角，逐能带 Berry phase：

$$\gamma_n = \mathrm{Im}\,\ln\prod_j [\mathbf{U}^\dagger(k_j) \mathbf{M}(k_j) \mathbf{U}(k_{j+1})]_{nn}$$

精确满足 $\sum_n \gamma_n = \gamma$。

#### 2.C.2 逐原子分配

Wannier 函数 $|w_n\rangle$ 的中心 $\langle r_n \rangle$ 直接给出原子归属：

$$\langle r_n \rangle = \frac{a_\alpha}{2\pi} \gamma_n$$

按 Wannier center 位置分配到最近原子。

#### 2.C.3 优势

- 最严格：给出精确的 Wannier center
- $\sum_n \gamma_n = \gamma$ 严格成立
- 逐原子分配基于物理位置（Wannier center），无任意性

#### 2.C.4 挑战

- 需要实现 Wannierization（迭代最小化 spread）
- 计算量最大（每次迭代需对所有 k 点做矩阵运算）
- 需要处理简并和收敛问题
- 可直接调用 Wannier90 库，但增加外部依赖

---

### 方案 D：利用 .mmn 文件直接计算

#### 2.D.1 核心思想

当 `towannier90=1` 时，ABACUS 生成 .mmn 文件，包含精确的重叠矩阵 $\mathbf{M}_{mn}(k, b) = \langle u_{m,k} | u_{n,k+b}\rangle$（周期部分重叠）。

DeltaP 模块可以直接读取 .mmn 文件，执行方案 A 的计算：

1. 从 .mmn 构建 Wilson loop 矩阵 $\mathbf{W} = \prod_j \mathbf{M}(k_j, k_{j+1})$
2. 对角化 $\mathbf{W}$ → 特征值 $\lambda_n$, 特征向量 $|v_n\rangle$
3. 从 .amn 文件读取投影 $A_{np}(k) = \langle u_{n,k} | \alpha_p \rangle$
4. 计算逐原子权重 $w^I_n = \sum_{p \in I} |A_{np}(k_0) \cdot v_{n,p}|^2$

#### 2.D.2 优势

- 复用现有 Wannier90 接口，零额外重叠计算
- .mmn 包含周期部分重叠（正确的 Berry phase 定义）
- .amn 提供现成的 SMO 投影
- 可与 Wannier90 的 MLWF center 交叉验证

#### 2.D.3 挑战

- .mmn 文件较大（nocc²×nkpts×nnn 个复数）
- 需要处理 .mmn 的 b-vector 映射（哪个 b 对应 gdir 方向）
- .amn 的投影数可能与 SMO 通道数不匹配（ABACUS LCAO 接口会添加额外投影）

---

### 方案 E：位置矩阵直接计算

#### 2.E.1 核心思想

Berry connection 的解析形式：

$$A_\alpha(k) = -2\,\mathrm{Im}\sum_n \langle u_{n,k} | i\partial_\alpha | u_{n,k}\rangle$$

在 LCAO 基组中，位置矩阵可通过 $\partial_k \mathbf{S}(k)$ 计算：

$$\mathbf{r}(k) = \mathbf{C}^\dagger(k) \cdot i\partial_k \mathbf{S}(k) \cdot \mathbf{C}(k)$$

逐原子分解：

$$A^I_\alpha(k) = -2\,\mathrm{Im}\sum_n \sum_{a \in I} \langle u_{n,k} | \alpha_a \rangle \langle \alpha_a | i\partial_\alpha | u_{n,k} \rangle$$

#### 2.E.2 优势

- 解析公式，无需有限差分
- 逐原子 sum rule 精确成立

#### 2.E.3 挑战

- 仍然是 Berry connection（dk→0 极限），不是 Berry phase
- $\partial_k \mathbf{S}(k)$ 需要解析计算（已在 berry_connection 方法中实现）
- 精度与离散 Berry connection 相同（都是 trace 近似）

**结论**：方案 E 与现有 berry_connection 方法本质相同，无改进。

---

## 3. 方案对比与推荐

| | 方案 A (W特征值) | 方案 B (dk外推) | 方案 C (Wannierization) | 方案 D (.mmn直读) |
|---|:---:|:---:|:---:|:---:|
| **精确性** | ✅ 精确 | ⚠ 外推近似 | ✅ 精确 | ✅ 精确 |
| **Sum rule** | ✅ 精确 | ✅ 精确 | ✅ 精确 | ✅ 精确 |
| **规范不变** | ✅ | ✅ | ✅ | ✅ |
| **实现难度** | 中等 | 低 | 高 | 中等 |
| **计算量** | 小 | 4× | 大 | 小（读文件） |
| **nproj>nocc** | ✅ 无影响 | ✅ 无影响 | ✅ 无影响 | ✅ 无影响 |
| **需要Wannier90** | ❌ | ❌ | 可选 | ✅ |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 推荐实施路径

**第一步（方案 A）**：实现 Wilson loop 特征值分解
- 在 `compute_wannier_polarization` 中，用矩阵乘法构建 W
- 对角化 W，获取特征值和特征向量
- 用 SMO 投影计算逐原子权重
- 验证 $\sum_n \gamma_n = \gamma$ 和 sum rule

**第二步（方案 D）**：添加 .mmn 直读模式
- 新增 `deltap_method = mmn`
- 读取 .mmn/.amn 文件执行方案 A
- 与 Wannier90 的 MLWF center 交叉验证

**第三步（方案 B）**：作为 fallback
- 如果方案 A 在某些体系有问题（如特征值简并）
- 用 dk 外推法作为备选

---

## 4. 方案 A 的数学严格性论证

### 4.1 Wilson loop 矩阵的定义

对一条闭合 k-string $k_0 \to k_1 \to \cdots \to k_{N-1} \to k_0$：

$$\mathbf{W} = \mathbf{M}(k_0, k_1) \cdot \mathbf{M}(k_1, k_2) \cdots \mathbf{M}(k_{N-1}, k_0)$$

$\mathbf{W}$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 矩阵。

### 4.2 规范不变性

在规范变换 $|u_{n,k}\rangle \to e^{i\varphi_n(k)} |u_{n,k}\rangle$ 下：

$$\mathbf{M}(k_j, k_{j+1}) \to \mathbf{\Phi}^\dagger(k_j) \cdot \mathbf{M}(k_j, k_{j+1}) \cdot \mathbf{\Phi}(k_{j+1})$$

其中 $\mathbf{\Phi}(k) = \mathrm{diag}(e^{i\varphi_1(k)}, \ldots, e^{i\varphi_{N_{\text{occ}}}(k)})$。

Wilson loop：

$$\mathbf{W} \to \mathbf{\Phi}^\dagger(k_0) \cdot \mathbf{M}_{01} \cdot \mathbf{\Phi}(k_1) \cdot \mathbf{\Phi}^\dagger(k_1) \cdot \mathbf{M}_{12} \cdot \mathbf{\Phi}(k_2) \cdots \mathbf{\Phi}(k_0)$$

中间项 $\mathbf{\Phi}(k_j) \cdot \mathbf{\Phi}^\dagger(k_j) = \mathbf{I}$ 逐一消去：

$$\mathbf{W} \to \mathbf{\Phi}^\dagger(k_0) \cdot \mathbf{W} \cdot \mathbf{\Phi}(k_0)$$

这是**酉相似变换**，不改变特征值：$\lambda_n(\mathbf{W}) = \lambda_n(\mathbf{\Phi}^\dagger \mathbf{W} \mathbf{\Phi})$。

**结论**：$\lambda_n$ 是规范不变的。✓

### 4.3 Sum rule 的精确性

$$\gamma = \mathrm{Im}\,\ln\det\mathbf{W} = \mathrm{Im}\,\ln\prod_n \lambda_n = \mathrm{Im}\sum_n \ln\lambda_n = \sum_n \arg(\lambda_n) = \sum_n \gamma_n$$

**严格成立**，不需要任何近似。✓

### 4.4 逐原子权重

特征向量 $|v_n\rangle$ 在规范变换下：

$$|v_n\rangle \to \mathbf{\Phi}^\dagger(k_0) |v_n\rangle$$

SMO 投影：

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$$

在规范变换下，$\langle v_n | \alpha_a \rangle \to \langle v_n | \mathbf{\Phi}(k_0) | \alpha_a \rangle$。

**问题**：$w^I_n$ 是规范依赖的！

但 $\sum_n w^I_n \gamma_n$ 是规范不变的，因为：
- $\gamma_n$ 是规范不变的
- 在简并子空间内，$w^I_n$ 的变化被 $\gamma_n$ 的不变性抵消
- 在非简并情况下，$|v_n\rangle$ 只差一个相位，$|\langle v_n | \alpha_a \rangle|^2$ 不变

**结论**：$w^I_n$ 在非简并情况下规范不变；在简并情况下，$\sum_n w^I_n \gamma_n$ 仍规范不变。✓

### 4.5 与 Wannier center 的关系

Wilson loop 特征值 $\lambda_n = e^{i\gamma_n}$，对应的 Wannier center：

$$\langle r_{n,\alpha} \rangle = \frac{a_\alpha}{2\pi} \gamma_n = \frac{a_\alpha}{2\pi} \arg(\lambda_n)$$

这与 Wannier90 给出的 MLWF center **完全一致**（在 Wannier gauge 中，MLWF center 就是 Wilson loop 特征值的 arg）。

**结论**：方案 A 给出的 $\gamma_n$ 与 Wannier90 的 Wannier center 是同一个量。✓

---

## 5. 实现伪代码

```python
def compute_per_atom_polarization(M_kpairs, D_I, nocc, natom, nproj_per_atom):
    """
    M_kpairs: list of nocc×nocc overlap matrices along k-string
    D_I: SMO projection matrix (nproj_total × nocc) at k_0
    """
    # Step 1: Build Wilson loop matrix
    W = identity(nocc)
    for M in M_kpairs:
        W = W @ M
    
    # Step 2: Diagonalize
    eigenvalues, eigenvectors = eig(W)
    
    # Step 3: Per-band Berry phase
    gamma_n = [arg(lam) for lam in eigenvalues]
    gamma_total = sum(gamma_n)
    # Verify: gamma_total == arg(det(W))  ← exact
    
    # Step 4: Per-atom weights
    # Project SMOs onto eigenvectors
    # D_I is (nproj_total × nocc), eigenvectors is (nocc × nocc)
    # projection[n, a] = <v_n | alpha_a> = (D_I @ eigenvectors)[a, n]
    proj = D_I.T @ eigenvectors  # (nproj_total × nocc)
    
    gamma_I = zeros(natom)
    offset = 0
    for I in range(natom):
        r = nproj_per_atom[I]
        w_In = sum(|proj[offset:offset+r, n]|^2, axis=0)  # (nocc,)
        gamma_I[I] = sum(w_In * gamma_n)
        offset += r
    
    # Sum rule: sum(gamma_I) ≈ gamma_total (exact if SMO complete)
    return gamma_I, gamma_total
```

---

## 6. 预期效果

### 6.1 对 BaTiO3 的预期

| 指标 | 当前 (trace) | 方案 A (特征值) |
|---|---|---|
| 总 γ | trace = -0.535 | det = 1.200 |
| Sum γ_n | 0.672 (≠γ) | = γ (精确) |
| Z*_Ti | 47.7 (错) | ≈ 6.69 (预期正确) |
| Z*_Ba | 13.9 (错) | ≈ 2.67 (预期正确) |

### 6.2 对 diamond 的预期

| 指标 | berry_phase (粗k) | 方案 A | Wannier90 |
|---|---|---|---|
| Z*_C | 4.0 (错) | ≈ 0 (预期正确) | -0.02 (正确) |

### 6.3 关键改进

- **消除 trace≠det 问题**：特征值分解给出精确的逐能带 Berry phase
- **消除 nproj>nocc 问题**：W 是 nocc×nocc，与 nproj 无关
- **保持 sum rule**：$\sum_n \gamma_n = \gamma$ 严格成立
- **与 Wannier90 一致**：$\gamma_n = \arg(\lambda_n)$ 就是 Wannier center
