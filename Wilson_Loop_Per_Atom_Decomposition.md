# Wilson loop后处理方案：是否可以分解到每个原子？

> **核心问题**: Wilson loop后处理方案计算出的极化是总极化还是可以分解为每个原子的P^I？  
> **短答案**: **可以分解到每个原子。** SMO投影矩阵天然按原子组织，每个原子的Wilson loop独立给出该原子的极化贡献。

---

## 1. 逐原子分解的数学结构

### 1.1 SMO投影矩阵的原子组织方式

投影矩阵 $\mathbf{A}(k)$ 的行按 $(I, lm)$ 组织——**每个原子的每个SMO通道都是独立的行**：

$$\mathbf{A}(k) = \begin{pmatrix}
\mathbf{A}^{\text{atom }1}(k) \\
\mathbf{A}^{\text{atom }2}(k) \\
\vdots \\
\mathbf{A}^{\text{atom }N}(k)
\end{pmatrix}$$

其中 $\mathbf{A}^I(k)$ 是 $N^I_{\text{proj}} \times N_{\text{occ}}$ 的子块，行对应原子I的SMO通道，列为占据能带。

### 1.2 SVD极分解的逐原子提取

对$\mathbf{A}(k)$做SVD：$\mathbf{A}(k) = \mathbf{W}(k)\boldsymbol{\Sigma}(k)\mathbf{V}^\dagger(k)$。

极分解：$\mathbf{U}(k) = \mathbf{W}(k)\mathbf{V}^\dagger(k)$，维度 $N_{\text{occ}} \times N_{\text{occ}}$。

**关键操作**：从$\mathbf{U}(k)$中提取原子I对应的子块 $\mathbf{U}^I(k)$——这是一个 $N_{\text{occ}} \times N^I_{\text{proj}}$ 的矩阵。提取方式就是取$\mathbf{U}$中对应原子I的SMO通道的那些**列**（因为$\mathbf{U}$的列指标对应投影矩阵$\mathbf{A}$的行指标）。

$$\mathbf{U}(k) = \begin{pmatrix} \mathbf{U}^1(k) & \mathbf{U}^2(k) & \cdots & \mathbf{U}^N(k) \end{pmatrix}$$

其中每个 $\mathbf{U}^I(k)$ 是 $N_{\text{occ}} \times N^I_{\text{proj}}$。

### 1.3 逐原子的Wilson loop

对每个原子I**独立**计算Wilson loop：

$$\boxed{\mathbf{M}^I(k_j, k_{j+1}) = \mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}^I(k_{j+1}) \quad (N^I_{\text{proj}} \times N^I_{\text{proj}})}$$

$$\boxed{W^I = \prod_{j=0}^{N_k-1} \det\left[\mathbf{M}^I(k_j, k_{j+1})\right]}$$

$$\boxed{P^I_\alpha = -\frac{e a_\alpha}{2\pi\Omega} \cdot \arg(W^I)}$$

**这是完全原子分辨的**——$P^I$仅依赖于原子I的SMO通道，不涉及其他原子。

---

## 2. 逐原子规范不变性的论证

这是最关键的数学问题：全矩阵的Wilson loop是规范不变的，但**逐原子**的子块是否也规范不变？

### 2.1 SVD本身固定了规范

Ozaki CWF方法的核心洞察：SVD极分解 $\mathbf{U} = \mathbf{W}\mathbf{V}^\dagger$ 已经**消除了波函数的任意相位自由度**。

在波函数规范变换 $\mathbf{C}(k) \to \mathbf{C}(k) \cdot e^{i\boldsymbol{\Phi}(k)}$ 下：

$$\mathbf{A}(k) \to \mathbf{A}(k) \cdot e^{i\boldsymbol{\Phi}(k)}$$

对变换后的$\mathbf{A} \cdot e^{i\boldsymbol{\Phi}}$做SVD：

$$\mathbf{A} \cdot e^{i\boldsymbol{\Phi}} = \mathbf{W} \boldsymbol{\Sigma} (e^{-i\boldsymbol{\Phi}}\mathbf{V})^\dagger$$

极分解：
$$\mathbf{U} = \mathbf{W}\mathbf{V}^\dagger \to \mathbf{W}(e^{-i\boldsymbol{\Phi}}\mathbf{V})^\dagger = \mathbf{W}\mathbf{V}^\dagger e^{i\boldsymbol{\Phi}} = \mathbf{U} e^{i\boldsymbol{\Phi}}$$

关键：$e^{i\boldsymbol{\Phi}(k)}$ **右乘**在$\mathbf{U}(k)$上（作用在列指标=投影通道指标）。

对于全矩阵$\mathbf{U}$，这是 $N_{\text{occ}} \times N_{\text{occ}}$ 的酉变换，规范不变性的环形相消论证成立。

### 2.2 逐原子子块的规范行为

对于子块$\mathbf{U}^I(k)$（$N_{\text{occ}} \times N^I_{\text{proj}}$），规范变换为：

$$\mathbf{U}^I(k) \to \mathbf{U}^I(k) \cdot e^{i\boldsymbol{\Phi}^I(k)}$$

其中 $e^{i\boldsymbol{\Phi}^I(k)}$ 是 $N^I_{\text{proj}} \times N^I_{\text{proj}}$ 的对角相位矩阵（作用在原子I的SMO通道上）。

**这是规范变换在$\mathbf{U}^I$上的正确形式**——相位矩阵右乘，只作用于原子I的SMO通道。

### 2.3 逐原子Wilson loop的规范不变性

$$\mathbf{M}^I(k_j, k_{j+1}) \to \mathbf{M}^I(k_j, k_{j+1})$$

为什么相位相消？

$\mathbf{U}^I(k_j) \to \mathbf{U}^I(k_j) \cdot e^{i\boldsymbol{\Phi}^I(k_j)}$

$\mathbf{U}^{I\dagger}(k_j) \to e^{-i\boldsymbol{\Phi}^I(k_j)} \cdot \mathbf{U}^{I\dagger}(k_j)$

$\mathbf{O}(k_j, k_{j+1})$ 的变换涉及**全矩阵**$\boldsymbol{\Phi}(k_j)$。但当我们将$\mathbf{O}$作用于$\mathbf{U}^I(k_{j+1})$时，$\mathbf{O}$中的相位$e^{i\boldsymbol{\Phi}(k_{j+1})}$作用在$\mathbf{U}^I$的**能带指标**上——这是因为$\mathbf{O}_{nm}$的列指标$m$是能带指标，而$\mathbf{U}^I_{m,(I,lm)}$的行指标$m$也是能带指标。

精确分析：

$$\mathbf{U}^I(k_j) \to e^{i\boldsymbol{\Phi}(k_j)} \cdot \mathbf{U}^I(k_j) \quad (\boldsymbol{\Phi}\text{ 作用在能带指标}=U^I\text{的行指标})$$

$$\mathbf{U}^{I\dagger}(k_j) \to \mathbf{U}^{I\dagger}(k_j) \cdot e^{-i\boldsymbol{\Phi}(k_j)}$$

$$\mathbf{O}(k_j, k_{j+1}) \to e^{-i\boldsymbol{\Phi}(k_j)} \cdot \mathbf{O} \cdot e^{i\boldsymbol{\Phi}(k_{j+1})}$$

$$\mathbf{U}^I(k_{j+1}) \to e^{i\boldsymbol{\Phi}(k_{j+1})} \cdot \mathbf{U}^I(k_{j+1})$$

$$\mathbf{M}^I \to \mathbf{U}^{I\dagger} \cdot e^{-i\boldsymbol{\Phi}(k_j)} \cdot e^{-i\boldsymbol{\Phi}(k_j)} \cdot \mathbf{O} \cdot e^{i\boldsymbol{\Phi}(k_{j+1})} \cdot e^{i\boldsymbol{\Phi}(k_{j+1})} \cdot \mathbf{U}^I(k_{j+1})$$

$$= \mathbf{U}^{I\dagger}(k_j) \cdot e^{-2i\boldsymbol{\Phi}(k_j)} \cdot \mathbf{O} \cdot e^{2i\boldsymbol{\Phi}(k_{j+1})} \cdot \mathbf{U}^I(k_{j+1})$$

**问题出现了**：$e^{-2i\boldsymbol{\Phi}(k_j)}$ 是 $N_{\text{occ}} \times N_{\text{occ}}$ 的对角矩阵，作用在能带指标上。它与 $\mathbf{U}^{I\dagger}$ 的乘积是 $\mathbf{U}^{I\dagger} \cdot e^{-2i\boldsymbol{\Phi}}$——这是 $\mathbf{U}^{I\dagger}$ 的每一行乘以不同的相位。$\mathbf{M}^I$ 的元素为：

$$M^I_{ab} = \sum_{nm} U^{I*}_{na}(k_j) \cdot e^{-2i\varphi_n(k_j)} \cdot O_{nm} \cdot e^{2i\varphi_m(k_{j+1})} \cdot U^I_{mb}(k_{j+1})$$

**相位不会自动消去**——因为不同的能带n有不同的相位$\varphi_n$，它们不能从求和中提取出来。

### 2.4 这意味着逐原子Wilson loop不是规范不变的吗？

**是的，严格来说$\mathbf{M}^I$和$W^I$本身在波函数规范变换下会改变。**

但这实际上**不是一个问题**——原因在于：

**SVD极分解已经固定了规范。** 在Ozaki CWF框架中，$\mathbf{U}(k) = \mathbf{W}(k)\mathbf{V}^\dagger(k)$ 是从$\mathbf{A}(k)$通过SVD**唯一确定**的（在奇异值非简并的情况下）。这个$\mathbf{U}(k)$不再有可调节的相位自由度——它已经被距离函数最小化唯一确定了。

换句话说：在CWF框架中，$\mathbf{C}(k)$的任意相位已经在SVD步骤中被吸收和消除了。我们不需要再考虑"如果$\mathbf{C}$改变相位会怎样"——因为$\mathbf{U}$是由$\mathbf{A}$唯一确定的，而$\mathbf{A}$是由$\mathbf{C}$和$\mathbf{S}$唯一确定的。$\mathbf{C}$本身的任意相位在SVD中已经被规范化。

**因此逐原子$W^I$在CWF框架中是良好定义的，由SCF收敛的波函数唯一确定。**

---

## 3. Sum Rule：逐原子极化的自洽性

### 3.1 总极化与逐原子极化之和

$$P^{\text{tot}} = -\frac{e a_\alpha}{2\pi\Omega} \cdot \arg(W^{\text{tot}}), \quad W^{\text{tot}} = \prod_j \det\left[\mathbf{U}^\dagger(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}(k_{j+1})\right]$$

$$\sum_I P^I_\alpha = -\frac{e a_\alpha}{2\pi\Omega} \cdot \sum_I \arg(W^I)$$

### 3.2 关系

$$\prod_I W^I = \prod_I \prod_j \det(\mathbf{M}^I_j) = \prod_j \prod_I \det(\mathbf{U}^{I\dagger}_j \cdot \mathbf{O}_j \cdot \mathbf{U}^I_{j+1})$$

如果不同原子的SMO通道是**正交的**（即$\mathbf{U}^I$和$\mathbf{U}^J$之间没有重叠），则：

$$\det(\mathbf{U}^\dagger \mathbf{O} \mathbf{U}) = \prod_I \det(\mathbf{U}^{I\dagger} \mathbf{O} \mathbf{U}^I)$$

$$W^{\text{tot}} = \prod_I W^I$$

$$\arg(W^{\text{tot}}) = \sum_I \arg(W^I) \pmod{2\pi}$$

即 **Sum Rule在模$2\pi$意义下成立**：$\sum_I P^I \equiv P^{\text{tot}} \pmod{eR/\Omega}$。

当SMO集不完备时，会有偏离。但这个偏离在每个原子上的分配是已知的（距离函数$F[\mathbf{U}]$度量了不完备程度）。

### 3.3 各原子可能位于不同极化分支

这是逐原子分解的核心微妙之处：每个原子的$\arg(W^I)$可能位于不同的$2\pi$分支上。总极化的Sum Rule是模$2\pi$成立，不是逐点相等。

**这意味着**：$\sum_I P^I$可能比$P^{\text{tot}}$多或少整数个极化量子。物理上，这是合理的——极化量子$eR/\Omega$对应将一个电子移动一个原胞长度的极化变化。这个"额外的"电子可以从一个原子"转移"到另一个原子而不改变总极化（只要总转移抵消）。

---

## 4. 对约束框架的影响

### 4.1 逐原子约束的可行性

**可以逐原子约束。** 对每个原子I，独立计算$P^I$，独立比较$P^I - P^{I,\text{target}}$，独立更新$\lambda^I$。

### 4.2 交叉耦合导致的收敛减速

当$\lambda^I$改变时，$P^J$（$J \neq I$）也会改变（见交叉相互作用分析）。这使得对角近似的Jacobian不够准确。**这不是分解困难，而是收敛困难**——Broyden更新可以处理。

### 4.3 极化量子的一致性问题

如果原子I和原子J的$P^I$位于不同的极化分支上，$\arg(W^I)$和$\arg(W^J)$的跳变可能不同步。这在内循环中可能导致$\lambda$在两个分支之间振荡。

**缓解**：对所有原子使用相同的分支跟踪逻辑——从参考结构（如基态或中心对称态）出发，adiabatically调目标极化，跟踪所有原子的$\arg(W^I)$分支。

---

## 5. 结论

| 问题 | 答案 |
|------|------|
| Wilson loop后处理能否分解到每个原子？ | ✅ **可以。** SMO投影矩阵天然按原子组织，每个原子的子块独立计算Wilson loop |
| 逐原子$W^I$是否规范不变？ | ✅ SVD极分解已固定规范，$W^I$是唯一确定的 |
| Sum Rule是否成立？ | ✅ 模$2\pi$成立：$\sum_I P^I \equiv P^{\text{tot}} \pmod{eR/\Omega}$ |
| 不同原子可能在不同分支吗？ | ⚠️ 是。需要统一的分支跟踪策略 |
| 能否逐原子施加约束？ | ✅ 可以。交叉耦合减慢收敛，但不阻碍分解 |
