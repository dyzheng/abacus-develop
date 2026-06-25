# SMO投影Wannier函数 — 多维度评估与文献对比

> **问题**: 评估"通过SMO构造原型Wannier函数"的好处、不足，与直接SMO投影的区别  
> **文献**: Ozaki CWF (2024), Solovyev (2007), Koepernik SCMPWF (2023)  
> **日期**: 2026-06-24

---

## 1. 执行摘要

方案六并非凭空设想——它与三个独立发展的前沿方法**数学上同构**：

| 方法 | 作者/年份 | 与方案六的关系 |
|------|----------|:--:|
| **Closest Wannier Functions (CWF)** | Ozaki, 2024 [1] | **完全等价**：投影+极分解 = SVD最小化距离函数 |
| **投影算符法Wannier** | Solovyev et al., 2007 [2] | 同一投影→Löwdin正交化流程 |
| **Symm. Conserving Max. Projected WF** | Koepernik et al., 2023 [3] | 同一思想，强调对称性保持 |

这给了方案六强有力的学术支撑。但三个文献也一致揭示了一个根本权衡，这构成了方案六的核心张力。

---

## 2. 与直接SMO投影的本质区别

### 2.1 直接SMO投影：只能"数电子"，不能"定位置"

DeltaSpin已有的SMO投影计算的是**占据数**：

$$n^{I}_{lmm'} = \sum_{\mathbf{R}} \sum_{\mu\nu} \rho_{\mu\nu}(\mathbf{R}) \langle\phi^0_\mu|\alpha^I_{lm}\rangle\langle\alpha^I_{lm'}|\phi^{\mathbf{R}}_\nu\rangle$$

这是一个**标量**——它告诉你"原子I的lm轨道上有多少电子"。但极化需要的是**矢量**——"这些电子的中心在哪里"。

在Berry phase形式alism中，电子的"位置信息"编码在波函数的**相位**中，而占据数（密度矩阵的对角元）恰好丢失了所有相位信息。

### 2.2 投影Wannier函数：保留了相位信息

方案六的关键步骤：

$$|\tilde{w}^I_{lm}\rangle = \hat{P}^I_{lm} |\psi^{\text{occ}}\rangle = |\alpha^I_{lm}\rangle\langle\alpha^I_{lm}|\psi^{\text{occ}}\rangle$$

这里 $|\tilde{w}^I_{lm}\rangle$ 是一个**波函数**（矢量），不是标量。它保留了：
- 占据态 $|\psi^{\text{occ}}\rangle$ 的相位信息
- SMO $|\alpha^I_{lm}\rangle$ 与占据态的相位差
- 从而保留了Wannier中心的**实空间位置**

### 2.3 直观对比

```
直接SMO投影 (DeltaSpin已有):
  ψ → ⟨α|ψ⟩ → |⟨α|ψ⟩|² → 标量 n → "这个轨道上有多少电子"

投影Wannier函数 (方案六):
  ψ → |α⟩⟨α|ψ⟩ → 波函数 |w̃⟩ → ⟨w̃|r|w̃⟩ → 矢量 r̄ 
  → "这些电子的中心在哪里"
```

**一句话总结**：直接SMO投影给了你"电子计数"，投影Wannier函数给了你"电子位置"。

---

## 3. 好处（与文献一致的论证）

### 3.1 非迭代、无局部极小值

这是CWF [1]和SCMPWF [3]共同强调的核心优势。

- MLWF需要迭代优化 $\Omega = \sum_n [\langle r^2\rangle_n - \langle \mathbf{r}\rangle_n^2]$，经常陷入局部极小值 [3]
- CWF/方案六：一次投影 + 一次SVD → 唯一解
- Ozaki [1] 证明：当投影矩阵 A(k) 的奇异值全正时，解是**唯一的**（Eq. 12-13的证明）

> "The minimization is directly achieved by a polar decomposition of a projection matrix via singular value decomposition, making iterative calculations and complications arising from the choice of the gauge irrelevant." — Ozaki 2024 [1]

### 3.2 规范不变性

这是三个文献共同强调的另一个关键优势。

CWF [1]通过极分解 $B(k) = U(k) = W(k)V^\dagger(k)$ 自动消除了Bloch函数的任意相位：

> "the non-unique phase of the Bloch functions is canceled out via the polar decomposition" — Ozaki 2024 [1]

这意味着：**独立SCF计算之间自动保持规范连续**——这正是你上一个问题中要求的核心特性。

### 3.3 化学直观性

SCMPWF [3]的核心论点：

> "If one knows from band characters and symmetry conditions which orbitals engender a certain band complex then the WFs obtained from projection onto these orbitals will be localized." — Koepernik 2023 [3]

SMO本身就是根据化学直觉构造的——截断的NAO ζ函数保留了原子的价轨道特征。用SMO做投影本质上是说："价带主要是这些原子的这些轨道组成的，把我投影上去"。

### 3.4 处理纠缠能带的天然能力

三个方法都通过窗口函数/能量选择自然地处理纠缠能带（disentanglement），不需要像MLWF那样需要显式的"解纠缠"步骤 [1,3]。

CWF引入光滑窗口函数 $w(\varepsilon)$：
$$w(\varepsilon) = 1 - \frac{\exp(x_0 + x_1)}{(1+\exp(x_0))(1+\exp(x_1))} + \delta$$

使得能带解纠缠被自动处理——在能量窗口外的态权重逐渐衰减到零，不会产生奇异性。

### 3.5 对称性保持

SCMPWF的设计核心就是对称性保持 [3]。方案六中的SMO天然具有：
- 球对称性（径向函数 × 球谐函数）
- 原子中心对称性
- 在ABACUS中的SMO已经具有这些性质 [DeltaSpin, Sec. II.B]

---

## 4. 不足与风险（同样来自文献的警示）

### 4.1 对投影轨道选择的敏感性 ⚠️ **最重要的风险**

这是三个文献**最一致**的警告。

Solovyev [2]用两带模型（Sec. IV.A）做了精辟的演示：
- 当投影轨道 $\beta=0$（选择与最大密度矩阵本征值对应的轨道）→ WF极度局域（99.9%权重在中心+最近邻）
- 当投影轨道 $\beta=90^\circ$（选择正交的轨道）→ WF极度弥散（需要~200个配位球才收敛）

> "a bad choice of trial orbitals can be linked to the discontinuity of phase of the Bloch waves in the reciprocal space, which leads to the delocalization of WFs in the real space" — Solovyev 2007 [2]

**对方案六的推论**：SMO的质量（调制半径rm、平滑参数σ）直接影响投影Wannier函数的局域性。如果SMO太小（rm太小），无法充分覆盖价带波函数，Wannier函数会弥散。

Ozaki [1]也指出：
> "The choice of the guiding functions ... may depend on the implementation"

### 4.2 Wannier中心精度低于MLWF

SCMPWF论文明确承认 [3]：

> "SCMPWF basically represents the zeroth order approximation of the maximally localized approach"

因为是"零阶近似"（没有经过spread最小化的迭代优化），投影Wannier函数的spread $\langle r^2\rangle - \langle \mathbf{r}\rangle^2$ 通常大于MLWF。

**量化影响**：Solovyev [2] Fig. 4显示，对V₂O₃的t₂g带，最优投影给出的 $\langle r^2\rangle = 2.47$ Å²，而不当选择可达3.18 Å²（增加~30%）。对于极化计算，spread本身不直接等于误差，但更大的spread意味着Wannier中心对远程原子轨道的敏感度增加。

### 4.3 $S^I$ 矩阵近奇异时的数值不稳定

当SMO投影在某个方向上的模趋于零时（某些原子的有效电子数很低），正交化步骤 $(S^I)^{-1/2}$ 可能数值不稳定。

Ozaki [1]通过引入小常数 $\delta = 10^{-12}$ 在窗口函数中缓解：
> "The violation from the positive definiteness of the singular values can be avoided by the small constant of δ"

方案六需要类似的数值正则化。

### 4.4 不适合"无原子"的Wannier函数

SCMPWF [3]指出：某些体系的Wannier函数是键中心（bond-centered）而非原子中心（atom-centered）的。对于这类体系，仅用原子中心的SMO做投影是不够的。

Koepernik [3]的解决方案是引入"分子轨道投影"（MO projectors）——允许用户定义原子轨道的线性组合作为投影器。方案六可以通过SMO的线性组合来扩展：

$$|\tilde{w}^{\text{bond}}_{AB}\rangle = \frac{|\tilde{w}^A\rangle + |\tilde{w}^B\rangle}{\sqrt{2(1 + \text{Re}\langle\tilde{w}^A|\tilde{w}^B\rangle)}}$$

### 4.5 对孤立能带组可能缺少足够投影

Ozaki [1]指出：对于拓扑绝缘体等有能带反转的体系，需要仔细选择投影轨道：

> "When picking isolated band complexes for Wannierization it must be avoided that the gaps above and below are topological. Otherwise, the WFs cannot be localized due to topological obstruction." — Koepernik 2023 [3]

这与"拓扑量子化学"的概念有关 [3, Ref. 15]——如果选取的投影函数不构成完整的"基本能带表示"（elementary band representation），则无法构造局域Wannier函数。SMO作为投影函数天然构成原子中心的化学基组，大概率符合此要求。

---

## 5. 文献中的数值证据

### 5.1 CWF的精度表现 [1]

Ozaki 对Si、Cu、TTF-TCNQ、Bi₂Se₃的测试显示：

| 体系 | Wannier内插能带精度 | DM函数值 |
|------|:---:|:---:|
| Si（价带） | 完美重现 | 0.444 |
| Cu（3d+4s+4p） | 完美重现 | 0.211 |
| Cu（仅3d解纠缠） | 3d带完美解纠缠 | 0.081 |
| TTF-TCNQ（分子轨道投影） | 4条前沿带完美重现 | 0.022 |
| Bi₂Se₃（含SOC） | 含能带反转，完美重现 | 0.184 |

DM函数值越小越好（0=完美投影），表明CWF通常非常接近原始原子轨道。

### 5.2 Solovyev 的局域性分析 [2]

对V₂O₃的t₂g带：
- 最优投影（$\beta=0$）：Wannier函数权重在~7.5Å内收敛（16个配位球）
- 不当投影（$\beta=60^\circ$）：需要~15Å（远距离尾更长）
- 极端不当（$\beta=90^\circ$）：20Å内仍不收敛

但对能带色散的重现：**三种选择都完美重现了LMTO能带**（Fig. 5）——这表明即使Wannier函数不够局域，tight-binding Hamiltonian仍然正确。对极化计算来说，关键是Wannier中心是否准确。

---

## 6. 对方案六的具体建议

### 6.1 SMO调制半径的优化

基于文献共识，SMO的rm选择至关重要。建议采用DeltaSpin已有的"最大磁矩策略" [4] 的类似逻辑——对于极化，选择使 $|\partial M/\partial r_m|$ 最大的rm，或使SMO投影效率最高的rm。

具体：扫描 $r_m \in [1.0, 5.0]$ Bohr，选择使 $\sum_{n,\mathbf{k}} |\langle\alpha^I_{lm\mathbf{k}}|\psi_{n\mathbf{k}}\rangle|^2$ 最大且稳定的rm。

### 6.2 与CWF方法的对齐

方案六的数学实现可以直接对齐CWF [1]的算法流程：

```
Step 1: 构造投影矩阵 A(k)_{(μ,n),(I,lm)} = ⟨φ_μ,k|α^I_lmk⟩ · C_{nμ}(k)
Step 2: SVD: A(k) = W(k) Σ(k) V†(k)
Step 3: 极分解: U(k) = W(k) V†(k)
Step 4: 正交Wannier Bloch和: |w^I_lm,k⟩ = Σ_n U(k)_{n,(I,lm)} |ψ_nk⟩
Step 5: Wannier中心: r̄^I_lm = ⟨w^I_lm,0|r|w^I_lm,0⟩
Step 6: 加和: P^I = -(e/Ω) Σ_lm n_lm r̄^I_lm
```

这比原始方案的"Löwdin正交化"更数值稳定（SVD自动处理近奇异情况）。

### 6.3 对称性检查

确保选用的SMO集合构成每个原子的"化学基组"的完备子集（如对过渡金属：至少包含s+p+d轨道）。这可以避免Koepernik警告的"分裂基本能带表示"问题。

---

## 7. 最终评估矩阵

| 维度 | 评分 | 说明 |
|------|:---:|------|
| **数学正确性** | ★★★★★ | 与CWF/SCMPWF等价，严格公式 |
| **无迭代/无局部极小** | ★★★★★ | 比MLWF的最大优势 |
| **跨SCF规范连续性** | ★★★★★ | 极分解自动消除随机相位 |
| **局域性** | ★★★★☆ | 通常良好，但不如MLWF最优 |
| **对SMO选择的敏感性** | ⚠️ 中风险 | 需要合理的rm和基组选择 |
| **实现复杂度** | ★★★☆☆ | SVD比MLWF简单，但比直接投影复杂 |
| **ABACUS集成难度** | ★★★☆☆ | 需要k空间SVD，ABACUS/HContainer可复用 |
| **已有学术验证** | ★★★★★ | 三个独立方法的共同验证 |

---

## 参考文献

[1] Ozaki, T. (2024). Closest Wannier functions to a given set of localized orbitals. *Physical Review B*, 110(12), 125115. [arXiv:2306.15296]

[2] Solovyev, I. V., Pchelkina, Z. V., & Anisimov, V. I. (2007). Construction of Wannier functions from localized atomic-like orbitals. *Physical Review B*, 75(4), 045110. [arXiv:cond-mat/0608528]

[3] Koepernik, K., Janson, O., Sun, Y., & van den Brink, J. (2023). Symmetry-conserving maximally projected Wannier functions. *Physical Review B*, 107(23), 235135. [arXiv:2111.09652]

[4] Zheng, D. et al. (2025). Integrating Deep-Learning-Based Magnetic Model and Non-Collinear Spin-Constrained Method. [arXiv:2501.14382]
