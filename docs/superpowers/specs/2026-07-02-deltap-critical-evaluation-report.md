# DeltaP 逐原子极化分解算法：批判性评估报告

> **日期**: 2026-07-02
> **分支**: `feat/deltap-wilson-per-atom`
> **代码版本**: ABACUS v3.11.0-beta.1 + DeltaP 模块
> **评估范围**: 算法 D（Wilson loop 特征值分解）及其衍生方案的完整开发周期
> **前序文档**: `2026-06-28-deltap-algorithm-evaluation.md` (算法理论与测试), `2026-06-30-deltap-three-system-test-results.md` (三体系对比)

---

## 1. 评估摘要

DeltaP 算法的核心目标是将晶体电子极化（Berry phase）分解为各原子的贡献，以计算 Born 有效电荷 $Z^*_I$。经过 30+ 次提交、6 种候选算法的理论分析与实现、4 个关键 bug 的修复，当前状态如下：

| 维度 | 状态 | 评价 |
|------|------|------|
| **总电子极化** | ✅ 精确 (≤0.1%) | 三体系 (BN, H₂O, 液态水) 均与 Berry Phase 电子部分一致 |
| **逐原子 sum rule** | ✅ 精确 | $\sum_I \gamma^I = \gamma$ 在 SMO 完备时成立 |
| **逐原子分配** | ⚠️ 不唯一 | SMO 权重给出连续分配，但与 Wannier90 硬归属差异 7-22% |
| **逐原子位移** | ❌ 未完成 | D_mat > 1 违反 Cauchy-Schwarz，Löwdin 正交化受阻 |
| **Born 有效电荷 $Z^*$** | ❌ 不可靠 | 3% 极化误差被 $1/\delta$ 放大，2π 分支跳变未解决 |
| **跨基组一致性** | ❌ 未验证 | SMO 权重与 Mulliken 类似，可能基组依赖 |

**核心判断**: DeltaP 在**总量级别**已经成功——总电子极化与 Berry Phase 精确一致。但在**逐原子级别**，无论是极化分配还是位移计算，都存在尚未解决的根本性困难。这些困难部分是数学本质的（极化分解不唯一、2π 分支切割），部分是实现层面的（D_mat 基矢不匹配、Löwdin 正交化受阻）。

---

## 2. 可取之处

### 2.1 总电子极化：精确且通用

修正 4 个关键 bug 后，DeltaP 的总电子极化在所有测试体系中与 Berry Phase 精确一致：

| 体系 | 能带色散 | DeltaP / Berry 偏差 |
|------|---------|---------------------|
| BN 闪锌矿 (强色散) | ~3 eV | 0.01% |
| H₂O 分子 (无色散) | 0 eV | 0.10% |
| 液态水 (弱色散) | <0.5 eV | 0.02% |

此前的"DeltaP 依赖能带色散、对分子体系失效"结论已被证明是代码 bug 导致的假象。修正后，DeltaP 对所有体系类型均适用，包括孤立分子和弱色散分子晶体。

**技术优势**：DeltaP 使用 `berryphase_overlap()` 构建 Wilson loop，确保与 Berry Phase 使用完全相同的重叠矩阵，消除了此前 `snap` vs `center2` 积分网格差异导致的 3% 误差。

### 2.2 算法 D 的理论严格性

Wilson loop 特征值分解法（算法 D）在理论上是最严格的逐原子分解路径：

1. **精确性**：$\sum_n \arg(\lambda_n) = \arg(\det \mathbf{W}) = \gamma$，严格成立，无近似
2. **规范不变**：特征值 $\lambda_n$ 是酉相似变换不变量
3. **与 Wannier center 等价**：$\gamma_n = \arg(\lambda_n)$ 就是 Wannier center 位置
4. **无需 Wannierization**：只需 $N_{\text{occ}} \times N_{\text{occ}}$ 矩阵对角化，计算量极小
5. **不受 $n_{\text{proj}} > N_{\text{occ}}$ 限制**：$\mathbf{W}$ 是 $N_{\text{occ}} \times N_{\text{occ}}$，与 SMO 通道数无关

相比之下，算法 A（trace）因 $\text{trace} \neq \det$ 有 44% 数学必然误差；算法 B 在 $n_{\text{proj}} > N_{\text{occ}}$ 时退化；算法 C 的 trace/det 比例随结构变号。算法 D 是唯一同时满足"精确 + 可线性分解 + 不受 nproj 限制"的方案。

### 2.3 连续 SMO 权重 vs 离散硬归属

DeltaP 的 SMO 投影权重 $w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$ 给出连续的 0~1 分配，而非 Wannier90 的 0 或 1 硬归属。在共价体系中，这更符合物理直觉：

- **BN**：DeltaP 给出 B 57% / N 43%（符合 B-N 电负性差异），Wannier90 给出 B 100% / N 0%（所有 WF 靠近 B，显然不合理）
- **H₂O**：DeltaP 给出 O 39% / H 61%，Wannier90 给出 O 48% / H 52%（两者都有偏差，但 DeltaP 的连续分配在概念上更合理）

### 2.4 无需 disentanglement

Wannier90 在 BN 体系中因 disentanglement 子空间跨 k 点不一致导致总极化 74% 偏差。DeltaP 直接对角化 $N_{\text{occ}} \times N_{\text{occ}}$ Wilson loop 矩阵，不需要子空间选择，从根本上避免了此问题。这是 DeltaP 相对于 Wannier90 的一个结构性优势。

### 2.5 计算效率

算法 D 的计算复杂度为 $O(N_k \cdot N_{\text{occ}}^3)$（Wilson loop 构造 + 对角化），远低于 Wannierization 的 $O(N_k \cdot N_{\text{occ}}^2 \cdot N_{\text{nnn}} \cdot N_{\text{iter}})$（通常 $N_{\text{iter}} \sim 10\text{-}100$）。在实际测试中，DeltaP 的对角化步骤几乎不耗时。

---

## 3. 不足之处

### 3.1 逐原子极化分解的本质不唯一性

**这是最根本的困难，不是 bug，而是数学本质。**

Berry phase $\gamma = \text{Im}\,\ln\det\mathbf{W}$ 是一个整体量（global quantity），将其分解为 $\gamma = \sum_I \gamma^I$ 没有唯一的分解方式。

#### 3.1.1 数学表述

设 Wilson loop 矩阵 $\mathbf{W} \in \mathbb{C}^{N_{\text{occ}} \times N_{\text{occ}}}$，其特征值分解为 $\mathbf{W} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{-1}$，则 Berry phase：

$$\gamma = \arg(\det \mathbf{W}) = \sum_{n=1}^{N_{\text{occ}}} \arg(\lambda_n)$$

逐原子分解需要定义权重 $w^I_n$，使得：

$$\gamma^I = \sum_{n=1}^{N_{\text{occ}}} w^I_n \cdot \arg(\lambda_n), \quad \sum_I w^I_n = 1$$

权重 $w^I_n$ 的定义**不唯一**。DeltaP 使用 SMO 投影权重：

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$$

Wannier90 使用最近原子硬归属：

$$w^I_n = \begin{cases} 1 & \text{if WF}_n\text{ closest to atom }I \\ 0 & \text{otherwise} \end{cases}$$

两种定义都满足 $\sum_I w^I_n = 1$（sum rule），但给出完全不同的逐原子分配。

#### 3.1.2 测试数据

**BN 闪锌矿**（`deltap_results.dat`，nocc=4，8×8×8 k-mesh）：

| 原子 | DeltaP $P_z$ (e/bohr²) | 占比 | Wannier90 WF 数 | Wannier90 占比 |
|------|------------------------|------|----------------|--------------|
| B | $3.756 \times 10^{-4}$ | 57.4% | 4 | 100% |
| N | $2.788 \times 10^{-4}$ | 42.6% | 0 | 0% |
| **总和** | $6.544 \times 10^{-4}$ | 100% | — | — |

Wannier90 将 4 个 WF 全部归属给 B（因 WF 中心更靠近 B），N 极化 = 0。这显然不合理——B-N 键的 4 个成键电子不可能全部属于 B。DeltaP 的 B 57%/N 43% 符合电负性差异（B 2.04, N 3.04），但也不精确。

**H₂O 分子**（`deltap_results.dat`，nocc=4，4×4×4 k-mesh）：

| 原子 | DeltaP $P_z$ (e/bohr²) | 占比 | Wannier90 $P_z$ (e/bohr²) | 占比 |
|------|------------------------|------|--------------------------|------|
| O | $-3.130 \times 10^{-5}$ | 49.2% | $-3.086 \times 10^{-5}$ | 48.5% |
| H1 | $-1.614 \times 10^{-5}$ | 25.4% | $-1.639 \times 10^{-5}$ | 25.8% |
| H2 | $-1.614 \times 10^{-5}$ | 25.4% | $-1.639 \times 10^{-5}$ | 25.8% |
| **总和** | $-6.359 \times 10^{-5}$ | 100% | $-6.363 \times 10^{-5}$ | 100% |

两种方法的逐原子分配在 H₂O 中偏差较小（~3%），但都给出 O ~49% / H ~51%，与物理理想值（O ~85%，因 O 电负性远大于 H）偏差很大。

**液态水**（`deltap_results.dat`，nocc=16，4×4×4 k-mesh）：

| 原子组 | DeltaP $P_z$ (e/bohr²) | 占比 | Wannier90 $P_z$ (e/bohr²) | 占比 |
|-------|------------------------|------|--------------------------|------|
| O 总和 (4个) | $7.085 \times 10^{-5}$ | 43.8% | $1.053 \times 10^{-4}$ | 65.1% |
| H 总和 (8个) | $9.102 \times 10^{-5}$ | 56.2% | $5.654 \times 10^{-5}$ | 34.9% |
| **总和** | $1.619 \times 10^{-4}$ | 100% | $1.619 \times 10^{-4}$ | 100% |

两种方法给出**相反的 O/H 比例**：DeltaP 给 H 更多（56%），Wannier90 给 O 更多（65%）。物理理想值是 O ~85%（O 电负性大，O-H 键电子偏向 O）。两种方法都严重偏离理想值。

#### 3.1.3 结论

逐原子极化分解的本质不唯一性意味着：**没有任何权重定义可以被证明是"物理正确的"**。不同的权重方案只是不同的数学约定，其物理合理性需要通过辅助判据（如声学求和规则、跨基组稳定性）来评估，而非通过"与某个参考一致"来判定。

---

### 3.2 D_mat 违反 Cauchy-Schwarz：基矢不匹配

**当前最严重的实现层面问题，阻塞了 Löwdin 正交化路径。**

#### 3.2.1 数学表述

D_mat 的设计意图是计算 SMO 投影矩阵：

$$D_{a,n}(k) = \langle \alpha_a(k) | \psi_{n,k} \rangle$$

其中 $\alpha_a$ 是原子 $I$ 的第 $a$ 个 SMO 通道（第一 zeta 轨道），$\psi_{n,k}$ 是归一化的占据态波函数。因 $\|\alpha_a\| = \|\psi_n\| = 1$，Cauchy-Schwarz 不等式给出：

$$|D_{a,n}| \leq \|\alpha_a\| \cdot \|\psi_n\| = 1$$

逐原子权重通过 Löwdin 正交化计算：

$$\tilde{D}_{a,n} = \sum_b (S^{-1/2})_{ab} D_{b,n}$$

$$w^I_n = \sum_{a \in I} |\tilde{D}_{a,n}|^2, \quad \sum_I w^I_n = \sum_a |\tilde{D}_{a,n}|^2 = \tilde{\mathbf{D}}^\dagger_n \tilde{\mathbf{D}}_n = \mathbf{D}^\dagger_n S^{-1} \mathbf{D}_n \leq 1$$

这里 $S_{ab} = \langle \alpha_a | \alpha_b \rangle$ 是 SMO 重叠矩阵。

#### 3.2.2 实测数据（H₂O 分子）

从 `deltap_smo_weights.dat` 读取的**原始（非正交）权重** $w^I_n = \sum_{a \in I} |D_{a,n}|^2$：

| 能带 $n$ | $w^O_n$ | $w^{H1}_n$ | $w^{H2}_n$ | $\sum_I w^I_n$ | 归一化 O 占比 |
|---------|---------|-----------|-----------|-------------|------------|
| 0 | 1.935 | 1.636 | 1.636 | **5.207** | 37.2% |
| 1 | 11.735 | 3.306 | 3.306 | **18.347** | 64.0% |
| 2 | 2.307 | 1.966 | 1.966 | **6.240** | 37.0% |
| 3 | 2.247 | 1.936 | 1.936 | **6.119** | 36.7% |

**关键发现**：$\sum_I w^I_n$ 远大于 1（5.2 ~ 18.3），严重违反 sum rule。这说明 $|D_{a,n}|^2$ 之和远超 1，即 $|D_{a,n}|$ 远超 1。

代码中的调试输出（`deltap_wannier.cpp:718`）确认：

```
DeltaP Dmat: m_dim=3 n_dim=4
D_mat[:, 0]: (1.417,0) (-0.076,0) (-0.076,0) ...
|D_mat[:,0]|^2 = 6.05
```

$|D_{\text{mat}[:,0]}|^2 = \sum_a |D_{a,0}|^2 = 6.05 \gg 1$，严重违反 Cauchy-Schwarz。

#### 3.2.3 根因分析

D_mat 通过 `intor_`（`overlap_orb_onsite`：`tabulate(*orb_, *orb_onsite_, 'S')`）计算，代码路径为：

```
compute_real_overlaps():
  intor_->snap(T1, L1, N1, M1, T0, dtau, nlm)
  → nlm[0][iw] = <phi_{T1,L1,N1,M1} | phi_onsite_{T0,iw}>
  
compute_S_k():
  S_k[iat][lm][mu] = overlap_R_[iat][ad].nlm[key][lm]
  → <phi_mu | phi_onsite_lm>

compute_D_I():
  D_I[iat][lm][n] = sum_mu conj(S_k[iat][lm][mu]) * psi_k[mu + n*nrow]
  = sum_mu <phi_onsite_lm | phi_mu> * c_{n,mu}
  = <phi_onsite_lm | psi_n>
```

**数学上**：$D_{a,n} = \langle \phi_{\text{onsite},a} | \psi_n \rangle = \sum_\mu \langle \phi_{\text{onsite},a} | \phi_\mu \rangle c_{n,\mu}$

在非正交 LCAO 基中，$\psi_n = \sum_\mu c_{n,\mu} \phi_\mu$ 的归一化条件是：

$$\langle \psi_n | \psi_n \rangle = \sum_{\mu,\nu} c^*_{n,\mu} \langle \phi_\mu | \phi_\nu \rangle c_{n,\nu} = \mathbf{c}^\dagger_n \mathbf{S}_{\text{LCAO}} \mathbf{c}_n = 1$$

而非 $\sum_\mu |c_{n,\mu}|^2 = 1$。因此 $|c_{n,\mu}|$ 可以大于 1。

但 $|D_{a,n}| = |\langle \phi_{\text{onsite},a} | \psi_n \rangle| \leq \|\phi_{\text{onsite},a}\| \cdot \|\psi_n\| = 1$ 仍然成立——**只要 $\phi_{\text{onsite},a}$ 是真正归一化的**。

$|D_{a,n}|^2 = 6.05 > 1$ 的唯一可能是：**`intor_` 的 `snap()` 返回值不是归一化的重叠积分**，或者 $\psi_n$ 未在 LCAO 基中正确归一化。

**进一步排查**：SMO 重叠矩阵 S 使用 `onsite_onsite_intor_`（`tabulate(*orb_onsite_, *orb_onsite_, 'S')`），实测 S 矩阵性质良好（trace=17, diagonal≈1, 对称, 正定），说明 `onsite_onsite_intor_` 的归一化是正确的。但 `intor_`（`tabulate(*orb_, *orb_onsite_, 'S')`）使用不同的 bra 基组（`orb_` 是完整 LCAO 基），其 `snap()` 返回值可能包含额外的归一化因子。

#### 3.2.4 基矢不匹配的数学表述

Löwdin 正交化要求 $S$ 和 $D$ 在**同一个基矢空间**中定义。当前实现中：

| 量 | 积分器 | 物理含义 | 基矢空间 |
|----|--------|---------|---------|
| $D_{a,n}$ | `intor_` | $\langle \phi_{\text{onsite},a} | \psi_n \rangle$ | onsite ⊗ LCAO |
| $S_{ab}$ | `onsite_onsite_intor_` | $\langle \phi_{\text{onsite},a} | \phi_{\text{onsite},b} \rangle$ | onsite ⊗ onsite |

$S$ 的维数是 $m_{\text{dim}} \times m_{\text{dim}}$（SMO 通道数），$D$ 的维数也是 $m_{\text{dim}} \times N_{\text{occ}}$。形式上可以做 $S^{-1/2} D$，但数学含义不自洽：

$$\tilde{D}_{a,n} = \sum_b (S^{-1/2})_{ab} D_{b,n} = \sum_b (S^{-1/2})_{ab} \langle \phi_{\text{onsite},b} | \psi_n \rangle$$

这要求 $\sum_b (S^{-1/2})_{ab} \langle \phi_{\text{onsite},b} |$ 是某个正交化算符的矩阵元，即 $\langle \tilde{\alpha}_a | = \sum_b (S^{-1/2})_{ab} \langle \phi_{\text{onsite},b} |$。但 $\psi_n$ 生活在 LCAO 空间（非正交），$\tilde{\alpha}_a$ 生活在 onsite 空间（正交化后），两者的内积 $\langle \tilde{\alpha}_a | \psi_n \rangle$ 不能简单地通过 $S^{-1/2} D$ 计算——还需要 LCAO 侧的正交化。

**正确的双正交化公式**应为：

$$\tilde{D}_{a,n} = \sum_{b,\mu,\nu} (S_{\text{onsite}}^{-1/2})_{ab} \langle \phi_{\text{onsite},b} | \phi_\mu \rangle (S_{\text{LCAO}}^{-1/2})_{\mu\nu} c_{n,\nu}$$

即：先对 LCAO 系数做正交化（$S_{\text{LCAO}}^{-1/2} \mathbf{c}$），再用正交化的交叉重叠矩阵投影。

#### 3.2.5 验证数据

代码中的验证输出（`deltap_wannier.cpp:755-783`）确认：

```
DeltaP: S * Sinv max_err = 0.0           (S^{-1} 正确)
DeltaP: S * Sinv * S - S max_err = 0.0   (S^{-1} 正确)
DeltaP: Sinv * Sinv * S - I max_err = 0.0 (S^{-1/2} 正确)
```

SMO 重叠矩阵 S 的计算和 $S^{-1/2}$ 的求取都是正确的。问题出在 $D$ 矩阵的基矢空间不匹配。

```
DeltaP: n=0 raw_sum=6.05 tilde_sum=??  (raw_sum >> 1, Löwdin 修正后仍异常)
```

---

### 3.3 2π 分支切割：特征值级别未解决

#### 3.3.1 数学表述

Wilson loop 特征值 $\lambda_n = |\lambda_n| e^{i\theta_n}$，定义 $\gamma_n = \arg(\lambda_n) \in (-\pi, \pi]$。

当 $\lambda_n$ 沿复平面运动越过负实轴时，$\arg(\lambda_n)$ 跳变 $2\pi$：

$$\arg(\lambda_n) \to \arg(\lambda_n) \pm 2\pi$$

**总量**不受影响：

$$\gamma = \sum_n \arg(\lambda_n) = \arg(\det \mathbf{W}) \pmod{2\pi}$$

不同 $n$ 的 $\pm 2\pi$ 跳变在求和中自动抵消（因 $\det \mathbf{W} = \prod_n \lambda_n$ 的辐角在 $(-\pi, \pi]$ 内）。

**逐原子量**受影响：

$$\gamma^I = \sum_n w^I_n \cdot \arg(\lambda_n)$$

因 $w^I_n$ 对不同 $n$ 不同，某 $n$ 的 $2\pi$ 跳变不被其他 $n$ 的跳变抵消，导致 $\gamma^I$ 突变 $2\pi \cdot w^I_n$。

#### 3.3.2 测试数据

**BaTiO₃ 大位移测试**（$\delta = 0.01$ direct = 0.0794 Bohr，Bug 修复前 3% 误差期）：

| 结构 | Berry $P_{\text{elec}}$ (e/bohr²) | DeltaP $P_{\text{elec}}$ (e/bohr²) | 比例 |
|------|----------------------------------|-----------------------------------|------|
| ref | $-5.790 \times 10^{-3}$ | $-5.636 \times 10^{-3}$ | 0.973 ✅ |
| Ti+0.01 | $-6.719 \times 10^{-3}$ | $-5.063 \times 10^{-3}$ | 0.754 ⚠️ |
| Ba+0.01 | $-7.073 \times 10^{-3}$ | $+3.753 \times 10^{-3}$ | **-0.531 ❌** |

Ba+0.01 的**符号反转**正是 2π 分支跳变的直接后果：某个 $\lambda_n$ 越过负实轴，$\arg(\lambda_n)$ 从 $+\pi^-$ 跳到 $-\pi^+$，导致 $\gamma^I$ 突变约 $2\pi$。

**BaTiO₃ 小位移测试**（$\delta = 0.001$ direct = 0.00794 Bohr）：

| 结构 | Berry $P_{\text{elec}}$ | DeltaP $P_{\text{elec}}$ | 比例 |
|------|------------------------|------------------------|------|
| ref | $-5.790 \times 10^{-3}$ | $-5.682 \times 10^{-3}$ | 0.981 ✅ |
| Ti+0.001 | $-5.882 \times 10^{-3}$ | $-6.125 \times 10^{-3}$ | 1.041 ✅ |
| Ba+0.001 | $-5.919 \times 10^{-3}$ | $-5.520 \times 10^{-3}$ | 0.933 ✅ |

小位移避免了分支跳变（所有比例 ≈ 0.93–1.04），但 $P_{\text{elec}}$ 的 2–7% 误差被 $1/\delta \approx 57133$ 放大。

**H₂O zeta 数据**（`deltap_zeta_debug.dat`）：

所有 16 个 k-string 的 zeta 值相同：$\zeta = 0.9894 - 0.1602i$，$\arg(\zeta) = -0.1605$ rad。这表明 H₂O（孤立分子，平坦能带）的 Wilson loop 接近单位矩阵，特征值 $\lambda_n \approx 1$，无分支跳变问题。

**液态水 zeta 数据**（`deltap_zeta_debug.dat`）：

前 4 个 string：$\zeta \approx 0.1776 - 1.0011i$，$\arg(\zeta) \approx -1.3952$ rad。
后 3 个 string：$\zeta \approx 1.0080 + 0.1851i$，$\arg(\zeta) \approx 0.1816$ rad。

不同 string 的 $\arg(\zeta)$ 差异大（$-1.40$ vs $+0.18$），但都在 $(-\pi, \pi]$ 内，无 2π 跳变。然而，逐特征值的 $\arg(\lambda_n)$ 可能有跳变（zeta 是所有特征值乘积，个体跳变在乘积中抵消）。

#### 3.3.3 已尝试方案及结果

| 方案 | 方法 | 结果 | 失败原因 |
|------|------|------|---------|
| 方案 A：相位展开 | 沿 k-string 逐步对角化 $\mathbf{W}_j$，用 $\arg(\lambda_n^{(j)} / \lambda_n^{(j-1)})$ 跟踪连续相位 | H₂O (4 bands) 成功；液态水 (16 bands) 失败 | 16 bands 时特征值交叉，最近邻匹配错乱 |
| zeta 级别 unwrap | berry_phase 的"除以平均"方法 | BaTiO₃ 使结果变差（比例从 0.97 变 -1.41） | 不同能带的 arg 跳变不均匀，比例缩放扭曲分配 |
| 比例缩放 | $\gamma^I_{\text{corrected}} = \gamma^I_{\text{raw}} \times \arg(\zeta) / \sum_I \gamma^I_{\text{raw}}$ | 总量正确，逐原子分配被等比例扭曲 | 假设所有原子按相同比例修正，实际不是 |

#### 3.3.4 根本困难

特征值级别的 2π 跳变需要**跨结构特征值跟踪**：

$$\Delta\gamma_n = \arg\left(\frac{\lambda_n^{\text{disp}}}{\lambda_n^{\text{ref}}}\right) \in (-\pi, \pi]$$

匹配通过特征向量重叠矩阵 $U_{mn} = \langle v_m^{\text{ref}} | v_n^{\text{disp}} \rangle$：$|U_{mn}|$ 最大的匹配对即为同一能带。

**困难**：当 $\lambda_m^{\text{ref}} \approx \lambda_{m'}^{\text{ref}}$（简并或近简并）时，特征向量 $|v_m\rangle, |v_{m'}\rangle$ 可以任意旋转，$U_{mn}$ 不确定。此时匹配失败。

---

### 3.4 Born 有效电荷 $Z^*$ 不可靠

#### 3.4.1 数学表述

Born 有效电荷定义为：

$$Z^*_{I,\alpha\beta} = \frac{\Omega}{e} \frac{\partial P_\alpha}{\partial \tau_{I,\beta}} \approx \frac{\Omega}{e} \frac{P_\alpha^{\text{disp}} - P_\alpha^{\text{ref}}}{\delta}$$

其中 $\delta$ 是原子 $I$ 沿方向 $\beta$ 的位移（Bohr）。

误差传播：

$$\delta Z^*_I = \frac{\Omega}{e} \frac{\delta P^{\text{disp}} - \delta P^{\text{ref}}}{\delta} \approx \frac{\Omega}{e} \frac{2 \epsilon_P}{\delta}$$

其中 $\epsilon_P$ 是单次极化计算的误差。当 $\epsilon_P / P \sim 0.1\%$ 时：

| 位移 $\delta$ (Bohr) | $1/\delta$ | $\delta Z^* / Z^*$（估计） |
|---------------------|-----------|------------------------|
| 0.0794 (0.01 direct) | 12.6 | ~1.3% |
| 0.00794 (0.001 direct) | 125.9 | ~12.6% |

但这是**总量**的误差估计。**逐原子** $Z^*_I$ 的误差取决于逐原子极化的精度，而逐原子分配本身就有 7-22% 的不确定性（见 3.1），使得 $Z^*_I$ 的实际误差远大于上述估计。

#### 3.4.2 测试数据（BaTiO₃，Bug 修复前 3% 误差期）

**大位移 ($\delta = 0.0794$ Bohr)**：

| 量 | berry_phase | DeltaP | 误差 |
|----|-----------|--------|------|
| $Z^*_{\text{Ti}}$ | 2.69 | 3.28 | 22% |
| $Z^*_{\text{Ba}}$ | 0.67 | 53.65 | **7866%** |

$Z^*_{\text{Ba}}$ 的 7866% 误差来自 Ba+0.01 结构的符号反转（3.3 中的 2π 分支跳变）。

**小位移 ($\delta = 0.00794$ Bohr)**：

| 量 | berry_phase | DeltaP | 误差 |
|----|-----------|--------|------|
| $Z^*_{\text{Ti}}$ | 2.80 | -25.29 | **1004%** |
| $Z^*_{\text{Ba}}$ | 0.67 | 9.26 | **1274%** |

小位移避免了分支跳变（所有 ratio ≈ 0.93–1.04），但 $P_{\text{elec}}$ 的 3% 误差被 $1/\delta = 125.9$ 放大到 ~378%，再叠加逐原子分配的不确定性，总误差超过 1000%。

#### 3.4.3 Bug 修复后的预期

Bug 修复后总量精度从 3% 提升到 0.1%。理论上：
- 大位移：$Z^*$ 误差 ~0.1% × 12.6 ≈ 1.3%（仅总量误差，不含逐原子不确定性）
- 小位移：$Z^*$ 误差 ~0.1% × 125.9 ≈ 12.6%

但 BaTiO₃ 的 $Z^*$ 测试尚未在 Bug 修复后重新运行。且即使总量误差降至 1%，逐原子分配的 7-22% 不确定性（见 3.1）仍使逐原子 $Z^*_I$ 不可靠。

---

### 3.5 SMO 权重的基组依赖性

#### 3.5.1 数学表述

SMO 投影权重：

$$w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$$

依赖于 SMO 基函数 $\alpha_a$ 的选择：
- **轨道截断半径** $r_{\text{cut}}$：增大 $r_{\text{cut}}$ 使 $\alpha_a$ 与更多近邻原子重叠，改变权重分配
- **zeta 数量**：多 zeta 基组下 SMO 只取第一 zeta，不同基组的第一 zeta 性质不同
- **角动量截断** $l_{\text{max}}$：$n_{\text{proj}} = (l_{\text{max}}+1)^2$，不同 $l_{\text{max}}$ 给出不同的投影通道数

这与 Mulliken 布居分析的基组依赖性在数学上同构：

$$w^I_{\text{Mulliken}} = \sum_{\mu \in I} \sum_\nu D_{\mu\nu} S_{\mu\nu}$$

Mulliken 权重和 SMO 权重都依赖于非正交基的重叠矩阵，都随基组改变而变化。

#### 3.5.2 理论分析

Löwdin 正交化（$S^{-1/2}$）可以**部分**消除基组依赖：

$$w^I_{n,\text{Löwdin}} = \sum_{a \in I} |\tilde{D}_{a,n}|^2, \quad \tilde{\mathbf{D}} = S^{-1/2} \mathbf{D}$$

Löwdin 正交化后的权重满足 $\sum_I w^I_n \leq 1$（等号在 SMO 完备时成立），且对基组变化更稳健。但由于 3.2 的基矢不匹配问题，Löwdin 方案当前无法正确实施。

#### 3.5.3 当前状态

当前代码使用**归一化的非正交权重**作为 fallback：

$$w^I_{n,\text{norm}} = \frac{w^I_n}{\sum_J w^J_n} = \frac{\sum_{a \in I} |D_{a,n}|^2}{\sum_a |D_{a,n}|^2}$$

这是 Mulliken 式的归一化，不消除基组依赖性。实测 H₂O 的归一化权重：

| 能带 $n$ | O 归一化权重 | H1 | H2 | O 占比 |
|---------|-----------|-----|-----|------|
| 0 | 0.372 | 0.314 | 0.314 | 37.2% |
| 1 | 0.640 | 0.180 | 0.180 | 64.0% |
| 2 | 0.370 | 0.315 | 0.315 | 37.0% |
| 3 | 0.367 | 0.316 | 0.316 | 36.7% |
| **平均** | — | — | — | **43.7%** |

O 平均占比 43.7%，远低于物理理想值 ~85%。这表明 SMO 权重方法在 H₂O 中系统性低估 O 的贡献。

---

### 3.6 液态水 O/H 分配反转

#### 3.6.1 测试数据

从 `deltap_results.dat` 读取的逐原子极化：

| 原子 | 类型 | DeltaP $P_z$ (e/bohr²) | Wannier90 $P_z$ (e/bohr²) |
|------|------|------------------------|--------------------------|
| O0 | O | $1.641 \times 10^{-5}$ | — |
| O1 | O | $2.088 \times 10^{-5}$ | — |
| O2 | O | $1.564 \times 10^{-5}$ | — |
| O3 | O | $1.792 \times 10^{-5}$ | — |
| **O 总和** | | $7.085 \times 10^{-5}$ (**43.8%**) | $1.053 \times 10^{-4}$ (**65.1%**) |
| H0 | H | $9.923 \times 10^{-6}$ | — |
| H1 | H | $9.168 \times 10^{-6}$ | — |
| H2 | H | $1.374 \times 10^{-5}$ | — |
| H3 | H | $1.060 \times 10^{-5}$ | — |
| H4 | H | $1.383 \times 10^{-5}$ | — |
| H5 | H | $1.177 \times 10^{-5}$ | — |
| H6 | H | $8.018 \times 10^{-6}$ | — |
| H7 | H | $1.397 \times 10^{-5}$ | — |
| **H 总和** | | $9.102 \times 10^{-5}$ (**56.2%**) | $5.654 \times 10^{-5}$ (**34.9%**) |
| **总和** | | $1.619 \times 10^{-4}$ | $1.619 \times 10^{-4}$ |

两种方法给出**相反的 O/H 比例**。物理理想值是 O ~85%（O 电负性 3.44，H 电负性 2.20，O-H 键电子密度偏向 O）。

#### 3.6.2 反转原因分析

**DeltaP H 偏高的机制**：

SMO 权重 $w^I_n = \sum_{a \in I} |\langle v_n | \alpha_a \rangle|^2$ 中，$\alpha_a$ 是 H 的 s 轨道。H 的轨道截断半径（6 Bohr，来自 `H_gga_6au_100Ry_2s1p.orb`）较大，使得 H 的 s 轨道在空间上延伸到 O-H 键中心区域。O-H 键的 WF（特征向量 $|v_n\rangle$）在键中心有较大振幅，与 H 的 s 轨道重叠较大，导致 H 获得过高的权重。

**Wannier90 O 偏高的机制**：

Wannier90 的 16 个 WF 中，11 个归属给 O（最近原子），5 个给 H。4 个 H 原子获得 0 个 WF（极化 = 0）。这种硬归属在多原子分子中系统性偏向重原子（O 的孤对电子 WF 明确在 O 上，但 O-H 键 WF 也被归属给 O 因 O 更重、WF 中心更靠近 O）。

#### 3.6.3 物理理想值的估计

液态水中每个 O-H 键的电子密度重心偏向 O 约 0.3 Å（从 Wannier90 位移数据可见 $\delta_{\text{H}} \approx 0.3$ Å）。因此 O 应获得约 85% 的电子极化贡献：

$$\text{O 占比}_{\text{ideal}} \approx \frac{N_{\text{lone pair}} \times 1.0 + N_{\text{O-H bond}} \times 0.7}{N_{\text{total}}} = \frac{2 \times 4 \times 1.0 + 8 \times 0.7}{16} = \frac{8 + 5.6}{16} = 85\%$$

DeltaP 给出 43.8%，偏差 $|85 - 43.8| = 41.2$ 个百分点。Wannier90 给出 65.1%，偏差 $|85 - 65.1| = 19.9$ 个百分点。两种方法都严重偏离物理理想值。

---

### 3.7 Wannier90 硬归属的系统性缺陷

#### 3.7.1 数学表述

Wannier90 的逐原子分配使用最近原子判据：

$$w^I_n = \begin{cases} 1 & \text{if } I = \arg\min_J |\langle \mathbf{r} \rangle_n - \boldsymbol{\tau}_J| \\ 0 & \text{otherwise} \end{cases}$$

其中 $\langle \mathbf{r} \rangle_n$ 是第 $n$ 个 WF 的中心。

这导致：
- $\sum_I w^I_n = 1$（sum rule 满足）
- 但 $w^I_n \in \{0, 1\}$（离散），不反映键合轨道的共享性质
- 部分 $I$ 获得 $w^I_n = 0$（无 WF 归属），极化贡献为零

#### 3.7.2 测试数据

**BN 闪锌矿**：4 个 WF 中心全部更靠近 B（距 B 0.04-0.42 Å，距 N 0.95-1.52 Å），全部归属给 B：

| WF | 距 B (Å) | 距 N (Å) | 归属 |
|----|---------|---------|------|
| 1 | 0.047 | 1.52 | B |
| 2 | 0.42 | 0.95 | B |
| 3 | 0.42 | 0.95 | B |
| 4 | 0.42 | 0.95 | B |

结果：$w^B_n = 1$（所有 $n$），$w^N_n = 0$（所有 $n$），N 极化 = 0。

**液态水**：16 个 WF 的归属分布：

| 原子 | 类型 | WF 数 | WF 数 | WF 数 | WF 数 |
|------|------|-------|-------|-------|-------|
| O0 | O | 4 | O2 | O | 2 |
| O1 | O | 2 | O3 | O | 3 |
| H0 | H | 0 | H4 | H | 1 |
| H1 | H | 0 | H5 | H | 1 |
| H2 | H | 1 | H6 | H | 0 |
| H3 | H | 1 | H7 | H | 1 |

3 个 H 原子（H0, H1, H6）获得 0 个 WF，极化贡献 = 0。O1 获得 2 个 WF，O0 获得 4 个，差异无物理原因（都是等价 O 原子）。

#### 3.7.3 失败的物理根源

WF 中心 $\langle \mathbf{r} \rangle_n$ 是**绝对空间位置**，不是相对于某个原子的位置。当 WF 是键合轨道时，$\langle \mathbf{r} \rangle_n$ 位于两个原子之间，其极化贡献 $-e\langle r_n \rangle / \Omega$ **同时属于两个原子**。强行归属给最近原子（0 或 1）丢失了这种共享性质。

对于离子晶体（如 NaCl），WF 高度局域化在原子位置附近，硬归属合理。对于共价/极性共价体系（BN, H₂O），WF 是键合轨道，硬归属不合理。

---

## 4. Wannier 中心作为参考的适用性分析

### 4.1 可以作为参考的场景

| 场景 | 原因 | 验证 |
|------|------|------|
| **总极化**（无 disentanglement） | $\arg(\det \mathbf{W})$ 与 Berry phase 等价 | H₂O: 0.25%, 液态水: 0.01% |
| **孤立带体系** | 无 disentanglement，子空间跨 k 一致 | H₂O (gap 7.8 eV), 液态水 |
| **离子晶体** | WF 为原子轨道样，硬归属合理 | — |
| **WF 为原子轨道**（孤对电子、芯态） | 空间位置明确，归属无歧义 | H₂O 的 2 个 O 孤对 WF |
| **逐能带 Berry phase** $\gamma_n$ | $\langle r_n \rangle = \frac{a}{2\pi}\arg(\lambda_n)$ 数学严格 | 与 DeltaP 特征值一致 |

### 4.2 不能作为参考的场景

| 场景 | 原因 | 后果 |
|------|------|------|
| **需要 disentanglement 的体系** | 子空间跨 k 点不一致，Wilson loop 连续性被破坏 | BN: 74% 总极化偏差 |
| **共价键体系**（逐原子分解） | 键合 WF 无法归属给单一原子 | BN: N = 0%，显然错误 |
| **多原子分子**（逐原子分解） | 部分 H 原子获得 0 个 WF | 液态水: 4 个 H 的极化 = 0 |
| **逐原子**总**极化** | 离子相位 mod 归约使逐原子之和 ≠ 总值 | H₂O: 逐原子离子和 = 2.078 ≠ 0.078 |
| **跨基组对比** | WF 中心依赖于初始投影选择 | 不同投影给不同 WF 中心 |
| **近简并能带** | Wannierization 可能不收敛 | — |

### 4.3 关键判断

**Wannier 中心作为"总极化"的参考是可靠的**（在无 disentanglement 的前提下），但作为"逐原子极化分解"的参考是**不可靠的**——硬归属在共价体系和多原子分子中系统性失败。

更准确地说：Wannier90 提供的 $\gamma_n = -2\pi \langle r_n \rangle / R$ 是**逐能带**的 Berry phase，这是严格正确的。但将 $\gamma_n$ 分配给各原子（即定义 $\gamma^I = \sum_{n \in I} \gamma_n$）需要"WF 归属"，这一步是不唯一且有缺陷的。

**结论**：Wannier 中心可以作为**逐能带 Berry phase** 的参考（验证 $\gamma_n = \arg(\lambda_n)$），但不能作为**逐原子极化**的参考（因为归属方案不合理）。

---

## 5. 后续需要重点解决的算法问题

### 5.1 问题定义：三层困难

DeltaP 的困难可以清晰地分为三个层次：

| 层次 | 困难 | 性质 | 难度 |
|------|------|------|------|
| **L1: 总量** | 总电子极化精度 | ✅ 已解决 (≤0.1%) | — |
| **L2: 逐能带** | $\gamma_n = \arg(\lambda_n)$ 的 2π 分支 | 数学本质 + 实现 | 高 |
| **L3: 逐原子** | $\gamma^I = \sum_n w^I_n \gamma_n$ 的权重定义 | 数学本质（不唯一） | 极高 |

L2 和 L3 的困难性质不同：L2 是技术性的（可通过更好的算法解决），L3 是本质性的（没有唯一正确答案）。

### 5.2 问题 1：D_mat 基矢不匹配（L3，实现层面）

**问题**：D_mat 使用 `intor_`（$\langle \phi_\mu | \phi_{\text{onsite}} \rangle$）计算，生活在完整 LCAO 空间；SMO 重叠矩阵 S 使用 `onsite_onsite_intor_`（$\langle \phi_{\text{onsite}} | \phi_{\text{onsite}} \rangle$），生活在 onsite 子空间。两者维数不同、基矢不同，直接做 $S^{-1/2} D$ 不自洽。

**解决方向**：

1. **方案 A：双正交化**
   - 计算 LCAO 重叠矩阵 $S_{\text{LCAO}} = \langle \phi_\mu | \phi_\nu \rangle$（已存在于 ABACUS 的 S 矩阵中）
   - 计算 LCAO-onsite 交叉重叠 $C_{\mu,a} = \langle \phi_\mu | \phi_{\text{onsite},a} \rangle$（即当前的 `intor_`）
   - 计算 onsite-onsite 重叠 $S_{\text{onsite}} = \langle \phi_{\text{onsite},a} | \phi_{\text{onsite},b} \rangle$（即当前的 `onsite_onsite_intor_`）
   - 正确的投影：$\tilde{D}_{a,n} = \sum_\mu (S_{\text{onsite}}^{-1/2})_{ab} C_{\mu,b} (S_{\text{LCAO}}^{-1/2})_{\mu\nu} c_{n,\nu}$
   - 即：先将 LCAO 系数正交化（$S_{\text{LCAO}}^{-1/2} c$），再用正交化的交叉重叠投影

2. **方案 B：放弃 Löwdin，使用规范固定**
   - 不做正交化，而是为每个能带 $n$ 选择一个"锚定原子"（SMO 投影最大的原子）
   - 该能带的全部 Berry phase 归属给锚定原子
   - 类似 Wannier90 的硬归属，但基于 SMO 投影而非空间距离
   - 优点：简单，无基矢不匹配问题
   - 缺点：离散归属，在共价体系中可能不合理

3. **方案 C：使用 Wannier90 的 WF 中心 + SMO 权重**
   - 从 Wannier90 获取 WF 中心 $\langle r_n \rangle$（逐能带，严格正确）
   - 用 SMO 权重 $w^I_n$（修正后的）将 $\langle r_n \rangle$ 分配给各原子
   - 优点：$\gamma_n$ 严格正确（来自 Wannier90），只需解决权重
   - 缺点：依赖外部 Wannier90，且 SMO 权重仍有基组依赖

**推荐**：方案 A 最严格但实现复杂；方案 C 最实用但引入外部依赖。建议先实现方案 A 验证是否解决 D_mat > 1 问题，若成功则作为主方案。

### 5.3 问题 2：2π 分支切割的特征值级别跟踪（L2）

**问题**：$\arg(\lambda_n)$ 的 $2\pi$ 跳变在逐原子分解中不抵消，导致逐原子极化错误。

**已尝试方案**：
- 沿 k-string 逐步跟踪：4 bands 成功，16 bands 失败（特征值交叉）
- zeta 级别 unwrap + 比例缩放：使结果变差

**解决方向**：

1. **跨结构特征值跟踪**
   - 对参考结构和位移结构分别对角化 $\mathbf{W}$
   - 用特征向量重叠 $U_{mn} = \langle v_m^{\text{ref}} | v_n^{\text{disp}} \rangle$ 匹配特征值
   - 差分 $\Delta\gamma_n = \arg(\lambda_n^{\text{disp}}) - \arg(\lambda_n^{\text{ref}})$ 消除 $2\pi$ 跳变
   - $Z^*_I \propto \sum_n w^I_n \Delta\gamma_n$
   - **困难**：简并/近简并时特征向量混合，匹配不确定

2. **Wilson loop 矩阵级别的跨结构跟踪**
   - 不在特征值级别跟踪，而在 $\mathbf{W}$ 矩阵级别跟踪
   - $\Delta \ln \det \mathbf{W} = \text{Tr}(\mathbf{W}^{-1} \Delta \mathbf{W})$（矩阵对数微分）
   - 逐原子：$\Delta \gamma^I = \sum_n w^I_n \cdot \text{Im}[\text{Tr}_n(\mathbf{W}^{-1} \Delta \mathbf{W})]$
   - **困难**：$\text{Tr}_n$ 需要特征值分解，回到原问题

3. **增大 k-mesh 使分支跳变消失**
   - 更密的 k-mesh 使 $\lambda_n$ 更接近 1（$|\arg(\lambda_n)|$ 更小），减少越过负实轴的概率
   - **困难**：计算量增大；不能根本解决问题

**推荐**：方案 1（跨结构特征值跟踪）是最直接的解决方案。对于简并情况，可以在简并子空间内用子空间对角化处理（简并特征值相同，子空间内任意线性组合不改变 $\sum_{n \in \text{deg}} w^I_n \gamma_n$）。

### 5.4 问题 3：逐原子极化分解的唯一性（L3，数学本质）

**问题**：$\gamma = \sum_I \gamma^I$ 的分解不唯一，不同权重给出不同分配，没有"正确答案"。

**这是数学本质问题，无法通过算法完全解决。** 但可以通过以下策略减轻：

1. **定义"最优"分解的物理标准**
   - 要求 $Z^*_I$ 满足声学求和规则（$\sum_I Z^*_{I,\alpha\beta} = 0$ for $\alpha \neq \beta$）
   - 要求 $Z^*_I$ 接近化学直觉（O 的 $Z^*$ 应大于 H 的）
   - 要求跨基组稳定性（不同基组给出相近的 $Z^*_I$）

2. **使用约束优化**
   - 在 sum rule $\sum_I \gamma^I = \gamma$ 约束下，最小化某种物理目标函数
   - 例如：最小化 $\sum_I |Z^*_I - Z^{\text{PaO}}_I|^2$（与 PaO 方法的偏差）
   - 或：最小化逐原子极化的基组依赖性

3. **接受不唯一性，提供多种分解**
   - 同时输出 SMO 权重、Wannier90 硬归属、Resta-Z 的结果
   - 让用户根据体系特性选择
   - 明确标注每种方法的适用范围和限制

**推荐**：方案 3 最务实。在文档中明确说明"逐原子极化分解不唯一"，提供多种方案供用户选择，而非追求单一"正确"答案。

### 5.5 问题 4：跨基组可转移性（L3，实现层面）

**问题**：用户要求方法跨基组可转移。SMO 权重依赖于 SMO 基函数的选择，类似于 Mulliken 布居的基组依赖。

**解决方向**：

1. **Löwdin 正交化**（如果 5.2 的基矢问题解决）：$S^{-1/2}$ 可以部分消除基组依赖
2. **使用规范不变的量**：如 Resta 的 $|z|^2$（局域化指示符），不依赖基组
3. **使用 Wannier90 的 WF 中心**：WF 中心在收敛后是基组不变的（但依赖初始投影）

**推荐**：解决 5.2 后验证 Löwdin 方案的跨基组稳定性。如果 Löwdin 仍有基组依赖，则需要寻找完全不同的分解方案（如基于密度泛函理论的分解）。

### 5.6 问题优先级排序

| 优先级 | 问题 | 影响 | 难度 | 建议时间线 |
|--------|------|------|------|-----------|
| **P0** | D_mat 基矢不匹配 (5.2) | 阻塞 Löwdin 正交化 | 中 | 1-2 周 |
| **P1** | 2π 分支切割 (5.3) | Z* 不可靠 | 高 | 2-4 周 |
| **P2** | 跨基组可转移性 (5.5) | 方法普适性 | 高 | 4-8 周 |
| **P3** | 分解唯一性 (5.4) | 物理正确性 | 极高 | 长期/接受 |

---

## 6. 对 Wannier90 逐原子分解的批判性评估

### 6.1 Wannier90 的三层能力

| 层次 | 能力 | 可靠性 |
|------|------|--------|
| 总极化（无 disentanglement） | ✅ $\arg(\det \mathbf{W})$ | 高 (≤0.25%) |
| 逐能带 Berry phase $\gamma_n$ | ✅ $\langle r_n \rangle = \frac{a}{2\pi}\arg(\lambda_n)$ | 高（数学严格） |
| 逐原子极化 $\gamma^I$ | ❌ 硬归属 | 低（共价体系失效） |

### 6.2 硬归属失败的物理根源

Wannier90 的 WF 中心 $\langle r_n \rangle$ 是**绝对空间位置**，不是相对于某个原子的位置。当 WF 是键合轨道时，$\langle r_n \rangle$ 位于两个原子之间，其极化贡献 $-e\langle r_n \rangle / \Omega$ **同时属于两个原子**。强行归属给最近原子（0 或 1）丢失了这种共享性质。

DeltaP 的 SMO 权重给出连续分配（0~1），在概念上更合理，但 SMO 权重本身也有问题（基组依赖、H 偏高）。

### 6.3 何时可以使用 Wannier90 逐原子分解

**可以使用**：
- 离子晶体（NaCl, MgO）：WF 为原子轨道样，硬归属无歧义
- 孤对电子（H₂O 的 O 孤对）：WF 明确在 O 上
- 芯态电子：WF 高度局域化

**不应使用**：
- 共价键（BN, Si, C-C）：键合 WF 跨越两原子
- 氢键体系（液态水）：部分 H 获得 0 个 WF
- 金属/半金属：WF 非局域化

### 6.4 Wannier90 作为 DeltaP 验证标准的合理性

| 验证目的 | Wannier90 是否合适 | 原因 |
|---------|-------------------|------|
| 总极化 | ✅ 合适（无 disentanglement 时） | $\arg(\det \mathbf{W})$ 严格正确 |
| 逐能带 $\gamma_n$ | ✅ 合适 | $\langle r_n \rangle$ 严格正确 |
| 逐原子 $\gamma^I$ | ⚠️ 仅作参考 | 硬归属不唯一，但可作为"另一种分解方案"对比 |
| 逐原子位移 $\delta^I$ | ❌ 不合适 | 硬归属导致部分原子 $\delta = 0$ |

---

## 7. 总结与建议

### 7.1 当前成果

DeltaP 算法在**总电子极化**计算上已经成功，与 Berry Phase 精确一致（≤0.1%），适用于所有体系类型（周期性晶体、分子晶体、孤立分子）。算法 D（Wilson loop 特征值分解）的理论框架是正确的，具有规范不变性、精确 sum rule、与 Wannier center 等价等优良性质。

### 7.2 当前差距

逐原子分解层面存在三个层次的困难：
1. **实现层面**：D_mat 基矢不匹配阻塞了 Löwdin 正交化
2. **技术层面**：2π 分支切割在特征值级别未解决
3. **本质层面**：极化分解不唯一，没有"正确答案"

### 7.3 建议的后续路线

**短期（1-2 月）**：
1. 解决 D_mat 基矢不匹配（方案 A：双正交化），验证 $|D_{a,n}| \leq 1$
2. 实现跨结构特征值跟踪，解决 2π 分支切割
3. 在 BaTiO₃ 上重新测试 $Z^*$，验证精度提升

**中期（3-6 月）**：
1. 验证 Löwdin 方案的跨基组可转移性（更换轨道截断/zeta 数量）
2. 扩展测试体系（金属氧化物、铁电体、界面体系）
3. 与 ABACUS 已有的 PaO（PBE+轨道）$Z^*$ 方法对比

**长期**：
1. 如果 Löwdin 方案跨基组不稳定，探索基于密度泛函理论的分解方案
2. 考虑将 Wannier90 WF 中心 + 修正后的 SMO 权重作为组合方案
3. 接受逐原子分解的不唯一性，提供多种方案供用户选择

### 7.4 最终判断

DeltaP 的**总量计算**已达到实用水平。**逐原子分解**仍处于研究阶段，需要解决基矢不匹配和分支切割两个关键技术问题后才能评估其实用性。逐原子分解的不唯一性是数学本质问题，应通过明确定义适用范围和提供多种方案来处理，而非追求单一"正确"答案。

Wannier90 可作为**总极化和逐能带 Berry phase** 的可靠参考（在无 disentanglement 的前提下），但**不应作为逐原子极化分解的绝对标准**——其硬归属方案在共价体系和多原子分子中存在系统性缺陷。

---

## 附录 A: 文档索引

| 文档 | 内容 | 状态 |
|------|------|------|
| `2026-06-28-deltap-algorithm-evaluation.md` | 6 种算法理论分析 + 三体系测试 + Bug 修复 | §1-10 已被 §11 修正 |
| `2026-06-30-deltap-berryphase-wannier90-comparison-guide.md` | 三方法对比公式与方法论 | 有效 |
| `2026-06-30-deltap-three-system-test-results.md` | 三体系完整测试数据 | 有效 |
| `2026-07-02-deltap-critical-evaluation-report.md` | 本报告：批判性评估 | 当前 |

## 附录 B: 代码文件索引

| 文件 | 内容 | 关键函数 |
|------|------|---------|
| `deltap.h` | 类定义 | DeltaP, AtomicPolarization |
| `deltap.cpp` | 初始化 + k-string | `init()`, `setup_kstring()` |
| `deltap_wannier.cpp` | Wilson loop 主逻辑 | `compute_wannier_polarization()`, `compute_S_dk_link()`, `compute_resta_z()` |
| `deltap_overlap.cpp` | SMO 重叠矩阵 | `compute_real_overlaps()`, `compute_smo_overlap_matrix()` |
| `deltap_berry.cpp` | S_k, D_I, Berry connection | `compute_S_k()`, `compute_D_I()` |
| `deltap_io.cpp` | 输出 | `write_results()` |
| `two_center_bundle.cpp` | 积分器构建 | `overlap_onsite_onsite` 新增 |
| `ctrl_scf_lcao.cpp` | DeltaP 入口 | init 调用 |
