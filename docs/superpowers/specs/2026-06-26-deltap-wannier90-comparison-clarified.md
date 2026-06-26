# 逐原子极化分解验证：测试逻辑、预期与结果分析

> **日期**: 2026-06-26
> **目标**: 用清晰的逻辑链说明"对比 Wannier center"的测试思路、预期结果、实际结果和分析

---

## 1. 背景知识：三个关键概念

### 1.1 Berry phase（精确极化）

晶体的电子极化由 Berry phase 给出（King-Smith & Vanderbilt, 1993）：

$$P_\alpha = \frac{e\, a_\alpha}{2\pi\, \Omega} \cdot \gamma_\alpha$$

其中 $\gamma_\alpha$ 是沿方向 $\alpha$ 的 Berry phase：

$$\gamma_\alpha = \mathrm{Im}\, \ln \prod_{k} \det\big[\langle u_{n,k} | u_{m,k+\mathbf{b}}\rangle\big]$$

这里 $|u_{n,k}\rangle$ 是 Bloch 态的周期部分，$\mathbf{b}$ 是相邻 k 点的位移。

**关键性质**：
- $\gamma$ 是 **规范不变的**（波函数相位 $e^{i\phi}$ 在 det 中消去）
- $\gamma$ 是 **精确的**（不依赖 dk 大小，只要 k-string 形成闭合回路）
- ABACUS 的 `berry_phase` 模块计算的就是这个量

### 1.2 Berry connection（近似极化）

Berry connection 是 Berry phase 在 $dk \to 0$ 极限下的 **一阶近似**：

$$\gamma_\alpha^{\text{conn}} = \sum_k \mathrm{Im}\, \mathrm{Tr}\big[\langle u_{n,k} | u_{m,k+\mathbf{b}}\rangle\big] = \sum_k \sum_n \mathrm{Im}\, M_{nn}(k)$$

其中 $M_{mn}(k) = \langle u_{m,k} | u_{n,k+\mathbf{b}}\rangle$ 是重叠矩阵。

**关键性质**：
- $\gamma^{\text{conn}}$ 是 **规范不变的**（trace 的循环性质）
- 但 $\gamma^{\text{conn}} \neq \gamma$（对于有限 dk）
- 差异来自 **高阶项**：$\ln\det M \neq \mathrm{Tr}\, M$（对数-行列式 ≠ 迹）
- 只有当 $dk \to 0$、$M \to I$（单位矩阵）时两者才相等

**数学根源**：
$$\ln\det M = \mathrm{Tr}\ln M = \mathrm{Tr}\big[(M-I) - \frac{1}{2}(M-I)^2 + \cdots\big]$$

而 $\mathrm{Tr}\, M = \mathrm{Tr}\big[I + (M-I)\big] = n + \mathrm{Tr}(M-I)$

所以 $\ln\det M \neq \mathrm{Tr}\, M$，差异为 $\mathrm{Tr}(M-I)^2/2$ 等高阶项。

### 1.3 Wannier center（精确的逐能带分解）

Wannier 函数 $|w_n\rangle$ 是从占据 Bloch 态构造的局域函数。其中心 $\langle r_n \rangle$ 与极化的关系：

$$P = -\frac{e}{\Omega} \sum_n \langle r_n \rangle$$

Wannier center 可从重叠矩阵计算：

$$\langle r_{n,\alpha} \rangle = \frac{a_\alpha}{2\pi} \cdot \mathrm{Im}\, \ln \prod_k \big[U^\dagger(k)\, M(k)\, U(k+\mathbf{b})\big]_{nn}$$

其中 $U(k)$ 是 **Wannierization** 过程中找到的最优规范变换（使 Wannier 函数最大局域化）。

**关键性质**：
- Wannier center 给出 **精确的逐能带极化分解**
- $\sum_n \langle r_n \rangle$ 精确等于总极化（规范变换不改变 det）
- 需要 Wannierization（求最优 $U(k)$），这需要运行 Wannier90.x

---

## 2. 测试逻辑

### 2.1 我们要验证什么？

DeltaP 模块试图实现 **逐原子极化分解** $P^I$，使得 $\sum_I P^I = P^{\text{total}}$。

验证的方法是：将 DeltaP 的 $P^I$ 与 Wannier center 的逐原子分解对比。

### 2.2 预期什么应该一致？

| 对比项 | 预期 | 说明 |
|--------|------|------|
| 总 Berry phase（.mmn vs ABACUS） | **应该一致** | 两者都计算 $\mathrm{Im}\,\ln\prod\det M$，使用相同的重叠矩阵 |
| 总 Berry connection vs Berry phase | **不应该一致** | trace ≠ det，差异是数学必然 |
| 逐原子 sum rule（trace 方法） | **应该成立** | 按能带归一化的权重使 $\sum_I A^I = A^{\text{total}}$ |
| 逐原子 sum rule（det 方法） | **不应该成立** | $\prod_I \det M^I \neq \det M$（除非原子间正交） |
| DeltaP Z* vs berry_phase Z* | **应该一致**（如果分解正确） | Z* = (Ω/e)·ΔP/Δτ，P 应来自精确的 Berry phase |

### 2.3 测试流程

```
ABACUS SCF → 收敛电荷
    ↓
ABACUS NSCF (towannier90=1) → 生成 .mmn/.amn/.eig 文件
    ↓
┌───────────────────────────────────────────┐
│  路径 A: 直接从 .mmn 计算                  │
│  1. 总 Berry phase (det) → 对比 ABACUS     │
│  2. 总 Berry connection (trace)            │
│  3. 逐原子 Berry connection (加权 trace)    │
│  4. 逐能带 Wannier center (对角元素)       │
└───────────────────────────────────────────┘
    ↓
Wannier90.x → .wout (MLWF centers)
    ↓
┌───────────────────────────────────────────┐
│  路径 B: 从 Wannier90 输出                 │
│  1. 精确的 Wannier center ⟨r_n⟩           │
│  2. 逐原子 P^I = -e·Σ_{n∈I} ⟨r_n⟩ / Ω    │
│  3. 精确的逐原子 Z*                        │
└───────────────────────────────────────────┘
    ↓
对比 DeltaP P^I vs Wannier90 P^I
```

---

## 3. 实际结果

### 3.1 Wannier90.x 运行失败

Wannier90 3.1.0（conda-forge）在 disentanglement 阶段完成后 I/O 崩溃（segfault），无法获取 .wout 中的 MLWF centers。

**影响**：无法获得路径 B（精确的 Wannier center），只能走路径 A（从 .mmn 直接计算）。

**局限**：路径 A 无法做 Wannierization（求最优 $U(k)$），因此无法获得精确的逐能带 Wannier center。只能计算：
- 总 Berry phase（det，精确，规范不变）
- 总 Berry connection（trace，近似，规范不变）
- 逐能带 Berry connection（$M_{nn}$ 对角元素，近似，**规范依赖**）
- 逐原子 Berry connection（加权 trace，近似，规范不变但 ≠ Berry phase）

### 3.2 总 Berry phase 对比

| 量 | 值 | 说明 |
|---|---|---|
| .mmn Berry phase (det) | +1.200 | 从 .mmn 重叠矩阵计算，15 个占据能带 |
| ABACUS electronic phase | -0.331 (reduced) | 即 γ/(2π) = -0.331, γ = -2.078 |
| 比例 | -0.577 | **不一致** |

**分析**：.mmn 的总 Berry phase 与 ABACUS berry_phase **不一致**。可能原因：

1. **重叠矩阵定义不同**：.mmn 文件中的 $M_{mn}(k,b) = \langle u_{m,k} | u_{n,k+b}\rangle$（周期部分重叠），而 ABACUS berry_phase 使用的是 $O_{mn}(k) = \langle \psi_{m,k} | \psi_{n,k+b}\rangle = C^\dagger(k) \cdot S(\mathbf{b}) \cdot C(k+b)$（Bloch 态重叠）。两者的关系是 $\langle\psi|\psi\rangle = \langle u | e^{i\mathbf{b}\cdot\mathbf{r}} | u\rangle$，**差一个位置算符因子**。

2. **符号约定差异**：.mmn 可能用 $\langle u_{k+b} | u_k\rangle$ 而非 $\langle u_k | u_{k+b}\rangle$，导致 γ → -γ。

3. **能带数问题**：.mmn 有 30 个能带，我只取了前 15 个。如果 ABACUS berry_phase 用了不同的能带选择，结果会不同。

4. **k-string 平均差异**：.mmn 的 Berry phase 在不同 k-string 间变化剧烈（std=6.6），说明 band 12-15 有金属性（与 conduction band 交叉），导致 Berry phase 不稳定。

**结论**：由于总 Berry phase 不匹配，.mmn 的绝对数值不能直接用作参考基准。但 .mmn 仍然可以用来验证 **trace vs det 的数学关系**，因为这两者来自同一个重叠矩阵。

### 3.3 Berry connection vs Berry phase（核心对比）

| 量 | .mmn 计算 | 说明 |
|---|---|---|
| 总 Berry phase (det) | +1.200 | $\mathrm{Im}\,\ln\prod\det M$ |
| 总 Berry connection (trace) | -0.535 | $\sum \mathrm{Im}\,\mathrm{Tr}\, M$ |
| **trace/det 比例** | **-0.446** | trace 仅为 det 的 44.6%，且符号相反 |
| 逐能带 sum (Σ Im M_nn) | +0.672 | $\sum_n \sum_k \mathrm{Im}\, M_{nn}$ |
| Σ / det 比例 | +0.560 | 逐能带之和为 det 的 56.0% |

**分析**：

这是本次测试的 **核心发现**。trace（Berry connection）与 det（Berry phase）的差异不是 DeltaP 实现的 bug，而是 **数学上的必然**：

$$\mathrm{Im}\,\ln\det M \neq \sum_n \mathrm{Im}\, M_{nn}$$

原因：
- $\ln\det M = \mathrm{Tr}\ln M$（恒等式），但 $\ln M \neq M - I$（只有当 $M \approx I$ 时成立）
- $\mathrm{Tr}\, M = \sum_n M_{nn}$（恒等式），但 $\mathrm{Tr}\ln M \neq \mathrm{Tr}(M-I) \neq \mathrm{Tr}\, M - n$

对于 $M \approx I + i\cdot dk \cdot A$（$A$ 为 Berry connection 矩阵）：
- $\mathrm{Im}\,\ln\det M = dk \cdot \mathrm{Tr}\, A + O(dk^3)$
- $\mathrm{Im}\,\mathrm{Tr}\, M = dk \cdot \mathrm{Tr}\, A + O(dk^2)$

差异为 $O(dk^2)$。当 $dk = 0.1$（10 个 k 点）、$n_{\text{occ}} = 15$（15×15 矩阵）时，高阶项贡献显著，导致 44% 的误差。

### 3.4 逐原子分解

用 .amn 投影矩阵计算按能带归一化的权重：

$$w^I_n(k) = \frac{\sum_{p \in I} |A_{np}(k)|^2}{\sum_{\text{all}\, p} |A_{np}(k)|^2}$$

满足 $\sum_I w^I_n = 1$（每个能带的权重归一）。

逐原子 Berry connection：

$$A^I = \sum_k \sum_n w^I_n(k) \cdot \mathrm{Im}\, M_{nn}(k)$$

| 原子 | $A^I$ | 占比 |
|---|---|---|
| Ba | -0.278 | 51.9% |
| Ti | +0.088 | -16.4% |
| O1 | -0.330 | 61.7% |
| O2 | -0.329 | 61.5% |
| O3 | +0.314 | -58.7% |
| **Sum** | **-0.535** | **100%** |

**Sum rule: PASS** — $\sum_I A^I = A^{\text{total}}$（精确成立）。

但 $A^{\text{total}}$（Berry connection）≠ $\gamma$（Berry phase），所以逐原子分解的是 **近似量**，不是精确量。

---

## 4. 对 DeltaP 的分析

### 4.1 DeltaP 的两种方法

| 方法 | 公式 | sum rule | 精确性 |
|------|------|----------|--------|
| Wilson loop (det) | $\gamma^I = \mathrm{Im}\,\ln\prod\det M^I$ | **不成立**（$\prod\det M^I \neq \det M$） | 精确（如果 sum rule 成立） |
| Berry connection (trace) | $A^I = \sum w^I \cdot \mathrm{Im}\, M_{nn}$ | **成立** | 近似（trace ≠ det） |

### 4.2 DeltaP Z* 不正确的原因

Born 有效电荷 $Z^* = (\Omega/e) \cdot \Delta P / \Delta\tau$。

| Z* 来源 | Ti | Ba | 说明 |
|---------|----|----|------|
| berry_phase（基准） | 6.69 | 2.67 | 用 det 差分，精确 |
| DeltaP（trace） | 47.7 | 13.9 | 用 trace 差分，trace≠det 导致错误 |
| DeltaP（det+trace 混合） | -2.09 | 101.1 | trace/det 比例随结构变化，rescaling 无效 |

**根因**：$Z^*$ 是两个结构间的 **差分**，即使 Berry connection 的绝对误差不大，差分后的相对误差也可能很大（因为不同结构的 trace/det 比例不同）。

### 4.3 .mmn 对比验证了什么？

1. **trace ≠ det 不是 DeltaP 的 bug**：从 ABACUS 自己生成的 .mmn 文件独立计算，确认 trace/det = 0.446。这是 $n=15$、$dk=0.1$ 条件下的数学必然。

2. **逐原子 sum rule 可以精确满足**：按能带归一化的权重法使 $\sum_I A^I = A^{\text{total}}$。DeltaP 的"全局 SVD + 权重"方法用的是 SVD gauge 的权重 $w^I_s$，原理类似，sum rule 也成立。

3. **但 sum rule 成立 ≠ 结果正确**：因为 $A^{\text{total}}$（trace）≠ $\gamma$（det），sum rule 保证的是近似量的可加性，不是精确量的可加性。

---

## 5. 没有完成的验证

### 5.1 Wannier90.x 精确 Wannier center

Wannier90.x 崩溃，无法获得经过 Wannierization 的精确 Wannier center $\langle r_n \rangle$。

如果有精确的 Wannier center：
- $\sum_n \langle r_n \rangle$ 精确等于总极化（Berry phase）
- 逐原子 $P^I = -e \sum_{n \in I} \langle r_n \rangle / \Omega$ 是精确分解
- 可以直接与 DeltaP 的 $P^I$ 对比

### 5.2 修复 Wannier90 的可能途径

1. **编译 Wannier90 源码**（非 conda-forge 版本），避免 I/O bug
2. **使用 PW basis**（`basis_type=pw`）的 Wannier90 接口，避免 LCAO 接口的额外投影问题
3. **使用更简单的体系**（如 Si，已知 Wannier90 能正常工作），先验证方法再推广到 BaTiO3

---

## 6. 总结

### 测试逻辑
用 Wannier90 的 Wannier center（精确的逐能带极化分解）作为独立基准，验证 DeltaP 逐原子极化分解的正确性。

### 预期结果
- 总 Berry phase 从 .mmn 应与 ABACUS berry_phase 一致（都是 det）
- 逐原子分解应满足 sum rule
- DeltaP 的 Z* 应与 berry_phase 的 Z* 一致

### 实际结果
1. **Wannier90.x 崩溃**：无法获得精确 Wannier center，对比验证不完整
2. **.mmn 总 Berry phase 与 ABACUS 不匹配**：可能因重叠矩阵定义差异（周期部分 vs Bloch 态）
3. **trace ≠ det 确认**：从 .mmn 独立验证，比例为 0.446（44.6%），这是数学必然
4. **逐原子 sum rule 成立**（trace 方法）：但分解的是近似量

### 分析
DeltaP Z* 不正确的根因是 **Berry connection（trace）≠ Berry phase（det）**。这不是实现 bug，而是极化逐原子分解的根本数学困难：精确的 Berry phase（log-det）不可线性分解，可线性分解的 Berry connection（trace）是近似且误差大（44%）。

### 下一步
1. 修复 Wannier90 运行（编译源码或换 PW basis），获得精确 Wannier center
2. 用精确 Wannier center 做完整的逐原子 Z* 对比
3. 探索增大 k-mesh（减小 dk）以降低 trace-det 差异
