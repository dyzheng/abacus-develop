# DeltaQS 电荷投影基扩展严格分析

## 1. 问题陈述

### 1.1 当前投影方案的局限

DeltaQS 的电荷约束依赖原子投影电荷：

$$N_I = \text{Tr}(\rho \cdot P_I), \quad P_I = \sum_{lm} |\alpha_{I,1,l,m}\rangle\langle\alpha_{I,1,l,m}|$$

当前投影 $P_I$ 使用每个角动量 $l$ 的**第一个 zeta 轨道**。以 Fe 为例（赝势价电子 3d⁶4s²，Z_val = 8）：

| 角动量 $l$ | 第一个 zeta | 可捕获电子数 | 实际价电子 |
|-----------|------------|-------------|-----------|
| s ($l=0$) | 第一个 s-zeta | 2 | 2 (4s²) |
| p ($l=1$) | 第一个 p-zeta | 6 | 0 |
| d ($l=2$) | 第一个 d-zeta | 6 | 6 (3d⁶) |

对于大多数元素，第一个 zeta 足以覆盖价电子。但对于存在多个同角动量壳层的元素（如碱土金属的 ns² 和 (n+1)s²），第一个 zeta 无法覆盖所有价电子。

### 1.2 ABACUS NAO 的 "严格 single-zeta" 定义

在 ABACUS 的数值原子轨道 (NAO) 框架中，**严格 single-zeta (SZ)** 的定义为：

> 每个价电子都有对应的轨道覆盖。若某角动量 $l$ 有 $n_e$ 个价电子，则需要 $\lceil n_e / (2l+1) \rceil$ 个 zeta 轨道。

**示例**：

| 元素 | 价电子构型 | s-zeta 数 | p-zeta 数 | d-zeta 数 | 总轨道数 |
|------|-----------|----------|----------|----------|---------|
| Fe | 3d⁶4s² | 1 (= ⌈2/1⌉) | 0 | 1 (= ⌈6/5⌉) | 1+5 = 7 (half) |
| Ca | 4s²3p⁶ | 2 (= ⌈8/1⌉) | 1 (= ⌈6/3⌉) | 0 | 2+3 = 5 (half) |
| Ti | 3d²4s² | 2 (= ⌈4/1⌉) | 0 | 1 (= ⌈2/5⌉) | 2+5 = 7 (half) |
| O  | 2s²2p⁴ | 1 (= ⌈2/1⌉) | 1 (= ⌈4/3⌉) | 0 | 1+3 = 4 (half) |

注意：这里的 "half" 指的是轨道数（不计自旋）。完整 SZ 基的轨道数 = 2 × half。

### 1.3 关键区别：严格 SZ vs 全部 zeta

| 方案 | 定义 | 轨道数 | 物理意义 |
|------|------|-------|---------|
| 第一个 zeta | 每个 $l$ 只用第一个 zeta | $\sum_l (2l+1)$ | 不完整投影 |
| 严格 SZ | 每个 $l$ 用 $\lceil n_e^l/(2l+1)\rceil$ 个 zeta | $\sum_l n_\zeta^l (2l+1) \geq n_e$ | 完备覆盖价电子空间 |
| 全部 zeta | 使用轨道文件中所有 zeta | $\sum_l n_{\text{zeta,file}}^l (2l+1)$ | 可能过度，捕获非价电子 |

**关键区别**：严格 SZ 保证投影空间的维度恰好等于（或略大于）价电子数，不会捕获非价电子。

---

## 2. 投影指标体系分析

### 2.1 当前单 zeta 指标方案

当每个 $l$ 只有一个 zeta 时，投影轨道指标为 $(l, m)$，其中 $m = -l, \ldots, l$。总轨道数：

$$N_{\text{proj}} = \sum_{l} (2l+1) = (l_{\max}+1)^2$$

指标到线性索引的映射：

$$\text{index}(l, m) = l^2 + (l + m)$$

这利用了 $1 + 3 + 5 + 7 + \ldots = L^2$ 的求和性质。当前代码中 `B_I_nproj[iat] = max_l_plus_1 * max_l_plus_1` 正是基于此。

### 2.2 扩展 SZ 指标方案

当每个 $l$ 有 $n_\zeta^l$ 个 zeta 时，指标变为 $(\zeta, l, m)$。线性索引映射：

$$\text{index}(\zeta, l, m) = \left[\sum_{l'<l} n_\zeta^{l'} (2l'+1)\right] + \zeta \cdot (2l+1) + (l + m)$$

总投影轨道数：

$$N_{\text{proj}}^{\text{SZ}} = \sum_{l} n_\zeta^l (2l+1)$$

**代码影响**：
- `B_I_nproj[iat]` 不再等于 `(max_l_plus_1)^2`
- 需要存储每个原子的 `nzeta_per_l` 数组
- `pre_hr` 的维度从 $(l_{\max}+1)^2$ 扩展到 $N_{\text{proj}}^{\text{SZ}}$
- `cal_moment_IJR` 中的步长需要相应修改

### 2.3 指标方案对比

| 属性 | 单 zeta | 扩展 SZ |
|------|--------|--------|
| 指标 | $(l, m)$ | $(\zeta, l, m)$ |
| 维度 | $(l_{\max}+1)^2$ | $\sum_l n_\zeta^l(2l+1)$ |
| 简洁公式 | ✓ $l^2 + l + m$ | ✗ 需要累积求和 |
| 与现有代码兼容 | ✓ | ✗ 需要修改索引 |

---

## 3. 正交性分析

### 3.1 NAO 与 SMO 的正交性

**NAO (Numerical Atomic Orbitals)**：
- 来自原子 DFT 计算（孤立原子）
- 同一 $l$ 的不同 zeta **自动正交**：$\langle \alpha_{\zeta_1, l, m} | \alpha_{\zeta_2, l, m} \rangle = \delta_{\zeta_1, \zeta_2}$
- 不同 $l$ 的轨道因球谐函数正交性自动正交

**SMO (Self-consistent Molecular Orbitals)**：
- 来自分子/周期性环境的自洽计算
- 同一 $l$ 的不同 zeta **不一定正交**：$\langle \alpha_{\zeta_1, l, m} | \alpha_{\zeta_2, l, m} \rangle \neq \delta_{\zeta_1, \zeta_2}$
- 非正交性来源于环境诱导的轨道杂化和变形

### 3.2 非正交性对投影的影响

**投影算符的正确形式**：

对于正交基 $\{|\alpha_i\rangle\}$（$\langle \alpha_i | \alpha_j \rangle = \delta_{ij}$）：

$$P_I = \sum_i |\alpha_i\rangle \langle \alpha_i|$$

$$P_I^2 = P_I \quad (\text{幂等性成立})$$

$$N_I = \text{Tr}(\rho \cdot P_I) = \sum_i \langle \alpha_i | \rho | \alpha_i \rangle$$

对于非正交基 $\{|\alpha_i\rangle\}$（$\langle \alpha_i | \alpha_j \rangle = S_{ij} \neq \delta_{ij}$）：

**方案 A：简单投影（当前实现）**

$$P_I^{\text{simple}} = \sum_i |\alpha_i\rangle \langle \alpha_i|$$

问题：
- $P_I^2 \neq P_I$（不是真正的投影算符）
- 会**重复计算**非正交轨道之间的重叠电荷
- $\sum_I N_I$ 可能**超过**总价电子数

**方案 B：Löwdin 正交化投影**

$$P_I^{\text{Löwdin}} = \sum_{ij} |\alpha_i\rangle (S_I^{-1})_{ij} \langle \alpha_j|$$

其中 $(S_I)_{ij} = \langle \alpha_i | \alpha_j \rangle$ 是原子 $I$ 的投影轨道重叠矩阵。

性质：
- $(P_I^{\text{Löwdin}})^2 = P_I^{\text{Löwdin}}$（幂等性成立）
- $\sum_I N_I = N_{\text{total}}$（电荷守恒）
- 但 $N_I$ 可能包含负的轨道布居（Löwdin 分析的特点）

**方案 C：Mulliken 布居分析**

$$N_I^{\text{Mulliken}} = \sum_{\mu \in I} \sum_\nu (P \cdot S)_{\mu\nu}$$

等价于将重叠电荷等分给两个原子。

### 3.3 正交性影响的定量估计

设投影轨道的非正交度为 $\epsilon = \max_{i \neq j} |S_{ij}|$。

**简单投影的误差**：

$$\Delta N_I = N_I^{\text{simple}} - N_I^{\text{correct}} \sim \mathcal{O}(\epsilon \cdot n_\zeta)$$

对于 SMO 轨道：
- 典型非正交度 $\epsilon \sim 0.01 - 0.1$
- 若 $n_\zeta = 2$，误差 $\Delta N_I \sim 0.02 - 0.2$ e
- 这对于化学精度（~0.1 e）来说**可能重要**

对于 NAO 轨道：
- $\epsilon = 0$（精确正交）
- 误差 $\Delta N_I = 0$

### 3.4 正交化方案推荐

| 轨道类型 | 正交性 | 推荐方案 | 理由 |
|---------|-------|---------|------|
| NAO | 正交 | 简单投影 $P_I = \sum |\alpha_i\rangle\langle\alpha_i|$ | 无需正交化，代码简单 |
| SMO | 非正交 | Löwdin 正交化 $P_I = \sum |\alpha_i\rangle (S^{-1})_{ij} \langle\alpha_j|$ | 保证幂等性和电荷守恒 |

**SMO 正交化的实现**：

需要在 `cal_pre_HR()` 中：
1. 计算投影轨道重叠矩阵 $(S_I)_{ij} = \langle \alpha_i | \alpha_j \rangle$
2. 计算 $S_I^{-1}$（对于小矩阵，直接求逆）
3. 使用 $S_I^{-1}$ 构造正确的投影算符

对于单 zeta 情况（NAO），$S_I = I$，退化为当前实现。

---

## 4. 对数据集一致性的影响

### 4.1 理想情况

使用严格 SZ + 正交投影基：

$$N_I = \text{Tr}(\rho \cdot P_I^{\text{correct}}), \quad \sum_I N_I = Z_{\text{val}}^{\text{total}}$$

对于同一元素在不同结构中：

| 结构 | Fe 的氧化态 | $N_I$ | $V_I = N_I - Z_{\text{val}}$ |
|------|-----------|-------|------------------------------|
| Fe metal | 0 | ~8.0 | 0 |
| FeO | +2 | ~6.0 | -2 |
| Fe₂O₃ | +3 | ~5.0 | -3 |

**一致性**：✓ 不同结构的价态标签可比较

### 4.2 非正交性的影响

若使用 SMO 投影但不做正交化：

| 结构 | $N_I^{\text{simple}}$ | 误差 $\Delta N_I$ | $V_I^{\text{apparent}}$ |
|------|---------------------|------------------|------------------------|
| Fe metal | 8.0 + 0.2 | +0.2 | -0.2（应为 0）|
| FeO | 6.0 + 0.15 | +0.15 | -2.15（应为 -2）|
| Fe₂O₃ | 5.0 + 0.1 | +0.1 | -3.1（应为 -3）|

**影响**：
- 误差量级 ~0.1-0.2 e，取决于化学环境
- 氧化态的**相对顺序**保持正确
- 但**绝对值**有系统偏差
- 对于 ML 训练，可能导致标签噪声

### 4.3 数据集一致性结论

| 方案 | 绝对一致性 | 相对一致性 | 推荐度 |
|------|----------|----------|-------|
| SZ + NAO（正交） | ✓ 完美 | ✓ 完美 | ★★★★★ |
| SZ + SMO + Löwdin | ✓ 完美 | ✓ 完美 | ★★★★☆ |
| SZ + SMO + 简单投影 | ✗ 有偏差 | ✓ 基本保持 | ★★★☆☆ |
| 单 zeta（当前） | ✗ 不完整 | ✗ 不完整 | ★★☆☆☆ |

---

## 5. 实现路线建议

### 5.1 Phase 1：NAO 轨道的 SZ 扩展（低风险）

**前提**：NAO 轨道已经正交，无需正交化处理。

**修改点**：
1. `cal_pre_HR()`：扩展投影轨道包含所有 SZ zeta
2. 索引系统：从 $(l,m)$ 扩展到 $(\zeta, l, m)$
3. `B_I_nproj`：从 $(l_{\max}+1)^2$ 改为 $\sum_l n_\zeta^l (2l+1)$

**验证**：
- 对已知化合物（FeO, Fe₂O₃, TiO₂ 等），验证 $\sum_I N_I \approx Z_{\text{val}}^{\text{total}}$
- 验证投影价态与化学直觉一致

### 5.2 Phase 2：SMO 轨道的正交化处理（中风险）

**修改点**：
1. 计算投影轨道重叠矩阵 $S_I$
2. 实现 Löwdin 正交化：$P_I \to S_I^{-1/2} P_I S_I^{-1/2}$
3. 验证电荷守恒和幂等性

**验证**：
- 对比 NAO 和 SMO 的投影电荷
- 验证 $\sum_I N_I^{\text{SMO}} \approx \sum_I N_I^{\text{NAO}}$

### 5.3 Phase 3：数据集基准测试

**目标**：验证 DeltaQS 价态标签在不同结构间的一致性。

**测试集**：
- Fe 氧化物系列：Fe, FeO, Fe₂O₃, Fe₃O₄
- Ti 氧化物系列：Ti, TiO, TiO₂, Ti₂O₃
- 对比已知氧化态和投影价态

---

## 6. 总结

### 6.1 核心结论

1. **扩展投影基是必要的**：当前单 zeta 投影无法覆盖所有价电子，导致电荷约束目标不可达。

2. **ABACUS 严格 SZ 是正确选择**：保证投影空间维度恰好覆盖价电子数，不会过度捕获。

3. **正交性处理取决于轨道类型**：
   - NAO：已正交，无需处理
   - SMO：需要 Löwdin 正交化

4. **数据集一致性要求**：
   - 必须使用完备投影基 + 正交化处理
   - 否则不同结构的价态标签不可比较

### 6.2 推荐实现优先级

1. **先实现 NAO 的 SZ 扩展**（Phase 1）
   - 代码修改较小
   - 无需正交化
   - 可立即验证物理正确性

2. **再处理 SMO 的正交性**（Phase 2）
   - 需要额外的重叠矩阵计算
   - 但物理上更严格

3. **最后进行数据集基准测试**（Phase 3）
   - 验证一致性
   - 校准参数

### 6.3 未解决问题

1. **轨道文件中 zeta 数量的确定**：如何从轨道文件或赝势中自动确定每个 $l$ 需要的 SZ zeta 数？
   - 需要从原子 DFT 计算中读取价电子构型
   - 或手动在 INPUT 中指定

2. **SMO 正交化的计算成本**：$S_I^{-1}$ 的计算和存储开销
   - 对于小矩阵（~10×10），成本可忽略
   - 但需要在每次 `cal_pre_HR()` 时重新计算

3. **与现有 DeltaSpin 的兼容性**：修改投影基是否影响磁矩约束？
   - 磁矩 $M_I = N_I^\uparrow - N_I^\downarrow$
   - 若上下自旋使用相同的扩展投影基，差值应不受影响
   - 但需要验证
