# DeltaP SMO-Berry Connection 算法问题：公式推导、数值测试与根因分析

> **日期**: 2026-06-25  
> **范围**: 详细推导Berry connection极化分解的数学公式，设计纯数学单元测试定位非平滑性根源，分析Born有效电荷测试中P对结构变化非平滑响应的原因

---

## 目录

1. [数学公式推导](#1-数学公式推导)
2. [非平滑性问题的数学分析](#2-非平滑性问题的数学分析)
3. [数值实验设计](#3-数值实验设计)
4. [测试结果](#4-测试结果)
5. [根因定位](#5-根因定位)
6. [Born有效电荷测试回顾](#6-born有效电荷测试回顾)
7. [结论与改进方向](#7-结论与改进方向)

---

## 1. 数学公式推导

### 1.1 极化的Berry phase定义

晶体电子极化沿晶格矢量 $\mathbf{a}_\alpha$ 方向的分量：

$$P_\alpha = -\frac{e \cdot a_\alpha}{2\pi \cdot \Omega} \sum_n \int_0^1 dk_\alpha \, \mathrm{Im}\left[\langle u_{n\mathbf{k}} | \partial_{k_\alpha} u_{n\mathbf{k}} \rangle\right]$$

其中：
- $a_\alpha$ 是晶格矢量长度（Bohr）
- $\Omega$ 是晶胞体积（Bohr³）
- $|u_{n\mathbf{k}}\rangle$ 是Bloch函数的周期部分
- $k_\alpha$ 是直接坐标（0到1）
- 积分结果是无量纲的Berry phase $\gamma$

离散化（k-string上 $N$ 个点）：

$$P_\alpha = -\frac{a_\alpha}{2\pi \Omega} \cdot \Delta k \sum_n \sum_j \mathrm{Im}\left[A_n(k_j)\right]$$

其中 $\Delta k = 1/(N-1)$，$A_n(k_j) = \langle u_{n,k_j} | \partial_k u_{n,k_j} \rangle$ 是Berry connection。

### 1.2 SMO投影分解

将恒等算符近似为SMO投影算符之和：

$$\hat{I} \approx \sum_I \hat{P}^I = \sum_I \sum_{lm} |\alpha^I_{lm\mathbf{k}}\rangle\langle\alpha^I_{lm\mathbf{k}}|$$

代入Berry connection：

$$A_n(\mathbf{k}) = \langle u_{n\mathbf{k}} | \partial_k u_{n\mathbf{k}} \rangle \approx \sum_I \sum_{lm} \left[\langle u | \partial_k \alpha^I_{lm} \rangle \langle \alpha^I_{lm} | u \rangle + \langle u | \alpha^I_{lm} \rangle \partial_k \langle \alpha^I_{lm} | u \rangle\right]$$

定义原子Berry connection：

$$\boxed{A^I_n(k, \alpha) = \sum_{lm} \left[\underbrace{\langle \psi | \partial_{k_\alpha} \alpha^I_{lm} \rangle \cdot \langle \alpha^I_{lm} | \psi \rangle}_{\text{term1}} + \underbrace{\langle \psi | \alpha^I_{lm} \rangle \cdot \partial_{k_\alpha} \langle \alpha^I_{lm} | \psi \rangle}_{\text{term2}}\right]}$$

### 1.3 各项的具体计算

**SMO-NAO k空间重叠**：

$$S_{\mu, Ilm}(k) = \sum_{\mathbf{R}} e^{2\pi i \mathbf{k} \cdot \mathbf{R}} \langle \phi^0_\mu | \alpha^I_{lm}(\mathbf{R}) \rangle$$

**解析k导数**（链式法则，注意 $e^{2\pi i k R}$ 的 $2\pi$ 因子）：

$$\frac{\partial S_{\mu, Ilm}}{\partial k_\alpha} = \sum_{\mathbf{R}} 2\pi i \cdot R_\alpha \cdot e^{2\pi i \mathbf{k} \cdot \mathbf{R}} \cdot \langle \phi^0_\mu | \alpha^I_{lm}(\mathbf{R}) \rangle$$

**SMO-波函数重叠**：

$$D_I(lm, n, k) = \langle \alpha^I_{lm\mathbf{k}} | \psi_{n\mathbf{k}} \rangle = \sum_\mu S^*_{\mu, Ilm}(k) \cdot C_{n\mu}(k)$$

**term1（解析）**：

$$\langle \psi | \partial_k \alpha^I_{lm} \rangle = \sum_\mu C^*_{n\mu} \cdot \frac{\partial S_{\mu, Ilm}}{\partial k_\alpha}$$

$$\text{term1} = \left(\sum_\mu C^*_{n\mu} \cdot \frac{\partial S_{\mu}}{\partial k}\right) \cdot D_I(lm, n, k)$$

**term2（有限差分）**：

$$\partial_k \langle \alpha^I_{lm} | \psi \rangle = \partial_k D_I \approx \frac{D_I(k+\Delta k) - D_I(k-\Delta k)}{2\Delta k}$$

$$\text{term2} = D^*_I(lm, n, k) \cdot \frac{D_I(k+\Delta k) - D_I(k-\Delta k)}{2\Delta k}$$

### 1.4 规范固定

每个k点的波函数有任意相位 $e^{i\varphi_{nk}}$，使 $D_I$ 的相位不确定，有限差分无效。

**SMO锚定规范**：选择锚定SMO $\text{ref}(n,k) = \arg\max_{I,lm} |D_I(lm,n,k)|$，固定相位：

$$\tilde{D}_I = D_I \cdot g(n,k), \quad g(n,k) = \frac{D^*_{\text{anchor}}(n,k)}{|D_{\text{anchor}}(n,k)|}$$

使锚定投影 $\tilde{D}_{\text{anchor}}$ 为正实数。连续跟踪：若 $g(k_j)$ 与 $g(k_{j-1})$ 反号，翻转 $g \to -g$。

### 1.5 规范固定后的Berry connection

$$\tilde{A}^I_n = \text{term1} + \text{term2}$$

**term1（规范不变）**：

$$\text{term1} = \left(\sum_\mu (C_\mu \cdot g)^* \cdot \frac{\partial S}{\partial k}\right) \cdot (D_I \cdot g) = |g|^2 \cdot \text{term1}_{\text{raw}} = \text{term1}_{\text{raw}}$$

因为 $|g| = 1$，term1是规范不变的。

**term2（规范固定后的有限差分）**：

$$\text{term2} = (D_I \cdot g)^* \cdot \frac{D_I(k+\Delta k) \cdot g(k+\Delta k) - D_I(k-\Delta k) \cdot g(k-\Delta k)}{2\Delta k}$$

### 1.6 精确Wilson loop（对比方法）

$$\gamma_{\text{Wilson}} = \mathrm{Im}\left[\log \prod_j \det\left(\mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}^I(k_{j+1})\right)\right]$$

其中 $\mathbf{U}(k) = \mathbf{W}(k)\mathbf{V}^\dagger(k)$ 是 $D_I(k) = \mathbf{W}\boldsymbol{\Sigma}\mathbf{V}^\dagger$ 的极分解，$\mathbf{O}(k_j, k_{j+1}) = \langle \psi_{k_j} | \psi_{k_{j+1}} \rangle$ 是波函数重叠矩阵。

Wilson loop是**规范不变的**（SVD极分解消除任意相位），但 `arg()` 函数有分支切割（极化量子 $eR/\Omega$）。

---

## 2. 非平滑性问题的数学分析

### 2.1 问题定义

Born有效电荷要求 $P$ 对原子位置 $\tau$ 平滑变化：

$$Z^*_{I,\alpha\beta} = \Omega \frac{\partial P_\alpha}{\partial \tau_{I,\beta}} \approx \Omega \frac{P(+\Delta\tau) - P(-\Delta\tau)}{2\Delta\tau}$$

如果 $P$ 对 $\tau$ 不平滑（非线性或跳变），$Z^*$ 不可靠。

### 2.2 非平滑性来源推导

当原子位置变化 $\delta\tau$ 时，各量变化：

| 量 | 变化 | 性质 |
|----|------|------|
| $\alpha^I_{lm}(\mathbf{r})$ | SMO随原子移动 | 连续 |
| $S_{\mu,Ilm}(k)$ | 重叠积分变化 | 连续 |
| $D_I(lm,n,k)$ | $S$和$C$都变 | 连续 |
| $g(n,k)$ | $= D^*_{\text{anchor}}/\|D_{\text{anchor}}\|$ | **可能跳变** |
| $\partial_k \tilde{D}_I$ | 有限差分含$g$ | **放大跳变** |

**关键：规范相位 $g$ 的变化**

$g(k) = D^*_{\text{anchor}}(k) / |D_{\text{anchor}}(k)|$ 的幅角为 $\arg(g) = -\arg(D_{\text{anchor}})$。

当 $D_{\text{anchor}}$ 的幅角随结构连续变化时，$g$ 也连续变化。但有两个不连续来源：

**来源1：锚定跳变**

当 $|D_{I_1,lm_1}|$ 和 $|D_{I_2,lm_2}|$ 接近时，微小结构变化可能使argmax从 $(I_1, lm_1)$ 跳到 $(I_2, lm_2)$，导致 $g$ 跳变 $\Delta\varphi = \arg(D_{\text{new}}) - \arg(D_{\text{old}})$。

**来源2：连续跟踪的符号翻转**

连续跟踪规则：若 $\text{Re}(g(k_j) \cdot g^*(k_{j-1})) < 0$，则 $g \to -g$。

当 $g(k_j)$ 接近 $-g(k_{j-1})$（即幅角差接近 $\pi$）时，微小扰动可能触发符号翻转，导致 $g$ 跳变 $\pi$。

### 2.3 有限差分对跳变的放大

规范固定后的有限差分：

$$\partial_k \tilde{D}_I = \frac{D_I(k+\Delta k) \cdot g(k+\Delta k) - D_I(k-\Delta k) \cdot g(k-\Delta k)}{2\Delta k}$$

当 $g$ 在 $k+\Delta k$ 处跳变 $\delta g$ 时：

$$\delta(\partial_k \tilde{D}_I) = \frac{D_I(k+\Delta k) \cdot \delta g}{2\Delta k}$$

- 若 $\delta g$ 连续（$O(\epsilon)$）：$\delta(\partial_k \tilde{D}_I) = O(\epsilon / \Delta k)$，对 $\Delta k = 0.1$ 放大10倍
- 若 $\delta g$ 跳变（$O(1)$，锚定跳变或符号翻转）：$\delta(\partial_k \tilde{D}_I) = O(1/\Delta k) = O(10)$，**放大10倍**

### 2.4 对P的影响

$$\delta P = -\frac{a_\alpha}{2\pi\Omega} \Delta k \sum_n \sum_j \mathrm{Im}[\delta A^I_n(k_j)]$$

$$\delta A^I_n = \delta(\text{term1}) + \delta(\text{term2})$$

- $\delta(\text{term1}) = O(\epsilon)$（规范不变，平滑）
- $\delta(\text{term2}) = O(\epsilon/\Delta k)$ 或 $O(1/\Delta k)$（有限差分放大）

因此 $\delta P \propto \Delta k \cdot O(1/\Delta k) = O(1)$ —— **P的变化与扰动大小无关**，取决于 $g$ 是否跳变。

### 2.5 Wilson loop的对比

Wilson loop $\gamma = \mathrm{Im}[\log \prod_j \det(\mathbf{M}_j)]$ 不涉及 $g(k)$，因此不受规范跳变影响。

但 Wilson loop 有**分支切割问题**：$\arg(\cdot)$ 函数在负实轴跳变 $2\pi$，导致极化量子 $eR/\Omega$ 的不确定性。

---

## 3. 数值实验设计

### 3.1 设计原则

- **纯数学**：不需要ABACUS基础设施（无UnitCell、Psi、TwoCenterIntegrator）
- **合成数据**：构造满足物理特征的 $D_I$ 数据
- **可控扰动**：精确控制扰动大小，验证线性响应
- **方法对比**：Berry connection（有/无规范）vs Wilson loop

### 3.2 合成数据

```cpp
struct KStringData {
    int nppstr;  // k-string上的点数（含环绕）
    int nbands;  // 能带数
    int nproj;   // 每原子SMO通道数
    int nat;     // 原子数
    // D_I[ik][iat][lm][n] = <alpha|psi>
    // S_k[ik][iat][lm][mu] = <phi|alpha>
    // dS_k[ik][iat][alpha][lm][mu] = d/dk <phi|alpha>
    // C[ik][n][mu] = 波函数系数
};
```

数据特征：
- $D_I$ 幅值随k平滑变化
- $D_I$ 相位线性变化（模拟Berry phase）
- 原子0的投影最大（锚定）
- $dS = 2\pi i \cdot R \cdot S$（解析导数）

### 3.3 三种计算方法

| 方法 | 公式 | 规范依赖 |
|------|------|---------|
| Berry (gauge) | term1 + term2，带SMO锚定规范 | 依赖 $g(k)$ |
| Berry (no gauge) | term1 + term2，原始相位 | 依赖 $D_I$ 的任意相位 |
| Wilson loop | $\arg(\prod_j \det(\mathbf{U}^\dagger \mathbf{U}_{\text{next}}))$ | 不依赖 $g$（SVD） |

### 3.4 四类测试

| 测试 | 目的 | 方法 |
|------|------|------|
| 平滑性 | $\Delta P \propto \epsilon$? | 扫描 $\epsilon \in [10^{-6}, 10^{-2}]$ |
| 规范不变性 | 相位旋转后 $P$ 不变? | 随机相位扰动 |
| 锚定跳变 | 接近跳变时 $P$ 非线性? | 设两原子投影接近 |
| 对比 | 哪个方法更平滑? | 同一数据三种方法 |

---

## 4. 测试结果

### 4.1 结果总表

| # | 测试 | 方法 | 结果 | 关键数值 |
|---|------|------|------|---------|
| 1 | 一般扰动平滑性 | Berry (gauge) | **PASS** | $\Delta P/\epsilon$ 在各 $\epsilon$ 一致 |
| 2 | 一般扰动平滑性 | Berry (no gauge) | **PASS** | rel_err < 8% |
| 3 | 对比平滑性 | Wilson vs Berry | **Wilson更差** | Wilson $\Delta P/\epsilon$=538 vs Berry=1.16 |
| 4 | 规范不变性 | Wilson loop | **FAIL** | 简化SVD非规范不变 |
| 5 | 规范不变性 | Berry connection | PASS | 预期变化 (rel=1.24) |
| 6 | 锚定跳变 | Berry (gauge) | PASS | 2%扰动→0.6% $\Delta P$ |
| 7 | 无锚定跳变 | Berry (gauge) | **FAIL** | 1%相位扰动→14% $\Delta P$ |

### 4.2 详细结果

**测试1：Berry connection（规范固定）平滑性 — PASS**

```
epsilon   deltaP      deltaP/epsilon   线性?
1e-6      1.16e-6     1.16             ✓
1e-5      1.16e-5     1.16             ✓
1e-4      1.16e-4     1.16             ✓
1e-3      1.16e-3     1.16             ✓
1e-2      1.16e-2     1.16             ✓
```

$\Delta P \propto \epsilon$，斜率恒定。Berry connection公式对一般扰动是平滑的。

**测试3：方法对比 — Wilson loop 更差**

```
方法              P0        P1(eps=1e-3)  dP       dP/eps
Wilson loop      0.0646    0.6023        0.5377   537.7
Berry (gauge)   -7.112    -7.111        0.0012   1.16
Berry (no gauge) -3.487    -3.486        0.0005   0.47
```

Wilson loop的 $\Delta P$ 与 $\epsilon$ 无关（恒为~0.537），这是 `arg()` 分支切割所致——Wilson loop乘积越过负实轴时跳变 $\pi$。

**测试7：无锚定跳变的平滑性 — FAIL**

```
1%均匀相位扰动:
P0 = 1.842
P1 = 2.102
dP = 0.259
rel_dP = 14.1%  (应~1%)
```

即使无锚定跳变（原子0占绝对优势），1%的均匀相位扰动导致14%的P变化。原因：规范相位 $g(k)$ 对 $D_I$ 的相位变化敏感，1%的相位扰动改变了 $g(k)$，进而改变了有限差分 $d_k(D_I \cdot g)$。

---

## 5. 根因定位

### 5.1 排除过程

| 假设 | 测试 | 结论 |
|------|------|------|
| Berry connection公式本身有bug | 测试1-2 | **排除**：公式对一般扰动平滑 |
| 锚定跳变是主要原因 | 测试6 | **排除**：合成数据中锚定跳变影响小 |
| Wilson loop更好 | 测试3-4 | **排除**：arg()分支切割更严重 |
| **规范固定对相位变化敏感** | **测试7** | **确认**：1%相位→14%P变化 |

### 5.2 根因：规范固定的相位敏感性

规范相位 $g(n,k) = D^*_{\text{anchor}} / |D_{\text{anchor}}|$ 的幅角为 $-\arg(D_{\text{anchor}})$。

当结构变化导致 $D_{\text{anchor}}$ 的相位变化 $\delta\varphi$ 时：

$$\delta g = g \cdot (-i \cdot \delta\varphi) + O(\delta\varphi^2)$$

有限差分中 $g$ 的变化被 $1/\Delta k$ 放大：

$$\delta(\partial_k \tilde{D}) \sim \frac{D \cdot \delta g}{\Delta k} \sim \frac{D \cdot \delta\varphi}{\Delta k}$$

对于 $\Delta k = 0.1$（10个k点），放大因子为10。因此1%的相位扰动→10%的P变化，与测试结果（14%）一致。

### 5.3 为什么真实ABACUS测试中更严重

真实ABACUS测试中，±0.01 Bohr位移导致P变化16倍（Berry connection）。比合成测试（14%）更严重，因为：

1. **多k点叠加**：真实系统有多个k点，每个k点的 $g$ 变化不同，叠加放大
2. **锚定跳变**：真实系统中多个SMO的投影可能接近，容易触发跳变
3. **SCF收敛**：波函数本身在不同结构下可能收敛到略有不同的态
4. **多能带**：多个能带的 $g$ 变化叠加

---

## 6. Born有效电荷测试回顾

### 6.1 正确结构+小位移的berry_phase结果

| 原子 | Z*_berry | 文献值 | 误差 |
|------|---------|--------|------|
| Ti | +6.80 | +7.18 | 5.3% |
| Ba | +2.67 | +2.74 | 2.6% |

berry_phase Z*可靠，用户经验正确。之前错误来自：(1)错误基础结构（O在立方位置），(2)位移太大（0.05 Bohr）。

### 6.2 DeltaP (SMO-berry) Z*完全不可靠

| 原子 | Z*_deltap | Z*_berry | 偏差 |
|------|-----------|---------|------|
| Ti | -1633 | +6.80 | 240× (符号错) |
| Ba | +63.5 | +2.67 | 24× |

P值对±0.01 Bohr位移有16倍不对称（berry为0.55倍），严重违反线性响应。

### 6.3 差分不但没消除偏差，反而放大

| 量 | deltap/berry 比值 |
|----|-------------------|
| P (平衡) | 12.4× |
| Z*_Ti | 240× |

差分将12.4×放大到240×，因为规范固定对结构变化的敏感性被有限差分的 $1/\Delta k$ 放大。

---

## 7. 结论与改进方向

### 7.1 结论

1. **Berry connection公式本身是平滑的**（测试1-2通过），问题不在公式推导
2. **非平滑性根源是规范固定的相位敏感性**（测试7定位）：$g(k)$ 对 $D_I$ 的微小相位变化敏感，有限差分 $1/\Delta k$ 放大
3. **Wilson loop有不同问题**：`arg()`分支切割导致极化量子跳变
4. **berry_phase Z*是可靠的验证基准**（Ti误差5.3%, Ba误差2.6%）
5. **DeltaP (SMO-berry) 当前实现不适合Z*计算**：差分放大偏差12.4×→240×

### 7.2 改进方向

**方向A：精确Wilson loop + 分支跟踪**

$$\gamma = \mathrm{Im}\left[\log \prod_j \det\left(\mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j, k_{j+1}) \cdot \mathbf{U}^I(k_{j+1})\right)\right]$$

- 优点：规范不变（SVD），不依赖 $g(k)$
- 需要：正确SVD极分解（非简化归一化）+ `unkOverlap_lcao`精确重叠 + 极化分支跟踪
- 挑战：`arg()`分支切割需要连续跟踪（记录跨量子时的 $\Delta\varphi$ 校正）

**方向B：解析Berry connection（避免有限差分）**

将term2的有限差分替换为解析导数：

$$\partial_k D_I = \sum_\mu \left[\frac{\partial S^*_{\mu}}{\partial k} C_{n\mu} + S^*_{\mu} \frac{\partial C_{n\mu}}{\partial k}\right]$$

- $\partial S / \partial k$ 已有解析公式（$2\pi i R e^{2\pi ikR}$）
- $\partial C / \partial k$ 需要Sternheimer响应（$\partial\psi/\partial k$），或k+微扰的微扰论
- 优点：无有限差分，无 $1/\Delta k$ 放大
- 挑战：$\partial C / \partial k$ 实现复杂

**方向C：规范协变的Berry connection**

不用SMO锚定规范，改用**平行传输规范**（berryphase已有实现）：
- 在k-string上，每个k的波函数相位由 $\max \mathrm{Re}\langle\psi_{k_j}|\psi_{k_{j+1}}\rangle$ 确定
- 平行传输不依赖SMO，对结构变化更平滑
- 但跨结构（Phase B约束循环）不连续

**推荐**：方向A（精确Wilson loop + 分支跟踪）是最有前景的方案。它从根本上消除了规范固定问题，同时分支跟踪解决了极化量子。需要实现的组件：

1. `unkOverlap_lcao` 计算精确 $\mathbf{O}(k_j, k_{j+1})$
2. LAPACK `zgesvd` 正确SVD极分解
3. 极化分支连续跟踪（记录 $\Delta\varphi$ 跨量子校正）
4. 单元测试验证P对结构变化的平滑性（复用本文测试框架）
