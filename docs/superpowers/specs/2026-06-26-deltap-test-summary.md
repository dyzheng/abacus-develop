# DeltaP 逐原子极化分解：测试结果总结

> **日期**: 2026-06-26
> **体系**: BaTiO3 四方铁电相, 10×10×10 k-mesh, LCAO basis, nocc=15

---

## 1. 测试不通过的结果一览

### 1.1 总极化 P (参考结构)

ABACUS 自带的 `berry_phase` 模块给出的结果作为**基准**（已验证 Z* 与文献一致）：

| ABACUS berry_phase 输出 | 值 |
|---|---|
| Ionic Phase (reduced) | +0.36000 |
| Electronic Phase (reduced) | -0.33085 |
| P_total = (a/Ω)×(0.36000-0.33085) | **+5.10×10⁻⁴ e/bohr²** |
| P_electronic = (a/Ω)×(-0.33085) | **-5.79×10⁻³ e/bohr²** |

### 1.2 DeltaP 两种实现的总极化（sum of per-atom Pz）

DeltaP 模块有两种 `deltap_method`，都**不通过**：

| 方法 | P_sum (e/bohr²) | vs P_elec | 状态 |
|---|---|---|---|
| `berry_connection` | +4.29×10⁻³ | -0.74× | ❌ 不匹配 |
| `wannier` (当前代码) | -1.48×10⁻² | 2.56× | ❌ 不匹配 |

### 1.3 Born 有效电荷 Z*

| 方法 | Z*_Ti | Z*_Ba | 文献值 | 状态 |
|---|---|---|---|---|
| berry_phase (基准) | 6.69 | 2.67 | 7.18 / 2.74 | ✅ |
| DeltaP berry_connection | -1633 | 63.5 | — | ❌ 完全错误 |
| DeltaP wannier (per-atom SVD+trunc) | -14.5 | 12.5 | — | ❌ |
| DeltaP wannier (global SVD+trace) | 47.7 | 13.9 | — | ❌ |
| DeltaP wannier (hybrid Wilson+trace) | -2.09 | 101.1 | — | ❌ |

**全部 DeltaP 方法的 Z* 均不正确。**

### 1.4 逐原子 Pz (参考结构, 两种 DeltaP 方法)

| 原子 | berry_connection | wannier |
|---|---|---|
| Ba | +4.82×10⁻³ | -1.47×10⁻³ |
| Ti | -1.22×10⁻⁴ | +1.46×10⁻³ |
| O1 | +1.32×10⁻³ | -4.26×10⁻³ |
| O2 | -1.47×10⁻³ | -6.74×10⁻³ |
| O3 | -2.55×10⁻⁴ | -3.81×10⁻³ |
| **Sum** | **+4.29×10⁻³** | **-1.48×10⁻²** |
| ABACUS P_elec | -5.79×10⁻³ | -5.79×10⁻³ |

两种方法的逐原子值完全不同，且都不等于 ABACUS 电子极化。

---

## 2. Wannier center 对应的量

### 2.1 定义

Wannier center 是 Wannier 函数的位置期待值 ⟨r_n⟩，单位为 **Bohr**。

与极化的关系：

$$P_{\text{elec}} = -\frac{e}{\Omega} \sum_n \langle r_n \rangle \quad [\text{e/bohr}^2]$$

逐原子分解：将每个 Wannier 函数按其投影中心归属到原子 I：

$$P^I_{\text{elec}} = -\frac{e}{\Omega} \sum_{n \in I} \langle r_n \rangle \quad [\text{e/bohr}^2]$$

### 2.2 与 Berry phase 的关系

Wannier center 从重叠矩阵 M(k) 计算：

$$\langle r_{n,\alpha} \rangle = \frac{a_\alpha}{2\pi} \cdot \mathrm{Im}\,\ln \prod_k \big[U^\dagger(k)\, M(k)\, U(k+\mathbf{b})\big]_{nn}$$

其中 U(k) 是 Wannierization 求得的最优规范变换。

**关键性质**：在 Wannier gauge 中，$\sum_n \langle r_n \rangle$ 精确等于总极化（因为 det = 特征值乘积，ln 是可加的）。

### 2.3 .mmn 文件给出的当前结果

ABACUS Wannier90 接口生成了 .mmn 文件（重叠矩阵 M），但 Wannier90.x 在 I/O 阶段崩溃（独立问题，另行处理）。

从 .mmn 直接计算（未经 Wannierization，在本征态规范下）：

| 量 | 值 | 说明 |
|---|---|---|
| .mmn Berry phase (det, 15 occ bands) | γ = 1.200, P = 3.34×10⁻³ e/bohr² | 精确总 Berry phase |
| .mmn Berry connection (trace) | γ = -0.535, P = -1.49×10⁻³ e/bohr² | 近似（trace≠det） |
| .mmn 逐能带 Wannier center ⟨r_z⟩ | 见下表 | 本征态规范（非 Wannier gauge） |

逐能带 Wannier center ⟨r_z⟩（本征态规范，近似值）：

| Band | ⟨r_z⟩ (Bohr) | | Band | ⟨r_z⟩ (Bohr) |
|---|---|---|---|---|
| 1 | 1.269 | | 9 | -0.445 |
| 2 | 0.101 | | 10 | -0.571 |
| 3 | 1.269 | | 11 | -0.815 |
| 4 | 0.782 | | 12 | -0.328 |
| 5 | -1.583 | | 13 | -0.120 |
| 6 | -1.145 | | 14 | 1.159 |
| 7 | 0.574 | | 15 | 0.790 |
| 8 | -0.088 | | | |
| **Σ** | **0.672/(2π)×a_z** | | .mmn det | **1.200/(2π)×a_z** |

Σ ≠ det：本征态规范下逐能带之和（0.672）≠ 总 Berry phase（1.200），因为 $\prod_n M_{nn} \neq \det M$。

### 2.4 .mmn 与 ABACUS berry_phase 的差异

**.mmn Berry phase ≠ ABACUS Electronic Phase 是预期的**，因为两者使用不同的重叠矩阵：

| | ABACUS berry_phase | .mmn (Wannier90 接口) |
|---|---|---|
| 重叠矩阵 | ⟨ψ_k \| ψ_{k+b}⟩（Bloch 态重叠） | ⟨u_k \| u_{k+b}⟩（周期部分重叠） |
| 关系 | = ⟨u_k \| e^{ib·r} \| u_{k+b}⟩ | = 不含位置算符 |
| 差异 | 含位置算符 e^{ib·r} | 不含 |
| γ (15 bands) | -2.079 | +1.200 |

两者差一个位置算符因子 e^{ib·r}。ABACUS 的 total（electronic + ionic）是正确的，因为位置修正与离子项抵消。但 .mmn 的 electronic 单独与 ABACUS 的 electronic 不可直接比较。

**注意**：DeltaP 两种方法使用的是 Bloch 态重叠（与 ABACUS berry_phase 一致），不是周期部分重叠（与 .mmn 一致）。因此 DeltaP ↔ ABACUS 是正确对比，DeltaP ↔ .mmn 不可直接对比。

---

## 3. 两种 DeltaP 方法的对比

### 3.1 berry_connection 方法

**计算内容**：
- 用解析 ∂_k S 和有限差分 ∂_k C 计算 Berry connection
- A^I = -2 Im Σ ⟨ψ_n|α^I⟩ ∂_k⟨α^I|ψ_n⟩
- 使用 Bloch 态 ψ（与 ABACUS berry_phase 一致）

**结果**：
- P_sum = +4.29×10⁻³ e/bohr²
- ABACUS P_elec = -5.79×10⁻³ e/bohr²
- 比例 = -0.74（不匹配）

**问题**：
1. 符号相反
2. 量级偏差 74%
3. Z* 完全不可靠（Ti=-1633, Ba=63.5）

### 3.2 wannier 方法

**计算内容**（当前代码经过多轮修改）：
- 全局 SVD → V(k), W(k)
- 逐原子权重 w^I_s = Σ_{a∈I} |W_{a,s}|²
- Berry connection: A^I = Σ_s w^I_s · Im[M_j(s,s)]
- 总量用 Wilson loop (det) 精确计算, 逐原子用 trace 比例分解

**结果**：
- P_sum = -1.48×10⁻² e/bohr²
- ABACUS P_elec = -5.79×10⁻³ e/bohr²
- 比例 = 2.56（不匹配）

**问题**：
1. 量级偏差 2.5 倍
2. trace/det 比例在不同结构间剧烈变化（-4.77 到 +4.29），导致 Z* 错误

### 3.3 两种方法都不正确的根本原因

两种方法本质上都计算 **Berry connection**（trace 类量），而正确的极化需要 **Berry phase**（det 类量）：

$$\underbrace{\mathrm{Im}\,\ln\det M}_{\text{Berry phase (精确)}} \neq \underbrace{\mathrm{Im}\,\mathrm{Tr}\, M}_{\text{Berry connection (近似)}}$$

对于 nocc=15 个占据能带和 dk=0.1（10 个 k 点）：
- trace/det 比例 ≈ 0.45-0.56（44-56% 误差）
- 这个误差不是 bug，是 trace（线性）与 log-det（非线性）的数学差异

**Berry phase 不可线性分解**为逐原子（或逐能带）贡献。Berry connection 可以线性分解，但它是 dk→0 极限下的近似，误差太大。

---

## 4. Wannier90 运行问题（另行处理）

Wannier90 3.1.0（conda-forge 安装）在 disentanglement 完成后 I/O 阶段 segfault。这阻止了获取精确的 Wannier center（需要 Wannierization 求最优 U(k)）。

**解决途径**（不影响上述分析）：
1. 从源码编译 Wannier90（避免 conda-forge I/O bug）
2. 使用 PW basis 的 Wannier90 接口（避免 LCAO 接口的投影数问题）
3. 使用更简单的体系（如 Si）先验证流程

---

## 5. 文件位置

| 内容 | 路径 |
|---|---|
| berry_connection 方法结果 | `/tmp/opencode/bto_methods/berry_conn/OUT.autotest/deltap_results.dat` |
| wannier 方法结果 | `/tmp/opencode/bto_methods/wannier/OUT.autotest/deltap_results.dat` |
| .mmn/.amn 文件 | `/tmp/opencode/w90_bto/equilibrium/OUT.autotest/seedname.*` |
| DeltaP 源码 | `source/source_lcao/module_deltap/deltap_wannier.cpp` |
| DeltaP Berry connection 源码 | `source/source_lcao/module_deltap/deltap_berry.cpp` |
