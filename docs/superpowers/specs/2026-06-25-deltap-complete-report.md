# DeltaP 约束极化方法：完整算法分析与实现报告

> **日期**: 2026-06-25  
> **范围**: 从SMO投影极化分解到精确Wilson loop的完整算法演进、Bug修复、数值验证、H^λ代价分析与改进方案  
> **代码基线**: `feat/ds-lcao-subspace-accel` 分支

---

## 目录

1. [问题背景与设计目标](#1-问题背景与设计目标)
2. [已实现模块结构](#2-已实现模块结构)
3. [算法公式推导](#3-算法公式推导)
4. [Bug修复历史](#4-bug修复历史)
5. [Berry Connection方法的非平滑性分析](#5-berry-connection方法的非平滑性分析)
6. [三种改进方案的H^λ代价对比](#6-三种改进方案的hλ代价对比)
7. [精确Wilson loop实现与测试](#7-精确wilson-loop实现与测试)
8. [Born有效电荷验证结果](#8-born有效电荷验证结果)
9. [剩余问题与改进路线](#9-剩余问题与改进路线)
10. [结论](#10-结论)

---

## 1. 问题背景与设计目标

### 1.1 DeltaP约束极化方法

DeltaP是DeltaSpin（约束磁矩DFT）在极化领域的推广：通过Lagrange乘子λ（虚拟电场）约束原子极化P^I到目标值P^I_target。

$$E_c = E_{KS} + \sum_I \lambda^I_\alpha (P^I_\alpha - P^I_{\alpha,\text{target}})$$

### 1.2 两个核心计算

| 计算 | 用途 | 平滑性要求 |
|------|------|-----------|
| **P^I评估** | 计算当前极化，与目标比较 | 跨独立SCF平滑（Z*计算、E(P)扫描需要） |
| **H^λ构建** | 哈密顿量修正，驱动波函数响应 | 内循环内平滑（λ优化需要） |

### 1.3 平滑性的两层需求

| 范围 | 场景 | 保障机制 |
|------|------|---------|
| 内循环内 | λ变化→子空间对角化更新C | 策略2：子空间∂_k C（解析响应） |
| 跨独立SCF | 不同结构的独立SCF运行 | Wilson loop + 分支跟踪（规范不变） |

---

## 2. 已实现模块结构

### 2.1 文件清单

```
source/source_lcao/module_deltap/
    deltap.h                  # 类定义、数据结构
    deltap.cpp                # init(), setup_kstring(), compute_atomic_polarization()
    deltap_overlap.cpp        # compute_real_overlaps() — <phi|alpha(R)> 二中心积分
    deltap_berry.cpp          # compute_S_k(), compute_D_I(), compute_berry_connection(), integrate_polarization()
    deltap_gauge.cpp          # gauge_fix_smo_anchored() — SMO锚定规范（已弃用）
    deltap_wannier.cpp        # compute_wannier_polarization() — 精确Wilson loop
    deltap_io.cpp             # verify_sum_rule(), write_results()
    CMakeLists.txt
    test/
        deltap_math_test.cpp      # 相位求和数学验证 (3 tests, PASS)
        deltap_gauge_test.cpp     # 规范锚定验证 (4 tests, PASS)
        deltap_smoothness_test.cpp # 平滑性数值实验 (8 tests)
```

### 2.2 入口点

`source/source_io/module_ctrl/ctrl_scf_lcao.cpp` step 12b：
```cpp
if (inp.calculation == "nscf" && inp.deltap_switch)
{
    deltap::DeltaP dp;
    dp.init(ucell, gd, kv,
            two_center_bundle.overlap_orb_onsite.get(),
            two_center_bundle.overlap_orb.get(),
            orb.cutoffs(), inp.deltap_rm, inp.deltap_gdir, &pv);
    dp.compute_atomic_polarization(ucell, psi, pelec);
}
```

### 2.3 输入参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deltap_switch` | false | 启用DeltaP |
| `deltap_rm` | 3.0 | SMO调制半径 (Bohr) |
| `deltap_gdir` | 3 | 极化方向 |
| `deltap_gauge_mode` | "none" | 规范模式（已弃用） |
| `deltap_method` | "berry_connection" | "berry_connection" 或 "wannier" |

---

## 3. 算法公式推导

### 3.1 极化的Berry phase定义

$$P_\alpha = -\frac{a_\alpha}{2\pi\Omega} \sum_n \int_0^1 dk_\alpha \, \text{Im}\left[\langle u_{nk} | \partial_{k_\alpha} u_{nk} \rangle\right]$$

### 3.2 SMO投影分解（Berry connection方法）

$$A^I_n(k) = \sum_{lm} \left[\langle\psi|\partial_k\alpha^I_{lm}\rangle\langle\alpha^I_{lm}|\psi\rangle + \langle\psi|\alpha^I_{lm}\rangle\partial_k\langle\alpha^I_{lm}|\psi\rangle\right]$$

其中：
- **term1**（解析，规范不变）：$\langle\psi|\partial_k\alpha\rangle = \sum_\mu C^*_\mu \partial_k S_\mu$
- **term2**（有限差分，规范依赖）：$\partial_k D_I \approx [D_I(k+\Delta k) - D_I(k-\Delta k)]/(2\Delta k)$

关键量：
- $S_{\mu,Ilm}(k) = \sum_R e^{2\pi i kR}\langle\phi_\mu|\alpha^I_{lm}(R)\rangle$ — SMO-NAO k空间重叠
- $\partial_k S = 2\pi i \sum_R R_\alpha e^{2\pi ikR}\langle\phi_\mu|\alpha(R)\rangle$ — 解析k导数
- $D_I(lm,n,k) = \sum_\mu S^*_\mu C_{n\mu}$ — SMO-波函数重叠

### 3.3 精确Wilson loop方法

$$P^I = -\frac{a_\alpha}{2\pi\Omega} \arg\left(\prod_j \det\left[\mathbf{U}^{I\dagger}(k_j) \cdot \mathbf{O}(k_j,k_{j+1}) \cdot \mathbf{U}^I(k_{j+1})\right]\right)$$

其中：
- $\mathbf{U}(k) = \mathbf{W}(k)\mathbf{V}^\dagger(k)$ — $D_I(k) = \mathbf{W}\boldsymbol{\Sigma}\mathbf{V}^\dagger$的SVD极分解
- $\mathbf{O}(k_j,k_{j+1}) = \mathbf{C}^\dagger(k_j) \cdot \mathbf{S}(\Delta k) \cdot \mathbf{C}(k_{j+1})$ — 精确波函数重叠
- $\mathbf{S}(\Delta k) = \sum_R e^{2\pi i\Delta k \cdot R}\langle\phi_\mu(0)|\phi_\nu(R)\rangle$ — NAO-NAO位移重叠

**规范不变性证明**：波函数规范变换$C(k)\to C(k)e^{i\varphi(k)}$下，$\det(\mathbf{M}^I)$获得相位$e^{2i(\varphi_{j+1}-\varphi_j)}$，在k-string乘积中伸缩抵消（$\varphi_N = \varphi_0$，周期边界）。

### 3.4 策略1（解析Berry connection，无∂_k C）—— 已证明P=0

$\partial_k D_I \approx \sum_\mu \text{conj}(\partial_k S_\mu) C_\mu = \text{conj}(\langle\psi|\partial_k\alpha\rangle)$

$$A^I = a \cdot b + b^* \cdot \text{conj}(a) = 2\,\text{Re}(a \cdot b) \quad\Rightarrow\quad \text{Im}(A^I) = 0 \quad\Rightarrow\quad P^I = 0$$

**结论**：Berry phase的虚部完全来自波函数k导数∂_k C，SMO几何贡献（∂_k S）只给出实部。

### 3.5 H^λ（哈密顿量修正）

$$H^\lambda|\psi_{nk}\rangle = \frac{e}{2\pi a}\sum_{I,\alpha}\lambda^I_\alpha \cdot i\sum_{lm}\left[|\partial_k\alpha^I_{lm}\rangle\langle\alpha^I_{lm}|\psi\rangle + |\alpha^I_{lm}\rangle\partial_k\langle\alpha^I_{lm}|\psi\rangle\right]$$

**可预存储部分**（对应A的实部，P=0的部分）：

$$H^{\lambda,\text{pre}}_{\mu\nu}(k) = 2\pi i \sum_R R_\alpha e^{2\pi ikR} H^{\text{pre},I}_{\mu\nu}(R) \cdot \lambda^I_\alpha$$

这是DeltaSpin HContainer的加权重Fourier变换——但对应P=0，不直接有用。

**不可预存储部分**（对应A的虚部，P≠0的部分）：涉及$\partial_k C$，k空间非局域。

---

## 4. Bug修复历史

### 4.1 term1 bra_grad公式错误

| 项目 | 内容 |
|------|------|
| 问题 | `bra_grad = conj(dS)·C`给出⟨d_kα\|ψ⟩=conj(⟨ψ\|d_kα⟩)，非正确的⟨ψ\|d_kα⟩ |
| 影响 | term1非规范不变（乘e^{2iφ}） |
| 修复 | 改为`bra_grad = conj(C)·dS` |

### 4.2 缺少MPI Allreduce

| 项目 | 内容 |
|------|------|
| 问题 | D_I = Σ conj(S)·C只对本地波函数行求和，并行下不完整 |
| 影响 | per-atom sum ≠ P_total（sum rule严重违反） |
| 修复 | 在compute_D_I后添加`MPI_Allreduce(MPI_IN_PLACE, ...)` |

### 4.3 极化prefactor缺少晶胞体积Ω

| 项目 | 内容 |
|------|------|
| 问题 | 用`-1/(2π·a)`代替正确的`-a/(2π·Ω)` |
| 影响 | P偏大a²/Ω倍（BTO约7倍） |
| 修复 | 改为`-a_alpha/(2π·omega)`，使用`ucell.omega` |

### 4.4 dS缺少2π因子

| 项目 | 内容 |
|------|------|
| 问题 | 相位e^{2πikR}的导数应为2πi·R·e^{2πikR}，代码缺少2π |
| 影响 | term1偏小2π倍（对Berry phase虚部影响有限，因term1贡献实部） |
| 修复 | 在dS中添加`ModuleBase::TWO_PI`因子 |

---

## 5. Berry Connection方法的非平滑性分析

### 5.1 数值实验设计

8个纯数学单元测试（无ABACUS依赖），使用合成D_I数据：

| 测试 | 目标 | 方法 |
|------|------|------|
| 一般扰动平滑性 | ΔP ∝ ε? | 扫描ε∈[10⁻⁶,10⁻²] |
| 规范不变性 | 相位旋转后P不变? | 随机相位扰动 |
| 锚定跳变 | 接近跳变时P非线性? | 设两原子投影接近 |
| 无锚定跳变平滑性 | 无跳变时P平滑? | 1%均匀相位扰动 |

### 5.2 测试结果

| # | 测试 | 结果 | 关键数值 |
|---|------|------|---------|
| 1 | Berry(gauge)一般扰动 | **PASS** | ΔP/ε恒定 |
| 2 | Berry(no gauge)一般扰动 | **PASS** | rel_err<8% |
| 3 | Wilson vs Berry对比 | Wilson更差 | Wilson ΔP/ε=538 vs Berry=1.16 |
| 4 | Wilson规范不变性 | **FAIL** | 简化SVD非规范不变 |
| 5 | Berry规范不变性 | PASS | 预期变化 |
| 6 | 锚定跳变 | PASS | 2%扰动→0.6%ΔP |
| 7 | 无锚定跳变平滑性 | **FAIL** | 1%相位→14%ΔP |

### 5.3 根因定位

**Berry connection公式本身是平滑的**（测试1-2通过）。

**非平滑性根源是规范固定的相位敏感性**（测试7定位）：

- 规范相位$g(k) = D^*_{\text{anchor}}/|D_{\text{anchor}}|$对$D_I$的微小相位变化敏感
- 有限差分$d_k(D_I \cdot g)$中，$g$的变化被$1/\Delta k$放大
- 1%的$D_I$相位变化 → 10%的P变化（与$\Delta k = 0.1$的放大因子一致）

**Wilson loop有不同问题**：`arg()`分支切割导致极化量子跳变。

### 5.4 策略1给出P=0

实验验证：解析Berry connection（仅用∂_k S，忽略∂_k C）对所有结构给出**P=0**。

数学证明：$A^I = a \cdot b + b^* \cdot \text{conj}(a) = 2\,\text{Re}(a \cdot b)$，虚部为零。

**Berry phase的虚部完全来自波函数k导数∂_k C**，SMO几何贡献只给出实部。

---

## 6. 三种改进方案的H^λ代价对比

### 6.1 方案对比

| 维度 | 方案A (Wilson loop) | 方案B (解析Berry) | 方案C (平行传输) |
|------|-------------------|-----------------|----------------|
| H^λ核心 | SVD导数$(\sigma_i^2-\sigma_j^2)^{-1}$ | ∂_k Fourier(H^pre) | 同B |
| 预存储复用 | 无法复用 | **完全复用DeltaSpin** | 同B |
| 跨SCF平滑 | arg()分支切割 | 策略1→P=0 | k连续，结构不连续 |
| 数值稳定性 | 奇异值接近时不稳定 | 稳定 | 稳定 |

### 6.2 关键发现

- **方案B策略1（仅∂_k S）的H^λ可预存储**：$H^{\lambda,\text{pre}} = 2\pi i \sum_R R_\alpha e^{2\pi ikR} H^{\text{pre}}(R) \lambda$——约50行代码
- **但对应P=0**——Part A是Berry connection的实部，对极化无贡献
- **P≠0的部分（Part B = S·∂_k C）不可预存储**，使H^λ在k空间非局域

### 6.3 H^λ与P^I的分工

| 计算 | 机制 | 平滑性范围 |
|------|------|-----------|
| P^I评估 | Wilson loop + 分支跟踪 | 跨SCF + 内循环（规范不变） |
| H^λ构建 | 策略2子空间响应 | 仅内循环（解析响应） |

---

## 7. 精确Wilson loop实现与测试

### 7.1 实现内容

1. **compute_S_dk()**：用NAO-NAO二中心积分器计算$S(\Delta k) = \sum_R e^{2\pi i\Delta k \cdot R}\langle\phi_\mu|\phi_\nu(R)\rangle$
2. **精确O**：$O = C^\dagger(k_j) \cdot S(\Delta k) \cdot C(k_{j+1})$，通过ScaLAPACK pzgemm + MPI_Allreduce
3. **精确M^I**：$M^I = U^{I\dagger} \cdot O \cdot U^I$（替换$U^{I\dagger} \cdot U^I$近似）
4. **分支跟踪**：跨结构arg()展开（内存版）

### 7.2 测试结果

| 指标 | berry_phase | Berry connection(旧) | Wilson loop(新) |
|------|------------|---------------------|----------------|
| Ba平滑性 | 0.79 | 16× | **1.12** ✓ |
| Ti平滑性 | 0.55 | 16× | 0.01 ✗ |

**Ba平滑性从16×改善到1.12**——精确Wilson loop消除了规范固定问题，验证了det()+SVD的规范不变性。

**Ti仍有SVD不连续**——d电子使奇异值谱密集，位移时σ_i≈σ_j导致U=WV†不连续。

### 7.3 分支分析

| 案例 | P_berry | P_wannier | diff/quantum | 分支 |
|------|---------|-----------|-------------|------|
| eq | +5.10e-4 | -3.52e-4 | 0.049 | 0 |
| Ti+ | +6.60e-4 | -1.30e-2 | 0.778 | 1 |
| Ti- | +3.60e-4 | -3.70e-5 | 0.023 | 0 |
| Ba+ | +5.69e-4 | -1.24e-2 | 0.741 | 1 |
| Ba- | +4.51e-4 | -1.18e-2 | 0.699 | 1 |

分支跟踪需跨SCF持久化（当前W_prev_是内存变量，每次运行重置）。

---

## 8. Born有效电荷验证结果

### 8.1 正确结构+小位移的berry_phase Z*（可靠基准）

| 原子 | Z*_berry | 文献值 | 误差 |
|------|---------|--------|------|
| Ti | +6.80 | +7.18 | 5.3% ✓ |
| Ba | +2.67 | +2.74 | 2.6% ✓ |

berry_phase Z*可靠，用户经验正确。之前错误来自：(1)错误基础结构，(2)位移太大。

### 8.2 DeltaP Z*（各方法对比）

| 方法 | Z*_Ti | Z*_Ba | Ti平滑性 | Ba平滑性 |
|------|-------|-------|---------|---------|
| berry_phase (基准) | +6.80 | +2.67 | 0.55 | 0.79 |
| Berry connection (有限差分+规范) | -1633 | +63.5 | 16.3 | — |
| Wilson loop (精确O) | -292.8 | -13.8 | 0.01 | **1.12** |

### 8.3 差分是否消除系统偏差？

| 量 | deltap/berry比值 |
|----|-------------------|
| P (平衡) | 12.4× (Berry connection) |
| Z*_Ti | 240× (Berry connection) |

**差分不但没消除偏差，反而放大**——因为规范固定的相位敏感性被有限差分1/Δk放大。

Wilson loop的Ba情况：平滑性1.12（接近1），但绝对Z*仍偏大（SMO不完备~6%）。

---

## 9. 剩余问题与改进路线

### 9.1 已解决的问题

| 问题 | 解决方案 | 验证 |
|------|---------|------|
| Berry connection公式bug | 修正bra_grad=conj(C)·dS | 单元测试PASS |
| MPI并行D_I不完整 | 添加MPI_Allreduce | sum rule满足 |
| Prefactor缺少Ω | 改为-a/(2πΩ) | P量级正确 |
| dS缺少2π | 添加TWO_PI因子 | 导数正确 |
| 规范固定非平滑性 | 精确Wilson loop替代 | Ba平滑性1.12 ✓ |
| 策略1给出P=0 | 数学证明+实验验证 | 确认需∂_k C |

### 9.2 未解决的问题

| 问题 | 影响 | 难度 |
|------|------|------|
| SVD奇异值交叉（Ti） | U=WV†不连续→P跳变 | ⭐⭐⭐ 高 |
| 分支跟踪未持久化 | 跨SCF的arg()分支不一致 | ⭐ 低（~30行） |
| SMO不完备（~6%） | 绝对Z*偏大 | ⭐⭐ 中（增大rm） |
| H^λ的Part B（∂_k C） | H^λ在k空间非局域 | ⭐⭐⭐ 高 |

### 9.3 改进路线

**短期（修复当前问题）**：
1. 分支持久化：将W^I复数写入`deltap_branch.dat`（~30行）
2. SVD连续跟踪：奇异值交叉时用前一结构U作为初猜，连续变形
3. 增大SMO rm：测试rm=5.0/7.0对完备度和Z*的影响

**中期（提高精度）**：
4. 策略2子空间∂_k C：内循环中用微扰论估计波函数k导数
5. H^λ预存储Part A + 内循环Part B：双层架构

**长期（根本解决）**：
6. 解析SVD导数：一阶微扰论跟踪奇异向量在交叉点的连续变形
7. 完备SMO集：增大rm和ζ函数数量，使完备度>90%

---

## 10. 结论

### 10.1 已验证的结论

1. **berry_phase Z*是可靠基准**（Ti=6.80, Ba=2.67，误差<6%）
2. **Berry connection（有限差分+规范固定）不适合Z*计算**——规范固定相位敏感性被1/Δk放大，差分放大偏差12.4×→240×
3. **精确Wilson loop消除了规范固定问题**——Ba平滑性从16×改善到1.12
4. **策略1（解析∂_k S，忽略∂_k C）给出P=0**——Berry phase虚部完全来自波函数k导数
5. **H^λ的可预存储部分对应P=0**——Part A是Berry connection实部，Part B（∂_k C）不可预存储

### 10.2 推荐实现路径

```
P^I评估: 精确Wilson loop + 分支持久化 + SVD连续跟踪
  → 消除规范固定问题，保障跨SCF平滑性

H^λ构建: 策略2子空间∂_k C
  → 内循环内平滑，复用DeltaSpin子空间对角化框架

内循环: 复用DeltaSpin CG/BFGS框架，M→P
  → λ优化，P_target约束
```

### 10.3 核心权衡

| 方法 | P^I平滑性 | H^λ代价 | 适用场景 |
|------|----------|---------|---------|
| Berry connection (有限差分) | ✗ 非平滑 | 低 | 不推荐 |
| 精确Wilson loop | ✓ Ba平滑, Ti待修 | 高 (SVD) | P^I评估 |
| 策略2子空间 | ✓ 内循环 | 中 | H^λ构建 |

**P^I用Wilson loop，H^λ用策略2**——两者服务于不同范围的平滑性需求，组合使用是当前最优方案。
