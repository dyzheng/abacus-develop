# DeltaP 增量开发设计 — 基于 DeltaSpin 基础设施

> **基线代码**: ABACUS DeltaSpin（SMO投影 + 双层Lagrange框架 + 力/应力修正）  
> **目标**: 在DeltaSpin已有轮子上实现DeltaP约束极化方法  
> **日期**: 2026-06-24

---

## 目录

1. [DeltaSpin基础设施回顾](#1-deltaspin基础设施回顾)
2. [DeltaSpin → DeltaP 映射分析](#2-deltaspin--deltap-映射分析)
3. [增量开发设计](#3-增量开发设计)
4. [代码修改清单](#4-代码修改清单)
5. [测试验证策略](#5-测试验证策略)
6. [风险点排查](#6-风险点排查)

---

## 1. DeltaSpin基础设施回顾

DeltaSpin在ABACUS中实现了非共线自旋约束DFT的完整框架，包含以下可复用组件：

### 1.1 SMO投影算子（Section II.B, Eq. 18-24）

```
核心算子: P^I_{lmm'} = |α^I_lm⟩⟨α^I_lm'|

其中 α_{Ilm}(r) = α_l(|r-τ_I|)Y_lm 是Smooth Modulation Orbital
```

**实现细节**（来自Note）：
- 投影轨道的径向函数通过截断NAO的ζ函数后平滑得到：`α(r) = χ(r)·g(r;σ) / ⟨χg|χg⟩^{1/2}`
- 平滑函数：`g(r;σ) = 1 - exp(-(r-rm)²/(2σ²))` for r < rm, 0 otherwise
- 在LCAO基组中occupation matrix：`n^{σσ'}_{Ilmm'} = Σ_R Σ_{μν} ρ^{σσ'}_{μν}(R) ⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm'|ϕ^R_ν⟩`
- 在PW基组中：`n^{σσ'}_{Ilmm'} = Σ_{nk} f_{nk} S^{σ*}_{Inklm} S^{σ'}_{Inklm'}`
- 原子磁矩：`M^p_I = Σ_{σσ'} Σ_{lmm'} σ^p_{σσ'} n^{σσ'}_{Ilmm'} δ_{mm'}`

### 1.2 Lagrange双层循环框架（Section II.C, Eq. 29-38）

```
E_c = E_KS[ρ,m] - Σ_I λ_I · (M_I[m] - M_{I,target})
```

**NAO基组中H^λ的矩阵表示**（Eq. 33-34）：

```
H^{λ,σσ'}_{μν}(R) = Σ_I f(I,σσ') · Σ_{lm} ⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm|ϕ^R_ν⟩
f(I,σσ') = [[λ^z_I, λ^x_I+iλ^y_I], [λ^x_I-iλ^y_I, -λ^z_I]]
```

**PW基组中的H^λ作用**（Eq. 38）：

```
ĥ_λ|c^σ_{nk}(G)⟩ = Σ_I Σ_{σ'} f(I,σσ') Σ_{lm} S^{σ'}_{Inklm} |α_{Ilm}(k+G)⟩
```

### 1.3 预存储HContainer（Note: 内循环中哈密顿量构建）

```
H^{pre,I}_{μν}(R) = Σ_{lm} ⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm|ϕ^R_ν⟩
H^{λ,σσ'}_{μν}(R) = Σ_I f(I,σσ') · H^{pre,I}_{μν}(R)
```

在SCF过程中预存储一次，内循环只需做加权求和。

### 1.4 外循环策略（Note: 外循环策略改进）

```
Phase 1: 固定λ的SCF → 收敛电荷密度ρ(r)（不运行λ内循环）
Phase 2: λ内循环+完整SCF
  Step a: SCF到目标threshold
  Step b: λ内循环优化M→M_target → 得到λ1
  Step c: update_pot（更新有效势以匹配新的磁密度）
  Step d: λ内循环优化M→M_target → 得到λ2（二次估计）
  Step e: 重置密度混合历史 → 继续正常DeltaSpin SCF
```

### 1.5 力与应力修正（Eq. 36-37）

LCAO基组需要额外计算Pulay项（2,3,5），PW基组无需额外计算。修正力的形式为：

```
F^{λ,p}_I = -2 Σ_R Σ_{μν} Σ_{σσ'} ρ^{σσ'}_{μν}(R) Σ_{R'} Σ_{lm}
            ⟨ϕ⁰_μ|α^{R'}_{Ilm}⟩ f(I,σσ') ∂/∂τ^p_I ⟨α^{R'}_{Ilm}|ϕ^R_ν⟩
```

### 1.6 高效λ内循环算法（Note: 内循环迭代算λ高效算法）

- 对共线磁矩(NSPIN=2)：可忽略δc，只计算δε→δw→δM
- 对非共线磁矩(NSPIN=4)：不可忽略δc
- λ更新：通过微扰响应计算∂M/∂λ，用CG优化δM-δλ

---

## 2. DeltaSpin → DeltaP 映射分析

### 2.1 结构级对比

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeltaSpin 结构            DeltaP 对应       │
├─────────────────────────────────────────────────────────────────┤
│ 约束量              M^p_I (原子磁矩)        P^I_α (原子极化)     │
│ Lagrange函数        E_KS - Σλ·(M-M_tgt)    E_KS + Σλ·(P-P_tgt)  │
│ 投影算符            |α^I_lm⟩⟨α^I_lm'|      同，但需k空间版本     │
│ 投影量计算          实空间ρ(R)            k空间A^I_{k,n,α}       │
│ H^λ结构             Σf(I)⟨ϕ|α⟩⟨α|ϕ⟩       |∂_kα⟩⟨α| + |α⟩∂_k⟨α| │
│ λ物理意义           磁扭矩(magnetic torque) 局域电场              │
│ 基组                NAO + PW               NAO + PW             │
│ 双层循环            内λ循环+外SCF循环       同框架               │
│ HContainer预存      H^{pre,I}(R)           需扩展为k依赖版本     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 可直接复用的组件 ✅

| 组件 | DeltaSpin | DeltaP复用方式 |
|------|-----------|---------------|
| SMO构造 | `α_{Ilm}(r) = α_l(r)Y_{lm}` | **完全复用**，无需修改 |
| SMO投影算符 | `P^I_{lmm'} = \|α^I_{lm}⟩⟨α^I_{lm'}\|` | **完全复用** |
| NAO occup. matrix | `n^{σσ'}_{Ilmm'} = Σρ⟨ϕ\|α⟩⟨α\|ϕ⟩` | 复用结构，但需k空间版本(见3.3) |
| PW overlap | `S^σ_{Inklm} = Σ_G α*(G) c^σ_{nk}(G)` | **完全复用** |
| Lagrange框架 | `E_c = E_KS + Σλ(M-M_tgt)` | 框架复用，改符号和物理量 |
| 双层循环 | 内λ循环+外SCF循环 | **框架复用** |
| HContainer预存 | `H^{pre,I}_{μν}(R)` | 结构复用，需k扩展(见3.5) |
| λ更新策略 | CG优化δM-δλ | 改为δP-δλ CG优化 |
| 力/应力修正 | Pulay项计算 | 框架复用，公式需调整 |
| 外循环两阶段 | Phase1固定λ+Phase2内循环 | **完全复用** |

### 2.3 需要新开发的部分 🆕

| 组件 | 说明 | 难度 |
|------|------|------|
| `|∂_k α^I_{lmk}⟩` 计算 | NAO基组：解析`iR_α e^{ikR} |φ⟩`；PW基组：解析`i(G+k)_α` | ⭐⭐ |
| 原子极化P^I_α计算 | k空间Berry connection积分（需after-SCF评估） | ⭐⭐⭐ |
| H^λ新结构 | `|∂_kα⟩⟨α| + |α⟩∂_k⟨α|` vs DeltaSpin的`|α⟩⟨α|` | ⭐⭐⭐ |
| k点依赖 | DeltaSpin是实空间框架，DeltaP本质是k空间量 | ⭐⭐ |
| H^λ非Hermitian处理 | Berry connection含虚部 | ⭐⭐ |
| 极化量子化处理 | P^I只能确定到模eR/Ω | ⭐ |

---

## 3. 增量开发设计

### Phase A: 后处理原型（SMO投影 → 原子极化计算）🟢 **推荐起点**

**目标**：不修改ABACUS SCF循环，利用SCF输出的波函数做后处理，验证SMO投影框架可以正确分解原子极化。

#### A.1 实现方案

```
输入: ABACUS SCF输出 (H(R), S(R), C_{nμ}(k), E_{nk})
       + SMO投影轨道 {α^I_{lm}}
输出: 每原子极化分量 P^I_α (α=x,y,z)
```

**核心技术路线**——在原Berry phase推导中，原子投影极化需要Berry connection在k空间的积分。但在DeltaP增量方案中，我们利用ABACUS已有的Berry phase基础设施来计算总极化，然后通过SMO投影建立原子分解。

**后处理原型不需要改ABACUS源码，只需编写独立Python脚本**：

```python
# Phase A prototype workflow
def compute_atomic_polarization():
    # Step 1: Read ABACUS outputs
    H_R, S_R = read_HSR("abacus_output/")
    C_nk, E_nk = read_wavefunctions("abacus_output/")
    kpts, kweights = read_kpoints()

    # Step 2: Load SMO projection orbitals (reuse DeltaSpin's orbital generation)
    smo_orbitals = load_smo_orbitals(atom_indices, l_channels, rm)

    # Step 3: For each k-point, compute atom-projected Berry connection
    for k_idx, k in enumerate(kpts):
        for n in occupied_bands:
            # --- Compute term1: ⟨ψ_nk|∂_kα α^I_lmk⟩⟨α^I_lmk|ψ_nk⟩ ---
            for I in atoms:
                for lm in channels[I]:
                    # Parse |∂_kα α^I_lmk⟩  (Eq. 3.1 in main design doc)
                    # NAO: analytic i·R_α·exp(ikR)|φ⟩
                    grad_alpha_k = compute_grad_alpha_analytic(I, lm, k, α)

                    # ⟨ψ_nk|∂_kα α⟩ = C†_n(k)·S(k)·grad_alpha_k_coeffs
                    bra1 = dot(C_nk.conj(), S_k, grad_alpha_k)

                    # ⟨α|ψ_nk⟩ = SMO projection (reuse DeltaSpin formula)
                    bra2 = smo_overlap(I, lm, C_nk, k)

                    term1 += bra1 * bra2

                    # --- Compute term2: ⟨ψ_nk|α^I_lmk⟩ ∂_kα⟨α^I_lmk|ψ_nk⟩ ---
                    # Finite diff of SMO overlap on neighboring k-points
                    bra_psi_alpha = smo_overlap(I, lm, C_nk, k)
                    grad_alpha_psi = finite_diff_overlap(I, lm, n, k, α, dk)
                    term2 += bra_psi_alpha * grad_alpha_psi

            A_I_k[n, α] = term1 + term2

    # Step 4: k-space integration → Berry phase → polarization
    for I in atoms:
        gamma_I_α = kweights[k] * Σ_n A_I_k[n, α]
        P_I_α = e * gamma_I_α / (2π * a_α)

    # Step 5: Verify sum rule
    P_total = sum(P_I_α for I in atoms)
    assert abs(P_total - P_abacus) / abs(P_abacus) < 0.01

    return P_I_α
```

#### A.2 从DeltaSpin复用的关键计算

```
复用: SMO overlap计算
  DeltaSpin公式 (Note Eq. 3):
    n^{σσ'}_{Ilmm'} = Σ_R Σ_{μν} ρ^{σσ'}_{μν}(R) ⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm'|ϕ^R_ν⟩

  DeltaP对应（k空间版本）:
    ⟨α^I_lmk|ψ_nk⟩ = Σ_μ (S_{μ,Ilm}(k))* · C_{nμ}(k)
    其中 S_{μ,Ilm}(k) = (1/√N) Σ_R e^{-ikR} ⟨ϕ⁰_μ|α^I_lm(r-R-τ_I)⟩
```

#### A.3 测试验证（见Section 5, T0-T1）

---

### Phase B: SCF集成的DeltaP

**目标**：将DeltaP的H^λ修正集成进ABACUS SCF循环，实现自洽约束极化计算。

#### B.1 H^λ算符的矩阵表示

从用户推导，NAO基组下：

```
H^λ |ψ_nk⟩ = (e/2πa) Σ_{I,α} λ^I_α · i Σ_{lm} [
    |∂_{k_α}α^I_{lmk}⟩⟨α^I_{lmk}|ψ_{nk}⟩
  + |α^I_{lmk}⟩ ∂_{k_α}⟨α^I_{lmk}|ψ_{nk}⟩
]
```

将其转化为密度矩阵泛函导数（类似DeltaSpin Eq. 30-33的链式法则）：

**关键差异**：DeltaSpin的H^λ ∝ Σ_{lm} ⟨ϕ|α⟩⟨α|ϕ⟩是**实空间、与k无关**的算符。而DeltaP的H^λ显含k点依赖和波函数依赖——不能简单写成H^{pre,I}的线性组合。

**但这不阻碍增量开发！** 可以将H^λ分为两部分处理：

**Part 1（与DeltaSpin同构）**：`|α⟩⟨α|ψ⟩` 类型的项

```
H^{λ,part1}_{μν}(R) 结构同DeltaSpin: Σ_{I,α} g(I,α) · Σ_{lm} ⟨ϕ⁰_μ|α^I_{lm}⟩⟨α^I_{lm}|ϕ^R_ν⟩
其中 g(I,α) 是 λ^I_α 和 ∂_k⟨α|ψ⟩系数的结合
```

这部分可以直接复用HContainer预存储。

**Part 2（新结构）**：`|∂_k α⟩⟨α|ψ⟩` 类型的项

```
H^{λ,part2}_{μν}(R) = Σ_I Σ_{lm} [...] ⟨ϕ⁰_μ|∂_k α^I_{lmk}⟩⟨α^I_{lm}|ϕ^R_ν⟩
```

需要新构造一个 `HContainer_grad` 预存储 `⟨ϕ⁰_μ|∂_k α^I_{lmk}⟩` 类型的矩阵元。

#### B.2 双层循环适配

DeltaP的内循环同样更新λ以匹配约束：

```
Algorithm: DeltaP λ内循环（复用DeltaSpin框架）

Input: 当前电荷密度ρ(r), Kohn-Sham势V_eff(r)
       目标极化 P^{I,target}_α
Output: 满足约束的 λ^I_α

1. 使用初猜λ构建 H_eff = H_KS + H^λ
2. 对角化求解 {C_{nμ}, E_{nk}}
3. 计算当前极化 P^I_α（调用Phase A后处理逻辑）
4. 更新λ:
   δλ = (P^I_α - P^{I,target}_α) · (∂P/∂λ)^{-1}
   第一步用微扰响应估计 ∂P/∂λ
   后续步骤用CG或Broyden优化
5. 判断收敛: max|P^I - P^{I,target}| < tol
6. 若不收敛，返回步骤1
```

**关键优化**（复用DeltaSpin Note的高效内循环思想）：
- 对绝缘体（非金属），ΔP由价带主导，可以只在费米面附近的能带做子空间对角化
- δP主要由δε贡献（占据数变化），δc贡献较小（波函数系数变化）
- 可以复用"先算δε→δw→δP，只在必要时算δc"的策略

#### B.3 HContainer扩展

DeltaSpin已有的 `HContainer` 存储 `⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm|ϕ^R_ν⟩`。

DeltaP需要两种新Container：

```cpp
// Type 1: 复用DeltaSpin原有 (⟨α|ϕ⟩ 投影)
HContainer H_pre_I_atom;  // 已有

// Type 2: 新增 |∂_kα⟩⟨α| 类型 (k-dependent!)
//          存储 ⟨ϕ⁰_μ|∂_{k_α}α^I_{lmk}⟩ 系数
//          NAO: ∂_{k_α}α^I_{lmk} = (i/√N)·Σ_R R_α·e^{ikR}|φ^I_{lm}(r-R-τ_I)⟩
HContainer_grad H_pre_grad_I_atom_k;  // 新增，依赖k和α

// Type 3: 新增 ⟨ϕ⁰_μ|α^I_{lm}⟩ 基本重叠（SMO→NAO basis transform）
//         用于快速计算 ⟨α^I_{lmk}|ψ_{nk}⟩ = Σ_μ S*_{μ,Ilm}(k)·C_{nμ}(k)  
OverlapContainer S_smo_nao;  // 新增，存储SMO与NAO基函数的重叠积分
```

### Phase C: PW基组实现（较低优先级）

PW基组的H^λ作用形式：

```
ĥ_λ|c^σ_{nk}(G)⟩ = (e/2πa) Σ_{I,α} λ^I_α · i Σ_{lm} [
    |∂_{k_α}α_{Ilm}(k+G)⟩ · S^{σ}_{Inklm}
  + |α_{Ilm}(k+G)⟩ · ∂_{k_α}S^{σ}_{Inklm}
]
```

其中：
- `S^{σ}_{Inklm} = Σ_G α*_{Ilm}(G) c^σ_{nk}(G)` 是DeltaSpin已有的SMO-wavefunction overlap
- `∂_{k_α}S^{σ}_{Inklm}` 是新的k导数项
- `|∂_{k_α}α_{Ilm}(k+G)⟩` 在PW基组下：`∂_{k_α}α_{Ilm}(k+G) = i(G_α+k_α)·α_{Ilm}(k+G)`（解析）

PW实现比NAO简单，因为不需要处理基组正交项和Pulay项。

---

## 4. 代码修改清单

### 4.1 ABACUS Source修改点

```
DeltaSpin已有 (不修改):
  source/module_hamilt_general/module_deltaspin/
    deltaspin_hamilt.cpp          # H^λ 构造与预存储
    deltaspin_force_stress.cpp     # 力与应力修正
    deltaspin_inner_loop.cpp       # λ内循环优化
    deltaspin_orbital.cpp          # SMO构造与投影

DeltaP新增/修改:
  source/module_hamilt_general/module_deltap/    [NEW]
    deltap_hamilt.cpp              # H^λ 构造（新结构：含|∂_kα⟩项）
    deltap_prestore.cpp            # HContainer_grad 预存储
    deltap_berry_connection.cpp    # k空间Berry connection计算
    deltap_inner_loop.cpp          # λ内循环（复用DeltaSpin框架，改M→P）
    deltap_force_stress.cpp        # 力/应力修正（公式需调整）

  source/module_io/                [MODIFY]
    input.cpp                      # 新增输入参数 deltap_* 

  source/module_elecstate/         [MODIFY]
    elecstate.cpp                  # SCF循环中插入DeltaP内循环

  source/module_cell/              [MODIFY]
    read_atoms.cpp                 # 读取原子极化目标

新的输入参数:
  deltap                  1          # 启用DeltaP
  deltap_target          0.5 0.0 0.0 # 目标极化 (C/m²)  
  deltap_atom             1          # 约束原子编号
  deltap_gdir             3          # 极化方向
  deltap_lambda_init      0.0        # 初始λ
  deltap_tol_p            1e-6       # 极化收敛阈值
  deltap_rm               3.0        # SMO modulation radius (复用DeltaSpin)
```

### 4.2 关键数据结构新增

```cpp
// 预存储的 |∂_kα⟩ 重叠矩阵元
struct GradAlphaContainer {
    // ⟨ϕ⁰_μ|∂_{k_α}α^I_{lmk}⟩ for each (k, I, lm, α)
    std::vector<std::complex<double>> coeffs;  // dim: [n_basis]
    int atom_idx, l, m, direction;  // direction: α ∈ {0,1,2}
    Vector3<double> kpoint;
};

// k空间SMO投影结果
struct SMOProjectionK {
    // ⟨α^I_{lmk}|ψ_{nk}⟩ for each (k, n, I, lm)
    std::vector<std::complex<double>> overlaps;  // dim: [n_bands]
    
    // ∂_k⟨α^I_{lmk}|ψ_{nk}⟩ 
    std::vector<std::complex<double>> grad_overlaps;  // dim: [n_bands][3]
};

// 原子极化中间量
struct AtomicPolarization {
    double P[3];          // 极化矢量 (x,y,z)
    double gamma[3];      // Berry phase
    std::vector<std::complex<double>> A_k;  // Berry connection at k-points
};
```

### 4.3 开发阶段与工时估计

| 阶段 | 内容 | 工时 | 依赖 |
|------|------|------|------|
| **A.1** | Python后处理原型：SMO投影→原子极化 | 3天 | 无（独立脚本） |
| **A.2** | Sum Rule验证 + 已知体系测试 | 2天 | A.1 |
| **B.1** | HContainer_grad预存储实现 | 3天 | A.1验证通过 |
| **B.2** | H^λ新结构实现（Part1+Part2） | 4天 | B.1 |
| **B.3** | λ内循环适配（M→P） | 2天 | B.2 |
| **B.4** | 力/应力修正公式实现 | 2天 | B.2 |
| **B.5** | 有限差分验证（力/应力/λ） | 2天 | B.3+B.4 |
| **B.6** | 完整体系收敛性测试 | 3天 | B.5 |
| **C.1** | PW基组DeltaP实现 | 3天 | B.6 |
| **总计** | | **~24天** | |

---

## 5. 测试验证策略

基于DeltaSpin已验证的有限差分测试框架 [DeltaSpin Fig. 4]，设计递进式五层测试：

### 测试层T0：单k点Berry连接元验证 🔬

**目的**：验证 `|∂_kα⟩` 解析公式 和 `∂_k⟨α|ψ⟩` 数值差分的正确性

**复用**：DeltaSpin的SMO投影框架

**做法**：
1. 选孤立原子（H或Li），手算 `|∂_kα^H_{1s,k}⟩` 并与解析公式对比
2. 对孤立原子，用有限差分（Δk=10^{-6}）计算 `∂_k⟨α|ψ⟩`，与中心差分对比
3. 验证 `⟨ψ|∂_kα⟩⟨α|ψ⟩ + ⟨ψ|α⟩∂_k⟨α|ψ⟩` 的模规范不变性

**通过判据**：解析 vs 有限差分 相对误差 < 10^{-8}

---

### 测试层T1：Sum Rule验证 ✅

**目的**：验证 `Σ_I P^I_α = P^{total}_α`（ABACUS内置Berry phase结果）

**复用**：ABACUS `berry_phase=1` 输出

**测试体系**：
- PbTiO3（铁电，自发极化 ~0.87 C/m²，已知基准）
- BaTiO3（铁电，~0.26 C/m²）
- AlAs（共价，~0，电子响应主导 [Diéguez-Vanderbilt 2006]）

**做法**：
1. 运行ABACUS标准Berry phase计算
2. 后处理计算 `Σ_I P^I_α`
3. 比较两者，验证基组完备性

**通过判据**：
- `|Σ P^I - P^{total}| / |P^{total}| < 0.01` (1%)
- 不同调制半径rm（2.0, 3.0, 4.0 Bohr）下的稳定性

---

### 测试层T2：λ=0极限验证 ⚡

**目的**：验证DeltaP在λ=0时不影响基态（同DeltaSpin的FM基态回归测试）

**复用**：DeltaSpin的外循环Phase 1策略（固定λ SCF）

**测试体系**：Si（中心对称，P=0），PbTiO3（基态极化）

**做法**：
1. 设置 `λ^I_α = 0`，运行DeltaP SCF
2. 对比标准SCF与DeltaP SCF（λ=0）的：总能量（差异<10^{-6} Hartree）、电荷密度（<10^{-5} e/Å³）、能带结构（<10^{-4} eV）
3. 验证 DeltaP输出 `P^I_α` 与后处理计算值一致

**通过判据**：同上 + DeltaP SCF在λ=0时输出和后处理计算一致

---

### 测试层T3：λ→P响应验证 📈

**目的**：验证λ导致正确的极化响应，且响应矩阵合理

**复用**：DeltaSpin的λ→M响应测试方法 [DeltaSpin Fig. 5]

**测试体系**：LiF（离子晶体，P~0.2 C/m²）

**做法**：
1. 对F原子施加 `λ^F_z`，保持Li的λ=0
2. 扫描 `λ_z ∈ [-0.05, 0.05]` a.u.（~10个点）
3. 记录 `P^F_z(λ_z)` 和 `P^Li_z(λ_z)`
4. 验证单调性、对称性、线性响应区

**通过判据**：
- `P^F_z` 单调依赖 `λ_z`
- `∂P^F_z/∂λ_z > 0`（物理：电场增大→极化增大，同DeltaSpin中M随λ增大）
- 交叉敏感性 `|∂P^Li_z/∂λ^F_z|/|∂P^F_z/∂λ^F_z| < 0.1`

---

### 测试层T4：有限差分验证 🔬🔬

**目的**：用有限差分法验证所有解析导数（力、应力、λ）

**复用**：DeltaSpin Fig. 4的验证框架

**测试体系**：PbTiO3（含位移自由度）、BaTiO3

**做法**：
1. 力验证：扰动原子位置 ~0.01 Bohr，有限差分 `ΔE/Δτ ≈ -F`
2. 应力验证：扰动晶格参数，有限差分 `ΔE/Δε ≈ Ω·σ`
3. λ验证：扰动目标极化，有限差分 `ΔE/ΔP ≈ λ`（模eR/Ω量子）

**通过判据**：
- 解析力与有限差分的最大偏差 < 10^{-4} Hartree/Bohr（同DeltaSpin标准）
- 解析应力与有限差分的最大偏差 < 10^{-4} Hartree/Bohr³
- λ与 `ΔE/ΔP` 的一致性

---

### 测试层T5：E(P)曲线基准对比 🎯

**目的**：与Diéguez-Vanderbilt (2006) PRL的恒总极化结果对比

**测试体系**：
- Ba(Ti_{1-δ},Ti_{1+δ})O3（SRV模型体系，δ=0, 0.2, 0.3, 0.4, 0.6）
- AlAs（电子主导极化）
- KNbO3（离子主导极化）

**做法**：
1. DeltaP约束总极化到目标值 P_target
2. 扫描 P_target 获得 E(P) 曲线
3. 与 D&V (2006) 图2, 3 对比

**通过判据**：
- E(P)曲线形状定性一致
- 自发极化值差异 < 5%
- 能量曲率（介电常数 ε = 1 + 4π/(d²E/dP²)）差异 < 10%

---

### 测试层级总览

| 测试层 | 名称 | DeltaSpin复用 | 时间 | 阻塞风险 |
|--------|------|---------------|------|---------|
| T0 | 单k点验证 | SMO投影 | 1h | 低 |
| T1 | Sum Rule | Berry phase输出 | 2h | 低 |
| T2 | λ=0极限 | Phase1外循环 | 2h | 低 |
| T3 | λ→P响应 | λ→M扫描 | 4h | 低 |
| T4 | 有限差分 | Fig.4框架 | 8h | 中 |
| T5 | E(P)基准 | - | 12h | 中 |

---

## 6. 风险点排查

### 风险1：H^λ的k点依赖导致无法简单预存储 ⚠️ **HIGH**

**问题**：DeltaSpin的H^λ ∝ ⟨ϕ|α⟩⟨α|ϕ⟩与k无关，可以在实空间预存储HContainer。DeltaP的H^λ含 `|∂_kα⟩⟨α|` 项，显含k依赖。

**缓解方案**：
1. NAO基组：`∂_{k_α}α^I_{lmk} = (i/√N)Σ_R R_α e^{ikR}|φ^I_{lm}(r-R-τ_I)⟩`。虽然与k有关，但预存储 `⟨ϕ⁰_μ|φ^I_{lm}(r-R-τ_I)⟩`（二中心重叠积分，与k无关！）。对每个k点，用解析权重 `i·R_α·e^{ikR}/√N` 做快速加权求和。
2. 存储开销：`n_kpoints × n_atoms × n_lm × n_basis × 3_directions` 个复数。对典型体系（100 k点，10原子，9轨道/原子，1000基函数），约 100×10×9×1000×3 ≈ 54 MB（可接受）。
3. 实际上可以只对不可约k点预存储，其他k点通过对称性生成。

**结论**：可控。存储需求在可接受范围。

### 风险2：Berry phase的量子化模 ⚠️ **HIGH**

**问题**：极化只能确定到模 eR/Ω。当约束P_target跨过极化量子时，λ会发生跳变。

**缓解方案**：
1. 约束收敛容差 tol_P < eR/Ω（极化量子/10），确保不会误收敛到相邻分支
2. 参考D&V (2006)的处理：从基态出发adiabatically调整P_target，跟踪极化分支
3. 检测λ跳变 > threshold时，检查是否跨过极化量子

### 风险3：ΔP^I与原子位移的耦合 ⚠️ **MEDIUM**

**问题**：原子极化P^I与原子位置τ耦合（Born有效电荷效应），外循环中λ更新和原子弛豫可能互相干扰。

**缓解方案**：
1. 复用DeltaSpin的外循环策略（Phase1固定λ收敛电荷密度→Phase2内循环）
2. 在结构弛豫时使用嵌套循环：外层更新原子位置，内层DeltaP SCF
3. 必要时使用D&V (2006)的扩展Newton方法（Eq. 4, 联合求解 ΔX 和 ΔE）

### 风险4：k点密度需求 ⚠️ **MEDIUM**

**问题**：Berry phase积分需要沿极化方向的稠密k点，增加计算成本。

**缓解方案**：
1. 沿gdir方向指数稠密k点（如8→16→32自动收敛测试）
2. 非极化方向的k点保持正常密度
3. 利用Wannier插值（ABACUS已有Wannier90接口）减少k点需求
4. 内存换取时间：更多k点但每个k点对角线化独立，天然并行

---

## 附录：DeltaSpin代码中DeltaP可复用的关键函数

```cpp
// === SMO投影 (完全复用) ===
// source/module_hamilt_general/module_deltaspin/deltaspin_orbital.cpp
void SMO::build_alpha(int atom_idx, int l, double rm);
    // 构造 SMO: α(r) = χ(r)·g(r;σ)/⟨χg|χg⟩^{1/2}

void SMO::compute_overlap_NAO(...);
    // 计算 ⟨ϕ⁰_μ|α^I_lm⟩ 二中心重叠积分

// === HContainer预存储 (结构复用) ===
// source/module_hamilt_general/module_deltaspin/deltaspin_hamilt.cpp
void DeltaSpin::prestore_H_container();
    // 预存储 H^{pre,I}_{μν}(R) = Σ_{lm} ⟨ϕ⁰_μ|α^I_lm⟩⟨α^I_lm|ϕ^R_ν⟩
    // → DeltaP扩展为 H^{pre_grad,I}_{μν}(R,k,α)

// === λ内循环 (框架复用) ===
void DeltaSpin::inner_loop_optimize_lambda();
    // 双层循环：固定ρ→优化λ→更新M→判断收敛
    // → DeltaP：固定ρ→优化λ→计算P→判断收敛

// === 力/应力修正 (框架复用) ===
void DeltaSpin::compute_lambda_force();
    // Pulay项计算 (Eq. Note: F^{λ,p}_I)
    // → DeltaP需要修改投影算符部分

// === 外循环策略 (完全复用) ===
void DeltaSpin::scf_outer_loop();
    // Phase 1: 固定λ SCF → Phase 2: λ内循环+完整SCF
    // → DeltaP完全复用
```
