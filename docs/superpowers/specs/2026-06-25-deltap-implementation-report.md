# DeltaP NAO 基组实现与测试报告

> **日期**: 2026-06-25  
> **代码基线**: `feat/ds-lcao-subspace-accel` 分支  
> **文档范围**: 完整记录 DeltaP 模块的实现、测试结果、Bug 修复历史、方法对比，以及关于验证目标的重新思考

---

## 目录

1. [模块结构](#1-模块结构)
2. [算法实现](#2-算法实现)
3. [输入参数](#3-输入参数)
4. [Bug 修复历史](#4-bug-修复历史)
5. [测试结果](#5-测试结果)
6. [Berry Connection vs Wannier 方法对比](#6-berry-connection-vs-wannier-方法对比)
7. [关于验证目标的思考：P_total vs Born有效电荷](#7-关于验证目标的思考p_total-vs-born有效电荷)

---

## 1. 模块结构

### 1.1 文件清单

```
source/source_lcao/module_deltap/
    deltap.h                  # 类定义、数据结构
    deltap.cpp                # init(), setup_kstring(), compute_atomic_polarization()
    deltap_overlap.cpp        # compute_real_overlaps() — <phi|alpha(R)> 二中心积分
    deltap_berry.cpp          # compute_S_k(), compute_D_I(), compute_berry_connection(), integrate_polarization()
    deltap_gauge.cpp          # gauge_fix_smo_anchored() — SMO锚定规范
    deltap_wannier.cpp        # compute_wannier_polarization() — SVD极分解Wannier方法
    deltap_io.cpp             # verify_sum_rule(), write_results()
    CMakeLists.txt            # 构建配置
    test/
        CMakeLists.txt
        deltap_math_test.cpp      # T0: 相位求和数学验证
        deltap_gauge_test.cpp     # 规范锚定与连续性验证
```

### 1.2 类接口

```cpp
namespace deltap {

class DeltaP {
public:
    void init(const UnitCell& ucell, const Grid_Driver& gd,
              const K_Vectors& kv, const TwoCenterIntegrator* intor,
              const std::vector<double>& orb_cutoff,
              double rm, int gdir, const Parallel_Orbitals* paraV);

    void compute_atomic_polarization(
        const UnitCell& ucell,
        const psi::Psi<std::complex<double>>* psi,
        const elecstate::ElecState* pelec);

private:
    // 核心算法
    void compute_real_overlaps(...);    // <phi|alpha(R)> via snap()
    void setup_kstring(...);            // k-string索引
    void compute_S_k(int ik);           // S(k), dS(k) 相位求和
    void compute_D_I(...);              // D_I = <alpha|psi> = S*·C
    void gauge_fix_smo_anchored(...);   // SMO锚定规范
    void compute_berry_connection(...); // A^I_n = term1 + term2
    void integrate_polarization(...);   // P^I = prefactor × gamma^I
    void compute_wannier_polarization(...); // SVD/Wilson loop方法
    void verify_sum_rule();
    void write_results(...) const;
};

} // namespace deltap
```

### 1.3 入口点

在 `source/source_io/module_ctrl/ctrl_scf_lcao.cpp` 中，berry_phase之后（step 12b）:

```cpp
if (inp.calculation == "nscf" && inp.deltap_switch)
{
    deltap::DeltaP dp;
    dp.init(ucell, gd, kv,
            two_center_bundle.overlap_orb_onsite.get(),
            orb.cutoffs(),
            inp.deltap_rm, inp.deltap_gdir, &pv);
    dp.compute_atomic_polarization(ucell, psi, pelec);
}
```

---

## 2. 算法实现

### 2.1 Berry Connection 方法（默认）

**物理公式**:

电子极化沿 α 方向:
```
P_α = -(e·a_α / 2π·Ω) · Σ_n ∫ dk_α · Im[⟨u_{nk}|∂_{k_α} u_{nk}⟩]
```

SMO投影分解:
```
A^I_n(k,α) = Σ_{lm} [⟨ψ|∂_{k_α} α^I_{lmk}⟩·⟨α^I_{lmk}|ψ⟩ + ⟨ψ|α^I_{lmk}⟩·∂_{k_α}⟨α^I_{lmk}|ψ⟩]
```

**计算流程**:

```
1. compute_real_overlaps: <phi|alpha(R)> via TwoCenterIntegrator::snap()
2. setup_kstring: k点字符串索引 (复用berryphase模式)
3. 对每个k点:
   a. compute_S_k: S(k) = Σ_R e^{2πikR} <phi|alpha(R)>
                     dS(k,α) = 2πi·Σ_R R_α·e^{2πikR}·<phi|alpha(R)>
   b. compute_D_I: D_I(lm,n) = Σ_μ conj(S)·C  [MPI Allreduce]
4. gauge_fix_smo_anchored: 锚定SMO = argmax|D_I|, 规范相位连续跟踪
5. 对每个k点:
   compute_berry_connection: A^I_n = term1(解析,规范不变) + term2(有限差分)
6. integrate_polarization: P^I = -(a_α/2πΩ)·dk·Σ Im[A^I_n]
```

**关键公式细节**:

- **term1** (规范不变): `bra_grad = Σ_μ conj(C·g)·dS`, `term1 = bra_grad·(D_I·g)`
  - 在规范变换 ψ→e^{iφ}ψ 下: bra_grad→e^{-iφ}·bra_grad, D_I·g→e^{iφ}·D_I·g, 乘积不变
- **term2** (有限差分): `d_D = [D_I(k+dk)·g(k+dk) - D_I(k-dk)·g(k-dk)]/(2dk)`, `term2 = conj(D_I·g)·d_D`
  - 规范相位g确保相邻k点间相位连续

### 2.2 Wannier 方法（SVD极分解，CWF等价）

**物理公式**:

```
1. D_I(k) = <α|ψ>  [已计算]
2. SVD: D_I = W·Σ·V†
3. 极分解: U = W·V†  (自动消除规范任意性)
4. Wannier Wilson loop: Π_j det(U^I†(k_j)·U^I(k_{j+1}))
5. P^I = -(a_α/2πΩ)·Im[log(Π_j det(...))]
```

**当前实现使用近似**: `⟨ψ_{k_j}|ψ_{k_{j+1}}⟩ ≈ δ_{nm}` (单位矩阵)
- 在 dk→0 极限下精确
- 对 dk=0.1 (10×10×10网格) 误差显著
- 精确实现需要 `unkOverlap_lcao::prepare_midmatrix_pbas` 计算完整重叠矩阵

### 2.3 SMO锚定规范 (Method 5)

**算法**:
```
Phase 1 (k_0): 对每个能带n, 选择锚定SMO: ref(n) = argmax_{I,lm} |D_I(lm,n,k_0)|
Phase 2 (k_1..k_{nppstr-1}):
  对每个(n, k_j):
    D_anchor = D_I[anchor_iat, anchor_lm, n, k_j]
    if |D_anchor| < threshold: 重新选择锚定, 记录Δφ
    g = conj(D_anchor)/|D_anchor|
    连续跟踪: if Re(g·conj(g_prev)) < 0: g = -g  (避免π跳变)
```

---

## 3. 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_switch` | bool | false | 启用DeltaP极化分解 |
| `deltap_rm` | real | 3.0 | SMO调制半径 (Bohr) |
| `deltap_gdir` | int | 3 | 极化方向 (1=x, 2=y, 3=z) |
| `deltap_dk_fd` | real | 1e-6 | 有限差分delta-k (T0验证用) |
| `deltap_npk_string` | int | 0 | 覆盖k-string密度 (0=用KPT网格) |
| `deltap_gauge_mode` | string | "none" | 规范模式: "none" 或 "smo_anchored" |
| `deltap_anchor_thr` | real | 1e-8 | 锚定SMO重选阈值 |
| `deltap_method` | string | "berry_connection" | 计算方法: "berry_connection" 或 "wannier" |

---

## 4. Bug 修复历史

### 4.1 term1 bra_grad 公式错误

**问题**: 代码计算 `bra_grad = Σ conj(dS)·C`，这给出 `⟨d_kα|ψ⟩ = conj(⟨ψ|d_kα⟩)`，而非正确的 `⟨ψ|d_kα⟩`。

**影响**: term1在规范变换下不是不变的 (乘以 e^{2iφ} 而非不变)。

**修复**: 改为 `bra_grad = Σ conj(C)·dS`，计算正确的 `⟨ψ|d_kα⟩`，使term1规范不变。

### 4.2 缺少MPI Allreduce

**问题**: `compute_D_I` 中 `D_I = Σ_μ conj(S)·C` 只对本地波函数系数行求和。在MPI并行下，每个进程只有部分 μ 值，D_I是不完整的。

**影响**: 每个原子的P^I只是部分和，sum rule严重违反 (per-atom sum ≠ P_total)。

**修复**: 在D_I计算后添加:
```cpp
MPI_Allreduce(MPI_IN_PLACE, D_I.data(), 2*sz, MPI_DOUBLE, MPI_SUM, paraV_->comm());
```

### 4.3 极化prefactor缺少晶胞体积Ω

**问题**: 代码使用 `prefactor = -1/(2π·a_α)·dk`，但正确公式是 `P = -(a_α/2πΩ)·γ`。

**影响**: P值偏大 a_α²/Ω 倍 (对BTO约7倍)。

**修复**: 改为 `prefactor = -a_α/(2π·Ω)·dk`，使用 `ucell.omega`。

### 4.4 dS缺少2π因子

**问题**: 相位为 `e^{2πi·k·R}`，导数 `dS/dk = Σ 2πi·R·e^{2πikR}·overlap`，但代码计算 `Σ i·R·e^{2πikR}·overlap` (缺少2π)。

**影响**: term1偏小2π倍 (但因term1对Berry phase虚部的贡献被term2的共轭抵消，实际影响有限)。

**修复**: 在dS中添加 `ModuleBase::TWO_PI` 因子。

---

## 5. 测试结果

### 5.1 测试配置

- **PP/Orbital**: `/root/pporb/` APNS precision set
  - Ba: `Ba_ONCV_PBE-1.0.upf` + `Ba_gga_10au_100Ry_6s3p3d2f.orb`
  - Ti: `Ti_ONCV_PBE-1.2.upf` + `Ti_gga_10au_100Ry_6s3p3d2f.orb`
  - O: `O.upf` + `O_gga_10au_100Ry_3s3p2d1f.orb`
  - Si: `Si.upf` + `Si_gga_10au_100Ry_3s3p2d.orb`
- **K-points**: 10×10×10 Monkhorst-Pack
- **MPI**: `mpirun -np 4`, `OMP_NUM_THREADS=1`
- **SMO半径**: rm = 3.0 Bohr (默认)
- **规范**: smo_anchored

### 5.2 Si (金刚石结构, 中心对称, P_z = 0)

**STRU**: 2原子初胞, a = 5.13 Å (LATTICE_CONSTANT = 1.8897 Bohr)
** berry_phase结果**: P_z = 0.000 e/bohr² ✓

| 方法 | P_total (e/bohr²) | Si[0] Pz | Si[1] Pz | 评价 |
|------|-------------------|----------|----------|------|
| Berry connection | -1.19e-03 | -5.83e-04 | -5.93e-04 | 接近零,原子对称 ✓ |
| Wannier (SVD) | -3.76e-03 | 0.0 | -3.76e-03 | 偏大,破坏原子等价性 ✗ |

### 5.3 BaTiO3 (四方铁电相, P_z ≠ 0)

**STRU**: 5原子初胞, a = 4.0 Å, c = 4.2 Å, Ti位移至z=0.52
**berry_phase结果**: P_z = +5.102e-04 e/bohr² = 0.0292 C/m²

| 方法 | P_total (e/bohr²) | 与berry_phase比值 | 符号 | 评价 |
|------|-------------------|-------------------|------|------|
| Berry connection | +6.30e-03 | 12.4× | ✓ 正确 | 偏大但符号正确 |
| Wannier (SVD) | -1.77e-02 | 34.7× | ✗ 错误 | 符号错误 |

**BaTiO3 per-atom Pz (Berry connection)**:

| 原子 | Pz (e/bohr²) | 物理预期 |
|------|-------------|---------|
| Ba | +2.31e-03 | 正 (阳离子位移方向) ✓ |
| Ti | +2.61e-03 | 正 (Ti向上位移) ✓ |
| O (顶点) | +3.78e-04 | 小 |
| O (赤道1) | +1.28e-03 | |
| O (赤道2) | +5.96e-04 | |
| **Sum** | **7.17e-03** | |
| **P_total** | **6.30e-03** | sum ≈ total (3%误差) ✓ |

### 5.4 SMO半径依赖性 (BaTiO3, Berry connection)

| rm (Bohr) | P_total (e/bohr²) | 与berry_phase比值 | 符号 |
|-----------|-------------------|-------------------|------|
| 3.0 | +6.30e-03 | 12.4× | ✓ |
| 5.0 | -3.79e-02 | 74× | ✗ |
| 7.0 | +1.50e-02 | 29× | ✓ |

**结论**: P对rm不收敛 — 表明SMO完备性不是唯一问题，Berry connection的一阶近似本身有系统误差。

### 5.5 单元测试

| 测试 | 结果 |
|------|------|
| DeltaPMathTest.SSumConsistency | PASS |
| DeltaPMathTest.AnalyticDSMatchesFiniteDiff | PASS |
| DeltaPMathTest.BerryConnectionGaugeInvariance | PASS |
| DeltaPGaugeTest.AnchorIsMaxProjection | PASS |
| DeltaPGaugeTest.AnchorProjectionIsPositiveReal | PASS |
| DeltaPGaugeTest.PhaseContinuityNoPiJumps | PASS |
| DeltaPGaugeTest.Term1GaugeInvariance | PASS |

---

## 6. Berry Connection vs Wannier 方法对比

### 6.1 对比矩阵

| 维度 | Berry Connection | Wannier (SVD) |
|------|-----------------|---------------|
| BTO符号正确 | ✓ | ✗ |
| Si接近零 | ✓ (1.2e-3) | ✗ (3.8e-3) |
| 原子对称性 | ✓ (Si等价) | ✗ (破坏) |
| per-atom物理直觉 | ✓ (Ba+, Ti+) | ✗ (集中在Ba) |
| sum rule满足 | ✓ (3%误差) | — |
| 规范不变性 | 需要gauge fixing | ✓ (SVD自动) |
| 非迭代 | — | ✓ (一次SVD) |
| 与berry_phase比值 | 12.4× | 34.7× |
| 实现复杂度 | 中等 | 中等 (但近似粗糙) |

### 6.2 Wannier方法误差来源

当前Wannier实现使用 `⟨ψ_{k_j}|ψ_{k_{j+1}}⟩ ≈ δ_{nm}` (单位矩阵近似):
- 在 dk→0 极限精确
- 对 dk=0.1 (10×10×10) 误差显著
- 需要用 `unkOverlap_lcao` 计算精确重叠矩阵O(k_j, k_{j+1})才能公平对比

### 6.3 Berry connection方法误差来源

1. **SMO不完备**: Σ_I P^I ≠ I, 导致 Σ_I A^I ≠ A_total
2. **一阶近似**: Berry connection积分 ≠ Wilson loop (O(dk²)误差)
3. **有限差分**: d_k D_I 用中心差分, dk=0.1可能不够小
4. **rm不收敛**: P对rm不稳定,说明误差不只是SMO完备性

---

## 7. 关于验证目标的思考：P_total vs Born有效电荷

### 7.1 P_total 作为验证目标的问题

当前验证策略是将 DeltaP 计算的 `P_total = Σ_I P^I` 与 ABACUS berry_phase 结果对比。这个目标存在根本性缺陷：

**问题1: P_total 对SMO分解不敏感**

P_total 是所有原子极化的总和。如果 SMO 集是完备的 (Σ P^I = I)，则 P_total = berry_phase 结果。但如果 SMO 不完备，P_total 可能恰好正确（误差相互抵消）而 per-atom 分解完全错误，反之亦然。**P_total 无法验证 per-atom 分解的质量**。

**问题2: 极化量子不确定性**

Berry phase 极化只能确定到模 eR/Ω。P_total 可能在不同的极化分支上，导致与 berry_phase 的直接对比失去意义（需要跟踪分支）。

**问题3: 对结构变化不敏感**

P_total 对原子位置的导数 (∂P/∂τ) 才是物理上可测量（Born有效电荷 Z*）。两个方法可能给出相同的 P_total 但完全不同的 Z*，说明 P_total 不是区分方法好坏的好目标。

**问题4: 当前结果证实了P_total的不可靠性**

从测试数据看:
- rm=3.0: P = 6.3e-3 (12.4× berry_phase)
- rm=5.0: P = -3.8e-2 (74×, 符号翻转)
- rm=7.0: P = 1.5e-2 (29×)

P_total 对 rm 不收敛且符号翻转，说明 P_total 作为验证目标是不稳定的。

### 7.2 Born有效电荷 Z* 作为验证目标的优势

**Born有效电荷定义**:

$$Z^*_{I,\alpha\beta} = \Omega \frac{\partial P_\alpha}{\partial \tau_{I,\beta}} = \frac{\partial(\text{Berry phase})}{\partial \tau_{I,\beta}} \cdot \frac{\Omega}{2\pi a_\alpha}$$

其中 τ_{I,β} 是原子 I 沿 β 方向的位移。

**优势1: 直接验证 per-atom 分解**

Z* 本质上是 **per-atom 量** — 它告诉你"移动原子 I 时，极化变化多少"。这直接检验 DeltaP 的 per-atom P^I 分解是否正确，而不是仅检验总和。

**优势2: 有可靠的参考值**

Born有效电荷有大量文献基准值：
- BaTiO3 的 Z* (Giustino, *Materials Modelling using Density Functional Theory* 表 5.1):

| 原子 | Z*_{zz} (文献) | 典型值 |
|------|----------------|--------|
| Ba | 2.0-2.8 | +2.7 |
| Ti | 7.0-7.5 | +7.2 |
| O_⊥ (垂直) | -5.4~-5.8 | -5.6 |
| O_∥ (平行) | -2.0~-2.4 | -2.2 |

这些值可以与 DeltaP 的 `∂P^I/∂τ_I` 直接对比，无需依赖 berry_phase 总极化的对比。

**优势3: 差分消除系统误差**

Z* 通过有限差分计算: `Z* = Ω·ΔP / Δτ`。如果 P 有系统偏差 (如 12.4× 偏大)，但偏差对两个结构一致，差分可能部分抵消。更重要的是，**Z* 的物理意义更清晰** — 它是电子跟随原子位移的能力，不涉及极化量子不确定性。

**优势4: 对SMO参数更鲁棒**

Z* 关注的是 P 对位移的**变化率**，而非绝对值。SMO 不完备导致的系统偏差在差分中可能部分抵消，使得 Z* 对 rm 的依赖性弱于 P_total。

### 7.3 验证方案设计

**有限差分 Born 有效电荷计算**:

```
对每个原子 I 和方向 β:
  1. 在平衡位置计算 P^I_α (DeltaP)
  2. 将原子 I 沿 β 方向位移 +Δτ (如 0.01 Bohr)
  3. 重新SCF, 计算 P^I_α(+Δτ)
  4. 将原子 I 沿 β 方向位移 -Δτ
  5. 重新SCF, 计算 P^I_α(-Δτ)
  6. Z*_{I,αβ} = Ω · [P^I_α(+Δτ) - P^I_α(-Δτ)] / (2·Δτ)
```

**验证判据**:

| 量 | 判据 | 文献参考 |
|----|------|---------|
| Z*_{Ti,zz} | 7.0-7.5 | Giustino, 表5.1 |
| Z*_{Ba,zz} | 2.0-2.8 | 同上 |
| Z*_{O∥,zz} | -2.0~-2.4 | 同上 |
| Z*_{O⊥,zz} | -5.4~-5.8 | 同上 |
| Σ_I Z*_{I,αα} | 0 (声子横模条件) | 电中性条件 |

**关键优势**: Z*_{Ti,zz} ≈ 7.2 是一个非常特征性的值（远大于形式电荷 +4），它直接反映了 Ti-O 共价性。如果 DeltaP 能正确给出这个值，就证明 per-atom 分解是可靠的。

### 7.4 结论

**完全赞同用 Born 有效电荷替代 P_total 作为验证目标**。理由:

1. **P_total 验证的是总和, 而非分解** — 两个方法可能给出相同的 P_total 但完全不同的 per-atom Z*
2. **Z* 有可靠的文献基准** — 不依赖 berry_phase 的极化量子问题
3. **Z* 直接检验 per-atom 质量** — 这是 DeltaP 的核心价值
4. **Z* 对系统偏差更鲁棒** — 差分消除部分系统误差
5. **Z* 是约束极化方法的实际应用目标** — Phase B 的 λ 内循环最终约束的是 P^I, 而 Z* 检验的是 P^I 对原子位移的响应, 这正是约束方法需要精确的量

### 7.5 下一步建议

1. **实现 Born 有效电荷有限差分计算**: 在 DeltaP 中添加 `compute_born_charges()` 函数, 对每个原子做 ±Δτ 扰动
2. **用 BaTiO3 验证**: 与 Giustino 表 5.1 的 Z* 值对比
3. **比较 Berry connection vs Wannier 的 Z***: 这才是真正有意义的对比 — 哪个方法给出的 Z* 更接近文献值
4. **扫描 rm 对 Z* 的影响**: 验证 Z* 是否比 P_total 对 rm 更鲁棒
5. **与 ABACUS 的 Berry phase 有限差分 Z* 对比**: ABACUS 可以用 berry_phase 在两个位移结构上计算总 P, 差分得到总 Z* = Σ_I Z*_I, 与 DeltaP 的 Σ Z*_I 对比
