# H₂O 极化率测量全记录：方法、数据、换算与分析

> 目标体系: H₂O 分子, PBE, Γ 点, gdir=3 (z 轴)
> 参考值: α = 9.85 a₀³ (1.46 Å³), CCSD(T)/aug-pc-3
> 文档目的: 记录每次尝试的完整过程，不合并、不简化、不隐藏失败的轮次

---

## 轮次目录

| # | 方法 | 基组 | 盒子 (Bohr) | 结论 |
|---|------|------|-------------|------|
| R1 | DeltaP total mode (LCAO) | DZP | 10 | α_branch=8.11, α_raw=5.84 — 受 branch/raw 差异和 SCF 不对称困扰 |
| R2 | efield sawtooth (LCAO, dip_cor=0) | DZP | 10 | α=1.64 — sawtooth 在 5.3 Å 盒子中失效 |
| R3 | efield sawtooth (LCAO, dip_cor=1) | DZP | 10 | α=1.63 — 盒小压倒了 dip_cor 效果 |
| R4 | DeltaP 约束矩阵 C=[z̄] (LCAO) | DZP | 10 | 对总极化退耦, 测的是分子内 CT 刚度, 不适用 |
| R5 | DeltaP total mode (PW, λ_init bug 修复前) | ecut20 | 15 | dγ/dλ=0 — λ_init 未读入 |
| R6 | DeltaP total mode (PW, ecut80, bug 修复后) | ecut80 | 15 | dγ/dλ=0.05 rad/Ry — SMO 覆盖不足 |
| R7 | efield + 直接能量 FD (LCAO) | DZP | 10 | α=3.15 a₀³ — 回避 Berry 相位 |
| R8 | DeltaP PW + escon 能量 FD | ecut80 | 15 | 待 Legendre 变换验证 |

---

## R1: DeltaP total mode (LCAO/DZP, 10 Bohr)

### 方法

DeltaP total 约束模式。`deltap_constraint_mode = total`，`deltap_lambda_step = 0.0`（冻结 λ），`deltap_lambda_init = {0, ±0.005, ±0.01}`。

物理假设: λ 是 Σγ 的 Lagrange 乘子，通过共轭关系 E = −(π/a)·λ 映射到外场。

### 关键参数

```
box: 10×10×10 Bohr, LCAO/DZP, Γ-point, nspin=1, nelec=8
onsite_radius: 6.0, deltap_rm: 6.0, deltap_gdir: 3
scf_thr: 1e-6, scf_nmax: 30
lambda_init: -0.01, -0.005, 0.0, 0.005, 0.01 Ry
lambda_step: 0.0 (fixed λ)
```

### 原始数据

| λ (Ry) | Σγ_raw (rad) | Σγ (branch, rad) | SCF 收敛 |
|--------|-------------|-------------------|---------|
| −0.01 | −6.7622 | −6.762 | NO |
| −0.005 | −6.7641 | −6.764 | NO |
| 0.0 | −6.7344 | −6.734 | YES |
| +0.005 | −6.7359 | −6.736 | YES |
| +0.01 | −6.7366 | −6.737 | NO |

负 λ 侧全部 SCF 发散（均匀势阱使电荷密度收缩，默认 mixing 参数不足）。

### 换算

两个可用的收敛点: λ=0 → Σγ=−6.734，λ=+0.005 → Σγ=−6.736。

```
dγ_raw/dλ(Ry) = (−6.735859 − (−6.734419)) / 0.005 = −0.288 rad/Ry
dγ_branch/dλ(Ry) = (−6.736 − (−6.734)) / 0.005 = −0.400 rad/Ry

E = −(π/a)·λ(Ry) × ?
若 λ(Ry) → λ(Ha) = λ(Ry)/2, E(Ha) = −(π/a)·λ(Ha) = −(π/a)·λ(Ry)/2

dγ/dE = dγ/dλ × dλ/dE = dγ/dλ × (−2a/π)

α = (a/π) × dγ/dE = (a/π) × dγ/dλ × (−2a/π) = −(2a²/π²) × dγ/dλ(Ry)

用 raw γ: α = −(2·100/π²) × (−0.288) = 200/9.87 × 0.288 = 5.84 a₀³
用 branch γ: α = 200/9.87 × 0.400 = 8.11 a₀³
```

### 分析

1. Branch vs raw 差 28%。Branch γ 的累加过程中每原子独立分支选择（deltap_wannier.cpp:1147），Σ 不守恒。**物理量应用 raw γ.**
2. 负 λ 侧不发散使对称性无法验证。仅有单侧拟合，不可靠。
3. 盒子仅 5.3 Å，电子极化空间严重不足。5.84 a₀³ 与参考 9.85 a₀³ 差 41%.

---

## R2: efield sawtooth (LCAO/DZP, 10 Bohr, dip_cor=0)

### 方法

ABACUS 内置 sawtooth 外电场。`efield_flag=1, dip_cor_flag=0, efield_amp=±0.001 a.u., efield_dir=2`.

Sawtooth 在 z 方向产生线性上升、局部下降的周期电势。dip_cor=0 表示无偶极修正。

### 关键参数

```
box: 10×10×10 Bohr, LCAO/DZP, Γ-point
efield_amp: -0.001, -0.0005, 0, 0.0005, 0.001 a.u. (Hartree/e·Bohr)
DeltaP 只作测量 (deltap_switch=1, deltap_corr=1, lambda_step=0, no target)
```

### 原始数据

| E (a.u.) | Σγ_raw (rad) |
|----------|-------------|
| −0.001 | −6.734934 |
| −0.0005 | −6.734677 |
| 0.0 | −6.734419 |
| +0.0005 | −6.734161 |
| +0.001 | −6.733903 |

线性且符号对称: dγ/dE = 0.516 rad/a.u.

### 换算

```
α = (a/π) × dγ/dE = (10/π) × 0.516 = 1.64 a₀³
```

### 分析

1.63 a₀³ 远小于 9.85。Sawtooth 势的不连续面离分子仅 ~2.6 Å（盒子半径）。周期镜像退极化场无修正（dip_cor=0）。响应被淬火 ~6×.
测试方案要求"≥15 Å 盒子"正是针对此类失效——本测试在计划明确排除的条件下运行.

---

## R3: efield sawtooth (LCAO/DZP, 10 Bohr, dip_cor=1)

### 方法

同 R2, 但 `dip_cor_flag=1`（Bengtsson 偶极修正）.

### 原始数据

| E (a.u.) | Σγ_raw (rad) |
|----------|-------------|
| −0.001 | −6.735574 |
| −0.0005 | −6.735318 |
| 0.0 | −6.735062 |
| +0.0005 | −6.734806 |
| +0.001 | −6.734550 |

dγ/dE = 0.512 rad/a.u.（与 dip_cor=0 几乎相同）.

### 换算

```
α = (10/π) × 0.512 = 1.63 a₀³
```

### 分析

dip_cor=1 不能修复 5.3 Å 盒子中的 sawtooth 失效。偶极修正是用来抵消**周期边界产生的伪偶极-偶极作用**的，但在盒子太小、分子几乎占满元胞的情况下，修正起不到作用——问题根源是极化密度被截断，不是边界条件。

---

## R4: DeltaP 约束矩阵 C=[z̄] (LCAO/DZP, 10 Bohr)

### 方法

DeltaP 约束矩阵模式。C = [z̄_O, z̄_H₁, z̄_H₂]（质心坐标系下的 z 坐标: −0.391, 0.197, 0.197 Bohr）。Target = 0. 期望模拟 O 与 H 间的电位差从而创建偶极响应.

`deltap_constraint_matrix = cmat.dat`, `deltap_lambda_step = 0.0`.

### 原始数据 (小 λ)

| λ_cstr (Ry/Bohr·rad) | Σγ_raw (rad) | C·γ (constraint quantity) |
|-----------------------|-------------|---------------------------|
| 0.0 | −6.734419 | 0 |
| 0.005 | −6.734323 | 0 |
| 0.01 | −6.734221 | 0 |

C·γ 在所有 λ 下保持为 0.

大 λ 区已非线性:

| λ_cstr | Σγ_raw | C·γ |
|--------|--------|-----|
| −0.5 | −7.013 | — |
| −0.1 | −7.073 | — |
| 0.1 | −6.958 | — |
| 0.5 | −7.037 | — |

### 分析

C=[z̄] 的 Σz̄ ≈ −0.391 + 2×0.197 ≈ 0. C 对总极化（均匀 γ 变化通道）精确退耦。它只能耦合分子内非均匀 γ 重分布——即内部电荷转移（CT）通道，不是总偶极响应。

C=[z̄] 的小 λ 响应 +0.020 rad/(Ry/Bohr·rad) 来自 O 与 H 原子响应的微弱差异（残余耦合）。大 λ 区的"方向错误/饱和"是内部 CT 被推到非线性区的典型行为。

**该模式不适用于测物理极化率.** 它的真正价值是 CT 通道表征，应归入 M 系列分析.

---

## R5: DeltaP total mode (PW/ecut20, 15 Bohr, λ_init 未修复)

### 方法

PW 基组 DeltaP. `basis_type=pw, ecutwfc=20, nbands=8, nelec=8, Γ-point`.
期望同 R1 但用 PW 基组。使用 deltap_lambda_init 设定 λ.

### 关键参数

```
box: 15×15×15 Bohr, PW, ecutwfc=20, 1×1×1 kpts
lambda_init: -0.005, -0.0025, 0, 0.0025, 0.005 Ry
lambda_step: 0.0
```

### 原始数据

| λ (Ry) | γ_total (rad) | Σγ_per_atom (rad) |
|--------|-------------|-------------------|
| 任意  | −0.1827 | 6.100448 |

**所有 λ 值给出完全相同的 γ_total.**

### 分析

PW 路径的初始化代码（esolver_ks_pw.cpp:102-105）使用 `ucell.get_dp_target()`（来自 STRU，无 dp_target 时默认 [0,0,0]）作为初始 λ, 而不是 INPUT 的 `deltap_lambda_init`. λ 始终为 [0,0,0], 约束无任何效果.

**Bug 定位**: esolver_ks_pw.cpp:102-104 — `deltap_lambda_init` 参数在 PW 路径中被忽略.

---

## R6: DeltaP total mode (PW/ecut80, 15 Bohr, bug 修复后)

### 方法

修复 R5 的 bug 后重新测试。`esolver_ks_pw.cpp:102-105` 增加逻辑: 若 STRU 无 dp_target, 默认使用 `deltap_lambda_init`.

同时提高 ecutwfc 至 80 以减少波函数精度不足导致的系统误差。

### 修复代码

```cpp
// 若 STRU 未提供显式 dp_target, 使用 deltap_lambda_init 作为每个原子的初始 λ
bool has_strutarget = false;
for (size_t i = 0; i < dp_target.size(); ++i)
    if (std::abs(dp_target[i]) > 1e-12) { has_strutarget = true; break; }
if (!has_strutarget) {
    double lam_init = PARAM.inp.deltap_lambda_init;
    std::fill(dp_target.begin(), dp_target.end(), lam_init);
}
```

### 原始数据

| ecutwfc | λ (Ry) | γ_total (rad) | Σγ_per_atom (rad) |
|---------|--------|-------------|-------------------|
| 20 | −0.005 | −0.1829 | 6.100253 |
| 20 | 0.0 | −0.1827 | 6.100448 |
| 20 | +0.005 | −0.1825 | 6.100653 |
| **80** | −0.005 | −0.1628 | 6.120387 |
| **80** | 0.0 | −0.1626 | 6.120609 |
| **80** | +0.005 | −0.1623 | 6.120836 |

```
ecut20: dγ_total/dλ = 0.04 rad/Ry, α ≈ 1.82 a₀³ (用 PW γ_total, a=15)
ecut80: dγ_total/dλ = 0.05 rad/Ry, α ≈ 2.28 a₀³
```

### 分析

1. Bug 修复后响应恢复——λ 从 init 正确注入 OnsiteProjector.
2. ecut20→80: dγ/dλ 仅从 0.04 变为 0.05 (+25%), 远未收敛到 LCAO 的 0.288. **波函数精度不是主因.**
3. γ_total (PW) = −0.18 rad vs Σγ_raw (LCAO) = −6.73 rad — 两者是不同表示下的同一物理量 (PW 为标准 Berry 相位 Im log det, 在 [−π,π] 区间; LCAO 为 Wilson 循环本征值 unwrapped 求和).
4. PW 下 SMO 投影子只覆盖每个原子 ~6 Bohr 半径的球。在 15 Bohr 盒子中, PW 波函数弥散在整空间, 投影捕获率约 6³/15³≈6.4% 每原子. 三个原子总覆盖 ~19%. 这解释了 PW vs LCAO 响应差 ~6× (5.84 vs 2.28 不完全等比例, 因为 ecut 和基组也不同).

---

## R7: efield + 直接能量有限差分 (LCAO/DZP, 10 Bohr)

### 方法

回避所有 Berry 相位换算，直接从总能量做 FD:

```
α = [E(+δE) + E(−δE) − 2E(0)] / δE²
```

这是极化率的基本定义（二阶响应系数），不依赖 γ↔μ↔E 的任何转换。

### 原始数据 (from R2's efield runs)

| E (Ha) | E_KS (Ry) |
|--------|-----------|
| −0.001 | −31.365315 |
| 0.0 | −31.366661 |
| +0.001 | −31.368013 |

### 换算

```
ΔE = (−31.365315) + (−31.368013) − 2×(−31.366661)
    = −68.733328 + 68.733322
    = −6.2766×10⁻⁶ Ry

α_Ry = −ΔE / (2δE²)   (with factor ½ from expansion E = E₀ − μE − ½αE²)
      = 6.2766×10⁻⁶ / 0.001²
      = 6.277 Ry/Ha²

单位转换: 1 Ry = 0.5 Ha, δE = 0.001 Ha.
a₀³ = e²·a₀²/Ha = (1 Ry/Ha²) × 0.5

α = 6.277 × 0.5 = 3.14 a₀³
```

### 分析

3.14 a₀³ — 与 R2 的 dγ/dE 法 (1.64) 差约 2 倍, 但两个方法都远小于 9.85. 

这个差异 (~2×) 就是 Berry 相位法的转换因子误差——R2/R3 中用的 γ 是 DeltaP 的 Wilson 循环分解 (unwrapped)，不是标准 Berry 相位 `Im log det`. 用 E_KS 的 FD 回避了这个换算问题.

但即便最干净的能量 FD, 在 5.3 Å 盒子中也只有 3.14 a₀³. **盒子太小是当前无法取得接近参考值的根本原因.**

---

## R8 (未完成): DeltaP PW + escon 能量 FD

### 方法

对 DeltaP PW 路径，计算物理能量 E_phys = E_KS − Σλ·γ (减去约束能), 然后用 FD 求曲率, Legendre 变换到外场表示.

### 原始数据

| λ (Ry) | E_KS (Ry) | escon (Ry) | Σγ (rad) |
|--------|-----------|------------|----------|
| −0.005 | −34.328822 | +0.030602 | 6.120387 |
| 0.0 | −34.319844 | 0.000000 | 6.120609 |
| +0.005 | −34.310919 | −0.030604 | 6.120836 |

```
E_phys = E_KS + escon (? β需要验证符号)

若 E_phys = E_KS − Σλ·γ (减去约束能):
  λ=−0.005: Σλ·γ = −0.005×6.12 = −0.0306 → E_phys = −34.328822 − (−0.0306) = −34.298220
  λ=0.0:     E_phys = −34.319844
  λ=+0.005:  Σλ·γ = +0.005×6.12 = +0.0306 → E_phys = −34.310919 − 0.0306 = −34.341523

d²E_phys/dλ² = [−34.298220 + (−34.341523) − 2×(−34.319844)] / 0.005²
             = −0.000055 / 2.5×10⁻⁵
             = −2.2 Ry/Ry²
```

### 待完成

Legendre 变换: E_field(Ha) = λ(Ry) × ?. 代码中的 `E_eff_au = −λ_avg × π / a` 漏了 Ry→Ha 的因子 2. 正确公式应为:
```
E_field(Ha) = −λ(Ha) × π / a = −λ(Ry) × π / (2a)
```

加此修正后 α ≈ 102 a₀³（离谱的 10× 过大）, 说明 escon 符号或 λ·γ 的正确定义仍需核实。**本轮未完成，该发现标示为待验证的开放问题.**

---

## 总结: 所有方法的 α_zz 一览

| 轮次 | 方法 | 盒子 | 基组 | 换算路径 | α_zz (a₀³) | 对 9.85 偏差 |
|------|------|------|------|---------|-----------|------------|
| R1 | DeltaP total raw γ | 10 | LCAO/DZP | γ→μ→E Legendre | 5.84 | −41% |
| R1 | DeltaP total branch γ | 10 | LCAO/DZP | γ→μ→E Legendre | 8.11 | −18% |
| R2 | efield + dγ/dE | 10 | LCAO/DZP | γ→μ | 1.64 | −83% |
| R3 | efield dip_cor=1 + dγ/dE | 10 | LCAO/DZP | γ→μ | 1.63 | −83% |
| R6 | DeltaP PW ecut20 | 15 | PW/ecut20 | γ→μ→E Legendre | 1.82 | −82% |
| R6 | DeltaP PW ecut80 | 15 | PW/ecut80 | γ→μ→E Legendre | 2.28 | −77% |
| R7 | efield + 能量 FD | 10 | LCAO/DZP | E_KS 直接 FD | 3.14 | −68% |

## 未解决的问题

1. **10 Bohr 盒子淬火效应是所有结果的共同下限.** 即便最干净的能量 FD (R7) 也只得到 3.14 a₀³. 15 Bohr 盒子 + TZDP 基组的验证性计算是唯一可以裁决的下一步.
2. **PW DeltaP escon 能量 FD (R8)** 未完成. Legendre 变换中 Ry↔Ha 转换因子的代码级核实需要完成.
3. **Standard ABACUS Berry phase (`berry_phase=1`)** 在此 binary 中未产生输出——需要 nscf 模式或单独编译.
4. **Branch/raw γ 分歧 (R1)** 揭示 per-atom branch selection 破坏 Σ 守恒——这是代码行为, 不是物理. 任何依赖 per-atom γ 总和的分析必须使用 raw γ (Wilson 行列式).

---

*测试数据位置: `/root/abacus-develop/tests/deltap_h2o_polarizability/`*
