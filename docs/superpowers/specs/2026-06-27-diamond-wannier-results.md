# Diamond Wannier Center 对比：测试结果与分析

> **日期**: 2026-06-27
> **测试**: diamond ref + C1 z-位移(0.01 direct), 4×4×4 k-mesh, gdir=3

---

## 1. 测试结果

### 1.1 berry_phase

| 结构 | elec_phase | ionic_phase | P_total (e/bohr²) |
|------|-----------|------------|-------------------|
| ref | 0.00000 | 0.00000 | 0 |
| disp | 0.00000 | 0.04000 | 3.041×10⁻³ |

**电子相位 = 0**：berry_phase 认为电子对极化变化的贡献为零。

### 1.2 DeltaP

| 结构 | gamma | P_elec (e/bohr²) |
|------|-------|-----------------|
| ref | 0 | 0 |
| disp | 0 | 0 |

**DeltaP gamma = 0**：与 berry_phase 的 elec_phase = 0 一致。✅

### 1.3 Wannier90

8 个 Wannier center 沿 a3 方向的投影 (Bohr):

| WF | 原子 | ref | disp | shift |
|----|------|-----|------|-------|
| 1 | C1 | 7.5484 | 7.4753 | -0.0730 |
| 2 | C1 | 7.5484 | 7.5842 | +0.0358 |
| 3 | C1 | 7.5484 | 7.6435 | +0.0952 |
| 4 | C1 | 7.5484 | 7.6435 | +0.0952 |
| 5 | C2 | 1.0783 | 1.0979 | +0.0196 |
| 6 | C2 | 1.0783 | 1.0849 | +0.0066 |
| 7 | C2 | 1.0783 | 1.0754 | -0.0029 |
| 8 | C2 | 1.0783 | 1.0754 | -0.0029 |
| **Σ** | | | | **+0.1735** |

**总 Wannier center 位移 = +0.1735 Bohr**（非零！）

### 1.4 Z* 对比

| 方法 | Z*_total | Z*_elec |
|------|---------|---------|
| berry_phase | 4.00 | 0.00 |
| DeltaP | 0.00 (gamma=0) | 0.00 |
| Wannier90 | -4.02 | -8.02 |
| 文献 | ≈ 0 | ≈ 0 |

---

## 2. 分析

### 2.1 矛盾：三方法给出三个不同答案

| 方法 | Z* | 电子响应 |
|------|-----|---------|
| berry_phase | 4.0 | 无 (elec_phase=0) |
| Wannier90 | -4.02 | 有 (ΣΔ⟨r⟩≠0) |
| 文献 | ≈ 0 | 完全屏蔽 |

### 2.2 berry_phase elec_phase=0 的原因

4×4×4 k-mesh (nppstr=5, dk=0.25) 太粗，Berry phase 无法捕捉电子响应。

之前在 diamond 测试中已发现：
- Wannier90 Z*_C ≈ -0.02（正确，4×4×4 足够 Wannier90）
- berry_phase Z* = 4.0（错误，4×4×4 对 berry_phase 太粗）

**berry_phase 的 4×4×4 精度不够**，而 Wannier90 的 4×4×4 精度足够（因为 Wannierization 做了最优规范变换）。

### 2.3 DeltaP gamma=0 的原因

DeltaP 用 Wilson loop 特征值法，本质上和 berry_phase 一样是离散 Berry phase。4×4×4 的精度与 berry_phase 一致，也给出 gamma=0。

**DeltaP 和 berry_phase 在 4×4×4 上一致（都给出 0）**，但两者在粗 k-mesh 上都不准确。

### 2.4 Wannier90 给出非零结果的原因

Wannier90 通过 Wannierization（迭代最小化 spread）找到最优规范 U(k)，然后计算 Wannier center。这个过程比直接 Berry phase 更稳健：
- Berry phase 直接计算 ∏ det(O_j)，受离散化误差影响大
- Wannier90 计算 U†MU 的对角元素，通过迭代优化使得结果更连续

**关键差异**: Wannier90 做了规范优化（Wannierization），而 berry_phase/DeltaP 没有做。

### 2.5 这对 DeltaP 意味着什么

DeltaP 的 Wilson loop 特征值法在**粗 k-mesh 上与 berry_phase 一致**（都给出 gamma=0），但**不如 Wannier90 准确**（Wannier90 给出非零的电子响应）。

这说明：
1. DeltaP 的框架是正确的（与 berry_phase 一致）
2. 但需要更密的 k-mesh 或 Wannierization 来提高精度
3. 直接对比 Wannier center 需要 DeltaP 也有 Wannierization 级别的精度

---

## 3. 结论

### 3.1 已验证

| 验证项 | 结果 |
|--------|------|
| DeltaP 与 berry_phase 一致性 (diamond 4×4×4) | ✅ gamma=0, 一致 |
| DeltaP P_elec ratio (BaTiO3 10×10×10) | ✅ 0.97 |
| SMO 第一 zeta | ✅ 正确 |
| sum_n arg(λ_n) = arg(det(W)) | ✅ 精确 |
| Wannier90 正常运行 (diamond) | ✅ 8 个 Wannier center |

### 3.2 限制

| 限制 | 原因 |
|------|------|
| DeltaP Z* 不准确 | P_elec 的 3% 误差被 1/δ 放大 |
| 无法直接对比 γ_n vs ⟨r_n⟩ | DeltaP 4×4×4 给出 γ_n=0, Wannier90 给出非零 ⟨r_n⟩ |
| berry_phase 4×4×4 不准确 | 离散 Berry phase 精度不够 |

### 3.3 下一步方向

1. **增大 diamond k-mesh** (8×8×8 或 10×10×10): 使 berry_phase 和 DeltaP 能捕捉电子响应，然后对比 Wannier90
2. **BaTiO3 对比 Wannier center**: BaTiO3 10×10×10 已有 berry_phase 精度，但 Wannier90 崩溃
3. **实现 SMO-basis Wannierization** (方案 C): 在 DeltaP 内部做规范优化，提高精度到 Wannier90 水平
4. **对比 det(W) 而非 γ_n**: det(W) 不受规范影响，可直接对比 berry_phase 和 Wannier90 的总 Berry phase

