# Diamond 8×8×8 测试结果与分析

> **日期**: 2026-06-27
> **体系**: Diamond, 8×8×8 k-mesh, nocc=4, gdir=3, symmetry=-1

---

## 1. 测试结果

### 1.1 berry_phase

| 结构 | elec_phase | ionic_phase | P_elec (e/bohr²) |
|------|-----------|------------|-----------------|
| ref | 1.00000 | 0.00000 | 7.601×10⁻² |
| disp | 0.96000 | 0.04000 | 7.297×10⁻² |
| Δ | -0.04000 | +0.04000 | -3.041×10⁻³ |

**电子相位从 1.0 变到 0.96**：8×8×8 足够捕捉电子响应（4×4×4 给出 0）。
总相位不变 (1.0→1.0)，Z* = 0 ✓（diamond 文献值）。

### 1.2 DeltaP

| 结构 | P_elec (e/bohr²) |
|------|-----------------|
| ref | 2.260×10⁻² |
| disp | -2.653×10⁻² |
| Δ | -4.913×10⁻² |

**ΔP/ΔP_berry = 16.2**：DeltaP 差分严重偏离 berry_phase。

### 1.3 Wannier90

4 个 Wannier center 沿 a3 方向 (Bohr):

| WF | ref | disp | shift |
|----|-----|------|-------|
| 1 | 7.5484 | 7.7987 | +0.2503 |
| 2 | 7.5483 | 7.5716 | +0.0233 |
| 3 | 7.5484 | 7.4549 | -0.0934 |
| 4 | 7.5484 | 7.4550 | -0.0933 |
| **Σ** | | | **+0.0869** |

### 1.4 极化中心偏移对比

| 方法 | Δr (Bohr) | vs berry |
|------|-----------|---------|
| berry_phase | +0.1725 | 1.000 |
| DeltaP | +2.7878 | 16.16 ❌ |
| Wannier90 | +0.0869 | 0.504 ❌ |

---

## 2. 分析

### 2.1 Wannier90 与 berry_phase 不一致

Wannier90 Δr = 0.0869，berry_phase Δr = 0.1725，比例 0.504。

**原因**: Wannier90 用 4 个 Wannier 函数 (num_wann=4)，但原始计算用 12 个能带 (nbands=12)。在 .mmn/.amn 截断到 4 个能带时，丢失了能带间的耦合信息。

之前 4×4×4 用 8 个 Wannier 函数 (num_wann=8, 含 disentanglement) 给出 Z* ≈ -0.02 (正确)。现在 8×8×8 用 4 个 Wannier 函数 (无 disentanglement) 给出错误结果。

**结论**: Wannier90 需要用 disentanglement (8 个 Wannier 函数) 才能正确处理 diamond。

### 2.2 DeltaP 与 berry_phase 不一致

DeltaP Δr = 2.79，berry_phase Δr = 0.17，比例 16.2。

**原因**: 
1. ref 结构: berry_phase elec_phase=1.0 (非零!), DeltaP P=2.26e-2
   - berry P_elec = 7.60e-2, DeltaP P = 2.26e-2, ratio = 0.30
   - **DeltaP 对 ref 结构的 P 偏差很大**

2. disp 结构: berry_phase elec_phase=0.96, DeltaP P=-2.65e-2
   - berry P_elec = 7.30e-2, DeltaP P = -2.65e-2, ratio = -0.36
   - **符号都反了**

**根本问题**: DeltaP 在 8×8×8 上对 diamond 的 P_elec 计算不正确。
- 4×4×4: gamma=0 (一致)
- 8×8×8: gamma≠0 但与 berry_phase 不一致

可能原因:
- 8×8×8 的 elec_phase=1.0 (整数)，意味着 Berry phase = 2π。这可能是分支跳变！
- arg(det(W)) 在 2π 附近不稳定

### 2.3 berry_phase elec_phase=1.0 的含义

elec_phase=1.0 (reduced) 意味着 γ_elec = 1.0 × 2π = 2π = 0 (mod 2π)。

但 berry_phase 报告的是 "unwrapped" 值 1.0，不是 0。这意味着实际 Berry phase 恰好在 2π 分支切割上。

DeltaP 用 arg(det(W)) ∈ (-π, π]，会给出 0 而不是 2π。这就是不一致的来源！

**berry_phase 用 "除以平均" 方法 unwrap 到 1.0，DeltaP 用 arg() 给出 0 或接近 0。**

### 2.4 验证

DeltaP ref: gamma (string 0) = -π = -3.14159。这说明 det(W) 在负实轴上，arg = ±π。

**这正是 2π 分支跳变的情形！** berry_phase unwrap 到 1.0 (= 2π/(2π))，DeltaP arg 给出 ±π。

---

## 3. 结论

### 3.1 8×8×8 上的不一致来自 2π 分支跳变

- berry_phase elec_phase = 1.0 (= γ/(2π) = 2π/(2π))，unwrap 到 1.0
- DeltaP gamma = ±π (arg(det(W)) 在分支切割上)
- Wannier90 (4 WF, 无 disentanglement) 不正确

### 3.2 4×4×4 上的一致性是"巧合"

- 4×4×4: berry_phase elec_phase=0, DeltaP gamma=0, 一致
- 但两者都是 0，可能只是精度不够（都给出 0），不是真正一致

### 3.3 核心困难确认

Berry phase 的 2π 分支歧义是 DeltaP 的核心困难：
- arg(det(W)) ∈ (-π, π]
- 当 det(W) 接近负实轴时，arg 在 ±π 之间跳变
- berry_phase 用 "除以平均" unwrap 处理这个问题
- DeltaP 的简单 arg 平均无法处理

### 3.4 下一步

1. **在 BaTiO3 上验证**: BaTiO3 的 elec_phase = -0.331 (不在分支切割上), DeltaP ratio=0.97 ✓
2. **避免分支切割**: 选择体系/方向使 Berry phase 远离 2π 倍数
3. **实现正确的 unwrap**: 在 det(W) 级别用 berry_phase 方法 unwrap 总量，不做逐原子缩放

