# DeltaP 逐原子极化分解：实现尝试与验证报告

> **日期**: 2026-06-25
> **目标**: 基于 `Wilson_Loop_Per_Atom_Decomposition.md` 和 `DeltaSpin_vs_DeltaP_Essential_Difficulties.md` 的方案，实现逐原子 Wilson loop 极化分解，验证连续性和 Born 有效电荷正确性

---

## 1. 已完成的开发工作

### 1.1 修复 M^I 共轭 bug

**问题**: 原始 `deltap_wannier.cpp` 中 M^I 矩阵的共轭位置错误。

代码约定:
- D = ⟨α|ψ⟩ (nproj × nocc), 行 = SMO 通道
- SVD 极分解: U_D = W·V† (nproj × nocc), 行按原子组织
- 文档约定: A = ⟨ψ|α⟩ = D†, U_A = U_D†

文档公式 M^I = U^{I†}_A · O · U^I_A 转换到 D 约定为:
```
M^I = U^I_D · O · U^{I†}_D   (左无共轭, 右共轭)
```

**原始代码** (错误):
```cpp
const auto uj = std::conj(U_k[j][idx_j]);        // 左共轭 ← 错误
sum += uj * Oj[n+m*n_dim] * U_k[j+1][idx_jp1];   // 右无共轭 ← 错误
```

**修复后**:
```cpp
const auto uj = U_k[j][idx_j];                     // 左无共轭 ← 正确
sum += uj * Oj[n+m*n_dim] * std::conj(U_k[j+1][idx_jp1]); // 右共轭 ← 正确
```

### 1.2 分支持久化 (cross-SCF 平滑性)

新增 `load_branch()` / `save_branch()` 方法:
- `load_branch()`: 在 `compute_wannier_polarization` 开头读取 `deltap_branch.dat`
- `save_branch()`: 在结尾写入每个原子的 W^I 复数值
- 文件格式: 第一行 = 原子数, 后续每行 = (Re, Im)
- 用途: 跨独立 SCF 运行保持 arg(W^I) 分支一致, 消除 2π 跳变

### 1.3 发现并诊断 nproj > nocc 导致 det(M^I) = 0

**关键发现**: BaTiO3 测试体系中:
- 所有原子的 nproj = (nwl+1)² = 16 (轨道文件含 l=0..3)
- nocc = 15 (O PP 含 8 价电子, 非 6)
- k_I = min(16, 15) = 15 = nocc

当 k_I = nocc 时, Vt 是 (nocc × nocc) 酉矩阵, det(Vt·O·Vt†) = det(O) → **所有原子的 Wilson loop 退化为总 Wilson loop**, 失去原子分辨性。

**实验验证**: 修复共轭 bug 后运行, 所有 5 个原子的 Pz 完全相同 (-2.675e-3), branch 文件中 W^I 全部 ≈ 0。

### 1.4 尝试方案 A: 逐原子 SVD + 奇异值截断

**思路**: 对每个原子单独做 SVD, 截断小奇异值使 k_eff < nocc。

**实现**:
- 对每个原子 I: SVD(D^I) = W^I Σ^I Vt^I
- 截断: 保留 σ > rel_thr · σ_max 的通道, k_eff < nocc
- M^I = Vt^I(k_j) · O_j · Vt^{I†}(k_{j+1}) (k_eff × k_eff)

**测试** (rel_thr = 0.1):
- Ba: k_eff=8, Ti: k_eff=8, O: k_eff=4~6
- Pz 值终于不同了! 但总 Z* 完全错误:

| | berry_phase | DeltaP (截断) |
|---|---|---|
| Z*_Ti | 6.69 | -14.5 |
| Z*_Ba | 2.67 | 12.5 |

**失败原因**: 不同原子的 V^I 独立计算, 不满足互补性 (Σ_I V^I·V^{I†} ≠ I), sum rule 严重破坏。

### 1.5 尝试方案 B: 全局 SVD + 逐原子权重 (Berry connection)

**思路**: 用全局 V (共享, 保证 sum rule), 通过权重 w^I_s = Σ_{a∈I} |W_{a,s}|² 分解。

**数学**:
- 全局 SVD: D = W Σ V†, V 是 (nocc × nocc) 酉
- 权重: w^I_s = Σ_{a∈I} |W_{a,s}|², 满足 Σ_I w^I_s = 1 (精确)
- Berry connection: A^I = Σ_j Σ_s w^I_s · Im[M_j(s,s)]
- Sum rule: Σ_I A^I = Im Tr(O_j) (精确)

**验证**: 权重和验证通过 (3.35+4.64+2.34+2.50+2.16 ≈ 15 = nocc)。

**但 Z* 仍然错误**:

| | berry_phase | DeltaP (trace) |
|---|---|---|
| Z*_Ti | 6.69 | 47.7 |
| Z*_Ba | 2.67 | 13.9 |

**失败原因**: Berry connection (Im Tr) ≠ Berry phase (Im ln det), 对于 nocc=15 和 nppstr=11, 差异巨大。

### 1.6 尝试 SVD 规范对齐 (Procrustes matching)

**问题**: 各 k 点 SVD 独立, 奇异向量符号不一致, 导致 Im[M_j(s,s)] 符号错误。

**实现**: 在每对相邻 k 点间, 求 Q = argmax Re Tr[V†(k_{j-1})·V(k_j)·Q], 对齐 V(k_j)。

**效果**: 修正了符号问题 (Z* 从负变正), 但量级仍然偏差 ~7×。

### 1.7 尝试方案 C: 混合方法 (Wilson 总量 + trace 比例)

**思路**: 用 Wilson loop 精确计算总 Berry phase, 用 trace 比例分解到原子:
```
γ^I = γ_total_Wilson × (A^I_trace / A_total_trace)
```

**结果**: 完全失败, 因为 Wilson/trace 比例在不同结构间剧烈变化:

| 结构 | γ_Wilson | γ_trace | 比例 |
|---|---|---|---|
| ref | -5.323 | 1.115 | -4.77 |
| ti_p | -5.454 | -1.883 | 2.90 |
| ba_p | 1.030 | 0.240 | 4.29 |

比例变号 → rescaling 无效。

### 1.8 发现单 k-string 问题

**根本原因定位**: 代码只处理第一个 k-string (k_index_[0][j]), 而 ABACUS berry_phase 对所有 100 个 k-string (10×10) 平均。

单 string 的 Wilson loop γ = -5.323, 而 ABACUS 电子 Berry phase = 2π × (-0.331) = -2.078, 比例 ≈ 2.56, 与 100 个 string 的平均不一致。

**这是所有方案失败的最终根因**: 总 Berry phase 本身就不对, 任何分解方法都无法给出正确的 Z*。

---

## 2. 测试验证结果

### 2.1 Berry phase 基准 (可靠)

| 位移 | P_berry (e/bohr²) | Z*_berry | 文献值 | 误差 |
|---|---|---|---|---|
| ref | 5.102e-4 | — | — | — |
| Ti+0.01 | 1.682e-3 | 6.69 | 7.18 | 7% |
| Ba+0.01 | 9.781e-4 | 2.67 | 2.74 | 3% |

**结论**: berry_phase Z* 是可靠基准。

### 2.2 DeltaP 各方案 Z* 汇总

| 方案 | Z*_Ti | Z*_Ba | 问题 |
|---|---|---|---|
| berry_phase (基准) | 6.69 | 2.67 | — |
| 原始 (共轭 bug) | — | — | det=0, W^I≈0 |
| A: 逐原子SVD+截断 | -14.5 | 12.5 | sum rule 破坏 |
| B: 全局SVD+权重(trace) | 47.7 | 13.9 | trace≠phase |
| C: 混合(Wilson+trace比例) | -2.09 | 101.1 | 比例随结构变号 |

### 2.3 连续性验证

分支持久化机制已实现 (`deltap_branch.dat` 读写), 但由于 Z* 不正确, 连续性验证的意义有限——即使 P^I 平滑, 数值也是错的。

---

## 3. 关键结论

### 3.1 逐原子 Wilson loop 分解的根本困难

通过 4 轮实现尝试, 确认了 `DeltaSpin_vs_DeltaP_Essential_Difficulties.md` 中分析的五个层级困难:

1. **nproj > nocc 退化** (新发现): 当 SMO 通道数 > 占据能带数时, 极分解因子 Vt 变为 (nocc×nocc) 酉矩阵, det(Vt·O·Vt†) = det(O), 逐原子 Wilson loop 退化为总 Wilson loop。BaTiO3 中 nproj=16 > nocc=15, **所有原子**都退化。

2. **sum rule 不可满足**: 逐原子 SVD 给出的 V^I 不互补 (Σ V^I·V^{I†} ≠ I), sum rule 严重破坏。全局 SVD 权重法 sum rule 精确, 但分解的是 Berry connection (trace) 而非 Berry phase (det)。

3. **Berry connection ≠ Berry phase**: 对于 nocc=15 和 nppstr=11 (dk=0.1), Im Tr(O) 与 Im ln det(O) 差异可达 2-7 倍, 且比例随结构变化, 无法通过 rescaling 修正。

4. **单 k-string 不充分**: 代码只处理 1/100 个 k-string, 总 Berry phase 本身就不对。

### 3.2 Wilson_Loop_Per_Atom_Decomposition.md 方案的适用条件

文档中的逐原子 Wilson loop 方案 (M^I = U^{I†}·O·U^I) 要求:
- **nproj_per_atom < nocc** (否则 det=0 或退化为总量)
- **SMO 子空间互补** (否则 sum rule 不成立)
- **dk → 0** (否则 Berry connection ≠ Berry phase)

BaTiO3 体系不满足第一和第三个条件。

### 3.3 可行的后续方向

1. **多 k-string 平均**: 修复单 string 问题, 使总 Berry phase 与 ABACUS berry_phase 一致。这是任何分解方法正确的前提。

2. **减少 SMO 通道**: 只使用价态 l 通道 (Ba: s, Ti: s+d, O: s+p), 使 nproj < nocc。需要修改 SMO 构造逻辑。

3. **精确逐原子 Wilson loop**: 在 nproj < nocc 且多 string 平均后, 重新测试 det(M^I) 方案。

4. **Berry connection 多 string 平均**: 即使 trace≠phase, 多 string 平均后的 Z* 比例可能更稳定, 可作为近似分解。

---

## 4. 代码修改清单

### 4.1 已提交的修改

| 文件 | 修改 | 状态 |
|---|---|---|
| `deltap_wannier.cpp` | 修复 M^I 共轭 (左无共轭, 右共轭) | ✅ |
| `deltap_wannier.cpp` | 新增 load_branch()/save_branch() | ✅ |
| `deltap_wannier.cpp` | 逐原子 SVD + 截断 (方案A) | ✅ (已验证不可用) |
| `deltap_wannier.cpp` | 全局 SVD + 权重 + Procrustes (方案B) | ✅ (已验证不可用) |
| `deltap_wannier.cpp` | 混合 Wilson+trace (方案C) | ✅ (已验证不可用) |
| `deltap.h` | 新增 load_branch()/save_branch() 声明 | ✅ |

### 4.2 测试文件

| 路径 | 内容 |
|---|---|
| `/tmp/opencode/bto_zstar/ref/` | BaTiO3 参考 SCF+NSCF |
| `/tmp/opencode/bto_zstar/ti_p/` | Ti+0.01 位移 SCF+NSCF |
| `/tmp/opencode/bto_zstar/ba_p/` | Ba+0.01 位移 SCF+NSCF |

---

## 5. 数值数据存档

### 5.1 奇异值 (ref 结构, k=0)

| 原子 | r | k_I | nocc | 前5个奇异值 |
|---|---|---|---|---|
| Ba | 16 | 15 | 15 | 1.47, 1.29, 1.07, 0.83, 0.76 |
| Ti | 16 | 15 | 15 | 2.54, 1.54, 1.30, 1.19, 1.00 |
| O(1) | 16 | 15 | 15 | 2.00, 1.24, 1.16, 0.29, 0.19 |
| O(2) | 16 | 15 | 15 | 1.94, 0.97, 0.48, 0.34, 0.26 |
| O(3) | 16 | 15 | 15 | 2.07, 1.09, 0.98, 0.34, 0.24 |

### 5.2 全局 SVD 权重和 (ref, k=0)

| 原子 | Σ_s w^I_s |
|---|---|
| Ba | 3.35 |
| Ti | 4.64 |
| O(1) | 2.34 |
| O(2) | 2.50 |
| O(3) | 2.16 |
| **总和** | **14.99 ≈ nocc** |

### 5.3 Wilson/trace 比例 (混合方案)

| 结构 | γ_Wilson | γ_trace | 比例 |
|---|---|---|---|
| ref | -5.323 | 1.115 | -4.77 |
| Ti+0.01 | -5.454 | -1.883 | 2.90 |
| Ba+0.01 | 1.030 | 0.240 | 4.29 |

比例变号 → 混合方案无效。
