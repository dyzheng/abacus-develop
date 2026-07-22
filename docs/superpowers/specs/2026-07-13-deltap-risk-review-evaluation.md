# DeltaP 风险评审 Rebuttal 评估

> 针对 `2026-07-13-deltap-risk-review-rebuttal.md` 的逐点回应  
> 日期: 2026-07-13

---

## 评估总表

| 风险 ID | 我的原评级 | Rebuttal 评级 | 最终共识 | 立场 |
|---------|-----------|--------------|---------|------|
| C2 | Critical | Critical | **Critical** | ✅ 完全一致，采用方案 B |
| C1 | Critical | High | **High** | ✅ 接受降级 |
| C3 | Critical | Medium | **Medium** | ✅ 接受降级 |
| H4 | High | High | **High** | ✅ 完全一致 |
| H3 | High | High | **High** | ✅ 完全一致 |
| H5 | High | Low | **Low** | ✅ 接受降级 |
| H1 | High | Medium(待验证) | **待验证** | ✅ 同意需先确认 max_asym |
| H2 | High | Medium(待验证) | **待验证** | ✅ 同意需先确认 Bloch 约定 |
| M1-M6 | Medium | Medium | **Medium** | ✅ 完全一致 |
| L1-L4 | Low | Low | **Low** | ✅ 完全一致 |
| B16 | 未评级 | **P0 阻滞** | **P0** | ✅ 同意是实际根因 |

---

## 达成共识（6 项）

### C2: Zeta rescaling 数学错误 → 采用方案 B

双方一致确认：
- 数学分析正确：`arg(det W)` 是主值，`Σγ_unwrapped` 可附加 2πk，scale 错误
- 方案 A（删除 rescaling）过于激进，SMO 不完备时 rescaling 有合法用途
- **方案 B 正确**：用 `Σγ_unwrapped / Σγ_raw` 替代 `arg(det W) / Σγ_raw`

### H3: 分支选择间距 → per-band spacing 正确

双方一致确认内联代码使用 `2π·scale` 而非 `2π·w^I_n·scale` 是错误，粗 mesh 影响精度。应启用 `select_branch_set`。

### H4: Gauge phase_corrections_ 未生效 → 需修复

双方一致确认 `phase_corrections_` 计算后从未应用，anchor 切换处 gauge 不连续。

### M1-M6, L1-L4 → 评级准确

双方对所有 Medium 和 Low 级风险点评级一致。

### B16 是实际阻滞因素

双方一致同意 Wilson loop 本征值非确定性是当前 SCF 不收敛的真正根因，优先于所有理论风险点。

---

## 接受降级（3 项）

### C1: Critical → High

**Rebuttal 论据**: Stage 2 实测 `dγ/dλ ≈ 0.1-0.3`，证明 HK 修正产生有物理意义的 γ 响应。若 S_dk 从根本上错误，不可能观测到系统性响应。

**接受理由**: 实测证据优先于纯理论分析。S_dk 不一致引入的是定量误差（位置修正量级 O(dk)），在 3×3×3 mesh 上 dk ≈ 0.33，误差 ~0.03 rad — 可感知但不阻滞功能。`compute_S_dk_link` 是正确方向，但作为精度提升（High）而非正确性修复（Critical）。

### C3: Critical → Medium

**Rebuttal 论据**:
1. 与外层 SCF 电荷 mixing 同质：`ρ_actual = ρ_old + β(ρ_new - ρ_old)`，下轮用 `ψ(ρ_actual)` 而非 `ψ(ρ_new)`，无人称此为 Critical
2. γ 对 ψ 变化鲁棒：实测 Δλ=0.1 仅改变 γ 约 0.01 rad
3. 修复方案（每步重对角化）代价高且因电荷密度未更新而无一致性保证

**接受理由**: 电荷 mixing 类比成立。标准 SCF 中 ψ(ρ_mixed) 同样不自洽，通过迭代趋于自洽。内循环虽然无法"迭代消除"不一致，但实测影响小。重对角化修复的收益/代价比不佳。

### H5: High → Low

**Rebuttal 论据**: `arg(eval) ∈ (-π, π]`，与 unwrapped γ 比较前**必须在某侧做 fmod**，否则无法进行模 2π 匹配。文档将 fmod 视为 bug 是对匹配算法的误解。匹配歧义是贪心算法的固有缺陷，应由匈牙利算法（L1）解决。

**接受理由**: 正确。fmod 是合理的 2π 约化方式。"丢失累积相位"的担忧不成立——匹配只需在单个 2π 窗口内识别特征向量归属，累积相位已在 `gamma_unwrapped` 中正确计算并用于 per-atom γ。

---

## 保留分歧（1 项）

### C1: 位置修正的阶数

| 方面 | 我的分析 | Rebuttal |
|------|---------|---------|
| 位置修正阶数 | O(dk)，提供 overlap 的领先虚部 | O(dk²) 精化项 |
| 对 Berry phase 的影响 | 一阶贡献 | 高阶修正 |
| 实际结论 | High（精度提升） | High（精度提升） |

**分歧分析**:

位置修正项 `-i·dk·⟨r⟩` 对 overlap 的贡献确实是 O(dk)。更关键的是，Bloch 重叠 `⟨ψ_k|ψ_{k+dk}⟩` 的虚部为 O(dk²)（因为实部 1 - O(dk²) 主导），而位置修正直接贡献 O(dk) 的虚部。Berry phase 正是从 overlap 的虚部提取的。因此从数学上看，位置修正是 Berry phase 的一阶贡献，不是高阶精化。

**但在实际影响层面，结论一致**: 对于密集 k-mesh（dk → 0），此项趋于 0。对于当前的 3×3×3 mesh，误差量级 ~0.03 rad，可感知但不阻滞。双方一致同意降为 High。

---

## Rebuttal 揭示的盲区

### 冻结电荷下 gamma 跨运行不同

Rebuttal 提供的关键实验证据：

| 现象 | 我的 18 项风险能否解释 |
|------|----------------------|
| λ=0 SCF 正常收敛 | ✅ C1-C3 非阻滞因子 |
| λ=0.05 gamma 振荡 | ⚠️ HK 缓存已消除 ψ-HK 自洽，仍振荡 |
| **冻结电荷 3 次运行 gamma 不同** | **❌ 无一能解释** |
| HR 延迟后两次运行 gamma 仍不同 | **❌ 无一能解释** |

冻结电荷意味着 ψ 完全相同。若 Wilson loop 计算是确定性的，gamma 必须相同。跨运行不同意味着计算内部存在**非确定性源**。

### 非确定性可能来源（my analysis 未覆盖）

1. **zgeev 特征值排序不保证**: LAPACK `zgeev` 不保证特征值返回顺序。若两次运行返回不同顺序，贪心匹配从不同初始条件出发，可能产生不同匹配结果 → 不同 gamma_unwrapped → 不同 per-atom γ。

2. **编译器浮点重排序**: `-O3` / `-ffast-math` 允许结合律变换。Wilson loop 的矩阵累积 `W = O_0·O_1·...·O_{N-1}` 涉及大量浮点乘加，不同结合顺序 → 不同舍入误差 → 特征值微差 → 匹配分支不同。

3. **Newton-Schulz 浮点非结合性**: 虽然算法本身是确定性的（无 LAPACK），但 20 次迭代的矩阵乘法在浮点运算中不完全结合，可能产生 1e-15 量级的差异。若此差异恰好使两个特征值的相位差跨越匹配阈值，会导致不同的匹配结果。

4. **MPI 归约顺序**: 若使用 MPI，`MPI_Allreduce` 的归约顺序依赖进程调度，不同运行可能产生不同的浮点舍入。

### 诊断建议

在 `compute_wannier_polarization` 末尾添加确定性诊断：

```cpp
// 输出 Wilson loop 矩阵的 Frobenius 范数
double frob_W = 0;
for (int i = 0; i < n_dim*n_dim; ++i) frob_W += std::norm(W_mat[i]);
std::cout << "  [DIAG] ||W||_F = " << std::setprecision(17) << std::sqrt(frob_W) << std::endl;

// 输出逐带 gamma_unwrapped
for (int n = 0; n < n_dim; ++n)
    std::cout << "  [DIAG] gamma_unwrapped[" << n << "] = " << gamma_unwrapped[n] << std::endl;

// 输出特征值
for (int n = 0; n < n_dim; ++n)
    std::cout << "  [DIAG] eval[" << n << "] = " << evals[n] << std::endl;
```

比较多次运行的这些输出，可定位非确定性出现在 Wilson loop 矩阵构建阶段还是特征值对角化/匹配阶段。

---

## 对建议优先级重排的评估

| Rebuttal 优先级 | 我的评估 | 备注 |
|----------------|---------|------|
| **P0**: B16 本征值非确定性 | ✅ **同意** | 唯一阻滞 Stage 3 |
| **P1**: C2 修复（方案 B） | ✅ **同意** | 唯一纯数学 Critical bug |
| **P1**: H4 修复 | ✅ **同意** | gauge 不连续直接影响 γ |
| **P2**: H1/H2 验证后修复 | ✅ **同意** | 先确认再修 |
| **P2**: H3 修复 | ✅ **同意** | 粗 mesh 精度 |
| **P3**: C1 改进 | ✅ **同意** | O(dk) 精度提升 |
| **P3**: M1-M6 维护性修复 | ✅ **同意** | 非功能性 |

---

## 最终共识总结

1. **1 个真正的 Critical 数学 bug**: C2（zeta rescaling 用错参照值），采用方案 B
2. **2 个 High 级修复**: H4（gauge phase_corrections_）、H3（分支选择间距）
3. **2 个待验证项**: H1（SMO 对称性）、H2（D_I 共轭）
4. **1 个未解释的非确定性现象**: B16 是实际阻滞因素，需运行时诊断定位根因
5. **原 3 个 Critical 中 2 个合理降级**: C1 → High（实测有效），C3 → Medium（类比电荷 mixing）
6. **我的静态分析遗漏了运行时非确定性**: 需要补充实验诊断，不能仅靠代码审查
