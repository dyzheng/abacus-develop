# DeltaP 风险评审 Rebuttal

> 针对 `2026-07-12-deltap-risk-points-and-solutions.md` 的逐条评估回复
> 日期: 2026-07-13

---

## 评估总表

| 风险 ID | 文档评级 | 我的评级 | 立场 | 核心理由 |
|---------|---------|---------|------|---------|
| C2 | Critical | **Critical** | ✅ 同意风险，不同意方案 A | 数学正确；应修 ζ rescaling 但用 unwrapped sum (方案 B)，不应删除 |
| C1 | Critical | **High** | ⚠️ 降级 | Stage 2 实测 dγ/dλ≈0.1-0.3，证明 S_dk 工作正常；缺失项为 O(dk²) |
| C3 | Critical | **Medium** | ⚠️ 降级，反对修复方案 | 与电荷 mixing 同质近似；两轮对角化代价大且无保证一致性 |
| H4 | High | **High** | ✅ 同意 | phase_corrections_ 确实疑似未应用，需代码确认 |
| H3 | High | **High** | ✅ 同意 | per-band spacing 在粗 mesh 影响精度 |
| H1 | High | **Medium** | ❓ 需验证 | 需确认 max_asym 是否为 0 |
| H2 | High | **Medium** | ❓ 需验证 | Bloch 约定问题，需确认符号 |
| H5 | High | **Low** | ❌ 不同意 | fmod 是正确的；匹配歧义应交给匈牙利算法 (L1) |
| M1-M6 | Medium | **Medium** | ✅ 同意 | 评级准确 |
| L1-L4 | Low | **Low** | ✅ 同意 | 评级准确 |

---

## 详细 Rebuttal

### C1: S_dk 不一致 → 不同意 "Critical"，降为 High

**文档主张**: compute_S_dk 缺少位置修正项 `-i·dk·⟨r⟩`，导致 HK 操作在与 Wilson loop 不同的流形上。

**实测证据**: Stage 2 验证（2026-07-12）已确认 `dγ/dλ ≈ 0.1-0.3` per unit λ. 若 S_dk 从根本上错误，HK 修正不可能产生有物理意义的 γ 响应. 位置修正为 O(dk²) 量级精化项，非一阶错误。

**结论**: 使用 `compute_S_dk_link` 是正确方向，但这是精度提升 (High)，非正确性阻滞 (Critical).

### C2: Zeta rescaling → 同意数学分析，不同意方案 A

**文档主张**: `arg(det W) / Σγ_unwrapped ≠ 1` 时 scale 错误缩放 per-atom γ.

**同意**: 数学推导正确。unwrap 后的 γ_n 可附加 2πk，导致 scale 偏离 1.

**不同意**: 方案 A（删除 rescaling）过于激进。rescaling 的合法用途是修正 SMO 基组不完备性 (Σ_I w^I_n ≠ 1). 正确方案是 **方案 B**: 用 `Σγ_unwrapped / Σγ_raw` 代替 `arg(det W) / Σγ_raw`.

### C3: psi-lambda 不一致 → 不同意 "Critical"，反对修复方案

**文档主张**: mixing 后 λ_actual ≠ λ_trial，但 ψ 是 H(λ_trial) 的本征态，导致 residual 不连续。

**Rebuttal**:
1. 此问题与外层 SCF 的电荷 mixing 同质：`ρ_actual = ρ_old + β*(ρ_new - ρ_old)`，下轮使用 ψ(ρ_actual) 而非 ψ(ρ_new)，无人称此为 Critical bug
2. γ 对 ψ 的变化鲁棒：Stage 2 实测 Δλ=0.1 仅改变 γ 约 0.01 rad
3. 文档提议的修复（内循环每步重对角化）代价高且不保证一致性，因电荷密度未更新

**结论**: 降为 Medium，不实施重对角化修复。

### H5: fmod 丢失累积相位 → 不同意

**文档主张**: `fmod(γ_unwrapped, 2π)` 丢失累积相位信息，可能导致匹配错误。

**Rebuttal**: `arg(eval)` 始终在 `(-π, π]` 范围内。与 unwrapped γ 比较前**必须在某一侧做 fmod**，否则无法进行模 2π 匹配。文档提议的"直接比较 unwrapped γ"不可行。匹配歧义是局部贪心算法的固有缺陷，应由匈牙利算法 (L1) 解决，非 fmod 引入的 bug.

**结论**: 降为 Low.

### H1, H2, M5: 需验证

| ID | 验证问题 | 验证方法 |
|----|---------|---------|
| H1 | SMO 重叠矩阵是否真的非对称？ | 检查输出中 max_asym 是否非零 |
| H2 | conj(S_k) 是否正确？ | grep ABACUS Bloch 相位约定 |
| M5 | 诊断检验公式是否真的 S·S^{-1/2}？ | 读取实际代码 |

---

## 实际实验 vs 文档评级的不一致

| 现象 | 文档可解释？ | 说明 |
|------|------------|------|
| λ=0 SCF 正常收敛 (15 轮) | ✅ | C1-C3 均非阻滞因子 |
| λ=0.05 SCF gamma 振荡 | ⚠️ | 文档将此归因于 C1/C3，但 HK 缓存 (λ 不变不重算) 已消除 ψ-HK 自洽——gamma 仍振荡 |
| 冻结电荷 3 次运行 gamma 不同 | ❌ | 文档 16 项风险点中无任何一项解释此现象 |
| HR 延迟后 λ=0.05 两次运行 gamma 仍不同 | ❌ | 同上 |

**结论**: 文档理论分析的价值在于提出改进方向 (C2, H3, H4)，但文档判定的 3 个 Critical 风险**均非当前 SCF 不收敛的根因**. 实际根因是 Wilson loop 本征值计算内部的非确定性 (B16)，尚待定位.

---

## 建议优先级重排

| 优先级 | 项目 | 理由 |
|--------|------|------|
| **P0** | 定位本征值非确定性根因 | 当前唯一阻滞 Stage 3 的问题 |
| **P1** | C2 修复（方案 B） | 唯一真正的 Critical bug |
| **P1** | H4 修复 | Gauge 不连续导致错误 γ |
| **P2** | H1/H2 验证后修复 | 可能影响精度 |
| **P2** | H3 修复 | 粗 mesh 精度 |
| **P3** | C1 改进（使用 compute_S_dk_link） | O(dk²) 精度提升 |
| **P3** | M1-M6 维护性修复 | 非功能性问题 |
