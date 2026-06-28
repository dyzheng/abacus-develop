# DeltaP Wilson loop 特征值法：进展总结与下一步规划

> **日期**: 2026-06-27
> **分支**: `feat/deltap-wilson-per-atom`
> **提交数**: 20+ commits

---

## 1. 已完成的工作

### 1.1 算法实现

**Wilson loop 特征值分解法**（方案 A）已完整实现：

```
输入: ψ(k), S(dk), D_I(k), kv
  ↓
[环节 A] O_j = C†(k_j) · S_dk · C(k_{j+1})
  - compute_S_dk_link: berry_phase 相位 + 位置算子修正
  - 相位: 2π(kvec_c·R_cart - dk_c·τ)
  - 位置: ⟨φ|φ(R)⟩ - i·dk·tpiba·⟨φ|r_local|φ(R)⟩
  - 缓存优化: snap + get_psi_r_psi 结果复用
  ↓
[环节 B] W = ∏ O_j (归一化) + det(W) = ∏ det(O_j)
  - 矩阵乘法 + zgetrf
  ↓
[环节 C] W = V · diag(λ_n) · V⁻¹
  - zgeev
  - γ_n = arg(λ_n), sum_n γ_n = arg(det(W)) ✓
  ↓
[环节 D] w^I_n = Σ_{a∈I} |⟨v_n|α_a⟩|², γ^I = Σ_n w^I_n · arg(λ_n)
  - SMO 第一 zeta (与 DeltaSpin 一致) ✓
  ↓
[环节 E] P^I = -0.5 × (a/2πΩ) × γ^I_avg
  - 自旋因子 -0.5 (nspin=1)
  - 多 k-string 平均 (100 strings)
```

### 1.2 Bug 修复

| Bug | 影响 | 修复 |
|-----|------|------|
| M^I 共轭位置错误 | det=0, W^I≈0 | 左无共轭, 右共轭 |
| onsite_radius=0 | segfault (无 overlap_orb_onsite) | 动态构建 |
| nmp=[0,0,0] (symmetry=-1) | 空 k-string, segfault | 从 k 点坐标推断 |
| kstring_data_ 跨 string 累积 | 数值爆炸 (10^114) | 每 string 清空 |
| 矩阵乘法溢出 | W 元素爆炸 | 每步归一化 |
| Bloch 态 vs 周期部分重叠 | P ratio ~5× | 加相位 + 位置算子修正 |
| 位置修正用了 full r_psi | P ratio ~2× | 减去 R1×overlap 得 local |
| 自旋因子缺失 | P ratio ~2× | 乘 -0.5 |
| berryphase_overlap gathering bug | P ratio ~3× | 回退到 compute_S_dk_link |

### 1.3 验证结果

| 验证项 | 结果 | 状态 |
|--------|------|------|
| sum_n arg(λ_n) = arg(det(W)) | 精确成立 | ✅ |
| SMO 第一 zeta | 与 DeltaSpin 一致 | ✅ |
| Diamond 平衡 P=0 | P≈0 | ✅ |
| BaTiO3 P_elec ratio | **0.966** (3% 误差) | ✅ |
| Z*_Ti (大位移 0.01) | 3.28 vs 2.69 (22%) | ⚠️ |
| Z*_Ba (大位移 0.01) | 分支跳变 | ❌ |
| Z* (小位移 0.001) | P 误差被 1/δ 放大 | ❌ |
| Wannier90 diamond 4×4×4 | 都给出 P=0 (巧合) | ⚠️ |
| Wannier90 diamond 8×8×8 | 2π 分支切割 | ❌ |

---

## 2. 当前状态

### 2.1 已验证正确

**P_elec 与 berry_phase 一致到 3%**:
- berry_phase: P_elec = -5.790×10⁻³ e/bohr²
- DeltaP: P_elec = -5.592×10⁻³ e/bohr²
- ratio = 0.966

**3% 误差来源**: 少数 k-string 的 arg(zeta) 接近 ±π 分支切割，简单平均未做 unwrap。berry_phase 用"除以平均"方法处理此问题。

### 2.2 已知未解决问题

| 问题 | 影响 | 难度 |
|------|------|------|
| **A. 3% P_elec 误差** | Z* 差分被 1/δ 放大 | 中 |
| **B. Z* 分支跳变** | Ba 位移导致 arg 跳 2π | 高 |
| **C. berryphase_overlap gathering bug** | 无法用 berry_phase 的精确 O_j | 中 |
| **D. 位置修正近似** | local r_psi 可能与 unkOverlap_lcao 不完全一致 | 低 |
| **E. SMO 不完备** | sum_I w^I_n ≠ 1 (影响逐原子) | 低 |

### 2.3 问题分析

#### 问题 A: 3% P_elec 误差

**根因**: 简单 arg(zeta) 平均 vs berry_phase 的"除以平均" unwrap。

**尝试过的修复**:
- berry_phase 式 unwrap: 失败（scale 因子随结构变化，破坏逐原子比例）
- 小位移: P 正确但 Z* 被 1/δ 放大

**可行方向**:
1. 在 **总量级别** 用 berry_phase unwrap，逐原子比例用原始值
2. 增大 k-mesh 减少 2π 跳变频率
3. 中心差分 (±δ) 消除一阶系统误差

#### 问题 B: Z* 分支跳变

**根因**: Wilson loop 特征值 λ_n 越过负实轴时 arg 跳 2π。

**berry_phase 的处理**: 在 zeta (= det(W)) 级别用"除以平均" unwrap，不做逐能带 unwrap。

**DeltaP 的困难**: 逐原子分解需要逐能带 γ_n = arg(λ_n)，但 arg(λ_n) 有 2π 歧义。

**可行方向**:
1. 在 zeta 级别 unwrap 总量，逐原子用比例分解
2. 在特征值级别做跨结构跟踪（匹配特征向量）
3. 避免分支切割（选择 Berry phase 远离 2π 的体系/方向）

#### 问题 C: berryphase_overlap gathering bug

**根因**: pzgemm 输出描述符 (para_orb.desc, nlocal×nlocal) 与实际输出矩阵 (occBands×occBands) 不匹配。

**影响**: O_j 矩阵的某些元素错误，det(O_j) 正确（只需对角元）。

**当前 workaround**: 用 compute_S_dk_link (snap + get_psi_r_psi) 替代。给出 P_elec ratio=0.966。

**根本修复**: 为 occBands×occBands 输出创建专门的 ScaLAPACK 描述符。

---

## 3. 下一步规划

### 3.1 短期（立即可做）

**任务 1: 验证 P_elec 在多个结构上的一致性**

在 BaTiO3 的 3 个结构 (ref, ti_p, ba_p) 上运行，检查每个结构的 P_elec ratio。
如果都 ≈ 0.97，说明 3% 误差是系统性的（来自位置修正近似）。
如果某些结构偏差大，说明有分支跳变。

**预期**: ref 和 ti_p ratio ≈ 0.97，ba_p 可能有分支跳变。

**任务 2: 中心差分 Z* 测试**

用 ±δ 位移 (δ=0.01) 计算 Z*:
- Z* = (Ω/2δ) × (P(+δ) - P(-δ))
- 消除一阶系统误差
- 如果 P(+δ) 和 P(-δ) 的 ratio 都 ≈ 0.97，Z* 误差 ≈ 0%

**前提**: P(+δ) 和 P(-δ) 都没有分支跳变。

### 3.2 中期（需要开发）

**任务 3: 修复 berryphase_overlap gathering**

为 occBands×occBands 输出创建专门的 ScaLAPACK 描述符:
```cpp
int desc_occ[9];
ScalapackConnector::create_desc(desc_occ, occBands, occBands, ...);
```

这样 O_j 矩阵将完全一致于 berry_phase，消除 3% 误差。

**任务 4: 实现 zeta 级别 unwrap**

在 **总量** 级别（不做逐原子缩放）实现 berry_phase 的"除以平均":
1. 收集所有 string 的 zeta = det(W)
2. cave = Σ zeta / N
3. γ_total_unwrapped = arg(cave) + Σ arg(zeta/cave) / N
4. 逐原子: γ^I = γ_total_unwrapped × (Σ γ^I_raw / Σ γ_total_raw)

**关键**: 只在最终平均值上做一次缩放，不在每个 string 上缩放。

### 3.3 长期（需要新功能）

**任务 5: 特征值级别分支跟踪**

跨结构匹配 Wilson loop 特征值:
1. 在 ref 结构计算 λ_n 和 |v_n⟩
2. 在 disp 结构计算 λ'_n 和 |v'_n⟩
3. 用 |v_n⟩ 和 |v'_n⟩ 的重叠匹配 λ_n ↔ λ'_n
4. 展开 arg(λ'_n) - arg(λ_n)（消除 2π 跳变）

**优势**: 直接给出逐能带 Z*，无 1/δ 放大。

**任务 6: SMO-basis Wannierization**

在 DeltaP 内部做规范优化（最小化 spread），提高精度到 Wannier90 水平。

---

## 4. 优先级排序

| 优先级 | 任务 | 预期收益 | 工作量 |
|--------|------|---------|--------|
| **P0** | 任务 1: 多结构 P_elec 一致性 | 确认 3% 误差性质 | 1 小时 |
| **P0** | 任务 2: 中心差分 Z* | 消除一阶误差，可能得到正确 Z* | 3 小时 |
| **P1** | 任务 3: 修复 gathering | 消除 3% 误差 | 1 天 |
| **P1** | 任务 4: zeta unwrap | 修复分支跳变 | 半天 |
| **P2** | 任务 5: 特征值跟踪 | 精确逐能带 Z* | 2 天 |
| **P3** | 任务 6: SMO Wannierization | Wannier90 级精度 | 1 周 |

---

## 5. 文件清单

### 5.1 源码

| 文件 | 内容 |
|------|------|
| `source/source_lcao/module_deltap/deltap.h` | DeltaP 类定义 |
| `source/source_lcao/module_deltap/deltap.cpp` | init + setup_kstring |
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | Wilson loop 特征值法主逻辑 |
| `source/source_lcao/module_deltap/deltap_overlap.cpp` | SMO 重叠计算 |
| `source/source_lcao/module_deltap/deltap_berry.cpp` | Berry connection 方法（旧） |
| `source/source_lcao/module_deltap/deltap_io.cpp` | I/O + sum rule 验证 |
| `source/source_io/module_unk/unk_overlap_lcao.cpp` | berryphase_overlap 函数（有 gathering bug） |
| `source/source_io/module_ctrl/ctrl_scf_lcao.cpp` | DeltaP 入口点 |

### 5.2 文档

| 文件 | 内容 |
|------|------|
| `docs/superpowers/specs/2026-06-27-deltap-wilson-dev-log.md` | 开发日志 |
| `docs/superpowers/specs/2026-06-27-deltap-detailed-analysis.md` | 详细推导与分析 |
| `docs/superpowers/specs/2026-06-27-deltap-algorithm-breakdown.md` | 算法拆解 |
| `docs/superpowers/specs/2026-06-27-deltap-zeta-comparison.md` | zeta 对比（发现 gathering bug） |
| `docs/superpowers/specs/2026-06-27-deltap-algorithm-optimization.md` | 优化方案理论分析 |
| `docs/superpowers/specs/2026-06-26-deltap-wilson-per-atom-report.md` | 初始实现报告 |
| `docs/superpowers/specs/2026-06-26-deltap-test-summary.md` | 测试结果总结 |
| `docs/superpowers/specs/2026-06-26-deltap-diamond-test.md` | Diamond 测试 |
| `DeltaSpin_vs_DeltaP_Essential_Difficulties.md` | 本质困难分析 |
| `Wilson_Loop_Per_Atom_Decomposition.md` | 分解方案数学 |
