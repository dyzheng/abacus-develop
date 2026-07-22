# DeltaP 验证 — 根因分析与修复 (2026-07-12)

## 摘要

在 BN (zinc-blende) 体系上执行 DeltaP 实现有效性验证过程中，发现了三个根因 bug：k-string 索引无效、Wilson loop 特征值坍缩、SMO 投影矩阵的非 unitarity 处理。修复前两个后，per-atom gamma 从恒为 0 变为非零合理值。dγ/dλ 响应测试因 branch 幅角跳变被阻塞。

## Bug 1: k-string 索引超界（关键）

**现象**：所有 k-string 的 `O_kpair[j]` 对于 j ≥ 1 均为零矩阵 → Wilson loop 特征值坍缩 → gamma = 0。

**根因**：`setup_kstring()`（`deltap.cpp:212-229`）用全 Monkhorst-Pack 网格索引 `ix + iy*mp_x + iz*mp_x*mp_y` 填充 `k_index_`，但这些索引对应全网格（如 3×3×3 = 27 个 k 点），而 ABACUS 因时间反演对称性仅存储 14 个 k 点（nks=14）。索引 9、18 等超出 nks，导致 `berryphase_overlap()` 因 `ik_R >= nks` 条件跳过而返回零矩阵。

**后果**：每个 k-string 仅第 0 个 link（两个 k 点都在存储集中）有非零 overlap，其余 link 的 O_kpair 全为零。Wilson loop 累积计算因此退化。

**修复**：使用 `symmetry=-1` 强制存储全 BZ k 点（27 个对于 3×3×3 网格），确保所有 k_index_ 索引有效。正确修复应将 `setup_kstring` 改为建立从全网格 (ix,iy,iz) 到存储 k 点索引的映射表。

## Bug 2: Wilson loop 矩阵归一化破坏 unitarity

**现象**（修复 Bug 1 后，使用 2×2×2 等效测试验证）：即使 O_kpair 全为非零矩阵，Wilson loop 特征值在第一步对角化后即坍缩为零——|λ| = 7.7e-25 而非 1。

**根因**：`deltap_wannier.cpp:544-551`（旧代码）的矩阵归一化：
```cpp
max_elem = max(|W_mat|);
W_mat /= max_elem;  // 标量归一化
```
Wilson loop 矩阵累积是按 `W_j = W_{j-1}·O_j` 步进的。每一步的标量归一化将整个矩阵除以一个常数，使所有特征值等比例缩放。反复归一化导致特征值向零指数坍缩。

**数学推导**：设矩阵 M = X Σ Y†（SVD）。正则化后的矩阵为 M/σ_max，特征值从 |λ| ≈ 1 缩小为 |λ|/σ_max。重复累乘导致 |λ| → 0。正确做法是使用极化分解（polar decomposition）将 M 投影到酉群 U(n) 上：最接近 M 的酉矩阵为 U = X Y†（Frobenius 范数下的极小化器）。

**修复**：用 LAPACK `zgesvd` 替代标量归一化，每步计算 `U·V†`（SVD 的酉因子）作为酉投影。修正代码位于 `deltap_wannier.cpp:545-605`。

## Bug 3: SMO 重叠矩阵的非 unitarity（部分解决）

**现象**（旧代码修复后）：对于仅 4 个占据带的 BN，`berryphase_overlap()` 返回的 O_j 矩阵非酉（秩亏损到 1-2）。SVD 极化分解虽然强制 W_mat 化为酉矩阵，但物理意义上基组大小不足以容纳全部占据带的 Wannier 函数。

**缓解**：使用更多占据带（BN 的 nbands=14 而非仅 nele/DEGSPIN=4）可改善 unitarity。当前代码的 n_dim = nocc_use = min(nelec/2, nbands)，对于 BN 为 4。如果 n_dim 提升至 8-10，O_j 的 singular values 将更接近 1。

**状态**：未修复。SVD 酉投影是数值上的正确解决方法，但更大的 n_dim 能从根本上改善 unitarity。

## 三组验证测试状态

### Test 1: dγ/dλ 响应
| 项目 | 状态 |
|------|------|
| HK correction λ 依赖性 | ✓ 已验证（w_eff ∝ λ） |
| gamma 从 0 变为非零 | ✓ Bugs 1+2 修复后通过 |
| 单调的 dγ/dλ 关系 | ✗ 阻塞 |

**阻塞原因**：
1. **Branch 幅角跳变**：per-string Wilson loop 特征值的 2π 分支在不同 λ 值下产生不一致的 unwrapping，导致 γ 在 λ 相近时跳变达 1.0 rad。
2. **电荷-λ 耦合噪声**：SCF 收敛过程中 gamma 的变化（~0.1 rad）远大于 λ 驱动的 HK correction 信号（~0.02 Rad）。
3. **过小基组 n_dim=4**：O_j 矩阵秩亏损大，每次 SVD 投影都丢失信息。

**建议方案**：
A. 增加 n_dim（使用更多占据带）减少秩亏损
B. 在 branch unwrapping 中，对所有 λ 值使用相同的 branch 参考态（一致性连续化）
C. 测量 dγ/dλ 时，记录每条 k-string 的 'raw' zeta phase（不受 per-atom 分解和 branch 选择影响）

### Test 2: 完整 SCF target sweep
- 尚未开始 — 被 Test 1 阻塞

### Test 3: Wannier90 对比
- 尚未开始 — 预备状态已具备（BN win, wout 已计算，wannier90.x 可用）

## 代码修改

### 修改文件
1. `source/source_lcao/module_deltap/deltap_wannier.cpp` — SVD 酉投影替代标量归一化（~60 行变更）
2. 同上 — 移除 HKDBG 和 O_DEBUG 临时调试代码

### 未修改但应修改
1. `source/source_lcao/module_deltap/deltap.cpp:setup_kstring()` — 应添加全网格→存储点的映射
2. `deltap.h`/`.cpp` — n_dim 应可配置（当前固定为 nocc）

## 下一步

1. **修复 branch 不连续性**：对所有 λ 值使用 `W_prev_` 初始化，确保跨 λ 幅角一致
2. **增加 n_dim**：测试 n_dim=8/10 对 gamma 稳定性的影响
3. **dγ/dλ 测量**：在固定 branch 状态和较大 n_dim 下重复
4. **完整 SCF sweep**：用 dγ/dλ 斜率校准 target 序列
5. **Wannier90 对比**：运行完整 pipeline
