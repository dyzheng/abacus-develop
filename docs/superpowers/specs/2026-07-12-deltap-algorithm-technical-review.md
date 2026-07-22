# DeltaP 算法技术文档与批判性分析

> 代码版本: 2026-07-12  
> 审查范围: `module_deltap/`, `module_operator_lcao/deltap_lcao.cpp`, `esolver_ks_lcao.cpp`

---

## 1. 物理动机与算法概述

### 1.1 目标

DeltaP 算法旨在实现**原子级极化分解**（atomic polarization decomposition），即将宏观极化 P 分解为每个原子的贡献 P^I：

```
P = Σ_I P^I
```

这在铁电体、极性半导体等体系中有重要应用，可以回答"哪个原子贡献了多少极化"的问题。

### 1.2 核心思想

传统 Berry phase 方法计算总极化：

```
P = (e/Ω) · R · γ / (2π)
```

其中 γ 是总 Berry phase（Wilson loop 特征值的辐角之和）。DeltaP 的关键创新是：

1. **SMO 投影**：用每个原子的 Spherical Mu-like Orbital（SMO，即每个 l 的第一个 zeta 函数）作为投影子
2. **权重分解**：计算每个 Wannier 函数 |v_n⟩ 到每个原子 SMO 的投影权重 w^I_n
3. **加权 Berry phase**：

```
γ^I = Σ_n w^I_n · γ_n
```

其中 γ_n 是第 n 个 Wilson loop 特征值的辐角。

### 1.3 约束 DFT 模式

当 `deltap_corr = true` 时，DeltaP 进入约束模式：通过调整 Lagrange 乘子 λ^I，使得每个原子的 γ^I 趋向于目标值 target^I。

---

## 2. 数学公式推导

### 2.1 Wilson Loop 与 Berry Phase

给定 k-strings 沿方向 gdir，相邻 k 点的 overlap 矩阵：

```
O_j[m,n] = ⟨ψ_m(k_j) | ψ_n(k_{j+1})⟩
```

**实际实现** (`deltap_wannier.cpp:468-471`): 使用 `berry_overlap_->berryphase_overlap()`，它计算的是周期性部分的重叠：

```
O_j[m,n] = ⟨u_m(k_j) | u_n(k_{j+1})⟩ = ⟨ψ_m(k_j) | e^{-i·dk·r} | ψ_n(k_{j+1})⟩
```

其中 dk 是 k-strings 的间距。

Wilson loop 矩阵是 overlap 矩阵的累积乘积：

```
W_j = O_0 · O_1 · ... · O_j
```

最终 W = W_{N-1} 是闭环（PBC）的 Wilson loop。

### 2.2 酉投影（Newton-Schulz 迭代）

**代码位置**: `deltap_wannier.cpp:554-627`

每一步累积后，W_j = W_{j-1} · O_j 不再是严格酉矩阵（因为 O_j 不完全酉）。代码使用 Newton-Schulz 迭代将 W_j 投影到最近酉矩阵：

```
X_0 = W_j / ||W_j||_F
X_{k+1} = X_k · (3I - X_k†·X_k) / 2
```

直到 ||X†X - I||_F < tol。

**批判性分析**:
- **风险 1**: Newton-Schulz 迭代收敛要求初始矩阵的奇异值在 (0, √3) 范围内。代码使用 `scale = 1/sqrt(frob2/n_dim)` 归一化，这假设奇异值接近均值。对于秩亏损矩阵（n_dim 过大），某些奇异值可能接近 0，导致收敛缓慢。
- **风险 2**: 迭代 20 次后若未收敛，代码静默继续，可能导致非酉矩阵进入对角化步骤。

### 2.3 特征值退绕（Phase Unwrapping）

**代码位置**: `deltap_wannier.cpp:654-718`

Wilson loop 特征值 λ_n = |λ_n|·exp(iζ_n)，其中 ζ_n ∈ (-π, π]。为获得连续的 Berry phase，需要退绕：

1. **首次对角化** (j=0): 按辐角排序，建立初始顺序
2. **后续步骤** (j>0): 对每个新特征值，找到与上一特征值辐角差最小的匹配，计算连续增量

```
γ_n^{(j)} = γ_n^{(j-1)} + arg(λ_n^{(j)} / λ_n^{(j-1)})
```

**批判性分析**:
- **风险 3 (分支跳变)**: 当两个特征值在复平面上交叉时，匹配可能错误，导致 γ_n 跳变。代码没有检测特征值简并或接近简并的情况。
- **风险 4 (全局 vs 局部)**: 退绕是沿 k-string 进行的，但跨不同 k-string 时没有一致性保证。不同 k-string 可能选择不同的分支，导致平均后的 γ^I 不连续。

### 2.4 Per-Atom 权重 w^I_n

**代码位置**: `deltap_wannier.cpp:800-926`

SMO 投影矩阵 D_I:

```
D_I[iat][lm][n] = Σ_μ S_k^*[iat][lm][μ] · ψ_k[μ + n·nrow]
```

其中 S_k[iat][lm][μ] = Σ_R exp(2πi·k·R) · ⟨φ_onsite^I_lm | φ_μ(R)⟩

**Löwdin 正交化**:

```
tilde_proj[a,n] = Σ_b (S^{-1/2})_{a,b} · proj[b,n]
```

其中 S_{a,b} = ⟨α_a | α_b⟩ 是 SMO 重叠矩阵，proj[a,n] = D_mat[a, m] · VR[m, n]。

权重:

```
w^I_n = Σ_{a ∈ I} |tilde_proj[a,n]|²
```

**批判性分析**:
- **风险 5 (SMO 基组不完备)**: SMO 仅使用每个 l 的第一个 zeta 函数。对于具有多个 zeta 的基组（如 double-zeta），SMO 空间不能完整表示占据带。这导致 Σ_I w^I_n < 1，即部分 Wannier 函数"泄漏"到 SMO 空间之外。
- **风险 6 (Löwdin 正交化的数值问题)**: S^{-1/2} 通过特征值分解计算（`deltap_overlap.cpp:348-397`）。对于近线性相关的 SMO 基组，S 的小特征值导致 S^{-1/2} 的大元素，放大数值噪声。代码使用阈值 1e-10 截断，但这可能破坏 S·S^{-1} = I 的精确性。

### 2.5 HK 修正（Berry Connection 算符）

**代码位置**: `deltap_wannier.cpp:1144-1322`

约束 DFT 需要在 Hamiltonian 中加入修正项：

```
H → H + H_K
```

其中 H_K 是 k-dependent 的修正，用于驱动 γ^I 趋向 target^I。

**推导** (从代码反推):

目标函数: L = Σ_I λ^I · (γ^I - target^I)

梯度: ∂L/∂ψ = Σ_I λ^I · ∂γ^I/∂ψ

由 γ^I = Σ_n w^I_n · γ_n，且 γ_n = -i·ln(λ_n)（Wilson loop 特征值），

HK 修正的物理意义是：在 k-space 的相邻 k 点之间插入一个"规范场"，改变 Berry connection。

**实际实现**:

```cpp
// Step 1: SC = S_dk · C_R
// Step 2: F = (i/2) · w_eff · SC  
// Step 3: M = F · C_L†
// Step 4: H_sym = (M + M†) / 2
```

其中:
- `S_dk` 是位移重叠矩阵: S_dk[μ,ν] = Σ_R exp(2πi·dk·R) · ⟨φ_μ(0) | φ_ν(R)⟩
- `w_eff[n] = Σ_I λ^I · w^I_n / S_I`（带 per-atom 预条件）
- `C_L`, `C_R` 是 k_L, k_R 处的波函数系数

**批判性分析**:
- **风险 7 (公式符号)**: 代码注释说 `F = (i/2) · w_eff · SC`，但 `half_i = complex(0, -0.5)`，所以实际是 `F = (-i/2) · w_eff · SC`。这与文档中的 HK correction sign (development log #1: "F = -(i/2)*w_eff*SC not +(i/2)") 一致，但容易混淆。
- **风险 8 (H_sym 的厄米性)**: 代码显式构造 `H_sym = (M + M†)/2`，但 M 本身依赖于 ψ，而 ψ 在内循环中会变化。这意味着 HK 修正不是严格的"外场"，而是自洽依赖波函数。
- **风险 9 (per-atom 预条件 S_I)**: `w_eff[n] = Σ_I λ^I · w^I_n / S_I`，其中 `S_I = avg_n Σ_{lm} |D_I[iat][lm][n]|²`。这个预条件的目的是归一化不同原子的灵敏度，但 S_I 本身依赖于波函数，在内循环中会变化，导致 w_eff 不稳定。

### 2.6 内循环优化（Fletcher-Reeves CG）

**代码位置**: `module_optimizer/bfgs.h`, `esolver_ks_lcao.cpp:646-686`

内循环使用 Fletcher-Reeves 共轭梯度法优化 λ:

```
r_k = γ(λ_k) - target           # residual
β_k = |r_k|² / |r_{k-1}|²       # FR beta
d_k = r_k + β_k · d_{k-1}       # search direction
λ_{k+1} = λ_k + α_opt · d_k     # update
```

线搜索使用 secant 方法:

```
α_opt = α_trial · (-r_k · Δr) / |Δr|²
```

其中 Δr = r_trial - r_k。

**批判性分析**:
- **风险 10 (非真实梯度)**: 代码将 residual r = γ - target 作为"梯度"方向，但这不是真正的 ∂L/∂λ。真正的梯度需要计算 dγ/dλ（Jacobian），这涉及 Wilson loop 对 λ 的导数。当前的 Fletcher-Reeves 方法实际上是在做"pseudo-gradient descent"，不保证收敛。
- **风险 11 (CG 历史重置)**: 每次外循环开始时，CG 历史被重置 (`start_outer`)。这意味着内循环每步只能做 steepest descent，无法累积共轭方向信息。
- **风险 12 (线搜索的可靠性)**: secant 方法假设 γ(λ) 是线性的，但对于 Wilson loop 这种高度非线性的函数，线搜索可能给出不合理的 α_opt。代码有 max_step 限制，但这只是截断，不能修复方向错误。

### 2.7 Gate 触发机制

**代码位置**: `esolver_ks_lcao.cpp:624-627`

```cpp
bool gate = iter > 1 && !dp_dp->inner_loop_cooldown()
              && (dp_dp->inner_loop_triggered()
                  || (this->drho > 0 && this->drho < PARAM.inp.deltap_inner_thr));
```

- 首次激活: `drho < deltap_inner_thr`（电荷基本收敛）
- 后续: 每 N_COOLDOWN=5 次外循环激活一次

**批判性分析**:
- **风险 13 (哈密顿量突变)**: gate 打开时，HK 修正突然加入 H，导致电荷密度振荡。没有平滑过渡（如 HK 的 gradual ramp-up）。
- **风险 14 (cooldown 的启发式)**: N_COOLDOWN=5 是经验值，没有理论依据。对于不同体系、不同 k-mesh，最优 cooldown 可能差异很大。

---

## 3. 实现细节与数据结构

### 3.1 关键数据结构

| 数据成员 | 类型 | 用途 |
|---------|------|------|
| `kstring_data_` | `vector<KSpaceData>` | 每条 k-string 的 S_k, dS_k, D_I |
| `D_I_all_` | `vector<vector<vector<vector<complex>>>>` | 所有 k 点的 D_I（用于 HK） |
| `S_dk_` | `vector<complex>` | 位移重叠矩阵（2D block-cyclic） |
| `smo_overlap_` | `vector<double>` | SMO 重叠矩阵 S_{ab} |
| `smo_overlap_inv_` | `vector<double>` | S^{-1/2}（Löwdin 正交化） |
| `W_prev_` | `vector<complex>` | 跨 SCF 迭代的分支追踪状态 |
| `bfgs_` | `BFGS` | 内循环优化器 |

### 3.2 并行化

- **k 点并行**: `paraV_` 管理 2D block-cyclic 分布
- **D_I 的 MPI 归约**: `deltap_wannier.cpp:419-436` 对每个 k 点的 D_I 做 Allreduce
- **HK 修正仅支持串行**: `deltap_wannier.cpp:1164-1169` 检查 `nrow == ncol`，并行时直接返回

### 3.3 分支追踪（Branch Tracking）

**代码位置**: `deltap_wannier.cpp:1327-1385`

为解决跨 SCF 迭代/跨运行的分支不一致问题（B16），代码使用文件 `deltap_branch.dat` 保存每个原子的 Wilson loop 乘积 W^I。下次运行时加载作为参考，选择最近的 2π 分支。

**批判性分析**:
- **风险 15 (文件覆盖策略)**: `save_branch()` 只在文件不存在时写入（首次运行建立 baseline）。但如果结构变化（如分子动力学），旧的 branch 参考不再适用。
- **风险 16 (per-atom vs per-string)**: 分支追踪是按原子进行的，但 γ^I 是从多条 k-string 平均得到的。不同 k-string 可能有不同的分支选择，平均后的 γ^I 可能位于"错误"的分支上。

---

## 4. 风险评估与改进建议

### 4.1 高风险问题

| ID | 风险 | 严重程度 | 建议 |
|----|------|---------|------|
| R3 | 特征值交叉导致退绕错误 | 高 | 添加简并检测，交叉时切换退化微扰论 |
| R7 | HK 公式符号混淆 | 中 | 统一文档与代码注释，添加单元测试 |
| R10 | 非真实梯度的伪 CG | 高 | 实现真正的 dγ/dλ Jacobian |
| R13 | HK 突变导致振荡 | 高 | 实现 gradual ramp-up |

### 4.2 中风险问题

| ID | 风险 | 严重程度 | 建议 |
|----|------|---------|------|
| R5 | SMO 基组不完备 | 中 | 支持多 zeta SMO，或 adaptive basis |
| R6 | Löwdin 正交化数值不稳定 | 中 | 使用 SVD 替代特征值分解，显式检查条件数 |
| R8 | HK 的自洽依赖 | 中 | 明确记录这是近似，或实现真正的外场 |
| R15 | 分支文件覆盖策略 | 中 | 添加结构指纹校验，自动检测结构变化 |

### 4.3 低风险问题

| ID | 风险 | 严重程度 | 建议 |
|----|------|---------|------|
| R1 | Newton-Schulz 收敛失败 | 低 | 添加 fallback 到 SVD polar decomposition |
| R4 | 跨 k-string 分支不一致 | 低 | 使用全局 zeta 约束替代 per-string 独立退绕 |
| R11 | CG 历史重置 | 低 | 保留部分历史或使用 L-BFGS |
| R12 | 线搜索不可靠 | 低 | 添加 trust region 或 backtracking |

---

## 5. 公式推导补充

### 5.1 S(dk) 的位移重叠矩阵

**代码位置**: `deltap_wannier.cpp:49-153`

```
S_dk[μ,ν] = Σ_R exp(2πi·dk·R) · ⟨φ_μ(0) | φ_ν(R)⟩
```

**物理意义**: 这是 Bloch 态重叠 ⟨ψ_μ(k) | ψ_ν(k+dk)⟩ 的 LCAO 表示。

**位置修正** (`deltap_wannier.cpp:155-285`): 代码还实现了将 Bloch 重叠转换为周期性部分重叠的修正：

```
⟨u_μ(k) | u_ν(k+dk)⟩ ≈ ⟨φ_μ | φ_ν(R)⟩ · exp(2πi·dk·R) - i·dk·⟨φ_μ | r | φ_ν(R)⟩ · exp(2πi·dk·R)
```

这使用了小 dk 近似：exp(-i·dk·r) ≈ 1 - i·dk·r。

**批判性分析**:
- **风险 17 (小 dk 近似的精度)**: 对于稀疏 k-mesh（如 2×2×2），dk 可能不够小，一阶近似误差显著。应提供精确 exp(-i·dk·r) 积分的选项。

### 5.2 Zeta 缩放（Zeta Rescaling）

**代码位置**: `deltap_wannier.cpp:996-1016`

代码在计算 γ^I 后做了一个"zeta rescaling"：

```cpp
double gamma_correct = std::arg(zeta_scalar);  // zeta = det(W) 的辐角
double scale = gamma_correct / gamma_raw_sum;
gamma_I_per_atom[iat] *= scale;
```

**物理意义**: 总 γ 应该等于 zeta 的辐角（因为 det(W) = Π_n λ_n）。这个缩放确保 Σ_I γ^I = arg(zeta)。

**批判性分析**:
- **风险 18 (破坏 per-atom 分解)**: 缩放假设 γ^I 的比例关系是正确的，只是整体幅度需要修正。但如果不同原子的 γ^I 有不同的系统误差（如不同的 SMO 投影质量），等比例缩放不能修复相对误差。
- **风险 19 (与 branch-set selection 的冲突)**: 代码先做 zeta rescaling，然后做 branch-set selection。但 branch-set selection 假设 2π 间隔是 `2π · w_sum^I`，而 rescaling 改变了这个间隔。

### 5.3 Resta-Z 方法（备选）

**代码位置**: `deltap_wannier.cpp:1476-1774`

代码还实现了基于 Resta 的 z^I 方法：

```
z^I = ⟨exp(-i·2π·r/R)⟩^I
⟨r_elec^I⟩ = -(R/2π) · Im[ln(z^I)]
```

但实际实现使用了 Mulliken 分析近似：

```
⟨r^I⟩ = Σ_{μ,ν ∈ I} D_{μ,ν} · r_{μ,ν} / Σ_{μ,ν ∈ I} D_{μ,ν} · S_{μ,ν}
```

**批判性分析**:
- **风险 20 (Mulliken 近似的 gauge 依赖性)**: Mulliken 分析不是 gauge 不变的，依赖于 LCAO 基组的选择。对于扩展基组，结果可能不合理。
- **风险 21 (仅支持 Gamma 点)**: 代码注释说 "Skip this k-point if not Gamma"（`deltap_wannier.cpp:1612`），但 Resta-Z 方法原则上需要所有 k 点。当前实现对于 multi-k 计算是不完整的。

---

## 6. 代码质量观察

### 6.1 正面

1. **模块化设计**: DeltaP 类封装良好，与 esolver 的接口清晰
2. **分支追踪机制**: 解决了跨运行的相位一致性问题
3. **Newton-Schulz 迭代**: 比 SVD 更高效，适合小矩阵
4. **SMO 预条件**: 归一化不同原子的灵敏度是合理的工程实践

### 6.2 待改进

1. **大量调试输出**: 生产代码中不应有 `std::cout` 输出中间变量
2. **硬编码常数**: `N_COOLDOWN=5`, `anchor_thr_=1e-8`, `max_iter=20` 等应可配置
3. **缺少单元测试**: Wilson loop、退绕、HK 修正等核心功能应有独立测试
4. **MPI 支持不完整**: HK 修正仅支持串行，限制了大规模应用

---

## 7. 总结

DeltaP 算法在概念上是正确的：通过 SMO 投影将总 Berry phase 分解为原子贡献，然后用约束 DFT 控制每个原子的极化。但实现中存在多个数值和算法风险，主要集中在：

1. **Wilson loop 的数值稳定性**: 酉投影、特征值退绕、分支追踪都需要更鲁棒的处理
2. **HK 修正的自洽性**: 当前实现将 HK 视为"外场"，但实际上它依赖波函数
3. **优化的可靠性**: 伪梯度 CG 不保证收敛，需要真正的 Jacobian 或更稳健的优化策略

建议优先处理 R3（特征值交叉）、R10（真实梯度）、R13（HK 平滑过渡）这三个高风险问题。
