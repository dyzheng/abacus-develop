# DeltaP 风险点与解决方案

> 基于 2026-07-12 深度代码审查  
> 所有行号引用均基于 `module_deltap/` 和 `esolver_ks_lcao.cpp` 当前版本

---

## 目录

- [一、Critical 级风险（立即修复）](#一critical-级风险立即修复)
  - [C1: HK 修正使用的 S_dk 与 Wilson loop 不一致](#c1-hk-修正使用的-s_dk-与-wilson-loop-不一致)
  - [C2: Zeta rescaling 数学错误](#c2-zeta-rescaling-数学错误)
  - [C3: 内循环 psi-lambda 不一致](#c3-内循环-psi-lambda-不一致)
- [二、High 级风险（本迭代修复）](#二high-级风险本迭代修复)
  - [H1: SMO 重叠矩阵非对称](#h1-smo-重叠矩阵非对称)
  - [H2: D_I 共轭可能错误](#h2-d_i-共轭可能错误)
  - [H3: 分支选择使用错误的间距](#h3-分支选择使用错误的间距)
  - [H4: Gauge 修复的 phase_corrections_ 未生效](#h4-gauge-修复的-phase_corrections_-未生效)
  - [H5: 最终特征值重排使用 fmod 丢失累积相位](#h5-最终特征值重排使用-fmod-丢失累积相位)
- [三、Medium 级风险（下迭代修复）](#三medium-级风险下迭代修复)
  - [M1: BFGS 类名误导——实际是 Fletcher-Reeves CG](#m1-bfgs-类名误导实际是-fletcher-reeves-cg)
  - [M2: Lambda mixing 导致 BFGS 内部状态不一致](#m2-lambda-mixing-导致-bfgs-内部状态不一致)
  - [M3: 伪梯度方向缺乏 Jacobian](#m3-伪梯度方向缺乏-jacobian)
  - [M4: Newton-Schulz 收敛条件不保证](#m4-newton-schulz-收敛条件不保证)
  - [M5: 诊断检验公式错误](#m5-诊断检验公式错误)
  - [M6: select_branch_set 函数为死代码](#m6-select_branch_set-函数为死代码)
- [四、Low 级风险（维护性改进）](#四low-级风险维护性改进)
  - [L1: 特征值追踪在简并点可能失败](#l1-特征值追踪在简并点可能失败)
  - [L2: 自适应步长的系统性偏差](#l2-自适应步长的系统性偏差)
  - [L3: SMO 权重注释声称 sum = 1 但实际 ≤ 1](#l3-smo-权重注释声称-sum--1-但实际--1)
  - [L4: HK 修正中 w_eff 预条件器缺乏理论推导](#l4-hk-修正中-w_eff-预条件器缺乏理论推导)
- [五、已验证正确的部分](#五已验证正确的部分)

---

## 一、Critical 级风险（立即修复）

### C1: HK 修正使用的 S_dk 与 Wilson loop 不一致

**位置**: `deltap_wannier.cpp:1171-1175`, `deltap_wannier.cpp:55-153` vs `unk_overlap_lcao.cpp:503-537`

**问题描述**:

HK 修正 (`compute_hk_correction`) 调用 `compute_S_dk()` 构建重叠矩阵。Wilson loop (`berryphase_overlap`) 通过 `prepare_midmatrix_pblas()` 构建重叠矩阵。两者存在两处不一致：

| 方面 | compute_S_dk (HK 用) | berryphase_overlap (Wilson loop 用) |
|------|----------------------|-------------------------------------|
| 相位 | `exp(2πi·(dk_frac·R_int - dk_frac·τ))` | `exp(2πi·(k_c_R·R - dk_c·τ))` |
| 位置修正 | **无** | `overlap += -i·dk·tpiba·⟨φ\|r\|φ(R)⟩` |

`compute_S_dk` 使用的是分数坐标 dk 且没有位置修正。`berryphase_overlap` 使用笛卡尔 k 向量 k_R 并包含位置修正 `-i·dk·tpiba·⟨r⟩`。

**数学推导**:

Wilson loop 的 overlap 矩阵应该是周期性部分的重叠：

```
O_j[m,n] = ⟨u_m(k_j)|u_n(k_{j+1})⟩
         = ⟨ψ_m(k_j)|e^{-i·dk·r̂}|ψ_n(k_{j+1})⟩
         ≈ ⟨ψ_m(k_j)|ψ_n(k_{j+1})⟩ - i·dk·⟨ψ_m(k_j)|r̂|ψ_n(k_{j+1})⟩
```

一阶展开的第二项（位置修正）对于将 Bloch 态重叠转换为周期性部分重叠是**必要的**。缺少这一项，HK 修正操作在与 Wilson loop 不同的流形上，导致约束力与计算的 γ 不一致。

**影响**: HK 修正方向错误 → λ 优化方向偏离真实梯度 → 收敛到错误的 λ 值或不收敛。

**解决方案**:

`compute_hk_correction` 应使用 `compute_S_dk_link()` 替代 `compute_S_dk()`。`compute_S_dk_link` 已存在于代码中（`deltap_wannier.cpp:160-285`），实现了正确的 berry_phase 相位约定和位置修正。

具体修改：
```cpp
// compute_hk_correction 中替换:
if (S_dk_.empty())
    compute_S_dk(ucell);

// 改为对每个 (ik_L, ik_R) link 调用:
compute_S_dk_link(ucell, kvec_d_R, kvec_c_L, kvec_c_R);
```

由于 `compute_S_dk_link` 有缓存机制（`S_dk_cache_`），性能影响有限。

---

### C2: Zeta rescaling 数学错误

**位置**: `deltap_wannier.cpp:1000-1016`

**问题描述**:

代码计算：
```cpp
double gamma_correct = std::arg(zeta_scalar);       // arg(det W) ∈ (-π, π]
double gamma_raw_sum = Σ_I γ^I_raw;
double scale = gamma_correct / gamma_raw_sum;
γ^I *= scale;
```

其中 `γ^I_raw = Σ_n w^I_n · γ_n`，`γ_n` 是**退绕后**的累积相位（可超出 (-π, π]）。

**数学错误**:

`arg(det W) = arg(Π_n λ_n) = Σ_n arg(λ_n) mod 2π`，其中 `arg(λ_n) ∈ (-π, π]`。

但 `Σ_n γ_n`（退绕后的相位之和）可能不等于 `arg(det W)`。如果某个 γ_n 在 k-string 上累积了额外的 2π，则：

```
Σ_n γ_n = arg(det W) + 2πk  (k ≠ 0)
```

此时：
```
scale = arg(det W) / (arg(det W) + 2πk) ≠ 1
```

rescaling 会引入一个与 2πk 成正比的错误缩放。

**举例**: 若 `arg(det W) = 0.5`，`Σ_n γ_n = 0.5 + 2π ≈ 6.78`，则 `scale ≈ 0.074`，将所有 per-atom gamma 缩小到原值的 7.4%。

**影响**: Per-atom γ^I 的绝对值被错误缩放，约束目标无法正确达到。

**解决方案**:

方案 A（推荐）：移除 zeta rescaling。当 SMO 基组完备时 `Σ_I w^I_n = 1`，`scale ≈ 1`，rescaling 无影响。当 SMO 不完备时，rescaling 是一个缺乏理论依据的启发式。正确做法是报告 `Σ_I w^I_n` 的值让用户判断 SMO 质量。

```cpp
// 删除 lines 1000-1016 的 rescaling 代码
// 替换为诊断输出:
double weight_sum = 0;
for (int n = 0; n < n_dim; ++n) {
    double w_n = 0;
    for (int iat = 0; iat < nat_; ++iat) w_n += w_In_matrix[n][iat];
    weight_sum += w_n;
}
std::cout << "   DeltaP: SMO weight sum = " << weight_sum / n_dim
          << " (should be ~1 for complete basis)" << std::endl;
```

方案 B：如果必须保留 rescaling，至少使用退绕后的总和 `Σ_n γ_n`（而不是 `arg(det W)`）作为参照：
```cpp
double gamma_unwrapped_sum = 0;
for (int n = 0; n < n_dim; ++n) gamma_unwrapped_sum += gamma_unwrapped[n];
double scale = gamma_unwrapped_sum / gamma_raw_sum;  // 保持相对分布，修正 SMO 不完全性
```

---

### C3: 内循环 psi-lambda 不一致

**位置**: `esolver_ks_lcao.cpp:670-685`

**问题描述**:

内循环中 lambda mixing 导致：
1. BFGS 提议 λ_BFGS = λ_old + dnu
2. mixing 后 λ_actual = λ_old + β·dnu
3. psi 是 H(λ_BFGS) 的本征态（来自 trial 对角化）
4. 但 Hamiltonian 被设置为 H(λ_actual)

**下一轮迭代**：BFGS 在 (ψ(λ_BFGS), λ_actual) 处评估 residual。ψ 不对应当前的 λ，导致 residual 表面不连续。

**数学分析**:

设 r(ψ, λ) 为 residual。BFGS 看到的是：
```
r_eff(λ) = r(ψ(λ_prev), λ)
```
而非正确的：
```
r(λ) = r(ψ(λ), λ)
```

差异量级：
```
|r_eff - r| ≈ |∂r/∂ψ| · |ψ(λ_prev) - ψ(λ)|
            ≈ |∂r/∂ψ| · |∂ψ/∂λ| · (1-β)·|dnu|
```

当 β 接近 0（强 mixing）时，差异很大；β 接近 1（弱 mixing）时，差异小。

**影响**: CG 加速被破坏，内循环退化为带噪声的 steepest descent。

**解决方案**:

方案 A（推荐）：在内循环的每步 accept_trial 后，使用 mixed lambda 重新对角化：
```cpp
// 现有代码 (lines 682-685):
dp_op->set_lambda(lambda);  // mixed lambda
hk_corr_adj = ...;
dp_op->set_hk_correction(hk_corr_adj);
// 新增：
hsolver_obj.solve(..., true);  // skip_charge=true, 重新对角化使 psi 与 mixed lambda 一致
```

代价：每步多一次对角化。但对于小 nscf（3-5步），总开销可接受。

方案 B：移除 lambda mixing（设 `deltap_lambda_mixing = 0`），让 BFGS 自由优化。依靠 cooldown 和外循环电荷 mixing 来稳定。

---

## 二、High 级风险（本迭代修复）

### H1: SMO 重叠矩阵非对称

**位置**: `deltap_overlap.cpp:241,319`

**问题描述**:

`compute_smo_overlap_matrix` 只填充 `smo_overlap_[a + b * m_dim]`（一个三角），未设置对称元素 `smo_overlap_[b + a * m_dim]`。理论上 `⟨α_a|α_b⟩ = ⟨α_b|α_a⟩*`（实基组下为 `⟨α_a|α_b⟩ = ⟨α_b|α_a⟩`），但代码没有利用这一性质。

**影响**: 若 dsyev 对非对称输入行为未定义或产生错误特征值，S^{-1/2} 将错误。

**验证**: 代码在 line 338 检查了对称性（`max_asym`），若输出中 `max_asym` 非零，则确实存在此问题。

**解决方案**: 在填充循环结束后，显式对称化：
```cpp
for (int i = 0; i < smo_m_dim_; ++i)
    for (int j = 0; j < i; ++j)
        smo_overlap_[j + i * smo_m_dim_] = smo_overlap_[i + j * smo_m_dim_];
```

---

### H2: D_I 共轭可能错误

**位置**: `deltap_berry.cpp:96-99`

**问题描述**:

```cpp
const std::complex<double> s_conj = std::conj(s_val);
D_I[iat][lm][n] += s_conj * psi_k[mu_local + n * nrow_local];
```

S_k 的物理意义是 k 空间重叠：
```
S_k[iat][lm][μ] = Σ_R exp(i·k·R) · ⟨α^I_lm | φ_μ(R)⟩
```

D_I 应该是：
```
D_I[iat][lm][n] = ⟨α^I_lm(k) | ψ_n(k)⟩ = Σ_μ S_k[iat][lm][μ] · c_{μ,n}
```

代码计算的是 `Σ_μ S_k* · c_{μ,n}` 而非 `Σ_μ S_k · c_{μ,n}`。

**分析**: 这取决于 ABACUS 的 Bloch 约定。如果 Bloch 函数定义为 `ψ_k = Σ_R exp(-ikR) φ(r-R)`（负号约定），则 `⟨α|ψ⟩ = Σ_μ S_k* · c_μ`，代码正确。如果定义为正号，则需要去掉 conj。

**解决方案**: 验证 ABACUS 的 Bloch 约定。检查 `berryphase_overlap` 或 `HamiltLCAO` 中 Bloch 相位的符号。

```bash
# 在 ABACUS 源码中搜索 Bloch phase 约定
grep -r "exp.*k.*R\|TWO_PI.*kvec" source/source_lcao/module_hcontainer/ | head -5
```

若为正号约定，将 line 96 改为：
```cpp
D_I[iat][lm][n] += s_val * psi_k[mu_local + n * nrow_local];
```

---

### H3: 分支选择使用错误的间距

**位置**: `deltap_wannier.cpp:1034-1039`

**问题描述**:

内联分支搜索使用均匀间距 `2π * scale`：
```cpp
double candidate = g + k * 2.0 * M_PI * scale;
```

正确的分支集合是：
```
Γ^I = {γ^I_0 + 2π · Σ_n w^I_n · k_n : k_n ∈ Z}
```

对于单带偏移（k_m = ±1），正确的间距应为 `2π · w^I_m · scale`（每个带的权重不同）。

**影响**: 当某个原子的 SMO 权重高度集中在少数带上（如 w^I_0 = 0.8, w^I_1 = 0.15, w^I_2 = 0.05），正确的最近分支在 `2π · 0.8 · scale` 处，但代码搜索 `2π · 1.0 · scale`。可能选错分支。

**解决方案**: 使用 `select_branch_set` 函数（已实现但从未被调用），它正确地使用 per-band 权重：

```cpp
// 替换 lines 1023-1052 的内联代码为：
std::vector<int> k_selected;
double g_selected = select_branch_set(smo_weights_per_atom[iat],
                                       gamma_unwrapped, n_dim,
                                       prev_gamma[iat], k_selected);
gamma_accum[iat] += g_selected - gamma_I_per_atom[iat];
gamma_I_per_atom[iat] = g_selected;
prev_gamma[iat] = g_selected;
```

需要将 `smo_weights_per_atom[iat]` 构造为 per-band 权重向量。

---

### H4: Gauge 修复的 phase_corrections_ 未生效

**位置**: `deltap_gauge.cpp:125,134-148`

**问题描述**:

当 anchor 切换时（`abs_D < anchor_thr_`），代码计算：
```cpp
double delta_phi = std::arg(D_new) - std::arg(D_old);
phase_corrections_[n] *= std::polar(1.0, -delta_phi);
```

但 `phase_corrections_[n]` **从未被应用**到后续的 `gauge_phase_[j][n]`。Anchor 切换后的 gauge phase 直接从新 anchor 计算，丢失了相位修正。

**影响**: Anchor 切换点处的 gauge 可能不连续，导致 Berry connection 计算出现人为跳变。

**解决方案**: 在 anchor 切换后，将 `phase_corrections_` 应用到所有后续 k 点的 gauge phase：

```cpp
// 在 lines 127-129 (anchor 切换) 之后添加:
// 回溯修正所有已计算的 gauge_phase
for (int jj = 0; jj < j; ++jj)
    gauge_phase_[jj][n] *= std::polar(1.0, -delta_phi);
```

或者更简洁地，在最终使用 `gauge_phase_` 时乘以 `phase_corrections_`。

---

### H5: 最终特征值重排使用 fmod 丢失累积相位

**位置**: `deltap_wannier.cpp:749-777`

**问题描述**:

```cpp
double diff = std::abs(std::arg(evals[m]) - std::fmod(gamma_unwrapped[n], 2.0*M_PI));
diff = std::min(diff, 2.0*M_PI - diff);
```

`gamma_unwrapped[n]` 是退绕后的累积相位（可远大于 2π），但 `fmod(gamma, 2π)` 丢失了累积信息。若两个带的 `fmod` 值接近，匹配可能错误。

**影响**: 错误匹配 → 特征向量 VR 分配给错误的带 → SMO 权重 w_In 对应到错误的 γ_n → per-atom gamma 错误。

**不影响**: 总 Berry phase（Σ_n γ_n）不受影响，因为 γ_unwrapped 本身已正确计算。

**解决方案**: 使用 `gamma_unwrapped` 的最近邻匹配，不需要 fmod：

```cpp
// 替换匹配逻辑:
for (int n = 0; n < n_dim; ++n) {
    double best_diff = 1e10;
    int best_m = -1;
    for (int m = 0; m < n_dim; ++m) {
        if (matched[m]) continue;
        double diff = std::abs(std::arg(evals[m]) - gamma_unwrapped[n]);
        // 将 diff 折回到 [0, π]
        diff = std::fmod(diff, 2.0*M_PI);
        if (diff > M_PI) diff = 2.0*M_PI - diff;
        if (diff < best_diff) { best_diff = diff; best_m = m; }
    }
    // ... same as before
}
```

---

## 三、Medium 级风险（下迭代修复）

### M1: BFGS 类名误导——实际是 Fletcher-Reeves CG

**位置**: `module_optimizer/bfgs.h` 全文

**问题描述**: 类名为 `BFGS`，但实现的是 Fletcher-Reeves 共轭梯度法。BFGS 需要维护 n×n 近似逆 Hessian 矩阵，代码中完全没有。

**影响**: 开发者误以为使用了 BFGS 的超线性收敛性质，实际只有 CG 的线性收敛。文档和注释也会误导。

**解决方案**:
1. 重命名类为 `FletcherReevesCG` 或 `ConjugateGradient`
2. 更新所有引用（`deltap.h:263`，`esolver_ks_lcao.cpp` 等）
3. 若需要真正的 BFGS 收敛，需要实现 L-BFGS（有限内存 BFGS）

---

### M2: Lambda mixing 导致 BFGS 内部状态不一致

**位置**: `esolver_ks_lcao.cpp:676-681`

**问题描述**:

BFGS 维护的 `dnu_` 是未 mixing 的累积步长。Mixing 后实际 λ 为 `λ_old + β·dnu`，但 BFGS 不知道。CG 方向基于错误的位移历史。

**数学分析**:

有效优化问题变为：
```
min_λ |r(λ_old + β(λ - λ_old))|²
```
与原始问题有相同极小值，但景观被 β 压缩。CG 的共轭性被破坏。

**解决方案**:

方案 A: 将 mixing 纳入 BFGS 内部。在 `get_lambda()` 中返回 mixed 值，在 `accept_trial()` 中用 mixed 值更新状态。

方案 B: 移除 lambda mixing，依赖 cooldown 和电荷 mixing 来稳定。

---

### M3: 伪梯度方向缺乏 Jacobian

**位置**: `module_optimizer/bfgs.h:180-195`

**问题描述**:

搜索方向 `d_k = r_k + β·d_{k-1}` 使用 residual `r = target - γ` 作为"梯度"。但真正的目标函数是 `F(λ) = |r(λ)|²`，梯度为：
```
∇F = -2 J^T r
```
其中 `J = ∂γ/∂λ` 是 Jacobian。代码没有计算 J，直接使用 r 作为搜索方向。

**当 J ≈ -cI（c > 0）时**：搜索方向是 ascent 方向而非 descent 方向。secant 线搜索会发现 α_opt < 0（负步长），自动纠正。但这意味着 CG 的共轭性完全失效。

**解决方案**:

计算数值 Jacobian。在内循环中，对每个原子 i 做有限差分：
```
J_{n,i} ≈ (γ_i(λ + δ_i) - γ_i(λ)) / δ
```

代价：每步多 n_atom 次 gamma 评估。对于小 nat（< 10），可接受。

替代方案：使用 Broyden 方法（维护 Jacobian 的低秩近似），这是约束 DFT 的标准方法。

---

### M4: Newton-Schulz 收敛条件不保证

**位置**: `deltap_wannier.cpp:570-627`

**问题描述**:

Newton-Schulz 迭代 `X_{k+1} = X_k·(3I - X_k†X_k)/2` 收敛要求 X_0 的奇异值满足 `σ_max²/σ_avg² < 2`（即 `||I - X_0†X_0|| < 1`）。

对于粗 k-mesh 或秩亏损矩阵，σ_max 可能远大于 σ_avg，导致发散。

**当前缓解**: 代码用 `scale = 1/sqrt(frob2/n_dim)` 归一化。但这对极端条件数无效。

**解决方案**:

1. 在迭代前检查条件数（通过 `frob2` 和 `max|X_ij|` 估计）
2. 若条件数过大，fallback 到 LAPACK `zgesvd` 计算 SVD polar factor `U·V†`
3. 添加收敛检查：若 20 次迭代后 `||X†X - I|| > tol`，输出警告并 fallback

```cpp
if (err > tol) {
    // Fallback to SVD
    // M = U Σ V† → polar factor = U V†
    zgesvd_('A', 'A', &n_dim, &n_dim, tmp.data(), ...);
    // UV† = Σ_k U[:,k] · V[:,k]†
    ...
}
```

---

### M5: 诊断检验公式错误

**位置**: `deltap_wannier.cpp:873-896`

**问题描述**:

代码检验 `S · Sinv ≈ I`（line 873），但 `Sinv = S^{-1/2}`（不是 `S^{-1}`），所以 `S · S^{-1/2} = S^{1/2} ≠ I`。

同样，line 885 检验 `S · Sinv · S ≈ S`，实际计算 `S^{1/2} · S ≠ S`。

**影响**: 诊断输出总是报告非零误差，误导用户认为 Löwdin 正交化有问题。

**解决方案**: 修正检验公式：
```cpp
// Line 873-883: 应检验 Sinv · Sinv · S ≈ I (已在 line 898-911 正确实现)
// 删除 line 873-896 的错误检验，只保留 line 898-911 的正确检验
```

---

### M6: select_branch_set 函数为死代码

**位置**: `deltap_wannier.cpp:1399-1474`, `deltap.h:171`

**问题描述**: `select_branch_set` 函数已实现但从未被调用。内联的分支选择逻辑（lines 1023-1052）使用了不同的（错误的）间距。

**解决方案**: 删除死代码，或将内联逻辑替换为 `select_branch_set` 调用（见 H3 的解决方案）。

---

## 四、Low 级风险（维护性改进）

### L1: 特征值追踪在简并点可能失败

**位置**: `deltap_wannier.cpp:682-718`

**问题描述**: 贪心最近邻匹配在两个特征值相位接近时可能错误交换。

**解决方案**: 使用匈牙利算法求全局最优匹配。对于 n_dim ≤ 20（典型值），开销可忽略。

---

### L2: 自适应步长的系统性偏差

**位置**: `module_optimizer/bfgs.h:255-257`

**问题描述**: `g = 1.5 · |α_opt|/α_trial` 中的因子 1.5 引入系统性上偏。几何平均分析表明，即使 |α_opt| 在 α_trial 附近振荡，α_trial 也会持续增长直到被 max_step 截断。

**解决方案**: 将 1.5 改为 1.0，或使用 `g = |α_opt|/α_trial` 的几何平均而非算术平均来驱动适应。

---

### L3: SMO 权重注释声称 sum = 1 但实际 ≤ 1

**位置**: `deltap_wannier.cpp:857`

**问题描述**: 注释 "satisfies sum_I w_In = 1" 仅在 SMO 基组完备时成立。

**解决方案**: 修改注释为 "satisfies sum_I w_In ≤ 1 (= 1 if SMO basis is complete)"。

---

### L4: HK 修正中 w_eff 预条件器缺乏理论推导

**位置**: `deltap_wannier.cpp:1208-1265`

**问题描述**: `w_eff[n] = Σ_I λ^I · w_In / S_I` 中的预条件器 `S_I = avg_n(Σ_{lm} |D_I[lm][n]|²)` 是经验性的，不是从 ∂L/∂ψ = 0 推导的。

**影响**: 改变了不同原子间约束力的相对大小。物理上合理（平衡不同原子的灵敏度），但缺乏理论保证。

**解决方案**: 文档化这是预条件器，不是真实梯度。考虑实现 Broyden 方法替代。

---

## 五、已验证正确的部分

| 组件 | 验证结论 |
|------|---------|
| Wilson loop 矩阵乘法顺序 | ✓ 正确：W_j = W_{j-1} · O_j，column-major 索引正确 |
| Newton-Schulz 迭代公式 | ✓ 正确：X·(3I - X†X)/2 收敛到最近酉矩阵 UV† |
| Löwdin 正交化 S^{-1/2} 计算 | ✓ 正确：dsyev → V·D^{-1/2}·V^T，矩阵乘法顺序正确 |
| SMO 权重的 Löwdin 投影 | ✓ 正确：tilde_proj = S^{-1/2} · proj |
| D_I 的 MPI Allreduce | ✓ 正确：D_I 大小一致（nbands），局部行索引一致 |
| S_k 局部索引 | ✓ 正确：global2local_row 一致 |
| HK 的厄米对称化 | ✓ 正确：H_sym = (M + M†)/2 保证厄米性 |
| k-string 索引无重叠 | ✓ 正确：不同 string 的 k 点索引不重叠 |
| Secant 线搜索公式 | ✓ 正确：最小化线性化的 |r(α)|² |

---

## 修复优先级总表

| ID | 风险 | 级别 | 修复难度 | 建议时间线 |
|----|------|------|---------|-----------|
| C1 | S_dk 不一致 | Critical | 中 | 立即 |
| C2 | Zeta rescaling 错误 | Critical | 低 | 立即 |
| C3 | psi-lambda 不一致 | Critical | 低 | 立即 |
| H1 | SMO 矩阵非对称 | High | 低 | 本周 |
| H2 | D_I 共轭 | High | 低（需验证约定） | 本周 |
| H3 | 分支选择间距 | High | 中 | 本周 |
| H4 | Gauge phase_corrections_ 死代码 | High | 低 | 本周 |
| H5 | 特征值重排 fmod | High | 低 | 本周 |
| M1 | BFGS 命名 | Medium | 低 | 下周 |
| M2 | Lambda mixing 状态不一致 | Medium | 中 | 下周 |
| M3 | 伪梯度缺 Jacobian | Medium | 高 | 下周 |
| M4 | NS 收敛条件 | Medium | 中 | 下周 |
| M5 | 诊断公式错误 | Medium | 低 | 下周 |
| M6 | select_branch_set 死代码 | Medium | 低 | 下周 |
| L1 | 特征值简并 | Low | 中 | 后续 |
| L2 | 步长偏差 | Low | 低 | 后续 |
| L3 | 注释错误 | Low | 低 | 后续 |
| L4 | 预条件器理论 | Low | 高 | 后续 |
