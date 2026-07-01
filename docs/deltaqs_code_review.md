# DeltaQS 代码审查报告

**日期**: 2026-07-01  
**审查范围**: DeltaQS 电荷约束 DFT 方法的完整实现  
**审查重点**: 方案一致性、正确性、性能

---

## 1. 总体架构

```
STRU (tc, cq, mu)
     │
     v
init_deltaqs() ── 存储 mu_, target_charge_, Ni_, constrain_charge_
     │
     v
run_qs_lambda_loop() ── 统一 CG 优化 (lambda + mu)
  │
  ├── apply_and_solve():
  │     1. 设置 lambda_[iat], mu_[iat]
  │     2. dspin_op->update_lambda()
  │     3. HSolverLCAO::solve() ← 全对角化
  │     4. cal_mi_lcao() + cal_ni_lcao()
  │
  └── compute_residual_and_rms():
        RMS = sqrt( sum((Mi-M_target)² + (Ni-N_target)²) / n_active )

contributeHR() ── 向实空间 H 添加 H_DS = mu*I + lambda·sigma
  │
  ├── cal_coeff_lambda_qs():
  │     nspin=2: {(mu+λz), (mu-λz)}
  │     nspin=4: {mu+λz, λx-iλy, λx+iλy, mu-λz}
  │
  └── dHR += coeff × pre_hr  (first-zeta 投影算符)

cal_escon() = -Σ(λ·M) - Σ(μ·N)  ──→  加入总能量 f_en.escon
```

---

## 2. 方案一致性审查

### 2.1 投影算符 P_I

| 组件 | 使用的投影算符 | 来源 |
|------|-------------|------|
| `cal_pre_HR()` | first-zeta: 每个角动量通道 l 只取第1个 zeta | `dspin_lcao.cpp:388-403` |
| `contributeHR()` | 同上 (`pre_hr[iat]`) | `dspin_lcao.cpp:171-205` |
| `cal_ni_lcao()` | 同上 (复用 DeltaSpin 的 `cal_moment()`) | `deltaqs.cpp:182-187` |
| `cal_moment()` | 同上 | `dspin_lcao.cpp:506-549` |
| `cal_escon()` | 使用 Mi, Ni（均基于上述投影） | `spin_constrain.cpp:38-61` |

**结论**: 所有组件使用**同一个** first-zeta 投影算符 P_I = Σ_{lm} |α_{lm}^(1)><α_{lm}^(1)|。✅ 一致。

### 2.2 哈密顿量 ↔ 能量修正 ↔ 电荷/磁矩计算

约束泛函:
$$E'[\rho] = E[\rho] + \sum_I \lambda_I \cdot (M_I - M_I^{target}) + \sum_I \mu_I (N_I - N_I^{target})$$

哈密顿量修正:
$$H_{DS} = \sum_I (\lambda_I \cdot \sigma + \mu_I \cdot I_{spin}) \otimes P_I$$

| 量 | 公式 | 代码 | 一致性 |
|----|------|------|--------|
| H 贡献 | H += (μ + λz) P_I (↑), (μ - λz) P_I (↓) | `cal_coeff_lambda_qs` | ✅ |
| 能带贡献 | ΔE_band = Σ(λ·M + μ·N) | 隐含在对角化中 | ✅ |
| 能量修正 | E_scon = -Σ(λ·M) - Σ(μ·N) | `cal_escon()` | ✅ |
| E_DFT | E_DFT = E_KS + E_scon | `fp_energy.cpp:19` | ✅ |
| 磁矩 | M_I = Tr(ρ_spin · P_I) | `cal_moment(dmr_diff)` | ✅ |
| 电荷 | N_I = Tr(ρ_total · P_I) | `cal_ni_lcao(dmr_total)` | ✅ |

**结论**: 哈密顿量、能量修正、电荷/磁矩计算三者完全一致。✅

### 2.3 nspin=2 与 nspin=4 的一致性

| | nspin=2 | nspin=4 |
|--|---------|---------|
| H_DS | (μ+λz)P↑, (μ-λz)P↓ | 2×2 自旋矩阵 × P_I |
| M_I | Tr(ρ_diff · P_I) | Tr(σ · ρ · P_I) |
| N_I | Tr(ρ_total · P_I) via switch_dmr(1) | Σ 对角元 × P_I (手动循环) |
| E_scon | -λz·Mz - μ·N | -λ·M - μ·N |

**结论**: nspin=2 和 nspin=4 路径物理等价，实现一致。✅

---

## 3. 投影轨道选择

### 3.1 当前方案: First-Zeta

每个角动量通道 l 只取第1个 zeta 轨道:
- 投影算符: P_I = Σ_{l=0}^{l_max} Σ_{m=-l}^{l} |α_{lm}^{(1)}><α_{lm}^{(1)}|
- 维度: (l_max + 1)² 个投影函数/原子
- 对 Fe (l_max=3): 16 个投影函数

**优点**:
- 近似正交: <α_{lm}^(1)|α_{l'm'}^(1)> ≈ δ_{ll'}δ_{mm'}
- 满足幂等性: P_I² ≈ P_I
- 物理意义清晰: 捕获局域价电子的核心部分

**缺点**:
- 不完备: ΣN_I < N_total (捕获约 60-90% 电荷)
- 间隙电荷不受约束控制

### 3.2 CSZ 方案 (已弃用)

`module_deltaqs/` 中实现了 Complete Single-Zeta 投影:
- 使用所有 n_zeta 个 zeta 轨道/每个 l
- 对 Fe (4s2p2d1f): 27 个投影函数
- **问题**: 同 l 不同 zeta 不正交 → P_I² ≠ P_I → 过完备

**当前状态**: CSZ 代码完整实现但**未接入**主流程。`cal_ni_lcao()` 使用 first-zeta。

**结论**: First-zeta 是正确选择。CSZ 正确地未被使用。✅

---

## 4. 发现的问题

### 4.1 [BUG] 能量修正门控条件缺失纯电荷模式

**文件**: `source/source_estate/elecstate_energy.cpp:330`

```cpp
if (PARAM.inp.sc_mag_switch)  // ← 只检查 sc_mag_switch
{
    this->f_en.escon = get_spin_constrain_energy();
}
```

**问题**: 当 `sc_charge_switch = true` 但 `sc_mag_switch = false`（纯 DeltaQ 模式）时，`E_scon` **不会被加入总能量**。

**影响**:
- E_DFT 缺少 -Σ(μ·N) 项
- 总能量不正确
- E(δN) 扫描的热力学一致性被破坏
- μ = dE/dN 关系不成立

**修复**:
```cpp
if (PARAM.inp.sc_mag_switch || PARAM.inp.sc_charge_switch)
{
    this->f_en.escon = get_spin_constrain_energy();
}
```

**严重度**: P0 — 纯 DeltaQ 模式计算结果全部错误。

**状态**: ✅ 已修复 (2026-07-01)

### 4.2 [PERF] QS 内循环无子空间加速

**文件**: `source/source_lcao/module_deltaspin/deltaqs.cpp:509-539`

**现状**: `apply_and_solve()` 在每次 CG 迭代中执行完整的 `HSolverLCAO::solve()` (全对角化)。

| 操作 | DeltaQS (当前) | DeltaSpin (有加速) |
|------|---------------|-------------------|
| 每步对角化 | 全对角化 O(N³) | 子空间 O(n_bands³) |
| 每步 solve 次数 | 2 (试步 + 线搜索) | 0-1 (子空间不需要) |
| nsc=40 总 solve | ~81 次全对角化 | ~5 次全对角化 + 子空间操作 |

**性能差距**: 约 15-20x (取决于 N/n_bands 比值)

**根因**: `run_qs_lambda_loop()` 未接入 `DiagonalizationEngine` 框架:
- 无 `SubspaceDiagonalizer` (子空间对角化)
- 无 `FirstOrderResponseEngine` (一阶响应)
- 无 `cal_mw_from_lambda()` (子空间缓存管理)

**修复方向**: 将 `apply_and_solve()` 替换为 `cal_mw_from_lambda()` 路径，复用 DeltaSpin 的子空间加速基础设施。需要将电荷残差计算集成到 `DiagonalizationEngine` 中。

**严重度**: P1 — 功能正确但性能不可接受。

### 4.3 [BUG] nspin=4 时 apply_and_solve 未调用 update_lambda()

**文件**: `deltaqs.cpp:524-526`

```cpp
auto* dspin_op = dynamic_cast<hamilt::DeltaSpin<hamilt::OperatorLCAO<std::complex<double>, double>>*>(
    this->p_operator);
if (dspin_op) dspin_op->update_lambda();
```

**问题**: nspin=4 时算符类型是 `DeltaSpin<OperatorLCAO<complex<double>, complex<double>>>`，此 `dynamic_cast` 返回 nullptr，导致 `update_lambda()` 从未被调用。H 不包含 DeltaSpin/DeltaQS 修正。

**对比**: `cal_mw_from_lambda` (lines 904-914) 正确 dispatch 了 nspin=2 和 nspin=4。

**严重度**: P0 — nspin=4 + DeltaQS 模式完全不工作。

**修复**: 此问题在 P1（统一内循环）中一并解决——将 `apply_and_solve` 替换为 `cal_mw_from_lambda` 后自然修复。

### 4.4 [MINOR] cal_charge_escon() 未被使用

**文件**: `deltaqs.cpp:305-316`

`cal_charge_escon()` 是一个独立的电荷能量修正函数，但从未被调用。`cal_escon()` 已经包含了电荷项。

**影响**: 死代码，无功能影响。

---

## 5. 性能分析（修正版）

### 5.1 DeltaSpin 内循环架构

`run_lambda_loop()` 调用 `cal_mw_from_lambda()`，后者支持**三种对角化模式**：

| 模式 | 分支 | 方法 | 计算量 | 使用条件 |
|------|------|------|--------|---------|
| 全空间 | Branch 3 | `HSolverLCAO::solve()` | O(N³) | 默认 / 早期迭代 |
| 子空间 | Branch 2b | `diag_hegvd()` in nbands×nbands | O(n_bands³) | RMS < 阈值后 |
| 一阶响应 | Branch 2a | 本征值平移，无对角化 | O(n_bands) | nspin=2, RMS 很小 |

子空间加速流程：
1. RMS 低于 `sc_acceleration_rms_thr` 时激活
2. 调用 `cal_mw_from_lambda(-2)` 构建子空间缓存 (H_sub, S_sub, P_I_sub)
3. 后续迭代使用子空间对角化，避免 O(N³) 全对角化

### 5.2 DeltaQS 内循环架构

`run_qs_lambda_loop()` 使用自己的 `apply_and_solve` lambda，**只有一种对角化模式**：

```cpp
auto apply_and_solve = [&](int step) {
    // 1. 设置 lambda 和 mu
    // 2. update_lambda() ← 只处理 nspin=2！(nspin=4 的 dynamic_cast 失败)
    // 3. HSolverLCAO::solve() ← 始终全对角化
    // 4. cal_mi_lcao() + cal_ni_lcao()
};
```

### 5.3 两套内循环的对比

| | `cal_mw_from_lambda` (DeltaSpin) | `apply_and_solve` (DeltaQS) |
|--|--------------------------------|---------------------------|
| 对角化方式 | 全空间 / 子空间 / 一阶响应 | 仅全空间 |
| 电荷约束 (μ) | ❌ 无 | ✅ 有 |
| 电荷观测量 (Ni) | ❌ 不调用 `cal_ni_lcao` | ✅ 调用 |
| nspin=4 | ✅ 正确 dispatch | ❌ `update_lambda()` 未调用 |
| PW 路径 | ✅ 有 | ❌ 无 |
| 子空间缓存 | ✅ H_sub, S_sub, P_I_sub | ❌ 无 |

### 5.4 性能差距根因

**不是算法设计问题，而是实现分裂问题。**

`cal_mw_from_lambda` 原本是为纯 DeltaSpin 设计的，不包含任何电荷约束逻辑。当开发 DeltaQS 时，由于需要同时处理 μ 和 Ni，开发者没有扩展 `cal_mw_from_lambda`，而是新写了一个 `apply_and_solve` lambda。这导致：

1. **子空间加速未移植**：DeltaQS 每步都是 O(N³) 全对角化
2. **nspin=4 bug**：`apply_and_solve` 中 `dynamic_cast<DeltaSpin<OperatorLCAO<complex, double>>*>` 对 nspin=4 返回 nullptr
3. **CG 算法差异**：DeltaSpin 用 Polak-Ribiere + `cal_alpha_opt` 线搜索；DeltaQS 用简化版 CG + `alpha_factor` 线性插值

### 5.5 性能差距量化

| 组件 | DeltaSpin (有加速) | DeltaQS (当前) |
|------|-------------------|---------------|
| 初始全对角化 | 1-2 次 | 1 次 |
| 子空间构建 | 1 次全对角化 | — |
| 每 CG 步 | 子空间对角化 O(n_bands³) | 2 次全对角化 O(N³) |
| nsc=20 总 solve | ~5 次全对角化 + 子空间 | ~41 次全对角化 |
| **总全对角化次数** | **~5** | **~41** |

### 5.6 修复方向

**正确方案**：扩展 `cal_mw_from_lambda` 以支持电荷约束，让 `run_qs_lambda_loop` 复用它。

具体步骤：
1. 在 `cal_mw_from_lambda` 的所有分支中添加 `cal_ni_lcao()` 调用（当 `charge_constraint_enabled_` 时）
2. 在 `cal_mw_from_lambda` 中处理 `mu_` 的设置（与 lambda 同步）
3. 让 `run_qs_lambda_loop` 将 `apply_and_solve` 替换为 `cal_mw_from_lambda`
4. 子空间加速中的 `calculate_delta_hcc_lcao` 需要同时处理 μ 贡献：`H_sub += Σ(λ·P_I_sub + μ·P_I_sub)`

预期效果：DeltaQS 自动获得子空间加速，性能提升 ~10-15x。

---

## 6. 一致性总结

| 审查项 | 状态 | 说明 |
|--------|------|------|
| 投影算符统一 | ✅ | H, N, M, E_scon 均用 first-zeta P_I |
| H ↔ E_scon 对应 | ✅ | H += μP + λσP, E_scon = -μN - λM |
| nspin=2/4 等价 | ⚠️ | `cal_mw_from_lambda` 正确；`apply_and_solve` 中 nspin=4 的 `update_lambda()` 未调用 |
| first-zeta 选择 | ✅ | 近似幂等，物理合理 |
| CSZ 弃用 | ✅ | 正确地未接入主流程 |
| 能量门控 | ✅ 已修复 | 纯 DeltaQ 模式 E_scon 现已正确加入总能量 |
| 子空间加速 | ❌ | 实现分裂：QS 用 `apply_and_solve`（全对角化），DS 用 `cal_mw_from_lambda`（支持子空间）。需统一入口 |
| 热力学一致性 | ⚠️ | 依赖能量门控修复后才能验证 |

---

## 7. 修复建议（按优先级）

### P0: 能量门控修复
```cpp
// elecstate_energy.cpp:330
- if (PARAM.inp.sc_mag_switch)
+ if (PARAM.inp.sc_mag_switch || PARAM.inp.sc_charge_switch)
```

### P1: 统一内循环（核心修复）

**问题本质**：`run_qs_lambda_loop` 和 `run_lambda_loop` 使用了两套独立的内循环实现（`apply_and_solve` vs `cal_mw_from_lambda`），导致 DeltaQS 无法复用 DeltaSpin 的子空间加速。

**修复方案**：扩展 `cal_mw_from_lambda` 支持电荷约束，让两套循环共用同一个对角化入口。

步骤：
1. `cal_mw_from_lambda` 中添加 `mu_` 设置和 `cal_ni_lcao()` 调用
2. 子空间分支中 `calculate_delta_hcc_lcao` 同时处理 μ 贡献
3. `run_qs_lambda_loop` 将 `apply_and_solve` 替换为 `cal_mw_from_lambda`
4. 修复 nspin=4 的 `update_lambda()` dispatch（当前 `apply_and_solve` 中缺失）

### P2: CG 改进
- RMS 连续上升时 beta → 0 (重启)
- spin 和 charge 使用独立的 Polak-Ribiere beta
- Armijo 线搜索替代简单线性插值
