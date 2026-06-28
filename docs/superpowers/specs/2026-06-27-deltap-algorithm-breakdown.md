# DeltaP 算法拆解与对比测试设计

> **日期**: 2026-06-27
> **目标**: 将 DeltaP Wilson loop 算法拆解为独立可验证环节，设计每个环节的对比测试

---

## 1. 算法数据流拆解

### berry_phase 的数据流（参考实现）

```
输入: ψ(k), S(k,k'), kv
  ↓
[环节 A] 构建重叠矩阵 O_j = C†(k_j) · M_j · C(k_{j+1})
  - M_j: prepare_midmatrix_pblas
    - phase = 2π(k_R·R_cart - dk·τ)
    - overlap = ⟨φ|φ(R)⟩ - i·dk·tpiba·⟨φ|r_local|φ(R)⟩
  - O_j = C†(k_L) · M · C(k_R) via pzgemm
  ↓
[环节 B] 计算 zeta_string = ∏_j det(O_j)
  - det via LU decomposition
  - zeta = complex number (模 = 1 if O_j is unitary)
  ↓
[环节 C] 多 k-string 平均 + unwrap
  - cave = Σ wistring · exp(i·zeta_string) / Σ wistring
  - theta0 = atan2(cave.imag(), cave.real())
  - phik = (theta0 + atan2((cphik/cave).imag(), (cphik/cave).real())) / (2π)
  - phik_ave = Σ wistring · phik
  ↓
[环节 D] 自旋因子
  - pdl_elec = 2 × phik_ave (nspin=1)
  - pdl_elec = phik_ave (nspin=2/4)
  ↓
输出: pdl_elec (reduced phase = γ/(2π))
```

### DeltaP 的数据流（当前实现）

```
输入: ψ(k), S(dk), D_I(k), kv
  ↓
[环节 A'] 构建重叠矩阵 O_j = C†(k_j) · S_dk · C(k_{j+1})
  - S_dk: compute_S_dk_link
    - phase = 2π(kvec_c·R_cart - dk_c·τ)
    - overlap = ⟨φ|φ(R)⟩ - i·dk·tpiba·⟨φ|r_local|φ(R)⟩  (via get_psi_r_psi)
  - O_j = C†(k_j) · S_dk · C(k_{j+1}) via pzgemm
  ↓
[环节 B'] 构建 Wilson loop W = ∏_j O_j (归一化)
  - 矩阵乘法，每步除以 max|element|
  ↓
[环节 C'] 对角化 W → λ_n, |v_n⟩
  - zgeev
  - γ_n = arg(λ_n), γ_total = arg(det(W))
  ↓
[环节 D'] 逐原子权重 + Berry phase
  - w^I_n = Σ_{a∈I} |⟨v_n|α_a⟩|²
  - γ^I = Σ_n w^I_n · arg(λ_n)
  - γ^I_avg = Σ_strings γ^I / N_strings
  ↓
[环节 E'] 自旋因子 + 极化
  - P^I = -0.5 × (a/2πΩ) × γ^I_avg
  ↓
输出: P^I (per-atom), P_total = Σ P^I
```

### 环节对应关系

| berry_phase 环节 | DeltaP 环节 | 差异 |
|-----------------|-------------|------|
| A: 构建 O_j | A': 构建 O_j | **应一致** (相同 phase + position correction) |
| B: zeta = ∏ det(O_j) | B'+C': W = ∏ O_j, det(W) | **数学等价**: det(∏O_j) = ∏det(O_j) |
| C: unwrap 平均 | C': 简单 arg 平均 | **不一致**: DeltaP 缺少 unwrap |
| D: 自旋因子 | E': -0.5 因子 | **已修正** |
| — | D': 逐原子分解 | **DeltaP 独有**: berry_phase 无逐原子 |

---

## 2. 逐环节验证方案

### 环节 A vs A': 重叠矩阵一致性

**验证**: 对同一条 k-string, 同一个 link j, 比较 det(O_j)^berry vs det(O_j)^DeltaP

**方法**:
1. 在 berry_phase 的 `stringPhase` 函数中添加 debug 输出: 每个 link 的 det(O_j)
2. 在 DeltaP 的 O_kpair 循环中添加 debug 输出: 每个 link 的 det(O_j)
3. 对比

**预期**: 如果 phase + position correction 一致，det(O_j) 应该相同。

**问题**: berry_phase 用 `det_berryphase` (返回 det, 不返回 O_j 矩阵)。DeltaP 用 `compute_S_dk_link` (返回 S_dk, 再 pzgemm 得 O_j)。

**简化验证**: 直接对比 zeta_string = ∏ det(O_j)（环节 B vs B'）。

### 环节 B vs B': zeta 一致性

**验证**: 对比每条 k-string 的 zeta = ∏_j det(O_j)

**berry_phase**: `stringPhase` 返回 `log(zeta).imag()` = Im(ln(zeta)) = arg(zeta)
**DeltaP**: `arg(det(W))` = arg(∏_j det(O_j)) = arg(zeta)

**数学上等价**，但数值上可能不同:
- berry_phase: 逐 link 计算 det(O_j), 乘积累积 zeta
- DeltaP: 矩阵乘法累积 W = ∏ O_j, 再计算 det(W)

**验证方法**: 在 DeltaP 中添加 zeta_string 的输出，与 berry_phase 的 stringPhase 对比。

### 环节 C vs C': 平均方法

**验证**: 对比平均后的 γ_total

**berry_phase**:
```
cave = Σ exp(i·γ_string) / N
θ₀ = arg(cave)
γ_unwrap_string = θ₀ + arg(exp(i·γ_string) / cave)
γ_total = Σ γ_unwrap_string / N
```

**DeltaP**:
```
γ_total = Σ γ_string / N  (简单平均)
```

**差异**: 当某个 γ_string 接近 ±π 时, arg 跳变, 简单平均错误。

**验证方法**: 在 DeltaP 中同时输出简单平均和 berry_phase 式 unwrap 平均，对比。

### 环节 D': 逐原子分解

**验证**: Σ_I γ^I = γ_total (sum rule)

**方法**: 检查 sum rule 是否精确成立。

**预期**: 当 SMO 完备时精确成立, 否则有偏差。

---

## 3. 对比测试设计

### 测试 1: BaTiO3 zeta_string 对比 (环节 B)

**体系**: BaTiO3 ref, 10×10×10, symmetry=-1

**步骤**:
1. 修改 berry_phase 代码: 在 `stringPhase` 中输出每个 string 的 zeta (复数)
2. 修改 DeltaP 代码: 输出每个 string 的 zeta = det(W)
3. 运行 NSCF (berry_phase + deltap 同时)
4. 对比 100 个 string 的 zeta

**预期**: |zeta_berry - zeta_DeltaP| < 1e-10 (如果环节 A 一致)

**优点**: 不涉及 unwrap, 直接对比复数 zeta

### 测试 2: BaTiO3 unwrap vs 简单平均 (环节 C)

**体系**: BaTiO3 ref, 10×10×10

**步骤**:
1. 从测试 1 获得 100 个 zeta_string
2. 用 berry_phase 方法计算 unwrap 平均
3. 用简单平均计算
4. 对比两者与 berry_phase 输出的 elec_phase

**预期**: unwrap 平均 = berry_phase elec_phase (精确), 简单平均有 3% 误差

### 测试 3: 逐原子 sum rule (环节 D')

**体系**: BaTiO3 ref

**步骤**:
1. 从 DeltaP 输出 γ^I (per-atom) 和 γ_total
2. 检查 Σ γ^I = γ_total

**预期**: 精确成立 (mod 2π)

### 测试 4: 逐原子 P^I 对比 (最终目标)

**体系**: BaTiO3 ref + ti_p + ba_p (大位移 0.01)

**步骤**:
1. 从测试 2 获得 unwrap 后的 γ^I
2. 计算 Z*_I = (Ω/δ) × Δγ^I
3. 与 berry_phase Z* 对比

**前提**: 测试 1-3 全部通过

---

## 4. 实施优先级

| 优先级 | 测试 | 预期收益 |
|--------|------|---------|
| **最高** | 测试 1: zeta_string 对比 | 确认环节 A+B 一致性 |
| **高** | 测试 2: unwrap vs 简单平均 | 修复 3% 误差 |
| 中 | 测试 3: sum rule | 确认逐原子分解 |
| 低 | 测试 4: Z* 对比 | 最终验证 (依赖 1-3) |

