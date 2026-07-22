# DeltaP 内循环 — H 增量更新与波函数相位分析

> 日期: 2026-07-17

---

## 一、H 增量更新: 已有机制, 无需 refresh()

### 1.1 DeltaP operator 的增量更新设计

DeltaP operator 内置了增量 HR 更新 (`dspin_lcao.cpp` 同样的模式):

```
dp_hr_done flag 状态机:
  true  → HR 已包含当前 λ 的贡献, 不需要更新
  false → λ 已改变, 下次 contributeHR() 需要添加增量

set_lambda(λ_new):
  lambda_ = λ_new
  dp_hr_done = false      ← 触发增量更新
  // 但不写 lambda_save_ — 保留旧值用于计算 dλ

contributeHR() (下次 diago 时自动调用):
  if (!hr_done && dp_hr_done): return   ← 无变化
  if (!hr_done): ...                      ← 全量重建 (SCF iter start)
  else:                                   ← 增量更新!
    dλ = lambda_ - lambda_save_          ← 仅 λ 的差分
    HR += dλ × tau × pre_hr             ← 只加差分, 不清零 HR
    lambda_save_ = lambda_
    dp_hr_done = true
```

**关键**: HR **不清零**, KS 部分保留。只加 `dλ × pre_hr[iat]` 的差分. 这正是内循环需要的——每次 trial λ 只需修改 HK 项的差分, 不需要全量重建 H.

### 1.2 内循环中的 H 更新流程

```
Step A: dp_op->set_lambda(lam_trial)   → dp_hr_done = false
Step B: dp->compute_hk_correction()     → 算 k-dependent HK 修正
Step C: dp_op->set_hk_correction(hk_corr)
Step D: hamilt->updateHk(ik) for each k:
          init(ik):
            contributeHR()  → dλ × pre_hr 追加到 HR
            contributeHk()  → hk_correction_ 加到 Hk
            base::contributeHk() → HR fold → Hk
Step E: hsolver.solve(H, ψ, ...)  ← 对角化用更新后的 H
```

**当前 Segfault 原因不在于 H 更新, 在于 solver 的重入问题** (见 §2).

### 1.3 与 DeltaSpin 的对比

| | DeltaSpin | DeltaP |
|---|---|---|
| 增量 flag | `sc_hr_done` (独立于 `hr_done`) | `dp_hr_done` |
| 增量计算 | `dλ × coeff × pre_hr` | `dλ × tau × pre_hr` |
| λ 更新 | `update_lambda()` → `sc_hr_done=false` | `set_lambda()` → `dp_hr_done=false` |
| k-dependent 项 | 无 (只有 HR) | 有 (`contributeHk` 添加 Berry connection 修正) |

**DeltaP 不需要 `sc_hr_done` 的独立 flag**——因为它不在 `refresh(true)` 被跳过的情况下运行 (内循环中 `refresh(false)` 保持 `hr_done=true`, 增量正确生效)。

---

## 二、波函数相位不确定性与分支一致性

### 2.1 相位来自哪里

每次对角化 `H·ψ = ε·S·ψ` (广义本征值), 任意本征矢 ψ_n 可以乘任意相位 e^{iθ}:

```
ψ'_n = e^{iθ_n} × ψ_n     (H 不变, ε 不变)
```

LAPACK (zgeev/zhegv) 返回的 ψ 有一致的内部归一化, 但**不同对角化之间相位不可控**.

### 2.2 相位影响 C†SC (O_j 矩阵)

```
O_j[n,m] = Σ_{μ,ν} conj(C_{μ,n}(k_L)) × S_{μ,ν}(dk) × C_{ν,m}(k_R)

若 C'_{μ,n}(k_L) = e^{iθ_n} × C_{μ,n}(k_L):
    O'_j[n,m] = e^{-iθ_n} × O_j[n,m] × e^{iθ_m}

结果: O_j 有任意左乘右乘的相位矩阵 (对角矩阵 diag(e^{iθ}))
```

**关键**: O_j 不是对角不变的, 但 W = O_0·O_1·... 的行列式 det(W) 是不变的:

```
det(O'_j) = det(e^{-iθ} × O_j × e^{iθ'}) = e^{iΣ(θ'-θ)} × det(O_j)
          = det(O_j)    (当 k_L 的 θ 被下个 j 的 θ' 抵消时)
```

对于完整的闭合 Wilson loop: det(W_final) = Π_j det(O_j) **规范不变**.

### 2.3 相位如何影响特征值追踪

问题在**单个** k-point pair, 不在闭合 loop:

```
W_j 对角化: evals_j 的 arg 受 ψ 相位影响

第一次对角化 (first_diag):
  → 按 arg 排序 → 初始 gamma_unwrapped[n] 取决于排序结果
  → 如果 ψ 相位使 arg(evals[0]) ≈ 0.1 和 0.2 交换,
     排序后 n=0 和 n=1 的对应物理带不同

后续对角化 (Hungarian):
  → cost 矩阵: |arg(evals_new[n]/evals_prev[m])|
  → 若 evals 的相位排列因 ψ 相位而不同, matching 可能错
```

### 2.4 现有保护机制

| 层级 | 机制 | 对相位的防护 |
|:---:|------|------|
| L0 | (未实现) 带交叉检测 | 检测 cost 接近二义 |
| L1 | Hungarian + deltap_match.dat | **跨 run 匹配一致** (冻结排列) |
| L2 | ref_gamma_unw_sum 固定 | zeta scale 跨 string 一致 |
| L3 | prev_gamma = target (0 或 W_prev_3d) | per-atom 分支锚定 |

### 2.5 内循环中相位的影响

内循环每步 solve 产生新的 ψ → 新的 ψ 相位:

```
inner=0: ψ 相位 {θ_0^0, θ_1^0, ...}
inner=1: ψ 相位 {θ_0^1, θ_1^1, ...}  (不同!)

O_j 矩阵因相位不同 → W_j evals 不同 → arg(evals) 排列可能不同

若 L1 match 已加载: → 相同的 match_to[] → gamma_unwrapped 一致 ✅
若 L1 match 未加载: → Hungarian 重新计算 → 匹配可能不同 → γ 跳变
```

**结论**: L1 match freeze (`deltap_match.dat`) 是内循环正确运行的前提。首次 outer SCF iter 保存 match, 后续 inner loop 复用.

### 2.6 为什么 DeltaSpin 不受影响

DeltaSpin 的约束量是 M = Tr(ρ·σ), 由密度矩阵的迹给出. 密度矩阵 D = Σ_n f_n·ψ_n·ψ†_n — 这里 ψ×ψ† 消除了相位 (e^{iθ}×e^{-iθ}=1). 所以 **DeltaSpin 可观测量天然规范不变, 不需要 eigenvalue 匹配**.

DeltaP 的约束量是 Wilson loop 的 per-atom 分解, 需要追踪每个带的相位. 因此 **DeltaP 比 DeltaSpin 多了一个 "eigenvalue ordering" 层次**.

---

## 三、内循环 segfault 的真正原因

排除了 H 更新问题后, 剩余可能:

1. **`compute_gamma_scf` 调了 `psi->fix_k()`**: 改变了 psi 的内部状态. 后续 `hsolver.solve()` 假设 psi 在某特定状态. → **修复**: solve 前调用 `psi->fix_k(0)` 重置.

2. **solver 对象状态**: `HSolverLCAO` 的 `solve()` 方法内部分配临时矩阵, 可能假设第一次调用时 `pelec->f_en.eband = 0` (line 410 in hamilt2rho_single). 第二次调用时 eband 非零. → **修复**: solve 前重置 `this->pelec->f_en.eband = 0`.

3. **Norm-conserving condition**: `HSolverLCAO.solve()` 内可能调用 `psi->fix_k()` 并假设所有 k 点的 ψ 可以覆盖写入. 如果 psi 的内部数组在上次 solve 后被破坏, segfault.

### 建议调试步骤

```
1. 内循环 nscf=1, 但 BEFORE solve 添加:
   this->pelec->f_en.eband = 0.0;
   this->pelec->f_en.demet = 0.0;
   
2. 如果步骤 1 仍然 segfault, 将内循环的 solve 改为:
   // 创建新的 hsolver 对象 (每次内循环重建)
   hsolver::HSolverLCAO<TK> hsolver_fresh(&(this->pv), PARAM.inp.ks_solver);
   hsolver_fresh.solve(...);
   
3. 如果步骤 2 解决, segfault 确认来自 solver 对象状态重用.
```
