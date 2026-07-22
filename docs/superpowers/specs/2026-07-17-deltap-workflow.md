# DeltaP 算法工作流——公式、状态与内存

> 日期: 2026-07-17

---

## 一、总览

```
SCF 迭代循环:
  iter=1..N:
    iter_init()          → 用当前 ρ 构建 H
    hamilt2rho_single()  → 对角化 H → 更新 ρ
    iter_finish()        → DeltaP: 计算极化 → 更新 λ → 修改 H (下一迭代生效)
```

约束极化项的 Hamilton 量:

```
H' = H_KS + H_HK

H_HK(k_j) = M(k_j) + M†(k_j)   (Hermitian 对称化)

M(k_j) = (i/2) × S(dk) × C(k_{j+1}) × W_eff(k_j) × C†(k_j)

W_eff[n] = Σ_I λ_I × w_I_n(k_j)    (per-atom λ × per-band SMO weight)
```

---

## 二、物理量状态表

### 2.1 持久态 (跨 SCF 迭代)

| 物理量 | 声明的函数 | 类型 | 初始化时机 | 更新时机 | 冻结条件 |
|--------|----------|------|-----------|---------|---------|
| **λ** (Lagrange乘子) | `before_scf` | `double[nat]` | `deltap_lambda_init` (默认 0) | `iter_finish` 中梯度下降或 BFGS | cooldown 期间冻结 |
| **hk_correction_** | `DeltaPOperator` | `map<int,vector<complex>>` | 空 | `iter_finish` 中 `compute_hk_correction` | — |
| **W_prev_3d** | `DeltaP` | `Vector3[nat]` | NaN (无历史) → `load_branch()` 读文件 | `iter_finish` 中保存 gamma | 首次运行后 `has_prev_=true` |
| **saved_matches_** | `DeltaP` | `int[alpha][string][j][n]` | 空 → `load_match()` 读文件 | Hungarian 匹配后保存 | — |
| **results_** | `DeltaP` | `AtomicPolarization` | 每 SCF iter 清空 | `compute_wannier_polarization` 末尾 | — |
| **scf_initialized_** | `DeltaP` | `bool` | `false` | SMO 等 init 完成后 `true` | 首次为 false → init 执行 → 后续跳过 |
| **deltap_scf_initialized_** | `ESolver` | `bool` | `false` | `iter_finish` 中 DeltaP 对象创建后 | 首次为 false → 创建 DeltaP |

### 2.2 半持久态 (跨 k-string/方向, 同一 SCF iter 内)

| 物理量 | 作用域 | 初始化 | 清除 | 内存 |
|--------|------|------|------|------|
| **S_dk_** | `DeltaP` 成员, per alpha | 空 → `compute_S_dk()` 填充 | 每次 `S_dk_.clear()` (alpha 切换) | nrow×ncol×16B |
| **kstring_data_[]** | `DeltaP` 成员, per string | `resize(nppstr_)` → clear each iteration | 每次 alpha 循环 resize | nat×nproj×nbands×... ~100KB |
| **prev_gamma** | alpha 循环局部, per alpha | 0.0 或 `W_prev_[alpha]` | 每 alpha 重新初始化 | nat×8B |
| **gamma_accum** | alpha 循环局部, per alpha | 全零 | 每 alpha 重新声明 | nat×8B |
| **n_strings_processed** | alpha 循环局部, per alpha | 0 | 每 alpha 重新声明 | int |

### 2.3 临时态 (per k-string, 同一 SCF iter 内)

| 物理量 | 作用域 | 内存 | 分配时机 |
|--------|------|------|------|
| **psi_k_ptrs** | istring 循环局部 | nppstr × 8B | 每 istring |
| **S_k (临时)** | kstring_data_[j].S_k | nat×nproj×NBASIS×16B | `compute_S_k(j)` 每次 fill |
| **D_I (临时)** | kstring_data_[j].D_I | nat×nproj×nbands×16B | `compute_D_I(j)` 每次 fill |
| **O_kpair[j]** | istring 循环局部 | nppstr × nocc² × 16B | 每 istring 的 O_kpair 循环 |
| **O_full** | j 循环局部 | nocc² × 16B | 每 j |
| **SC (快速路径)** | istring 循环局部 | nrow×nocc×16B | 每 istring |
| **W_mat** | j 循环局部 | nocc² ×16B | 每 istring 初始化 I, j 循环更新 |
| **cost** | Hungarian 匹配 | nocc² × 8B | j 循环中 Hungarian |
| **evals_j, evals_prev** | j 循环局部 | nocc×16B | j 循环中 zgeev |
| **gamma_unwrapped** | j 循环累积 | nocc×8B | 跨 j 循环跟踪 |
| **proj, tilde_proj** | istring 循环局部 | m_dim×nocc×16B | per istring (D_mat × VR) |
| **w_In_matrix** | istring 循环局部 | nocc×nat×8B | per istring (tilde_proj norm) |
| **gamma_I_per_atom** | istring 循环局部 | nat×8B | per istring |

---

## 三、分阶段工作流

### 3.1 初始化阶段

```
ESolver_KS_LCAO::before_scf():
  ├── DeltaPOperator 创建, 添加到 HamiltLCAO
  │   └── lambda_ 初始化为 deltap_lambda_init (默认 0)
  └── 无 DeltaP 对象 (延迟到 iter_finish)

ESolver_KS_LCAO::iter_init():
  └── 构建 H = H_KS + 0 (λ=0, 无 HK 修正)

[SCF iter = 1]
  hamilt2rho_single():
    └── 对角化 H → ψ, 更新 ρ (无 DeltaP)

  iter_finish() — 首次:
    ├── deltap_scf_initialized_ == false:
    │   ├── build_orb_onsite()       ← TwoCenter integrator
    │   ├── r_overlap_scf_ ← new    ← 位置矩阵计算器
    │   ├── berry_ovl_scf_ ← new    ← (已废弃: 用快速路径替代)
    │   ├── dp ← new DeltaP()
    │   ├── dp->init(ucell, gd, kv, intor, ...)
    │   │   ├── gd_ 指针保存
    │   │   ├── kv_ 指针保存
    │   │   └── 无 SMO, 无 k-string (延迟到首次 compute_wannier_polarization)
    │   ├── dp->load_branch()        ← 读 deltap_branch.dat (若存在)
    │   ├── dp->load_match()         ← 读 deltap_match.dat (若存在)
    │   ├── deltap_target_ 初始化    ← 0.0 或从文件读
    │   └── dp->init_inner_loop()    ← nscf_, bfgs_.init()
    │
    ├── dp->compute_gamma_scf()      ← compute_wannier_polarization()
    │   └── (见 §3.3)
    │
    ├── λ 更新: 梯度下降或 BFGS (若 inner_loop_active)
    │   └── dp_op->set_lambda()
    │
    ├── dp->compute_hk_correction()  ← 用新 λ 算 HK → 下一 iter 生效
    └── dp_op->set_hk_correction()
```

### 3.2 公式: HK 修正 (compute_hk_correction)

```
输入: 当前 ψ, λ[0..nat-1]
输出: hk_correction[ik][NBASIS × NBASIS] (column-major Hermitian)

对每条 k-string (istring), 每个 link (j):

  k_L = k_index_[istring][j]
  k_R = k_index_[istring][j+1]

  w_eff[n] = Σ_I λ[I] × Σ_{lm∈I} |D_I(k_L)[lm][n]|² / S_I
    其中 S_I = avg_{n} Σ_{lm} |D_I[lm][n]|²  (per-atom preconditioner)

  SC = S_dk × C(k_R)              (NBASIS × nocc)
  F  = (i/2) × w_eff × SC         (NBASIS × nocc)
  M  = F × C†(k_L)                (NBASIS × NBASIS)

  H_sym = (M + M†) / 2             (Hermitian 对称化)
  hk_correction[k_L] += H_sym
```

**关键**: `half_i = (0, -0.5)` → `F = -(i/2) × w_eff × SC`. 负号驱动 γ 向更负方向 (已验证).

**依赖**: `S_dk_` 必须在调用前已计算 (每个 alpha 方向仅一次).

### 3.3 极化计算 (compute_wannier_polarization)

```
输入: ψ, ρ (通过 pelec 间接), gdir_ 指定的方向
输出: results_.P_I[iat], results_.gamma_I[iat], results_.P_total

阶段 0: SMO 初始化 (首次调用)
  ├── compute_real_overlaps()     ← 重叠积分, O(nat × neighbors)
  ├── setup_kstring()             ← k-string 拓扑 (依赖 gdir_)
  └── compute_smo_overlap_matrix() ← S^{-1/2} (Löwdin)

[alpha loop: α=0,1,2 (x,y,z)]
  gdir_ = α+1; setup_kstring(); S_dk_.clear()

  [istring loop: s=0..total_string_-1]
    ├── 阶段 1a: S_k, D_I (每 k 点)
    │    compute_S_k(j): 对 kstring_data_[j] 填 S_k, dS_k
    │    compute_D_I(j): D_I[lm][n] = Σ_mu conj(S[mu]) × psi[mu][n]
    │
    ├── 阶段 1b: 保存 D_I_all_
    │
    ├── 阶段 2: O_kpair (k-point pair overlap)
    │    for j=0..nppstr_-2:
    │      快速路径: O = C†(k_L) × S_dk × C(k_R)     NBASIS×nocc × 2 GEMM
    │      或: berry_overlap_->berryphase_overlap()    (已废弃)
    │
    ├── 阶段 3a: zeta = Π_j det(O_j)    (zgetrf)
    │
    ├── 阶段 3b: Wilson Loop 构建 + 特征值追踪
    │    for j=0..nppstr_-2:
    │      W_j = W_{j-1} × O_j
    │      diagonalize W_j → evals_j
    │      if first: 按 arg 排序, gamma_unwrapped[n] = arg(evals[n])
    │      else: Hungarian 匹配 → gamma_unwrapped[n] = gamma_prev + diff
    │
    ├── 阶段 3c: 最终对角化 + evals 匹配
    │    zgeev(W_final) → reorder to match gamma_unwrapped
    │
    ├── 阶段 4: Per-atom 投影
    │    D_mat = D_I(k_0)                  (m_dim × nocc)
    │    proj = D_mat × VR                 (D_I × eigenvectors)
    │    tilde_proj = S^{-1/2} × proj     (Löwdin orthogonalization)
    │    w_In[n][iat] = Σ_{a∈I} |tilde_proj[a][n]|²
    │    gamma_I[iat] = Σ_n w_In[n][iat] × gamma_unwrapped[n]
    │
    ├── 阶段 5: Zeta rescaling + 分支选择
    │    ref_gamma_unw_sum = gamma_unw_sum (第一条 string)
    │    scale = ref / gamma_raw_sum
    │    gamma_I *= scale
    │    
    │    for iat:
    │      if |g - prev| < π: no branch correction
    │      else: search ±2π × w_In[n][iat] per band
    │      prev_gamma[iat] = corrected_gamma
    │
    └── n_strings_processed++

  [alpha 循环结束: 累加器只用当前 alpha 的数据]
    gamma_I[iat][alpha] = gamma_accum[iat] / n_strings_processed
    W_prev_[iat][alpha] = gamma_I[iat][alpha]   ← 保存到 deltap_branch.dat

[三方向完成]
  P_total = Σ_iat P_I[iat]
  verify_sum_rule()   ← 打印 (Px, Py, Pz)
  save_branch()       ← W_prev_3d → deltap_branch.dat
  save_match()        ← saved_matches_ → deltap_match.dat
```

### 3.4 分支选择流程

```
per string, per atom:

prev_gamma 初始值:
  SCF iter 1: 0.0 (默认目标极化)
  SCF iter N: W_prev_[iat][alpha] (上一轮 SCF 的 gamma)

第一条 string: 
  使用初始 prev_gamma → 选最接近的 branch

后续 string:
  prev_gamma = 上一条 string 选择的 gamma

分支校正:
  if |gamma_I - prev| < π:
      无需校正
  else:
      for each band n:
          for sign in {-1, +1}:
              candidate = gamma_I + sign × 2π × w_In[n][iat]
          选 |candidate - prev| 最小的

跨方向 (alpha) 隔离:
  每个 alpha 独立 prev_gamma (在 alpha 循环内声明)
```

### 3.5 特征值匹配流程

```
每条 k-string 的每个 k-point pair j>0:

构建 cost 矩阵:
  cost[m][n] = |arg(evals_j[n] / evals_prev[m])|   (wrap ±π)

检查是否有保存的匹配 (load_match):
  if has_saved_match[alpha][istring][j]:
      直接使用保存的 match_to[] (确定性的 λ sweep)
  else:
      Hungarian 全局最优匹配
      保存 match_to 到 saved_matches_

NaN 守卫:
  若 evals_j 或 evals_prev 含 NaN → 贪心回退

展开相位:
  gamma_new[match] = gamma_prev[match] + arg(evals_j[n] / evals_prev[match])
```

---

## 四、关键状态转换

### 4.1 λ 和 H

```
iter=1:
  H = H_KS + 0               ← 无 HK (λ=0)
  iter_finish: γ 计算, λ 更新
  H_HK = f(λ_new)            ← 构建 HK 修正, 存入 operator

iter=2:
  H = H_KS + H_HK(λ_old)     ← 用上一轮 λ 的 HK
  iter_finish: γ 测量 (受 λ_old 影响), λ 可能更新
  H_HK = f(λ_new)            ← 构建新 HK

iter=N:
  H = H_KS + H_HK(λ_{N-1})
  ...
```

**关键**: HK 修正总是滞后一个 SCF 迭代。当前迭代的 γ 是在上次 λ 的 HK 修正下计算的。

### 4.2 ψ 和 ρ 冻结

```
Normal SCF:
  H → diagonalize → ψ → DM → ρ → mix → H' → ...

Gradient Descent (无内循环):
  iter_finish 只改 λ, 不改 ψ 或 ρ
  下次 iter_init 用新 λ 构建 H → 重新 diagonalize

Inner Loop (设计, 未实现):
  lam_trial → H_HK(lam_trial) → re-diagonalize → ψ' (ρ 不变)
  多次内迭代 → 找到最优 λ
  然后 ρ 更新 (外循环)
```

### 4.3 内存生命周期

```
DeltaP 对象创建:    iter_finish iter=1
  ├── 持久:  S_dk_, kstring_data_, A_nk_, results_
  ├── 半持久: prev_gamma, gamma_accum (per alpha 生命周期)
  └── 临时:   所有 istring/j 循环局部变量 (栈分配, 自动释放)

SMO init:          首次 compute_wannier_polarization
  └── smo_overlap_, smo_overlap_inv_ (permanent)

k-string setup:    setup_kstring() 每次 alpha 切换
  └── k_index_, nppstr_, total_string_ (覆盖写入)

匹配保存:          每次 compute_wannier_polarization 结束
  ├── deltap_branch.dat  ← W_prev_3d
  └── deltap_match.dat   ← saved_matches_
```

---

## 五、与 DeltaSpin 对比

| | DeltaSpin | DeltaP (当前) |
|---|---|---|
| 约束量 | 原子磁矩 M_I | 原子极化 γ_I |
| λ 更新位置 | `hamilt2rho_single` (内循环) | `iter_finish` (外循环) |
| λ 优化器 | BFGS (内循环) | 梯度下降 (无内循环) |
| ρ 冻结 | 内循环内冻结 | 无冻结 (每次 iter 都更新) |
| H 修正 | contributeHk 中 `λ×σ_z` | contributeHk 中 `(M+M†)/2` |
| 可观测量 | `cal_mi_lcao_wrapper` | `compute_gamma_scf` |

**DeltaP 缺失的关键**: 内循环 (`hamilt2rho_single` 中的 BFGS-CG, 类似 DeltaSpin 的 `run_lambda_loop`). 已在 `esolver_ks_lcao.cpp:605-680` 有未完成实现 (segfault, 因为 `hsolver_lcao_obj.solve()` 不可重入).
