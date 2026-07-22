# DeltaP 分支选择初始化 Bug 分析

> 日期: 2026-07-19

---

## 问题现象

λ sweep 测试（deltap_nscf=0, mixing_beta=0）显示 gamma 值在不同 λ_init 之间正负号交替：

| λ_init | γ_B | γ_N | λ_1 (B) | λ_1 (N) |
|--------|-----|-----|---------|---------|
| 0.0 | -4.26 | -4.99 | -0.43 | -0.50 |
| 0.1 | 0.94 | 0.98 | 0.57 | 0.59 |
| 0.5 | -4.28 | -4.98 | -1.64 | -1.99 |
| 1.0 | 0.86 | 1.02 | 1.43 | 1.51 |
| 2.0 | 14.6 | 17.4 | 9.29 | 10.7 |
| 5.0 | -4.26 | -5.00 | 2.87 | 2.50 |

---

## 诊断输出

```
DeltaP branch-set: atom 0 rescaled=-4.920255e+00 selected=-8.856205e-01 
                   prev=0.000000e+00 delta=4.034634e+00
DeltaP branch-set: atom 1 rescaled=-5.764008e+00 selected=-1.051320e+00 
                   prev=0.000000e+00 delta=4.712688e+00
```

**关键信息**：
- `prev=0.000000e+00` — 第一次迭代的分支参考值初始化为 0
- `rescaled=-4.92` — zeta rescaling 后的 per-atom gamma（远偏离 0）
- `selected=-0.886` — 分支选择强制拉向 prev=0
- `delta=4.03` — 分支修正量（约 2π × 0.64）

---

## 根因分析

### 1. 分支选择代码逻辑

```cpp
// deltap_wannier.cpp, branch selection
for (int iat = 0; iat < nat; ++iat)
{
    double best = gamma_I[iat][alpha];
    double best_dist = std::abs(best - prev_gamma[iat][alpha]);
    
    // 尝试 ±2π × w_sum 修正
    for (int k = -1; k <= 1; k += 2)
    {
        double candidate = gamma_I[iat][alpha] + k * 2.0 * M_PI * w_sum_I[iat];
        double dist = std::abs(candidate - prev_gamma[iat][alpha]);
        if (dist < best_dist)
        {
            best = candidate;
            best_dist = dist;
        }
    }
    gamma_I_corrected[iat][alpha] = best;
}
```

**分支选择标准**：选择距离 `prev_gamma` 最近的分支。

### 2. 第一次迭代的初始化问题

在第一次 SCF 迭代（iter=1）的 `iter_finish` 中：
```cpp
// W_prev_3d 初始化
W_prev_3d_.assign(ucell.nat, ModuleBase::Vector3<double>(0, 0, 0));
```

分支参考值初始化为 **0**。

### 3. 问题链条

1. **Zeta rescaling** 将 per-atom gamma 从原始值（约 1.9）缩放到约 -5
2. **分支选择** 以 `prev=0` 为参考，强制将 gamma 拉到接近 0 的分支
3. **修正后的 gamma**（约 -0.9）与 rescaling 前的值（约 -5）相差 4 rad
4. **梯度下降** 使用修正后的 gamma 更新 λ：`λ += step × γ_corrected`
5. **不同 λ_init** 导致不同的 rescaling 因子，使得修正后的 gamma 落在不同的 2π 分支

### 4. 为什么不同 λ_init 导致符号交替

- λ_init=0.0: raw γ ≈ -4.92, 修正后 ≈ -0.89 → λ_1 ≈ -0.43
- λ_init=0.1: raw γ ≈ 1.57, 修正后 ≈ 1.57（已在 [-π, π] 内）→ λ_1 ≈ 0.57
- λ_init=0.5: raw γ ≈ -4.94, 修正后 ≈ -0.89 → λ_1 ≈ -1.64
- λ_init=1.0: raw γ ≈ 1.73, 修正后 ≈ 1.73 → λ_1 ≈ 1.43

**模式**：raw gamma 的符号取决于 λ_init，而分支选择总是将 gamma 拉到接近 0 的分支。当 raw gamma 远离 0 时，修正量很大；当 raw gamma 接近 0 时，修正量很小。

---

## 影响

### 1. λ→γ 响应测试失效

由于分支选择强制将 gamma 拉到接近 0，不同 λ_init 的测试点实际上在不同的 2π 分支上测量。这导致：
- γ(λ) 不是 λ 的连续函数
- 无法通过插值找到 λ_opt
- accept_trial 的线性插值假设不成立

### 2. 内循环收敛困难

内循环的 `bfgs.step()` 假设 γ 是 λ 的平滑函数，但分支选择的不连续性破坏了这一假设。即使在内循环内部（同一 SCF 迭代），不同 trial λ 的 gamma 也可能落在不同分支。

---

## 修复方案

### 方案 A：改进 prev_gamma 初始化

**问题**：`prev=0` 不是合理的分支参考。

**修复**：使用第一次迭代的 raw gamma 作为初始参考：

```cpp
// iter_finish, 第一次迭代
if (!deltap_scf_initialized_)
{
    // 先计算 gamma（不做分支修正）
    dp->compute_gamma_scf_raw(ucell, psi, pelec);  // 新增方法
    
    // 使用 raw gamma 初始化 W_prev_3d_
    for (int iat = 0; iat < ucell.nat; ++iat)
        W_prev_3d_[iat] = ModuleBase::Vector3<double>(
            gamma_raw[iat][0], gamma_raw[iat][1], gamma_raw[iat][2]);
    
    // 然后初始化 dp_scf_
    ...
}
```

**优点**：第一次迭代的 gamma 不被强制拉到 0 附近。

**缺点**：需要修改 `compute_gamma_scf` 接口，增加 raw gamma 输出。

### 方案 B：禁用第一次迭代的分支选择

**修复**：第一次迭代不做分支修正，直接使用 raw gamma：

```cpp
// deltap_wannier.cpp
if (first_iteration && !deltap_scf_initialized_)
{
    // 跳过分支选择，使用 raw gamma
    return gamma_I;  // 不修正
}
else
{
    // 正常分支选择
    ...
}
```

**优点**：简单，不改变接口。

**缺点**：第一次迭代的 gamma 可能在错误的分支上，影响后续迭代的收敛。

### 方案 C：使用连续分支追踪

**修复**：在 λ sweep 过程中，使用连续追踪而不是独立分支选择：

```cpp
// 对于 λ sweep 测试
double prev_gamma = gamma_at_lambda_0;
for (double lambda : lambda_values)
{
    double gamma = compute_gamma_at_lambda(lambda);
    // 追踪到最近的分支
    gamma = branch_select(gamma, prev_gamma);
    prev_gamma = gamma;
    // 记录 (lambda, gamma)
}
```

**优点**：确保 λ→γ 曲线连续。

**缺点**：需要在测试脚本中实现，不能依赖 ABACUS 内部逻辑。

---

## 推荐方案

**短期**：方案 C（连续分支追踪），用于 λ sweep 测试。

**长期**：方案 A（改进 prev_gamma 初始化），解决根本问题。

**验证标准**：
- λ sweep 测试中 γ(λ) 应该是 λ 的连续函数
- 内循环中 trial λ 的 gamma 应该在同一分支上

---

## 相关文件

- `source/source_lcao/module_deltap/deltap_wannier.cpp`: 分支选择代码
- `source/source_esolver/esolver_ks_lcao.cpp:758`: W_prev_3d_ 初始化
- `source/source_lcao/module_deltap/deltap.h:107`: W_prev_3d_ 定义
