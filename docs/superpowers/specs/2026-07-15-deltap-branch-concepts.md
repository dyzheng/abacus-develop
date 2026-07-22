# DeltaP 分支选择机制——概念、参数与带交叉分析

> 日期: 2026-07-15  
> 目标: 完整解释 Wilson Loop 分支选择、带交叉检测、以及所有关键参数的含义

---

## 一、极化计算的完整链路

```
k-string 上的 k 点序列: k₀ → k₁ → ... → k_N

  1. O_j = C†(k_j) · S(dk) · C(k_{j+1})          重叠矩阵 (nocc × nocc)
  2. W_j = O₀ × O₁ × ... × O_j                    Wilson Loop 矩阵
  3. Diagonalize W_j → evals[1..nocc]              特征值
  4. Track eigenvalues across j → γ_unwrapped[n]   展开相位
  5. gamma_I[iat] = Σ_n w_I_n[iat] × γ_unwrapped[n]  per-atom 极化
  6. Zeta rescaling + branch selection             分支修正
  7. Average over strings → P_I, P_total          最终极化
```

---

## 二、逐个概念详解

### 2.1 O_j 矩阵 (Overlap Matrix)

```
O_j[n,m] = <ψ_n(k_j) | ψ_m(k_{j+1})>
         = [ C†(k_j) · S(dk) · C(k_{j+1}) ]_{n,m}
```

| 符号 | 含义 | 维度 (BN) |
|------|------|:---:|
| n, m | 占据带索引 | 4 |
| C | LCAO 系数矩阵 | 26×4 |
| S(dk) | LCAO 重叠矩阵 (含 Bloch 相位) | 26×26 |
| O_j | k_j→k_{j+1} 的波函数重叠 | 4×4 |

**物理意义**: 相邻 k 点间 Bloch 波函数的 overlap。O_j 越接近单位矩阵, k 空间采样越密。O_j 的行列式 (det) 贡献了总 Berry phase 的一部分: `arg(zeta) = Σ_j arg(det(O_j))`。

**可变性**: O_j 随波函数 C (从而随 SCF 电荷密度 / λ 值) 改变。不同 λ → 不同 H → 不同 C → 不同 O_j → 不同 evals → Hungarian 匹配可能不同。

---

### 2.2 特征值匹配与 Hungarian 算法 (P0 修复)

#### 2.2.1 为什么需要匹配

W_j 对角化后 LAPACK (zgeev) 返回 nocc 个特征值 evals_j[n]。**返回顺序不保证**——同一矩阵两次对角化可能返回不同排列。

追踪每个 band 沿 k-string 的相位连续性要求匹配 evals_j 和 evals_{j-1}:

```
cost[m][n] = |phase_diff(evals_j[n] / evals_{j-1}[m])|  (wrapped ±π)
```

用 Hungarian 最小化 Σ cost → 全局最优匹配 → 保证确定性。

#### 2.2.2 Hungarian 算法的局限: 带交叉

当两个 evals 的相位差 < ε 时:

```
正常:  evals_prev = {0.30, 1.80}   evals_new = {0.35, 1.75}
  cost = [[0.05, 1.45], [1.50, 0.05]] → 匹配确定 (0→0, 1→1)

带交叉: evals_prev = {0.30, 0.31}   evals_new = {0.35, 0.34}
  cost = [[0.05, 0.04], [0.05, 0.03]] → cost[0][0]+cost[1][1]=0.08
                                       cost[0][1]+cost[1][0]=0.09
                                       差 0.01 → 匹配确定但近乎二义
```

**当前代码: 无带交叉检测**。当 cost 矩阵中两行/两列的值几乎相等时, Hungarian 做确定性匹配但**无法验证匹配是否正确**。

**检测方案**:

```cpp
// 在 Hungarian 匹配后检查
double best_cost = total_cost;
double alt_cost = cost[0][1] + cost[1][0];  // 假设交换带 0 和 1
if (fabs(best_cost - alt_cost) < 1e-6) {
    // 带 n 和 m 的匹配不唯一——标记为 ambiguous
    mark_crossing(n, m);
}
```

**带交叉时的处理: 合并跟踪**。当检测到 ambiguity 时, 将两个带合并为一个"复合带":

```
gamma_unwrapped_merged = gamma_prev[n] + gamma_prev[m]   // 总相位不变
w_In_merged = w_In[n] + w_In[m]                          // 权重合并
```

合并后降维 (n_dim-1), Hungarian 问题简化。交叉后在下一个 k 点尝试重新分解 (如果 evals 差异变得显著)。

**为什么合并有效**: 两个交叉带的总相位贡献是 gauge invariant 的 (等于 `arg(evals[n]) + arg(evals[m])`)。单个带的分量在交叉点丧失物理意义, 但总和不。

#### 2.2.3 BN 的带结构

从实际测试数据:

```
n=0: raw_sum=0.034  → BN 成键带 (π)
n=1: raw_sum=0.061  → BN 成键带 (σ)
n=2: raw_sum=1.625  → N 孤对电子带
n=3: raw_sum=0.016  → 反键带 (above EF, 近乎零投影)
```

| 带 | 化学性质 | SMO 总投影 | 对极化贡献 |
|:---:|------|:---:|:---:|
| 0 | B-N 成键 | 0.034 | ~2% |
| 1 | B-N 成键 | 0.061 | ~4% |
| 2 | N 孤对 | 1.625 | **~93%** |
| 3 | 反键 | 0.016 | ~0.4% |

- **带 2 孤立** (w_sum 远大于其他带), 几乎不可能与其他带交叉
- **带 0 和带 1 接近** (w_sum 都在 0.03-0.06), 可能交叉
- **带 3 贡献可忽略** (w_sum < 0.02)

**关键结论**: 带 0 和带 1 的 w_In 相近 → 即使交叉, 对 per-atom gamma 的影响也很小 (Σ w_I_k × γ_k 在交换时近似不变)。这就是 BN 当前 branch selection 虽然不完美但仍能得到合理结果的原因。

---

### 2.3 gamma_unwrapped[n] (展开相位)

```
第一次 (j=0): gamma_unwrapped[n] = arg(evals_0[n])   排序后
后续 (j>0): gamma_unwrapped[n] = gamma_prev[n] + diff  通过匹配追踪
                                   其中 diff = arg(evals_new[n]/evals_prev[matched])
```

**物理意义**: 每个占据带的累积 Berry phase, 消除了 2π 分支跳变。

**不唯一性**: 虽然 matching 是确定性的, 但 `arg(evals_0[n])` 的初始值在 (-π,π] 内。不同字符串的 k_0 不同 → evals_0 不同 → 初值不同。当带交叉时, 匹配可能交换 → 后续展开可能相差 2π 的整数倍。

---

### 2.4 Per-atom Gamma

```
gamma_I_per_atom[iat] = Σ_n w_I_n[iat] × gamma_unwrapped[n]
```

| 符号 | 含义 |
|------|------|
| w_I_n[iat] | SMO 投影权重: <br>w_I_n = Σ_{lm∈I} |⟨α_lm^I | ψ_n⟩|²<br>经 Löwdin 正交化后 Σ_I w_I_n = 1 |
| gamma_unwrapped[n] | 带 n 的展开 Berry phase |

**物理意义**: 每个原子对极化的贡献, 由 SMO 局域轨道投影分解。O 电负性 > H → O 的 w_I_n 更大 → O 的 gamma_I 更大。

**跨 string 变化**: 不同 (ix,iy) 的 string 经过不同 k 点 → SMO 投影权重 w_I_n 不同 → raw gamma 不同 (通常 ~10-30%)。

**带交叉时的 invariance**: 如果带 n 和 m 的 w_I_n ≈ w_I_m, 交换时 γ_I 近似不变。如果 w_I_n ≫ w_I_m, 交换时 γ_I 会显著变化。

---

### 2.5 Zeta Rescaling 和 ref_gamma_unw_sum

```
gamma_raw_sum  = Σ_iat gamma_I_per_atom[iat]           per-atom gamma 之和
gamma_unw_sum  = Σ_n gamma_unwrapped[n]                展开相位之和

理想 (Löwdin 完美): gamma_raw_sum ≈ gamma_unw_sum     (Σ_I w_I_n = 1)
实际 (SMO 不完备): gamma_raw_sum ≠ gamma_unw_sum      (Σ_I w_I_n < 1)

scale = gamma_unw_sum / gamma_raw_sum                  (~0.9-1.1, 修正不完备性)
gamma_I_corrected[iat] = gamma_I_raw[iat] × scale
```

**ref_gamma_unw_sum** (Bug 1 修复): 每条 string 的 gamma_unwrapped 可能因 eigenvalue 展开不同而相差 2π。修复: **所有 string 共用第一条 string 的 gamma_unw_sum**:

```
string 0: ref_gamma_unw_sum = gamma_unw_sum_0
string N: 使用 ref_gamma_unw_sum 作为 scale = ref / gamma_raw_sum
```

**未修复前**: 不同 string 用不同的 gamma_unw_sum → scale 不一致 → per-atom gamma 跨 string 跳变 → BRANCH INCONSISTENT。

---

### 2.6 Per-atom 分支参考 (prev_gamma)

```
第一个 string: prev_gamma = target (默认 0.0)
               → 分支选择: 选离 0 最近的 per-band branch
后续 string:   prev_gamma = 前一条 string 的 gamma_I (校正后)
               → 分支选择: 选离前一条 string 最近的 branch
```

| 参数 | 初始值 | 更新时机 | 作用范围 |
|------|------|------|------|
| prev_gamma[iat] | 0.0 或 W_prev_[iat][alpha] | 每条 string 处理后 | 当前 alpha 内跨 string |

**NaN 的 bug** (已修复): 原来初始化为 NaN → 第一条 string **跳过分文选择** (|g-NaN| 永远是 false) → 随机锁定分支。

---

### 2.7 分支选择 (Branch Selection)

对每个原子 iat, 当前 string 的 gamma_I_corrected[iat] 与 prev_gamma[iat] 比较:

```
if |g - prev| < π:
    无需校正 (同一 branch)
else:
    搜索每个带 n 的 ±2π·w_In 偏移:
        candidate = g + sign × 2π × w_In_matrix[n][iat]
        选 |candidate - prev| 最小的 candidate
```

**物理意义**: per-atom gamma 可以相差 2π × w_I_n (每个带贡献 2π 的整数倍)。分支选择确保相邻 string 的 gamma 连续。

---

### 2.8 W_prev_3d (跨 SCF 分支持久化)

| 参数 | 类型 | 存储 | 作用 |
|------|------|------|------|
| W_prev_[iat] | Vector3<double> | deltap_branch.dat | 保存每个原子在 (x,y,z) 三个方向的 gamma |

**生命周期**:
```
SCF iter 1: has_prev_ = false → prev_gamma = 0.0
            → 分支选择以 0 为参考
            → 保存 W_prev_3d[iat] = (γ_x, γ_y, γ_z)

SCF iter N: load_branch() → has_prev_ = true
            → prev_gamma = W_prev_3d[iat][alpha]
            → 分支选择以上一轮的 gamma 为参考
```

**未修复前** (Bug 2): W_prev_ 只存最后一个方向 (z) → x 和 y 的方向被 z 污染。

---

## 三、分支不确定性的三个层次

从底层到顶层, 需要"冻结"以确保跨 λ / 跨 SCF 的一致性:

| 层级 | 对象 | 失效条件 | 当前状态 |
|:---:|------|------|:---:|
| L0 | 带交叉 (per k-point pair) | |eval[n]|-|eval[m]|| → 0 | ❌ 无检测 |
| L1 | 特征值匹配顺序 (per k-pair) | Hungarian cost 二义性 | ✅ 确定但跨 λ 不一致 |
| L2 | Zeta scale 参考值 (per alpha) | 不同 string 的不同 unwrapped sum | ✅ ref_gamma_unw_sum 固定 |
| L3 | Per-atom 分支参考 (per alpha) | prev_gamma 缺失或错误 | ✅ prev_gamma=0 或 W_prev_3d |
| L4 | 跨 SCF 分支持久化 | W_prev_ 跨方向污染 | ✅ W_prev_3d Vector3 |

### 3.1 L0: 带交叉检测与合并

**当前**: 无检测。Hungarian 在 cost 接近时做确定性匹配, 但可能错误。

**建议**: 添加 Hungarian 结果二义性检查:
```cpp
// Hungarian 后
double cost_swapped = 0;
for (int n = 0; n < N; ++n) {
    int m = match_to[n];
    int m_alt = (n + 1) % N;  // assume swap with neighbor
    cost_swapped += cost[m_alt][n];
}
if (fabs(best_cost - cost_swapped) < 1e-6)
    merge_bands(n, n+1);  // → compound band tracking
```

### 3.2 L1: 特征值匹配冻结

**当前**: Hungarian 在每次调用时重新计算, 跨 λ 值不一致。

**建议**: 首次运行保存 `match_to[]` 到文件; 后续运行跳过 Hungarian 直接复用:
```cpp
if (has_match_history_) {
    match_to = load_match(istring, j);     // replay saved matching
} else {
    hungarian(cost, match_to);              // compute fresh
    save_match(istring, j, match_to);       // save for next run
}
```

### 3.3 L2-L4: 已完成

`ref_gamma_unw_sum` (L2), `prev_gamma=0|W_prev_3d` (L3), `W_prev_3d` (L4) 均已修复。

---

## 四、BN 测试数据解读

### 4.1 SCF 收敛后的极化 (2×2×2, λ=0)

```
P = (-1.14e-2, +1.63e-2, +1.63e-2)  ← Px≠Py=Pz, 立方采样不均
σ(跨 string) = 3e-3 rad            ← 同一 run 内分支一致
σ(跨 lambda, frozen) = 振荡         ← L1 未冻结
```

### 4.2 带结构与交叉可能

| 带 | SMO 投影 | 角色 | 交叉可能性 |
|:---:|:---:|------|:---:|
| 0 | 0.034 | B-N 成键 | ⚠️ 与带 1 可能交叉 |
| 1 | 0.061 | B-N 成键 | ⚠️ 与带 0 可能交叉 |
| 2 | 1.625 | N 孤对 | ✅ 孤立 (主导极化) |
| 3 | 0.016 | 反键 | ✅ 可忽略 |

带 0 和 1 的 w_In ≈ 0.03-0.06, 对 per-atom γ 贡献小 (~6%), 即使交叉影响也有限。
带 2 贡献 ~93% 的极化, 不参与交叉 → **BN 当前没有带交叉问题**。

### 4.3 Stage 2 λ sweep 振荡的根因

振荡不是来自带交叉 (L0), 也不是 L2-L4 (已修复)。根源在 **L1 未冻结**:
- λ=0 的 Hungarian 匹配排列与 λ=0.05 的不同
- 不同排列 → 不同 gamma_unwrapped → 不同 per-atom gamma
- 即使在无交叉的正常情况下, 浮点差异也可能改变 cost 的极小值位置

---

## 五、修复状态总结

| 场景 | 修复前 | 修复后 |
|------|------|------|
| 同一 run 内跨 string | σ 可达 1.6 rad, BRANCH INCONSISTENT | σ ~ 3×10⁻³, OK |
| 同一 run 跨 alpha | gamma_accum 共享 → P 值污染 | 独立 accum ✅ |
| 跨 SCF iter (同 λ) | NaN prev → 随机选 branch | W_prev_3d 持久化 ✅ |
| 跨 λ 值 (frozen charge) | 每条 string 独立 unwrapped sum | ref 固定 ✅ |
| 跨 λ 值 (matching) | Hungarian 重新计算 | 未冻结 ❌ |
| 带交叉检测 | 无 | 待实现 ❌ |

---

## 六、下一步实施路线

| 优先级 | 任务 | 预计行数 |
|:---:|------|:---:|
| P1 | L1 匹配冻结: 保存/加载 match_to[] per k-pair | ~80 |
| P2 | L0 带交叉检测: cost 二义性 + 合并跟踪 | ~60 |
| P3 | 缩减 n_dim: 添加 deltap_nbands 参数 (大体系) | ~20 |
