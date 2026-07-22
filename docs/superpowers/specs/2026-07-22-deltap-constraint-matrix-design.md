# 方向 2：通用线性约束矩阵 — 详细设计方案

**日期**: 2026-07-22
**状态**: 设计文档

---

## 1. 目标

将 DeltaP 的约束能力从当前两种固定模式（per_atom / total）扩展到**任意线性组合**：

```
C · γ = t

C: n_constraints × n_atoms 矩阵
γ: 每原子 Berry 相位向量 [n_atoms]
t: 靶标向量 [n_constraints]
```

当前两种模式是 C 的特殊情况：
- `per_atom`: C = I（单位矩阵），t = target.dat 的每原子值
- `total`: C = [1, 1, ..., 1] / nat，t = [目标总和 / nat]

新增的有物理意义的约束：

| 约束矩阵 C | 物理含义 | λ 的物理意义 |
|-----------|---------|-------------|
| `[1, -1, 0, ...]` | γ_A - γ_B = target | A↔B 电荷转移的"广义力" |
| `[w_1, w_2, ..., w_n]` | 加权总极化 | 加权有效电场 |
| `[[1,1,0],[0,0,1]]` | 部分总约束 + 单原子约束 | 混合模式 |
| `[Z*_1, Z*_2, ..., Z*_n]` | Born 有效电荷方向的极化 | Z* 约束刚度 |

---

## 2. 数学框架

### 2.1 拉格朗日形式

约束条件：Σ_i C_αi · γ_i = t_α，对每个约束 α = 0..m-1

```
L = E_KS + Σ_α λ_α · ( Σ_i C_αi · γ_i - t_α )
  = E_KS + Σ_i ( Σ_α λ_α C_αi ) · γ_i - Σ_α λ_α t_α
```

每个原子的**等效 λ** 是约束 λ 的线性组合：

```
λ_i^eff = Σ_α λ_α · C_αi
```

### 2.2 λ 更新（梯度下降）

约束 α 的残差：

```
r_α = Σ_i C_αi · γ_i - t_α
```

梯度下降更新：

```
λ_α_new = λ_α + step × r_α
```

混合：

```
λ_α = mixing × λ_α_raw + (1-mixing) × λ_α
```

### 2.3 哈密顿修正

当前代码（per-atom 模式）：

```cpp
w_eff[n] = Σ_i λ_i · w_In[n][i]
```

推广到约束矩阵：

```cpp
// 预计算每个约束的投影权重
for (int α = 0; α < n_constraints; ++α)
    for (int n = 0; n < nocc; ++n)
        w_proj[α][n] += C[α][i] × w_In[n][i]

// 有效权重
w_eff[n] = Σ_α λ[α] × w_proj[α][n]
```

代码改动量：在 `compute_hk_correction` 中将单一的 `lambda[iat]` 替换为上述两层求和。约 15 行改动。

### 2.4 分支选择：约束空间的顺序贪心搜索

这是实现的核心难点。总模式（C = [1,1,...,1]）的顺序贪心可直接推广：

**算法**：对每原子 iat，在"其他原子取期望值"的假设下，选出最佳 k-vector。

```
输入：C (m×n), t (m×1), raw_γ (n×1), shift_amp (n×nbands)
输出：best_γ (n×1)

γ_running = [0, 0, ..., 0]  // n_atoms 维

for iat = 0 to nat-1:
    // 1. 前驱原子：已选定
    // 2. 当前原子：待优化
    // 3. 后继原子：取其 raw γ 作为"期望值"

    γ_expect = [γ_running[0], ..., γ_running[iat-1],
                ???,                       // 当前原子待定
                raw_γ[iat+1], ..., raw_γ[nat-1]]

    // 约束残差（当前原子 + 后继原子的 raw 贡献）：
    r_remain = t - C · γ_running          // m 维
    // 后继原子贡献的估计：
    C_future_contrib = Σ_{j>iat} C[:,j] × raw_γ[j]  // m 维
    // 当前原子需要承担的残差：
    r_current = r_remain - C_future_contrib          // m 维

    // 当前原子的列向量：
    c_i = C[:, iat]  // m 维

    // 有效靶标：使 |c_i × γ_i - r_current| 最小的 γ_i
    // = argmin_{γ_i} Σ_α (C_αi · γ_i - r_current[α])²
    // = (c_i^T · r_current) / (c_i^T · c_i)      (最小二乘)
    effective_target = dot(c_i, r_current) / dot(c_i, c_i)
    // 如果 c_i = 0（此约束不涉及该原子），跳过

    // 2. 穷举搜索 K=5：
    for all k-vectors:
        γ_candidate = raw_γ[iat] + Σ_n k[n] × shift_amp[iat][n]
        if |γ_candidate - effective_target| < best_dist:
            best_γ = γ_candidate

    γ_running[iat] = best_γ

输出 best_γ
```

**关键性质**：

- 对 C = I：`c_i = [0,...,1,...,0]` → `effective_target = r_current[iat] / 1 = t[iat]`（原 per-atom 模式）
- 对 C = [1,1,...,1]：`c_i = [1]` → `effective_target = r_current / 1 = total_t - Σ_{j>iat} raw_γ[j]`（原 total 模式贪心）
- 对 C = [1,-1,0]：`c_0 = [1], c_1 = [-1], c_2 = [0]` → atom 2 不受约束（effective_target = raw_γ[2]，即不改变）

---

## 3. 代码改动清单

### 3.1 输入参数（~30 行）

文件：`input_parameter.h`, `read_input_item_other.cpp`

```cpp
std::string deltap_constraint_matrix = "";  // 约束矩阵文件路径
```

删除或废弃 `deltap_constraint_mode`（被 C 矩阵取代，向后兼容保留）。

### 3.2 矩阵加载（~30 行）

文件：`esolver_ks_lcao.cpp` — `deltap_init` 函数

```cpp
// 读取约束矩阵
std::vector<std::vector<double>> constraint_matrix_;  // m × n
std::vector<double> constraint_target_;                // m 维

if (!PARAM.inp.deltap_constraint_matrix.empty()) {
    std::ifstream ifs(PARAM.inp.deltap_constraint_matrix);
    int m, n;
    ifs >> m >> n;
    constraint_matrix_.resize(m, std::vector<double>(n));
    constraint_target_.resize(m);
    for (int a = 0; a < m; ++a) {
        for (int i = 0; i < n; ++i) ifs >> constraint_matrix_[a][i];
        ifs >> constraint_target_[a];
    }
    // 设置 DeltaP 的 target_gamma_ = constraint_target_ / diag(C) 近似
    // （用于分支选择的初始锚定）
} else if (PARAM.inp.deltap_constraint_mode == "total") {
    // 向后兼容：自动构建 C = [1,1,...,1]/nat
} else {
    // 向后兼容：C = I
}
```

### 3.3 分支选择 — 约束空间贪心（~80 行）

文件：`deltap_wannier.cpp` — 替换 per_atom/total 的 if-else 分支

将当前 `if (total_mode) { ... } else { ... }` 替换为统一的约束空间贪心搜索。

### 3.4 λ 更新（~25 行）

文件：`esolver_ks_lcao.cpp` — `deltap_update_lambda`

将 lambda 向量从 n_atoms 维改为 n_constraints 维。

```cpp
std::vector<double> lambda(n_constraints, 0.0);  // 约束空间的 λ

for (int a = 0; a < n_constraints; ++a) {
    double residual = 0.0;
    for (int i = 0; i < n_atoms; ++i)
        residual += C[a][i] * gamma_I[i][alpha];
    residual -= constraint_target_[a];
    lambda_raw[a] += step * residual;
    lambda[a] = mixing * lambda_raw[a] + (1-mixing) * lambda[a];
}
```

### 3.5 HK 修正（~20 行）

文件：`deltap_wannier.cpp` — `compute_hk_correction`

将 `lambda[iat]` 改为 `Σ_α λ[α] · C[α][iat]`。

### 3.6 DeltaP 输出格式（~20 行）

文件：`esolver_ks_lcao.cpp` — 输出格式

显示约束空间的 λ 向量和残差：

```
[DeltaP P3] iter=50  γ=(3.999, 3.495, 3.507)
  constraints: λ=(+2.15, -1.08) μRy  |C·γ-t|=(0.003, 0.002)
```

---

## 4. 测试计划

### 4.1 回归测试

| 测试 | C 矩阵 | 靶标 | 预期 |
|------|--------|------|------|
| Per-atom 等价 | I | 同 per-atom | 与当前 per-atom 模式一致 |
| Total 等价 | [1,1,...,1]/nat | 同 total | 与当前 total 模式一致 |
| H2O 单原子 | 同 H2O PES 靶标 | 同 H2O PES | 再现已有结果 |

### 4.2 新物理测试

| 测试 | C 矩阵 | 靶标 | 测量量 |
|------|--------|------|--------|
| 电荷转移刚度 | [1, -1, 0] | target_diff | dλ/d(Δγ) → 原子间电荷转移刚度 |
| 加权极化 | [w_O, w_H, w_H] | weighted_total | 不同权重的总极化刚度 |
| 部分冻结 | [[1,0,0],[0,1,0]] | γ_O, γ_H1 固定 | γ_H2 自由演化 |

---

## 5. 边界情况与鲁棒性

### 5.1 欠定约束 (m < n)

约束数少于原子数 → 系统有 n - m 个零模式。贪心搜索中，零模式对应的原子 `effective_target = raw_γ`（保持不变）。

### 5.2 过定约束 (m > n)

不可能精确满足所有约束。贪心搜索中，每个原子取最小二乘有效靶标，最终残差非零。

### 5.3 零列向量 (c_i = 0)

原子 i 不参与任何约束。`effective_target = raw_γ[iat]`，不进行分支搜索。

### 5.4 与当前 total 模式的兼容

```
C = [1/nat, 1/nat, ..., 1/nat]   (归一化，使得 sum = t × nat)
```

等价于当前 total 模式（除归一化因子外）。

---

## 6. 文件 constraint.mat 格式

```
3  5                    # m=3个约束, n=5个原子
1  0  0  0  0   t1     # 约束1: γ_0 = t1
0  1 -1  0  0   t2     # 约束2: γ_1 - γ_2 = t2
1  1  1  1  1   t3     # 约束3: Σγ = t3
```

空白分隔，`#` 注释。每行最后的数字是靶标值 t_α。

---

## 7. 实施优先级

| 步骤 | 工作量 | 依赖 |
|------|--------|------|
| 3.2 矩阵加载 + 3.3 分支选择 | 最高 | — |
| 3.4 λ 更新 | 高 | 3.2 |
| 3.5 HK 修正 | 中 | 3.2 |
| 3.6 输出格式 | 低 | 3.4 |
| 回归测试 | 中 | 全部 |

总代码量：~200 行（分布在 4 个文件中）。

---

## 8. 与当前代码的关系

- **不删除** `deltap_constraint_mode`: 作为快捷方式保留。当 `constraint_matrix` 为空时，使用 `constraint_mode` 自动构建 C 矩阵（向后兼容）。
- **不修改** per-atom 分支选择的核心穷举搜索代码：K=5 的 odometer 循环完全复用。
- **新增**约束空间贪心搜索函数 `branch_search_constrained(C, t, raw_γ, shift_amp)`：独立的 ~50 行函数。

---

## 9. 设计审查：发现的问题与风险

### 9.1 数学框架审查 ✓

**拉格朗日形式 (2.1)**：正确。约束项的标准形式。

**λ 更新 (2.2)**：正确。残差和梯度下降是标准的。

**HK 修正 (2.3)**：正确。投影公式正确。

**分支选择算法 (2.4)**：正确。顺序贪心的逻辑经过验证：
- `γ_running` 初始化为 `[0,...,0]` 是合理的，因为 `C_future_contrib` 会正确加上未来原子的 raw γ
- 每个原子的 `effective_target` 计算正确：当前原子需要承担的残差 = 总残差 - 未来原子的 raw 贡献

### 9.2 发现的问题

#### 问题 1：伪逆数值稳定性（Section 2.4）

**问题**：当 `dot(c_i, c_i)` 接近零时（原子 i 不参与任何约束），除法会数值不稳定。

**修复**：添加阈值检查：
```cpp
double c_i_norm_sq = dot(c_i, c_i);
if (c_i_norm_sq < 1e-10) {
    // 此原子不受任何约束，保持 raw γ
    effective_target = raw_γ[iat];
} else {
    effective_target = dot(c_i, r_current) / c_i_norm_sq;
}
```

**位置**：Section 2.4 算法伪代码中需要添加此检查。

#### 问题 2：顺序依赖偏差（Section 2.4）

**问题**：算法按 0, 1, ..., nat-1 顺序处理原子。最后一个原子承担最多的约束"负担"。对于 C = [1,...,1] 这是可接受的（所有原子贡献相等），但对于一般 C 矩阵可能引入偏差。

**示例**：C = [[1, 0], [0, 1]]（两个独立约束），t = [5, 10]，raw_γ = [3, 4]
- iat=0：effective_target = 5，best_γ ≈ 5
- iat=1：effective_target = 10，best_γ ≈ 10
结果：γ = [5, 10]，正确。

但如果 C = [[1, 1], [1, -1]]（耦合约束），t = [15, -5]，raw_γ = [3, 4]：
- iat=0：r_current = [15 - 4, -5 - (-4)] = [11, -1]，c_0 = [1, 1]
  effective_target = (11 + (-1)) / (1 + 1) = 5
- iat=1：r_current = [15 - 5, -5 - 5] = [10, -10]，c_1 = [1, -1]
  effective_target = (10 + (-10)) / (1 + 1) = 0
结果：γ = [5, 0]，但最优解应该是 γ = [5, 10]（满足两个约束）。

**原因**：顺序贪心是局部最优，不是全局最优。对于耦合约束，原子之间的选择相互影响。

**缓解方案**：
1. **多次迭代**：对原子顺序进行 2-3 轮迭代（每轮用上一轮的 best_γ 作为初始值）
2. **按约束耦合度排序**：优先处理参与最多约束的原子
3. **接受局部最优**：对于大多数物理场景（C 接近对角或接近 [1,...,1]），顺序贪心足够好

**建议**：初始实现采用单次顺序贪心，在文档中注明此限制。如果测试中发现偏差，再添加多轮迭代。

#### 问题 3：Lambda 向量大小变更（Section 3.4）

**问题**：文档说"将 lambda 向量从 n_atoms 维改为 n_constraints 维"，但没有讨论实现细节。

**调查**：需要检查 `dp_operator` 中 lambda 的存储方式：
- 如果是 `std::vector<double>`，可以 resize
- 如果是固定大小数组，需要改接口

**执行风险**：如果 lambda 在多处被读写（get_lambda, set_lambda, 输出格式），改动可能波及多个文件。

**建议**：在开发前先调查 dp_operator 的 lambda 接口，确定改动范围。

#### 问题 4：向后兼容的 C 矩阵构建（Section 3.2）

**问题**：文档说"向后兼容：自动构建 C = I"和"C = [1,1,...,1]/nat"，但没有展示具体代码。

**修复**：添加具体的构建代码：
```cpp
if (PARAM.inp.deltap_constraint_matrix.empty()) {
    if (PARAM.inp.deltap_constraint_mode == "total") {
        // C = [1/nat, 1/nat, ..., 1/nat]
        constraint_matrix_.resize(1, std::vector<double>(ucell.nat, 1.0 / ucell.nat));
        constraint_target_.resize(1);
        // 从 target.dat 读取总和
        std::ifstream ifs(PARAM.inp.deltap_target_file);
        ifs >> constraint_target_[0];
    } else {
        // C = I (单位矩阵)
        constraint_matrix_.resize(ucell.nat, std::vector<double>(ucell.nat, 0.0));
        for (int i = 0; i < ucell.nat; ++i)
            constraint_matrix_[i][i] = 1.0;
        constraint_target_.resize(ucell.nat);
        std::ifstream ifs(PARAM.inp.deltap_target_file);
        for (int i = 0; i < ucell.nat; ++i)
            ifs >> constraint_target_[i];
    }
}
```

#### 问题 5：分支选择回归风险（Section 3.3）

**问题**：文档说"替换 per_atom/total 的 if-else 分支"，这意味着用新的统一算法替换已测试的代码。

**风险**：如果新算法有 bug，会导致 per_atom 和 total 模式都失败。

**建议**：保留旧代码作为 fallback：
```cpp
if (PARAM.inp.deltap_constraint_matrix.empty()) {
    // 使用旧的 per_atom / total 分支选择
    if (total_mode) { ... } else { ... }
} else {
    // 使用新的约束空间贪心
    branch_search_constrained(C, t, raw_γ, shift_amp);
}
```

这样，只有显式设置 `deltap_constraint_matrix` 时才使用新算法，旧模式不受影响。

#### 问题 6：w_proj 预计算位置（Section 3.5）

**问题**：文档说"预计算每个约束的投影权重"，但没有指定在哪里预计算。

**选项**：
1. **在 `deltap_init` 中预计算**：一次性计算，存储为成员变量。优点：高效。缺点：需要存储 w_proj 数组。
2. **在 `compute_hk_correction` 中每次计算**：每次调用时重新计算。优点：简单。缺点：每次 SCF 迭代都重复计算。

**建议**：在 `deltap_init` 中预计算，因为 w_In 在 SCF 过程中不变（只依赖于轨道基组）。

#### 问题 7：输出格式歧义（Section 3.6）

**问题**：当 n_constraints = n_atoms（per-atom 模式）时，输出应该显示 per-atom λ 还是 constraint-space λ？

**建议**：始终显示 constraint-space λ。对于 per-atom 模式（C = I），constraint-space λ 就是 per-atom λ，所以没有歧义。

### 9.3 执行风险总结

| 风险 | 严重性 | 缓解措施 |
|------|--------|----------|
| 伪逆数值不稳定 | 中 | 添加阈值检查 |
| 顺序贪心偏差 | 低 | 初始实现接受局部最优，后续可添加多轮迭代 |
| Lambda 向量大小变更 | 中 | 开发前调查 dp_operator 接口 |
| 向后兼容 C 矩阵构建 | 低 | 添加具体构建代码 |
| 分支选择回归 | 高 | 保留旧代码作为 fallback |
| w_proj 预计算位置 | 低 | 在 deltap_init 中预计算 |
| 输出格式歧义 | 低 | 始终显示 constraint-space λ |

---

## 10. 开发 TODO 列表（细化）

### Phase 0：调查与准备（1-2 小时）

- [ ] **TODO 0.1**：调查 `dp_operator` 的 lambda 存储方式
  - 检查 `source/source_lcao/module_operator/dp_operator.h` 和 `.cpp`
  - 确认 lambda 是 `std::vector<double>` 还是固定数组
  - 确认 `get_lambda()` 和 `set_lambda()` 的接口
  - 确认 lambda 在哪些地方被读写
  - **输出**：文档记录 lambda 的存储和访问方式

- [ ] **TODO 0.2**：调查 `w_In` 的计算和存储
  - 确认 `w_In[n][i]` 在 SCF 过程中是否变化
  - 确认 `w_In` 的存储位置（DeltaP 成员变量还是临时变量）
  - **输出**：确认 w_In 可以在 `deltap_init` 中预计算

### Phase 1：输入与矩阵加载（2-3 小时）

- [ ] **TODO 1.1**：添加输入参数 `deltap_constraint_matrix`
  - 文件：`source/source_io/module_parameter/input_parameter.h`
  - 添加：`std::string deltap_constraint_matrix = "";`
  - 文件：`source/source_io/module_parameter/read_input_item_other.cpp`
  - 添加读取逻辑（参考 `deltap_target_file` 的实现）
  - **验证**：编译通过，`INPUT` 文件中可以设置 `deltap_constraint_matrix`

- [ ] **TODO 1.2**：在 `esolver_ks_lcao.h` 中添加成员变量
  - 添加：`std::vector<std::vector<double>> constraint_matrix_;`
  - 添加：`std::vector<double> constraint_target_;`
  - 添加：`std::vector<std::vector<double>> w_proj_;`（预计算的投影权重）

- [ ] **TODO 1.3**：实现矩阵加载逻辑
  - 文件：`source/source_esolver/esolver_ks_lcao.cpp` — `deltap_init` 函数
  - 实现 Section 3.2 的加载逻辑，包括：
    - 从 `constraint.mat` 文件读取 C 和 t
    - 向后兼容：当 `deltap_constraint_matrix` 为空时，根据 `deltap_constraint_mode` 构建 C
    - 添加 Section 9.2 问题 4 的具体构建代码
  - **验证**：可以加载 `constraint.mat` 文件，打印 C 和 t 确认正确

- [ ] **TODO 1.4**：实现 w_proj 预计算
  - 在 `deltap_init` 中，加载 C 矩阵后，预计算 `w_proj[α][n] = Σ_i C[α][i] × w_In[n][i]`
  - 需要访问 `w_In`（从 DeltaP 对象获取）
  - **验证**：打印 w_proj 确认正确

### Phase 2：分支选择算法（4-6 小时）

- [ ] **TODO 2.1**：实现约束空间贪心搜索函数
  - 文件：`source/source_lcao/module_deltap/deltap_wannier.cpp`
  - 新增函数：`void branch_search_constrained(const std::vector<std::vector<double>>& C, const std::vector<double>& t, const std::vector<double>& raw_gamma, const std::vector<std::vector<double>>& shift_amp, std::vector<double>& best_gamma)`
  - 实现 Section 2.4 的算法，包括：
    - Section 9.2 问题 1 的伪逆阈值检查
    - 顺序贪心循环
  - **验证**：单元测试（手动构造 C, t, raw_gamma，检查 best_gamma 是否符合预期）

- [ ] **TODO 2.2**：集成到全局分支选择
  - 文件：`source/source_lcao/module_deltap/deltap_wannier.cpp` — `compute_wannier_polarization` 函数
  - 在 Section 9.2 问题 5 建议的位置添加分支：
    ```cpp
    if (PARAM.inp.deltap_constraint_matrix.empty()) {
        // 旧的 per_atom / total 分支选择
        if (total_mode) { ... } else { ... }
    } else {
        // 新的约束空间贪心
        branch_search_constrained(C, t, raw_γ, shift_amp, best_γ);
    }
    ```
  - **验证**：编译通过

### Phase 3：Lambda 更新与 HK 修正（3-4 小时）

- [ ] **TODO 3.1**：修改 lambda 向量大小
  - 文件：`source/source_lcao/module_operator/dp_operator.h` 和 `.cpp`
  - 根据 TODO 0.1 的调查结果，修改 lambda 的存储和接口
  - 如果 lambda 是 `std::vector<double>`，在 `set_lambda` 中 resize 到 n_constraints
  - **验证**：编译通过，lambda 可以存储 n_constraints 个值

- [ ] **TODO 3.2**：修改 lambda 更新逻辑
  - 文件：`source/source_esolver/esolver_ks_lcao.cpp` — `deltap_update_lambda` 函数
  - 实现 Section 3.4 的更新逻辑：
    ```cpp
    for (int a = 0; a < n_constraints; ++a) {
        double residual = 0.0;
        for (int i = 0; i < n_atoms; ++i)
            residual += C[a][i] * gamma_I[i][alpha];
        residual -= constraint_target_[a];
        lambda_raw[a] += step * residual;
        lambda[a] = mixing * lambda_raw[a] + (1-mixing) * lambda[a];
    }
    ```
  - **验证**：编译通过

- [ ] **TODO 3.3**：修改 HK 修正逻辑
  - 文件：`source/source_lcao/module_deltap/deltap_wannier.cpp` — `compute_hk_correction` 函数
  - 将 `w_eff[n] = Σ_i lambda[i] * w_In[n][i]` 改为 `w_eff[n] = Σ_α lambda[α] * w_proj[α][n]`
  - 需要访问 `w_proj`（从 esolver 传递或从 DeltaP 成员变量获取）
  - **验证**：编译通过

### Phase 4：输出格式（1-2 小时）

- [ ] **TODO 4.1**：修改 DeltaP 输出格式
  - 文件：`source/source_esolver/esolver_ks_lcao.cpp` — `iter_finish` 函数
  - 实现 Section 3.6 的输出格式：
    ```
    [DeltaP P3] iter=50  γ=(3.999, 3.495, 3.507)
      constraints: λ=(+2.15, -1.08) μRy  |C·γ-t|=(0.003, 0.002)
    ```
  - 计算并打印约束残差 `|C·γ-t|`
  - **验证**：编译通过，输出格式正确

### Phase 5：回归测试（2-3 小时）

- [ ] **TODO 5.1**：Per-atom 模式回归测试
  - 使用 BN 测试用例（`tests/deltap_bn_test`）
  - 不设置 `deltap_constraint_matrix`（使用旧的 per_atom 模式）
  - 运行 SCF，检查 γ 和 λ 是否与修改前一致
  - **验证**：γ 和 λ 与修改前完全一致（数值误差 < 1e-10）

- [ ] **TODO 5.2**：Total 模式回归测试
  - 使用 H2O 测试用例（`tests/deltap_h2o_test`）
  - 设置 `deltap_constraint_mode = total`（不设置 `deltap_constraint_matrix`）
  - 运行 SCF，检查 Σγ 和 λ 是否与修改前一致
  - **验证**：Σγ 和 λ 与修改前完全一致

- [ ] **TODO 5.3**：C = I 等价测试
  - 使用 BN 测试用例
  - 创建 `constraint_I.mat` 文件：C = I，t = per-atom targets
  - 设置 `deltap_constraint_matrix = constraint_I.mat`
  - 运行 SCF，检查 γ 和 λ 是否与 per_atom 模式一致
  - **验证**：γ 和 λ 与 per_atom 模式一致（数值误差 < 1e-6）

- [ ] **TODO 5.4**：C = [1,...,1] 等价测试
  - 使用 H2O 测试用例
  - 创建 `constraint_total.mat` 文件：C = [1,1,1]/3，t = total target
  - 设置 `deltap_constraint_matrix = constraint_total.mat`
  - 运行 SCF，检查 Σγ 和 λ 是否与 total 模式一致
  - **验证**：Σγ 和 λ 与 total 模式一致（数值误差 < 1e-6）

### Phase 6：新物理测试（3-4 小时）

- [ ] **TODO 6.1**：电荷转移刚度测试
  - 使用 H2O 测试用例
  - 创建 `constraint_diff.mat` 文件：C = [1, -1, 0]，t = target_diff
  - 运行多个 target_diff 值（-0.2, -0.1, 0.0, 0.1, 0.2）
  - 记录 λ 和 E，计算 dλ/d(Δγ)
  - **验证**：λ 与 Δγ 呈线性关系，斜率为电荷转移刚度

- [ ] **TODO 6.2**：加权极化测试
  - 使用 H2O 测试用例
  - 创建 `constraint_weighted.mat` 文件：C = [2, 1, 1]（O 权重 2，H 权重 1），t = weighted target
  - 运行 SCF，检查加权总极化是否被约束
  - **验证**：Σ C[i]·γ[i] ≈ t（残差 < 0.01）

- [ ] **TODO 6.3**：部分冻结测试
  - 使用 H2O 测试用例
  - 创建 `constraint_partial.mat` 文件：C = [[1,0,0],[0,1,0]]，t = [γ_O, γ_H1]
  - 运行 SCF，检查 γ_O 和 γ_H1 是否被约束，γ_H2 是否自由演化
  - **验证**：γ_O ≈ t[0]，γ_H1 ≈ t[1]，γ_H2 不受约束

### Phase 7：文档与清理（1-2 小时）

- [ ] **TODO 7.1**：更新用户文档
  - 文件：`docs/deltap_user_guide.md`（如果存在）
  - 添加 `deltap_constraint_matrix` 参数的说明
  - 添加 `constraint.mat` 文件格式说明
  - 添加示例（per-atom, total, 差分约束）

- [ ] **TODO 7.2**：代码清理
  - 移除调试用的 `std::cout` 语句
  - 添加必要的注释（特别是 Section 9.2 问题 1 的阈值检查）
  - 确保所有新增函数有 Doxygen 风格的注释

- [ ] **TODO 7.3**：提交代码
  - 创建新分支 `feature/constraint-matrix`
  - 提交所有改动
  - 创建 Pull Request

---

## 11. 时间估算

| Phase | 时间 | 累计 |
|-------|------|------|
| Phase 0：调查与准备 | 1-2 小时 | 1-2 小时 |
| Phase 1：输入与矩阵加载 | 2-3 小时 | 3-5 小时 |
| Phase 2：分支选择算法 | 4-6 小时 | 7-11 小时 |
| Phase 3：Lambda 更新与 HK 修正 | 3-4 小时 | 10-15 小时 |
| Phase 4：输出格式 | 1-2 小时 | 11-17 小时 |
| Phase 5：回归测试 | 2-3 小时 | 13-20 小时 |
| Phase 6：新物理测试 | 3-4 小时 | 16-24 小时 |
| Phase 7：文档与清理 | 1-2 小时 | 17-26 小时 |

**总计**：17-26 小时（2-3 个工作日）

---

## 12. 验收标准

1. **功能正确性**：
   - Per-atom 和 total 模式回归测试通过（与修改前一致）
   - C = I 和 C = [1,...,1] 等价测试通过
   - 差分约束、加权极化、部分冻结测试通过

2. **数值稳定性**：
   - 伪逆阈值检查生效（c_i = 0 时不崩溃）
   - 约束残差 < 0.01（对于合理的 target）

3. **向后兼容**：
   - 不设置 `deltap_constraint_matrix` 时，行为与修改前完全一致
   - 旧的 `deltap_constraint_mode` 参数仍然有效

4. **代码质量**：
   - 编译无警告
   - 所有新增函数有注释
   - 无调试用的 `std::cout`

5. **文档完整性**：
   - 用户文档更新
   - 示例文件提供
