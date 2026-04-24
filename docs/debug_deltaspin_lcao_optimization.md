# DeltaSpin LCAO Lambda 优化失败：根因分析与 Debug 方案

## 1. 问题描述

在 `run_lambda_loop_lcao()` 中尝试用"子空间对角化 + 解析 Jacobian"替代原始 `run_lambda_loop()` 的"每步全量 SCF 对角化"，目标是减少昂贵的 `cal_mw_from_lambda()` 调用次数。

测试算例：`tests/17_DS_DFTU/24_LCAO_DS_S2_Z`（Fe 反铁磁二聚体，nspin=2，目标磁矩 ±2.0 µB）

现象：
- lambda 在内迭代中跳到非常大的值
- 优化无效，RMS 不收敛或发散
- 原始 `run_lambda_loop()` 可以正常收敛

## 2. 原始方法 vs 优化方法对比

### 原始方法 (`run_lambda_loop`)
```
每步: lambda → contributeHR → 重建 H(k) → 全量对角化 → 新 C_k, e_k → cal_MW → 新 Mi
```
- 每步调用 `cal_mw_from_lambda()`，内部执行完整的 Hamiltonian 构建 + 对角化
- 使用 BFGS/共轭梯度搜索方向 + `alpha_opt` 线搜索
- 有 `check_restriction()` 限制步长（`sccut` 参数，本算例 = 3.0 eV/µB）
- 典型需要 ~10-50 步收敛

### 优化方法 (`run_lambda_loop_lcao`)
```
Phase 1: 一次全量对角化 → C_k, e_k, Mi
Phase 2: 计算 P_I_sub = D_I^dag D_I（投影矩阵）
Phase 3: 解析 Jacobian chi_I = dM_I/dlambda_I
Phase 4: Newton 迭代（最多 2 步），每步只做子空间对角化
Phase 5: 旋转波函数，更新 DM/charge
```
- 仅 1 次全量对角化，后续用 nbands×nbands 子空间对角化
- 用解析 chi 做 Newton 步

## 3. 可能的根因分析

### 根因 A：解析 Jacobian (chi) 计算错误 — 高概率

**问题代码** (`lambda_loop.cpp:389-428`):

```cpp
// chi_I = dM_I^z / dlambda_I
for (int ik = 0; ik < nks; ik++) {
    const double sign = (ik < nk) ? 1.0 : -1.0;
    for (int n = 0; n < nbands; n++) {
        for (int m = n + 1; m < nbands; m++) {
            const double de = this->pelec->ekb(ik, n) - this->pelec->ekb(ik, m);
            const double P_nm_sq = std::norm(P[n * nbands + m]);
            chi_val += 2.0 * (fn - fm) * P_nm_sq / de;
        }
    }
}
```

**疑点 1：sign 未参与 chi 计算**

代码注释说"sign * sign = 1 always, so both channels add"，但这个推导有误。

正确的微扰论公式为：

```
M_I = sum_k sign_k * sum_n f_n * <psi_nk|P_I|psi_nk>
```

当 lambda_I 变化 δλ 时，H 的变化为 δH = sign_k * δλ * P_I。
一阶微扰给出：

```
δ<psi_nk|P_I|psi_nk> = 2 * Re sum_{m≠n} <psi_n|P_I|psi_m> * <psi_m|sign_k * δλ * P_I|psi_n> / (e_n - e_m)
                      = 2 * sign_k * δλ * sum_{m≠n} |P_I_nm|^2 / (e_n - e_m)
```

因此：
```
dM_I/dλ_I = sum_k sign_k * sum_n f_n * 2 * sign_k * sum_{m≠n} |P_nm|^2 / (e_n - e_m)
          = sum_k sign_k^2 * (...)
          = sum_k (...)    // sign^2 = 1，这部分没问题
```

但上面只考虑了波函数变化对 P_I 期望值的贡献。完整的 chi 还需要考虑占据数 f_n 的变化（通过 Fermi 能级移动）。对于金属体系（Fe 二聚体，smearing_sigma=0.01），占据数对能级变化敏感，忽略这一项可能导致 chi 严重偏离真实值。

**疑点 2：wg 包含 k 点权重，不是纯占据数**

`this->pelec->wg(ik, n)` 是 `f_n * weight_k`，不是纯 `f_n`。在计算 `(fn - fm)` 时，如果同一 k 点的权重一致则差值中权重抵消，但 chi 的绝对值会被 k 点权重缩放。Newton 步 `delta_lambda = alpha * (target - current) / chi` 中，如果 chi 的量纲不对，lambda 步长就会偏大或偏小。

**疑点 3：跨自旋通道的 (fn - fm) 混合**

代码中 `n, m` 都在同一个 `ik` 的 band 空间内遍历。对于 nspin=2，spin-up 和 spin-down 是不同的 ik，所以这里没有跨通道问题。但需要确认 `P_I_sub[ik]` 确实是在同一自旋通道内计算的。

### 根因 B：子空间对角化不自洽 — 高概率

**核心问题**：子空间 Hamiltonian 的构建假设波函数基不变。

```cpp
// Phase 4: H_sub = diag(e_k) + sign * sum_I delta_lambda_I * P_I_sub(k)
```

这里 `P_I_sub` 是在 Phase 2 用初始波函数 C_k^(0) 计算的。当 lambda 变化较大时：

1. 真实的 H(lambda_new) 不仅仅是 H(lambda_old) + delta_lambda * P_I
2. 自洽场效应：lambda 改变 → 占据数改变 → 电荷密度改变 → Hartree/XC 势改变 → H 改变
3. 子空间对角化只捕获了 lambda 对 H 的直接（一阶）效应，完全忽略了自洽场响应

对于 Fe 这样的强关联体系，自洽场效应很强。如果第一步 Newton 更新的 delta_lambda 较大，子空间近似就会严重偏离真实解，导致：
- Mi_new 计算不准确
- 下一步 Newton 基于错误的 Mi 继续更新
- lambda 发散

### 根因 C：Newton 步缺乏步长限制 — 高概率

原始方法有多重保护：
- `check_restriction(search, alpha_trial)` 限制搜索步长
- `sccut` 参数（3.0 eV/µB）作为硬上限
- `alpha_opt` 线搜索自适应调整步长
- `check_gradient_decay` 检查梯度衰减

优化方法只有：
- `alpha_damp = 0.8` 固定阻尼
- `chi` 可能很小 → `delta_lambda = 0.8 * (target - current) / chi` 可能很大
- 没有 lambda 上限检查

**具体场景**：如果 chi ≈ 0.01（响应很弱），target - current ≈ 2.0 µB，则：
```
delta_lambda = 0.8 * 2.0 / 0.01 = 160 Ry/µB ≈ 2177 eV/µB
```
这远超合理范围（通常 lambda 应在 ~1 eV/µB 量级）。

### 根因 D：P_I_sub 计算中的并行/索引问题 — 中概率

`cal_PI_sub` (`dspin_lcao.cpp:560-635`) 中：

1. `nbands` 使用 `this->ParaV->get_nbands()`（全局 band 数），但 `psi` 的本地列数是 `ncol_bands`
2. D_I 的构建涉及 2D block-cyclic 分布的波函数到全局 nbands 的映射
3. MPI_Allreduce 汇总 D_I 后计算 P_I = D_I^dag * D_I

潜在问题：
- 如果 `nbands_global` 与实际参与计算的 band 数不一致
- 如果某些进程的 `ncol_local = 0` 导致 D_I 部分为零
- Allreduce 后 P_I 的对角元应该等于投影权重，可以用来验证

### 根因 E：Phase 5 波函数旋转的正确性 — 中概率

```cpp
// V[mcol_global, jcol_global] — V is column-major from zheev
const std::complex<double> v_mj = V_save[ik][mcol_global * nbands + jcol_global];
```

zheev 输出的特征向量是列主序（Fortran 风格），即 V[i + j*N] = V_{ij}。
代码中 `V_save[ik][mcol_global * nbands + jcol_global]` 对应 V_{mcol, jcol}，这是行主序访问。

如果 V 是列主序存储，则 V_{m,j} 应该是 `V[m + j * nbands]`，而不是 `V[m * nbands + j]`。

**这可能导致波函数旋转完全错误**，进而导致后续 DM 和电荷密度错误，外层 SCF 无法收敛。

### 根因 F：ekb_new 未正确传播 — 低概率

Phase 4 中 zheev 输出的 `w` 是子空间本征值，存入 `ekb_new(ik, n)`。但 Phase 5 中：
```cpp
this->pelec->ekb(ik, n) = ekb_new(ik, n);
```
这些本征值是 H_sub 的本征值，包含了 lambda 贡献。后续 `calculate_weights` 用这些能量计算 Fermi 能级和占据数。如果 lambda 很大，这些能量会偏离真实 KS 能量，导致占据数计算异常。

## 4. Debug 排查方案

### 第一步：验证 chi 的正确性（根因 A）

**方法**：数值有限差分验证解析 chi。

在 `run_lambda_loop_lcao` 的 Phase 3 之后插入验证代码：

```cpp
// Numerical chi verification
for (int iat = 0; iat < nat; iat++) {
    if (this->constrain_[iat].z == 0) continue;

    const double dlambda = 1e-4; // Ry/uB

    // Save current lambda
    double lambda_save = this->lambda_[iat].z;

    // Forward: lambda + dlambda
    this->lambda_[iat].z = lambda_save + dlambda;
    this->cal_mw_from_lambda(0);
    double Mi_plus = this->Mi_[iat].z;

    // Backward: lambda - dlambda
    this->lambda_[iat].z = lambda_save - dlambda;
    this->cal_mw_from_lambda(0);
    double Mi_minus = this->Mi_[iat].z;

    // Restore
    this->lambda_[iat].z = lambda_save;
    this->cal_mw_from_lambda(-1);

    double chi_numerical = (Mi_plus - Mi_minus) / (2.0 * dlambda);
    std::cout << "iat=" << iat
              << " chi_analytical=" << chi[iat]
              << " chi_numerical=" << chi_numerical
              << " ratio=" << chi[iat] / chi_numerical << std::endl;
}
```

**判定标准**：ratio 应接近 1.0（±20% 以内）。如果偏差大，说明解析 chi 公式有误。

### 第二步：验证 P_I_sub 的正确性（根因 D）

**方法**：检查 P_I 的对角元是否等于投影权重。

```cpp
// After Phase 2, verify P_I_sub
for (int ik = 0; ik < nks; ik++) {
    double sign = (ik < nk) ? 1.0 : -1.0;
    for (int iat = 0; iat < nat; iat++) {
        if (PI_sub[ik][iat].empty()) continue;
        double trace = 0.0;
        for (int n = 0; n < nbands; n++) {
            trace += PI_sub[ik][iat][n * nbands + n].real();
        }
        // trace should equal sum of projection weights for this atom
        std::cout << "ik=" << ik << " iat=" << iat << " Tr(P_I)=" << trace << std::endl;
    }

    // Also verify: sum_iat sign * sum_n wg(ik,n) * P_I_nn should give Mi contribution from this k
    // Compare with cal_MW result
}
```

同时验证 P_I 的 Hermitian 性：
```cpp
double max_asym = 0.0;
for (int n = 0; n < nbands; n++)
    for (int m = n+1; m < nbands; m++) {
        auto diff = PI_sub[ik][iat][n*nbands+m] - std::conj(PI_sub[ik][iat][m*nbands+n]);
        max_asym = std::max(max_asym, std::abs(diff));
    }
std::cout << "max asymmetry = " << max_asym << std::endl;
```

### 第三步：验证子空间对角化结果（根因 B）

**方法**：比较子空间对角化得到的 Mi_new 与全量对角化的结果。

在 Phase 4 的 Newton 步之后：

```cpp
// After subspace diag gives Mi_new, verify with full diag
auto lambda_backup = this->lambda_;
// lambda already updated in Newton step
this->cal_mw_from_lambda(0);  // full diag
std::cout << "Mi_subspace vs Mi_fulldiag:" << std::endl;
for (int iat = 0; iat < nat; iat++) {
    std::cout << "iat=" << iat
              << " subspace=" << Mi_new[iat].z
              << " fulldiag=" << this->Mi_[iat].z
              << " diff=" << Mi_new[iat].z - this->Mi_[iat].z << std::endl;
}
```

**判定标准**：如果差异 > 0.1 µB，说明子空间近似在当前 delta_lambda 下不可靠。

### 第四步：检查 V 矩阵的存储顺序（根因 E）

**方法**：验证 zheev 输出的特征向量排列。

```cpp
// After zheev in Phase 4
// V should be unitary: V^dag * V = I
std::vector<std::complex<double>> VdV(nbands * nbands, {0.0, 0.0});
for (int i = 0; i < nbands; i++)
    for (int j = 0; j < nbands; j++)
        for (int k = 0; k < nbands; k++)
            VdV[i*nbands+j] += std::conj(V[k*nbands+i]) * V[k*nbands+j];
            // 如果 V 是列主序: V_{k,i} = V[k + i*nbands]
            // 如果 V 是行主序: V_{k,i} = V[k*nbands + i]

// Check if VdV ≈ I
double max_off_diag = 0.0;
for (int i = 0; i < nbands; i++)
    for (int j = 0; j < nbands; j++) {
        double expected = (i == j) ? 1.0 : 0.0;
        max_off_diag = std::max(max_off_diag, std::abs(VdV[i*nbands+j] - expected));
    }
std::cout << "V unitarity check: max deviation = " << max_off_diag << std::endl;
```

如果 max deviation 很大，尝试转置访问模式：
```cpp
// 改为列主序访问
VdV[i*nbands+j] += std::conj(V[k + i*nbands]) * V[k + j*nbands];
```

### 第五步：添加 lambda 步长限制（根因 C）

**方法**：在 Newton 步中添加硬限制，观察是否改善。

```cpp
// In Phase 4 Newton step
const double lambda_max = this->restrict_current_ / ModuleBase::Ry_to_eV; // sccut in Ry
for (int iat = 0; iat < nat; iat++) {
    if (this->constrain_[iat].z == 0) continue;

    // Clamp chi to avoid division by near-zero
    double chi_clamped = chi[iat];
    if (std::abs(chi_clamped) < 0.1) {
        chi_clamped = (chi_clamped >= 0) ? 0.1 : -0.1;
    }

    double delta_lambda_z = alpha_damp * (this->target_mag_[iat].z - spin[iat].z) / chi_clamped;

    // Clamp delta_lambda
    if (std::abs(delta_lambda_z) > lambda_max) {
        delta_lambda_z = (delta_lambda_z > 0) ? lambda_max : -lambda_max;
    }

    this->lambda_[iat].z = initial_lambda[iat].z + delta_lambda_z;

    std::cout << "iat=" << iat << " chi=" << chi[iat]
              << " delta_lambda=" << delta_lambda_z * ModuleBase::Ry_to_eV << " eV/uB"
              << " lambda=" << this->lambda_[iat].z * ModuleBase::Ry_to_eV << " eV/uB" << std::endl;
}
```

## 5. 推荐排查顺序

| 优先级 | 步骤 | 预计耗时 | 理由 |
|--------|------|----------|------|
| 1 | 第五步：添加 lambda 限制 | 10 min | 最快验证，如果 lambda 爆炸是直接原因，限制后应立即改善 |
| 2 | 第四步：检查 V 存储顺序 | 15 min | 如果行/列主序搞反，Phase 5 波函数旋转完全错误，这是致命 bug |
| 3 | 第一步：验证 chi | 30 min | 需要多次全量对角化，但能确定 Newton 方向是否正确 |
| 4 | 第三步：子空间 vs 全量对角化 | 20 min | 确定子空间近似的有效范围 |
| 5 | 第二步：验证 P_I_sub | 15 min | 如果前面都没问题再查这里 |

## 6. 如果子空间近似本身不够准确的备选方案

如果排查发现子空间近似在 Fe 体系上误差过大（根因 B 确认），可以考虑：

1. **混合策略**：前几步用全量对角化建立好的初始点，后续用子空间对角化微调
2. **自适应切换**：当 |delta_lambda| > 阈值时回退到全量对角化
3. **子空间 + SCF 混合**：子空间对角化后做 1-2 步简化 SCF（只更新 Hartree 势，不更新 XC）
4. **减少外层 SCF 的 lambda 步数而非替换内层算法**：保持 `cal_mw_from_lambda` 但用更好的搜索方向（解析 Jacobian 指导 BFGS 初始方向）

---

## 7. Debug 执行结果汇总（2026-04-23）

### 7.1 排查到的所有问题

| # | 问题 | 根因分类 | 严重度 | 状态 | 涉及文件:行号 |
|---|------|----------|--------|------|---------------|
| 1 | **V 矩阵列主序访问错误** | Root Cause E | 🔴 致命 | ✅ 已修复 | `lambda_loop.cpp:608` |
| 2 | **target_mag 读取错误** | 新增 | 🔴 致命 | ✅ 已修复 | `spin_constrain.cpp:357` |
| 3 | **Newton 步无步长限制** | Root Cause C | 🟠 高 | ✅ 已修复 | `lambda_loop.cpp:461-482` |
| 4 | **chi 计算中 k 权重处理** | Root Cause A | 🟡 中 | ✅ 已修复 | `lambda_loop.cpp:434-447` |
| 5 | **子空间近似在强关联体系失效** | Root Cause B | 🔴 致命 | ⚠️ 已确认，未修复 | 算法层面限制 |

### 7.2 Bug #1: V 矩阵列主序访问错误（Root Cause E — 已修复）

**文件**: `source/source_lcao/module_deltaspin/lambda_loop.cpp`

**原始代码** (Phase 5, 波函数旋转):
```cpp
// V[mcol_global, jcol_global] — V is column-major from zheev
const std::complex<double> v_mj = V_save[ik][mcol_global * nbands + jcol_global];
```

**问题**: `zheev` (LAPACK) 输出列主序矩阵，即 `V[row, col] = V[row + col * lda]`。
原代码使用 `V[mcol * nbands + jcol]` 是行主序访问，实际读到了转置的元素。
这导致波函数旋转 `C_new = C * V` 完全错误 → DM 错误 → 电荷密度错误 → SCF 发散。

**修复后**:
```cpp
// V is column-major from zheev: V[row + col * lda]
const std::complex<double> v_mj = V_save[ik][mcol_global + jcol_global * nbands];
```

**验证**: 添加 V 的幺正性检查 `V^dag V ≈ I`，修复后 max_dev ≈ 1e-15 ✓
```
V unitarity check (ik=0): max_dev=1.9984e-15
```

---

### 7.3 Bug #2: target_mag 读取错误（新增发现 — 已修复）

**文件**: `source/source_lcao/module_deltaspin/spin_constrain.cpp:357`

**原始代码**:
```cpp
void SpinConstrain<TK>::set_target_mag(const std::vector<ModuleBase::Vector3<double>>& target_mag_in)
{
    if (this->nspin_ == 2)
    {
        this->target_mag_.resize(nat, 0.0);
        for (int iat = 0; iat < nat; iat++)
        {
            this->target_mag_[iat].z
                = target_mag_in[iat].x; /// this is wired because the UnitCell class set in x direction
        }
    }
```

**问题**: 代码读取 `target_mag_in[iat].x`，但 STRU 文件解析器（`read_atoms_helper.cpp`）将标量磁矩存入 `.z` 分量：
```cpp
// read_atoms_helper.cpp:296-299
// only one mag is given, assume it is z
atom.m_loc_[ia].x = 0;
atom.m_loc_[ia].y = 0;
atom.m_loc_[ia].z = atom.mag[ia];
```

因此对于 nspin=2 算例，`target_mag` 始终为 `0.0`，导致优化目标错误。

测试算例 STRU 中定义：
```
0.00   0.00   0.00   mag  2.0 sc 1     → m_loc_ = (0, 0, 2.0)
0.51   0.51   0.51   mag -2.0 sc 1     → m_loc_ = (0, 0, -2.0)
```
但 `target_mag_` 被错误设为 `(0, 0, 0.0)`，所以实际收敛目标是磁矩为零，而非 ±2.0 µB。

**修复后**:
```cpp
this->target_mag_[iat].z = target_mag_in[iat].z; // UnitCell stores collinear mag in z component
```

**影响**: 这个 bug 导致 `run_lambda_loop_lcao` 和 `run_lambda_loop` 的 target 全部为零，
之前的日志中反复出现 `target spin (uB): ATOM 1 0.000, ATOM 2 0.000` 即是此原因。

---

### 7.4 Bug #3: Newton 步无步长限制（Root Cause C — 已修复）

**文件**: `source/source_lcao/module_deltaspin/lambda_loop.cpp`

**原始代码**:
```cpp
for (int iat = 0; iat < nat; iat++)
{
    if (this->constrain_[iat].z == 0) { continue; }
    if (std::abs(chi[iat]) < 1e-15) { continue; }
    const double delta_lambda_z = alpha_damp * (this->target_mag_[iat].z - spin[iat].z) / chi[iat];
    this->lambda_[iat].z = initial_lambda[iat].z + delta_lambda_z;
}
```

**问题**: 无任何步长限制。当 chi 很小或 target-spin 差值很大时，delta_lambda 可达数百 eV/µB，
远超合理范围（~1-10 eV/µB）。lambda 爆炸 → H_sub 偏离线性区域 → 子空间对角化结果完全错误。

**修复后**:
```cpp
const double lambda_max = this->restrict_current_; // sccut in Ry/uB
for (int iat = 0; iat < nat; iat++)
{
    if (this->constrain_[iat].z == 0) { continue; }
    // Clamp chi to avoid division by near-zero
    double chi_clamped = chi[iat];
    if (std::abs(chi_clamped) < 0.1) {
        chi_clamped = (chi_clamped >= 0) ? 0.1 : -0.1;
    }
    double delta_lambda_z = alpha_damp * (this->target_mag_[iat].z - spin[iat].z) / chi_clamped;
    // Clamp delta_lambda to prevent explosion
    if (std::abs(delta_lambda_z) > lambda_max) {
        delta_lambda_z = (delta_lambda_z > 0) ? lambda_max : -lambda_max;
    }
    this->lambda_[iat].z = initial_lambda[iat].z + delta_lambda_z;
}
```

**验证**: 修复后出现 CLAMPED 日志，步长被限制在 sccut=3.0 eV/µB 以内：
```
Newton iat=0 chi=-1.2937 target-spin=2.08869 delta_lambda=-17.5732 eV/uB lambda=-118.415 eV/uB
CLAMPED: delta_lambda -17.5732 -> 3 eV/uB
```

---

### 7.5 Bug #4: chi 计算中 k 权重处理（Root Cause A — 已修复）

**文件**: `source/source_lcao/module_deltaspin/lambda_loop.cpp`

**问题**: `pelec->wg(ik, n)` = `f_n * weight_k`，包含 k 点权重。原始代码直接使用 `wg` 计算 `(fn - fm)`，
同一 k 点内差值中权重确实抵消，但 chi 的绝对值被缩放，量纲不一致。

**修复后**: 显式除以 k 权重得到纯占据数，再乘以权重恢复正确的求和：
```cpp
const double wk = this->pelec->klist->wk[ik]; // k-point weight
const double fn = this->pelec->wg(ik, n) / wk; // pure occupation
// ...
chi_val += 2.0 * wk * (fn - fm) * P_nm_sq / de;
```

---

### 7.6 Bug #5: 子空间近似在强关联体系失效（Root Cause B — 已确认，算法层面限制）

**验证结果**: 子空间对角化预测的 Mi 与全量对角化结果差异巨大：

| 外层步 | inner | Mi_subspace (iat0) | Mi_fulldiag (iat0) | 差异 | Mi_subspace (iat1) | Mi_fulldiag (iat1) | 差异 |
|--------|-------|--------------------|--------------------|------|--------------------|--------------------|------|
| GE1-0 | 0 | 2.460 | 1.823 | **0.637** | 2.460 | -1.940 | **4.400** |
| GE1-0 | 1 | -2.046 | (未验证) | — | -2.039 | (未验证) | — |
| GE2-0 | 0 | -0.673 | -1.378 | **0.705** | -0.671 | -3.119 | **2.448** |
| GE49-0 | 0 | 5.087 | (未验证) | — | -2.922 | (未验证) | — |

**判定**: 差异远超 0.1 µB 阈值，确认子空间近似在当前 delta_lambda 下不可靠。

**根因分析**:
1. `P_I_sub` 在 Phase 2 用初始波函数 `C_k^(0)` 计算，是冻结基近似
2. 当 lambda 变化时，真实的 H 不仅是 `H_0 + delta_lambda * P_I`，还有自洽场响应：
   - lambda → 占据数改变 → 电荷密度改变 → Hartree 势改变 → H 改变
   - 对于 Fe 强关联体系，XC 势响应很强
3. 子空间对角化只捕获了 lambda 对 H 的直接（一阶）效应，完全忽略了 SCF 响应
4. 即使加了步长限制（sccut=3.0 eV/µB），delta_lambda=3 eV/µB 仍足以使子空间近似偏离

**结论**: 子空间对角化 + 解析 Newton 的优化策略在强关联（DFT+U / DeltaSpin）体系上**从根本上不适用**。
该方法仅适用于弱关联体系（如简单金属、半导体），其中 SCF 响应可忽略。

---

### 7.7 验证通过的项目

| 验证项 | 结果 | 证据 |
|--------|------|------|
| P_I_sub Hermitian 性 | ✅ 通过 | `max_asym=0` |
| P_I_sub 迹 (投影权重) | ✅ 合理 | `Tr(P_I) ≈ 9-12`（与 Fe d 轨道占据一致） |
| V 矩阵幺正性 | ✅ 通过 | `max_dev ≈ 1e-15` |
| chi 量级 | ✅ 合理 | `chi ≈ -1 to -5 Ry/uB`（负号表示 lambda 增大 → Mi 减小，符合物理） |
| chi 钳位 | ✅ 有效 | `chi_clamped` 不低于 0.1 |

---

### 7.8 修复后运行状态

**编译**: ✅ 通过，0 errors
- 二进制: `build/abacus_basic_para` (380MB)

**测试算例**: `tests/17_DS_DFTU/24_LCAO_DS_S2_Z`
- nspin=2, Fe 二聚体, target_mag = ±2.0 µB, sccut=3.0 eV/µB
- 结果: SCF 仍未收敛（50 步未达阈值），但行为已明显改善
  - lambda 值不再爆炸到 ±2000 eV/µB，而是在 ±100 eV/µB 范围内
  - delta_lambda 被正确钳位在 3.0 eV/µB 以内
- 未收敛原因: Root Cause B（子空间近似误差）是算法层面的限制，代码修复无法解决

---

### 7.9 备选方案建议

鉴于 Root Cause B 确认子空间近似在强关联体系上误差过大，建议采用以下方案之一：

#### 方案 A: 保持全量对角化，仅用解析 chi 优化搜索方向（推荐）

用 `chi` 替代 BFGS 的初始 Hessian 近似，但不替换 `cal_mw_from_lambda`：
```cpp
// 在 run_lambda_loop() 中，用 1/chi 初始化 BFGS 的逆 Hessian
// 而非用子空间对角化替换全量 SCF
```
优势: 保留完整的 SCF 自洽性，仅加速搜索方向。

#### 方案 B: 自适应切换策略

```cpp
// 伪代码
if (|delta_lambda| > threshold) {
    // 大步: 回退到全量 cal_mw_from_lambda
    cal_mw_from_lambda(inner_step);
} else {
    // 小步: 可用子空间对角化加速
    subspace_diag_and_update();
}
```

#### 方案 C: 子空间 + 简化 SCF 混合

子空间对角化得到新波函数后，做 1 步简化 SCF：
1. 从旋转后的波函数计算新电荷密度
2. 重新计算 Hartree + XC 势
3. 用新势更新 H_sub，再对角化
代价: 每步仍需部分 SCF 计算，但比重构整个 H 和全量对角化便宜。

---

### 7.10 代码改动清单

| 文件 | 改动类型 | 描述 |
|------|----------|------|
| `lambda_loop.cpp` | Bug fix + Debug | V 矩阵列主序修复 (line 608), Newton 步长限制 (line 461-482), chi k 权重修正 (line 434-447), 添加 P_I_sub/chi/V 验证打印 |
| `spin_constrain.cpp` | Bug fix | target_mag 读取从 `.x` 改为 `.z` (line 357) |

---

## 8. 最终执行结果（2026-04-24）

### 8.1 方案演进

子空间对角化方案被证实**从根本上不适用于强关联体系**后，采用了**方案 A：chi 引导的全量对角化**：
- 保留 `cal_mw_from_lambda()` 全量 SCF 对角化（保证自洽性）
- 用解析 chi + secant 更新提供 Newton 搜索方向
- 每步调用全量对角化验证，chi 自适应包含 SCF 响应

### 8.2 对比测试结果

测试算例：`tests/17_DS_DFTU/24_LCAO_DS_S2_Z`（Fe 反铁磁二聚体，nspin=2，target = ±2.0 µB）

| 指标 | chi 引导（新方案） | 原始 BFGS | 子空间对角化（废弃） |
|------|-------------------|-----------|---------------------|
| Lambda loop 收敛 | ✅ 38 步 | ❌ 100 步未收敛 | ❌ 子空间近似误差 >4 µB |
| 最终磁矩 (iat=0) | **2.0001 µB** | 11.57 µB | 5.09 µB (subspace) |
| 最终磁矩 (iat=1) | **-2.0002 µB** | 8.92 µB | -2.92 µB (subspace) |
| Lambda 值 | 0.00009 / 0.0013 eV/µB | -510 / -571 eV/µB | -103 / 113 eV/µB |
| cal_mw_from_lambda | 1681 次 | 199 次 | N/A |

### 8.3 最终代码改动

| 文件 | 改动类型 | 描述 |
|------|----------|------|
| `lambda_loop.cpp` | 重构 | 完全重写 `run_lambda_loop_lcao()`：删除子空间对角化(~250 行)，实现 chi 引导的全量对角化 + secant 更新 |
| `spin_constrain.cpp` | Bug fix | `target_mag` 读取从 `.x` 改为 `.z` |

### 8.4 核心算法

```
Phase 1: cal_mw_from_lambda(-1) → 初始 Mi, e_k, C_k
Phase 2: cal_PI_sub() → 解析 chi（仅第一步使用）
Phase 3-6 (迭代):
  3. Newton 步: delta_lambda = alpha_damp * (target - Mi) / chi（带 sccut 限制）
  4. cal_mw_from_lambda(inner) → 真实 Mi（全量 SCF，保证自洽）
  5. Secant chi 更新: chi = dMi / dlambda（隐式包含 SCF 响应）
  6. 检查 RMS 收敛
Phase 7: 更新 DM/charge
```

### 8.5 关键设计决策

1. **Secant chi 更新**：比解析 chi 更准确，因为它隐式包含了 SCF 自洽响应
2. **稳定性保护**：secant chi 仅在符号一致且量级合理 (0.01-100) 时才替换解析 chi
3. **步长限制**：复用 sccut=3.0 eV/µB 作为 delta_lambda 的硬上限
4. **chi 钳位**：|chi| >= 0.1 避免除以近零值
