# DeltaP SCF 约束计算：设计与测试计划

> **日期**: 2026-07-08
> **分支**: `feat/deltap-wilson-per-atom`
> **目标**: 将分支选择后的逐原子极化加入 SCF 迭代，验证哈密顿量修正和内循环收敛
> **前序**: `2026-07-08-deltap-branch-set-design.md`（分支集合方法）, `2026-07-08-deltap-branch-choice-impact.md`（分支选择影响）

---

## 1. 架构

顺序更新模式：每个 SCF 步计算一次 γ^I、更新一次 λ，SCF 密度与 lambda 同步收敛。

```
SCF iter loop:
  1. Build H = H_KS + H^λ                    ← DeltaPOperator::contributeHR() (已有)
  2. Diagonalize → ψ_{n,k}                   ← 已有
  3. iter_finish:
     a. [首次] 初始化 Wilson loop 基础设施    ← 新增
     b. 计算 Wilson loop(ψ) → {λ_n} → γ^I   ← 新增: compute_gamma_scf()
     c. 分支选择: γ^I_eff = nearest(γ^I, prev) ← 已有
     d. 更新: λ^I += step · (γ^I_eff - target) ← 新增
     e. dp_op->set_lambda(λ)                 ← 已有接口
  4. 收敛检查: |γ^I - target| < tol AND SCF收敛
```

---

## 2. 组件设计

### 2.1 组件 1: SCF 内 Wilson loop 计算

**新增方法** `DeltaP::compute_gamma_scf()`：

是 `compute_wannier_polarization()` 的轻量版，仅计算 γ^I，不写文件、不计算 Resta-Z。

```cpp
/// Lightweight Wilson loop computation for SCF inner loop.
/// Returns per-atom Berry phase gamma_I with branch selection.
void DeltaP::compute_gamma_scf(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const elecstate::ElecState* pelec);
```

内部流程：
1. 首次调用时：`setup_kstring()`、`compute_real_overlaps()`、`compute_smo_overlap_matrix()`（复用已有代码）
2. 每次调用：
   - 遍历 k-strings，构建 Wilson loop 矩阵（使用 `berry_overlap_` 计算重叠矩阵）
   - 对角化 → 特征值 λ_n
   - 计算 SMO 权重 w^I_n
   - Zeta 缩放 + 分支选择（使用 `W_prev_` 作为 prev_gamma）
   - 平均 k-strings → γ^I
   - 更新 `W_prev_`（供下一步使用）
3. 结果存入 `results_.gamma_I`，不调用 `write_results()`

**初始化问题**：Wilson loop 需要 `unkOverlap_lcao`（berry phase 重叠矩阵），当前只在 NSCF 路径初始化。解决方案：

在 `ESolver_KS_LCAO` 中新增成员：
```cpp
// deltap SCF members
std::unique_ptr<deltap::DeltaP> dp_scf_;
unkOverlap_lcao berry_ovl_scv_;
cal_r_overlap_R r_overlap_scf_;
bool deltap_scf_initialized_ = false;
```

首次 `iter_finish` 调用时初始化（复用 `ctrl_scf_lcao.cpp:370-397` 的逻辑）：
```cpp
if (!deltap_scf_initialized_)
{
    // Build orb_onsite if needed
    if (!two_center_bundle.overlap_orb_onsite)
    {
        two_center_bundle.build_orb_onsite(PARAM.inp.deltap_rm);
        two_center_bundle.tabulate();
    }
    // Build position matrix and berry overlap
    r_overlap_scf_.init(ucell, *this->pv, this->orb_);
    berry_ovl_scv_.init(ucell, this->kv.get_nkstot(), this->orb_);
    berry_ovl_scv_.cal_R_number(ucell, this->gd);
    berry_ovl_scv_.cal_orb_overlap(ucell);
    // Create DeltaP
    dp_scf_ = std::make_unique<deltap::DeltaP>();
    dp_scf_->init(ucell, this->gd, this->kv,
                  two_center_bundle.overlap_orb_onsite.get(),
                  two_center_bundle.overlap_orb.get(),
                  two_center_bundle.overlap_onsite_onsite.get(),
                  this->orb_.cutoffs(),
                  PARAM.inp.deltap_rm, PARAM.inp.deltap_gdir,
                  this->pv, &r_overlap_scf_, &berry_ovl_scv_);
    dp_scf_->load_branch();  // load prev from file if exists
    deltap_scf_initialized_ = true;
}
```

### 2.2 组件 2: Lambda 更新

替换 `esolver_ks_lcao.cpp:663-688` 的占位代码：

```cpp
// 3c) DeltaP SCF constraint
if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
{
    auto* dp_op = hamilt_lcao->get_dp_operator();
    if (!dp_op) WARNING_QUIT("ESolver_KS_LCAO", "dp_op is null");

    // Initialize Wilson loop infrastructure (first call only)
    if (!deltap_scf_initialized_)
    { /* ... see 2.1 ... */ }

    // Compute per-atom Berry phase with branch selection
    dp_scf_->compute_gamma_scf(ucell, this->psi, this->pelec);

    // Get results
    const int alpha = PARAM.inp.deltap_gdir - 1;
    const auto& gamma_I = dp_scf_->get_results().gamma_I;

    // Lambda update: gradient descent
    std::vector<double> lambda = dp_op->get_lambda();
    double step = PARAM.inp.deltap_lambda_step;
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        double gamma = gamma_I[iat][alpha];
        double target = deltap_target_[iat];
        double dlambda = step * (gamma - target);
        lambda[iat] += dlambda;
    }
    dp_op->set_lambda(lambda);  // triggers HR recompute

    // Print status
    std::cout << " [DeltaP] iter=" << iter;
    for (int iat = 0; iat < ucell.nat; ++iat)
        std::cout << " gamma=" << gamma_I[iat][alpha]
                  << " lambda=" << lambda[iat];
    std::cout << std::endl;
}
```

**算符工作机制**（已有，无需修改）：

`DeltaPOperator::contributeHR()` 计算增量：
```
dlambda = lambda_[iat] - lambda_save_[iat]
coeff = dlambda * tau_alpha
HR += coeff * pre_hr[iat]  (即 dlambda * tau * P̂^I)
```

- 首次：lambda_save_=0，dlambda=lambda，应用完整 lambda
- 后续：dlambda=lambda-lambda_save，仅应用增量
- `set_lambda()` 设置 `dp_hr_done=false`，触发下次 `contributeHR()` 重算

### 2.3 组件 3: 输入参数

**已有参数**（`input_parameter.h:625-627`）：
- `deltap_corr` (bool): 启用 SCF 修正
- `deltap_lambda_step` (double): 更新步长，默认 0.5
- `deltap_nscf` (int): 最大内循环数（顺序模式不用）

**新增参数**：
- `deltap_target_file` (string): 目标 γ^I 文件路径，每行一个原子

文件格式 `deltap_target.dat`：
```
# Per-atom target Berry phase gamma^I
# atom_index  gamma_target
0  0.82
1  -0.68
2  -0.20
```

读取逻辑在 `iter_finish` 首次调用时执行。

### 2.4 算符 gdir 修正

当前 `DeltaPOperator` 构造时 `gdir_=3`（硬编码）。在 `hamilt_lcao.cpp:422` 后添加：
```cpp
this->dp_operator->set_gdir(PARAM.inp.deltap_gdir);
```

---

## 3. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `source/source_lcao/module_deltap/deltap.h` | +`compute_gamma_scf()` 声明 |
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | +`compute_gamma_scf()` 实现（轻量版 Wilson loop） |
| `source/source_esolver/esolver_ks_lcao.h` | +`dp_scf_` 等成员 |
| `source/source_esolver/esolver_ks_lcao.cpp` | 替换 iter_finish 占位代码 |
| `source/source_lcao/hamilt_lcao.cpp` | +`set_gdir()` 调用 |
| `source/source_io/module_parameter/input_parameter.h` | +`deltap_target_file` |
| `source/source_io/module_parameter/read_input_item_other.cpp` | +`deltap_target_file` 读取 |

---

## 4. 测试计划

### T1: λ=0 回归测试

**目的**：验证 `deltap_corr=1` + λ=0 不影响基态。

**方法**：
1. 标准 SCF（`deltap_corr=0`）→ 记录总能量 E₀、密度 ρ₀
2. `deltap_corr=1`，λ=0（不设 target 或 target=γ^I_0）→ 记录 E₁、ρ₁
3. 比较：|E₁ - E₀| < 10⁻⁶ Ha

**体系**：H₂O 分子，4×4×4 k-mesh

**INPUT**：
```
deltap_switch    1
deltap_corr      1
deltap_method    wannier
deltap_rm        3.0
deltap_gdir      3
deltap_gauge_mode    smo_anchored
deltap_lambda_step  0.0    # zero step = no lambda update
```

**通过判据**：
- 总能量差异 < 10⁻⁶ Ha
- SCF 收敛步数一致
- 日志显示 `dlambda=0`（HR 未被修改）

### T2: 线性响应测试

**目的**：验证 H^λ 能有效驱动 γ^I 变化，且方向正确。

**方法**：
1. 标准 SCF → 记录 γ^I_0（从 NSCF 后处理或 SCF 内计算）
2. 固定 λ_O = 0.1（其他 λ=0），跑一个 SCF 步 → 记录 γ^I_1
3. 固定 λ_O = 0.2 → γ^I_2
4. 检查 Δγ^I = γ^I(λ) - γ^I(0) ∝ λ

**体系**：H₂O

**INPUT**：手动设置 λ（通过代码或 `deltap_lambda_step=0` + 初始 lambda 文件）

**通过判据**：
- γ^O 随 λ_O 单调变化
- R² > 0.9 的线性关系
- 其他原子的 γ^I 变化远小于 γ^O（局域性）

### T3: 约束收敛测试

**目的**：验证 lambda 内循环能收敛到使 γ^I = target。

**方法**：
1. 标准 SCF → γ^I_0
2. 设 target = γ^I_0 + δ（δ=0.1 for O，0 for others）
3. `deltap_corr=1`, `deltap_lambda_step=0.1` → 跑 SCF
4. 监控 λ^I 和 γ^I 的收敛

**体系**：H₂O

**通过判据**：
- λ^O 收敛到非零值（|Δλ| < 10⁻⁴）
- γ^O → target（偏差 < 5%）
- 其他 λ^I ≈ 0（未约束）
- SCF 在 50 步内收敛

### T4: 分支连续性测试

**目的**：验证分支选择在 SCF 步间保持一致（k_sum=0）。

**方法**：在 T3 的运行中，检查每步的分支选择日志。

**通过判据**：
- 所有 SCF 步的 k_sum = 0（无 2π 跳变）
- γ^I 连续变化（步间变化 < π）

---

## 5. 风险与回退

| 风险 | 概率 | 回退方案 |
|------|------|---------|
| Wilson loop 在 SCF 内计算过慢 | 中 | 减少 k-string 数量（只取第一条 string） |
| 算符-变量不一致（λ 驱动方向错误） | 中 | T2 线性响应测试检测，若方向错误则翻转 step 符号 |
| SCF 不收敛 | 高 | 减小 `deltap_lambda_step`（0.1→0.01），加 mixing |
| nocc 计算错误 | 已知 | H₂O nocc=4 正确（已验证）；液态水需单独修复 |

---

## 6. 实现顺序

1. **组件 3**（参数）→ 最简单，无依赖
2. **组件 1**（compute_gamma_scf）→ 核心，复用已有 Wilson loop 代码
3. **组件 2**（lambda 更新）→ 依赖组件 1
4. **T1**（λ=0 回归）→ 验证不破坏基态
5. **T2**（线性响应）→ 验证算符驱动
6. **T3**（收敛）→ 验证内循环
7. **T4**（分支连续）→ 验证分支选择
