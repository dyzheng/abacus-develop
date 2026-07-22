# DeltaP 内循环嵌套 SCF 设计 (2026-07-10)

## 1. 目标

将当前同步 λ 更新方案（每 SCF 步更新一次 λ）改为嵌套 SCF 方案：
内循环优化 λ（冻结电荷密度），外循环收敛电荷密度。消除 SCF 振荡，实现 gamma → target 收敛。

参考：deltaspin 的 `lambda_loop.cpp` + `hamilt2rho_single()` 嵌套模式。

## 2. 架构

```
外循环 (iter = 1..maxniter)           ← 电荷密度收敛
├─ iter_init()                         ← 电荷混合、DFT+U、EXX
├─ hamilt2rho_single()
│   ├─ if deltap_active && drho < deltap_inner_thr:
│   │   run_deltap_inner_loop(iter)
│   │       for inner_step = 0..nsc-1:
│   │           1. compute gamma via Wilson loop
│   │           2. compute residual: R = gamma - target
│   │           3. BFGS: Δλ = optimize(R, history)
│   │           4. λ = initial_lambda + Δλ
│   │           5. compute_hk_correction(λ) → update operator
│   │           6. rebuild H(k) 各 k 点（含 HK correction）
│   │           7. HSolver.solve(skip_charge=true) → 新 psi
│   │           8. check: max|gamma-target| < deltap_conv_thr → break
│   │       HSolver.solve(skip_charge=false) [if needed] → update rho
│   │   skip_solve = true
│   └─ else:
│       normal HSolver diagonalization
└─ iter_finish()                       ← 电荷混合、报告
```

## 3. 模块划分

### 3.1 BFGS 优化器 (新建)

从 deltaspin `lambda_loop.cpp` 抽象通用 BFGS 共轭梯度优化器。

```
source/module_optimizer/bfgs.h
source/module_optimizer/bfgs.cpp
```

接口：
```cpp
class BFGS {
public:
    // n_dim: 优化变量维度 (例如 nat, 每原子 1 个 λ)
    void init(int n_dim, double initial_alpha, double decay_grad);

    // 每次 iter_finish 开始时重置内循环状态
    void reset_inner();

    // 返回下一步的 Δλ 增量
    // total_lambda: 输出，完整的 λ = λ₀ + Δλ
    // residual: γ - target (目标函数残差)
    // step: 内循环步号 (0, 1, 2, ...)
    // converged: 输出，是否收敛
    void next_step(const std::vector<double>& residual,
                   int step,
                   std::vector<double>& total_lambda,
                   bool& converged);

private:
    void check_restriction(std::vector<double>& search, double& alpha);
    double line_search(...);
    // Polak-Ribiere conjugate gradient state
};
```

### 3.2 DeltaP 内循环 (修改 DeltaP 类)

新增方法：`inner_loop_step()`

```
deltap_wannier.cpp → inner_loop_gamma_scf()
```

职责：
- 在固定电荷密度下调用 HSolver 对角化
- 计算 Wilson loop gamma
- 调用 BFGS 更新 λ
- 循环直到收敛或达到 max 步数

### 3.3 esolver 修改 (修改 ESolver_KS_LCAO)

`esolver_ks_lcao.cpp → hamilt2rho_single()`:
- 添加 deltap 内循环激活逻辑
- skip_solve 机制
- 从 iter_finish 中移除 lambda 更新代码（改为内循环负责）

## 4. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deltap_inner_thr` | 1.0e-4 | drho 阈值，低于此值启动内循环 |
| `deltap_nsc` | 5 | 内循环最大步数 |
| `deltap_conv_thr` | 1.0e-3 | |gamma-target| 收敛阈值 |
| `deltap_bfgs_init_step` | 1.0 | BFGS 初始步长 (Ry) |

## 5. 数据流

```
iter_finish (每次外循环结束):
  仅报告 gamma、lambda、收敛状态
  不更新 lambda（由内循环负责）

hamilt2rho_single (外循环的哈密顿量到电荷映射):
  if deltap_active && drho < deltap_inner_thr:
    // ---- 内循环 ----
    BFGS.reset_inner()
    for inner_step = 0..deltap_nsc-1:
      1. if inner_step > 0:
           // 使用上一步的 λ 重建 H(k) 各 k 点
           更新 operator 中的 hk_correction
           重建每个 k 点的 HK
      2. HSolver.solve(skip_charge=true) → 新 psi
      3. compute_gamma_scf → γ_I
      4. residual = γ_I - target
      5. check convergence: max|residual| < deltap_conv_thr → break
      6. BFGS.next_step(residual, inner_step, λ, converged)
      7. compute_hk_correction(λ) → 存入 operator (供下一步使用)
    // ---- 内循环结束 ----
    // 用最终收敛的 λ 更新电荷密度
    HSolver.solve(skip_charge=false) → final psi + rho
    skip_solve = true

  else:
    正常 HSolver 对角化（不内循环）

iter_finish (同步模式的 lambda 更新仅保留为 fallback):
  if (!deltap_inner_active):
    旧同步模式: lambda += step * (gamma - target)
                  compute_hk_correction(λ)
```

## 6. 测试计划

1. H₂O 小体系验证：设置 target = γ₀ + 0.01，验证内循环收敛
2. Lambda continuity 测试（同 2026-07-10-deltap-lambda-continuity-test.md）
3. 并行模式测试（2 process MPI）

## 7. 风险

- BFGS 抽象可能引入接口不匹配
- skip_charge 模式下的 HSolver 需要兼容 deltap 的 k-dependent 操作符
- 现有 iter_finish 中的 lambda 更新逻辑需要安全迁移

## 8. 文件修改清单

| 文件 | 修改 |
|------|------|
| `source/module_optimizer/bfgs.h` | 新建 BFGS 类 |
| `source/module_optimizer/bfgs.cpp` | 新建 BFGS 实现 |
| `source/source_lcao/module_deltap/deltap.h` | 添加 inner_loop_step 声明 |
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | 实现内循环 |
| `source/source_esolver/esolver_ks_lcao.cpp` | hamilt2rho_single 添加内循环激活 |
| `source/source_esolver/esolver_ks_lcao.cpp` | iter_finish 迁移 lambda 更新到内循环 |
| `source/source_io/module_parameter/input_parameter.h` | 添加新参数 |
| `source/source_io/module_parameter/read_input_item_other.cpp` | 参数读取 |
