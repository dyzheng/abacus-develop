# DeltaQS 统一框架实现文档

## 概述

本文档记录了将 DeltaSpin（自旋约束 DFT）扩展为 DeltaQS（电荷-自旋联合约束 DFT）的完整实现。DeltaQS 将 DeltaSpin 和 DeltaQ 统一为同一 Lagrange 框架在两个正交子空间上的投影，实现了电荷和磁矩的联合约束与基态优化。

## 理论基础

### 统一 Lagrange 泛函

$$E_{\text{total}} = E_{KS} + \sum_I \mu_I (N_I - N_I^{\text{target}}) + \sum_I \lambda_I (M_I - M_I^{\text{target}})$$

有效势：
$$v_{\text{eff}}^\alpha = v_{KS}^\alpha + \sum_I (\mu_I + \lambda_I) w_I$$
$$v_{\text{eff}}^\beta = v_{KS}^\beta + \sum_I (\mu_I - \lambda_I) w_I$$

其中 μ_I 是电荷 Lagrange 乘子，λ_I 是自旋 Lagrange 乘子。

### 三个模式的关系

| 模式 | μ | λ | 有效势修正 |
|------|---|---|-----------|
| DeltaSpin | 0 | λ | α: +λw, β: -λw |
| DeltaQ | μ | 0 | α: +μw, β: +μw |
| DeltaQS | μ | λ | α: (μ+λ)w, β: (μ-λ)w |

### 梯度信息（包络定理）

$$\frac{\partial E}{\partial N_I^{\text{target}}} = -\mu_I, \qquad \frac{\partial E}{\partial M_I^{\text{target}}} = -\lambda_I$$

一次 SCF 计算同时给出 E 和梯度 (−μ, −λ)，不需要有限差分。

## 文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `source/source_lcao/module_deltaspin/deltaqs.cpp` | DeltaQS 全部 S1-S6 功能实现 |

### 修改文件

| 文件 | 变更说明 |
|------|---------|
| `source/source_io/module_parameter/input_parameter.h` | 新增 9 个 DeltaQS INPUT 参数 |
| `source/source_io/module_parameter/read_input_item_other.cpp` | 注册新参数的解析、验证和文档 |
| `source/source_cell/atom_spec.h` | Atom 结构体新增 `target_charge`, `mu`, `constrain_charge` 字段 |
| `source/source_cell/read_atoms_helper.cpp` | 分配新字段 + STRU 解析 `tc`/`cq`/`mu` 关键词 |
| `source/source_cell/unitcell.h` | 新增 3 个 getter 方法声明 |
| `source/source_cell/unitcell.cpp` | 实现 `get_target_charge()`, `get_mu()`, `get_constrain_charge()` |
| `source/source_lcao/module_deltaspin/spin_constrain.h` | SpinConstrain 类扩展：数据成员、方法声明 |
| `source/source_lcao/module_deltaspin/spin_constrain.cpp` | `cal_escon()` 增加电荷约束能量项 |
| `source/source_lcao/module_deltaspin/deltaspin_lcao.cpp` | Facade 层集成 DeltaQS 初始化和联合循环 |
| `source/source_lcao/module_deltaspin/template_helpers.cpp` | `double` 特化的所有新方法 no-op stubs |
| `source/source_lcao/module_deltaspin/CMakeLists.txt` | 添加 `deltaqs.cpp` |
| `source/source_lcao/module_operator_lcao/dspin_lcao.h` | 新增 `mu_save` 成员 |
| `source/source_lcao/module_operator_lcao/dspin_lcao.cpp` | `contributeHR()` 支持 (μ+λ, μ-λ) 势 |

## 分阶段实现详情

### S1: DeltaQS SCF 核心模块

#### S1.1 SpinConstrain 类扩展

新增数据成员（`spin_constrain.h`）：

```cpp
// 电荷约束数据（每原子）
std::vector<double> mu_;              // 电荷 Lagrange 乘子 (Ry/e)
std::vector<double> target_charge_;   // 目标投影电荷 (electrons)
std::vector<double> Ni_;              // 当前计算投影电荷
std::vector<int> constrain_charge_;   // 电荷约束标志 (0=free, 1=constrained)

// 配置参数
bool charge_constraint_enabled_;
std::string qs_mode_;                 // "deltaspin", "deltaq", "deltaqs", "auto"
double sc_charge_thr_;                // 电荷收敛阈值
double charge_alpha_trial_;           // μ 更新步长 (Ry/e²)
double charge_restrict_current_;      // μ 最大步长 (Ry/e)
bool ground_state_search_;
int outer_max_iter_;
double outer_thr_;
bool gradient_output_;
```

#### S1.2 Operator 修改（dspin_lcao.cpp）

核心变更：`contributeHR()` 中的增量势计算。

**原 DeltaSpin (nspin=2):**
```
coeff_up   = +delta_lambda_z
coeff_down = -delta_lambda_z
```

**DeltaQS (nspin=2):**
```
coeff_up   = delta_mu + delta_lambda_z
coeff_down = delta_mu - delta_lambda_z
```

新增 `cal_coeff_lambda_qs()` 函数，同时对 nspin=2 (double) 和 nspin=4 (complex) 提供重载。`contributeHR()` 中通过 `sc.is_charge_constraint_enabled()` 判断是否使用 QS 系数。

`mu_save` 向量用于增量更新：`delta_mu = mu_current - mu_save`，与 `lambda_save` 平行管理。

#### S1.3 联合 Lambda 循环

`run_qs_lambda_loop()` 实现交替优化策略：

1. 先运行标准 `run_lambda_loop()` 收敛自旋约束 (λ)
2. 计算电荷投影 `cal_ni_lcao()`
3. 梯度下降更新 μ：`μ_I += κ_μ · (N_I - N_I^target)`
4. 重新对角化，检查电荷 RMS
5. 迭代直到电荷 RMS < sc_charge_thr

#### S1.4 电荷投影计算

`cal_ni_lcao()` 复用 DeltaSpin 的 `pre_hr` 投影框架：

- nspin=2: 使用总密度矩阵 `switch_dmr(0)` + `cal_moment()`
- nspin=4: 使用旋量密度矩阵，求迹得到总电荷

#### S1.5 约束能量修正

`cal_escon()` 扩展：

```
E_scon = -Σ_I (λ_I · Mi_I) - Σ_I (μ_I · Ni_I)
```

#### S1.6 INPUT 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sc_charge_switch` | bool | false | 启用电荷约束 |
| `sc_qs_mode` | string | "auto" | 模式选择 |
| `sc_charge_thr` | double | 1e-4 | 电荷收敛阈值 (e) |
| `sc_charge_alpha` | double | 0.01 | μ 步长 (eV/e²) |
| `sc_charge_sccut` | double | 3.0 | μ 最大步长 (eV/e) |
| `sc_ground_state_search` | bool | false | 外层优化开关 |
| `sc_outer_max_iter` | int | 50 | 外层最大迭代 |
| `sc_outer_thr` | double | 1e-4 | 外层收敛阈值 (eV) |
| `sc_gradient_output` | bool | false | 梯度输出开关 |

#### S1.7 STRU 格式扩展

```
ATOMIC_POSITIONS
Direct
Fe
0.0
2
0.00 0.00 0.00  mag 2.0  sc 1 1 1  tc 6.5  cq 1  mu 0.0
```

新增关键词：
- `tc <value>`: 目标电荷 (electrons)
- `cq <flag>`: 电荷约束标志 (0=free, 1=constrained)
- `mu <value>`: 初始电荷乘子 (eV/e，内部转为 Ry/e)

#### S1.8 SCF 集成

Facade 函数 `init_deltaspin_lcao()` 在 `init_sc()` 后调用 `init_deltaqs()` 初始化电荷约束数据。`run_deltaspin_lambda_loop_lcao()` 根据 `is_charge_constraint_enabled()` 选择调用 `run_lambda_loop()` 或 `run_qs_lambda_loop()`。

### S2: 梯度提取与输出

#### 梯度文件输出

`write_gradient_file()` 输出 `deltaqs_gradient_<step>.dat`：

```
# Atom  Ni  Mi_z  target_N  target_M  mu(Ry)  lambda_z(Ry)  mu(eV)  lambda_z(eV)
Fe_0  6.52  2.01  6.50  2.00  0.012  0.003  0.163  0.041
```

#### 打印函数

- `print_Ni()`: 输出各原子的 Ni、目标值、偏差
- `print_Charge_Force()`: 输出各原子的 μ 乘子
- `cal_charge_escon()`: 电荷约束能量修正

### S3: 网格扫描框架

`run_qs_grid_scan()` 实现 2D E(N, M) 势能面映射：

```
输入: scan_atom, N_min, N_max, N_step, M_min, M_max, M_step
输出: deltaqs_grid_scan.dat
```

对每个 (N, M) 网格点：
1. 设置 target_charge 和 target_mag
2. 运行 `run_qs_lambda_loop()` 收敛
3. 记录 E, μ, λ, Ni, Mi

输出文件包含完整势能面数据，可直接用于 CP-3 验证。

### S4: 梯度下降优化器

`run_qs_gradient_descent()` 实现 2D 势能面上的梯度下降：

```
输入: max_steps, step_size, conv_thr
输出: deltaqs_gradient_descent.dat
```

每步：
1. 运行 DeltaQS SCF → 得到 E, μ, λ
2. 构造梯度: g_N = -μ_I + μ_ref, g_M = -λ_I
3. 检查收敛: max|grad| < conv_thr
4. 更新目标: N_target += step_size · μ, M_target += step_size · λ

### S5: L-BFGS 优化器

`run_qs_lbfgs()` 实现高维 L-BFGS 优化：

```
输入: max_steps, conv_thr, history_size=5
输出: deltaqs_lbfgs.dat
```

优化变量: x = (N_1, ..., N_{k-1}, M_1, ..., M_k)，其中原子 k 为电荷缓冲池。

算法流程：
1. 运行 DeltaQS SCF → 得到 E, μ, λ
2. 构造梯度向量（维度 = active_charge - 1 + active_spin）
3. L-BFGS two-loop recursion 计算搜索方向
4. 线搜索步长 α = 0.1（可配置）
5. 更新目标值，维护历史 {s_k, y_k} 对

支持配置历史窗口大小（默认 m=5）。

### S6: 差异归因工具

`run_qs_attribution()` 实现 CP-6 差异归因分析：

输出 `deltaqs_attribution.dat`，包含：
- 各原子的 Ni, Mi, μ, λ 详细数据
- 总磁矩 S* = Σ Mi
- 归因分类判断：
  - **A**: 同一 M，不同磁构型（FM vs AFM）
  - **B**: M 不在扫描范围内
  - **C**: 多体效应（分数自旋态）

## 使用指南

### 基本 DeltaQS 计算

INPUT 文件：
```
sc_mag_switch      1
sc_charge_switch   1
sc_qs_mode         auto
sc_charge_thr      1e-4
sc_charge_alpha    0.01
sc_scf_thr_mode    immediate
```

STRU 文件：
```
V
0.0
2
0.00 0.00 0.00  mag 1.0  sc 1  tc 3.5  cq 1
0.00 0.00 3.00  mag 1.0  sc 1  tc 3.5  cq 1
```

### 基态搜索

```
sc_ground_state_search  true
sc_outer_max_iter       50
sc_outer_thr            1e-4
sc_gradient_output      true
```

### 网格扫描（CP-3 验证）

通过代码调用：
```cpp
sc.run_qs_grid_scan(scan_atom, N_min, N_max, N_step, M_min, M_max, M_step);
```

## 验证检查点

| 检查点 | 验证内容 | 判据 |
|--------|---------|------|
| CP-0 | DeltaQS 与标准 SCF 等价性 | |E_QS - E_ref| < 1e-5 Ha |
| CP-1 | 电荷梯度 μ 验证 | |g_FD - (-μ)| / (|g_FD| + η) < 0.01 |
| CP-2 | 自旋梯度 λ 验证 | |g_FD^M - (-λ)| / (|g_FD^M| + η) < 0.01 |
| CP-3 | 2D E(N,M) 势能面 | 无断崖，最小值处 μ₁≈μ₂, λ≈0 |
| CP-4 | 梯度下降轨迹 | ≥90% 轨迹收敛到全局最小 |
| CP-5 | 高维 L-BFGS | E_global ≤ E_M-scan^min |
| CP-6 | 差异归因 | 每个 ΔE<0 案例可归入 A/B/C |

## 向后兼容性

- 当 `sc_charge_switch = false` 时，所有 DeltaQS 代码路径不执行，行为与原 DeltaSpin 完全一致
- `sc_qs_mode = "auto"` 自动根据开关推断模式
- 现有测试用例不受影响

## 单位系统

| 量 | 内部单位 | INPUT 单位 | 转换 |
|----|---------|-----------|------|
| μ | Ry/e | eV/e | × Ry_to_eV |
| λ | Ry/μB | eV/μB | × Ry_to_eV |
| Ni | electrons | electrons | - |
| Mi | μB | μB | - |
| E | Ry | Ry | - |

## 后续开发方向

1. **PW 基组支持**: 当前 `cal_ni_lcao()` 仅支持 LCAO，需添加 `cal_ni_pw()` 路径
2. **nspin=4 联合优化**: `run_qs_lambda_loop()` 中的 nspin=4 分支需要完善
3. **子空间加速集成**: DeltaQS 的 μ 更新与 LCAO subspace acceleration 的兼容
4. **自动步长选择**: L-BFGS 中的 Armijo/Wolfe 线搜索替代固定步长
5. **MPI 并行**: 网格扫描和 L-BFGS 多起点的 MPI 任务分配
