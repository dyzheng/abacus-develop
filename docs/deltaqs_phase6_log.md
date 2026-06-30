# Phase 6 开发记录：网格扫描与优化器

## 日期
2026-06-30

## 目标
实现 2D E(N, M) 势能面扫描和优化器，用于寻找基态电荷/磁矩配置。

## 实现内容

### 1. 2D 网格扫描 (run_qs_grid_scan)

**功能**：系统地扫描目标电荷 (N) 和磁矩 (M)，计算每个网格点的能量。

**实现位置**：`deltaqs.cpp:571-631`

**参数**：
```cpp
void run_qs_grid_scan(
    int scan_atom = 0,           // 扫描的原子索引
    double N_min = 13.0,         // 最小目标电荷 (e)
    double N_max = 14.0,         // 最大目标电荷 (e)
    double N_step = 0.1,         // 电荷步长 (e)
    double M_min = 1.5,          // 最小目标磁矩 (μB)
    double M_max = 2.5,          // 最大目标磁矩 (μB)
    double M_step = 0.1          // 磁矩步长 (μB)
)
```

**输出**：
- 文件：`deltaqs_grid_scan.dat`
- 格式：`N  M  E(Ry)  mu(Ry/e)  lambda(Ry/μB)  N_actual(e)  M_actual(μB)`
- 自动记录最小能量点和对应的 (N, M)

**算法**：
```
for N_target in [N_min, N_max]:
    for M_target in [M_min, M_max]:
        1. 设置 target_charge[scan_atom] = N_target
        2. 设置 target_mag[scan_atom].z = M_target
        3. 运行 DeltaQS 优化 (run_qs_lambda_loop)
        4. 记录 E, μ, λ, N_actual, M_actual
        5. 更新最小能量点
```

### 2. 梯度下降优化器 (run_qs_gradient_descent)

**功能**：使用梯度信息迭代更新目标 (N, M)，寻找能量最小值。

**实现位置**：`deltaqs.cpp:634-705`

**参数**：
```cpp
void run_qs_gradient_descent(
    int max_steps = 50,          // 最大优化步数
    double step_size = 0.1,      // 步长
    double conv_thr = 0.01       // 梯度收敛阈值 (Ry)
)
```

**输出**：
- 文件：`deltaqs_gradient_descent.dat`
- 格式：`step  E(Ry)  max|grad|(Ry)  N_target(e)  M_target(μB)  mu(Ry/e)  lambda(Ry/μB)`

**算法**：
```
for step in [0, max_steps]:
    1. 运行 DeltaQS 优化
    2. 计算梯度范数：
       grad_norm = sqrt(Σ μ_i² + Σ λ_i²)
    3. 如果 grad_norm < conv_thr，收敛退出
    4. 更新目标：
       N_target += step_size * μ  (因为 dE/dN = -μ)
       M_target += step_size * λ  (因为 dE/dM = -λ)
```

**梯度方向说明**：
- 包络定理：dE/dN = -μ, dE/dM = -λ
- 梯度下降方向：-∇E = (μ, λ)
- 因此：N_new = N_old + step_size * μ
- 因此：M_new = M_old + step_size * λ

### 3. L-BFGS 优化器 (run_qs_lbfgs)

**功能**：使用有限内存 BFGS 算法加速收敛。

**实现位置**：`deltaqs.cpp:706-829`

**参数**：
```cpp
void run_qs_lbfgs(
    int max_steps = 50,          // 最大优化步数
    double conv_thr = 0.01,      // 梯度收敛阈值 (Ry)
    int history_size = 5         // L-BFGS 历史步数
)
```

**输出**：
- 文件：`deltaqs_lbfgs.dat`
- 格式：`step  E(Ry)  grad_norm(Ry)`

**算法**：
```
初始化：
    n_vars = 约束变量总数
    s_history = []  // x_{k+1} - x_k
    y_history = []  // g_{k+1} - g_k
    x_prev, g_prev = 当前状态

for step in [0, max_steps]:
    1. 运行 DeltaQS 优化
    2. 提取状态向量 x = [N_target, M_target, ...]
    3. 提取梯度向量 g = [-μ, -λ, ...]
    4. 计算梯度范数
    5. 如果 grad_norm < conv_thr，收敛退出
    
    6. L-BFGS 方向计算：
       a. 计算 ρ_i = 1 / (s_i · y_i)
       b. 第一循环：计算 α_i
       c. 初始 Hessian：H_0 = γI, γ = (s·y)/(y·y)
       d. 第二循环：计算搜索方向 z = H*g
    
    7. 更新：x_new = x_old - z
    
    8. 存储历史：
       s = x_new - x_old
       y = g_new - g_old
       如果 len(history) > history_size，删除最旧的
    
    9. 应用新目标
```

**L-BFGS 优势**：
- 使用曲率信息（通过 s, y 对）
- 比梯度下降收敛更快（超线性收敛）
- 内存需求小（只存储 history_size 步）

## 使用示例

### 示例 1: 网格扫描
```cpp
// 在代码中调用
spinconstrain::SpinConstrain<std::complex<double>>& sc = 
    spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();

// 扫描 Fe 原子的 (N, M) 空间
sc.run_qs_grid_scan(
    0,              // 扫描第 0 个原子
    13.0, 14.0, 0.1,  // N: 13.0 到 14.0，步长 0.1
    1.5, 2.5, 0.1     // M: 1.5 到 2.5，步长 0.1
);
```

### 示例 2: 梯度下降优化
```cpp
// 从当前配置开始优化
sc.run_qs_gradient_descent(
    50,    // 最多 50 步
    0.1,   // 步长 0.1
    0.01   // 梯度阈值 0.01 Ry
);
```

### 示例 3: L-BFGS 优化
```cpp
// 使用 L-BFGS 快速优化
sc.run_qs_lbfgs(
    50,    // 最多 50 步
    0.01,  // 梯度阈值 0.01 Ry
    5      // 存储最近 5 步历史
);
```

## 测试结果

### 编译验证
✅ `deltaspin` 模块编译成功
✅ `abacus_basic_para` 链接成功

### 功能验证
⏳ 待创建测试用例（需要添加 INPUT 参数或自定义 driver）

## 文件变更

### 新增函数
- `run_qs_grid_scan()` - deltaqs.cpp:571-631
- `run_qs_gradient_descent()` - deltaqs.cpp:634-705
- `run_qs_lbfgs()` - deltaqs.cpp:706-829

### 头文件声明
- spin_constrain.h:889-892（已存在）

## 已知问题

### 1. 缺少 INPUT 参数触发
当前这些函数只能通过代码调用，无法通过 INPUT 文件触发。

**解决方案**：
- 添加 INPUT 参数：`sc_optimization_mode` (none/grid/gradient/lbfgs)
- 或者创建自定义 driver 程序

### 2. 多原子优化
当前实现假设单原子优化（只扫描/优化一个原子）。

**改进方向**：
- 支持多原子同时优化
- 添加原子选择掩码

## 下一步

### Phase 7: 归因分析与数据集工具
1. 基态搜索（多起点优化）
2. 数据集生成工具
3. 归因分析（分析优化轨迹）

## 总结

Phase 6 实现了三个关键工具：
1. **网格扫描**：系统探索 (N, M) 空间
2. **梯度下降**：简单可靠的优化器
3. **L-BFGS**：快速收敛的高级优化器

这些工具为后续的基态搜索和数据集生成奠定了基础。
