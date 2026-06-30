# DeltaQS 独立模块开发规划

## 1. 问题陈述

### 1.1 现有 DeltaSpin 投影的根本缺陷

DeltaSpin 使用 **first-zeta 投影**：

$$P_I^{\text{current}} = \sum_{l,m} |\alpha_{I,0,l,m}\rangle\langle\alpha_{I,0,l,m}|$$

其中只取每个角动量 $l$ 的 $N=0$ zeta 轨道。

**缺陷**：ABACUS 轨道文件中 zeta 的排序（N 字段）仅是顺序索引（0, 1, 2, ...），**不保证 N=0 是最重要或布居最大的 zeta**。Mulliken 分析证实：

| 元素 | 轨道文件 | p-zeta 1 布居 | p-zeta 2 布居 | 结论 |
|------|---------|-------------|-------------|------|
| Fe | 4s2p2d1f | 0.90 | **5.31** | N=0 不是最大 |
| C  | 2s2p1d | 待验证 | 待验证 | - |

这导致电荷投影不完备，缺失约 30% 的价电子。

### 1.2 为什么不能从轨道文件获取排序规则

1. **N 字段是顺序索引**，不是主量子数
2. **没有元数据**标记 zeta 的物理含义（3p vs 4p）
3. **排序规则因生成方法而异**（SG15, ONCVPSP, 自定义等）
4. **无法可靠地从轨道文件推断**哪个 zeta 最重要

### 1.3 为什么不能从赝势获取价电子构型

当前赝势解析器只读取 `z_valence`（总价电子数），**不读取每个 $l$ 通道的电子数**。UPF 文件中的 `PP_INPUTFILE` 部分包含参考构型（如 `3 1 6.00` 表示 3p⁶），但 ABACUS 未解析这些信息。

---

## 2. 解决方案：Complete Single-Zeta (CSZ) 投影

### 2.1 CSZ 的定义

**ABACUS 严格 single-zeta**：每个角动量 $l$ 使用足够的 zeta 轨道，使得投影空间能覆盖该 $l$ 通道的所有价电子。

对于 $l$ 通道有 $n_e^l$ 个价电子：

$$n_\zeta^l = \lceil n_e^l / (2l+1) \rceil$$

CSZ 投影算符：

$$P_I^{\text{CSZ}} = \sum_{l} \sum_{\zeta=0}^{n_\zeta^l - 1} \sum_{m=-l}^{l} |\alpha_{I,\zeta,l,m}\rangle\langle\alpha_{I,\zeta,l,m}|$$

### 2.2 确定 $n_\zeta^l$ 的方法

ABACUS 轨道文件中 zeta 的排列是**顺序的**（N=0, 1, 2, ...），CSZ 投影从 N=0 开始连续取 $n_\zeta^l$ 个 zeta。所需数量从赝势的价电子构型中分析得到。

**步骤**：
1. 解析 UPF 文件的 `PP_INPUTFILE` 段，提取 $(n, l, f)$ 三元组
2. 对每个 $l$ 累加 $f$（occupation），得到 $n_e^l$
3. 计算 $n_\zeta^l = \lceil n_e^l / (2l+1) \rceil$
4. 从轨道文件中取 N=0 到 N=$n_\zeta^l - 1$ 的 zeta

**示例**（Fe，赝势价电子 3s²3p⁶3d⁶4s²）：

| $l$ | $n_e^l$ | $n_\zeta^l = \lceil n_e^l/(2l+1)\rceil$ | 取 N= |
|-----|---------|--------------------------------------|-------|
| s (0) | 2+2=4 | ⌈4/1⌉ = 4 | 0,1,2,3 |
| p (1) | 6 | ⌈6/3⌉ = 2 | 0,1 |
| d (2) | 6 | ⌈6/5⌉ = 2 | 0,1 |
| f (3) | 0 | 0 | 无 |

CSZ 投影轨道总数：4×1 + 2×3 + 2×5 = **20**

**错误处理**：如果轨道文件中某 $l$ 通道的 zeta 数 < $n_\zeta^l$，**直接报错退出**（ABACUS 生成的最小基组就是 single-zeta，不会出现 zeta 不足的情况；若出现则说明轨道文件与赝势不匹配）。

**真正的挑战**：从 double-zeta、TZDP 等多 zeta 基组中准确识别并提取前 $n_\zeta^l$ 个 zeta 作为 CSZ 部分。

### 2.3 投影轨道的正交性

**NAO 正交性**：同一原子、同一 $l$ 的不同 zeta 在 NAO 框架中是正交的（由径向方程的边界条件保证）。因此：

$$\langle \alpha_{\zeta_1, l, m} | \alpha_{\zeta_2, l, m} \rangle = \delta_{\zeta_1, \zeta_2} \delta_{m, m'}$$

CSZ 投影算符满足幂等性 $P^2 = P$，无需 Löwdin 正交化。

---

## 3. 模块架构

### 3.1 目录结构

```
source/source_lcao/module_deltaqs/
├── CMakeLists.txt
├── deltaqs.h                    # 主类声明
├── deltaqs.cpp                  # 主类实现
├── deltaqs_projector.h          # CSZ 投影算符
├── deltaqs_projector.cpp        # CSZ 投影实现
├── deltaqs_operator.h           # DeltaQS 算符（继承 OperatorLCAO）
├── deltaqs_operator.cpp         # 算符实现
├── deltaqs_charge.h             # 电荷投影计算
├── deltaqs_charge.cpp           # 电荷计算实现
├── deltaqs_lambda_loop.cpp      # 联合 μ+λ 优化循环
├── deltaqs_outer_loop.cpp       # 外层基态搜索（S4-S5）
├── deltaqs_grid_scan.cpp        # 网格扫描（S3）
├── deltaqs_attribution.cpp      # 归因分析（S6）
├── deltaqs_upf_parser.h         # UPF 价电子构型解析器
├── deltaqs_upf_parser.cpp       # UPF 解析实现
└── test/
    ├── test_projector.cpp
    ├── test_charge_projection.cpp
    └── test_deltaqs_scf.cpp
```

### 3.2 类关系

```
DeltaQS (主控制器, Singleton)
├── DeltaQSProjector (CSZ 投影算符)
│   └── UPFValenceParser (从 UPF 解析 n_e^l)
├── DeltaQSOperator (Hamiltonian 算符)
│   ├── contributeHR(): H += (μ+λ)P↑ + (μ-λ)P↓
│   └── cal_charge(): N_I = Tr(ρ·P_I)
├── DeltaQSLambdaLoop (联合优化)
│   ├── run_inner_loop(): BFGS 优化 μ, λ
│   └── cal_mw_from_lambda(): 应用约束 → 求解 → 计算 N_I, M_I
├── DeltaQSOuterLoop (外层基态搜索)
│   ├── gradient_descent_2d()
│   └── lbfgs_optimizer()
└── DeltaQSGridScan (网格扫描)
```

### 3.3 与现有模块的关系

```
module_deltaqs/ (新, 独立)          module_deltaspin/ (现有)
├── 完全独立的投影系统               ├── first-zeta 投影
├── 电荷约束 + 自旋约束              ├── 仅自旋约束
├── 需要完整轨道信息                 ├── 仅需 first-zeta
└── 面向数据集构建                   └── 面向单点计算

共享:
├── basic_funcs.h (向量运算)
├── source_hsolver/ (对角化)
├── source_estate/ (电子态)
└── source_base/ (基础工具)
```

---

## 4. 分阶段实现计划

### Phase 0: UPF 价电子构型解析器（1 天）

**目标**：从 UPF 文件自动解析每个 $l$ 通道的价电子数。

**实现**：
```cpp
struct ValenceConfig {
    double zv_total;                    // 总价电子数
    std::map<int, double> electrons_per_l; // l -> n_e^l
    std::map<int, int> csz_per_l;       // l -> n_zeta (complete single-zeta)
};

ValenceConfig parse_upf_valence(const std::string& upf_file);
```

**解析逻辑**：
1. 读取 `PP_INPUTFILE` 段
2. 提取 `(n, l, f)` 三元组
3. 对每个 $l$ 累加 $f$（occupation）
4. 计算 $n_\zeta^l = \lceil n_e^l / (2l+1) \rceil$

**验证**：
- Fe UPF: s→4, p→6, d→6 → CSZ: s=4, p=2, d=2
- O UPF: s→2, p→4 → CSZ: s=2, p=2
- 对比手动分析

### Phase 1: CSZ 投影算符（2 天）

**目标**：实现完整 single-zeta 投影。

**核心逻辑**：
1. 从赝势解析得到每个元素的 $n_\zeta^l$
2. 从轨道文件取 N=0 到 N=$n_\zeta^l - 1$ 的 zeta（从前往后数）
3. 校验轨道文件中该 $l$ 通道的可用 zeta 数 ≥ $n_\zeta^l$，否则报错退出

**核心类**：
```cpp
class DeltaQSProjector {
public:
    void init(const UnitCell& ucell);
    // 内部自动：
    //   1. 读取每个元素的 UPF → ValenceConfig
    //   2. 计算 n_zeta_per_l
    //   3. 校验轨道文件 zeta 数
    //   4. 调用 cal_pre_hr_csz()
    
    void cal_pre_hr_csz(const UnitCell& ucell,
                         const TwoCenterIntegrator* intor,
                         const Grid_Driver* gridD);
    // 与 DeltaSpin 的 cal_pre_HR 类似，但遍历前 n_zeta 个 zeta
    // 而非仅 N=0
    
    int get_nproj(int iat) const;
    const HContainer<double>* get_pre_hr(int iat) const;
    
private:
    std::vector<ValenceConfig> valence_configs_;
    std::vector<HContainer<double>*> pre_hr_csz_;
};
```

**指标系统**：

CSZ 的指标 $(\zeta, l, m)$ 的线性映射：

```cpp
int index(int zeta, int l, int m, const std::vector<int>& nzeta_per_l) {
    int offset = 0;
    for (int ll = 0; ll < l; ll++) {
        offset += nzeta_per_l[ll] * (2*ll + 1);
    }
    return offset + zeta * (2*l + 1) + (l + m);
}
```

总投影轨道数：$N_{\text{proj}} = \sum_l n_\zeta^l (2l+1)$

**cal_pre_hr_csz() 与 DeltaSpin cal_pre_HR() 的差异**：

DeltaSpin 只取 target_L 的第一个 zeta：
```cpp
// DeltaSpin: 只取 L0 == target_L 的第一个
if (L0 == target_L) { ... target_L++; }
```

DeltaQS CSZ 取前 $n_\zeta^l$ 个 zeta：
```cpp
// DeltaQS: 取前 n_zeta 个 zeta
int zeta_count = 0;
for (iw ...) {
    if (L0 == target_L) {
        if (zeta_count < n_zeta_per_l[target_L]) {
            // 包含此 zeta 的所有 m 分量
            for (m ...) { nlm_target[index] = nlm[zeta_count][iw+m]; }
            zeta_count++;
        }
    }
}
```

**验证**：
- 对 Fe: N_proj = 4×1 + 2×3 + 2×5 = 20
- 对 O: N_proj = 2×1 + 2×3 = 8
- 对比 first-zeta: Fe N_proj = 9

### Phase 2: DeltaQS 算符（2 天）

**目标**：实现约束势对 Hamiltonian 的贡献。

**核心类**：
```cpp
class DeltaQSOperator : public OperatorLCAO<TK, TR> {
public:
    // H += (μ_I + λ_I) P_I↑ + (μ_I - λ_I) P_I↓
    void contributeHR() override;
    
    // 计算投影电荷 N_I = Tr(ρ · P_I)
    std::vector<double> cal_charge(const HContainer<double>* dmR_total);
    
    // 计算投影磁矩 M_I = Tr((ρ↑-ρ↓) · P_I)
    std::vector<double> cal_moment(const HContainer<double>* dmR_diff);
    
    // 更新约束乘子
    void update_constraints(const double* mu, const double* lambda, int nat);
    
private:
    DeltaQSProjector* projector_;
    std::vector<double> mu_;      // 电荷乘子
    std::vector<double> lambda_;  // 自旋乘子
    std::vector<double> mu_save_;
    std::vector<double> lambda_save_;
};
```

**contributeHR() 实现**：

对 nspin=2（两个独立 spin 通道）：
```
对于每个约束原子 I:
  coeff_up   = (mu_I + lambda_I) - (mu_save_I + lambda_save_I)
  coeff_down = (mu_I - lambda_I) - (mu_save_I - lambda_save_I)
  
  对于 pre_hr_I 中的每个 <J, R> 对:
    HR_up[J,R]   += coeff_up   * pre_hr_I[J,R]
    HR_down[J,R] += coeff_down * pre_hr_I[J,R]
```

**cal_charge() 实现**：

使用 switch_dmr(1) 获取 ρ_total = ρ↑ + ρ↓：
```
N_I = Σ_{μ,ν} (ρ↑_μν + ρ↓_μν) * pre_hr_I[μ,ν]
    = Σ_{μ,ν} ρ_total_μν * pre_hr_I[μ,ν]
```

**验证**：
- 无约束 SCF → N_I 应接近 Mulliken 总布居
- 对 Fe₂: Σ N_I ≈ Z_val_total = 32

### Phase 3: 联合 Lambda 循环（2 天）

**目标**：同时优化 μ 和 λ 以满足电荷和磁矩约束。

**算法**：
```
run_qs_lambda_loop():
  1. 初始化: μ = 0, λ = 0
  2. 计算初始 N_I, M_I
  3. 循环直到收敛:
     a. 计算残差: δN_I = N_I - N_I_target, δM_I = M_I - M_I_target
     b. 计算 RMS_charge, RMS_spin
     c. 如果 RMS_charge < thr_charge 且 RMS_spin < thr_spin → 收敛
     d. 更新 μ: μ_I += α_charge * δN_I
     e. 更新 λ: λ_I += α_spin * δM_I  (使用 BFGS)
     f. apply_constraints(μ, λ)
     g. 重新对角化 H(μ, λ)
     h. 计算新 N_I, M_I
```

**核心实现**：
```cpp
class DeltaQSLambdaLoop {
public:
    bool run(DeltaQS& qs, int max_iter, 
             double thr_charge, double thr_spin);
    
private:
    // BFGS 状态
    std::vector<double> search_direction_;
    double alpha_trial_;
    double alpha_opt_;
};
```

**验证**：
- 对 Fe₂ 设置 N_target = 16 ± 0.5, M_target = ±2.0
- 验证收敛后 N_I, M_I 匹配目标值
- 对比纯 DeltaSpin 结果（λ only）

### Phase 4: SCF 集成（1 天）

**目标**：将 DeltaQS 集成到 SCF 循环中。

**修改 ESolver**：
```cpp
// esolver_ks_lcao.cpp 的 iter_init()
if (PARAM.inp.sc_charge_switch || PARAM.inp.sc_mag_switch) {
    if (use_deltaqs) {
        deltaqs.run_qs_lambda_loop(iter);
    } else {
        sc.run_lambda_loop(iter - 1);
    }
}
```

**INPUT 参数**：
```
# DeltaQS 参数
sc_charge_switch     1        # 启用电荷约束
sc_qs_mode           deltaqs  # deltaspin/deltaq/deltaqs
sc_charge_thr        1e-4     # 电荷收敛阈值 (electrons)
sc_charge_alpha      0.01     # μ 步长 (eV/e²)
```

注：CSZ 投影的 zeta 数量由赝势自动确定，无需用户指定。

**STRU 关键词**：
```
Fe  0.0  2
0.00 0.00 0.00  mag 2.0  sc 0 0 1  tc 16.0  cq 1  mu 0.0
0.51 0.51 0.51  mag -2.0 sc 0 0 1  tc 16.0  cq 1  mu 0.0
```

其中 `tc` 在 absolute 模式下表示目标投影电荷（CSZ 投影下应接近 Z_val），在 valence 模式下表示目标价态（N_projected - Z_val）。

**验证**：
- 完整 SCF 计算
- 检查能量、电荷、磁矩收敛

### Phase 5: 梯度提取与验证（1 天）

**目标**：输出 μ_I, λ_I 梯度，验证 CP-1/CP-2。

**实现**：
```cpp
void DeltaQS::write_gradient_file(int step) {
    // 输出: atom, Ni, Mi_z, target_N, target_M, mu, lambda_z
}
```

**验证**：
- CP-1: 有限差分 ∂E/∂N_I ≈ -μ_I
- CP-2: 有限差分 ∂E/∂M_I ≈ -λ_I
- 判据：|g_FD - g_analytical| / (|g_FD| + η) < 0.01

### Phase 6: 网格扫描与优化器（2 天）

**目标**：实现 S3-S5 的高级功能。

**网格扫描**：
```cpp
void DeltaQSGridScan::run_2d(
    int scan_atom, 
    double N_min, double N_max, double N_step,
    double M_min, double M_max, double M_step);
// 输出: E(N, M), μ(N, M), λ(N, M)
```

**梯度下降**：
```cpp
void DeltaQSOuterLoop::gradient_descent_2d(
    int max_steps, double step_size, double conv_thr);
```

**L-BFGS**：
```cpp
void DeltaQSOuterLoop::lbfgs_optimizer(
    int max_steps, double conv_thr, int history_size = 5);
```

### Phase 7: 归因分析与数据集工具（1 天）

**目标**：实现 S6 归因分析和数据集构建工具。

**归因分析**：
```cpp
void DeltaQSAttribution::analyze(
    const std::string& ref_label,
    const std::vector<double>& E_mscan);
// 分类: A(磁构型), B(扫描范围), C(多体效应)
```

**数据集工具**：
```cpp
void DeltaQSDataset::export_labels(
    const std::string& output_file);
// 输出: structure_id, atom_id, valence, mu, lambda
```

---

## 5. 验证检查点

### CP-0: CSZ 投影完备性
**判据**：对 Fe₂, Σ N_I / Z_val_total > 0.95（CSZ 应捕获 ≥95% 价电子）

### CP-1: 电荷梯度验证
**操作**：固定 M, 变化 N_V1 ± 0.05
**判据**：|g_FD - (-μ)| / (|g_FD| + η) < 0.01

### CP-2: 自旋梯度验证
**操作**：固定 N, 变化 M_V1 ± 0.05
**判据**：|g_FD - (-λ)| / (|g_FD| + η) < 0.01

### CP-3: 2D 势能面光滑性
**判据**：E(N, M) 无断崖，全局最小处 μ₁ ≈ μ₂, λ ≈ 0

### CP-4: 梯度下降轨迹
**判据**：≥90% 轨迹收敛到同一全局最小

### CP-5: 高维 L-BFGS
**判据**：E_global(Q) ≤ E_M-scan^min(Q)

---

## 6. 时间表

| Phase | 任务 | 预计时间 | 依赖 |
|-------|------|---------|------|
| 0 | UPF 解析器 | 1 天 | 无 |
| 1 | CSZ 投影算符 | 2 天 | Phase 0 |
| 2 | DeltaQS 算符 | 2 天 | Phase 1 |
| 3 | 联合 Lambda 循环 | 2 天 | Phase 2 |
| 4 | SCF 集成 | 1 天 | Phase 3 |
| 5 | 梯度提取 (CP-1/2) | 1 天 | Phase 4 |
| 6 | 网格扫描+优化器 | 2 天 | Phase 5 |
| 7 | 归因分析+数据集 | 1 天 | Phase 6 |
| **总计** | | **12 天** | |

---

## 7. 风险评估

### 7.1 UPF 解析兼容性（中风险）
- **问题**：不同 UPF 版本（1.0, 2.0.1）的 `PP_INPUTFILE` 格式可能不同
- **缓解**：支持 UPF 1.0 和 2.0.1 两种格式；解析失败时报错退出（不 fallback）
- **测试**：收集 10+ 种 UPF 文件验证解析

### 7.2 轨道文件与赝势不匹配（低风险）
- **问题**：轨道文件的某 $l$ 通道 zeta 数 < 赝势要求的 $n_\zeta^l$
- **处理**：直接报错退出，提示用户更换匹配的轨道文件
- **说明**：ABACUS 生成的最小基组就是 single-zeta，正常情况不会出现

### 7.3 投影轨道线性相关（低风险）
- **问题**：多 zeta 可能导致 pre_hr 矩阵病态
- **缓解**：NAO 正交性保证不会线性相关
- **监控**：检查 pre_hr 的条件数

### 7.4 计算成本增加（中风险）
- **问题**：CSZ 投影轨道数 > first-zeta（Fe: 20 vs 9）
- **影响**：pre_hr 内存增加 ~2.2×，cal_charge 时间增加 ~2.2×
- **缓解**：可接受范围内，不是瓶颈

---

## 8. 与现有代码的兼容性

### 8.1 DeltaSpin 保持不变
- 现有 `module_deltaspin/` 代码不修改
- 用户可通过 `sc_charge_switch = 0` 继续使用纯 DeltaSpin

### 8.2 共享基础设施
- `basic_funcs.h`: 向量运算
- `lambda_loop_helper.cpp`: BFGS 辅助函数（可复用）
- `source_hsolver/`: 对角化求解器
- `source_estate/`: 电子态管理

### 8.3 独立编译
- `module_deltaqs/` 作为独立的 OBJECT 库
- 仅在 `sc_charge_switch = 1` 时链接
- 不影响现有 DeltaSpin 的编译和运行
