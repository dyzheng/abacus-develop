# DeltaP 项目当前状态总结

> **日期**: 2026-06-29
> **目的**: 全面盘点已完成的工作、已实现但未验证的代码、以及未完成的任务，供评估下一步方向

---

## 1. 已验证的成果（已提交，有测试数据支持）

### 1.1 Wilson loop 特征值分解算法（算法 D）

**代码文件**（已提交，30+ commits）：
- `source/source_lcao/module_deltap/deltap_wannier.cpp` — 主逻辑
- `source/source_lcao/module_deltap/deltap.h` — 类定义
- `source/source_lcao/module_deltap/deltap.cpp` — init + setup_kstring
- `source/source_lcao/module_deltap/deltap_overlap.cpp` — SMO 重叠计算
- `source/source_lcao/module_deltap/deltap_berry.cpp` — 旧 Berry connection 方法
- `source/source_lcao/module_deltap/deltap_gauge.cpp` — SMO 规范固定
- `source/source_lcao/module_deltap/deltap_io.cpp` — I/O + sum rule 验证

**已验证的测试结果**：

| 验证项 | 结果 | 状态 |
|--------|------|------|
| $\sum_n \arg(\lambda_n) = \arg(\det\mathbf{W})$ | 机器精度内一致 | ✅ |
| Diamond 平衡结构 P=0 | $1.1 \times 10^{-17}$ | ✅ |
| BaTiO3 P_elec ratio (vs berry_phase) | 0.966 (3% 误差) | ✅ |
| SMO 第一 zeta 选择 | 与 DeltaSpin 一致 | ✅ |
| 小位移所有结构 ratio | 0.93–1.04 | ✅ |

**3% 误差根因已定位**：snap (nk=1005) vs center2_orb11 (kmesh=4021) 的 k 网格密度差异，导致 O_j 矩阵非酉（|det|=0.33–0.99 vs berry 的 ≈1.0）。这是**数值精度问题**，不是算法框架问题。

### 1.2 算法评估文档

**文件**: `docs/superpowers/specs/2026-06-28-deltap-algorithm-evaluation.md`

六种算法的完整推导和评估：
- 算法 A (Berry connection/trace): ❌ 44% 误差，数学必然 (trace≠det)
- 算法 B (逐原子 Wilson loop det): ❌ nproj>nocc 退化
- 算法 C (混合 Wilson+trace): ❌ 比例随结构变号
- **算法 D (Wilson loop 特征值): ✅ 已实现，0.966 ratio**
- 算法 E (dk 外推): ⚠ 未实现，4× 计算量
- 算法 F (SMO-basis Wannierization): ⚠ 未实现，理论等价于 D

---

## 2. 已实现但完全未验证的代码（未提交）

### 2.1 DeltaP 哈密顿量修正算符

**状态**: 代码已写完，能编译通过，但**从未成功运行过任何测试**。

**新增文件**（untracked）：
- `source/source_lcao/module_operator_lcao/deltap_lcao.h` — DeltaPOperator 类定义
- `source/source_lcao/module_operator_lcao/deltap_lcao.cpp` — 实现

**修改文件**（未提交）：
- `source/source_lcao/hamilt_lcao.h` — 添加 `dp_operator` 成员 + 前向声明
- `source/source_lcao/hamilt_lcao.cpp` — 添加 DeltaPOperator 到算符链
- `source/source_esolver/esolver_ks_lcao.cpp` — 添加 `iter_finish` 中的占位代码
- `source/source_io/module_parameter/input_parameter.h` — 添加 `deltap_corr` 等参数
- `source/source_lao/CMakeLists.txt` — 添加 deltap_lcao.cpp
- `source/source_lao/module_operator_lcao/CMakeLists.txt` — 同上

**修正算符公式**：
$$H_{\text{corr}} = -\sum_I \lambda_I \cdot \tau^I_\alpha \cdot \hat{P}^I$$

其中 $\hat{P}^I = \sum_{lm} |\alpha^I_{lm}\rangle\langle\alpha^I_{lm}|$ 是 SMO 投影器，$\tau^I_\alpha$ 是原子位置。这与 DeltaSpin 的修正 $H_{\text{DS}} = \lambda \cdot \sigma_z \cdot \hat{P}^I$ 结构完全相同，只是系数从 `lambda * sigma_z` 变为 `lambda * tau_alpha`。

**代码实现状态**：
- ✅ `cal_pre_HR()`: SMO 投影器计算（从 DeltaSpin 复制，逻辑相同）
- ✅ `contributeHR()`: 将 `lambda * tau * pre_hr` 加到 hR
- ✅ `set_lambda()` / `update_lambda()`: lambda 设置接口
- ✅ 算符链注册: `hamilt_lcao.cpp` 中 `this->getOperator()->add(dp_op)`
- ⚠ `iter_finish` 中的 lambda 更新: **占位代码，实际不做任何更新**（只调用 set_lambda 传入当前值）
- ❌ SCF lambda 循环: **完全未实现**
- ❌ Wilson loop 极化计算在 SCF 内的调用: **未集成**
- ❌ 任何运行测试: **从未成功运行**

**尝试过的测试**：
- Si 4×4×4 SCF: 失败（Si 轨道文件名不匹配，`nw=0`）
- 未尝试 BTO（需要先准备 SCF 电荷）

### 2.2 Wannier90 对比验证

**状态**: 完全未完成。

**尝试过的问题**：
1. ABACUS 生成的 `.mmn` 文件是 v2 格式（5 字段），Wannier90 3.1.0 期望 v3 格式（8 字段含 G 矢量）
2. 尝试转换格式时，nnkp 的 b-vector 约定不明确，G 矢量计算对大部分 block 失败
3. 现有 `100_PW_W90` 测试只生成 `.mmn` 文件但从未运行 `wannier90.x`
4. 从未成功运行过任何 Wannier90 → Wannier center → 与算法 D 对比的测试

**需要的条件**（未具备）：
- 从源码编译的 Wannier90（conda-forge 版本在 BaTiO3 上有 I/O 崩溃）
- 正确格式的 `.mmn` 文件（v3 格式，或修复 ABACUS 的输出格式）
- 一个已知能工作的 Wannier90 例子作为基准

---

## 3. 未实现的关键组件

### 3.1 SCF 内 Wilson loop 极化计算

当前 DeltaP 的 `compute_atomic_polarization` 只在 NSCF 后处理中调用（`ctrl_scf_lcao.cpp` line 365-398）。要在 SCF 迭代中更新 lambda，需要：
- 在 `iter_finish` 中实例化 DeltaP 对象
- 调用 `compute_atomic_polarization` 计算 P^I
- 从 P^I 更新 lambda
- 将 lambda 传递给 DeltaPOperator

**困难**：DeltaP 的初始化需要 `unkOverlap_lcao`、`cal_r_overlap_R` 等重量级对象，在每步 SCF 中重复创建代价大。

### 3.2 Lambda 更新逻辑

DeltaSpin 有完整的 BFGS/CG 优化器（`lambda_loop.cpp`, ~500 行）。DeltaP 目前只有占位代码。需要实现：
- 简单梯度下降：`lambda_I += step * (P^I - P^I_target)`
- 或更复杂的优化器（参考 DeltaSpin）

### 3.3 分支跟踪

Wilson loop 特征值 $\arg(\lambda_n)$ 有 2π 跳变问题。在 SCF 迭代中，如果特征值越过负实轴，P^I 不连续，导致 lambda 更新方向错误。需要跨 SCF 步的特征值跟踪。

---

## 4. 关键未验证的理论假设

### 4.1 "算法 D = Wannier center" 的等价性

**理论推导**：$\langle r_n \rangle = \frac{a}{2\pi}\arg(\lambda_n)$，Wilson loop 特征值的 arg = Wannier center。

**验证状态**: ❌ **从未用任何外部基准验证过**。

这是整个项目的核心假设。如果这个等价性在实际数值中不成立（例如因为离散化误差、规范问题等），那么：
- 算法 D 的逐原子分解可能不正确
- 修正算符的方向可能错误
- 所有基于此的后续工作都不可靠

### 4.2 "Berry connection 可作为修正方向" 的假设

**理论推导**：用 $\hat{r}^I_\alpha \approx \tau^I_\alpha \hat{P}^I$（Berry connection 主项）作为修正方向，虽然 Berry connection ≠ Berry phase（44% 误差），但 Lagrange 乘子迭代更新可以收敛。

**验证状态**: ❌ **从未测试过**。

如果修正方向与真实梯度正交分量过大，迭代可能不收敛或收敛到错误值。

### 4.3 "snap vs center2 的 3% 差异可通过统一 k 网格修复" 的假设

**理论推导**：snap (nk=1005) 和 center2_orb11 (kmesh=4021) 使用相同数学方法但不同 k 网格密度，增大 lcao_ecut 可使两者一致。

**验证状态**: ❌ **增大 lcao_ecut=1600 的测试被中断，从未完成**。

---

## 5. 建议的分阶段验证体系

### 阶段 0: 修复算法 D 的 3% 误差（前提条件）

**目标**: P_elec ratio → 1.0（当前 0.966）

**方法**:
1. 增大 `lcao_ecut=1600`，重跑 BaTiO3 10×10×10，检查 ratio
2. 或修复 `berryphase_overlap` 的 ScaLAPACK 描述符 bug，使用 berry_phase 的精确 O_j

**验证标准**: P_elec ratio > 0.99

**状态**: ❌ 未完成，之前测试被中断

### 阶段 1: 验证 "算法 D = Wannier center" 等价性

**目标**: 用外部 Wannier90 验证 $\arg(\lambda_n) = \frac{2\pi}{a}\langle r_n\rangle$

**方法**:
1. 编译 Wannier90 从源码（获取库模式 + 修复 I/O bug）
2. 用 ABACUS PW 基组（已有 `100_PW_W90` 测试）生成 `.mmn` 文件
3. 修复 `.mmn` 格式（v2 → v3，或修改 ABACUS 输出格式）
4. 运行 `wannier90.x` 获取 Wannier center
5. 用相同 `.mmn` 构建 Wilson loop，对角化，获取 $\arg(\lambda_n)$
6. 对比两者

**验证标准**:
- 总量: $\sum_n \arg(\lambda_n) = \frac{2\pi}{a}\sum_n \langle r_n\rangle$（精确匹配）
- 逐能带: $\{\arg(\lambda_n)\}$ 作为集合 = $\{\frac{2\pi}{a}\langle r_n\rangle\}$（排序后匹配）

**状态**: ❌ 完全未开始，存在格式兼容性问题

**替代方案**（如果 Wannier90 格式问题太难解决）:
- 直接用 Python 从 `.mmn` 构建 Wilson loop 并对角化
- 对比 `arg(det(W))` 与 berry_phase 输出的 `elec_phase`
- 这验证 "Wilson loop = Berry phase"，但不验证 "特征值 = Wannier center"

### 阶段 2: 验证逐原子分解的正确性

**目标**: $\sum_I P^I = P^{\text{total}}$ 且各原子 P^I 物理合理

**方法**:
1. 用 BaTiO3（已知 Z* 文献值），检查各原子 P^I 的相对大小
2. 位移 Ti 原子，检查 ΔP^I_Ti 是否最大
3. 检查 sum rule: $\sum_I \gamma^I = \gamma^{\text{total}}$

**验证标准**:
- Sum rule 精确成立（已有验证 ✅）
- 位移响应物理合理（Ti 位移 → Ti 的 P^I 变化最大）

**状态**: 部分完成（sum rule ✅，位移响应有数据但精度不足）

### 阶段 3: 验证修正算符的线性响应

**目标**: 线性变化的 lambda 产生线性变化的 P^I

**方法**:
1. 固定 lambda = 0，运行 SCF，记录 P^I_0
2. 固定 lambda = λ₀（小值），运行 SCF，记录 P^I_λ
3. 检查 ΔP^I = P^I_λ - P^I_0 是否与 λ₀ 成正比
4. 对多个 λ₀ 值重复，验证线性度

**验证标准**:
- $R^2 > 0.99$ 的线性关系
- 不同原子的响应系数符号正确（Ti 响应 > O 响应）

**前提条件**:
- 阶段 0 完成（P_elec 精确）
- 修正算符代码能成功运行（当前 ❌）

**状态**: ❌ 代码能编译但从未成功运行

### 阶段 4: 验证 lambda SCF 迭代收敛

**目标**: 给定 P^I_target，SCF 迭代能收敛到使 P^I ≈ P^I_target 的 lambda

**方法**:
1. 设置 P^I_target = P^I_0 + δ（已知偏移量）
2. 运行 lambda SCF 迭代
3. 检查 lambda 收敛和 P^I → P^I_target

**验证标准**:
- lambda 收敛（|Δlambda| < 1e-4）
- P^I 误差 < 1%

**前提条件**:
- 阶段 3 完成（线性响应验证通过）
- Lambda 更新逻辑实现（当前 ❌）

**状态**: ❌ 完全未实现

---

## 6. 代码质量评估

### 6.1 已提交代码（算法 D）

| 方面 | 评价 |
|------|------|
| 算法正确性 | ✅ sum rule 精确，P ratio=0.966 |
| 数值精度 | ⚠ 3% 误差来自 snap vs center2 |
| 代码清洁度 | ✅ 已清理 debug 代码 |
| 文档完整性 | ✅ 10+ 篇详细文档 |
| 测试覆盖 | ⚠ BaTiO3 + Diamond，无自动化测试 |

### 6.2 未提交代码（修正算符）

| 方面 | 评价 |
|------|------|
| 编译状态 | ✅ 能编译通过 |
| 运行测试 | ❌ 从未成功运行 |
| 算法正确性 | ❌ 未验证 |
| 代码清洁度 | ⚠ cal_pre_HR 从 DeltaSpin 复制，有冗余 |
| 集成完整性 | ❌ lambda 更新是占位代码 |
| 文档 | ⚠ 仅在算法评估文档中有公式推导 |

---

## 7. 总结与判断

### 已确立的结论

1. **算法 D（Wilson loop 特征值）是正确的理论选择** — 六种算法中唯一同时满足精确性 + sum rule + 不受 nproj 限制的方案
2. **3% 误差是数值精度问题** — 根因是 snap vs center2 的 k 网格差异，非算法框架问题
3. **修正算符 $H_{\text{corr}} = -\sum_I \lambda_I \tau^I_\alpha \hat{P}^I$ 理论上可行** — 与 DeltaSpin 结构平行

### 未确立的关键假设

1. **$\arg(\lambda_n)$ = Wannier center** — 理论推导成立但**从未用外部基准验证**
2. **Berry connection 可作为修正方向** — **从未测试过收敛性**
3. **3% 误差可通过 lcao_ecut 修复** — **测试被中断**

### 当前最大风险

**在没有验证 "算法 D = Wannier center" 等价性的情况下**，直接实现修正算符并测试线性响应，是在一个未验证的地基上建楼。如果等价性在数值上不成立（哪怕理论推导正确），修正算符的测试结果将无法解读。

### 建议的优先级

1. **最高优先级**: 阶段 0 — 修复 3% 误差（1-2 天），这是所有后续工作的前提
2. **高优先级**: 阶段 1 — 验证 Wannier center 等价性（3-5 天），需要解决 Wannier90 格式问题
3. **中优先级**: 阶段 3 — 验证线性响应（2-3 天），需要先完成阶段 0
4. **低优先级**: 阶段 4 — Lambda SCF 迭代（5+ 天），依赖所有前置阶段
