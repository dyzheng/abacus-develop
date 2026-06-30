# DeltaQS 开发进度总结

## 总体状态

**项目**：DeltaQS 统一框架 - 电荷-自旋联合约束 DFT  
**分支**：feat/deltaqs-unified-framework  
**最后更新**：2026-06-30  
**当前进度**：Phase 0-4 完成 (5/8)

---

## 阶段完成状态

### ✅ Phase 0: UPF 价电子构型解析器

**目标**：确定每个元素的 Complete Single-Zeta (CSZ) 投影基

**实现**：
- `upf_valence_parser.h/cpp`: 解析赝势 zv 和轨道文件 zeta 数
- 策略：从赝势读取 zv，从轨道文件读取 l_nchi，使用所有可用 zeta

**关键发现**：
- ABACUS 在 `read_pp.cpp` 中过滤相对论通道（j=l+0.5），但不更新 occupations
- 导致 `pp.oc` 数据错乱，无法直接从 PP_PSWFC 获取价电子构型
- 解决方案：改用 zv + 轨道文件 zeta 数

**验证**：
- Fe (4s2p2d1f): zv=16, CSZ=27 投影函数 (4s+6p+10d+7f)
- O (2s2p1d): zv=6, CSZ=13 投影函数 (2s+6p+5d)

**文件**：
- `source/source_lcao/module_deltaqs/upf_valence_parser.h`
- `source/source_lcao/module_deltaqs/upf_valence_parser.cpp`
- `docs/deltaqs_phase0_log.md`

**Commit**: `ee62144a3`

---

### ✅ Phase 1: CSZ 投影算符

**目标**：实现使用所有 zeta 的投影算符

**实现**：
- `deltaqs_projector.h/cpp`: CSZProjector 类
- 修改 `cal_pre_hr_csz()`: 使用 `csz_per_l` 中指定的多个 zeta
- 集成到 `init_deltaqs()`: 构建 CSZ 投影器

**关键发现**：
- CSZ 投影器给出 ~25 e⁻/atom（过度计数）
- first-zeta 投影器给出 ~13.6 e⁻/atom（不足）
- Mulliken 分析给出 16.0 e⁻/atom（正确）
- **根本原因**：CSZ 的多个 zeta 不正交，导致重复计数

**正确公式**：
$$N_I = \sum_{i,j} \langle\alpha_i|\rho|\alpha_j\rangle \cdot S^{-1}_{ji}$$

**临时方案**：使用 first-zeta 投影器（正交，无重复计数）

**验证**：
- Fe: CSZ=27 proj, first-zeta=16 proj
- Charge comparison: CSZ=25.07, first-zeta=13.64, Mulliken=16.00

**文件**：
- `source/source_lcao/module_deltaqs/deltaqs_projector.h`
- `source/source_lcao/module_deltaqs/deltaqs_projector.cpp`
- `docs/deltaqs_phase1_log.md`

**Commit**: `a6849fc54`

---

### ✅ Phase 2: DeltaQS 算符

**目标**：实现 (μ+λ)P↑ + (μ-λ)P↓ 的 Hamiltonian 贡献

**实现**：
- `cal_coeff_lambda_qs()`: nspin=2 和 nspin=4 两个版本
- `contributeHR()`: 检查 `charge_enabled` 并调用相应函数
- `mu_save`: 跟踪 μ 用于增量更新

**nspin=2 公式**：
```cpp
coeff_up = μ + λ_z
coeff_down = μ - λ_z
```

**nspin=4 公式**（Pauli 矩阵）：
```cpp
coeff[0] = μ + λ_z      (up-up)
coeff[1] = λ_x - iλ_y   (up-down)
coeff[2] = λ_x + iλ_y   (down-up)
coeff[3] = μ - λ_z      (down-down)
```

**关键发现**：Phase 2 已在 S1 阶段实现！

**文件**：
- `source/source_lcao/module_operator_lcao/dspin_lcao.cpp:84-96, 160-168`

**Commit**: 包含在 `aa1dd3e79` 中

---

### ✅ Phase 3: 联合 Lambda 循环

**目标**：实现 μ 和 λ 的联合优化

**实现**：
- `run_qs_lambda_loop()`: 交替优化 λ 和 μ
  1. 先运行 `run_lambda_loop()` 优化 λ（自旋约束）
  2. 然后运行 `update_mu_simple()` 优化 μ（电荷约束）
  3. 每次 μ 更新后重新求解 Hamiltonian
  4. 重新计算 Ni 和 Mi

**μ 更新算法**：
```cpp
mu_I^(n+1) = mu_I^(n) + alpha * (Ni - Ni_target)
```

**关键发现**：Phase 3 已在 S1 阶段实现！

**验证**（Fe₂ 测试）：
- 步长 0.01: 收敛慢（10步，RMS 0.196→0.192）
- 步长 0.1: 收敛快（1步，RMS 0.0005 < 0.001）✅
- 最终结果：Ni=13.50 e, Mi=±1.99 μB

**参数调优**：
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| sc_charge_alpha | 0.1 | μ 更新步长 |
| sc_charge_sccut | 3.0 | μ 最大步长 |
| sc_charge_thr | 0.001 | 电荷收敛阈值 |
| nsc | 50 | 最大 μ 迭代数 |

**文件**：
- `source/source_lcao/module_deltaspin/deltaqs.cpp:484-566`
- `docs/deltaqs_phase2-3_log.md`

**Commit**: `aa1dd3e79`

---

### ✅ Phase 4: SCF 集成

**目标**：将 DeltaQS 集成到 ESolver 的 SCF 循环

**实现**：
- 修改 `init_deltaspin_lcao()`: 接受 Grid_Driver, TwoCenterIntegrator, orb_cutoff, hR
- 修改 `esolver_ks_lcao.cpp`: 传递这些参数
- 修改 `run_deltaspin_lambda_loop_lcao()`: 调用 `run_qs_lambda_loop()` 而非 `run_lambda_loop()`
- 修改 `cal_mi_lcao_wrapper()`: 同时调用 `cal_ni_lcao()` 和 `cal_mi_lcao()`

**关键修改**：
```cpp
// esolver_ks_lcao.cpp:148-160
auto* hamilt_lcao_tmp = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt);
void* hR_ptr = nullptr;
if (hamilt_lcao_tmp) {
    hR_ptr = static_cast<void*>(hamilt_lcao_tmp->getHR());
}

init_deltaspin_lcao<TK>(ucell, PARAM.inp, &(this->pv), this->kv, 
                        this->p_hamilt, this->psi, this->dmat.dm, this->pelec,
                        static_cast<void*>(&this->gd),
                        static_cast<void*>(two_center_bundle_.overlap_orb.get()),
                        orb_.cutoffs(),
                        hR_ptr);
```

**验证**：Fe₂ 测试成功运行，电荷和自旋约束同时满足

**文件**：
- `source/source_esolver/esolver_ks_lcao.cpp:148-160`
- `source/source_lcao/module_deltaspin/deltaspin_lcao.cpp:45-128`
- `source/source_lcao/module_deltaspin/deltaspin_lcao.h:27-47`

**Commit**: 包含在 `aa1dd3e79` 中

---

### ⏳ Phase 5: 梯度提取与验证

**目标**：计算并验证 ∂E/∂N_I = -μ_I 和 ∂E/∂M_I = -λ_I

**任务**：
1. 实现梯度计算函数
2. 设计有限差分验证测试（CP-1, CP-2）
3. 生成梯度文件用于后续优化

**状态**：待开始

---

### ⏳ Phase 6: 网格扫描与优化器

**目标**：2D E(N,M) 势能面扫描，梯度下降，L-BFGS 优化

**任务**：
1. 实现 2D 网格扫描（固定一个原子，扫描 N 和 M）
2. 实现梯度下降优化器
3. 实现 L-BFGS 优化器
4. 比较不同优化器的性能

**状态**：待开始

---

### ⏳ Phase 7: 归因分析与数据集工具

**目标**：基态搜索，数据集生成，归因分析

**任务**：
1. 实现基态搜索（多起点优化）
2. 实现数据集生成工具
3. 实现归因分析（分析优化轨迹）

**状态**：待开始

---

## 关键技术决策

### 1. CSZ 投影器正交化问题

**问题**：多个 zeta 不正交导致重复计数

**解决方案**：
- 短期：使用 first-zeta 投影器（正交，无重复计数）
- 长期：实现 Löwdin 正交化（需要计算 S⁻¹）

**影响**：
- first-zeta 只捕获 ~85% 价电子
- 需要用户校准目标值（先运行无约束 SCF 获取参考电荷）

### 2. 投影电荷 vs 价态模式

**问题**：target_charge 应该设置为投影电荷还是价态？

**决策**：
- 投影电荷模式：target_charge = 投影电荷（推荐，物理意义明确）
- 价态模式：target_charge = 价态（需要额外转换）

**实现**：两种模式都支持，通过 `sc_charge_mode` 参数选择

### 3. μ 更新步长

**问题**：步长太小收敛慢，步长太大可能震荡

**决策**：
- 默认步长：0.1 eV/e²
- 用户可调：通过 `sc_charge_alpha` 参数
- 步长限制：通过 `sc_charge_sccut` 参数

**验证**：Fe₂ 测试中，步长 0.1 可在 1 步内收敛

---

## 已知问题

### 1. CSZ 投影器过度计数（Phase 1）

**状态**：已记录，使用 first-zeta 作为临时方案

**后续**：Phase 1b 实现 Löwdin 正交化（可选）

### 2. 目标电荷校准

**问题**：用户如何确定合理的 target_charge？

**当前方案**：
1. 先运行无约束 SCF
2. 从 gradient 文件读取参考 Ni
3. 根据物理需求设置 target_charge（如 Ni ± 0.5）

**后续**：实现自动校准工具

### 3. 收敛阈值选择

**问题**：sc_charge_thr = 0.001 是否合理？

**当前方案**：用户可调

**后续**：基于测试数据给出推荐值

---

## 下一步计划

### Phase 5: 梯度提取与验证

**目标**：验证 ∂E/∂N_I = -μ_I 和 ∂E/∂M_I = -λ_I

**步骤**：
1. 实现梯度计算函数
2. 设计 CP-1 测试：固定 M，变化 N，验证 ∂E/∂N = -μ
3. 设计 CP-2 测试：固定 N，变化 M，验证 ∂E/∂M = -λ
4. 生成梯度文件

**预计用时**：2 天

---

## 文件结构

```
source/source_lcao/module_deltaqs/
├── CMakeLists.txt
├── upf_valence_parser.h
├── upf_valence_parser.cpp
├── deltaqs_projector.h
├── deltaqs_projector.cpp
└── test/
    ├── CMakeLists.txt
    ├── test_upf_valence_parser.cpp
    └── test_csz_projector.cpp

docs/
├── deltaqs_module_development_plan.md
├── deltaqs_projector_analysis.md
├── deltaqs_phase0_log.md
├── deltaqs_phase1_log.md
└── deltaqs_phase2-3_log.md
```

---

## 测试用例

### Fe₂ (BCC)

**设置**：
- 赝势：Fe.upf (zv=16)
- 轨道：Fe_gga_6au_100Ry_4s2p2d1f.orb
- 自旋约束：atom 0 → M_z = +2.0 μB, atom 1 → M_z = -2.0 μB
- 电荷约束：atom 0 → N = 13.5 e, atom 1 → N = 13.5 e
- 投影器：first-zeta (16 proj/atom)

**结果**：
- Charge RMS: 0.0005 e (threshold 0.001 e) ✅
- Spin: Mi_z = ±1.993 μB (target ±2.0 μB) ✅
- Charge: Ni = 13.5005 e (target 13.50 e) ✅

**文件**：`/tmp/deltaqs_test/`

---

## Commit 历史

| Commit | 内容 | 文件数 |
|--------|------|--------|
| `ee62144a3` | Phase 0: CSZ 基确定 | 10 |
| `a6849fc54` | Phase 1: CSZ 投影算符 | 10 |
| `aa1dd3e79` | Phase 2&3: 验证 | 1 |

---

## 参考文献

1. 原始设计文档：`/personal/DeltaQS_unified_framework.md`
2. CSZ 投影分析：`docs/deltaqs_projector_analysis.md`
3. 各阶段详细日志：`docs/deltaqs_phase*_log.md`

---

## 联系

**开发者**：AI Assistant  
**监督者**：dyzheng  
**分支**：feat/deltaqs-unified-framework  
**远程**：zdy/feat/deltaqs-unified-framework
