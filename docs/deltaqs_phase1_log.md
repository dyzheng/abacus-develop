# Phase 1 开发记录：CSZ 投影算符

## 日期
2026-06-30

## 目标
实现 Complete Single-Zeta (CSZ) 投影算符，使用所有可用的 zeta 轨道（而非仅第一个 zeta）来构建投影矩阵 pre_hr_csz。

## 实现内容

### 新增文件
| 文件 | 说明 |
|------|------|
| `module_deltaqs/deltaqs_projector.h` | CSZProjector 类声明 |
| `module_deltaqs/deltaqs_projector.cpp` | CSZ 投影算符实现 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `module_deltaqs/CMakeLists.txt` | 添加 deltaqs_projector.cpp |
| `module_deltaspin/spin_constrain.h` | 添加 csz_projector_ 成员 |
| `module_deltaspin/deltaspin_lcao.h` | init_deltaspin_lcao 添加 CSZ 参数 |
| `module_deltaspin/deltaspin_lcao.cpp` | 传递 CSZ 参数到 init_deltaqs |
| `module_deltaspin/deltaqs.cpp` | init_deltaqs 中构建 CSZ 投影器 |
| `source_esolver/esolver_ks_lcao.cpp` | 传递 Grid_Driver, TwoCenterIntegrator, orb_cutoff, hR |

### 核心实现

**CSZProjector::build()**：
- 复用 DeltaSpin cal_pre_HR 的框架（邻接原子搜索、AtomPair 拓扑、HContainer 分配）
- **关键改动**：nlm 选择逻辑从 "仅 first-zeta" 改为 "前 n_zeta_l 个 zeta"
- 索引系统：顺序索引 over (zeta, l, m)，总投影轨道数 = Σ_l(n_zeta_l × (2l+1))

**CSZProjector::cal_hr_ijr()**：
- 复用 DeltaSpin cal_HR_IJR 的逻辑（nspin=2 路径）
- 正确处理 data_pointer 的步进（npol 相关）

**CSZProjector::cal_charge()**：
- 计算 N_I = Tr(ρ × pre_hr_I)
- 与 DeltaSpin cal_moment 逻辑相同

## 测试验证

### Fe 测试（4s2p2d1f 轨道文件）

**CSZ 投影器构建**：
```
Element: Fe (Z_val = 16)
l  orbital_zetas  csz_zetas  proj_orbitals
0       4            4            4
1       2            2            6
2       2            2           10
3       1            1            7
Total CSZ projection orbitals: 27
```
✅ CSZ 投影器成功构建，每原子 27 个投影函数

### 电荷计算对比

| 方法 | Atom 0 | Atom 1 | 总和 | 预期 |
|------|--------|--------|------|------|
| first-zeta | 13.64 | 13.64 | 27.28 | 32 |
| CSZ (无正交化) | 25.07 | 25.07 | 50.14 | 32 |
| Mulliken | 16.00 | 16.00 | 32.00 | 32 |

### Mulliken 分析（正确参考）

| 角动量 | 电子数 |
|--------|--------|
| s (4 zeta) | 3.01 |
| p (2 zeta) | 6.21 |
| d (2 zeta) | 6.73 |
| f (1 zeta) | 0.05 |
| **总计** | **16.00** |

## 关键发现：CSZ 投影器过度计数问题

### 问题描述
CSZ 投影器给出 ~25 电子/原子，比正确值 16 多出 ~56%。

### 根本原因
CSZ 投影器的多个 zeta 轨道之间**不正交**：

$$P_I^{\text{CSZ}} = \sum_{\zeta,l,m} |\alpha_{\zeta,l,m}\rangle\langle\alpha_{\zeta,l,m}|$$

当计算 $N_I = \text{Tr}(\rho \cdot P_I)$ 时，由于 $\langle\alpha_{\zeta_1}|\alpha_{\zeta_2}\rangle \neq 0$（$\zeta_1 \neq \zeta_2$），重叠区域的电荷被**重复计数**。

### 正确公式
需要使用投影器的重叠矩阵逆：

$$N_I = \sum_{i,j} \langle\alpha_i|\rho|\alpha_j\rangle \cdot (S^{-1})_{ji}$$

其中 $S_{ij} = \langle\alpha_i|\alpha_j\rangle$ 是投影器之间的重叠矩阵。

### 为什么 first-zeta 不会过度计数
first-zeta 投影器只用一个 zeta per l，不同 l 的轨道因球谐函数正交性自动正交：
$$\langle\alpha_{l_1,m_1}|\alpha_{l_2,m_2}\rangle = \delta_{l_1,l_2}\delta_{m_1,m_2}$$

所以 first-zeta 投影器是正交的，不会过度计数。但它是**不完备**的（只用部分 zeta）。

## Phase 1 临时方案

鉴于 CSZ 正交化的复杂性，Phase 1 采用**first-zeta 投影器**作为临时方案：

1. **优点**：
   - 已实现且经过验证
   - 正交，不会过度计数
   - 代码改动最小

2. **缺点**：
   - 不完备，只捕获 ~85% 的价电子
   - 目标值需要校准

3. **校准方法**：
   - 先运行无约束 SCF，获取 first-zeta 投影的参考电荷 N_ref
   - 用户指定价态变化 δN
   - 目标电荷 = N_ref + δN

## 后续计划

### Phase 1b（可选）：CSZ 正交化
实现 Löwdin 正交化的 CSZ 投影器：

1. **计算重叠矩阵** $S_{ij} = \langle\alpha_i|\alpha_j\rangle$
   - 需要修改 `intor->snap()` 以支持 alpha-alpha 重叠积分
   - 或从轨道文件直接计算

2. **计算逆矩阵** $S^{-1}$
   - 小矩阵（27×27），可直接求逆

3. **修改 cal_charge()**：
   ```cpp
   N_I = Σ_{i,j} Tr(ρ · |α_i><α_j|) × S^{-1}_{ji}
   ```

4. **验证**：
   - 对比 Mulliken 分析
   - 确保 Σ N_I = Z_val_total

### Phase 2：DeltaQS 算符
实现 (μ+λ)P↑ + (μ-λ)P↓ 的 Hamiltonian 贡献。

### Phase 3：联合 Lambda 循环
实现 μ 和 λ 的联合优化。

## 文件变更统计
- 新增文件：2
- 修改文件：6
- 新增代码行：~450
- 测试用例：1（Fe2 系统）

## 下一步
1. 提交 Phase 1 代码
2. 开始 Phase 2：DeltaQS 算符实现
