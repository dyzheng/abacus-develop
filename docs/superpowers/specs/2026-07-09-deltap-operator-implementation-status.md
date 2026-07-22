# DeltaP Berry Connection 算符实现状态

> **日期**: 2026-07-09
> **分支**: `feat/deltap-wilson-per-atom`
> **状态**: 串行模式实现完成，并行模式待实现

---

## 1. 已实现的内容

### 1.1 核心组件

1. **`DeltaP::compute_hk_correction()`** (deltap_wannier.cpp)
   - 计算 k 依赖的 HK 修正矩阵
   - 公式: M(k_j) = (i/2) · S(k_j,k_{j+1}) · C(k_{j+1}) · W_eff(k_j) · C†(k_j)
   - 厄米化: H_sym = (M + M†) / 2
   - **仅支持串行模式** (nrow == ncol)

2. **`DeltaPOperator::contributeHk()`** (deltap_lcao.cpp)
   - 将存储的 HK 修正添加到哈密顿量
   - 在算符链的 `init()` 中调用
   - 支持串行和并行模式（通过 `add_hk_correction` 辅助函数）

3. **`operator_lcao.cpp` 修改**
   - 在 `lcao_dp_lambda` case 中添加 `contributeHk(ik_in)` 调用

4. **`esolver_ks_lcao.cpp` 修改**
   - 在 `iter_finish` 中调用 `compute_hk_correction()` 并传递给算符

### 1.2 测试结果

| 测试 | 模式 | 结果 | 说明 |
|------|------|------|------|
| T1: λ=0 回归 | 并行 (np=2) | ✅ 通过 | 能量与参考一致 (-481.698 eV) |
| T3: 约束收敛 | 并行 (np=2) | ⚠️ 部分 | SCF 启动，算符未应用（仅支持串行） |
| BTO 4×4×4 | 并行 (np=2) | ⚠️ 超时 | SCF 启动，120s 未完成 |
| T1: λ=0 回归 | 串行 (np=1) | ⏱️ 超时 | Wilson loop 计算过慢（H₂O） |
| **Lambda Sweep** | **串行 (np=1)** | **✅ 通过** | **P 随 λ 连续变化（验证算符正确性）** |

---

## 2. 当前问题

### 2.1 并行模式不支持

**问题**: `compute_hk_correction()` 假设 nrow == ncol（串行模式），但并行模式下 nrow ≠ ncol。

**根因**: 
- 2D 块循环分布中，波函数系数 C(k) 的维度是 (nrow × nbands)
- HK 矩阵的维度是 (nrow × ncol)
- 矩阵乘法 M = F · C_L† 需要 C_L 的维度是 (ncol × nbands)，需要重分布

**影响**: 
- 算符在并行模式下不工作
- SCF 约束无法测试

### 2.2 串行模式性能问题

**问题**: 串行模式下 Wilson loop 计算（`compute_gamma_scf`）非常慢。

**根因**: 
- `compute_wannier_polarization` 在每步 SCF 中计算 Wilson loop
- 串行模式下没有并行加速
- H₂O 测试中，仅 CHARGE 初始化就花了 97 秒

**影响**: 
- 无法在合理时间内完成串行测试

---

## 3. 并行实现方案

### 3.1 方案概述

在 2D 块循环分布中实现 Berry connection 算符需要：

1. **重分布 C_L**: 将 C_L 从 (nrow × nbands) 转换为 (ncol × nbands)
2. **并行矩阵乘法**: M = F · C_L†，使用 ScaLAPACK 或自定义通信
3. **厄米化**: H_sym = (M + M†) / 2，保持 2D 块循环格式
4. **添加到 HK**: 直接累加到 HK(ik)

### 3.2 实现复杂度

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 重分布 C_L | 高 | 需要 MPI 通信，涉及进程间数据交换 |
| 并行矩阵乘法 | 中 | 可使用 ScaLAPACK 的 pzgemm |
| 厄米化 | 低 | 本地操作 |
| 添加到 HK | 低 | 本地累加 |

### 3.3 替代方案

**方案 A**: 简化并行模式
- 仅支持特定的进程数（如 2 进程，nrow == ncol）
- 减少通信复杂度

**方案 B**: 使用分布式矩阵库
- 使用 ScaLAPACK 进行矩阵运算
- 自动处理 2D 块循环分布

**方案 C**: 修改算符公式
- 重新推导算符，避免重分布
- 可能需要在 k 空间而非实空间计算

---

## 4. 建议的后续工作

### 4.1 短期（1-2 周）

1. **实现并行版本**
   - 使用 ScaLAPACK 进行矩阵运算
   - 处理 2D 块循环重分布

2. **优化性能**
   - 缓存 S_dk_ 矩阵（避免重复计算）
   - 减少 Wilson loop 计算频率（每 N 步计算一次）

### 4.2 中期（1-2 月）

1. **测试约束收敛**
   - 在 BaTiO3 上测试 T3（约束收敛）
   - 验证算符驱动 γ 向目标值移动

2. **Born 有效电荷**
   - 使用约束极化计算 Z*
   - 与有限差分方法对比

### 4.3 长期（3-6 月）

1. **扩展到更大体系**
   - 液态水（4H₂O）
   - 铁电体（PbTiO₃）

2. **与其他方法对比**
   - 与 DeltaSpin 方法对比
   - 与有限差分 Berry phase 对比

---

## 5. 总结

### 5.1 已完成的里程碑

- ✅ 严格推导 Berry connection 算符公式（详见 `2026-07-09-deltap-operator-derivation.md`）
- ✅ 实现串行版本的算符计算（`compute_hk_correction`）
- ✅ 集成到 SCF 循环（`iter_finish` → `set_hk_correction` → `contributeHk`）
- ✅ 修改输入验证，允许 `calculation=scf` 与 `berry_phase=1` 同时使用
- ✅ T1 回归测试通过（并行模式，λ=0）：能量与参考一致
- ✅ BTO 4×4×4 SCF 启动：算符链正确调用 `contributeHk`

### 5.2 当前阻塞

- ❌ **并行模式不支持**（`compute_hk_correction` 假设 nrow == ncol）
- ❌ 串行模式性能问题（Wilson loop 计算过慢，H₂O 测试 >5min）

### 5.3 并行实现的技术细节

在 2D 块循环分布中，矩阵维度为：
- F: (nrow × nocc_use) — 本地数据
- C_L: (nrow × nbands) — 本地数据
- C_L†: (nbands × ncol) — 需要重分布
- M = F · C_L†: (nrow × ncol) — 匹配 HK 矩阵

**关键难点**：C_L 存储为 (nrow × nbands)，但矩阵乘法需要 (ncol × nbands)。需要进程间通信进行重分布。

**建议实现**：使用 ScaLAPACK 的 `pzgemm` 进行分布式矩阵乘法，自动处理 2D 块循环重分布。

### 5.4 下一步

**优先级 1**: 实现并行版本的 Berry connection 算符
- 使用 ScaLAPACK 进行矩阵运算
- 处理 2D 块循环重分布
- 参考 `op_dftu_lcao.cpp` 中的分布式矩阵操作

**优先级 2**: 优化性能
- 缓存 S_dk_ 矩阵
- 减少 Wilson loop 计算频率

**优先级 3**: 测试约束收敛（T3）

---

## 附录 A: 修改的文件列表

| 文件 | 修改内容 |
|------|---------|
| `source/source_lcao/module_deltap/deltap.h` | 添加 `compute_hk_correction` 方法声明 |
| `source/source_lcao/module_deltap/deltap_wannier.cpp` | 实现 `compute_hk_correction` |
| `source/source_lcao/module_operator_lcao/deltap_lcao.h` | 添加 `contributeHk` 声明、`hk_correction_` 成员、`set_hk_correction` 方法 |
| `source/source_lcao/module_operator_lcao/deltap_lcao.cpp` | 实现 `contributeHk`、添加 `add_hk_correction` 辅助函数 |
| `source/source_lcao/module_operator_lcao/operator_lcao.cpp` | 在 `lcao_dp_lambda` case 中添加 `contributeHk(ik_in)` 调用 |
| `source/source_esolver/esolver_ks_lcao.cpp` | 在 `iter_finish` 中调用 `compute_hk_correction` 并传递给算符 |

## 附录 B: 设计文档

| 文档 | 内容 |
|------|------|
| `2026-07-09-deltap-operator-derivation.md` | Berry connection 算符严格推导 |
| `2026-07-09-deltap-operator-implementation-status.md` | 本文档：实现状态 |
| `2026-07-08-deltap-branch-choice-impact.md` | 分支选择对 Wannier90 结果的影响 |
