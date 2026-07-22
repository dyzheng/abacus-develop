# DeltaP Berry Connection 算符：并行实现状态

> **日期**: 2026-07-09
> **分支**: `feat/deltap-wilson-per-atom`
> **状态**: 串行模式完成并验证，并行模式需要 ScaLAPACK 或自定义通信

---

## 1. 已完成的里程碑

### 1.1 核心实现（串行模式）

| 组件 | 状态 | 文件 |
|------|------|------|
| 算符公式推导 | ✅ 完成 | `2026-07-09-deltap-operator-derivation.md` |
| `compute_hk_correction()` | ✅ 串行完成 | `deltap_wannier.cpp` |
| `contributeHk()` | ✅ 完成 | `deltap_lcao.cpp` |
| `deltap_lambda_init` 参数 | ✅ 已添加 | `input_parameter.h`, `read_input_item_other.cpp` |
| 输入验证修改 | ✅ 已修改 | `read_input_item_postprocess.cpp` |
| `operator_lcao.cpp` 修改 | ✅ 已修改 | 添加 `contributeHk` 调用 |
| `esolver_ks_lcao.cpp` 修改 | ✅ 已修改 | `iter_finish` 中调用算符 |

### 1.2 测试验证（串行模式）

| 测试 | 结果 | 说明 |
|------|------|------|
| Lambda Sweep | ✅ 通过 | P 随 λ 连续变化（5 个 λ 值） |
| T1: λ=0 回归 | ✅ 通过 | 能量与参考一致 |
| Linear Target | ⚠️ 符号问题 | λ 和 γ 都远离目标 |

### 1.3 Lambda Sweep 结果

```
lambda     g_O            g_H1=H2        Δg_H per Δλ
0.000     -8.0483e-02    -3.9851e-02    —
0.005     -8.0382e-02    -3.9616e-02    +4.7e-3
0.010     -8.0404e-02    -3.9392e-02    +4.6e-3
0.015     -8.0488e-02    -3.9220e-02    +3.4e-3
0.020     -8.0726e-02    -3.9066e-02    +3.1e-3
```

**氢原子 (H1=H2)**：g_H 随 λ **单调递增**，变化连续 ✅

---

## 2. 并行实现挑战

### 2.1 核心问题

在 2D 块循环分布下：
- `F`: `(nrow × nocc_use)` — 局部行
- `C_L`: `(nrow × nbands)` — 局部行
- `HK`: `(nrow × ncol)` — 局部块
- `H_sym` 修正: `(nrow × ncol)` — 需要匹配 HK

**矩阵乘法 `M = F · C_L†`** 要求：
- `F[alpha, p]`: 可用（alpha 是局部行）
- `C_L[beta, p]`: beta 是局部列索引，但 `C_L` 按局部行存储

**需要重分布**：将 `C_L` 从 `(nrow × nbands)` 重分布到 `(nbands × ncol)`，或实现自定义通信。

### 2.2 实现方案

**方案 A：使用 ScaLAPACK `pzgemm`**
- 自动处理 2D 块循环重分布
- 需要 ScaLAPACK 依赖
- 实现复杂度：中等

**方案 B：自定义通信**
- 手动实现进程间数据交换
- 避免 ScaLAPACK 依赖
- 实现复杂度：高

**方案 C：简化并行（仅对角块）**
- 只计算 `nrow == ncol` 的情况（对角块）
- 跳过非对角块
- 实现复杂度：低，但精度降低

### 2.3 建议

**短期**：保持串行模式（当前实现），用于测试和验证。

**中期**：实现方案 A（ScaLAPACK），参考 `op_dftu_lcao.cpp` 中的 `cal_eff_pot_mat_complex` 实现。

**长期**：优化性能，减少 Wilson loop 计算频率。

---

## 3. 已知问题

### 3.1 Linear Target 测试符号问题

**现象**：λ 和 γ 都远离目标值。

**可能原因**：
1. Lambda 更新符号错误
2. 算符修正符号错误
3. 数值噪声（单步 SCF）

**当前状态**：未解决，需要进一步调试。

### 3.2 并行模式未实现

**现象**：`compute_hk_correction` 在 `nrow != ncol` 时跳过。

**影响**：并行 SCF 计算中算符修正不被应用。

**当前状态**：未实现，需要 ScaLAPACK 或自定义通信。

---

## 4. 下一步

### 4.1 短期（1 周）

1. **调试 Linear Target 符号问题**
   - 检查 lambda 更新符号
   - 检查算符修正符号
   - 尝试多步 SCF 减少数值噪声

2. **文档完善**
   - 编写用户使用指南
   - 添加输入参数说明

### 4.2 中期（1-2 月）

1. **实现并行版本**
   - 使用 ScaLAPACK `pzgemm`
   - 参考 DFT+U 算符实现

2. **性能优化**
   - 缓存 `S_dk_` 矩阵
   - 减少 Wilson loop 计算频率

### 4.3 长期（3-6 月）

1. **SCF 约束收敛测试**
   - 设置目标 `gamma^I`
   - 运行完整 SCF
   - 验证收敛性

2. **扩展到更大体系**
   - 液态水（4H₂O）
   - 铁电体（PbTiO₃）

---

## 5. 技术细节

### 5.1 算符公式

Berry connection 算符（Hermitian 对称化后）：

$$H^\lambda_{\alpha\beta}(k_j) = \frac{1}{2}\left(M_{\alpha\beta} + M^*_{\beta\alpha}\right)$$

其中：

$$M_{\alpha\beta} = \sum_p F_{\alpha p} c^*_{L,\beta p}$$

$$F_{\alpha p} = \frac{i}{2} w^{\text{eff}}_p \sum_\gamma S_{\alpha\gamma}(k_j, k_{j+1}) c_{R,\gamma p}$$

$$w^{\text{eff}}_p = \sum_I \lambda^I w^I_p$$

### 5.2 关键修改文件

| 文件 | 修改内容 |
|------|---------|
| `input_parameter.h` | 添加 `deltap_lambda_init` 参数 |
| `read_input_item_other.cpp` | 添加 `deltap_lambda_init` 读取 |
| `read_input_item_postprocess.cpp` | 允许 `scf` + `berry_phase` + `deltap_corr` |
| `deltap_lcao.cpp` | 使用 `deltap_lambda_init` 初始化 lambda |
| `deltap_wannier.cpp` | 实现 `compute_hk_correction`（串行） |
| `deltap_lcao.h` | 添加 `contributeHk` 声明、`hk_correction_` 成员 |
| `operator_lcao.cpp` | 在 `lcao_dp_lambda` case 中调用 `contributeHk` |
| `esolver_ks_lcao.cpp` | 在 `iter_finish` 中调用 `compute_hk_correction` |

### 5.3 并行实现伪代码

```cpp
// 并行版本 compute_hk_correction
void DeltaP::compute_hk_correction_parallel(...)
{
    // 获取局部维度
    int nrow = paraV_->get_row_size();
    int ncol = paraV_->get_col_size();
    
    // 对于每个局部 HK 元素 (alpha, beta)
    for (int alpha = 0; alpha < nrow; ++alpha)
    {
        for (int beta = 0; beta < ncol; ++beta)
        {
            // 计算 M_{alpha, beta}
            // 需要 C_L[global_beta, p]，需要通信获取
            
            // 计算 H_sym_{alpha, beta}
            // H_sym = (M + M†) / 2
            
            // 存储到 hk_correction
        }
    }
}
```

---

## 附录 A: 相关文档

| 文档 | 内容 |
|------|------|
| `2026-07-09-deltap-operator-derivation.md` | Berry connection 算符严格推导 |
| `2026-07-09-deltap-lambda-sweep-validation.md` | Lambda Sweep 验证报告 |
| `2026-07-08-deltap-branch-choice-impact.md` | 分支选择对 Wannier90 结果的影响 |
| `2026-07-09-deltap-operator-implementation-status.md` | 实现状态 |

## 附录 B: 测试脚本

| 脚本 | 位置 | 说明 |
|------|------|------|
| `run_test.sh` | `/tmp/opencode/lambda_sweep/` | Lambda Sweep 测试 |
| `run_test.sh` | `/tmp/opencode/linear_target_test/` | Linear Target 测试 |
