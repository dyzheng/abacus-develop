# PW DFTU Debug Progress

## 当前状态 (2026-04-19)

### 通过测试
| Test | nspin | Spin | DFTU | DeltaSpin | 状态 |
|------|-------|------|------|-----------|------|
| 200 LCAO SPIN S2 Z | 2 | collinear Z | ✗ | ✓ | PASS |
| 201 LCAO SPIN S4 XYZ | 4 | non-collinear | ✗ | ✓ | PASS |
| 202 LCAO DFTU S2 Z | 2 | collinear Z | ✓ | ✗ | PASS |
| 203 LCAO DFTU S2 XY | 2 | collinear XY | ✓ | ✗ | PASS |
| 204 LCAO DFTU S4 XYZ | 4 | non-collinear | ✓ | ✗ | PASS |
| 220 PW SPIN S2 Z | 2 | collinear Z | ✗ | ✓ | PASS |
| 221 PW SPIN S4 XYZ | 4 | non-collinear | ✗ | ✓ | PASS |
| 250 PW DS S2 Z | 2 | collinear Z | ✗ | ✓ | PASS |
| 251 PW DS S2 XY | 2 | collinear XY | ✗ | ✓ | PASS |
| 260 PW DFTU+DS S2 Z | 2 | collinear Z | ✓ | ✓ | PASS |
| 261 PW DFTU+DS S2 XY | 2 | collinear XY | ✓ | ✓ | PASS |
| 300-315 LCAO DeltaSpin/DFTU+DS | 2/4 | all | ✓/✗ | ✓ | PASS |

### 失败测试
| Test | nspin | Spin | DFTU | DeltaSpin | 失败位置 |
|------|-------|------|------|-----------|----------|
| **222 PW DFTU S2 Z** | 2 | collinear Z | ✓ | ✗ | iter=2 Davidson hpsi → NaN |
| 223 PW DFTU S2 XY | 2 | collinear XY | ✓ | ✗ | 同上 |
| 225 PW DFTU S2 FeO | 2 | collinear Z | ✓ | ✗ | iter=2 Davidson hpsi → NaN |

## 根因分析

### 已排除
- ~~iter=1 cal_occ_pw 调用时机~~ → 即使跳过 iter=1 cal_occ_pw，iter=2 仍然崩溃
- ~~iter=1 VU 矩阵差异~~ → 跳过 iter=1 cal_occ_pw 后，iter=2 的 locale/VU 完全对齐
- ~~DFTU 投影计算 (cal_becp/onsite_proj)~~ → iter=1 前 3 次 hpsi 调用完全一致
- ~~tab_atomic_ 计算~~ → ik=1 时 vkb 值完全一致
- ~~psi 初始化~~ → iter=1 第 1 轮 ik=0/1 的 psi 基本一致（仅符号差异）

### 核心发现

#### 1. Davidson 求解器的 hpsi 范数分叉
**关键数据**：iter=1, ik=1, m=28 的前 5 个 band 的 hpsi 范数（平方和）对比：

| 调用次数 | zdy-tmp 范数 (Band 0..4) | pw-port 范数 (Band 0..4) | 状态 |
|---------|--------------------------|--------------------------|------|
| #1 | `33.97 11.04 11.04 11.04 0.58` | `34.93 10.61 11.40 10.18 0.59` | ✅ **一致** (~3% 差异) |
| #2 | `476.8 397.8 541.9 603.0 473.6` | `478.9 397.5 565.8 598.0 456.9` | ✅ **一致** (~0.4% 差异) |
| #3 | `39.74 37.26 14.16 14.26 14.24` | `38.06 28.59 11.86 10.69 10.90` | ✅ **一致** (~10% 差异) |
| **#4** | **`32.27 71.27 19.00 19.39 21.45`** | **`390.5 299.8 380.1 267.6 350.2`** | ❌ **严重分叉！** (10x 差异) |
| #5 | `37.18 34.58 12.86 12.85 12.62` | `40.66 30.13 13.22 12.04 12.22` | ✅ **恢复一致** |

**推论**：
- 分叉发生在第 4 次 Hψ 调用，但第 5 次恢复。这说明 **H 算子本身没问题**，问题出在 Davidson 子空间迭代产生的 trial vector。
- 第 4 次迭代时，pw-port 的 preconditioner 可能放大了残差，产生了一个“异常”的 trial vector，导致 Hψ 范数激增。

#### 2. psi 符号差异
iter=1 第 1 轮中：
- ik=0 psi: `(-0.000434, 0.001126)` vs `(0.000434, -0.001126)` ⚠️ 符号相反
- ik=2 psi: `(-0.001334, -0.000412)` vs `(0.001334, 0.000412)` ⚠️ 符号相反

#### 3. 数值爆炸链
```
iter=2 ik=0: hpsi → 1e84 → NaN → assertion failure (bpcg_kernel_op.cpp:170)
```

### 下一步排查方向

1. **检查 Preconditioner 输出**：在第 4 次调用前，打印 preconditioner 作用后的 residual norm。
2. **对比 trial vector**：打印 Davidson 生成的新 trial vector 的范数。
3. **检查 k-point 交错影响**：确认 pw-port 为何在 iter=1 时交错处理 k-point。

## 文件变更记录

### 已修改
- `source/source_lcao/module_dftu/dftu.cpp` - DFTU 类重构
- `source/source_lcao/module_dftu/dftu_pw.cpp` - cal_occ_pw 实现及诊断
- `source/source_pw/module_pwdft/op_pw_proj.cpp` - OnsiteProj 实现及 hpsi 诊断
- `source/module_hamilt_pw/hamilt_pwdft/operator_pw/onsite_proj_pw.cpp` - zdy-tmp hpsi 诊断

### 测试记录
- `tests/integrate/2xx/` - 完整的测试案例库及运行输出 (run.out)
- `DEBUG_PROGRESS.md` - 详细调试记录
- `TEST_STATUS.md` - 测试状态汇总
