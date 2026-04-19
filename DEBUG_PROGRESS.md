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
| 224 PW DFTU S4 XYZ | 4 | non-collinear | ✓ | ✗ | 待验证 |
| 225 PW DFTU S2 FeO | 2 | collinear Z | ✓ | ✗ | iter=2 Davidson hpsi → NaN |
| 252 PW DS S2 XYZ | 2 | collinear XYZ | ✗ | ✓ | 待验证 |
| 253-255 PW DS S4 | 4 | non-collinear | ✗ | ✓ | 待验证 |
| 262-265 PW DFTU+DS | 2/4 | all | ✓ | ✓ | 待验证 |

## 根因分析

### 已排除
- ~~iter=1 cal_occ_pw 调用时机~~ → 即使跳过 iter=1 cal_occ_pw，iter=2 仍然崩溃
- ~~iter=1 VU 矩阵差异~~ → 跳过 iter=1 cal_occ_pw 后，iter=2 的 locale/VU 完全对齐
- ~~DFTU 投影计算 (cal_becp/onsite_proj)~~ → iter=1 前 3 次 hpsi 调用完全一致
- ~~tab_atomic_ 计算~~ → ik=1 时 vkb 值完全一致
- ~~psi 初始化~~ → iter=1 第 1 轮 ik=0/1 的 psi 基本一致（仅符号差异）

### 核心发现

#### 1. Davidson 求解器的 k-point 间 psi 符号差异
iter=1 第 1 轮中：
- ik=0 psi: `(-0.000434, 0.001126)` vs `(0.000434, -0.001126)` ⚠️ 符号相反
- ik=1 psi: `(0.001300, -0.000510)` vs `(0.001300, -0.000510)` ✅ 完全相同
- ik=2 psi: `(-0.001334, -0.000412)` vs `(0.001334, 0.000412)` ⚠️ 符号相反

**问题**：为什么 ik=0 和 ik=2 符号相反，但 ik=1 相同？

#### 2. hpsi 分叉点
iter=1 ik=1 的第 4 次 hpsi 调用时分叉：
```
zdy-tmp hpsi[0]: (-0.57525, 0.0189382)
pw-port hpsi[0]: (-0.123483, 0.56216)
```

#### 3. 数值爆炸链
```
iter=2 ik=0: hpsi → 1e84 → NaN → assertion failure (bpcg_kernel_op.cpp:170)
```

### 下一步排查方向

1. **Davidson 子空间对角化的 k-point 独立性**
   - 检查 `need_subspace` 在 istep=0, iter=1 时为 false 的行为
   - 对比两个分支在 ik=0→ik=1 切换时的 subspace 状态

2. **psi 初始化的符号确定性**
   - 检查 `p_wf_init->initialize_psi` vs `stp.init` 的实现差异
   - 确认 pw_seed 是否正确使用（pw_seed=1 在 INPUT 中）

3. **Hamiltonian 构建的差异**
   - 对比两个分支在 iter=1 时 Hamiltonian 的完整构建过程
   - 检查 `hsolver_pw_obj.solve` vs `phsol->solve` 的差异

4. **gemm_op 调用差异**
   - zdy-tmp: `gemm_op()(this->ctx, ...)`
   - pw-port: `gemm_op()(...)` 缺少 ctx 参数
   - 这可能影响计算精度或设备上下文

## 文件变更记录

### 已修改
- `source/source_lcao/module_dftu/dftu.cpp` - DFTU 类重构
- `source/source_lcao/module_dftu/dftu_pw.cpp` - cal_occ_pw 实现
- `source/source_pw/module_pwdft/op_pw_proj.cpp` - OnsiteProj 实现
- `TODO.md` - 项目进度追踪

### 诊断代码（需清理后提交）
- `source/source_pw/module_pwdft/op_pw_proj.cpp` - HPSI 诊断输出
- `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp` - BECP 诊断输出
- `source/source_pw/module_pwdft/onsite_projector.cpp` - PSI/BECP 诊断输出
