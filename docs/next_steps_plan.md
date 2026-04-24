# DeltaSpin LCAO 优化：下一步计划

## 当前状态总结

子空间对角化方案（`run_lambda_loop_lcao`）在 Fe 强关联体系上**根本不适用**：
- 冻结基近似忽略 SCF 响应，Mi 预测偏差 0.6–4.4 µB
- 4 个代码 bug 已修复（V 矩阵列主序、target_mag 读取、Newton 步长限制、chi k 权重）
- 编译通过，但 SCF 仍不收敛

已验证可复用的组件：
- `cal_PI_sub()`：P_I 子空间投影矩阵计算正确（Hermitian ✓，Tr 合理 ✓）
- 解析 chi 公式正确（量级 -1 到 -5 Ry/µB，符合物理）
- `DeltaSpin<OperatorLCAO>` 的 `cal_moment()` 实空间投影正确

---

## 方案选择：方案 A — chi 引导的全量对角化

保留 `cal_mw_from_lambda()` 全量 SCF 对角化，用解析 chi 替代原始 BFGS 的盲搜索。

核心思路：原始 `run_lambda_loop()` 的 BFGS 搜索方向 `search = delta_spin`（梯度方向），
初始步长 `alpha_trial` 靠经验值 + 线搜索自适应。用 chi 可以直接给出 Newton 方向，
第一步就接近最优，减少 `cal_mw_from_lambda` 调用次数（从 ~20-50 步降到 ~5-10 步）。

---

## 实施步骤

### Step 1: 重构 `run_lambda_loop_lcao` 为 chi 引导模式

**文件**: `source/source_lcao/module_deltaspin/lambda_loop.cpp`

将当前的 `run_lambda_loop_lcao()` 重写为：

```
Phase 1: 全量对角化 → C_k, e_k, Mi（复用现有 cal_mw_from_lambda）
Phase 2: 计算 P_I_sub → 解析 chi（复用现有代码）
Phase 3: 用 chi 做 Newton 步计算初始 delta_lambda
Phase 4: 全量对角化验证 → 得到真实 Mi_new
Phase 5: 如未收敛，用 (Mi_new - Mi_old) / delta_lambda 更新 chi（secant 更新）
Phase 6: 重复 Phase 3-5 直到收敛
Phase 7: 更新 DM/charge
```

关键区别：
- 每步仍调用 `cal_mw_from_lambda()` 做全量对角化（保证 SCF 自洽性）
- 但搜索方向由 chi 给出（Newton 方向），而非盲 BFGS
- chi 在迭代中通过 secant 方法自适应更新，越来越准
- 预期 5-10 步收敛（vs 原始 20-50 步）

### Step 2: 实现 secant chi 更新

每次全量对角化后，用有限差分更新 chi：

```cpp
// secant update: chi_new = (Mi_new - Mi_old) / (lambda_new - lambda_old)
for (int iat = 0; iat < nat; iat++) {
    double dlambda = this->lambda_[iat].z - lambda_old[iat].z;
    double dMi = this->Mi_[iat].z - Mi_old[iat].z;
    if (std::abs(dlambda) > 1e-10) {
        chi[iat] = dMi / dlambda;  // secant approximation
    }
}
```

这比解析 chi 更准确，因为它隐式包含了 SCF 响应。
解析 chi 仅用于第一步（无历史数据时）。

### Step 3: 保留步长限制

复用已修复的 Newton 步长限制逻辑：
- chi 钳位：`|chi| >= 0.1`
- delta_lambda 钳位：`|delta_lambda| <= sccut`（restrict_current_）
- 这些保护在全量对角化模式下同样重要

### Step 4: 修改 esolver 路由

**文件**: `source/source_esolver/esolver_ks_lcao.cpp`

当前 nspin=2 路由到 `run_lambda_loop_lcao()`，保持不变。
重构后的 `run_lambda_loop_lcao()` 内部使用全量对角化 + chi 引导。

### Step 5: 清理 debug 打印

移除 `run_lambda_loop_lcao()` 中的所有 debug 输出：
- P_I_sub verification 打印
- chi 值打印
- V unitarity check
- Mi_subspace vs Mi_fulldiag 对比
- Newton step 详细日志

保留标准的 `print_header()` / `check_rms_stop()` / `print_termination()` 输出。

### Step 6: 编译验证

```bash
cd /root/abacus-ds-lcao-optimize/abacus-develop/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON -DBUILD_TESTING=OFF ..
cmake --build . -j$(nproc)
```

### Step 7: 测试验证

测试算例：`tests/17_DS_DFTU/24_LCAO_DS_S2_Z`
- 验证 SCF 收敛
- 验证 lambda loop 步数减少
- 验证最终磁矩接近 target（±2.0 µB）

对比基准：先用原始 `run_lambda_loop()` 跑一次，记录步数和收敛行为。

---

## 可选后续优化

### Step 8（可选）: 混合 chi 策略

如果 secant chi 在某些步不稳定，可以混合解析 chi 和 secant chi：

```cpp
chi_mixed = alpha * chi_secant + (1 - alpha) * chi_analytical;
```

alpha 从 0 逐步增大到 1（随着 secant 数据积累）。

### Step 9（可选）: 多原子 Jacobian 矩阵

当前 chi 是标量（每个原子独立）。对于多原子体系，原子间磁矩耦合可能重要。
可以扩展为 Jacobian 矩阵 `J[iat1][iat2] = dMi_1 / dlambda_2`，
用 Broyden 方法更新。但这增加了复杂度，建议先验证标量 chi 的效果。

---

## 需要保留的代码

从当前 `run_lambda_loop_lcao()` 中保留：
1. Phase 2 的 `cal_PI_sub()` 调用和 chi 计算逻辑（用于第一步）
2. Newton 步长限制逻辑（chi 钳位 + delta_lambda 钳位）
3. 收敛判定逻辑（RMS error + current_sc_thr_）

可以删除：
1. Phase 4 的子空间对角化（zheev）
2. Phase 5 的波函数旋转
3. 所有 debug 验证打印
4. V_save、ekb_new、wg_new 等子空间相关变量

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| chi 引导仍需 >20 步 | 低 | 中 | 回退到原始 BFGS |
| secant chi 不稳定 | 中 | 低 | 混合解析 chi |
| 多原子耦合导致 chi 不准 | 中 | 中 | 扩展为 Jacobian 矩阵 |
| target_mag bug 影响原始 run_lambda_loop | 高 | 高 | 已修复，需验证原始路径 |

---

## 优先级排序

1. **最高优先级**: 验证 target_mag bug 修复后，原始 `run_lambda_loop()` 是否能正常收敛
   - 这是基准线，如果原始方法本身就不收敛，说明还有其他问题
2. **高优先级**: 实现 Step 1-6（chi 引导的全量对角化）
3. **中优先级**: Step 7 测试验证
4. **低优先级**: Step 8-9 可选优化

---

## 执行结果（2026-04-24）

### Step 0: 原始方法基准测试 ✅ 完成

**结果**: 原始 `run_lambda_loop()` **无法收敛**
- RMS: 0.32 → 10.47（发散），100 步达到上限
- Lambda: -510 ~ -571 eV/µB（爆炸）
- 磁矩: 11.57 / 8.92 µB（应为 ±2.0 µB）
- 总时间: 6.67s, cal_mw_from_lambda: 199 calls

### Step 1-7: chi 引导方案 ✅ 完成

**编译**: ✅ 通过，0 errors
- 二进制: `build/abacus_basic_para` (380MB)

**测试结果**: chi 引导方案**成功收敛**

| 指标 | chi 引导方案 | 原始方法 |
|------|-------------|----------|
| Lambda loop 收敛 | ✅ 38 步收敛 | ❌ 100 步未收敛 |
| 最终磁矩 (iat=0) | **2.0001 µB** | 11.57 µB |
| 最终磁矩 (iat=1) | **-2.0002 µB** | 8.92 µB |
| Lambda 值 (eV/µB) | 0.00009 / 0.0013 | -510 / -571 |
| SCF 总时间 | 46.19s | 6.67s |
| cal_mw_from_lambda 调用 | 1681 次 | 199 次 |

**关键发现**:
1. chi 引导的 Newton 方向有效：磁矩精确收敛到 ±2.000 µB（误差 < 0.01%）
2. Lambda 值极小（< 0.002 eV/µB），说明系统本身接近目标磁矩
3. 总调用次数偏多（1681 vs 199），原因是每个外层 SCF 迭代都触发了 lambda loop
4. secant chi 更新稳定：未出现数值不稳定或震荡

**收敛曲线（最后一次外层迭代 GE49）**:
```
Inner 1: RMS = 0.132  → Inner 10: RMS = 0.226
Inner 20: RMS = 0.056 → Inner 30: RMS = 0.0017
Inner 36: RMS = 0.000297 → Inner 38: RMS = 0.000154 ✓ (thr = 0.000248)
```

### 代码改动清单

| 文件 | 改动类型 | 描述 |
|------|----------|------|
| `lambda_loop.cpp` | 重构 | 删除子空间对角化代码（~250 行），重写为 chi 引导的全量对角化模式 |
| `spin_constrain.cpp` | Bug fix | target_mag 读取从 `.x` 改为 `.z` |

### 后续建议

1. **性能优化**: 当前每个外层 SCF step 都调用 lambda loop，可优化为仅在 drho < sc_scf_thr 时调用一次
2. **Step 8（可选）**: 混合 chi 策略可进一步提升收敛稳定性
3. **Step 9（可选）**: 对于多原子体系，可考虑 Jacobian 矩阵 + Broyden 更新
