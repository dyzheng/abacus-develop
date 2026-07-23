# BN 体系 DeltaP 两阶段阈值模式测试

## 测试目标
验证以下改进在 BN 体系上的效果：
1. **两阶段阈值模式**：drho 降到阈值后一次性更新 λ
2. **多带组合 target-aware 分支选择**：将 γ 对齐到 target ±0.006 rad
3. **mixing_reset**：清除 Broyden 历史避免电荷晃动

## 运行方法

```bash
cd /root/abacus-develop/tests/deltap_bn_test

# 确保使用改进后的代码
export OMP_NUM_THREADS=1

# 运行测试
/root/abacus-develop/build/abacus_basic_para 2>&1 | tee output.log
```

## 关键参数说明

- `deltap_switch = 1`：启用 DeltaP
- `deltap_corr = 1`：启用约束修正
- `deltap_nscf = 0`：使用梯度下降（非 BFGS 内循环）
- `deltap_lambda_init = 0.0`：从 λ=0 开始
- `deltap_inner_thr = 1e-3`：drho 阈值，低于此值触发 λ 更新
- `deltap_lambda_mixing = 0.1`：λ 更新混合系数
- `deltap_target_file = target.dat`：目标 γ 值（B: 4.0, N: 3.5）

## 预期输出检查

### Phase 1：无约束 SCF 收敛（λ=0）
```
[DeltaP] iter=N max|gamma-target|=? g0=? g1=? l0=0.0000 l1=0.0000
```
- 检查 γ 值是否稳定（多带分支选择应使其接近 target）
- 预期：max|gamma-target| < 0.01 rad（如果多带选择成功）

### Phase 2：λ 更新触发（drho < 1e-3）
```
[DeltaP] iter=N max|gamma-target|=? g0=? g1=? l0=? l1=?
```
- 检查 λ 是否从 0 跳变到小值（预期 ~1e-6）
- 检查 γ 是否保持在 target 附近（±0.006 rad）

### Phase 3：固定 λ 继续收敛
- drho 应继续下降（预期达到 1e-6 ~ 1e-7）
- 能量应稳定振荡（幅度 < 1e-3 Ry）
- 不应出现 drho 反弹或发散

## 成功标准

✅ **γ 对齐**：max|gamma-target| < 0.01 rad
✅ **λ 小**：最终 λ < 1e-5 Ry
✅ **drho 收敛**：drho < 1e-6（或至少 < 1e-5）
✅ **无振荡**：drho 单调下降或小幅振荡（< 1e-6）

## 与 H2O 对比

| 指标 | H2O (target=4.0, 3.0) | BN (target=4.0, 3.5) |
|------|----------------------|---------------------|
| γ 对齐精度 | ±0.006 rad | 预期 ±0.006 rad |
| λ 大小 | ~1e-6 Ry | 预期 ~1e-6 Ry |
| drho 收敛 | 2e-7 | 预期 1e-6 ~ 1e-7 |
| 能量振荡 | 2e-4 Ry | 预期 < 1e-3 Ry |

## 故障排查

### 如果 γ 偏差大（> 0.1 rad）
- 多带分支选择可能失败
- 检查 deltap_branch_enum.dat 中的权重矩阵
- 可能需要增大 K 值（当前 K=3）

### 如果 drho 振荡
- mixing_reset 可能未生效
- 检查代码中 p_chgmix->mix_reset() 是否被调用
- 或尝试增大 mixing_restart 参数

### 如果 λ 过大（> 1e-4）
- γ 未充分对齐，梯度下降步长过大
- 减小 deltap_lambda_step（当前 0.01）
- 或减小 deltap_lambda_mixing（当前 0.1）

## 输出文件

- `output.log`：完整运行日志
- `OUT.bn/`：标准 ABACUS 输出目录
  - `running_scf.log`：SCF 迭代详情
  - `deltap_branch_enum.dat`：分支枚举数据
  - `deltap_gamma.dat`：γ 值历史记录
- `target.dat`：目标 γ 值
