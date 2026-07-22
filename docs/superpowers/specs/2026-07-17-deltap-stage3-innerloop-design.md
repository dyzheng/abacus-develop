# DeltaP Stage 3 内循环实现 — 设计与现状

> 日期: 2026-07-17

---

## 一、问题诊断

当前梯度下降 + cooldown 方法无法使约束极化 SCF 收敛, 根本原因:

1. **电荷弛豫后 λ 才更新**: 每次 λ 更新后, Broyden mixing 历史被破坏, SCF 需要重新收敛, 但 cooldown 5 步不够
2. **λ 更新在电荷密度更新之后**: `iter_finish` 中 lambda 更新发生在 SCF 迭代结束后, HK 修正要到下一次迭代才生效 → λ 反馈滞后一个 SCF cycle
3. **缺少内循环**: DeltaSpin 用 BFGS 内循环在同一 SCF 步内反复对角化 + 调整 λ, 不动电荷密度。当前 DeltaP 缺少这个机制

## 二、正确流程 (参考 DeltaSpin)

DeltaSpin 在 `hamilt2rho_single` 中 (`esolver_ks_lcao.cpp:452-585`):

```
hamilt2rho_single:
  DeltaSpin 内循环:
    bfgs.start_outer(lambda_init);
    for (inner = 0; inner < nscf; inner++):
      bfgs.step(residual) → trial_λ
      set_lambda(trial_λ) → update H   (不动电荷!)
      diagonalize → get ψ
      compute observable (spin moment)
      bfgs.accept_trial(residual)
    set_lambda(optimized_λ)
    skip_solve = true  (正常对角线已被内循环完成)
  
  正常 SCF 电荷密度更新 (使用内循环优化后的 λ)
```

DeltaP 需要在 `hamilt2rho_single` 的 DeltaSpin 块之后添加类似的 DeltaP 块:

```
  DeltaP 内循环:
    bfgs.start_outer(lambda_init);
    for (inner = 0; inner < deltap_nscf; inner++):
      bfgs.step(residual) → trial_λ
      dp_op->set_lambda(trial_λ)
      dp_op->update_lambda()
      hamiltLCAO->refresh_H()        ← 只需重建含 HK 修正的 H 部分
      diagonalize → get ψ
      dp->compute_gamma_scf() → γ
      residual = target - γ
      bfgs.accept_trial(residual)
    dp_op->set_lambda(optimized_λ)
    skip_solve = true
```

## 三、当前代码状态

### 已有的基础设施

| 组件 | 状态 | 位置 |
|------|:---:|------|
| FletcherReevesCG 优化器 | ✅ | `bfgs.h` |
| DeltaP::compute_gamma_scf | ✅ | `deltap_wannier.cpp` |
| DeltaP::compute_hk_correction | ✅ | `deltap_wannier.cpp` |
| DeltaPOperator::set_lambda | ✅ | `deltap_lcao.h:38` |
| DeltaPOperator::update_lambda | ✅ | `deltap_lcao.h:36` |
| DeltaPOperator::contributeHk | ✅ | `deltap_lcao.cpp` |
| 快速 O_kpair 路径 | ✅ | `deltap_wannier.cpp` |

### 需要新增

| 组件 | 行数 | 位置 |
|------|:---:|------|
| DeltaP 内循环块 | ~60 | `esolver_ks_lcao.cpp:hamilt2rho_single` |
| 移除旧梯度下降 | -20 | `esolver_ks_lcao.cpp:iter_finish` |
| BFGS 初始化调用 | ~10 | `esolver_ks_lcao.cpp:before_scf` |

## 四、风险

1. **hamilt2rho_single 访问 dp_scf_**: 需要确认 DeltaP 对象在内循环中可访问 (通过 `dp_op->get_dp()` 或类似接口)
2. **H 重建**: `update_lambda` 标记后, `hamiltLCAO->refresh_H()` 是否正确只重建 HK 部分? 需要验证
3. **BFGS 状态**: 每次 outer SCF 迭代需要 reset BFGS search direction (`start_outer`)

## 五、建议

实现量为 ~70 行。建议在下次会话中实施, 需要完整测试 DeltaP 内循环 + Stage 1/2 回归。
