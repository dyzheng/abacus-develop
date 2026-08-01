# BN DeltaP 约束掩码（constrain mask 0/1 混合）测试

## 测试目标
验证 STRU 中 `dp_constrain 0/1` 逐原子约束掩码的端到端行为
（`esolver_ks_lcao.cpp` → `DeltapScfSolver::update_lambda_gd` → `deltap_common::gd_update`）：

1. **B 原子**（`dp_constrain 1`，默认）：受约束，P2 阶段 λ 更新。
2. **N 原子**（`dp_constrain 0`）：自由原子，P1/P2/P3 全程 λ 保持 0.0。
3. γ 测量与 branch selection 对两个原子照常进行（掩码只作用于 λ 更新）。

## 运行方法
```bash
cd /root/abacus-develop/tests/deltap_bn_sampling/test_mask
rm -rf OUT.bn
OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para > test_mask.log 2>&1
```

## 关键输入
- `target.dat`：`4.0 / 3.5`（per-atom target；与 center 用例一致）
- `STRU`：N 原子行追加 `dp_constrain 0`
- 其余参数与 `center/` 一致（`deltap_lambda_step 0.01`、`mixing 0.1`、`scf_nmax 50`）

## 预期输出检查
```
[DeltaP P2] iter=11 drho=... < 1.00e-03 → λ updated, mix_reset()
[DeltaP P3] iter=11  γ=(..., ...) λ=(2.58e-03, 0.0e+00) |γ-t|=...
```
- **关键断言**：P3 起 `λ[1] == 0.0e+00`（N 自由，不被更新）；`λ[0] != 0`（B 受约束）。
- 若 N 的 λ 非零，说明掩码未传递到 `gd_update`（回归信号）。

## 现状（2026-08-01 R5）
- 通过：P3 起 λ=(2.58e-03, 0.0e+00)，掩码生效；SCF 50 步内未达 1e-8
  （BN+deltap_corr 历史性振荡 ~1e-4，与 center/test_C_I 行为一致）。
