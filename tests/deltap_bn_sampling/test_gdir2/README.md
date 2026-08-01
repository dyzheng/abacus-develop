# BN DeltaP gdir=2（y 方向约束）测试

## 测试目标
验证 `deltap_gdir 2` 沿 y 晶格方向做 Wilson loop / 极化测量的端到端路径
（`deltap_wannier.cpp` k-string 构建 + `DeltapScfSolver` 状态机），补 gdir=1/3
之外的方向覆盖。

## 运行方法
```bash
cd /root/abacus-develop/tests/deltap_bn_sampling/test_gdir2
rm -rf OUT.bn
OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para > test_gdir2.log 2>&1
```

## 关键输入
- `deltap_gdir 2`：约束/测量方向 = y
- `target.dat`：`4.0 / 3.5`（per-atom target，与 center 一致）
- 其余参数与 `center/` 一致

## 预期输出检查
- P1 起 γ 即为 y 方向分支选择结果；P2 触发后 λ 更新，P3 中 γ 收敛到 target 附近：
```
[DeltaP P3] iter=11  γ=(4.000, 3.500) λ=(-6.49e-10, 2.83e-07) |γ-t|=2.833e-04
```
- SCF DRHO 应单调下降至 ~3e-8（本机实测 GE50 DRHO=3.4e-8，接近 scf_thr 1e-8）。

## 现状（2026-08-01 R5）
- 通过：gdir=2 下 γ 收敛到 target（|γ-t| ≤ 2.8e-4），λ 量级 ~1e-7，SCF 稳定。
