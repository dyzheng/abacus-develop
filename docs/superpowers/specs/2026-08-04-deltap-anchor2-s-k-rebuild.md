# 2026-08-04 DeltaP 锚点重建 #2（S_k 键修复 + 相位配对 sort 修复后）

> 触发：执行 TODO Stage 0.3（`2026-08-04-deltap-execution-todo.md`）。
> 锚点重建 #2 是计划内第二次（S_k 修复后一次、Route A+ operator 模式一次）。
> 驱动原因：`deltap_wannier.cpp` 相位配对 sort 负距离 bug 修复改变串行逐原子 γ
> （co 串行 (-6.711,-9.209) → (-6.702,-9.219)），λ≠0 锚点全部需要按
> commit `f49f9fcae` 重建。

## 1. 测试计划

1. 12 用例全部 rc=0：bn_sampling 9-label（1-rank）+ deltap_bn_test（4-rank）+
   test_stru_target（4-rank, inner loop）+ deltap_relax（1-rank, 3 离子步）。
2. E'/λ/γ/branch 轨迹存档：`/tmp/deltap_anchor2/`（run.log +
   deltap_lambda_gamma.dat + deltap_branch.dat）。
3. `tests/deltap_bn_sampling/results.csv` 用新锚点重新生成。
4. 与 B-6 时代锚点（commit `1b2625fdd`）对比，标注差异来源。

## 2. 测试设置

- 二进制：`build/abacus_basic_para`（f49f9fcae 修复后，MPI+Release）。
- 脚本：`/tmp/deltap_anchor2.sh`（bn_sampling 3 并行、bn_test/test_stru_target
  4-rank、relax 1-rank；OMP_NUM_THREADS=1）。
- 用例参数（沿用 B-6 轮）：bn_sampling/bn_test/test_stru_target =
  ecutwfc 100 + ecutrho 400 + scf_thr 1e-8；relax = ecutwfc 100 + ecutrho 400 +
  scf_thr 1e-6 + relax_nmax 3。环境伪势/轨道 `/root/pporb/apns-*`。
- **格式注意**：`deltap_lambda_gamma.dat` 是 B-6 轮临时补丁产物（当前二进制不写）。
  本轮从 run.log 的 `[DeltaP P*]` 行重新提取轨迹（内容等同），归档与仓库
  `*.dat` 均已替换为新鲜提取；drho 列为 P2 触发行（iter≈11）的 drho 值
  （当前二进制仅在此处打印 drho）。

## 3. 结果

### 3.1 12 用例 rc=0 ✅（全部正常终止）

| 用例 | rank | rc | SCF | 备注 |
|---|---|---|---|---|
| bn_sampling 9-label | 1 | 0 ✅ | 50 iter 未收敛（deltap 协议内） | 全部正常终止 |
| deltap_bn_test | 4 | 0 ✅ | 50 iter | E/γ 与 1-rank 同量级 |
| test_stru_target | 4 | 0 ✅ | 30 iter + inner loop 3 步 | 内循环收敛 λ 应用后 γ→target |
| deltap_relax | 1 | 0 ✅ | 3 离子步 | 能量单调下降 |

### 3.2 results.csv（新锚点，commit f49f9fcae）

| label | γB | γN | λB | λN | E_tot(eV) | drho | iter |
|---|---|---|---|---|---|---|---|
| center | 3.986 | 3.532 | −2.9e-6 | −1.7e-6 | −338.713148 | 6.92e-6 | 50 |
| x_plus | 4.097 | 3.534 | −4.2e-6 | +2.3e-5 | −338.713786 | 6.92e-6 | 50 |
| x_minus | 3.900 | 3.499 | −7.2e-6 | +2.3e-5 | −338.713636 | 6.92e-6 | 50 |
| y_plus | 4.025 | 3.698 | −6.6e-6 | +2.1e-5 | −338.713652 | 6.92e-6 | 50 |
| y_minus | 3.965 | 3.526 | −6.6e-6 | +6.9e-7 | −338.713026 | 6.92e-6 | 50 |
| diag_plus | 4.100 | 3.605 | −4.2e-6 | +2.1e-5 | −338.713755 | 6.92e-6 | 50 |
| diag_minus | 3.902 | 3.382 | −7.2e-6 | +6.9e-7 | −338.712997 | 6.92e-6 | 50 |
| anti_plus | 4.099 | 3.400 | −4.2e-6 | +6.9e-7 | −338.713145 | 6.92e-6 | 50 |
| anti_minus | 3.920 | 3.514 | −7.2e-6 | +2.1e-5 | −338.713673 | 6.92e-6 | 50 |

（γ/λ 为最后 P3 iter=50 值；完整逐 iter 轨迹见归档
`/tmp/deltap_anchor2/bn_sampling/<label>/deltap_lambda_gamma.dat`。）

### 3.3 与 B-6 时代锚点（1b2625fdd）的差异——关键结论

| 量 | B-6 时代（相位 bug 生效） | 本轮（修复后） |
|---|---|---|
| 末态 \|γ−t\|（9-label 中位数） | 2.6–5.8 rad（5/9 远偏目标） | **1e-3–1.3e-1 rad（全部近目标）** |
| λ 量级 | ±(2.2–2.7)e-3 | **±(0.7–23)e-6（近零）** |
| E_tot(eV) | −338.73 ~ −338.94（label 间差 ~0.21） | **−338.712997 ~ −338.713786（label 间差 <1e-3）** |
| bn_test γ | (4.081, 3.557) | (4.016, 3.515) |
| bn_test λ | (4.74e-5, 8.3e-7) | (8.4e-7, 7.2e-6) |

**分析**：相位配对 bug 修复前，近简并带的 Hungarian 平局被负距离贪心错误配对，
γ 权重配到错误带上 → 残差方向错 → λ 被驱动到 ±2e-3 且 γ 停在错误分支
（|γ−t|~2.6–5.8）。修复后约束达到（|γ−t| 全部 <0.13 rad，多数 <0.04），
λ 自然衰减到 ~1e-6（约束近乎自洽）。这**不是回归**，是 S_k/sort 修复恢复
约束一致性的直接证据（与 Stage 0.2 的 co/hf f0 逐原子 γ 串行==4-rank 闭合一致）。

### 3.4 bn_test / test_stru_target / relax 锚点

| 用例 | E_tot(eV) | 末态 γ | 末态 λ | escon(Ry) | 备注 |
|---|---|---|---|---|---|
| bn_test (4-rank) | −338.713628 | (4.016, 3.515) | (8.4e-7, 7.2e-6) | −2.9e-5 | \|γ−t\|=1.6e-2 |
| test_stru_target (4-rank, inner) | −339.929763 | (4.007, 3.497) | (1.50e-2, 1.31e-2) | −0.1059 | inner loop 3 步，\|γ−t\|=6.7e-3 |
| relax step1 | −483.568406 | (−7.959, −2.379, −2.379) | (−7.96e-3, −2.38e-3, −2.38e-3) | −0.0747 | grad 0.589 eV/Å |
| relax step2 | −485.400820 | (−7.938, −2.374, −2.374) | (−1.59e-2, −4.75e-3, −4.75e-3) | −0.1488 | grad 1.814 eV/Å |
| relax step3 | −487.314027 | (−7.953, −2.379, −2.379) | (−2.38e-2, −7.13e-3, −7.13e-3) | −0.2236 | grad 0.288 eV/Å |

（relax 3 离子步能量单调下降 −483.57 → −485.40 → −487.31，λ 随步增长；
γ 恒在 ~(−7.95, −2.38)（target 0），约束未达——relax_nmax=3 限制，非崩溃。）

## 4. 分析

1. **锚点整体换代**：9-label 的 γ/λ/E 全部改变，是 S_k 键修复 + 相位 sort 修复的
   预期结果；新锚点"约束自洽"（γ→target、λ→0）比旧锚点物理上更合理。
2. **Stage 0 出口检查**：git 干净（除待提交文档）；hf/co 4-rank vs 串行
   逐原子 γ 一致已在上轮 dev log 记录（-9.173,-2.767 / -6.702,-9.219）。
3. **轨迹存档完整性**：12 用例 run.log + deltap_lambda_gamma.dat +
   deltap_branch.dat 齐备；`*.dat` gitignored 不提交，归档在
   `/tmp/deltap_anchor2/`。

## 5. 下一步

1. Stage 1.1a/b：`deltap.h` 加 `gamma_op_` 成员 + `compute_operator_observable()`
   声明；`compute_gamma_scf` 顺带累加 Γ_I^HR。
2. Stage 1.1c/d：`compute_hk_correction` 按原子拆 Γ_I^HK；T0 对照
   （per-k vs 实空间 hhrdbg p_hat，<1e-10，不一致即停）。
3. Stage 1.2/1.3：observable 状态机切换 + 外循环 secant。
