# 2026-08-03 DeltaP 锚点重建轮（B-6 后，生产设置 ecutwfc=100 + ecutrho=400）

> 触发：评审确认 B-6 修复 + 执行要求（commit → 锚点重建 → A2/C → D-D）。
> 锚点挂在 commit `53f94042d`（相位修复 + B-6 + 防护注释，调试打印已 flag
> 包裹）。本轮：按生产设置重建全部 λ≠0 锚点，内容 = E' + λ 轨迹 + γ +
> deltap_branch.dat；顺带完成 D2 内循环 λ×L 核实。

## 1. 测试计划

1. 生产设置统一：全部锚点用例 INPUT 显式加 `ecutrho 400`（ecutwfc=100
   下 4× 默认本就 400，显式化以对齐 D-D 处方）；relax 用例 ecutwfc
   50→100。
2. 重建范围（评审要求 4）：bn_sampling 9-label + deltap_bn_test +
   deltap_relax + test_stru_target 内循环。
3. 锚点内容（评审要求 3）：E'（FINAL_ETOT_IS）+ λ 轨迹 + γ（逐 iter，
   存 deltap_lambda_gamma.dat 工件）+ deltap_branch.dat（分支选择工件）。
4. D2 核实：内循环（BFGS）模式下 B-6 前后 λ 是否按 L 重收敛
   （算符不变性），用 center + deltap_inner_nmax=5 做 pre/post A/B。
5. 确认 hhrdbg/hkchk/fsdbg 调试打印（已 #if 0）不改变被锚定输出行格式。

## 2. 测试设置

- 二进制：commit `53f94042d` 构建（B-6 生效、调试打印关闭）；
  D2 的 pre-B-6 二进制 = 临时把 3 处 taud 改回 tau（保留相位修复）。
- 运行目录：`/tmp/b6_smoke/anchors/<case>/`；工件
  `deltap_lambda_gamma.dat`（全 iter λ/γ/escon）、`deltap_branch.dat`。
- 用例参数：9-label / bn_test / test_stru_target = ecutwfc 100 + ecutrho
  400 + scf_thr 1e-8；relax = ecutwfc 100 + ecutrho 400 + scf_thr 1e-6 +
  relax_nmax 3。bn_test / test_stru_target 4-rank，其余 1-rank。
- 环境：伪势/轨道 `/root/pporb/apns-*`。

## 3. 结果

### 3.1 results.csv（新锚点，已更新 `tests/deltap_bn_sampling/results.csv`）

| label | γB | γN | λB | λN | E_tot(eV) | drho | iter |
|---|---|---|---|---|---|---|---|
| center | 6.572 | 1.180 | 2.58e-3 | −2.32e-3 | −338.944878 | 1.55e-5 | 50 |
| x_plus | 3.955 | 3.768 | 2.48e-3 | −2.32e-3 | −338.765803 | 3.40e-5 | 50 |
| x_minus | −0.972 | −0.491 | 2.68e-3 | −2.32e-3 | −338.731448 | 2.33e-5 | 50 |
| y_plus | 4.000 | 3.544 | 2.58e-3 | −2.42e-3 | −338.776653 | 2.35e-5 | 50 |
| y_minus | −1.692 | −2.037 | 2.58e-3 | −2.22e-3 | −338.751945 | 6.17e-6 | 50 |
| diag_plus | 3.941 | 3.346 | 2.48e-3 | −2.42e-3 | −338.775754 | 4.13e-5 | 50 |
| diag_minus | 3.655 | 3.400 | 2.68e-3 | −2.22e-3 | −338.780179 | 2.91e-6 | 50 |
| anti_plus | −1.695 | −2.035 | 2.48e-3 | −2.22e-3 | −338.754092 | 8.68e-6 | 50 |
| anti_minus | −1.699 | −2.036 | 2.68e-3 | −2.42e-3 | −338.758172 | 2.41e-5 | 50 |

（γ/λ 为最后 P3 iter 值；完整逐 iter 轨迹见各用例
`deltap_lambda_gamma.dat` 工件——`*.dat` 被仓库 .gitignore 排除，不提交，
路径 `/tmp/b6_smoke/anchors/<case>/`。）

### 3.2 轨迹要点（分支翻转是常态，须按轨迹锚定）

- λ 冻结统一在 iter≈11（P3 起始，drho<1e-3 触发）；9-label 的 λ 均为
  ±(2.2–2.7)e-3 量级。
- γ 分支翻转计数（|Δγ|>1.5 rad/iter）：18–41 次/50 iter——γ 在目标附近
  振荡或跳到其他分支（center/x_minus/y_minus/anti_* 末态 |γ−t|≈2.6–5.8）。
- **与 Jul 版 results.csv 的差异不是 B-6 单独造成**：旧值（γ≈目标、
  λ~1e-5）来自相位修复前 + B-6 前 + R5 重构前的二进制；新锚点整体取代
  旧锚点（相位修复改变 E_HK→SCF，B-6 削弱约束驱动 L 倍）。

### 3.3 其他用例锚点

| 用例 | E'(eV) | 末态 γ | 末态 λ | escon(Ry) | 备注 |
|---|---|---|---|---|---|
| bn_test (4-rank) | −338.716023 | (4.081, 3.557) | (4.74e-5, 8.3e-7) | −1.96e-4 | γ 已近目标 |
| tst_stru_target (4-rank, inner) | −338.310090 | (4.001, 3.500) | (−5.00e-3, −4.39e-3) | +0.0354 | 内循环收敛 γ→target；λ 符号对 SCF 噪声敏感（两次运行 ±，噪声主导，见 3.4） |
| relax (ecutwfc=100) | step1 −489.0203 / step2 ≈−483.9 / step3 ≈−489.02 | 逐离子步见轨迹 | step3 λ=(−1.66e-2, −1.08e-2, −1.08e-2) | −0.169 | 3 离子步未收敛（grad 0.67 eV/Å，relax_nmax=3）；E_HK 随 λ 增（−8.6e-2→−2.9e-1 Ry） |

relax 逐离子步 λ/γ 轨迹已存 `deltap_lambda_gamma.dat` 工件（110 行）。

### 3.4 D2 内循环 λ×L 核实（center + inner_nmax=5，1-rank）

| | pre-B-6（lat0 τ） | post-B-6（分数 τ） | 比值 |
|---|---|---|---|
| inner loop done λ | (−2.0890e-3, −3.5877e-3) | (−2.2711e-3, −9.8011e-4) | l0: 0.92 / l1: **3.66** |
| 末态 γ（iter50） | (4.000, 3.500) | (4.000, 3.500) | 相同（|γ−t| 2.6e-4 / 1.1e-4） |
| E'(eV) | — | −338.559546 | |

**结论**：BFGS 内循环下 λ 按 **L=3.615** 重收敛（N 原子 l1 比值 3.66，
1.2% 吻合）——B-6 算符不变性在收敛 λ 模式成立；B 原子（τ=0，λτ≡0）
不受影响（比值 0.92，噪声级）。**D2 的"内循环 λ ×L"假说确认**；
同步模式（单步冻结）λ 不重收敛（B-6 轮已证），两模式行为差异如实记录。

### 3.5 调试打印格式确认

flag 包裹后 h2o base 复跑：0 条 hhrdbg/fsdbg/hkdbg/hkchk 输出；
λ/γ/escon/E_HK/FINAL_ETOT_IS 与被锚定行逐字节一致（B-6 轮 §3.2 数据
复现）——被锚定输出行格式不受调试打印影响。

## 4. 分析

1. **γ 收敛能力退化（预期，非回归 bug）**：B-6 使同步模式 H_HR 弱 L 倍，
   bn_sampling 的 γ 目标常不可达/分支振荡（9-label 中 4 个近目标、
   5 个远偏）。这不是代码错，是 λ 标定随 τ 约定变化——`deltap_lambda_step`
   若按 F1 备忘录 ×L 重标定可恢复旧驱动强度。锚点如实记录现状。
2. **test_stru_target 的 λ 符号敏感**：γ(λ=0) 已在目标附近（|γ−t|~1e-3），
   内循环 α_opt 首步符号由 SCF 噪声决定 → λ 末态 ±5e-3 皆可出现，γ 均
   保持近目标。该用例不宜做 λ 量级判据（×L 核实改用 center+inner5）。
3. **relax 锚点**：3 离子步 λ 从 −5.5e-3 增长到 −1.66e-2（γ 未达目标 0），
   E_HK 随 λ 增大——约束在 relax 中持续施力，relax 未收敛属 relax_nmax
   限制，非崩溃（B-1 修复回归面 OK）。
4. **锚点可复现性**：全部用例 rc=0；E' 与逐 iter λ/γ 轨迹工件齐备；
   D2 pre/post 二进制的差异仅限 3 处 taud↔tau。

## 5. 下一步

1. A2/C 成对实现 + 成对验证（冻结 λ FD；新预算 A2=+1.85e-3 / C=+3.70e-3
   Ry/Bohr，hhrdbg 打印翻回 #if 1 使用）。
2. D-D 高精度验收（ecutwfc=100 + ecutrho≥400 + scf_thr 1e-8，与锚点同
   设置，run_fd.sh 已支持 ECUTWFC/ECUTRHO/SCF_THR 覆盖）。
3. F1 备忘录：λ_step ×L 重标定决策（bn_sampling 驱动强度）。
4. 清理 hhrdbg/hkchk/fsdbg 打印（A2/C 完成后）；MPI/ASAN 回归。
