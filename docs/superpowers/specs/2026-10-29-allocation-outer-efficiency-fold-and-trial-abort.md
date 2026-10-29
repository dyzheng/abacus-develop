# 2026-10-29 Allocation 外层效率：trial 早停（负结果）+ fold_potential（正结果）

## 1. 测试计划

在 mixing-reset 修复（82→55 步）之后，继续缩减 Fe2 同靶 ±3.8 μB 的外层成本。
两个独立假设：

1. **R-A trial 早停**：trial SCF 中途若某自旋矩相对已接受锚点塌缩 >30%，
   立即中止该 trial（回到锚点、缩小 trust、重试），避免把注定失败的分支
   SCF 跑完（Fe2 原第 3 步浪费 ~13 次迭代）。
2. **fold_potential**：把 allocation 的局域势 `Σμ_i w_i` 折进已有 Veff 变换，
   使算符只加目标投影修正，省掉每 band 一次 `recip2real→real2recip`
   （针对单次 hPsi 2.2× 的成本）。

**实现前先记录假设**：线性响应无法预测磁分支跳变（同一 trial 线性预测
Δq≈0.065，实际 −1.17），因此 R-A 只能"早发现、早止损"，不能"预测"。

## 2. 测试设置

- Fe2-bcc：同前（`nspin=2`、27 bands、20/80 Ry、kpar=2、`scf_thr=1e-8`、
  目标 ±3.8 μB、`constraint_alloc_target_l=2`、`subspace_rotation=1`）；
  4 MPI，单线程 BLAS/OMP；`build_rel/abacus_basic_para`。
- LiCl：`tests/constraint_s5_parity/licl_alloc`（charge、delta=0、l=0）。
- 基线（mixing-reset 后）：Fe2 **55 迭代 / 15–16 s**，
  ETOT −6370.6101512101。

## 3. 结果

### 3.1 R-A trial 早停：**负结果（已回退）**

- 实现：每次 SCF 迭代用 allocation 矩与 `MuSolver` 的 trial 锚点比较；
  `|q| < 0.7|q_anchor|` 且锚点已进入约束区（`|q_anchor|>0.5|t|`）时，
  调用 `MuSolver::reject` 回退并缩 trust，置 mixing reset。单测
  `AllocTrialGuardAbortsCollapsedBranch` 通过。
- Fe2：guard 在 `outer_step=2` 正确触发一次
  （`moment 2.66 < 0.7 × 3.83`），但运行结果：

  ```text
  200 次电子迭代，final status: RUNNING，ETOT = -6370.610194（另一状态）
  ```

  **比不早停的 55 步更差**。
- 分析：中止 SCF 只能恢复 μ，**不能恢复密度**；trial 的坏支密度留在内存里，
  重试时 SCF 要从坏支弛豫回参考支，代价高于让该 trial 收敛后由 trust 在
  收敛点一次性拒绝。现有"收敛后按 rho 拒绝"是正确的机制，R-A 已回退。

### 3.2 fold_potential：**正结果（默认改为开启）**

同二进制、同 55 迭代、ETOT 逐位一致：

| 配置 | 迭代 | 墙钟 | hPsi 总 | 单次 hPsi | hamilt2rho |
|---|---:|---:|---:|---:|---:|
| `fold_potential=0` | 55 | 16 s | 10.80 s / 1543 | **7.00 ms** | 14.04 s |
| `fold_potential=1` | 55 | **13 s** | 6.06 s / 1543 | **3.93 ms** | 8.90 s |
| DeltaSpin 参照 | 13 | 2 s | 0.97 s / 289 | 3.36 ms | 1.43 s |

- 单次 hPsi **1.78×** 改善，相对 DeltaSpin 仅 **1.17×**（评审门槛 `<1.5×`）；
  墙钟 −19%，ETOT/迭代数不变。
- LiCl（charge，delta=0）：fold 前后 `q=2.947887372`、ETOT 完全相同。
- 结论：`constraint_alloc_fold_potential` 默认值由 `false` 改为 **`true`**
  （`input_parameter.h` + INPUT 说明同步；`hamilt_pw.cpp:106/283` 已正确门控，
  非 allocation 路径不受影响）。

### 3.3 剩余差距的构成

- 迭代数：allocation 55 vs DeltaSpin 13（**4.2×**）；
- 单次 hPsi：1.17×（已接近）；
- 墙钟：13 s vs 2 s（6.5×，固定开销占比在 DeltaSpin 的 13 步里更高）。

因此剩余差距**几乎全部是外层迭代数**。

## 4. 分析

1. R-A 的失败说明：磁分支问题不能靠"中途止损"解决，因为止损点上的密度已经
   在坏支；需要的是**不进入坏 trial**（更保守的 step/continuation）或
   **在坏支收敛后快速拒绝**（现状）。
2. fold_potential 把单步成本拉回评审门槛以内，说明"per-hPsi 2.2×"是可以用
   工程手段消掉的；剩余 4.2× 的迭代差是算法结构问题。
3. 线性响应层（`J = Q_ρ(I−T_ρ)^{-1}T_μ`）对**收敛尾巴**（55 步中约 3 步）
   有效，但对 Fe2 的分支跳变无效；要减少外层步数，真正需要的是
   **响应层 + 分支延拓/更保守 trial** 的组合，或放弃 allocation 的通用约束
   路径、只用 DeltaSpin。

## 5. 下一步

1. **响应层仍按 D6/D7 时间盒评估**，但明确目标应是"减少**已接受**步数
   （尾巴）"而非"避免分支跳变"；分支跳变需 continuation/更小 trust。
2. **电荷互补归一化**（`dsum_n=0.376`）仍未修，优先级高于进一步性能。
3. 若不做响应层：当前状态（mixing-reset 55 步 + fold 3.93 ms/hPsi）可作为
   allocation 的实用基线；Fe 类 d 矩约束仍推荐 DeltaSpin。
