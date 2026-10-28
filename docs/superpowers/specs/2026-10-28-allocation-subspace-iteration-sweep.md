# 2026-10-28 Allocation 子空间内层迭代扫描 + 观测量快路径 A/B

## 1. 测试计划

针对批判性对比评估给出的效率缺口（Fe2 同靶：allocation 约 82 次电子迭代 /
80+ 倍原始墙钟，DeltaSpin 约 13 次），验证两个具体假设：

1. **观测量快路径**：把 `compute_pw_alloc_observables` 限制在"会被消费"的迭代
   （SCF 收敛、armed ALM、inner 调度），能否降低墙钟而不改变任何被接受量。
2. **内层子空间迭代数**：把 `constraint_alloc_subspace_iterations` 从 1 提到
   4 / 8（在 `subspace_rotation=1` 下），能否像 DeltaSpin 的 λ 内环那样减少
   外层 SCF 迭代。

## 2. 测试设置

- 体系与参数同 `2026-10-28-allocation-full-residual-accept.md` §2：
  Fe2-bcc，`nspin=2`，27 bands，`ecutwfc/ecutrho = 20/80 Ry`，`kpar=2`，
  `scf_thr=1e-8`，`scf_nmax=200`，目标 `-3.8/+3.8 μB`，
  `constraint_trust_region=1`、`constraint_step_max=0.02`、
  `constraint_alloc_switch=1`、`constraint_alloc_target_l=2`。
- 二进制：`build_rel/abacus_basic_para`；4 MPI；OMP/OpenBLAS/MKL 单线程。
- 用例副本：`/tmp/opencode/fe2_base`（基线）与 `/tmp/opencode/exp_*`（扫描）。
- 基线（无 `subspace_rotation`）：87 次电子迭代、6 个外层步、CONVERGED、
  `FINAL_ETOT = -6370.6101512716 eV`、墙钟 27 s。

## 3. 结果

### 3.1 观测量快路径 A/B（已回退）

| 配置 | 电子迭代 | 外层步 | FINAL_ETOT / eV | 墙钟 |
|---|---:|---:|---:|---:|
| 未加门控（基线） | 87 | 6 | -6370.6101512716 | 27 s |
| 门控 `conv_esolver \|\| alm_armed \|\| inner_active` | 87 | 6 | -6370.6101512720 | 27 s |

- 结果逐位一致（ETOT 差 4×10⁻¹⁰ eV），迭代数相同，墙钟无可测差异。
- 计时分解（基线，同一日志）：

  ```text
  Operator hPsi            21.36 s  3126 calls  75.8%
  Operator veff_pw         10.70 s  3126 calls
  PW_Basis_K recip2real    10.87 s  87124 calls
  Diago_DavSubspace cal_grad 16.93 s 2778 calls
  ElecStatePW psiToRho      1.30 s   87 calls
  TOTAL (Driver total)     28.17 s
  ```

- 结论：hPsi/Davidson 占主导，观测量不在可测范围内。**该优化已回退**，
  不进入代码。

### 3.2 内层子空间迭代扫描

| 配置 | 电子迭代 | 最终状态 | FINAL_ETOT / eV |
|---|---:|---|---:|
| `rotation=1, iterations=1` | **82** | CONVERGED | -6370.6101511942 |
| `rotation=1, iterations=4` | 180 | 未在 `scf_nmax=200` 内收敛 | 无 |
| `rotation=1, iterations=8` | 200 | RUNNING（预算耗尽） | -6370.6103793233（低 0.23 meV 的另一状态） |
| `rotation=1, exact_subspace=1`（单独） | 116 | 未在 200 内收敛 | 无 |

- `iterations=1` 复现了先前记录的 82 步 / ETOT 基线；
- 增大内层迭代使外层 SCF 显著恶化（180/200 步），`iterations=8` 还落到
  另一个能量低 0.23 meV 的状态——与 `2026-10-28-allocation-full-residual-accept.md`
  中 `alpha=0.5` 的落支现象一致；
- `exact_subspace=1` 单独使用（无 `full_residual_accept`）走 predictor 安装
  路径，116 步不收敛，与安全版 85 步的差异说明该路径必须保持"仅诊断"。

## 4. 分析

1. **观测量不是瓶颈**：hPsi + Davidson 占 75–90%，观测量在计时分辨率以下；
   门槛 #1 的真正内容是"λ=0 不进入算符"，而这已由
   `AllocConstraintPW::get_fold_potential` / `act()` 的零乘子快路径实现。
2. **内层迭代数不能替代响应层**：增加固定密度子空间内的 μ 更新，等价于让
   μ 在冻结密度下"过冲"，随后的密度松弛把 SCF 推入更长/另一条轨迹。这与
   D2 结论（冻结密度 Jacobian 缺失密度响应，`||JB-J0||/||JB|| = 0.45–3.18`）
   互相印证：**在缺少 `(I - T_rho)^{-1}` 响应信息时，任何内层加速都会以
   外层步数或收敛分支为代价**。
3. **语义问题仍未解决**：同一目标 `±3.8 μB` 下，allocation 的
   `q = Tr[ρ A_i]` 达到目标，但传统 orbital-sum 只有约 ±2.93 μB，DeltaSpin
   的 orbital-sum 与目标一致；两者约束的不是同一算符。效率优化之前，必须
   先固定该观测量与 onsite/Becke/实验量的关系。

## 5. 下一步

1. **暂停性能调参**（观测量、内层迭代、预测器均已证否）。
2. 优先完成批判性评估的第 4 条（语义闭环）：在 Fe2 / LiCl 上并列报告
   `q=Tr[ρA_i]`、onsite 轨道投影矩、Becke 矩与 `target_l` 敏感性，明确
   allocation 坐标到底对应哪个物理量；若无法自洽定义，按第 8.3 节门槛
   停止其作为通用约束框架。
3. 若语义闭环通过，再时间盒评估显式响应层（D6/D7：`(I-T_rho)^{-1}` 的
   Krylov 成本与 `J` 条件数），否则保留只读 `ALLOC-PW` 诊断 + exact 基线。
