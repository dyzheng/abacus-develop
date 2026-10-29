# 2026-10-29 Allocation 电荷归一化：根因定位与分阶段修复计划

> **状态：已修复**（约定 C 全路径统一），验证见
> `2026-10-29-allocation-complex-k-conjugation-fix.md`。

## 1. 测试计划

定位 `ALLOC-SPLIT audit` 的 `sum_n - nelec = +0.376 e`（Fe2 两片段完整划分）。
把偏差拆成三个可独立测量的量：

1. 目标行归属：是否有未归属行（`sum_n_T_all` vs owned）；
2. 波函数/占据归一化：`sum wg |psi_G|^2` 是否等于 `nelec`；
3. 投影恒等式：`sum_a |<phi_a|psi>|^2 + |Q_c psi|^2 == |psi|^2`（G 空间 vs 实空间）。

## 2. 测试设置

同前：Fe2-bcc、`nspin=2`、27 bands、20/80 Ry、kpar=2、目标 ±3.8 μB、
`constraint_alloc_target_l=2`；4 MPI；`build_rel/abacus_basic_para`。
新增诊断输出 `n_T_all/m_T_all/n_band_norm/n_real_norm` 并在
`ALLOC-SPLIT audit` 行打印。

## 3. 结果

### 3.1 三项诊断（修复前）

```text
sum_n=32.3764 nelec=32 dsum_n=0.376354
sum_n_T_all=11.8625  (== owned 目标之和 -> 无未归属行)
sum_n_c=20.5139
all_target_plus_c=32.3764   (目标 + 互补 > N)
sum_band_norm=32.0000       (波函数/占据归一化正确)
sum_real_norm=20.5139       (实空间互补范数 == G 空间互补范数)
```

结论：
- **不是**未归属目标（全部归属）；
- **不是**波函数/占据归一化（band norm 精确等于 N）；
- **不是** FFT/实空间归一化（G 空间与实空间互补范数一致）；
- 是**投影恒等式被破坏**：`|Pψ|² + |Q_cψ|² > |ψ|²`。

### 3.2 正交化约定（数值实验）

用随机复 Hermitian Gram `S` 与复向量做 2×2 数值实验（`H = S^{-1/2}`）：

| 约定 | \|P−P_exact\| | \|P\|²+\|Q\|²−\|ψ\|² |
|---|---:|---:|
| A（现行代码）：`phi_orth = H φ`，`overlap = conj(H) raw` | 2.16e+00 | **+1.18e+01** |
| B：`H φ`，`H raw` | 9.92e-01 | +2.19e+00 |
| **C：`conj(H) φ`，`H raw`** | **1.7e-15** | **−3.6e-15** |
| D：`conj(H) φ`，`conj(H) raw` | 1.02e+00 | +2.28e+00 |

- **约定 C 是唯一正确的配对**；现行代码是 A（`phi_orth=Gφ` 配
  `overlap=conj(G)raw`），在复 Hermitian `G` 下给出**共轭投影**。
- Γ 点 `G` 实数，A 与 C 完全一致 → LiCl（Γ）无法暴露该问题；Fe2 的
  2×2×2 k 网格（广义 k）才暴露，偏差量级 0.376 e 与 1.18e+01 的离散同源。
- 新增单测 `AllocLinalgTest.HermitianInvSqrtIdentity` 固定
  `S^{-1/2} S S^{-1/2} = I`（复 Hermitian），确认 `herm_inv_sqrt` 本身正确
  ——问题只在"旋转算符与重叠系数的共轭配对"。

### 3.3 只改两处 helper 的后果：**SCF 发散**

把 `orthogonalize_pw_overlaps`、`build_pw_alloc_orthogonal_phi` 及导数切到
约定 C 后：

- 分区审计立即正确：`dsum_n = -3.6e-11`（此前 0.376）；
- 但 Fe2 在第一次 μ 注入后的第一个 SCF 迭代 drho 爆到 `2.6e+49`，
  `mpirun` 以 exit 1 结束（无 ABACUS 错误文本，属数值爆炸）。

说明 **operator / force 路径里还有与 helper 不一致的共轭配对**，只改
helper 会让观测量与注入算符错配。已确认至少一处：`build_band_force_data`
（`op_pw_alloc.cpp`）使用 `conj(G) raw` 配 **raw φ** 构造互补，与约定 C
不一致（该路径只在力/应力时调用，不是 SCF 发散的来源，但同样必须统一）。
SCF 的注入算符路径（`context.phi_orth` + 缓存 products）需要逐项审计。

**处置**：为保证分支可用，已回退 helper 的约定切换（Fe2 恢复 55 步收敛、
`dsum_n=0.376` 回到已知状态）；**保留诊断输出与新的单测**。

## 4. 分析

1. 根因是**复 k 下正交化共轭约定的不一致**，不是归一化常数、不是未归属行、
   也不是 FFT 因子。它只影响广义 k；Γ-only 用例（LiCl）永远看不到。
2. 这是一个**跨路径的一致性缺陷**：value/observable、operator、subspace、
   force、stress 五条路径必须使用同一配对。单独改 observer 会让观测量与
   注入算符错配，导致 SCF 发散——这解释了为什么"看起来只是两行"的修复
   不能直接合入。
3. 正确的配对（约定 C）已经用数值实验锁定，且分区恒等式可达 1e-11；
   修复路径明确，但需要系统性重构与全面重验。

## 5. 下一步（分阶段）

1. **盘点所有配对**：grep 全部 `ST_inv_root` / `phi_orth` / `raw_overlap`
   的使用点，列出"旋转算符 × 重叠系数"的组合（至少 value、operator、
   subspace、force、stress 五处）。
2. **统一到约定 C**：`phi_orth = conj(G) φ`、`overlap = G raw`、导数为
   `d(phi_orth) = conj(dG) φ + conj(G) dφ`、`d(overlap) = dG raw + G d raw`；
   包括 `build_band_force_data` 与应力路径。
3. **加矩阵级回归**：对复 Hermitian Gram + 复向量，断言
   `|Pψ|² + |Q_cψ|² = |ψ|²` 至 1e-12（不依赖 Fe2 运行）；再断言
   `Σ_i A_i = I`。
4. **重跑验证**：Fe2 分区审计 `|dsum_n| < 1e-8`；LiCl 电荷读数不变（Γ）；
   力/应力 FD 判据不破（性质2 口径）。
5. 完成前 **charge allocation 不得进入生产路径**；自旋通道因差分抵消
   数值自洽，但 m_c 幅值同样受该约定影响，修复后需重跑语义闭环表。

**产出**：本轮新增 `ALLOC-SPLIT` 诊断（`sum_n_T_all`、`sum_band_norm`、
`sum_real_norm`）、`HermitianInvSqrtIdentity` 单测、本根因报告；
23/23 约束测试通过；代码回退到收敛基线。
