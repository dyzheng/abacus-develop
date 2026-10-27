# Fe2-bcc 与 LiCl 约束方案对照测试

- 日期：2026-10-27
- 分支：`feat/unified-local-moment-allocation`
- 运行方式：`mpirun -np 4`；`OMP_NUM_THREADS=1`、
  `OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`
- 输入目录：
  `tests/constraint_s5_parity/{fe2_bcc_alloc,fe2_bcc_ds,licl_free,licl_alloc}`

## Test plan

1. Fe2-bcc：在相同结构、k 网格、截断能和目标磁矩下，比较 unified
   allocation 约束与原 native DeltaSpin。目标是确认 allocation 在真实
   multi-k、`nspin=2` 体系中能收敛，并量化总能、约束读数、SCF 成本差异。
2. LiCl：在完全相同的电子密度上，比较 allocation 局域电荷与既有 Becke
   空间权重电荷。目标是区分“同一密度下的不同分片口径”，而不是比较两个
   不同 SCF 解。

## Test setup

### Fe2-bcc

- bcc Fe2，`LATTICE_CONSTANT=8.190`，`2x2x2` Monkhorst-Pack，`kpar=2`
- `ecutwfc=20 Ry`、`ecutrho=80 Ry`、`nspin=2`、`nbands=27`
- `scf_thr=1e-8`、Broyden mixing、`onsite_radius=5.9`
- 共同目标：Fe0 `-3.8 μB`、Fe1 `+3.8 μB`
- native DeltaSpin：`sc_mag_switch=1`、`sc_thr=1e-4`、`nsc=100`
- allocation：`constraint_alloc_switch=1`、`constraint_alloc_target_l=2`、
  `constraint_mu_schedule=outer`、`constraint_trust_region=1`、
  `constraint_thr=1e-4`、`constraint_step_max=0.02`、
  `constraint_step_probe=0.005`

### LiCl

- 15 Å 立方盒；Li `(7.5,7.5,7.5) Å`，Cl `(7.5,7.5,9.52) Å`
- `ecutwfc=100 Ry`、`ecutrho=400 Ry`、Γ 点、`nspin=1`、`nelec=10`
- `scf_thr=1e-8`、`onsite_radius=6.0`、`nbands=8`
- 两套输入仅差 `constraint_alloc_switch`：Becke 为 `0`，allocation 为 `1`
- 共同约束：Li `delta=0.0 e`，初始 `mu=0`。因此两边应严格得到同一密度和
  总能，只比较局域电荷分片。

## Results

### Fe2-bcc

| 指标 | unified allocation | native DeltaSpin |
|---|---:|---:|
| 总能 / eV | -6370.6101496879373 | -6369.985258222084 |
| 约束读数 / μB | `q0=-3.800054394`，`q1=+3.800065059` | `m0=-3.8000818583`，`m1=+3.8000808496` |
| 约束残差 / μB | `6.505918454e-5` | `8.60e-5` |
| 最终乘子 | `mu=-0.01504901489/+0.01443699914 Ry` | `lambda=0.8537309590/-0.8570173106 eV/μB` |
| 电子迭代 | 81 | 13 |
| 墙钟 | 84 s（4 MPI ranks） | 4 s（4 MPI ranks） |
| 状态 | `CONVERGED` | `SCF IS CONVERGED` |

allocation 额外读数：

```text
ALLOC-PW atom=0 n_T=5.93119257 m_T=-3.59872797 n=15.9835093 m=-2.81896681
ALLOC-PW atom=1 n_T=5.93108462 m_T= 3.60022662 n=16.0164907 m= 2.81897748
Orbital Sum Mag(z): Fe1=-2.9267574220, Fe2=+2.9257101666 uB
sum_n=32.00000000, dsum_n=1.33e-11, tot_mag=1.07e-5 uB
```

native DeltaSpin 的轨道电荷分析给出 `Fe1=-3.8001045015`、
`Fe2=+3.8000621499 μB`，与其约束目标一致。

**能量差**：`E_alloc - E_DS = -0.624891466 eV`（相对 `0.00981%`）。

### LiCl

| 方案 | Li 局域电荷 / e | Cl 互补电荷 / e | 总能 / eV |
|---|---:|---:|---:|
| Becke 空间权重 | 3.095906027 | 6.904093973 | -643.97565238003517 |
| unified allocation | 2.947887372 | 7.052112628 | -643.97565238003517 |
| 差 | -0.148018655 (-4.7811%) | +0.148018655 | 0 |

allocation 原始读数：

```text
CONSTRAINT_AUDIT c[0] kind=charge q=2.947887372 t=2.947887372 mu=0 res=0
ALLOC-PW atom=0 n_T=1.99987921 n=3.09590603 n_c=1.09602682
ALLOC-PW atom=1 n_T=1.99579288 n=6.90409397 n_c=4.90830109
sum_n=10.00000000, dsum_n=-2.19e-12
```

## Analysis

- Fe2 的 allocation 在其自身约束定义下严格收敛；数值稳定性依赖
  `constraint_trust_region=1`。不带 trust-region 的普通 secant 在陡峭
  `Q(mu)` 分支上会振荡，不能作为最终结果。
- 相同 `±3.8 μB` 数值目标下，两方案的总能相差 `0.625 eV`，且 allocation
  的传统 orbital-sum 磁矩只有约 `±2.93 μB`。原因不是 SCF 未收敛，而是
  allocation 的 `q=Tr[rho A_i]` 与 DeltaSpin 的 onsite orbital magnetization
  不是同一个算符：前者使用目标壳层投影加互补空间电荷，后者直接约束
  onsite 原子轨道磁矩。因此“同目标数值”不等价于“同物理态”。
- allocation 的 `ALLOC-PW n_T` 是目标壳层占据（Li 约 `2 e`），不是总局域
  电荷；Fe2 中 `m_T` 也不同于最终约束审计量 `q`。对比时必须使用
  `CONSTRAINT_AUDIT q`。
- LiCl 两边总能完全相同且 allocation `mu=0`，说明比较发生在同一密度上。
  Becke 的 `3.095906027 e` 是按空间权重分片；allocation 的
  `2.947887372 e` 是正交目标子空间加互补电荷。两者相差 `0.148 e`
  （`4.78%`），属于定义差异，不是数值误差。`ALLOC-PW n=3.09590603 e`
  恰好接近 Becke 值，但它仍是空间电荷读数，不能替代 `q`。
- 成本上 native DeltaSpin 在本例快约一个数量级（4 s 对 84 s），主要因为
  allocation 每个外层步都需要重建目标空间和互补电荷。

## Verification

- `cmake --build build --target abacus_basic_para -j 14`：PASS
- `ctest -R MODULE_ESTATE_constraint_alloc`：10/10 PASS
- `ctest -R constraint`：23/23 PASS
- 同步修复了参考 SCF 阶段 `targets_` 为空导致
  `constraint_energy_functional()` 越界崩溃的问题；修复后
  `tests/01_PW/211_PW_constraint_h2o` 恢复正常收敛。

## Next steps

1. 定义两套方法之间的可观测量映射，在同一个 `q` 定义下比较 Fe2 总能和力，
   而不是仅比较目标数值。
2. 为 Fe2 allocation 增加自动回归，固定 `±3.8 μB`、trust-region 配置和
   残差阈值，防止后续算子改动破坏收敛。
3. 评估是否将 trust-region 设为 multi-k spin allocation 的推荐默认值，并
   优化目标空间/互补密度的每步缓存成本。

