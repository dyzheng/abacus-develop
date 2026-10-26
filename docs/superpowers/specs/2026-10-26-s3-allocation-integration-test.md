# S3 统一分配约束集成与冻结力测试

日期：2026-10-26

## 1. 测试计划

按 `docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
的 S3 闸门验证：

1. `constraint_alloc_switch` 把 `Σμ_i K_i` 注入 Gamma-only LCAO
   Hamiltonian，并由 ALM 驱动到目标 `Tr(DMR K_i)`；
2. 0.01 μB 自旋算例在 `scf_nmax=100` 内收敛，两帧 relax 均收敛；
3. 固定 λ 的位置有限差分与解析 O-z 力一致，判据 `<0.0129 eV/A`；
4. 固定 λ 的目标值有限差分满足 `dE*/dt=-λ`，判据 `<0.02 eV/μB`；
5. 全原子约束集满足 `Σm=M_tot`，判据 `<1e-8`；
6. 约束模块回归和 `abacus_basic_para` 构建不得回退。

## 2. 测试设置

- 构建：`cmake --build build -j 32 --target abacus_basic_para`。
- 回归：`ctest --test-dir build -R constraint --output-on-failure`。
- 体系：H2O，LCAO、Gamma-only、`nspin=2`、`scf_nmax=100`、
  `constraint_mu_schedule=alm`、`constraint_alloc_target_l=2`。
- 自旋目标：O 原子 `delta=0.01 μB`，`constraint_thr=1e-4`。
- 位置 FD：固定 `mu=-0.01221724227 Ry`，O 的 z 坐标为中心差分，
  半位移 `0.005 Bohr = 0.002645886 A`；三个 INPUT 均设
  `constraint_branch_tol=0`、`scf_thr=1e-8`。
- 目标 FD：同一几何与固定 μ，只把目标文件从 `0.02` 改到 `0.0 μB`。
- 求和闸门：O/H/H 三个自旋约束，`constraint_alloc_target_l=1`，
  目标 `[0.01, 0, 0] μB`；粗网格 `ecutwfc=20/ecutrho=80`，细网格
  `ecutwfc=60/ecutrho=240`。
- 两帧 relax：`calculation=relax`、`relax_method=cg`、`relax_nmax=2`、
  `relax_scale_force=0.01`、`force_thr_ev=0`、`constraint_warm_start=1`。
- Becke 对照：同体系同目标，但 `constraint_alloc_switch=0`，走旧
  `Tr[W.DM]` 路径。

## 3. 结果

### 3.1 构建与回归

```text
MODULE_ESTATE_constraint_alloc_constraint     PASS
ctest -R constraint                           18/18 PASS (51.50 s)
abacus_basic_para                             build/link PASS
git diff --check                              PASS
```

构建只有项目既有的 C++17 扩展警告。

### 3.2 单帧自旋 ALM

```text
SCF converged at iteration 63
CONSTRAINT_AUDIT c[0] kind=spin q=0.009999463971
                                  t=0.01000000002
                                  mu=-0.01221724227
                                  res=-5.360541546e-07
[constraint] final status: CONVERGED (targets reached within 0.0001 mu_B)
[constraint] ALM: mu_phys = -0.01221756973 Ry, k = 0.6108621137 Ry/mu_B
!FINAL_ETOT_IS -466.4018565607724 eV
```

`M3b runtime audit: SKIPPED in allocation mode` 正常出现，说明旧
`Tr[W_i.DMR]` 审计没有和 S3 的 `Tr[K_i.DMR]` 口径混用。

### 3.3 两帧 ALM relax

| frame | SCF 状态 | Q (μB) | target (μB) | residual (μB) |
|---:|---|---:|---:|---:|
| 0 | CONVERGED | 0.009999463971 | 0.01000000002 | 5.3605e-7 |
| 1 | CONVERGED | 0.0100763985 | 0.01007639851 | 1.7711e-11 |

两帧残差均 `<1e-4 μB`，末能量 `-466.4054020750246 eV`。第二帧 target
按 `delta` 语义随新几何参考值更新，因此不是固定 0.01。

### 3.4 位置 FD

| 量 | 数值 |
|---|---:|
| `E_plus` (eV) | -466.3984612074121401 |
| `E_minus` (eV) | -466.4048061261813700 |
| `F_FD = -(E+ - E-)/(2*0.002645886 A)` (eV/A) | -1.1990159004 |
| 解析 O-z 力 (eV/A) | -1.2069823210 |
| 绝对差 (eV/A) | **0.0079664206** |
| `E+ + E- - 2E0` (eV) | 4.457879465e-4 |

判据 `<0.0129 eV/A`：PASS。

### 3.5 目标 FD

固定 `mu=-0.01221724227 Ry`，目标 `t_plus=0.02 μB`、`t_minus=0.0 μB`：

| 量 | 数值 |
|---|---:|
| `E_plus` (eV) | -466.4001943196828961 |
| `E_minus` (eV) | -466.4035188018572171 |
| `dE*/dt` (eV/μB) | 0.1662241087161 |
| `-mu` (eV/μB) | 0.1662240491349 |
| 绝对差 (eV/μB) | **5.958116e-8** |

判据 `<0.02 eV/μB`：PASS。

### 3.6 全原子 `Σm=M_tot`

三个自旋约束的 `CONSTRAINT_AUDIT total_charge` 在 allocation 模式下就是
`ΣQ`。细网格末行：

```text
total_charge=4.084049765e-10
ALLOC audit sum_m=-4.242679897e-10 tot_mag=-4.242680765e-10
|ΣQ - M_tot| = 8.33e-10 < 1e-8
```

粗网格 `20/80` 的同一差值为 `2.18e-8`，超过闸门；提高 FFT 网格后降到
`8.33e-10`。该三约束 SCF 在 100 步预算内仍为 `RUNNING`，只用于求和
不变量审计，不作为收敛依据。

### 3.7 Becke 对照与耗时

| 路径 | μ (Ry) | k (Ry/μB) | residual (μB) | SCF 总时间 |
|---|---:|---:|---:|---:|
| S3 allocation | -0.01221724227 | 0.6108621137 | 5.36e-7 | 42.1 s |
| Becke | -0.008282561534 | 0.4141280767 | 3.55e-7 | 40.4 s |

Becke 的 M3b 矩阵/网格审计为 `4.27e-11 e`。两者 μ 相差约 1.47 倍，
不是设计预期的 10 倍；只作记录，不作判据。

## 4. 分析

1. 算子已注册在 Gamma-only `HamiltLCAO` 分支，multi-k 只保留配置期拒绝，
   避免 `constraint_alloc_switch=1` 时静默退回 Becke 注入。
2. S3 的 ALM 读取完整分配量 `Tr[(D_up+D_dn)K_i]` 或
   `Tr[(D_up-D_dn)K_i]`；`n_T/m_T` 只是目标子空间诊断，不能驱动约束。
3. 冻结力通过 `Q_c D Q_c` 和现有权重导数核计算。位置 FD 在细设置的
   `scf_thr=1e-8` 下通过，目标 FD 证明乘子符号与能量修正口径一致。
4. 全原子求和依赖 `Σ_i K_i=S`。粗网格上 `W_i` 的数值积分误差会放大到
   `2e-8`；细网格把误差压到 `1e-10` 以下。这是求积误差，不是算子恒等式
   失败，但 S4/S6 的求和审计应固定使用细网格或改进矩阵积分口径。
5. 两帧 relax 和第二帧极小的残差说明 warm reinit 没有让 ALM/目标空间
   跨几何失效。

## 5. 下一步

1. 进入 S4.1：实现完整 `∂Q̂`，以 S3 冻结力为基线比较，目标
   `<0.005 eV/A`。
2. S4.2 实现应力并做应变 FD；S4.3 补非共线通道。
3. 在 S4 回归中保留细网格全原子求和审计，避免把粗网格求积误差误判为
   分配算符错误。
4. 记录 S3/S4 力差和应力相对误差后，再判断是否可以进入 PW 平价阶段。
