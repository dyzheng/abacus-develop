# S2 LCAO 统一分配只读读数测试

日期：2026-10-26

## 1. 测试计划

按 `docs/superpowers/plans/2026-10-21-unified-allocation-execution.md` 的 S2 闸门验证：

1. 稠密 LCAO 核心的 `S_T=C^TSC`、`C'=CS_T^{-1/2}`、`n_T`、`m_T` 与逐态求和一致；
2. 奇异目标交叠按秩截断，并报告 `rank` 与 `cond`；
3. 生产钩子只在 `constraint_alloc_report=true` 时输出 `ALLOC`，默认关闭不改变 SCF；
4. h2o_dimer 单帧与两帧 warm relax 中，`sum_n` 与电子数、`sum_m` 与当前密度总磁矩一致；
5. 记录非磁、自旋极化、约束与非约束场景下 `m_T/m_c/m` 的量级和 onsite 目标对照；
6. 约束模块既有测试不得回退。

## 2. 测试设置

- 构建：`cmake --build build -j 32 --target abacus_basic_para`
- 单测：`ctest --test-dir build -R constraint --output-on-failure`
- 生产输入：LCAO、Gamma-only、`ecutwfc=20`、`scf_nmax=100`；
  `constraint_alloc_report=1`，`constraint_alloc_target_l=2`。
- 两帧 relax：`h2o_dimer` 的 `tests/constraint_ct_dimer/m4_r1.tqugAb`，
  两个 Becke 电荷约束，`relax_nmax=2`，`constraint_warm_start=1`，
  `constraint_mu_schedule=alm`。
- 自旋对照：H2O，`nspin=2`，O 原子 onsite 目标分别为 `delta=0.02` 与
  `delta=0.1`，Becke 权重，100 步预算。
- 默认关闭对照：同一 H2O 自旋算例，仅令 `constraint_alloc_report=0`。

## 3. 结果

### 3.1 单元与回归

```text
MODULE_ESTATE_constraint_alloc_lcao                 3/3 PASS
ctest -R constraint                                17/17 PASS (51.28 s)
abacus_basic_para                                  build PASS
git diff --check                                   PASS
```

`alloc_lcao` 覆盖三件事：与逐态求和逐元一致（阈值 `1e-10`）、
选择型目标重构 `C'^T S C'=I`（阈值 `1e-10`）、奇异方向截断且
`rank=1`、`cond=inf`。

### 3.2 h2o_dimer 非磁约束单帧

```text
[constraint] ALLOC frame=0 target_l=2 ntarget=10 norb=46 cond=1.057935545 rank=10
ALLOC audit sum_n=20 nelec=20 dsum_n=-1.978861519e-12
            sum_m=0 tot_mag=0 dsum_m=0
```

O 的两个目标轨道子空间 `n_T` 分别为 `0.1200`、`0.1126`，其余原子没有
`l=2` 目标，`n_T=0`。非磁密度下所有 `m_T/m_c/m` 均为 0。

### 3.3 h2o_dimer 两帧 warm relax

两帧均在 100 步预算内收敛，末能量为 `-903.9374066604425 eV`。

| frame | cond | rank | `n_T(O0,O1)` | `dsum_n` | `dsum_m` |
|---:|---:|---:|---|---:|---:|
| 0 | 1.057935545 | 10 | 0.120002404, 0.112554960 | -2.188e-12 | 0 |
| 1 | 1.057595940 | 10 | 0.119748832, 0.112458076 | -2.494e-12 | 0 |

两帧均为 `CONVERGED`；`sum_n=20`、`sum_m=0`，`dsum_n` 在 `1e-12`
量级，帧间 `n_T` 变化约 `2.5e-4`。非磁两帧验证了生产钩子在
`after_scf` 中能逐帧读取同一 mixed `rho`，不存在跨帧残留。

### 3.4 自旋极化 onsite 对照

O 的目标为 `delta=0.02` 和 `delta=0.1`（即总磁矩目标约 0.02/0.1 μB）。

| 目标 Δ | 原子 | `m` | `m_T` | `m_c` | `m_T/Δ` | onsite 状态 |
|---:|---|---:|---:|---:|---:|---|
| 0.02 | O | 0.02002730 | -0.00296323 | 0.02299053 | -14.8% | 达到总矩目标 |
| 0.02 | H0/H1 | -0.01001359 / -0.01001362 | 0 / 0 | 同左 | 0 | 反平行补偿 |
| 0.10 | O | 0.09986136 | -0.01476481 | 0.11462617 | -14.8% | 达到总矩目标 |
| 0.10 | H0/H1 | -0.04993067 / -0.04993066 | 0 / 0 | 同左 | 0 | 反平行补偿 |

两个目标下 `m_T` 与 onsite 总矩同向相反，幅值稳定在目标的约 15%。
这说明 onsite 约束的“总矩”不能解释为目标 `d` 子空间投影矩；S2 的
`m_T` 是当前实现下必须单独报告的诊断量。两例在 100 步内均为
`RUNNING`，按资源纪律未提高上限，因此这里只用于符号/比例审计，
不作为 S3 收敛验收数据。

### 3.5 默认关闭

关闭 `constraint_alloc_report` 后日志中 `ALLOC` 出现次数为 0；与开启
算例相比，除日期、计时等非物理字段外 SCF 数值一致，两边末能量均为
`-466.4010255694707 eV` 且均 `CONVERGED`。新增报告为纯只读钩子。

## 4. 分析

1. `sum_n` 与 `sum_m` 的闭合误差分别处于 `1e-12` 和 `1e-15`，说明
   DMR 目标投影与按原子 Becke 权重的实空间积分使用了同一电子数口径。
2. 两帧 warm relax 的 `cond` 仅变化 `3.4e-4`，目标子空间稳定；
   `rank=ntarget=10`，未触发截断。
3. onsite 对照暴露了关键物理差异：当前 `onsite=` 约束是原子总自旋矩，
   而 S2 目标集是原子 `d` NAO 子空间；`m_T` 与总矩反向且约为其 15%。
   后续 S3 注入必须约束 `m_T` 本身，不能把现有 `onsite=` 数值直接当作
   分配目标。
4. 报告路径显式拒绝多 k、非共线、复数 DMR 和 dipole 权重；这些路径只
   输出 `ALLOC skipped`，避免把非 R0 数据误当作稠密 Gamma 矩阵。

## 5. 下一步

1. 进入 S3.1：实现 `ΔH=V-PV-VP+PVP+Σλ_iQ_i` 的 LCAO 注入，并以
   小基组稠密构造逐元对拍；
2. 复用 S2 的 `n_T/m_T` 投影驱动 ALM，目标改为 `m_T`，输出每步
   `Σm=M_tot` 审计；
3. S3.2 实现互补投影密度的冻结力，并做位置/目标有限差分；
4. S3 闸门（目标 FD、位置 FD、ALM 100 步收敛）未过前不进入 S4。
