# S5.1 PW 分配读数生产接线与 LCAO 平价：测试报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S5.1。

## 1. 测试计划

本轮验证以下四件事：

1. 从真实 PW 收敛波函数和 `OnsiteProjector` 读取
   `becp=<alpha|psi>`，构造目标 shell 的 `S_T`、正交化目标空间并输出
   `n_T/m_T`。
2. 用同一 `WeightGrid` 和实空间密度输出原子 `n/m` 及补集 `n_c/m_c`，
   并满足 `sum(n)=nelec`、`sum(m)=tot_mag`。
3. 在同一 H2O 构型、同一目标 shell、同一 Becke 权重、同一固定
   `fixed_mu` 下，PW 与 LCAO 的 `m_T`、原子总磁矩满足 S5 平价阈值：
   `|Delta m_T|<0.02 uB`、`|Delta M|<0.01 uB`。
4. 修复真实 PW 接线暴露的 shell 计数与轨道占据索引问题后，既有
   constraint 单元测试和 PW DFT+U 集成路径保持通过。

## 2. 测试设置

### 2.1 系统与输入

- H2O，单 Γ 点，15 Angstrom 立方盒，8 电子，`nbands=8`，`nspin=2`。
- `ecutwfc=20 Ry`、`ecutrho=80 Ry`、Gaussian smearing
  `sigma=0.002 Ry`、Broyden `mixing_beta=0.7`。
- 数值轨道：`O_gga_7au_60Ry_2s2p1d.orb`、
  `H_gga_8au_60Ry_2s1p.orb`。
- 目标 shell：O 的 `l=2` d shell，`constraint_alloc_target_l=2`。
- `onsite_radius=6.9 Bohr`，Becke 权重，目标文件：

```json
{"constraints": [{"type": "spin", "target": 0.5, "atoms": [0], "fixed_mu": 0.01}]}
```

- PW 与 LCAO 均使用 `scf_thr=1e-7`、`scf_nmax=100`、
  `constraint_alloc_report=1`，且固定同一个 `fixed_mu`。
- LCAO 另设 `gamma_only=1`，其余几何、轨道、约束和权重参数与 PW 相同。

运行目录：

```text
tests/constraint_s5_parity/pw
tests/constraint_s5_parity/lcao
```

### 2.2 实现范围

- 新增 `source_pw/module_pwdft/alloc_pw_report.{h,cpp}`：从
  `OnsiteProjector` 选择目标 `l` shell、读取真实 `becp`、构建低秩目标空间，
  输出 `ALLOC-PW` 块。
- `ESolver_KS_PW::after_scf()` 在
  `constraint && constraint_alloc_report` 时调用只读适配器；不改变 SCF
  势或能量。PW allocation 约束算符尚未接线时，
  `constraint_alloc_switch=1` 显式拒绝，避免关闭网格势却没有替代算符。
- `n_T/m_T` 在 k-pool 之间用
  `Parallel_Reduce::reduce_double_allpool()` 归约。
- `OnsiteProjector` 每类原子的 shell 数由错误的 `nwl` 改为 `nwl+1`。
- `cal_occupations()` 的 `becp` band stride 改为 `npol*nkb`，并分别处理
  共线标量通道与非共线 spinor 通道。

## 3. 结果

### 3.1 非零自旋态 PW/LCAO 平价

PW：

```text
[constraint] ALLOC-PW frame=0 target_l=2 ntarget=5 nband=8
             cond=1.0000734467 rank=5
ALLOC-PW atom=0 n_T=1.24622917e-01 m_T=1.90575050e-03
                  n=6.25543872e+00 n_c=6.13081580e+00
                  m=-1.38249883e-02 m_c=-1.57307388e-02
ALLOC-PW atom=1 n_T=0.00000000e+00 m_T=0.00000000e+00
                  n=8.72280706e-01 n_c=8.72280706e-01
                  m=6.91250032e-03 m_c=6.91250032e-03
ALLOC-PW atom=2 n_T=0.00000000e+00 m_T=0.00000000e+00
                  n=8.72280575e-01 n_c=8.72280575e-01
                  m=6.91248797e-03 m_c=6.91248797e-03
ALLOC-PW audit sum_n_T=1.24622917e-01 sum_m_T=1.90575050e-03
                  sum_n=8.00000000e+00 nelec=8.00000000e+00
                  dsum_n=-1.93178806e-12
                  sum_m=4.04190570e-16 tot_mag=-3.38983969e-15
                  dsum_m=3.79403026e-15
```

LCAO：

```text
[constraint] ALLOC frame=0 target_l=2 ntarget=5 norb=23 cond=1 rank=5
ALLOC c[0] atom=0 n_T=0.1224672456 n_c=6.28544882
                    n=6.407916065 m_T=0.001786301482
                    m_c=-0.01385912568 m=-0.0120728242
ALLOC c[1] atom=1 n_T=0 n_c=0.7960419554 n=0.7960419554
                    m_T=0 m_c=0.006036402904 m=0.006036402904
ALLOC c[2] atom=2 n_T=0 n_c=0.7960419793 n=0.7960419793
                    m_T=0 m_c=0.006036420452 m=0.006036420452
ALLOC audit sum_n=8 nelec=8 dsum_n=-9.681144775e-14
            sum_m=-8.416155663e-10 tot_mag=-8.416154603e-10
            dsum_m=-1.059542335e-16
```

平价差：

```text
|Delta n_T(O-d)| = |0.124622917 - 0.1224672456| = 2.1557e-03 e
|Delta m_T(O-d)| = |0.0019057505 - 0.0017863015| = 1.1945e-04 uB
|Delta M(O)|     = |-0.0138249883 - (-0.0120728242)| = 1.7522e-03 uB
|Delta sum_m|    = 8.42e-10 uB
```

`|Delta m_T|` 和 `|Delta M|` 分别比 S5 阈值小约 167 倍和 5.7 倍。PW 与
LCAO 各自满足 `sum(n)=nelec`；两边的 `sum(m)` 均在 `1e-9 uB` 内等价于零，
因为 O 与两个 H 的原子磁矩相互抵消。

零 `fixed_mu` 基线也已运行：PW `n_T=0.124629573`、LCAO
`n_T=0.1224614715`，`m_T` 均在 `1e-8 uB` 以下；它只验证零态求和，不作为
非零磁矩平价判据。

### 3.2 收敛与输出完整性

- PW：`#SCF IS CONVERGED#`，`FINAL_ETOT_IS=-4.4209007894161573e+02 eV`。
- LCAO：`#SCF IS CONVERGED#`，`FINAL_ETOT_IS=-4.664010255694707e+02 eV`。
- 两边总能不同来自 PW/NAO 基组差异；本轮平价判据只针对同一目标 shell 的
  局域占据量，不比较总能量。
- PW 的标准 `Orbital Charge Analysis` 在修复后不再出现 NaN；O 的
  `s/p/d` 电荷分别约为 `1.9826/5.3321/0.1230 e`，总计 `7.4377 e`。

### 3.3 回归

```text
cmake --build build --target abacus_basic_para -j 14        PASS
ctest --test-dir build -R MODULE_ESTATE_constraint_alloc_pw PASS (4/4)
ctest --test-dir build -R constraint                        PASS (23/23, 109.38 s)
tests/17_DS_DFTU: 11_PW_DFTU_S2_FeO, MPI 4                  PASS (2/2, 3.354 s)
```

## 4. 分析

首个真实 PW 读数只有 `ntarget=0`，根因不是目标选择器，而是
`OnsiteProjector` 把 `nwl`（最大角动量）当成 shell 数，只生成
`l=0..nwl-1`。O 的轨道文件 `Lmax=2`，因此 d shell 完全缺失。改为
`nwl+1` 后 O/H 分别有 `(2+1)^2=9`、`(1+1)^2=4` 个 onsite 行，
`target_l=2` 正确选出 O 的 5 个 d 轨道，`rank=5`、`cond` 约为 1。

扩展 shell 后，`cal_occupations()` 的旧索引暴露出独立越界：它固定使用
`ib*2*nkb`，但 `nspin=2` 的标量波函数每个 band 的 stride 是 `nkb`。
越界读取导致轨道电荷表出现 NaN。改为 `npol*nkb` 并对共线/非共线分别组装
2x2 密度后，标准输出恢复有限值。

PW/LCAO 的 `n_T` 仍有约 `2.2e-3 e` 差异。这是两种基组对同一局域 d 目标
的不同表示，而不是求和或投影错误：两边各自求和规则均通过，且
`m_T` 与原子总磁矩远优于 S5 闸门。当前目标径向函数口径也仍不同：PW 使用
`smoothgen(..., onsite_radius)` 的 onsite 平滑轨道，LCAO 报告使用 NAO
shell。该差异被记录为平价余量；若后续需要收紧 `n_T`，应让 LCAO 报告改用
同一 `build_orb_onsite()` 口径。

## 5. 下一步

1. 接 PW allocation 约束算符到 `HamiltPW`，实现真实 `V psi`、`V alpha` 与
   `apply_pw_alloc_delta_h()`，再做固定目标的 SCF 与位置 FD 力
   `<0.0129 eV/A`。
2. 在 `KPAR>1` 的生产输入上复跑 `ALLOC-PW`，确认新增 all-pool 归约；
   检查 `cond/rank` 是否也需要跨 pool 最大值归约。
3. 生成同一电子态下 Becke 与 Hirshfeld 的 `n_T/m_T/n/m` 三量对照表，并
   完成 S5.2 剩余验收。
4. 对齐 PW/LCAO 目标径向轨道口径，评估是否能将 `|Delta n_T|` 从
   `2.2e-3 e` 进一步降低。
5. 单独调查旧 v1 JSON 在 `iter_finish` 的 segfault；本轮所有输入均使用
   v2 `fixed_mu`，未覆盖该历史路径。
