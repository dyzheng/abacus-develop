# 2026-10-21 M5 偶极约束坐标 -- 测试报告

计划：`docs/superpowers/plans/2026-10-21-m5-dipole-constraint.md`
设计：`docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §4

## 1. 测试计划

1. **IO/参数层**：验证 dipole JSON v2 的 frame/axis/saw/unit 解析、V -> e*Bohr
   换算、单全局约束限制，以及 atoms/efield/dip_cor/nspin=4/fixed_mu 自动选面等互斥守卫。
2. **权重与观测层**：验证 saw/molecular 权重加离子偏移与被动
   `DipoleObserver` 逐值对拍，并覆盖 MPI 下偏移只能归约一次。
3. **力与循环层**：验证 saw/molecular 的原子偏移导数、解析力、自动 saw 平面、
   ALM/审计和状态单位。
4. **集成层**：h2o dimer LCAO，np8/OMP1，固定 `scf_nmax=100`，显式
   `saw_pos=0.9`、delta 目标 0.01 e*Bohr。验证靶值闭合、Q 与被动 `p_z`
   同值、deltaV 口径、M3b 矩阵/网格审计，以及有限约束力。
5. **预算纪律**：100 步不收敛时只记录 drho/残差，禁止提高 `scf_nmax`。

## 2. 测试设置

- 机器：Intel(R) Core(TM) Ultra 5 225H；ABACUS v3.11.0-beta.1。
- 集成并行：`OMP_NUM_THREADS=1 mpirun --allow-run-as-root -np 8`。
- 基态：`tests/constraint_ct_dimer/m5_dipole/`，15 Å 立方盒、6 原子、LCAO、
  `nelec=20`、`ecutwfc=60`、`ecutrho=240`、`scf_thr=1e-7`、
  `scf_nmax=100`、Broyden 0.7、`constraint_thr=1e-4`。
- 约束：单个 `type=dipole`，`frame=saw`、`axis=z`、`saw_pos=0.9`、
  `saw_pos_dec=0.1`、`unit=ebohr`、`constraint_mu_schedule=alm`。
- 目标：`constraint_target_mode=delta`、`target=0.01 e*Bohr`。因此最终
  target 为 `Q_ref + 0.01 = -6.7209103342 e*Bohr`。
- 运行目录中的 `OUT.*` 与 `run.*.log` 均为 scratch，不进入提交。

## 3. 结果

### 3.1 单元与构建

- 首轮全量 `ctest -R constraint`：**13/13 PASS**，61.67 s。
- 最终 MPI 偏移修复后：`MODULE_ESTATE_constraint_weight_grid_mpi`、
  `..._mpi_4np`、`MODULE_ESTATE_constraint_observe` **3/3 PASS**。
- 新增 `WeightGridMpiTest.DipoleObserveReducesOffsetOnce`：4 rank 下
  `Q[0]` 与被动 `saw_dipole` 在 `1e-9` 内一致，并检查所有 rank 同位。
- 最终全量回归在用户要求提交时中止；此前同一 constraint 路径已全量通过，
  最终修复后重点 MPI/observe 目标通过，主程序 `abacus_basic_para` 构建成功。

### 3.2 集成 1：0.05 e*Bohr，暴露 MPI 偏移重复计数

首跑 100 步未收敛，末步 `DRHO=1.8774e-6`；审计 `Q=140.4501459`，而被动
报告 `p_z=-6.68068 e*Bohr`，两者严重不一致。根因为
`ConstraintObserver` 在 pool allreduce **之前**加入离子偏移，np8 下偏移被累加
8 次。修复为“先归约密度积分，再加一次 rank-independent offset”。

修复后同目标仍为 100 步未收敛：

| 量 | 结果 |
|---|---:|
| 末步 drho | 1.8774e-6 |
| Q / target / residual | -6.681156004 / -6.680910334 / -2.4567e-4 e*Bohr |
| 状态 | RUNNING（SCF 未在 100 步内收敛） |
| 审计 Q 与被动 p_z | 已同量级并对应同一 saw 平面 |

分析：ALM 第二步在 iter 77 得到 residual `1.4739e-4`，仅略高于
`constraint_thr=1e-4`；第三步后反向到 `-2.4567e-4`。SCF 本身在收敛，
但 100 步预算内没有完成最后一次 ALM settle。未提高 `scf_nmax`。

### 3.3 集成 2：0.01 e*Bohr，最终验收

改为更小的 0.01 e*Bohr delta 后，在 iter 77 同时完成 SCF 与靶值判据：

```text
[constraint] final status: CONVERGED (targets reached within 0.0001 e*Bohr)
CONSTRAINT_AUDIT c[0] kind=dipole q=-6.72088832 t=-6.720910334 mu=-0.000376661988 res=2.201376926e-05
[constraint] ALM: mu_phys = -0.00037446061104 Ry, k = 0.1 Ry/e
[constraint] dipole coordinate: frame=saw, axis=z, target=-6.7209103342 e*Bohr, saw_pos=0.9, saw_pos_dec=0.1, deltaV=-2.8602705391 V
  p      = (0, 0, -6.7208883204) e*Bohr
  p_ref  = (0, 0, -6.7309103342) e*Bohr
```

| 判据 | 结果 |
|---|---|
| Q 命中 target | residual 2.201e-5 < 1e-4 e*Bohr |
| 审计 Q 与被动 p_z | 差值 8.7e-9 e*Bohr（同一收敛密度） |
| deltaV | `4*pi*Q/A` = -2.8602705391 V，与报告一致 |
| M3b `Tr[W.DM]` vs `int w rho` | 1.3663e-6 e |
| 力 | `#TOTAL-FORCE` 正常输出，全部有限，最大分量约 9.80 eV/Å |
| 墙钟 | 71 s（含 1.15 s 力计算） |

drho 轨迹（关键点）：

| iter | 30 | 31 | 53 | 54 | 65 | 73 | 77 |
|---|---:|---:|---:|---:|---:|---:|---:|
| drho | 7.71e-8 | 3.85e-3 | 8.84e-8 | 4.19e-3 | 1.35e-4 | 2.26e-7 | 7.92e-8 |

分析：reference 在 iter 30 收敛；ALM 探针与最终 μ 修正各引入一次可预期的
drho 跳变，随后都在预算内回到 `scf_thr`。没有出现 0.05 目标下的末段振荡。

### 3.4 M3b 审计口径修复

审计 Q 包含 dipole 的离子偏移，而 `Tr[W.DM]` 只表示密度权重项。首版直接把
含偏移的 Q 传给 M3b，得到 `21.0187587872 e` 的假偏差。修复为比较前减去
`WeightGrid::observable_offset(alpha)`；普通 charge/spin 偏移为 0，历史结果不变。
修复后最终偏差 `1.3662975924e-6 e`。

## 4. 遗留风险

1. `saw_pos` 的坐标约定沿用 ABACUS 分数 `taud.z`；saw 偶极对跳变面位置敏感，
   集成必须显式给出或确认自动选面落在真空区。
2. 自动 saw 平面与 warm start / fixed_mu 互斥；当前只在冷 reference-first
   路径支持自动选面。
3. molecular + absolute 对带电体系仍依赖 Cartesian 原点；delta 语义可消去。
4. 0.05 e*Bohr 目标在 ALM 单层调度下需要超过 100 步完成最后一次 settle；
   本次按预算纪律保留 RUNNING 证据，不提高上限。

## 5. 下一步

1. 补跑最终被中止的完整 `ctest -R constraint`，确认仅文档/审计改动后仍为
   13/13。
2. 若后续仍需 0.05 e*Bohr 单点闭合，评估 ALM settle 的末段阻尼或更优初始
   μ 预测，而不是提高 `scf_nmax`。
3. M5 合入后进入依赖偶极坐标的界面/电场场景验证，并保持 saw 平面自动选择
   的真空区守卫。
