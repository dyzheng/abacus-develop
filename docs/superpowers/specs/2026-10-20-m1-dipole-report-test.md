# 2026-10-20 M1 偶极报告口径 — 测试报告

计划：`docs/superpowers/plans/2026-10-20-m1-dipole-report.md`
设计：`docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §3 模块 A

## 1. 测试计划

1. **单元层**：验证 `DipoleObserver` 的分子偶极（中性/带电、原点无关性）、离子中心、
   q_dip 符号、saw 形状、saw 偶极、ΔV、自动选面与跳变面守卫；验证 INPUT→`ConstraintConfig`
   管道与守卫；验证 `ConstraintLoop` 的 reporting 生命周期与 p_ref 快照。
2. **回归层**：默认 `constraint_dipole_report = false` 时日志无 dipole block；报告开启/关闭
   两种运行的 SCF 密度残差轨迹逐位相同（纯观测）。
3. **集成层**：h2o_dimer LCAO（molecular + saw 两个 frame）；LiCl PW 对拍性质1 §5.2 的
   6.8665 D。
4. **资源约束**：按用户要求，所有 SCF 运行 `scf_nmax = 100`；100 步不收敛则记录 drho
   轨迹并分析原因，不继续增加步数。

## 2. 测试设置

### 2.1 构建与单元测试

- 构建目录：`build`（Debug，MPI）+ `build_rel`（Release，MPI）。Release 仅用于集成运行，
  两套构建均包含本轮 `constraint_dipole.cpp`。
- 单元测试目标：`MODULE_ESTATE_constraint_dipole`（新增）、
  `MODULE_ESTATE_constraint_io`、`MODULE_ESTATE_constraint_loop`。
- 命令：
  ```bash
  cmake --build build -j 32 --target MODULE_ESTATE_constraint_dipole
  ctest --test-dir build -R 'MODULE_ESTATE_constraint_(dipole|io|loop)' --output-on-failure
  ```

### 2.2 h2o_dimer（LCAO）

- 用例：`tests/constraint_ct_dimer/h2o_dimer/`（15 Å 立方盒，6 原子，LCAO，PP_ORB）。
- SCF 上限：`scf_nmax 100`（原用例 400）；`scf_thr 1e-7`；`OMP_NUM_THREADS=1`；
  Release 二进制，`mpirun -np 1`。
- 约束目标：v2 列表，acceptor `[0,2,3]` +0.05 e，donor `[1,4,5]` -0.05 e。
- 三个运行：
  1. 默认关闭（无 dipole 参数）；
  2. `constraint_dipole_report 1` + `constraint_dipole_frame molecular`；
  3. `constraint_dipole_report 1` + `constraint_dipole_frame saw`。

### 2.3 LiCl（PW）

- 结构：15 Å 立方盒，Li (7.5,7.5,7.5) Å、Cl (7.5,7.5,9.52) Å。
- 赝势：`/root/pporb/apns-pseudopotentials-v1/{Li,Cl}.upf`（APNS-v1，Z*=3/7）。
- 参数：`basis_type pw`、`gamma_only 1`、`ecutwfc 100`、`ecutrho 400`、
  `scf_thr 1e-8`、`scf_nmax 100`、`nelec 10`、`smearing gauss 0.002`。
- 约束：`{"constraints":[{"type":"charge","target":0.0,"atoms":[0]}]}`（δ=0 空操作）。
- 运行：Release，`mpirun -np 8`，`OMP_NUM_THREADS=1`。
- 说明：本轮在本机执行（Bohrium 未使用）；同一套 APNS 赝势、同一公式，对照性质1 §5.2。

## 3. 结果

### 3.1 单元测试（PASS）

`MODULE_ESTATE_constraint_dipole`：8/8 通过

- `MolecularNeutralOriginIndependent`：中性盒 p=(0,0,-6) e·Bohr，原点 (0,0,0) 与
  (3.3,-1.7,5.1) 给出同一结果。
- `MolecularChargedOriginShift`：带电玩具 p(0)=(-20,-20,-20) e·Bohr；
  原点平移 (1,2,3) 后 Δp=(2,4,6) e·Bohr = -Q_net Δo。
- `IonicCenter`：片段 {0}→O 坐标；{1,2}→H 中点。
- `ChargeTransferSign`：电子 -2e 由 a→b 给出 q_dip=+2 e；反向为 -2 e。
- `SawShape`：端点连续、周期、斜坡斜率 1。
- `SawDipoleAndPotentialDrop`：电荷转移玩具 p_z=-20 e·Bohr，ΔV=4πp/A<0。
- `LowestDensityPlane`：自动面远离 z_frac=0.5 的原子。
- `DecayWindowGuard`：窗内原子距离 0；远离窗 9.0 Bohr。

`MODULE_ESTATE_constraint_io`：全部通过，新增 `DipoleReportPlumbing`
（默认值、saw 参数透传、未知 frame / pos_dec / pos_max / 非 z 正交晶胞 / nspin=4 守卫）。

`MODULE_ESTATE_constraint_loop`：全部通过，新增
`DipoleReportLifecycle`（observe 得到 p_z=-6，REFERENCE→CONSTRAINED 快照 p_ref，
q_dip=0）、`DipoleReportDefaultOff`（reporter 惰性、q_dip invalid）、
`SawQdipProjectsLateralOffset`（dx=-4、dz=-5 Bohr 时 q_dip=dp_z/d_z，而非
3D 点积的 25/41 倍）与 `SawQdipRequiresZSeparation`（Δz=0 → q_dip invalid）。

全量 `ctest -R constraint`：12 个测试中 10 通过；失败的 2 个是
`MODULE_ESTATE_constraint_weight_grid_mpi{,_4np}`，原因是本机 OpenMPI
`ompi_rte_init failed: Unable to start a daemon`（环境问题，与本轮改动无关）。

### 3.2 h2o_dimer：默认关闭回归

`OUT.autotest/running_scf.log` 中 `dipole report` 出现次数 = **0**。
默认关闭运行的 `Electron density deviation` 序列与 molecular / saw 两次开启运行的序列
**逐位相同**（`diff` 三者输出 identical），证明报告是纯观测、不改变约束动力学。

### 3.3 h2o_dimer：molecular 报告块

```text
[constraint] dipole report (frame=molecular, origin=(14.1483, 14.2824, 14.0693) Bohr)
  p      = (0.775769, -0.276136, -0.745384) e*Bohr  (|p| = 2.82313 Debye)
  p_ref  = (0.926877, -0.18301, -1.47622) e*Bohr (reference phase)
  q_dip  = -0.135554 e  (dipole-derived; fragment vs rest, R = 5.4801 Bohr; ...)
```

`final status`：`RUNNING (SCF ended before the outer loop converged — raise scf_nmax)`，
这是 100 步上限造成的预期截断（见 §3.5）。p 的量级 O(1) Debye，符合水二聚体；
q_dip 处于第二次 μ 更新后的瞬态，不能当作收敛电荷转移读数。

### 3.4 h2o_dimer：saw 报告块

```text
[constraint] dipole saw frame resolved: pos_max = 0.777778 (auto), pos_dec = 0.1
[constraint] dipole report (frame=saw, pos_max=0.777778, pos_dec=0.1)
  p      = (0, 0, -19.5267) e*Bohr  (|p| = 49.6318 Debye)
  p_ref  = (0, 0, -20.5277) e*Bohr (reference phase)
  deltaV = -8.31015 V  (4*pi*p_saw/A; positive: potential higher at +z)
  q_dip  = -0.183269 e  (dipole-derived; fragment vs rest, R = 5.46192 Bohr; ...)
```

自动选面落在 z_frac=0.7778 的低密度区；`deltaV` 行存在。数值对应 100 步截断的瞬态密度。
`R` 是 saw frame 下两片段中心的 **z 投影距离**（评审修复后），不是 3D 距离。

### 3.5 h2o_dimer：100 步内的 drho 轨迹与收敛分析

Reference 相在 **iteration 30** 收敛：drho = 7.72e-8 < 1e-7。
第一次 μ 注入（±0.05 Ry）后：

| iteration | drho | 说明 |
|---|---|---|
| 30 | 7.72e-8 | reference SCF 收敛，outer step 1 |
| 31 | 1.25e-2 | μ 注入后密度跳变 |
| 36 | 1.25e-2 | 平台（混合器重建响应） |
| 66 | 5.77e-6 | 已进入收敛尾段 |
| 73 | 9.32e-8 | 第一次约束 SCF 收敛，outer step 2 |
| 74-82 | 1.84e-1 | 第二次 μ 注入（μ=-0.08646 Ry）后平台 |
| 83 | 3.91 | 混合器明显过渡态 |
| 96 | 1.33e-2 | 尚未回到收敛尾段 |
| 100 | 1.27e-2 | 达到 100 步上限，SCF 未收敛 |

**分析**：
- 参考相健康（30 步到 1e-7）；慢的是 μ 注入后的约束相，drho 先跳变、再缓慢下降。
- 100 步内只完成 2 次 outer step；旧 400 步归档结果（`results/d_p0.05.audit`）
  显示同一用例需 **318 步 / 7 次 outer step** 才 CONVERGED。
- 报告开关不改变轨迹（§3.2），所以该收敛成本属于既有约束动力学，不是 M1 引入的。
- 结论：100 步上限适合作为冒烟/资源护栏；真正的收敛加速属于 M2（热启动/定 μ）
  与 M3（信赖域/渐进 SCF/鲁棒性）的目标。

### 3.6 LiCl：与性质1 §5.2 对拍

运行 18 步 SCF 收敛（drho=1.87e-9 < 1e-8），`final status: CONVERGED`。

```text
[constraint] dipole report (frame=molecular, origin=(14.172951583, 14.172951583, 16.845025388) Bohr)
  p      = (6.8624036968e-07, 2.5742199281e-07, -2.7029348628) e*Bohr  (|p| = 6.8701738759 Debye)
  p_ref  = (6.8624036968e-07, 2.5742199281e-07, -2.7029348628) e*Bohr (reference phase)
  q_dip  = 0 e  (dipole-derived; fragment vs rest, R = 3.8172482931 Bohr; ...)
```

```text
CONSTRAINT_AUDIT nconstraint=1 e_con=0 max_residual=0 total_charge=3.095941178 nelec=10 maxdev=0
CONSTRAINT_AUDIT c[0] kind=charge q=3.095941178 t=3.095941178 mu=0 res=0
```

| 量 | 本轮 | 预期/性质1 | 偏差 |
|---|---|---|---|
| 分子偶极 \|p\| | 6.8702 D | 6.8665 ± 0.01 D | +0.0037 D (+0.054%) |
| q_dip(δ=0) | 0.000 e | 0.000 e | exact |
| Li Becke 电荷 | 3.095941178 e | 3.095937431 e | +3.7e-6 e |

### 3.7 评审判定后的修复（With fixes → 可合入）

独立评审确认物理公式、网格约定、MPI 归约、p_ref 快照点、fixed_mu 前布防、
默认关闭回归与 AGENTS.md 注释规范均正确，无 Critical 问题。本轮修复：

1. **Important — saw frame 的 q_dip 投影**：saw 偶极只有 z 分量有物理意义。
   `ConstraintLoop::init` 解析片段中心后在 saw frame 下把 `frag_a_/frag_b_` 的
   x,y 置零，并新增 `|Δz| < 1e-6 → frag_valid_=false` 守卫。h2o_dimer saw 运行的
   `R` 由 5.4801（3D）变为 5.46192 Bohr（z 投影），q_dip 由 -0.182055 变为
   -0.183269 e；重复运行 drho 轨迹与修复前逐位相同。
2. **Minor 1**：`min_atom_decay_distance` 头注释改为"窗口内为 0，否则为到最近边的
   周期距离"，不再声称返回负值。
3. **Minor 2**：auto 选面解析后补跑一次 1 Bohr 跳变面警告；显式 pos_max 仍由 init
   检查，避免重复告警。
4. **Minor 3**：设计文档 §3.1 回写实现口径——参考点取全胞离子电荷中心，
   `q_dip` 由 `p−p_ref` 消去原点依赖；saw frame 的 d 为 z 投影并带 Δz→0 守卫。
5. **Minor 4**：新增打印串中的 em-dash 改为 `--`。
6. **Minor 5**：新增两个 loop 级测试，直接覆盖投影与 Δz=0 守卫。

修复后 `MODULE_ESTATE_constraint_{dipole,io,loop}` 3/3 目标全部通过。

## 4. 分析

- **口径闭合**：LiCl 的 `p` 从本实现直接输出即 6.87 D，与性质1 的同一密度后处理值
  （6.8665 D）在 0.01 D 内一致；符号为负 z（Li→Cl 方向电子转移），与性质1 的
  `ΣZz − ∫ρz` 负值一致。
- **纯观测性质**：默认关闭与开启报告的三次 h2o_dimer 运行 drho 轨迹逐位相同，
  且默认运行无 dipole block ⇒ M1 满足"默认路径输出不变"。
- **q_dip 语义**：δ=0 时 p=p_ref ⇒ q_dip 恒为 0（LiCl 精确 0），说明 delta 语义与
  p_ref 快照点正确；非零 q_dip 只在约束相瞬态出现。
- **saw 通道**：自动选面与 ΔV 行均正常；h2o_dimer 为正交 z 盒，满足 configure 守卫。
- **已知限制**：h2o_dimer 在 100 步内只到 outer step 2，报告读到的是瞬态密度；
  这是刻意遵守 100 步资源上限的结果，不是报告实现错误。

## 5. 下一步

1. M2 计划：C1 热启动 + C4 定 μ 产品化，直接针对本轮观测到的"μ 注入后 SCF 重建慢"
   （h2o_dimer 318 步）。
2. 若要获得 h2o_dimer 的收敛偶极读数，可在 M2 落地后用同一 100 步上限重跑；当前
   不提高步数。
3. 全量 `cmake --build build` 仍有一个与轮次无关的链接失败：
   `MODULE_ESTATE_charge_mixing` 缺 `InfoNonlocal::{ctor,dtor}` mock。建议单独修复该
   测试目标（本轮未触碰）。
4. 提交前运行 `ctest -R constraint`（排除环境性 MPI 两例）并归档本报告。
