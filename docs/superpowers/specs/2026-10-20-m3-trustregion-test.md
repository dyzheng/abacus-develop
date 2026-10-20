# 2026-10-20 M3 信赖域 + 渐进 SCF + 鲁棒性 — 测试报告

计划：`docs/superpowers/plans/2026-10-20-m3-trustregion-progressive-robustness.md`
设计：`docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §5.2/§5.3/§7

## 1. 测试计划

1. **单元层**：全量 `constraint` 测试，覆盖 C2 信赖域/Broyden/reject、
   C3 渐进阈值状态机、E1 磁分支锚定、E2 `branch_tol` auto、E3 UNREACHABLE 文案。
2. **C3 集成**：h2o_dimer LCAO，0.001 e absolute，warm 两帧，`scf_nmax=100`。
3. **C2 集成**：同一 0.001 e 用例；再跑 M2 失败的 0.01 e BFGS 用例。
4. **C2+C3 组合**：0.01 e BFGS，记录每帧 SCF 迭代、drho 抽样、末 residual、墙钟。
5. **资源纪律**：所有集成运行固定 `scf_nmax=100`；不收敛只记录 drho 与 residual，
   不提高上限。

## 2. 测试设置

### 2.1 构建与单元命令

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
  cmake --build build -j 32 --target abacus_basic_para
ctest --test-dir build -j 16 -R constraint --output-on-failure
```

### 2.2 集成共同参数

- 用例：`tests/constraint_ct_dimer/h2o_dimer/`，15 Å 立方盒，6 原子，LCAO，
  `nelec=20`，`ecutwfc=60`，`ecutrho=240`，`scf_thr=1e-7`，`mixing broyden 0.7`，
  `constraint_thr=1e-4`，`constraint_mu_max=5.0`，`OMP_NUM_THREADS=1`，
  `mpirun -np 8`。
- 0.001 e 目标：`9.993852052 / 10.006147948`。
- 0.01 e 目标：`10.002852052 / 9.997147948`。
- 0.001 e 运行：`relax_method cg`、`relax_nmax 2`、`relax_scale_force 0.01`。
- 0.01 e 运行：`relax_method bfgs`、`relax_nmax 2`。
- 所有运行 `constraint_warm_start=1`，`scf_nmax=100`。

### 2.3 运行目录

| 运行 | 目录 | 开关 |
|---|---|---|
| C3 0.001 e | `/tmp/m3_c3_small.aoMlSE` | `constraint_progressive_scf=1` |
| C2 0.001 e（修复前） | `/tmp/m3_c2_small.Xij06G` | `constraint_trust_region=1` |
| C2 0.001 e（修复后） | `/tmp/m3_c2fix_small.RS2nnn` | 同上 |
| C2 0.01 e | `/tmp/m3_c2_01.7DUImj` | 同上 |
| C2+C3 0.01 e | `/tmp/m3_c2c3_01.tIKFkI` | trust + progressive |
| M2 基线 | `/tmp/m2f_warm_small.WYukoi`、`/tmp/m2_warm01.oIEtW2` | 两开关均关 |

## 3. 结果

### 3.1 单元测试：12/12 PASS

```text
MODULE_ESTATE_constraint_mu_solver ............. Passed
MODULE_ESTATE_constraint_io .................... Passed
MODULE_ESTATE_constraint_accounting ............ Passed
MODULE_ESTATE_constraint_dipole ................ Passed
MODULE_ESTATE_constraint_weight_grid_mpi ....... Passed
MODULE_ESTATE_constraint_weight_grid_mpi_4np ... Passed
MODULE_ESTATE_constraint_observe ............... Passed
MODULE_ESTATE_constraint_inject_pw ............. Passed
MODULE_ESTATE_constraint_inject_lcao ........... Passed
MODULE_ESTATE_constraint_weight_grid ........... Passed
MODULE_ESTATE_constraint_loop .................. Passed
MODULE_ESTATE_constraint_deriv ................. Passed

100% tests passed, 0 tests failed out of 12
Total Test time (real) = 47.73 sec
```

`MODULE_ESTATE_constraint_loop` 43 个用例、`MODULE_ESTATE_constraint_mu_solver`
16 个用例全部通过。新增覆盖：

- C2：`TrustRegionExpandsOnGoodStep`、`TrustRegionRejectsBadStep`、
  `BroydenCapturesCoupling`、`RejectStepRestoresMu`、
  `BadRhoUpdatesModelInsteadOfRepeatingTheStep`；
- C2 loop：`TrustRegionConfigReachesSolver`、`TrustRejectionRequestsMixReset`、
  `TrustAuditAppendsDiagnosticsOnlyWhenEnabled`；
- C3：`ProgressiveScfDefaultOff`、`ProgressiveScfRelaxesProbeAndTightensFinal`、
  `ProgressiveScfInertForInnerAndFixed`；
- E1：`MagAnchorRejectsBranchJump`、`MagAnchorInertWhenDisabled`；
- E2/E3：io 侧 `BranchTolAutoForSpin`；`FuseUnreachable` 追加 protocol-cap 文案断言。

### 3.2 C3 0.001 e

| 运行 | 帧 1 SCF | 帧 2 SCF | 墙钟/s | 最终状态 |
|---|---:|---:|---:|---|
| M2 warm 基线 | 88 | 46 | 176 | 两帧 CONVERGED |
| C3 | 92 | 46 | 165 | 两帧 CONVERGED |

日志触发两次 tightening：

```text
frame1: outer residual 9.31438e-4 < 1e-3 -> strict final SCF
frame2: outer residual 6.62068e-7 < 1e-3 -> strict final SCF
```

帧 1 多出 4 个 SCF 迭代；帧 2 不变。C3 在该用例上没有迭代收益，墙钟差异
（165 vs 176 s）落在运行波动范围内，不能归因于算法收益。

### 3.3 C2 0.001 e：修复前后

| 运行 | 帧 1 SCF | 帧 2 SCF | 墙钟/s | 帧 1 状态 |
|---|---:|---:|---:|---|
| M2 warm 基线 | 88 | 46 | 176 | CONVERGED |
| C2 修复前 | 100 | 94 | 214 | **RUNNING** |
| C2 修复后 | **75** | **46** | **154** | CONVERGED |

修复前日志显示 SCF 58/59/60/61 连续四次 outer step，`mu` 不变而
`trust_radius` 从 0.05 依次降到 0.025/0.0125/0.00625/0.003125，`trust_rej`
从 1 增到 4：坏 rho 拒绝后用旧 Jacobian 重走，步长未被半径截断时重复同一步。

修复后同一用例仅 1 次拒绝，半径序列 `0.05 -> 0.025 -> 0.1`，帧 1 迭代减少 13
（88 -> 75，-14.8%），帧 2 保持 46；总墙钟 154 s（基线 176 s，-12.5%）。
末态 `c[1] res = 1.373e-6 e`。

### 3.4 C2 0.01 e（100 步上限）

| 运行 | 帧 1 SCF | 帧 2 SCF | 墙钟/s | 帧 1 | 帧 2 | 帧 2 末 drho | 帧 2 末 residual |
|---|---:|---:|---:|---|---|---:|---:|
| M2 warm 基线 | 97 | 100 | 230 | CONVERGED | RUNNING | 2.73e-4（iter 100） | -1.1128e-2 e |
| C2 | 84 | 100 | 212 | CONVERGED | RUNNING | 2.4034e-4 | +3.265e-4 e |
| C2+C3 | **78** | 100 | **201** | CONVERGED | RUNNING | **3.7018e-5** | +3.263e-4 e |

C2 帧 2 外层在 iter 40、72、96 各更新一次 μ；最后一次更新后只剩 4 个 SCF
迭代。C2+C3 帧 2 外层在 iter 40、63、87，随后 tightening 并跑到 100。

帧 2 drho 抽样（C2 / C2+C3）：

| iter | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 99 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 | 8.286e-2 | 7.634e-4 | 3.507e-6 | 6.546e-8 | 8.193e-3 | 1.780e-5 | 3.339e-7 | 4.111e-4 | 4.588e-6 | -- | 2.403e-4 |
| C2+C3 | 8.287e-2 | 7.638e-4 | 3.508e-6 | 6.487e-8 | 8.241e-3 | 1.974e-5 | 3.204e-3 | 9.531e-6 | 2.402e-4 | 1.111e-4 | 3.702e-5 |

两条轨迹仍是“每次 μ 注入后 drho 重新起跳，再收敛”的两层循环签名。C2+C3 把
帧 2 末 drho 从 2.4e-4 压到 3.7e-5、帧 1 从 97 步降到 78 步，但都没有在
100 步内完成帧 2 的 target（末 residual 3.26e-4 e > 1e-4 e）。按要求未提高
`scf_nmax`。

### 3.5 E1/E2/E3

- E1：TMAG 从 1.0 跳到 2.0 mu_B、tol=0.5 时，`mag_rejections()==1`，半径减半、
  μ 从被拒 trial 回退并重走；恢复参考分支后正常 CONVERGED。默认关闭时同一跳变
  不产生任何拒绝。
- E2：spin 非 fixed + `branch_tol=-1` 解析为 1e-3 Ry；charge 与全 fixed 解析为 0；
  显式 0 保持关闭（`ConstraintIOTest.BranchTolAutoForSpin`）。
- E3：UNREACHABLE 最终行包含
  `protocol cap constraint_mu_max=<cap> Ry reached; increase constraint_mu_max to extend`。

## 4. 分析

- **C2 的根因修复**：首版在坏 rho 拒绝后沿用旧 J 重走；当原步长小于半径时，
  重走产生完全相同的 μ，连续 4 次拒绝后才把半径压到步长以下。修正为：
  拒绝的 trial 仍是已收敛的 Q(μ) 观测，先用它更新 Broyden/对角模型再重走。
  对该用例，首次步长 0.00333 Ry 后 J 从 -0.3 学到约 -0.58，重走步长降到
  0.00173 Ry，一次拒绝即完成。`BadRhoUpdatesModelInsteadOfRepeatingTheStep`
  锁定该行为。
- **符号守卫**：`update_trial_model` 只在所有被移动分量的 `dQ/dmu` 符合通道
  预期符号时更新 J；否则退回 `response_sign*kappa_min`，避免非单调通道把
  Newton 方向翻反（`TrustRegionRejectsBadStep` 仍保持正向重走）。
- **C3 收益有限**：`scf_thr=1e-7` 时探针阈值 `max(1e-5, 100*scf_thr)=1e-5`，
  与严格阈值只差两个数量级；该用例的 SCF 成本主要在 μ 注入后的 mixing 重建，
  收敛尾部只占约 4 步，因此 C3 单独使用反而多 4 步。组合中 C3 的收益主要来自
  “近目标时冻结 μ 再跑严格 SCF”，而不是放宽探针本身。
- **C2 对 0.01 e 的作用**：帧 1 迭代 97->84，帧 2 末 residual 从 1.11e-2 降到
  3.27e-4 e（约 34 倍改善），说明耦合 Jacobian 明显改善了外层方向；但帧 2 最后
  一次 μ 更新发生在 iter 96，剩余 4 步不足以完成 SCF。这是外层调度/预算问题，
  不是收敛到错误解；按用户要求停在 100 步并记录。
- **E1/E2/E3**：默认全部关闭或保持旧语义；单元测试直接覆盖跳变回退、auto
  解析和文案，未改变默认回归。

## 5. 下一步

1. 合入 C2（默认关闭）与 C2 修复；保留 C3 开关但记录“本用例无收益”的实测结论，
   后续只有在 SCF 收敛尾部占比较大的体系再评估。
2. 0.01 e 的 100 步失败应交给 M4 ALM / 外层调度处理，而不是提高 `scf_nmax`；
   需要让最后一次 μ 更新留足至少一个完整 SCF 重建预算。
3. 补充 M3 开发日志与提交；默认关闭路径的 M2 基线数值继续作为回归基准。
