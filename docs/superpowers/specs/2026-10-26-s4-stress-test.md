# S4.2 统一分配约束应力：测试报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S4.2。

## 1. 测试计划

本阶段验证三件事：

1. Becke 权重对均匀应变的解析导数在普通点和跨周期镜像点都与中心差分一致。
2. 网格约束应力核在完整密度/多通道路径上满足应变中心差分、对称性和对
   multiplier 的线性性。
3. LCAO 统一分配路径的 `CONSTRAINT STRESS` 和 `TOTAL-STRESS` 能与总能应变
   差分对照；同时用无约束同构型基线区分“实现误差”和“参考曲线本身非线性”。

应力符号采用 ABACUS 现有 KS 口径：`sigma_ab = -(1/Omega) dE/d eps_ab`。

## 2. 测试设置

### 2.1 单元与回归测试

- 编译目标：`MODULE_BASE_GRID_test_partition`、
  `MODULE_ESTATE_constraint_weight_grid`、`MODULE_ESTATE_constraint_deriv`、
  `abacus_basic_para`。
- 网格回归：`ctest --test-dir build -R
  'MODULE_BASE_GRID_test_partition|constraint' --output-on-failure`。
- 新增/扩展测试：
  - `WeightGridTest.StrainDerivativeMatchesFiniteDifference`：普通网格点的
    `d w/d eps_ab` 对晶格应变中心差分。
  - `WeightGridTest.PeriodicImagePositionDerivative`：固定跨镜像点
    `ir=1659`，同时检查位置导数和应变导数。
  - `ConstraintDerivTest.StressMatchesStrainFiniteDifference`：网格观测量
    `Q=int rho w dV` 的应变中心差分对 `constraint_stress()`。

### 2.2 集成应力差分

- 体系：H2O，LCAO，Gamma-only，`nspin=2`，`scf_nmax=100`。
- 约束：单一 spin 约束，`constraint_alloc_switch=1`，固定
  `mu=-0.01221724241 Ry`；目标值为标签，不在 fixed-mu 运行中执行 ALM。
- 应力：`cal_stress=1`，`test_stress=1`。
- 应变：只施加 `eps_zz`，同时按同一因子缩放 `LATTICE_VECTORS` 的 z 分量和
  所有原子 z 坐标；固定分数坐标。
- 基准体积：`Omega0=22775.65615090413 Bohr^3`。
- 单位换算：`1 Ry/Bohr^3 = 147105.2279754489 kbar`。
- 中央差分：`sigma_zz^FD = -(E+ - E-)/(2 eps Omega0)`，能量用
  `!FINAL_ETOT_IS`。
- 相对误差定义：`|sigma_analytic - sigma_FD| / |sigma_FD|`。

约束运行目录：

```text
base          /tmp/s42_stress_probe.W7rDMf
eps=+1e-4     /tmp/s42_zz_0.0001_plus.cct1fgue
eps=-1e-4     /tmp/s42_zz_0.0001_minus.wl15ibv6
eps=+5e-4     /tmp/s42_zz_0.0005_plus.r3fg8p_5
eps=-5e-4     /tmp/s42_zz_0.0005_minus.rqq_dwep
eps=+1e-3     /tmp/s42_zz_0.001_plus.tsfr8axz
eps=-1e-3     /tmp/s42_zz_0.001_minus.qmhg6kbh
```

无约束基线目录：

```text
base          /tmp/s42_nocon_zz_W7rDMf.88om410y
eps=+1e-4     /tmp/s42_nocon_zz_0.0001_plus.DAPzqn
eps=-1e-4     /tmp/s42_nocon_zz_0.0001_minus.CTzbT3
eps=+5e-4     /tmp/s42_nocon_zz_0.0005_plus.mRLa0M
eps=-5e-4     /tmp/s42_nocon_zz_0.0005_minus.aS2AsK
eps=+1e-3     /tmp/s42_nocon_zz_001_plus.uvcvyf_t
eps=-1e-3     /tmp/s42_nocon_zz_001_minus.yoa68fn3
```

所有 10 个约束/无约束点均 `#SCF IS CONVERGED#`，没有提高
`scf_nmax=100`。

## 3. 结果

### 3.1 单元与回归测试

- `MODULE_ESTATE_constraint_weight_grid`：16/16 PASS；跨镜像位置/应变导数
  均通过。
- `MODULE_ESTATE_constraint_deriv`：`StressMatchesStrainFiniteDifference`
  PASS，含非对角对称应变 `strain_factor=2` 的检查。
- `MODULE_BASE_GRID_test_partition`：PASS。
- 完整 constraint 回归：19/19 PASS。
- `git diff --check`：PASS。

### 3.2 约束体系总能应力差分

基准解析 `TOTAL-STRESS(zz)=0.4756819938 kbar`，
`CONSTRAINT STRESS(zz)=0.0008874390 kbar`。

| eps | E+ (eV) | E- (eV) | FD total zz (kbar) | analytic total zz (kbar) | rel. |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | -466.4019580852 | -466.4017550842 | 0.481842060 | 0.4756819938 | 1.278% |
| 5e-4 | -466.4023009906 | -466.4013788523 | 0.437756494 | 0.4756819938 | 8.664% |
| 1e-3 | -466.4022713636 | -466.4007443496 | 0.362451215 | 0.4756819938 | 31.240% |

因此 `1e-4` 满足 `<5%` 闸门；`5e-4` 和 `1e-3` 不满足，但下节的无约束基线
显示这两个点的参考曲线同样不满足。

### 3.3 无约束基线对照

基准解析 `TOTAL-STRESS(zz)=0.4743997999 kbar`。

| eps | E+ (eV) | E- (eV) | FD total zz (kbar) | analytic total zz (kbar) | rel. |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | -466.4011267958 | -466.4009243904 | 0.480428393 | 0.4743997999 | 1.255% |
| 5e-4 | -466.4014711159 | -466.4005495675 | 0.437476458 | 0.4743997999 | 8.440% |
| 1e-3 | -466.4014434924 | -466.3999162955 | 0.362494640 | 0.4743997999 | 30.871% |

约束与无约束的相对偏差在三档应变下分别相差仅约 `0.02`、`0.22`、`0.37`
个百分点。

### 3.4 张量与固定 multiplier 审计

- `CONSTRAINT STRESS` 所有非对角分量在两三角上逐项相同，最大差
  `<=1e-10 kbar`。
- `TOTAL-STRESS` 的 `xz/zx`、`yz/zy` 在打印精度内相同。
- 固定 `mu=-0.01221724241 Ry` 全程不变。六次扰动运行的
  `q=Q/mu_B` 范围为 `0.009987111988` 到 `0.01000074895`，相对目标
  `0.01` 的变化约 `1e-6` 到 `1.3e-5`；约束残差 `|q-0.01|` 为
  `7.49e-7` 到 `1.29e-5`，均小于 fixed-mu 运行设定的
  `constraint_thr=1e-4` 量级。
- `CONSTRAINT STRESS(zz)` 随 `eps` 的变化为
  `0.0008873815` 到 `0.0008874967 kbar`，在固定目标/固定 multiplier 条件下
  连续且量级稳定。

## 4. 分析

1. **S4.2 的网格核已闭合。** 普通点和跨镜像点的解析应变导数均与晶格中心
   差分一致；`constraint_stress()` 对 `Q=int rho w dV` 的中心差分在
   `2e-8` 级通过。周期镜像修复中的 `pair_dir` 现在同时覆盖位置导数和应变
   导数，避免了旧实现从 `drR/eR` 重建原子对方向造成的错误。
2. **集成 `1e-4` 通过。** 约束体系总能应力相对误差 `1.278% < 5%`，且与
   无约束基线的 `1.255%` 几乎相同。
3. **`5e-4/1e-3` 的失败不是 S4.2 特有。** 无约束同构型在三档应变下给出
   `1.255%/8.440%/30.871%`，与约束体系 `1.278%/8.664%/31.240%` 几乎逐点
   重合。大应变误差来自当前 LCAO 总能/基组在有限应变下的高阶或非光滑参考
   曲线，不能作为新增应力项错误的判据。按执行方案闸门的严格要求，这两档
   仍记为“未通过 `<5%`”，不粉饰为通过。
4. **`CONSTRAINT STRESS` 不直接与 `TOTAL-STRESS` 差分同口径比较。**
   前者是约束算符显式项的投影贡献，后者还含约束引起的自洽密度变化；固定
   multiplier 的目标只是标签，不做 moving-target 重算。因此本阶段用网格级
   `Q` 差分验证核，用总能差分验证总应力，`CONSTRAINT STRESS` 只做连续性、
   对称性和固定-mu 审计。
5. **公式覆盖。** 已实现 `dW/d eps`、目标空间 `dS/d eps`（复用 S4.1
   伴随 `E`）、二者的交叉项（由反向模式伴随覆盖）和体积/Jacobian 项
   `-mu Q delta_ab`。当前 fixed-mu/fixed-target 路径没有 moving-target 项；
   非共线 `lambda_i.sigma` 留给 S4.3。dipole 权重不是 Becke 分区，显式拒绝
   其应力请求，避免静默给出错误张量。

## 5. 下一步

1. 提交 S4.2：周期镜像导数修复、权重应变核、约束应力核、LCAO 接线、测试与
   本文档。
2. 进入 S4.3：非共线三分量 `lambda_i.sigma`、能量/观测/力/应力通道，以及
   无 SOC 整体自旋旋转 `|Delta E|<1e-6 Ry` 检查。
3. 在后续应力验收中固定使用无约束基线共同报告；若要挑战 `5e-4/1e-3` 的
   `<5%`，应先定位并改进 LCAO 总能对有限应变的非线性/网格误差，而不是在
   约束应力核上做经验补偿。
