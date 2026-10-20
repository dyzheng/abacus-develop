# 2026-10-20 M2 热启动 + 定 μ 产品化 — 测试报告

计划：`docs/superpowers/plans/2026-10-20-m2-warmstart-fixedmu.md`
设计：`docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §5.1/§5.4

## 1. 测试计划

1. **单元层**：全量 `constraint` 测试，覆盖 INPUT / v2 `fixed_mu` schema、
   `MuSolver::rebind()/slope()`、冷/暖 init、delta `Q_ref` 反推、fixed μ 单次 SCF、
   warm INNER 状态清理。
2. **验收层（absolute 主路径）**：h2o_dimer LCAO，两个离子步，同一 absolute 目标，
   warm 与 cold 各跑一次。按计划核对：
   - 第 2 个离子步的 SCF 迭代数；
   - warm 迭代数是否 ≤ cold 的 1/3；
   - 第 2 步收敛后的 `q` 与 `mu` 差异是否 < 1%；
   - 若 100 步内未收敛，只记录 drho 轨迹并分析，不提高 `scf_nmax`。
3. **delta 冒烟**：同一用例切 `constraint_target_mode delta`，确认暖启动日志、
   `Q_ref` 反推与 target 随几何更新。
4. **fixed_mu 冒烟**：v2 target 文件写入 `fixed_mu`，确认每帧一次 SCF、
   `final status: FIXED_MU`、Q 与同 μ 的 cold 收敛值一致。
5. **资源约束**：所有 SCF 运行 `scf_nmax = 100`。0.01 e / 0.05 e 的失败用例只记录
   drho 与末态 residual，不延长 SCF。

## 2. 测试设置

### 2.1 构建与单元测试

- 构建目录：`build`（MPI）。
- 构建命令：
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target abacus_basic_para
  ```
- 测试命令：
  ```bash
  ctest --test-dir build -j 32 -R constraint --output-on-failure
  ```
- `abacus_basic_para` 构建成功。全量 `cmake --build build` 仍被与本轮无关的既有测试
  目标阻断（`MODULE_ESTATE_charge_{test,mixing,extra}`、`elecstate_energy` 链接失败；
  `elecstate_base/print` 缺 `InfoNonlocal` 类型），因此以 `abacus_basic_para` +
  `constraint` 测试作为本轮验收。

### 2.2 h2o_dimer 与自由相数值来源

- 用例：`tests/constraint_ct_dimer/h2o_dimer/`（15 Å 立方盒，6 原子，LCAO，
  `PP_ORB`，`nelec = 20`）。
- 共同参数：`ecutwfc 60`、`ecutrho 240`、`scf_thr 1e-7`、`scf_nmax 100`、
  `mixing broyden 0.7`、`constraint_thr 1e-4`、`constraint_mu_max 5.0`、
  `OMP_NUM_THREADS=1`、`mpirun -np 8`。
- **自由相参考**（`/tmp/m2_h2o_ref.puGog8/OUT.autotest/running_scf.log`，100 步内收敛）：
  ```text
  CONSTRAINT_AUDIT c[0] kind=charge q=9.992852052 t=9.992852052 mu=0 res=0
  CONSTRAINT_AUDIT c[1] kind=charge q=10.00714795 t=10.00714795 mu=0 res=0
  ```
  两片段合计 20 e。计划中的示例 target（10.04285205 / 9.957147948）来自另一参考口径；
  本轮按计划的“数值来源纪律”改用本机实测自由相值。
- **absolute 目标（0.001 e 转移）**：
  ```json
  {"constraints":[
    {"type":"charge","target":9.993852052,"atoms":[0,2,3]},
    {"type":"charge","target":10.006147948,"atoms":[1,4,5]}]}
  ```
- **relax 参数**：`relax_nmax 2`、`relax_method cg`、`relax_new 1`、
  `relax_scale_force 0.01`、`constraint_warm_start 1`（cold 运行改为 0）。

### 2.3 运行目录

| 运行 | 目录 | 说明 |
|---|---|---|
| warm absolute | `/tmp/m2f_warm_small.WYukoi` | 0.001 e，CG，2 帧 |
| cold absolute | `/tmp/m2f_cold_small.PfusS6` | 同目标，`warm_start 0` |
| delta 冒烟 | `/tmp/m2f_delta_small.eaoETY` | target ±0.001 e |
| fixed_mu | `/tmp/m2f_fixedmu.BVKVvv` | `fixed_mu = ∓0.002533935749 Ry` |
| 0.01 e warm | `/tmp/m2_warm01.oIEtW2` | BFGS，100 步失败诊断 |
| 0.01 e cold | `/tmp/m2_cold01.bDF5bo` | BFGS，100 步失败诊断 |
| 0.05 e 首帧 | `/tmp/m2_warm.dkylSe` | BFGS，100 步失败诊断 |

## 3. 结果

### 3.1 单元测试：12/12 PASS

```text
1/12  MODULE_ESTATE_constraint_mu_solver ............. Passed
2/12  MODULE_ESTATE_constraint_io .................... Passed
3/12  MODULE_ESTATE_constraint_accounting ............ Passed
4/12  MODULE_ESTATE_constraint_dipole ................ Passed
5/12  MODULE_ESTATE_constraint_weight_grid_mpi ....... Passed
6/12  MODULE_ESTATE_constraint_weight_grid_mpi_4np ... Passed
7/12  MODULE_ESTATE_constraint_observe ............... Passed
8/12  MODULE_ESTATE_constraint_inject_pw ............. Passed
9/12  MODULE_ESTATE_constraint_inject_lcao ........... Passed
10/12 MODULE_ESTATE_constraint_weight_grid ........... Passed
11/12 MODULE_ESTATE_constraint_loop .................. Passed
12/12 MODULE_ESTATE_constraint_deriv ................. Passed

100% tests passed, 0 tests failed out of 12
Total Test time (real) = 45.71 sec
```

`MODULE_ESTATE_constraint_loop` 共 35 个用例全部通过。新增的
`WarmStartClearsInnerSettleState` 覆盖了代码自审发现的暖启动串状态问题：
在没有清理块时，暖 init 后 `inner_steps()` 仍为上一几何的 2，且下一次 SCF 会因
陈旧 `settle_armed_` 记录一次伪 bounce；加入清理后两处断言均恢复为 0。

### 3.2 warm vs cold（absolute，0.001 e）

两个运行的第 1 帧都在 88 个 SCF 迭代内收敛。第 2 帧：

| 运行 | 第 2 帧 SCF 迭代 | c[0] q (e) | c[0] mu (Ry) | 末 residual (e) | 总墙钟 |
|---|---:|---:|---:|---:|---:|
| warm | **46** | 9.993850881 | -0.002530262458 | -1.1707e-6 | 176 s |
| cold | **77** | 9.993852985 | -0.002533935749 | 9.3294e-7 | 204 s |

- 迭代减少：`(77-46)/77 = 40.3%`。
- 墙钟减少：`(204-176)/204 = 13.7%`。
- `q` 差 `2.104e-6 e`（相对 9.99 e 约 `2.1e-7`）。
- `mu` 差 `3.673e-6 Ry`，相对 cold `0.002533935749 Ry` 为 **0.145%**。
- 判据核对：
  - **未达到** `warm ≤ cold/3`：`46 > 25.67`；
  - **满足** `q`、`mu` 差异 < 1%；
  - 第 2 帧均在 100 步内收敛，无需失败诊断。

### 3.3 delta 反推冒烟

日志（`/tmp/m2f_delta_small.eaoETY/OUT.autotest/running_relax.log`）：

```text
[constraint] WARM_START: geometry rebuilt, mu carried (|mu|=0.00345276 Ry), phase=constrained; delta Q_ref will be re-derived at the first converged SCF
[constraint] WARM_START delta: Q_ref re-derived from the new geometry (targets rebuilt)
  c[0]: mu_prev=-0.00172638 Ry, Q_ref=9.99239 e
  c[1]: mu_prev=0.00172638 Ry, Q_ref=10.0076 e
```

第 2 帧首个约束收敛点（SCF iteration 20）：

```text
CONSTRAINT_AUDIT c[0] kind=charge q=9.993386406 t=9.993386406 mu=-0.001726380281 res=0
CONSTRAINT_AUDIT c[1] kind=charge q=10.00661359 t=10.00661359 mu=0.00172638028 res=0
```

target 从首帧的 `9.993852052` 更新为 `9.993386406`，变化 `-4.66e-4 e`；carried μ
保持不变，residual 为 0。该 0 residual 是 delta 反推的构造结果：
`target = (Q_obs - kappa*mu_prev) + delta`，在当前 μ 下按同一 kappa 线性外推即回到
当前观测点。它验证了“target 随几何重建”的管道，不等价于证明绝对目标已满足；
absolute 路径才是本轮主验收。总墙钟 157 s。

### 3.4 fixed_mu v2 冒烟

`fixed_mu` 取自同一用例 cold 收敛 μ：`∓0.002533935749 Ry`。

| 帧 | SCF 迭代 | c[0] q (e) | c[0] residual (e) | final status |
|---|---:|---:|---:|---|
| 1 | 30 | 9.994320161 | 4.681e-4 | `FIXED_MU` |
| 2 | 19 | 9.993853451 | 1.399e-6 | `FIXED_MU` |

两帧日志均为：

```text
[constraint] final status: FIXED_MU (mu fixed by the v2 target file; targets are labels, not enforced)
```

第 2 帧 Q 与 cold 同 μ 收敛值 `9.993852985` 的差为 `4.66e-7 e`，远小于判据
`1e-2 e`。总墙钟 102 s（warm 176 s / cold 204 s）。

注意：fixed 分支的 audit 行仍打印 `outer step 1`，这是既有审计格式的标签复用；
实际没有外层 secant 迭代，且 `targets` 明确只作标签、不被执行。

### 3.5 100 步失败诊断（不提高 scf_nmax）

#### 0.01 e 转移（BFGS，第 2 帧）

warm（`/tmp/m2_warm01.oIEtW2`）drho：

| iter | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| drho | 8.29e-2 | 7.63e-4 | 3.51e-6 | 6.45e-8 | 9.78e-3 | 1.17e-4 | 1.03e-6 | 1.56e-2 | 2.08e-2 | 2.73e-4 |

末态 `c[0] res = -1.1128e-2 e`，`final status: RUNNING`。外层在 SCF 40、75 各更新一次 μ。

cold（`/tmp/m2_cold01.bDF5bo`）drho：

| iter | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| drho | 1.40e-3 | 2.90e-6 | 1.88e-1 | 3.65e-1 | 2.28e-2 | 6.44e-3 | 2.16e-5 | 1.40e-7 | 1.89e-2 | 2.14e-3 |

末态 `c[0] res = 9.6146e-2 e`，`final status: RUNNING`。外层在 SCF 27、81 各更新一次 μ。

#### 0.05 e 转移（BFGS，首帧）

`/tmp/m2_warm.dkylSe` 在 GE100 的 `drho = 1.5735e-2`，末态
`c[0] res = -2.1086e-2 e`，100 步内未收敛；外层只完成 2 次 μ 更新（SCF 30、74）。

**共性**：drho 不是单调下降，而是在每次 μ 注入后重新起跳，随后再次降到 1e-6~1e-8，
下一次外层更新又回到 1e-2 量级。这是“外层 secant + 每步重跑 SCF”两层循环的代价签名：
几何变化后约束相密度仍需重新收敛，热启动只省掉部分外层试探与参考相，不能消除每次
μ 更新后的 SCF 重建。

## 4. 分析

- **功能闭合**：暖启动只在 `md/relax/cell-relax`、`istep > 0`、上一帧 `CONVERGED`、
  约束列表不变时触发；absolute 与 delta 都正确 rebind，delta 的 `Q_ref/target` 随几何
  更新；fixed μ 每帧一次 SCF，优先级高于暖启动；默认 `constraint_warm_start = true`
  不改变单点 `scf` 路径。
- **收益与判据**：absolute 第 2 帧迭代减少 **40.3%**、墙钟减少 **13.7%**，说明热启动
  有真实收益；但计划要求的 `warm ≤ cold/3` **未达到**。原因是本用例几何变化后最优 μ
  从 `-0.0017264` 移到 `-0.0025303 Ry`，warm 仍需 2 个外层步（cold 3 个），
  且首个约束 SCF 本身就要约 20 步；参考相/外层步的节省不足以把总迭代压到 1/3。
  这是实测结论，不应通过提高 `scf_nmax` 或挑选有利几何来掩盖。
- **fixed μ 是更强的生产杠杆**：同一用例 2 帧共 49 个 SCF 迭代、102 s，明显少于
  warm 的 134 个迭代 / 176 s 和 cold 的 165 个迭代 / 204 s；Q 与同 μ 收敛值差
  `4.66e-7 e`。代价是 target 不再被执行，只作为标注。
- **100 步诊断**：0.01 e / 0.05 e 的失败均呈 drho 锯齿，且 warm/cold 都未在 100 步内
  收敛，确认瓶颈是几何变化后的约束相 SCF 重建，而不是热启动接线错误。按用户要求
  停在 100 步并记录轨迹。
- **代码自审修复**：暖启动复用实例时，INNER 的 `settle_armed_`、bounce 计数与步数预算
  属于上一几何。`warm_reinit()` 现显式清零这 4 项 + `inner_step_this_iter_`，
  防止新几何首个 SCF 被旧 settle 结论误判。回归测试在移除清理块时稳定失败
  （`inner_steps=2` 且 `settle_failures=1`），恢复后通过。
- **审计格式**：fixed 模式沿用 `outer step 1` 字符串；测试/日志解析应以
  `final status: FIXED_MU` 为准，不能把该标签当作外层迭代次数。

## 5. 下一步

1. **M2 关闭**：接受“功能正确、迭代减少 40.3%、1/3 判据未达”的实测结论；在主路径
   保持 absolute 暖启动，生产定 μ 场景优先使用 v2 `fixed_mu`。
2. **M3 立项**：按设计 §5.2/§5.3 实施信赖域/自适应 SCF，直接针对本轮 drho 锯齿与
   每次 μ 更新后的 SCF 重建成本；warm-start 状态持久化作为 M3 信赖域与 M4 ALM 的
   共同基础。
3. **后续验证**：若要重新评估 1/3 判据，应换用几何变化更小或响应更线性的载体，并在
   计划中预先写明判据；本轮数据保留为基线，不修改阈值。

