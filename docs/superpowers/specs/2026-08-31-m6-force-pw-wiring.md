# 2026-08-31: Task 2.5.3——PW 力接线（Forces::cal_force + 驻点守卫）

> 上游：`docs/superpowers/plans/2026-08-31-task25-m6-force-detail.md` 2.5.3。
> 接线点：`source/source_pw/module_pwdft/forces.h/.cpp`；力核仍是 2.5.2 的
> `constraint_force`（PW/LCAO 共用，本 Task 只接 PW 一侧）。
> 零新依赖/算法；总力累加仿 `forces_onsite` dspin 先例位置。

## 测试计划（失败测试先行）

1. 单测 `ComputeForceZeroWhenDisabled`：环未启用时 `compute_force` 为 no-op，
   缓冲原值不被触碰。
2. 单测 `ComputeForceConvergedMatchesKernel`：外环驱动到 CONVERGED 后，
   `compute_force` 输出与直调 `constraint_force`（同一密度、同一 μ、同一
   权重场）**逐位一致**——证明"观测量==注入算符"延续到力，且接线（密度指针、
   通道、共享权重场）正确。
3. 单测 `ComputeForceZeroWhenMuZero`：target==参考电荷 → μ*=0 → 力**精确零**
   （核的零乘子短路）。
4. 集成冒烟（PW 约束 H₂O，`relax_nmax 1` 以触发 cal_force）：
   - μ≠0 相：`test_force` 打印 `#CONSTRAINT  FORCE (Ry/Bohr)#` 且非零；
   - μ=0 参考相：同块**恒零**。

## 测试设置

- 单测：沿用 constraint_loop_test 的 H₂O fixture（40³ 网格，O 权重 σ=1.5），
  归一化线性响应 mock（rho(mu) = rho_ref − mu·w/S，Q(mu)=Q_ref−mu）。
- 集成：`tests/01_PW/211_PW_constraint_h2o/`（ecutwfc 20/ecutrho 80、15 Å 盒、
  target +0.1 e on O），`mpirun -np 2`（覆盖 reduce_pool 池归约路径）；
  INPUT 追加 `test_force 1` + `calculation relax` + `relax_nmax 1`。
  构建树二进制 `build/abacus_basic_para`（LCAO+MPI 配置，ccache 增量构建）。

## 结果

| 项 | 判据 | 实测 |
|---|---|---|
| 单测 10/10（含 3 个新测试） | PASS | 10/10 PASS |
| constraint_deriv 4/4 | PASS | PASS |
| constraint_weight_grid 10/10 | PASS | PASS |
| ctest -R "constraint\|partition" | 全绿 | 12/12 |
| 主库编译（forces.cpp / elecstate 含 constraint_deriv、constraint_loop） | 无警告无错误 | 通过 |
| 集成 μ≠0 相 | 力块非零 | O z=+0.0993、H1/H2 x=∓0.0963、z=+0.0722（Ry/Bohr） |
| 集成 μ=0 参考相 | 力块恒零 | 全零（精确 0.0000000000） |
| 集成外环 | 6 外步 CONVERGED | μ=−0.17655 Ry（README 记载 −0.1765，吻合） |
| 驻点守卫 | 收敛态不触发 WARNING | warning.log 无守卫记录（仅例行初始密度告警） |

## 分析

- **接线结构**：`Forces::cal_force` 内新增 `cal_force_constraint(forcecon, chr)`
  编排（在 `cal_force_cc` 前调用，位置对齐 dspin/onsite 先例）；总力累加
  `if (PARAM.inp.constraint) force += forcecon`；`test_force` 下以 Ry/Bohr
  打印独立分量块（不污染 eV/Å 各分量块）。`ConstraintLoop::compute_force`
  懒建导数网格（每几何一次）、收敛态判定后调 2.5.2 核。
- **驻点守卫（方案 §5.1 推论 1）**：`mu_norm()>0 && status_!=CONVERGED` 时
  `WARNING`"constraint force from an unconverged outer loop: residual
  O(|Q−t|)"——不阻断（力仍可输出供诊断），但明确披露包络定理前提不满足。
  本次冒烟两相均收敛，守卫未触发；未收敛路径由单测 `IgnoresUnconvergedScf`
  及守卫分支逻辑保证（守卫仅在非收敛态发警告，不改变数值）。
- **测试期间修复 1 处**：`ComputeForceConvergedMatchesKernel` 直调
  `constraint_force` 时 fixture 权重网格未建导数缓存，触发核的
  WARNING_QUIT 守卫（进程退出码 1）——在直调前补 `wg->build_derivatives()`
  （注释说明环路径在 compute_force 内懒建）。此为测试编排缺陷，非产品缺陷；
  反向证明核守卫真实生效。
- **集成单位与量级自检**：O z 分量 0.0993 Ry/Bohr ≈ 2.55 eV/Å，与
  μ=−0.1765 Ry × 权重导数网格积分同量级；H 原子受力非零符合物理（Becke
  权重 w_O 依赖全部原子位置，∂w_O/∂R_H ≠ 0）。Σ_J F_J 不恒零符合 2.5.2 认知
  （精确恒等式为 Σ_J F_J = −Σ_α μ_α dQ_α/dt，非"≡0"）。
- **MPI**：-np 2 实跑覆盖 `reduce_pool` 池归约，力块数值与串行单测口径一致
  （H1/H2 分量呈镜面对称，池归约无重复累加）。

## 下一步

- Task 2.5.4：LCAO 力接线——`FORCE.h/.cpp` 调**同一个** `constraint_force`
  核（ρ 走 pw_rhod），验证"双基组同一核"架构声明；FD 全部留 2.6。
