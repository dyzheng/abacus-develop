# 2026-10-30 更快的 mu 控制器：暴露 constraint_kappa_min，让 secant 用实测坡度跳跃

## 0. 根因
外层 mu 由 `MuSolver::step_secant` 更新：`kappa = clamp(|dQ/dmu|, kappa_min, kappa_max)`，
`dmu = -(Q-target)/kappa`，再被 `step_max` 截断。
- LiCl MBIS 的**真实**响应 `|dQ/dmu| ≈ 0.036`（很 stiff）。
- 默认 `kappa_min = 0.3` 把 0.036 **抬到 0.3** ⇒ `dmu = res/0.3 ≈ 0.17`，又被
  `step_max = 0.05` 截断 ⇒ 每外步仅 **0.05** ⇒ ~28 外步（≈650 SCF），100 步内不可能收敛。

## 1. 修改
- 新增输入 `constraint_kappa_min`（默认 **0.3**，行为不变）：secant 保留的最小 `|dQ/dmu|`。
- `ConstraintConfig.kappa_min`；`configure_constraint` 从 `PARAM.inp.constraint_kappa_min`
  取值；`constraint_loop` 传给 `MuSolverParams::kappa_min`。
- stiff 通道可把它调到真实坡度以下（如 0.01），配合 `constraint_step_max` 放大，
  secant 即按实测坡度做 Newton 跳跃。

## 2. 验证（LiCl, delta=−0.05, 目标 1.994532, `-np 2`）
`constraint_kappa_min 0.01`、`constraint_step_max 1.5`、`scf_nmax 200`：
```
step1 mu=1.5    q=2.004468 res= 9.94e-3
step2 mu=1.872  q=1.996542 res= 2.01e-3
step3 mu=1.966  q=1.994483 res=-4.94e-5
[constraint] final status: CONVERGED (targets reached within 0.0001 e)
```
- **3 个外步**收敛（此前 ~28 步且 100 步内不收敛）；Li 电荷 1.99448（|res|<1e-4）。
- 默认（`kappa_min=0.3`, `step_max=0.05`）**行为不变**（安全）；快速路径为**显式 opt-in**。
- 模块测试 12/12 不变。

## 3. 说明
- 未改默认值：大 `step_max` 在**无 trust region** 时对大步敏感，故保守默认；建议 stiff
  通道（MBIS/强关联电荷转移）显式设 `constraint_kappa_min≈0.01` 与
  `constraint_step_max≈1.5`（配合 `constraint_trust_region 1` 更稳）。
- 文件：`input_parameter.h`、`read_input_item_other.cpp`、`constraint_io.{h,cpp}`、
  `constraint_loop.cpp`、本日志。
