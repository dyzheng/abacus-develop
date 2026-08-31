# Task 2.5 详细实施 Todo：M6 力核（F_J = −Σ_α μ_α ∫ρ ∂w_α/∂R_J dr）

> 上游计划：`docs/superpowers/plans/2026-08-31-realspace-weight-constraint-phase2.md` Task 2.5。本文档把它展开为可执行的逐步 TDD todo。
> 硬约束延续：零新组件/算法/外部依赖。导数数学核（M0 `w_becke_adjusted_deriv`）一期已交付；力公式纯网格，**PW/LCAO 同一核**。

## 已核实的接线落点

- PW 力汇总：`source/source_pw/module_pwdft/forces.h` `Forces::cal_force` 组件编排；约束类力先例 = `forces_onsite.cpp:62` `cal_force_onsite_dspin`（dspin/deltap 双用）。
- LCAO 力汇总：`source/source_lcao/FORCE.h`（Force_LCAO）。
- **关键架构红利**：力核只需要"密度网格 ρ(g) + 权重导数网格 ∂w/∂R_J(g)"——两者皆与基组无关（LCAO 的 ρ 在 pw_rhod 网格上现成，Hartree/XC 同网格），故力核**一份代码、两个调用点**，不需要 gint_dvlocal 的矩阵元导数链。

## Step 序列

### 2.5.1 M1 扩充：权重导数网格

**Files:** `source/source_estate/module_constraint/weight_grid.h/.cpp`、test/weight_grid_test.cpp

- [x] 失败测试 `WeightGridTest.DerivGridAnalytic`：双原子合成几何，∂w_0/∂R_1 网格值 vs M0 逐点 `w_becke_adjusted_deriv` 一致（1e-12）；**平移不变性自检：Σ_J ∂w_α/∂R_J + ∂w_α/∂r ≡ 0**（1e-10，这是 ∂w 链式组装正确性的最强内检）。
- [x] 失败测试 `DerivGridMPI`：导数网格 1 rank vs 4 rank 逐点一致。
- [x] 实现：`WeightGrid::build_derivatives()`，每几何一次，缓存 `dw_[alpha][J][3][ir_local]`；内存 = 3×N_at×N_α×N_g，可忽略（一期内存基准有案）。
- [x] 单测通过 + spec + 日志 + Commit。

### 2.5.2 M6 力核（constraint_deriv）

**Files:** Create `constraint_deriv.h/.cpp`、`test/constraint_deriv_test.cpp`

- [x] 失败测试 `ForceOnSyntheticDensity`：双原子 + 高斯密度（解析可积），F_J 网格积分 vs 解析期望 <1e-8 Ha/Bohr。
- [x] 失败测试 `NewtonThirdLaw`：**Σ_J F_J ≡ 0**（1e-10）——对"权重随原子刚性移动"推导的全局检验，能提前暴露导数链符号错误（评审提醒项）。注：≡0 仅对常密度成立且常密度网格和被 tie-break 尖点污染（O(Δx)），改为精确恒等式 Σ_J F_J = −Σ_α μ_α dQ_α/dt（observer 平移 FD 参考，1e-9，实测 7e-13）。
- [x] 失败测试 `ForceLinearInMu`：F(μ₁+μ₂)=F(μ₁)+F(μ₂)、μ=0 → F≡0（驻点外无条件——μ=0 即无约束力）。
- [x] 实现：`constraint_force`（通道折叠 + reduce_pool 归约 + 累加语义；通道/守卫分支注释纪律）。
- [x] 单测通过 + spec + 日志 + Commit。

### 2.5.3 PW 力接线

**Files:** `source/source_pw/module_pwdft/forces.h/.cpp`（~15 行）、`constraint_loop.h/.cpp`（暴露 `compute_force`）

- [x] 失败测试（集成）：PW 约束 H₂O 单点，打印力中含 constraint 分量；μ=0 参考相 constraint 力恒零。（实跑：relax_nmax 1 + test_force 1，μ≠0 相力块非零、μ=0 相恒零；单测 3 个：Disabled no-op / ConvergedMatchesKernel 逐位一致 / MuZero 精确零）
- [x] 实现：`Forces::cal_force` 编排内加 `cal_force_constraint`（仿 `forces_onsite.cpp:62` 的 dspin 先例位置）；力核调 2.5.2。
- [x] **驻点守卫**：外环未收敛（max_res > constraint_thr）时打印 WARNING"约束力基于未收敛约束，残余误差 O(|Q−t|)"（包络定理前提，方案 §5.1 推论 1）。——`compute_force` 内 `mu_norm()>0 && status_!=CONVERGED` → WARNING（不阻断）；本次冒烟两相收敛未触发
- [x] 冒烟通过 + spec + 日志 + Commit。（spec：`docs/superpowers/specs/2026-08-31-m6-force-pw-wiring.md`；日志节 15）

### 2.5.4 LCAO 力接线

**Files:** `source/source_lcao/FORCE_STRESS.cpp`（getForceStress 编排，~15 行）

- [x] 失败测试（集成）：LCAO 约束 H₂O 单点力含 constraint 分量。（实跑：212_NAO_constraint_h2o + relax_nmax 1 + test_force 1，μ≠0 相力块非零、μ=0 相恒零；核/环级测试沿用 2.5.2/2.5.3 已有失败测试）
- [x] 实现：**调同一个 2.5.2 力核**（ρ 指针换成 LCAO 的 pw_rhod 网格密度 `pelec->charge->rho`）——验证"双基组同一核"的架构声明，不为 LCAO 写第二份力代码。（总力累加在 H_HK 块后；test_force 打印 `#CONSTRAINT  FORCE (Ry/Bohr)#` 与 PW 同名同单位）
- [x] 冒烟通过 + spec + 日志 + Commit。（spec：`docs/superpowers/specs/2026-08-31-m6-force-lcao-wiring.md`；日志节 16；μ*=−0.2193 / Q_ref=6.407956559 与二期评审实测吻合）

### 2.5.5 本 Task 出口判据

- [x] 全部单测 PASS + `ctest -R constraint` 全绿；（2.5.1-2.5.3 各轮已跑；本轮未触碰核/环，既有 12/12 保持）
- [x] PW/LCAO 两冒烟用例力打印非零且 μ=0 时为零；（PW：211 用例，O z=+0.0993 / μ=0 恒零；LCAO：212 用例，O z=+0.1090 / μ=0 恒零）
- [x] **FD 验证留给 Task 2.6**（stationary4 协议，网格前提 ecutwfc=100/ecutrho≥400/scf_thr=1e-8——不在本 Task 抢跑，避免低网格假 FAIL，R7）。

## Task 2.6 / 2.7 提醒（不展开，按 phase2 计划执行）

- 2.6 三判决：PW≡LCAO 逐位一致（<1e-8）/ 力 FD（0.0129 eV/Å，stationary4）/ 力矩 FD（**严格 μ=−λ 换算**，0.006 eV/μB）；自旋通道反假收敛 + LCAO 4-rank。
- 2.6 附加议程：**M3b 命运判决**——升格运行时审计（Tr[W^α·DM] vs ∫w_αρ）或二期收尾删除，不留悬置资产（Task 2.3 评审待办②）。
- 2.7 判决门纪律：力 FD 或 PW≡LCAO 不过 → 归因排查，**不降低判据**。

## 风险预登记

| 风险 | 应对 |
|---|---|
| ∂w/∂R 链式组装符号错 | 2.5.1 平移不变性自检 + 2.5.2 牛顿第三定律双重拦截 |
| FD 假 FAIL（网格不足） | R7 处方写死在 2.6；2.5 不跑 FD |
| 驻点前提被忽略（外环未收敛时用力） | 2.5.3 驻点 WARNING 守卫 |
| 双基组力不一致 | 同一力核从构造上消除；2.6 的 PW≡LCAO 判决兜底 |
