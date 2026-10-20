# M3 信赖域 + 渐进 SCF + 鲁棒性 Implementation Plan

> **For agentic workers:** 按 Task 粒度实现、测试、提交。步骤用 checkbox 跟踪。

**Goal:** 在 M2 的热启动/定 μ 之上，降低约束外层循环的 SCF 重建成本并提高多解体系鲁棒性：C2 自适应信赖域（含多约束 Broyden 耦合）、C3 外层探测期渐进 SCF 收紧、E1 磁分支锚定回退、E2 spin 通道默认能量分支守卫、E3 UNREACHABLE 措辞纠正。

**Architecture:**
- `MuSolver` 新增独立 trust-region 路径：`ρ = 实际残差下降 / 预测残差下降`，好步半径×2（上限 1 Ry）、坏步半径×0.5 并回退 μ；多约束用 Broyden 秩一 Jacobian 求解耦合步。默认关闭时保留现有分量 secant 路径逐位不变。
- `ConstraintLoop` 把 cfg 开关映射到 solver，记录 trust 回退请求供 esolver 重置 mixing，并在审计行按需追加 trust 诊断。
- C3 由 `ConstraintLoop::scf_threshold(strict)` 向 `ESolver_KS::iter_finish` 提供本迭代的 drho 门槛：外层探测期用 `max(1e-5, 100×scf_thr)`；`max|res| < 10×constraint_thr` 后冻结 μ、切回 `scf_thr` 跑终步。
- E1 在每个几何的参考收敛点记录 `TMAG`，约束相超过 `constraint_mag_tol` 时调用 solver 的强制回退；E2 用 `constraint_branch_tol = -1` 作为 auto 哨兵（spin 默认 1e-3 Ry，charge 默认 0，显式 0 仍关闭）；E3 只改 UNREACHABLE 日志措辞。

**Tech Stack:** C++17、gtest、CMake（AddTest）、ABACUS module_constraint。

**Spec:** `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §5.2（C2）、§5.3（C3）、§7（E1-E3）、§5.5（验证）。
**前置:** M2 已合入（`b493fdcbc` 之后）。

**默认与兼容纪律：**
- 新行为默认关闭：`constraint_trust_region=false`、`constraint_progressive_scf=false`、`constraint_mag_anchor=false`。
- `constraint_branch_tol` 的 INPUT 默认改为 `-1`（auto）：任何 spin 约束且非全 fixed -> 1e-3 Ry；否则 0。显式 `0` 始终关闭。
- trust 关闭时 `MuSolver::step` 走原路径，旧测试与 M1/M2 数值逐位不变；审计行不追加新字段。
- C3 只对 `mu_schedule=outer`、非 fixed、非 warm probe、CONSTRAINED 相生效；REFERENCE/INNER/fixed 使用严格 `scf_thr`。

**资源约束：** 所有集成 SCF 运行 `scf_nmax=100`；不收敛只记录 drho 与 residual，不提高上限。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| 改 `source/source_io/module_parameter/input_parameter.h` | `constraint_trust_region`、`constraint_progressive_scf`、`constraint_mag_anchor`、`constraint_mag_tol`；`constraint_branch_tol` auto 哨兵 |
| 改 `source/source_io/module_parameter/read_input_item_other.cpp` | 新 INPUT 的声明、默认值、描述 |
| 改 `source/source_estate/module_constraint/constraint_io.h/.cpp` | `ConstraintConfig` 新字段；branch_tol auto 解析；`mag_anchor` 与 trust 的互斥守卫 |
| 改 `source/source_estate/module_constraint/mu_solver.h/.cpp` | trust/Broyden/reject API、Jacobian 与半径状态、诊断访问器 |
| 改 `source/source_estate/module_constraint/constraint_loop.h/.cpp` | solver 参数映射、trust 回退请求、C3 状态机、E1 TMAG 锚定、E3 措辞、审计扩展 |
| 改 `source/source_esolver/esolver_ks.cpp` | 用 `ConstraintLoop::scf_threshold()` 替换本次 `chgmixing_ks` 的 `scf_thr` |
| 改 `source/source_esolver/esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp` | trust 回退后 `mix_reset()` |
| 改 `source/source_estate/module_constraint/test/mu_solver_test.cpp` | 半径增长/拒绝回退/Broyden 耦合/reject API |
| 改 `source/source_estate/module_constraint/test/constraint_io_test.cpp` | 新参数透传、branch_tol auto、互斥守卫 |
| 改 `source/source_estate/module_constraint/test/constraint_loop_test.cpp` | C3 阈值状态机、E1 回退、trust 审计/请求 |
| 新建 `docs/superpowers/specs/2026-10-20-m3-trustregion-test.md` | 测试报告（五段） |
| 改 `docs/superpowers/specs/deltap-development-log.md` | 本轮日志 |

**关键既有事实：**
- `ESolver_KS::iter_finish` 在 `source/source_esolver/esolver_ks.cpp:239` 调用 `module_charge::chgmixing_ks`，`scf_thr` 从成员传入；PW/LCAO 随后在 `on_scf_converged` 中可覆盖 `conv_esolver`。
- `ConstraintLoop::on_scf_converged` 的顺序是：INNER settle -> 未收敛早退 -> 能量分支守卫 -> warm finish -> `outer_step`。
- `MuSolver::step` 的旧路径（分量 secant、cap、plateau fuse）必须原样保留；新逻辑走 `params_.trust_region` 分支。
- `MuSolver::rebind` 保留 `last_slope_`；trust 路径需同时把 Jacobian 重置为对角 retained slope。
- `ucell_->magnet.tot_mag` 在 `ESolver_KS::iter_finish` 的 `ucell.magnet.compute_mag(...)` 后已更新；loop 的 `ucell_` 指针在 `build_geometry` 设置。
- `constraint_branch_tol` 的 fixed_mu 互斥守卫在 `constraint_loop.cpp` 初始化段；auto 解析必须把全 fixed 列表解析为 0，避免误触发。

---

### Task 1: INPUT + Config 管道

**Files:** `input_parameter.h`、`read_input_item_other.cpp`、`constraint_io.h/.cpp`、`constraint_io_test.cpp`

- [ ] Step 1: 写失败测试
  - `ConstraintIOTest.TrustProgressiveMagPlumbing`：直接调用扩展 `configure_constraint` 后，在 `configure_from_inputs` 路径检查 4 个新字段默认值与显式值。
  - `ConstraintIOTest.BranchTolAutoForSpin`：`nspin=2` + spin v1 + `branch_tol=-1` -> `cfg.branch_tol=1e-3`；charge -> 0；显式 0 -> 0；spin 全 fixed v2 -> 0。
  - `ConstraintIOTest.MagAnchorRequiresTrust`：`mag_anchor=true` 且 `trust_region=false` -> ERROR，错误串含 `constraint_mag_anchor`。
  - `mag_tol <= 0` 且 anchor 开 -> ERROR。
- [ ] Step 2: 实现
  - `ConstraintConfig` 增加：`bool trust_region=false; bool progressive_scf=false; bool mag_anchor=false; double mag_tol=0.5;`
  - `Input_para` 增加对应 4 项；`constraint_branch_tol` 默认改 `-1.0`。
  - `configure_from_inputs` 透传 4 项；扩展 `configure_constraint` 接受 `branch_tol=-1`，在 specs 解析完成后解析 auto。
  - 守卫：`mag_anchor && !trust_region`、`mag_anchor && mag_tol<=0`、`progressive_scf` 不做硬性互斥（INNER/fixed 下 inert）。
- [ ] Step 3: 构建 + 测试
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target MODULE_ESTATE_constraint_io
  ctest --test-dir build -R MODULE_ESTATE_constraint_io --output-on-failure
  ```
- [ ] Step 4: Commit `feat(constraint): M3 trust/progressive/mag input plumbing`

### Task 2: MuSolver 信赖域 + Broyden + reject

**Files:** `mu_solver.h/.cpp`、`mu_solver_test.cpp`

- [ ] Step 1: 写失败测试
  - `TrustRegionExpandsOnGoodStep`：线性 mock，一步后 ρ≈1，半径增大；后续步长大于初始 `step_max`。
  - `TrustRegionRejectsBadStep`：`Q(mu)=2-0.5mu+100mu^2`，首步预测下降但实际残差上升；断言 `trust_reject_count()>0`、半径缩小、`mu` 回到旧值后重走更小步。
  - `BroydenCapturesCoupling`：`J=[[-1,0.5],[0.5,-1]]` 的二维线性 mock；断言 `jacobian(0,1)` 非零且收敛。
  - `RejectStepRestoresMu`：pending 后调用 `reject()`，μ 恢复且半径减半。
  - 全部旧测试在 `trust_region=false` 下保持通过。
- [ ] Step 2: 实现
  - `MuSolverParams` 增加：`bool trust_region=false; bool broyden=true; double trust_radius_max=1.0; double trust_eta_good=0.5; double trust_eta_bad=0.1; double trust_shrink=0.5; double trust_grow=2.0; double trust_radius_min=1e-8; int trust_max_rejects=20;`
  - `MuSolver` 增加状态：`trust_radius_`、`trust_pending_`、`mu_before_step_`、`Q_before_`、`res_before_`、`pred_res_`、`J_`、`trust_accept_/reject_count_`、`trust_fallback_`。
  - `step()` 分派：trust off -> 原 `step_secant`（代码原样搬移）；trust on -> `step_trust`。
  - `step_trust`：评估 pending 的 ρ；坏步回退 μ、缩半径、用旧残差与 J 重新出步；好步接受并 Broyden 更新 J；求解 `J dmu = -res`，按半径/μ cap 截断，记录 pending。
  - 多约束用 reduced active set（已收敛分量 dmu=0）；奇异矩阵回退对角 secant。
  - `reject(Q,target,mu)` 公开：强制回退 pending 步并重走（E1 使用）；无 pending 时用 `mu_prev_/Q_prev_` 作锚。
  - `rebind`：保留对角 slope，重置 J 为对角、清 pending、半径回 `step_max`。
  - 诊断：`trust_radius()`, `trust_accept_count()`, `trust_reject_count()`, `last_rho()`, `jacobian(i,j)`, `trust_fallback()`。
- [ ] Step 3: 构建 + 测试
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target MODULE_ESTATE_constraint_mu_solver
  ctest --test-dir build -R MODULE_ESTATE_constraint_mu_solver --output-on-failure
  ```
- [ ] Step 4: Commit `feat(constraint): M3 trust-region Broyden solver`

### Task 3: ConstraintLoop C2 集成

**Files:** `constraint_loop.h/.cpp`、`constraint_loop_test.cpp`、`esolver_ks_pw.cpp`、`esolver_ks_lcao.cpp`

- [ ] Step 1: 写失败测试
  - `TrustRegionConfigReachesSolver`：cfg trust on -> `loop.trust_radius()`、`loop.trust_enabled()` 初始值正确。
  - `TrustRejectionRequestsMixReset`：构造坏步（同 Task 2 的非线性 mock）驱动一次 outer step，断言 `consume_trust_rollback()==true` 且第二次调用为 false。
  - `TrustAuditAppendsDiagnosticsOnlyWhenEnabled`：默认审计行不含 `trust_`；开启后含 `trust_radius=` 与 `trust_rej=`。
- [ ] Step 2: 实现
  - cold init 的 `MuSolverParams` 映射 trust/broyden 开关。
  - `outer_step` / `take_inner_step`：调用前后比较 reject count，设置 `trust_rollback_pending_`。
  - 新增 `bool consume_trust_rollback()`；PW/LCAO 在 `on_scf_converged` 后若为 true 调 `p_chgmix->mix_reset()`。
  - 审计行在 trust 开启时追加 `trust_radius`/`trust_rej`；默认关闭保持旧格式。
- [ ] Step 3: 构建 + 测试
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop abacus_basic_para
  ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure
  ```
- [ ] Step 4: Commit `feat(constraint): M3 trust-region loop integration`

### Task 4: C3 渐进 SCF

**Files:** `constraint_loop.h/.cpp`、`esolver_ks.cpp`、`constraint_loop_test.cpp`

- [ ] Step 1: 写失败测试
  - `ProgressiveScfDefaultOff`：`scf_threshold(1e-7)==1e-7`。
  - `ProgressiveScfRelaxesProbeAndTightensFinal`：REFERENCE 严格；CONSTRAINED 返回 `1e-5`；残差 `5e-4`（<10×c_thr）时 `on_scf_converged` 不推进 μ、`conv=false`，阈值切回 `1e-7`；下一严格收敛步完成 CONVERGED。
  - `ProgressiveScfInertForInnerAndFixed`：INNER/fixed 始终返回严格阈值。
- [ ] Step 2: 实现
  - `ConstraintLoop` 增加 `tighten_pending_`、`tighten_notice_done_`。
  - `double scf_threshold(double strict_thr) const`：仅 `progressive_scf && enabled && phase==CONSTRAINED && schedule==outer && !fixed_mu_ && !warm_pending_ && !tighten_pending_` 时返回 `max(1e-5, 100*strict_thr)`，否则 strict。
  - `on_scf_converged` 在 warm finish 后、`outer_step` 前插入 tightening 分支：残差 < `10*cfg.thr` -> `tighten_pending_=true`、`conv_esolver=false`、打印一次；否则正常 outer step。
  - 每次 μ 实际变化时清 `tighten_pending_`（回到探测期）。
  - `ESolver_KS::iter_finish` 用 `scf_threshold(this->scf_thr)` 传入 `chgmixing_ks`；仅当 `PARAM.inp.constraint`。
- [ ] Step 3: 构建 + 测试
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop abacus_basic_para
  ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure
  ```
- [ ] Step 4: Commit `feat(constraint): M3 progressive SCF tightening`

### Task 5: E1/E2/E3 鲁棒性

**Files:** `constraint_loop.h/.cpp`、`constraint_io.cpp`、`constraint_loop_test.cpp`、`constraint_io_test.cpp`

- [ ] Step 1: 写失败测试
  - `MagAnchorRejectsBranchJump`：reference TMAG=1.0；约束相 TMAG=2.0 且 tol=0.5 -> `mag_rejections()==1`、μ 回退、`conv=false`；TMAG 恢复后正常收敛。
  - `MagAnchorInertWhenDisabled`：默认无行为变化。
  - `BranchTolAutoForSpin`：Task 1 覆盖。
  - E3：新增 `ConstraintLoop::final_status_line()`（仅测试/日志复用）或在 final_report 调用后检查字符串含 `protocol cap` 与 `constraint_mu_max`。
- [ ] Step 2: 实现
  - E1：`ConstraintConfig::mag_anchor/mag_tol`；loop 在 REFERENCE 收敛与暖几何首收敛记录 `mag_ref_` + `mag_ref_valid_`；CONSTRAINED 收敛时若 `|tot_mag-mag_ref_| > tol`，调用 `mu_solver_.reject(Q_, targets_, mu_)`，`conv_esolver=false`，置 rollback 请求，打印 `MAG_ANCHOR` 日志；连续回退计数暴露为 `mag_rejections()`。
  - E2：Task 1 的 branch_tol auto 解析；spin 非 fixed 默认 1e-3 Ry，charge/全 fixed 默认 0。
  - E3：UNREACHABLE 最终行改为 `UNREACHABLE (protocol cap constraint_mu_max=<cap> Ry reached; increase constraint_mu_max to extend)`。
- [ ] Step 3: 构建 + 测试
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop MODULE_ESTATE_constraint_io
  ctest --test-dir build -R 'MODULE_ESTATE_constraint_(loop|io)' --output-on-failure
  ```
- [ ] Step 4: Commit `feat(constraint): M3 magnetic anchor + spin branch guard default`

### Task 6: 集成验证 + 文档

**Files:** `docs/superpowers/specs/2026-10-20-m3-trustregion-test.md`、`deltap-development-log.md`

- [ ] Step 1: 全量单元回归
  ```bash
  env CCACHE_DIR=/tmp/ccache CCACHE_MAXSIZE=5G \
    cmake --build build -j 32 --target abacus_basic_para
  ctest --test-dir build -j 32 -R constraint --output-on-failure
  ```
- [ ] Step 2: h2o_dimer 集成（LCAO，np8，`scf_nmax 100`，2 帧）
  - 基线：复用 M2 的 0.001 e warm absolute（46 iter / 176 s）。
  - C3：`constraint_progressive_scf 1`，记录每帧 SCF 迭代与 drho。
  - C2：`constraint_trust_region 1`，先跑 0.001 e；再跑 M2 失败的 0.01 e BFGS 用例，检查是否在 100 步内收敛。
  - C2+C3 组合：同一 0.01 e 用例，记录 drho 抽样、末 residual、墙钟。
  - 所有失败用例只记录，不提高 `scf_nmax`。
- [ ] Step 3: 写五段测试报告：计划/设置/结果/分析/下一步；如实记录 C2/C3 是否达到预期。
- [ ] Step 4: 更新 `deltap-development-log.md`：提交列表、文件、bug/fix、验证数据、M4 下一步。
- [ ] Step 5: `git diff --check` + `git status --short`，提交
  ```bash
  git add docs/superpowers/specs/2026-10-20-m3-trustregion-test.md \
          docs/superpowers/specs/deltap-development-log.md
  git commit -m "docs(constraint): M3 trust-region + progressive SCF test round"
  ```

---

## 验收标准

1. **默认回归**：trust/progressive/mag 全关时，所有既有 `constraint` 测试通过，M2 的 absolute/delta/fixed 数值不变。
2. **C2**：坏步被拒绝且 μ 回退，半径按 0.5 缩小；好步半径按 2 增长且不超过 1 Ry；二维耦合 mock 的 Broyden 收敛。
3. **C3**：探测期阈值 `max(1e-5, 100*scf_thr)`，接近目标后切回 strict 并完成终步；INNER/fixed 不受影响。
4. **E1/E2/E3**：TMAG 跳变触发回退；spin 自动 branch_tol=1e-3 Ry（显式 0 可关）；UNREACHABLE 文案含 protocol cap 与 `constraint_mu_max`。
5. **集成**：h2o_dimer 100 步内记录 C2/C3 的实际收益与失败轨迹；不通过提高 SCF 上限规避。

