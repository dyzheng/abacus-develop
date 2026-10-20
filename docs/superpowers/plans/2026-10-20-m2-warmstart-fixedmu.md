# M2 μ 热启动 + 定 μ 产品化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 MD/relax 每个离子步重跑自由相 + 从零割线的浪费。首离子步保留完整 REFERENCE 相；后续离子步重建几何（WeightGrid / 偶极状态）但保留 μ、割线斜率与 CONSTRAINED phase，直接以 μ_prev 进入新几何。同时把 `ABA_CONSTRAINT_FIXED_MU` 实验开关产品化为 v2 约束条目 `"fixed_mu": <Ry>`，支持一次 SCF 一帧的定 μ 生产 MD。

**Architecture:**
- `ConstraintLoop::init()` 增加 `istep` 参数并拆成两条路径：**冷启动**（`reset()` + REFERENCE 相）与 **热重建**（只重建几何，保留求解器状态）。
- `MuSolver` 增加 `rebind(mu, Q)`（把历史锚到新几何的第一个收敛点）与 `slope(i)`（读取上一几何保留的 κ，供 delta 模式反推 Q_ref）。
- delta 模式在热启动几何的第一个收敛 SCF 上做 `Q_ref(new) = Q_obs(μ_prev) − κ·μ_prev`，再重建 `targets = Q_ref + delta`；absolute 模式 targets 不变。
- 定 μ 模式从 `ConstraintSpec::fixed_mu` 读取，优先级高于环境变量；外层循环不启动，一次 SCF 即一帧。
- 所有新行为由 `constraint_warm_start`（默认 true，仅 MD/relax）与 `fixed_mu` 门控；单点 `calculation=scf` 与默认路径逐字节不变。

**Tech Stack:** C++17、gtest、CMake（AddTest）、ABACUS module_constraint。

**Spec:** `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §5.1（C1）与 §5.4（C4）。
**前置:** M1 已合入（commit `a37c740a7` 之后）。

**计划序列：** 本计划只覆盖 M2。M3（C2 信赖域 + C3 渐进 SCF + E 鲁棒性）、M4（D ALM）、M5（B 偶极约束坐标）各自独立成计划。

**代码规范（AGENTS.md，必须遵守）：**
- 每个 if-else 分支与非平凡 if 块前必须有注释（说明分支做什么、为什么）
- 禁 goto / setjmp / longjmp；函数 >300 行拆分；优先 RAII / std::vector

**资源约束（本轮用户令）：** 所有集成 SCF 运行 `scf_nmax <= 100`；100 步不收敛则记录 drho 轨迹并分析，不提高步数。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| 改 `source/source_io/module_parameter/input_parameter.h` | `constraint_warm_start` |
| 改 `source/source_io/module_parameter/read_input_item_other.cpp` | `constraint_warm_start` Input_Item |
| 改 `source/source_estate/module_constraint/constraint_io.h` | `ConstraintConfig::warm_start`；`ConstraintSpec::{fixed_mu, fixed_mu_value}` |
| 改 `source/source_estate/module_constraint/constraint_io.cpp` | INPUT 管道；v2 `"fixed_mu"` 解析与 all-or-none 守卫；`|fixed_mu| <= mu_max` 守卫 |
| 改 `source/source_estate/module_constraint/mu_solver.h/.cpp` | `rebind()` / `slope()` / 保留 κ |
| 改 `source/source_estate/module_constraint/constraint_loop.h/.cpp` | `istep` 参数、冷/热双路径、warm finalize、定 μ 优先、审计字符串 |
| 改 `source/source_esolver/esolver_ks_pw.cpp` | `init(..., istep)` |
| 改 `source/source_esolver/esolver_ks_lcao.cpp` | `init(..., istep)` |
| 改 `source/source_estate/module_constraint/test/constraint_io_test.cpp` | 参数/JSON 测试 |
| 改 `source/source_estate/module_constraint/test/mu_solver_test.cpp` | rebind/slope 测试 |
| 改 `source/source_estate/module_constraint/test/constraint_loop_test.cpp` | 冷/热、delta 反推、定 μ 测试 |
| 新建 `docs/superpowers/specs/2026-10-20-m2-warmstart-fixedmu-test.md` | 集成测试报告 |
| 改 `docs/superpowers/specs/deltap-development-log.md` | 本轮日志 |

**关键既有事实（实现时直接引用）：**
- `ConstraintLoop::init` 当前两个重载（`constraint_loop.h:71-99`）；生产入口是 specs-bearing 版本。`reset()` 清空全部状态。
- `reset()` 中 `cfg_ = ConstraintConfig(); wg_.reset(); mu_solver_.reset(); mu_.clear(); Q_.clear(); targets_.clear(); phase_=IDLE`。
- 生产调用点：`esolver_ks_pw.cpp:210`、`esolver_ks_lcao.cpp:273`，二者都在每个离子步的 `before_scf(ucell, istep)` 内。
- `istep` 语义：relax driver 传 `istep-1`（首个离子步为 0，`relax_driver.cpp:46`）；MD 传当前 md step（首个为 0，`md_func.cpp:259`）。
- `calculation` 取值见 `driver_run.cpp:77-90`：`md` / `scf` / `relax` / `cell-relax` / `nscf`。
- `PARAM.inp.calculation` 是 `std::string`；`PARAM.inp` 是 `PARAM.input` 的 const 引用。
- 固定 μ 环境变量分支：`constraint_loop.cpp:240-275`（`ABA_CONSTRAINT_FIXED_MU`），目前是标量、跳过 REFERENCE、在 `outer_step` 首次收敛时 DONE。
- `MuSolver::step` 的历史是 `mu_prev_/Q_prev_/has_history_`；无历史时 κ 回退 `response_sign*kappa_min`。
- delta 目标构建：`ConstraintLoop::outer_step` 的 `targets_[a] = Q_ref_[a] + specs_[a].target`（`constraint_loop.cpp:515-540`）。
- 偶极状态 armed 在 `init` 的 WeightGrid 构建之后（`constraint_loop.cpp:170-250`），热重建必须复用同一段逻辑。
- h2o_dimer 100 步实测：reference 在 iter 30 收敛；第一约束相在 iter 73 收敛；第二 μ 注入后 iter 100 仍 drho=1.27e-2（M2 的直接动机）。

---

### Task 1: INPUT 参数 + v2 `fixed_mu` 解析 + Config 管道

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_other.cpp`
- Modify: `source/source_estate/module_constraint/constraint_io.h`
- Modify: `source/source_estate/module_constraint/constraint_io.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_io_test.cpp`

- [ ] **Step 1: 写失败测试**

在 `constraint_io_test.cpp` 末尾追加：

```cpp
TEST(ConstraintIOTest, WarmStartPlumbing)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_warm_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Default: warm start is on (MD/relax only; the loop enforces the scope).
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(cfg.warm_start);
    PARAM.input.constraint_warm_start = false;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.warm_start);
    std::remove(path.c_str());
}

TEST(ConstraintIOTest, FixedMuParsingAndGuards)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "absolute";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_fixed_mu_test.json";
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // All constraints fixed -> values flow into the specs.
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.12},
            {"type":"charge","target":5.0,"atoms":[1,2],"fixed_mu":0.08}]})";
    }
    PARAM.input.constraint_target_file = path;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_TRUE(specs[0].fixed_mu);
    EXPECT_TRUE(specs[1].fixed_mu);
    EXPECT_DOUBLE_EQ(specs[0].fixed_mu_value, -0.12);
    EXPECT_DOUBLE_EQ(specs[1].fixed_mu_value, 0.08);

    // Mixed fixed/free -> ERROR (phase-1 C4 keeps one dynamics mode per run).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.12},
            {"type":"charge","target":5.0,"atoms":[1,2]}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("fixed_mu"), std::string::npos);

    // Fixed value outside the component cap -> ERROR (loud, no silent clamp).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-7.5}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("fixed_mu"), std::string::npos);

    // v1 files can not express fixed_mu: the legacy path stays untouched.
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_FALSE(specs[0].fixed_mu);
    std::remove(path.c_str());
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -5
```
预期：编译错误（无 `warm_start` / `constraint_warm_start` / `fixed_mu`）。

- [ ] **Step 3: 实现参数、解析与管道**

`input_parameter.h`（`constraint_inner_nmax` 之后）：

```cpp
    // M2 (C1): keep mu / secant history / CONSTRAINED phase across ionic
    // steps of MD/relax.  Single-point scf is unaffected (see loop init).
    bool constraint_warm_start = true;
```

`read_input_item_other.cpp`（`constraint_inner_nmax` item 之后）：

```cpp
    {
        Input_Item item("constraint_warm_start");
        item.annotation = "warm start the constraint multiplier across ionic steps";
        item.category = "Constraint";
        item.type = "Boolean";
        item.description = "MD/relax only. true (default): the first ionic step runs the full free reference SCF, later steps rebuild the geometry but keep mu, the secant slope and the CONSTRAINED phase (one outer update per step instead of a fresh reference). false: every ionic step restarts from the free reference. Single-point calculation=scf is unaffected.";
        item.default_value = "True";
        item.unit = "";
        item.availability = "constraint is true";
        read_sync_bool(input.constraint_warm_start);
        this->add_item(item);
    }
```

`constraint_io.h`：

```cpp
    bool warm_start = true;  // M2 C1: reused across MD/relax ionic steps
```

`ConstraintSpec` 追加：

```cpp
    // M2 C4 (fixed-mu productization): "fixed_mu" in a v2 file pins this
    // component; the outer loop does not run.  The field is all-or-none
    // across the list (configure guard), so mixed dynamics can not happen.
    bool fixed_mu = false;
    double fixed_mu_value = 0.0; // Ry
```

`constraint_io.cpp`：

1. `parse_constraint_object` 在 `mu_max` 分支之后追加：

```cpp
    // Branch F: optional "fixed_mu" (M2 C4).  The value is a Lagrange
    // multiplier in Ry, not a target; |value| is validated against the
    // component cap once the cap is known.
    pos = 0;
    if (find_key(obj, "fixed_mu", pos))
    {
        skip_ws(obj, pos);
        if (pos >= obj.size() || obj[pos] != ':')
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: expected ':' after \"fixed_mu\"";
            return false;
        }
        ++pos;
        skip_ws(obj, pos);
        if (!parse_number(obj, pos, spec.fixed_mu_value))
        {
            error = "constraint c[" + std::to_string(index)
                    + "]: malformed number in \"fixed_mu\"";
            return false;
        }
        spec.fixed_mu = true;
    }
```

2. `parse_v2_list` 在逐条解析循环之后追加：

```cpp
    // Guard (M2 C4): fixed mu is a run-level dynamics mode.  A mixed list
    // would need a two-mode outer loop; refuse instead of silently freezing
    // only part of the run.
    int n_fixed = 0;
    for (const ConstraintSpec& s : specs)
    {
        if (s.fixed_mu)
        {
            ++n_fixed;
        }
    }
    if (n_fixed > 0 && n_fixed != static_cast<int>(specs.size()))
    {
        error = "constraint file: \"fixed_mu\" must be set for all "
                "constraints or none (mixed fixed/free runs are not "
                "implemented)";
        return false;
    }
    // Guard: a fixed value beyond the component cap is a user error, not a
    // value to clamp silently (the audit would otherwise report a mu that
    // was never applied).
    for (const ConstraintSpec& s : specs)
    {
        if (s.fixed_mu && std::abs(s.fixed_mu_value) > s.mu_max)
        {
            error = "constraint file: |fixed_mu| = "
                    + std::to_string(std::abs(s.fixed_mu_value))
                    + " exceeds the component mu_max = "
                    + std::to_string(s.mu_max);
            return false;
        }
    }
```

3. `configure_from_inputs` 在 M1 dipole 管道旁追加：

```cpp
    cfg.warm_start = PARAM.inp.constraint_warm_start;
```

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_io --output-on-failure 2>&1 | tail -5
```
预期：全部通过。

- [ ] **Step 5: Commit**

```bash
git add source/source_io/module_parameter/input_parameter.h source/source_io/module_parameter/read_input_item_other.cpp source/source_estate/module_constraint/constraint_io.h source/source_estate/module_constraint/constraint_io.cpp source/source_estate/module_constraint/test/constraint_io_test.cpp
git commit -m "feat(constraint): M2 warm-start input + fixed_mu v2 schema"
```

---

### Task 2: MuSolver `rebind()` / `slope()`（保留 κ，清除跨几何陈旧历史）

**Files:**
- Modify: `source/source_estate/module_constraint/mu_solver.h`
- Modify: `source/source_estate/module_constraint/mu_solver.cpp`
- Test: `source/source_estate/module_constraint/test/mu_solver_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
TEST(MuSolverTest, RebindRetainsSlope)
{
    constraint::MuSolverParams p;
    p.step_max = 0.05;
    p.kappa_min = 0.3;
    p.response_sign = -1;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0};
    // First observation at mu=0 -> conservative first step.
    EXPECT_EQ(s.step({1.0}, {0.0}, mu), constraint::MuStatus::RUNNING);
    const double mu1 = mu[0];
    // Second observation gives a measured slope dQ/dmu = -2
    // (mu fell by 0.05 and Q rose by 0.1: the physical charge response).
    EXPECT_EQ(s.step({1.1}, {0.0}, mu), constraint::MuStatus::RUNNING);
    const double slope = s.slope(0);
    EXPECT_NEAR(slope, -2.0, 1e-12);

    // New geometry: anchor at the carried mu with a fresh Q.  The retained
    // slope must survive rebind, while the stale cross-geometry Q history
    // must not.
    s.rebind({mu1}, {0.95});
    EXPECT_NEAR(s.slope(0), slope, 1e-12);
    const double mu_before = mu[0];
    s.step({0.95}, {0.0}, mu); // dmu_obs == 0 -> retained slope path
    EXPECT_NE(mu[0], mu_before);
}

TEST(MuSolverTest, RebindResetsFuseWindow)
{
    constraint::MuSolverParams p;
    p.step_max = 0.05;
    p.mu_max = 0.05;
    p.plateau_window = 2;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0};
    // Drive the component onto the cap with a flat residual.
    s.step({1.0}, {0.0}, mu);
    s.step({1.0}, {0.0}, mu);
    // A stale plateau must not fuse the new geometry immediately.
    s.rebind(mu, {1.0});
    const constraint::MuStatus st = s.step({1.0}, {0.0}, mu);
    EXPECT_EQ(st, constraint::MuStatus::RUNNING);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_mu_solver 2>&1 | tail -5
```
预期：无 `rebind/slope`。

- [ ] **Step 3: 实现**

`mu_solver.h` public:

```cpp
    // M2 warm start: anchor the secant history at a new geometry's first
    // converged observation (mu_at_obs, Q_at_obs).  The retained slope from
    // the previous geometry survives; mu_prev/Q_prev are replaced so the
    // geometry change can not be mistaken for a mu response.  The residual
    // window / fuse state are cleared (a new geometry is a new problem).
    void rebind(const std::vector<double>& mu,
                const std::vector<double>& Q);
    // Last accepted raw dQ/dmu for component i (negative for charge/spin),
    // or the conservative response_sign*kappa_min when none was measured.
    double slope(const int i) const;
```

private:

```cpp
    std::vector<double> last_slope_; // retained across rebind()
```

`mu_solver.cpp`:

```cpp
void MuSolver::rebind(const std::vector<double>& mu,
                      const std::vector<double>& Q)
{
    const int n = static_cast<int>(Q.size());
    has_history_.assign(n, true);
    mu_prev_ = mu;
    Q_prev_ = Q;
    residual_window_.assign(n, std::vector<double>());
    fuse_component_ = -1;
    fuse_mu_ = 0.0;
    fuse_residual_ = 0.0;
    // last_slope_ intentionally NOT cleared: it is the κ carried across
    // ionic steps (C1).
    if (static_cast<int>(last_slope_.size()) != n)
    {
        const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
        last_slope_.assign(n, resp * params_.kappa_min);
    }
}

double MuSolver::slope(const int i) const
{
    if (i >= 0 && i < static_cast<int>(last_slope_.size()))
    {
        return last_slope_[i];
    }
    const double resp = (params_.response_sign > 0) ? 1.0 : -1.0;
    return resp * params_.kappa_min;
}
```

在 `step()` 中：

1. `has_history_` 首次初始化时同时 `last_slope_.assign(n, resp*kappa_min)`。
2. 计算 `kappa` 后记录 `last_slope_[i] = kappa;`（仅当 `dmu_obs != 0` 的真实测量；sign-flip 回退也记录回退值）。
3. 无历史或 `dmu_obs == 0` 时优先使用 `last_slope_[i]`：

```cpp
        double kappa = slope(i);
        if (has_history_[i])
        {
            const double dmu_obs = mu_at_obs - mu_prev_[i];
            const double dQ_obs = Q[i] - Q_prev_[i];
            if (dmu_obs != 0.0)
            {
                const double raw = dQ_obs / dmu_obs;
                // sign-flip / clamp branches unchanged ...
                last_slope_[i] = kappa;
            }
        }
```

在 `reset()` 中清空 `last_slope_`。

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_mu_solver 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_mu_solver --output-on-failure 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(constraint): MuSolver rebind/slope for cross-geometry warm start (M2)"
```

---

### Task 3: ConstraintLoop 冷/热双路径 + esolver 传 `istep`

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h`
- Modify: `source/source_estate/module_constraint/constraint_loop.cpp`
- Modify: `source/source_esolver/esolver_ks_pw.cpp`
- Modify: `source/source_esolver/esolver_ks_lcao.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
// Warm start: the second geometry rebuilds the weights but keeps mu, phase
// and the solver history; a cold re-init resets everything.
TEST_F(ConstraintLoopTest, WarmStartKeepsMuAndPhase)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.warm_start = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, /*istep=*/0);
    const double* rp[1] = {rho_ref.data()};
    loop.observe(1, rp, 1);
    bool conv = true;
    loop.on_scf_converged(2, conv);
    ASSERT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    const double mu1 = loop.mu()[0];
    ASSERT_NE(mu1, 0.0);

    // New geometry: move an atom so the rebuilt WeightGrid differs.
    ucell->atoms[1].taud[0].x += 0.05;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, /*istep=*/1);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_DOUBLE_EQ(loop.mu()[0], mu1);
    EXPECT_TRUE(loop.warm_started_last_init());

    // Cold path (switch off) must still return to the reference phase.
    cfg.warm_start = false;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, /*istep=*/2);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::REFERENCE);
    EXPECT_DOUBLE_EQ(loop.mu()[0], 0.0);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -5
```

- [ ] **Step 3: 实现**

`constraint_loop.h`：

```cpp
    void init(const UnitCell& ucell,
              const ModulePW::PW_Basis* rho_basis,
              const ConstraintConfig& cfg,
              const std::vector<double>& radii,
              const double nelec,
              const int istep = 0);
    void init(const UnitCell& ucell,
              const ModulePW::PW_Basis* rho_basis,
              const ConstraintConfig& cfg,
              const std::vector<ConstraintSpec>& specs,
              const std::vector<double>& radii,
              const double nelec,
              const int istep = 0);
```

public 诊断：

```cpp
    bool warm_started_last_init() const { return warm_started_last_init_; }
```

private：

```cpp
    bool warm_started_last_init_ = false;
    bool warm_pending_ = false; // delta Q_ref re-derivation at next convergence
    bool geometry_state_valid_ = false;
    void build_geometry(const UnitCell& ucell,
                        const ModulePW::PW_Basis* rho_basis,
                        const std::vector<double>& radii,
                        const std::vector<ConstraintSpec>& specs);
    bool warm_start_eligible(const ConstraintConfig& cfg,
                             const std::vector<ConstraintSpec>& specs,
                             const int istep) const;
    void warm_reinit(const UnitCell& ucell,
                     const ModulePW::PW_Basis* rho_basis,
                     const ConstraintConfig& cfg,
                     const std::vector<ConstraintSpec>& specs,
                     const std::vector<double>& radii,
                     const double nelec);
```

`constraint_loop.cpp` 顶层：

```cpp
namespace
{
// Warm start is a multi-geometry feature: single-point scf keeps the
// historical reference-first semantics.
bool warm_start_calculation()
{
    return PARAM.inp.calculation == "md" || PARAM.inp.calculation == "relax"
           || PARAM.inp.calculation == "cell-relax";
}
} // anonymous namespace
```

`ConstraintLoop::init` 全量入口开头：

```cpp
    // NOTE (Task 5): the fixed-mu product branch (v2 "fixed_mu") is inserted
    // HERE, before the warm-start branch, so it wins over warm start.  Task 3
    // lands the cold/warm split first; Task 5 adds that branch with its
    // all-or-none configure guard.
    // Branch: warm start (M2 C1) — only MD/relax, only after a converged
    // previous geometry, only with an unchanged constraint list.
    if (warm_start_eligible(cfg, specs, istep))
    {
        warm_reinit(ucell, rho_basis, cfg, specs, radii, nelec);
        return;
    }
    // Branch: cold start (first step / scf / warm_start=false / changed
    // constraint list) — the pre-M2 path.
```

`warm_start_eligible`：

```cpp
bool ConstraintLoop::warm_start_eligible(const ConstraintConfig& cfg,
                                         const std::vector<ConstraintSpec>& specs,
                                         const int istep) const
{
    // Guard: the feature is scoped to multi-geometry runs and needs a
    // previous converged geometry state in this process.
    if (!cfg.enabled || !cfg.warm_start || istep <= 0
        || !geometry_state_valid_ || !warm_start_calculation())
    {
        return false;
    }
    // Guard: an unreachable / branch-flipped previous frame must not be
    // recycled; the run needs a clean reference instead.
    if (status_ != MuStatus::CONVERGED)
    {
        return false;
    }
    // Guard: warm start reuses the solver vectors; a changed list would
    // silently pair mu with the wrong fragment.
    if (specs.size() != specs_.size())
    {
        return false;
    }
    for (size_t a = 0; a < specs.size(); ++a)
    {
        if (specs[a].kind != specs_[a].kind || specs[a].atoms != specs_[a].atoms)
        {
            return false;
        }
    }
    return true;
}
```

`warm_reinit`（不调用 `reset()`）：

```cpp
void ConstraintLoop::warm_reinit(...)
{
    warm_started_last_init_ = true;
    cfg_ = cfg;
    specs_ = specs;
    nelec_ = nelec;
    kinds_.clear(); channels_.clear(); mu_caps_.clear();
    for (const ConstraintSpec& s : specs_) { ... same fill as cold init ... }
    build_geometry(ucell, rho_basis, radii, specs_); // WeightGrid + dipole arm
    // The first SCF of the new geometry is a probe at the carried mu: no
    // inner updates may move mu before Q(μ_prev) is observed.
    inner_active_ = false;
    warm_pending_ = (cfg_.target_mode == "delta");
    status_ = MuStatus::RUNNING;
    phase_ = LoopPhase::CONSTRAINED;
    saw_resolved_ = false;
    // Q_ from the previous geometry is stale; rebind happens at the first
    // converged observation (Task 4).
    GlobalV::ofs_running
        << "[constraint] WARM_START: geometry rebuilt, mu carried (|mu|="
        << mu_norm() << " Ry), phase=constrained"
        << (warm_pending_ ? "; delta Q_ref will be re-derived at the first "
                            "converged SCF"
                          : "")
        << "\n";
}
```

`build_geometry` 抽出现有 WeightGrid 构建 + M1 dipole arming。冷 init 与 warm reinit 共用；区别仅冷 init 在 `reset()` 后调用。

`geometry_state_valid_ = true` 在全量 init 末尾与 warm_reinit 末尾设置；`reset()` 置 false、`warm_started_last_init_=false`、`warm_pending_=false`。

esolver 调用点改为：

```cpp
constraint::ConstraintLoop::instance().init(
    ucell, this->pw_rhod, cfg, specs, radii, PARAM.inp.nelec, istep);
```

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(constraint): M2 cold/warm init split + esolver istep plumbing"
```

---

### Task 4: delta 模式 Q_ref 反推 + 首 SCF 探针

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h/.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
// Delta warm start: the first converged observation of the new geometry is
// taken at the carried mu; Q_ref is re-derived as Q_obs - kappa*mu, and the
// targets follow the new geometry instead of the old reference.
TEST_F(ConstraintLoopTest, WarmStartDeltaRedefinesTarget)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.warm_start = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    const double* rp[1] = {rho_ref.data()};
    loop.observe(1, rp, 1);
    bool conv = true;
    loop.on_scf_converged(2, conv);
    const double target_old = loop.targets()[0];

    // New geometry: rebuild with warm start, then hand in the first
    // converged observation at the carried mu.
    ucell->atoms[1].taud[0].x += 0.05;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 1);
    std::vector<double> rho = rho_ref;
    const double* rp2[1] = {rho.data()};
    loop.observe(3, rp2, 1);
    bool conv2 = true;
    loop.on_scf_converged(4, conv2);
    // Q_ref moved with the geometry: the new target differs from the old.
    EXPECT_NE(loop.targets()[0], target_old);
}
```

Absolute 模式测试：

```cpp
TEST_F(ConstraintLoopTest, WarmStartAbsoluteKeepsTarget)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.target_mode = "absolute";
    cfg.targets[0].value = 9.9;
    cfg.warm_start = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    const double* rp[1] = {rho_ref.data()};
    loop.observe(1, rp, 1);
    bool conv = true;
    loop.on_scf_converged(2, conv);
    const double target_old = loop.targets()[0];
    ucell->atoms[1].taud[0].x += 0.05;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 1);
    loop.observe(3, rp, 1);
    loop.on_scf_converged(4, conv);
    EXPECT_DOUBLE_EQ(loop.targets()[0], target_old);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -5
```

- [ ] **Step 3: 实现**

在 `on_scf_converged` 的 `outer_step(iter, conv_esolver)` 之前加入：

```cpp
    // Branch: the first converged SCF of a warm-started geometry.  Q_ is the
    // observation at the carried mu; anchor the secant history there and (in
    // delta mode) re-derive the geometry's free reference before stepping.
    if (warm_pending_)
    {
        finish_warm_start();
    }
```

```cpp
void ConstraintLoop::finish_warm_start()
{
    // Anchor at the new geometry: the previous Q_prev belongs to another
    // ionic configuration and must not enter the slope.
    mu_solver_.rebind(mu_, Q_);
    // Branch: delta mode -> the target is measured relative to THIS
    // geometry's free reference.  Q(μ_prev) = Q_ref + κ μ_prev, so
    // Q_ref = Q_obs − κ μ_prev with the κ carried across the step.
    if (cfg_.target_mode == "delta")
    {
        for (size_t a = 0; a < mu_.size(); ++a)
        {
            Q_ref_[a] = Q_[a] - mu_solver_.slope(static_cast<int>(a)) * mu_[a];
            targets_[a] = Q_ref_[a] + specs_[a].target;
        }
        GlobalV::ofs_running
            << "[constraint] WARM_START delta: Q_ref re-derived at mu_prev="
            << mu_[0] << " Ry (c[0]); targets rebuilt from the new geometry\n";
    }
    // The probe is over: re-arm the INNER schedule (if selected) so later
    // SCF iterations may update mu again.
    inner_active_ = inner_schedule_;
    warm_pending_ = false;
}
```

`finish_warm_start` 声明为 private；多约束时逐分量打印请用循环（上面只打印 c[0] 作为提示）。

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(constraint): M2 delta Q_ref re-derivation on warm start"
```

---

### Task 5: 定 μ 产品化（v2 fixed_mu，优先级高于环境变量）

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h/.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
TEST_F(ConstraintLoopTest, FixedMuFromSpecsRunsOneScf)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    std::vector<constraint::ConstraintSpec> specs;
    constraint::ConstraintSpec s;
    s.kind = constraint::ConstraintKind::Charge;
    s.chan = constraint::build_channel_profile(s.kind);
    s.atoms = {0};
    s.target = 3.0;
    s.mu_max = 5.0;
    s.fixed_mu = true;
    s.fixed_mu_value = -0.12;
    specs.push_back(s);
    constraint::ConstraintConfig cfg = make_cfg(0.0);
    loop.init(*ucell, rhopw, cfg, specs, radii, 10.0, 0);
    ASSERT_TRUE(loop.enabled());
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    EXPECT_DOUBLE_EQ(loop.mu()[0], -0.12);
    const double* rp[1] = {rho_ref.data()};
    loop.observe(1, rp, 1);
    bool conv = true;
    loop.on_scf_converged(2, conv);
    EXPECT_TRUE(loop.done());
    EXPECT_TRUE(loop.fixed_mu_mode());
}

TEST_F(ConstraintLoopTest, FixedMuEnvironmentStillWorks)
{
    setenv("ABA_CONSTRAINT_FIXED_MU", "-0.07", 1);
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.0), radii, 10.0, 0);
    EXPECT_TRUE(loop.fixed_mu_mode());
    EXPECT_DOUBLE_EQ(loop.mu()[0], -0.07);
    unsetenv("ABA_CONSTRAINT_FIXED_MU");
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -5
```

- [ ] **Step 3: 实现**

`constraint_loop.h`：

```cpp
    bool fixed_mu_mode() const { return fixed_mu_; }
```

`constraint_loop.cpp` 的 init 中，在环境变量分支之前：

```cpp
    // Branch: v2 fixed_mu (M2 C4) -> product fixed-mu mode.  Every component
    // has a value (configure guard); the outer loop never runs and one SCF
    // is one frame.  The legacy env switch below stays as the v1 path.
    bool json_fixed = !specs_.empty();
    for (const ConstraintSpec& s : specs_)
    {
        json_fixed = json_fixed && s.fixed_mu;
    }
    if (json_fixed)
    {
        // Guard: the env experiment must not silently override a file that
        // says exactly what mu to use.
        if (std::getenv("ABA_CONSTRAINT_FIXED_MU") != nullptr)
        {
            ModuleBase::WARNING("ConstraintLoop::init",
                "ABA_CONSTRAINT_FIXED_MU is ignored because the v2 target "
                "file sets fixed_mu");
        }
        for (size_t a = 0; a < specs_.size(); ++a)
        {
            mu_[a] = specs_[a].fixed_mu_value;
            targets_[a] = specs_[a].target;
        }
        fixed_mu_ = true;
        inner_active_ = false;
        phase_ = LoopPhase::CONSTRAINED;
        status_ = MuStatus::RUNNING;
        return;
    }
```

`outer_step` 的 fixed 分支保持现状（首次 SCF 收敛后 DONE），但审计/最终状态区分来源：

```cpp
        GlobalV::ofs_running
            << "FIXED_MU (mu fixed by input/env; targets are labels, not "
               "enforced)\n";
```

`print_audit_line` 不新增字段；`CONSTRAINT_AUDIT` 的 `q/t/res` 照常输出供标注。

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(constraint): M2 fixed_mu v2 product mode"
```

---

### Task 6: 全量回归 + 集成验证（h2o_dimer relax，100 步上限）

**Files:**
- New: `docs/superpowers/specs/2026-10-20-m2-warmstart-fixedmu-test.md`
- Modify: `docs/superpowers/specs/deltap-development-log.md`

- [ ] **Step 1: 全量单元回归**

```bash
cmake --build build -j 32 2>&1 | tail -3
ctest --test-dir build -j 32 -R constraint --output-on-failure 2>&1 | tail -15
```
预期：除本机环境性 MPI 两例外全部通过；若 MPI 正常应 12/12。

- [ ] **Step 2: 冷/热对比（h2o_dimer relax，2 个离子步）**

在临时工作目录构造：

```
calculation      relax
relax_nmax       2
relax_method     bfgs
scf_nmax         100
constraint_warm_start 1     # warm run
constraint_target_mode absolute   # MD 推荐，避免 t 随几何漂移
constraint_target_file constraint_target.json
```

目标文件（absolute，固定片段目标）：

```json
{"constraints":[
  {"type":"charge","target":10.04285205,"atoms":[0,2,3]},
  {"type":"charge","target":9.957147948,"atoms":[1,4,5]}]}
```

运行两次（同结构、同 SCF 上限）：

```bash
mpirun -np 8 abacus_basic_para > run_warm.log 2>&1   # warm_start=1
mpirun -np 8 abacus_basic_para > run_cold.log 2>&1   # warm_start=0
```

判据：
- 从 `running_scf.log` 统计每个离子步的 SCF 迭代数；
- 第 2 个离子步 warm 的 SCF 数 ≤ cold 的 1/3；
- 两个运行在第 2 步收敛后的 `q` 与 `mu` 差异 < 1%（absolute 模式 target 相同）；
- 若第 2 步在 100 步内未收敛，记录 drho 轨迹（每离子步的 iter/deviation）并分析，不提高上限。

- [ ] **Step 3: delta 反推冒烟**

同一用例 `constraint_target_mode delta`、`constraint_warm_start 1`，`scf_nmax 100`。检查日志出现：

```
[constraint] WARM_START: geometry rebuilt, mu carried ...
[constraint] WARM_START delta: Q_ref re-derived ...
```

且 `targets` 随几何更新（比对两次 audit 的 `t=`）。若 100 步内未走到第二个离子步，记录并转 absolute 路径作为验收。

- [ ] **Step 4: 定 μ 60 步冒烟**

同一 h2o_dimer，`calculation md` 或 `relax`，目标文件改为：

```json
{"constraints":[
  {"type":"charge","target":10.04285205,"atoms":[0,2,3],"fixed_mu":-0.0654},
  {"type":"charge","target":9.957147948,"atoms":[1,4,5],"fixed_mu":0.0654}]}
```

判据：
- 每个离子步只跑一次 SCF（`final status: FIXED_MU`，无 outer step）；
- `CONSTRAINT_AUDIT` 的 `q` 可用于标注；
- 与 warm-start 扫描同一 μ 的 Q 偏差 < 1e-2 e。

- [ ] **Step 5: 写测试文档 + 更新开发日志**

按 AGENTS.md 五段格式写 `2026-10-20-m2-warmstart-fixedmu-test.md`，并在
`deltap-development-log.md` 追加本轮条目（做了什么 / 文件 / bug / 下一步 M3）。

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-10-20-m2-warmstart-fixedmu-test.md docs/superpowers/specs/deltap-development-log.md
git commit -m "docs(constraint): M2 warm-start + fixed_mu test round"
```

---

## Self-Review 记录

1. **Spec 覆盖**：§5.1 的冷/热拆分、μ/κ/phase 保留、delta Q_ref 反推、absolute 推荐、仅 MD/relax 生效；§5.4 的 v2 `fixed_mu`、一次 SCF 一帧、审计 Q 输出、env 兼容——全部映射到 Task 1-6。✅
2. **默认路径**：`constraint_warm_start` 默认 true 但 `istep<=0` / `calculation=scf` / 无上一几何状态时走冷路径；单点输出不变。✅
3. **风险**：①delta 反推依赖跨几何 κ，首个 warm 步可能偏差大——Task 4 用 rebind 清除陈旧 Q 历史并以保留 κ 起步，集成测试用 absolute 主验收；②fixed_mu 与 warm_start 同开时 fixed 优先（每个几何重置到输入 μ）；③`warm_start_eligible` 要求上一帧 `CONVERGED`，未收敛帧回退冷启动并打印警告。✅
4. **类型一致性**：`ConstraintConfig::warm_start`、`ConstraintSpec::{fixed_mu,fixed_mu_value}`、`ConstraintLoop::{init(...,istep),warm_started_last_init(),fixed_mu_mode()}`、`MuSolver::{rebind,slope}` 在全部 Task 间一致。✅
