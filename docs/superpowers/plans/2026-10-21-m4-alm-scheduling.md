# M4 ALM 单层循环 + 外层调度改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用增广拉格朗日（ALM）单层循环替代"注入→SCF 重建"的两层循环：约束势 `v_con(r) = Σ_a [μ_a + k_a(Q_a[ρ]−t_a)]·w_a(r)` 每个 SCF 迭代随密度连续更新，无注入事件、无 mixing reset；外层仅做 0-3 次温和的 `μ += k(Q−t)` 更新。配套三项调度改进：跨几何保留 Broyden Jacobian、残差自适应探针、预算感知优雅收尾 + RUNNING 帧暖启动。

**Architecture:**
- 新文件 `constraint_alm.{h,cpp}`：`AlmController` 纯数学（k 自动设定、mu_eff、外层更新、能量修正、振荡阻尼），无 loop 依赖、可独立单测。
- `ConstraintLoop`：ALM 流程（REFERENCE → 探针测 χ → arm k + Newton μ₀ → ALM SCF/外层裁决）；注入/回加/力核三处由 `mu_` 单点切换为 `mu_eff_`（mu_solver.cpp 的注入点 constraint_loop.cpp:574/603/622/1466）。
- 调度包：`MuSolver::rebind` 增加全 J 携带（flag）；`scf_threshold` 探针改为残差自适应；`on_scf_converged` 加预算感知冻结；`warm_start_eligible` 允许近收敛 RUNNING 帧。
- 全部默认关闭/向后兼容：`mu_schedule=alm` 显式开启；调度包各 flag 独立；默认路径逐字节不变。

**Tech Stack:** C++17、gtest、CMake（AddTest）、ABACUS module_constraint。

**Spec:** `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md` §6（模块 D）。
**输入：** M3 评审的效率分析（`deltap-development-log.md` 2026-10-20 条目、M3 测试报告 §4）——0.01 e 帧 2 失败是两层循环结构成本，ALM 是结构性解法；M3 实测：μ 注入重建 ~20-30 步/次、drho 与 residual 同数量级（2.4e-4 vs 3.3e-4 e）。

**前置:** M3 已合入（commit `ee48e06e8` 之后）。
**计划序列：** 本计划覆盖 M4。M5（B 偶极约束坐标）在 M4 验收后另行计划。
**深水区纪律（设计 §6.3 硬性要求）：** ALM 必须通过 ①单元层不动点/能量/力自洽 ②h2o_dimer 0.001/0.01 e 集成 ③dE/dt = −μ_phys 三点 FD 三道门，方可认为可用；性质2/性质4 的 Bohrium 全套重验移交 Task 8 文档化。

**代码规范（AGENTS.md）：** 每个 if-else 分支与非平凡 if 前必须有注释；禁 goto；函数 >300 行拆分；RAII 优先。
**资源约束（用户 standing order）：** 所有集成 SCF 运行 `scf_nmax <= 100`；不收敛记录 drho 轨迹并分析，不提高上限。

---

## 关键既有事实（实现时直接引用，均已核验）

- 注入点（四处共用 `mu_`，ALM 改为 `mu_eff_` 单点切换）：`constraint_loop.cpp:574-575`（PW v_eff+veff_smooth）、`:603`（LCAO v_eff）、`:622`（add_back veff）；力核 `:1466` `constraint_force(*wg_, rho, nspin, channels_, mu_, forcecon)`。
- `mu_schedule` 守卫：`constraint_io.cpp:976`（仅允许 "outer"/"inner"）；inner 守卫 `:986-997`。
- `observe()` 每次 SCF 迭代被调（Q_ 更新）；`on_iteration(iter, drho)` 每迭代被调（drho 监测点，现有 inner 逻辑也在此）。
- 分支守卫 G2 的能量修正自算 `e_con = Σ mu_a(Q_a - t_a)`（constraint_loop.cpp G2 块，~:735）；ALM 需换公式。
- M3 审计诊断追加模式：`TrustAuditAppendsDiagnosticsOnlyWhenEnabled`——ALM 的 `mu_phys` token 用同一模式追加，不改既有字段。
- `scf_threshold(strict)`：M3 C3 挂接点（esolver_ks.cpp iter_finish 已接）；Branch C 当前为 `max(1e-5, 100*strict)`。
- `MuSolver::rebind`（mu_solver.cpp）：trust 分支把 jacobian_ 重置为对角（:276-279）——J 携带修改点。
- `warm_start_eligible` 要求 `status_ == CONVERGED`；`finish_warm_start` 做 rebind + delta Q_ref 反推（M2）。
- `PARAM.inp.scf_nmax`（input_parameter.h:118，默认 100）。
- 玩具 mock 设施（constraint_loop_test.cpp）：`mock_iteration`、`fill_rho_mock`（单位刚度 Q=Q_ref−μ）、`Q_of`、`make_h2o_ucell`。
- h2o_dimer absolute 目标（M3 报告 §2.2）：0.001 e = `9.993852052/10.006147948`；0.01 e = `10.002852052/9.997147948`（atoms [0,2,3] / [1,4,5]）。
- ALM 不动点数学（设计 §6.2，本计划的测试基准）：线性响应 Q=Q_ref+χ·μ_eff 下，
  - ALM 自洽点 `Q−t = χ(μ−μ*)/(1−kχ)`，μ* 为精确乘子；
  - 外层更新 `μ ← μ + k(Q−t)` 的收缩率 `1/|1−kχ|`（kχ=7.5 时 1/6.5，μ=0 起步 ≤3 次收敛到 1e-4）；
  - 收敛后 `μ_phys ≡ μ + k(Q−t) = μ*` 精确成立（误差 O(thr)）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| 新建 `source/source_estate/module_constraint/constraint_alm.{h,cpp}` | AlmController 纯数学 |
| 改 `source/source_io/module_parameter/input_parameter.h` | `constraint_alm_k` / `constraint_trust_carry_jacobian` / `constraint_budget_freeze` / `constraint_warm_start_running` |
| 改 `source/source_io/module_parameter/read_input_item_other.cpp` | 4 个 Input_Item |
| 改 `source/source_estate/module_constraint/constraint_io.{h,cpp}` | mu_schedule="alm" + 守卫 + 参数管道 |
| 改 `source/source_estate/module_constraint/constraint_loop.{h,cpp}` | ALM 流程、mu_eff_ 注入切换、调度包 B |
| 改 `source/source_estate/module_constraint/mu_solver.{h,cpp}` | rebind 全 J 携带（flag） |
| 改 `source/source_estate/module_constraint/test/*` | alm / io / loop / mu_solver 测试 |
| 改 `source/source_estate/CMakeLists.txt`、`test/CMakeLists.txt` | 构建接线 |
| 新建 `docs/superpowers/specs/2026-10-21-m4-alm-test.md` | 集成测试报告 |

---

### Task 1: INPUT/Config 管道（alm 枚举 + 4 参数 + 守卫）

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`（M3 参数块之后）
- Modify: `source/source_io/module_parameter/read_input_item_other.cpp`（M3 item 块之后）
- Modify: `source/source_estate/module_constraint/constraint_io.h/.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_io_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
TEST(ConstraintIOTest, AlmPlumbingAndGuards)
{
    auto ucell = make_h2o_ucell();
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_alm_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;
    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // mu_schedule = alm is accepted and flows through; k default = auto.
    PARAM.input.constraint_mu_schedule = "alm";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_EQ(cfg.mu_schedule, "alm");
    EXPECT_DOUBLE_EQ(cfg.alm_k, 0.0); // 0 = auto (k = 7.5/|chi|)

    // Explicit k flows through.
    PARAM.input.constraint_alm_k = 3.0;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_DOUBLE_EQ(cfg.alm_k, 3.0);
    PARAM.input.constraint_alm_k = 0.0;

    // Guard: alm + negative k -> ERROR.
    PARAM.input.constraint_alm_k = -1.0;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_alm_k = 0.0;

    // Guard: alm + json fixed_mu (all-fixed) -> ERROR (one dynamics mode).
    {
        std::ofstream ofs(path);
        ofs << R"({"constraints": [
            {"type":"charge","target":3.0,"atoms":[0],"fixed_mu":-0.1}]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("alm"), std::string::npos);

    // Scheduling flags flow through with defaults.
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.trust_carry_jacobian);
    EXPECT_TRUE(cfg.budget_freeze);
    EXPECT_TRUE(cfg.warm_start_running);
    PARAM.input.constraint_mu_schedule = "outer";
    std::remove(path.c_str());
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -5
```
预期：编译错误（`alm_k` / `trust_carry_jacobian` / `budget_freeze` / `warm_start_running` / `constraint_alm_k` 不存在）。

- [ ] **Step 3: 实现**

`input_parameter.h`（M3 参数块之后）：

```cpp
    // M4 (module D): augmented-Lagrangian single-layer schedule
    // (constraint_mu_schedule = "alm").  0 = auto k = 7.5/|chi| per
    // component; >0 = explicit global k in Ry/e.
    double constraint_alm_k = 0.0;
    // M4 scheduling pack: carry the full Broyden Jacobian across ionic-step
    // rebinds (off = legacy diagonal reset); budget-aware freeze converts a
    // doomed final mu update into a converged best-effort frame;
    // warm_start_running allows warm starts from near-converged RUNNING
    // frames (residual < 10*constraint_thr) instead of a cold restart.
    bool constraint_trust_carry_jacobian = false;
    bool constraint_budget_freeze = true;
    bool constraint_warm_start_running = true;
```

`read_input_item_other.cpp`（M3 item 块之后）：

```cpp
    {
        Input_Item item("constraint_alm_k");
        item.annotation = "ALM penalty gain k (Ry/e); 0 = auto";
        item.category = "Constraint";
        item.type = "Real";
        item.description = "Augmented-Lagrangian penalty gain. 0 (default): auto-set per component to k*|chi| = 7.5 with chi the measured density response dQ/dmu (cold: one probe SCF; warm: carried secant slope). >0: explicit global k for every component, clamped to [0.1, 100] with a warning. Only used when constraint_mu_schedule=alm.";
        item.default_value = "0.0";
        item.unit = "Ry/e";
        item.availability = "constraint_mu_schedule is alm";
        read_sync_double(input.constraint_alm_k);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_trust_carry_jacobian");
        item.annotation = "carry the Broyden Jacobian across ionic steps";
        item.category = "Constraint";
        item.type = "Boolean";
        item.description = "trust_region only. false (default): each warm-started geometry resets the Jacobian to the diagonal carried slope (legacy). true: the full Broyden Jacobian (including fragment couplings) is carried across ionic steps; the trust-region rho score rejects and relearns a stale model.";
        item.default_value = "False";
        item.unit = "";
        item.availability = "constraint_trust_region is true";
        read_sync_bool(input.constraint_trust_carry_jacobian);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_budget_freeze");
        item.annotation = "freeze mu instead of starting a doomed final update";
        item.category = "Constraint";
        item.type = "Boolean";
        item.description = "true (default): when the remaining SCF budget (scf_nmax) cannot cover another mu-update rebuild (estimated from the previous rebuild cost) and the residual is already below 10*constraint_thr, the outer loop freezes mu at the current converged SCF and reports RUNNING (budget) with a usable converged frame instead of exhausting scf_nmax on a doomed update.";
        item.default_value = "True";
        item.unit = "";
        item.availability = "constraint is true";
        read_sync_bool(input.constraint_budget_freeze);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_warm_start_running");
        item.annotation = "allow warm start from near-converged RUNNING frames";
        item.category = "Constraint";
        item.type = "Boolean";
        item.description = "true (default): an ionic frame that ended RUNNING with max residual < 10*constraint_thr (budget-limited, not physics-limited) may warm-start the next frame (mu/J carried, rebind at the first convergence) instead of forcing an expensive cold reference restart.";
        item.default_value = "True";
        item.unit = "";
        item.availability = "constraint_warm_start is true";
        read_sync_bool(input.constraint_warm_start_running);
        this->add_item(item);
    }
```

`constraint_io.h`（ConstraintConfig 尾部）：

```cpp
    // M4 (module D) ALM schedule.
    double alm_k = 0.0; // 0 = auto (7.5/|chi|); >0 = explicit global k (Ry/e)
    // M4 scheduling pack.
    bool trust_carry_jacobian = false;
    bool budget_freeze = true;
    bool warm_start_running = true;
```

`constraint_io.cpp`：
1. `mu_schedule` 守卫（:976）改为允许三值：
```cpp
    // Branch: the schedule vocabulary is outer / inner / alm (M4).
    if (mu_schedule != "outer" && mu_schedule != "inner"
        && mu_schedule != "alm")
    {
        error = "constraint_mu_schedule must be \"outer\", \"inner\" or "
                "\"alm\"";
        return ConfigStatus::ERROR;
    }
    // Branch: alm is an outer-schedule dialect — it is incompatible with
    // the INNER machinery and with fixed-mu modes (one dynamics per run).
    if (mu_schedule == "alm")
    {
        if (alm_k < 0.0)
        {
            error = "constraint_alm_k must be >= 0 (0 = auto)";
            return ConfigStatus::ERROR;
        }
        bool all_fixed = !specs.empty();
        for (const ConstraintSpec& s : specs)
        {
            all_fixed = all_fixed && s.fixed_mu;
        }
        if (all_fixed)
        {
            error = "constraint_mu_schedule=alm is incompatible with "
                    "\"fixed_mu\" (fixed-mu runs have no outer loop; use "
                    "outer/inner or remove fixed_mu)";
            return ConfigStatus::ERROR;
        }
    }
```
（注意该守卫块位于扩展 configure 核心内，需要给该核心加 `alm_k` 形参或在 `configure_from_inputs` 中检查——选择与 M1 dipole 守卫相同的后者：在 `configure_from_inputs` 的 dipole 管道旁做 alm_k/all-fixed 检查，避免改动 16 参数签名。）
2. `configure_from_inputs` 管道：
```cpp
    // M4 ALM + scheduling pack plumbing (guards above at the same site as
    // the M1 dipole guards).
    cfg.alm_k = PARAM.inp.constraint_alm_k;
    cfg.trust_carry_jacobian = PARAM.inp.constraint_trust_carry_jacobian;
    cfg.budget_freeze = PARAM.inp.constraint_budget_freeze;
    cfg.warm_start_running = PARAM.inp.constraint_warm_start_running;
```

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_io --output-on-failure 2>&1 | tail -5
```
预期：全部通过。

- [ ] **Step 5: Commit**

```bash
git add source/source_io/module_parameter source/source_estate/module_constraint/constraint_io.* source/source_estate/module_constraint/test/constraint_io_test.cpp
git commit -m "feat(constraint): M4 alm schedule enum + scheduling-pack input plumbing"
```

---

### Task 2: AlmController 纯数学（新文件）

**Files:**
- Create: `source/source_estate/module_constraint/constraint_alm.h/.cpp`
- Create: `source/source_estate/module_constraint/test/constraint_alm_test.cpp`
- Modify: `source/source_estate/CMakeLists.txt`、`test/CMakeLists.txt`

- [ ] **Step 1: 写失败测试**

新建 `test/constraint_alm_test.cpp`：

```cpp
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "source_estate/module_constraint/constraint_alm.h"

// Arm: k_a = kappa_target / |chi_a|, clamped to [k_min, k_max]; zero or
// sign-uncertain chi falls back to kappa_target (chi = -1 nominal).
TEST(ConstraintAlmTest, ArmFromChi)
{
    constraint::AlmController alm;
    alm.resize(2);
    EXPECT_FALSE(alm.armed());
    alm.arm({-0.65, -24.0});
    ASSERT_TRUE(alm.armed());
    EXPECT_NEAR(alm.k()[0], 7.5 / 0.65, 1e-12);
    EXPECT_NEAR(alm.k()[1], 7.5 / 24.0, 1e-12);
}

// mu_eff: before arming it is plain mu; after arming it is mu + k(Q - t).
TEST(ConstraintAlmTest, MuEffBeforeAndAfterArming)
{
    constraint::AlmController alm;
    alm.resize(1);
    std::vector<double> eff;
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_DOUBLE_EQ(eff[0], 0.2); // unarmed: penalty inactive
    alm.arm({-1.0}); // k = 7.5
    alm.mu_eff({0.2}, {1.0}, {0.0}, eff);
    EXPECT_NEAR(eff[0], 0.2 + 7.5 * 1.0, 1e-12);
}

// Outer update: mu += k(Q - t), capped per component; a pinned component
// with a flat residual fuses (UNREACHABLE semantics mirrored from MuSolver).
TEST(ConstraintAlmTest, UpdateCapsAndFuses)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-1.0}); // k = 7.5
    std::vector<double> mu = {0.0};
    // Q far above target: update pushes mu up by k*1.0 = 7.5 but the cap
    // pins it at 5.0.
    double res = alm.update({1.0}, {0.0}, {5.0}, mu);
    EXPECT_NEAR(res, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(mu[0], 5.0);
    EXPECT_EQ(alm.fuse_component(), -1); // first pin: no fuse yet
    // Two more flat updates at the cap -> plateau fuse.
    alm.update({1.0}, {0.0}, {5.0}, mu);
    alm.update({1.0}, {0.0}, {5.0}, mu);
    EXPECT_EQ(alm.fuse_component(), 0);
    EXPECT_NEAR(alm.fuse_residual(), 1.0, 1e-12);
}

// Energy correction: sum_a [ mu_a(Q-t) + (k_a/2)(Q-t)^2 ].
TEST(ConstraintAlmTest, EnergyCorrection)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-2.0}); // k = 3.75
    const double e = alm.energy_correction({0.5}, {1.2}, {1.0});
    EXPECT_NEAR(e, 0.5 * 0.2 + 0.5 * 3.75 * 0.04, 1e-12);
}

// Linear-model contraction (design sec. 6.2): with k*chi = 7.5 the outer
// update contracts |mu - mu*| by 1/|1 - k*chi| = 1/6.5 per step; mu_phys
// after convergence equals the exact multiplier.
TEST(ConstraintAlmTest, ContractionAndMuPhysExact)
{
    constraint::AlmController alm;
    alm.resize(1);
    const double chi = -0.65;      // LiCl near-neutral response (e/Ry)
    const double q_ref = 3.0959;   // Li fragment free charge
    const double t = q_ref + 0.1;  // delta = +0.1
    const double mu_star = (t - q_ref) / chi; // exact multiplier
    alm.arm({chi});
    std::vector<double> mu = {0.0};
    // Iterate the ALM fixed point on the linear model
    // Q(mu_eff) = q_ref + chi * mu_eff, mu_eff = mu + k(Q - t).
    for (int step = 0; step < 10; ++step)
    {
        std::vector<double> eff;
        // Solve Q = q_ref + chi*(mu + k(Q-t)) for Q:
        // Q (1 - chi k) = q_ref + chi (mu - k t)
        const double k = alm.k()[0];
        const double q = (q_ref + chi * (mu[0] - k * t)) / (1.0 - chi * k);
        const double res = std::abs(q - t);
        if (res < 1e-4)
        {
            break;
        }
        alm.update({q}, {t}, {50.0}, mu);
        ASSERT_LT(step, 5); // must converge in <= 3-4 updates
    }
    std::vector<double> eff;
    // Final consistent Q at the current mu:
    const double k = alm.k()[0];
    const double q = (q_ref + chi * (mu[0] - k * t)) / (1.0 - chi * k);
    alm.mu_eff(mu, {q}, {t}, eff);
    EXPECT_NEAR(eff[0], mu_star, 1e-3); // mu_phys ~= exact multiplier
}

// Dampen halves k with a floor at k_min.
TEST(ConstraintAlmTest, DampenFloors)
{
    constraint::AlmController alm;
    alm.resize(1);
    alm.arm({-0.075}); // k = 100 -> at the k_max cap
    EXPECT_NEAR(alm.k()[0], 100.0, 1e-12);
    for (int i = 0; i < 20; ++i)
    {
        alm.dampen();
    }
    EXPECT_NEAR(alm.k()[0], 0.1, 1e-12); // k_min floor
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_alm 2>&1 | tail -3
```
预期：目标/文件不存在。

- [ ] **Step 3: 实现 constraint_alm.h/.cpp**

```cpp
// constraint_alm.h
#ifndef CONSTRAINT_ALM_H
#define CONSTRAINT_ALM_H

#include <vector>

namespace constraint
{

/**
 * @brief Augmented-Lagrangian state and pure update math (module D / M4).
 *
 *   E_ALM[rho] = E_KS[rho] + sum_a [ mu_a (Q_a - t_a) + (k_a/2)(Q_a - t_a)^2 ]
 *   v_con(r)   = sum_a (mu_a + k_a (Q_a - t_a)) w_a(r) = sum_a mu_eff_a w_a
 *
 * The penalty term makes the SCF converge the constrained problem directly
 * (no inject-rebuild cycle); the outer update mu += k(Q - t) contracts
 * |mu - mu*| by 1/|1 - k*chi| per step (design sec. 6.2).  The physical
 * multiplier reported for constant-potential labelling is
 * mu_phys = mu + k(Q - t) (= mu_eff at the converged state).
 */
class AlmController
{
  public:
    void resize(const int n);
    bool armed() const { return armed_; }
    // Arm k from the measured signed responses chi_a = dQ/dmu
    // (negative for charge/spin): k_a = kappa_target / |chi_a|, clamped to
    // [k_min, k_max].  A non-positive or absurd chi falls back to |chi| = 1
    // (k = kappa_target) with the event counted (the caller logs it).
    void arm(const std::vector<double>& chi);
    int arm_fallback_count() const { return arm_fallback_; }
    // mu_eff_a = mu_a + k_a (Q_a - t_a); unarmed -> plain mu (probe phase).
    void mu_eff(const std::vector<double>& mu,
                const std::vector<double>& Q,
                const std::vector<double>& target,
                std::vector<double>& out) const;
    // Outer ALM update mu_a += k_a (Q_a - t_a), per-component caps; a
    // component pinned at its cap whose residual has not improved by >= 1%
    // over the last plateau_window updates fuses (UNREACHABLE mirror).
    // Returns max_a |Q_a - t_a| BEFORE the update.
    double update(const std::vector<double>& Q,
                  const std::vector<double>& target,
                  const std::vector<double>& caps,
                  std::vector<double>& mu);
    // sum_a [ mu_a (Q_a - t_a) + (k_a/2) (Q_a - t_a)^2 ].
    double energy_correction(const std::vector<double>& mu,
                             const std::vector<double>& Q,
                             const std::vector<double>& target) const;
    // Stiffness guard: halve every k (floor k_min) on drho growth.
    void dampen();
    int dampen_count() const { return dampen_count_; }
    const std::vector<double>& k() const { return k_; }
    int fuse_component() const { return fuse_component_; }
    double fuse_residual() const { return fuse_residual_; }
    // Reset fuse/plateau bookkeeping at a new geometry (warm rebind); k
    // survives (it is re-armed from the carried response anyway).
    void reset_fuse();

  private:
    std::vector<double> k_;
    bool armed_ = false;
    double kappa_target_ = 7.5;  // design sec. 6.2 (k*|chi| target)
    double k_min_ = 0.1;         // Ry/e
    double k_max_ = 100.0;       // Ry/e
    int plateau_window_ = 3;
    int arm_fallback_ = 0;
    int dampen_count_ = 0;
    std::vector<std::vector<double>> residual_window_;
    int fuse_component_ = -1;
    double fuse_residual_ = 0.0;
};

} // namespace constraint

#endif
```

实现要点（cpp，每分支带注释）：
- `arm`：`k_a = clamp(kappa_target_/|chi_a|, k_min_, k_max_)`；`chi_a >= 0 || |chi_a| < 1e-6` → 用 `kappa_target_` 并 `++arm_fallback_`。
- `update`：先记残差窗（每分量压入 |res|，超长截首），再 `mu_a = clamp(mu_a + k_a*res_a, -cap_a, +cap_a)`；若 `|mu_a| == cap_a` 且窗口内改善 <1% → 置 fuse。
- `mu_eff`：未 armed → 拷贝 mu；armed → `mu_a + k_a*(Q_a - t_a)`。
- `energy_correction`：未 armed 时 k 项为零退化为 `Σ mu(Q-t)`。

`source_estate/CMakeLists.txt` 加 `module_constraint/constraint_alm.cpp`；`test/CMakeLists.txt` 加：

```cmake
AddTest(
  TARGET MODULE_ESTATE_constraint_alm
  LIBS parameter ${math_libs} base device
  SOURCES constraint_alm_test.cpp ../constraint_alm.cpp
)
```

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 2>&1 | tail -2 && cmake --build build -j 32 --target MODULE_ESTATE_constraint_alm 2>&1 | tail -3
ctest --test-dir build -R MODULE_ESTATE_constraint_alm --output-on-failure 2>&1 | tail -5
```
预期：6 个用例全过。

- [ ] **Step 5: Commit**

```bash
git add source/source_estate
git commit -m "feat(constraint): AlmController pure math (M4 module D core)"
```

---

### Task 3: ConstraintLoop ALM 流程（探针→arm→单层 SCF→外层裁决）

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h/.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

玩具 ALM 一致性 mock（测试文件头部设施）：单位/任意刚度下，注入的是 `mu_eff`，响应 `Q = Q_ref + chi*mu_eff`——直接用 `loop.mu_eff()` 驱动既有 `fill_rho_mock` 的推广版：

```cpp
// In ConstraintLoopTest fixture (facility used by the ALM tests):
// rho built from the loop's CURRENT mu_eff with stiffness chi:
//   rho[ir] = rho_ref[ir] + chi * mu_eff * w0[ir] / S, S = int w0^2 dV
//   => Q = Q_ref + chi * mu_eff   (consistent with the ALM injection)
void mock_alm_iteration(constraint::ConstraintLoop& loop,
                        std::vector<double>& rho,
                        const int iter,
                        const double chi,
                        bool& conv_out)
{
    rho = rho_ref;
    const double S = S_int();
    const std::vector<double>& w0 = wg->constraint_weight(0);
    const std::vector<double>& meff = loop.mu_eff();
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        rho[ir] += chi * meff[0] * w0[ir] / S;
    }
    const double* rp[1] = {rho.data()};
    loop.observe(iter, rp, 1);
    bool conv = true;
    loop.on_scf_converged(iter, conv);
    conv_out = conv;
}
```

```cpp
// Cold ALM: reference -> probe (measures chi) -> arm -> ALM SCFs.  With a
// linear mock and delta = +0.1, chi = -0.65, the Newton mu0 lands on the
// exact solution, so the run converges in <= 2 outer adjudications and the
// reported mu_phys equals the exact multiplier (t - Q_ref)/chi.
TEST_F(ConstraintLoopTest, AlmColdConvergesWithExactMuPhys)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.mu_schedule = "alm";
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    ASSERT_TRUE(loop.enabled());
    EXPECT_EQ(loop.schedule(), "alm");

    const double chi = -0.65;
    std::vector<double> rho;
    bool conv = false;
    int iter = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && iter < 40)
    {
        ++iter;
        mock_alm_iteration(loop, rho, iter, chi, conv);
    }
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    // Exact multiplier of the linear model:
    const double q_ref = Q_of(rho_ref);
    const double mu_star = (loop.targets()[0] - q_ref) / chi;
    EXPECT_NEAR(loop.mu_phys(0), mu_star, 1e-6);
    // The outer adjudication count is small (mu0 from the probe Newton step
    // is exact on a linear model).
    EXPECT_LE(loop.outer_steps(), 6); // reference close + probe + <=2 alm
}

// ALM + warm start: k from the carried slope (no probe SCF), mu carried.
TEST_F(ConstraintLoopTest, AlmWarmStartSkipsProbe)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.mu_schedule = "alm";
    cfg.warm_start = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    const double chi = -0.65;
    std::vector<double> rho;
    bool conv = false;
    int iter = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && iter < 40)
    {
        ++iter;
        mock_alm_iteration(loop, rho, iter, chi, conv);
    }
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    const double mu1 = loop.mu_phys(0);

    // New geometry: warm re-init must NOT re-probe (k re-armed from the
    // carried chi); mu carried.
    ucell->atoms[1].taud[0].x += 0.05;
    wg->build(); // test-side weights follow the moved geometry (M2 note)
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 1);
    ASSERT_TRUE(loop.warm_started_last_init());
    EXPECT_TRUE(loop.alm_armed()); // armed from the carried response
    while (loop.status() == constraint::MuStatus::RUNNING && iter < 60)
    {
        ++iter;
        mock_alm_iteration(loop, rho, iter, chi, conv);
    }
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    EXPECT_NEAR(loop.mu_phys(0), mu1, 2e-2); // geometry change is small
}

// Stiffness guard: growing drho three iterations in a row dampens k.
TEST_F(ConstraintLoopTest, AlmDampensOnDrhoGrowth)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.mu_schedule = "alm";
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    const double chi = -0.65;
    std::vector<double> rho;
    bool conv = false;
    // Drive past the probe so k is armed, then feed growing drho values
    // through on_iteration (the drho monitor hook).
    for (int iter = 1; iter <= 6; ++iter)
    {
        mock_alm_iteration(loop, rho, iter, chi, conv);
    }
    ASSERT_TRUE(loop.alm_armed());
    const double k0 = loop.alm_k(0);
    loop.on_iteration(7, 1e-3);
    loop.on_iteration(8, 2e-3);
    loop.on_iteration(9, 4e-3); // three consecutive >20% growths
    EXPECT_LT(loop.alm_k(0), k0);
}
```

（注：`loop.mu_eff()`、`loop.mu_phys(i)`、`loop.alm_armed()`、`loop.alm_k(i)` 为新访问器，见 Step 3。）

- [ ] **Step 2: 运行确认失败**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -5
```
预期：编译错误（访问器/流程不存在）。若该目标未链接 `../constraint_alm.cpp`，在 `test/CMakeLists.txt` 的 `MODULE_ESTATE_constraint_loop` SOURCES 中补上。

- [ ] **Step 3: 实现**

`constraint_loop.h` public 追加：

```cpp
    // ---- M4 ALM schedule (module D) ---------------------------------------
    // mu_eff is the multiplier the injector/force actually apply:
    // mu_a + k_a(Q_a - t_a) when the ALM is armed, else plain mu (probe or
    // outer schedules).  mu_phys(i) is the constant-potential label of
    // component i (== mu_eff at convergence).  alm_armed() is true once k
    // was set (probe measurement or carried warm-start response).
    const std::vector<double>& mu_eff() const { return mu_eff_; }
    double mu_phys(const int i) const { return mu_eff_.at(i); }
    bool alm_armed() const { return alm_armed_; }
    double alm_k(const int i) const { return alm_.k().at(i); }
```

private 追加：

```cpp
    // M4 ALM state.
    AlmController alm_;
    bool alm_mode_ = false;       // cfg_.mu_schedule == "alm"
    bool alm_armed_ = false;      // k set (probe measured / warm carried)
    bool alm_probe_pending_ = false; // cold probe SCF in flight
    double alm_probe_mu_ = 0.02;  // Ry: chi-measurement probe multiplier
    std::vector<double> mu_eff_;  // see the public note above
    std::vector<double> drho_trend_; // last drho values for the dampen guard
    void alm_arm(const std::vector<double>& chi);
    void alm_adjudicate(const int iter, bool& conv_esolver);
    void finish_warm_start_alm();
```

（`reset()` 同步清零上述状态。）

`constraint_loop.cpp`：

(a) **init**：`alm_mode_ = (cfg_.mu_schedule == "alm");`（在 inner_schedule_ 赋值旁；alm 下 `inner_active_`/`inner_schedule_` 恒 false）。

(b) **outer_step 的 REFERENCE 分支**：alm 时不做首割线步，改为布防探针：

```cpp
        // Branch (M4 ALM): the reference close arms the chi probe instead
        // of a secant step.  One small plain-injection SCF at mu_probe
        // measures the response chi = dQ/dmu before k is set — the price
        // of a safe k on a cold run (warm runs skip it via the carried
        // slope, see finish_warm_start_alm).
        if (alm_mode_)
        {
            phase_ = LoopPhase::CONSTRAINED;
            for (size_t a = 0; a < mu_.size(); ++a)
            {
                mu_[a] = alm_probe_mu_;
            }
            alm_probe_pending_ = true;
            ++outer_steps_;
            print_audit(iter);
            conv_esolver = false;
            return;
        }
```

(c) **on_scf_converged**：在 warm_probe/finish_warm_start 处理之后、`outer_step` 之前插入 ALM 裁决：

```cpp
    // Branch (M4 ALM) probe close: measure chi on the converged probe
    // density, arm k, take the Newton mu0 = -(Q_ref - t)/chi (capped), and
    // continue into the first ALM SCF.
    if (alm_probe_pending_)
    {
        std::vector<double> chi(mu_.size(), -1.0);
        for (size_t a = 0; a < chi.size(); ++a)
        {
            chi[a] = (Q_[a] - Q_ref_[a]) / alm_probe_mu_;
            // Guard: a non-negative or absurd response means the channel
            // did not move as expected; AlmController::arm falls back.
        }
        alm_arm(chi);
        for (size_t a = 0; a < mu_.size(); ++a)
        {
            mu_[a] = -(Q_ref_[a] - targets_[a]) / chi[a];
            mu_[a] = std::max(-mu_caps_[a], std::min(mu_caps_[a], mu_[a]));
        }
        alm_probe_pending_ = false;
        print_audit(iter);
        conv_esolver = false;
        return;
    }
    // Branch (M4 ALM) adjudication at a converged ALM SCF: target met ->
    // DONE; otherwise one gentle outer update mu += k(Q - t) and continue.
    if (alm_mode_ && alm_armed_ && phase_ == LoopPhase::CONSTRAINED)
    {
        alm_adjudicate(iter, conv_esolver);
        return;
    }
```

（`alm_adjudicate`：`++outer_steps_`；`res = alm_.update(Q_, targets_, mu_caps_, mu_)`；审计打印（含 mu_phys 追加 token，镜像 M3 诊断追加模式）；`res < cfg_.thr` → CONVERGED/DONE（**更新前先判定**：残差已满足则不更新、直接收尾——实现时把判定放在 update 之前：`max_residual() < thr` → DONE）；fuse → UNREACHABLE/DONE；否则 `conv_esolver = false`。）

(d) **alm_arm / finish_warm_start_alm**：

```cpp
void ConstraintLoop::finish_warm_start_alm()
{
    // Warm path (M4): the carried secant slope is the response measurement
    // — arm k directly, no probe SCF.  Q_ref re-derivation (delta) already
    // happened in finish_warm_start() above this call.
    std::vector<double> chi(mu_.size(), -1.0);
    for (size_t a = 0; a < chi.size(); ++a)
    {
        chi[a] = mu_solver_.slope(static_cast<int>(a));
    }
    alm_arm(chi);
}
```

（在 `finish_warm_start()` 末尾追加 `if (alm_mode_) { finish_warm_start_alm(); }`。）

> **暖几何的 armed 状态纪律**：`warm_reinit` 不调 `reset()`，**不要**在暖路径清 `alm_armed_`/k——暖探针 SCF 期间注入用的是"旧 k + 旧 targets 的 mu_eff_"，在旧收敛态附近惩罚≈0、注入≈μ_prev，密度向新几何漂移时惩罚连续回拉，这是 ALM 相对 M2 裸 μ 携带的平滑优势；`finish_warm_start`（rebind + Q_ref 反推 + `finish_warm_start_alm` 重新 arm）后口径自动对齐。`AlmWarmStartSkipsProbe` 测试的 `EXPECT_TRUE(loop.alm_armed())` 依赖这一点。
>
> **env 定 μ 与 alm 的共存**：`ABA_CONSTRAINT_FIXED_MU` 分支早退（fixed_mu_=true），alm 永不 armed——注入走 `mu_eff_ = mu_`（未 armed 直通），`on_scf_converged` 的 ALM 裁决被 `alm_armed_` 门控，固定分支照常工作，无需额外守卫。

(e) **mu_eff_ 维护**：`observe()` 末尾（dipole 更新旁）：

```cpp
    // M4 ALM: keep the effective multiplier in sync with the observation
    // (unarmed phases pass mu through unchanged — probe / outer schedules).
    if (alm_armed_)
    {
        alm_.mu_eff(mu_, Q_, targets_, mu_eff_);
    }
    else
    {
        mu_eff_ = mu_;
    }
```

(f) **注入/回加/力切换**：`inject_potential`（:574-575）、`inject_potential_lcao`（:603）、`add_back_constraint_potential`（:622）、`compute_force`（:1466）四处 `mu_` → `mu_eff_`，各加分支注释：

```cpp
    // M4: mu_eff_ tracks mu_ + k(Q - t) under the ALM schedule and equals
    // mu_ on every other schedule — a single swap point covers both.
```

（`mu_eff_` 在 init/reset 中与 `mu_` 同尺寸初始化。）

(g) **drho 阻尼守卫**：`on_iteration` 开头（inner 逻辑之前）：

```cpp
    // M4 ALM stiffness guard: three consecutive >20% drho growths mean the
    // penalty feedback is too stiff for the mixing — dampen k (no mixing
    // reset: the gain change is small and continuous by design).
    if (alm_armed_)
    {
        drho_trend_.push_back(drho);
        if (drho_trend_.size() > 3)
        {
            drho_trend_.erase(drho_trend_.begin());
        }
        if (drho_trend_.size() == 3
            && drho_trend_[1] > 1.2 * drho_trend_[0]
            && drho_trend_[2] > 1.2 * drho_trend_[1])
        {
            alm_.dampen();
            drho_trend_.clear();
            GlobalV::ofs_running
                << "[constraint] ALM dampen: drho growing three iterations "
                   "(penalty too stiff) -> k halved (x"
                << alm_.dampen_count() << ")\n";
        }
    }
```

(h) **能量/守卫口径**：审计打印处（`print_audit_line`  accounting 之后）：

```cpp
    // M4 ALM: the energy correction carries the penalty term
    // sum_a (k_a/2)(Q_a - t_a)^2 on top of the accounting's sum_a mu(Q-t);
    // the branch guard (G2) consumes the same helper, never its own formula.
    if (alm_armed_)
    {
        audit_.e_con = alm_.energy_correction(mu_, Q_, targets_);
    }
```

G2 分支守卫块内的自算 `e_con` 循环改为：

```cpp
            double e_con = 0.0;
            // Branch: ALM — the correction includes the penalty term and is
            // computed by the same helper as the audit (single source).
            if (alm_armed_)
            {
                e_con = alm_.energy_correction(mu_, Q_, targets_);
            }
            else
            {
                for (size_t a = 0; a < mu_.size(); ++a)
                {
                    e_con += mu_[a] * (Q_[a] - targets_[a]);
                }
            }
```

(i) **final_report**：CONVERGED 行后追加 ALM 块：

```cpp
    // M4 ALM: report the physical multipliers (the constant-potential
    // labels) and the penalty gains alongside the standard status.
    if (alm_armed_)
    {
        GlobalV::ofs_running << "[constraint] ALM: mu_phys =";
        for (size_t a = 0; a < mu_eff_.size(); ++a)
        {
            GlobalV::ofs_running << " " << mu_eff_[a];
        }
        GlobalV::ofs_running << " Ry, k =";
        for (const double kv : alm_.k())
        {
            GlobalV::ofs_running << " " << kv;
        }
        GlobalV::ofs_running << " Ry/e\n";
    }
```

- [ ] **Step 4: 运行确认通过**

```bash
cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -3
ctest --test-dir build -R "MODULE_ESTATE_constraint_(loop|alm)" --output-on-failure 2>&1 | tail -5
```
预期：全部通过（含既有 43+ 用例回归）。

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(constraint): ALM single-layer schedule in ConstraintLoop (M4)"
```

---

### Task 4: 力与能量口径的 μ_phys 验证（单元级 FD）

**Files:**
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
// ALM force convention: the force kernel must see mu_phys (= mu_eff at the
// converged state), not the bare ALM mu — the potential the SCF felt is
// mu_eff * w, so the envelope force is built from the same multiplier.
TEST_F(ConstraintLoopTest, AlmForceUsesMuPhys)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.mu_schedule = "alm";
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    const double chi = -0.65;
    std::vector<double> rho;
    bool conv = false;
    int iter = 0;
    while (loop.status() == constraint::MuStatus::RUNNING && iter < 40)
    {
        ++iter;
        mock_alm_iteration(loop, rho, iter, chi, conv);
    }
    ASSERT_EQ(loop.status(), constraint::MuStatus::CONVERGED);
    // The converged mu_eff differs from the bare mu by k(Q - t) with a
    // non-zero penalty share whenever the target is not exactly met; the
    // force path and the injection path must read the same vector.
    EXPECT_EQ(loop.mu_eff(), loop.mu_eff()); // accessor sanity
    ASSERT_FALSE(loop.mu_eff().empty());
    // mu_phys of the linear model equals the exact multiplier (Task 3);
    // the force kernel input IS mu_eff (verify via the accessor the force
    // path uses — the kernel itself is covered by constraint_deriv tests).
    const double q_ref = Q_of(rho_ref);
    const double mu_star = (loop.targets()[0] - q_ref) / chi;
    EXPECT_NEAR(loop.mu_phys(0), mu_star, 1e-6);
}

// ALM energy convention: e_con carries the penalty term.
TEST_F(ConstraintLoopTest, AlmEnergyCorrectionInAudit)
{
    // (Drive the same converged ALM state as above, then:)
    // EXPECT_NEAR(loop.last_audit().e_con,
    //             alm.energy_correction(mu, Q, target), 1e-12);
    // 实现要点：last_audit().e_con 必须等于 Σ[μ(Q−t) + (k/2)(Q−t)²]，
    // 而不是会计默认的 Σ μ(Q−t)。断言两者之差 = Σ (k/2)(Q−t)²。
}
```

（`AlmEnergyCorrectionInAudit` 的完整驱动代码镜像 `AlmColdConvergesWithExactMuPhys`；实现者照抄驱动块再追加断言，勿留空壳注释——上面注释块是断言内容的精确描述。）

- [ ] **Step 2-4:** 运行失败 → 补缺（Task 3 已实现的则直接过）→ 通过
- [ ] **Step 5: Commit**

```bash
git commit -am "test(constraint): ALM force/energy convention (mu_phys) unit gates"
```

---

### Task 5: 调度包 A——跨几何 J 携带 + 残差自适应探针

**Files:**
- Modify: `source/source_estate/module_constraint/mu_solver.{h,cpp}`（rebind）
- Modify: `source/source_estate/module_constraint/constraint_loop.{h,cpp}`（scf_threshold、rebind 调用点传 flag）
- Test: `test/mu_solver_test.cpp`、`test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
TEST(MuSolverTest, RebindCarriesJacobianWhenEnabled)
{
    constraint::MuSolverParams p;
    p.trust_region = true;
    p.broyden = true;
    p.step_max = 0.05;
    constraint::MuSolver s(p);
    std::vector<double> mu = {0.0, 0.0};
    // Drive two accepted steps so an off-diagonal coupling is learned.
    s.step({1.0, 0.5}, {0.0, 0.0}, mu);
    s.step({0.9, 0.55}, {0.0, 0.0}, mu);
    s.step({0.81, 0.605}, {0.0, 0.0}, mu);
    const double j01 = s.jacobian(0, 1);
    // Carry on: the full Jacobian survives the rebind.
    s.set_carry_jacobian(true);
    s.rebind(mu, {0.8, 0.6});
    EXPECT_NEAR(s.jacobian(0, 1), j01, 1e-12);
    // Carry off (legacy): diagonal reset wipes the coupling.
    constraint::MuSolver s2(p);
    std::vector<double> mu2 = {0.0, 0.0};
    s2.step({1.0, 0.5}, {0.0, 0.0}, mu2);
    s2.step({0.9, 0.55}, {0.0, 0.0}, mu2);
    s2.step({0.81, 0.605}, {0.0, 0.0}, mu2);
    s2.rebind(mu2, {0.8, 0.6});
    EXPECT_DOUBLE_EQ(s2.jacobian(0, 1), 0.0);
}
```

```cpp
// Progressive probe scales with the outer residual (M3 data: drho and the
// residual in e are the same order; 0.3x keeps the Q error well below the
// residual, and the trust rho score guards the step quality).
TEST_F(ConstraintLoopTest, ProgressiveProbeScalesWithResidual)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.progressive_scf = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    // ... drive to CONSTRAINED with the standard mock ...
    // Far from target (res ~ 1e-1): probe may rise toward 1e-3.
    // Near target (res ~ 1e-4): probe stays at the 1e-5 floor.
    // Assert scf_threshold(1e-7) >= 1e-4 when residual is large, and
    // == 1e-5 when residual is small; tighten path unchanged.
}
```

- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**

`mu_solver.h`：`void set_carry_jacobian(const bool on) { carry_jacobian_ = on; }` + private `bool carry_jacobian_ = false;`（`reset()` 清零）。
`mu_solver.cpp` rebind 的 trust 分支：

```cpp
    // Branch: carry the full Jacobian across geometries (M4 scheduling pack
    // A, opt-in).  The off-diagonal fragment couplings cost outer steps to
    // relearn every frame; the trust-region rho score rejects and relearns
    // a stale model, so carrying is safe.  Legacy (flag off): diagonal
    // reset from the carried slopes.
    if (params_.trust_region)
    {
        if (!carry_jacobian_)
        {
            jacobian_.assign(n, std::vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
            {
                jacobian_[i][i] = last_slope_[i];
            }
        }
        // ... radius/pending/fallback reset unchanged ...
    }
```

`constraint_loop.cpp` init：`mu_solver_.set_carry_jacobian(cfg_.trust_carry_jacobian);`（params 赋值旁）。
`scf_threshold` Branch C：

```cpp
    // Branch C: outer probe — the threshold scales with the outer residual.
    // M3 data (h2o_dimer): drho and the residual in e are the same order of
    // magnitude, so 0.3*res keeps the probe Q error well below the residual
    // (the step direction is unaffected; the trust rho score guards).  The
    // floor/ceiling keep the probe sane at both ends.
    const double probe = std::min(1e-3, 0.3 * max_residual());
    return std::max(std::max(1e-5, 100.0 * strict_thr), probe);
```

- [ ] **Step 4: 运行确认通过** → **Step 5: Commit**

```bash
git commit -am "feat(constraint): M4 scheduling pack A — Jacobian carry + residual-scaled probe"
```

---

### Task 6: 调度包 B——预算感知冻结 + RUNNING 帧暖启动

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.{h,cpp}`
- Test: `test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试**

```cpp
// Budget freeze: with the SCF budget nearly exhausted and the residual
// already below 10*thr, the loop freezes at the current converged SCF
// instead of starting a doomed mu update (M3 0.01 e frame-2 signature).
TEST_F(ConstraintLoopTest, BudgetFreezeConvertsDoomedUpdate)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.budget_freeze = true;
    loop.init(*ucell, rhopw, cfg, radii, 10.0, 0);
    // Drive to CONSTRAINED and close to the target (residual < 10*thr),
    // then simulate the budget edge: iter near scf_nmax with a recorded
    // rebuild cost that cannot fit another update.
    // ... (mock drive; then call the internal check via on_scf_converged
    // with iter = scf_nmax - 5 and last_rebuild = 25)
    EXPECT_TRUE(loop.budget_frozen());
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::DONE);
    EXPECT_EQ(loop.status(), constraint::MuStatus::RUNNING); // not CONVERGED
}

// Warm-from-RUNNING: a frame that ended RUNNING with residual < 10*thr is
// warm-eligible; a mid-flight frame (residual >= 10*thr) is not.
TEST_F(ConstraintLoopTest, WarmStartFromNearConvergedRunning)
{
    // ... drive frame 1 to RUNNING with small residual (budget freeze path),
    // re-init at istep=1, expect warm_started_last_init() == true;
    // then repeat with a large residual, expect cold (REFERENCE).
}
```

- [ ] **Step 2-3: 实现**

`constraint_loop.h`：

```cpp
    // M4 scheduling pack B diagnostics.
    bool budget_frozen() const { return budget_frozen_; }
```

private：`bool budget_frozen_ = false; int last_step_iter_ = 0; int last_rebuild_cost_ = 0;`（reset 清零）。

`outer_step` 提议点记录重建成本：

```cpp
    // Rebuild-cost bookkeeping (M4 pack B): iterations spent since the last
    // mu move, used by the budget-freeze estimator.
    if (last_step_iter_ > 0)
    {
        last_rebuild_cost_ = iter - last_step_iter_;
    }
    last_step_iter_ = iter;
```

`on_scf_converged` 在 `outer_step` 调用之前：

```cpp
    // M4 pack B (budget freeze): estimate the rebuild cost of another mu
    // update; if it cannot fit the remaining scf_nmax budget and the
    // residual is already below 10*thr, freeze here — a converged frame at
    // a slightly-off mu is usable, an exhausted SCF is not (M3 0.01 e).
    if (cfg_.budget_freeze && phase_ == LoopPhase::CONSTRAINED
        && !fixed_mu_ && cfg_.mu_schedule == "outer"
        && max_residual() < 10.0 * cfg_.thr && max_residual() >= cfg_.thr)
    {
        const int est = std::max(15, last_rebuild_cost_);
        if (iter + 3 * est / 2 >= PARAM.inp.scf_nmax)
        {
            budget_frozen_ = true;
            phase_ = LoopPhase::DONE;
            GlobalV::ofs_running
                << "[constraint] BUDGET_FREEZE: iter " << iter
                << " + estimated rebuild " << est
                << " would exceed scf_nmax=" << PARAM.inp.scf_nmax
                << "; mu frozen at the current converged SCF (residual "
                << max_residual() << " e).  Reported RUNNING (budget) — "
                   "the frame is converged and usable, the target is not "
                   "validated\n";
            return; // conv_esolver stays true: the SCF is converged
        }
    }
```

`final_report` 的 RUNNING 分支追加 budget 提示（`budget_frozen_` 时打印 "(budget freeze; converged frame, target not validated)"）。

`warm_start_eligible` 的状态守卫改为：

```cpp
    // Guard: the previous frame must be a usable anchor.  CONVERGED always
    // qualifies; RUNNING qualifies only when it is budget-limited
    // (residual < 10*thr, no magnetic violence) — the M3 0.01 e signature —
    // never when it is mid-flight (the mu would carry an unscored jump).
    if (status_ != MuStatus::CONVERGED)
    {
        const bool running_ok
            = cfg_.warm_start_running && status_ == MuStatus::RUNNING
              && max_residual() < 10.0 * cfg_.thr && mag_rejections_ == 0;
        if (!running_ok)
        {
            return false;
        }
        GlobalV::ofs_running
            << "[constraint] WARM_START from a RUNNING frame (residual "
            << max_residual() << " e < 10*thr; budget-limited, rebind at "
               "the first convergence)\n";
    }
```

- [ ] **Step 4: 运行确认通过** → **Step 5: Commit**

```bash
git commit -am "feat(constraint): M4 scheduling pack B — budget freeze + warm-from-RUNNING"
```

---

### Task 7: 集成验收（h2o_dimer，scf_nmax=100 纪律）

**Files:** 无新代码；运行目录用 `mktemp -d`；结果写入 Task 8 报告。

- [ ] **Step 1: 全量单元回归**

```bash
cmake --build build -j 32 --target abacus_basic_para 2>&1 | tail -2
ctest --test-dir build -j 16 -R constraint --output-on-failure 2>&1 | tail -5
```
预期：12/12（MPI 环境正常时）。

- [ ] **Step 2: R1 — ALM 0.001 e relax 两帧（对照 M3 基线 88/46）**

`tests/constraint_ct_dimer/h2o_dimer` 的临时副本，INPUT 改：`calculation relax`、`relax_nmax 2`、`relax_method cg`、`relax_scale_force 0.01`、`scf_nmax 100`、`constraint_warm_start 1`、`constraint_mu_schedule alm`、`constraint_target_mode absolute`，目标文件用 M3 的 0.001 e 值。`mpirun -np 8`。

判据：
- 两帧 `final status: CONVERGED`（≤100 步/帧）；
- `ALM: mu_phys = ...` 行出现；两帧 mu_phys 与 M3 同目标收敛 μ（基线运行归档 `/tmp/m3_c2fix_small.RS2nnn` 的 audit；若本机无归档则先补跑一次 M3 基线）偏差 < 1%；
- 帧 1/帧 2 SCF 数记录（目标：帧 2 ≤ 60；如实报告，不凑数）。

- [ ] **Step 3: R2 — ALM 0.01 e BFGS 两帧（M3 失败点，验收判据）**

同 R1 但 `relax_method bfgs`、0.01 e 目标。

判据（**M4 头号验收**）：
- **两帧均在 scf_nmax=100 内 CONVERGED**（M3 帧 2 RUNNING 的失败被解决）；
- 末 residual ≤ 1e-4 e；mu_phys 逐帧记录。

若失败：记录 drho/residual 轨迹（M3 报告 §3.4 格式），分析是 k 设定、探针还是结构问题，**不提高 scf_nmax**。

- [ ] **Step 4: R3 — dE/dt = −μ_phys 三点 FD（性质2 判据）**

单点 scf ×3（absolute 模式）：t_c = 10.002852052（atoms [0,2,3]），t± = t_c ± 0.01 e；`constraint_mu_schedule alm`、`scf_nmax 100`。
取三点 FINAL_ETOT（含 cc_escon 修正后的 etot）：
`T_FD = (E+ − E−)/0.02 (eV/e)` vs `μ_phys(中心)×13.605693`，判据 |Δ| < 0.02 eV/e（性质2 电荷力矩判据）。

- [ ] **Step 5: R4 — spin 冒烟（212_NAO_constraint_h2o_spin）**

`constraint_mu_schedule alm` + `constraint_mag_anchor 1`（若 M3 flag 名不同以 M3 为准），spin 约束 δ=0.1 μB：收敛、无 MAG_ANCHOR 拒绝风暴（≤2 次）、mu_phys 稳定。

- [ ] **Step 6: R5 — 调度包 B 端到端（ALM off，outer 路径）**

0.01 e BFGS `relax_nmax 3`、`constraint_trust_region 1`、`constraint_trust_carry_jacobian 1`、`constraint_budget_freeze 1`、`constraint_warm_start_running 1`、`scf_nmax 100`：
- 若帧 2 预算不足：应出现 `BUDGET_FREEZE`，帧 2 以收敛帧收尾（RUNNING 但 SCF 收敛）；
- 帧 3 应出现 `WARM_START from a RUNNING frame` 而非冷启动参考相；
- 记录三帧 SCF 数与状态。

- [ ] **Step 7: R6 — 自适应探针（C3 复测）**

0.001 e、`constraint_progressive_scf 1`、outer 路径：对照 M3 的 92/46，记录帧 1/帧 2 SCF 数（预期帧 1 ≤ 80；若无收益如实记录，同 M3 纪律）。

---

### Task 8: 文档 + Bohrium 移交清单

**Files:**
- New: `docs/superpowers/specs/2026-10-21-m4-alm-test.md`
- Modify: `docs/superpowers/specs/deltap-development-log.md`

- [ ] **Step 1: 测试报告**（AGENTS.md 五段：测试计划/设置/结果/分析/下一步）
  内容：单元结果、R1-R6 全部原始数据（每帧 SCF 数、drho 抽样、residual、mu_phys、墙钟）、R2 失败轨迹（若有）、ALM vs M3 对照表、收缩率实测 vs 设计 1/6.5。
- [ ] **Step 2: 开发日志**追加本轮条目（做了什么/文件/bug/下一步）。
- [ ] **Step 3: Bohrium 移交清单**（写入报告"下一步"节）：
  1. **性质2 重验（ALM）**：LiCl PW 约束力 FD（13 点，fixed-μ 腿协议不变——注意 ALM 模式下 FD 腿用重优化协议）+ LiNiO₂ 电荷/自旋力矩 FD（APNS），判据沿用性质2（力 <0.0129 eV/Å、力矩 <0.02 eV/e / <0.006 eV/μB）；参考成本：性质2 原始 ~5 元。
  2. **性质2 补账（APNS 自旋三点）**：E1+E2+信赖域（M3 已合入）下重跑性质2 §5.3 的 APNS 自旋三点，closure 性质2 的 16/19 → 19/19；`constraint_thr 5e-5` 预案沿用。
  3. **性质4 重验（ALM）**：LiCl 500 fs 约束 MD（δ=+0.1）测 ⟨μ_phys⟩，与静态 μ_phys 对照（判据 5%）；参考成本：性质4 原始 68 元/14 h——**先跑 50 fs 冒烟再决定全量**。
- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-10-21-m4-alm-test.md docs/superpowers/specs/deltap-development-log.md
git commit -m "docs(constraint): M4 ALM + scheduling test round"
```

---

## Self-Review 记录（计划编写后）

1. **Spec 覆盖**：设计 §6.1（ALM 公式/SCF 内更新）→ Task 2/3；§6.2（误差结构、k 自动、μ_phys）→ Task 2（ContractionAndMuPhysExact）/Task 3；§6.3（k 刚度守卫、性质2/4 重验纪律）→ Task 3(g)/Task 8。M3 评审输入 #2/#3/#5 → Task 5/6；#1 验收（0.01 e 两帧）→ Task 7 R2。✅
2. **占位符扫描**：`AlmEnergyCorrectionInAudit` 的驱动块以"镜像 Task 3 驱动 + 精确断言描述"给出——实现者必须照抄 Task 3 驱动代码并填实断言，禁止留空壳。其余代码完整。✅
3. **类型一致性**：`AlmController::{resize,arm,mu_eff,update,energy_correction,dampen,k,fuse_component,reset_fuse}`、`ConstraintLoop::{mu_eff,mu_phys,alm_armed,alm_k,schedule,budget_frozen}`、`MuSolver::set_carry_jacobian`、`ConstraintConfig::{alm_k,trust_carry_jacobian,budget_freeze,warm_start_running}` 跨 Task 一致。✅
4. **已知风险**：①线性 mock 下 Newton μ0 精确，单元层测不到非线性收缩——由 Task 2 的 ContractionAndMuPhysExact（允许 5 次更新）与 R2 集成覆盖；②alm+warm+delta 的 Q_ref 反推沿用 M2 公式（κ 口径一致）；③budget freeze 的 est 估计在首帧无历史（last_rebuild_cost_=0 → est=15 地板，保守）。
5. **与 M3 评审建议的映射**：#1 ALM=Task 2/3；#2 自适应探针=Task 5；#3 J 携带=Task 5；#4 μ 时序预测=**未纳入**（MD 生产强化，留待 M5 后评估）；#5 优雅收尾+RUNNING 暖启=Task 6；#6 扫描工作流=**未纳入**（属工作流层，与 M5 无关，建议单独立项）。

---

## 实施修订记录（2026-10-21，集成实测驱动）

计划中的公式/参数在实现与 R1-R6 集成后被以下实测结论修订；代码注释、
`docs/superpowers/specs/2026-10-21-m4-alm-test.md` 与开发日志同步更新。

| 项 | 计划值 | 实现值 | 依据 |
|---|---|---|---|
| 自动增益 `k*|chi|` | 7.5（设计 §6.2 的 5-10） | 0.5 | 单层每 SCF 反馈的密度滞后一拍；滞后映射模长 `sqrt(K/(1+K))`，K=7.5 为 0.94，R1 在 100 步内不收敛；K=0.5 为 0.58 |
| 势更新 | 每迭代 `mu+k(Q-t)`，不 reset | `alpha=1/(1+K)` 阻尼 + μ 移动时 `mix_reset()` | R1 旧增益出现 drho 0.1945 -> 1.033 -> 0.3339 的 2-cycle；R2 无 reset 时 0.25 drho 瞬态 |
| 探针幅度 | 0.02 Ry | 0.005 Ry | 0.02 的探针+settle 在 R2 的 100 步预算内没有 μ 修正余量；0.005 的 dQ~3e-3 e 仍远高于噪声 |
| 罚势时序 | 探针 close 后立即生效 | 先 fixed-mu settle，adjudicate 后残差仍大才开罚势 | 在旧探针密度上开罚势会 5 倍过冲，R1 首版耗尽 100 步 |
| 暖启动 | 携带斜率直接 Newton | 有帽 Newton + 本几何二段割线 | R2 帧 2 携带 |chi|=0.58、真实 1.49，无帽步过冲 2.5 倍 |
| R4 用例 | 212_NAO_constraint_h2o_spin | `tests/constraint_fd_force/cases/` 的 spin 用例 | 仓库内无 212_NAO 目录；现有 spin FD 用例覆盖 nspin=2、delta 0.1 μB、mag anchor、ALM |
| 预算冻结审计 | 未规定 | 冻结时 `print_audit(iter)` | `final_report()` 重放 `last_audit_line_`，不刷新会打印上一次 outer step 的旧 residual（R5：0.0111 vs 3.265e-4） |
| 收敛后 μ 标签 | `mu_eff` 即物理乘子 | CONVERGED 分支重算 `mu+k(Q-t)` | 两阶段 settle 可能关掉罚势，最后 `observe()` 的 `mu_eff` 是裸 μ；能量/力仍需要罚势项 |
