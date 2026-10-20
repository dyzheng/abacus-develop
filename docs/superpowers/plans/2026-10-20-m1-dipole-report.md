# M1 偶极报告口径（Dipole Reporting）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在约束审计输出中增加偶极观测量（分子偶极 p、偶极导出电荷转移 q_dip、周期 saw 口径的电势降 ΔV），与 Becke 片段电荷并列，解决"Becke 电荷是记账数、无物理意义"的读数问题（性质1 §5.3）。

**Architecture:** 新增 `DipoleObserver`（纯观测类，不进入任何循环逻辑），挂在 `ConstraintLoop::observe()`（每次 SCF 迭代已被两个 esolver 调用）与 `final_report()`（每离子步已被调用）上——**不需要改任何 esolver 文件**。p_ref 在 REFERENCE→CONSTRAINED 转移点随 Q_ref 同点快照（constraint_loop.cpp `Q_ref_ = Q_` 处）。所有功能由 `constraint_dipole_report`（默认 false）门控，默认路径输出逐字节不变。

**Tech Stack:** C++17、gtest、CMake（AddTest 宏）、ABACUS module_constraint 现有分层。

**Spec:** `docs/superpowers/specs/2026-10-20-constraint-improvement-design.md`（§3 模块 A）。

**计划序列说明：** 本计划只覆盖里程碑 M1。M2（C1 热启动 + C4 定 μ 产品化）、M3（C2 信赖域 + C3 渐进 SCF + E 鲁棒性）、M4（D ALM 单层循环）、M5（B 偶极约束坐标）各自独立成计划，在前序里程碑合入后以其时点的代码为准另行编写——后续任务级细节依赖前序落地结果（如 M4 依赖 M3 加固后的求解器）。

**代码规范（AGENTS.md，必须遵守）：**
- 每个 if-else 分支与非平凡 if 块前必须有注释（说明分支做什么、为什么）
- 禁 goto / setjmp / longjmp；函数 >300 行拆分；优先 RAII / std::vector

---

## 文件结构

| 文件 | 职责 |
|---|---|
| 新建 `source/source_estate/module_constraint/constraint_dipole.h/.cpp` | DipoleObserver：分子偶极、saw 偶极、ΔV、q_dip、片段离子中心、自动选面、跳变面守卫 |
| 新建 `source/source_estate/module_constraint/test/constraint_dipole_test.cpp` | DipoleObserver 单元测试（玩具密度解析对拍） |
| 改 `source/source_io/module_parameter/input_parameter.h:766`（constraint_inner_nmax 后） | 4 个新 INPUT 字段 |
| 改 `source/source_io/module_parameter/read_input_item_other.cpp`（constraint_inner_nmax item 块后） | 4 个 Input_Item |
| 改 `source/source_estate/module_constraint/constraint_io.h`（ConstraintConfig） | dipole_report/dipole_frame/dipole_saw_pos_max/dipole_saw_pos_dec 字段 |
| 改 `source/source_estate/module_constraint/constraint_io.cpp`（configure_from_inputs 尾部） | PARAM→cfg 管道 + 守卫 |
| 改 `source/source_estate/module_constraint/constraint_loop.h/.cpp` | 报告器状态、observe 钩子、p_ref 快照、final_report 偶板块 |
| 改 `source/source_estate/module_constraint/test/constraint_io_test.cpp` | 管道与守卫测试 |
| 改 `source/source_estate/module_constraint/test/constraint_loop_test.cpp` | loop 级偶极测试 |
| 改 `source/source_estate/CMakeLists.txt:54` 后 | 新源文件入 estare 库 |
| 改 `source/source_estate/module_constraint/test/CMakeLists.txt` | 新 AddTest |

**关键既有事实（实现时直接引用，不要重新发现）：**
- 网格索引模式（镜像 `efield.cpp:179-198` 与 `constraint_loop_test.cpp:66-77`）：
  `i = ir/(ny*nplane); j = ir/nplane − i*ny; k = ir%nplane + startz_current`；分数坐标 `(i/nx, j/ny, k/nz)`；笛卡尔 = `rfrac * ucell.latvec * ucell.lat0`；`dV = omega/nxyz`。
- saw 语义（镜像 `efield.cpp:234-251`，x∈[0,1]，a=pos_max，b=pos_dec）：`fac=1−b`；`x≤a: x−a+0.5·fac`；`x>a+b: x−a−1+0.5*fac`；否则 `0.5·fac − fac·(x−a)/b`。上升沿斜率 1，下降沿在 `[a, a+b]`（跳变面区）。
- 常量：`ModuleBase::FOUR_PI`、`ModuleBase::Hartree_to_eV`(=27.211396)、`ModuleBase::BOHR_TO_A`、`ModuleBase::PI`（`source_base/constants.h`）。
- 离子价电荷：`ucell.atoms[it].ncpp.zv`；坐标：`ucell.atoms[it].taud[ia]`（分数）× latvec × lat0。
- 并行归约：`Parallel_Reduce::reduce_pool(scalar)` 与 `reduce_pool(double*, int)`（serial 下 no-op）。
- Q_ref 快照点：`constraint_loop.cpp:322`（`Q_ref_ = Q_;`，REFERENCE→CONSTRAINED 转移内）。
- fixed_mu 早退分支：`constraint_loop.cpp` init 内（`ABA_CONSTRAINT_FIXED_MU` 分支 `return`）——偶极 init 代码必须放在该分支**之前**。
- 现有 loop 测试驱动：`mock_iteration(loop, rho, iter, mu_obs, drho, conv_out)`（constraint_loop_test.cpp:237-258）。

---

### Task 1: INPUT 参数与 ConstraintConfig 管道

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`（constraint 参数区尾部，line 766 `constraint_inner_nmax` 之后）
- Modify: `source/source_io/module_parameter/read_input_item_other.cpp`（constraint_inner_nmax item 块之后）
- Modify: `source/source_estate/module_constraint/constraint_io.h`（ConstraintConfig 尾部）
- Modify: `source/source_estate/module_constraint/constraint_io.cpp`（configure_from_inputs 尾部，`return st;` 之前）
- Test: `source/source_estate/module_constraint/test/constraint_io_test.cpp`

- [ ] **Step 1: 写失败测试**

在 `constraint_io_test.cpp` 末尾追加：

```cpp
TEST(ConstraintIOTest, DipoleReportPlumbing)
{
    auto ucell = make_h2o_ucell(); // O, H, H -> 3 atoms
    Set_GlobalV_Default();
    PARAM.input.constraint = true;
    PARAM.input.constraint_type = "charge";
    PARAM.input.constraint_weight_type = "becke";
    PARAM.input.constraint_target_mode = "delta";
    PARAM.input.constraint_mu_max = 5.0;
    PARAM.input.constraint_thr = 1e-4;
    const std::string path = "constraint_target_dipole_test.json";
    {
        std::ofstream ofs(path);
        ofs << R"({"targets": [0.1], "atoms": [[0]]})";
    }
    PARAM.input.constraint_target_file = path;

    constraint::ConstraintConfig cfg;
    std::vector<constraint::ConstraintSpec> specs;
    std::vector<double> radii;
    std::string error;

    // Default off: the flag is false and the frames carry their defaults.
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_FALSE(cfg.dipole_report);
    EXPECT_EQ(cfg.dipole_frame, "molecular");
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_max, -1.0);
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_dec, 0.1);

    // Enabled with saw frame: values flow through.
    PARAM.input.constraint_dipole_report = true;
    PARAM.input.constraint_dipole_frame = "saw";
    PARAM.input.constraint_dipole_saw_pos_max = 0.45;
    PARAM.input.constraint_dipole_saw_pos_dec = 0.08;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::OK)
        << error;
    EXPECT_TRUE(cfg.dipole_report);
    EXPECT_EQ(cfg.dipole_frame, "saw");
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_max, 0.45);
    EXPECT_DOUBLE_EQ(cfg.dipole_saw_pos_dec, 0.08);

    // Guard: unknown frame -> ERROR.
    PARAM.input.constraint_dipole_frame = "circular";
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("dipole_frame"), std::string::npos);

    // Guard: saw decay width out of (0, 0.5] -> ERROR.
    PARAM.input.constraint_dipole_frame = "saw";
    PARAM.input.constraint_dipole_saw_pos_dec = 0.6;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_dipole_saw_pos_dec = 0.1;

    // Guard: saw pos_max outside [-1, 1) -> ERROR.
    PARAM.input.constraint_dipole_saw_pos_max = 1.5;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    PARAM.input.constraint_dipole_saw_pos_max = -1.0;

    // Guard: non-collinear spin not implemented -> ERROR.
    PARAM.input.nspin = 4;
    EXPECT_EQ(constraint::configure_from_inputs(cfg, specs, *ucell, radii,
                                                error),
              constraint::ConfigStatus::ERROR);
    EXPECT_NE(error.find("nspin"), std::string::npos);
    PARAM.input.nspin = 1;
    PARAM.input.constraint_dipole_report = false;
    PARAM.input.constraint_dipole_frame = "molecular";
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -5
```

预期：编译错误（`ConstraintConfig` 无 `dipole_report` 成员 / `PARAM.input` 无 `constraint_dipole_report`）。

- [ ] **Step 3: 实现参数声明、读取与管道**

`input_parameter.h`（`int constraint_inner_nmax = 20;` 行之后）：

```cpp
    // Module A (M1, plan 2026-10-20): passive dipole reporting alongside the
    // Becke fragment charges.  Pure observation; never enters the loop logic.
    bool constraint_dipole_report = false;   ///< print dipole / q_dip / deltaV block
    std::string constraint_dipole_frame = "molecular";  ///< "molecular" | "saw"
    double constraint_dipole_saw_pos_max = -1.0;  ///< saw maximum (fractional z); -1 = auto lowest-density plane
    double constraint_dipole_saw_pos_dec = 0.1;   ///< saw decay width (fractional z)
```

`read_input_item_other.cpp`（`constraint_inner_nmax` 的 item 块之后——搜索 `constraint_inner_nmax` 定位，该文件内 constraint 系列 item 连续排布）：

```cpp
    {
        Input_Item item("constraint_dipole_report");
        item.annotation = "report the molecular / saw-frame dipole alongside the fragment charges";
        item.category = "Constraint";
        item.type = "Boolean";
        item.description = "When true, the final constraint report adds a dipole block: the molecular-frame dipole p (e*Bohr and Debye), the dipole-derived charge transfer q_dip (e), and (frame=saw) the potential drop deltaV = 4*pi*p_saw/A (V). Pure observation; the constraint dynamics is unchanged. Default false keeps the historical output byte-identical.";
        item.default_value = "False";
        item.unit = "";
        item.availability = "constraint is true";
        read_sync_bool(input.constraint_dipole_report);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_dipole_frame");
        item.annotation = "dipole frame: molecular or saw";
        item.category = "Constraint";
        item.type = "String";
        item.description = "molecular (default): p = sum Z*R - int rho*r dr about the ionic charge center (origin-independent for neutral cells). saw: saw-function-referenced p_z for periodic slabs (the decay plane must sit in a low-density region; deltaV is reported). Only read when constraint_dipole_report is true.";
        item.default_value = "molecular";
        item.unit = "";
        item.availability = "constraint_dipole_report is true";
        read_sync_string(input.constraint_dipole_frame);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_dipole_saw_pos_max");
        item.annotation = "saw maximum position (fractional z)";
        item.category = "Constraint";
        item.type = "Real";
        item.description = "Fractional z of the saw maximum; the decay (jump) plane occupies [pos_max, pos_max+pos_dec] and must lie in a low-density region. -1 (default) auto-places it at the lowest plane-averaged density plane, resolved on the first SCF iteration of each geometry.";
        item.default_value = "-1.0";
        item.unit = "";
        item.availability = "constraint_dipole_frame is saw";
        read_sync_double(input.constraint_dipole_saw_pos_max);
        this->add_item(item);
    }
    {
        Input_Item item("constraint_dipole_saw_pos_dec");
        item.annotation = "saw decay width (fractional z)";
        item.category = "Constraint";
        item.type = "Real";
        item.description = "Fractional width of the saw decay region; must be in (0, 0.5].";
        item.default_value = "0.1";
        item.unit = "";
        item.availability = "constraint_dipole_frame is saw";
        read_sync_double(input.constraint_dipole_saw_pos_dec);
        this->add_item(item);
    }
```

`constraint_io.h`（ConstraintConfig 尾部，`int inner_nmax` 之后）：

```cpp
    // Module A (M1): passive dipole reporting (plan 2026-10-20).  Observation
    // only; the outer loop never reads these.  dipole_report=false (default)
    // keeps the historical output byte-identical.
    bool dipole_report = false;
    std::string dipole_frame = "molecular"; // "molecular" | "saw"
    double dipole_saw_pos_max = -1.0;       // -1 = auto lowest-density plane
    double dipole_saw_pos_dec = 0.1;        // fractional decay width
```

`constraint_io.cpp`（`configure_from_inputs` 内，radii 填充循环之后、`return st;` 之前）：

```cpp
    // Module A (M1) dipole-report plumbing: passive observation, filled
    // straight from INPUT (no interaction with the target list).  The
    // legacy single-type configure entry leaves the defaults (report off),
    // which is exactly its pre-M1 behaviour.
    cfg.dipole_report = PARAM.inp.constraint_dipole_report;
    cfg.dipole_frame = PARAM.inp.constraint_dipole_frame;
    cfg.dipole_saw_pos_max = PARAM.inp.constraint_dipole_saw_pos_max;
    cfg.dipole_saw_pos_dec = PARAM.inp.constraint_dipole_saw_pos_dec;
    // Branch: report requested -> the frame and saw parameters must be
    // valid; anything else refuses to run rather than silently reporting a
    // wrong dipole (the same "loud guard" policy as the rest of this file).
    if (cfg.dipole_report)
    {
        // Branch A: unknown frame.
        if (cfg.dipole_frame != "molecular" && cfg.dipole_frame != "saw")
        {
            error = "constraint_dipole_frame must be \"molecular\" or "
                    "\"saw\", got \"" + cfg.dipole_frame + "\"";
            return ConfigStatus::ERROR;
        }
        // Branch B: saw-frame parameter sanity.
        if (cfg.dipole_frame == "saw"
            && (cfg.dipole_saw_pos_dec <= 0.0 || cfg.dipole_saw_pos_dec > 0.5
                || cfg.dipole_saw_pos_max >= 1.0
                || cfg.dipole_saw_pos_max < -1.0))
        {
            error = "constraint_dipole_saw_pos_dec must be in (0, 0.5] and "
                    "constraint_dipole_saw_pos_max in [-1, 1)";
            return ConfigStatus::ERROR;
        }
        // Branch C: non-collinear spin (nspin == 4) reads a 4-component
        // density the reporter is not wired for -> refuse loudly.
        if (PARAM.inp.nspin == 4)
        {
            error = "constraint_dipole_report is not implemented for "
                    "nspin = 4";
            return ConfigStatus::ERROR;
        }
    }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_io 2>&1 | tail -3 && ctest --test-dir build -R MODULE_ESTATE_constraint_io --output-on-failure 2>&1 | tail -5
```

预期：`100% tests passed`（含既有全部用例 + 新用例）。

- [ ] **Step 5: Commit**

```bash
cd /root/abacus-develop && git add source/source_io/module_parameter/input_parameter.h source/source_io/module_parameter/read_input_item_other.cpp source/source_estate/module_constraint/constraint_io.h source/source_estate/module_constraint/constraint_io.cpp source/source_estate/module_constraint/test/constraint_io_test.cpp && git commit -m "feat(constraint): dipole-report INPUT params + config plumbing (M1, default off)"
```

---

### Task 2: DipoleObserver 分子通道

**Files:**
- Create: `source/source_estate/module_constraint/constraint_dipole.h`
- Create: `source/source_estate/module_constraint/constraint_dipole.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_dipole_test.cpp`

- [ ] **Step 1: 写失败测试**

新建 `test/constraint_dipole_test.cpp`：

```cpp
#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

#include "constraint_test_utils.h"

// Mock symbols required by the cell_info object library in unit tests.
Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
#include "source_estate/module_constraint/constraint_dipole.h"

using ModuleBase::PI;
using ModuleBase::Vector3;

// Normalized Gaussian centered at 'center' with width sigma (Bohr).
static double gaussian(const std::array<double, 3>& r,
                       const std::array<double, 3>& center,
                       const double sigma)
{
    const double dx = r[0] - center[0];
    const double dy = r[1] - center[1];
    const double dz = r[2] - center[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double norm = std::pow(2.0 * PI * sigma * sigma, 1.5);
    return std::exp(-r2 / (2.0 * sigma * sigma)) / norm;
}

class ConstraintDipoleTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell;
    ModulePW::PW_Basis* rhopw = nullptr;
    std::vector<std::array<double, 3>> pos;
    // Grid Cartesian position of local point ir (mirrors the established
    // efield.cpp / constraint test indexing).
    Vector3<double> grid_pos(const int ir) const
    {
        const int i = ir / (rhopw->ny * rhopw->nplane);
        const int j = ir / rhopw->nplane - i * rhopw->ny;
        const int k = ir % rhopw->nplane + rhopw->startz_current;
        const Vector3<double> rfrac(static_cast<double>(i) / rhopw->nx,
                                    static_cast<double>(j) / rhopw->ny,
                                    static_cast<double>(k) / rhopw->nz);
        return rfrac * ucell->latvec * ucell->lat0;
    }
    // Toy density: per-atom Gaussians with electron counts 'nelec_atom',
    // optionally centered off-atom via 'centers' (empty = on-atom).
    void fill_rho(const std::vector<double>& nelec_atom,
                  const std::vector<std::array<double, 3>>& centers,
                  std::vector<double>& rho) const
    {
        const double sigma = 1.0;
        rho.assign(rhopw->nrxx, 0.0);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const Vector3<double> rc = grid_pos(ir);
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                const std::array<double, 3>& c
                    = centers.empty() ? pos[I] : centers[I];
                rho[ir] += nelec_atom[I] * gaussian(r, c, sigma);
            }
        }
    }

    void SetUp() override
    {
        Set_GlobalV_Default();
        ucell = make_h2o_ucell(); // O at (10,10,10), H at (11.2,10,10),(8.8,10,10) Bohr
        rhopw = new ModulePW::PW_Basis;
        rhopw->initgrids(1.0, ucell->latvec, 40, 40, 40);
        rhopw->distribute_r();
        pos = h2o_positions(*ucell);
    }
    void TearDown() override { delete rhopw; }
};

// Neutral cell (N_I = Z*_I = {6,1,1}), O Gaussian shifted by +1 Bohr in z:
//   p = sum Z* (R_I - c_I) = 6 * (R_O - (R_O + z)) = (0, 0, -6) e*Bohr,
// independent of the origin.
TEST_F(ConstraintDipoleTest, MolecularNeutralOriginIndependent)
{
    std::vector<double> rho;
    std::vector<std::array<double, 3>> centers = pos;
    centers[0][2] += 1.0; // O electron cloud displaced +1 Bohr in z
    fill_rho({6.0, 1.0, 1.0}, centers, rho);
    const double* rp[1] = {rho.data()};

    const Vector3<double> o1(0.0, 0.0, 0.0);
    const Vector3<double> o2(3.3, -1.7, 5.1);
    const Vector3<double> p1 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o1);
    const Vector3<double> p2 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o2);
    EXPECT_NEAR(p1.z, -6.0, 1e-6);
    EXPECT_NEAR(p1.x, 0.0, 1e-6);
    EXPECT_NEAR(p1.y, 0.0, 1e-6);
    EXPECT_NEAR(p2.z, -6.0, 1e-6); // origin-independent for a neutral cell
    EXPECT_NEAR(p2.x, 0.0, 1e-6);
}

// Charged toy (N_I = {8,1,1} vs Z* = {6,1,1}, net electron excess 2):
// on-atom Gaussians -> p(o) = -Q_net * o with Q_net = sum Z* - sum N = -2,
// i.e. p(o) = 2*o; shifting the origin shifts p by -Q_net*(o2 - o1).
TEST_F(ConstraintDipoleTest, MolecularChargedOriginShift)
{
    std::vector<double> rho;
    fill_rho({8.0, 1.0, 1.0}, {}, rho);
    const double* rp[1] = {rho.data()};
    const Vector3<double> o1(0.0, 0.0, 0.0);
    const Vector3<double> o2(1.0, 2.0, 3.0);
    const Vector3<double> p1 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o1);
    const Vector3<double> p2 = constraint::DipoleObserver::molecular(
        *ucell, rhopw, rp, 1, o2);
    // Q_net = (6+1+1) - (8+1+1) = -2; p(o) = p(0) - Q_net * o, p(0) = 0.
    EXPECT_NEAR(p1.x, 0.0, 1e-6);
    EXPECT_NEAR(p2.x, 2.0, 1e-6);
    EXPECT_NEAR(p2.y, 4.0, 1e-6);
    EXPECT_NEAR(p2.z, 6.0, 1e-6);
}

// Z*-weighted ionic center: fragment {0} -> R_O; fragment {1,2} -> midpoint.
TEST_F(ConstraintDipoleTest, IonicCenter)
{
    const Vector3<double> c0 = constraint::DipoleObserver::ionic_center(
        *ucell, {0});
    EXPECT_NEAR(c0.x, pos[0][0], 1e-12);
    EXPECT_NEAR(c0.z, pos[0][2], 1e-12);
    const Vector3<double> c12 = constraint::DipoleObserver::ionic_center(
        *ucell, {1, 2});
    EXPECT_NEAR(c12.x, 0.5 * (pos[1][0] + pos[2][0]), 1e-12);
    EXPECT_NEAR(c12.z, 0.5 * (pos[1][2] + pos[2][2]), 1e-12);
}

// q_dip = (p - p_ref) . (R_a - R_b) / |R_a - R_b|^2; electrons moved from
// fragment a to fragment b by delta give q_dip = +delta (a's excess positive
// charge; the LiCl convention, property-1 sec. 5.2).
TEST_F(ConstraintDipoleTest, ChargeTransferSign)
{
    const Vector3<double> p_ref(0.0, 0.0, 0.0);
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(0.0, 0.0, 4.0);
    // Moving +2 e of ELECTRON charge from a to b changes p by
    // -2 * (R_b - R_a) = (0, 0, -8).
    const Vector3<double> p(0.0, 0.0, -8.0);
    const double q = constraint::DipoleObserver::charge_transfer(p, p_ref, a, b);
    EXPECT_NEAR(q, 2.0, 1e-12);
    // Reverse direction flips the sign.
    const double q_rev = constraint::DipoleObserver::charge_transfer(
        p, p_ref, b, a);
    EXPECT_NEAR(q_rev, -2.0, 1e-12);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_dipole 2>&1 | tail -5
```

预期：失败（目标不存在 / 头文件不存在）。（CMake 目标在 Task 6 接线；本步允许 "No rule to make target"。）

- [ ] **Step 3: 实现 constraint_dipole.h/.cpp（分子通道 + 两个助手）**

`constraint_dipole.h`：

```cpp
#ifndef CONSTRAINT_DIPOLE_H
#define CONSTRAINT_DIPOLE_H

#include <vector>

#include "source_base/vector3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

namespace constraint
{

/**
 * @brief Passive dipole observables (module A / M1, plan 2026-10-20).
 *
 * Pure observation on the charge-density grid: nothing here feeds back into
 * the constraint loop.  Two frames:
 *  - molecular: p = sum_I Z*_I (R_I - o) - int rho (r - o) dr  [e*Bohr];
 *    origin-independent for neutral cells, printed with its origin for
 *    charged ones.
 *  - saw: p_z referenced to the Efield-style saw function for periodic
 *    slabs; the derived quantity is the potential drop dV = 4 pi p_z / A.
 *
 * The dipole-derived charge transfer q_dip (charge_transfer) is the
 * dipole-consistent counterpart of the Becke fragment charge (property-1
 * sec. 5.2): for a two-fragment system q_dip = (p - p_ref).d / |d|^2 with d
 * the fragment separation.
 */
class DipoleObserver
{
  public:
    // Dimensionless saw, periodic on x in [0,1); mirrors
    // Efield::saw_function (a = pos_max, b = pos_dec) so the two modules can
    // never disagree on the frame shape.  The decay (jump) region is
    // [pos_max, pos_max + pos_dec].
    static double saw(const double pos_max, const double pos_dec,
                      const double x);

    // Molecular-frame dipole about 'origin' [e*Bohr].  rho[is][ir] is the
    // per-spin density on rho_basis (nspin = 1/2; nspin = 4 is guarded out
    // at configure time).  MPI: pool-reduced.
    static ModuleBase::Vector3<double> molecular(const UnitCell& ucell,
            const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin,
            const ModuleBase::Vector3<double>& origin);

    // Saw-frame dipole along the third lattice vector [e*Bohr]:
    //   p_z = L_z * [ sum_I Z*_I saw(x_I) - int rho(r) saw(x(r)) dr ]
    // with x the fractional third coordinate and L_z the cell height.
    // Guarded (configure time) to cells with a z-aligned third lattice
    // vector orthogonal to a,b.  MPI: pool-reduced.
    static double saw_dipole(const UnitCell& ucell,
            const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin,
            const double pos_max, const double pos_dec);

    // Fractional third-lattice coordinate of the lowest plane-averaged
    // electron density plane (auto placement of the saw decay plane).
    static double lowest_density_plane(const ModulePW::PW_Basis* rho_basis,
            const double* const* rho, const int nspin);

    // Potential drop across the cell [Hartree]: dV = 4 pi p_z / A_perp.
    // Sign convention (locked by the unit test): positive sheet above
    // negative sheet (larger z) -> dV > 0.  Convert with
    // ModuleBase::Hartree_to_eV (1 Hartree = 27.211396 V for a potential).
    static double potential_drop(const double p_saw, const UnitCell& ucell);

    // Dipole-derived charge transfer [e]:
    //   q_dip = (p - p_ref) . (R_a - R_b) / |R_a - R_b|^2
    // = fragment a's excess positive charge relative to the reference.
    static double charge_transfer(const ModuleBase::Vector3<double>& p,
            const ModuleBase::Vector3<double>& p_ref,
            const ModuleBase::Vector3<double>& center_a,
            const ModuleBase::Vector3<double>& center_b);

    // Z*-weighted ionic center of a fragment [Bohr, Cartesian].
    static ModuleBase::Vector3<double> ionic_center(const UnitCell& ucell,
            const std::vector<int>& fragment);

    // Smallest distance [Bohr] from any atom to the saw decay window
    // [pos_max, pos_max + pos_dec] (fractional third coordinate); negative
    // when no atom sits inside the window.  Caller warns below ~1 Bohr.
    static double min_atom_decay_distance(const UnitCell& ucell,
            const double pos_max, const double pos_dec);
};

} // namespace constraint

#endif
```

`constraint_dipole.cpp`（关键实现；完整）：

```cpp
#include "constraint_dipole.h"

#include <algorithm>
#include <cmath>

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"

namespace constraint
{

double DipoleObserver::saw(const double pos_max, const double pos_dec,
                           const double x)
{
    // Shape mirror of Efield::saw_function (efield.cpp): ramp up with slope
    // 1 on [0, pos_max], decay on [pos_max, pos_max+pos_dec], periodic.
    const double fac = 1.0 - pos_dec;
    // Branch A: rising ramp (left of the maximum).
    if (x <= pos_max)
    {
        return x - pos_max + 0.5 * fac;
    }
    // Branch B: right of the decay window (ramp continues periodically).
    else if (x > pos_max + pos_dec)
    {
        return x - pos_max - 1.0 + 0.5 * fac;
    }
    // Branch C: inside the decay window (steep descent).
    else
    {
        return 0.5 * fac - fac * (x - pos_max) / pos_dec;
    }
}

ModuleBase::Vector3<double> DipoleObserver::molecular(
    const UnitCell& ucell, const ModulePW::PW_Basis* rho_basis,
    const double* const* rho, const int nspin,
    const ModuleBase::Vector3<double>& origin)
{
    // Ionic part: sum_I Z*_I (R_I - origin).
    ModuleBase::Vector3<double> p(0.0, 0.0, 0.0);
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double> r
                = ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
            p.x += zv * (r.x - origin.x);
            p.y += zv * (r.y - origin.y);
            p.z += zv * (r.z - origin.z);
        }
    }
    // Electronic part: int rho(r) (r - origin) dr.  Grid indexing mirrors
    // Efield::cal_elec_dipole (efield.cpp:179-198) so the two modules read
    // the same grid points; startz_current keeps the MPI plane split exact.
    const int nspin0 = (nspin == 2) ? 2 : 1;
    ModuleBase::Vector3<double> pe(0.0, 0.0, 0.0);
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int i = ir / (rho_basis->ny * rho_basis->nplane);
        const int j = ir / rho_basis->nplane - i * rho_basis->ny;
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        const ModuleBase::Vector3<double> rfrac(
            static_cast<double>(i) / rho_basis->nx,
            static_cast<double>(j) / rho_basis->ny,
            static_cast<double>(k) / rho_basis->nz);
        const ModuleBase::Vector3<double> r
            = rfrac * ucell.latvec * ucell.lat0;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        pe.x += rho_tot * (r.x - origin.x);
        pe.y += rho_tot * (r.y - origin.y);
        pe.z += rho_tot * (r.z - origin.z);
    }
    Parallel_Reduce::reduce_pool(pe.x);
    Parallel_Reduce::reduce_pool(pe.y);
    Parallel_Reduce::reduce_pool(pe.z);
    const double dV = ucell.omega / static_cast<double>(rho_basis->nxyz);
    p.x -= pe.x * dV;
    p.y -= pe.y * dV;
    p.z -= pe.z * dV;
    return p;
}

ModuleBase::Vector3<double> DipoleObserver::ionic_center(
    const UnitCell& ucell, const std::vector<int>& fragment)
{
    ModuleBase::Vector3<double> c(0.0, 0.0, 0.0);
    double zsum = 0.0;
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia, ++iat)
        {
            // Branch: atom in the fragment accumulates into the center.
            if (std::find(fragment.begin(), fragment.end(), iat)
                != fragment.end())
            {
                const ModuleBase::Vector3<double> r
                    = ucell.atoms[it].taud[ia] * ucell.latvec * ucell.lat0;
                c.x += zv * r.x;
                c.y += zv * r.y;
                c.z += zv * r.z;
                zsum += zv;
            }
        }
    }
    // Guard: an empty fragment can not reach here (configure rejects empty
    // fragments), but never divide by zero.
    if (zsum > 0.0)
    {
        c.x /= zsum;
        c.y /= zsum;
        c.z /= zsum;
    }
    return c;
}

double DipoleObserver::charge_transfer(
    const ModuleBase::Vector3<double>& p,
    const ModuleBase::Vector3<double>& p_ref,
    const ModuleBase::Vector3<double>& center_a,
    const ModuleBase::Vector3<double>& center_b)
{
    const ModuleBase::Vector3<double> d = center_a - center_b;
    const double r2 = d * d; // |d|^2
    // Guard: coincident centers carry no charge-separation direction.
    if (r2 < 1e-20)
    {
        return 0.0;
    }
    return ((p - p_ref) * d) / r2;
}

} // namespace constraint
```

（saw_dipole / lowest_density_plane / potential_drop / min_atom_decay_distance 在 Task 3 添加。）

- [ ] **Step 4: 接线 CMake 并运行测试**

`source/source_estate/CMakeLists.txt`（`module_constraint/constraint_deriv.cpp` 行之后）加一行：

```cmake
    module_constraint/constraint_dipole.cpp
```

`test/CMakeLists.txt` 末尾追加：

```cmake
AddTest(
  TARGET MODULE_ESTATE_constraint_dipole
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_dipole_test.cpp
    ../constraint_dipole.cpp
    ../../../source_io/module_output/output.cpp
)
```

运行：

```bash
cd /root/abacus-develop && cmake --build build -j 32 2>&1 | tail -3 && cmake --build build -j 32 --target MODULE_ESTATE_constraint_dipole 2>&1 | tail -3 && ctest --test-dir build -R MODULE_ESTATE_constraint_dipole --output-on-failure 2>&1 | tail -5
```

预期：4 个测试全过（`MolecularNeutralOriginIndependent`、`MolecularChargedOriginShift`、`IonicCenter`、`ChargeTransferSign`）。若 Gaussian 截断误差超 1e-6，把容差放宽到 1e-5 并在注释说明（20 Bohr 盒、sigma=1 的尾部 ~1e-12，正常应到 1e-8）。

- [ ] **Step 5: Commit**

```bash
cd /root/abacus-develop && git add source/source_estate/module_constraint/constraint_dipole.h source/source_estate/module_constraint/constraint_dipole.cpp source/source_estate/module_constraint/test/constraint_dipole_test.cpp source/source_estate/CMakeLists.txt source/source_estate/module_constraint/test/CMakeLists.txt && git commit -m "feat(constraint): DipoleObserver molecular frame + q_dip helpers (M1)"
```

---

### Task 3: DipoleObserver saw 通道 + ΔV + 自动选面 + 跳变面守卫

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_dipole.h`（声明已在 Task 2 给出）
- Modify: `source/source_estate/module_constraint/constraint_dipole.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_dipole_test.cpp`

- [ ] **Step 1: 写失败测试（追加到 constraint_dipole_test.cpp）**

```cpp
// saw() shape: periodic, continuous at pos_max and pos_max+pos_dec, unit
// ramp slope outside the decay window (locks the Efield mirror semantics).
TEST_F(ConstraintDipoleTest, SawShape)
{
    const double a = 0.4, b = 0.1; // pos_max, pos_dec
    const double fac = 1.0 - b;
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, 0.0),
                -a + 0.5 * fac, 1e-15);
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, a), 0.5 * fac, 1e-15);
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, a + b), -0.5 * fac,
                1e-15);
    // Periodicity: saw(0) == saw(1-eps) in the limit.
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, 0.0),
                constraint::DipoleObserver::saw(a, b, 1.0 - 1e-12), 1e-9);
    // Unit ramp slope in the rising region: saw(x+h) - saw(x) = h.
    const double x = 0.2, h = 1e-6;
    EXPECT_NEAR(constraint::DipoleObserver::saw(a, b, x + h)
                    - constraint::DipoleObserver::saw(a, b, x),
                h, 1e-12);
}

// Saw dipole of a charge-transfer toy: neutral atomic background (N_I = Z*,
// cancels exactly) + 2 e moved from the z = 5 Bohr plane to the z = 15 Bohr
// plane (Gaussians in z, uniform in x-y).  Both planes sit in the ramp
// region (slope 1), so p_z = -2 * (15 - 5) = -20 e*Bohr, and
// dV = 4 pi p_z / A < 0 (electrons moved up -> potential higher at the
// bottom).  Box: 20 x 20 x 20 Bohr -> A = 400 Bohr^2.
TEST_F(ConstraintDipoleTest, SawDipoleAndPotentialDrop)
{
    // Density: neutral on-atom background + transfer pair.
    std::vector<double> rho(rhopw->nrxx, 0.0);
    const double sigma = 1.0;
    const std::vector<double> nelec_atom = {6.0, 1.0, 1.0};
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        const Vector3<double> rc = grid_pos(ir);
        const std::array<double, 3> r{rc.x, rc.y, rc.z};
        for (size_t I = 0; I < pos.size(); ++I)
        {
            rho[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
        }
        // Transfer pair: +2 e at z = 15, -2 e at z = 5 (electrons), each a
        // z-Gaussian smeared uniformly over x-y (amplitude normalized in z
        // only; the x-y integral is supplied by the uniform factor 1/A_xy
        // with A_xy = 400 Bohr^2).
        const double gz_up = std::exp(-std::pow((rc.z - 15.0) / sigma, 2) / 2.0)
                             / (std::sqrt(2.0 * PI) * sigma) / 400.0;
        const double gz_dn = std::exp(-std::pow((rc.z - 5.0) / sigma, 2) / 2.0)
                             / (std::sqrt(2.0 * PI) * sigma) / 400.0;
        rho[ir] += 2.0 * (gz_up - gz_dn);
    }
    const double* rp[1] = {rho.data()};
    // Frame: decay window [0.45, 0.55] in fractional z = [9, 11] Bohr, far
    // from both sheets and from the atoms (H at x = 8.8/11.2 Bohr sit at
    // z = 10 Bohr!  -> use a frame whose decay window is off the atoms:
    // pos_max = 0.95, pos_dec = 0.05 -> decay in [19, 20] Bohr).
    const double p = constraint::DipoleObserver::saw_dipole(
        *ucell, rhopw, rp, 1, 0.95, 0.05);
    // Atoms at z = 10 Bohr (x_frac = 0.5): saw = 0.5 - 0.95 + 0.5*0.95 =
    // 0.025 for both background and ions -> they cancel (N_I = Z*_I).
    // Sheets at z = 5/15 Bohr (x_frac 0.25/0.75, both in the ramp):
    // saw(0.25) = 0.25 - 0.95 + 0.475 = -0.225; saw(0.75) = 0.75 - 0.95 +
    // 0.475 = 0.275.  p_z = L_z * [0 - 2*(saw(0.75) - saw(0.25))] =
    // 20 * (-2 * 0.5) = -20 e*Bohr.
    EXPECT_NEAR(p, -20.0, 1e-6);
    const double dv = constraint::DipoleObserver::potential_drop(p, *ucell);
    // dV = 4 pi p / A = 4 pi (-20) / 400 Hartree.
    EXPECT_NEAR(dv, -20.0 * ModuleBase::FOUR_PI / 400.0, 1e-12);
    EXPECT_LT(dv, 0.0); // electrons up -> potential higher at the bottom
}

// Auto placement: the lowest plane-averaged density of the transfer toy is
// far from the sheets and atoms; the returned fractional z must lie in the
// [0.55, 0.95] low-density arc (between the atoms at 0.5 and the upper sheet
// at 0.75 the density dips, but the global minimum sits in the empty top
// region around z ~ 0.85-0.95 or bottom 0.55-0.7; assert loosely).
TEST_F(ConstraintDipoleTest, LowestDensityPlane)
{
    std::vector<double> rho(rhopw->nrxx, 0.0);
    const double sigma = 1.0;
    const std::vector<double> nelec_atom = {6.0, 1.0, 1.0};
    for (int ir = 0; ir < rhopw->nrxx; ++ir)
    {
        const Vector3<double> rc = grid_pos(ir);
        const std::array<double, 3> r{rc.x, rc.y, rc.z};
        for (size_t I = 0; I < pos.size(); ++I)
        {
            rho[ir] += nelec_atom[I] * gaussian(r, pos[I], sigma);
        }
    }
    const double* rp[1] = {rho.data()};
    const double x = constraint::DipoleObserver::lowest_density_plane(
        rhopw, rp, 1);
    // Atoms sit at z_frac = 0.5; the minimum must be far from them.
    const double d = std::fabs(x - 0.5);
    const double dwrap = std::min(d, 1.0 - d); // periodic distance
    EXPECT_GT(dwrap, 0.1);
    EXPECT_GE(x, 0.0);
    EXPECT_LT(x, 1.0);
}

// Decay-window guard: atoms at z = 10 Bohr; window [9.5, 10.5] Bohr
// (pos_max 0.475, pos_dec 0.05) contains them -> distance 0; window
// [19, 20] Bohr -> distance ~9 Bohr.
TEST_F(ConstraintDipoleTest, DecayWindowGuard)
{
    EXPECT_NEAR(constraint::DipoleObserver::min_atom_decay_distance(
                    *ucell, 0.475, 0.05),
                0.0, 1e-12);
    const double d = constraint::DipoleObserver::min_atom_decay_distance(
        *ucell, 0.95, 0.05);
    EXPECT_GT(d, 8.9);
    EXPECT_LT(d, 9.1);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_dipole 2>&1 | tail -5
```

预期：链接错误（`saw_dipole` 等未定义）。

- [ ] **Step 3: 实现 saw 通道（追加到 constraint_dipole.cpp）**

```cpp
double DipoleObserver::saw_dipole(const UnitCell& ucell,
        const ModulePW::PW_Basis* rho_basis,
        const double* const* rho, const int nspin,
        const double pos_max, const double pos_dec)
{
    // Effective cell height along the third lattice vector.  Configure-time
    // guard restricts this frame to z-aligned third vectors orthogonal to
    // a,b, so latvec(2,2) * lat0 is the exact height.
    const double lz = ucell.latvec(2, 2) * ucell.lat0;
    // Ionic part: sum_I Z*_I saw(x_I), x_I = fractional third coordinate.
    double p_ion = 0.0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const double zv = ucell.atoms[it].ncpp.zv;
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            double x = ucell.atoms[it].taud[ia].z;
            // Wrap into [0, 1): taud can carry small out-of-range values.
            x -= std::floor(x);
            p_ion += zv * saw(pos_max, pos_dec, x);
        }
    }
    // Electronic part: int rho(r) saw(x(r)) dr (same indexing as molecular).
    const int nspin0 = (nspin == 2) ? 2 : 1;
    double p_elec = 0.0;
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        const double x = static_cast<double>(k) / rho_basis->nz;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        p_elec += rho_tot * saw(pos_max, pos_dec, x);
    }
    Parallel_Reduce::reduce_pool(p_elec);
    const double dV = ucell.omega / static_cast<double>(rho_basis->nxyz);
    return lz * (p_ion - p_elec * dV);
}

double DipoleObserver::lowest_density_plane(
    const ModulePW::PW_Basis* rho_basis,
    const double* const* rho, const int nspin)
{
    const int nspin0 = (nspin == 2) ? 2 : 1;
    // Plane-averaged profile over the global z index; the pool reduction
    // hands every rank the complete profile.
    std::vector<double> profile(rho_basis->nz, 0.0);
    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
    {
        const int k = ir % rho_basis->nplane + rho_basis->startz_current;
        double rho_tot = 0.0;
        for (int is = 0; is < nspin0; ++is)
        {
            rho_tot += rho[is][ir];
        }
        profile[k] += rho_tot;
    }
    Parallel_Reduce::reduce_pool(profile.data(),
                                 static_cast<int>(profile.size()));
    const int kmin = static_cast<int>(std::min_element(
                          profile.begin(), profile.end())
                      - profile.begin());
    return static_cast<double>(kmin) / rho_basis->nz;
}

double DipoleObserver::potential_drop(const double p_saw,
                                      const UnitCell& ucell)
{
    // A_perp = |a x b| * lat0^2 (z-aligned third vector, configure guard).
    const double axb = ucell.latvec(0, 0) * ucell.latvec(1, 1)
                       - ucell.latvec(0, 1) * ucell.latvec(1, 0);
    const double area = std::abs(axb) * ucell.lat0 * ucell.lat0;
    // dV = 4 pi p_z / A [Hartree]; sign locked by SawDipoleAndPotentialDrop.
    return ModuleBase::FOUR_PI * p_saw / area;
}

double DipoleObserver::min_atom_decay_distance(const UnitCell& ucell,
        const double pos_max, const double pos_dec)
{
    const double lz = ucell.latvec(2, 2) * ucell.lat0;
    double dmin = 1e30;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            double x = ucell.atoms[it].taud[ia].z;
            x -= std::floor(x);
            // Distance into the decay window [pos_max, pos_max+pos_dec]
            // (periodic): 0 inside, else the gap to the nearer edge,
            // converted to Bohr.
            double d = 0.0;
            // Branch A: atom inside the decay window.
            if (x >= pos_max && x <= pos_max + pos_dec)
            {
                d = 0.0;
            }
            // Branch B: outside — gap to the nearer window edge, periodic.
            else
            {
                const double d_left
                    = std::fmod(x - (pos_max + pos_dec) + 1.0, 1.0);
                const double d_right
                    = std::fmod(pos_max - x + 1.0, 1.0);
                d = std::min(d_left, d_right);
            }
            dmin = std::min(dmin, d * lz);
        }
    }
    return dmin;
}
```

并在 configure 守卫中加 saw 晶胞形状检查（`constraint_io.cpp` 的 Branch B 之后追加 Branch B2）：

```cpp
        // Branch B2: the saw frame requires a z-aligned third lattice
        // vector orthogonal to a,b (the dipole and deltaV formulas use
        // latvec(2,2) as the exact cell height).
        if (cfg.dipole_frame == "saw")
        {
            const ModuleBase::Matrix3& lv = ucell.latvec;
            const double off = std::abs(lv(0, 2)) + std::abs(lv(1, 2))
                               + std::abs(lv(2, 0)) + std::abs(lv(2, 1));
            if (off > 1e-10 || lv(2, 2) <= 0.0)
            {
                error = "constraint_dipole_frame = saw requires a z-aligned "
                        "third lattice vector orthogonal to a,b";
                return ConfigStatus::ERROR;
            }
        }
```

（`constraint_io.cpp` 需要 `#include <cmath>` 若尚无；`ucell.latvec` 在该文件已可用——`configure_from_inputs` 接收 `const UnitCell&`。对应地给 Task 1 的测试补一个用例：非正交晶胞 + saw → ERROR。把 `make_h2o_ucell()` 返回值的 `ucell->latvec(0,2) = 0.1;` 后调用即可。）

- [ ] **Step 4: 运行确认通过**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_dipole MODULE_ESTATE_constraint_io 2>&1 | tail -3 && ctest --test-dir build -R "MODULE_ESTATE_constraint_(dipole|io)" --output-on-failure 2>&1 | tail -5
```

预期：全部通过。

- [ ] **Step 5: Commit**

```bash
cd /root/abacus-develop && git add -A source/source_estate/module_constraint && git commit -m "feat(constraint): DipoleObserver saw frame + deltaV + auto placement + decay guard (M1)"
```

---

### Task 4: ConstraintLoop 集成（p_ref 快照 + final_report 偶板块）

**Files:**
- Modify: `source/source_estate/module_constraint/constraint_loop.h`
- Modify: `source/source_estate/module_constraint/constraint_loop.cpp`
- Test: `source/source_estate/module_constraint/test/constraint_loop_test.cpp`

- [ ] **Step 1: 写失败测试（追加到 constraint_loop_test.cpp）**

```cpp
// M1 dipole reporting through the loop: init arms the reporter from cfg,
// observe() refreshes the value, and the REFERENCE->CONSTRAINED transition
// snapshots p_ref alongside Q_ref.
TEST_F(ConstraintLoopTest, DipoleReportLifecycle)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    constraint::ConstraintConfig cfg = make_cfg(0.1);
    cfg.dipole_report = true;
    cfg.dipole_frame = "molecular";
    loop.init(*ucell, rhopw, cfg, radii, 10.0);
    ASSERT_TRUE(loop.enabled());
    ASSERT_EQ(loop.phase(), constraint::LoopPhase::REFERENCE);

    // Density: neutral on-atom Gaussians (N_I = Z*_I = {6,1,1}) with the O
    // cloud displaced +1 Bohr in z -> p = (0, 0, -6) e*Bohr.
    std::vector<double> rho(rhopw->nrxx, 0.0);
    {
        const double sigma = 1.0;
        std::vector<std::array<double, 3>> centers = pos;
        centers[0][2] += 1.0;
        const std::vector<double> ne = {6.0, 1.0, 1.0};
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            const int i = ir / (rhopw->ny * rhopw->nplane);
            const int j = ir / rhopw->nplane - i * rhopw->ny;
            const int k = ir % rhopw->nplane + rhopw->startz_current;
            const ModuleBase::Vector3<double> rfrac(
                static_cast<double>(i) / rhopw->nx,
                static_cast<double>(j) / rhopw->ny,
                static_cast<double>(k) / rhopw->nz);
            const ModuleBase::Vector3<double> rc
                = rfrac * ucell->latvec * ucell->lat0;
            const std::array<double, 3> r{rc.x, rc.y, rc.z};
            for (size_t I = 0; I < pos.size(); ++I)
            {
                rho[ir] += ne[I] * gaussian(r, centers[I], sigma);
            }
        }
    }
    const double* rp[1] = {rho.data()};
    loop.observe(1, rp, 1);
    EXPECT_NEAR(loop.dipole().z, -6.0, 1e-6);
    EXPECT_FALSE(loop.dipole_ref_valid());

    // Close the reference phase: the snapshot must carry the same dipole.
    bool conv = true;
    loop.on_scf_converged(2, conv);
    EXPECT_EQ(loop.phase(), constraint::LoopPhase::CONSTRAINED);
    ASSERT_TRUE(loop.dipole_ref_valid());
    EXPECT_NEAR(loop.dipole_ref().z, -6.0, 1e-6);

    // q_dip: single constraint {0} -> counter-fragment {1,2}; at the
    // reference point p == p_ref, so q_dip = 0.
    bool qd_valid = false;
    const double qd = loop.q_dip(qd_valid);
    ASSERT_TRUE(qd_valid);
    EXPECT_NEAR(qd, 0.0, 1e-12);
}

// Default off: the reporter stays inert and dipole_ref never validates.
TEST_F(ConstraintLoopTest, DipoleReportDefaultOff)
{
    constraint::ConstraintLoop& loop = constraint::ConstraintLoop::instance();
    loop.init(*ucell, rhopw, make_cfg(0.1), radii, 10.0);
    const double* rp[1] = {rho_ref.data()};
    loop.observe(1, rp, 1);
    bool conv = true;
    loop.on_scf_converged(2, conv);
    EXPECT_FALSE(loop.dipole_ref_valid());
    bool qd_valid = true;
    EXPECT_NEAR(loop.q_dip(qd_valid), 0.0, 1e-12);
    EXPECT_FALSE(qd_valid);
}
```

（注：`gaussian` 与 `pos`、`rho_ref` 均为该测试文件既有设施。）

- [ ] **Step 2: 运行确认失败**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -5
```

预期：编译错误（`dipole()` / `dipole_ref_valid()` / `q_dip()` 不存在）。

注意：constraint_loop 的 AddTest 目前可能未链接 constraint_dipole.cpp——检查 `test/CMakeLists.txt` 中 `MODULE_ESTATE_constraint_loop` 的 SOURCES，若含 `../constraint_loop.cpp` 则需同时加 `../constraint_dipole.cpp`。

- [ ] **Step 3: 实现 ConstraintLoop 集成**

`constraint_loop.h`——public 区（`const std::vector<double>& charges()` 访问器附近）追加：

```cpp
    // ---- Module A (M1) passive dipole report -----------------------------
    // Pure observation: armed by cfg.dipole_report, refreshed by observe(),
    // never read by the loop logic.  dipole_ref() is the REFERENCE-phase
    // snapshot (the q_dip origin, delta semantics; valid only after the
    // reference phase closes).  q_dip(valid) is the dipole-derived charge
    // transfer of the constrained fragment vs the rest [e]; valid=false
    // when the constraint list can not define a fragment pair (0 or >2
    // constraints) or the report is off.
    const ModuleBase::Vector3<double>& dipole() const { return dipole_latest_; }
    const ModuleBase::Vector3<double>& dipole_ref() const { return dipole_ref_; }
    bool dipole_ref_valid() const { return dipole_ref_valid_; }
    double delta_v() const { return delta_v_latest_; } // saw frame, Hartree
    double q_dip(bool& valid) const;
```

private 区追加：

```cpp
    // Module A reporter state (see the public accessors above).
    void update_dipole(const double* const* rho, const int nspin);
    void print_dipole_report() const;
    bool dipole_report_ = false;
    std::string dipole_frame_ = "molecular";
    double dipole_saw_pos_max_ = -1.0; // -1 = auto (lowest-density plane)
    double dipole_saw_pos_dec_ = 0.1;
    bool saw_resolved_ = false;        // auto frame resolved this geometry
    double saw_pos_max_live_ = 0.5;    // resolved pos_max actually in use
    const UnitCell* ucell_ = nullptr;  // per-geometry, set in init()
    ModuleBase::Vector3<double> dipole_origin_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> dipole_latest_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> dipole_ref_{0.0, 0.0, 0.0};
    bool dipole_ref_valid_ = false;
    double delta_v_latest_ = 0.0;
    ModuleBase::Vector3<double> frag_a_{0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> frag_b_{0.0, 0.0, 0.0};
    bool frag_valid_ = false;
```

并在 reset() 末尾追加（`last_reset_iter_ = 0;` 之后）：

```cpp
    dipole_report_ = false;
    dipole_frame_ = "molecular";
    dipole_saw_pos_max_ = -1.0;
    dipole_saw_pos_dec_ = 0.1;
    saw_resolved_ = false;
    saw_pos_max_live_ = 0.5;
    ucell_ = nullptr;
    dipole_origin_ = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    dipole_latest_ = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    dipole_ref_ = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    dipole_ref_valid_ = false;
    delta_v_latest_ = 0.0;
    frag_a_ = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    frag_b_ = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    frag_valid_ = false;
```

`constraint_loop.cpp`：

(a) 头部 include 加 `#include "constraint_dipole.h"`。

(b) init() 中 **WeightGrid build 之后、fixed_mu 早退分支之前**插入：

```cpp
    // ---- Module A (M1) passive dipole reporter arming --------------------
    // Must run before the fixed-mu early return below so FIXED_MU runs keep
    // the report.  Pure observation: nothing here feeds the loop logic.
    dipole_report_ = cfg_.dipole_report;
    dipole_frame_ = cfg_.dipole_frame;
    dipole_saw_pos_max_ = cfg_.dipole_saw_pos_max;
    dipole_saw_pos_dec_ = cfg_.dipole_saw_pos_dec;
    saw_resolved_ = false;
    dipole_ref_valid_ = false;
    delta_v_latest_ = 0.0;
    ucell_ = &ucell; // per-geometry; the esolver outlives the loop usage
    // Branch: reporter armed -> resolve the origin and the fragment pair.
    if (dipole_report_)
    {
        // Origin for the molecular frame: ionic charge center of the whole
        // cell (origin-independent for neutral cells, the printed convention
        // for charged ones).
        std::vector<int> all_atoms(ucell.nat);
        std::iota(all_atoms.begin(), all_atoms.end(), 0);
        dipole_origin_ = DipoleObserver::ionic_center(ucell, all_atoms);
        // Fragment pair for q_dip: exactly one constraint -> that fragment
        // vs the rest of the cell; exactly two -> the pair; anything else ->
        // q_dip undefined (the report prints the dipole only).
        frag_valid_ = false;
        if (specs_.size() == 1 || specs_.size() == 2)
        {
            frag_a_ = DipoleObserver::ionic_center(ucell, specs_[0].atoms);
            std::vector<int> other;
            if (specs_.size() == 2)
            {
                other = specs_[1].atoms;
            }
            else
            {
                // Branch: complement of specs_[0] in the whole cell.
                for (int i = 0; i < ucell.nat; ++i)
                {
                    if (std::find(specs_[0].atoms.begin(),
                                  specs_[0].atoms.end(), i)
                        == specs_[0].atoms.end())
                    {
                        other.push_back(i);
                    }
                }
            }
            if (!other.empty())
            {
                frag_b_ = DipoleObserver::ionic_center(ucell, other);
                frag_valid_ = true;
            }
        }
        // Saw frame: warn when an atom sits in / near the decay window.
        if (dipole_frame_ == "saw" && dipole_saw_pos_max_ >= 0.0)
        {
            const double d = DipoleObserver::min_atom_decay_distance(
                ucell, dipole_saw_pos_max_, dipole_saw_pos_dec_);
            if (d < 1.0)
            {
                ModuleBase::WARNING("ConstraintLoop::init",
                    "an atom sits within 1 Bohr of the saw decay window; "
                    "the saw-frame dipole is ill-defined there — move "
                    "constraint_dipole_saw_pos_max");
            }
        }
    }
```

（`constraint_loop.cpp` 需 `#include <numeric>`——已有；`#include <algorithm>`——检查，若无则加。）

(c) observe() 末尾追加：

```cpp
    // Module A (M1): passive dipole observation; never feeds the loop.
    if (dipole_report_)
    {
        update_dipole(rho, nspin);
    }
```

(d) `Q_ref_ = Q_;`（约 line 322，REFERENCE→CONSTRAINED 转移内）之后追加：

```cpp
        // Module A: snapshot the reference dipole with the reference
        // charges (the q_dip origin, delta semantics).
        dipole_ref_ = dipole_latest_;
        dipole_ref_valid_ = dipole_report_;
```

(e) 新成员函数实现：

```cpp
void ConstraintLoop::update_dipole(const double* const* rho, const int nspin)
{
    // Branch: saw frame -> resolve the auto placement once per geometry on
    // the first density the loop sees (init has no density yet).
    if (dipole_frame_ == "saw")
    {
        if (!saw_resolved_)
        {
            saw_pos_max_live_
                = (dipole_saw_pos_max_ >= 0.0)
                      ? dipole_saw_pos_max_
                      : DipoleObserver::lowest_density_plane(
                            wg_->rho_basis(), rho, nspin);
            saw_resolved_ = true;
            GlobalV::ofs_running << "[constraint] dipole saw frame resolved:"
                " pos_max = " << saw_pos_max_live_
                << (dipole_saw_pos_max_ >= 0.0 ? " (input)" : " (auto)")
                << ", pos_dec = " << dipole_saw_pos_dec_ << "\n";
        }
        const double p = DipoleObserver::saw_dipole(
            *ucell_, wg_->rho_basis(), rho, nspin, saw_pos_max_live_,
            dipole_saw_pos_dec_);
        dipole_latest_ = ModuleBase::Vector3<double>(0.0, 0.0, p);
        delta_v_latest_ = DipoleObserver::potential_drop(p, *ucell_);
    }
    // Branch: molecular frame (default).
    else
    {
        dipole_latest_ = DipoleObserver::molecular(
            *ucell_, wg_->rho_basis(), rho, nspin, dipole_origin_);
        delta_v_latest_ = 0.0;
    }
}

double ConstraintLoop::q_dip(bool& valid) const
{
    valid = dipole_report_ && dipole_ref_valid_ && frag_valid_;
    // Guard: without a reference snapshot or a fragment pair the transfer
    // is undefined (the report prints the raw dipole only).
    if (!valid)
    {
        return 0.0;
    }
    return DipoleObserver::charge_transfer(dipole_latest_, dipole_ref_,
                                           frag_a_, frag_b_);
}

void ConstraintLoop::print_dipole_report() const
{
    // Debye: 1 e*Bohr = 2.541746 D.
    const double eBohr_to_debye = 2.541746;
    GlobalV::ofs_running << "[constraint] dipole report (frame="
                         << dipole_frame_;
    if (dipole_frame_ == "molecular")
    {
        GlobalV::ofs_running << ", origin=(" << dipole_origin_.x << ", "
                             << dipole_origin_.y << ", " << dipole_origin_.z
                             << ") Bohr";
    }
    else
    {
        GlobalV::ofs_running << ", pos_max=" << saw_pos_max_live_
                             << ", pos_dec=" << dipole_saw_pos_dec_;
    }
    GlobalV::ofs_running << ")\n";
    GlobalV::ofs_running << "  p      = (" << dipole_latest_.x << ", "
                         << dipole_latest_.y << ", " << dipole_latest_.z
                         << ") e*Bohr  (|p| = "
                         << dipole_latest_.norm() * eBohr_to_debye
                         << " Debye)\n";
    if (dipole_ref_valid_)
    {
        GlobalV::ofs_running << "  p_ref  = (" << dipole_ref_.x << ", "
                             << dipole_ref_.y << ", " << dipole_ref_.z
                             << ") e*Bohr (reference phase)\n";
    }
    if (dipole_frame_ == "saw")
    {
        GlobalV::ofs_running
            << "  deltaV = " << delta_v_latest_ * ModuleBase::Hartree_to_eV
            << " V  (4*pi*p_saw/A; positive: potential higher at +z)\n";
    }
    bool qd_valid = false;
    const double qd = q_dip(qd_valid);
    if (qd_valid)
    {
        const ModuleBase::Vector3<double> d = frag_a_ - frag_b_;
        GlobalV::ofs_running << "  q_dip  = " << qd
                             << " e  (dipole-derived; fragment vs rest, R = "
                             << d.norm() << " Bohr; Becke q above is the "
                                "bookkeeping counterpart — property-1 5.3)\n";
    }
}
```

(f) final_report() 末尾（`GlobalV::ofs_running << last_audit_line_ << "\n";` 之后）追加：

```cpp
    // Module A (M1): the dipole block rides the final report; off by
    // default, so the historical output is byte-identical when disabled.
    if (dipole_report_)
    {
        print_dipole_report();
    }
```

(g) `test/CMakeLists.txt` 中 `MODULE_ESTATE_constraint_loop` 的 SOURCES 加 `../constraint_dipole.cpp`（若该目标链接了 constraint_loop.cpp）。

- [ ] **Step 4: 运行确认通过**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target MODULE_ESTATE_constraint_loop 2>&1 | tail -3 && ctest --test-dir build -R MODULE_ESTATE_constraint_loop --output-on-failure 2>&1 | tail -5
```

预期：全部通过（既有用例 + 2 个新用例）。

- [ ] **Step 5: Commit**

```bash
cd /root/abacus-develop && git add source/source_estate/module_constraint && git commit -m "feat(constraint): wire dipole report into ConstraintLoop observe/final_report (M1)"
```

---

### Task 5: 全量回归 + abacus 构建

**Files:** 无新增（验证任务）。

- [ ] **Step 1: 全量单元测试回归**

```bash
cd /root/abacus-develop && cmake --build build -j 32 2>&1 | tail -3 && ctest --test-dir -j 32 -R constraint --output-on-failure 2>&1 | tail -15
```

预期：`100% tests passed`（module_constraint 全部 9 个目标：weight_grid / observe / deriv / mu_solver / io / inject_pw / inject_lcao / loop / dipole）。

- [ ] **Step 2: 构建 abacus 主程序**

```bash
cd /root/abacus-develop && cmake --build build -j 32 --target abacus 2>&1 | tail -3
```

预期：链接成功，生成 `build/abacus`（或 `build/source/abacus`，以 CMakeCache 为准——`find build -name abacus -type f` 确认）。

- [ ] **Step 3: Commit（若有构建接线遗漏的修复）**

```bash
cd /root/abacus-develop && git add -A && git commit -m "build(constraint): M1 dipole report — full regression green" || true
```

---

### Task 6: 集成验证（h2o_dimer 本地 + LiCl Bohrium）

**Files:**
- 集成用例：`tests/constraint_ct_dimer/h2o_dimer/`（LCAO，PP_ORB 仓库内置赝势）
- 新建测试文档：`docs/superpowers/specs/2026-10-XX-m1-dipole-report-test.md`（AGENTS.md 要求，日期填实际执行日）

- [ ] **Step 1: 回归验证——默认关闭时输出逐字节不变**

```bash
cd /root/abacus-develop/tests/constraint_ct_dimer/h2o_dimer && /root/abacus-develop/build/abacus > out.log 2>&1; grep -c "dipole report" OUT.autotest/running_scf.log || echo "0 (expected: no dipole block)"
```

预期：`0`——日志无 dipole block，与基线逐字节一致（可用 git stash 前后各跑一次 diff 验证，或与既有 `results/` 归档对比）。

- [ ] **Step 2: 开启报告——dipole block 出现且数值合理**

在 INPUT 追加两行：

```
constraint_dipole_report 1
constraint_dipole_frame molecular
```

重跑：

```bash
cd /root/abacus-develop/tests/constraint_ct_dimer/h2o_dimer && /root/abacus-develop/build/abacus > out_dip.log 2>&1; grep -A6 "dipole report" OUT.autotest/running_scf.log
```

预期：
- 出现 `[constraint] dipole report (frame=molecular, origin=...)` 块，含 p、p_ref、q_dip 三行；
- H2O 二聚体（10 分子）偶极量级 ~1-3 Debye（H2O 单体 1.85 D），q_dip 与 delta 目标（见 constraint_target.json）同量级；
- 约束收敛行、SCF 收敛行与 Step 1 完全一致（报告是纯观测）。

- [ ] **Step 3: saw 通道冒烟**

INPUT 改为 `constraint_dipole_frame saw`（h2o_dimer 是 z 正交盒则直接可跑；否则跳过本步并记录原因）。重跑后 grep：

```bash
grep -A6 "dipole report" OUT.autotest/running_scf.log
```

预期：出现 `pos_max=... (auto)` 行与 `deltaV = ... V` 行。

- [ ] **Step 4: LiCl Bohrium 复算（对照性质1 的 6.8665 D）**

复用性质1 的 LiCl 输入（APNS-v1 `Li.upf`/`Cl.upf`，15 Å 盒，PW 100/400 Ry，δ=0 空操作约束），加 `constraint_dipole_report = true`。预期：

| 量 | 预期值 | 来源 |
|---|---|---|
| 分子偶极 \|p\| | 6.8665 ± 0.01 D | 性质1 §5.2 后处理值（同一份密度、同一公式） |
| q_dip(δ=0) | 0.000 e | delta 语义（p = p_ref） |
| Li Becke 电荷 | 3.095937431 e | 性质1 §3.2（回归不变） |

提交 Bohrium 后把 running_scf.log 的 dipole block 摘录进测试文档。

- [ ] **Step 5: 写测试文档 + 更新开发日志**

按 AGENTS.md 格式写 `docs/superpowers/specs/2026-10-XX-m1-dipole-report-test.md`（测试计划/设置/结果/分析/下一步），并在 `deltap-development-log.md` 追加本轮条目（做了什么/文件清单/bug 清单/下一步=M2 计划）。

- [ ] **Step 6: Commit**

```bash
cd /root/abacus-develop && git add docs/superpowers/specs/2026-10-*-m1-dipole-report-test.md docs/superpowers/specs/deltap-development-log.md && git commit -m "docs(constraint): M1 dipole report test round (h2o_dimer + LiCl 6.87 D closure)"
```

---

## Self-Review 记录（计划编写后）

1. **Spec 覆盖**：设计文档 §3（模块 A）的每一项——分子通道、带电参考点、saw 通道、ΔV、q_dip、自动选面、跳变面守卫、final_report 输出、默认关闭回归、LiCl 6.8665 D 验证——均映射到 Task 1-6。✅
2. **占位符扫描**：无 TBD/TODO；所有代码步骤含完整代码。✅
3. **类型一致性**：`DipoleObserver::{molecular, saw_dipole, lowest_density_plane, potential_drop, charge_transfer, ionic_center, min_atom_decay_distance}`、`ConstraintLoop::{dipole, dipole_ref, dipole_ref_valid, delta_v, q_dip(bool&)}`、`ConstraintConfig::{dipole_report, dipole_frame, dipole_saw_pos_max, dipole_saw_pos_dec}`、`PARAM.inp.constraint_dipole_*` 在全部 Task 间一致。✅
4. **已知风险**：①Gaussian 玩具密度的数值积分容差（1e-6 太紧则放宽到 1e-5，Task 2 Step 4 已注明）；②`MODULE_ESTATE_constraint_loop` 的 AddTest SOURCES 需加 `../constraint_dipole.cpp`（Task 4 Step 2 已注明检查）；③h2o_dimer 若非 z 正交盒则 saw 冒烟跳过（Task 6 Step 3 已注明）。
