# 实空间权重约束框架（一期·判决性）开发计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ABACUS 中落地"统一实空间权重约束框架"一期：PW 基组 + Becke 权重的电荷约束（单/双约束），含外环 μ 求解器、记账审计与判决性验证 V1/V2/V3+反假收敛。

**Architecture:** 五层解耦（权重数学核 M0 → 网格权重 M1 → 读数 M2 → PW 势注入 M3a → 外环求解器 M4 + 记账 M5 + I/O M7），注入侧与读数侧共享同一权重实例（观测量=注入算符，dspin 恒等式前提）。冻结权重、μ 只在外环更新。

**Tech Stack:** C++17、GoogleTest、ABACUS PW 栈；复用 `module_grid/partition.h`（Becke 原语）、`module_pot/efield`（veff 注入先例）、`spin_constrain lambda_loop`（外环骨架）、`module_dipole`（网格读数归约先例）。

**依据文档：** `/root/abacus-develop/plan-architecture.md`（架构）+ 4 份评审（`docs/superpowers/specs/2026-08-26-realspace-weight-*.md`）。

**代码纪律（AGENTS.md）：** 每个 if-else 分支前置注释；禁 goto/setjmp；RAII 优先（`std::vector::data()` 优于裸指针）；函数 >300 行拆分；每轮修改+测试产出 `docs/superpowers/specs/YYYY-MM-DD-<topic>.md` 并更新 `deltap-development-log.md`。

---

## 文件结构（一期全部落点）

| 文件 | 职责 |
|---|---|
| `source/source_base/module_grid/partition.h/.cpp` | M0：扩展 Becke 异核修正 χ_ij + 解析位置导数 |
| `source/source_base/module_grid/test/test_partition.cpp` | M0 单测（扩充现有） |
| `source/source_estate/module_constraint/weight_grid.h/.cpp` | M1：网格权重构造+缓存+审计 |
| `source/source_estate/module_constraint/constraint_observe.h/.cpp` | M2：Q_α=∫w_α·d_α 网格读数 |
| `source/source_estate/module_constraint/constraint_inject_pw.h/.cpp` | M3a：PW veff 注入 |
| `source/source_estate/module_constraint/mu_solver.h/.cpp` | M4：逐分量 secant+护栏+熔断 |
| `source/source_estate/module_constraint/constraint_accounting.h/.cpp` | M5：E_con+审计行 |
| `source/source_estate/module_constraint/constraint_io.h/.cpp` | M7：最小输入解析+语义守卫 |
| `source/source_estate/module_constraint/constraint_loop.h/.cpp` | 外环编排（PW esolver 钩子） |
| `source/source_estate/module_constraint/test/*.cpp` | M1/M2/M4/M5/M7 单测 |
| `source/source_estate/module_parameter/`（修改） | 新 INPUT 参数注册 |
| `source/source_esolver/esolver_ks_pw.cpp`（修改 ~10 行） | 外环钩子（仿 226 行 deltaspin 先例） |

---

## Task 1: M0 Becke 异核修正与解析位置导数

**Files:**
- Modify: `source/source_base/module_grid/partition.h`（声明）、`partition.cpp`（实现）
- Test: `source/source_base/module_grid/test/test_partition.cpp`（扩充）

背景：现有 `w_becke(nR0, drR, dRR, nR, iR, c)` 只有距离参数（无半径、无导数）。缺口=异核半径比修正 χ_ij（Becke 1988 size adjustment）与 ∂w_i/∂R_J。

- [x] **Step 1: 写失败测试——异核修正**

在 `test_partition.cpp` 追加（参照现有测试风格）：

```cpp
// 异核修正：R_i > R_j 时中点权重应偏向 j（小原子得权重少、切换面向大原子侧移）
TEST(PartitionTest, BeckeHeteronuclearMidpoint)
{
    // 两中心相距 2 Bohr，网格点在中点：drR={1,1}, dRR=[0,2;2,0]
    double drR[2] = {1.0, 1.0};
    double dRR[4] = {0.0, 2.0, 2.0, 0.0};
    double radii[2] = {1.5, 0.5};   // 大原子、小原子（Bragg-Slater 量级）
    int iR[2] = {0, 1};
    // 无修正时中点 w=0.5；有修正时大原子权重 > 0.5
    double w_big = Grid::Partition::w_becke_adjusted(2, drR, dRR, radii, 2, iR, 0);
    EXPECT_GT(w_big, 0.5);
    EXPECT_NEAR(w_big + Grid::Partition::w_becke_adjusted(2, drR, dRR, radii, 2, iR, 1), 1.0, 1e-12);
}
```

- [x] **Step 2: 写失败测试——解析导数 vs 中心差分**

```cpp
TEST(PartitionTest, BeckeDerivFD)
{
    // 三中心随机几何，∂w_0/∂R_1 的 x 分量解析 vs δ=1e-5 Bohr 中心差分，误差 <1e-6
    // （几何量与 ∂w/∂dRR、∂w/∂drR 链式组装，实现见 Step 4）
    ...
}
```

- [x] **Step 3: 运行测试确认失败**

Run: `cmake --build build --target partition_test && ./build/source/source_base/module_grid/test/partition_test --gtest_filter='PartitionTest.Becke*'`
Expected: 编译错误（`w_becke_adjusted` 未声明）

- [x] **Step 4: 实现**

`partition.h` 追加声明（保持旧接口不动，新接口并存）：

```cpp
// 异核半径修正版 Becke 权重。radii[I] 为中心 I 的分区半径（共价半径表）。
// 修正公式（Becke 1988）：
//   u_ij = (chi_ij - 1)/(chi_ij + 1),  chi_ij = radii[I]/radii[J]
//   a_ij = clip(u_ij/(u_ij*u_ij - 1), -0.5, 0.5)
//   mu'_ij = mu_ij + a_ij * (1 - mu_ij^2)   // 切换面等比例移动
double w_becke_adjusted(int nR0, const double* drR, const double* dRR,
                        const double* radii, int nR, const int* iR, int c);

// 位置导数：∂w_c/∂R_J（J 为 iR 中的中心下标），输出 3 分量。
// 链式：∂w/∂mu_ij = 经 s_becke 连乘求导；∂mu_ij/∂R_J 由 drR/dRR 几何导数给出（闭式）。
void w_becke_adjusted_deriv(int nR0, const double* drR, const double* dRR,
                            const double* radii, const double* eR, // eR[3*I+d]: r->I 方向余弦
                            int nR, const int* iR, int c, int J, double* dw);
```

实现要点（partition.cpp）：复用现有 `s_becke` 与其连乘结构；`s_becke` 导数 `s'` 用 f_3 的解析导数（多项式闭式）；分支均前置注释（AGENTS.md）。

- [x] **Step 5: 运行测试确认通过**

Run: `./build/.../partition_test --gtest_filter='PartitionTest.*'`
Expected: 全部 PASS（含原有测试不回归）

- [x] **Step 6: 写本轮 spec 文档 + 更新日志 + Commit**

`docs/superpowers/specs/2026-08-30-m0-becke-heteronuclear-deriv.md`（测试计划/设置/结果/分析/下一步五段式）。

```bash
git add source/source_base/module_grid/ docs/superpowers/specs/2026-08-30-m0-becke-heteronuclear-deriv.md
git commit -m "feat(grid): Becke heteronuclear size adjustment + analytic position derivatives"
```

---

## Task 2: M1 权重网格模块（构造+缓存+审计）

**Files:**
- Create: `source/source_estate/module_constraint/weight_grid.h`、`weight_grid.cpp`
- Test: `source/source_estate/module_constraint/test/weight_grid_test.cpp`

职责：对每个约束 α，在 PW 密度网格上构造 w_α(r_g)（调 M0），近邻表缓存（Stratmann/截断半径筛掉无关中心），并行域分解沿用电荷网格划分。

- [x] **Step 1: 写失败测试——逐点 sum rule 硬断言**

```cpp
TEST(WeightGridTest, PartitionOfUnity)
{
    // H2O 三原子 + 40 Bohr 立方盒、网格间距 0.2 Bohr（合成网格，不经 SCF）
    WeightGrid wg(ucell, ng, radii, WeightType::Becke);
    wg.build();
    double maxdev = wg.max_partition_deviation();   // max_g |Σ_I w_I(g) - 1|
    EXPECT_LT(maxdev, 1e-10);
}
```

- [x] **Step 2: 写失败测试——对称性与并行一致性**

```cpp
TEST(WeightGridTest, SymmetryAndMPI)
{
    // 对称分子两 H 权重逐点相等（1e-12）；1 rank vs 2/4 rank 逐点一致（含非方网格）
}
```

- [x] **Step 3: 运行确认失败**（模块不存在，编译失败）

- [x] **Step 4: 实现**——`WeightGrid::build()` 双循环（网格点×近邻中心），内部调 `Grid::Partition::w_becke_adjusted`；缓存 `std::vector<std::vector<double>> w_[alpha][ir_local]`；构造后立刻跑 sum rule 审计并把 maxdev 存为成员（供 M5 打印）。每几何重建一次（MD 每步），禁止依赖密度。

- [x] **Step 5: 测试通过 + 基准**：打印 N_g×N_at×N_neigh 实测耗时，确认 < 一步 SCF 的 1%。

- [x] **Step 6: spec 文档 + 日志 + Commit**

```bash
git add source/source_estate/module_constraint/
git commit -m "feat(constraint): M1 weight grid construction with partition-of-unity audit"
```

---

## Task 3: M2 约束读数（已知密度对拍）

**Files:**
- Create: `source/source_estate/module_constraint/constraint_observe.h/.cpp`
- Test: `test/constraint_observe_test.cpp`

职责：`Q_α = Σ_g w_α(g)·d_α(g)·ΔV`，并行归约仿 `source_io/module_dipole/write_dipole.cpp` 的网格归约模式。

- [x] **Step 1: 写失败测试——原子叠加密度解析对拍**

```cpp
TEST(ConstraintObserveTest, AtomicSuperposition)
{
    // 构造 ρ = Σ_I ρ_I^atomic（Becke 权重下解析期望：N_I = 该原子电子数，精确）
    // 断言 Q_I 与期望差 < 1e-8；Σ_I Q_I == N_el（1e-10）
}
```

- [x] **Step 2: 运行确认失败**

- [x] **Step 3: 实现**——`ConstraintObserver::observe(const WeightGrid&, const double* rho, int nspin, std::vector<double>& Q)`；自旋通道 m=ρ↑−ρ↓ 预留接口（一期不接）。片段=原子权重求和后同一函数。

- [x] **Step 4: 测试通过**

- [x] **Step 5: spec 文档 + 日志 + Commit**

```bash
git commit -m "feat(constraint): M2 grid-based constraint observable with analytic benchmark test"
```

---

## Task 4: M4 外环 μ 求解器（合成映射 mock 先行）

**Files:**
- Create: `source/source_estate/module_constraint/mu_solver.h/.cpp`
- Test: `test/mu_solver_test.cpp`

设计（对照 R2/R3 缺口）：逐分量 secant + κ 限幅 + 翻号检测 + 单步限幅 + 联合熔断。一期不做 Broyden。

- [x] **Step 1: 写失败测试——已知根收敛**

```cpp
TEST(MuSolverTest, SecantKnownRoot)
{
    // 合成映射 Q(mu) = 2.0 - 0.5*mu（chi=0.5），target=1.9 → 根 mu*=0.2
    MockResponse mock(2.0, 0.5);
    MuSolver solver({0.0}, {1.9}, MuSolverParams{.step_max=0.05, .mu_max=5.0,
                                                 .kappa_min=0.3, .kappa_max=20.0});
    // 逐步喂 (mu, Q)，断言 6 步内 |Q-t|<1e-6，且 mu 单调逼近 0.2
}
```

- [x] **Step 2: 写失败测试——翻号/死通道熔断**

```cpp
TEST(MuSolverTest, SignFlipClampAndFuse)
{
    // (a) 非单调映射：κ 翻号时断言更新方向被截断/翻转保护触发（不发散）
    // (b) 双侧不可达映射（Q(mu)=Q0 常数）：断言 μ 顶到 mu_max 后状态=UNREACHABLE
    //     且附带残差平台报告，而非无限增大
    // (c) 反假收敛：mu=0、Q==target 时首步即 CONVERGED；Q!=target 时不得报 CONVERGED
}
```

- [x] **Step 3: 运行确认失败**

- [x] **Step 4: 实现**

```cpp
struct MuSolverParams { double step_max=0.05; double mu_max=5.0;
                        double kappa_min=0.3, double kappa_max=20.0;
                        double conv_tol=1e-4; int plateau_window=3; };
enum class MuStatus { RUNNING, CONVERGED, UNREACHABLE };

class MuSolver {
  // 每约束独立历史 (mu_prev, Q_prev)
public:
  MuStatus step(const std::vector<double>& Q, const std::vector<double>& target,
                std::vector<double>& mu);
private:
  // 逐分量：kappa_i = clamp(dQ_i/dmu_i, kappa_min, kappa_max)（符号必须为负，翻号→回退半步+标记）
  // dmu_i = -(Q_i - t_i)/kappa_i；|dmu_i| 截断 step_max
  // 熔断：|mu_i|>mu_max 且最近 plateau_window 步残差下降 <1% → UNREACHABLE（附 Q(mu) 端点）
};
```

所有分支前置注释（发散护栏的每个 if 写明对应历史事件：T-4a' 翻号、T-5' 顶限）。

- [x] **Step 5: 测试通过**

- [x] **Step 6: spec 文档 + 日志 + Commit**

```bash
git commit -m "feat(constraint): M4 component-wise secant mu solver with sign-flip guard and fuse"
```

---

## Task 5: M7 最小 I/O + 语义守卫

**Files:**
- Modify: `source/source_estate/module_parameter/`（参数注册，先 grep `deltaspin` 在该目录的注册位置照抄模式）
- Create: `source/source_estate/module_constraint/constraint_io.h/.cpp`、测试

- [x] **Step 1: 失败测试——schema 与守卫**

```cpp
TEST(ConstraintIOTest, GuardsAndDefaults)
{
    // weight_type=becke（一期唯一合法值；hirshfeld 报"未实现"而非静默跑）
    // target_mode 缺省=delta；absolute 模式触发 WARNING（口径已从 ~e 变为 ~0.2-0.3 e，R12）
    // mu_max 缺省 5.0 Ry；缺 target → WARNING_QUIT（不允许"无 target 隐式约束"——DeltaP 4.3 教训）
}
```

- [x] **Step 2-4: 实现参数**：`constraint`（bool）、`constraint_type=charge`、`weight_type=becke`、`constraint_target_file`（JSON，复用 `module_deltaspin/sc_parse_json.cpp` 解析模式）、`constraint_target_mode=delta|absolute`、`constraint_mu_max`、`constraint_thr=1e-4`。

- [x] **Step 5: spec 文档 + 日志 + Commit**

---

## Task 6: M3a PW 势注入

**Files:**
- Create: `source/source_estate/module_constraint/constraint_inject_pw.h/.cpp`
- Test: `test/constraint_inject_pw_test.cpp`

先例：`source_estate/module_pot/efield.cpp`（`Efield::add_efield` 计算网格势、`v_eff += ...` 注入模式）。

- [x] **Step 1: 失败测试——单步注入逐点正确**

```cpp
TEST(ConstraintInjectPWTest, PointwiseInjection)
{
    // 固定 veff 数组 + 给定 mu/w，调用注入后断言：
    // veff_new(ir) == veff_old(ir) + Σ_α mu[α]*w[α][ir]（机器精度，nspin=1）
    // 共享实例断言：注入侧与读数侧 w 指针相同（观测量=注入算符，原则 2）
}
```

- [x] **Step 2-4: 实现**——`ConstraintInjectPW::inject(const WeightGrid&, const std::vector<double>& mu, ModuleBase::matrix& veff)`；nspin=2 时自旋通道 ±μ 预留（一期只接 charge）。

- [x] **Step 5: spec 文档 + 日志 + Commit**

---

## Task 7: M5 记账与审计行

**Files:**
- Create: `constraint_accounting.h/.cpp`、测试

- [x] **Step 1: 失败测试**

```cpp
TEST(AccountingTest, EsconAndAudit)
{
    // E_con = Σ_α mu_α (Q_α - t_α)（沿用 escon 形式）
    // 审计行格式化输出：Σ_I N_I vs N_el、残差、mu、maxdev（M1），机器可读（key=value）
}
```

- [x] **Step 2-4: 实现 + 接入** elecstate 总能量路径（仿 deltaspin escon 汇入 `elecstate_pw.cpp` 的先例）。

- [x] **Step 5: spec 文档 + 日志 + Commit**

---

## Task 8: 外环编排 + PW esolver 接线

**Files:**
- Create: `constraint_loop.h/.cpp`
- Modify: `source/source_esolver/esolver_ks_pw.cpp`（仿 226 行 `pw::run_deltaspin_lambda_loop(iter-1, this->drho, PARAM.inp)` 先例，加 `constraint::run_constraint_loop(iter-1, ...)`）

- [x] **Step 1: 失败测试（集成冒烟）**——H₂O PW 单点 + 单约束 delta=+0.1 e：跑通 SCF、审计行出现、μ 非零。

- [x] **Step 2: 运行确认失败**（钩子不存在）

- [x] **Step 3: 实现编排**：预处理建 M1 → 参考态 SCF 记 Q_ref → 约束 SCF（每 iter 后注入 M3a、读数 M2）→ SCF 收敛后 M4.step → 收敛/熔断退出。两阶段门控仿 deltaspin sc_scf_thr_mode。

- [x] **Step 4: 冒烟通过 + 回归**（`ctest -R deltaspin` 等不回归）

- [x] **Step 5: spec 文档 + 日志 + Commit**

---

## Task 9–11: 判决性验证（一期验收门）

- [x] **Task 9 V1 sum rule**：H₂O 无约束 SCF，审计行 Σ_I N_I ≡ 8（<1e-8）。
- [ ] **Task 10 V2 口径基准**：同一收敛密度导出 cube，Multiwfn 算 Becke 电荷对拍（消电子结构差异，纯验分账规则，判据 1e-4 e）。
  - **阻塞（外部工具）**：本环境无 Multiwfn，未闭环 → 勾选失信修正（评审 P1）。
    协议已备：ABACUS `out_chg` 导出收敛密度 cube → Multiwfn "Becke 电荷"
    逐原子对拍；内部覆盖由 M2 高斯基准（1e-8）承担。V2 作为未清债务转入
    二期前置项：在 Multiwfn 补拍完成前，不作"基组无关/口径正确"的对外声明
    （尤其 PW≡LCAO 可发表主张）。
- [x] **Task 11 V3 可达性 + 反假收敛**：delta 模式 q=±0.05/0.1/0.2/0.3 e 扫描，全部达标无封顶；μ=0 自由跑不得收敛到任何非自然靶点；故意设不可达靶点触发熔断。
- [x] **Task 12 止损判决**：若 V3 物理可达域过窄（κ 封顶，R4），记录 Q(μ) 端点曲线并止损归档；否则开二期。

---

## 二期/三期大纲（详见 plan-architecture.md §4，本计划不展开）

- 二期：M3b（LCAO Gint 新通道，按新建估）+ M6 力/力矩（stationary4 FD + 对标 DeltaSpin 力矩 FD）+ 自旋通道 + PW≡LCAO 逐位一致。
- 三期：应力（先补推导，R6）+ Hirshfeld 权重 + Broyden（R3）+ 偶极 V4（R5 细化协议）+ 输出规范。

---

## 自检（Self-Review）

- Spec 覆盖：架构文档 M0–M7 一期模块全部有对应 Task（M0=T1，M1=T2，M2=T3，M4=T4，M7=T5，M3a=T6，M5=T7，编排=T8，V 项=T9–12）✔；二/三期为架构既定大纲，不在本计划展开 ✔。
- 占位符扫描：T9–T11 为验收实验而非代码任务，参数与判据已给具体值 ✔。
- 类型一致性：`WeightGrid`/`MuSolver`/`MuStatus` 命名全文一致 ✔。
