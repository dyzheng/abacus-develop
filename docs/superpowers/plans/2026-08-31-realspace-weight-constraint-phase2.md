# 实空间权重约束框架（二期）开发计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 二期交付——LCAO 基组通道（M3b）、力/力矩导数（M6）、自旋通道（±μ），达成 PW≡LCAO 逐位一致 + 力/力矩 FD 验收；同步清零一期评审债务（P1/P2/P4 + V2 重定义落地）。

**Architecture:** 沿用一期五层架构（plan-architecture.md），不新增层、不新增模块。M6 力公式 F_J=∫ρ ∂w/∂R_J dr 为纯实空间网格运算——**双基组同一核**（权重定义在网格上，与基组解耦的架构红利在此兑现）。

**硬约束（用户指定）：严格不引入原框架外的任何新组件、新算法、外部依赖。** 合规映射见 §4 检查表。明确不做（全部三期）：Hirshfeld 权重、Broyden、偶极/多极子、应力（R6 待证）、Hirshfeld-I。

**依据文档：** plan-architecture.md、`2026-08-31-phase1-review.md`（P1–P4 债务）、`2026-08-31-v2-redefinition.md`（V2 重定义）、`2026-08-31-v1-v3-validation.md`。

---

## 文件结构（二期全部落点）

| 文件 | 职责 | 新/改 |
|---|---|---|
| `docs/superpowers/plans/2026-08-30-realspace-weight-constraint-phase1.md` | P1：Task 10 勾选修正 | 改 1 行 |
| `source/source_estate/module_constraint/constraint_loop.cpp` | P2：inject 返回值契约 | 改 ~5 行 |
| `source/source_estate/module_constraint/constraint_inject_pw.h` | P2：契约注释对齐 | 改注释 |
| `source/source_estate/module_constraint/constraint_loop.cpp:113` | 注释与分支错位修正 | 改注释 |
| `source/source_estate/module_constraint/test/constraint_observe_test.cpp` | V2b：C++ 独立参考实现（不用 numpy） | 扩充 |
| `tests/constraint_pw_h2o/` + `tests/CMakeLists.txt` | P4：集成用例注册 ctest | 改 |
| `source/source_estate/module_constraint/constraint_inject_lcao.h/.cpp` | M3b：W^α_μν Gint 核 | 新建 |
| `source/source_estate/module_constraint/weight_grid.h/.cpp` | M6 前置：∂w/∂R_J 导数网格缓存 | 扩充 |
| `source/source_estate/module_constraint/constraint_deriv.h/.cpp` | M6：力核（双基组共用） | 新建 |
| `source/source_estate/module_constraint/test/constraint_inject_lcao_test.cpp`、`constraint_deriv_test.cpp` | 新单测 | 新建 |
| `source/source_esolver/esolver_ks_lcao.cpp` | LCAO 三处薄钩子（仿 spinconstrain 先例 446/566/600 行） | 改 ~30 行 |
| `source/source_estate/module_constraint/constraint_io.cpp` | 自旋通道守卫（nspin=2、constraint_type=spin） | 扩充 |
| `tests/constraint_lcao_h2o/`、`tests/constraint_pw_h2o_spin/` | 集成用例（注册 ctest） | 新建 |

---

## Task 2.1: 一期债务清零（P1/P2/P4 + V2 落地）

**Files:** 见上表前六行

- [x] **Step 1: P2 修复——inject 返回值契约**（评审修复轮 293f53aa8 完成）（先于一切，5 分钟）

`constraint_loop.cpp:88-89` 改为检查返回值：

```cpp
// 注入失败（约束数不匹配）属内部状态错误：按头文件契约立即中止，
// 不得静默继续（契约见 constraint_inject_pw.h:27-29）
if (!constraint_inject_pw::inject(*wg_, mu_, veff))
{
    ModuleBase::WARNING_QUIT("ConstraintLoop::inject_potential",
                             "constraint potential injection failed: mu/weight size mismatch");
}
```

同步修正 `constraint_loop.cpp:113` 注释错位（"delta mode" 注释移到 else 分支，或改写注释匹配 `if (absolute)`）。

- [x] **Step 2: V2a 网格收敛（零新代码，纯跑测试）**（量化门通过；严格单调因切换面对齐振荡不成立，口径偏差登记见 v2a-grid-convergence.md）

用 `tests/constraint_pw_h2o/` 同一输入，ecutrho=80/160/320 三档重跑参考态，比较 Q_I：
判据：相邻档差 <1e-4 e 且单调收敛。结果记入 `docs/superpowers/specs/2026-08-31-v2a-grid-convergence.md`。

- [x] **Step 3: V2b C++ 独立参考（不用 numpy，遵守零外部依赖）**（IndependentReferenceBecke，1e-8 PASS）

在 `constraint_observe_test.cpp` 追加：测试内**从 Becke 1988 原始公式独立重写**权重（手写 f_3 迭代多项式与连乘，不调 `Grid::Partition::w_becke_adjusted`），在测试自建的合成密度上与被测 M1+M2 链路对拍逐原子 Q_I，判据 1e-8。独立性=独立代码路径；约定健全性由 V2c（文献带宽 ~0.05 e）宽松核对。

```cpp
// 独立参考：直接从 Becke 1988 公式重写（故意不调用被测实现）
static double ref_f3(double mu) { /* f1(x)=(3x-x^3)/2，迭代 3 次 */ }
static double ref_s(double mu)  { return 0.5*(1.0 - ref_f3(mu)); }
// ref_weight(i, r) = Π_{j≠i} ref_s(mu_ij) / Σ_k Π_{j≠k} ref_s(mu_kj)
TEST(ConstraintObserveTest, IndependentReferenceBecke) { /* 对拍 Q_I < 1e-8 */ }
```

- [x] **Step 4: P4 集成用例注册 ctest**（评审修复轮 293f53aa8：迁入 01_PW 并注册）

`tests/constraint_pw_h2o/` 接入 tests/ 集成测试体系（照抄同目录其他 PW 用例的注册模式），README 去掉绝对路径。

- [x] **Step 5: P1 勾选修正**（Task 10 已勾回并标注 V2a+V2b 替代完成）

V2a+V2b 通过后，phase1 计划 Task 10 改注"V2=V2a+V2b 替代完成（Multiwfn 降级可选项）"再勾回 `[x]`。

- [ ] **Step 6: spec + 日志 + Commit**

```bash
git commit -m "fix(constraint): phase-1 review debts P1/P2/P4 + V2 internal closed-loop validation"
```

---

## Task 2.2: M3b LCAO 约束矩阵 Gint 核

**Files:**
- Create: `constraint_inject_lcao.h/.cpp`
- Test: `test/constraint_inject_lcao_test.cpp`

职责：W^α_μν = ∫ φ_μ(r) w_α(r) φ_ν(r) dr，每几何一次（对 μ 线性，每 SCF 迭代仅 μ 加权稀疏加）。**严格复用 module_gint 现有积分基建**（gint_env_gamma/gint_env_k 双变体），不写新积分框架。

- [x] **Step 1: 失败测试——与网格直积对拍**（constraint_inject_lcao_test.cpp 4 测试，见 2026-08-31-m3b spec）

```cpp
TEST(ConstraintInjectLCAOTest, MatrixElementVsDirectGrid)
{
    // 单原子单轨道玩具：φ 为数值原子轨道，w 已知；
    // W_μν 经 Gint 核计算 vs 同一网格上 φ_μ*w*φ_ν 直接求和，相对差 <1e-10
    // sum rule 推论：Σ_α W^α_μν == ⟨φ_μ|φ_ν⟩（重叠矩阵元，机器精度）
}
```

- [x] **Step 2: 运行确认失败**（反向验证：移除 build() 内核调用 → 恰两个核相关测试 FAIL，恢复 4/4 PASS）

- [x] **Step 3: 实现**——`ConstraintInjectLCAO::build` 复用生产 `cal_gint_vl` vlocal 核（权重场扮演局域势），输出 HContainer（与现有 LCAO 哈密顿容器同型，直接可加进 H）；`add_weighted` 每 SCF 迭代 μ 加权稀疏加。gamma/k 共用实空间核（k 点由 esolver H(R)→H(k) 变换处理，与 Veff 完全同构）。分支前置注释。

- [x] **Step 4: 测试通过 + sum rule 推论验证**（Σ_α W^α = S，实测 2.0e-15；W↔直积 3.4e-15，均优于判据；constraint ctest 10/10 全绿）

- [x] **Step 5: spec + 日志 + Commit**

```bash
git commit -m "feat(constraint): M3b LCAO constraint matrix via existing Gint infrastructure"
```

---

## Task 2.3: LCAO esolver 接线（薄钩子 + 一期技术债下移）

**Files:** Modify `esolver_ks_lcao.cpp`（~30 行）、`constraint_loop.h/.cpp`

先例：`esolver_ks_lcao.cpp:446/566/600` 的 `spinconstrain::SpinConstrain<TK>::getScInstance()` + `run_lambda_loop(iter-1)` 钩子模式。

- [ ] **Step 1: 失败测试**——LCAO H₂O 冒烟（delta=+0.1 e，6±2 外步内 CONVERGED）。
- [ ] **Step 2: 运行确认失败**（LCAO 钩子不存在）
- [ ] **Step 3: 实现**——before_scf（配置+W^α 预建）/hamilt2rho 内 H += Σμ_α W^α/iter_finish（读数+外步+cc_escon 汇入，镜像 PW 的 fp_energy 路径）。**顺带还一期技术债**：把 PW 侧 before_scf 的 ~50 行配置块下移至 `constraint_io`/`constraint_loop` 模块内函数，PW/LCAO 共用（review 轻微项）。
- [ ] **Step 4: 冒烟通过 + 回归**（`ctest -R "constraint|lcao"` 全绿）
- [ ] **Step 5: spec + 日志 + Commit**

---

## Task 2.4: 自旋通道（nspin=2，±μ）

**Files:** `constraint_observe.cpp`（m 通道）、`constraint_inject_pw.cpp` / `constraint_inject_lcao.cpp`（±μ）、`constraint_io.cpp`（守卫）、测试

一期已预留接口（M2 的 DensityChannel、M3a 的 charge-only nspin=2 测试）。二期接通：

- [ ] **Step 1: 失败测试**

```cpp
TEST(ConstraintSpinTest, SplitInjection)
{
    // constraint_type=spin：V_↑ += μ·w、V_↓ −= μ·w；读数通道 m=ρ↑−ρ↓
    // 守卫：nspin!=2 + constraint_type=spin → WARNING_QUIT（不静默跑错）
}
```

- [ ] **Step 2-4: 实现 + 守卫 + 集成**（`tests/constraint_pw_h2o_spin/` 注册 ctest）
- [ ] **Step 5: spec + 日志 + Commit**

---

## Task 2.5: M6 力核（双基组同一实现）

**Files:**
- Modify: `weight_grid.h/.cpp`（∂w/∂R_J 导数网格，调 M0 已交付的 `w_becke_adjusted_deriv`）
- Create: `constraint_deriv.h/.cpp` + 测试

**架构要点**：力公式 F_J = −Σ_α μ_α ∫ ρ(r) ∂w_α/∂R_J dr 是纯网格运算，**PW/LCAO 共用同一核**——不需要 gint_dvlocal 的矩阵元导数链（那是基组空间路径，本框架绕开）。LCAO 的 ρ 网格现成（Hartree/XC 同网格）。

- [ ] **Step 1: 失败测试——解析力 vs 合成密度解析期望**

```cpp
TEST(ConstraintDerivTest, ForceOnSyntheticDensity)
{
    // 双原子 + 高斯密度（解析可积），F_J 解析期望 vs 力核网格积分 <1e-8 Ha/Bohr
    // 牛顿第三定律自检：Σ_J F_J == 0（1e-10）
}
```

- [ ] **Step 2-4: 实现 + 单测通过 + 接入力输出**（PW/LCAO 各自的 FORCE 汇总点，仿 dp/onsite 力汇入先例）
- [ ] **Step 5: spec + 日志 + Commit**

---

## Task 2.6: 二期判决验证（三个 FD/一致性验收）

**严格按历史处方，缺一即假 FAIL：**

- [ ] **Step 1: PW≡LCAO 逐位一致**——同一 H₂O、同一约束靶点，两基组 Q_α 读数逐位一致（<1e-8）；μ* 差 <1%。**二期核心交付（增量 1 可发表点的数据）**。
- [ ] **Step 2: 力 FD（stationary4 协议）**——冻结 t*、δ=0.005 Bohr、判据 0.0129 eV/Å；网格前提写死：ecutwfc=100 + ecutrho≥400 + scf_thr=1e-8（R7）。PW 与 LCAO 各过一遍。
- [ ] **Step 3: 力矩 FD（对标 DeltaSpin）**——扰动靶点磁矩 δM，λ_解析（外环收敛 μ 值）vs ∂E/∂M 数值微分，判据 <0.006 eV/μB（"能力与 DeltaSpin 一致"判决实验）。
- [ ] **Step 4: 反假收敛 + MPI**——自旋通道 μ=0 自由跑不得收敛到非自然靶点；LCAO 4-rank（含非方网格）逐位一致。
- [ ] **Step 5: spec（`2026-XX-XX-phase2-validation.md`）+ 日志 + Commit**

---

## Task 2.7: 二期判决门

- [ ] 全部 Step 通过 → 二期闭合，一期 V2c'（Multiwfn 可选）与三期（应力/Hirshfeld/Broyden/偶极）评估续行；
- [ ] 力 FD 超判据 → 按 DeltaP 先例归因链排查（记账恒等式 → 驻点条件 → 网格精度），不豁免判据；
- [ ] PW≡LCAO 不一致 → 排查 Gint 网格精度与密度网格口径，**不降低判据**（该一致性是框架核心卖点，放水即否决架构）。

---

## §4 零新依赖合规检查表

| Task | 复用的现成基建 | 新组件？ |
|---|---|---|
| 2.1 | 现有测试体系、Becke 1988 公式（参考文献非依赖） | 否（V2b 用 C++ 内测，不用 numpy） |
| 2.2 | module_gint（gamma/k 双变体现成）、HContainer 现成 | 否 |
| 2.3 | esolver_ks_lcao spinconstrain 钩子先例 | 否 |
| 2.4 | 一期 M2/M3a 预留接口、DeltaSpin ±μ 语义 | 否 |
| 2.5 | M0 已交付导数核、网格密度现成 | 否 |
| 2.6 | stationary4 协议（历史工具）、现有 FD 方法论 | 否 |

**显式禁止清单**：Hirshfeld/promolecule、Broyden、偶极/多极子权重、应力、Hirshfeld-I、第三方工具（Multiwfn/numpy）、新积分框架、新 JSON 库。
