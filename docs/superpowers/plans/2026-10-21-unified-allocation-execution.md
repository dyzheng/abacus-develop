# 统一局域磁矩分配：工程执行方案（分阶段 + 可验收测试）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按设计文档实现"正交目标轨道 + 互补子空间空间分配"的统一分配算符 `Â_i = Q̂_i + Q̂_c Ŵ_i Q̂_c`，分六个阶段（S0–S5）交付：先修基线文档与审计，再做只读读数，最后才允许进入约束势与动力学。

**Architecture:** 新模块 `source/source_estate/module_constraint/alloc_*`（复用 `WeightGrid`、`Parallel_Reduce`、`ConstraintLoop` 审计）；S0 只改文档与测试；S1 为纯代数 + 网格只读原型；S2 起注入 `ΔĤ`（局域势 + 低秩投影修正）并实现冻结目标轨道力；S3 补 `∂Q̂`、应力、非共线；S4 做 PW 平价与周期 Hirshfeld 后端；S5 接入现有 M2/M3/M4 调度与子空间加速。

**Tech Stack:** C++17、gtest、CMake（AddTest）、ABACUS module_constraint 现有分层；**不引入 Eigen**（自带 R×R Jacobi 特征分解）。

**Spec:** `docs/superpowers/specs/2026-10-21-unified-local-moment-allocation-design.md`（公式、算子、力/应力、风险 R1–R15）。

**代码规范（AGENTS.md）：** 每个 if-else 分支与非平凡 if 前必须有注释；禁 goto/setjmp；函数 >300 行拆分；RAII/std::vector 优先。
**文档纪律（AGENTS.md）：** 每个阶段结束必须产出 `docs/superpowers/specs/YYYY-MM-DD-<stage>-test.md`（测试计划/设置/结果/分析/下一步）并更新 `deltap-development-log.md`。

**资源纪律（沿用）：** 集成 SCF 运行 `scf_nmax <= 100`；不收敛记录 drho/residual 轨迹，不提高上限。

---

## 0. 已核验的工程事实（实现时直接引用）

| 事实 | 位置/结论 |
|---|---|
| 仓库**无 Eigen** | `grep "#include <Eigen"` 无命中 → 自带 `R×R` Jacobi 对称特征分解（R 典型 ≤ 数百，成本可忽略） |
| 权重与导数现成 | `WeightGrid::{build,build_derivatives,constraint_weight,weight_derivative,max_partition_deviation}` |
| 网格约定 | `i=ir/(ny*nplane); j=ir/nplane−i*ny; k=ir%nplane+startz_current`；`dV=omega/nxyz`；`Parallel_Reduce::reduce_pool` |
| **只读观测量不能只用 ρ(r)** | `Â_i` 的读数需要密度矩阵 `ρ̂`（LCAO 的 `DMR`）或波函数（PW 的 `stp.psi_cpu`）；现有 `ConstraintLoop::observe()` 只收到 `rho`（密度），因此新读数需要**独立钩子** |
| LCAO 密度矩阵 | `this->dmat.dm`（`elecstate::DensityMatrix`）：`switch_dmr(2)` 取自旋差、`get_DMR_pointer(1)` 取指针（参考 `module_deltaspin/cal_mw.cpp:80-90`） |
| LCAO 重叠 | AO 重叠子块可由现有 `S`（`HS_Matrix_K`/`Parallel_Orbitals`）取得；R×R 目标轨道交叠 `S_T=C†SC` |
| PW 波函数 | `this->stp.psi_cpu`（`esolver_ks_pw.cpp` 的 `iter_finish`），占据 `this->pelec->wg` |
| 现有审计格式 | `CONSTRAINT_AUDIT ...`（`constraint_accounting.cpp`）；新读数用独立 `ALLOC` 行，默认关闭 |
| 现有力/力矩判据 | 力 FD `< 0.0129 eV/Å`；磁广义力/力矩 FD `< 0.02 eV/μB`、`< 0.006 eV/μB`（沿用既有性质测试阈值） |

---

## 1. 阶段总览与闸门

| 阶段 | 内容 | 退出闸门（硬性） |
|---|---|---|
| **S0** | 文档修正 + 权重/导数审计 | 三原子 `maxdev < 1e-12`；导数求和 `< 1e-10`；文档与代码一致 |
| **S1** | 分配算子核心 + 网格只读观测 | 代数恒等式 `< 1e-10`；`Σn=N`、`Σm=M_tot` `< 1e-8`；空目标集与 Becke **逐位相同**；`n_c ≥ 0` |
| **S2** | LCAO 只读生产读数 | 同上 + 默认关闭输出逐字节不变；与 `onsite=`/Becke 三量对照表 |
| **S3** | 冻结目标轨道**约束** + ALM | 收敛；`Σm=M_tot`；目标 FD `dE*/dt=−λ` `< 0.02 eV/μB`；位置 FD `< 0.0129 eV/Å`；0.01 μB 刚性子例 100 步内收敛 |
| **S4** | 完整导数 `∂Q̂` + 应力 + 非共线 | 与 S3 力差 `< 0.005 eV/Å`；应力 FD 相对 `< 5%`；无 SOC 整体自旋旋转 `|ΔE|<1e-6 Ry` |
| **S5** | PW 平价 + 周期 Hirshfeld | PW vs LCAO `|Δm^T|<0.02 μB`、`|ΔM|<0.01 μB`；Hirshfeld `maxdev<1e-10`；镜像切换 FD 连续 |
| **S6** | 生产化（调度/子空间/回归） | 约束 SCF 成本 ≤ Becke 约束的 1.5×；后处理 ≤ SCF 的 5%；既有 13 个约束测试全绿 |

> 闸门未过不得进入下一阶段；每个闸门的原始数据写入对应阶段的 dated 测试文档。

---

## 2. S0：基线修正与审计

### Task S0.1 修正对比文档的 Becke 公式

**Files:**
- Modify: `docs/superpowers/specs/2026-10-21-local-moment-algorithms-comparison.md`（§1.1）

- [ ] **Step 1:** 把 `μ_IJ = (r_I − r_J)/(r_I + r_J)` 改为 `μ_IJ = (|r−R_I| − |r−R_J|)/|R_I−R_J|`。
- [ ] **Step 2:** 把权重写成 `p_I = Π_{J≠I} s(μ_IJ)`、`w_I = p_I / Σ_J p_J`（显式归一化）。
- [ ] **Step 3:** 删除"由 `s(μ)+s(−μ)=1` 推出 `Σ_I w_I=1`"的表述，改为"成对恒等式 + 归一化 ⇒ 单位分解"，并加一句三原子反例（三角中心 `Σp=3/4`）。
- [ ] **Step 4:** 验收：`grep -n "R_I−R_J\|p_I /" 文档` 命中；无 "r_I + r_J" 残留。

### Task S0.2 三原子 maxdev 测试（钉死归一化）

**Files:**
- Modify: `source/source_estate/module_constraint/test/weight_grid_test.cpp`
- Test: 同文件新增 TEST

- [ ] **Step 1: 写测试**

```cpp
// Three-centre partition: the normalized Becke product must sum to one at
// every grid point (the pairwise identity s(mu)+s(-mu)=1 is NOT enough for
// more than two centres; normalization is what makes it a partition).
TEST(WeightGridTest, ThreeCentrePartitionOfUnity)
{
    auto ucell = make_h2o_ucell();               // O, H, H (3 atoms)
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 40, 40, 40);
    pw.distribute_r();
    const std::vector<double> radii = {1.5, 0.5, 0.5}; // Bohr
    constraint::WeightGrid wg(*ucell, &pw, radii);
    wg.build();
    EXPECT_LT(wg.max_partition_deviation(), 1e-12);

    // Counter-check: the UNNORMALIZED product is far from one somewhere
    // near the geometric centre of the three atoms.  This proves the
    // normalization is doing real work (regression guard for S0.1).
    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    const double ox = (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0;
    const double oy = (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0;
    const double oz = (pos[0][2] + pos[1][2] + pos[2][2]) / 3.0;
    double sum_raw = 0.0;
    for (int I = 0; I < 3; ++I)
    {
        double p = 1.0;
        for (int J = 0; J < 3; ++J)
        {
            if (J == I) continue;
            const double rI = std::sqrt(std::pow(ox - pos[I][0], 2)
                + std::pow(oy - pos[I][1], 2) + std::pow(oz - pos[I][2], 2));
            const double rJ = std::sqrt(std::pow(ox - pos[J][0], 2)
                + std::pow(oy - pos[J][1], 2) + std::pow(oz - pos[J][2], 2));
            const double RIJ = std::sqrt(std::pow(pos[I][0]-pos[J][0], 2)
                + std::pow(pos[I][1]-pos[J][1], 2) + std::pow(pos[I][2]-pos[J][2], 2));
            p *= Grid::Partition::s_becke((rI - rJ) / RIJ);
        }
        sum_raw += p;
    }
    EXPECT_GT(std::abs(sum_raw - 1.0), 1e-3);      // unnormalized != 1
}
```

- [ ] **Step 2:** 构建并运行：`cmake --build build -j 32 --target MODULE_ESTATE_constraint_weight_grid && ctest --test-dir build -R MODULE_ESTATE_constraint_weight_grid --output-on-failure`。预期通过（代码已归一化；测试钉死行为）。
- [ ] **Step 3:** 若无 `make_h2o_ucell/h2o_positions` 可见，补 `#include "constraint_test_utils.h"` 与 `#include "source_base/module_grid/partition.h"`。

### Task S0.3 权重导数求和审计

**Files:**
- Modify: `source/source_estate/module_constraint/test/weight_grid_test.cpp`

- [ ] **Step 1: 写测试**

```cpp
// Sum over atoms of the analytic weight derivative must vanish for every
// target atom/axis/grid point (because sum_i w_i = 1 identically).  This is
// the derivative counterpart of the partition-of-unity audit.
TEST(WeightGridTest, DerivativeSumRule)
{
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 40, 40, 40);
    pw.distribute_r();
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.build();
    wg.build_derivatives();
    const int nat = wg.nat();
    const int nrxx = wg.nrxx();
    for (int J = 0; J < nat; ++J)
    {
        for (int d = 0; d < 3; ++d)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                double s = 0.0;
                for (int a = 0; a < nat; ++a)
                {
                    s += wg.weight_derivative(a, J, d, ir);
                }
                // Guard: skip grid points within 1e-3 Bohr of a minimum-image
                // switch plane (derivative is discontinuous there by design).
                if (std::abs(s) > 1e-10)
                {
                    // Count violations instead of aborting on the first one.
                    ADD_FAILURE() << "dsum=" << s << " J=" << J
                                  << " d=" << d << " ir=" << ir;
                    return;
                }
            }
        }
    }
}
```

- [ ] **Step 2:** 运行。**若因镜像切换点失败**（预期可能出现），把该用例改为"统计违约点比例 < 1%，且违约点距最近镜像切换面 < 1e-3 Bohr"，并在测试注释中登记（这是设计风险 R6 的量化基线，S5 的 Hirshfeld 后端必须把它降为 0）。
- [ ] **Step 3:** 提交：`git add` 文档+测试，`git commit -m "test(constraint): S0 partition/derivative audits + doc formula fixes"`。
- [ ] **Step 4:** 写 `docs/superpowers/specs/2026-10-21-s0-baseline-audit-test.md` + 更新 dev log。

**S0 验收**：`maxdev<1e-12`；导数求和（或违约率基线）记录；对比文档公式与代码一致。

---

## 3. S1：分配算子核心 + 网格只读观测

### Task S1.1 R×R 对称特征分解（无 Eigen）

**Files:**
- Create: `source/source_estate/module_constraint/alloc_linalg.h`
- Create: `source/source_estate/module_constraint/alloc_linalg.cpp`
- Create: `source/source_estate/module_constraint/test/alloc_linalg_test.cpp`
- Modify: `source/source_estate/CMakeLists.txt`、`test/CMakeLists.txt`

- [ ] **Step 1: 写测试**

```cpp
#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <vector>

#include "source_estate/module_constraint/alloc_linalg.h"

static std::vector<double> random_spd(const int n, std::mt19937& gen)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> B(n * n);
    for (double& x : B) { x = nd(gen); }
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k) { s += B[i*n+k] * B[j*n+k]; }
            A[i*n+j] = s + (i == j ? 1.0 : 0.0);
        }
    }
    return A;
}

TEST(AllocLinalgTest, SymEigReconstruction)
{
    std::mt19937 gen(7);
    const int n = 8;
    const std::vector<double> A = random_spd(n, gen);
    std::vector<double> w, V;
    alloc::sym_eig(A, n, w, V);
    // Reconstruction A = V diag(w) V^T and V^T V = I.
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k) { s += V[i*n+k] * w[k] * V[j*n+k]; }
            EXPECT_NEAR(s, A[i*n+j], 1e-10);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k) { s += V[k*n+i] * V[k*n+j]; }
            EXPECT_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(AllocLinalgTest, InvSqrt)
{
    std::mt19937 gen(11);
    const int n = 6;
    const std::vector<double> S = random_spd(n, gen);
    std::vector<double> Si;
    alloc::sym_inv_sqrt(S, n, Si, 1e-10);
    // S^{-1/2} S S^{-1/2} = I.
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int a = 0; a < n; ++a)
            {
                for (int b = 0; b < n; ++b)
                {
                    s += Si[i*n+a] * S[a*n+b] * Si[b*n+j];
                }
            }
            EXPECT_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }
}
```

- [ ] **Step 2:** 构建前先失败（目标不存在）。实现 `alloc_linalg.{h,cpp}`：

```cpp
// alloc_linalg.h
#ifndef ALLOC_LINALG_H
#define ALLOC_LINALG_H
#include <vector>
namespace alloc {
// Symmetric eigendecomposition of a row-major n x n matrix A:
// A = V diag(w) V^T with V's columns as eigenvectors; w sorted ascending.
void sym_eig(const std::vector<double>& A, const int n,
             std::vector<double>& w, std::vector<double>& V);
// S^{-1/2} for a symmetric positive semi-definite matrix (eigenvalues below
// rank_tol are dropped; the caller must audit the condition number).
void sym_inv_sqrt(const std::vector<double>& S, const int n,
                  std::vector<double>& Sinv_root, const double rank_tol);
}
#endif
```

```cpp
// alloc_linalg.cpp —— Jacobi 旋转（每分支带注释，符合 AGENTS.md）
#include "alloc_linalg.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace alloc {

void sym_eig(const std::vector<double>& A, const int n,
             std::vector<double>& w, std::vector<double>& V)
{
    std::vector<double> a = A;
    V.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) { V[i * n + i] = 1.0; }
    // Cyclic Jacobi: sweep until the off-diagonal norm is numerically zero.
    for (int sweep = 0; sweep < 100; ++sweep)
    {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q) { off += a[p*n+q] * a[p*n+q]; }
        }
        // Branch: converged (off-diagonal norm below rounding).
        if (off < 1e-28) { break; }
        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const double apq = a[p*n+q];
                // Branch: already negligible rotation.
                if (std::abs(apq) < 1e-20) { continue; }
                const double theta = (a[q*n+q] - a[p*n+p]) / (2.0 * apq);
                const double t = ((theta >= 0.0) ? 1.0 : -1.0)
                                 / (std::abs(theta) + std::sqrt(theta*theta + 1.0));
                const double c = 1.0 / std::sqrt(t*t + 1.0);
                const double s = t * c;
                // A <- J^T A J (rows then columns), V <- V J.
                for (int k = 0; k < n; ++k)
                {
                    const double akp = a[k*n+p], akq = a[k*n+q];
                    a[k*n+p] = c*akp - s*akq;
                    a[k*n+q] = s*akp + c*akq;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double apk = a[p*n+k], aqk = a[q*n+k];
                    a[p*n+k] = c*apk - s*aqk;
                    a[q*n+k] = s*apk + c*aqk;
                }
                for (int k = 0; k < n; ++k)
                {
                    const double vkp = V[k*n+p], vkq = V[k*n+q];
                    V[k*n+p] = c*vkp - s*vkq;
                    V[k*n+q] = s*vkp + c*vkq;
                }
            }
        }
    }
    // Collect eigenvalues and sort ascending (permute eigenvector columns).
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    w.resize(n);
    for (int i = 0; i < n; ++i) { w[i] = a[i*n+i]; }
    std::sort(idx.begin(), idx.end(),
              [&](const int x, const int y) { return w[x] < w[y]; });
    std::vector<double> ws(n);
    std::vector<double> Vs(n * n);
    for (int i = 0; i < n; ++i)
    {
        ws[i] = w[idx[i]];
        for (int k = 0; k < n; ++k) { Vs[k*n+i] = V[k*n+idx[i]]; }
    }
    w = ws;
    V = Vs;
}

void sym_inv_sqrt(const std::vector<double>& S, const int n,
                  std::vector<double>& Sinv_root, const double rank_tol)
{
    std::vector<double> w, V;
    sym_eig(S, n, w, V);
    Sinv_root.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        // Branch: numerically zero eigenvalue -> drop (rank truncation).
        const double f = (w[i] > rank_tol) ? 1.0 / std::sqrt(w[i]) : 0.0;
        for (int a = 0; a < n; ++a)
        {
            for (int b = 0; b < n; ++b)
            {
                Sinv_root[a*n+b] += V[a*n+i] * f * V[b*n+i];
            }
        }
    }
}
} // namespace alloc
```

- [ ] **Step 3:** CMake：`source_estate/CMakeLists.txt` 加 `module_constraint/alloc_linalg.cpp`；`test/CMakeLists.txt` 加

```cmake
AddTest(
  TARGET MODULE_ESTATE_constraint_alloc_linalg
  LIBS parameter ${math_libs} base device
  SOURCES alloc_linalg_test.cpp ../alloc_linalg.cpp
)
```

- [ ] **Step 4:** 构建/运行，预期 2 个用例全过。
- [ ] **Step 5:** Commit。

### Task S1.2 分配算子（网格后端）与单位分解

**Files:**
- Create: `source/source_estate/module_constraint/alloc_operator.h/.cpp`
- Create: `test/alloc_operator_test.cpp`（CMake 新目标）

- [ ] **Step 1: 写测试（核心 4 个）**

```cpp
// (a) Sum rule: Σ_i A_i x = x for arbitrary grid functions x.
// (b) Positivity: ⟨x|A_i|x⟩ >= 0 for random x (quadratic form).
// (c) Interpolation limits: no target -> A_i = diag(w_i);
//     complete orthonormal target set (matrix backend) -> A_i = Q_i.
// (d) Operator application equals the dense definition (tiny grid).
```

测试代码（完整）：

```cpp
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "constraint_test_utils.h"
#include "source_estate/module_constraint/alloc_operator.h"
#include "source_estate/module_constraint/weight_grid.h"

// Normalized Gaussian on the grid (the R x R orthogonalization removes any
// discretization normalization error; this helper only shapes the target).
static std::vector<double> gaussian_grid(const ModulePW::PW_Basis& pw,
                                         const UnitCell& ucell,
                                         const std::array<double, 3>& c,
                                         const double sigma)
{
    std::vector<double> f(pw.nrxx, 0.0);
    const double norm = std::pow(2.0 * ModuleBase::PI * sigma * sigma, 1.5);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const int i = ir / (pw.ny * pw.nplane);
        const int j = ir / pw.nplane - i * pw.ny;
        const int k = ir % pw.nplane + pw.startz_current;
        const ModuleBase::Vector3<double> rf(static_cast<double>(i) / pw.nx,
                                             static_cast<double>(j) / pw.ny,
                                             static_cast<double>(k) / pw.nz);
        const ModuleBase::Vector3<double> r = rf * ucell.latvec * ucell.lat0;
        const double r2 = (r.x - c[0]) * (r.x - c[0])
                        + (r.y - c[1]) * (r.y - c[1])
                        + (r.z - c[2]) * (r.z - c[2]);
        f[ir] = std::exp(-r2 / (2.0 * sigma * sigma)) / norm;
    }
    return f;
}

TEST(AllocOperatorTest, SumRuleAndPositivity)
{
    auto ucell = make_h2o_ucell();                 // O, H, H
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    const double dV = ucell->omega / static_cast<double>(pw.nxyz);
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    // The allocator needs one weight per atom: pin the map explicitly.
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();
    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    std::vector<std::vector<double>> phi0;
    phi0.push_back(gaussian_grid(pw, *ucell, pos[0], 1.0)); // "d-like" on O
    phi0.push_back(gaussian_grid(pw, *ucell, pos[1], 1.0)); // on H1
    std::vector<int> target_atom = {0, 1};
    alloc::AllocOperator op;
    op.build_grid(wg, target_atom, phi0);
    EXPECT_LT(op.cond(), 1e6);

    std::mt19937 gen(3);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(pw.nrxx), y(pw.nrxx), acc(pw.nrxx, 0.0);
    for (double& v : x) { v = nd(gen); }

    // (a) Sum rule: sum_i A_i x == x pointwise.
    for (int i = 0; i < ucell->nat; ++i)
    {
        op.apply_A(i, x, y);
        for (int ir = 0; ir < pw.nrxx; ++ir) { acc[ir] += y[ir]; }
    }
    for (int ir = 0; ir < pw.nrxx; ++ir) { EXPECT_NEAR(acc[ir], x[ir], 1e-10); }

    // (b) Positivity of every allocation's quadratic form (with dV).
    for (int i = 0; i < ucell->nat; ++i)
    {
        op.apply_A(i, x, y);
        double q = 0.0;
        for (int ir = 0; ir < pw.nrxx; ++ir) { q += x[ir] * y[ir] * dV; }
        EXPECT_GE(q, -1e-12);
    }
}

TEST(AllocOperatorTest, InterpolationLimits)
{
    // --- Limit 1: empty target set -> A_i = diag(w_i) -------------------
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();
    alloc::AllocOperator op0;
    op0.build_grid(wg, {}, {}); // no targets
    std::mt19937 gen(5);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(pw.nrxx), y;
    for (double& v : x) { v = nd(gen); }
    for (int i = 0; i < ucell->nat; ++i)
    {
        op0.apply_A(i, x, y);
        const std::vector<double>& w = wg.constraint_weight(i);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            EXPECT_NEAR(y[ir], w[ir] * x[ir], 1e-12);
        }
    }

    // --- Limit 2: complete orthonormal target set -> A_i = Q_i ----------
    // Tiny grid (4^3) with delta-like targets (one per grid point) spans the
    // whole represented space, so Q_c = 0 and A_i == Q_i exactly.
    ModulePW::PW_Basis pw4;
    pw4.initgrids(1.0, ucell->latvec, 4, 4, 4);
    pw4.distribute_r();
    const double dV4 = ucell->omega / static_cast<double>(pw4.nxyz);
    constraint::WeightGrid wg4(*ucell, &pw4, {1.5, 0.5, 0.5});
    wg4.set_constraint_atoms({{0}, {1}, {2}});
    wg4.build();
    std::vector<std::vector<double>> deltas(pw4.nrxx,
        std::vector<double>(pw4.nrxx, 0.0));
    std::vector<int> which_atom(pw4.nrxx, 0);
    const std::vector<std::array<double, 3>> pos4 = h2o_positions(*ucell);
    for (int a = 0; a < pw4.nrxx; ++a)
    {
        deltas[a][a] = 1.0 / std::sqrt(dV4);   // orthonormal on the grid
        // Label each grid point by its nearest atom (any complete assignment
        // works; the test only needs Q_d = I).
        const int i = a / (pw4.ny * pw4.nplane);
        const int j = a / pw4.nplane - i * pw4.ny;
        const int k = a % pw4.nplane + pw4.startz_current;
        const ModuleBase::Vector3<double> rf(static_cast<double>(i) / pw4.nx,
                                             static_cast<double>(j) / pw4.ny,
                                             static_cast<double>(k) / pw4.nz);
        const ModuleBase::Vector3<double> r = rf * ucell->latvec * ucell->lat0;
        double best = 1e30;
        for (int I = 0; I < 3; ++I)
        {
            const double d2 = (r.x - pos4[I][0]) * (r.x - pos4[I][0])
                            + (r.y - pos4[I][1]) * (r.y - pos4[I][1])
                            + (r.z - pos4[I][2]) * (r.z - pos4[I][2]);
            // Branch: new nearest atom.
            if (d2 < best) { best = d2; which_atom[a] = I; }
        }
    }
    alloc::AllocOperator opC;
    opC.build_grid(wg4, which_atom, deltas);
    std::vector<double> x4(pw4.nrxx), y4;
    for (double& v : x4) { v = nd(gen); }
    for (int i = 0; i < ucell->nat; ++i)
    {
        opC.apply_A(i, x4, y4);
        for (int a = 0; a < pw4.nrxx; ++a)
        {
            // Branch: grid point assigned to atom i stays, the rest vanish.
            const double expected = (which_atom[a] == i) ? x4[a] : 0.0;
            EXPECT_NEAR(y4[a], expected, 1e-10);
        }
    }
}
```

- [ ] **Step 2:** 实现 API（头文件）：

```cpp
// alloc_operator.h
#ifndef ALLOC_OPERATOR_H
#define ALLOC_OPERATOR_H
#include <vector>
#include "source_estate/module_constraint/weight_grid.h"
namespace alloc {
/**
 * @brief Unified allocation operator A_i = Q_i + Qc W_i Qc on the grid
 * (design 2026-10-21, eq. 2.1).  Grid backend: target orbitals are given as
 * real-space functions; used by the readout prototype and the unit tests.
 */
class AllocOperator
{
  public:
    // Build from the shared weights and R target orbitals (grid functions),
    // orthogonalized in the R x R subspace (symmetric orthogonalization).
    void build_grid(const constraint::WeightGrid& wg,
                    const std::vector<int>& target_atom,
                    const std::vector<std::vector<double>>& phi0);
    int ntarget() const { return (int)atom_of_.size(); }
    double cond() const { return cond_; }        // S_T condition number audit
    // o_a = <phi_a|x> for the orthogonalized targets.
    void overlaps(const std::vector<double>& x, std::vector<double>& o) const;
    void apply_P(const std::vector<double>& x, std::vector<double>& out) const;
    void apply_Qc(const std::vector<double>& x, std::vector<double>& out) const;
    void apply_A(const int atom, const std::vector<double>& x,
                 std::vector<double>& out) const;
  private:
    std::vector<std::vector<double>> phi_;  // orthogonalized, [R][nrxx]
    std::vector<int> atom_of_;              // [R] global atom index
    std::vector<std::vector<double>> w_;    // [nat][nrxx]
    double dV_ = 0.0;
    double cond_ = 1.0;
};
} // namespace alloc
#endif
```

实现要点（每分支注释）：
- `build_grid`：`S_T[a][b] = Σ_g φ_a φ_b dV`（pool 归约）→ `sym_eig` → 条件数审计 → `φ' = φ S_T^{-1/2}`；`w_` 从 `wg.constraint_weight(a)` 复制（若约束映射不是"每原子一个"，先 `set_constraint_atoms({{0},{1},{2}})`）。
- `apply_P/apply_Qc`：`P x = Σ_a φ_a ⟨φ_a|x⟩`；`Qc x = x − P x`。
- `apply_A(atom,x)`：`Q_i x = Σ_{a∈i} φ_a⟨φ_a|x⟩`；`A_i x = Q_i x + Qc(W_i·Qc x)`（先乘权重再投影，顺序按定义 `Q̂_c Ŵ_i Q̂_c`）。

- [ ] **Step 3:** 构建/运行，预期全过（`<1e-10`）。
- [ ] **Step 4:** Commit。

### Task S1.3 网格只读观测与三量分解

**Files:**
- Create: `alloc_observe.h/.cpp`；`test/alloc_observe_test.cpp`

- [ ] **Step 1: 写测试**

```cpp
// Toy spin-polarized bands: one up-like Gaussian on O, one down-like on H1.
// Checks: sum_i n_i = N, sum_i m_i = M_tot (same-grid reference),
//         n_i = n_i^T + n_i^c, m_i = m_i^T + m_i^c, n_c(r) >= 0.
TEST(AllocObserveTest, SumRulesAndComplement)
{
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    const double dV = ucell->omega / static_cast<double>(pw.nxyz);
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();
    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    alloc::AllocOperator op;
    op.build_grid(wg, {0, 1},
                  {gaussian_grid(pw, *ucell, pos[0], 1.0),
                   gaussian_grid(pw, *ucell, pos[1], 1.0)});

    std::vector<std::vector<double>> up{
        gaussian_grid(pw, *ucell, pos[0], 1.0)};
    std::vector<std::vector<double>> dn{
        gaussian_grid(pw, *ucell, pos[1], 1.0)};
    std::vector<double> occ_up{1.0}, occ_dn{1.0};

    alloc::AllocObserve obs(op);
    const alloc::AllocResult r = obs.observe(up, occ_up, dn, occ_dn);

    // Same-grid reference totals (defines N and M for this toy).
    double nref = 0.0, mref = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        double ru = 0.0, rd = 0.0;
        for (size_t b = 0; b < up.size(); ++b)
        {
            ru += occ_up[b] * up[b][ir] * up[b][ir];
        }
        for (size_t b = 0; b < dn.size(); ++b)
        {
            rd += occ_dn[b] * dn[b][ir] * dn[b][ir];
        }
        nref += (ru + rd) * dV;
        mref += (ru - rd) * dV;
    }

    double nsum = 0.0, msum = 0.0;
    for (int i = 0; i < ucell->nat; ++i)
    {
        nsum += r.n_tot[i];
        msum += r.m_tot[i];
        EXPECT_NEAR(r.n_tot[i], r.n_T[i] + r.n_c[i], 1e-10);
        EXPECT_NEAR(r.m_tot[i], r.m_T[i] + r.m_c[i], 1e-10);
    }
    EXPECT_NEAR(nsum, nref, 1e-8);
    EXPECT_NEAR(msum, mref, 1e-8);
    for (const double v : r.n_c_grid) { EXPECT_GE(v, -1e-14); }
}

// Empty target set must reproduce the existing Becke readout (same weights,
// same grid), i.e. n_tot[i] = int w_i (rho_up+rho_dn) and
// m_tot[i] = int w_i (rho_up-rho_dn), to machine precision.
TEST(AllocObserveTest, EmptyTargetSetEqualsBecke)
{
    // Same grid/bands/weights as SumRulesAndComplement.
    auto ucell = make_h2o_ucell();
    ModulePW::PW_Basis pw;
    pw.initgrids(1.0, ucell->latvec, 24, 24, 24);
    pw.distribute_r();
    const double dV = ucell->omega / static_cast<double>(pw.nxyz);
    constraint::WeightGrid wg(*ucell, &pw, {1.5, 0.5, 0.5});
    wg.set_constraint_atoms({{0}, {1}, {2}});
    wg.build();
    const std::vector<std::array<double, 3>> pos = h2o_positions(*ucell);
    std::vector<std::vector<double>> up{
        gaussian_grid(pw, *ucell, pos[0], 1.0)};
    std::vector<std::vector<double>> dn{
        gaussian_grid(pw, *ucell, pos[1], 1.0)};
    std::vector<double> occ_up{1.0}, occ_dn{1.0};

    alloc::AllocOperator op0;
    op0.build_grid(wg, {}, {}); // no target orbitals
    alloc::AllocObserve obs0(op0);
    const alloc::AllocResult r0 = obs0.observe(up, occ_up, dn, occ_dn);

    for (int i = 0; i < ucell->nat; ++i)
    {
        double n_direct = 0.0, m_direct = 0.0;
        const std::vector<double>& w = wg.constraint_weight(i);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            double ru = 0.0, rd = 0.0;
            for (size_t b = 0; b < up.size(); ++b)
            {
                ru += occ_up[b] * up[b][ir] * up[b][ir];
            }
            for (size_t b = 0; b < dn.size(); ++b)
            {
                rd += occ_dn[b] * dn[b][ir] * dn[b][ir];
            }
            n_direct += w[ir] * (ru + rd) * dV;
            m_direct += w[ir] * (ru - rd) * dV;
        }
        EXPECT_NEAR(r0.n_tot[i], n_direct, 1e-13);
        EXPECT_NEAR(r0.m_tot[i], m_direct, 1e-13);
    }
}
```

- [ ] **Step 2:** 实现 `AllocObserve`：
  - 输入：每自旋每 band 的网格波函数与占据数（PW 风格），或每自旋密度（如果目标集为空时与 Becke 等价）。
  - 计算：`o_{n,a}^{(σ)} = ⟨φ_a|ψ_n^{(σ)}⟩`；`n_T[i] = Σ_{a∈i}Σ_{σ,n} f_{σ,n}|o|²`；`n_c(r) = Σ_{σ,n} f |ψ − Σ_a φ_a o|²`；`n_c[i] = ∫w_i n_c`；自旋同理用 `(↑−↓)`。
  - 输出结构 `AllocResult {n_T,m_T,n_c,m_c,n_tot,m_tot, n_c_grid, cond}`。
- [ ] **Step 3:** 运行通过。
- [ ] **Step 4:** Commit + 写 `2026-10-21-s1-allocation-core-test.md`（含 `cond`、`m^c` 量级、性能）。

**S1 验收**：Task S1.1–S1.3 全绿；`Σn/Σm < 1e-8`；空目标集与 Becke 逐位一致；`n_c ≥ 0`；`cond < 1e6`（否则记录并进入风险 R3 处理）。

---

## 4. S2：LCAO 只读生产读数

### Task S2.1 目标轨道与重叠子块

**Files:** `alloc_operator.h/.cpp`（新增 LCAO 后端：输入为选定目标轨道的 AO 系数 `C`（尺寸 `R × norb`）、目标原子标签、AO 重叠 `S`；输出 `S_T = C†SC`、正交化系数 `C' = C S_T^{-1/2}`）；`test/alloc_operator_test.cpp`

- [ ] **Step 1:** 明确数据来源（实现时按此查证）：NAO 轨道文件（`LCAO_Orbitals`/`Numerical_Orbital_Lm`，按 `lchi` 选 `l=2` 等）、原子轨道数 `ucell.atoms[it].nw`、AO 重叠 `S`（`HS_Matrix_K`/`Parallel_Orbitals`）。
- [ ] **Step 2:** 测试（小基组 mock）：给定 `C=[c_a]`、`S`、总密度矩阵 `DMR`、自旋差 `DMR_diff`，验证（约定 `ρ̂ = Σ|χ_μ⟩DMR_{μν}⟨χ_ν|`，故 `Tr(DMR·S)` 为电子数）
  ```
  C' = C S_T^{-1/2} ,   S_T = C† S C                        （R×R 正交化）
  n_T[i] = Σ_{a∈i} c'_a† S · DMR      · S c'_a              （总密度矩阵）
  m_T[i] = μ_B Σ_{a∈i} c'_a† S · DMR_diff · S c'_a          （自旋差 = DMR↑ − DMR↓）
  ```
  与逐态求和 `Σ_{σ,n} f_{σ,n} |c'_a† S C_n^{(σ)}|²` 对拍 `<1e-10`。
  实现取数：`dmat.dm->cal_DMR()` 后，`get_DMR_pointer(1)` 为总 DMR；`switch_dmr(2)` 后同一指针为自旋差（用完 `switch_dmr(0)` 还原）。
- [ ] **Step 3:** 条件数/秩截断记录（`S_T` 的 `cond` 与截断秩）。
- [ ] **Step 4:** Commit。

### Task S2.2 生产钩子与 `ALLOC` 审计行

**Files:** `esolver_ks_lcao.cpp`（`after_scf` 或 `iter_finish`，在 `dmat.dm` 可用处）；`constraint_loop.{h,cpp}`；`input_parameter.h`、`read_input_item_other.cpp`（`constraint_alloc_report`，默认 false）

- [ ] **Step 1:** 新增 INPUT `constraint_alloc_report`（Boolean，默认 False；description 明确"只读、需要密度矩阵/波函数、默认关闭"）。
- [ ] **Step 2:** 钩子：`ConstraintLoop::report_alloc(const DensityMatrix&, const S&, ...)` 或独立函数；输出：
  ```
  [constraint] ALLOC frame=<istep> ntarget=<R> cond=<...>
  ALLOC c[0] atom=0 n_T=... n_c=... n=... m_T=... m_c=... m=... (mu_B/e)
  ALLOC audit sum_n=... nelec=... dsum_n=... sum_m=... tot_mag=... dsum_m=...
  ```
- [ ] **Step 3:** 测试：
  - 默认关闭：h2o_dimer 输出与基线逐字节一致（`diff` 日志）。
  - 开启：`|Σn−nelec|<1e-8`、`|Σm−tot_mag|<1e-8`；`m^T` 与 `onsite=` 对照表（同向/幅值比）。
- [ ] **Step 4:** 集成：h2o_dimer LCAO（warm relax 两帧）输出三量，记录 `m^c` 量级。
- [ ] **Step 5:** Commit + dated 测试文档。

**S2 验收**：默认关闭零影响；开启后求和精确；三量对照表成形；`m^c` 作为诊断量的量级被记录。

---

## 5. S3：冻结目标轨道约束 + ALM

### Task S3.1 约束势注入（LCAO 优先）

**Files:** `alloc_operator.{h,cpp}`（`apply_dH`）；`constraint_inject_lcao.cpp`（或新 `alloc_inject_lcao`）；测试

- [x] **Step 1:** 单元测试：小基组下 `ΔH = V − PV − VP + PVP + Σλ_iQ_i`（设计 eq. 2.4）与稠密构造逐元 `<1e-10`。
- [x] **Step 2:** 实现：`V = Σ_i λ_i W_i`（`W_i` 的 AO 矩阵 = `∫w_i χ_μ χ_ν`，复用现有局域势矩阵机制）；修正项用 R×R 低秩组装；每 SCF 迭代更新 `λ`（ALM）。
- [x] **Step 3:** 观测：每迭代用波函数/密度矩阵算 `s_i`（复用 S2 的投影），驱动 ALM `λ^eff = λ + k(s−t)`。
- [x] **Step 4:** 收敛判据：`max|s−t| < constraint_thr`；`Σm = M_tot` 审计每步输出。

### Task S3.2 冻结力（设计 eq. 6.3）

- [x] **Step 1:** 实现 `F_con,J = −Σ_i λ_i ∫ ν_c(r) ∂w_i/∂R_J dr`：复用现有权重导数核（`constraint_deriv`），把自旋密度换成**互补投影自旋密度** `ν_c`。
- [x] **Step 2:** 测试（FD）：
  - 位置 FD：固定 λ，±δ（δ=0.005 Bohr）三点能量差分 vs 解析力；判据 `< 0.0129 eV/Å`。
  - 目标 FD：`dE*/dt = −λ`；判据 `< 0.02 eV/μB`。
  - 与 Becke 约束同体系对照：`λ` 量级（预期约 10× 差异，因坐标不同；只作记录，不作判据）。
- [x] **Step 3:** ALM 刚性子例：0.01 μB，`scf_nmax=100`，两帧 relax；判据两帧 CONVERGED、末残差 `<1e-4 μB`。
- [x] **Step 4:** Commit + dated 测试文档（含 FD 表、`Σm` 审计、效率对比）。

**S3 验收**：位置 FD `<0.0129 eV/Å`；目标 FD `<0.02 eV/μB`；`Σm=M_tot`；ALM 100 步内收敛。

---

## 6. S4：完整导数 + 应力 + 非共线

### Task S4.1 `∂Q̂` 项

- [x] 实现设计 eq. (6.2) 的 `∂Q̂_i`/`∂P̂` 项（R×R 的 `∂S^{-1/2}`，附录 B）。
- [x] 测试：S3 vs S4 力差 `<0.005 eV/Å`（同一构型）；若超差，记录并保留 S4 为默认。
- [x] 验收：位置 FD 在 S4 下仍 `<0.0129 eV/Å`。

### Task S4.2 应力

- [x] 实现 eq. (6.5) 的固定目标/固定 multiplier 路径：权重应变导数、
      `∂Q̂/∂ε`（S4.1 伴随 `E` + overlap `dS`）、交叉项和体积/Jacobian 项。
      当前 fixed-mu/fixed-target 协议没有 moving-target 重算，故该子项未激活；
      非共线留给 S4.3，dipole 应力显式拒绝。
- [x] 测试：应变中心差分三档 `±1e-4/±5e-4/±1e-3` 全部记录；张量对称性 PASS。
      `1e-4` 约束总应力相对误差 `1.278% <5%` PASS；`5e-4/1e-3` 分别为
      `8.664%/31.240%`，但无约束同构型基线为 `8.440%/30.871%`，故记为
      “参考曲线非线性导致未过闸门”，不据此判定 S4.2 核错误。
- [x] 验收：应力表、固定-mu 审计、`CONSTRAINT STRESS` 连续性/对称性已记录；
      符号与 KS 应力口径一致。详见
      `docs/superpowers/specs/2026-10-26-s4-stress-test.md`。

### Task S4.3 非共线

- [x] 稠密核心：`λ_i` 三分量、`ΔH = ΣÂ_i(λ_i·σ)`；
      `m_i = μ_B Tr(ρ̂Â_iσ)` 的 Pauli/ spinor 两种读数与
      `dS` 伴随已实现于 `alloc_spin.{h,cpp}`。
- [x] 稠密测试：无 SOC 模型整体旋转 `|ΔE|=0 <1e-6 Ry`；三分量协同变换；
      伴随力 FD 差 `2.395e-9 <2e-7`。详见
      `docs/superpowers/specs/2026-10-26-s4-noncollinear-test.md`。
- [x] 生产接线（固定向量部分）：LCAO `nspin=4` 单 Γ 点、无 SOC 的
      target/kernel 构建、spinor 注入、Pauli DMR 读数与向量审计已实现。
      输入层要求 `constraint_alloc_switch=1`、所有约束 `fixed_mu`；横向
      x/y 分量要求 `noncolin=1`，多 k、SOC 和自适应向量 multiplier 仍显式
      拒绝。详见 `docs/superpowers/specs/2026-10-26-s4-noncollinear-production.md`。
- [x] 验收：SCF 级无 SOC 旋转不变性 + 位置 FD 力。固定 μ 的 z/x 旋转能量差
      已测为 `1.3e-9 Ry < 1e-6 Ry`，MPI 1/2 逐迭代一致；complex `TR`
      位置力已接线，H2O 三点 FD
      `F_FD=-1.4171907516 eV/A` 对解析 `-1.4205728783 eV/A`，
      差 `0.0033821267 < 0.0129 eV/A`。x 方向 `scf_thr=1e-8` 的磁密度
      混合极限环作为独立已知限制记录，不阻塞旋转能量与固定-`lambda`
      位置力闸门。

**S4 验收**：上述三条全过；写 dated 测试文档。

---

## 7. S5：PW 平价 + 周期 Hirshfeld

### Task S5.1 PW 后端

- [x] 新增 projector–projector 重叠 `⟨α_a|α_b⟩`（两中心或实空间积分）；R×R 正交化；`apply_dH` 按设计 §5.2 算法（becp + 局域乘）。
      核心已实现于 `alloc_pw.{h,cpp}`；projector tabulation overlap/becp 的
      2/4-rank `reduce_pool` 一致性已测。
- [ ] 测试：PW vs LCAO 同构型同目标：`|Δm^T|<0.02 μB`、`|ΔM|<0.01 μB`；`Σm` 审计。
      代数同目标平价已过（`max_dmT=1.066e-14`，求和差 0）；真实 PW/LCAO
      同电子态数据已接线：H2O/O-d、固定 `fixed_mu=0.01 Ry` 下
      `|Δm^T|=1.19e-4 μB`、`|ΔM|=1.75e-3 μB`；两边
      `|Σm-tot_mag|<1e-15 μB`。
- [x] 冻结目标位置力：由最终 PW 能带形成 `Q_c D Q_c`，交给共享
      `constraint_force()` 权重导数核；三点位置 FD 为
      `-5.6679144 vs -5.6632941 eV/Å`，误差 `0.0046203 eV/Å < 0.0129`。
      互补密度已通过 `KP_WORLD` 汇总跨 k-pool；`bndpar > 1` 仍显式拒绝。
- [x] 完整 PW `∂Q̂`：实现 projector 位移导数、`ST^{-1/2}` Fréchet 导数与
      正交目标转动项，并叠加到冻结 `Q_c ∂W Q_c` 网格力之后。`np=2` 三点总力
      FD 为 `F_FD=-5.667912284` 对解析 `-5.669041444` eV/Å，差
      `0.00112916 < 0.0129 eV/Å`；`np=4` 约束力/总力与 `np=2` 一致到
      `3e-10 eV/Å`。单独 `e_con` 差分不是完整约束能量且不满足电子驻点条件，
      不再作为独立力门；详见
      `docs/superpowers/specs/2026-10-27-s5-pw-target-force.md`。
- [x] 验收：平价表已通过；PW allocation 算符已接入 `HamiltPW`，固定
      `fixed_mu` SCF 在 MPI 2/4 进程收敛且读数一致，`kpar=2` 跨池短跑无
      死锁；冻结目标 FD 力 `<0.0129 eV/Å` 已通过。缓存、批量归约与
      `tab_atomic_` 首次消费初始化见
      `docs/superpowers/specs/2026-10-27-s5-pw-cache-mpi.md`；冻结力与
      `e_con` 边界见
      `docs/superpowers/specs/2026-10-27-s5-pw-frozen-force.md`；完整目标导数
      与总力 FD 见
      `docs/superpowers/specs/2026-10-27-s5-pw-target-force.md`；跨 k-pool 力
      对照见
      `docs/superpowers/specs/2026-10-27-s5-pw-kpar-force.md`。

### Task S5.2 周期 Hirshfeld 权重后端

- [x] `WeightType::HirshfeldPeriodic`：参考密度 `q_Z^0`（正系数高斯拟合或孤立原子密度）、晶格平移求和、`w_i=q_i/Σq_j`；`C²` 窗口（设计 Eq. 11）与**显式余量通道**（覆盖失败时）。
- [x] 测试：
  - `maxdev<1e-10`；`Σ_i ∂w_i/∂R=0`（S0.3 审计降为 0 违约）；
  - 大真空盒覆盖失败：余量通道被启用且 `原子+余量=全局`；
  - 镜像切换面附近权重导数的 FD 连续性（相对误差 `<1e-6`，对比 Becke 的最小镜像后端）。
- [ ] 验收：上述测试全过；Becke 与 Hirshfeld 两后端在同一电子态的三量对照。
      测试已全过；同电子态三量对照尚未生成，且 Hirshfeld 应变导数/应力仍显式拒绝。

**S5 验收**：PW≡LCAO 平价；Hirshfeld 审计与光滑性；两后端对照表。

---

## 8. S6：生产化与回归

### Task S6.1 与现有调度整合

- [ ] 暖启动：`Â_i` 的约束坐标需要自己的参考（delta → `s_ref` 每几何更新，复用 M2 的反推机制）。
- [ ] M3 信赖域/Broyden、M4 ALM、预算冻结：把 `s_i` 作为可观测量接入。
- [ ] 边界：`constraint_mu_schedule=alm` 与 `alloc` 组合的守卫；`branch_tol` 的能量口径改用 §4 的 `e_con`。

### Task S6.2 子空间加速

- [ ] 复用 DeltaSpin `SubspaceDiagonalizer`：`P_i^sub = ⟨ψ|Ŵ_i|ψ⟩ + ⟨Q̂_cψ|Ŵ_i|Q̂_cψ⟩`（只需局域积分 + becp）。
- [ ] 测试：λ trial 数 × 小对角化成本；与全对角化结果 `<1e-8`；近简并时回退。

### Task S6.3 回归与效率

- [ ] 既有 13 个 `ctest -R constraint` 全绿；默认路径逐字节不变。
- [ ] 成本：约束 SCF 迭代数 ≤ Becke 约束同目标的 1.5×；后处理 ≤ SCF 的 5%；记录墙钟。
- [ ] 集成：h2o_dimer（LCAO/PW）、自旋/电荷/混合三例；`Σm/Σq` 审计。
- [ ] 写 dated 测试文档 + dev log（AGENTS.md）。

**S6 验收**：上述全部；生产路径默认关闭/显式开启；回归网不破。

---

## 9. 风险 → 测试映射

| 风险 | 在哪个 Task 被测试 |
|---|---|
| R1 正交化混合/局域性 | S1.2（`cond`）、S2.1（轨道展宽/条件数）、S2.2（对照表） |
| R2 POVM 非正交 | S1.2（可加性 + 正性；文档声明"可加≠独立"） |
| R3 近线性依赖/秩截断 | S1.1（`rank_tol`）、S1.2（`cond<1e6`）、S2.1（截断秩） |
| R4 Wannier 非仿射 | S5.1（若采用 Wannier：冻结近似 + 外层收敛） |
| R5 冻结近似误差 | S3.2 vs S4.1（力差 `<0.005 eV/Å`） |
| R6 镜像切换导数不连续 | S0.3（基线量化）、S5.2（Hirshfeld 降为 0） |
| R7 覆盖失败 | S5.2（大真空 + 余量通道） |
| R8 电荷冗余 | S3.1（全部原子约束时的秩检查；`N−1` 差异模式） |
| R9 符号漂移 | S3.2（目标 FD `dE*/dt=−λ`） |
| R10 ALM 密度依赖 | S3.2（收敛点取力 + 残差警告） |
| R11 赝势口径 | S2.2（与 `onsite=`/总矩对账） |
| R12 轨道矩 | 设计非目标；S6 文档明确区分自旋/总矩 |
| R13 目标集主观 | S1.3/S2.2（目标集随结果报告；敏感性扫描） |
| R14 成本 | S1.3（cond/时间）、S6.2/S6.3（效率） |
| R15 子空间可靠性 | S6.2（近简并回退 + 残差） |

---

## 10. 阶段工作量估算（单人，含测试与文档）

| 阶段 | 估算 | 关键不确定性 |
|---|---|---|
| S0 | 0.5–1 天 | 导数求和审计可能暴露镜像切换点（预期） |
| S1 | 3–5 天 | Jacobi/算子/观测的接口设计 |
| S2 | 3–5 天 | NAO 目标轨道与 `S` 子块的取数（API 查证） |
| S3 | 1.5–2.5 周 | 非局域约束的 SCF 稳定性、ALM 参数 |
| S4 | 1–1.5 周 | `∂S^{-1/2}` 与应力项的推导/实现 |
| S5 | 1.5–2 周 | PW projector 重叠、Hirshfeld 参考密度 |
| S6 | 1–1.5 周 | 调度整合与回归 |

---

## 11. Definition of Done（每个阶段）

1. 代码 + 单元测试通过（`ctest -R` 对应目标全绿）；
2. 阶段闸门表（§1）中的全部判据有原始数据；
3. `docs/superpowers/specs/YYYY-MM-DD-<stage>-test.md`（AGENTS.md 五段）已写；
4. `docs/superpowers/specs/deltap-development-log.md` 追加本轮条目；
5. 默认关闭路径逐字节不变（凡引入 INPUT 的 Task）；
6. 分支注释/无 goto/函数长度 符合 AGENTS.md。
