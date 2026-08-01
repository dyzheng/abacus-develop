# 2026-07-31 DeltaP esolver 代码简化重构设计方案

> 状态：**已实施（R1 07-31 / R2 08-01 / R3 08-01 / R4 08-01 全部完成，见各轮 dated 文档）**
> 范围：`source/source_esolver/`（esolver_ks_lcao / esolver_ks_pw / deltap_common / deltap_solver）、
> `source/source_pw/module_pwdft/deltap_pw.*`、`source/source_lcao/module_operator_lcao/deltap_lcao.*`
> 结论先行：**可以，且应该重构**。当前 esolver 侧 DeltaP 逻辑存在约 260 行死代码、4 处重复实现同一数学、两套不一致的状态机；但数值核心（module_deltap）是干净的，不需要动。

---

## 1. 现状审查

### 1.1 代码分布与行号锚点

| 位置 | 内容 | 问题密度 |
|---|---|---|
| `esolver_ks_lcao.h:89-99` | 8 个成员 + 3 个 flag + 3 个 `void*` | 高 |
| `esolver_ks_lcao.cpp:620-632` | `hamilt2rho_single` 中 inner-loop 接线 | 中 |
| `esolver_ks_lcao.cpp:695-835` | `iter_finish` 内联 DeltaP 块（约 140 行） | 高 |
| `esolver_ks_lcao.cpp:936-1054` | `deltap_init`（raw `new`，无 delete） | 中 |
| `esolver_ks_lcao.cpp:1057-1090` | `deltap_compute_gamma` | 中 |
| `esolver_ks_lcao.cpp:1092-1256` | `deltap_inner_loop`（BFGS/FR-CG） | 高 |
| `esolver_ks_lcao.cpp:1258-1410` | `deltap_update_lambda`（梯度下降） | 高 |
| `esolver_ks_pw.cpp:99-121,189,243-245,301-306` | PW 接线 | 中 |
| `deltap_pw.cpp:18-28` | 文件级全局单例状态 | 高 |
| `deltap_pw.cpp:130-160,200-260,299` | no-op + 死分支 + 死函数 | 高 |
| `deltap_common.h` | 4 个函数，仅 1 个被用 | 高 |
| `deltap_solver.h` | 自声明 deprecated，0 处 include | 高 |

### 1.2 问题分类

#### A. 死代码 / 废弃脚手架（建议直接删）

1. **`deltap_solver.h`**：回调抽象从未落地，头文件自注释已声明可删，全仓库 0 引用。
2. **`deltap_common.h`**：`update_lambda` / `to_effective_lambda` / `compute_max_residual` 无调用者，只有 `compute_dp_escon` 被 `esolver_ks_lcao.cpp:745` 使用。讽刺的是，esolver 里重复实现的三份逻辑正是这些"未用函数"该干的事。
3. **PW `run_deltap_lambda_loop`**（`deltap_pw.cpp:130`）：恒返回 `false` 的 no-op，却每个 SCF 迭代都被调用；Phase A 语义可以并入 `deltap_iter_finish` 或干脆删除。
4. **PW inner-loop 死分支**（`deltap_pw.cpp:200-260`）：`inner_nmax > 0` 先 `WARNING_QUIT`（同文件 ~190 行），后续 `if (inner_nmax > 0 && s_hamilt ...)` 永远不可达，`inner_loop_ok` 恒 false。
5. **`s_hamilt` / `set_deltap_pw_hamilt`**：唯一写点 `esolver_ks_pw.cpp:189`，唯一读点在不可达分支 → 整体死代码。遵循 `void*` 存 Hamilt 的 DeltaSpin 模式，但 DeltaP 从未实现。
6. **`compute_per_atom_gamma_from_becp`**（`deltap_pw.cpp:299`）：定义无调用者，与 `compute_per_atom_gamma_kstring` 功能重叠（后者在用）。
7. **LCAO raw `new` 泄漏**：`deltap_init` 中 `new deltap::DeltaP` / `new unkOverlap_lcao` / `new cal_r_overlap_R` 存入 `void*`，析构函数（`esolver_ks_lcao.cpp:45-50`）只释放 `psi` → 每次运行泄漏 3 个对象。

#### B. 同一数学重复实现 4+ 份

1. **残差计算** `r = C·γ − t`（或 `γ − t`，含 total 模式）：
   - `deltap_compute_gamma`（`esolver_ks_lcao.cpp:1073-1088`）
   - `deltap_inner_loop` 初值（:1115-1133）与每次 trial（:1204-1222）
   - `deltap_update_lambda`（:1284-1340）
   - `deltap_common::update_lambda` / `compute_max_residual`（未用）
   - `deltap_pw.cpp` 内联（:180-185、:210-230、:250-262）
2. **约束空间 → 逐原子 λ**（`λ_eff[i] = Σ_a λ[a]·C[a][i]`）：`deltap_inner_loop` 2 处（:1163-1170、:1226-1233）、`deltap_update_lambda`（:1305-1312）、`deltap_common::to_effective_lambda`（未用）。
3. **梯度下降 + mixing**：`deltap_update_lambda` 与 `deltap_pw.cpp` 各写一遍，且 total 模式分支各自实现。
4. **2π 分支跟踪（跨 SCF 步 unwrap）**：LCAO 在 `deltap_wannier.cpp`（`select_branch_set`），PW 在 `deltap_pw.cpp:264-279` 各自实现；后者是简化版（按上一帧最近分支）。
5. **`dp_escon = −Σ λ·γ`**：LCAO 与 PW 各写一遍。

#### C. 状态管理混乱（两条路径两套状态机）

| 维度 | LCAO | PW |
|---|---|---|
| 存放 | ESolver 类成员（8 个 + 3 flags + 3 void*） | 文件级全局单例（`deltap_pw.cpp:18-28`） |
| 生命周期 | 依赖 `deltap_scf_initialized_` 惰性 init | `set_deltap_pw_hamilt` 里 `s_lambda_set=false` |
| 离子步重置 | `iter==1` 清 `deltap_lambda_set_`/`deltap_inner_loop_done_` | 无显式离子步重置（靠 setter 时机） |
| 两阶段门控 | `drho>0 && drho<thr` 才更新 λ（`deltap_update_lambda`） | `drho<=0 || drho>=thr` 直接 return |
| 一致性 | C-01/C-12 已踩过（escon 全 rank、λ reset） | C-19pw 已踩过（λ/目标混淆） |

两套语义细节还不一致（例如 LCAO 要求 `drho>0` 严格大于，PW 允许 `drho<=0` 时跳过），同一算法两种行为，后续任何"统一修正"都要双倍成本。

#### D. `iter_finish` 巨型内联块（`esolver_ks_lcao.cpp:695-835`）

单块混合了 6 种职责：
1. 空指针/基组检查（dynamic_cast + WARNING_QUIT × 2）
2. 惰性 init + 离子步 flag 重置
3. γ 计算 + max_dev
4. λ 更新
5. escon 记账
6. 三种诊断输出：[rawG]（:751-766）、[DeltaP P1/P3]（:767-810）、[E-field]（:811-830）

嵌套结构 `if (switch && corr) { if constexpr complex { … } else { WARNING } }`：对 `TK=double` 实例，`else` 的 WARNING 每个 SCF 迭代打一次；这个检查应在 init 处一次性完成。rank-0 打印块缩进混乱（`} // rank 0 scope` 前 30 行未缩进），可读性差。

#### E. 潜在 bug / 一致性隐患

1. `deltap_init` 回退路径 `deltap_constraint_lambda_.assign(ucell.nat, 0.0)`（`esolver_ks_lcao.cpp:1049-1050`）：约束矩阵模式下应为 `m` 维，写成 `nat` 维；仅在矩阵文件加载失败时触发，是潜伏 bug。
2. 约束矩阵列数不匹配只 `cerr`（:1042），未 `WARNING_QUIT`，后续会用错误尺寸继续跑。
3. target / 约束矩阵由**每个 MPI rank 各自读文件**（`deltap_init`、`esolver_ks_pw.cpp:99-121`），跨 rank 一致性依赖"同一文件系统 + 相同路径"的假设；应 rank-0 读取 + Bcast（与 λ 的 Bcast 模式对齐，见 `esolver_ks_lcao.cpp:1360-1368`）。
4. `deltap_scf_initialized_` 在 `iter_finish` 才置位 → `hamilt2rho_single` 的 inner-loop 调用点在第一个 SCF 迭代必然被跳过，行为靠"iter_finish 先于下一次 hamilt2rho_single"的隐式顺序，脆弱。

---

## 2. 设计目标与原则

1. **ESolver 只接线，不做决策**：约束求解的状态机、门控、λ 更新、报告全部下沉到独立组件，`iter_finish`/`hamilt2rho_single` 各保留 ≤10 行。
2. **一份数学，一份实现**：残差、梯度下降、`Cᵀ` 转换、escon、2π 分支跟踪各只有一个实现，LCAO/PW 共用。
3. **消灭死代码**：删 `deltap_solver.h`、未用 common 函数、PW 死分支/死函数/`s_hamilt`。
4. **单一状态机**：LCAO/PW 共用同一套 P1→P2→P3 语义与离子步重置规则，差异只体现在 backend 回调。
5. **每轮可编译、可回归**：迁移分 4 轮，每轮结束都有可运行的中间态，行为（数值 + stdout 格式）逐轮 diff 对齐。

---

## 3. 目标架构

```
source/source_esolver/
  deltap_scf.h/.cpp      ← 新增：DeltapScfSolver 状态机（basis-independent 控制流）
  deltap_common.h        ← 精简为纯函数库：residual / gd_update / to_effective / escon / unwrap
  deltap_solver.h        ← 删除
source/source_lcao/module_deltap/   ← 数值核心（Wilson loop / gauge / branch）不动
source/source_lcao/module_operator_lcao/deltap_lcao.*  ← operator 不动（get/set_lambda、hk_correction）
source/source_pw/module_pwdft/deltap_pw.*  ← 精简为 PW 数值适配：gamma 计算 + backend 回调
```

### 3.1 `DeltapScfSolver`（核心新增）

```cpp
// deltap_scf.h —— 约束 SCF 状态机
namespace deltap_scf {

struct DeltapParams {                    // init 时从 Input_para 快照，运行期只读
    int    gdir;                         // 1/2/3 → alpha = gdir-1
    double inner_thr;                    // drho 门控
    double lambda_step, lambda_mixing;
    int    nscf;                         // inner loop 步数（0 = 同步两阶段）
    double conv_thr, lambda_init;
    std::string constraint_mode;         // "total" / "per_atom" / "matrix"
    std::vector<double>          target; // per-atom γ 目标（matrix 模式为空）
    std::vector<int>            constrain;
    std::vector<std::vector<double>> C;  // 约束矩阵（空 = 非 matrix 模式）
    std::vector<double>               t; // 矩阵目标
    bool verbose;                        // 是否打印 [rawG]/[E-field]
};

struct DeltapState {                     // 全部可变状态收拢于此（替换散落的成员）
    bool initialized = false;
    bool lambda_set = false;             // 两阶段：λ 已更新并冻结
    bool inner_loop_done = false;
    std::vector<double> lambda_eff;      // 逐原子 λ [nat]
    std::vector<double> lambda_cstr;     // 约束空间 λ [m]（per_atom 模式与 lambda_eff 同）
    std::vector<double> gamma_I;         // 最近一次逐原子 γ [nat]
    double max_res = 0.0;
    double dp_escon = 0.0;
    std::vector<double> gamma_prev;      // 2π 分支跟踪（跨 SCF 步）
};

class DeltapScfSolver {
public:
    // 基组相关操作以回调注入；ESolver 负责把回调绑到 operator/hsolver/proj
    struct Backend {
        std::function<void(const std::vector<double>&)> set_lambda;        // operator/onsite_proj
        std::function<void(const std::vector<double>&)> set_hk_correction; // LCAO only，PW 传空
        std::function<std::vector<double>()> compute_gamma;                // DeltaP::compute_gamma_scf / PW kstring
        std::function<void()> solve_frozen;                                // HSolver(skip_charge=true)
        std::function<void()> sync_rho_from_dm;                            // LCAO only（skip_solve 后 dm2rho）
        std::function<void()> reset_charge_mixing;                         // P2 转换 mix_reset
    };

    void init(const DeltapParams& p, Backend b);
    void reset_ionic_step();            // iter==1：清 lambda_set / inner_loop_done / gamma_prev
    bool inner_loop(int iter, double drho);  // 返回 skip_solve（nscf==0 时恒 false）
    void iter_finish(int iter, double drho); // γ → 门控 λ 更新 → escon → report()
    const DeltapState& state() const;
    // 诊断：report() 内部统一输出 [DeltaP P1/P3]、[rawG]、[E-field]
private:
    DeltapParams params_;
    DeltapState  state_;
    Backend      backend_;
    // 所有残差/λ 更新/转换/分支跟踪都调用 deltap_common 纯函数
};
} // namespace deltap_scf
```

要点：
- `compute_gamma` 回调返回**已折叠到 gdir 方向**的一维逐原子 γ（LCAO 内部取 `gamma_I[iat][alpha]`，PW 取 k-string 结果），状态机不再关心 `AtomicPolarization` 结构。
- `inner_loop` 与 `iter_finish` 共用同一份残差/更新逻辑，`nscf==0` 时 `inner_loop` 直接返回 false（同步两阶段），`nscf>0` 时走 BFGS/FR-CG（BFGS 对象由 backend 持有：LCAO 用 `dp->bfgs()`，PW 当前不支持则 init 时拒绝）。
- `reset_ionic_step()` 统一替代 LCAO 的 `iter==1` 内联重置与 PW 的 setter 时机重置，两条路径同一语义。

### 3.2 `deltap_common.h`（纯函数，唯一实现）

```cpp
namespace deltap_common {
// r = C·γ − t（matrix）或 γ − t（per_atom，支持 total 模式与 constrain 掩码）
std::vector<double> compute_residual(const std::vector<std::vector<double>>& C,
                                     const std::vector<double>& t,
                                     const std::vector<double>& gamma,
                                     const std::vector<int>& constrain,
                                     bool total_mode);
double max_norm(const std::vector<double>& r);
// λ ← mixing·(λ + step·r) + (1−mixing)·λ
void gd_update(std::vector<double>& lambda, const std::vector<double>& r,
               double step, double mixing);
// λ_eff[i] = Σ_a λ[a]·C[a][i]；C 空 = identity
std::vector<double> to_effective_lambda(const std::vector<double>& lambda_cstr,
                                        const std::vector<std::vector<double>>& C, int nat);
double dp_escon(const std::vector<double>& lambda, const std::vector<double>& gamma);
// 跨 SCF 步 2π 分支跟踪：选离上一帧最近的 γ
std::vector<double> unwrap_2pi(const std::vector<double>& gamma,
                               const std::vector<double>& prev);
}
```

- 全部无状态、可单测（`source/source_esolver/test/` 有现成测试目录可加）。
- PW/LCAO 各自数值核心中的"专用"分支选择（`select_branch_set` 多带权重版本）保留在 module_deltap，不强行合并；`unwrap_2pi` 只收敛两处相同的"跨 SCF 步最近分支"逻辑。

### 3.3 ESolver 最终形态

**LCAO（`esolver_ks_lcao.cpp`）：**

```cpp
// 头文件：8 成员 + 3 flags + 3 void* → 一个成员
std::unique_ptr<deltap_scf::DeltapScfSolver> deltap_scf_;  // 构造即创建，init 在 first iter_finish

// iter_finish —— 从 ~140 行缩到 ~10 行
if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr) {
    if constexpr (std::is_same<TK, std::complex<double>>::value) {
        deltap_scf_->iter_finish(iter, this->drho);
    }
}

// hamilt2rho_single —— 原 620-632
if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr && deltap_scf_)
    skip_solve = deltap_scf_->inner_loop(iter, this->drho);

// cal_force —— 保留 store_lambda_for_force 接线（force 需要 λ）
```

- 基组检查（dynamic_cast、dp_op null）移到 backend 绑定处一次性完成；real 实例的 WARNING 移到 init 一次性。
- `deltap_init`/`deltap_compute_gamma`/`deltap_inner_loop`/`deltap_update_lambda` 四个成员函数删除，逻辑进 `DeltapScfSolver`。
- 三个 raw `new` 的数值对象改为 `std::unique_ptr` 成员（或由 `DeltapScfSolver` 持有），修泄漏。

**PW（`esolver_ks_pw.cpp` + `deltap_pw.cpp`）：**

```cpp
// esolver_ks_pw.cpp: iter_finish
if (PARAM.inp.deltap_switch && PARAM.inp.deltap_corr)
    this->pelec->f_en.dp_escon = pw_deltap::deltap_scf()->iter_finish(ucell, iter, this->drho, psi_cpu, kv, pw_wfc, pw_rho);

// deltap_pw.cpp: 全局单例 → 实例对象；只留数值部分
class DeltapPwBackend { ... };  // compute_gamma = compute_per_atom_gamma_kstring + 分支跟踪
                                // set_lambda = set_deltap_pw_lambda（onsite_proj 读取）
```

- 删除：`s_lambda_set`/`s_gamma_prev`/`s_hamilt`/`s_active` 等全局态（`s_lambda`/`s_targets`/`s_constrain` 改由 backend 读入状态机）；`run_deltap_lambda_loop`；`compute_per_atom_gamma_from_becp`。
- `inner_nmax>0` 的拒绝语义保留：在 `DeltapScfSolver::init` 时对 PW backend 一次性 `WARNING_QUIT`。

---

## 4. 迁移路线（4 轮，每轮独立可编译可回归）

### Round 1：清理死代码 + 纯函数唯一化（零行为变化）
- 删 `deltap_solver.h`（含 CMake/Makefile.Objects 若引用）；删 `deltap_common.h` 未用函数。
- 删 PW：`run_deltap_lambda_loop` 的调用与实现（`esolver_ks_pw.cpp:243-245` 保留 no-op 位置改为注释）、`inner_nmax>0` 死分支、`s_hamilt`/`set_deltap_pw_hamilt`、`compute_per_atom_gamma_from_becp`。
- LCAO `deltap_init` 的 `new` 改 `unique_ptr`（行为不变，修泄漏）。
- **测试**：P01 H₂O 冒烟（LCAO + PW），stdout 与重构前 diff 为空。

### Round 2：LCAO 抽取 `DeltapScfSolver`（v1，仅 LCAO 使用）
- 新增 `deltap_scf.h/.cpp`（挂到 `source/source_esolver/CMakeLists.txt` + `Makefile.Objects`）。
- 迁入 `deltap_init`/`deltap_compute_gamma`/`deltap_inner_loop`/`deltap_update_lambda` 逻辑；残差/更新/转换/escon 全部走 `deltap_common`。
- `iter_finish` DeltaP 块收缩；`[rawG]`/`[DeltaP]`/`[E-field]` 输出原样搬进 `report()`（格式逐字符保持，测试解析依赖）。
- **测试**：P01 复跑，γ/λ/escon 时间序列与重构前一致（diff 运行日志）；`deltap_common` 加单测（`test/` 现有框架）。

### Round 3：PW 接入同一状态机
- `deltap_pw.cpp` 全局单例 → `DeltapScfSolver` 实例 + PW backend。
- 统一两阶段门控与离子步重置：删 `s_lambda_set`，用 `reset_ionic_step()`。
- **测试**：P13（PW 通道）冒烟；LCAO/PW 同输入下 γ 序列对齐（容许数值差异，时序语义一致）。

### Round 4：MPI 收敛 + 修潜伏 bug + 文档
- target/约束矩阵读取改 rank-0 + Bcast（对齐 λ 的 Bcast 模式）。
- 修 `deltap_constraint_lambda_` 回退维度 bug（nat → m）；矩阵尺寸不匹配改 `WARNING_QUIT`。
- real 实例的 WARNING 收口到 init。
- 更新 `deltap-development-log.md`、本设计文档状态改为"已实施"。

---

## 5. 风险与验收

| 风险 | 缓解 |
|---|---|
| inner-loop `skip_solve` + `dm2rho` 路径行为漂移（LCAO） | Round 2 单独 diff 该路径日志；保留 `dm2rho` 回调语义 |
| stdout 被测试/脚本解析 | 输出字符串逐字符保真迁移，Round 1/2 用 diff 验收 |
| PW `inner_nmax>0` 拒绝语义丢失 | 拒绝逻辑移到 `init`，保留 `WARNING_QUIT` |
| 两套门控语义合并导致收敛行为变化 | Round 3 以"P01/P13 收敛轨迹"为验收锚点，不追求逐迭代一致 |
| 模板实例化（double/complex）误入复杂分支 | `if constexpr` 保留在 ESolver 接线层，init 校验收口 |

**总验收**：`git diff` 显示 `esolver_ks_lcao.cpp` DeltaP 净删除 ≥350 行、`deltap_pw.cpp` 净删除 ≥150 行；`deltap_solver.h` 消失；`deltap_common` 四个函数全被调用且有单测；P01/P13 回归通过；MPI 2-rank 运行 C-01/C-12 相关场景无 escon/λ 不一致。

---

## 6. 本轮结论

- esolver 侧 DeltaP 逻辑**可以显著简化**：问题主要是"抽象未落地 + 重复实现 + 双状态机 + 死代码"，不是算法本身复杂。
- 数值核心 `module_deltap`（Wilson loop / gauge / branch）不在本次范围内，保持不动，降低回归风险。
- 建议按 Round 1→4 推进；Round 1 纯删除零风险，可先行。
