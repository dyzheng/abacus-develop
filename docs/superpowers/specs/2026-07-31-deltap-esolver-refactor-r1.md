# 2026-07-31 DeltaP esolver 重构 R1：清理死代码 + 修泄漏

> 设计依据：`2026-07-31-deltap-esolver-refactor-design.md` §4 Round 1
> 目标：**零行为变化**地删除死代码、精简 `deltap_common.h`、修复 LCAO 内存泄漏。

## 1. Test plan

1. 编译通过：`esolver`、`module_pwdft`、`abacus_basic_para` 三个目标。
2. 行为不变（A/B 对比）：旧代码二进制 vs R1 二进制，同一算例的 stdout
   DeltaP 相关行逐字节一致；全日志仅允许时间戳/计时列差异。
3. 冒烟收敛：H₂O 30 Bohr，LCAO 与 PW 两通道均 SCF 收敛，exit=0。

## 2. Test setup

- 系统：本机（14 核），Release 构建，MPI 单进程（`mpirun -np 1`），ccache 禁用（沙箱只读缓存目录）。
- 二进制：`build/abacus_basic_para`（旧版 = HEAD 源码；新版 = R1 改动）。
- LCAO 算例 `/tmp/deltap_r1_smoke/h2o_lcao/`：P01 H₂O STRU/KPT（30 Bohr，KPT 1 1 2），
  INPUT：`basis_type=lcao, ecutwfc=50, scf_thr=1e-6, scf_nmax=80, ks_solver=genelpa,
  deltap_switch=1, deltap_corr=1, deltap_inner_nmax=0, deltap_lambda_init=0, deltap_inner_thr=1e-3, deltap_gdir=3`。
- PW 算例 `/tmp/deltap_r1_smoke/h2o_pw/`：P16 STRU_30BOHR/KPT，INPUT：
  `basis_type=pw, ecutwfc=30, nbands=8, berry_phase=1, gdir=3, deltap_switch=true, deltap_corr=1,
  deltap_constraint_mode=total, deltap_inner_thr=1e-2, onsite_radius=2.5`。
- 对比方法：旧源码（`git show HEAD:` 还原）→ 重建 → 采集 `run_pre_{lcao,pw}.log`；
  恢复 R1 源码 → 重建 → 采集 `run_post_{lcao,pw}.log`；过滤时间戳后 diff。

## 3. Results

| 项 | 结果 |
|---|---|
| 编译（esolver / module_pwdft / abacus_basic_para） | PASS（仅既有 `if constexpr` C++17 警告） |
| LCAO 冒烟 | PASS：48 s 收敛，`[rawG] Σγ_raw=-1.268e+01`、`[DeltaP P1]`、`[E-field]` 正常 |
| PW 冒烟 | PASS：40 s 收敛，`[DeltaP-PW]` γ_total/λ/escon 正常 |
| A/B DeltaP 行 diff（LCAO） | IDENTICAL |
| A/B DeltaP 行 diff（PW） | IDENTICAL |
| A/B 全日志 diff | 仅日期行、`DONE(...SEC)` 计时、GE/CG 行末计时列不同（非确定项） |

代码量变化：7 个文件，**+53 / −388**（`deltap_solver.h` 59 行删除；`deltap_common.h` 155→19；
`deltap_pw.cpp` −144；其余为接线调整）。

## 4. Analysis

- R1 改动均为纯删除 + RAII 化，不触碰任何数值路径：
  - 删 `deltap_solver.h`（0 引用）；`deltap_common.h` 删 3 个未用函数（`update_lambda`/
    `to_effective_lambda`/`compute_max_residual`），保留 `compute_dp_escon`。
  - PW：删 `run_deltap_lambda_loop`（恒 false no-op）、`s_active`/`set_deltap_pw_active`/
    `is_deltap_pw_active`（只写不读）、`s_hamilt`/`set_deltap_pw_hamilt`（死代码 + 唯一写点）、
    `inner_nmax>0` 后不可达的 inner-loop 分支、`compute_per_atom_gamma_from_becp`（无调用者）。
  - **保留行为**：`set_deltap_pw_hamilt` 内含"每 SCF 周期重置 `s_lambda_set`/`s_gamma_prev`"，
    重命名为 `reset_deltap_pw_scf_cycle()` 原样保留调用点；`inner_nmax>0` 的 `WARNING_QUIT`
    拒绝语义保留（移到同步步之前，与原来等价）。
  - LCAO：`void* dp_scf_/berry_ovl_scf_/r_overlap_scf_` + 3 个 raw `new`（从不 delete）
    → `std::unique_ptr` + 前向声明，析构自动释放（修 C-21 泄漏）。
- A/B 全等证明数值路径未受影响；唯一一次编译错误（`max_res` 重定义）是删除死分支时
  把块内变量提升到函数作用域所致，已顺手删除该块内从未被读取的死变量，行为不变。

## 5. Next steps

1. R2：LCAO 抽取 `DeltapScfSolver` 状态机（`deltap_scf.{h,cpp}` + CMake/Makefile 挂载），
   `iter_finish` DeltaP 块（esolver_ks_lcao.cpp:695-835）收缩到 ≤10 行接线，
   `deltap_common` 重新引入 `compute_residual/gd_update/to_effective_lambda/unwrap_2pi` 供状态机使用。
2. R3：PW 全局单例 → `DeltapScfSolver` 实例 + backend，统一两阶段门控与离子步重置。
3. R4：MPI rank0+Bcast 读 target/约束矩阵（C-23）、修 `deltap_constraint_lambda_` 回退维度
   （C-22）、real 实例 WARNING 收口（C-26 剩余部分）。
