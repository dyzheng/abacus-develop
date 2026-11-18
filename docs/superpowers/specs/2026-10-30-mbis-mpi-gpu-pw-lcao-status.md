# 2026-10-30 MBIS/约束：MPI 并行、GPU 异构、PW/LCAO 覆盖度评估

## 结论速览
| 维度 | 状态 | 关键证据 / 说明 |
|---|---|---|
| **MPI 正确性** | ✅ 完成 | observe `reduce_pool`；`weight_grid` `MPI_Allgatherv` gather；inject 秩本地；alloc `Allreduce/Allgatherv`；**np=1/2/4 的 Q 完全一致** |
| **MPI 可扩展性** | ✅ **已优化** | rank0 计算 + `MPI_Bcast`(N/ζ) + `MPI_Scatterv`(本地权片, 非 root 无 O(nxyz) 副本)；`fill_weights`/`r`/`N,ζ` 网格循环 **OpenMP**；np=4 与 np=1/2 一致 |
| **GPU 异构** | ✅ **已做（PW + LCAO 网格权重）** | CUDA/HIP 后端 `cuda|rocm/mbis_alm_gpu.*`：pro+权重、N/r 归约、KL；`device=gpu` 的 PW/LCAO 约束已放行（PW 用 `sync_veff_smooth_to_device` 补注入 H2D，LCAO 的 gint 每次自行 H2D）；LiCl PW 端到端 **2.9×**（179 s → 61.5 s），LCAO 约束外步/μ/Q 与 CPU 一致、ETOT 差 3e-10 eV。详见 `2026-11-17-mbis-gpu-heterogeneous-optimization.md` |
| **PW（共线 charge/spin）** | ✅ 布线+测试 | `esolver_ks_pw`：init/inject_potential/observe/energy；MBIS 早期 e2e 在 PW（S6.38） |
| **LCAO（共线 charge/spin）** | ✅ 布线+测试 | `esolver_ks_lcao`：inject_potential_lcao 经 `v_eff`→`cal_gint_vl`→H；LiCl/FeO e2e 在 LCAO |
| **非共线 nspin4** | ⚠️ 部分（**安全**：spin 会显式中止）| nspin4 **charge** 可注入；**spin** 时 `inject` 返回 false 且调用方 `WARNING_QUIT`（`constraint_loop.cpp:920/968`，**非静默**）；完整 nspin4 spin 需 4 分量 inject（未实现）|
| **MBIS 力（生产）** | ✅ **已实现 + 端到端 FD 验证**（S6.82/83）| `constraint_force` 用 `dw_`；`build_derivatives` 无 `is_mbis` 分支 → **落到 Becke 公式**（对 MBIS 错）。模块内 `constraint_force`（dmin 导数）仅用于**单测**；生产路径错 |
| **MBIS 应力（生产）** | ✅ **已实现 + 端到端 FD 验证**（S6.84）| `dwe_` 应变导数；`MbisStressEndToEndFD` PASS |

## 1. MPI
- **正确性已达成**：`ConstraintObserver::observe` 做秩本地积分 + `Parallel_Reduce::reduce_pool`；
  `WeightGrid::build_mbis` 用 `MPI_Allgatherv`（z-最快重排）gather 密度、`compute_mbis` 后 scatter；
  注入写秩本地 `v_eff`；alloc 路径 `MPI_Allreduce/Allgatherv`。实测 `np=1/2/4` Q 一致。
- **扩展性不足**：`compute_mbis` 是**全局串行**（每 rank 冗余全网格迭代），内存 O(nxyz)/rank、
  重复计算 ×np；无 OpenMP。**大体系会内存/算力受限**。
- **改进**：rank0 计算 + `MPI_Bcast`（权重），或 MBIS 域分解/近场；OpenMP 并行 `pro_density_at` 循环。

## 2. GPU
- **已实现（PW + LCAO 网格权重）**：`module_mbis_alm` 新增 CUDA/HIP 后端
  （`mbis_alm_gpu.h`、`cuda|rocm/mbis_alm_gpu.*`、CPU stub）。
  - `pro_weights_kernel`（融合 pro 求和 + 归一化权重 + 最小像距离）、
    `reduce_atom_kernel`（每原子块归约）、`kl_kernel`（KL 诊断）；
  - `w`/`r_min` 常驻显存，主机每步只交换 `N/ζ`；
  - `MbisConfig::use_gpu`，失败回退 CPU（不静默出错）。
- **接线**：`WeightGrid::build_mbis` 在 `device=gpu` 时启用并打印
  `[constraint] MBIS fixed point on GPU`；`constraint_io` 放行
  PW + 非 allocation 的 GPU 约束；`Potential::sync_veff_smooth_to_device`
  在 `inject_potential` 后把宿主 `veff_smooth` 重新上传，GPU PW 哈密顿量才看得到约束势。
- **实测**：LiCl/PW/MBIS/images=1，192³ 网格 6 步 SCF，179 s(CPU) → 61.5 s(GPU)，
  ETOT 一致到 5e-13 eV；模块 gtest CPU/GPU 在 `images=0/1` 下 `N,ζ,D,w` < 1e-8。
- **LCAO**：`gint_vl_gpu` 每次 `cal_gint` 把宿主 `vr_eff` H2D，故注入写宿主 `v_eff` 即可；LiCl/LCAO/MBIS 外步轨迹与 CPU 一致（μ/Q），ETOT 差 3e-10 eV。
- **力/应力**：新增 `module_constraint/cuda|rocm/constraint_mbis_gpu.*` 直算核；
`build_derivatives` 在 MBIS+GPU 时不物化 `dw_/dwe_`（192³/2 原子约省 1.7 GB），
`constraint_force/stress` 直接从 `w_/cw_/zeta/dens/mu` 收缩；gtest 与 CPU 通用核
逐分量一致 <1e-9，FD 力/应力测试仍通过。
- **仍未做**：allocation GPU、多卡每 rank 绑卡、ROCm 实机验证。

## 3. PW vs LCAO
- **两者都已接线**：PW（`inject_potential` 到 PW `v_eff`）、LCAO（`inject_potential_lcao` 经 `v_eff`）。
  MBIS 权重由 `rho_basis`（两基组同一实空间网格）构建，天然共用。
- **测试**：MBIS 早期 e2e 为 **PW**；LiCl/FeO 主 e2e 为 **LCAO**。**共线 charge/spin 两边都通**。
- **非共线 nspin4**：`ConstraintLoop` 有 nspin4 分支、LCAO `alloc_spin_lcao` 有 spinor；
  但 **PW 的 `ConstraintInjectPW::inject` 对 nspin4（veff.nr=4）直接 `return false`** →
  PW 非共线**不支持**；LCAO WeightGrid 的 x/y 自旋注入也**受限**。

## 4. 力 / 应力（**重点缺口**）
- **力（MBIS，生产）**：`constraint::constraint_force` 用 `wg.weight_derivatives()`（`dw_`）。
  `WeightGrid::build_derivatives()` 只有 dipole/Hirshfeld/Becke 分支，**MBIS 落入 Becke** →
  `dw_` 用 Becke 公式，**对 MBIS 电荷/自旋权重是错的**。
  （模块 `mbis_alm::constraint_force` 用 `modulo_cell`/dmin 导数，**正确但仅单测**，
  与生产 WeightGrid 路径**不是同一实现**。）
- **应力（MBIS，生产）**：`constraint_stress` 对 dipole/Hirshfeld `WARNING_QUIT`，
  **无 MBIS 分支/guard** → 静默用 Becke `dw_` → **错**。
- **影响**：SCF-only e2e **不触发**力/应力 → 该缺口**未被测试暴露**；relax/MD/应力会**错**。

## 5. 建议（按优先级）
### P0（正确性）
1. **MBIS 力**：给 `build_derivatives` 加 `is_mbis()` 分支——对**冻结的 `(N,ζ)`** 计算
   `∂w_A/∂R`（Slater `pro` 的 dmin 导数 + 商法则），或把模块 force 提升为生产实现；
   在未实现前**至少 QUIT**（避免静默错）。
2. **MBIS 应力**：加 `is_mbis()` guard（QUIT）或实现应变导数。
3. **PW 非共线**：`inject` 支持 `nspin4`（4 分量）或**显式 QUIT**；LCAO x/y 自旋注入补齐/守卫。
### P1（性能/扩展）
4. MPI：rank0+`Bcast`（去冗余）；OpenMP 循环；必要时 MBIS 域分解。
5. 应力/力在 **PW/LCAO** 的 FD 校验（含 MBIS）。
### P2（GPU）
6. ~~device kernel for `pro_density_at`/`fill_weights`/`compute_mbis`~~ ✅ 已完成
   （CUDA/HIP，PW + LCAO 网格权重）。后续：力/应力 device 核、allocation GPU、多卡 MPI。

## 6. 证据（代码位置）
- `constraint_observe.cpp:106`（reduce_pool）；`weight_grid.cpp` build_mbis（Allgatherv）；
  `alloc_spin_lcao.cpp` / `alloc_pw.cpp`（Allreduce/Allgatherv）。
- `weight_grid.cpp:658 build_derivatives`（无 `is_mbis` → Becke）；
  `constraint_deriv.cpp:212 constraint_stress`（dipole/Hirshfeld QUIT，无 MBIS guard）。
- `constraint_inject_pw.cpp:58`（nspin4 spin inject 拒绝）。
- `esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp`（init/inject/observe/energy 接线）。
- `module_mbis_alm/test/CMakeLists.txt`（`remove_definitions(-D__CUDA)`）。
- 无 `#pragma omp` 于 `module_mbis_alm`/`weight_grid.cpp`。
