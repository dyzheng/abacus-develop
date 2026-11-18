# 2026-11-17 MBIS GPU 异构加速与性能优化

分支：`mbis`（`zdy/abacus-develop`，HEAD `bd84bccad`）
目标：把 MBIS（迭代 Slater Stockholder，带布居约束的不动点）从纯 CPU
`std::vector`/`std::exp` 循环迁移到 GPU 异构后端，并优化 CPU 热循环，使主约束
循环在 `device=gpu` 的 PW 运行中真正可用。

## 1. 测试计划

1. **正确性（模块级）**：GPU 与 CPU 的 `compute_mbis` 在 `images=0/1` 下逐原子
   `N, ζ`、KL 诊断 `D`、权重 `w` 一致（相对误差 < 1e-8）。
2. **正确性（端到端）**：LiCl / PW / nspin1 / MBIS / `images=1` 的 6 步 SCF，
   GPU 与 CPU 的总能量一致；GPU 日志明确显示 `MBIS fixed point on GPU`。
3. **性能（模块级）**：64³ 网格、4 原子、`images=1` 的固定点计时。
4. **性能（端到端）**：同上的 192³ LiCl MBIS 运行 wall time。
5. **回归**：既有 MBIS 单测（13 项）与 constraint 模块单测全部通过。

## 2. 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti (sm_86, 8 GB) |
| CUDA | 12.9（conda `nvcc`），驱动 CUDA UMD 13.3 |
| CPU | 12th Gen Intel i7-12700KF，20 线程 |
| 构建 | `build_fix_cuda`，Release，`USE_CUDA=ON`，`ENABLE_MPI=ON`，`ENABLE_LCAO=ON` |
| 体系 | LiCl，PW，nspin1，`ecutwfc=100`，192×192×192 FFT，`constraint_mbis_images=1` |

## 3. 实现内容

### 3.1 设备后端（新增）

- `source/source_estate/module_mbis_alm/mbis_alm_gpu.h`
  设备后端接口：`mbis_gpu_available()` / `compute_mbis_gpu(...)`，返回
  `bool`，失败时由调用方回退 CPU，绝不静默返回错误结果。
- `source/source_estate/module_mbis_alm/cuda/mbis_alm_gpu.cu`
  CUDA 核：
  - `pro_weights_kernel`：每线程一个网格点，融合「Slater pro 求和 +
    高斯归一化权重 + 最小像距离」，一次写出 `w`、`P`、`r_min`；
  - `reduce_atom_kernel`：`gridDim.y = atom`，块内树形归约 + 每块一次
    `atomicAdd`，得到 `N_A`、`∫r w_A ρ`；
  - `kl_kernel`：KL 诊断 `D`（用 `pro=wP` 避免额外存 pro）；
  - 迭代固定点保留在 host（每步仅 2·nat 个 double 往返），`w`/`r_min`
    常驻显存，避免每步重建。
- `source/source_estate/module_mbis_alm/rocm/mbis_alm_gpu.hip.cu`
  HIP 镜像（同一核，API 宏映射），供 ROCm 构建。
- `source/source_estate/module_mbis_alm/mbis_alm_gpu_stub.cpp`
  纯 CPU 构建的「不可用」定义。

### 3.2 接入

- `MbisConfig::use_gpu`（默认 false）；`compute_mbis` 在 `use_gpu` 时先尝试
  设备后端，失败回退 CPU。
- `WeightGrid::build_mbis`：`cfg.use_gpu = (PARAM.inp.device == "gpu")`，
  仅 pool root 计算；日志打印 `[constraint] MBIS fixed point on GPU`，
  GPU 不可用时给出 `WARNING` 并回退。
- `constraint_io.cpp` 守卫：允许 **PW + 非 allocation + double + 非
  double_grid** 的 GPU 约束；LCAO / allocation / single / double_grid 仍然
  显式拒绝（不静默出错）。
- `Potential::sync_veff_smooth_to_device()`：注入 `μw` 发生在
  `update_from_charge()` 已经上传 `veff_smooth` 之后，GPU 的 PW 哈密顿量读的是
  设备缓冲 `d_veff_smooth`，因此注入后重新上传（CPU 为 no-op）。
  `esolver_ks_pw` 在 `inject_potential` 后调用。
- CMake：`.cu` 加入 `device` 对象库（CUDA/ROCm）；CPU-only 时
  `elecstate` 加入 stub；新增 GPU gtest 目标；constraint 测试补齐 MBIS 源。

### 3.3 CPU 热循环优化

- `fill_weights` 复用已分配权重缓冲，避免每次迭代 `assign`。
- `observe` / `constraint_potential` / KL `D` 加 OpenMP（原子/网格独立）。
- `WeightGrid::build_mbis` 串行 pool 直接用 `swap` 接管打包权重，去掉一次
  `nat*nxyz` 拷贝。

## 4. 结果

### 4.1 模块正确性 + 性能（gtest）

```
[  PASSED  ] 4 tests.   (MODULE_ESTATE_mbis_alm_gpu)
  MbisGpu.BackendAvailable
  MbisGpu.MatchesCpuWithoutImages   (N, ζ, D, w < 1e-8)
  MbisGpu.MatchesCpuWithImages      (N, ζ, D < 1e-8)
  MbisGpu.ProductionLikeGridTiming
[mbis_gpu] 64^3 grid, 4 atoms, images=1:
  CPU 0.55–0.62 s, GPU 0.19–0.29 s  (~2–3x)
```

既有 MBIS 单测：`13/13 PASSED`。

### 4.2 端到端 LiCl / PW / MBIS

| 设备 | wall time | FINAL_ETOT (eV) |
|---|---|---|
| CPU（20 线程） | **179 s** | −643.9746485961186 |
| GPU（RTX 3070 Ti） | **61.5 s** | −643.9746485961191 |

- 端到端加速 **≈2.9×**。
- 总能量差 **5×10⁻¹³ eV**（浮点归约顺序差异），即 GPU 与 CPU 数值一致。
- GPU 日志确认：`[constraint] MBIS fixed point on GPU (device backend)`。
- 该算例每步 SCF 在 reference 相都重建 MBIS（192³ 网格、2 原子、
  `images=1` → 每次固定点约 3.8×10⁸ 次 exp），是主要开销来源，故端到端
  收益显著。

### 4.3 LCAO GPU（第二轮）

放开 `basis_type=lcao` 的 GPU 约束（注入写宿主 `v_eff`，`gint_vl_gpu` 每次
`cal_gint` 都会把它 H2D，无需额外同步）。LiCl/LCAO/nspin1/MBIS/images=1：

| 设备 | wall time | FINAL_ETOT (eV) | 外步轨迹 |
|---|---|---|---|
| CPU（20 线程） | 71.7 s | −643.2645362540702 | iter 20/57/87/111，μ = −1/−0.168/−0.220 |
| GPU（3070 Ti） | 79.8 s | −643.2645362537546 | 同上，Q 差 <1e-8 |

- 外步号、每次 `μ`、`Q` 轨迹一致；ETOT 差 **3×10⁻¹⁰ eV**，证明 GPU 注入的
  约束势确实进入了 LCAO 哈密顿量（Q 从 2.0437 → 2.1032 → 2.0504 → 2.0527）。
- 该 LCAO 小算例整体耗时几乎不变：GPU 只加速 MBIS 固定点，其余 LCAO 部分
  （gint/DM）仍在 CPU 侧占主导；大网格 PW 的端到端收益更明显。

### 4.4 力/应力 device 直算（第三轮）

MBIS 的通用力/应力核会先物化
`dw_[nalpha][nat*3*nrxx]` 与 `dwe_[nalpha][9*nrxx]`；对 192³、2 原子、
2 约束，这分别约 **0.68 GB + 1.02 GB**。冻结分割下导数有闭式表达，
因此在 GPU 上直接从 `w_/cw_/zeta/dens/mu` 收缩，只回传 `nat*3` 力与
`3×3` 应力：

- 新核 `cuda|rocm/constraint_mbis_gpu.*`：`mbis_force_kernel_scaled`、
  `mbis_stress_kernel`；每块归约后一次 `atomicAdd`。
- `WeightGrid::build_derivatives` 在 `is_mbis() && device=gpu` 时**只置标志**，
  不分配 `dw_/dwe_`；`constraint_force` / `constraint_stress` 走直算分支，
  应力仍补上体积/Jacobian 对角项。
- `mu` 全零（reference 相）时直接返回零，跳过设备上传。
- 新增 gtest `MbisGpuForceStressMatchesCpu`：同一 MBIS 权重下，GPU 直算与
  CPU 通用 `dw_/dwe_` 收缩逐分量一致（< **1e-9**）。
- MD 生产路径冒烟：`calculation md` + MBIS + `device=gpu` 正常输出
  `CONSTRAINT FORCE`（reference 相力为零，符合预期）。
- 原有 `MbisForceEndToEndFD` / `MbisStressEndToEndFD` 仍通过。

### 4.5 MPI 分布式 GPU 固定点（第四轮）

多 rank pool 不再 gather 到 root：每个 rank 用自己的 GPU 计算本地 z-plane 块，
每次固定点迭代只对 `N_A` 与 `∫r w_A ρ` 做 `MPI_Allreduce`（2·nat 个 double），
`w` 直接留在本地，省掉 Gatherv/Scatterv：

- `MbisGpuLocalWorkspace`：一次性上传 rho/原子/镜像，迭代内只传 `N/ζ`；
  `step()` 复用设备缓冲（不再每步 `cudaMalloc` + 常量 H2D）。
- `mbis_gpu_local_step`：局部网格（`nplane`/`startz_current`）的
  pro+权重+归约设备步；`want_weights=false` 时跳过 D2H 权重回传。
- `WeightGrid::build_mbis`：`pool_np>1 && device=gpu` 时走分布式循环，
  `MPI_Allreduce` 后按 CPU 相同规则 damp/收敛判定；最终再算一次权重。
- LCAO `kpar` 保持 1，pool 内多 rank 共享网格，因此走分布式路径；
  PW 的 GPU 校验会把 `kpar` 自动设为 `NPROC`（`poolnproc=1`），仍走
  serial-pool 全网格 GPU 路径（不冗余、也无需 gather）。

| 项 | 结果 |
|---|---|
| `MODULE_ESTATE_constraint_weight_grid_mpi` np=4 | 新增 `MbisDistributedGpuMatchesSerial`：GPU 分布式 vs CPU 串行权重 < **1e-8**；6/6 PASS |
| LiCl/LCAO/nspin1/MBIS np=2 | GPU 分布式 `FINAL_ETOT` −643.2969955462979 eV vs CPU np=2 −643.2969955462986 eV（差 **7e-13 eV**） |
| 同算例 wall time | GPU 16.3 s（持久 workspace）vs CPU np=2 28.4 s |

### 4.6 回归

```
MODULE_ESTATE_constraint_weight_grid      Passed
MODULE_ESTATE_constraint_observe          Passed
MODULE_ESTATE_constraint_deriv            Passed
MODULE_ESTATE_constraint_inject_pw        Passed
MODULE_ESTATE_constraint_accounting       Passed
MODULE_ESTATE_constraint_loop             Passed
MODULE_ESTATE_constraint_alloc_operator   Passed
MODULE_ESTATE_constraint_alloc_observe    Passed
```

## 5. 分析

1. 设备后端每步只做 3 个核（pro+weight、reduce、KL 仅最后一次），
   主机只交换 `N/ζ` 标量；`w`、`r_min` 常驻显存，符合「大网格、少量原子」
   的 MBIS 形态。
2. 将最小像距离在 pro 核内顺带求出并存 `r_min`，把 reduce 核从
   `O(nat·nxyz·nimg)` 的 sqrt 循环降为纯内存+FLOPs，是模块加速的主因之一；
   代价是多一份 `nat·nxyz` 显存。
3. 端到端 2.9× 低于理想值，原因是：reference 相每步 SCF 重建固定点导致
   每步都伴随 `w` 主机回传与 MPI Scatterv；双精度 `exp` 在 GPU 上吞吐有限。
4. 设备守卫从「GPU 全禁」放宽到「PW 网格权重允许」，并用
   `sync_veff_smooth_to_device` 补上注入后的 H2D；这是让 GPU 路径真正
   进入哈密顿量的关键正确性修复。

## 6. 风险 / 已知限制

- 每原子每网格点存 `r_min`，显存 ≈ `2 · nat · nxyz · 8 B`（另加 `w`）。
  超大体系（nat 数百、网格 >10⁷）需要监控显存；失败会回退 CPU。
- PW 与 LCAO 的网格权重都已允许 GPU；allocation / single precision /
  double_grid 仍显式拒绝。LCAO 的收益只覆盖 MBIS 固定点（其余 gint/DM 仍 CPU）。
- 设备后端使用双精度 `exp`；未引入 `__expf` 等降精度快路径。
- 归约核用 `atomicAdd`（每块一次），块间顺序不定，重复 GPU 运行的 ETOT
  末位有 ~1e-12 eV 抖动；与 CPU 的差异仍在 1e-12 eV 量级，非实现错误。
- ROCm 代码为 CUDA 的镜像，未在 AMD 硬件上实测。
- 多 rank 分布式路径已用 LCAO np=2 与 MPI 单测 np=4 验证；PW 因 ABACUS
  GPU 校验自动设 `kpar=NPROC`，`poolnproc=1`，走 serial-pool 全网格 GPU 路径。
  分布式循环已用持久 `MbisGpuLocalWorkspace` 缓存常量（rho/R/images），
  迭代内只上传 `N/ζ`；真实多卡扩展性仍需实测。

## 7. 下一步

1. 在真实 GPU 节点用多 MPI rank（每 rank 绑卡）验证 pool root 计算 + Scatterv
   的设备路径与显存行为。
2. 视显存压力把 `r_min` 改为 `float` 或按预算切换到「重算 sqrt」的 reduce 核。
3. ~~评估 LCAO GPU 的 gint/potential 设备同步后放开 LCAO 约束~~ ✅ 已完成并验证。
4. ~~为 `constraint_force` / `build_mbis_derivatives`（力/应力）增加设备核~~ ✅ 已完成
   （直算，跳过 `dw_/dwe_`）。
5. allocation GPU（`alloc_observe`/`alloc_pw` 的设备归约）仍未做。
6. ~~多卡：每个 pool rank 绑卡~~ ⚠️ 已实现分布式 GPU 固定点（LCAO/多 rank pool）；
   还需减少每步常量重传，并在真实多卡节点做扩展性测量。
