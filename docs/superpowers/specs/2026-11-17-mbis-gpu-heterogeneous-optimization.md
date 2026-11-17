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

### 4.3 回归

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
- 仅 PW 网格权重允许 GPU；LCAO / allocation / single / double_grid 仍拒绝。
- 设备后端使用双精度 `exp`；未引入 `__expf` 等降精度快路径。
- 归约核用 `atomicAdd`（每块一次），块间顺序不定，重复 GPU 运行的 ETOT
  末位有 ~1e-12 eV 抖动；与 CPU 的差异仍在 1e-12 eV 量级，非实现错误。
- ROCm 代码为 CUDA 的镜像，未在 AMD 硬件上实测。
- 多 MPI rank + GPU 的 pool-root 计算路径未在本次实测（np=2 的 gamma-only
  1 k 点算例被 ABACUS 的 k 点分发生效失败挡住，属输入限制）；单 rank 已端到端验证。

## 7. 下一步

1. 在真实 GPU 节点用多 MPI rank（每 rank 绑卡）验证 pool root 计算 + Scatterv
   的设备路径与显存行为。
2. 视显存压力把 `r_min` 改为 `float` 或按预算切换到「重算 sqrt」的 reduce 核。
3. 评估 LCAO GPU 的 gint/potential 设备同步后放开 LCAO 约束。
4. 为 `constraint_force` / `build_mbis_derivatives`（力/应力）增加设备核。
