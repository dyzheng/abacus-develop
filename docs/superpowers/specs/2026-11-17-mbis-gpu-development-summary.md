# 2026-11-17 MBIS GPU 异构开发总结（第一轮）

> 权威细节与原始数据见 `2026-11-17-mbis-gpu-heterogeneous-optimization.md`。
> 本文是**压缩版总结**：做了什么、验证到什么程度、还剩什么。

## 0. 一句话

MBIS 的迭代 Slater Stockholder 不动点已从纯 CPU 循环迁移到 CUDA/HIP 设备后端，
`device=gpu` 的 PW 网格权重约束已可端到端运行：LiCl/MBIS 算例 **179 s → ~60 s
（≈3×）**，总能量与 CPU 一致到 **5×10⁻¹³ eV**。

## 1. 动机与瓶颈

MBIS 每步固定点需要
`pro_A(x)=N_A ζ_A³/(8π) Σ_T exp(−ζ_A|x−R_A−T|)`、归一化权重、以及
`N_A, ∫r w_A ρ`。主约束循环在 reference 相每个 SCF 步都重建权重，
在 192³ 网格、2 原子、`images=1` 时每次约 3.8×10⁸ 次 `exp`，是主要开销。
原实现为 `std::vector` + `std::exp` 的 CPU 循环，并且 `constraint` 框架
**显式拒绝** `device=gpu`。

## 2. 架构与改动

### 2.1 设备后端（新）

| 文件 | 作用 |
|---|---|
| `module_mbis_alm/mbis_alm_gpu.h` | `mbis_gpu_available()` / `compute_mbis_gpu()`，返回 bool；失败回退 CPU |
| `cuda/mbis_alm_gpu.cu` | CUDA 实现 |
| `rocm/mbis_alm_gpu.hip.cu` | HIP 镜像 |
| `mbis_alm_gpu_stub.cpp` | 纯 CPU 构建的不可用定义 |

核划分：

1. `pro_weights_kernel`：每线程一个网格点；原子/镜像循环内同时得到
   Slater pro、分区函数 `P`、最小像距离 `r_min`；就地写 `w_A`。
2. `reduce_atom_kernel`：`gridDim.y = atom`；块内树归约 `N_A`、`∫r w_A ρ`；
   每块一次 `atomicAdd`，原子间无竞争。
3. `kl_kernel`：KL 诊断 `D`（用 `pro=wP`，不额外存 pro）。
4. `w`、`r_min` 常驻显存；主机每步只交换 `N/ζ`（2·nat 个 double）。

### 2.2 接线

- `MbisConfig::use_gpu`（默认 false），`compute_mbis` 首行尝试设备后端。
- `WeightGrid::build_mbis`：`device=gpu` 时启用；仅 pool root 计算；
  第一次打印 `[constraint] MBIS fixed point on GPU`，不可用时 WARNING + CPU 回退。
- `constraint_io.cpp`：守卫从「GPU 全禁」放宽为
  **PW/LCAO + 非 allocation + double + 非 double_grid**；allocation 仍显式拒绝。
- `Potential::sync_veff_smooth_to_device()`：约束 `μw` 注入发生在
  `update_from_charge()` 之后，GPU PW 哈密顿量读 `d_veff_smooth`；
  注入后重新上传，否则约束势在 GPU 上静默丢失。`esolver_ks_pw` 调用。
- LCAO：`gint_vl_gpu` 每次 `cal_gint` 都把宿主 `v_eff` H2D，所以注入写宿主
  `v_eff` 即生效，不需要额外同步。
- 力/应力：`build_derivatives` 在 MBIS+GPU 时只置标志，不物化
  `dw_/dwe_`（192³/2 原子约 1.7 GB）；`constraint_force/stress` 用
  `constraint_mbis_gpu` 的 device 核直接从 `w_/cw_/zeta/dens/mu` 收缩，
  只回传 `nat*3` / `3×3`。`mu=0` 时跳过设备调用。
- MPI 分布式：`pool_np>1 && device=gpu` 时每个 rank 只用本地 z-plane 块和
  自己的 GPU，每步仅 `MPI_Allreduce` N/MR；`w` 留在本地，去掉 Gatherv/Scatterv。
  PW 的 GPU 校验会令 `kpar=NPROC`（poolnproc=1），仍走 serial-pool 全网格路径。

### 2.3 CPU 侧顺带优化

- `fill_weights` 复用权重缓冲；`observe`/`constraint_potential`/KL 加 OpenMP；
- 串行 pool 用 `swap` 接管打包权重，去掉一次 `nat*nxyz` 拷贝。

## 3. 验证矩阵

| 项 | 条件 | 结果 |
|---|---|---|
| 模块正确性 | 40³/36³，`images=0/1` | CPU vs GPU `N,ζ,D,w` < **1e-8**，4/4 PASS |
| 模块性能 | 64³，4 原子，`images=1` | CPU ~0.51–0.61 s，GPU ~0.18 s（**2.8–3.3×**） |
| 端到端 | LiCl/PW/nspin1/MBIS/192³/6 SCF | CPU **179 s**，GPU **~60 s**（≈3×） |
| 端到端数值 | 同上 | `FINAL_ETOT` −643.9746485961 eV，CPU/GPU 差 **~5e-13 eV** |
| LCAO 约束 | LiCl/LCAO/MBIS/images=1，150 步 | 外步 iter 20/57/87/111、μ/Q 与 CPU 一致；ETOT 差 **3e-10 eV**（GPU 注入的约束势确实生效） |
| 力/应力 | 48³ MBIS，3 约束 | GPU 直算 vs CPU `dw_/dwe_` 收缩逐分量 < **1e-9**；FD 力/应力测试仍通过 |
| 生产力路径 | LiCl/PW/MBIS，`calculation md` | `CONSTRAINT FORCE` 正常输出（reference 相为零） |
| MPI 分布式 | `weight_grid_mpi` np=4 | GPU 分布式 vs CPU 串行权重 < **1e-8**；6/6 PASS |
| MPI 端到端 | LiCl/LCAO/MBIS np=2 | GPU 分布式 ETOT 与 CPU np=2 差 **7e-13 eV**；18.4 s vs 28.4 s |
| 回归 | MBIS 单测 | 13/13 PASS |
| 回归 | constraint 8 项 | 全部 PASS |
| 构建 | CPU-only `elecstate` + MBIS 单测 | 通过（stub 生效） |

## 4. 关键结论

1. 把 `r_min` 在 pro 核内顺带求出，使归约核从 `O(nat·nxyz·nimg)` 的 sqrt
   循环变成纯访存+FLOPs，是模块加速的主因之一；代价是一份 `nat·nxyz` 显存。
2. 端到端 3× 低于理想，因为 reference 相每步 SCF 都重建固定点，且每次
   伴随 `w` 主机回传 + MPI Scatterv；双精度 `exp` 在消费级 GPU 上吞吐有限。
3. 守卫放宽 + 注入后 H2D 同步，是让 GPU 路径真正进入哈密顿量的关键正确性修复；
   只做 MBIS 计算而不做同步会得到「计算很快但约束无效」的假结果。

## 5. 已知限制

- 每原子每网格点存 `w` 和 `r_min`（≈`2·nat·nxyz·8 B`），超大体系需监控显存；
  失败会自动回退 CPU。
- PW 与 LCAO 网格权重都可用 GPU；allocation、single precision、double_grid 仍拒绝。
  LCAO 的收益只覆盖 MBIS 固定点（gint/DM 仍 CPU 主导）。
- 力/应力直算在 GPU 上每次重新收缩（CPU 路径可复用 dw_/dwe_）；换来的是
  不再分配约 1.7 GB 的导数网格（192³/2 原子）。
- 归约用 `atomicAdd`，重复运行 ETOT 末位有 ~1e-12 eV 抖动。
- 分布式循环每步重新上传 rho/R/images 等常量；后续可缓存成持久设备状态以
  降低 PCIe/分配开销，并在真实多卡节点测量扩展性。
- ROCm 仅代码镜像，未在 AMD 实机验证；多 MPI rank + GPU 路径未在本次实测。

## 6. 下一轮（其他部分的 GPU 异构）

1. ~~LCAO GPU 约束~~ ✅ 已完成：`gint_vl_gpu` 每次 `cal_gint` 自行 H2D，
   外步/μ/Q 与 CPU 一致，ETOT 差 3e-10 eV。
2. ~~力/应力~~ ✅ 已完成：设备直算，跳过 `dw_/dwe_`，与 CPU 收缩 <1e-9。
3. **allocation GPU**：`alloc_observe`/`alloc_pw` 的 `Allreduce/Allgatherv`
   与设备归约合并。
4. ~~多卡~~ ⚠️ 已实现分布式固定点（LCAO/多 rank pool）；下一步减少每步常量重传
   并在真实多卡节点测量扩展性。
5. ROCm 实机回归。

## 7. 复现命令

```bash
# 模块正确性 + 计时
cmake --build build_fix_cuda --target MODULE_ESTATE_mbis_alm_gpu -j
./build_fix_cuda/source/source_estate/module_mbis_alm/test/MODULE_ESTATE_mbis_alm_gpu

# 端到端：LiCl/PW/MBIS/images=1，device=GPU vs CPU
# 输入见 docs/superpowers/specs/2026-11-17-mbis-gpu-heterogeneous-optimization.md §2
```
