# 2026-10-30 MBIS 主循环 MPI gather（支持并行 pool）

## 0. 背景
`build_mbis` 原要求 `poolnproc==1`（串行），并行 SCF 触发守卫退出。

## 1. 实现
- 去掉串行守卫。`rho_total` 是**秩本地**密度（`size = nrxx`）。
- 实空间布局（`pw_grid_bridge.h` / `Parallel_Grid::reduce`）为 **z 最快**：
  全局 `ig = ixy*nz + iz`，本地 `ir = ixy*nplane + iz_local`。
- **Gather**：`MPI_Allgatherv` 把各 rank 的 `nrxx` 块拼成连续缓冲，再**重排**（置换）
  到全局 z 最快网格 `rho_full[ixy*nz + z0 + izl] = cat[base + ixy*npl + izl]`。
- 在 `rho_full` 上运行（全局）`compute_mbis`。
- **Scatter**：`w_[ixy*nplane + izl] = st.w[ixy*nz + startz_current + izl]`。
- 串行池（`nplane == nz`）退化为恒等。

## 2. 验证（LiCl, ecutwfc 50, delta=−0.05, scf_nmax 100）
| np | q(step1) | q(step2) | q(step3) | q(step4) |
|---|---|---|---|---|
| 1 | 2.044532 | 2.042667 | 2.040863 | 2.039116 |
| 2 | 2.044532 | 2.042667 | 2.040863 | — |
| 4 | 2.044532 | 2.042667 | 2.040863 | 2.039116 |

- `-np 1/2/4` **完全一致**；未约束 q=2.0445 与独立 MBIS 2.0447 一致 ⇒ gather 正确。
- 无守卫错误；模块测试 **12/12**。

## 3. 说明 / 后续
- 当前每个 pool rank 都**冗余**运行全局 `compute_mbis`（结果确定、一致）；可优化为
  rank 0 运行 + `MPI_Bcast`（后续）。
- 慢外循环仍是 mu 控制器（item 2）。

## 4. 文件
`source_estate/module_constraint/weight_grid.cpp`、本日志。
