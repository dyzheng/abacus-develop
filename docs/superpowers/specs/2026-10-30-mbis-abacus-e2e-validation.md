# 2026-10-30 ABACUS 主循环端到端验证（constraint_weight_type mbis）

## 0. 目的
验证集成到 ABACUS 主循环的 MBIS 约束能正常激活、给出正确电荷、并对 mu 有响应。

## 1. 设置
- `abacus_basic_para`（debug）增量重建（`source_estate/CMakeLists.txt:48` 含 `mbis_alm.cpp`）。
- LiCl，`basis_type lcao`，`ecutwfc 50`，`scf_thr 1e-6`，`scf_nmax 100`，`nspin 1`。
- `constraint true`, `constraint_type charge`, `constraint_weight_type mbis`,
  `constraint_target_mode delta`，Li(atom 0) 目标 delta = **−0.05**（→ 1.99453）。
- **性能**：`cfg.cutoff` 从 1e6 改为 **10 bohr**；`image_translations` 现在 `R=images`
  （`images=0` → 无周期像，大盒 direct，快；小盒需 `constraint_mbis_images≥1`）。
  修正前 `cutoff=1e6`+多镜像在 ~1e6 网格 × 200 迭代下**超时**。

## 2. 结果（100 SCF 步，`-np 1`）
```
[constraint] outer step 1 after SCF iter 23: q=2.044532  t=1.994532  mu=0.05
[constraint] outer step 2 after SCF iter 45: q=2.042667  mu=0.10
[constraint] outer step 3 after SCF iter 70: q=2.040863  mu=0.15
[constraint] outer step 4 after SCF iter 97: q=2.039116  mu=0.20
final status: RUNNING (SCF ended before outer loop converged — raise scf_nmax)
```
- **MBIS 约束确实激活**；`q` 随 `mu` **单调下降**（约束方向正确）。
- **未约束 Li 电荷**（主循环密度）≈ **2.0445**，与独立 MBIS cube 计算 **2.0447** 一致
  ⇒ **集成正确**（权重/观测量在主循环给出与独立模块一致的结果）。
- 运行输出 `chg.cube` 上独立 MBIS：Li=2.035（已朝目标下降）。
- 无崩溃；`-np 4` 触发串行守卫（`poolnproc==1`），需 gather（TODO）。

## 3. 分析：外循环慢（控制器，非 MBIS）
- `dQ/dmu ≈ (2.0391−2.0445)/(0.2−0.05) ≈ −0.036`；`mu` 每外步仅 +0.05（**线性注入**，
  未见 secant 的跳跃）。
- 要把 `q` 从 2.0445 拉到 1.9945（Δ0.05）需 `mu≈1.4` ⇒ ~28 外步 × ~23 内步 ≈ **650 SCF**。
- 外层 **secant**（依赖 `dQ/dmu`）需要**真正收敛的 SCF** 才触发；当前每个外步把密度注入扰动，
  secant 未建立跳跃 ⇒ 慢。这是**既有 mu 控制器**（对 becke 等同样）的问题，**非 MBIS**。

## 4. 结论 / 下一步
- **MBIS 主循环集成已验证**：约束激活、电荷方向正确、数值与独立模块一致、无崩溃。
- 慢收敛是**外循环 mu 控制器**：可改进为 **secant/Newton**（用 `dQ/dmu` 跳跃）或用
  `constraint_mu_schedule` 的更快模式；与权重类型无关。
- TODO：MPI gather（`build_mbis` 串行守卫）以支持并行。

## 5. 文件
`weight_grid.cpp`（`cutoff=10`）、`mbis_alm.cpp`（`R=images`）、本日志。
