# 2026-10-30 阶段 3：MBIS 权重接入 `constraint_loop`（运行验证）

## 0. 结论

`constraint_weight_type mbis` 已接通：`constraint_loop` 在 `observe()` 里用当前
密度重建 MBIS 权重，首个注入前零初始化；**LiCl 自由（delta=0）运行收敛**
（ETOT −643.97565，与 Becke 自由态一致），`CONSTRAINT_AUDIT` 正常输出。
绝对 MBIS 值已由 cube 组件测试验证（Li 2.0447）。**受约束运行（绝对目标）仍慢，
且 ALM 步长未按 MBIS 响应标定**——列为下一阶段。

## 1. 改动

- `constraint_io.cpp`：`constraint_weight_type` 允许 `"mbis"`。
- `constraint_loop.cpp`：
  - 构造：`WeightType::MBIS`；MBIS 时 **`init_zero()`**（首个注入在参考相 mu=0，
    零权重即精确）；非 MBIS 仍 `build()`。
  - `observe()`：`is_mbis()` 时用 `Σ_is rho[is][ir]` 调 `wg_->build_mbis()`，
    再 `ConstraintObserver::observe`。
- `weight_grid.{h,cpp}`：`WeightType::MBIS`、`build_mbis(rho_total)`、`init_zero()`、
  温启动缓存 `mbis_N_/mbis_zeta_`。
- `mbis_alm.{h,cpp}`：`MbisConfig::warm_N/warm_zeta`（跨 observe 热启动，大幅降低
  后续 KL 成本）。
- `source_estate/CMakeLists.txt`：`module_mbis_alm/mbis_alm.cpp` 入主库。

## 2. 运行结果（LiCl PW, serial, −np 1）

- **delta=0（no-op）**：SCF CONVERGED，ETOT **−643.9756525**（与 Becke 自由态
  −643.9756523 一致）；`CONSTRAINT_AUDIT c[0] q=2.032314099 t=2.0323 res=0`。
  - 说明：delta 模式下 target 取自**初始密度**的首个观测（2.0323），因此约束态
    Q 被保持在 2.0323；**收敛自由密度的 MBIS 是 2.0447**（cube 组件测试），
    两者差 0.012 是 delta 语义而非 bug。
- **绝对值 target=2.0**：900 s 内未收敛（末次 `q=2.032, mu=0.05, res=0.032`）。

## 3. 待办（下一阶段）

1. **性能**：MBIS 重建不宜每个 SCF 迭代都做；改为**外层 μ 步/每 K 次**重建
   （热启动已把后续 KL 降到几步，但首次/偶发仍贵）。可加 `mbis_obs % K` 节流。
2. **ALM 标定**：现有 `mu` 步长（`mu=0.05`）未按 MBIS 响应 `chi` 标定；
   需把 `calibrate_alm` 的响应探针接入 `ConstraintLoop` 的 ALM arm
   （当前 arm 可能基于 Becke 导数/固定 alpha）。
3. **MPI**：`build_mbis` 串行守卫；并行需 gather 或 MPI-aware KL。
4. 力/应力、bcc `symmetrize` 开关。

## 4. 意义

- MBIS 权重**真正进入 ABACUS 主约束循环**（编译、运行、自由态收敛、审计）。
- 组件正确性（cube 2.0447、11/11 测试）与运行时集成已分别验证；
  受约束生产化差在**性能节流**与**ALM 响应标定**两项。
