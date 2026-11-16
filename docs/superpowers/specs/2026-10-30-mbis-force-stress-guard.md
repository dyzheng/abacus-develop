# 2026-10-30 P0 修复：MBIS 力/应力不再静默使用 Becke 导数（改为显式 QUIT）

## 0. 缺陷
`WeightGrid::build_derivatives()` 只有 dipole/Hirshfeld/Becke 分支；**MBIS 落入 Becke**
（`w_becke_adjusted_deriv`），对 MBIS 的 Slater/密度依赖权重是**错的**。
`constraint_stress()` 只对 dipole/Hirshfeld `WARNING_QUIT`，**MBIS 静默用 Becke `dw_`**。
SCF-only 路径**不调用** `build_derivatives`（仅力/应力时调用），故 e2e 未暴露；
但 **relax/MD/应力** 会**静默给出错误结果**。

## 1. 修复（本轮，P0）

**更新（S6.82/83）**：力**已实现并端到端 FD 验证**（`build_mbis_derivatives`；`MbisWeightDerivativeFD` 公式 + `MbisForceEndToEndFD` 的 `∫dw_ρ` vs 扰动分数坐标重建 MBIS 的 `dQ/dτ` 均 PASS）⇒ relax/MD 力可信；**应力也已实现并 FD 验证**（`build_mbis_derivatives` 同时构建 `dwe_`；`MbisStressEndToEndFD` PASS）。

- `weight_grid.cpp::build_derivatives()`：开头加 `if (is_mbis()) WARNING_QUIT(...)`，
  明确说明"生产 WeightGrid 未实现冻结分区 MBIS 的 `d w/dR`；SCF 正常，relax/MD/stress 会中止"。
- `constraint_deriv.cpp::constraint_stress()`：加 `if (wg.is_mbis()) WARNING_QUIT(...)`（纵深防御）。
- **SCF 不受影响**（`build_derivatives` 仅在力/应力求值时调用）。
- 编译通过（`abacus_basic_para`）。

## 2. 已推导的正确公式（下一步实现 `build_mbis_derivatives`）
MBIS 权重 `w_A = pro_A/Σ_B pro_B`，`pro_A = N_A ζ_A³/8π · exp(−ζ_A d_A)`；
在**冻结 `(N,ζ)`** 下，原子 A 的 pro 只依赖**自身中心** `R_A`：
```
∂pro_A/∂R_A = −ζ_A · pro_A · u_A ,   u_A = (R_A − r)/d_A   (周期最小镜像, d_A=|r−R_A|)
∂w_alpha/∂R_J = (1_{J∈alpha} − w_alpha(r)) · (∂pro_J/∂R_J) / (Σ_B pro_B)
              = (1_{J∈alpha} − w_alpha(r)) · (−ζ_J) · w_J(r) · u_J
```
（用到 `w_J(r)=pro_J/Σpro` 与 `w_alpha=cw_[alpha]`；数据源：`mbis_zeta_`、`w_`、`cw_`、原子/网格点、
`min_image_displacement`——与 Hirshfeld 导数同框架。）
布局：`dw_[alpha][(J*3+d)*nrxx_+ir] = ∂w_alpha/∂R_J^d`。
**应力**（应变导数 `∂w/∂ε`）更复杂（需 `d_A`、`w_J` 的应变导数），本轮**未实现**（QUIT）。

## 3. 影响
- 当前：MBIS **SCF** 正常；MBIS **relax/MD/应力** 会**显式中止**（而非静默错）。
- 后续实现 `build_mbis_derivatives`（上述公式）+ 应变导数 + 力/应力 FD 校验后，方可放开。

## 4. 文件
`weight_grid.cpp`、`constraint_deriv.cpp`、本日志。
