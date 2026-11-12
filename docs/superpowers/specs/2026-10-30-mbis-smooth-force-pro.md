# 2026-10-30 光滑周期 pro 用于力（恢复 force FD 测试）

## 0. 问题

最小镜像 pro 在 **wrap 边界不连续** ⇒ 解析力（光滑）与能量 FD 在边界不一致，
`ForceFiniteDifference` 失败。

## 1. 方案：**光滑周期 pro（晶格和）+ 最小镜像均径**

- `pro_density_at`：改为**光滑晶格和** `Σ_T exp(−ζ|x−R−T|)`（对 `R` 光滑，导数=FD）。
- `r`（ζ 的均径）仍用**最小镜像**距离（ζ 更新需要精确对称；力用**冻结 ζ**，故其不连续性不影响力）。
- `constraint_force`：`pro`/`dpro` 都改为**同一晶格和**及其解析导数
  `Σ_T (ζ/|d_T|) exp(−ζ|d_T|) pre·d_T` ⇒ 与 FD **一致**。加 `int images` 参数。

## 2. 结果

- **12/12 测试通过**（`ForceFiniteDifference` 恢复）。
- 真实 cube：
  - **Fe2 bcc (0.50)**：Picard `N=15.9998/16.0001`、`M=±2.849`、ζ=2.569、**ρ(J)=0.309<1**
    （R=1 精度~3e-4；R=2 收敛到 15.99994/15.99994 精确对称）。
  - **LiCl**：N=2.04466/7.95537、ρ=0.295 稳定（不变）。
- coset 探测 2、平移群正确；Picard 为**对称稳定**解。

## 3. 已知问题 / TODO

- `compute_mbis_newton_fp` 在**真实 Fe2 的 sum 映射**上**发散**（诊断解，非主路径；
  主循环用 Picard）。Newton 需加阻尼/线搜索或改用 min-image 映射，后续处理。
- 生产可用 `constraint_mbis_images>=1`（默认 0 → R=1 足够，ζ 尖锐）。

## 4. 文件

`mbis_alm.{h,cpp}`（`pro_density_at` 光滑和、`constraint_force` 和式解析导数 + `images`）、
`test/mbis_alm_test.cpp`（force FD 恢复）、本日志。
