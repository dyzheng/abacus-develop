# 2026-10-30 不动点 Jacobian 判稳 + 约束 Newton（保 sum rule）

## 0. 实现

- `compute_mbis_newton_fp(rho, grid, atoms, cfg, n_starts)`：
  对**约束不动点** `G(p)=F(p)−p=0` 做 Newton（`F` 为 Picard 一步：
  `N_A=∫w_Aρ, ζ_A=3N_A/∫r w_Aρ`）。`F` 的输出**天然满足** `ΣN=∫ρ`，故不动点保 sum rule。
  多初值 + 最低 D 选解。
- `mbis_fixedpoint_spectral_norm(rho, grid, atoms, cfg, state)`：
  FD 求 `J=∂F/∂p`，算 `J^T J` 的对称特征值 → `||J||_2`（最大奇异值，即
  **Picard 谱半径的保守上界**）。判据：`||J||_2<1` ⇒ 必收缩；`≥1` ⇒ **不可证收缩**
  （可能病态/不稳定）。

## 1. 结果（`make_toy`）

```
[FP] picard D=6.2911e+02  newton D=6.2911e+02  ||dF/dp||_2=32.7311
```
- **Newton 与 Picard 到达同一不动点**（`D` 相同），且 **sum rule 保持**（`ΣN=∫ρ`）。
- `||∂F/∂p||_2 = 32.7 ≫ 1` ⇒ **不能证明 Picard 收缩**：不动点映射可能病态/不稳定
  （存在大奇异值/慢模）。这解释了为何 Picard 可能慢甚至破缺。

## 2. 意义

- **正确求解器**是**约束不动点**（保 sum rule）；Newton（`G=F−p`）比 Picard 更可靠，
  且同样保 sum rule。
- **稳定性**由不动点映射 `∂F/∂p` 决定；自由 D 的 Hessian **不是**该稳定性对象
  （自由最小化本身违反 sum rule，已删除）。
- 当前通用框架：
  1. 通用周期图像和（`detect_cosets`，含 bcc 体心平移）；
  2. 约束不动点求解（`compute_mbis` / `compute_mbis_newton_fp`）；
  3. 多初值 + 最低 D 选解（`compute_mbis_unique`）；
  4. Jacobian 谱范数（**保守**稳定性诊断）+ 自由-D Hessian（启发式）。

## 3. 局限 / 下一步

- `||J||_2` 只是**谱半径的保守上界**；要得**真实稳定性** `ρ(J)`，需对**非对称** `J`
  求**特征值**（小型 QR 特征求解器），而非奇异值。下一步实现。
- `ρ(J)>1` 的方向即**不稳定/破缺方向**；可用其**左/右特征向量**定位并处理。
- 合成 bcc toy 病态；最终需在**真实 LiCl/Fe2** 上：探测平移 + 稳定不动点 + 认证。
- 11/11 测试通过。
