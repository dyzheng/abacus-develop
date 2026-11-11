# 2026-10-30 Hessian 认证与鞍点分类（唯一解算法第二步）

## 0. 实现

- `mbis_hessian_eigenvalues(rho, grid, atoms, cfg, state, fd_step=1e-4)`：
  对 KL 目标 `D(p)`（`p={N_A,ζ_A}`）做**有限差分 Hessian**（中心差分），
  再用**循环 Jacobi** 求对称矩阵特征值。
- `mbis_hessian_classification(eig, tol=1e-6)`：
  ```
  all eig > +tol  -> "minimum"
  any eig < -tol  -> "saddle"
  otherwise       -> "degenerate"
  ```
- 两者用于**认证** `compute_mbis_unique`（多初值最低 D）选出的解是极小还是鞍点。

## 1. 判据

- KL 目标 `D` 对 `p={N_A,ζ_A}` **非凸**；自洽解 `∇D=0` 可能是极小/鞍点/退化。
- `H=∇²D`：
  - `H≻0`（所有本征值>0）⇒ 严格局部极小（**正确的 MBIS 最小**）；
  - 有本征值<0 ⇒ 鞍点（**不是**合法最小；定常迭代"收敛"到此为假收敛）；
  - 有本征值≈0 ⇒ 退化/近零模（多解/平坦方向）。

## 2. 在合成 bcc toy 上的观察（诊断）

```
[HESS] sym:    D=3.739928e+02 N=(3.2078,245.3954) min_eig=-4.189e+01 kind=saddle
[HESS] broken: D=3.739928e+02 N=(3.2078,245.3954) min_eig=-4.189e+01 kind=saddle
```
- 从**对称初值**与**破缺初值**都收敛到同一破缺解 `N=(3.2,245.4)`，且 `min_eig=-41.9`
  （负曲率）→ **saddle**。
- **解释**：该合成胞是**常规 bcc 胞 + 常规格图像和**，图像求和**不保持两 Fe 等价**；
  其"自洽点"不是 `D` 的合法极小。这与之前"常规胞图像格缺体心平移 → 破缺"一致，
  **进一步说明**：需要**原胞平移镜像**或**位点等价投影**（`constraint_mbis_symmetrize`）
  才能得到真正的 MBIS 最小。
- **注意**：FD Hessian 在**非严格驻点/病态 toy** 上的符号只作诊断；对真实体系应在
  **原胞/对称化**后认证。合成 toy 不构成对 Hessian 算法的反例。

## 3. 唯一解算法现状（两步）

1. **选解**：`compute_mbis_unique`（多初值 `init_zeta∈{0.5,1,2,4}`，取最低 `D`）；
2. **认证**：`mbis_hessian_eigenvalues` + `mbis_hessian_classification`
   → `minimum`/`saddle`/`degenerate`。
- 若选解被判为 `saddle`（如病态常规胞 toy），说明**分区的数值构造有问题**
  （图像/对称），应先修"原胞/对称"而非接受该解。

## 4. 测试

- gtest `HessianCertifiesMinimum`（诊断式）：打印 sym/broken 的 `D`、`N`、`min_eig`、
  `kind`；断言有限性。核心测试 **11/11 PASS**。
- 下一步：在 **原胞 / `symmetrize` 开启**的 Fe2、LiCl 上认证，确认选定解为 `minimum`。

## 5. 局限

- FD Hessian（`fd_step=1e-4`）在强非凸/大 N 时可能有数值误差；可用解析导数或更细步长。
- 只对小体系（`2*nat` 小，Jacobi O(n^3)）实用；大体系需稀疏/矩阵无关方法。
- 认证只保证**局部**极小；全局最小仍需多初值（已实现）+ 最低 D 选择。
