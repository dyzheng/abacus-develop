# S5.1 PW allocation 的复数 Hermitian 目标 overlap

日期：2026-10-27

计划：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`，Task S5.1。

本轮把 PW allocation 的 `ST=<alpha_a|alpha_b>` 从实对称投影扩展为复数 Hermitian，
使非 Gamma k 点的 projector overlap 虚部进入正交化、观测量与几何导数。

## 1. 测试计划

1. 验证复数 Hermitian 特征分解与重建。
2. 用有限差分验证 `ST^{-1/2}` 的复数 Hermitian Fréchet 导数。
3. 用随机复数 projector 验证目标正交化及几何导数，不再丢弃虚部。
4. 复跑全部 allocation 单测和 2/4-rank MPI 单测。
5. 复跑 Gamma 点与 `1×1×2` 非 Gamma k 点生产算例。

## 2. 实现

1. `alloc_linalg` 新增 `herm_eig()`、`herm_inv_sqrt()` 与
   `herm_inv_sqrt_derivative()`；谱分解使用 LAPACK `zheev`。
2. `PwTargetSpace::ST/ST_inv_root` 改为 `std::complex<double>`。
3. `build_pw_projector_overlap()` 保存完整 Hermitian 矩阵：
   `ST[a,b]=conj(ST[b,a])`。
4. `build_pw_target_derivative()` 使用 `dST=D+D^H`，不再取实部。
5. 标量 overlap 旋转改为 `G^H raw`、`dG^H raw + G^H d_raw`；目标向量本身仍用
   `G phi` 与 `dG phi + G d phi`。

## 3. 结果

### 3.1 单测

```text
ctest -R 'MODULE_ESTATE_constraint_alloc'                         PASS (10/10)
AllocLinalgTest.HermitianEigReconstruction                       PASS
AllocLinalgTest.HermitianInvSqrtDerivativeMatchesFiniteDifference PASS
AllocPwTest.TargetDerivativeMatchesFiniteDifference
  complex random projectors: max_diff=3.070e-10                  PASS
AllocPwTest.AllocationOperatorDerivativeDecomposition
  alpha=0: analytic=-2.568560e+01  fd=-2.568560e+01  diff=-2.557e-09
  alpha=1: analytic=-3.775249e+01  fd=-3.775249e+01  diff=-7.238e-09
MODULE_ESTATE_constraint_alloc_pw_mpi_2np/4np                    PASS
```

### 3.2 生产回归

Gamma 点、`np=4`：

| 量 | 复数表示前 | 复数表示后 |
|---|---:|---:|
| total energy (eV) | -442.158788373908 | -442.158788373908 |
| constraint force O-z (Ry/Bohr) | -0.0003401303 | -0.0003401303 |
| total force O-z (eV/A) | -5.6690414444 | -5.6690414418 |

含非 Gamma k 点的 `1×1×2`、`kpar=1`、`np=4`：

| 量 | 复数表示前 | 复数表示后 |
|---|---:|---:|
| total energy (eV) | -442.114230442330 | -442.114230442337 |
| constraint force O-z (Ry/Bohr) | -0.0005615148 | -0.0005615158 |
| total force O-z (eV/A) | -5.6821171917 | -5.6821172083 |

后两行差异分别为 `7.1e-12 eV`、`1.0e-9 Ry/Bohr`、`1.7e-8 eV/A`，属于保留
虚部后的浮点路径差异，不是物理分支变化。

## 4. 分析

1. 非 Gamma k 点下 projector overlap 并非实对称；先前取实部会破坏正交条件，
   并使目标力缺少合法虚部贡献。
2. 复数化后 `G` 为 Hermitian。目标向量按 `G phi` 旋转，而 overlap 标量按
   `G^H raw` 旋转；这两处共轭约定不同，是本次实现的关键。
3. `m_orth=G m_raw G` 在 Hermitian `G` 下仍成立，因为
   `(G m G)_{ab}=sum_cd G_ac m_cd G_db` 已等价于 `G m G` 的 Hermitian 形式。
4. Gamma 点结果保持不变，非 Gamma 点差异在数值噪声量级，说明改动没有改变已有
   约束定义。

## 5. 下一步

1. 补 PW allocation 应力。
2. 验证 `bndpar > 1` 的能带归属并决定是否解除剩余守卫。
3. 做更密 k 网格与完整 `kpar=2` 收敛回归。
