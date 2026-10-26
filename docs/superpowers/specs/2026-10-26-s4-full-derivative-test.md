# S4.1 完整目标轨道导数与约束力测试

日期：2026-10-26

## 1. 测试计划

按 `docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
的 S4.1 闸门验证：

1. 实现固定 `W_i` 下 `K_i(S,W_i)` 对重叠矩阵 `S` 的完整显式导数；
2. 用 `K_i(S+δS)`/`K_i(S-δS)` 中心差分校验稠密伴随矩阵 `E_i`；
3. 把 `E=Σμ_iE_i` 通过现有二中心 overlap 导数核接到 LCAO 力路径，
   得到 `F_S=-Tr(E∂S)`；
4. 固定 `μ` 下比较 S3 冻结力与 S4 完整力，判据 `<0.005 eV/A`；
5. 固定 `μ` 下位置 FD 仍满足 `<0.0129 eV/A`；
6. `abacus_basic_para` 构建及 constraint 回归不得回退。

## 2. 测试设置

- 构建：`cmake --build build --target abacus_basic_para -j 16`。
- 回归：`ctest --test-dir build -R constraint --output-on-failure`。
- 单测：`MODULE_ESTATE_constraint_alloc_constraint`，新增
  `SAdjointMatchesKernelFiniteDifference`。
- 体系：H2O，LCAO、Gamma-only、`nspin=2`、`scf_nmax=100`、
  `constraint_alloc_target_l=2`、`constraint_thr=1e-4`。
- 位置 FD：固定 `μ=-0.01221724241 Ry`，O 的 z 坐标中心差分，
  半位移 `0.005 Bohr = 0.002645886055 A`。
- 固定 μ 使用 JSON v2 条目：

```json
{"constraints": [
  {"type":"spin","target":0.01,"atoms":[0],"fixed_mu":-0.01221724241}
]}
```

三个 INPUT 均设 `constraint_mu_schedule=outer`、
`constraint_branch_tol=0`、`scf_thr=1e-8`。

## 3. 实现

1. `alloc_constraint.{h,cpp}` 新增
   `build_alloc_constraint_s_adjoint()`。对
   `K_i=W_i-UB^TW_i-W_iBU^T+U(B^TW_iB+Q_i)U^T`
   做反向模式微分，并用 `S_T` 的 `R×R` 谱分解计算
   `S_T^{-1/2}` 的 Fréchet 导数；
2. `AllocConstraintLcao` 在几何构建时缓存全局 `S`、`W_i`，并新增
   `s_derivative_force_density()`，按约束通道组合
   `D_i=read_up D_up+read_dn D_dn`，返回
   `E=Σμ_iE_i`；
3. `cal_alloc_constraint_dS_force()` 将 `E` 填入与 DMR 同布局的
   HContainer，调用现有 `hamilt::Overlap::cal_force_stress()`，得到
   `-Tr(E∂S)` 并累加到约束力；
4. S3 的 `Q_cDQ_c` 权重导数力保持不变；S4 只补 `∂S/∂B` 项，避免重复
   计入 `∂W_i`。

## 4. 结果

### 4.1 单测与回归

```text
MODULE_ESTATE_constraint_alloc_constraint     PASS
  SAdjointMatchesKernelFiniteDifference       PASS  (误差阈值 2e-7)
ctest -R constraint                           18/18 PASS (52.48 s)
abacus_basic_para                             build/link PASS
git diff --check                              PASS
```

### 4.2 固定 μ 位置 FD

固定 μ 三点能量：

| 构型 | 总能 (eV) |
|---|---:|
| plus | -466.3984612074578422 |
| minus | -466.4048061262116107 |
| base | -466.4018565608167251 |

解析力与 FD：

| 量 | 数值 |
|---|---:|
| `F_FD = -(E+ - E-)/(2*0.002645886055 A)` (eV/A) | -1.1990158728 |
| S3 冻结解析 O-z 力 (eV/A) | -1.2069823210 |
| S4 完整解析 O-z 力 (eV/A) | -1.2071200771 |
| S4 绝对误差 (eV/A) | **0.0081042043** |
| S4 相对误差 | 0.676% |

判据 `<0.0129 eV/A`：PASS。

### 4.3 S3 与 S4 力差

固定 μ 同一构型下，S3/S4 全原子三分量最大差为
`1.377561e-4 eV/A`，出现在 O-z 分量。判据 `<0.005 eV/A`：PASS。

完整力表：

| atom | S3 Fx | S4 Fx | S3 Fz | S4 Fz |
|---|---:|---:|---:|---:|
| O | -0.0000120319 | -0.0000120319 | -1.2069823210 | -1.2071200771 |
| H1 | -0.8788291326 | -0.8787038541 | 0.6034908881 | 0.6035597661 |
| H2 | 0.8788411645 | 0.8787158860 | 0.6034914329 | 0.6035603109 |

## 5. 分析

1. 稠密伴随单测独立验证了 `Tr(D∂K_i)=<E_i,∂S>`；S4 的改变量虽小，
   但符号与量级和 `∂Q̂/∂P̂` 项一致。
2. S4 力仍存在约 `8.1e-3 eV/A` 的位置 FD 误差，略大于 S3 的
   `7.97e-3 eV/A`；两项都在闸门内。三点能量的二阶差分
   `E+ + E- - 2E0 = 4.458e-4 eV` 非零，说明固定步长中心差分仍含
   曲率/SCF 数值残差；当前数据不足以把剩余误差单独归因于
   `∂Q̂` 公式，但没有出现符号错误或量级跳变。
3. 初次用 ALM 直接重跑 plus/minus 得到 `F_FD=-1.1896 eV/A`，但两帧
   的 `μ` 分别漂到 `-0.01231/-0.01213 Ry`，与 base 的固定 μ 不同，
   该数据无效并已丢弃。合法协议必须使用 JSON v2 `fixed_mu`。
4. S4 新增力路径只修改 `cal_force`；固定 μ 的 plus/minus 总能与 S3
   记录逐帧相差 `<5e-11 eV`，说明能量路径未受影响。

## 6. 下一步

1. S4.2：实现 eq. (6.5) 的应力项，优先复用本次 `E_i` 伴随矩阵与
   现有 overlap 应力核，并补 `∂Q̂/∂ε`；
2. 对应变做三档中心差分，目标相对误差 `<5%`，检查张量对称性；
3. S4.3：补非共线三分量 `λ·σ` 与无 SOC 整体自旋旋转不变性；
4. 文档记录后提交 S4.2/S4.3，并继续 S5 的 PW 平价验证。
