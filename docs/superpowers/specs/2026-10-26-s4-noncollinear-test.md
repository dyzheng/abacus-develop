# S4.3 统一分配约束非共线核心：测试报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S4.3。

## 1. 测试计划

本轮先验证非共线分配的稠密数学核心，不把尚未接线的生产路径标成完成：

1. `Delta H = K_i tensor (lambda_i . sigma)` 的 spinor 展开与 Pauli 读数
   `m_i = Tr[rho sigma A_i]` 必须一致。
2. 无 SOC 模型中，同时旋转 spinor 密度和约束方向后，Kohn-Sham 项与约束项
   的总能量必须不变。
3. 三分量 multiplier 的 `dS` 伴随必须满足
   `Tr(E dS) = d[sum_i lambda_i . m_i]` 的中心差分。

本轮的 `m_i` 沿用现有 allocation 路径的无量纲归一化（即 `mu_B` 单位）。

## 2. 测试设置

### 2.1 新增实现

- `source/source_estate/module_constraint/alloc_spin.h`
- `source/source_estate/module_constraint/alloc_spin.cpp`

新增接口：

- `build_alloc_spin_kernel()`：把标量 `K_i` 展开为
  `K_i tensor lambda_i.sigma` 的交错 spinor 矩阵。
- `observe_alloc_spin_moments()`：从 `rho0/rhox/rhoy/rhoz` 读取每个约束的
  电荷和三分量矩。
- `observe_alloc_spinor_moments()`：直接从 `2*norb` spinor 密度求迹，
  用来交叉校验 Pauli 分量读数。
- `build_alloc_spin_s_adjoint()`：对
  `E_c = sum_i lambda_i . m_i` 构造 `Tr(E dS)` 的稠密伴随。

### 2.2 测试目标

`MODULE_ESTATE_constraint_alloc_spin`：

- `SpinorAndPauliReadoutsAgree`
- `NoSocGlobalSpinRotationInvariant`
- `SAdjointMatchesSpinFiniteDifference`

无 SOC 旋转模型使用

```text
H_KS = K0 tensor I
H_c  = Kc tensor (lambda . sigma)
rho  = 1/2 [rho0 I + rho_x sigma_x + rho_y sigma_y + rho_z sigma_z]
```

对 spinor 态施加 `U = exp(-i theta n.sigma/2)`，同时对 `lambda` 施加同一个
三维旋转，检查 `m' = R m` 和总能量差。

## 3. 结果

```text
alloc_spin readout: max_dev=1.776e-15 energy_dev=0.000e+00
alloc_spin rotation: max_moment_dev=4.441e-16 dE=0.000e+00 Ry
alloc_spin s-adjoint: predicted=-6.154833622048e+00 fd=-6.154833619654e+00 diff=2.395e-09
```

- `SpinorAndPauliReadoutsAgree`：PASS；直接 spinor 求迹与 Pauli 分量读数最大
  差 `1.776e-15`。
- `NoSocGlobalSpinRotationInvariant`：PASS；三分量矩旋转最大差
  `4.441e-16`，总能量差 `0.000e+00 Ry < 1e-6 Ry`。
- `SAdjointMatchesSpinFiniteDifference`：PASS；解析伴随与中心差分差
  `2.395e-09 < 2e-7`。
- 既有相关回归：`alloc_lcao`、`alloc_constraint`、`weight_grid`（含 1/4 MPI
  进程）、`constraint_deriv` 等 7/7 PASS。
- `git diff --check`：PASS。

## 4. 分析

本轮证明了三件事：

1. Pauli 展开的符号与 ABACUS 现有非共线约定一致，`sigma_x/sigma_y/sigma_z`
   的上下自旋块没有符号漂移。
2. 无 SOC 时，密度与约束方向协同旋转不改变总能量；三分量矩按同一旋转矩阵
   变换。
3. 对固定权重矩阵，三分量约束能的 overlap 导数可逐分量复用 S4.1 标量伴随，
   不需要新的矩阵微积分实现。

本轮**没有**完成生产 LCAO `nspin=4` 接线，原因不是代数失败，而是当前路径仍有
三个结构限制：

1. allocation 的 target/kernel 构建只接受 Gamma-only 实 `HContainer`，而
   ABACUS 输入层明确禁止 `nspin=4` 与 `gamma_only=1` 同时使用。
2. `ConstraintLoop`、ALM 和审计行仍以每约束一个标量 multiplier 为中心；
   三分量 multiplier 需要独立的向量 residual/收敛语义。
3. `AllocConstraintLcaoOperator` 的 `TR=complex<double>` 分支当前是 no-op；
   真正注入需要 spinor block 展开，并决定非 Gamma k 点的 `K_i(k)` 折叠方式。

因此，本轮只能勾掉 S4.3 的“稠密三分量核心 + 旋转/伴随测试”；SCF 级
`nspin=4` 旋转和位置 FD 仍保留为未完成项，不能据此宣称 S4 闸门通过。

## 5. 下一步

1. 为 LCAO `nspin=4` 增加生产构建入口：
   - 从 spinor `S(R)` 的 `(0,0)` block 提取标量空间 overlap；
   - 用 `Gint_vl_nspin4` 的 `{w,0,0,0}` 通道构造标量权重矩阵；
   - 在 Gamma-only 单 k 特例或完整 `S(k)/W(k)` 后端中构建 `K_i(k)`。
2. 扩展 `AllocConstraintLcaoOperator<complex<double>, complex<double>>`：
   把 `K_i(k) tensor sigma_a` 注入 spinor `H(k)`/`H(R)`，先用单 Gamma 点闭环。
3. 从 spinor DMK/DMR 读取 `rho0/rho_x/rho_y/rho_z`，使用本轮
   `observe_alloc_spin_moments()`，再接向量 multiplier 的循环/审计。
4. 向量 multiplier 的收敛规则需单独定义：建议先固定 `lambda` 做旋转不变性
   和位置 FD，再实现逐分量 ALM/信赖域。
5. 集成验收顺序：dense algebra（本轮 PASS）→ LCAO 单 k 能量旋转
   `<1e-6 Ry` → 三分量协同变换 → 位置 FD `<0.0129 eV/A` → 完整回归。
