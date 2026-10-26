# S4.3 非共线分配约束 LCAO 生产接线：测试报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S4.3。

## 1. 测试计划

本轮把已通过稠密代数测试的三分量分配约束接入 LCAO `nspin=4` 单 Γ 点生产路径，
验证四件事：

1. 从 spinor `S(R)` 提取的标量 overlap、Gint 标量权重和 `K_i` 能展开为
   `K_i tensor (lambda_i . sigma)` 并注入 complex `H(R)`。
2. 从 ABACUS 的 Pauli 打包 DMR 读出
   `rho0/rho_x/rho_y/rho_z`，得到的 `q_i` 与注入核使用同一套标量指标。
3. MPI 1/2 进程逐迭代能量、约束观测量一致，排除 2D block-cyclic 局部块从
   spinor 分量 1 起始时的映射错误。
4. 无 SOC、固定三分量 `lambda` 下，z 取向与 x 取向的 Kohn-Sham 能量差
   `<1e-6 Ry`；并记录 SCF 密度收敛状态，不能以能量稳定替代完整收敛声明。

位置 FD 力不属于本轮已完成范围：complex `TR` 的 allocation 力路径尚未接线，
因此本轮不关闭 S4.3 验收。

## 2. 测试设置

### 2.1 实现范围

- 新增 `AllocSpinConstraintLcao`：标量 target/`W_i`/`K_i` 构建、spinor kernel
  重建、Pauli DMR 投影。
- `ConstraintLoop` 增加向量 multiplier、target、observable 工作区；非共线
  审计能量使用 `sum_i lambda_i . (q_i - t_i)`。
- `AllocConstraintLcaoOperator<complex<double>, complex<double>>` 注入
  R=0-only complex kernel；`HamiltLCAO` 的 complex 算子链注册该 operator。
- `before_scf` 对 `nspin=4` 限制为：`constraint_alloc_switch=1`、无 SOC、
  单 Γ 点、所有约束 `fixed_mu`。未设置 `noncolin=1` 时，只允许 z 分量；
  任何横向 target/multiplier 都显式拒绝。
- Pauli DMR 和 spinor kernel 的局部块映射改为使用
  `Parallel_Orbitals::get_indexes_row/col`，分别从局部 spinor index 提取
  `s=row%2` 与 `mu=base+row/2`，不再假设本地块从 spinor 0 对齐。

### 2.2 系统与参数

H2O 分子，LCAO，8 电子，`nbands=8`，`ecutwfc=20 Ry`，`ecutrho=80 Ry`，
`nspin=4`，`noncolin=1`，`gamma_only=0`，KPT 仅 Γ 点，
`constraint_alloc_switch=1`，`constraint_alloc_target_l=0`，
`constraint_target_mode=delta`，`constraint_thr=1e-4`，
`constraint_mu_max=5`，Broyden `mixing_beta=0.7`，`smearing_sigma=0.002`。

两个方向使用同一构型和同一约束强度：

```json
{"constraints": [{"type": "spin", "target": [0,0,0.5], "atoms": [0], "fixed_mu": [0,0,0.1]}]}
{"constraints": [{"type": "spin", "target": [0.5,0,0], "atoms": [0], "fixed_mu": [0.1,0,0]}]}
```

主验收运行使用 `scf_thr=1e-8`、`scf_nmax=100`、MPI 2 进程。另有
`scf_nmax=10` 的 MPI 1/2 进程成对运行，用于逐迭代布局一致性检查。

## 3. 结果

### 3.1 MPI 2 进程主运行

z 取向：

```text
q = (0, 0, -0.179919)
CONSTRAINT_AUDIT e_con = -0.06799185831 Ry
max_residual = 0.3200814169
E_KohnSham = -34.2888147117 Ry
SCF converged at iteration 38
FINAL_ETOT_IS = -467.448334436218 eV
```

x 取向：

```text
q = (-0.179919, 0, ~1e-7)
E_KohnSham = -34.2888147104 Ry
SCF density deviation at iteration 100 = 4.08734e-6
SCF IS NOT CONVERGED (density threshold 1e-8)
```

x 的 `E_KohnSham` 与 z 的差为：

```text
Delta E_KohnSham = 1.3000018839e-09 Ry
```

两者在收敛末段固定 `lambda` 的约束能均为约 `-0.0679918583 Ry`，因此
`E_KS + e_con` 的旋转差仍为 `1.3e-9 Ry < 1e-6 Ry`。x 未达到 1e-8 密度阈值
是混合极限环，不把该运行标记为“SCF converged”。

### 3.2 MPI 1/2 逐迭代一致性

在相同 `scf_nmax=10` 的短运行中：

- z：10 步 `E_KohnSham` 的 MPI 1/2 差全部为 `0.000e+00 Ry`；每步
  `q_z` 相同。
- x：10 步最大能量差 `1.0e-10 Ry`（第 9 步，输出末位舍入）；每步 `q` 相同。

这排除了本轮修正前“本地块从 spinor 0 对齐”假设在 MPI 下产生的错配。

### 3.3 回归与构建

```text
cmake --build build --target MODULE_ESTATE_constraint_alloc_spin abacus_basic_para -j 16  PASS
ctest --test-dir build -R constraint --output-on-failure
19/19 tests passed (82.73 s)
```

## 4. 分析

首次 x/z 冒烟的能量差约 `0.093 eV`，原因不是旋转不变性破缺，而是两个早期
输入都遗漏了 `noncolin=1`。ABACUS 在 `nspin=4, noncolin=0` 时设置
`domag_z=true, domag=false`，只合并 z 自旋通道；此时 x 约束仍注入
`K tensor sigma_x`，但交换关联势没有对应的 x/y 响应，必然得到不同的固定点。
补上 `noncolin=1` 后，z/x 的 Kohn-Sham 能量在 `1.3e-9 Ry` 内一致，且
`q` 按同一自旋旋转从 z 轴转到 x 轴，证明 Pauli 展开与 Gint 非共线通道闭合。

MPI 映射是另一个真实缺陷：2D block-cyclic 分布的原子局部块可能以 spinor
分量 1 起始，旧实现的 `2*i+s` 会错把本地第 0 行当成全局 spinor 0。改用
`get_indexes_row/col` 后，MPI 1/2 逐迭代结果一致。

x 取向在 `scf_thr=1e-8` 下的密度残差在 `4e-6` 附近振荡，但能量和三分量矩
稳定到远优于旋转闸门的精度。这是非共线混合路径的收敛速度问题，当前证据不支持
把它归因于 allocation kernel。后续需要单独调整 `mixing_angle`/磁密度混合参数，
或在报告中采用能量与约束观测量双判据；在解决前不宣称完整 SCF 收敛。

## 5. 下一步

1. 接 complex `TR` 的 allocation 位置导数力：复用 S4.1 的
   `d[sum lambda_i . m_i]/dS` 伴随，并生成与 spinor DMR 同布局的实/复
   能量矩阵，走 `Overlap::cal_force_stress()`。
2. 对 H2O x/z 两方向做固定 `lambda` 位置 FD，验收 `<0.0129 eV/A`；完成后
   才能勾选 S4.3 的 SCF 级验收。
3. 增加 adapter 单元测试：Pauli DMR 2x2 block -> 四分量、kernel 的
   Hermitian/block 结构、spin-off-diagonal overlap 忽略规则。
4. 单独处理非共线 SCF 密度极限环（`mixing_angle`/磁密度混合），使 x 方向
   在 `scf_thr=1e-8` 内达到 CONVERGED。
5. 向量 multiplier 的外部调度/ALM 仍保持未实现；当前只支持固定 `fixed_mu`。
