# 2026-08-31 M3a：PW veff 约束势注入

## 1. Test plan
- `PointwiseInjectionNspin1`：固定 veff 背景（sin 摆线）+ 全原子 μ，
  逐点断言 veff_new(ir) == veff_old(ir) + Σ_α μ_α w_α(ir)（机器精度，
  nspin=1）。
- `ChargeChannelOnlyNspin2`：nspin=2 时两个自旋通道各加同一 +dV（耦合
  总电荷）；自旋差通道（上 +dV / 下 −dV，二期磁性）必须保持不变。
- `ObservableEqualsInjectionOperator`：单约束 ∫ρ·(μ w_0)dr == μ Q_0；
  双约束线性性 ∫ρ·(μ0 w0 + μ1 w1)dr == μ0 Q0 + μ1 Q1（1e-10 相对）。
  共享同一 WeightGrid 实例 ⇒ 观测量 == 注入算符（架构原则 2）。
- `SizeMismatchGuard`：μ 长度 ≠ nconstraint 时拒绝注入且 veff 原样保留。
- M1 回归新增 `SetAtomsBeforeBuild`：set_constraint_atoms 在 build() 前
  调用与后调用逐点一致（防御分支 A 锁定）。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest，`MODULE_ESTATE_constraint_inject_pw`、
  `MODULE_ESTATE_constraint_weight_grid`。
- 输入：H2O 20 Bohr 盒 40³ 网格、原子叠加高斯密度（σ=1.0）、
  radii={1.5,0.5,0.5} Bohr；veff 用 `ModuleBase::matrix(nspin, nrxx)`。

## 3. Results
- 4/4 PASS（inject_pw）+ 6/6 PASS（weight_grid，含新增 1 个）。
- 首跑 3 项失败均为测试侧问题，非实现缺陷：① 逐点断言用 EXPECT_DOUBLE_EQ
  对"注入器逐 α 累加 vs 参考单次求和"两种求和顺序做位相等 → 改为
  EXPECT_NEAR 1e-12；② nspin=2 自旋差通道 (2+dv)−(1+dv)≠1 的 1-ulp
  相消误差 → 改 1e-12；③ 测试把 set_constraint_atoms 放在 build() 前，
  触发 w_ 为空 → 越界段错误。
- 修复 1 个鲁棒性缺陷：`WeightGrid::set_constraint_atoms` 在 build() 前
  调用时 `w_` 为空导致越界；加防御分支——w_ 空时仅存 constraint_atoms_
  （build() 会据其派生 cw_），非空时照常重派生，两种调用顺序行为一致。

## 4. Analysis
- 注入公式 veff(ispin, ir) += Σ_α μ_α w_α(ir)：μ 单位 Ry（与 veff 一致），
  权重无量纲；一阶能量贡献恰为 Σ_α μ_α Q_α（E_con 的变分来源）。
- nspin=2 的 charge 通道 = 两自旋通道同加 +dV（耦合 ρ↑+ρ↓）；±μ 自旋差
  耦合属二期磁性约束，明确不在本期实现（有测试锁定其"必须不出现"）。
- 尺寸守卫返回 bool 而非静默跳过：调用方（外环）收到 false 必须
  WARNING_QUIT，避免"约束势没注入但继续跑"的静默错误。

## 5. Next steps
- Task 7 (M5)：constraint_accounting（E_con = Σ μ_α(Q_α−t_α) + 审计行
  key=value 输出）。
- Task 8：外环编排 constraint_loop + esolver_ks_pw 钩子 + H2O 冒烟。
