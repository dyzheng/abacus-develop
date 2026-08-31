# 2026-08-31: Task 2.5.2——M6 力核 constraint_deriv（F_J = −Σ_α μ_α ∫ρ ∂w_α/∂R_J dr）

> 上游：`docs/superpowers/plans/2026-08-31-task25-m6-force-detail.md` 2.5.2。
> 纯网格力核：密度网格 × 2.5.1 的导数网格，PW/LCAO 共用同一份代码。
> 零新依赖/算法；通道（charge/spin）分支与 reduce_pool 归约沿用 observe 模式。

## 测试计划（失败测试先行）

1. `ForceOnSyntheticDensity`：合成高斯密度 + 双原子几何，核力 vs 两路独立参考
   （M0 核固定点求积、以及 Q 求积的 5 点差分）<1e-8。
2. `NewtonThirdLaw`：刚性平移恒等式 `Σ_J F_J = −Σ_α μ_α dQ_α/dt`（网格
   observer 平移 FD 参考）1e-9。
3. `ForceLinearInMu`：F(μ₁+μ₂)=F(μ₁)+F(μ₂) 1e-12；μ=0 恒零。
4. `SpinChannelReadsMagnetization`：spin 通道读 m=ρ↑−ρ↓（ρ↓=0 时与 charge
   通道一致；ρ↑=ρ↓ 时恒零）1e-12。

## 测试设置

- 系统：H₂O 类几何（O 9.4,9.4,9.4；H1 12.6；H2 6.2，键长 3.2 Bohr，全部避开
  min-image tie-break 边界）；20 Bohr 立方盒；网格 80³（0.25 Bohr）。
- 密度：σ=1.5 Bohr 高斯叠加（nelec 8/1/1），h/σ=1/6。
- 参考求积：delley(35) × baker(140, rcut=14)，固定 O 心单中心。
- 构建：`cmake --build . --target MODULE_ESTATE_constraint_deriv`。

## 结果

| 测试 | 判据 | 实测（最大偏差） |
|---|---|---|
| ForceOnSyntheticDensity（M0 求积参考） | <1e-8 | 2.2e-10 |
| ForceOnSyntheticDensity（Q-FD 参考） | <1e-8 | 2.4e-10 |
| NewtonThirdLaw（observer 平移 FD） | <1e-9 | 7e-13 |
| ForceLinearInMu | 1e-12 / 恒零 | PASS |
| SpinChannelReadsMagnetization | 1e-12 / 恒零 | PASS |
| ctest -R constraint（10 既有 + 本目标） | 全绿 | 11/11 + partition 4/4 |

## 分析

- **求积参考必须复刻 min-image 周期约定**：参考求积最初用直接距离，O 心
  rcut=14 的球越出胞外且横穿 tie-break 平面（H1 平面 x=2.6 处 ρ_H2~2e-3），
  导数积分整片算错分支 → 符号翻转（kernel 与 FD 参考一致、M0 求积差 −3×）。
  复刻 `WeightGrid::min_image_displacement` 的 wrap 后三路一致到 1e-10。
- **多中心求积 ≠ 单中心**：每中心都覆盖全空间，多中心求和会把同一积分叠加
  nat 次（−3× 的"×3"来源）。力参考改用固定单中心（点不随原子移动，无
  移动中心项）。
- **计划"Σ_J F_J ≡ 0"需修正为精确恒等式**：对固定外场密度，Σ_J F_J =
  −Σ_α μ_α∫w_α∇ρ dr ≠ 0（约束同时推密度场）；"≡0"仅对常密度成立，而常密度
  网格和会被 min-image tie-break 尖点污染（O(Δx)，测度零，2.5.1 已登记）。
  连续求积的 ∫w_α∇ρ 对尖的 H 权重收敛差（nrad 140→300 仍移动 1.8e-6）。
  最终参考 = 网格 observer 的刚性平移 5 点 FD（同网格、仅用权重值、无导数
  核）——正是计划"权重随原子刚性移动"的原意，实测与核和一致到 7e-13。
- 力核实现：通道折叠为单密度缓冲（spin=m、charge=nspin1/2 三分支前置注释），
  μ 零乘子短路，`reduce_pool` 池归约 + 累加语义；守卫：导数网格未构建 /
  spin 需 nspin=2 / μ 长度 / force 形状 → WARNING_QUIT。
- 单位：Ry/Bohr（μ in Ry，ρ in e/Bohr³，e=1）。

## 下一步

- 2.5.3：PW 力接线（Forces::cal_force 编排 + 驻点 WARNING 守卫），仿
  forces_onsite.cpp:62 dspin 先例；约束 H₂O 单点力打印非零、μ=0 恒零。
