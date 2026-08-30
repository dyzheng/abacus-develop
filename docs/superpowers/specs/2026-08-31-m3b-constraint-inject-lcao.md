# 2026-08-31 M3b：LCAO 约束矩阵 W^α_μν（Gint 核）

## 1. Test plan
- `MatrixElementVsDirectGrid`：2 原子 H 二聚体玩具（每原子 1 个数值 s 轨道），
  解析权重场 w1=0.5(1+x/L)、w2=1−w1 定义在 pw_rho 网格上；`build()` 经
  生产 `ModuleGint::cal_gint_vl`（权重扮演局域势）产出每约束
  `HContainer<double>`，与**同一网格上的直接求积**（φ_μ·w·φ_ν·dV 逐点
  求和，dV=Ω/nrxx）对拍全部 4 个矩阵元，相对差 <1e-10。
- `PartitionSumRuleEqualsOverlap`：Σ_α W^α_μν == ⟨φ_μ|φ_ν⟩（重叠矩阵，
  直接求积 S_ref），绝对差 <1e-10（实测 ~2e-15）——单位分解的矩阵元级
  审计。
- `AddWeightedMatchesLinearCombination`：`add_weighted` 逐元素
  H == Σ_α μ_α W^α（μ={0.5,−0.25}），EXPECT_DOUBLE_EQ 位相等。
- `ZeroMuIsNoOp`：μ_α=0 的约束被跳过，H 与去掉该项的手动组合位相等。
- 独立参考说明：直接求积路径在测试内**从 `psi_uniform/dpsi_uniform`
  插值表重建** GintAtom 的立方 Hermite 求值（不调用 Gint 的 set_phi），
  网格点→权重映射独立写（z 最快 FFT 布局），验证的是 Gint 网格-基组
  循环、体元 dr3、HContainer 组装的整体正确性。

## 2. Test setup
- 平台：容器 gcc C++14 + GoogleTest，新增 `MODULE_ESTATE_constraint_inject_lcao`
  （链接 planewave_serial/cell_info/neighbor/container + 显式编译
  module_gint/module_hcontainer/module_ao 源文件——这些对象库按 __MPI
  编译，与串行测试不兼容，故按 module_gint 测试先例直接编译源文件）。
- 输入：20 Bohr 立方盒 40³ 网格（0.5 Bohr 间距，bx=by=bz=1），H 原子
  (8.3,10,10)/(12.3,10,10) Bohr，rcut=5 Bohr，单一归一化高斯 s 轨道
  ψ∝e^{−r²}（Simpson 归一化 ∫r²ψ²dr=1，dr=0.01、dr_uniform=0.005、
  nk=201）；邻居搜索 sr=2·rcut+0.001；GintInfo 全构造（真实 esolver 路径：
  nbx..nbzp 由 PW_Basis_Big 串行 distribute_r 派生）；PARAM gamma_only=true、
  nlocal=2、nspin=1；`itia2iat` 索引表按 LCAO 测试脚手架手工填充。

## 3. Results
- 4/4 PASS（新目标）；constraint 全量 ctest 10/10 PASS（含既有 9 个）。
- 数值余量（调试打印实测）：W↔直接求积最大**相对差 3.41e-15**（判据
  1e-10）；ΣW↔S 最大**绝对差 1.998e-15**（判据 1e-10）；
  W1[0,0]=0.7075、W2[0,0]=0.2925 → 和=1.0000=S[0,0]（归一化轨道对角
  重叠=1 自检成立）；S[0,1]=3.35e-4。
- TDD 反向验证：故意移除 build() 内 cal_gint_vl 调用 → 恰好
  MatrixElementVsDirectGrid 与 PartitionSumRuleEqualsOverlap 两测试 FAIL，
  恢复后 4/4 PASS——测试具备判别力（不是空转断言）。
- 首跑 3 处失败均为测试侧脚手架问题，非实现缺陷：① pw_basis_big.h 需
  先包含 pw_basis.h/pw_basis_sup.h（基类成员不可见）；② PARAM.globalv
  为 const 引用，须写 PARAM.sys（private 成员，靠测试的 private/public
  展开）；③ GintInfo 依赖 iat2it/iat2ia/itia2iat 索引表，UcellTestPrepare
  不填充（LCAO 脚手架模式手工填充后解决）。

## 4. Analysis
- 架构兑现：W^α 走**现成** module_gint vlocal 核（Gint_vl::cal_gint +
  phi_mul_vldr3 + phi_mul_phi → compose/transfer），零新积分框架；权重
  场在 LCAO 侧即 pw_rho 实空间网格（Veff 同款 `get_eff_v` 布局），
  gamma/k 双变体共用同一实空间核（k 点由 esolver 的 H(R)→H(k) 变换处理，
  与 Veff 完全同构）。
- 每几何一次：build() 预积分 W^α（HContainer 与 LCAO 哈密顿同型，
  直接可加）；每 SCF 迭代仅 μ 加权稀疏加（add_weighted，μ=0 跳过），
  对 μ 严格线性——外环 μ 更新不触发重新积分。
- sum rule Σ_α W^α = S 是单位分解的矩阵元级审计：w1+w2=1 逐点成立 ⇒
  任意基组空间下约束矩阵和 ≡ 重叠矩阵，机器精度验证通过。
- 体元一致性：dr3_ = meshgrid_volume = Ω/(nx·ny·nz)，与直接求积 dV
  一致；网格点映射 meshgrids_local_idx ↔ pw_rho 串行 FFT 布局（z 最快）
  逐点吻合。

## 5. Next steps
- Task 2.3：esolver_ks_lcao 三处薄钩子（before_scf 配置+W^α 预建 /
  hamilt2rho 内 H += ΣμW / iter_finish 读数+外步），PW/LCAO 共用
  before_scf 配置块下移；H₂O LCAO 冒烟 + 回归。
- Task 2.4：自旋通道 ±μ（m=ρ↑−ρ↓ 读数、V_↑/V_↓ 分离注入、nspin≠2 守卫）。
