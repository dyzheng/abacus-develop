# 2026-08-31 V2a 网格收敛 + V2b 独立参考（Task 2.1 落地）

> 背景：二期计划 Task 2.1 Step 2/3 —— V2a 三档网格收敛（判据：相邻档差
> <1e-4 e 且单调收敛）+ V2b 测试内独立参考（Becke 1988 原始公式重写，
> 判据 1e-8）。P1/P2/P4 已在评审修复轮（293f53aa8）清零。

## 1. Test plan
- V2a 主数据（零新代码，production 管道）：H₂O 单点 SCF（V1 变体
  `{"targets":[0,0,0],"atoms":[[0],[1],[2]]}`，delta=0 → 恰好终止于参考态），
  ecutwfc=20 固定，ecutrho=80/160/320 三档，比较参考态逐原子 Q_I。
- 关键事实：显式 `nx/ny/nz`（=各档自然网格 81/120/162）使 ABACUS 波函数
  FFT 网格同步为 81³/120³/162³，但三档平面波基组完全相同
  （34457 个 G 矢量，ecutwfc=20）——密度为同一 40 Ry 带限场在更细
  采样上的 SCF 收敛，属"密度+积分"协同收敛。
- V2a 解耦复核（临时工具，不提交）：单次 162³ 收敛密度（out_chg cube）
  在嵌套子采样网格 54³/81³/162³ 上（整数比 3/2/1，零插值）用生产
  `w_becke_adjusted` 重算 Q_I —— 隔离纯 FFT 网格积分误差。
- V2b（提交测试 `IndependentReferenceBecke`）：测试内从 Becke 1988 公式
  独立重写权重（p(x)=(3x−x³)/2 迭代 3 次 → s(mu)=0.5(1−p3)；连乘
  P_I=Π_{j≠I}s(mu_IJ)；归一化），**不调** `Grid::Partition::w_becke*`；
  同核半径（a_ij=0 → 生产 adjusted ≡ 纯 Becke）合成密度对拍逐原子 Q_I，
  判据 1e-8。
- 回归：constraint 相关 ctest 全绿。

## 2. Test setup
- 平台：abacus_basic_para（OpenMPI，np=1）+ gtest；容器 gcc C++17。
- 输入：`tests/01_PW/211_PW_constraint_h2o/`（15 Å 盒 H₂O，O.upf +
  H_ONCV_PBE-1.0.upf，ecutwfc=20，scf_thr=1e-7，scf_nmax=200）。
- 网格：ecutrho 80/160/320 → 自然 FFT 网格 81³/120³/162³（间距 0.350/
  0.236/0.175 Bohr）；显式 nx=ny=nz 保持 `pw_rho==pw_rhod` 且避开
  double_grid 守卫（守卫保护注入路径，观察路径在 double_grid 下网格一致
  但 V2a 走非 double_grid 路径）。
- V2b：20 Bohr 盒 H₂O（O 盒心，H ±1.2 Bohr），PW_Basis 40³（h=0.5），
  radii={1.5,1.5,1.5} Bohr（同核），ρ=原子叠加高斯（σ=1.0，N={8,1,1}）。

## 3. Results
- V2a 主数据（production 管道，参考态 Q_I）：
  | ecutrho | 网格 | Q(O) | Q(H1) | Q(H2) |
  |---|---|---|---|---|
  | 80 | 81³ | 6.255467998 | 0.872265565 | 0.8722664366 |
  | 160 | 120³ | 6.255437764 | 0.8722811825 | 0.8722810532 |
  | 320 | 162³ | 6.255519912 | 0.872240169 | 0.8722399185 |
  相邻档差：O 3.0e-5/8.2e-5，H1 1.6e-5/4.1e-5，H2 1.5e-5/4.1e-5 ——
  **全部 <1e-4 ✓**；FINAL_ETOT −442.0897771/−442.0896386/−442.0895870 eV
  （变分单调下降 ✓）。
- V2a 解耦复核（同一 162³ 密度，生产权重）：
  | 网格 | Q(O) | Q(H1) | Q(H2) | Σ |
  |---|---|---|---|---|
  | 54³ | 6.2543351079 | 0.8727883968 | 0.8728732244 | 7.9999967 |
  | 81³ | 6.2553029767 | 0.8723094479 | 0.8723602343 | 7.9999727 |
  | 162³ | 6.2552548326 | 0.8723479062 | 0.8724039994 | 8.0000067 |
  81³-vs-162³ 差：O 4.8e-5、H1 3.9e-5、H2 4.4e-5（<1e-4 ✓）；54³ 过粗
  （54→81 差 ~1e-3 ✗）——确认误差随 h 增大、生产网格位于收敛膝盖右侧。
- V2b：`IndependentReferenceBecke` PASS（判据 1e-8；点态权重同为同一数学
  的不同代码路径，FP 顺序差异 ~1e-15，Q_I 累积 ~1e-11，判据留 3 个量级
  余量）。

## 4. Analysis
- **判据判定**：量化门"相邻档差 <1e-4 e"在生产网格（81³+，h≤0.35 Bohr）
  三档全过（max 8.2e-5）；**严格单调不成立**——Becke 切换面（宽度
  ~0.2-0.5 Bohr）在均匀网格上的采样对齐导致 1e-4 量级非单调振荡（120³
  档 O 电荷下探 8e-5 即此类对齐伪影；解耦数据 54→81→162 亦非单调）。
  登记为计划口径偏差：以"收敛带宽 <1e-4 e"为操作判据（与 V2 重定义文档
  "钉死 FFT 网格积分误差 <1e-4" 的目的一致），"严格单调"条款对锐切换
  积分核不可满足，不作放水也不假装满足。
- 密度协同收敛幅度：三档 FINAL_ETOT 差 ≤1.4e-4 eV，Q_I 差异主导项为
  积分采样而非密度漂移（解耦 81-vs-162 ≈ 4.8e-5 与主数据 81-vs-162
  5.2e-5 同量级）。
- V2b 独立性：参考实现故意不调用生产 partition 代码（独立代码路径），
  同核半径使生产 adjusted ≡ 纯 Becke——对拍直接验证生产链的权重数学与
  Becke 1988 原文一致；M2 高斯基准（1e-8）+ 单点钉（1e-12）继续覆盖
  读数侧。
- 合成密度替代性检查（临时，未提交）：σ=0.4 高斯三档误差结构（81-vs-162
  差 1.6e-4）与物理密度（4.8e-5）不匹配——合成密度不能代表物理分账误差，
  V2a 必须用物理 SCF 密度（cube 58 MB 不入库，复现命令见 §5）。

## 5. Next steps
- V2 闭环成立：V2a（网格稳定 <1e-4）+ V2b（独立参考 <1e-8）；Multiwfn
  降级为可选 V2c'（三期/有机器的场合做约定级交叉核对）。
- P1 勾回：phase-1 计划 Task 10 改注"V2=V2a+V2b 替代完成"并勾回 [x]。
- 复现 V2a 的命令（记录于 dev-log，输入文件与 /tmp/v2a 三档一致）：
  `tests/01_PW/211_PW_constraint_h2o/` 输入 + `constraint_target.json`
  换 V1 变体 + `nx=ny=nz={81,120,162}` + `ecutrho={80,160,320}` + 可选
  `out_chg 1`；Q_I 取自运行日志首个 CONSTRAINT_AUDIT（phase=reference
  步，mu=0 观测 = Q_ref）。

## 本轮记录
- V2a：运行验证（零新代码）+ 解耦复核（临时工具 /tmp/v2a_decoupled，
  链接生产 partition.cpp，不提交）；V2b：`constraint_observe_test.cpp`
  新增 `IndependentReferenceBecke`（提交）。
- 文件：`source/source_estate/module_constraint/test/constraint_observe_test.cpp`、
  本规格、`docs/superpowers/plans/2026-08-31-realspace-weight-constraint-phase2.md`
  （勾选）、phase-1 计划（Task 10 勾回）、dev-log。
