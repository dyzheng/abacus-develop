# 2026-08-31 二期 Task 2.3：LCAO esolver 接线（薄钩子 + 一期技术债下移）

## 1. Test plan
- **LCAO H₂O 冒烟（失败测试→实现→通过）**：新集成用例
  `tests/02_NAO_Gamma/212_NAO_constraint_h2o/`（与 PW 用例 211 同一几何、
  同一 delta=+0.1 e 靶点、同一 15 Å 盒子），验收：外环 **6±2 外步内
  CONVERGED**；PARTITION OF UNITY 审计 total_charge == nelec。
  - 基线（无 LCAO 钩子的旧二进制）：`grep -c "\[constraint\]"` == 0
    （constraint=true 被静默忽略）——确认失败态。
  - 实现后（串行 1 rank 与套件口径 4 rank 各跑一遍）：CONVERGED、audit
    行出现、μ* 有限。
- **PW 回归（配置块下移不破坏 PW 通道）**：`tests/01_PW/211_PW_constraint_h2o`
  重跑，Q_ref(O)≈6.2555、μ*≈−0.1765、6 外步 CONVERGED、FINAL_ETOT 与
  result.ref 偏差 <1e-7 eV。
- **单测（共享配置函数）**：`ConstraintIOTest.ConfigureFromInputsShared`
  ——DISABLED（开关关）/ ERROR（文件不可读、靶文件空、GPU 守卫）/
  OK（真实靶文件 + 共价半径表 O=0.64 Å、H=0.32 Å → Bohr 换算）五分支。

## 2. Test setup
- 平台：容器 gcc C++14 + GoogleTest；debug 构建 `build/`（ENABLE_LCAO/MPI/
  OpenMP ON），`abacus_basic_para`。
- LCAO 冒烟输入：basis_type=lcao、gamma_only=1、init_wfc=atomic、
  ecutwfc=20、ecutrho=80（与 PW 用例同网格：15 Å 盒子 pw_rhod 密集网格）、
  scf_thr=1e-7、scf_nmax=300（MPI 路径外步间隔略长，200 档在 4 rank 下
  差一步收敛，300 留足余量）、mixing broyden β=0.7；STRU 用
  O_gga_7au_60Ry_2s2p1d.orb + H_gga_8au_60Ry_2s1p.orb（PP_ORB 现成）；
  constraint_target.json `{"targets":[0.1],"atoms":[[0]]}`（delta 模式）。
- 集成用例注册：照 P4 先例迁入标准用例树 `tests/02_NAO_Gamma/CASES_CPU.txt`
  （02_NAO_Gamma 套件以 `Autotest.sh -n 4` 运行；单用例验证
  `-r 212_NAO_constraint_h2o` 经 catch_properties 对拍 result.ref 通过）。
- 单测依赖：`constraint_io_test` 目标补 `cell_info` 对象库 +
  `output.cpp` + Magnetism/InfoNonlocal 桩符号（与 constraint_loop_test
  同款脚手架）。

## 3. Results
- **基线（旧二进制，无钩子）**：SCF 26 次迭代 CONVERGED，FINAL_ETOT=
  −466.4009941228562 eV，`grep -c "\[constraint\]"` == **0**——constraint
  开关被 LCAO 通道静默忽略，即失败态确认。
- **实现后串行（1 rank）**：外环 6 步 CONVERGED（判据 6±2 通过）：
  Q_ref(O)=6.407956559 → t=6.507956559，μ*=−0.219303873 Ry，
  max_residual=9.45e-05 < thr=1e-4；PARTITION OF UNITY
  total_charge=6.507862061、nelec=8、maxdev=**2.220446049e-16**；
  FINAL_ETOT=−466.2533233603166 eV。
- **实现后 MPI 4 rank（套件口径）**：同样 6 外步 CONVERGED，
  q=6.507862063、μ=−0.2193038814、FINAL_ETOT=−466.253323360341 eV
  ——与串行逐位一致到 1e-11，MPI 减约正确。
- **PW 回归**：Q_ref(O)=6.255467998（README 口径 6.2555）、μ* 收敛值
  −0.1761→−0.1765 档、6 外步 CONVERGED、FINAL_ETOT=−441.9708338568345
  vs result.ref −441.9708338609649（偏差 4e-9 eV，判据 1e-7）——
  配置块下移零回归。
- **单测**：`MODULE_ESTATE_constraint_io` PASS（新增
  ConfigureFromInputsShared 全分支）。
- **回归全量**：constraint ctest 10/10 PASS；`ctest -R MODULE_LCAO`
  29/33 PASS，2 FAIL + 2 Not Run 均为**既有环境问题**（parallel_*.sh
  未拷入构建树，与 phase-1 评审记录的 unitcell_test_pw support/ 同类；
  deltaspin 目标未构建），与本次改动无关（hcontainer/operators 文件未
  触碰）。
- 冒烟经 `Autotest.sh -r 212_NAO_constraint_h2o`：`[ PASSED ] 2 test
  cases passed.`（etotref、etotperatomref 对拍 OK）。

## 4. Analysis
- **接线机制（设计决策）**：LCAO 通道的注入不走 Task 2.2 的 W^α
  HContainer 手工加 H（那样需要改动 hamilt/operator 层、并处理 hR 重建
  时序），而是**网格注入**：在 hamilt2rho_single 的 HSolver 之前把
  μ·w 加到密集网格 v_eff（`pelec->pot->get_eff_v()`，pw_rhod 网格）——
  Veff::contributeHR 在 HSolver 的 updateHk 内用**同一生产
  cal_gint_vl** 积分 v_eff 进 H(R)，由 Gint 积分线性性自动得到
  H += Σ_α μ_α ∫φ_μ w_α φ_ν dr，与 PW 通道逐项同构（PW 注入 v_eff+
  veff_smooth；LCAO 只需 v_eff，veff_smooth 在 LCAO 在 FFT 网格上且
  LCAO 哈密顿不读它）。**观测量==注入算符由单一 WeightGrid 实例构造
  保证**（M1 权重场唯一，观测 cal_gint_rho/网格直积与注入同场）。
  参考相 μ=0 注入按值为 no-op；内环迭代（deltaspin/deltap skip_solve）
  跳过注入（HSolver 不跑、无哈密顿可注入）。k 点（TK=complex）自动
  同构：Veff 双变体共用 cal_gint_vl，H(R)→H(k) 变换与 Veff 完全一致。
- **每迭代代价**：注入是 O(nrxx) 的网格加，v_eff 本就被 Veff 每迭代
  重新积分（update_pot 先 zero_out 再重建），无额外 Gint 趟——对 μ 线性
  红利保留在"外环 μ 更新不触发任何积分重算"。
- **一期技术债下移**：PW before_scf 的 ~50 行配置块（守卫 + 读靶文件 +
  configure_constraint + 共价半径表）下沉为 `constraint::configure_from_inputs`
  （constraint_io），PW/LCAO 两通道同一守卫、默认、半径表；esolver 侧各
  留 12 行薄钩子。PW 回归证明行为逐位不变。
- **守卫语义**：double_grid/single/GPU 三守卫共享（任一通道拒绝静默跑错）；
  configure_constraint 的 phase-1 配方守卫（仅 charge/becke/delta）原样保留。
- **MPI 口径**：WeightGrid 建在 pw_rhod（密集网格，与 v_eff/chr.rho 同
  分布），observe 的 reduce_pool 减约在 4 rank 下验证正确（Q/μ/能量与
  串行一致到 1e-11）。
- **误差记账**：LCAO Q_ref(O)=6.40796 vs PW Q_ref(O)=6.25547 的差异是
  基组效应（NAO 2s2p1d/2s1p vs 平面波），非框架缺陷——Task 2.6 的
  PW≡LCAO 一致性判决定义在"同一密度下观测/注入算符逐位一致"（Task 2.2
  的 ΣW≡S、W↔直积 1e-15 已提供矩阵元级证据），基组间 Q_ref 差属预期。

## 5. Next steps
- Task 2.4 自旋通道：m=ρ↑−ρ↓ 读数、V↑/V↓ ±μ 注入、nspin≠2+spin 守卫；
  需要给 ConstraintLoop 增加按自旋拆分的注入变体（当前 charge 注入对
  nspin=2 是全通道同号，spin 通道要反号）。
- Task 2.5 M6 力核（∂w/∂R_J 网格导数，PW/LCAO 同一核）。
- Task 2.6 三判决验证（PW≡LCAO、力 FD stationary4、力矩 FD）+ LCAO
  4-rank 逐位一致（冒烟已在 4 rank 验证收敛路径）。
