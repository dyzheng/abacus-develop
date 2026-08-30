# 2026-08-31 M8：外环编排 + PW esolver 接线（T8 完成）

## 1. Test plan
- 单测（mock SCF 响应，无 esolver 依赖）：
  - `ReferenceThenConstrained`：参考 SCF 收敛 → 记 Q_ref、target=Q_ref+delta、
    首步 secant μ<0、conv_esolver 被置 false 强制续跑、审计行出现。
  - `ConvergesOnLinearResponse`：归一化线性响应 Q(μ)=Q_ref−μ → 3 个外步
    CONVERGED、μ 收敛到 −delta、最终 Q==target、SCF 正常结束。
  - `FuseUnreachable`：μ 无响应通道 → μ 顶到 mu_max + 平台窗 → UNREACHABLE、
    phase DONE、conv 保持 true（熔断干净退出）。
  - `InjectMatchesObserver`：循环注入后 ∫ρ·ΔV == μ·Q（共享实例），
    veff_smooth 与 v_eff 注入一致。
  - `IgnoresUnconvergedScf`：conv=false 时外步不推进（两阶段门控）。
  - `DisabledNoop`：enabled=false 全 no-op。
- 集成冒烟：H₂O PW 单点 + 单约束 delta=+0.1 e（O 片段）→ SCF 跑通、
  审计行出现、μ 非零、最终 CONVERGED。
- 回归：ctest -R "constraint|partition|read_input|elecstate_pw" 全绿。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest + abacus_basic_para。
- 输入（集成）：`tests/01_PW/211_PW_constraint_h2o/`（15 Å 盒 H₂O、O.upf +
  H_ONCV_PBE-1.0.upf、ecutwfc=20、scf_thr=1e-7、scf_nmax=200、
  constraint_target.json `{"targets": [0.1], "atoms": [[0]]}`）。

## 3. Results
- 单测 6/6 PASS（constraint_loop）+ 既有 constraint 测试全绿。
- 集成冒烟：参考 SCF 14 步收敛（Q_ref(O)=6.2555 e）→ 约束相 μ=−0.05→
  −0.1765，第 6 个外步 CONVERGED（res=3.06e-5 < 1e-4），
  FINAL_ETOT=−441.9708 eV，exit 0。
- 修 1 个设计缺陷：`on_scf_converged` 未按 conv_esolver 门控 → 外步在每个
  （未收敛）SCF 迭代都执行，secant 追混合噪声不收敛；加门控后收敛干净。
- 修 1 个解析缺陷：constraint_io 嵌套 fragments `[[0], [1], [2]]` 在分隔
  逗号后带空白时误报 "unterminated atoms array"（while 条件前缺 skip_ws）；
  新增 `NestedSingleElementFragments` 单测锁定。

## 4. Analysis
- 注入路径：hamilt2rho_single 在 run_deltaspin_lambda_loop 先例处注入
  get_eff_v() + get_veff_smooth()（CPU double 下 operator 直接读
  veff_smooth.c）；update_pot 每轮重建 veff 后由下一迭代重新注入。
- 两阶段门控（DeltaP 4.3 谱系）：参考相 μ=0 首个 SCF 收敛记 Q_ref；
  约束相每 SCF 收敛执行 M4.step，RUNNING 时强制 conv=false 续跑，
  CONVERGED/UNREACHABLE 时正常结束。未收敛 SCF 一律不推进外步。
- E_con 汇入总能量：fenergy 新增 cc_escon（calculate_etot 含入），
  迭代末尾刷新；收敛端点 e_con→0（约束自洽后能量修正消失），
  FINAL_ETOT 含 cc_escon。
- 半径来源：ModuleBase::CovalentRadius（Å→Bohr）；phase-1 守卫
  （double/CPU/非 double_grid）WARNING_QUIT 拒绝静默跑错。
- 性能：81³ 网格单步 ~1.5 s，参考 SCF 14 步，每个约束外步 ~10 SCF 步。

## 5. Next steps
- Task 9-12（V1/V2/V3）：sum rule 8/8 已验（V1）；V3 可达性扫描
  ±0.05/0.1/0.2/0.3 + 熔断；V2 需 Multiwfn 对拍（外部工具）。
