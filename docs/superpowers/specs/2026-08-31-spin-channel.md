# 2026-08-31 二期 Task 2.4：自旋通道（nspin=2，±μ 拆分注入）

## 1. Test plan
- **失败测试先行（TDD，历史假收敛纪律 T4a'）**：
  - `ConstraintInjectPWTest.SplitInjectionSpin`——type=spin 拆分注入
    `V_up += μ·w、V_dn −= μ·w`（判别量：自旋差势必须真实改变，
    电荷通道不改变它）；单通道缓冲（nspin=1）+ spin → inject 返回 false
    且缓冲原样（P2 契约的 spin 侧）。
  - `ConstraintObserverTest.SpinChannelMagnetizationReading`——读数
    m=ρ↑−ρ↓：网格 δ 探针把自旋读数钉死在电荷读数上（半探针自旋=0、
    电荷=2w，半探针自旋=w、电荷=w）。
  - `ConstraintIOTest.SpinTypeGuard` + `ConfigureFromInputsShared` spin
    分支——type=spin && nspin≠2 → ERROR（含共享路径 PARAM.nspin=1）；
    spin+nspin=2 → OK。
  - `ConstraintLoopTest.SpinChannelConvergesOnLinearResponse`——外环
    全链路：μ=0 自由跑 m≈0 ≠ 靶点 delta（**反假收敛**：不得自由跑
    收敛到非自然靶点），第一割线步把 μ 推离零；随后按线性响应收敛。
- **反向破坏验证（每道守卫各一次）**：
  - 移除 `constraint_io.cpp` nspin 守卫 → 恰 `SpinTypeGuard` +
    `ConfigureFromInputsShared` 2 个 FAIL；恢复 → PASS。
  - 翻转 spin 注入正负号（`V_up +=` → `V_up -=`）→ 恰
    `SplitInjectionSpin` FAIL；恢复 → PASS。
  - 把 `constraint_loop.cpp` 的 response_sign 设 +1 → 恰
    `SpinChannelConvergesOnLinearResponse` FAIL（m 被驱动到 −2 而非
    +0.02）；恢复 −1 → PASS。
- **自旋集成用例**：`tests/01_PW/212_PW_constraint_h2o_spin/`（PW
  nspin=2 H₂O、15 Å 盒子、delta=+0.1 μB on O、初始 mag=0.5），验收：
  外环 CONVERGED + CONSTRAINT_AUDIT 行 + μ* 有限；注册
  `tests/01_PW/CASES_CPU.txt`，`Autotest.sh -n 4` 对拍 result.ref。
- **守卫集成验证**：同一用例 nspin=1 + type=spin → before_scf
  WARNING_QUIT 退出（不静默跑错）。
- **回归**：`ctest -R constraint` 10/10；PW 211（电荷通道）逐位复现；
  037_PW_FM（nspin=2 无约束）不回归；LCAO 212_NAO_constraint_h2o
  （共享 loop 代码）不回归。

## 2. Test setup
- 平台：容器 gcc C++14 + GoogleTest；debug 构建 `build/`（ENABLE_LCAO/
  MPI/OpenMP ON），`abacus_basic_para`（集成用例用 mpirun -n 4）。
- 集成输入：basis_type=pw、nspin=2、nbands=8、init_wfc=random、
  pw_seed=1、ecutwfc=20、ecutrho=80、scf_thr=1e-7、scf_nmax=200、
  smearing gauss σ=0.002、mixing broyden β=0.7；STRU 含 `mag 0.5`（O）
  初始磁矩列；`constraint_target.json {"targets":[0.1],"atoms":[[0]]}`
  （delta 模式）；constraint_thr=1e-4、mu_max=5.0。

## 3. Results
- **单测**：constraint ctest **10/10 全绿**（含 5 个新/扩测试）。
- **反向破坏验证**：三项各恰中目标测试 FAIL（守卫真实判别，非摆设）。
- **自旋集成（4 rank，62.9 s 生成 ref / 60.8 s 校验）**：
  - 参考相（μ=0）：SCF 17 次迭代收敛，m_ref(O)=5.94e-6（H₂O 天然
    无磁——自由跑 m≈0 ≠ 靶点 0.1，反假收敛成立，外环强制续跑）。
  - 外步：step1 μ=−0.05 → q=0.06913；step2 μ=−0.07234 → q=0.09998；
    step3 **CONVERGED**（res=−2.3e-5 < 1e-4），μ*=−0.0723385445 Ry。
  - 实测响应斜率 dQ_m/dμ ≈ **−1.38 e/Ry**（两段 −1.382 / −1.383）。
  - CONSTRAINT_AUDIT：q=0.09998265137 t=0.1000059449 maxdev=2.2e-16；
    etotref=−442.0408674948625 eV（对拍 OK）。
- **守卫集成**：nspin=1 + spin → EXIT=1，warning.log：
  `ESolver_KS_PW::before_scf warning : constraint_type=spin requires
  nspin=2 (...)`——WARNING_QUIT 路径实证。
- **回归**：constraint ctest 10/10；PW 211（μ*=−0.1765、6 外步、
  Q_ref=6.2555）**OK**；037_PW_FM（5 属性）**OK**；LCAO
  212_NAO_constraint_h2o（μ*=−0.2193）**OK**。

## 4. Analysis
- **关键实证发现（推翻实现期假设，如实登记）**：实现期假设"自旋通道
  正响应（dQ_m/dμ>0）"，理由是"V_up += μw 使自旋上更吸引"。集成用例
  实测否定：dQ_m/dμ ≈ −1.38 < 0。物理上，对稳定基态，密度响应函数对角
  元为负（对 μ>0，`V_up += μw` 排斥自旋上、`V_dn -= μw` 吸引自旋下，
  m=ρ↑−ρ↓ 减小）——与 DeltaSpin 的 `E'=E−λ(M−M_t)` 约定同构，其
  "positive λ pushes moment +z" 对应本框架 **μ = −λ**。故自旋通道
  与电荷通道同为负响应，`response_sign` 统一 −1（默认值）；spin 特殊
  +1 覆写已移除，`MuSolverParams::response_sign` 保留为通用参数。
- **符号关系（2.6 力矩 FD 的接线注记）**：本框架收敛 μ 与 DeltaSpin λ
  符号相反（μ=−λ）；"磁力 = −μ"（约束泛函 ∂F/∂M_t = −μ）。2.6
  对标 DeltaSpin 时按此换算，勿直接比 λ。
- **验收口径修正**：评审预期"外环 CONVERGED + μ>0"——μ>0 基于错误的
  正响应假设。实测 delta=+0.1 → μ*=−0.07234（负），与电荷通道符号
  模式一致（delta=+0.1 e → μ*=−0.1765 Ry）。行为正确，仅符号方向与
  原预期相反；已在 README/spec 中如实写明。
- **反假收敛**：μ=0 时 m=5.9e-6，自由跑不得收敛到非自然靶点 0.1——
  参考相结束后外环强制续跑（loop 单测 `SpinChannelConvergesOnLinearResponse`
  亦显式断言），T4a' 纪律在自旋通道落实。

## 5. Next steps
- **Task 2.5 M6 力核**：F_J=−Σ_α μ_α ∫ρ ∂w_α/∂R_J dr（纯网格运算，
  PW/LCAO 同一核），复用 M0 导数核；合成密度解析对拍 + 牛顿第三定律。
- **Task 2.6 判决验证**：PW≡LCAO 逐位一致、力/力矩 FD（对标 DeltaSpin
  时按 μ=−λ 换算）、反假收敛 + LCAO 4-rank。
- **M3b 悬置资产**（评审待办②）：Task 2.6 升格运行时审计（Tr W^α·DM
  vs ∫w_αρ dr）或收尾评审评估删除。

## 附：改动文件
- 实现：`constraint_observe.h/.cpp`（DensityChannel + spin 读数 + nspin
  守卫）、`constraint_inject_pw.h/.cpp`（channel 参数 + 拆分注入 + 单通道
  return false）、`constraint_io.h/.cpp`（nspin 参数 + spin 类型守卫）、
  `mu_solver.h/.cpp`（response_sign 通用参数，默认 −1）、`constraint_loop.cpp`
  （channel 接线 + response_sign=−1 注释）。
- 测试：constraint_io/inject_pw/observe/mu_solver/loop 五个单测目标；
  集成用例 `tests/01_PW/212_PW_constraint_h2o_spin/`（注册 CASES_CPU.txt）。
