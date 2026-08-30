# 2026-08-31 一期评审修复轮（P1–P4 + 轻微项登记）

> 背景：`2026-08-31-phase1-review.md` 裁定一期通过可开二期，附 1 处失信
> （P1）+ 3 处建议修复（P2/P3/P4）+ 3 处轻微记录。本轮回执全部裁定项。

## 1. Test plan
- P1 计划勾选：Task 10 (V2) 改回 `[ ]` 并标注阻塞原因与二期前置转记——
  勾选状态作为验收依据必须诚实。
- P2 inject 返回值契约：`inject_potential` 检查 `ConstraintInjectPW::inject()`
  两路返回值，任一失败 `WARNING_QUIT`；回归 = 既有 constraint 单测全绿 +
  集成用例复跑（构造保证 mu_ 长度派生自权重对象，正常路径不触发）。
- P4 ctest 注册：用例迁入 `tests/01_PW/211_PW_constraint_h2o/`，INPUT 相对
  路径化（pseudo_dir/constraint_target_file），`CASES_CPU.txt` 注册，
  `Autotest.sh -g` 生成 result.ref，非 -g 模式比对通过（与 CI np=4 同参）。
- P3 措辞：核实 deltaspin ctest 实测状态，文档按"4 注册/2 PASS/2 Not Run"
  表述。
- 回归：ctest -R "constraint|partition|read_input|elecstate_pw|weight_grid"
  全绿；deltaspin 2 PASS + 2 Not Run（既有，非本轮回退）。

## 2. Test setup
- 平台：容器 gcc C++17 + GoogleTest + OpenMPI，`build/abacus_basic_para`。
- 集成用例：`tests/01_PW/211_PW_constraint_h2o/`（15 Å 盒 H₂O、O.upf +
  H_ONCV_PBE-1.0.upf、ecutwfc=20、scf_thr=1e-7、constraint delta=+0.1 e
  O 片段、`constraint_target.json` 相对路径、`suffix autotest`）。
- Autotest 命令：`bash ../integrate/Autotest.sh -a <bin> -n 4 -r
  '^211_PW_constraint_h2o$'`（-g 生成参照，无 -g 比对）。

## 3. Results
- P1：计划第 352 行 Task 10 改回 `[ ]`，附阻塞说明（无 Multiwfn）与二期
  前置转记；内部覆盖由 M2 高斯基准承担。
- P2：`constraint_loop.cpp:88-96` 两路返回值检查 + `WARNING_QUIT`
  （头文件契约兑现）；`constraint_loop.cpp:113` 附近 Branch A/B 注释与
  `if (absolute)` 分支对齐修正。
- P4：`git mv tests/constraint_pw_h2o tests/01_PW/211_PW_constraint_h2o`；
  INPUT `suffix autotest` + 相对路径；`CASES_CPU.txt` 追加注册；
  生成 result.ref（`etotref -441.9708338609649`）；非 -g 比对 2/2 OK
  PASSED（etotref/etotperatomref）。运行日志复核：参考相 Q_ref(O)=6.2555、
  约束相 mu=−0.1765509398、第 6 外步 CONVERGED、审计行
  `nconstraint=1 ... total_charge=6.3555 nelec=8 maxdev=2.2e-16`。
- P3：`ctest -N -R deltaspin` = 4 目标注册（#165/#166/#167/#168），
  实跑 #165+#168 PASS、#166+#167 Not Run（可执行文件缺失，既有问题）。
- 回归：constraint 相关 13/13 PASS（含 loop/inject/io/mu_solver/accounting/
  observe/weight_grid(+mpi)）；集成用例 np=4 双跑（生成+比对）PASS。

## 4. Analysis
- P1 是诚实性修复：验收依据文档必须与事实一致，V2 未闭环即 `[ ]`。
  V2 债务 = Multiwfn 补拍（或按 `2026-08-31-v2-redefinition.md` 的
  V2a/V2b 仓库内闭环方案清债），转入二期阻塞性前置项；在补拍完成前
  不作"基组无关/口径正确"对外声明。
- P2 根因：契约（头文件注释）与实现（调用点）分离导致静默失效风险；
  修复采用"守调用点契约"而非"改契约"——因为注入失败静默跑无约束
  SCF 是错误结果而非可接受行为。正常路径不可达（mu_ 长度与权重对象
  同源），但防御成本一行。
- P4 采用 `tests/01_PW/` 规范注册（Autotest 框架），而非自建 ctest：
  与 200+ 既有 PW 用例同一 CI 通道；`result.ref` 以 np=4（CI 同参）
  生成，etot 判据 1e-7 eV。旧 `tests/constraint_pw_h2o/README.md` 详细
  复现说明由 `2026-08-31-v1-v3-validation.md` + m8 spec 覆盖，故删除，
  保留 1 行式 autotest README。
- P3：评审引用我此前聊天报告措辞（"从未生成"），实际文档（dev-log
  评审轮）已用正确表述"2/4 未构建"；本轮回执确认无文档残留错误措辞。
- 轻微项登记（不阻断，转二期）：
  - esolver 钩子实际 ~90 行/4 挂载点（before_scf 配置块 ~50 行、
    hamilt2rho_single 注入、iter_finish 记账、after_scf 终审计）vs 计划
    "~10 行/单钩子"——功能正确，属计划口径偏差；before_scf 配置块
    下移至模块内的重构列入二期。
  - M7 自研 JSON 子集解析器（398 行）未复用 sc_parse_json——功能无问题，
    技术债登记；二期评估复用或保持独立（sc_parse_json 依赖面更大）。

## 5. Next steps
- 二期前置（阻塞）：V2 清债——Multiwfn 补拍（外部）或 V2a+V2b 仓库内
  闭环方案；完成后 Task 10 勾回 `[x]`。
- 二期候选：before_scf 配置块下移模块内；M7 JSON 解析器复用评估；
  deltaspin 未构建目标（#166/#167）补构建（既有，非本阶段）。
- 集成用例已入 01_PW 套件，后续改动须保持 result.ref 同步（-g 重生成）。

## 本轮记录
- 修复轮，代码改动：`constraint_loop.cpp`（P2 + 注释对齐）、
  `tests/constraint_pw_h2o → tests/01_PW/211_PW_constraint_h2o`（P4 迁移+
  注册）、`tests/01_PW/CASES_CPU.txt`、计划文档（P1）、m8 spec/dev-log
  路径同步；文档：本文件 + dev-log 追加 + 评审文档补修复回执。
