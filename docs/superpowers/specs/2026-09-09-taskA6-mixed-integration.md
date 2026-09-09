# Task A6：H₂O 混合集成用例 + 三旧回归 + μ 耦合实测（G6）

> 批复：`docs/superpowers/specs/2026-09-09-taskA5-review.md`（A5 G5 通过，A6 解锁）。
> 计划：`docs/superpowers/plans/2026-09-08-mixed-charge-spin-stageA.md` Task A6。
> 范围：新集成用例 `tests/01_PW/213_PW_constraint_h2o_mixed/`（注册 CASES_CPU.txt）、
> 三旧用例逐位回归（211/212/212_NAO）、μ 耦合测量（阶段 B 立项数据）。轻量 PW SCF
> （15 Å H₂O、20 Ry、nspin=2），无驻点/力重算。

---

## 1. 测试方案（Test plan）

- **T1 新用例 213**（G6 门）：nspin=2、v2 JSON 约束列表
  `[charge +0.1 e @{O}, spin +0.1 μB @{O}]`（同原子双类型——最直接用户场景），
  跑通 PW SCF 并验收：
  - 双分量均 CONVERGED（res < 1e-4，per-constraint kind= 审计行齐备）；
  - maxdev = 2.2e-16（partition-of-unity 审计）；总电荷守恒；
  - result.ref 入库（etotref 等）。
- **T2 μ 耦合实测**（P3 数据）：记录混合场景 μ_c/μ_s 相对单约束参考
  （211 μ_c=−0.1765、212 μ_s=−0.07234）的偏移；为排除"nspin 1→2"混淆，补一个
  nspin=2 charge-only 基线（同文件 spin target=0 → μ_s≡0 休眠），分解
  耦合偏移 vs nspin 效应。
- **T3 三旧用例回归**：211（PW charge）/212（PW spin）/212_NAO（LCAO charge）
  走旧 v1 格式兼容路径逐位复现——能量与 result.ref 之差、μ 与历史记录一致。
- **T4 sabotage 复验 + 模块全绿**：临时恢复 A1 staging mixed 守卫（扩展 configure
  核）→ 恰中混合测试 FAIL；还原后 `ctest -R MODULE_ESTATE_constraint` 11/11 全绿。

## 2. 测试设置（Setup）

- 平台：容器（`build/`，feat/deltap HEAD 3bf9f4c7d + A5 评审 2 笔未入库）；二进制
  `build/abacus_basic_para` 增量重链（A5 后首链，~10 s）。
- 命令：各用例目录 `mpirun -np 1 /root/abacus-develop/build/abacus_basic_para > log.txt`
  （np1 确定性；与 CI 的 np 无关——能量逐位复现核验以本机 np1 前后对照为准）。
  211 ~112 s、212 ~154 s、212_NAO ~78 s、213 ~395 s、基线 ~216 s。
- 213 输入：INPUT 抄 212（nspin=2、mag 0.5 on O、ecutwfc 20、scf_thr 1e-7、
  broyden 0.7、random init pw_seed 1），run 级 `constraint_type charge`（v2 文件不触发
  supersede）；`constraint_target.json` = v2 `{"constraints":[...]}`；
  STRU/KPT 与 212 一致（15 Å box、O mag 0.5）。
- 基线：213 副本置于 /tmp，仅改 spin target 0.0 + pseudo_dir 绝对路径。

## 3. 结果（Results）

- **T1 213 混合用例**（np1，395 s）：外步 15、SCF 迭代 117 后双通道 CONVERGED；
  审计（running_scf.log 尾部）：
  `CONSTRAINT_AUDIT c[0] kind=charge q=6.355355684 t=6.355369488 mu=-0.1811730734 res=-1.380425046e-05`
  `CONSTRAINT_AUDIT c[1] kind=spin q=0.09992737007 t=0.1000059449 mu=-0.08154434822 res=-7.857485846e-05`
  `CONSTRAINT_AUDIT nconstraint=2 e_con=8.9e-06 max_residual=7.86e-05 total_charge=6.455283054 nelec=8 maxdev=2.220446049e-16`
  `!FINAL_ETOT_IS -441.9159885324229 eV`
  → res_c/res_s 均 <1e-4 ✓、maxdev=2.2e-16 ✓、result.ref 入库 ✓。
- **T2 μ 耦合测量**（P3 数据）：

  | 场景 | μ_c (Ry) | μ_s (Ry) | 说明 |
  |---|---|---|---|
  | 211（nspin1，charge +0.1e） | −0.176548 | — | 单约束参考（历史） |
  | 212（nspin2，spin +0.1μB） | — | −0.072339 | 单约束参考（历史） |
  | nspin2 charge-only 基线（spin target 0） | −0.176354 | 0（休眠） | 本批实测 |
  | **213 混合（同原子）** | **−0.181173** | **−0.081544** | 本批实测 |
  | 耦合偏移（vs 同 nspin 基线/单约束） | −0.004819 | −0.009206 | 两分量同向变负 |

  nspin 1→2 对 μ_c 的影响仅 +1.9e-4（211 vs 基线）；混合耦合偏移 charge −4.8e-3、
  spin −9.2e-3——**耦合不可忽略且同向刚化**（对方通道存在 → 本通道需更负 μ）。
- **T3 三旧回归**（np1，旧 v1 格式兼容路径）：211 CONVERGED μ_c=−0.176548、etot 差
  =4.1e-9 a.u.；212 CONVERGED μ_s=−0.072339、etot 差=9.5e-11 a.u.；212_NAO CONVERGED
  μ_c=−0.219304、etot 差=2.4e-11 a.u.——能量逐位复现、μ 与历史一致。
- **T4 sabotage 复验**：临时恢复 A1 staging mixed 守卫于扩展 configure 核 →
  恰中 3 FAIL（io `MixedGuards`、loop `MixedConvergesOnLinearResponse`、
  `MixedFuseHonorsPerComponentCap`），legacy 全绿；还原后模块 `ctest` **11/11 PASS**
  （~49 s）。

## 4. 分析（Analysis）

- **同原子双类型是 P3 耦合的最干净探针**：w_c = w_s = w_O，观测/注入操作符同几何，
  唯一差异是密度通道；混合 μ 相对单约束的偏移即跨类型耦合（无片段错位混淆）。
- **耦合方向（同向刚化）的物理解读**：电荷 +0.1 e 使体系电子增多、泡利/库仑斥力使
  自旋响应变弱 → 达成 +0.1 μB 需要更负 μ_s（−0.0092）；磁化约束抬高自旋极化态的
  能量也使电荷响应微变 → μ_c 略更负（−0.0048）。量级 ~5-9 meV/Ry·e，非一阶可忽略
  ——**阶段 B 的 Broyden/Jacobian 立项获得直接实测依据**（对角 μ 不再充分）。
- **外步数 15 vs 单约束 3**：混合耦合把 secant 外环收敛拉长约 5×（SCF 迭代 117）。
  这是跨类型耦合的第二个观测证据（收敛速率退化），同样支撑阶段 B 的 Jacobian 更新。
- **诚实登记**：μ_c 偏移分解用 nspin2 charge-only 基线（spin target 0 休眠）；
  "休眠" spin 约束 μ_s≡0 由构造保证（Q_s 自然 ~5.9e-6 < 1e-4 thr）。213 的 μ_c 相对
  211 的总偏移 = nspin 效应（+1.9e-4）+ 耦合（−4.8e-3），表中分列不混报。
- **范围纪律**：未触偶极/Broyden/松紧 SCF 开发（仅登记实测）；无驻点/力重算
  （force FD 留给 A6 后 stationary4 验收轮）；result.ref 为本地 np1 实测入库，
  CI 按既有容差（能量 2 位小数以上）自动比对。

## 5. 下一步（Next Steps）

- 提交本 Task（含 A5 评审归档：review spec 入库、dev-log (4)→(22) 重编号并移至
  (21) 后）并推送 zdy；向用户汇报请求裁决。
- 批准后启动 **Task A7**（收尾文档）：用户手册 §2/§3（新 JSON 格式 + 旧格式
  deprecation + 混合约束边界）；开发者指南（ConstraintSpec 数据模型 + 逐约束
  channel 数据流 + 风险表刷新）；进展总结更新 + 评审申请。
- 阶段 B 立项输入（本批实测归档）：混合耦合偏移（charge −4.8e-3 / spin −9.2e-3）
  与外环收敛退化（3→15 步）——Broyden 立项判据"跨类型耦合是否可忽略"回答为否。
- 开放项跟踪（延续）：① 严格 PW≡LCAO 需只读观测口（A 工具补件）；
  ② torque 脚本 E' 口径修复；③ DeltaSpin 量级对等测量（阶段 B）。

---

## 本轮记录

- 代码改动：`tests/01_PW/CASES_CPU.txt`（注册 213）；新用例
  `tests/01_PW/213_PW_constraint_h2o_mixed/`（INPUT/STRU/KPT/constraint_target.json/
  README/result.ref）。无模块源码改动（sabotage 已还原）。
- 文档：spec（本文件）；计划 A6 勾选；dev-log (22)=A5 评审重编号归档 + (23)=A6 完成。
- 实测命令与时长：213 集成 np1 395 s；基线 np1 216 s；三旧回归 211/212/212_NAO 各
  ~112/154/78 s；模块 ctest ~49 s（11/11）。
