# 2026-09-14 `constraint_inner_thr` 三档标定（用户优先级 2）

- 前置：双迭代调度落地（commit afb97690d/a95014bd9）与 C-29 修复（commit 098091b4d）；
  用户批复："随后启动优先级 2：inner_thr 三档扫描（1e-3/1e-4/1e-5 on 212/213）"。
- 动机：`constraint_inner_thr` 默认 1e-3 是沿用 DeltaSpin 口径的**未标定值**（手册
  §5.9 已注明）。它决定"drho 降到多低才开始在 SCF 内更新 μ"，直接决定内环更新次数、
  mixing 复位次数与逼近段长度——是 INNER 调度唯一还有自由度的旋钮。
- 证据目录：`tests/deltap_inner_thr/`（README + run 脚本 + `results/*.audit` +
  `summary.txt` + `mgo_crosscheck.txt`）。

## 1. 测试计划

| # | 问题 | 判据 |
|---|---|---|
| Q1 | 门控值改变是否改变**答案**（μ*、E_tot）？ | μ* 相对差 < 1%、`E_tot` 差 < 1e-6 eV（沿用 Q1–Q3） |
| Q2 | 门控值如何影响**成本**（SCF 迭代数 = 对角化次数代理）？ | 相对同体系 OUTER 基线的变化率 |
| Q3 | 机制读数（内环 μ 更新次数、`MIX_RESET` 次数、reset cost、settle 事件）随门控如何变？ | 逐 run 审计行统计 |
| Q4 | 门控收紧会不会让 settle 检查"失效"（即它只对松门控有意义）？ | 三档下 settle 通过/反弹计数 |
| Q5 | 是否应改默认值？ | Q1+Q2 在**多体系**上的一致方向 |

## 2. 测试环境

- 算例：`tests/01_PW/212_PW_constraint_h2o_spin`（PW，nspin 2，自旋约束 δ=+0.1 μB）、
  `tests/01_PW/213_PW_constraint_h2o_mixed`（PW，混合 charge+spin 各 δ=+0.1）；
  补充：`tests/deltap_mgo_scan` 的 S3 δ=+0.5 e（LCAO，MgO 体相 8 原子胞）。
- 参数：同 Q1–Q3 对照口径——同二进制 `build_rel/abacus_basic_para`（Release，当前
  HEAD）、同网格/靶点/初猜，**只有 `constraint_inner_thr` 变**；`np=4`，
  `OMP_NUM_THREADS=1`（开发者文档 §3.5）；`constraint_inner_nmax 200`。
- MgO 补充点热启动源为 160/640 的旧扫描（跨基组重启）——即原 C-29 触发模式，
  本轮已由 `read_rhog` 守卫修复，顺带复验。

## 3. 结果

主表（SCF = 总 SCF 迭代数；inn/MIX = 内环 μ 更新/`mix_reset` 次数）：

| 体系 | OUTER | INNER 1e-3 | INNER 1e-4 | INNER 1e-5 |
|---|---|---|---|---|
| 212 PW 自旋 | 42（outer 3） | 44，inn 27，MIX 26 | **38**，inn 11，MIX 10 | 39，inn 10，MIX 8 |
| 213 PW 混合 | 117（outer 15） | 63，inn 28，MIX 26 | 62，inn 25，MIX 23 | 64，inn 24，MIX 22 |
| MgO 体相电荷 | 381（outer 23） | **94**，inn 31，MIX 30 | 107，inn 22，MIX 21 | 129，inn 22，MIX 21 |

- Q1（正确性等价）：全部 6+2 个 run `CONVERGED`；以各体系 1e-3 为参照，
  `max|Δμ|/|μ|` ≤ 0.34%（212 0.34% / 213 0.16% / MgO 0.0115%），
  `max|ΔE_tot|` ≤ 4.2e-7 eV（212）/ 5.2e-8（213）/ 2.4e-8（MgO）——**全部命中**判据。
- Q2（成本，相对 OUTER）：
  - 212：+4.8%（1e-3）→ **−9.5%**（1e-4）→ −7.1%（1e-5）；
  - 213：−46.2% → −47.0% → −45.3%（基本平）；
  - MgO：**−75%**（1e-3）→ −72%（1e-4）→ −66%（1e-5）（收紧变差）。
- Q3（机制）：门控收紧 ⇒ 内环 μ 更新次数单调下降（212 27→11→10、213 28→25→24、
  MgO 31→22→22），`MIX_RESET` 同降；reset cost（`mixing recovered after K`）随之间隔
  变大（212 1→1→2；213 2→3→4；MgO 4→?）。
- Q4（settle）：**每个门控值下 settle 仍在触发**（212 1e-5 抓到 1 次反弹、213 三档各
  1 次、MgO 三档 0 次）⇒ settle 不是"松门控的补丁"，而是独立的安全网。
- Q5：方向在体系间**翻转**（212 要收紧、MgO 要放松、213 无所谓）⇒ 不存在"更好"的
  全局默认；1e-3 在三个体系里两个最优（MgO、与 213 实质并列）。

## 4. 分析

**门控的作用机制**：`drho < inner_thr` 才允许内环读观测量并更新 μ。因此门控收紧有
两个相反后果——(a) 每次 μ 更新都在更"干净"的密度上发生、更新次数更少 ⇒ mixing 复位
更少、churn 更小（省）；(b) 达到门控所需的 SCF 迭代更多、逼近段更长，且 μ 更新更少
意味着同样要走的 μ 路被摊到更多 SCF 迭代上（亏）。**哪个占上风取决于体系的 SCF
收敛尺度对 μ 的敏感度**：

- 212（小 PW、单自旋约束、outer 仅 3 步）：SCF 本身就快，μ 的扰动是主要成本 ⇒ 收紧
  后"少扰动"立刻变现（44 → 38，反而优于 OUTER 42）；
- MgO（LCAO 体相、单个大位移、outer 23 步）：每一步 μ 变更都要重走长 SCF ⇒ 拉长逼近
  段的代价占优（94 → 107 → 129，但相对 OUTER 仍是 −66%~−75%）；
- 213（混合双约束）：两种效应抵消（63/62/64）。

**结论（决策用）**：默认**保持 1e-3**；`constraint_inner_thr` 定位为**逐体系旋钮**，
"INNER 打不过 OUTER 时先试收紧到 1e-4" 是可用启发式，但必须实测确认（MgO 上会变差）。
这条应回填用户手册 §5.9 的决策表。

**局限**：每点 1 个 run（无重复）；MgO 补充点无 repeat；且 212/213 是 20 Ry 小 PW 体系，
"体系相关"结论的普适性应以更多体系（如 FeO，待暂缓解除）复核。

## 5. 下一步

1. 回填文档：用户手册 §5.9 决策表 + §5 审计行说明补一行"门控效应符号体系相关"；
   开发者文档 §3.4 的实测成本画像补 MgO/212 的三档数字（可并入下次 docs/test commit）。
2. 已登记的测试卫生项：`MODULE_IO_input_test_para{,_4}` / `read_item_serial` 的陈旧
   `sc_scf_thr_mode` 期望值（评审 §三）。
3. 可选（低成本）：把门控扫描并入 `tests/deltap_mgo_scan`/`run_inner_thr_scan.sh` 的
   常规入口，作为"新体系接入 INNER 前的一次性标定"动作。
4. FeO 双稳体系的 INNER/OUTER 对照仍按用户令暂缓。
