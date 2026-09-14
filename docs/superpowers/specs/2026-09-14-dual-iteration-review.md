# 双迭代策略（INNER 调度 + Q1–Q3 定量对照）严格评审

> 评审对象：`2026-09-14-dual-iteration-inner-schedule.md` + 未提交改动（12 文件 + 2 新路径）。
> 本轮无代码改动。证据核实方式：ctest 亲测 + 证据文件逐项比对 + 内部一致性算术检验（未独立复跑算例）。

## 裁定：✅ 通过——实现正确、定量结论坚实、两个用户关切的机制问题都有了直接证据。批准提交（建议 2 个 commit），下一优先级 C-29 定位。

## 一、独立核实

| 项 | 核实结果 |
|---|---|
| 单测 | `ctest -R MODULE_ESTATE_constraint` 亲测 **11/11**（loop 19→25，6 新测试在案） |
| OUTER 零回归 | 对照二进制方法正确；211 前后 −441.9708337535537 eV 逐位一致 + 7 条审计行迭代号全等——`print_audit_line` 重构未改输出 |
| 证据链算术一致性 | summary.txt 与 audit 文件交叉一致：211 inner（28 步/26 复位/2 反弹/降级 1）、213 noreset（RUNNING、res=0.055、scf_nmax 用满）；**"内步−复位=CONVERGED 步数"契约在四组数据全部成立**（28−26=2↔settle_fail=2；27−26=1↔pass；31−30=1↔pass） |
| 复位代价 | audit 日志实证 `mixing recovered after 2 SCF iteration(s)`——复位成本 K=1–2 迭代，廉价 |
| C-30 | settle 反弹撤回判决但不撤回 status_ 的缺陷真实存在过（InnerSettleCheck 红灯暴露），修复正确（回 RUNNING）——TDD 价值的又一实证 |

## 二、两个用户关切问题的证据评价（本轮最有价值的产出）

1. **"INNER 有助于 SCF 难收敛"——部分证实，边界清晰**：MgO −75%（381→94）、213 混合 −46%（117→63），但 211/212 反亏（+14%/+4.8%）。收益=省掉的"每外步一次 SCF 重收敛"，代价=每次内更新的复位重启——**交叉点在外步 ≈10** 的机制解释与数据自洽。决策表（≲7 用 outer、≳15 用 inner）有数据支撑，入册正确。
2. **"charge–λ 耦合破坏 DIIS"——证实为真，且对策有效**：不复位对照（ABA_CONSTRAINT_INNER_NO_RESET）下 211 慢 2.2×、213 用满 scf_nmax 不收敛——mixing 历史腐败是真实机制，`mix_reset` 是刚需；且复位代价仅 1–2 迭代。这是"担心点"能得到的最好结局：**机制真实存在，但廉价对策已内建**。
3. **Settle check 的实战价值**：抓到 3 次"内环宣告收敛、密度一松弛靶点即破"——若无 settle，这 3 次都会误报 CONVERGED。历史陷阱（2026-07-20）的硬设计对策被证明不是过度设计。

## 三、提交批复

建议 2 个 commit（内容与历史惯例一致）：
1. `feat(constraint): dual mu-schedule (outer/inner) with settle check and mix-reset`——代码 + 单测 + INPUT（12 个改动文件中除文档外部分）；
2. `test(constraint): dual-iteration Q1-Q3 comparison + decision table + C-30 fix docs`——证据目录（tests/deltap_dual_iteration/）、spec、手册/开发者文档更新、dev-log。

提交前把 `tests/deltap_dual_iteration/` 里的运行产物（如有 OUT.*/log 大件）清理或裁剪——保留 README/results/*.audit/summary.txt 作为证据，别把 SCF 输出全量入库。

## 四、后续优先级（同意并排序）

1. **C-29 定位（ASAN/gdb，最高优先）**——现已同时阻塞 MgO 长跑收尾与远侧扫描；注意新事实：崩溃发生在 !FINAL_ETOT_IS 打印**之后**（收尾段），物理数据在崩溃前完整——定位范围可收窄到"能量决算后的清理路径"。
2. inner_thr 与 scf_thr 关系标定（1e-3/1e-4/1e-5 三档扫 212/213，低成本中价值）。
3. FeO 对照继续暂缓（同意：II-1b 锚点未收口 + S4/S5 已令缓）。
4. OMP 可复现性纪律入 SOP：对照/验收跑一律 `OMP_NUM_THREADS=1`（autotest 阈值 1e-7 eV 与该噪声同量级——这是阈值边缘性的真实来源，值得在 dev-guide 写明）。

---

## 本轮记录

- 评审轮，无代码改动。证据链内部一致性（契约算术）全部成立；未独立复跑算例（证据密度足够）。
