# `constraint_inner_thr` 三档标定（用户优先级 2）

计划：`docs/superpowers/plans/2026-09-11-dual-iteration-strategy.md`；
调度落地：`tests/deltap_dual_iteration/`；本轮 spec：
`docs/superpowers/specs/2026-09-14-inner-thr-calibration.md`。

**问题**：INNER 调度的密度门控 `constraint_inner_thr` 默认 1e-3 是沿用 DeltaSpin
口径的**未标定值**。收紧/放宽它，成本与答案各是什么？该不该改默认？

**口径**：同二进制（`build_rel/abacus_basic_para`，Release，当前 HEAD）、同网格、
同靶点、同初猜，只有 `constraint_inner_thr` 变（1e-3 / 1e-4 / 1e-5）；
`np=4`，`OMP_NUM_THREADS=1`。正确性判据沿用 Q1–Q3：μ* 相对差 < 1%、
`E_tot` 绝对差 < 1e-6 eV。

## 数据（`results/`）

- `2{12,13}_thr1e{3,4,5}.audit`：逐 run 的约束审计行节选（含 `MIX_RESET`/
  reset cost/settle/`FINAL_ETOT_IS`）；
- `summary.txt`：主表 + OUTER 基线对比 + 结论（由 `tools/summarize.py` 生成）；
- `mgo_crosscheck.txt`：**补充**对照——MgO 体相/LCAO 的 1e-4、1e-5 两点
  （1e-3 取自已入库的 `tests/deltap_dual_iteration`）。

## 结论摘要

| 体系 | OUTER | INNER 1e-3 | INNER 1e-4 | INNER 1e-5 |
|---|---|---|---|---|
| 212 PW 自旋 | 42 | 44 (+4.8%) | **38 (−9.5%)** | 39 (−7.1%) |
| 213 PW 混合 | 117 | 63 (−46.2%) | 62 (−47.0%) | 64 (−45.3%) |
| MgO 体相电荷 | 381 | **94 (−75%)** | 107 (−72%) | 129 (−66%) |

（括号为相对 OUTER 的 SCF 迭代数变化）

1. **正确性：门控是纯成本旋钮**。三体系、三个门控值下 μ* 相对散布 ≤ 0.34%、
   `|ΔE_tot|` ≤ 4.2e-7 eV——全部落在等价判据内，**没有改默认值的正确性理由**。
2. **成本：门控效应符号随体系翻转**。收紧门控永远 = 更少的内环 μ 更新
   （212 27/11/10、213 28/25/24、MgO 31/22/22），但"省下的 mixing churn" vs
   "被拉长的逼近段"孰大孰小是**体系相关**的：212 变好、213 基本平、MgO 变差。
3. **建议**：**默认保持 1e-3**；把 `constraint_inner_thr` 当**逐体系旋钮**——
   INNER 若打不过 OUTER，可做 2–3 点扫描，但可能救回来（212）也可能更差（MgO），
   不可盲调。
4. settle 检查在每个门控值下都还在触发（每点 1–2 次反弹，MgO 0 次），
   说明它**不是**"门控太松"的产物，应无条件保留。

## 复现

```
bash run_inner_thr_scan.sh 4                     # 212/213 × 1e-3/1e-4/1e-5
THRS="1e-4 1e-5" bash run_inner_thr_scan.sh 4    # 单点补跑
python3 tools/summarize.py /tmp/inner_thr_scan results/summary.txt
```

MgO 补充点需按 `tests/deltap_mgo_scan` 的 S3 δ=+0.5 输入追加
`constraint_mu_schedule inner` + `constraint_inner_thr <值>` +
`constraint_inner_nmax 200`，热启动源 `read_file_dir` 指向 160/640 的旧扫描
（**C-29 已修**，跨基组重启不再破坏堆）。

## ⚠️ 与 C-29 的关系

MgO 补充点的热启动正是历史上触发 C-29 的跨基组模式；本轮用含 `read_rhog`
守卫的 Release 二进制跑，`rc=0`、0 条堆破坏、`final status: CONVERGED`
——顺带第二次在 Release 上确认 C-29 修复有效。
