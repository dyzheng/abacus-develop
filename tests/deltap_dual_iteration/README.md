# 双迭代调度对照研究（OUTER vs INNER μ 更新）

计划：`docs/superpowers/plans/2026-09-11-dual-iteration-strategy.md`（§3）。
落地 spec：`docs/superpowers/specs/2026-09-14-dual-iteration-inner-schedule.md`。

**问题**：把 μ 更新放进 SCF 迭代内（`constraint_mu_schedule=inner`，
DeltaSpin `lambda_loop` 血统 + `mix_reset` + settle 检查）相对现状
（`outer`，SCF 完整收敛后一次 M4 秒差步）在**成本**与**正确性**上如何取舍？
两个调度都用同一二进制、同一网格、同一靶点、同一初猜，逐 run 落盘。

## 度量

- **成本主指标 = SCF 迭代数**（PW/LCAO 每迭代一次对角化，等价于"总对角化次数"）；
- **正确性等价判据**：μ* 相对差 < 1%、`E_tot` 绝对差 < 1e-6 eV；
- **机制读数**：`MIX_RESET` 次数与"复位代价"（`mixing recovered after K SCF
  iteration(s)`）、settle 通过/反弹次数、降级事件。

## 数据（2026-09-14，`np=4`，`OMP_NUM_THREADS=1`，Release `build_rel/abacus_basic_para`）

见 `results/summary.txt`（逐 run 明细）与 `results/*.audit`（日志节选）。

## 结论摘要

1. **正确性等价**：4 个体系（H₂O 电荷/自旋/混合 + MgO 体相电荷）上
   INNER 与 OUTER 的 μ* 相对差 ≤ 0.17%、`E_tot` 差 ≤ 8.5e-7 eV——**全部命中**判据。
2. **成本交叉点**：OUTER 外步数 ≲7 时 INNER 略差（+5…+14%）；外步数 ≳15 时
   INNER 显著更优（−46% / −75%）。即 **INNER 的收益 ≈ 省掉的"SCF 重收敛"次数**。
3. **mixing 复位是刚需**（Q3 对照）：关掉 `mix_reset` 后 211 迭代 72→161、
   213 直接用满 `scf_nmax` 且未收敛、MgO 145 次内更新后 settle 连败两次降级。
   ⇒ mixing 历史腐败是真机制，"内环 λ 破坏 DIIS 前提"的担心成立。
4. **settle 检查不是摆设**：全campaign 抓到 3 次"内环宣告 CONVERGED 但密度
   一松弛靶点就破"（211 两次、213 一次），否则会误报 CONVERGED。

## 复现

- H₂O 三用例：把 `tests/01_PW/{211,212,213}_PW_constraint_*` 的 INPUT 复制一份，
  追加 `constraint_mu_schedule inner`（`constraint_inner_thr 1e-3`）跑即可；
- MgO：`tests/deltap_mgo_scan` 的 S3 δ=+0.5 点，按同一 INPUT 追加 inner 三行；
- 复位对照：设 `ABA_CONSTRAINT_INNER_NO_RESET=1`（诊断开关，默认关）。

## ⚠️ 已知阻塞：C-29（预存在）

MgO δ=+0.5 的 OUTER **与** INNER 运行都在写完 `!FINAL_ETOT_IS` 之后于收尾阶段
堆破坏（`free(): invalid next size` / `corrupted size vs. prev_size`，四 rank 同中）。
用**父提交二进制**跑同一输入逐位复现 ⇒ 与双迭代改动无关；C-29 的已知范围应扩展到
"MgO 电荷约束 LCAO 运行的收敛后收尾路径"（不只是非收敛路径）。
物理量（μ*、`E_tot`、审计行）在崩溃前已完整打印，故上表可用：

- OUTER `-7659.260116348592 eV` / μ*=−1.049024275 / 381 iters；
- INNER `-7659.26011634944 eV` / μ*=−1.049044143 / 94 iters。

**未修 C-29 前**：MgO 类长跑必须 `timeout` 包裹，并只信崩溃前已落盘的
`CONVERGED`/审计行。
