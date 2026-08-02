# 2026-08-02 DeltaP MPI 一致性 commit 评审（d983206a1）

> 评审对象：`d983206a1` fix(deltap): sync LCAO lambda write-back and PW gamma
> across MPI ranks (T1,T3)。方法：逐行 diff + 调用链推演（sync/set_lambda 全
> 调用点、通信域、KPAR/psi 布局）+ 与上一轮验证记录交叉核对。

## 1. 总体结论

T1/T3 方向正确、实现最小且无回归：PW 1-rank 逐字节一致、单测 16/16、跨 rank 一致
性经临时诊断实跑证明（PW 2-rank escon/γ、LCAO 4-rank λ）。可保留，但存在 5 条
批评（均非阻塞）与 1 条 latent 清理项，见 §3 D1–D6。

## 2. 逐项评审意见

### 正面项

1. **T1 写回位置与幂等语义正确**：`sync_lambda` 在 `update_lambda_gd` 中紧跟
   `set_lambda` 调用（`deltap_scf.cpp:346-348`），写回 `dp_op->set_lambda(lam)`
   仅复位 `dp_hr_done`（值不变），对 rank0/单 rank 为无副作用 no-op；非 rank0 的
   operator λ 与 escon 因此与 rank0 一致，语义与 PW 侧 `s_lambda = lam` 对齐。
2. **rank0-wins 事实源契约统一**：T3 与历史 gamma 同步（`750ee179d`）同一策略；
   LCAO γ 本就在 `compute_gamma_scf`（`deltap_wannier.cpp:1624-1633`）按
   `paraV_->comm()` rank0 Bcast，T3 补齐 PW 侧后两端事实源一致。
3. **验证方法学扎实**：临时诊断（RANKDBG/LAMDBG）→ 实跑证明 → 移除，最终二进制
   无残留输出；带诊断/无诊断复跑逐字节一致（确定性）；stash 对照实验正确归因
   h2o_lcao 4-rank 崩溃为既有问题，未误判为回归。
4. **回归面控制严格**：PW 1-rank 与 R5 基线逐字节一致（T3 对 NPROC=1 no-op）、
   LCAO P2 iter=11 drho=6.92e-06 锚点保持、单测 16/16。
5. **文档规范**：dated 文档五段式齐全、dev-log 同步、新增注释符合 AGENTS.md
   分支注释规则。

### 批评（五条）

1. **C1（通信域与布局约束未显式化）**：T3 用 `GlobalV::NPROC > 1` +
   `Parallel_Common::bcast_double`（`MPI_COMM_WORLD`），LCAO 用 `pv.comm()`
   （POOL）——两种约定并存。且 `unkdotp_G`/`unkdotp_G0` 注释仍声称
   "must make GlobalV::KPAR = 1"，而本轮 2-rank 实跑（h2o_pw：2 k 点 2 rank，
   KPAR=2）γ 跨 rank 一致、γ_total 与 1-rank 逐位一致——说明 psi 在 deltap
   测量路径上每 rank 有效完整，KPAR=1 约束已陈旧。约束与实现的错位若不固化，
   未来 KPAR>1 多 pool 配置下 world-Bcast 可能用 pool0 的 γ 覆盖其他 pool 的
   合法测量。
2. **C2（λ 同步覆盖不完整）**：`sync_lambda` 只在同步模式
   （`update_lambda_gd`）触发；内循环路径 `inner_loop` 的
   `backend_.set_lambda`（`deltap_scf.cpp:210,235`）不走同步。多 rank 下若各
   rank 的 trial γ 有浮点差异，BFGS 轨迹可跨 rank 漂移，operator λ 失去一致
   性保证。LCAO 内循环（nscf>0）从未在 2/4-rank 验证过（h2o_inner 仅 1-rank）。
3. **C3（LCAO 侧验证不闭环）**：LAMDBG 只验证了 λ，未验证 escon/γ_report
   的 per-rank 一致性；LCAO γ 一致依赖 `compute_gamma_scf` 内部 Bcast，属
   推理保证而非实跑断言。PW 侧 RANKDBG 验证了 escon+γ 双项，LCAO 侧缺同等
   断言。
4. **C4（2-rank PW 数值基线未固化）**：1/2-rank 的 rank0 γ 差 ~0.02%
   （4.738603 vs 4.737743）、escon 差 7e-6 Ry；上轮文档记为"既有 nproc 敏感
   性"但未追根因（becp/重叠并行求和顺序）也未立项；2-rank 参考输出仅存
   `/tmp/t1t3_pw2.log`，未入仓库/文档表格，无回归锚点。
5. **C5（同步正确性无回归保护）**：临时诊断已删除，无持久化的跨 rank 一致性
   校验（如 verbose/debug 门控的 max-dev Allreduce 断言）；2-rank PW / 4-rank
   LCAO 冒烟未纳入 CI 矩阵，未来改动可悄悄破坏一致性而不被捕获。

### latent 清理项

- `pelec->f_en.dp_escon` 在 `esolver_ks_lcao.cpp:722` 与 `esolver_ks_pw.cpp:291`
  无 rank 守卫地写本地 state 值——T1/T3 后跨 rank 一致故当前安全，但依赖隐式
  同步契约，建议加注释/守卫。
- LCAO `sync_lambda` 对空 `lam`（`MPI_Bcast` count=0）与 PW 侧空 vector 防御
  （评审 T10/T11）同族，可一并统一。

## 3. D1–D6 修订 TODO

| 级别 | ID | 事项 | 位置 | 验收 |
|---|---|---|---|---|
| P1 | D1 | 统一 PW/LCAO 同步通信域（POOL）+ 显式化 KPAR/psi 布局约束（修正 `unkdotp` 陈旧注释或加守卫） | `deltap_pw.cpp` make_backend / `unk_overlap_pw.cpp` | 2-rank、4-rank 实跑不变；文档化 KPAR>1 行为 |
| P1 | D2 | 内循环路径 λ 同步下沉：`inner_loop` 的 `set_lambda` 后接 `sync_lambda`（或统一出口包装） | `deltap_scf.cpp` inner_loop | LCAO 内循环 2/4-rank 实跑各 rank `get_lambda` 一致 |
| P2 | D3 | LCAO 4-rank 补 escon/γ_report per-rank 断言（闭环 T1 验收，与 PW RANKDBG 同等） | 测试/临时诊断 | 4-rank 下各 rank escon、γ_report 一致 |
| P2 | D4 | 固化 2-rank PW 参考值（入库/文档表格）+ nproc 敏感性根因专项（becp/重叠求和顺序） | 文档/测试 + `deltap_pw.cpp` | 参考值可复现；根因有结论或明确接受 |
| P2 | D5 | 跨 rank 一致性校验沉淀为可复用诊断（verbose 门控 max-dev Allreduce 断言）；2-rank PW / 4-rank LCAO 冒烟入 CI | `deltap_scf.cpp` / CI | CI 矩阵含 MPI 冒烟；破坏一致性即失败 |
| P3 | D6 | `f_en.dp_escon` rank 赋值守卫/注释 + LCAO/PW 空 λ vector 防御统一 | `esolver_ks_lcao.cpp` / `esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp` sync_lambda | 无守卫依赖隐式契约；空 vector 安全 |

## 4. 结论与建议

- **D1/D2 先行**（功能正确性）：D1 是通信域契约，D2 补内循环同步，均可在下一次
  MPI 实跑中验证。
- **D3 随 D1/D2 一起做**（验收闭环，一次 MPI 迭代内完成三件事）。
- D4/D5 为基线与 CI 加固，随测试轮推进；D6 为清理项。
- 保持"临时诊断 → 实跑证明 → 移除"的验证方法学；建议把该流程固化为文档中的
  标准 MPI 验收清单（本文件 §2 正面项 3 可作为模板）。
