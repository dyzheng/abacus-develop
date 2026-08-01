# 2026-08-01 DeltaP 提交评审与 TODO 分级

> 评审对象：
> - `b825fed52` refactor(deltap): unify LCAO/PW SCF into DeltapScfSolver state machine (R1-R4)
> - `139380f64` test(deltap): add mask/gdir2 cases, relax reproducer, fix force-path null deref (R5)
> 评审方法：逐文件 diff + 与重构前基线（`6beb70bc3`/`aeb9a0f9c`）行为比对 + MPI/边界路径推演。

## 1. 总体结论

重构方向正确、实现质量良好，**无 P0 级问题**：

- 状态机收敛成功：LCAO/PW 共用同一 `DeltapScfSolver`（`deltap_scf.{h,cpp}`），
  `deltap_common.h` 纯函数化且 10 例单测覆盖；esolver 净删 ~700 行、deltap_pw 净删 ~150 行。
- 验证充分：R1–R4 三路 A/B 冒烟逐字节一致；R5 复验 test_C_I GE14–16 与旧日志逐字节一致；
  单测 16/16；MPI LCAO 4-rank / PW 2-rank 实跑通过。
- 头文件/CMake 改动干净（`unique_ptr` 替换 `void*`、公开面收敛、测试目标挂载正确）。
- 主要遗留集中在 **MPI 一致性（LCAO λ 写回）**、**PW 多离子步初始化频率**、
  **force 路径（既有崩溃）** 三类，均已分级列入 §3。

## 2. 逐项评审意见

### 2.1 状态机核心（deltap_scf.cpp / deltap_scf.h）— 质量好，3 处需跟进

1. **`init` rank0+Bcast（C-23）正确**：target/约束矩阵均 rank0 解析后 Bcast，
   非 MPI 构建走纯 rank0 路径；矩阵尺寸不匹配 `WARNING_QUIT`。
2. **`reset_ionic_step` 语义正确**：清 `lambda_set`/`inner_loop_done`/`gamma_prev`/`gamma_report`，
   保留 `lambda_cstr`（约束乘子跨离子步延续，合理）。
3. **内循环（nscf>0）**：drho gate + `inner_loop_done` 单次收敛，与旧 `inner_loop_active()`
   （`nscf_ > 0`）语义一致；`apply_hk_correction`/`solve_frozen` 可选回调防御良好。
4. **`state_.lambda_eff` 死字段（P2）**：`iter_finish`/`update_lambda_gd` 从不写入；
   若未来 backend 缺 `get_lambda`，escon 会静默为 0。建议删除或同步维护。
5. **total 模式 `report` 的 `lambda[0]`（P3 防御）**：`lambda` 为空 vector 时 UB（当前
   LCAO/PW backend 均提供 `get_lambda`，不会触发）。
6. **target 文件解析不校验 EOF（P3）**：行数不足时静默补 0（历史一致但易错）。

### 2.2 LCAO 接线（esolver_ks_lcao.cpp / .h）— 干净，1 处 MPI 遗留

1. **`deltap_make_backend` 职责清晰**：set/get_lambda、compute_gamma(_raw)、apply_hk、
   solve_frozen、sync_lambda、on_phase2、get_optimizer、lattice_period 全部注入；
   real 实例用 `if constexpr` 丢弃 gamma/HK 分支。
2. **`sync_lambda` Bcast 后未写回 operator（P1，最高优先级）**：
   ```cpp
   backend_.set_lambda(lambda);        // 用本地（未 Bcast）λ
   backend_.sync_lambda(lambda);       // Bcast 只改局部副本，非 rank0 的 operator 仍是本地值
   ```
   PW 侧 R4 已修（`s_lambda = lam` 写回），LCAO 侧沿袭旧代码（`6beb70bc3` 同样先 set 后
   Bcast 不写回）——非回归，但多 rank 下非 rank0 的 `dp_op` λ 与 escon 可能与 rank0 不一致
   （4-rank 冒烟未暴露，因 λ 量级小、escon 未跨 rank 校对）。对齐 PW 模式即可。
3. **`iter_finish` 每迭代 `apply_hk_correction` + `reset_ionic_step`（iter==1）**：与旧行为一致。
4. **头文件**：`void*` → `unique_ptr` + 前向声明，模板实例化无泄漏。

### 2.3 PW 接线（deltap_pw.cpp / esolver_ks_pw.cpp / deltap_pw.h）— 收敛良好，2 处跟进

1. **`sync_lambda` 写回 `s_lambda`（R4 修复）正确**；`[DeltaP-PW]`/`report_pw` rank0 守卫正确。
2. **`deltap_init` 在 `before_all_runners` 每离子步重复执行（P1）**：`g_solver.init` 重置
   `state_` + `s_lambda` 回到 `lambda_init`，而 LCAO 是惰性 init + `reset_ionic_step`
   （λ 跨离子步延续）。当前 PW 只支持 scf（relax 在 force 路径崩），多离子步语义差异暂不暴露；
   建议 PW 对齐 LCAO（惰性 init / 每离子步仅 reset）。
3. **γ 测量跨 rank 一致性未核对（P1）**：`compute_per_atom_gamma_kstring` 各 rank 本地计算，
   2-rank 冒烟跑通但未验证各 rank `gamma_report` 一致 → escon 跨 rank 一致性需在 2-rank
   下加断言核对（或对 gamma 做 Allreduce，与旧 `750ee179d` 的 gamma 同步对齐）。
4. **注释与实现不符（P3）**：`deltap_pw.h` 写 "Called once from before_all_runners"，
   实际每离子步调用。
5. **`deltap_iter_finish` 的 `iter=0`**：PW 自打印 `[DeltaP-PW]`（verbose=false），无影响。

### 2.4 deltap_common.h + 单测 — 质量好

- 纯函数化后 10 例单测覆盖 residual/mask/mixing/total/effective/escon/unwrap，
  掩码语义（`constrain[i]==0` 自由）与 STRU `dp_constrain` 默认 1 一致，端到端 test_mask 佐证。

### 2.5 R5（139380f64）— 算例/文档/修复

- **空指针守卫修复正确**：force 路径 `hR=nullptr` 不再崩；`this->paraV` 仅 `cal_pre_HR`
  （SCF 路径）使用，force 路径取 `dmR->get_paraV()`，守卫安全。
- **force 路径双释放（P1，既有）**：`cal_force_stress` OMP 区 heap corruption，
  根因（混合基组 nlm 越界读）+ 背靠栈已记录于 `tests/deltap_relax/README.md`；
  修复需统一 nlm 布局 + 长度守卫 + 补 H_HK/∂τ/∂R 力项，属 C-02 专项。
- 新算例（test_mask/test_gdir2）断言清晰、README 完整；测试日志 `*.log` gitignore 不入库合理。

## 3. P0–P3 分级 TODO 表

| 级别 | ID | 事项 | 位置 | 说明/验收 |
|---|---|---|---|---|
| P0 | — | 无 | — | 评审未发现必须立即修复的正确性/崩溃问题 |
| P1 | T1 | LCAO `sync_lambda` Bcast 后写回 operator λ（对齐 PW） | `esolver_ks_lcao.cpp` backend | 多 rank 非 rank0 `dp_op` λ/escon 一致；验收：2/4-rank 运行后各 rank `get_lambda` 一致 |
| P1 | T2 | PW `deltap_init` 改惰性 init + 每离子步 `reset_ionic_step`（对齐 LCAO） | `esolver_ks_pw.cpp` / `deltap_pw.cpp` | 多离子步 λ 延续语义；验收：relax（修复 T7 后）两步 λ 不归零 |
| P1 | T3 | PW γ 测量跨 rank 一致性核对/同步 | `deltap_pw.cpp` | 2-rank 下断言各 rank `gamma_report`/escon 一致；必要时 Allreduce |
| P1 | T7 | 修 `cal_force_stress` 双释放 + nlm 越界读 | `deltap_force_stress.hpp` | 复跑 `tests/deltap_relax` 不崩；FD 力按 `run_fd.sh` 判据验收（需先补 H_HK/∂τ/∂R 力项） |
| P2 | T4 | 删/维护 `DeltapState::lambda_eff` 死字段 | `deltap_scf.h/cpp` | backend 无 `get_lambda` 时 escon 不再静默 0 |
| P2 | T5 | PW 无 `on_phase2`（P2 后无 mix_reset/cooldown） | `deltap_pw.cpp` | 与 LCAO 行为统一评估；PW 收敛性改善时验证 |
| P2 | T6 | target 文件解析 EOF 校验 + WARNING | `deltap_scf.cpp` init | 行数不足不再静默补 0 |
| P2 | T8 | PW 接 `deltap_target_file`/`deltap_constraint_matrix`；PW total 模式统一 | `deltap_pw.cpp` | F13/F14；与 LCAO 状态机参数对齐 |
| P3 | T9 | 修正 `deltap_pw.h` "called once" 注释 | `deltap_pw.h` | 与实现一致 |
| P3 | T10 | total 模式 `report` 空 `lambda` 防御 | `deltap_scf.cpp` | `if (lambda.empty())` 提前返回 |
| P3 | T11 | `update_lambda_gd` 空 `get_lambda` fallback 防御 | `deltap_scf.cpp` | 避免空 vector `set_lambda` |
| P3 | T12 | 内层短 if 补独立注释（`bfgs_converged`/`apply_hk_correction`） | `deltap_scf.cpp` | 满足 AGENTS.md 注释规则 |
| P3 | T13 | BN 用例 SCF 不收敛（center/test_stru_target/mask）补"已知行为"断言或调参；P01–P18 用例入 CI | `tests/deltap_*` | 明确振荡为既有行为，防未来误判回归 |

## 4. 验证状态确认（评审复核）

- 与重构前行为一致性：test_C_I（约束矩阵、λ 恒 0）GE14–16 逐字节一致 → SCF 物理未变；
  P1 轨迹差异可归因于旧日志 INPUT 参数/分支文件漂移（R5 文档 §4.1 已记录）。
- total 模式：λ 更新数学与 `6beb70bc3` 逐行一致；目标均分语义由 `8e73e0f82` 引入（早于 R1–R4）。
- MPI：C-23（rank0+Bcast）、C-28（PW λ 同步/rank 守卫）实跑生效；LCAO 2-rank 方阵限制
  （`cbb37b7ae`）为既有约束。
