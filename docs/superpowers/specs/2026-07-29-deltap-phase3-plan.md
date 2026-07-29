# 2026-07-29 DeltaP 第三阶段开发修复计划

> 基于：`2026-07-29-deltap-validation-test-report.md`（50 项测试，45 通过，1 MPI 失败，4 预存）
> 前置：Phase 1 + Phase 2 共 17 commits 已推送到 dyzheng/abacus-develop feat/deltap

---

## 1. 测试报告中未通过的 5 项分析

### 1.1 MPI np=2 segfault（第 17/159/187 行）

**报告归因**：C-06（nrow≠ncol 越界）。**实际分析**：

- C-06 的 WARNING_QUIT 守卫只作用于 `compute_hk_correction`，该函数仅在 `deltap_corr=1` 时调用
- 报告的 Si nscf 测试使用 `deltap_switch=1` 但 `deltap_corr` 未设置（默认 0），所以 `compute_hk_correction` **不会被调用**
- 在 nscf 分解模式下（无约束），可能 crash 的其他位置：

| 候选 | 文件 | 问题 |
|------|------|------|
| berryphase_overlap ScaLAPACK | `unk_overlap_lcao.cpp:680+` | O_matrix 矩阵尺寸与 occ_bands 匹配？np=2 时 desc 可能不兼容 |
| MPI Allreduce on D_I | `deltap_wannier.cpp:436` | `paraV_->comm()` 在 nscf 独立进程中可能未正确初始化 |
| ctrl_scf_lcao 中的 unkOverlap_lcao 初始化 | `ctrl_scf_lcao.cpp:380+` | `cal_R_number`/`cal_orb_overlap` 在 MPI 下可能依赖全局状态 |

**下一步**：
1. 获取报告中的精确 crash 堆栈（gdb backtrace）
2. 如果是 paraV comm 问题 → 在 `ctrl_scf_lcao` 中创建 DeltaP 对象前检查 comm 有效性
3. 如果是 ScaLAPACK desc 问题 → 在 `berryphase_overlap` 中添加 desc 校验

**注意**：在当前环境（Si 2×2×2, np=2, OMP_NUM_THREADS=1）下**无法复现**，exit code = 0，结果文件正常产出。需在报告测试环境中复现。

### 1.2 BN 9 点 SCF 50 步未收敛（第 101-102 行）

**原因**：BN 的电子极化刚度接近零（Hessian ~ 1e3 μRy/rad²，见 07-22 文档），λ 驱动几乎不改变 γ，SCF 电荷密度反复振荡。

**修复方案**（2 选 1）：
- **方案 A**：增大 `scf_nmax` 到 100-150，配合更紧的 `scf_thr`
- **方案 B**：启用 `deltap_inner_nmax > 0`（内循环 BFGS 优化 λ），在冻结电荷密度下快速搜索 λ
- **推荐 B**：内循环专为此场景设计，且 BN 50 步未收敛恰好是该模式的典型用例

### 1.3 Smoothness 4/8 预存失败（第 120-138 行）

**状态**：与我的修改无关，原分支 HEAD `93a0d7825` 上同样 4 个失败。
**影响**：正常 SCF 流程不受影响；仅在极端微扰下规范固定鲁棒性有改进空间。
**处理**：本轮不修，在 `deltap-development-log.md` 中记录。

---

## 2. 第三阶段 TODO 清单

### TODO-1：MPI crash 诊断与修复（1-2 天）

```
1. 获取 crash stack trace（在报告测试环境中运行）
2. 根据 stack trace 定位 crash 位置
3. 可能修复点（按优先级）：
   a. 如果 crash 在 berryphase_overlap → 检查 desc 维度和 occ_bands 一致性
   b. 如果 crash 在 MPI_Allreduce → 检查 comm 有效性
   c. 如果 crash 在 cal_R_number → 加 comm 守卫
4. 用 Si 2×2×2 + BN 4×4×4 各跑一次 np=2 验证
```

### TODO-2：BN 收敛性改进（0.5 天）

```
1. 修改 INPUT：scf_nmax 100，增加 drho 输出
2. 如果仍不收敛 → 启用 deltap_inner_nmax=10
3. 检查 iter_finish 中 deltap_update_lambda 的 lambda 值是否稳定
4. 9 点全部收敛后，检查 |γ-t| 偏差是否 < 0.05 rad（C-11 验证）
```

### TODO-3：FD 力验证前置准备（0.5 天，无需集群）

```
1. 完善 tests/deltap_fd_force/h2o/ 的 STRU/KPT（目前只有 target.dat）
2. 添加 BN 体系的 FD 测试输入
3. 写验证脚本的预期输出对比逻辑（自动判断 pass/fail）
```

### TODO-4：文档更新（0.5 天）

```
1. 更新 deltap-development-log.md：
   - 记录验证报告中的 4 个预存 smoothness 失败
   - 记录 BN 收敛性问题和修复方案
   - 更新 test plan 状态
2. 更新 2026-07-29-deltap-phase2-validation.md：
   - 添加报告中的测试结果
   - 标记 MPI crash 为待修复
```

---

## 3. 执行顺序与依赖

```
Day 1 上午: TODO-2（BN 收敛性）          ← 独立，可在当前环境做
Day 1 下午: TODO-1（MPI crash 诊断）      ← 需要报告测试环境的 crash 信息
Day 2 上午: TODO-3（FD 准备）            ← 独立
Day 2 下午: TODO-4（文档）               ← 依赖 TODO-2 结果
```

---

## 4. 后续（集群级验证，本轮不执行）

| 项 | 内容 | 环境要求 |
|----|------|----------|
| FD 力验证 | `run_fd.sh h2o 0.005 2` | 12 次 SCF × 2 位移，需 MPI |
| C-11 回归 | BN 9 点 |t| 检查 | 收敛后的结果对比 |
| gdir=1/2/3 约束 | 三方向各一次 SCF | MPI np=2 |
| CI baseline | 重新生成 deltap_results.dat | 全量测试套件 |

---

## 5. 未修复的已知问题（维护清单更新）

| ID | 简要 | 状态变化 |
|----|------|----------|
| C-06 MPI crash | np=2 nscf segfault | **新增**（验证报告发现，此环境未复现） |
| BN 收敛 | 50 步 SCF 不收敛 | **新增**（需改参数） |
| Smoothness 4/8 | 预存测试失败 | 已记录，本轮不修 |
| C-02 ∂τ/∂R | 力缺项 | 不变（待 FD 验证后决定） |
| C-02 H_HK 力 | Berry 联络无力 | 不变 |
