# DeltaP Stage 5：文档与清理（dev-guide v3 + INPUT 手册 + #if 0 清理）

- 日期：2026-08-17
- 前置：Stage 4 全部完成（4.1–4.4，`95639fcb0` 之前 4 个 commit）；Stage 5
  为收尾轮：文档对齐 + 代码清理。
- 范围：① `#if 0` 调试块清理（fsdbg/hkdbg/hkstr/T0）并回归；② dev-guide
  v3（记账推导 + dspin 恒等式 + Route A+ 双路径分层 + Γ-path relax 用法）；
  ③ INPUT 用户手册补 12 个缺失关键词 + Route A+ 快速入门/场景；④ dev log +
  总览文档刷新。

## 1. 测试计划

1. **清理回归**：删除 `FORCE_STRESS.cpp`（3 块 fsdbg）与 `deltap_wannier.cpp`
   （12 块 T0/hkdbg/hkstr/hkforce，含 kern_zz_u/dw、U_phase 等）的全部
   `#if 0` 调试块；增量构建 + deltap 单测；center 目标驱动用例（bn center）
   与锚点 #3 逐位一致（纯清理无行为变化的硬约束）。
2. **文档验收**：dev-guide v3 覆盖三种记账（proxy/hk/ow/PW + 非正交全迹）、
   dspin 恒等式落点、T3 PASS/T3' FAIL 双路径并存表述、Γ-path relax 生产
   用法；INPUT 手册含全部 12 个缺失关键词（deltap_observable/drive/
   operator_mode/secant/proxy_target_file/outer_nmax/outer_thr/
   branch_anchor/branch_write/inner_scheme/lambda_init_file/dk_fd）且语义与
   `deltap_scf.h` DeltapParams 注释一致。

## 2. 测试设置

- 系统：同仓库单机；OMP_NUM_THREADS=1；一次一个任务。
- 构建：`build/abacus_basic_para` 增量构建；`ctest -R deltap`。
- 回归：`/tmp/deltap_43/regr_center`（bn center 目标驱动，锚点 #3 口径）。

## 3. 结果

### 3.1 #if 0 清理（提交 `95639fcb0`）

- `FORCE_STRESS.cpp`：−26 行（fsdbg-deltap / fsdbg f_hk / fsdbg-preprint）。
- `deltap_wannier.cpp`：−223 行（T0 per-k/band 分解、hkdbg、hkstr、hkforce、
  kern_zz_u/dw、U_phase 等，全部引用在被删块内）。
- 合计 249 行纯删除，0 新增；增量构建通过；`ctest -R deltap` 5/5 PASS。
- **逐位一致回归**：bn center E' = −338.7136166249902 eV、λ 与锚点 #3
  全同 → 纯清理无行为变化。

### 3.2 dev-guide v3（`2026-08-02-deltap-force-stress-dev-guide.md`）

- Status 矩阵刷新：v2 的"未实现/必崩"全部关闭；新增 ow 实验性与场模式力
  侧已知缺口。
- §2.2 三种记账推导：proxy（τ_α·P̂）/hk（H_HK only，F-2b）/ow（θ_n·P̂）/
  PW（P̂^onsite）；非正交全迹教训（T·Π 全迹 vs 对角迹，18% 差，T2 判决）。
- §2.3 dspin 恒等式三条件逐条核对（v2 的"①②不满足"被 Route A+ 推翻）。
- §2.5 双路径分层：**T3 PASS=数学自洽 / T3' FAIL=物理不可用**并存表述，
  防"proxy 驱动 relax 可用"误读。
- §4 FD 协议更新为 stationary4（t_Γ* 冻结 → disp± 重收敛 λ → 中心差）；
  §7 公式集 F1–F12；§8 check-list 对齐当前验收面；§10 Γ-path relax 生产
  用法（4.3 实测参数与禁止项）。

### 3.3 INPUT 用户手册（`docs/deltap_user_manual.md`）

- 头部双路径说明（operator/gamma 两变量）；新增 §1.3 Route A+ 快速入门
  （约束 + 场模式两用法）。
- 12 个缺失关键词全部补入：`deltap_dk_fd`（§2.2）、`deltap_lambda_init_file`
  （§2.3）、其余 10 个（§2.8 Route A+ 控制表），语义对齐 `deltap_scf.h`
  DeltapParams 注释。
- 新增 §7.5 Γ-path relax 生产用法（INPUT 模板 + 机制 + 禁止项）。

## 4. 分析

- 清理是纯删除（0 新增），逐位一致回归证明无行为变化；12 个关键词的
  语义直接取自 INPUT 回显实现（`read_input_item_other.cpp`）与 DeltapParams
  注释，手册与实现零漂移。
- dev-guide v3 是 v2 之后两周工程结果的压缩沉淀：三种记账、恒等式、
  双路径分层、Γ-path 用法四个知识点都是后来者最容易误读/误用的地方；
  T3/T3' 的并存表述（数学自洽 vs 物理不可用）是本轮防误读的核心。
- E-field 语义更新项：主算法文档 §8 已被能力边界文档取代（既有评审
  记录）；公式 (b) + ×1.6 校准注释已在 `deltap_scf.cpp:873` 打印，手册
  §八 已含——本轮无新增代码动作。

## 5. 下一步

- TODO Stage 5 ✅；dev log 追加本轮记录；总览文档（progress-and-plan）
  刷新；提交文档轮。
- 后续：P 系列 P01–P18 入 CI（遗留小项）；Stage 3 hk MPI 生产面已闭，
  L2 应力已闭；约束模式封口、场模式力侧挂 EFC（能力边界文档 §6）。
