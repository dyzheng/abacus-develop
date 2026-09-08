# 2026-09-07: Task 2.6.0 前置评审完成 + np=4 MPI 空目标段错误修复

> 上游：`docs/superpowers/plans/2026-08-31-task26-judgment-validations.md`
> Task 2.6.0（评审 c984b708e M3b 升格 + 低风险复核清零）。
> 归因发现 c984b708e 引入的真实 MPI 缺陷（np=4 LCAO 段错误），本次为
> 计划许可的唯一代码改动（bug 修复），随后进入 Task 2.6.1 判决。

## 1. Test plan

- 评审 c984b708e（M3b 升格运行时审计）四项：
  (a) paraV MPI 分布传递与 `transferSerials2Parallels` 正确性；
  (b) `trace()` 位一致布局配对与拒绝路径；
  (c) 运行时审计接入 LCAO esolver 且默认开销可忽略；
  (d) 升格后单测仍绿。
- 低风险未复核项清零：037_PW_FM 回归复跑、三道 sabotage 重做。
- MPI 深挖：LCAO constraint 集成用例在 np=1/2/4 下跑通，重点验证 np=4
  （计划 Task 2.6.4 的 LCAO 4-rank 一致性前置）。
- 修复后回归：constraint/partition ctest 全绿；np=1/2/4 集成数值一致。

## 2. Test setup

- 平台：容器 gcc debug 构建 `build/`（MPI/OpenMP ON），
  `abacus_basic_para`；ccache 只读失败 → `CCACHE_DISABLE=1` 重编。
- 集成用例：`tests/02_NAO_Gamma/212_NAO_constraint_h2o`（LCAO gamma，
  H₂O 15 Å 盒，becke 电荷约束 +0.1 e on O，ecutwfc 20/ecutrho 80，
  scf_thr 1e-7，scf_nmax 300）；`mpirun --allow-run-as-root -np N`。
- 对拍：FINAL_ETOT vs `result.ref`（etotref −466.253323360341）；
  审计行 `[constraint] M3b runtime audit` 数值跨 rank 比对。
- PW/自旋回归（前期本轮完成，记录备查）：037_PW_FM 5/5、
  211/212_PW_constraint_h2o(_spin) 4/4。

## 3. Results

### 3.1 评审结论（c984b708e，四项）

- (a) **发现真实 MPI 缺陷**：`build()` 的 MPI 目标容器
  `HContainer(paraV, nullptr, &gint_info->get_ijr_info())` 用 Gint
  网格导出的 IJR 列表逐 rank 过滤。当某 rank 的实空间网格子域不覆盖
  任何原子时，其 IJR 列表为空 → 目标 HContainer 空（nnr=0）→
  `cal_gint_vl` → `transferSerials2Parallels` → `HTransPara` 构造
  解引用 `get_atom_pair(0)` 越界 → 段错误。生产 DM/HR 布局由
  **网格无关的邻接表**（DensityMatrix::init_DMR / Veff::initialize_HR）
  按 rank 组装，永不为空，故生产路径不崩——缺陷只在审计路径暴露。
- (b) `trace()`：nnr + 完整 IJR 双重布局守卫后平面点积，拒绝路径返回
  false 且不改写 trace_out——评审确认正确。但守卫把"审计可跑性"绑在
  两容器的布局巧合上；修复后（见 §3.3）布局由构造保证一致，守卫成为
  双保险而非前提。
- (c) 接线：`ESolver_KS_LCAO::iter_finish` 在约束外环 DONE 时每几何
  一次，`constraint_audit_done_` before_scf 重置；charge→总 DM、
  spin→磁化 DM（nspin=2 switch_dmr(2)），结束 switch_dmr(0) 复原；
  无约束时 cloop.enabled()==false 直接跳过——默认开销为零。评审确认
  正确。
- (d) 单测：constraint/partition ctest 12/12 PASS（本次复核）。

### 3.2 低风险复核清零（本轮与前轮）

- 037_PW_FM：`Autotest.sh -a ... -n 4 -r 037_PW_FM` → 5/5 PASS。
- 三道 sabotage（评审者亲做，各恰中目标后恢复）：
  去 nspin 守卫 → SpinTypeGuard + ConfigureFromInputsShared 2 FAIL；
  翻转自旋注入符号 → SplitInjectionSpin FAIL；
  response_sign=+1 → SpinChannelConvergesOnLinearResponse FAIL。
  恢复后全部 PASS + constraint ctest 11/11。
- PW constraint 集成：211_PW_constraint_h2o + 212_PW_constraint_h2o_spin
  → 4/4 PASS。

### 3.3 np=4 崩溃复现与修复

- 复现：212_NAO_constraint_h2o `-np 4` 在 M3b audit 首次调用处
  exit=139（段错误），backtrace 指向
  `ConstraintInjectLCAO::build → cal_gint_vl → transferSerials2Parallels
  → HTransPara → 空 get_atom_pair(0)`；np=1/2 正常。
- 临时诊断：rank0 的 W nnr=0（IJR 过滤后无本地行），其它 rank 非空。
- 修复：`build()` 增加可选 `dm_layout` 参数；MPI 分支优先用
  **生产 DM 容器的逐位布局孪生**（`HContainer(*dm_layout)` copy ctor）
  作为 W 目标——DM 布局由邻接表按 rank 组装、永不为空，且 W 与 DM
  位一致配对由构造保证。`audit_weighted_trace()` 传入其 dmr 引用。
  无 dm_layout 的 MPI 调用保留原 IJR 派生分支（A2，文档注明风险）。
- 文件：`constraint_inject_lcao.h`（签名+文档）、
  `constraint_inject_lcao.cpp`（A1/A2 分支 + audit 传参）。

### 3.4 修复后验证

- 重编 `abacus_basic_para`；212_NAO np=4：exit=0，6 外步 CONVERGED，
  FINAL_ETOT=−466.2533233603385 eV（ref 偏差 ~3e-12），
  `M3b runtime audit: max |Tr[W.DM] − ∫wρ| = 1.51262771197e-08 e`。
- np=2 复核：FINAL_ETOT=−466.2533233617352、
  audit=1.51262806725e-08 e；np=1（前期）同量级——三档 rank 审计值
  一致到 ~1e-12 相对（归约顺序差），能量与 ref 一致。
- constraint/partition ctest 12/12 PASS（含 constraint_inject_lcao、
  weight_grid mpi/4np）。
- 全量 `cmake --build .` 在 `source/source_estate/test/` 若干目标失败
  （undefined `InfoNonlocal`/`get_dftu_energy`）——与本次改动无关的
  既有构建配置问题（未触碰那些文件，constraint 模块全量编译通过）。

## 4. Analysis

- 根因链：Gint 实空间网格按 rank 分域（big grids）；每个 rank 的
  `GintInfo::ijr_info_` 只含其网格域所覆盖原子的邻近对 → np=4 小体系
  下 rank 可分到"无原子网格域"→ IJR 空。生产 H(R)/DM 的布局按
  邻接表（与网格分域无关）+ paraV 本地行列构造，rank 不空。M3b 审计
  复用了生产 transfer 机制但错误地以网格导出的 IJR 为目标布局，是
  与生产口径的唯一不一致点（即评审重点项 (a) 的实体）。
- 修复为何正确：W 是 ∫φ_μ w φ_ν dr 的矩阵表示，其**本地块布局**只需
  与 DM 相同即可承载同一 transfer 机制的散播结果（与生产 HR 完全同
  构）；DM 布局包含所有邻接对，某对若任何 rank 的 Gint 域都未计算则
  相应块保持零——数学上该矩阵元本为零，不影响 Tr。np=2 下新旧布局
  恰好一致（审计值 1.5e-8 不变）佐证此点。
- 审计值 1.5e-8 e = SCF 残差量级（观察用混合后密度、DM 为混合前，
  差 O(drho·∫w)），与 M3b 审计 spec 记录一致；矩阵/网格两路径在
  np=4 生产运行中逐位吻合。
- 附带结论：M3b 升格评审 (a)-(d) 全部闭环；遗留 A2 分支仅作 API
  回退，当前无调用者，文档已警示。

## 5. Next steps

- 进入 Task 2.6.1 PW≡LCAO 逐位一致：同一 H₂O 几何、同一收敛密度
  （PW 导出 → LCAO init_chg 同网格读数对拍）Q_α <1e-8；同靶点 μ*
  差 <1%；M3b 审计随跑。数据机器可读块入库。
- 其后 2.6.2 力 FD stationary4（18+18 腿，判据 0.0128555 eV/Å）、
  2.6.3 力矩 FD、2.6.4 自旋反假收敛 + LCAO 4-rank 逐位一致。
- np=4 修复随 2.6.1 高网格腿复跑验证（audit 行每几何出现）。
