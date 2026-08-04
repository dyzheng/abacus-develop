# DeltaP 开发进展与计划（2026-08-04 总览）

> 本文档取代 `2026-07-13-deltap-progress-and-plan.md` 成为最新总览入口。
> 分支 `feat/deltap`，HEAD ≈ 4c446b950+（未提交工作含 Tier-1 骨架与 D_I 修复前期工作）。
> 详细逐轮记录见 `deltap-development-log.md`（压缩索引）。

---

## 0. 一句话现状

**SCF 单点约束功能（LCAO+PW）已达到高工程质量并通过系统性验证；
力/relax 链路经 T7 专项实现全部 Pulay 力项（A1/A2/B）并逐项验证，
但驻点约束判决 FD 结构性 FAIL（软约束 + 分支阶梯 + 代理差距三件套）；
最新发现 D_I MPI 混带 bug（P0，未修）——修复前 LCAO 多 rank 数值不可信。**

---

## 1. 状态仪表盘

### ✅ 已完成并验证

| 领域 | 内容 | 时间/证据 |
|------|------|-----------|
| 核心算法 | Wilson loop + Newton-Schulz 投影 + Hungarian 全局匹配 + gauge 锚定修正 + per-band 分支间距 | 07-13 前（P0/P1 轮） |
| 性能 | unkOverlap_lcao 快速 O_kpair 路径（~30000×） | 07-20（Z01） |
| esolver 重构 | `DeltapScfSolver` 状态机统一 LCAO/PW 控制流；esolver 净删 ~580 行；每轮 A/B 逐字节保真 | 07-31~08-01（R1-R4） |
| 测试覆盖 | mask/gdir2/relax 复现器入库；LCAO 4-rank、PW 1/2-rank 实跑 | 08-01（R5） |
| MPI 一致性（KPAR=1） | λ/γ/escon 跨 rank 一致 + 回归守卫；KPAR>1 WARNING_QUIT 守卫 | 08-01~02（T1/T3→D1-D6） |
| 力项实现 | A1（SMO Pulay）、A2（τ 导数，4ppm 闭合）、B（H_HK 解析力，双闭合） | 08-02~03（T7-a/b/c） |
| 重大 bug 修复 | B-1 力路径越界崩溃、B-6 τ 单位（L 倍放大）、S_dk 相位单位、g==0 相位链 | 08-02~03 |
| 回归锚点 | 12 用例 @ ecutwfc=100/ecutrho=400，含 E'/λ/γ 轨迹 + branch 文件 | 08-03（1b2625fdd） |
| FD 验证设施 | `run_fd.sh` 双组协议（冻结 λ / 重收敛 λ），支持 ECUTWFC 等覆盖 | 08-02（T7-b） |

### ❌ 未解决（按严重度）

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | **D_I MPI 混带 bug**：`psi->get_nbands()` 返回本地带数，D_I 按本地带分配 → 奇数 NBANDS 崩溃（co/h2o_asym 4-rank TRUNCATE），偶数 NBANDS 混带（hf 4-rank 逐原子 γ 漂移 ~0.8 rad，Σγ 巧合守恒） | **LCAO 多 rank 数值不可信**（此前"MPI 合理"判断作废） | P0 未修，方案 A' 已定（见 §5） |
| 2 | **驻点组② FD 结构性 FAIL**（84.8 eV/Å，5 位闭合于 λ-leakage） | 强约束 relax 力不一致 | 归因三件套（分支阶梯/γ 软性/代理差距），分支分解判别实验待做 |
| 3 | 内循环 SCF 极限环（drho 骑阈值振荡，1v4 rank 发散） | relax 生产化阻塞 | 待三界定实验 |
| 4 | C/R 力项未实现 | 仅冻结 λ 扫描场景（relax 不需要，已论证） | 暂缓 |
| 5 | relax/MD 端到端 | 不可用 | 受 #2/#3 阻塞 |

### ⏳ 待办小项

smoothness 单测参考更新（B-6 约定变更）· Tier-1 筛选 1/2 未完成 · PW FD 力验证 · PW 应力 · D-D 正式验收（暂停，等判决）· P01-P18 入 CI

---

## 2. 功能可用性矩阵（当前可信面）

| 场景 | 状态 | 备注 |
|------|------|------|
| LCAO SCF 单点 γ/λ（串行） | ✅ | 约束矩阵/total/per-atom/mask/gdir 均验证 |
| LCAO 内循环 λ 优化（串行） | ✅ | test_stru_target、D2 λ×L 已核实 |
| LCAO 多 rank | ❌ | D_I 混带 bug 修复前不可信（方阵网格 4-rank 恰好整除的用例外貌正常） |
| PW 单点（KPAR=1） | ✅ | 无 target 文件/约束矩阵接线；total 模式是"假 total" |
| PW 多 rank | ✅ 2-rank 验证 | KPAR>1 被守卫拒绝 |
| 力（串行） | ⚠️ 分项验证通过 | A1/A2/B 各自闭合；端到端 relax 不可用（#2/#3） |
| 应力 | ⚠️ S1 实现未 FD 验证 | PW 应力完全未实现 |
| relax/MD | ❌ | 见 #2/#3 |
| KPAR>1 | ❌ | WARNING_QUIT 守卫（k-string 跨全 k 列表） |

---

## 3. 近期进展时间线（07-13 之后）

| 日期 | 轮次 | 要点 |
|------|------|------|
| 07-20~29 | 第二阶段修复 | Z01 快速路径核实；61 项风险评审（C-01~C-20）；修 C-05/C-11/C-02-Step1/C-07；BN 收敛性诊断（内循环解决不收敛） |
| 07-21 | BN PES 采样 | **电子极化刚度 <1e-6 Ry/rad² ≈ 0**（物理发现：BN 的 γ 是近自由度）；分支发散记录 |
| 07-30 | P 系列测试集 | P01-P18 共 18 个可靠性用例构建；冒烟修复 KPT/gamma_only/F2=2 自旋因子 |
| 07-31~08-01 | **esolver 重构 R1-R5** | DeltapScfSolver 状态机；删死代码 ~580 行；修 C-21~C-28 系列；deltap_common 单测 10 例；逐字节 A/B 保真链 |
| 08-01~02 | MPI 一致性 | T1（LCAO λ 写回）/T3（PW γ Bcast）→ 严格评审 → D1-D6 修订（KPAR 守卫、apply_lambda 统一出口、γ 一致性守卫、3 用例 MPI 冒烟） |
| 08-02 | T7-a/b | 修 B-1 力路径崩溃（nlm 多 ζ 布局越界）；FD 双组协议首次量化：B-7（H_HK 力缺失 ~2.6 eV/Å）主导、B-6（τ 单位）、A2/C 预算 |
| 08-03 | T7-c | H_HK 解析力实现（g==0 相位链修复，单原子+均匀平移**双闭合**）；A2 实现（4ppm）；**S_dk 相位 τ 单位破案**（15.87×=L）；B-6 修复（E_H_HR 缩 15.90×） |
| 08-03 | 锚点 + 判决 | 锚点 @ecutwfc=100 重建（12 用例）；D2 内循环 λ×L 核实（3.66 vs 3.615）；**驻点组② FD 结构性 FAIL**（84.8 eV/Å = 224.4 eV/Ry × 0.378 Ry/Å，5 位闭合）→ O5 触发 |
| 08-03~04 | 分支/Tier-1 | 分支分解判别方案；Tier-1 非对称体系（hf/co/h2o_asym）构建；筛选 3 三体系 PASS；**发现 D_I MPI 混带 bug（P0）** |

---

## 4. 关键技术结论（知识库摘要，详见 `2026-08-03-deltap-force-knowledge-update.md` v2）

1. **力分解**：F = F_KS + A1 + A2 + B + C + R；前三项=已实现 H_c 的 Pulay（已验证）；
   C+R 在约束激活时相消（dspin 定理）——relax 正确性只需 A1+A2+B。
2. **dspin 定理前提**：λ 须在约束驻点；同步单步 GD 不满足 → 组② 残差是协议伪差；
   驻点协议下仍 FAIL ⇒ 深层三件套：**γ 分支阶梯性 × γ 软性（∂γ/∂λ~10³ rad/Ry
   放大 dλ*/dR）× 代理差距（Γ_op≠γ，∂E'/∂λ≈224 eV/Ry≠0）**。
3. **E' 原点敏感性** = 代理差距的直接定量（H_HR 比真 γ 大 ~8 倍，同号不相消）。
4. **FD 验收处方**：ecutwfc=100 + 显式 ecutrho≥400 + scf_thr 1e-8；判据级结论
   在低 ecut 下不可信（egg-box ~0.006 eV/Å ≈ 判据一半）；归因分析可用隔离对照豁免。
5. **均值扣除陷阱**（FORCE_STRESS.cpp:599）：力不守恒被摊平成均匀偏移；
   FD 用未扣除力；ΣF=0 单列验收。
6. **τ 一律分数坐标（taud）**：B-6 为量纲 bug（相位链证明），E=−πλ/(2a) 只在
   分数 τ 成立；历史 λ 解读 ×1/L 重标定。

---

## 5. 当前 TODO（优先级序，详见 `2026-08-04-deltap-tier1-screening-status.md` §5）

### P0 — D_I MPI 修复（一切多 rank 工作的前提）
- 方案 A'：D_I 按全局 nbands 分配清零 → 各 rank 只填自己拥有的全局带槽位
  （n_local→n_global 用 paraV 列带分布映射）→ 单次全通信子 Allreduce
  （行部分和与列带槽位天然不相交，无需子通信子管理）。
- **同族审计**：compute_D_I 另两调用点（:1689/:1897）、compute_hk_correction
  （c_R 本地列索引）、compute_berry_connection。
- 回归三件套：hf 4-rank 逐原子 γ 对串行 ±0.01 rad；co/h2o_asym 4-rank 不崩；
  hf 恢复 ~37 iter 收敛；MPI smoke 补奇数 NBANDS 用例（co）。

### P1 — 筛选与判决（串行部分可与 P0 并行）
- Tier-1 筛选 1（γ(λ) 三点）/2（±δ 分支零翻转）/3（重跑补录）——工作区迁仓库侧
  gitignored 目录；判决体系预期 hf。
- 分支分解判别：方案 A（离线 λ 轨迹重放，零成本）→ 方案 B（连续性-only 重跑）。
- smoothness 单测参考更新。

### P2 — 决策与固化
- hf MPI-vs-串行 γ 对照正式化为回归锚点（修复后必须闭合）。
- O5 决策（算符重构 / escon 改 ⟨Ô⟩ 记账 / 弱约束产品化 O5c）——等分支分解数据。
- 内循环极限环：三界定实验（范围/混沌 vs MPI/门控磁滞）→ 修复。

### P3 — 收尾与增强
- D-D 正式验收（高 ecut 全矩阵 FD，锚点已就位）。
- PW FD 力验证（ecutwfc 80-100）；PW 应力立项。
- P01-P18 纳入 CI；`compute_S_dk_link` 接线或删除；BTO W90 验证。

---

## 6. 主要文档索引

| 主题 | 文档 |
|------|------|
| 本总览 | 本文档（最新入口） |
| 压缩索引/长上下文 | `deltap-development-log.md` |
| 力求解知识库（名词解释+难点+公式集） | `2026-08-03-deltap-force-knowledge-update.md`（v2 详解版） |
| 力/应力开发指南（公式推导+check-list） | `2026-08-02-deltap-force-stress-dev-guide.md`（v2） |
| 驻点组② FAIL 与归因 | `2026-08-03-deltap-force-stationary-group2-fd.md` |
| 分支分解判别方案 | `2026-08-03-deltap-branch-decomposition-plan.md` |
| D_I MPI bug 与筛选状态 | `2026-08-04-deltap-tier1-screening-status.md` |
| 重构 R1-R5 | `2026-07-31-deltap-esolver-refactor-design.md` + R1~R5 dated 文档 |
| 风险台账（C-01~C-20） | `2026-07-29-deltap-risk-assessment-review.md` |
| P 系列测试集 | `2026-07-30-pseries-*`（A/B/C/D 组 + 整体验证） |

---

## 本轮记录

- 本文档为总览更新轮（取代 07-13 版），无代码改动。
- 数据综合自 §6 所列各 dated 文档。
