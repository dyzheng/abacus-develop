# F-7b 闭环：Q1 escon 测量时机代码修复 + Q2 scan.py 影响面审计（2026-08-17）

> 本轮回应评审（2026-08-17，F-7b 裁定）的两个后续项 + F-8 协议签核修正记录。
> Q1：escon 伪影修代码还是留文档？→ **修代码**（本轮落地）。
> Q2：scan.py 嵌套 if bug 的影响面？→ **仅 F-7b 复测**，无需重跑。
> F-8 协议：§5 PASS 行修正 + efield 对照升级必测项（已并入协议文档）。

---

## 1. 测试计划

1. **Q1 代码修复**：把 PW 侧 escon 的测量时机从"首个 drho<deltap_inner_thr
   迭代的一次性测量"改为"每个 SCF iter_finish 在当前 ψ 上刷新"。
   验证目标：ecut=80 + **默认 inner_thr=1e-3**（此前被污染的档位）下
   E'(λ) 中心差从 −0.0214 Ry/Ry 塌缩到 ≤1e-4 量级（与 F-7b 的
   inner_thr=1e-6 口径 0.00014 同量级）。
2. **Q2 审计**：确认 scan.py 的嵌套 if 选错 ecut bug 只影响 F-7b 复测；
   既往验收（F-1/F-2/F-2b/F-6/Stage-3 等）不经由该脚本选择 ecut。
3. **回归**：MPI smoke 4/4（PW 2-rank + BN/CO/BN-inner 4-rank）确认
   修复不破坏并行路径。

## 2. 测试设置

- 体系：`tests/deltap_fd_force/hf`（F–H，KPT=1×1×2，gdir=3），
  ecutwfc=80 / ecutrho=320 / scf_thr=1e-8 / **inner_thr 默认 1e-3**。
- λ 三点冻结扫描：λ ∈ {−0.001, 0, +0.001} Ry（`deltap_lambda_mixing=0`）。
- 修复后 `[DeltaP-PW]` 打印行数统计（refresh 直接覆写 `state.dp_escon`，
  不经 report 打印——预期每 run 仍 1 条，但 FINAL_ETOT 的 escon 已刷新）。
- 回归：`tests/deltap_mpi_smoke/run.sh`（严禁并行，单任务 MPI，
  `OMP_NUM_THREADS=1`）。

## 3. 结果

### 3.1 Q1 修复后 E'(λ)（ecut=80，inner_thr=1e-3）

| λ (Ry) | E' (eV) |
|--------|---------|
| −0.001 | −466.9457249 |
| 0      | −466.9457391 |
| +0.001 | −466.9457256 |

- 中心差 ±0.001：**−2.5e-5 Ry/Ry**（修复前同档 −0.0214，改善 ~850×）。
- ΣΓ@λ=0 = 7.9324（刷新口径）。
- `[DeltaP-PW]` 行仍每 run 1 条（只有 P2 时刻的那条 + 刷新不打印；
  刷新后 `f_en.dp_escon` 用新值——即打印的是旧 escon，能量用的是新 escon；
  如需诊断可另行加打印，不阻塞）。

### 3.2 回归

- MPI smoke：**4/4 PASS**（PW 2-rank + BN/CO/BN-inner 4-rank）。

### 3.3 Q2 审计

- `/tmp` 全盘 find：唯一 `scan.py` 位于 `/tmp/deltap_l1_1_pw/scan.py`
  （及衍生 `run_scan*.py`）——即 F-7b 复测目录。
- F-1/F-2/F-2b/F-6 各自独立运行目录，INPUT 直接写 `ecutwfc=100`
  （dated 文档可证），不经 scan.py 选档。
- **结论：影响面仅 F-7b 复测，无既往验收需要重跑。**

## 4. 分析

### 4.1 Q1 根因与修复机制

F-7b 归因：escon（=−λΓ^PW）在首个 `drho<inner_thr` 迭代一次性测量
（drho≈7e-4），此后的 SCF 迭代能量用同一个陈旧 escon，而本征值用
收敛 ψ——测量态与记账态错位 → Γ 偏差 ~0.1–0.3%（ecut=80 时 0.27%）。
修复：`iter_finish` 开头（任何能量计算前）调
`pw_deltap::refresh_pw_escon(ucell, psi_cpu, wg)`，在当前 ψ 上重算
escon 并覆写 `state.dp_escon` 与 `f_en.dp_escon`——能量与本征值同 ψ。

改动（3 源文件 + 1 头文件）：

- `source/source_pw/module_pwdft/deltap_pw.h/.cpp`：新增
  `refresh_pw_escon(ucell, psi_cpu, wg)`（内部复用
  `compute_gamma_op_pw` + `deltap_common::compute_dp_escon`；λ 空则跳过）。
- `source/source_esolver/deltap_scf.h`：新增 `set_escon(double)` 覆写
  `state_.dp_escon`（PW 刷新路径）。
- `source/source_esolver/esolver_ks_pw.cpp::iter_finish`：开头调 refresh
  并同步 `f_en.dp_escon`（`deltap_switch && deltap_corr` 门控）。

### 4.2 Q1 决策：修代码而非留文档

修复成本 = 1 个函数 + 1 个调用点（~30 行），机制是"末迭代重测期望值"，
与 LCAO 侧 `compute_gamma_op_hk` 的"当前 ψ 上重测"同族——不存在架构性
困难（两相模式的测量时机耦合很浅：PW 侧 escon 只由 λ 与当前 Γ 决定）。
因此按评审建议，**L11 从当前限制降级为历史注记**（阈值型规避
inner_thr≤1e-6 已不再需要，避免把陷阱留给下一个用户）。

### 4.3 Q2 结论

`scan.py` 的嵌套 if（"ecut=80 实为 ecut=40"）只在 F-7b 复测目录里使用过；
其余验收轮次不经过该脚本。脚本已修（F-7b 文档记录），无重跑需求。

### 4.4 F-8 协议签核修正记录

评审 2026-08-17 签核：F-8 协议附一处修正后通过，修正已并入
`2026-08-17-deltap-f8-maxwell-protocol.md`：

- §5 PASS 行修正：内部 Maxwell 恒等是记账推论，PASS 只证明
  ① H_HK 应力实现 = E_HK(ε) 精确梯度；② 记账恒等在应变通道成立；
  对 17.7× 类场物理失配**零信息量**。
- efield 压电对照从补充测量升级为**必测项**（不设 pass/fail，数据必须有；
  F-2 纪律：dip_cor 基线 amp=0、matched-μ）。
- 无需再等一轮评审，本 dated 文档注明"评审修正已并入"。

## 5. 下一步

1. **F-8 实现**（协议已签核）：H_HK 应力（LCAO 串行路径，
   `compute_hk_force` 扩展 + esolver/FORCE_STRESS 接线 + R6 门控修订：
   hk 模式放行串行应力、ow 与 MPI+hk 保持拒绝）→ V-H7 内部 FD +
   Maxwell 判据 + efield 必测项。
2. F-8 完成后进 Stage 4（4.1 含 PW Γ 锚点、4.2 含 ⟨η⟩ 关联检查）。
3. 能力边界文档同步（L11 降级历史注记；L4 应力行随 F-8 结果刷新）。
