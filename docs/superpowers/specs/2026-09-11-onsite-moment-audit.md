# 审计行 on-site 投影矩（最小 II-1b 仪器）落地

> 批复顺序第 4 项的前半（`2026-09-11-ii1-triage-review.md` §三.4："审计行
> on-site 矩打印（最小 II-1b）+ Fe 半径敏感性研究"）。本轮交付**仪器**（打印）；
> **半径敏感性研究（4b）显式顺延**到下一轮，与本轮给出的 `BRANCH_TOL`/重锚定
> 工作合批（理由见 §5）。

## 1. 测试计划（Test plan）

| # | 目标 | 判据 |
|---|---|---|
| P1 | 审计行新增 `onsite=`（DFT+U 运行时） | 每条约束的 `c[i]` 行出现 `onsite=`，数值 = 该片段原子 on-site 矩之和 |
| P2 | 与 `atomic mag` 同量 | `Plus_U::onsite_moment` 等于 DFT+U `atomic mag` 行的 `sum0[0]−sum0[1]`（迹 = 本征值和的恒等式） |
| P3 | 无 DFT+U 时输出逐位不变 | H₂O 四个集成用例（无 DFT+U）审计行无 `onsite=`，其余 token 与数值不变 |
| P4 | 失败响亮、不静默给错数 | 原子索引越界 → 整个 token 被丢弃（而不是部分求和） |
| P5 | 纯信息性 | `set_onsite_moments` 不进入注入/求解/力任何路径；`reset()` 清空 |
| P6 | 端到端 | FeO（PW+DFT+U）δ=+0.1 μB 点：同一行读到 `q` 与 `onsite`，并定量给出脱钩/跟随幅度 |

## 2. 测试设置（Test setup）

- 平台：容器 16 GB/14 核（OMP 3）；`build/`（debug，单测）+ `build_rel/abacus_basic_para`（生产）。
- 单测：`make MODULE_ESTATE_constraint_accounting MODULE_ESTATE_constraint_loop` +
  `ctest -R constraint`。新增 `OnsiteMomentFragmentSum`（accounting，含 legacy
  无 token 与越界丢 token 两个判别）与 `OnsiteMomentsReachTheAuditLine`（loop）。
- 端到端 1（有 DFT+U）：FeO `S3L_fe2_0p1` 的设置（热启动自 `S3L_donor`，
  δ=+0.1 μB on Fe2，scf_thr 1e-7，`onsite_radius 3.0`），np4；审计痕迹入库
  `tests/deltap_feo_spin_scan/results/onsite/ONSITE_fe2_p01.audit`。
- 端到端 2（无 DFT+U）：H₂O `211_PW_constraint_h2o`（+`constraint_branch_tol 1e-3`），
  检查 `onsite=` 出现 0 次。

## 3. 结果（Results）

- 单测：`ctest -R constraint` **11/11 全绿**（~50 s）；accounting 4 → 5 用例、
  loop 18 → 19 用例，全部 PASS。
- 端到端 1（FeO δ=+0.1 μB，CONVERGED，μ* = −0.06549837621 Ry，与上一轮 S3L
  的 −0.0655152 差 0.03%（本轮步长默认可比））：

| 状态 | `q`（Becke）[μB] | `t` [μB] | `onsite`（d 投影）[μB] | Δq | Δonsite |
|---|---|---|---|---|---|
| 参考相 μ=0 | 3.384107449 | 3.484107449 | **3.714721308** | — | — |
| 收敛 δ=+0.1 | 3.484068604 | 3.484107449 | **3.788478732** | **+0.09996** | **+0.07376** |

  ⇒ 良态点上两观测量**同向但不等幅**：on-site 只跟上 Becke 响应的 **74%**；
  与 II-1b 的换态点（Becke −1.28 vs onsite −0.20，反向脱钩）合起来给出完整的
  "同向跟随 → 反向脱钩"图景。
- 端到端 2（H₂O，无 DFT+U）：`onsite=` 出现 **0 次**，`c[0] kind=charge
  q=... t=... mu=... res=...` 与改动前逐位同格式；final status CONVERGED。
- 数值一致性（P2）：收敛点的 `onsite=3.788478732` 与同一 iteration 的
  `atomic mag: 2 3.78847873` 一致到 1e-8（同一量、同一路径）；与**上一**迭代的
  `atomic mag` 差 ~4.5e-5 μB（`locale` 是上一迭代占据，见 §4）。

## 4. 分析（Analysis）

1. **同量恒等式**：`write_occup_m` 打印的 `atomic mag` 用"占据矩阵本征值之和"，
   而本征值之和 ≡ 迹；`onsite_moment` 直接取迹，因此两条路径报告同一个数（实测
   1e-8 一致），但省掉每次迭代的 5×5 对角化，代价 O(2l+1)。
2. **片段求和的口径正确性**：`q_α = ∫w_α d_α`，`w_α = Σ_{I∈fragment} w_I`；
   `onsite_α = Σ_{I∈fragment} m_I`。两者在同一 `constraint_atoms()` 片段上求和，
   所以 `q` vs `onsite` 的差**只**来自投影子（Becke 盆地 vs 关联轨道投影球），
   正是 II-1b 要归因的那一项。
3. **陈旧度**：`locale` 在 PW 的 `iter_init`（`cal_occ_pw`）计算，即审计行用的是
   **上一迭代**的占据；收敛点上偏差 ~1e-4 μB（本例 3.788433751 → 3.788478732）。
   这一点写进开发者文档，避免把 1e-5 级差异当成 bug。
4. **零侵入**：`onsite` 是 `ConstraintAudit` 的附加字段，`audit_line` 仅在
   `onsite.size() == Q.size()` 时打印 token；loop 只在 esolver 显式喂入时才有值，
   所以①无 DFT+U 的用例输出逐位不变（P3 实测 0 次 token），②注入/求解/力路径
   一行未改（`set_onsite_moments` 只写成员）。
5. **能量口径的交叉验证**：本轮 FeO δ=+0.1 的 `!FINAL_ETOT_IS = −7652.9640439879 eV`
   与上一轮 S3L 记录的 −7652.9640435864 eV 相差 4e-7 eV（同基线、同步长），说明
   +0.04392 eV 的能升结论在本轮二进制上复现。
6. **为什么 4b（半径敏感性）顺延**：Becke 分区半径来自 `ModuleBase::CovalentRadius`
   表（`configure_from_inputs` 内一次性生成），要做"Fe 半径敏感性"必须①新增一个
   半径覆写/缩放 INPUT（本案目前没有），②每个半径下重跑参考相（改变半径 = 改变
   观测量定义，不是收敛参数），③这些运行必须锚在**已定死的参考态**上——而 II-1 的
   锚点正是评审要求下一轮处理的（重锚定 (a) 换 k 网格 + (c) 只报 ±0.1 窗口）。
   在锚点未定时做半径敏感性，量到的是"亚稳盆地 + 半径"的混合效应。因此本轮先交
   仪器（本轮已可读 `q` vs `onsite`），半径研究并入下一轮的重锚定批次。

## 5. 下一步（Next steps）

1. **4b + II-1 重锚定合批**：新增 Becke 半径覆写 INPUT（名字/语义待定，建议
   `constraint_radius_scale`，默认 1.0 = 现行为）→ 在已定死的锚点上跑
   Fe 半径扫描（如 0.8/1.0/1.2 × δ=+0.1）→ 用 `q` vs `onsite` 定量脱钩幅度；
2. 批复第 5 项：能力边界文档补"Becke 矩/d 局域矩脱钩"条目（本轮手册 §5.8 已先给
   用户侧要点，能力边界表待补）；
3. 批复第 6 项：I-1 MgO 继续；FeO S4/S5 仍暂缓；
4. 顺带：FeO 扫描脚本可把 `onsite` 一并抓进 `audit()`（本轮仅入了单点快照）。
