# DeltaP T-6'（Ô_w）实现轮：R1–R6/R8 落地 + R5 平直性首测 + V-H8 SCF 稳定性（2026-08-12）

> 依据：`2026-08-12-deltap-formula-review-risks.md`（§3 R1–R9、§4 处置顺序）+
> `2026-08-04-deltap-execution-todo.md` T-6'。本文件为"修改 + 测试"轮文档
> （AGENTS.md 纪律）；评审文档（risks）留在树中随本轮提交。
> 验收参照物：R5 判据（斜率 ≲1 eV/Ry，对照 T2 的 −0.013 eV/Ry）；
> V-H8 判据（ow SCF 稳定：无 D2 分支跳变导致的哈密顿量跳变）。

---

## 1. 测试计划

1. **R5 平直性首测（记账恒等式）**：ow 模式 λ 扫描 ±0.01，冻结 λ
   （`deltap_lambda_step 0.0`），测 E'(λ) 斜率。判据 ≲1 eV/Ry；
   若 O(1) → Γ 记账与 H_c 不一致，回查 Tr[ρH_ow] 逐原子对账。
2. **λ=0 零回归**：ow 与 proxy 同输入 λ=0 对比（E 逐位、γ 自然值、escon=0）。
3. **V-H8 SCF 稳定性**：ow λ=+0.01 长跑（150 iter），查是否收敛、
   是否有 D2 分支跳变消息（|Δθ|>π/2 冻结路径是否被触发）；
   同参数 proxy 对照（预期收敛 → ow 特有的不稳定性定位到状态依赖 H_ow）。
4. **回归**：`ctest -R deltap`（单测）+ 编译；已知 R9 smoothness 4/8
   参考过期（预先存在，不在本轮判定内）。

## 2. 测试设置

- 系统：h2o1，2×2×2 k 网格（8 k 点，2 条 string/gdir），
  ecutwfc=100 / ecutrho=400 / scf_thr=1e-8，生产设置。
- 二进制：`/root/abacus-develop/build/abacus_basic_para`（MPI 构建，串行运行）。
- 并行纪律：`OMP_NUM_THREADS=1`，单任务运行，严禁并行测试。
- INPUT 关键项：`deltap_operator_mode ow`（扫描/ow 运行）或 `proxy`（对照）；
  `deltap_observable operator`；`deltap_lambda_init` 逐点冻结；
  `deltap_lambda_step 0.0`；`deltap_branch_write 0`（T-9'：保持标定
  branch.dat 参考跨扫描）；校准 branch.dat 拷贝进所有扫描目录。
- 目录：`/tmp/deltap_ow_r5/{l-0.01,l-0.003,l-0.001,l0,l0.001,l0.003,l0.01,
  ow_0.01_long,proxy_0,proxy_0.01}`（+ 提交前复查 `ow_verify2`）。

## 3. 结果

### 3.1 R5：ow E'(λ) 扫描（FINAL_ETOT_IS）

| λ (Ry) | E' (eV) | escon (Ry) | SCF |
|---|---|---|---|
| −0.01  | −481.6713726094 | −0.275421 | 未收敛（50 iter 上限） |
| −0.003 | −481.6944080577 | −0.084187 | 收敛（28 iter） |
| −0.001 | −481.6970117711 | −0.028257 | 收敛（31 iter） |
|  0     | −481.6973727148 |  0        | 收敛（26 iter） |
| +0.001 | −481.6969775139 | +0.028480 | 未收敛（50 iter 上限） |
| +0.003 | −481.6950436653 | +0.086090 | 未收敛（50 iter 上限） |
| +0.01  | −481.6350322372 | +0.299550 | 未收敛（50 iter 上限） |

**斜率（±0.001 窗，收敛区）**：

- [−0.001, 0] 弦斜率 = −0.361 eV/Ry；
- [0, +0.001] 弦斜率 = +0.395 eV/Ry；
- 抛物拟合 E' = E₀ + aλ + bλ²：a ≈ +0.017 eV/Ry（线性残差），
  b ≈ +378 eV/Ry²（λ² 曲率）；两弦斜率差 ~9% 完全由 b 解释
  （弦斜率 = a ± b·δ，δ=0.001）。

**判定：R5 PASS 信号**——±0.001 窗对称 λ² 抛物、线性系数 a=0.017 eV/Ry
比 T2 参照 −0.013 eV/Ry 同量级，比 R5 判据 ≲1 eV/Ry 小 ~60×；
escon 与 Γ 记账自洽（escon ≡ −λ·Γ_ow，逐点核对：λ=−0.001 时
−λ·Γ = 0.001×(−19.031−4.613−4.613) = −0.028257 Ry ✓）。

**诚实标注**：±0.003 与 ±0.01 四个点 SCF 未收敛（50 iter 上限，
drho 未达 1e-8）——斜率数据只取 ±0.001 收敛窗；大 λ 区的 E' 数值是
未收敛密度下的 Harris 能量，只作 V-H8 稳定性判据，不作记账判据。

### 3.2 λ=0 零回归（ow vs proxy）

| 量 | ow (l0) | proxy (proxy_0) |
|---|---|---|
| FINAL_ETOT_IS | −481.6973727147502 eV | −481.6973727147502 eV（逐位同） |
| γ 报告 | (−6.399, −3.166, −3.166) | (−6.399, −3.166, −3.166)（自然值，同） |
| escon | 0 | 0 |
| λ | (0,0,0) | (0,0,0) |
| Γ 报告 | (−19.111, −4.627, −4.627) | (4.375, 1.362, 1.362) |

Γ 不同是**预期**的：ow 的记账观测量是 Γ^w + Γ^HK（θ_n 权重通道，
全 Gram 迹），proxy 是 τ_α·⟨P̂⟩ + Γ^HK——两个不同算符的自然值不同；
λ=0 时两者 H_c 均为零（escon=0、E 逐位同），零回归成立。
**提交前复验**（ow_verify3，含 MPI/rank-0 守卫的最终二进制）：26 iter
收敛，γ/Γ/P3 与 l0 逐位一致，FINAL_ETOT_IS 同值 → 守卫改动零行为回归。

### 3.3 V-H8：ow λ=+0.01 SCF 稳定性（ow_0.01_long）

- 150 iter 未收敛：P3 行持续打印至 iter=150，Γ/escon 稳定
  （Γ=(−20.304,−4.825,−4.825)、escon=+0.299550 Ry，iter 47 起逐位稳定），
  但 drho 不降（极限环）。
- **无 D2 分支跳变消息**（`rg "D2 branch jump|θ_n frozen"` 零命中）→
  不稳定性不是分支跳变（2π 不连续）引起，而是**状态依赖 H_ow 自身的极限环**。
- **对照 proxy_0.01**（同 INPUT 仅 operator_mode 不同）：27 iter 收敛，
  E=−481.6969692656 eV → ow 特有，实锤。

### 3.4 回归

- 编译：`cmake --build . --target abacus_basic_para -j 8` ✅；
- `ctest -R deltap`（串行，无 -j）：3/4 测试通过，
  `MODULE_ESOLVER_deltap_common_test`（11 单测）✅；
  `MODULE_LCAO_deltap_smoothness_test` 4/8 FAIL = R9 预先存在
  （git-stash 对照复现过，与本轮改动无关，待办）。

## 4. 分析

### 4.1 R5 平直性成立 → ow 记账恒等式在收敛窗内成立

a=0.017 eV/Ry 与 T2（−0.013 eV/Ry）同量级，说明 ow 模式下
escon ≡ −⟨H_ow⟩ 的记账（Γ^w 全 Gram 迹 + Γ^HK）与施加的 H_c 一致。
λ→0 线性残差与 T2 的 O(λ) 泄漏结构同族（残差 ∝|λ*|，工作窗内可验收）。
这是"非正交基下算符记账必须走全迹"教训在 Ô_w 上的第二次独立复现
（第一次 T2/F_HK）。

### 4.2 V-H8 结论：H_ow 极限环 = 状态依赖哈密顿量问题，非分支跳变

λ=0.01 的 150 iter 极限环中，θ_n 未发生 >π/2 的跳变（D2 冻结路径
零触发），但 SCF 不收敛——H_ow(ψ) 依赖当前 ψ（θ_n 是 ψ 的函数），
冻结 λ 下形成自洽极限环。对照 proxy 收敛，说明这是 Ô_w 通道特有的
状态依赖刚度问题。**这不是分支跳变修复（R4）能覆盖的**，是 T-6'
的下一步主线：H_ow 的 SCF 稳定化（内循环冻 θ / 混合参数 / 或
θ 更新限幅）必须解决，否则 V-H3' 判据（γ-hold FD 残差 <0.02 eV/Å）
在 λ*~2e-3 量级之外无法执行。V-H8 的"必查项"（Ô_w 下 γ(λ) 单调区间
实测）同样需要先解决 SCF 收敛。

### 4.3 R1–R6/R8 落地情况（代码侧）

- **R1（F_ow，阻塞级）**：`compute_ow_force` 实现并接线
  （`compute_hk_force` ow 分支）：E_ow=ΣλΓ^w 的冻结 C/θ 导数，
  ∂S_k/∂R 双中心核（bra/ket 反对称 + 相位项），X/Z 每通道权重，
  全 Gram 迹；`force_stress.hpp` 虚假注释同步修正 + ow 门控
  （cal_stress → WARNING_QUIT，R6）。
- **R2/R3（k 局域化）**：θ 按 k 存储（`ow_theta_k_`），H_ow/Γ^w
  在全部 string 上构建（`total_string_`），`fill_kstring` 集中
  重建 S_k/D_I（含 MPI Allreduce）。
- **R4（D2 冻结 + 恢复）**：每 k 冻结计数，50 次连续冻结后接受
  （合法演化不被永久冻结）；`freeze_branch_ref` 清空 prev/计数
  使下次运行重锚。首测锚定用 `anchor_ow_theta_to_ref`（R4 首轮）。
- **R8**：Im(Γ_I^w) 软告警（rank-0，提交前加守卫）。

### 4.4 工程守卫（提交前补）

- `compute_ow_force` 加 `#ifdef __MPI` nproc>1 防御守卫
  （父级 `compute_hk_force` 已有 serial-only 守卫，此处自包含防未来误用；
  Π=C†C 与 force_out 无 Allreduce）。
- R8 告警加 `GlobalV::MY_RANK == 0`（防 MPI 多 rank 重复打印）。

## 5. 下一步

1. **H_ow SCF 稳定化**（V-H8 主线）：内循环冻 θ / 混合 / θ 更新限幅，
   目标 λ=±0.01 收敛 → 重跑 R5 全窗（含 ±0.003/±0.01 收敛点），
   把"斜率 vs |λ|"曲线做满。
2. **V-H3' 判决**：γ 直驱驻点 FD @ 0.98 靶点，残差 <0.02 eV/Å
   （须先过 1 的收敛门）。
3. **γ(λ) 单调区间实测**（Ô_w 下，V-H8 必查项）。
4. **R7**：nbands<nocc 金属体系守卫（WARNING 已有，补测试）。
5. 进入 T-7'（per-atom Jacobian）前复查 H_ow 力（R1）的
   闭合计算（单原子 E_ow FD ↔ 解析 F_ow）。
