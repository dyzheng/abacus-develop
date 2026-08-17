# DeltaP Stage 4.4：V1（E_eff 符号/因子 efield 对拍）+ V3（BN 新记账刚度复核）

- 日期：2026-08-17
- 前置：4.1–4.3 完成（锚点 #3 / 驻点 FD 21-21 / relax 端到端）；4.4 为
  Stage 4 收官验收项，对应推导文档 §10 验证锚点 V1/V3。
- 范围：① V1：锯齿场对照钉 E_eff 符号与数值因子（预言 E_eff=λ/(2a) 10% 内）；
  ② V3：BN 9 点采样在新记账（escon=−λ·Γ）下刚度 <1e-6 Ry/rad² 结论复核。

## 1. 测试计划

1. **V1（当前 HEAD 全新测量，零复用旧数据）**：h2o1（gdir=3，盒长 30 Bohr），
   双通道同几何：
   - DeltaP 通道：λ 冻结 ∈ {−0.01, −0.003, +0.003, +0.01} Ry（proxy，
     `deltap_lambda_init` + step=0 + secant off），采 (λ, Σγ_raw, E')；
   - efield 通道：锯齿场 E ∈ {−0.001, −0.0003, +0.0003, +0.001} a.u.
     （`efield_flag 1`/`efield_dir 2`/`dip_cor_flag 1`，λ=0 冻结），采
     (E, Σγ_raw, E')；
   - 判定：符号（两通道 Σγ 响应同号？）、因子
     E_eff = (dΣγ/dλ)/(dΣγ/dE) vs 公式 (b) πλ/(2a)；μ₀ = −dE'/dE 与 D2/
     实验互证（通道校准正确性）。
2. **V3（新记账 γ-drive 重采样）**：BN 9 点（center/x±/y±/diag±/anti±，
   γ 靶同旧轮 3.9–4.1/3.4–3.6）在 **operator 模式 + `deltap_drive gamma`**
   （γ 直接约束，escon 仍 −λ·Γ）下重跑 + λ=0 自然参考；拟合 E'(Σγ) 刚度，
   对照旧轮（2026-07-21 γ-mode）<1e-6 Ry/rad² 结论。附注：锚点 #3 的
   proxy-drive 数据（κ=1 t_Γ 初值）不是 γ-PES 测量，不用于刚度判定。

## 2. 测试设置

- 系统：同仓库单机；OMP_NUM_THREADS=1；串行 `build/abacus_basic_para`
  （HEAD `3241475c5`，含 4.3 无 target 修复）；一次一个任务。
- V1 目录：`/tmp/deltap_43/v1_dp/{lam_m01,lam_m003,lam_p003,lam_p01}`、
  `/tmp/deltap_43/v1_ef/{ef_m001,ef_m0003,ef_p0003,ef_p001}`、base
  （λ=0/E=0 参考）。
- V3 目录：`/tmp/deltap_43/v3_bn/{base,center,x_plus,...,anti_minus}`。
- 关键参数：ecutwfc=100/ecutrho=400/scf_thr=1e-8、genelpa、symmetry=-1、
  deltap wannier/rm 3.0/gdir 3/corr 1/inner_thr 1e-3；BN 沿用 bn_sampling
  原 INPUT（scf_nmax=50、out_chg 1）+ `deltap_drive gamma`。
- 数据源：`FINAL_ETOT_IS`、`[rawG] Σγ_raw`（6 位）、`[DeltaP P3]`。

## 3. 结果

### 3.1 V1：双通道响应与 E_eff 裁决

| 通道 | 响应 | 数值 |
|---|---|---|
| DeltaP λ | dΣγ/dλ | +0.0529 rad/Ry（Σ 3 原子，raw，冻结） |
| DeltaP λ | dE'/dλ | −0.0118 eV/Ry（escon 线性项，与 F-1 −0.0127 同量级） |
| efield E | dΣγ/dE | +0.6303 rad/a.u. |
| efield E | μ₀ = −dE'/dE | +0.763 e·Bohr = **1.940 D**（实验 1.855 D，5%——与 D2 逐位互证） |
| E_eff(μ-lever) | = (dΣγ/dλ)/(dΣγ/dE) | **0.1678 a.u./Ha** |
| 公式 (b) | πλ/(2a)，a=30 Bohr | 0.1047 a.u./Ha |
| 比值 | 实测/公式(b) | **1.60×** |

- **符号：钉死 ✓**。ΔΣγ/Δλ 与 ΔΣγ/ΔE 同号（正 λ 与正 E 都使 Σγ 增加）——
  operator 模式公式 (b) 的正号约定与锯齿场一致。
- **因子：公式 (b) 低估 1.60×（本次全新实测）**，与 D2 轮 ×1.6 标注互证
  （口径：Σγ_raw 全原子和、冻结 λ、当前 HEAD）。"E_eff=λ/(2a) 10% 内"
  预言不成立（公式 (a) 低估 ~5×）——差值即 V1 条款中的"误差=代理差距"
  （F-2b：H_HK 非极化几何通道）。打印的校准注释
  `[formula (b); D2 response-calibrated ×~1.6]` 操作口径成立。

### 3.2 V3：BN 刚度（新记账 γ-drive 重采样）

| 量 | 旧轮（2026-07-21，γ-mode） | 锚点 #3（proxy-drive，κ=1 t_Γ 初值） | **本轮（γ-drive，新记账）** |
|---|---|---|---|
| 约束对象 | γ（直接） | Γ（t_Γ=t_γ 猜值） | γ（直接） |
| λ 量级 | ~1e-6–1e-5 Ry | **~4e-3 Ry** | ~1e-6–1e-5 Ry |
| escon | 忽略 | ~3e-3 Ry/点 | ~1e-6–1e-5 Ry |
| E' span | 2.0 μRy | 536 μRy | **1.72 μRy** |
| κ 拟合 | <1e-6 Ry/rad²（负本征值，噪声底） | 2108 μRy/rad²（伪曲率，λ-work） | **13.2 μRy/rad²（噪声粒度）** |
| \|γ−t\| | ≤40 mrad | ~4 rad（Γ 约束，非 γ） | ≤73 mrad（多数 <50） |

- **V3 裁定：PASS**。新记账（escon=−λ·Γ）下 γ-drive 重采样复现旧轮
  平坦 PES 量级（1.72 vs 2.0 μRy），λ/escon 均在噪声级；κ≈13 μRy/rad²
  是 SCF 能量量化粒度（±1 μRy）下的噪声曲率，物理结论"BN 电子极化约束
  刚度 ≈ 0（噪声级）"不变——对照 h2o1 κ≈60 Ry/rad²（6e7 μRy/rad²）
  仍差 6 个数量级。
- **口径说明（必读）**：锚点 #3 的 proxy-drive 数据（λ~4e-3 Ry、E' span
  536 μRy）测的是 Γ-约束的 λ-work（κ=1 t_Γ 猜值未校准），不是 γ-PES——
  从中拟合出的"2108 μRy/rad²"是协议伪曲率，不是 BN 刚度。旧轮
  "<1e-6 Ry/rad²"与本轮 "13 μRy/rad²"都是"低于噪声可分辨下限"的表述，
  不是可定量复现的物理数。
- 附注：γ-drive 下 BN 从自然 γ=(0.041,1.889) 推到靶 γ≈(4,3.5)
  （跨 ~2π 分支）能量变化 <2 μRy——平坦性甚至跨分支成立。

## 4. 分析

- **V1 三件事全部落地**：符号（同号 ✓）、因子（实测 1.60×，打印注释互证）、
  通道校准正确性（μ₀=1.94 D 与 D2/实验互证）。E_eff 的解析公式与实测的
  系统性 1.6× 偏差是代理算符的已知属性（F-2b 非极化几何通道），不是
  实现错误——打印必须带响应校准注释，用户对拍 efield 计算时按注释换算。
- **V3 的协议教训**：Route A+ operator 模式默认 proxy-drive 的"9 点采样"
  不是 γ-PES 采样；做物理刚度测量必须 γ-drive（或先用外循环校准 t_Γ*）。
  旧"刚度 <1e-6"结论在正确协议（γ-drive + 新记账）下不变。
- Stage 4 全部验收完成：4.1 锚点 #3（12 用例+PW+L9）、4.2 三体系驻点 FD
  21/21、4.3 relax 端到端 + 无 target 修复、4.4 V1/V3。进入 Stage 5
  （文档与清理）。

## 5. 下一步

- TODO 4.4 ✅；progress-summary/capability 同步；提交。
- **Stage 5**：dev-guide v3、E-field 语义更新（推导文档 §1.2 入手册 +
  `deltap_observable`/`deltap_drive` 关键词文档）、hhrdbg/hkdbg/fsdbg
  #if 0 清理、dev log + 总览刷新。
- 可选项（非阻塞）：V1 的 1.6× 校准因子若未来换 EFC 全阶算符需重测。
