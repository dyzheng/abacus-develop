# 2026-08-04 DeltaP T2 判定：Γ^HK 记账与 H_c 不一致 → T·Π 全迹修复（Route A+ 第一个硬信号）

> 分支 `feat/deltap`，工作区未提交。触发：TODO Stage 2 T2（∂E'/∂λ 重测）。
> 首轮实测 FAIL（斜率 −13.3 eV/Ry）→ 按 TODO 偏离动作"回 1.1"定位并修复
> → 复测 PASS（−0.013 eV/Ry）。T1 顺带判定（构造性恒等式）。

## 1. 测试计划（可证伪）

1. **T2 首测**：h2o1/base λ∈{−0.01,−0.001,0,+0.001,+0.01} Ry 扫描，测
   E'(λ)=FINAL_ETOT_IS 斜率。预言：≲1 eV/Ry（O(λ)）；偏离 → 停，回 1.1。
2. **归因**（若 FAIL）：escon 记账 vs 实际施加的 H_c 耦合
   （dE_band/dλ = Σ wg·⟨ψ|∂H/∂λ|ψ⟩ 为耦合的严格 HF 口径）逐项核对 HR/HK。
3. **修复**：使 Γ 记账 = 实际算符期望（Tr[ρ·H_c]/λ 的逐原子拆分）。
4. **T2 复测**：同批扫描斜率 ≲1 eV/Ry 且 E'(λ) 满足变分下界 E' ≥ E_KS(ρ₀)。
5. 回归：单测 11/11、6/6。

## 2. 测试设置

- 二进制：`build/abacus_basic_para`（MPI+Release，`CCACHE_DISABLE=1 cmake --build`）。
- 用例：`tests/deltap_fd_force/h2o1/base`（ecutwfc=100/ecutrho=400/scf_thr=1e-8，
  `1 1 2` k 网格，gdir=3，1-rank，`OMP_NUM_THREADS=1`，全部串行单任务执行）。
- 参数模板：`deltap_lambda_step 0.0`、`deltap_lambda_mixing 1.0`、
  `deltap_lambda_init <λ>`、`deltap_observable operator`、target.dat 保持 t_γ。
- 运行目录：`/tmp/deltap_t2_scan`（首测）、`/tmp/t2_scan_fixed`（复测，同 INPUT）。
- 伪势/轨道：`/root/pporb/apns-*`。

## 3. 结果

### 3.1 T2 首测（修复前，rc=0 全部）

| λ (Ry) | FINAL_ETOT_IS (eV) | Γ 末态 | escon (Ry) |
|---|---|---|---|
| −0.01 | −481.568902 | (7.196, 2.219, 2.219) | +0.116335 |
| −0.001 | −481.684490 | (7.195, 2.218, 2.218) | +0.011632 |
| 0 | −481.697770 | (7.195, 2.218, 2.218) | 0 |
| +0.001 | −481.711131 | (7.195, 2.218, 2.218) | −0.011632 |
| +0.01 | −481.834779 | (7.195, 2.218, 2.218) | −0.116303 |

**斜率 = (−481.8348+481.5689)/0.02 = −13.3 eV/Ry（O(1)，FAIL）**。
同时：dE_band/dλ = 142.7 eV/Ry（Σ wg ε 的 HF 口径耦合）、Γ_escon = 158.3 eV/Ry
（ΣΓ=11.63×13.6057）、dE_Harris/dλ = 145.0 eV/Ry。**Γ_escon − dE_band/dλ = 15.6**
（~10% 高估），E'(λ) 斜率全部来自该记账差。

### 3.2 归因探针（修复前，λ=+0.01 收敛态）

在 `compute_hk_correction` 内联探针，同一 ψ 下两种口径：

```
E_HK_conv   = 0.05739 Ry  （E_HK-split 对角约定：−0.5·Im[Σ f_p w_eff T_pp]）
E_HK_actual = 0.04702 Ry  （实际施加算符期望：Σ f_p·Re[c_p† H_sym c_p]）
```

Γ^HR 侧由 T0 验证 12 位一致（per-k ⟨P̂⟩ == 实空间 Tr[DMR·pre_hr]，80.25 eV/Ry）。
故偏差 100% 归因于 **Γ^HK 的对角约定**：非正交 LCAO 基下
c_p†·H_sym·c_p = −0.5·Im[Σ_{p'} w_eff[p']·T_{pp'}·Π_{p'p}]（Π=C_L†C_L 占据块 Gram 矩阵），
对角约定假设 Π=I，把耦合高估 ~18%。

### 3.3 修复

`deltap_wannier.cpp`（`compute_hk_correction` + `compute_gamma_op_hk`）：
Γ_I^HK 从 `−0.5·Im[Σ_j Σ_p f_p·w_IJ[p][I]·T_pp]` 改为按原子拆分的实际算符期望

```
Γ_I^HK = −0.5·Σ_j Σ_p wg(ik_L,p)·Im[ Σ_{p'} w_IJ[p'][I]·T_{pp'}·Π_{p'p} ]
T_{pp'} = (C_L† S_dk C_R)_{pp'},  Π_{p'p} = (C_L† C_L)_{p'p}
```

（H_sym 对 λ 线性：w_eff[n]=Σ_I λ_I·w_IJ[n][I]，逐原子拆分良定义。）
`compute_hk_force`（力路径 E_HK/F_HK）未动（Route A+ 力代表 A1/A2/B 保持，T3 验证）。

### 3.4 T2 复测（修复后，rc=0 全部，串行单任务）

| λ (Ry) | FINAL_ETOT_IS (eV) | Γ 末态 | escon (Ry) |
|---|---|---|---|
| −0.01 | −481.696217 | (6.723, 1.987, 1.987) | +0.106977 |
| −0.001 | −481.697756 | (6.700, 1.978, 1.978) | +0.010657 |
| 0 | −481.697770 | (6.698, 1.977, 1.977) | 0 |
| +0.001 | −481.697756 | (6.696, 1.977, 1.977) | −0.010649 |
| +0.01 | −481.696471 | (6.676, 1.969, 1.969) | −0.106138 |

- **斜率（端点） = −0.013 eV/Ry**（判据 ≲1 的 ~1/80；修复前 −13.3 的 1/1000）。
- ±0.001 两点 E' 逐位一致（−481.6977557 vs −481.6977557，8 位）。
- E'(±0.01) − E'(0) = +1.55/+1.30 meV（~λ² 抛物，量级符合设计预言 ~3e-4 eV）；
  **E' ≥ E_KS(ρ₀) 变分下界恢复**（修复前 −0.01 侧违反）。
- E_Harris/E_band 各分量与修复前逐位相同（修复只改可观测量/escon，不动 H_c）。

### 3.5 回归

- `MODULE_ESOLVER_deltap_common_test`：11/11 PASS；`esolver_dp_test`：6/6 PASS。
- gamma 模式路径未触碰（`compute_gamma_op_hk`/`compute_hk_correction` 为
  operator 模式专用；gamma 模式 escon/残差用 γ）。hk_correction 施加的 H_HK 不变。

## 4. 分析

1. **T2 硬信号 PASS**：E'(λ) 从 224（旧记账）→ −13.3（Route A+ 首轮，Γ 记账与 H_c
   不一致）→ **−0.013 eV/Ry**（修复后）。Route A+ 的核心恒等式
   E' = E_Harris − ⟨H_c⟩ ≡ E_KS(ψ*) 现在是**精确构造**：escon = −ΣλΓ，
   Γ^HR 由 T0 验证、Γ^HK 改为实际算符期望 → ΣλΓ ≡ ⟨H_c⟩（无近似）。
2. **根因**：Γ^HK 的 E_HK-split 对角约定（沿用 compute_hk_force 的 E_HK 拆分）与
   "算符期望"语义错位。非正交基里 H_HK 的期望必须用全 T·Π 迹；对角 T_pp 约定
   在 h2o1 高估 18%（耦合 142.7 vs 记账 158.3 eV/Ry）。
3. **T1 判定**：E' ≡ E_KS(ψ*) 由构造精确成立（代数恒等，<1e-8 机器精度）；
   独立数值证据 = T0（12 位）+ 探针（E_HK_actual == escon 的 HK 部分，迭代噪声级
   ~1e-5 eV）。
4. **Q1 数据（T4 前置）**：dΓ/dλ ≈ −4.2 Ry/Ry（ΣΓ 10.697→10.614），
   dγ/dλ ≈ −0.3 rad/Ry。Γ→0 需 λ*≈2.5 Ry（大）；γ 对 t_Γ 斜率 ≈ 0.07 rad/单位
   → 外循环 κ=1 首轮不足，T4 需实测 Δγ/Δt_Γ 更新 κ（Q2 翻号逻辑已就位）。
5. **T3 冻结协议（Q3）**：`deltap_proxy_target_file`（init 加载冻结 t_Γ*，MPI bcast）
   + `deltap_secant off`（`secant_update_proxy` 短路）已接线 → disp± 只重收敛 λ，
   无 t_Γ(R) 漂移污染。

## 5. 下一步

1. T3（判决点）：h2o1 O1-z 三几何。协议 = base 单点 secant 校准 t_Γ*（γ→t_γ）
   → 写 `proxy_target.dat` → disp± 用 `deltap_secant off` + 冻结文件重收敛 λ
   （驻点判据 |Γ−t_Γ*|<1e-3）→ 三几何 E' 中心差 FD vs 解析力。预言残差
   84.8 → ≤0.02 eV/Å。
2. T4（外循环）：t_γ=0.9γ_natural，secant ≤5 步 |γ−t_γ|<1e-2；Q2 翻号逻辑验证。
3. Stage 1+2 改动保持未提交，T3 通过后统一评审提交。
