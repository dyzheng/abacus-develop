# DeltaP 当前算法公式重推导 + 代码评审 + 风险提示（2026-08-12）

> 评审对象：HEAD `9378f1526` + 工作区未提交 T-6'（Ô_w）实现（8 文件 +312 行）。
> 本文三部分：§1 公式重推导（当前版本的完整公式链）、§2 代码评审结论、
> §3 风险提示（按严重度排序，R1 为阻塞级）。

---

## 1. 当前算法公式（严格推导）

### 1.1 测量链（不变）

k-string（k_j, j=0..nppstr−1，末槽为 PBC 包裹副本）：

```
M_j = C_L†·S_dk(k_j,k_{j+1})·C_R          （S_dk 相位 exp(2πi(dk·R−dk·τ_bra))，τ 分数）
W   = Π_j M_j；  θ_n = arg(eig W)（Hungarian 跨迭代匹配 + 连续性锚定）
w_In = Σ_{lm∈I} |D_{I,lm,n}|²，  D_I = ⟨SMO_I|ψ_n⟩
γ_I = Σ_n w_In·θ_n      （物理观测量；target-aware 报告值不可作判据）
```

### 1.2 约束哈密顿量（两种 operator 模式）

**proxy 模式（历史 Route A+）**：
```
H_c = H_HR + H_HK
H_HR = Σ_I λ_I·τ_α(I)·P̂_I,  P̂_I,μν = Σ_{lm∈I} c_μ(Ilm)c_ν(Ilm)
H_HK(k_L) = sym[(i/2)·S_dk·C_R·W_eff·C_L†],  W_eff[n] = Σ_I λ_I·w_In
```

**ow 模式（T-6'，工作区）**：
```
H_c = H_ow + H_HK        （H_HR 由 contributeHR 门控关闭）
H_ow(k_L) = sym[ Σ_n θ_n·(A·C_L)_n·(C_L†)_n ],  A = Σ_I λ_I·P̂_I
  即 Ô_w|ψ_n⟩ = θ_n·P̂_I|ψ_n⟩ 的对称化最小扩展（Route A++ §1.3 的精确权重通道算符）
  θ_n：compute_gamma_scf 捕获（gdir 首条 string），D2 跳变冻结（|Δθ|>π/2 保持前值）
```

### 1.3 记账观测量与能量恒等式

```
Γ_I = [proxy: Γ_I^HR = τ_I·Σ_k w_k Σ_n f_n w_In(k)] 或 [ow: Γ_I^w] + Γ_I^HK

Γ_I^HK = −0.5·Im[ Σ_j Σ_p f_p Σ_{p'} w_{I,p'}·T_{pp'}·Π_{p'p} ]
         （全 Gram 迹；T = C_L†S_dk C_R，Π = C_L†C_L——非正交基必须全迹）
Γ_I^w  = Σ_links Σ_m f_m·Re[ Σ_n θ_n·(C†P̂_I C)_{mn}·Π_{nm} ]

escon = −Σ_I λ_I·Γ_I  ⇒  E' = E_tot + escon ≡ E_KS(ψ*)   【恒等式】
```

验证锚：Γ_I^w 恰为 Tr[ρ·H_ow] 的逐原子拆分（E_ow = Σ_I λ_I Γ_I^w 逐行成立）；
Γ_I^HK 同（T2 判决：E'(λ) 斜率 224 → −0.013 eV/Ry）。

### 1.4 λ 驱动与力

```
残差：proxy 驱动 → Γ − t_Γ（t_Γ 由外层 secant 校准使 γ→t_γ）
      gamma 驱动 → γ_report − t_γ（T-4'；γ(λ) 映射弱/非单调区失效，T3' 实测）
更新：同步单步 GD / 内循环冻密度 BFGS（per-atom Jacobian 未实现）

F = F_std + Pulay(λÔ)：
  A1 = −Σρ·λ_I τ_α·∂(c_μc_ν)/∂R_J        （H_HR 基组导数；ow 模式门控关闭）
  A2 = −λ_J·⟨P̂_J⟩·(L⁻¹)_{αβ}/lat0        （H_HR τ 导数；ow 模式门控关闭）
  B  = −∂E_HK/∂R（全迹，冻结 C/W，含相位链）  （两种模式都在）
  F_ow = −Tr[ρ·∂(Σ_n θ_n·A_n)/∂R]|_{θ,C 冻结}   【ow 模式需要——当前未实现，见 R1】

路径泄漏：−λ·(∂Γ/∂λ)·dλ*/dR
  Γ-hold 路径：T3 实测 0.0138 eV/Å（闭合率 1.07）
  γ-hold 路径：T3' 实测 0.502 eV/Å（λ* 大 ~14×，O(λ)² 放大）——生产口径
```

### 1.5 应力 / E_eff

```
σ = −(1/Ω)dE'/dε：标准模块 + S1（H_HR 对形式，ow 模式同被门控）；
    H_HK/H_ow 应力未实现；∂τ/∂ε=0（分数坐标约定）
E_eff = λ/(2·a_α) a.u.（算符斜坡定义；旧 π 公式退役；符号/因子 V1 待定）
```

---

## 2. 代码评审结论

### 已提交部分（HEAD）：维持此前评审结论，无新问题

Route A+ 机器（Γ 计算、escon 切换、双驱动、连续性锚、D_I/S_k MPI 修复、
锚点 #2）均经判决实验验证。`p_hat_accum` 为 per-alpha 隔离累加（:383 在
alpha 循环内声明），多方向网格无三倍计数问题（本次专项核查）。

### 工作区 T-6'（Ô_w）：结构设计方向正确，但有 1 个阻塞级缺口和 3 个高/中风险

做对了的：
- H_ow 构造（M_ow = A_C·diag(θ)·C_L† 后 sym）与 Ô_w|ψ_n⟩=θ_n·P̂_I|ψ_n⟩
  的最小厄米扩展一致；A_C = Σ_I λ_I·S_k·D_I 正确实现 Σ_I λ_I·P̂_I·C；
- Γ_I^w 全 Gram 迹与 H_ow 的 sym 构造逐行一致（E_ow = Σλ_IΓ_I^w 成立）；
- H_HR/A1/A2 在 ow 模式的门控成对关闭（算符与其 Pulay 力同步退），
  legacy/proxy 路径零回归设计到位；
- D2 跳变冻结有了雏形。

发现的问题 → 见 §3 R1/R2/R3/R4。

---

## 3. 风险提示（按严重度）

### R1【阻塞级】ow 模式的几何力 F_ow 完全缺失——且代码注释谎称它存在

- `deltap_force_stress.hpp` 的 ow 门控注释写道：Ô_w 的几何力"computed
  separately by the esolver (DeltaP::compute_hk_force, 'ow' branch)"。
  **该分支不存在**（compute_hk_force 无任何 ow 代码，esolver 侧也无）。
- 后果：ow 模式下 H 含 H_ow 但其 Pulay 力整体缺失，ow 模式 FD 必然按
  Tr[ρ·∂H_ow/∂R] ~ **主导约束力量级**失败（这不是小修正项，是 ow 模式的 A2）。
- 处置（T-6' 完成前必须做）：
  1. 实现 F_ow：冻结 θ/C 下，对 A_C = Σ_Iλ_I·S_k·D_I 的 SMO 系数求导
     （snap cal_deri=1 现成），公式 F_J ⊃ −λ_I Σ_mn f_m θ_n
     ∂[(C†P̂_I C)_{mn}Π_{nm}]/∂R_J|_{θ,C} —— 与 A1 同族的双中心导数；
  2. 修正注释（在 F_ow 落地前不得写"computed separately"）；
  3. ow 模式验收必须包含总 E' FD（冻结 θ 协议），预言残差 = F_ow 量级，
     实现后应收敛到 O(λ) 窗口。

### R2【高】H_ow 只施加在 string-0 的 k_L 上——多 string 网格算符不完整

- H_ow 目前骑在 compute_hk_correction 的 link 循环里（该循环只跑
  k_index_[0]，首条 string）。2×2×2（BN，gdir=3）有 4 条 string，
  **只有 2/8 个 k 点拿到 H_ow**；Γ_I^w 同步只覆盖 1/4 BZ——
  escon 恒等式在"同一错误覆盖"下仍闭合（T2 测不出），但物理算符不完整。
- Ô_w 是 **k 局域算符**（Route A++ §2.1 明确"对 ψ_n(k) 施加"），
  根本不需要 string 机制。处置：H_ow 构造移出 link 循环，
  **按全部 nks 逐 k 独立构建**（θ 由该 k 所属 string 的 Wilson 环给出）——
  顺带绕开 nrow==ncol 方阵限制（k 局域不需要 S_dk 全矩阵），
  这同时是 R10 的解脱通道。
- h2o1（1×1×2，单 string）不受影响，所以现有判决数据不失效——
  但 BN 类多 string 体系在 ow 模式下结果不可信。

### R3【高】θ_n 只来自首条 string（istring==0 捕获）

同一限制的另一半：多 string 网格中非 string-0 的 k 点没有 θ_n。
与 R2 同一修复（按 k 所属 string 捕获 θ），否则 ow 模式只能声明
"单 string 网格可用"。

### R4【中】D2 跳变冻结无恢复路径，且与 γ 驱动存在失配窗口

- 当前逻辑：跳变 → θ 冻结于前值，**且 ow_theta_prev_ 不再更新**——
  若系统合法地演化过跳变点，算符永久冻结在旧 θ（无 hysteresis 恢复）。
  建议：冻结计数（N 步后接受新值）或与连续性锚的参考同步重置。
- 冻结期间：H_ow 与 Γ^w 用冻结 θ（自洽 ✓），但 γ_report 继续演化——
  gamma 驱动下 λ 残差响应的是"跳变后的 γ"，算符钉的是"跳变前的 θ"，
  驱动-算符失配。V-H8 必须覆盖该场景（构造一次分支跳变，检查驱动行为）。
- 首轮无 prev 时直接采用裸 θ：若首轮测量落在坏分支上，算符从错误的
  θ 起步——首轮应与连续性锚初始化（ref_gamma_）一致。

### R5【中】ow 模式的 T2 类恒等式验证未做

proxy 模式的 E'(λ) 平直性是 T2 判决过的；**ow 模式换记账观测量后
（Γ^w+Γ^HK），同一验证必须重跑**（同一 λ 扫描协议，预言斜率同样
≲1 eV/Ry）。这是 ow 模式所有后续实验的地基，成本 ~10 分钟。

### R6【中】Ô_w 的应力路径未定义

cal_force_stress 的 ow 门控把 stress 也清零了——ow 模式下 DeltaP 应力
= 0 参与汇总。若用户 ow + 变胞，应力静默缺失（无 WARNING）。
建议在 F_ow 实现前，ow + cal_stress 时 WARNING_QUIT。

### R7【低】首轮/参考一致性细节

- ow_theta_.size() >= nocc_use 的守卫假设 string-0 的 θ 维度覆盖占据带；
  nbands < nocc 体系（金属小 nbands）会静默跳过 H_ow（只有 WARNING 打印）。
- `deltap_operator_mode` 的 INPUT 描述已写入 V-H3' 判据（dγ/dλ≥3 rad/Ry）——
  好；但该判据同时依赖 dΓ/dλ 记账（Γ^w 的响应），V-H3 执行时两者都要测。

### R8【低】Γ^w 取 .real() 的舍去

per-atom 拆分 e_w_I 取实部（H_ow 是 sym 的，总量实；逐原子虚部应为 0）。
与 HK 的 −0.5·Im 约定不同但各自自洽。建议加一次性断言
|Im(e_w_I)| < 1e-12，防止索引错位被 .real() 吞掉（R2 类错误的早期信号）。

### R9【现状重申，非新风险】

- hk_correction/hk_force/H_ow 全部串行-only（nrow==ncol 守卫）——
  ow 模式的多 rank 同样不可用；R2 的修复（k 局域化）是唯一的结构性出路。
- PW 仍 gamma 旧记账（L1.1 待办）；smoothness 单测 4/8 参考过期（待办）。

---

## 4. 对 T-6' 的处置建议（顺序）

1. **先补 R5**（ow 模式 T2 类 E'(λ) 平直性，~10 分钟）——验证记账恒等式
   在 ow 下成立，这是继续的地基；
2. **R1 F_ow 实现**（阻塞 ow 的任何力实验）；
3. **R2/R3 k 局域化重构**（把 H_ow/θ 捕获移出 string-0 link 循环）——
   一次改动同时解除多 string 完整性和方阵限制；
4. **R4 冻结恢复路径 + 首轮锚定**；
5. 然后才进 V-H8（SCF 稳定性）/ V-H3'（dγ/dλ ≥ 3 rad/Ry 判决）。

在 R1+R2 修复前，**ow 模式的任何数值结果都不应作为证据引用**
（当前仅能用于编译与机制冒烟）。
