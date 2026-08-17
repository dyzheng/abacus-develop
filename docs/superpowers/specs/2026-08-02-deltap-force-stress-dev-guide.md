# DeltaP Force/Stress 开发指导文档（v3，Route A+ 记账 + Stage 4 验收版）

> 2026-08-17 · v3 修订（Stage 5 文档轮）：Status 矩阵全面刷新（Stage 1–4 后
> 力/应力/记账全部实现并验证）；新增三种记账推导（§2.2，Route A+
> escon=−λΓ 恒等式 + 非正交全迹教训）；dspin 恒等式落点（§2.3）；Route A+
> 双路径分层（§2.5：T3 PASS=数学自洽 / T3' FAIL=物理不可用）；FD 协议更新为
> stationary4 协议（§4）；公式集/check-list 对齐当前验收面（§7/§8）；新增
> Γ-path relax 生产用法（§10）。v2（2026-08-02）内容已由 git 历史归档，
> 本文档为现行权威。配套：算法完整推导
> `2026-08-05-deltap-algorithm-derivation-route-a-plus.md`（§8 已被
> `2026-08-13-deltap-capability-boundaries.md` 取代）。

---

## 1. 现状实现矩阵（Stage 4 后）

| 基组 | 项 | 状态 | 代码位置 |
|------|----|------|----------|
| LCAO | H_HR 代理算符（τ_α·P̂，A1 Pulay 力） | ✅ 实现并验证（T7-a 修 nlm 多 ζ 越界后 FD 闭合） | `deltap_force_stress.hpp:216 cal_force_IJR` |
| LCAO | A2 τ 显式导数力（Born 电荷型） | ✅ 实现，4ppm 闭合（T7-c） | `cal_force_IJR` |
| LCAO | H_HK 力（B 项，**全迹版**） | ✅ 实现并双闭合（单原子 + 均匀平移，B-7 范式） | `compute_hk_force`（`deltap_wannier.cpp`） |
| LCAO | H_HK MPI（pzgemm + A' Allreduce） | ✅ 实现（F-6，串行 A/B 逐位一致；h2o_asym 非方本地块解锁，3.2） | `compute_hk_correction`（`deltap_wannier.cpp`） |
| LCAO | H_HK 应力（σ^HK） | ✅ 实现（F-8，V-H7 内部 FD 6.8% / 内部 Maxwell 7.0% 闭合） | `compute_hk_force` 应变导数核 |
| LCAO | escon 记账（Route A+：escon=−λΓ） | ✅ 实现并逐点验证（T2/F-1/F-7b，E'≡E_KS(ψ\*)） | `deltap_common.h` / `deltap_scf.cpp`（每 SCF 迭代刷新，F-7b） |
| LCAO | ow 模式（Ô_w=θ_n·P̂） | ⚠️ 实验性：SCF/记账可用，**F_ow 力未实现**，应力 WARNING_QUIT | `deltap_lcao.cpp` / `deltap_force_stress.hpp` |
| PW | Γ^PW 记账（L1.1，escon=−λΓ^PW） | ✅ 实现（F-7，E'(λ) 线性项 0.00014 Ry/Ry @inner_thr=1e-6） | `deltap_pw.cpp` |
| PW | onsite 投影力 | ✅ 复用 dspin 核 | `forces_onsite.cpp` |
| PW | onsite 应力 | ✅ 复用 dspin 核（f48e59723） | `stress_onsite.cpp` |
| 接线 | LCAO 力/应力入口 | ✅ `FORCE_STRESS.cpp`（F_HK 静态存储，T7-c） | `FORCE_STRESS.cpp:433-453,529-` |
| 单测 | math/gauge/l1/smoothness | ✅ 11/11（smoothness 4/8 为既有 L9 注记） | `tests/` |

**结论**：力/应力链路已全部实现并经过双闭合验证（单原子 + 均匀平移 +
stationary4 三体系 21/21 腿）；relax 端到端在 Γ-path（自然靶点）下可用。
历史遗留（v2 的"未实现/必崩"）全部关闭，仅 ow 力与场模式力侧为已知缺口。

---

## 2. 约束泛函、能量记账与力分解

### 2.1 总能量定义（Route A+）

```
E' = E_tot + escon,   escon = − Σ_I λ_I · Γ_I          (Ry)
```

（`deltap_common.h:155-166`；F-7b 起 escon 每 SCF 迭代在末次测量态重估，
`deltap_scf.h:set_escon`。）

**核心恒等式（T2/F-1 逐点验证）**：

```
escon ≡ −⟨H_c⟩  ⇒  E' ≡ E_KS(ψ*)        【记账恒等式，无近似】
```

E' 等于约束态波函数的纯 KS 能量——物理干净，可跨构型/约束态比较，
并使力框架恢复 dspin 结构（§2.3）。

### 2.2 三种记账（算符 — 观测量 — 恒等式）

| 路径 | H_c 算符 | 观测量 Γ_I | 恒等式 | 状态 |
|------|----------|-----------|--------|------|
| **proxy**（默认） | H_HR = Σ_I λ_I τ_α(I) P̂_I（SMO 投影，几何代理） | Γ_I^HR = τ_α(I)·⟨P̂_I⟩ + Γ_I^HK | escon=−λΓ，E'=E_KS(ψ\*) ✓ | ✅ 主力路径（T2/T3/4.x 全用它） |
| **hk**（F-2b 场模式） | H_c = H_HK only（无 H_HR；离散 Berry 联络场算符） | Γ_I^HK | escon=−λΓ^HK 保持恒等式 ✓ | ✅ 能量/应力闭合；**力侧非真实场力**（Maxwell 失配 17.7×，§6） |
| **ow**（EFC L3.1，实验性） | H_ow = Σ_n θ_n·(P̂_λ|ψ_n⟩)⟨ψ_n|（逐带 Wilson 相位权重） | γ 权重通道 | 同上 | ⚠️ SCF/记账可用，力未实现 |
| **PW** | H_c,PW = Σ_I λ_I P̂_I^onsite | Γ^PW = ⟨P̂^onsite⟩ | escon=−λΓ^PW ✓（L1.1，F-7） | ✅ |

**非正交全迹教训（T2 轮，通用纪律 #11）**：LCAO 基 C†SC=I 但 C†C≠I，
算符期望必须走 **T·Π 全迹**（T=C_L†·S_dk·C_R，Π=C_L†·C_L Gram 阵），
而非对角 T_pp 迹——对角约定高估 ~18%（0.0574 vs 0.0470 Ry），escon 与
真实耦合错开，E'(λ) 斜率残留 −13.3 eV/Ry；全迹修正后 −0.013 eV/Ry
（T2 判决，三数量级压平、λ² 抛物、变分下界恢复）。

### 2.3 dspin 恒等式与 DeltaP 落点

dspin（磁矩约束）Pulay-only 力**精确**的三个条件：

1. escon 的观测量 = H_c 算符的 expectation（M_I = Tr[ρ·Ô_I]）；
2. 因此 E_band 中 TrρλÔ 与 escon 的 −λM **恒等相消**，E' = E_KS(ψ\*)；
3. 约束激活（M≡m ⇒ dM/dR=0）时，响应项 −λ(δM/δψ)(dψ/dR) 恰等于
   +Trρ·∂(λÔ)/∂R（Pulay 项）。

DeltaP 逐条核对（v2 表，结论不变）：

| dspin 条件 | DeltaP（Route A+） | 后果 |
|------------|-------------------|------|
| ① 观测量=算符 expectation | **满足**（operator 模式：Γ_I 定义为 H_c 的 expectation，proxy 用 τ·P̂ 几何代理，hk/ow 用严格权重算符） | E_band 与 escon 恒等相消 |
| ② 能量相消 | **满足**（escon=−λΓ，T2/F-1 逐点验证） | E' = E_KS(ψ\*) |
| ③ 约束激活 | **满足**（驻点 |Γ−t_Γ*|<1e-3 时 dΓ/dR≈0） | FD 协议成立（§4） |

⇒ 驻点力 = F_std + A1 + A2 + B，残差 O(λ)（§4.2），与 dspin 同构。
v2 时代"①②不满足（三个不同对象）"的判断已被 Route A+ 记账修复推翻。

### 2.4 力分解（当前实现全貌）

```
F_J = F_J^KS(ψ*)                                        [已有]
      − Σρ · ∂(λ_I τ_α P̂_I)/∂R_J  = A1 + A2              [✅ 已实现]
      − ∂E_HK/∂R_J（全迹版，含 Π 权重、p≠p' 离对角项） = B  [✅ 已实现]
      + O(λ) 驻点泄漏：−λ(∂Γ/∂λ)·dλ*/dR                   [FD 定量，不解析实现]
```

- **A1**（SMO 系数导数 Pulay）：−Σρ·λ_I·τ_α·Σ_{lm} ∂(c_μc_ν)/∂R_J。
  符号模式与 dspin 一致（force1 += tmp / force2 −= tmp）。
- **A2**（τ 显式导数，HF 项）：F_J,β ⊃ −λ_J·⟨P̂_J⟩·∂τ_α/∂R_{J,β}
  = −λ_J·⟨P̂_J⟩·(L⁻¹)_{αβ}（lat0 取向，T7-c 4ppm 闭合）。
- **B**（H_HK 力）：全迹约定下 ∂E_HK/∂R 的解析导数，含 Π_{p'p} 非对角
  权重（F_HK 全迹对齐，T3 轮）；单原子 + 均匀平移双闭合。
- **C**（响应残差）：λ·dΓ/dR 家族，O(λ)，FD 定量；T3 闭合率 1.07，
  Stage 4.2 三体系最大残差 0.00658 eV/Å（判据 1.95× 裕度）。

### 2.5 Route A+ 双路径分层（T3 PASS / T3' FAIL 的并存表述）

**两条约束路径，两条不同判据——必须分开读：**

1. **Γ-约束面（proxy 驱动，T3 验证）**：力 = 所实现能量 E'(R,λ*(R))
   在 Γ-约束面上的精确梯度。T3：残差 0.0138 eV/Å（判据 0.0129 的
   1.07×，O(λ) 响应项，闭合率 1.07）；Stage 4.2 三体系 21/21 腿 PASS。
   **T3 PASS = 数学自洽**：实现的力与记账一致。
2. **γ-hold（生产 relax 名义路径，T3' 验证）**：每离子步钉住 γ=t_γ
   的真实约束路径上，残差 **0.502 eV/Å（判据 25×）**——λ² 泄漏爆炸
   （dγ/dλ=−0.3 弱耦合 ⇒ λ* 被逼进 0.02 Ry 区）。
   **T3' FAIL = 物理不可用**：Γ-约束面 ≠ γ-约束面，0.502 恰好是两
   个面之间的力差。

**使用含义**：proxy 驱动 relax 的力对 Γ-约束面精确，但用户名义约束是
γ；生产 relax 必须走 **Γ-path**（t_Γ*=自然 Γ 冻结，§10），在自然靶点
附近 λ→0、两约束面重合、力与纯 DFT 一致。任何"严格钉 γ 到远离自然值"
的 relax 当前不可用（弱耦合是代理算符控制权限的定量边界）。

### 2.6 应力分解

```
σ_αβ = σ^KS + σ^HR(S1) + σ^HK + escon 应变项
S1  = (1/Ω) Σ_pairs F_α(pair)·R_β(cart)      [✅ cal_stress_IJR]
σ^HK = +0.5·Im(Σ kern)/Ω                     [✅ F-8，V-H7/Maxwell 闭合]
PW  = λ·P̂^onsite 应力（dspin 核复用）          [✅ f48e59723]
```

1. τ_α 是分数坐标：固定分数坐标应变约定下 ∂τ_α/∂ε=0，应力无 A2 类项；
   约束随晶格"共形拉伸"是设计选择（影响压电解释，见 §6）。
2. F-8 裁定：H_HK 的非极化几何通道主要经由**原子位移**进入（力路径
   52×），均匀应变下晶格缩放大部分是极化通道（应力侧 1.78×）——
   **17.7× 类失配是力路径特有**。
3. ow 模式应力 WARNING_QUIT（未定义）；hk 模式应力串行 only。

### 2.7 单位与符号约定

| 量 | 单位 | 出处 |
|----|------|------|
| λ_I | Ry | INPUT `deltap_lambda_init` |
| Γ_I | rad（算符期望，无需分支） | `compute_gamma_op` |
| γ_I | rad（Wilson loop，mod 2π 分支管理） | `compute_gamma` |
| escon | Ry | `compute_dp_escon` = −ΣλΓ |
| E-field 等效 | E = +πλ/(2a)（公式 (b)，**Ha/bohr**） | `deltap_scf.cpp:873` 打印 + 校准注记（实测 ×1.6，D2/V1） |
| force | Ry/Bohr（内部）；打印转 eV/Å | `forces_onsite.cpp` / `print_force` |
| stress | Ry/Bohr³（÷Ω 后） | `deltap_force_stress.hpp:201-209` |
| H_HK 符号 | +(i/2)·w_eff·SC 后 symmetrize | **不可翻转**（dev log #1） |

---

## 3. 历史 bug 台账（已关闭）

| 编号 | 内容 | 关闭轮 |
|------|------|--------|
| B-1 | nlm 布局多 ζ 越界（力路径堆损坏） | T7-a（按 (L,N) 双索引） |
| B-2 | hR=nullptr、缺 τ 因子、多 ×2、应力量纲 | 07-29 round 2 |
| B-6 | τ 单位（L 倍放大）、S_dk 相位单位 | T7-c |
| D_I | MPI 混带（本地带索引当全局） | F-6（A' 全带归约） |
| S_k | 本地列索引混带同族 | F-6 |
| H_sym | 行/列轨道语义错位（nrow==ncol 计数巧合） | F-6（pzgemm 分布式 GEMM） |
| escon 陈旧 | 首个 drho<inner_thr 迭代一次性测 Γ | F-7b（每 SCF 迭代刷新，d3a95e40e） |
| 无 target 隐式 Γ→0 | 全零 target 非空 → 隐式约束 Γ→0，λ 积累 | 4.3（`p.target.clear()`，LCAO；PW 保持文档化 γ→0 语义） |
| scan.py 嵌套 if | ecut 选择错档（仅 F-7b 受影响，审计干净） | F-7b 注记 |

---

## 4. FD 协议（现行：stationary4 驻点协议）

T3 冻结协议（Q3 评审定案）基础上扩展为三体系全矩阵（Stage 4.2）：

1. **base 几何 secant 校准 t_Γ\***（γ=t_γ，或直接取 λ=0 自然 Γ）；
2. **冻结 t_Γ\***（`deltap_secant off`，disp± 各腿同 t_Γ*，杜绝 t_Γ(R)
   漂移污染）；
3. **disp± 只重收敛 λ** 使 |Γ−t_Γ*|<1e-3（驻点判据在 Γ 上，不在 γ 上）；
4. 中心差 FD：F_FD = −[E'(R+δ)−E'(R−δ)]/2δ，δ=0.005 Bohr；
5. 判据：|F_FD − F_ana| < 0.0128555 eV/Å（5e-4 Ry/Bohr）。

脚本：`tests/deltap_fd_force/tools/run_stationary4.sh`（串行、OMP=1、
一次一任务）。Stage 4.2 结果：hf/co/h2o_asym 21/21 腿 PASS，最大残差
0.00658 eV/Å（h2o_asym atom1 x）；⟨η⟩ 关联阴性。

**泄漏预算（工作窗定义）**：残差 ∝ λ\*² 与 (dγ/dλ)⁻¹；生产验收表述
为"在 |λ\*|≤λ₀ 工作窗内残差 ≤X"，λ→0 时残差→0 线性（T3 评审）。

---

## 5. 代码 ↔ 公式映射速查

| 公式项 | 代码 | 备注 |
|--------|------|------|
| H_HR 构建 | `deltap_lcao.cpp` `cal_pre_HR` / `cal_HR_IJR` | SMO 外积 Σ_lm c_μc_ν |
| H_ow 构建 | `deltap_lcao.cpp:114-125`（operator_mode=ow/hk 门控） | H_HR 关闭 |
| λ 增量进 hR | `contributeHR`（dλ；hR 重建全量补加） | — |
| H_HK 构建 | `deltap_wannier.cpp compute_hk_correction` | pzgemm（MPI）/ 串行核 |
| H_HK 力 | `compute_hk_force` | 全迹 Π 权重 |
| H_HK 应力 | `compute_hk_force` 应变导数核 | +0.5·Im/Ω |
| escon | `deltap_common.h:157` / `deltap_scf.cpp` | −ΣλΓ，每 SCF 刷新 |
| A1/A2 力 | `deltap_force_stress.hpp cal_force_IJR` | 符号=dspin 模板 |
| S1 应力 | `cal_stress_IJR` | F·R_cart / Ω |
| 力入口 | `FORCE_STRESS.cpp:433-453` | 临时算符 + 静态 λ/F_HK 传递 |
| PW 力/应力 | `forces_onsite.cpp` / `stress_onsite.cpp` | dspin 核复用 |
| λ 更新（同步） | `deltap_scf.cpp update_lambda_gd` | 残差门 tgt 非空 |
| 内循环 | `inner_loop`（cg 标量 α / jacobi 逐分量 secant） | T-7' |
| t_Γ secant | `secant_update_proxy` | 外层/跨离子步 |
| 无 target 修复 | `esolver_ks_lcao.cpp` `p.target.clear()` | 4.3 |

---

## 6. 开放问题（截至 Stage 4）

- **O-A（场模式力侧）**：F-2b 力/极化 Maxwell 失配 17.7×（hf 52×）——
  H_HK 能量的 R 依赖走极化 + 基组几何双通道，真实场力只允许第一条。
  第一性解法 = EFC（未立项，动机唯一化为力级）；半经验 Z\* 补丁可选
  （须标注非从头算、2–3 几何验证可迁移性）。
- **O-B（ow 力）**：Ô_w 的 F_ow 未实现（EFC L3.1 前置，挂起等"大 Δγ
  应用需求"）；多 string 网格算符不完整（仅 string-0）。
- **O-C（γ-hold 路径）**：T3' 0.502 eV/Å 不可用——弱耦合是算符-观测
  量耦合的物理强度，不是 bug；约束模式已封口（proxy 驱动 + LIMITATION）。
- **O-D（压电解释带宽）**：场模式应力的定量压电解释带 ~2× 场物理杠杆
  （对比力侧 52×）；约束模式应力（内部自洽，如定 λ 变胞）不受此限。
- **O-E（C 响应族 ~7% 隙）**：L12，解析力/应力的冻结-C 近似隙 ~7%
  （T3 1.07 / V-H7 6.8% / Maxwell 7.0% 同源）；改进路径 = 线性响应 C
  导数（未立项，需求驱动）。

---

## 7. 公式集（开发时必须 follow）

```
[F1] E' = E_tot + escon,  escon = −Σ_I λ_I Γ_I                    （记账唯一口径）
[F2] H_c = Σ_I λ_I τ_α(I) P̂_I  +  H_HK        （proxy 模式；hk=仅 H_HK；ow=Σθ_n P̂）
[F3] P̂_I,μν = Σ_{lm∈I} c_μ(Ilm) c_ν(Ilm)      （SMO 投影，cal_deri=0/1）
[F4] A1: F_J ⊃ −Σ_{μν} ρ_νμ · λ_I τ_α(I) · Σ_{lm} ∂(c_μc_ν)/∂R_J
[F5] A2: F_J,β ⊃ −λ_J ⟨P̂_J⟩ (L⁻¹)_{αβ},  ⟨P̂_J⟩ = Σρ·Σ_{lm∈J} c_μc_ν
[F6] B:  F_J ⊃ −∂E_HK/∂R_J（全迹：Σ_p Σ_{p'} w_{I,p'} T_{pp'} Π_{p'p} 的 R 导数）
[F7] C:  E' 对 Γ 的共轭力 ∂E'/∂Γ_I = −λ_I；残差 ∝ λ*·(dΓ/dλ)·(dλ*/dR)（O(λ)）
[F8] S1: σ_αβ ⊃ (1/Ω) Σ_pairs F_α(pair) R_β(cart)；∂τ_α/∂ε = 0（分数坐标约定）
[F9] FD: F_FD = −[E'(R+δ) − E'(R−δ)] / 2δ,  δ = 0.005 Bohr（stationary4 协议，§4）
[F10] E-field：E_eff = +πλ/(2a) Ha/bohr（公式 (b)）；实测响应校准 ×1.6（proxy）
[F11] 恒等式：escon ≡ −⟨H_c⟩ ⇒ E' ≡ E_KS(ψ*)（与驱动信号无关，T-4' 一行定理）
[F12] 工作窗：|λ*| ≤ λ₀（残差 ≤X 的 λ 反解）；γ-hold 路径 λ 预算结构性不足（T3'）
```

---

## 8. Check-list（现行验收面）

**代码正确性**
- [x] nlm 按 (L,N) 双索引；多 ζ 无越界（T7-a，ASAN）
- [x] A1 符号 = dspin 模板；A2 含 lat0/L⁻¹ 取向（T7-c 4ppm）
- [x] B 全迹（Π_{p'p} 非对角权重）；单原子 + 均匀平移双闭合
- [x] H_HK MPI：串行 A/B 逐位一致（硬约束）；非方本地块解锁
- [x] escon 每 SCF 迭代刷新（F-7b）；E' 提取链 = etot + escon
- [x] 无 target 隐式 Γ→0 修复（4.3）

**数值验证**
- [x] T3：Γ-约束面残差 0.0138 eV/Å，闭合率 1.07（O(λ) 归因）
- [x] Stage 4.2：三体系驻点 FD 21/21 腿，最大残差 0.00658 eV/Å
- [x] F-8：σ^HK V-H7 6.8% / 内部 Maxwell 7.0%；efield 压电必测项（1.78× vs 52×）
- [x] 4.3：Γ-path relax 与纯 DFT 逐点一致（≤1.4e-4 eV / ≤1.2e-3 eV/Å）
- [ ] ow 力（F_ow）——未立项（O-B）

**回归**
- [x] 单测 11/11；MPI smoke 4/4；锚点 #3 12 用例 + PW Γ 用例逐位一致
- [x] 文档：dated 文档 + dev log 追加（AGENTS.md 纪律）

---

## 9. TODO 状态（2026-08-17）

| 阶段 | 状态 |
|------|------|
| Stage 1（T0/T1/T2/T3） | ✅ 全部 PASS（记账恒等式 T2 成立，T3 判据 1.07× 闭合） |
| Stage 2（γ-drive T3'） | ✅ FAIL 归档为"已闭合负结果"（物理不可用，数学自洽） |
| Stage 3（hk MPI / L2 应力 / L1 三件） | ✅ F-6/F-7/F-8 全部 PASS |
| Stage 4（锚点 #3 / 驻点 FD / relax / V1-V3） | ✅ 4.1–4.4 全部 PASS（2026-08-17） |
| Stage 5（文档与清理） | 本轮（dev-guide v3 / INPUT 手册 / #if 0 清理 / dev log） |

---

## 10. Γ-path relax 生产用法（Stage 4.3 验证）

**合法工作区**：靶点靠近体系自然极化（λ→0），即界面反场补偿类应用。

```
deltap_switch           1
deltap_corr             1
deltap_proxy_target_file t_gamma_star.dat   ← λ=0 自然 Γ（每原子一行）
deltap_secant           off                  ← 冻结 t_Γ* 跨离子步
deltap_observable       operator            （默认）
deltap_drive            proxy               （默认）
deltap_inner_nmax       0                    ← 同步 λ 更新（内循环冻密度对 γ 结构性失效）
deltap_lambda_step      0.01
deltap_lambda_mixing    0.1
deltap_inner_thr        1e-3
```

- 机制：每离子步 λ 在 SCF 内重收敛使 |Γ−t_Γ*|<1e-3；力 = 4.2 已验证的
  驻点力；λ 自动落在 1e-5–1e-4 Ry 合法区，escon ≤ ~7e-3 eV。
- 4.3 实测：h2o1 4 离子步收敛，与纯 DFT 基线逐点一致（能量差 ≤1.4e-4 eV、
  力差 ≤1.2e-3 eV/Å），无 deltap 特有退化；BFGS 过冲非单调 = 优化器
  行为（纯 DFT 同样出现），判据表述为"与纯 DFT 基线一致 + 收敛"。
- **禁止**：无 target 文件/STRU 靶标的 relax（4.3 前隐式 Γ→0 已修；
  LCAO 无 target = 自由跑 λ≡0）；任意大靶点（γ-hold 路径，T3' 0.502
  eV/Å 不可用）。
- 应力/变胞：约束模式应力（内部自洽，如定 λ 变胞）可用；场模式应力
  压电解释带 ~2× 杠杆（O-D）。

---

## 11. 本轮记录（v3 修订，2026-08-17）

- 依据 Stage 1–4 全部 dated 文档与 dev log 重写：Status 矩阵、记账推导
  （§2.2 三种路径 + 非正交全迹）、dspin 恒等式落点、双路径分层
  （T3/T3' 并存表述，防"proxy 驱动 relax 可用"误读）、FD 协议
  （stationary4）、公式集（F1–F12）、Γ-path relax 用法。
- 与 v2 的差异：v2 的"①②不满足、A2 未实现、B 未实现、O1–O5 开放"全部
  被后续轮次关闭；本版为现行权威，v2 见 git 历史。
