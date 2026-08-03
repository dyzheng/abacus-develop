# DeltaP Force/Stress 开发指导文档（v2，dspin 比对修正版）

> 2026-08-02 · v2 修订：依据 DeltaSpin 代码比对（`spin_constrain.cpp`、
> `cal_mw.cpp`、`dspin_force_stress.hpp`）重写力分解推导（§2.3），修正 FD 协议
> （§4 T7-b 双组方案），新增公式集（§7）与 check-list（§8）、TODO 清单（§9）。
> 依据：源码全量审查（`deltap_lcao.{h,cpp}`、`deltap_force_stress.hpp`、
> `FORCE_STRESS.cpp`、`forces_onsite.cpp`、`deltap_wannier.cpp:compute_hk_correction`、
> `deltap_common.h`、`deltap_scf.cpp` + dspin 三文件比对）。

---

## 1. 现状实现矩阵

| 基组 | 项 | 状态 | 代码位置 |
|------|----|------|----------|
| LCAO | H_HR Pulay 力（A1，SMO 投影导数） | **已实现，未验证，必崩（B-1）** | `deltap_force_stress.hpp:158-170` → `cal_force_IJR`（:216-283，符号模式逐行复制自 FD 验证模板的 dspin_force_stress.hpp:298-308） |
| LCAO | H_HR Pulay 应力（S1，F·R 对形式） | **已实现，未验证** | `cal_stress_IJR`（:285-368） |
| LCAO | ∂τ_α/∂R Hellmann-Feynman 力项（A2） | **未实现，优先级最高** | 头注释 :19-23 自声明 |
| LCAO | H_HK 力/应力项（B） | **未实现** | 头注释 :19-23 自声明 |
| LCAO | escon 一致性（C，λ dγ/dR 记账） | **未实现** | — |
| PW | onsite 投影 Pulay 力 | **已实现，未验证**（复用 `cal_force_onsite_dspin`，dspin 同款） | `forces_onsite.cpp:64-109` |
| PW | 应力 | **完全未实现** | — |
| PW | τ_α 因子 | PW 的 H_c 本身无 τ（设计差异，§2.2-c） | — |
| 接线 | LCAO 力/应力入口 | 已接线 | `FORCE_STRESS.cpp:433-453`（临时算符，hR=nullptr，`store_lambda_for_force` 静态传 λ） |
| 接线 | PW 力入口 | 已接线 | `forces.cpp:76-77,175` → `cal_force_onsite` |

**结论**：框架在、A1 符号与 dspin 模板一致、但数学未闭环、验证为零。
relax/MD 当前产出物理上不可信。

---

## 2. 约束泛函、能量记账与力分解

### 2.1 总能量定义

```
E' = E_tot + escon,   escon = − Σ_I λ_I · γ_I          (Ry)
```

（`deltap_common.h:155-166`；E_tot 的 band energy 已含 H_c 贡献。）
与 dspin 完全同构：`escon_dspin = −Σ_I λ_I·M_I`（`spin_constrain.cpp:48-50`）。

### 2.2 约束哈密顿量 H_c

**(a) H_HR（LCAO，同步 + 内循环模式都在用）**

```
H_HR,μν(R) = Σ_I λ_I · τ_α(I) · Σ_{lm∈I} c_μ(Ilm; R) · c_ν(Ilm; R)
```

- `c_μ(Ilm)`：SMO 展开系数，双中心积分 `intor_->snap`（SCF 路径 cal_deri=0，
  `deltap_lcao.cpp:205`；力路径 cal_deri=1，`deltap_force_stress.hpp:104`）。
- `τ_α(I)`：原子 I 沿 `deltap_gdir` 的**分数坐标**（`deltap_lcao.cpp:92`）。
  **这是与 dspin 的本质区别**：dspin 的 H_c = Σλ·Ô 中 λ 与 Ô 均不显式依赖 R；
  DeltaP 的 τ_α(R) 使 H_c 成为"位置算符"约束（类比电场焓 −E·P，见 §2.5）。
- λ 增量进 hR：`contributeHR` 只加 dλ = λ − λ_save（`deltap_lcao.cpp:94-118`）；
  hR 重建（新离子步）全量补加（:60-72）。

**(b) H_HK（k 空间 Berry 联络修正，沿 k-string 逐 link）**

```
w_eff(n) = Σ_I λ_I Σ_{lm} |D_{I,lm,n}|²
F        = (i/2)·w_eff(n)·S_dk·c_R,n          （deltap_wannier.cpp:1781-1789）
H_HK(k_L) = sym[ Σ_n F_n⟨c_L,n| ]              （:1791-1816）
```

H_HK 按 δγ/δψ 构造——**它才是 γ 的正确共轭算符**。符号约定 +(i/2)·w_eff·SC
经 symmetrize 等效负号驱动（dev log #1，**不可翻转**）。

**(c) PW**：`H_c,PW = Σ_I λ_I·P̂_I^onsite`（无 τ、无 H_HK），力复用 dspin 核
（`forces_onsite.cpp:80`）。

### 2.3 力分解（dspin 比对严格版）

**dspin 模板为什么 Pulay-only 力是精确的**——三个恒等条件：

1. escon 的观测量 = H_c 算符的 expectation：M_I = Tr[ρ·Ô_I]，Ô 与 H_c 中同一
   （`cal_mw.cpp` 头注释明确 M_i = Tr[P_at·ρ]）；
2. 因此 E_band 中的 TrρλÔ 与 escon 的 −λM **恒等相消**，E' = E_KS(ψ\*)（+const）；
3. 约束激活（M ≡ m ⇒ dM/dR = 0）时，响应项 −λ(δM/δψ)(dψ/dR) 恰好等于
   +Trρ·∂(λÔ)/∂R，即 Pulay 项。

⇒ dspin 实现 `F = F_KS(ψ*) + F_Pulay(λÔ)` 在约束激活时**精确**。

**DeltaP 逐条核对**：

| dspin 条件 | DeltaP 是否满足 | 后果 |
|------------|----------------|------|
| ① 观测量=算符 expectation | **不满足**：escon 减 λγ（Wilson loop），H_c 加 λτP̂（SMO 投影）+ H_HK——三个不同对象 | E_band 与 escon 无相消；Pulay-only 只是"代理算符近似" |
| ② 能量相消 | **不满足**（①的推论） | E' ≠ E_KS(ψ\*)，FD 微分对象必须显式含 escon |
| ③ 约束激活 | 可满足（γ≈t 时 dγ/dR≈0） | **FD 协议的关键**（§4 T7-b 双组方案） |

**正确的 Pulay 项应属于 expectation=γ 的算符，即 H_HK；H_HR 只是代理。**
DeltaP Pulay-only 力的误差 = [H_HK Pulay 缺失] + [H_HR 与 γ 算符的代理差距]
+ [τ_α 显式导数缺失]。

**由此分解（实现清单）**：

```
F_J = F_J^KS(ψ*)                                        [已有]
      − Σρ · ∂(λ_I τ_α P̂_I)/∂R_J  = A1 + A2              [A1 已实现；A2 未实现]
      − Σ_k ρ(k) · ∂H_HK(k)/∂R_J                          [B：未实现，正确算符的 Pulay]
      + Σ_I λ_I dγ_I/dR_J 的响应残差                       [C：FD 定量，不解析实现]
```

- **A1**（SMO 系数导数 Pulay）：−Σρ·λ_I·τ_α·Σ_{lm} ∂(c_μc_ν)/∂R_J。
  已实现，符号模式与 dspin 一致（`force1 += tmp, force2 −= tmp`）。
- **A2**（τ_α 显式导数，HF 项，**DeltaP 独有、优先级最高**）：
  ```
  F_J,β ⊃ −λ_J · ⟨P̂_J⟩ · ∂τ_α(J)/∂R_{J,β} = −λ_J · ⟨P̂_J⟩ · (L⁻¹)_{αβ}
  ```
  ⟨P̂_J⟩ = Σ_{μν} ρ_νμ Σ_{lm∈J} c_μc_ν（可用 dmR 与 pre_hr[iat] 收缩得到）；
  ∂τ_α/∂R_β = (L⁻¹)_{αβ}（注意 lat0 因子；代码参照 `ucell->latvec`/`ucell->G`）。
- **B/C**：不解析实现，由 T7-b 双组 FD 定量残差后写 LIMITATION（§6-O1）。

### 2.4 应力分解

```
σ_αβ = σ^KS + (1/Ω) Σ_pairs F_α(pair)·R_β(cart)          [S1：已实现，cal_stress_IJR]
       + H_HK 应变项（同 B，未实现）+ escon 应变项（同 C）
```

1. **τ_α 是分数坐标**：固定分数坐标应变约定下 ∂τ_α/∂ε = 0，应力无 A2 对应项；
   但约束随晶格"共形拉伸"是设计选择，须声明（影响压电/应力解释，§6-O2）。
2. S1 的 R_cart 已修直角坐标（:316-319），÷Ω 在 :201-209。
3. PW 应力整体缺失（§6-O4）。

### 2.5 物理类比（τ_α 项的重要性）

H_HR = λ·τ_α·P̂ 形式上是极化现代理论中的电场焓耦合 −E·P：A2 对应
−E·Z\* 型 Born 电荷力——**约束极化/ Berry 相位时的主导物理力，不是小修正**。
同理 C 项 λ·dγ/dR 即"Born 有效电荷型"耦合。A2 缺失意味着 relax 中原子
感受不到约束的均匀场力——这是当前 relax 物理错误的最大单项来源。

### 2.6 单位与符号约定

| 量 | 单位 | 出处 |
|----|------|------|
| λ_I | Ry | INPUT `deltap_lambda_init` |
| γ_I | rad | Wilson loop，mod 2π 分支管理 |
| escon | Ry | `compute_dp_escon` = −Σλγ |
| E-field 等效 | E = −πλ/(2a)（**Ha**/bohr） | esolver_ks_lcao.cpp；Ry↔Ha 因子 2（T-13） |
| force | Ry/Bohr（内部）；PW 打印转 eV/Å | `forces_onsite.cpp:97` |
| stress | Ry/Bohr³（÷Ω 后） | `deltap_force_stress.hpp:201-209` |
| H_HK 符号 | +(i/2)·w_eff·SC 后 symmetrize | **不可翻转**（dev log #1） |

---

## 3. 已知 bug 与根因

### B-1（阻塞，必崩）：`cal_force_stress` nlm 布局越界（`85b2af322` 引入）

`deltap_force_stress.hpp:76-77`：`length = (nwl+1)²`，`nlm_target(length*4)`——
按**每个 L 一个块**分配；但 :110-133 提取循环按 `iw` 遍历**所有轨道**
（多 ζ 逐一计数，`index` 持续自增，不区分 N/zeta）：ζ≥2 时
`index + channel*length` 越界写 → 堆损坏，OMP 区内表现为
heap corruption / double-free（`tests/deltap_relax` backtrace）。
注意 `cal_pre_HR:207-219` 同模式但有 target_L++ 覆盖语义——不越界但**多 ζ
丢轨道贡献**（§6-O3，同根）。dspin 版同样按 L 块布局，但 dspin 的投影算符
只取 sc_lambda 指定的单 (L,N) 通道，不遍历多 ζ——**复制模板时语义已偏离**。

**修复方向**：nlm 布局按 (L,N) 双索引（对齐 `atom1->iw2n`），或按 nw 分配并
同步改 `cal_force_IJR` 索引。验收：ASAN + `tests/deltap_relax` ≥3 离子步。

### B-2（已修备查）
hR=nullptr 空指针（R5）；缺 τ_α 因子、多 ×2、应力整数晶格矢量量纲（07-29 round 2）。

### B-3（待复核）
C-12/13 relax 跨离子步 λ 状态：`contributeHR` 已有 hR 重建全量补加（:60-72），
状态机有 `reset_ionic_step`——T7-a 复跑 `deltap_relax` 核对 λ 轨迹即可关闭。

### B-4（PW 侧）
PW 把 DeltaP 当 dspin-z 处理（`lam[iat].z = λ_I`，`forces_onsite.cpp:73`）：
与 PW 的 H_c = λ·P̂^onsite 自洽，但 P̂^onsite ≠ γ 的 k-string 算符，escon 仍按
−Σλγ 记账 → 同样不闭环（B/C 类）。PW 应力完全缺失。PW FD 验证优先级低于 LCAO。

---

## 4. 开发计划（T7 三步，每步独立可验收）

### T7-a：修崩（0.5–1 天）
1. 按 §3-B1 修 nlm 布局（(L,N) 双索引），顺带处理 O3 多 ζ 覆盖语义；
2. ASAN 构建 + `tests/deltap_relax` ≥3 离子步；
3. 核对 B-3（λ 轨迹跨离子步正确、无重复累加）；
4. 验收：exit=0、ASAN 干净、`deltap_mpi_smoke/run.sh` 三用例 + 单测 16/16 不回归。

### T7-b：FD 验证（双组协议，1–2 天）★ 修正版

微分对象 **E' = E_tot + escon**（提取 running log 的 etot + [DeltaP] escon；
`tests/deltap_fd_force/run_fd.sh` 需补 escon 项）。位移 ±0.005 Bohr，H₂O。

- **组① frozen λ**：直接 FD。判据之外的残差 ≈ λ·dγ/dR + A2 项——
  组①残差同时给出 A2 大小的独立测量（与解析式 −λ_J⟨P̂_J⟩(L⁻¹)_{αβ} 对照）。
- **组② 每位移点重新收敛 λ（γ≈t）**：对应 relax 实际受力。dspin 论证
  （§2.3 条件③）此时 Pulay-only 应在代理近似内精确——**组②才是 relax
  可用性的判决性测试**。
- 判据：|F_analytic − F_FD| < 5e-4 Ry/Bohr（参照 dspin 容差惯例；
  dspin 自身 FD 文档本仓库未查到，"与 dspin 同构"不构成豁免）。
- 两组残差分离：组②残差 − 组①残差 ≈ λ·dγ/dR（C 项量级），写 LIMITATION。

### T7-c：补 A2 + 定量 B/C（2–3 天）
1. 实现 A2（§2.3）：⟨P̂_J⟩ 用 dmR·pre_hr 收缩；L⁻¹ 用 `ucell->G`（注意倒格矢/
   lat0）；力路径 tmp 算符需能访问 pre_hr（重建或传入）。
2. 实现 ∂E'/∂γ = λ 诊断打印（对标 dspin "magnetic force" 打印，
   `spin_constrain.cpp:857`——零成本，助 FD 调试）。
3. FD 双组复验；应力 FD（变胞 ±0.1%）验 S1。
4. 关闭 C-02；`deltap_relax` 与 efield 对照（P17）跑通。
5. 可选：PW FD 力验证；PW 应力立项（O4）。

---

## 5. 代码 ↔ 公式映射速查

| 公式项 | 代码 | 备注 |
|--------|------|------|
| H_HR 构建 | `deltap_lcao.cpp:125 cal_pre_HR` / `:253 cal_HR_IJR` | SMO 外积 Σ_lm c_μc_ν |
| λ 增量进 hR | `:50 contributeHR` | dλ；hR 重建全量补加（:60-72） |
| H_HK 构建 | `deltap_wannier.cpp:1639 compute_hk_correction` | nrow==ncol 方阵限制（C-06） |
| H_HK 进 hsk | `deltap_lcao.cpp:309 contributeHk` | double 实例取 real |
| escon | `deltap_common.h:157` / `deltap_scf.cpp:310` | −Σλγ，rank 一致 |
| A1 力 | `deltap_force_stress.hpp:216 cal_force_IJR` | 符号=dspin 模板 |
| S1 应力 | `:285 cal_stress_IJR` | F·R_cart / Ω |
| 力入口 | `FORCE_STRESS.cpp:433-453` | 临时算符 + 静态 λ 传递 |
| PW 力 | `forces_onsite.cpp:64-109` | 复用 dspin 核（lam.z=λ） |
| dspin 力模板 | `dspin_force_stress.hpp:298-308` | FD 验证模板（本仓库无 FD 文档） |
| dspin escon | `spin_constrain.cpp:38-52` | −Σλ·M，M=Tr[P_at·ρ]（cal_mw.cpp） |
| dspin 诊断 | `spin_constrain.cpp:857` | "magnetic force"=λ 打印 → DeltaP 应仿照 |
| FD 脚本 | `tests/deltap_fd_force/run_fd.sh` | 备好未跑；E 提取需补 escon |

---

## 6. 开放问题

- **O1**：B/C 净误差量级——T7-b 双组 FD 残差回答；决定 H_HK 力项是否必须解析实现。
- **O2**：τ_α 分数坐标 vs Cartesian 的应变耦合（§2.4-1）——物理语义需与用户确认。
- **O3**：`cal_pre_HR` 多 ζ 覆盖语义丢轨道——与 B-1 同根，T7-a 一并处理并 FD 复验 γ。
- **O4**：PW 应力整体缺失；PW H_c 与 γ 算符差距未定量。
- **O5**：H_HR 代理算符与 γ 算符的差距（§2.3 表）——若组② FD 残差超判据，
  需考虑把 H_c 换成与 γ 一致的算符形式（H_HK -only 模式）。

---

## 7. 公式集（开发时必须 follow）

```
[F1] E' = E_tot + escon,  escon = −Σ_I λ_I γ_I                    （记账唯一口径）
[F2] H_c = Σ_I λ_I τ_α(I) P̂_I  +  H_HK        （LCAO；PW 只有 λ_I P̂_I^onsite）
[F3] P̂_I,μν = Σ_{lm∈I} c_μ(Ilm) c_ν(Ilm)      （SMO 投影，cal_deri=0/1）
[F4] A1: F_J ⊃ −Σ_{μν} ρ_νμ · λ_I τ_α(I) · Σ_{lm} ∂(c_μc_ν)/∂R_J
[F5] A2: F_J,β ⊃ −λ_J ⟨P̂_J⟩ (L⁻¹)_{αβ},  ⟨P̂_J⟩ = Σρ·Σ_{lm∈J} c_μc_ν
[F6] B:  F_J ⊃ −Σ_k Σ_{μν} ρ_νμ(k) ∂H_HK,μν(k)/∂R_J    （不实现，FD 定量）
[F7] C:  E' 对 γ 的共轭力 ∂E'/∂γ_I = −λ_I；relax 判据 γ→t 时 λ→λ*
[F8] S1: σ_αβ ⊃ (1/Ω) Σ_pairs F_α(pair) R_β(cart)；∂τ_α/∂ε = 0（分数坐标约定）
[F9] FD: F_FD = −[E'(R+δ) − E'(R−δ)] / 2δ,  δ = 0.005 Bohr
[F10] E-field 等效：E = −πλ/(2a) Ha/bohr；γ 与 μ 换算须先 unwrap 再 ÷自旋因子 2
```

## 8. Check-list（每轮开发逐项打勾）

**代码正确性**
- [ ] nlm 布局按 (L,N) 双索引；多 ζ 算例（Si/混基）无越界（ASAN）
- [ ] A1 符号 = dspin 模板（force1 += tmp / force2 −= tmp）
- [ ] A2 实现含 lat0 与 L⁻¹ 取向（用 gdir=1/2/3 三方向 FD 分别验证）
- [ ] λ 跨离子步轨迹：无重复累加、无丢失（deltap_relax ≥3 步打印核对）
- [ ] escon 与 [DeltaP] 打印一致；E' 提取链 = etot + escon

**数值验证**
- [ ] FD 组①（frozen λ）通过判据 5e-4 Ry/Bohr（A1+A2 后）
- [ ] FD 组②（每点位 λ 重收敛）通过判据——relax 可用性判决
- [ ] 组②−组①残差 = C 项量级，写入 LIMITATION
- [ ] 应力 FD（变胞 ±0.1%）验 S1
- [ ] gdir=1/2/3 三方向各至少一例 FD

**回归**
- [ ] 单测 16/16；`tests/deltap_mpi_smoke/run.sh` 三用例 PASS
- [ ] SCF 端不回归：test_C_I 能量锚点逐字节一致
- [ ] 文档：dated 文档 + dev log 追加（AGENTS.md 要求）

## 9. TODO-list（优先级序）

| # | 事项 | 依赖 | 预估 |
|---|------|------|------|
| T7-a | 修 B-1 nlm 布局（含 O3 多 ζ）+ ASAN 验收 + B-3 核销 | — | 0.5–1 天 |
| T7-b | FD 双组协议（run_fd.sh 补 escon 提取；组①+组②） | T7-a | 1–2 天 |
| T7-c | 实现 A2 + λ 诊断打印 + FD 复验 + 应力 FD → 关闭 C-02 | T7-b | 2–3 天 |
| T7-d | P17 relax 对照（deltap vs efield）端到端 | T7-c | 0.5 天 |
| O2 | τ_α 应变耦合语义确认（用户决策） | 随时 | — |
| O4 | PW 应力立项；PW FD 力验证 | T7-c 后 | 另立 |
| O5 | 若组②残差超判据：H_c 算符形式重构评估 | T7-b 数据 | 另立 |

---

## 10. 本轮记录（v2 修订）

- 依据 dspin 比对（spin_constrain/cal_mw/dspin_force_stress 源码）重写 §2.3：
  dspin Pulay-only 精确性三条件 → DeltaP 逐条核对（①②不满足、③可满足）；
  A2 提为最高优先（Born 电荷型主导力，§2.5）；FD 改双组协议（§4 T7-b）；
  新增 §7 公式集、§8 check-list、§9 TODO。
- 未改运行时代码。
