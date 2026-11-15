# ABACUS 约束 DFT 算法完整综述：DeltaSpin → DeltaP → Allocation → MBIS（2026-05-17 至今）

> 本文从 **2026-05-17（DeltaSpin 参数统一）** 起，沿 **DeltaP（Berry 相位/极化约束）**、
> **统一局域矩分配（Allocation）** 到 **MBIS（迭代 stockholder）** 的完整演进，给出各代
> **公式、物理机制、优劣、实测结果**，并从**物理机制**与**实际有效性**两维对比。
> 数值取自本仓库实测文档（§12 映射）；标注"指示性"者为粗网格/旧胞。
> 更细的逐算法公式见 `local-moment-constraint-algorithms-review.md`；
> 更正台账见 `mbis-development-history.md`。

---

## 0. 演进总览与共享主线

**一条共同主线**（所有代数都遵守）：
```
选一个"可观测量" Q_a（电荷/自旋/极化/壳层矩）
给每个约束加势 V_a = μ_a · W_a（或非局域算符），使 SCF 解满足 Q_a = t_a
报告能量 E' = E_KS + Σ_a μ_a (Q_a − t_a)          (约束泛函)
或物理能 E  = E_KS（把约束场贡献扣掉：escon = −Σ_a μ_a Q_a）
λ(μ) 由 secant / BFGS / 增广拉格朗日 更新
```

| 日期 | 里程碑 | 关键决定 / 贡献 |
|---|---|---|
| **2026-05-17** | **DeltaSpin 参数统一** | `sc_scf_thr`/`sc_scf_thr_mode`（`threshold`/`immediate`）、`sc_direction_only`、`mixing_restart` 逻辑、λ 循环激活时机；**把 ΔSpin 的 λ 回路做成可靠工程基础** |
| 2026-07～09 | **DeltaP（Berry/极化约束）** | 可观测量泛化到 **Berry 相位/极化** `Γ_I`；确立 `escon=−ΣλΓ` 记账；λ 两阶段/BFGS；`operator`/`gamma` 两种模式；力/relax 边界 |
| 2026-10-21～29 | **统一局域矩分配 Allocation** | 正交目标轨道投影 + 互补子空间空间权重；`WeightGrid`/`ConstraintLoop`/通道/ALM 通用框架；大量性能/一致性优化 |
| **2026-10-30** | **MBIS** | 迭代 Slater stockholder（密度自适应）；修正固定形状的离子性偏差；约束不动点；min-image/能量 `−ΣμQ`/MPI/快速 μ |

---

## 1. 共享约束框架（`module_constraint`）

- **权重网格** `WeightGrid`：`w_[iat][ir]`（原子权重，`Σ_i w=1`）、`cw_[α]`（片段=原子和）、
  `dw_`（位置导数，力核输入）。权重类型：Becke / Hirshfeld(周期) / **MBIS**（§5–6）。
- **读数** `ConstraintObserver::observe`：`Q_α = ∫ w_α·χ dr`（`χ=ρ`（电荷）或 `m=ρ↑−ρ↓`（自旋）；
  nspin=2 双通道，`read_up/read_dn`）。
- **注入** `ConstraintInjectPW/LCAO`：`V_eff[σ] += Σ_α μ_α·inj_σ·w_α`
  （电荷 `inj_up=inj_dn=+1`；自旋 `+1/−1`）。LCAO 侧隐式经 `cal_gint_vl` 进 `H`。
- **记账** `ConstraintAccounting`：`escon`、`CONSTRAINT_AUDIT`（key=value）。
- **力** `constraint_force`（PW/LCAO 共用）。
- **外循环** `ConstraintLoop`：`μ` 更新（secant/trust）；目标 `delta|absolute`；分支守卫 `constraint_branch_tol`。
- **能量抵消（关键，S6.72）**：约束场进 `V_eff` 后，报告/守卫须扣掉场能量
  `cc_escon = −Σ_α μ_α Q_α`（DeltaSpin `−ΣλM` 形式），否则大布居（如 Fe Q≈14）会假 `BRANCH_FLIP`。

**求解器（`μ` 控制）对比**（线性模型实测，`χ=dQ/dμ`）：
| 方案 | 更新 | 收敛 | 备注 |
|---|---|---|---|
| 全局 Lagrange | `μ←μ+α(Q−t)`, `α<2/χ_m` | **23016 步** | 被近零模拖死 |
| **ALM** | 内层 `μ_eff=μ+k(Q−t)`，`k=κ/χ`；外层同 | **10–34 步**，精确 | 推荐 |
| penalty-only | `μ_eff=k(Q−t)` | 1 步但 `Q*−t=(Q0−t)/(1+kχ)` **有偏** | 仅预条件 |
| secant/trust（MBIS 主循环） | `κ=clamp(|dQ/dμ|,κ_min,κ_max)`, `|Δμ|≤step_max` | stiff 通道需 `κ_min≈0.01` | LiCl 3 步 |

---

## 2. 第一代：DeltaSpin（onsite 轨道磁矩约束，05-17 工程化）

**可观测量（onsite 轨道投影）**
```
M_I = Tr[ P_I · ρ_diff ] ,   ρ_diff = ρ↑ − ρ↓ ,   P_I = 原子 I 的轨道投影
```
**约束势 / 能量**
```
H += λ · σ_z (onsite 非局域投影修正，collinear)
escon = − Σ_I λ_I · M_I            (cal_escon, spin_constrain.cpp:38)
E'   = E_KS + escon
λ 更新：两阶段（sc_scf_thr 门控）+ 方向 BFGS / secant；sc| 掩码选约束原子
```
- **05-17 贡献**：`sc_scf_thr_mode={"threshold","immediate"}`（替代 `10.0` 魔法数）、
  `sc_direction_only` 两阶段、`mixing_restart` 与 λ 激活时机的显式化——**让 λ 回路可靠、
  可调、可与 mixing 共存**。
- **机制**：直接约束**原子 onsite 轨道磁化**（TM 的 d 矩），非空间盆。
- **实测**：Fe2 目标 `±3.8 μB` → onsite 报告 **±3.800**（匹配）；其"传统 orbital-sum"
  仅 ±2.93（口径差）。
- **优**：直接命中壳层/onsite 磁矩、成熟、力/relax 完备。
- **劣**：投影族（非空间 AIM）；APNS 赝势下多点翻磁、TM d 通道脱钩等已知陷阱；
  `σ_z` 对 SOC/非共线受限；只覆盖磁矩（电荷需另设）。

---

## 3. 第二代：DeltaP（Berry 相位/极化约束，07–09）

**可观测量（Berry 相位/极化，per-atom 可加和）**
```
Γ_I = Γ_I^HR + Γ_I^HK          (约束算符期望；HR=实空间项, HK=k 空间/Hermitian 修正)
γ_I = Wilson loop Berry 相位    (旧模式 deltap_observable=gamma)
```
**约束势 / 能量**
```
Route A+ (operator, 推荐)：H 中加入约束项（deltap_corr 1），
escon = − Σ_I λ_I · Γ_I ,   E' = E_KS(ψ*)
场模式：均匀 λ ≈ 长程电场 E_eff = +πλ/(2a)（响应校准 ×1.6；|λ|≤0.007 Ry 内与锯齿场 <1 meV）
```
**λ 控制**
```
两阶段(inner_nmax=0)：Phase1 λ=0 自然 SCF → 梯度下降 μ + mix_reset → Phase3 冻结 λ 收紧
内层 BFGS(inner_nmax=3)：冻结密度，BFGS 优化 λ（3–5 次对角化），远快于两阶段
```
- **机制**：把"约束"从**磁矩**推广到**极化/Berry 相位**（可观测量由算符期望给出）；
  **确立 `escon=−ΣλΓ` 的物理能量记账**与 λ 求解范式。
- **实测/边界**：operator(Route A+) 力/relax 完备；gamma(旧) 力/relax **封口**
  （弱耦合 + γ-hold 泄漏）；场模式响应校准 ×1.6。
- **优**：**可观测量最通用**（可拼任意线性组合）；能量记账范式被后续全部继承；λ 控制成熟。
- **劣**：`gamma` 模式的力/relax 不闭合；`SMO`/规范（gauge）细节多；实现复杂。

---

## 4. 第三代：统一局域矩分配（Allocation，10-21～29）

**约束坐标（正交目标投影 + 互补子空间空间权重）**
```
A_i = Q_i + Q_c W_i Q_c ,   P = Σ_i Q_i ,  Q_c = I − P
q_i = Tr[ρ A_i] = Tr[ρ Q_i] + Tr[ρ Q_c W_i Q_c]
报告：n_tot,i = Tr[ρ W_i] = Σ w_i ρ dV ,   n_T,i = Tr[ρ Q_i] ,  n_c,i = n_tot − n_T
```
**约束势 / 能量**：`V_eff += μ_a w_a`（空间权重，通用框架）；
`cc_escon = −Σμ_a t_a`（allocation，因 eband 已含 μQ）；审计 `e_con=Σμ(Q−t)`。

- **机制**：把**目标壳层投影 `Q_i`** 与**互补空间权重 `W_i`** 混合，使求和精确且可
  重分配 `n_T/n_c`；配套 `WeightGrid`/通道/ALM **通用框架**。
- **实测**：LiCl `q=3.083`（≈Becke 3.096，**非净电荷**）；Fe2 `m_T=∓3.599/3.600`，
  orbital-sum `±2.93`；与 DeltaSpin 在 `±3.8` 下总能差 **0.625 eV**。
- **关键坑**：`q` 是**建模坐标≠净电荷**；算符互补 `Tr[ρ Q_cWQ_c] ≠ 报告 n_tot−n_T`
  （Fe2 `m_c` 算符 −0.20 vs 报告 +0.78）；每外步重建、成本高。
- **优**：求和精确、可定向壳层、通用框架（后续 MBIS 复用）。
- **劣**：坐标语义非净电荷、算符/报告不一致、成本高。

---

## 5. 第四代：MBIS（迭代 stockholder，10-30）——当前电荷/自旋约束

**定义（每原子 Slater-s，密度自适应）**
```
pro_A(r) = N_A · ζ_A³/(8π) · exp(−ζ_A · d_min(r,R_A))
w_A(r)   = pro_A / Σ_B pro_B ,   N_A = ∫ w_A ρ ,  ζ_A = 3N_A / ∫ r w_A ρ
自旋 M_A = ∫ w_A m dr ;  Σ_A N_A = ∫ρ (求和规则自动)
```
**求解**：Picard **不动点**（物理起点 ζ~1；稳定性 `ρ(∂F/∂p)<1`）；**D 仅诊断**
（`ζ→0` 时 D=0 退化，**不用于选解**）；Newton 退化回退 Picard。
**周期**：`constraint_mbis_images=k` → `(2k+1)³` 晶格平移；`0`=direct（大盒），
`≥1`=最小镜像（小胞必须）。**均径 r 与 pro 同镜像集**；**力**用光滑晶格和。
**能量**：`cc_escon=−ΣμQ`（DeltaSpin 形式，扣约束场）；`CONSTRAINT_AUDIT e_con=Σμ(Q−t)`。

- **机制**：与 Hirshfeld **同形式** `∫wρ`，但 `ζ_A` **随密度自洽迭代** → 离子化收缩
  （Li⁺ ζ≈4.6）→ **抓住离子性**。是**带布居约束的不动点**（非自由 D 最小化）。
- **实测**：LiCl `2.0447 e`（+0.955）、**偶极 6.48 D**（实验 ~6.3）；bcc Fe2 `16/16`、
  空间自旋 `±2.647`（cube）/ `±2.85`（min-image 0.50）、`ρ≈0.31` 稳定；
  **盆内 l 投影 d=3.069 ≈ Mulliken 3.034 ≈ d-投影 2.985**。
- **优**：物理最合理（离子性/偶极/AFM 局域矩）、通用、可扩壳层（盆内 l）、与 ORCA 同族。
- **劣/代价**：密度依赖→每外步迭代；周期小胞需 min-image；大 Q 能量需 `−ΣμQ`；
  外循环较慢（需 secant/kappa_min）。

---

## 6. 空间分片旁支：Becke / Hirshfeld（固定形状）

### 6.1 Becke
```
μ_IJ=(r_I−r_J)/R_IJ ; p(x)=0.5x(3−x²) ; s(μ)=0.5(1−p(p(p(μ))))
p_I=Π_{J≠I}s(μ_IJ) ; w_I=p_I/Σ_J p_J ; 异核 a_IJ=clip(u/(u²−1),±0.5)
```
纯几何、固定半径、单位分解精确（实测 maxdev=0）。LiCl Li `3.096`、Fe2 `2.285`。
**优**：便宜/光滑/确定；**劣**：不随离子化收缩→低估离子性、TM 局域矩偏低。

### 6.2 Hirshfeld（周期）
```
w_A(r)=ρ_A^free(r−R_A)/Σ_B ρ_B^free(r−R_B)   (free 参考固定, 非迭代)
+C2 窗口 + 晶格镜像和 + 真空余量通道
```
单位分解 maxdev<1e-12、导数 FD 通过。与 Becke 同族偏差（参考固定→离子性偏中性）。

---

## 7. 投影旁支：Mulliken / d-投影

- **Mulliken**：`q_A=Σ_{μ∈A}(DS)_{μμ}`；LiCl 2.483、Fe2 3.082；**基组依赖强**，不宜作 AIM。
- **d-投影**：`m_T=Tr[ρ σ_z Q_{A,l}]`；Fe2 d=`2.985`；壳层特定，与 DeltaSpin 同族。

---

## 8. 统一数值对比

### 8.1 LiCl（电荷/偶极，同一自由密度）
| 方法 | Li N(e) | Li 净 | 偶极 |
|---|---|---|---|
| 实验 | — | — | ~6.3 D |
| **MBIS** | **2.0447** | **+0.955** | **6.48 D** |
| Mulliken | 2.4830 | +0.517 | 5.0 D |
| Allocation | 3.0833 | −0.083 | — |
| Becke | 3.0959 | −0.096 | — |

### 8.2 Fe2（自旋四口径 + 盆内 l）
| 口径 | 自旋 | 备注 |
|---|---|---|
| Becke 盆 | 2.285 | 固定半径 |
| **MBIS 盆** | **2.647**/2.85 | ζ≈1.08；min-image 0.50 |
| d-投影 | 2.985 | 壳层 l=2 |
| Mulliken | 3.082 | 轨道布居 |
| **MBIS 盆内 l** | **d=3.069** | ≈ Mulliken/投影 |

### 8.3 DeltaSpin / DeltaP / Allocation（口径与能量）
| 项 | 值 |
|---|---|
| DeltaSpin Fe2 目标 ±3.8 → onsite 报告 | ±3.800（匹配）；orbital-sum ±2.93 |
| Allocation Fe2 `m_T` / orbital-sum | ∓3.60 / ±2.93；与 ΔSpin ±3.8 下 ΔE=0.625 eV |
| DeltaP 场模式 | `E_eff=+πλ/(2a)`，响应 ×1.6；|λ|≤0.007 Ry 内 <1 meV |

---

## 9. 物理机制对比（核心）

1. **可观测量层次**：DeltaSpin/DeltaP/投影 → **算符期望**（壳层/极化，基组/规范敏感）；
   Becke/Hirshfeld/MBIS → **空间盆**（AIM，求和精确）；Allocation → **两者混合**（语义需小心）。
2. **固定形状 vs 密度自适应**：Becke/Hirshfeld 固定 → **不响应离子化**（Li 3.1）；
   **MBIS 迭代 `ζ`** → **复现离子性**（Li 2.04、偶极 6.48）。**主因是权重形状（~1 e）**。
3. **空间盆 vs 壳层矩**：MBIS 盆自旋 2.65（含非 d 分量）< d 投影 2.99；
   **盆内再做 l 投影 → d≈3.07**，与 Mulliken/投影一致 ⇒ "空间 AIM + 壳层 l" 两步架构。
4. **能量记账必须扣约束场**：`escon=−ΣμQ`（DeltaSpin 形式）；大 Q 若不扣 → 假 BRANCH_FLIP
   （FeO 7 eV 异常，S6.72 已修）。
5. **近零模求解**：全局 Lagrange 慢；**ALM k=κ/χ 精确快**；penalty 有偏；MBIS 用 Picard+secant。

---

## 10. 实际有效性对比

| 维度 | DeltaSpin | DeltaP | Allocation | **MBIS** | Becke | Hirshfeld | Mulliken |
|---|---|---|---|---|---|---|---|
| 约束对象 | onsite 磁矩 | 极化/任意算符 | 壳层+空间 | **电荷/自旋** | 空间 | 空间 | 轨道 |
| 离子性/偶极 | 否(磁) | 否(极化) | 低 | **高** | 低 | 低 | 中低 |
| 壳层可分辨 | **是** | 是(算符) | 是 | **是**(盆内 l) | 否 | 否 | 是 |
| 基组依赖 | 中 | 中 | 中 | **弱** | 无 | 无 | **强** |
| 周期小胞 | 好 | 好 | 好 | **需 min-image** | 好 | 好 | 好 |
| 成本 | 中 | 中 | 高 | 中高 | **低** | 低 | 低 |
| 力/relax | **完备** | operator 完备 | 完备 | 完备 | — | — | — |
| 能量记账 | `−ΣλM` | `−ΣλΓ` | `−Σμt` | **`−ΣμQ`** | 简单 | 简单 | — |
| 已验证 | Fe2 | 极化算例 | LiCl/Fe2 | **LiCl/Fe2/e2e/MPI** | LiCl/Fe2 | 测试级 | LiCl/Fe2 |

---

## 11. 结论与推荐

- **电荷（离子性/偶极）**：**MBIS**（密度自适应，Li⁺/偶极），`constraint_mbis_images≥1`（小胞）
  + `cc_escon=−ΣμQ`（大 Q）。
- **壳层/onsite 磁矩**：**DeltaSpin / d-投影**（直接命中 d 矩）或 **MBIS 盆内 l**（d≈3.0 一致）。
- **极化/任意线性算符**：**DeltaP Route A+（operator）**。
- **快速/定性分片**：**Becke / 周期 Hirshfeld**（不求定量离子性）。
- **求解器**：**ALM** 或 **Picard+secant(kappa_min)**；避免纯 Lagrange/penalty。
- **勿**：Mulliken 当 AIM、allocation `q` 当净电荷、跨口径直接比数。

---

## 12. 文档映射
- 05-17：`docs/superpowers/specs/2026-05-17-deltaspin-parameter-unification-design.md`（+ plans）。
- DeltaP：`docs/deltap_user_manual.md`、`docs/superpowers/specs/2026-08-*deltap*`、dev log（2026-07～09）。
- Allocation：`docs/superpowers/specs/2026-10-21..29-allocation-*`、`2026-10-29-allocation-development-summary.md`。
- MBIS：`docs/superpowers/specs/2026-10-30-mbis-*`、`mbis-development-history.md`、
  `local-moment-constraint-algorithms-review.md`、dev log S6.34–S6.74。
- 框架：`docs/constraint_developer_guide.md`、`docs/constraint_user_manual.md`。
