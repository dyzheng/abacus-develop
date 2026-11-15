# ABACUS 约束 DFT 算法完整综述（2026-05-17 → 今）
### DeltaSpin → DeltaP → Allocation → MBIS：公式、物理机制与实际有效性

> 本文从 **2026-05-17 的 DeltaSpin 参数统一**起，沿 **DeltaP（Berry/极化约束）**、
> **统一局域矩分配（Allocation）** 到 **MBIS（迭代 stockholder）** 的完整演进，系统给出
> 各代**公式、物理机制、优劣与实测**，并从**物理机制**与**实际有效性**两维详细对比。
> 数值取自本仓库实测文档（§16 映射）；标注"粗网格/旧胞"者为指示性。
> 配套：`mbis-development-history.md`（更正台账/当前规范）、
> `local-moment-constraint-algorithms-review.md`（算法细节子集）、
> `deltap-development-log.md`（逐轮日志 S6.34–S6.75）。

## 目录

- §0 引言、范围与记号
- §1 统一数学框架（可观测量 / 约束势 / 泛函 / escon 记账 / 力 / 近零模）
- §2 第一代 DeltaSpin（onsite 轨道磁矩；05-17 参数统一；λ 回路/BFGS）
- §3 第二代 DeltaP（Berry 相位/极化；`Γ_I`；operator/gamma；场模式）
- §4 通用约束框架 `module_constraint`（WeightGrid/通道/注入/记账/力/ALM）
- §5 Becke（固定多边形分片）
- §6 Hirshfeld（周期 stockholder，固定参考）
- §7 Allocation（混合投影 + 空间互补）
- §8 MBIS（迭代 stockholder，当前电荷/自旋约束）
- §9 投影旁支 Mulliken / d-投影
- §10 约束求解器（Lagrange/ALM/penalty/secant/BFGS）
- §11 统一数值对比
- §12 物理机制对比
- §13 实际有效性对比
- §14 失败模式与更正台账
- §15 结论、推荐与 TODO
- §16 文档映射与阅读指引

---

# §0 引言、范围与记号

## 0.1 为什么需要"约束 DFT"

标准 Kohn–Sham（KS）DFT 只给出**总能量极小**的电子结构；但很多物理问题关心的是
**局域/片段的电子占据、磁矩、极化**等**非总能量最小的自洽约束态**（如强制某 TM 原子
处于给定 d 占据/自旋态、固定某片段的电荷转移量、约束 Berry 相位/极化）。约束 DFT 的做法是：

1. 选一个**可观测量** `Q_a`（可由密度/波函数线性或非线性给出）；
2. 给体系加一个**约束势**（拉格朗日乘子 `μ_a` 乘以权重/投影算符）；
3. 反复 SCF 并更新 `μ_a`，直到 `Q_a = t_a`（目标）。

这类"**受约束的自洽态**"是理解强关联、电荷转移、磁性、铁电等问题的常用工具，也是
DeltaSpin / DeltaP / 本仓库 MBIS 方案的共同出发点。

## 0.2 本文范围：2026-05-17 → 今

本文整合本仓库**从 2026-05-17 的 DeltaSpin 参数统一**起，到当前 **MBIS** 方案的完整演进：

- **2026-05-17**：DeltaSpin 的**工程化**（`sc_scf_thr_mode`、`sc_direction_only`、
  `mixing_restart`、λ 回路激活时机）——把"磁矩约束"做成可靠基础。
- **2026-07～09**：DeltaP——把约束**推广到 Berry 相位/极化**（`Γ_I`），确立
  `escon=−ΣλΓ` 的**物理能量记账**与 λ 求解范式。
- **2026-10-21～29**：统一局域矩分配（Allocation）——正交目标投影 + 互补空间权重，
  并落地通用的 `WeightGrid / ConstraintLoop / 通道 / ALM` 框架。
- **2026-10-30**：MBIS——**密度自适应**的迭代 Slater stockholder，修正固定形状的
  离子性偏差；约束**不动点**；min-image / 能量 `−ΣμQ` / MPI / 快速 μ。

> 更细的算法公式见 `local-moment-constraint-algorithms-review.md`；更正/撤回台账见
> `mbis-development-history.md`；本文是**面向物理机制与有效性的完整综述**。

## 0.3 读者指引

- 只想看"当前怎么做电荷/自旋约束"：读 §8（MBIS）+ §4（框架）+ §10（求解器）。
- 想理解"为什么 allocation 给出 3.08、MBIS 给出 2.04"：读 §5–§8 + §11–§12。
- 想知道"哪些早期结论被推翻"：读 §14 + `mbis-development-history.md` §2。
- 想比较"磁矩/极化 vs 电荷/空间分片"：读 §2–§3 vs §5–§8。

## 0.4 记号（Hartree 原子单位，除非注明）

| 符号 | 含义 |
|---|---|
| `ρ(r)` | 总电子密度；`ρ↑,ρ↓` 自旋密度；`m=ρ↑−ρ↓` 自旋密度 |
| `ρ_A` | 原子 A 的**分片密度** `ρ_A = w_A ρ` |
| `w_A(r)` | 空间权重（AIM），`Σ_A w_A(r)=1`；片段 `w_a=Σ_{A∈a}w_A` |
| `N_A = ∫ρ_A` | 原子 A 的电子数（MBIS 布居）；净电荷 `q_A = Z_A − N_A` |
| `M_A = ∫ w_A m` | 原子 A 的空间盆自旋矩 |
| `pro_A(r)` | MBIS/Hirshfeld 的**原（参考）密度**（Slater/Gaussian） |
| `ζ_A` | MBIS Slater pro 的宽度（以 `e^{−ζ r}`） |
| `d_min(r,R_A)` | 周期体系到原子 A 的**最小镜像**距离 |
| `t_a` | 约束目标；`μ_a`（或 `λ_I`）拉格朗日乘子 |
| `V_con(r)` | 约束势 `Σ_a μ_a W_a(r)`（或非局域投影项） |
| `E_KS, E'` | KS 总能量；受约束能量 |
| `escon` | 约束能量修正（`−Σ μ Q` 物理能 / `Σ μ(Q−t)` 泛函） |
| `P_I` | 原子/壳层投影算符（DeltaSpin/投影族） |
| `Γ_I` | Berry 相位/极化算符期望（DeltaP） |
| `Q_i, W_i, Q_c` | Allocation 的目标投影、空间权重、互补投影 |
| `χ = dQ/dμ` | 约束响应（近零模问题的主角） |
# §1 统一数学框架

所有代数（DeltaSpin/DeltaP/Allocation/MBIS/Becke/Hirshfeld）都落在同一个"约束 SCF"数学框架里。
本节把这个框架写清楚，后文只需替换**可观测量**与**约束势**的定义。

## 1.1 可观测量

**线性可观测量**（大多数情况）：
```
Q_a[ρ] = ∫ W_a(r) χ(r) dr
```
- 电荷约束：`χ(r)=ρ(r)`，`W_a=w_a`（空间权重）或投影（`Tr[ρ O_a]`）。
- 自旋约束：`χ(r)=m(r)=ρ↑−ρ↓`，`W_a=w_a`（空间盆）或 `σ_z` 投影（onsite）。
- 极化（DeltaP）：`Q_I=Γ_I`（Berry 相位算符期望，`Γ_I=Γ_I^HR+Γ_I^HK`）。
- 片段/任意线性组合：`w_a=Σ_{A∈a} w_A`；投影算符亦可线性组合。

**非线性可观测量**：如 MBIS 的 `N_A=∫w_A(N_A,ζ_A)ρ`——权重**依赖 ρ**，需要**自洽不动点**
（§8），不可写成一次性线性泛函。

## 1.2 约束势与受约束能量

约束通过**拉格朗日乘子** `μ_a` 引入。两种等价写法：

**(a) 势形式**（空间权重族）：把约束势加进有效势
```
V_eff(r) ← V_eff(r) + V_con(r),   V_con(r) = Σ_a μ_a W_a(r)
```
自旋按通道分裂：`V↑ += μ_a w_a, V↓ −= μ_a w_a`（等价 `σ_z` 结构）。

**(b) 算符形式**（投影族/DeltaP）：把约束项加进哈密顿量
```
H ← H + Σ_a μ_a O_a        (O_a 为投影/`Γ` 算符，可能非局域)
```

**受约束泛函**（SCF 实际极小化）
```
E'[ρ] = E_KS[ρ] + Σ_a μ_a (Q_a[ρ] − t_a)
```
- 对 `ρ` 变分 ⇒ `δE_KS/δ ρ + Σ μ_a δQ_a/δρ = μ_KS`，即约束势进入 KS 方程；
- 对 `μ_a` 变分 ⇒ `Q_a = t_a`（约束满足）。

## 1.3 能量记账：泛函 vs 物理能

**关键**：约束势是**人工**的，报告给用户的总能量必须是**物理的 KS 能**，或明确是
"受约束泛函"。两种约定：

1. **约束泛函**：`E_tot = E_KS[ρ*] + Σ μ_a (Q_a − t_a)`（`ρ*` 为约束解）。
2. **物理能（扣约束场）**：约束场贡献 `∫V_con ρ = Σ μ_a Q_a` 必须从电子总能中**扣除**：
   ```
   E_phys = E_KS[ρ*]   (等价于在 E_tot 中用 escon = −Σ μ_a Q_a 抵消约束场)
   ```
   **DeltaSpin 的 `escon = −Σ λ_I M_I`** 就是这一约定（`cal_escon`，`spin_constrain.cpp:38`）。

**踩过的坑（S6.72）**：若用 legacy `cc_escon = Σ μ(Q−t)` 却 **未扣除** `Σ μQ`，对**大布居**
（如 Fe `Q≈14`，`μQ≈0.6 Ry ≈ 8 eV`）会把受约束总能低估数 eV，触发**假 `BRANCH_FLIP`**。
修复：`cc_escon = −Σ μ_a Q_a`（DeltaSpin 形式）。LiCl（`Q≈2`）因 `μQ` 小而未暴露。

## 1.4 力 / 应力

受约束能量的位置导数（Hellmann–Feynman + 权重的显式导数）：
```
dE'/dR_J = ∂E_KS/∂R_J (显式) + Σ_a μ_a ∂Q_a/∂R_J
```
- 空间权重族：`∂Q_a/∂R_J = ∫ (∂w_a/∂R_J) χ dr`（`WeightGrid` 存 `dw_`）。
- 投影族/DeltaP：保留非局域投影的显式导数；**冻结 `μ`** 时力完备。
- 未收敛时可用 `μ_eff=μ+k(Q−t)` 近似，并报告残差；收敛后用物理 `μ`。

## 1.5 收敛判据与"近零模"问题

- **约束满足**：`max_a |Q_a − t_a| < constraint_thr`。
- **SCF 自洽**：`drho < scf_thr`；两者需**同时**满足（外层 μ + 内层 SCF）。
- **近零模**：响应 `χ_a = dQ_a/dμ_a` 可极小（局部量），使
  - 全局 Lagrange `μ←μ+α(Q−t)` 需 `α<2/χ_max`→对最小 `χ` 极慢（§10：23016 步）；
  - **ALM**（`k=κ/χ`）按模态自适应，给每个模固定收缩率 `1/(1+kχ)`，精确且快。

## 1.6 三种"权重/算符"来源（后文展开）

| 来源 | 形式 | 章节 |
|---|---|---|
| **空间分片** | `w_A(r)`（Becke/Hirshfeld/**MBIS**），`Q=∫wρ` | §5–§8 |
| **混合投影+空间** | `A_i=Q_i+Q_c W_i Q_c`（Allocation），`Q=Tr[ρA_i]` | §7 |
| **算符/onsite** | `Tr[ρ O]`（DeltaSpin `P_Iσ_z`、DeltaP `Γ_I`、Mulliken、d-投影） | §2,§3,§9 |

> 一句话：**约束 DFT = 选可观测量 + 加约束势 + 解近零模 + 正确记账**。后文所有算法都是
> 这四步的不同选择。
# §2 第一代：DeltaSpin（onsite 轨道磁矩约束）

DeltaSpin 是本仓库**最早的自洽约束模块**，也是 2026-05-17 文档（参数统一）的工程主体。
它约束的是**原子 onsite 轨道磁化**，而非空间盆电荷。

## 2.1 可观测量：onsite 轨道磁矩

以原子轨道投影 `P_I` 给出 onsite 磁化：
```
M_I = Tr[ P_I · ρ_diff ] ,   ρ_diff = ρ↑ − ρ↓
P_I = Σ_{μ,ν∈I} |φ_μ⟩ S^{−1}_{μν} ⟨φ_ν|   (或按目标壳层的投影)
```
- 也可写成 per-atom 的轨道分量（如 d 壳层）`M_I = Tr[ρ σ_z O_I]`。
- 这是**投影族**量（依赖基组/壳层选择），**不是**空间盆矩。

## 2.2 约束势与能量

onsite 非局域投影修正进入哈密顿量（collinear 用 `σ_z`）：
```
H ← H + λ · σ_z O_I          (对每个受约束原子 I)
escon = − Σ_I λ_I · M_I       (cal_escon, spin_constrain.cpp:38)
E'    = E_KS + escon
```
- 与空间势 `V↑+=μw, V↓−=μw` **同构**（都是 `σ_z` 结构），但这里是**非局域投影**。
- `escon=−ΣλM` 是**物理能量记账**（扣约束场），这正是后续 MBIS 能量修复参照的范式。

## 2.3 λ 回路与 05-17 参数统一

DeltaSpin 的外层用 `λ`（拉格朗日乘子）迭代，直到 `M_I = m_I^target`。核心工程问题在
**λ 回路何时激活 / 如何与电荷混合共存**：

- **`sc_scf_thr_mode`**（05-17 引入）：
  - `threshold`（默认）：`drho < sc_scf_thr` 时首次进入 λ 回路；
  - `immediate`：从第一个有效波函数迭代（`iter≥2`）即进入（替代旧 `sc_scf_thr=10.0` 魔法数）。
- **`sc_direction_only`**（两阶段）：Phase1 只更新方向、Phase2 之后进入 BFGS；`mixing_restart`
  与 λ 激活时机被显式解耦（避免 Phase1 触发意外 `init_mixing`）。
- **λ 更新**：两阶段（SCF 收敛→一步 λ + `mix_reset`→冻结）或**方向 BFGS/secant**（更快）。
- **约束原子**：由 STRU 的 **`sc|` 掩码**指定；单分量 `mag` 经后处理写 `m_loc_.z`。
- **`sc_mag_switch` / `sc_thr`**：启用与约束阈值。

> 05-17 文档虽是"参数统一"，但它把 λ 回路从"能用但脆弱"变成**可预测、可与 mixing 共存**，
> 是后续所有约束模块（DeltaP/Allocation/MBIS）共用外循环思路的工程起点。

## 2.4 物理机制

- 直接对**原子 onsite 磁化**加"有效磁场"（`λ`），把 TM 的 d 矩驱动到目标；
- 是**投影族**约束：`M_I` 来自原子轨道投影，壳层特定、非空间盆；
- 对**共线**（`nspin=2`）自然；对 SOC/非共线需 `σ` 向量推广（受限）。

## 2.5 实测

- **Fe2**：目标 `±3.8 μB` → onsite 报告 **±3.800**（匹配）；其"传统 orbital-sum"
  仅 **±2.93**（口径差，见 §11）。
- 与 Allocation 在相同 `±3.8` 目标下总能相差 **0.625 eV**（因 `q=Tr[ρA]` 与 onsite 投影
  不是同一算符）。

## 2.6 优劣

**优**
- 直接命中**壳层/onsite 磁矩**（TM d 矩的物理量）；
- 成熟、力/relax 完备（`λ` 冻结时力闭合）；
- `escon=−ΣλM` 能量记账正确（被后续继承）。

**劣**
- **投影族口径**（非空间 AIM）：基组/壳层依赖；
- 已知陷阱：APNS 赝势下多点翻磁分支、TM 体系与 d 通道矩"脱钩"（迹公式/邻域截断近似）；
- `σ_z` 对 SOC/非共线受限；**只约束磁矩**（电荷需另设）；
- `λ` 回路对近零模敏感，需两阶段/BFGS 而非朴素梯度。

## 2.7 在演进中的位置

DeltaSpin 证明了"**可观测量 + λ + escon**"这条范式；DeltaP（§3）把它从**磁矩**推广到
**极化/Berry 相位**，并强化了能量记账与 λ 求解；Allocation/MBIS（§7/§8）则转向**空间
权重/密度自适应**来刻画**电荷**。三者共享 §1 的框架，是同一主线的三个方向。
# §3 第二代：DeltaP（Berry 相位 / 极化约束）

DeltaP 把"约束 DFT"从**磁矩**推广到**电子极化 / Berry 相位**，并**确立、强化**了两件事：
(a) 约束能量的**物理记账** `escon = −ΣλΓ`；(b) 可处理**近零模**的 λ 求解范式。
它是"可观测量可任意线性组合"这一通用约束思想的关键一步。

## 3.1 可观测量：Berry 相位 / 极化

DeltaP 约束的是**原子级 Berry 相位（电子极化）**，支持 per-atom、总约束、任意线性组合。
在当前推荐的 **operator（Route A+）** 模式下，可观测量是**约束算符期望**：
```
Γ_I = Γ_I^HR + Γ_I^HK
```
- `Γ_I^HR`：实空间（real-space）项；`Γ_I^HK`：k 空间 / Hermitian 修正项。
  （实现细节见 DeltaP 设计文档与 `deltap_*` 代码；本文只给结构。）
- 旧模式 `gamma`：直接约束 **Wilson loop Berry 相位** `γ_I`（功能可用，但力/relax 已封口）。
- 与磁矩约束（§2）对照：DeltaSpin 的可观测量是 `Tr[P_I σ_z]`；DeltaP 是可加和的 `Γ_I`，
  本质仍是"**算符期望 + per-atom 分解**"。

## 3.2 约束势与能量

**(a) 约束模式（`deltap_corr 1`）**：在哈密顿量中加入约束项，
```
escon = − Σ_I λ_I · Γ_I
E'    = E_KS(ψ*)           (报告能量 = 物理 KS 能；约束场由 escon 抵消)
```
- 这正是后来 MBIS 能量修复（`cc_escon=−ΣμQ`）所对齐的 **DeltaSpin 形式**。
- **记账恒等式**：`E'=E_KS(ψ*)` 要求 `escon` 精确抵消约束场；否则大可观测量会漂移。

**(b) 场模式**（均匀 `λ`）：所有原子 `λ` 相同，等效于**长程电场**
```
E_eff = + π·λ / (2a)           (实测响应校准 ×1.6)
```
在窗口 `|λ| ≤ 0.007 Ry` 内，能量/极化响应与锯齿场（sawtooth）对拍 **< 1 meV**。
这提供了"约束 ↔ 外场"的物理标定，也说明 λ 的量纲/标定需**响应校准**。

**(c) 仅诊断（`deltap_corr 0`）**：只计算 `γ`/`Γ`，不改哈密顿量——用于取自然极化参考
（靶点应靠近自然极化）。

## 3.3 λ 控制（两阶段 vs 内层 BFGS）

DeltaP 的 λ 更新是"**外层 SCF + λ 内层**"的两级结构，有两种模式：

**两阶段阈值模式（`deltap_inner_nmax=0`，默认）**
```
Phase 1: λ = 0，SCF 自然收敛
Phase 2: 梯度下降更新 λ（deltap_lambda_step），mix_reset()
Phase 3: λ 冻结，SCF 带约束继续收敛
```
- 简单稳健，但**重收敛**成本高（每步 λ 都要重跑 SCF）。

**内层 BFGS 模式（`deltap_inner_nmax=3–5`，推荐）**
```
冻结密度：对 λ 做 BFGS 优化（3–5 次对角化），得到最优 λ
冻结层：λ 固定，SCF 继续收敛
```
- 实测 **3 步内找到最优 λ**，远快于两阶段重收敛。
- `deltap_lambda_init/_init_file`（逐原子初值）、`_step`、`_mixing`、`deltap_inner_thr`、
  `deltap_conv_thr` 控制该内层。

**λ 驱动信号**（`deltap_drive`）：`proxy`（默认，λ 残差驱动 `Γ` 对 `t_Γ`，secant 校准翻译层）
或 `gamma`（λ 直接驱动报告 `γ` 对 `t_γ`）。另有 `deltap_secant` 冻结 `t_Γ*`（relax 必须）。

## 3.4 物理机制

- 约束的是**几何/电子极化**（Berry 相位），与"磁矩""电荷布居"正交，是约束 DFT 的
  **第三种可观测量**；
- **per-atom 可加和** ⇒ 可约束总极化或任意片段/线性组合；
- 通过均匀 λ 可模拟**长程电场**（场模式），把"约束"与"外场"物理打通；
- 能量记账 `escon=−ΣλΓ` 保证报告量是物理 KS 能。

## 3.5 实测/边界

- **场模式**：`E_eff=+πλ/(2a)`，响应校准 **×1.6**；`|λ|≤0.007 Ry` 内 vs 锯齿场 **<1 meV**。
- **operator（Route A+）**：力/relax **完备**（λ 冻结时闭合）。
- **gamma（旧）**：力/relax **封口**（弱耦合 + γ-hold 路径泄漏，见
  `2026-08-13-deltap-capability-boundaries.md`）。
- 依赖 `SMO` 重叠截断（`deltap_rm`）、规范固定（`deltap_gauge_mode` 等）。

## 3.6 优劣

**优**
- **可观测量最通用**（能量/极化/任意线性组合）；
- 能量记账范式（`escon=−ΣλΓ`）被后续全部继承；
- λ 控制成熟（两阶段/内层 BFGS/场模式）；
- 与"外场"物理打通，可标定。

**劣**
- `gamma` 力/relax 不闭合（弱耦合限制）；
- `SMO`/规范（gauge）/锚定细节多，实现复杂；
- `Γ_I^HK` 项与近零模需谨慎处理；
- 聚焦**极化**，非电荷/磁矩（需另设可观测量）。

## 3.7 在演进中的位置

DeltaSpin（§2，磁矩）与 DeltaP（§3，极化）共同证明了"**可观测量 + 约束场 + `escon` 记账 +
λ 内层**"这一通用范式。2026-10 的 Allocation/MBIS 转向**空间权重**来刻画**电荷**，但
**能量记账（§1.3、§3.2 的 `escon`）与近零模求解（§10）**沿用了这两代确立的做法。
# §4 通用约束框架 `module_constraint`

Allocation/MBIS 与 Becke/Hirshfeld 都跑在同一套**实空间权重约束框架**上。本节日志化其
模块地图、数据格式、逐约束通道、能量记账与求解器接入，作为 §5–§9 的公共底座。

## 4.1 模块地图（M 编号，见 `docs/constraint_developer_guide.md`）

| 层 | 编号 | 文件 | 职责 |
|---|---|---|---|
| L0 | M0 | `source_base/module_grid/partition.*` | Becke 数学核：`w_becke_adjusted` + 解析位置导数 |
| L1 | M1 | `weight_grid.*` | 权重网格 `WeightGrid`：构造/缓存/单位分解审计/导数网格 |
| L1 | M2 | `constraint_observe.*` | 读数核 `ConstraintObserver::observe`（逐约束通道） |
| L2 | M3a | `constraint_inject_pw.*` | PW `veff` 注入核（逐约束通道） |
| L2 | M3b | `constraint_inject_lcao.*` | LCAO 侧（隐式经 `v_eff`→`cal_gint_vl` 进 `H`） |
| L3 | M4 | `constraint_loop.*` | 外循环 `ConstraintLoop`：μ 更新、phase、守卫、审计 |
| L3 | M5 | `constraint_accounting.*` | 记账 `escon` + `CONSTRAINT_AUDIT`（key=value） |
| L4 | M6 | `constraint_deriv.*` | 力核 `constraint_force`（PW/LCAO 共用） |
| L5 | M7 | `constraint_io.*` | 输入/靶点 JSON → `ConstraintConfig`/`ConstraintSpec` |
| — | M8 | `alloc_*` | Allocation 专用（投影 + 互补 + 报告） |

**结构性原则**：读/注/记/力四核"**单一实现 + 薄适配**"；逐约束 `channels` 是唯一实现，
没有第二路径（避免语义漂移）。

## 4.2 权重网格（`WeightGrid`）内存布局

| 字段 | 维度 | 含义 |
|---|---|---|
| `w_[iat][ir]` | 原子 × 局域点 | 逐原子权重 `w_A`（`Σ_A w_A=1`） |
| `cw_[alpha][ir]` | 约束 × 局域点 | 片段权重 `w_α=Σ_{A∈α} w_A` |
| `dw_[alpha][(J*3+d)*nrxx+ir]` | 约束×(原子×3)×点 | `∂w_α/∂R_J^d`（力核输入） |

权重类型：`Becke`、`HirshfeldPeriodic`、`MBIS`（密度依赖）、`DipoleSaw`、`DipoleMolecular`。
**单位分解** `Σ_A w_A=1` 逐点审计（`maxdev`，MPI 归约）。

## 4.3 逐约束"通道"（charge / spin）

靶点 JSON v2（推荐）逐约束给出 `type`/`target`/`atoms`：
```json
{"constraints":[
  {"type":"charge","target":0.1,"atoms":[0]},
  {"type":"spin","target":0.1,"atoms":[1],"mu_max":3.0}]}
```
`kind ∈ {Charge, Spin}` ⇒ 工厂 `build_channel_profile(kind)` **唯一**构建通道：

| kind | `read_up` | `read_dn` | `inj_up` | `inj_dn` | 观测量 |
|---|---|---|---|---|---|
| charge | +1 | +1 | +1 | +1 | `∫ w ρ`（总电荷） |
| spin | +1 | −1 | +1 | −1 | `∫ w m`（自旋），需 `nspin=2` |

- 读：`Q_α = ∫ w_α (read_up·ρ↑ + read_dn·ρ↓)`。
- 注：`V_eff[↑] += μ_α·inj_up·w_α`，`V_eff[↓] += μ_α·inj_dn·w_α`。
- **混合 charge+spin 列表**在同一框架内（`kinds_/channels_/mu_caps_` 平行数组）。
- 守卫：`spin` 且 `nspin≠2` → ERROR；`type` 越界 → ERROR。

## 4.4 一步 SCF 内的数据流

```
ψ → elecstate → ρ_out (charge.rho, 与 pw_rhod 同网格)
ConstraintObserver.observe(ρ) → Q_α
ConstraintLoop 更新 μ_α（secant/trust/ALM）
ConstraintInjectPW.inject(V_eff) ← Σ μ_α·inj·w_α
Hamiltonian 用 V_eff（PW）或 cal_gint_vl（LCAO）→ 新 ψ
ConstraintAccounting.audit → escon + CONSTRAINT_AUDIT 行
```
- **ψ 从不进约束模块**；约束读数在 **ρ 网格**上做（PW≡LCAO 同口径的根源）。
- **Hamiltonian 不被直接改**（LCAO 依赖 `v_eff` 的积分线性性 ⇒ `H += Σμ∫φwφ`）。

## 4.5 能量记账（框架内统一）

```
calculate_etot: etot = eband + deband + ... + cc_escon
legacy 空间权重：  cc_escon = − Σ_α μ_α Q_α        (扣约束场, DeltaSpin 形式; S6.72 修复)
allocation：       cc_escon = − Σ_α μ_α t_α        (eband 已含 μQ)
audit：            e_con    = Σ_α μ_α (Q_α − t_α)  (物理泛函, 仅报告/残差)
```
- 修复前 legacy 用 `Σμ(Q−t)`，对**大 Q**（Fe ~14）漏扣 `ΣμQ`（~8 eV）→ **假 BRANCH_FLIP**；
  对 LiCl（Q~2）无害，故旧结论在小 Q 上"看似正确"。
- 分支守卫 `constraint_branch_tol`：约束相 `etot` 比 mu=0 参考低超 tol ⇒ `BRANCH_FLIP`
  （拒绝验证；须用**物理** `etot`，即 `cc_escon=−ΣμQ`）。

## 4.6 求解器接入（外循环）

- `ConstraintLoop` 负责 `μ`：默认 **secant**（`κ=clamp(|dQ/dμ|,kappa_min,kappa_max)`，
  `|Δμ|≤step_max`）+ 可选 **trust-region**（`constraint_trust_region`）；`constraint_mu_schedule`
  可选 `outer`/`inner`/`alm`。
- `constraint_kappa_min`（默认 0.3）：stiff 通道（MBIS `|dQ/dμ|≈0.04`）需 ≈0.01，配合
  `constraint_step_max≈1.5` → 数步收敛（LiCl 3 步）。
- ALM（`k=κ/χ`）在独立模块验证最优（10–34 步），见 §10。

## 4.7 在演进中的位置

该框架把 §2/§3 的"可观测量 + 约束场 + `escon` + λ"**工程化、可插拔**：权重后端可换
（Becke/Hirshfeld/**MBIS**），通道可混（charge+spin），求解器可换（secant/ALM/trust）。
MBIS（§8）只是把权重后端从"固定形状"换成"密度自适应"，其余全部复用本框架。
# §5 空间分片旁支：Becke（固定多边形分片）

Becke 是本仓库最早的**纯几何空间权重**（`source_base/module_grid/partition.*`，M0）。它不依赖
密度，是 Allocation 的默认空间后端、也是 Hirshfeld/MBIS 的对照基线。

## 5.1 公式（Becke 1988 + 异核调整）

两中心：以原子 `I` 与 `J` 的分界平面为基准，定义
```
r_I = |r − R_I| ,   R_IJ = |R_I − R_J|
μ_IJ(r) = (r_I − r_J) / R_IJ  ∈ [−1, 1]      (沿 I→J 的归一化坐标)
```
平滑切换多项式（三次迭代）：
```
p(x) = 0.5·x·(3 − x²)
s(μ) = 0.5·(1 − p(p(p(μ))))                 (s(μ)+s(−μ)=1)
```
多中心权重（**显式归一化**）：
```
p_I(r) = Π_{J≠I} s(μ_IJ(r))
w_I(r) = p_I(r) / Σ_J p_J(r)
片段:  w_a(r) = Σ_{I∈a} w_I(r)
```
异核尺寸调整（不同原子半径/共价半径）：
```
χ_IJ = R_I^cov / R_J^cov
u_IJ = (χ_IJ − 1)/(χ_IJ + 1)
a_IJ = clip( u_IJ/(u_IJ² − 1), −0.5, +0.5 )
μ'_IJ = μ_IJ + a_IJ·(1 − μ_IJ²)              (等半径 a=0 → 标准 Becke)
```
**位置导数**：`∂w_I/∂R_J` 解析给出（`w_becke_adjusted_deriv`），供力核（M6）使用。

## 5.2 为什么必须"显式归一化"

`s(μ)+s(−μ)=1` 只是**成对**恒等式，**不能**推出多中心 `Σ_I p_I = 1`：
三个等半径、等距中心（等边三角形中心）处每个 `s(μ_IJ)=1/2`，
`Σ_I p_I = 3×(1/2)² = 3/4 ≠ 1`；**归一化** `w_I=p_I/Σp_J` 后 `Σ_I w_I=1`。
`WeightGrid::build()` 逐点校验 `Σ_I w_I=1`，并做 MPI 全局归约得审计量 `maxdev`（实测 **0**）。

## 5.3 物理机制

- **纯几何**：`w_I` 只依赖几何（原子位置/半径），**建一次**即可；
- **固定形状**：分片边界由**固定半径**决定，**不随电子结构响应**；
- **平滑**：`s(μ)` 三阶多项式保证权重及导数连续（避免最小镜像切换面跳变）；
- **单位分解精确**：`Σ_A w_A=1` ⇒ `Σ_A N_A=∫ρ` 自动成立。

## 5.4 实测

| 体系 | 量 | 值 |
|---|---|---|
| LiCl（LCAO） | Li 电荷 `∫w_Li ρ` | **3.096 e**（净 −0.096，近中性） |
| bcc Fe2 | 空间盆自旋 `∫w_Fe m` | **+2.285 μB** |
| 任意 | 单位分解 `maxdev` | **0**（实测） |

## 5.5 优劣

**优**
- **便宜**（每几何一次，无密度迭代）；**光滑可微**（导数解析）；
- **单位分解精确**；**确定性强**、无多解；
- 作为 Allocation 的空间后端、Hirshfeld/MBIS 的基线。

**劣**
- **固定形状 ⇒ 不响应离子化收缩**：Li⁺（小）会被"固定大半径"多分 → Li 3.1 e（近中性/
  微阴），**低估离子性**；
- TM 局域 d 自旋偏低（Fe 2.29 < d 投影 2.99），因固定半径抓不住紧凑 d；
- 定量 AIM（电荷/偶极）不宜；适合"快速、稳定、定性分片"。

## 5.6 与其他算法的关系

- 是 **Allocation** 的默认 `W_i`（Becke 互补权重）；
- 与 **Hirshfeld** 同属"固定形状 stockholder/多边形"族（§6），差异只在形状来源；
- 与 **MBIS** 形状相同但**不迭代**——这正是 §11 中 LiCl 3.1 vs 2.04 主因（权重形状，非投影）。
# §6 空间分片旁支：Hirshfeld（周期）

Hirshfeld 是"**stockholder 分片**"：按**预先给定的自由原子参考密度**把空间分给各原子。
周期版（`WeightGrid` 的 `HirshfeldPeriodic`）是本框架中与 Becke 并列的空间后端，MBIS 与它
**同泛函形式**但把参考密度改为**迭代 Slater**（§8）。

## 6.1 公式

给定自由原子（或固定参考）的**球形平均**密度 `ρ_A^free(r)`，原子权重为
```
w_A(r) = ρ_A^free(r − R_A) / Σ_B ρ_B^free(r − R_B)
原子电荷: N_A = ∫ w_A(r) ρ(r) dr
```
- 与 Becke 的差别：Becke 用**多边形切换**；Hirshfeld 用**参考密度之比**（"谁更像谁"）。
- **关键**：`ρ^free` 是**固定的**（非迭代）——这一点决定它与 MBIS 的物理差异（§8.6）。

## 6.2 周期化：C2 窗口 + 余量通道

直接对周期体系用上式会在**最小镜像切换面**处导数跳变。周期 Hirshfeld 的处理：
- **C2 窗口**：把参考密度在原子邻域内平滑截断（`C²` 连续），避免硬切换；
- **晶格镜像求和**：`ρ_A^free(r) = Σ_n ρ_A^free(r − R_A − L n)`（周期延拓）；
- **真空余量通道**（remainder）：在远离所有原子的真空区，用显式"余量"权重接管，
  保证 `Σ_A w_A + w_rem = 1` 且导数连续；
- **商法则导数**：`∂w_A/∂R_J` 由参考密度的解析导数经商法则给出，覆盖周期边界。

## 6.3 物理机制

- **stockholder 分片**：把每个空间点分给"该点参考密度最大"的原子（按比例）；
- **固定参考**：`ρ^free` 不随分子/晶体密度变化 → **不响应离子化/成键导致的收缩**；
- **多极可扩**：参考密度可含多极（提升偶极等），但本实现以球形平均为主。

## 6.4 实测

| 测试 | 结果 |
|---|---|
| 单位分解 `HirshfeldPeriodicPartitionAndRemainder` | `maxdev < 1e-12` |
| 权重及余量的解析位置导数求和 | 通过（含全覆盖区域） |
| 周期边界导数（原子分数坐标 `x=0.98`） | 通过 |
| `ConstraintLoop` 到达（`HirshfeldWeightTypeReachesLoop`） | 通过 |
| LiCl Li 电荷 | 与 Becke 同族偏差（近中性，远高于 MBIS 2.04） |

（周期 Hirshfeld 的 C2 窗口/镜像求和/商法则导数满足设计；余量通道在真空区显式接管，
避免 Becke 最小镜像切换面处的导数跳变。）

## 6.5 优劣

**优**
- 比 Becke 更有"原子密度"意义（stockholder）；**光滑**、**周期一致**；
- **真空余量**稳健（大真空/分子体系友好）；**便宜**（参考固定，建一次）；
- 解析导数完备（含周期边界）。

**劣**
- 参考密度**固定** ⇒ 与 Becke **同类偏差**：**低估离子性/不响应收缩**（Li 偏中性，
  Fe 局域矩偏低）；
- 依赖自由原子参考的选取（多极/球形、基组）；
- 定量 AIM（离子性/偶极）不宜；适合快速稳定分片。

## 6.6 与 MBIS 的关系（核心）

Hirshfeld 与 MBIS 是**同一公式形式** `w_A ∝ ρ_A^pro/Σρ_B^pro`、`N_A=∫w_Aρ`；
唯一区别是 **`pro_A` 的宽度是否随密度自洽**：
- **Hirshfeld**：`ζ`（宽度）**固定**（自由原子）⇒ 不响应离子化；
- **MBIS**：`ζ_A=3N_A/⟨r⟩_A` **随 ρ 迭代** ⇒ Li⁺ 收缩（ζ≈4.6）⇒ 离子性正确。

**因此"固定形状 vs 密度自适应"= Hirshfeld/Becke vs MBIS 的本质差异**，也是 §11 中
LiCl Li 3.1 vs 2.04 的主因（权重的**形状**，而非投影算符）。
# §7 第三代：统一局域矩分配（Allocation）

Allocation 是 2026-10-21～29 的主体：把"**目标壳层投影**"与"**互补空间权重**"混合，
建立可精确求和的约束坐标；同时落地了 §4 的通用 `WeightGrid/ConstraintLoop/通道/ALM` 框架。

## 7.1 约束坐标（公式）

```
A_i = Q_i + Q_c W_i Q_c ,      P = Σ_i Q_i ,   Q_c = I − P
q_i = Tr[ρ A_i] = Tr[ρ Q_i] + Tr[ρ Q_c W_i Q_c]        (约束坐标)
```
- `Q_i`：原子 i 上**目标 NAO 轨道（角动量 l）的正交投影**；
- `W_i`：实时**空间权重**（Becke，或固定 Gaussian 的周期 Hirshfeld）；
- `Q_c = I − Σ_i Q_i`：**互补子空间**（目标壳层之外）。
- 求和：`Σ_i q_i = Tr[ρ(P + Σ_i Q_c W_i Q_c)]`；在 `Σ_i W_i=1`（单位分解）且投影完备时精确。

## 7.2 报告量（易与约束坐标混淆）

代码（`alloc_lcao_report.cpp`）报告三个量：
```
n_tot,i = Tr[ρ W_i] = Σ_ir w_i(r) ρ(r) dV      (纯空间权重电子数; 与 target_l 无关)
n_T,i   = Tr[ρ Q_i]                             (目标壳层投影占据; 诊断)
n_c,i   = n_tot,i − n_T,i                        (报告的"差"; 不是算符互补)
```
**关键事实**：约束算符的互补项 `Tr[ρ Q_c W_i Q_c]` **≠** `n_tot − n_T`。已定量证实：
Fe2 上**算符** `m_c=−0.20` vs **报告** `m_c=+0.78`（符号/数值皆不同）。因此：

> **不要把 `q_i=Tr[ρA_i]` 当作净电荷**；它也是**建模坐标**。LiCl 上 `q=3.083` 实际接近
> Becke 的空间电子数 `3.096`（§11），而非净电荷（近中性）。

## 7.3 三条差异来源（按贡献排序）

以 LiCl 的 Li 为例（`2026-10-30-allocation-vs-mbis-formula-analysis.md`）：

**(A) pro/权重宽度（主因，~1 e）**
固定 Gaussian/Becke 的 `w^fixed`（Li 半径偏大）vs MBIS 迭代 Slater（Li⁺ 紧凑 ζ≈4.6）：
```
q^alloc − q^MBIS = ∫ [ w^fixed(r) − w^MBIS(r) ] ρ(r) dr ≈ 1.04 e
```
Li 附近 `w^fixed > w^MBIS` → 多分 ~1 e。

**(B) 目标壳层投影 `Q_i`（语义/基组依赖）**
`Q_i` 只重分配 `n_T/n_c`，**不改变总量**（`n` 与 `target_l` 无关：l=0/1/2 均为 `3.0833`）。
它影响**约束算符**（非纯权重），但对报告的净电荷无贡献。

**(C) 算符互补 vs 报告差**（§7.2）——语义闭环不一致。

> 结论：Allocation 偏离 MBIS 的 ~1 e 主要是 **(A) 权重形状**，与"投影到壳层"无关。

## 7.4 工程化（2026-10-27～29 的优化与修复）

Allocation 阶段完成了大量**性能与一致性**工作（文档见 `2026-10-27..29-allocation-*`）：
- **子空间**：`exact-subspace`、`subspace-iteration-sweep`、`subspace-warm-start`；
- **性能**：`band-batching`、`fft-buffer-trim`、`per-k-cache`；
- **MPI**：`s5-pw-cache-mpi`、`kpar-force`；
- **正确性**：`complex-k-conjugation-fix`、`charge-normalization-root-cause`；
- **响应/收敛**：`fd-response-layer`、`outer-efficiency-fold`、`trust-region`；
- **对齐**：`ds-mulliken-alignment`。

这些把框架打磨到可生产（`ConstraintLoop`/通道/ALM 即在此成熟），并**直接复用于 MBIS**。

## 7.5 物理机制

- **混合投影+空间**：把"壳层特定"（`Q_i`）与"空间互补"（`Q_c W_i Q_c`）结合，
  使求和精确且可"重分配目标壳层占据"；
- 是**建模坐标**（非净电荷、非纯空间盆），语义随 `target_l` 与 `W_i` 变；
- `W_i` 固定（Becke/Gaussian）⇒ **不响应离子化**（同 §5/§6 偏差）。

## 7.6 实测

| 体系 | 量 | 值 |
|---|---|---|
| LiCl（LCAO） | `q_i` / `n_tot`（Becke） | **3.0833** / ≈3.096（≈Becke，非净电荷） |
| LiCl | l 依赖（l=0/1/2 的 `n`） | 均 **3.0833**（不变；只 `n_T/n_c` 变） |
| Fe2 | `m_T`（原子 0/1） | ∓**3.599/3.600**；orbital-sum **±2.93** |
| Fe2 vs DeltaSpin @±3.8 | 总能差 | **0.625 eV** |

## 7.7 优劣

**优**
- **求和精确**、可定向"目标壳层"重分配（`n_T/n_c`）；
- 通用框架（`WeightGrid`/通道/ALM）在此成熟，直接复用；
- 工程优化充分（子空间/MPI/缓存/信任域）。

**劣**
- `q` 是**建模坐标 ≠ 净电荷**（易误用；LiCl 3.08 vs MBIS 2.04 是定义差）；
- **算符互补 ≠ 报告差**（`m_c` 符号/数值不一致）⇒ 语义闭环需谨慎；
- 权重固定 ⇒ 不响应离子化；每外步重建、成本高。

## 7.8 在演进中的位置

Allocation 的价值有二：(1) 证明了"**混合投影+空间**"的可行性并暴露其**语义陷阱**；
(2) 沉淀出 §4 的**通用框架**。MBIS（§8）正是在**同一框架**里把 `W_i` 换成**密度自适应**
的 Slater stockholder，从而修正 (A) 类偏差、给出净电荷/偶极正确的约束。
# §8 第四代：MBIS（迭代 stockholder，当前电荷/自旋约束）

MBIS（Minimal Basis Iterative Stockholder）是本方案的核心：把 §6 Hirshfeld 的**固定参考**
换成**随密度自洽迭代**的 Slater pro，从而同时获得**空间 AIM 的求和精确性**与**对离子化/
局域化的物理正确响应**。它接入 §4 的通用框架，作为 `constraint_weight_type mbis`。

## 8.1 定义（Slater-s，密度自适应）

每个原子 A 的 pro-density 与权重：
```
pro_A(r) = N_A · ζ_A³/(8π) · exp(−ζ_A · d_min(r, R_A))       (Slater-s)
w_A(r)   = pro_A(r) / Σ_B pro_B(r)                            (Σ_A w_A=1)
N_A = ∫ w_A(r) ρ(r) dr                                        (布居)
ζ_A = 3 N_A / ∫ r w_A(r) ρ(r) dr = 3 / ⟨r⟩_A                  (均值半径)
自旋:  M_A = ∫ w_A(r) m(r) dr ,   m = ρ↑ − ρ↓
```
更新用阻尼（`damp`）迭代至 `|ΔN|<tol`。收敛时 `Σ_A w_A=1` ⇒ `Σ_A N_A = ∫ρ`（**求和规则
自动**）。`d_min` 为**周期最小镜像**距离（见 §8.4）。

## 8.2 求解语义：**约束不动点**，不是自由 D 最小化（重要更正）

- 早期曾把 MBIS 理解为"最小化 KL 目标 `D(p)`、多初值取最低 D"。**已更正**：
  MBIS 是**带布居约束的自洽不动点** `p = F(p)`，其中 `F` 为 Picard 一步
  （`N_A=∫w_Aρ, ζ_A=3N_A/⟨r⟩`）。`F` 天然保 `ΣN=∫ρ`。
- **`D` 是退化的**：`ζ→0`（pro 均匀，`w_A=1/nat`）时 `D=Σ(ρ/nat)ln((ρ/nat)/(ρ/nat))=0`，
  是**全局最小** ⇒ **"最低 D 选解"无效**（会把平凡均匀解选中）。`D` **仅作诊断**。
- **规范求解器** = 物理起点（`ζ~1`）的 **Picard 不动点**（稳定）。
- **稳定性判据** = 不动点映射的**真实谱半径** `ρ(∂F/∂p) < 1`
  （`mbis_fixedpoint_spectral_radius`，Faddeev–LeVerrier + Durand–Kerner）。
  `‖∂F/∂p‖₂`（奇异值上界）**过松**，曾误判"不稳"（§14）。
- `compute_mbis_newton_fp`（约束 Newton）为诊断；退化（`ζ<0.05`/`D≤0`）时**回退 Picard**。

## 8.3 周期：`constraint_mbis_images` 的 0/1 语义

代码（`image_translations`）：`k` → 晶格平移壳层半径 `R=k`，共 **`(2k+1)³`** 个平移。
```
images = 0 → {0}                (仅 home cell, direct pro; 只适合大盒)
images = k → [-k,k]³ 晶格平移    (含最近/更远周期像; ≈ 最小镜像)
```
- **小胞必须 ≥1**：否则 direct pro **漏掉原子自身周期像**，corner 处严重低估（见 §8.4）。
- **大盒**（原子远离边界）用 `0` 即可且更快（LiCl）。

## 8.4 真正根因：direct-pro 的 corner 低估 → min-image 修复（S6.63）

- **病理**：直接 `pro_A=e^{−ζ|x−R_A|}` **忽略原子自身周期像**；bcc 小胞的 **corner** 原子
  密度是 **8 重周期像之和**，direct 只计 1 份 ⇒ corner 原子 pro **被严重低估** ⇒ `w` 偏向
  另一原子 ⇒ 落到**极端不对称**（Fe2：`N=0.48/31.52`；FeO：O `1.4/8.7`、Fe `0.4/33.5`）。
- **错误尝试**：`coset`（半格矢）图像和（对 Z2 重复计数）、截断晶格和（扩散瞬态收敛慢）、
  同种原子 `symmetrize`（不通用，已移除）——均未抓住根因。
- **修复（min-image）**：`pro_A(r)=N_Aζ_A³/8π·exp(−ζ_A·d_min(r,R_A))`，`d_min` 为**最近周期像**；
  等价原子（`R1=R0+g`，`2g` 格矢）满足 `d_min(x,R1)=d_min(x−g,R0)` ⇒ **精确对称**。
- **均径 `r` 用与 pro 相同的 images 集**（`images=0`→direct；`≥1`→min-image），保 ζ 对称。
- **结果**：bcc Fe2（`images=1`）**对称稳定** `N=16.00/16.00`、`M=±2.85`、`ρ≈0.31`。

## 8.5 力：光滑晶格和 pro（S6.64）

min-image pro 在 **wrap 边界不连续** ⇒ 解析力（光滑）与能量 FD 不一致。**力**改用
**光滑晶格和** `pro_A=Σ_T exp(−ζ_A|x−R_A−T|)`（对原子位置可微），解析导数与 FD 一致
（`ForceFiniteDifference` 恢复）；**观测/迭代**仍用 §8.4 的 pro/r。

## 8.6 能量记账：`cc_escon = −ΣμQ`（S6.71/72）

约束势注入 `V_eff` 后，`eband+deband` 仍带约束场能量 `∫V_con ρ=ΣμQ`。报告/守卫须扣除：
```
legacy：cc_escon = −Σ_α μ_α Q_α     (DeltaSpin −ΣλM 形式)
audit ：e_con    = Σ_α μ_α(Q_α−t_α)  (物理泛函, 仅报告/残差)
```
- **原状（错误）**：legacy `cc_escon=Σμ(Q−t)` **漏扣 `ΣμQ`**；对**大 Q**（Fe ~14，`μQ≈0.6 Ry≈8 eV`）
  `FINAL_ETOT` 被低估数 eV → **假 `BRANCH_FLIP`**（FeO 曾 −7643.9 vs 参考 −7637）。
- **修复后**：FeO `FINAL_ETOT` 回 **−7636.2 eV**（≈参考），`de=+0.087 Ry>0`（约束升能，物理
  正确），**假 flip 消失**。LiCl（Q~2）不受影响。

## 8.7 主循环：MPI gather（S6.67）

`build_mbis` 在**并行 pool** 上：按 **z-最快**布局
（全局 `ig=ixy*nz+iz`，本地 `ir=ixy*nplane+iz_local`）用 `MPI_Allgatherv` + 重排 **gather**
本地密度 → 全局 `compute_mbis` → 再 **scatter** 权重回本地。验证 `-np 1/2/4` 结果**完全一致**。

## 8.8 外循环：Picard + secant（快速 μ）

- 主循环用 Picard（`compute_mbis`）+ **secant/trust**（`κ=clamp(|dQ/dμ|,kappa_min,kappa_max)`，
  `|Δμ|≤step_max`）。
- MBIS 的 `|dQ/dμ|≈0.04`（stiff）：默认 `kappa_min=0.3, step_max=0.05` 需 ~28 外步；
  **`constraint_kappa_min≈0.01` + `constraint_step_max≈1.5`** ⇒ **LiCl 3 外步 CONVERGED**。
- 独立模块验证 **ALM（k=κ/χ）** 最优（10–34 步，精确），见 §10。

## 8.9 盆内 l 投影（壳层分解；S6.65 前后）

MBIS 空间盆给**盆内总自旋**（含原子间/非 d 分量），不等于 d 投影。**在 MBIS 盆内再做 l 投影**：
```
m_{A,l} = ∫ w_A(r) m_l(r) dr    (或加权重叠 Tr[w_A φφ]D，精确版待 C++ 落地)
```
在 **MBIS 平均半径 R=3/ζ≈2.78** 处，Fe2 的 **d 分量 `≈3.069`**，与 **Mulliken 3.034**、
**d-投影 2.985** 高度一致 ⇒ 支持"**空间 AIM + 盆内 l 投影**"两步架构。

## 8.10 物理机制

- **stockholder + 迭代宽度**：`ζ_A` 随密度自洽 ⇒ 离子化收缩（Li⁺ ζ≈4.6）、TM d 局域化，
  **复现离子性/偶极/AFM 局域矩**；
- **空间 AIM**：`ΣN=∫ρ`、可加和、可片段化；
- **约束不动点**：区别于自由 D 最小化（无退化选解问题）；
- **可扩壳层**：盆内 l 投影给出 d 贡献（与投影族一致）。

## 8.11 实测

| 体系 | 量 | 值 |
|---|---|---|
| LiCl（LCAO） | Li `N` / 偶极 | **2.0447 e**（净 +0.955）/ **6.48 D**（实验 ~6.3） |
| bcc Fe2（cube） | `N` / 空间 `M` / ζ | 16.00/16.00 / ±2.647 / 1.08 |
| bcc Fe2（min-image .50） | `N` / `M` / ρ(∂F/∂p) | 16.00/16.00 / **±2.85** / **≈0.31（稳定）** |
| Fe2 盆内 l（R=2.78） | d 分量 | **3.069** ≈ Mull 3.034 ≈ 投影 2.985 |
| 主循环 e2e（LiCl） | 收敛 | 3 外步 CONVERGED；np=1/2/4 一致 |
| FeO（charge+spin，无 U） | 参考 Q / 能量 | Fe 14.27、M 2.85（与独立一致）；能量 `−ΣμQ` 后自洽；**但 PBE 无 U 磁性不稳需 DFT+U** |

## 8.12 优劣

**优**
- **物理最合理**（离子性/偶极/AFM 局域矩）、**通用**（任意元素/体系）、**可扩壳层**（盆内 l）；
- 空间 AIM 求和精确；与 ORCA MBIS 同族；
- 已在 LiCl/Fe2/e2e/MPI 上验证；能量记账已修。

**劣 / 工程代价**
- **密度依赖 → 每外步迭代**（含 warm start）；
- **周期小胞必须 min-image**（`constraint_mbis_images≥1`）；主循环默认 `0` 只适合大盒；
- **大 Q 能量**需 `−ΣμQ`（已修）；
- 外循环较慢，需 secant/`kappa_min`（或 ALM）；
- 盆内 l 的**精确权重重叠版**待 C++ 落地（当前径向平均近似）。

## 8.13 在演进中的位置

MBIS 是 §5/§6（固定形状空间分片）与 §2/§3（算符期望约束）的**融合产物**：
- 采用**空间分片**的求和精确/可加和（Becke/Hirshfeld 血统）；
- 采用**密度自适应**（比 Hirschfeld 多一步自洽）获得离子性（§6.6）；
- 复用 §4 的**通用框架**与 §2/§3 的**能量记账/λ 求解**；
- 通过**盆内 l** 与投影族（Mulliken/d-投影/DeltaSpin，§9）桥接。

它是当前**电荷/自旋约束的推荐方案**（配合 min-image、`−ΣμQ`、secant/ALM、DFT+U 做磁性体系）。
# §9 投影旁支：Mulliken / d-投影

这两类属于"**轨道/壳层投影**"（`Tr[ρ O]`），与 §2 DeltaSpin、§8 盆内 l 同族，用作**壳层
可分辨**的对照。它们**不是空间盆**，与 MBIS/Becke/Hirshfeld 的口径不同。

## 9.1 Mulliken（轨道布居）

**公式**（Löwdin/对称化变体类似）：
```
q_A = Σ_{μ∈A} (D S)_{μμ}          (D 密度矩阵, S 重叠矩阵)
     = Tr[ ρ · P_A ] 的 Mulliken 分摊（跨原子重叠对称平分）
```
- 本质：把**基函数**按原子归属分配，跨原子重叠 `S_{μν}` 对称摊到两端。
- **自旋**：`m_A = Σ_{μ∈A}(D^↑−D^↓)S_{μμ}`。

**机制**：**非空间**、**基组依赖**（基组越大/越弥散，分摊越不同）；无"基组极限"的确定值。

**实测**：
| 体系 | 量 | 值 |
|---|---|---|
| LiCl | Li `N` | **2.483**（净 +0.517） |
| Fe2 | 自旋 | **3.082** |

**优**：极便宜、通用、常见。
**劣**：**强基组依赖**、跨原子重叠分摊任意、无空间盆意义 → **不宜作 AIM/定量电荷**；
LiCl 明显低估离子性（+0.52 vs MBIS +0.96）。

## 9.2 d-投影（壳层矩）

**公式**：
```
m_T = Tr[ ρ σ_z Q_{A,l} ] ,   Q_{A,l} = 原子 A 上角动量 l 的正交投影
```
- 只取该原子**目标壳层（如 d）**的贡献；`l` 可 s/p/d/f。
- **自旋**：`m_{A,d} = Tr[(D^↑−D^↓) Q_{A,l}]`。

**机制**：**壳层特定**——物理上对应 TM 的 d 矩；与 DeltaSpin（`P_Iσ_z`）同族。

**实测**：
| 体系 | 量 | 值 |
|---|---|---|
| Fe2 | `m_T`(d) | **2.985** |
| Fe2 | 与 Mulliken(3.082)/盆内 l(3.069) | 一致量级 |

**优**：直接给**壳层矩**（TM d 矩）；与 DeltaSpin 同口径。
**劣**：只覆盖壳层（忽略 s/p 及原子间分量）；投影求和/邻域截断近似；基组依赖。

## 9.3 与 MBIS 的关系（桥接）

- MBIS **空间盆**给出**盆内总自旋**（含原子间/非 d）：Fe2 ±2.65（< d 投影）。
- **盆内 l 投影**（§8.9）在 MBIS 盆上再做 `∫w_A m_l`：Fe2 **d≈3.069 ≈ Mulliken 3.034
  ≈ d-投影 2.985**。
- 因此三者一致并非偶然：**"空间 AIM（MBIS）+ 盆内 l"** 是可同时保留"空间求和精确"与
  "壳层可分辨"的两步架构；Mulliken/d-投影是**纯投影口径**的近似对照。

## 9.4 口径注意

- **跨方法不可直接比数**：Mulliken 3.08、d-投影 2.99、MBIS 盆 2.65、盆内 l 3.07 是**不同
  定义**（基组/壳层/空间盆）；
- 定量 AIM（离子性/偶极）**用 MBIS**；壳层矩**用 d-投影/DeltaSpin 或 MBIS 盆内 l**；
- **勿**把 Mulliken 当净电荷、**勿**把空间盆矩当 d 矩。
# §10 约束求解器（μ / λ 的更新）

约束 SCF 的外层是解 `Q_a(μ)=t_a`。因响应 `χ_a=dQ_a/dμ_a` 可极小（**近零模**），朴素梯度
极慢；各代采用不同求解器。本节给公式、收敛、偏置与选择。

## 10.1 全局 Lagrange（梯度下降）

```
μ_a ← μ_a + α_a (Q_a − t_a) ,   α_a < 2/χ_max
```
- 收敛率由**最小 χ**决定（`1−αχ`）；对 `χ=0.01` 类极端慢。
- **实测（线性模型）**：**23016 步**（χ=0.01 主导）。
- **优**：简单、无参数结构；**劣**：被近零模拖死。

## 10.2 增广拉格朗日（ALM，推荐）

**内层**（固定密度，解 `μ`）：
```
μ_eff,a = μ_a + k_a (Q_a − t_a)
```
**外层**（每 SCF 收敛后）：
```
μ_a ← μ_a + k_a (Q_a − t_a) ,   k_a = κ / χ_a  (按模态自适应)
```
- 每模收缩率 `1/(1+k_a χ_a) = 1/(1+κ)`：**与 χ 无关**，近零模也被固定速率收敛。
- **实测**：`κ=1` → **34 步**；`κ=10` → **10 步**；**精确**（`Q=t` 可达 1e-11）。
- `damping 1/(1+k|χ|)` 用于步内势迭代，防 2-周期。
- **优**：**快 + 精确 + 自适应近零模**；**劣**：需响应测量/`χ` 标定，`κ` 大时内层变硬。

## 10.3 纯罚（penalty-only）

```
μ_eff,a = k_a (Q_a − t_a)
```
- 单步即"收敛"，但**有偏**：
```
Q* − t = (Q0 − t) / (1 + k χ)     (k=κ/χ 时 = (Q0−t)/(1+κ))
```
- 实测：`k=1/χ` 残差 **0.50**；`k=10/χ` 残差 **0.091**；**永不精确**。
- **用途**：仅作"先粗收敛再切 ALM"的**预条件**，不能作为最终约束。

## 10.4 secant / trust-region（DeltaSpin/DeltaP/μ_solver，MBIS 主循环）

```
κ_a = clamp( |dQ_a/dμ_a|, kappa_min, kappa_max )     (secant 斜率, 下限防死通道)
Δμ_a = − (Q_a − t_a) / κ_a ,   |Δμ_a| ≤ step_max    (首步可用 step_probe)
```
- **trust-region**（`constraint_trust_region`）：按 `ρ=实际/预测残差下降` 接受/回滚：
  `ρ>0.5` 加倍半径（≤1 Ry），`ρ<0.1` 减半并回滚。
- **坑（S6.68）**：`kappa_min=0.3` 对 stiff 通道（MBIS `|dQ/dμ|≈0.04`）**高估响应** ⇒ 步过小，
  再被 `step_max=0.05` 截断 ⇒ 线性慢（~28 步）。
  **修复**：暴露 `constraint_kappa_min`；stiff 通道 `≈0.01` + `constraint_step_max≈1.5`
  ⇒ 用**实测坡度** Newton 跳跃，**LiCl 3 外步**。
- **优**：无需显式 `χ` 标定、鲁棒（trust）、可与 SCF 交替；**劣**：`kappa_min/step_max` 需按通道调，
  大 Q 需能量记账正确（§8.6）。

## 10.5 BFGS（DeltaP/DeltaSpin 内层）

- DeltaP 内层：**冻结密度**，对 λ 做 **BFGS**（3–5 次对角化）→ 3 步内最优 λ（§3.3）；
- DeltaSpin：两阶段/方向 BFGS（§2.3）。
- **优**：少对角化、对多维 λ 耦合好；**劣**：需冻结密度、Hessian 近似。

## 10.6 对比与选择

| 求解器 | 公式核心 | 收敛（实测） | 精确 | 适用 |
|---|---|---|---|---|
| Lagrange | `μ←μ+α(Q−t)` | 23016 | 是 | 不推荐 |
| **ALM** | `k=κ/χ` 内外层 | **10–34** | **是** | **独立模块/无 SCF 耦合** |
| penalty | `k(Q−t)` | 1（有偏） | **否** | 仅预条件 |
| **secant+trust** | `κ=clamp(dQ/dμ)` | LiCl 3 外步 | 是（收敛后） | **主循环（MBIS/DeltaSpin）** |
| BFGS | 冻结密度优化 λ | 3 步内层 | 是 | DeltaP 内层 |

**推荐**：主循环用 **secant + trust**（`kappa_min` 按通道自适应，stiff 用 0.01）；无 SCF 耦合
或需要保证精确收敛时用 **ALM（κ≈1–10）**；penalty 仅作预条件。**统一前提**：能量记账必须
扣约束场（`cc_escon=−ΣμQ`），否则大 Q 会假发散/假 `BRANCH_FLIP`。
# §11 统一数值对比

所有数字取自本仓库实测文档（粗网格/旧胞者已标注）。**跨方法不可直接比数**——此处并列意在
展示**口径差异**，而非"谁对谁错"。

## 11.1 LiCl（同一自由密度，LCAO）— 电荷 / 偶极

| 方法 | Li `N`(e) | Li 净电荷 | 分子偶极 |
|---|---|---|---|
| 气相实验 | — | — | ~6.3 D |
| **MBIS** | **2.0447** | **+0.955** | **6.48 D**（含原子偶极） |
| Mulliken | 2.4830 | +0.517 | 5.0 D（点电荷） |
| Allocation（Becke `n`） | 3.0833 | −0.083 | 不适用（`q` 非电荷） |
| Becke 空间 | 3.0959 | −0.096 | — |

**读法**：Li⁺ 离子性 `MBIS(+0.96) ≫ Mulliken(+0.52) ≫ Becke≈Allocation(≈0)`；偶极 MBIS 最接近
实验。差 ~1 e 的**主因是权重形状**（固定半径 vs 迭代 Slater 收缩），非投影（§7.3）。

## 11.2 Fe2（同一自由密度）— 自旋四口径 + 盆内 l

| 口径 | 自旋 (μB) | 说明 |
|---|---|---|
| Becke 空间盆 | 2.285 | 固定半径 |
| **MBIS 空间盆** | **2.647**（cube）/ **2.85**（min-image 0.50） | ζ≈1.08；密度自适应 |
| d-投影 `m_T` | 2.985 | 壳层 l=2 |
| Mulliken | 3.082 | 轨道布居 |
| **盆内 l（MBIS 盆，R=2.78）** | **d=3.069** | ≈ Mulliken 3.034 ≈ 投影 2.985 |

**排序**：`Becke 2.29 < MBIS 2.65 < d-投影 2.99 < Mulliken 3.08`；MBIS 空间盆落在 Becke 与
d-投影之间；**盆内 l → d≈3.07**，与 Mulliken/投影一致。电荷：MBIS `16.00/16.00`（中性 Fe）。

## 11.3 约束/极化/分配（口径与能量）

| 项 | 值 |
|---|---|
| DeltaSpin Fe2：目标 `±3.8` → onsite 报告 | **±3.800**（匹配）；orbital-sum ±2.93 |
| Allocation Fe2：`m_T` / orbital-sum | ∓**3.599/3.600** / ±2.93 |
| Allocation vs DeltaSpin @±3.8 总能差 | **0.625 eV** |
| DeltaP 场模式 | `E_eff=+πλ/(2a)`，响应校准 **×1.6**；`|λ|≤0.007 Ry` 内 vs 锯齿场 **<1 meV** |

## 11.4 收敛 / 成本

| 方法 | 每步成本 | 周期需求 | 收敛（实测） |
|---|---|---|---|
| Becke / Hirshfeld | **低**（建一次） | 好（Hirshfeld 真空余量） | — |
| Allocation | 高（每外步重建+投影） | 好 | Fe2 收敛（自身定义） |
| DeltaSpin | 中（非局域投影） | 好 | Fe2 匹配 ±3.8 |
| DeltaP | 中 | 好 | 内层 BFGS 3 步 |
| **MBIS** | 中高（每外步迭代+min-image） | **小胞需 min-image** | **LiCl 3 外步；Fe2 对称稳定 ρ≈0.31** |
| Lagrange/ALM/penalty | — | — | 23016 / **10–34** / 1(偏) |

## 11.5 一句话总结

- **电荷/偶极**：MBIS 最接近物理（Li 2.04、6.48 D）；
- **自旋**：空间盆（Becke 2.29 / MBIS 2.65）< 壳层投影（d 2.99 / Mulliken 3.08）；
  **盆内 l 桥接**（d≈3.07）；
- **磁矩约束**：DeltaSpin onsite ±3.8（口径 ≠ 空间盆）；
- **求解器**：ALM/secant 远优于 Lagrange，penalty 有偏。
# §12 物理机制对比（核心差异）

数值差异背后是四个**物理/数学机制**的分野。理解它们即可解释"为什么 2.04/2.48/3.08/2.29/2.99"
这些数，以及何时该用哪种。

## 12.1 固定形状 vs 密度自适应（决定**离子性**）

- **固定形状**（Becke 固定半径、Hirshfeld 固定自由原子参考）：`w_A` 只依赖几何/参考，
  **对离子化收缩无反馈** —— Li⁺（小）仍被大半径多分 → Li 3.1 e（近中性）、TM d 局域矩偏低。
- **密度自适应**（MBIS 迭代 `ζ_A`）：离子化使 `ζ_A` 收缩（Li⁺ ζ≈4.6），`w_A` 随之收紧 →
  Li 2.04 e、偶极 6.48 D、AFM 局域矩合理。
- **定量**：LiCl `∫(w^fixed−w^MBIS)ρ ≈ 1.04 e` —— 这正是 3.08 vs 2.04 的来源。
- **结论**：要**离子性/偶极**，必须**密度自适应**（MBIS）；固定形状只适合定性分片。

## 12.2 空间盆 vs 壳层投影（决定**自旋语义**）

- **空间盆**（Becke/Hirshfeld/MBIS）：把空间点分给原子，`M_A=∫w_A m` 是**盆内总自旋**
  （含原子间、非 d 分量）→ Fe2 MBIS 2.65（< d 投影）。
- **壳层投影**（Mulliken/d-投影/DeltaSpin）：`Tr[ρ O_l]` 只取**壳层** → d 2.99 / onsite ±3.8。
- **桥接**：**MBIS 盆内再做 l 投影**（`∫w_A m_l`）→ d≈3.07，同时保留空间 AIM（可加和、
  求和精确）与壳层信息。
- **结论**：说"Fe 的 d 矩"要用投影/盆内 l（3.0）；说"Fe 盆内总自旋"用 MBIS 2.65；二者**不是
  同一量**，不可混用。

## 12.3 算符/口径一致性（决定**可用性**）

- **Mulliken**：基组依赖、无基组极限 → 不宜 AIM。
- **Allocation**：`q=Tr[ρA]` 是**建模坐标 ≠ 净电荷**；且**算符互补 ≠ 报告差**
  （Fe2 `m_c` 算符 −0.20 vs 报告 +0.78）→ 语义闭环需谨慎。
- **DeltaSpin**：onsite 投影口径（±3.8）≠ 空间盆（2.65）。
- **结论**：**跨方法比较必须在同一 `Q` 定义下**；LiCl 3.08 vs 2.04 是**定义差**，不是数值误差。

## 12.4 能量记账（决定**长时程/大体系稳定**）

- 约束势 `V_con` 进 `V_eff` 后，电子总能含 `∫V_con ρ=ΣμQ`；报告/守卫须**扣除**：
  `escon=−ΣμQ`（DeltaSpin 形式）。
- **未扣**（旧 legacy `Σμ(Q−t)`）对**大 Q**（Fe ~14，`μQ≈8 eV`）→ 假 `BRANCH_FLIP`；
  小 Q（LiCl ~2）误判不明显，故旧结论"看似正确"。
- **结论**：能量记账是**正确性前提**，不是可选优化；大布居体系尤其关键。

## 12.5 近零模与求解（决定**收敛**）

- 响应 `χ=dQ/dμ` 可极小（局部量）→ 全局 Lagrange 被拖死。
- **ALM**（`k=κ/χ`）给每模固定收缩 `1/(1+κ)`，精确快；**secant+trust** 用实测坡度 +
  自适应 `kappa_min`。
- **penalty** 快但**有偏**（永不精确），仅预条件。
- **结论**：约束算法 = **可观测量 + 约束场 + 正确记账 + 近零模求解**四要素缺一不可。

## 12.6 规范 / 基组 / 周期（工程约束）

- **周期小胞**：空间 pro 必须 **min-image**（否则 corner 低估→破缺，§8.4）。
- **基组**：投影族（Mulliken/DeltaSpin）基组依赖；空间盆（Becke/Hirshfeld/MBIS）无基组依赖。
- **规范**（DeltaP）：Berry 相位需规范固定/锚定；`gamma` 模式力/relax 不闭合。
- **SOC/非共线**：DeltaSpin 的 `σ_z` 受限；空间盆（MBIS）权重由电荷建，不受 SOC 影响。
# §13 实际有效性对比（工程维度）

同一算法在"物理正确"之外，还要看**成本、周期稳健、稳定性、力/relax、已验证程度**。
本节给一张工程矩阵与逐项说明。

## 13.1 工程矩阵

| 维度 | DeltaSpin | DeltaP | Allocation | **MBIS** | Becke | Hirshfeld | Mulliken |
|---|---|---|---|---|---|---|---|
| 约束对象 | onsite 磁矩 | 极化/任意算符 | 壳层+空间 | **电荷/自旋** | 空间 | 空间 | 轨道 |
| 离子性/偶极正确 | 否(磁) | 否(极化) | 低 | **高** | 低 | 低 | 中低 |
| 壳层可分辨 | **是** | 是(算符) | 是 | **是**(盆内 l) | 否 | 否 | 是 |
| 基组依赖 | 中 | 中 | 中 | **弱** | 无 | 无 | **强** |
| 周期小胞稳健 | 好 | 好 | 好 | **需 min-image** | 好 | 好 | 好 |
| 每步成本 | 中 | 中 | 高 | 中高 | **低** | 低 | 低 |
| 收敛/稳定性 | 中 | 中 | 中 | **高**(ρ<1) | 高(几何) | 高(几何) | — |
| 力/relax | **完备** | operator 完备 | 完备 | 完备 | — | — | — |
| 能量记账 | `−ΣλM` | `−ΣλΓ` | `−Σμt` | **`−ΣμQ`** | 简单 | 简单 | — |
| 已在真体系验证 | Fe2 | 极化算例 | LiCl/Fe2 | **LiCl/Fe2/e2e/MPI** | LiCl/Fe2 | 测试级 | LiCl/Fe2 |

## 13.2 逐项说明

- **成本**：Becke/Hirshfeld **建一次**（几何确定），最便宜；MBIS **每外步迭代**（密度依赖），
  中高；Allocation 每外步重建投影+互补，最高；DeltaSpin/DeltaP 非局域投影，中。
- **周期稳健**：Hirshfeld 有**真空余量**最稳；MBIS **小胞必须 min-image**（否则破缺）；
  投影族在周期上需镜像/截断处理。
- **稳定性**：MBIS 是**约束不动点**，实测 `ρ(∂F/∂p)≈0.31<1` **稳定**；Becke/Hirshfeld
  几何确定、无多解；Allocation/DeltaSpin 需信任域/两阶段防近零模振荡。
- **力/relax**：DeltaSpin（λ 冻结）与 DeltaP **operator** 完备；DeltaP **gamma** 封口；
  空间权重族力核统一（`constraint_force`），MBIS 力用**光滑晶格和**（§8.5）。
- **能量记账**：DeltaSpin `−ΣλM`、DeltaP `−ΣλΓ`、Allocation `−Σμt`、**MBIS `−ΣμQ`**；
  正确的记账是大 Q/长时程稳定与 `BRANCH_FLIP` 守卫的前提。
- **已验证**：MBIS 已在 LiCl（电荷）、bcc Fe2（自旋，对称稳定）、主循环 e2e（np=1/2/4）、
  独立 cube（C++ vs Python）上验证；FeO（charge+spin，无 U）**机制与能量已验证**，但
  PBE 磁性不稳需 DFT+U 做受控验证。

## 13.3 适用建议（工程）

| 需求 | 推荐 | 关键设置 |
|---|---|---|
| 离子性/偶极/净电荷 | **MBIS** | `constraint_mbis_images≥1`（小胞）；`cc_escon=−ΣμQ` |
| TM d/onsite 磁矩 | **DeltaSpin / d-投影** | `sc|` 掩码/λ 回路；或 MBIS 盆内 l |
| 极化/任意线性算符 | **DeltaP（operator）** | `deltap_corr 1`；`gamma` 力勿用 |
| 快速/定性分片 | **Becke / 周期 Hirshfeld** | 固定半径/自由原子参考 |
| 约束收敛 | **ALM** 或 **secant+trust** | ALM `κ≈1–10`；stiff `kappa_min≈0.01` |
| 并行生产 | MBIS + pool gather | `np` 验证一致 |
# §14 失败模式与更正台账（含撤回项）

本方案多轮迭代，若干**早期结论被推翻**。为避免重复踩坑，集中列出"旧结论 → 更正/依据"。

## 14.1 概念/算法层

| 旧结论 | 状态 | 更正 / 依据 |
|---|---|---|
| MBIS = **自由最小化** KL `D(p)` | ❌ 撤回 | MBIS 是**带布居约束的不动点** `p=F(p)`；S6.59 `constrained-objective-correction` |
| 多初值 + **最低 D** 选"正确解" | ❌ 撤回 | `D` 在 `ζ→0`（均匀分割）=0 为**全局最小**（退化）；S6.65 `d-degeneracy-fallback` |
| **自由 D 的 Hessian** 认证/鞍点 | ❌ 对象错 | 稳定性判据是 **`ρ(∂F/∂p)<1`**（不动点映射）；S6.61 `fixedpoint-spectral-radius` |
| `‖J‖₂=32.7 ⇒ Picard 可能不稳` | ⚠️ 更正 | 真实谱半径 **0.81<1 稳定**；S6.61 |
| 同种原子 `symmetrize` 修复破缺 | ❌ 移除 | 根因是 **direct-pro corner 低估** → **min-image**；S6.63 `minimum-image-symmetry-fix` |
| coset（半格矢）图像和 | ❌ 错 | 对 Z2 协集**重复计数**；改用**完整晶格/min-image** |
| 截断晶格和 `R=2,3` | ❌ 不足 | 对扩散瞬态**收敛慢**（仍 19/13）；min-image 才对 |
| Fe2 "16/16 对称" 与 `images` 无关 | ⚠️ 条件 | `images=0`(direct) 会破缺；0.51 胞两 Fe 本不等价；S6.62 |
| legacy `cc_escon=Σμ(Q−t)` 正确 | ❌ 错 | 应 **`−ΣμQ`**（DeltaSpin 形式）；S6.72 |
| FeO `BRANCH_FLIP` = 参考非全局极小 / 权重 bug | ❌ 否 | 是**大 Q 能量记账**致假 flip；权重/观测正确；S6.70/71 |

## 14.2 收敛/求解层

| 旧尝试 | 结果 | 更正 |
|---|---|---|
| in-SCF ALM（`alm` 调度）加速 | ❌ 负结果 | 罚项过刚、dampen 砍 k 过慢；改用外层 secant/ALM |
| 大 h FD 探针 / 固定小步 | ❌ 不可靠 | 响应测量失真；用 secant+trust |
| `mixing_beta` 调参 | ❌ 对外层无效 | 瓶颈在近零模，非混合 |
| `kappa_min=0.3` 默认对 stiff | ⚠️ 慢 | 暴露 `constraint_kappa_min`，stiff 用 ≈0.01 + `step_max≈1.5` |

## 14.3 工程/性能层

| 早期 | 更正 |
|---|---|
| `build_mbis` 串行（guard QUIT） | MPI pool gather（z-fastest 重排），np=1/2/4 一致；S6.67 |
| `cutoff=1e6` 主循环 | `cutoff=10` bohr（物理衰减，1e6 网格可算）|
| `image_translations R=max(1,images)` | `R=images`（0→direct，大盒快；小胞手动 ≥1）|

## 14.4 教训

1. **先定性再定量**：不收敛/不对称时，先查**定义/口径/根因**（如 direct-pro corner），
   而非调参。
2. **能量记账是正确性前提**：`−ΣμQ` 必须扣；大 Q 才暴露。
3. **跨口径不可比**：Li 3.08 vs 2.04 是定义差；比较须同 `Q`、同 `images`、同网格。
4. **近零模**：用 ALM/secant，不用朴素梯度/固定步。
5. **周期小胞必 min-image**；**大盒用 direct**。
# §15 结论、推荐与 TODO

## 15.1 结论（一句话）

- **约束 DFT = 选可观测量 + 加约束场 + 正确能量记账 + 解近零模**；各代只是这四步的不同选择。
- **电荷/离子性/偶极**：**MBIS**（密度自适应）正确（Li⁺ +0.96、6.48 D）；固定形状
  （Becke/Hirshfeld）与投影（Mulliken/Allocation `q`）各有偏差。
- **磁矩**：DeltaSpin（onsite）/ d-投影（壳层）直接；MBIS 空间盆是"盆内总矩"，需**盆内 l**
  才能与 d 投影（≈3.0）对齐。
- **极化/任意算符**：DeltaP（operator）通用；能量记账 `escon=−ΣλΓ`。
- **求解**：ALM 或 secant+trust 远优于 Lagrange；penalty 有偏。

## 15.2 推荐（按用途）

| 需求 | 方案 | 关键设置 |
|---|---|---|
| 净电荷/离子性/偶极 | **MBIS** | `constraint_mbis_images≥1`（小胞）；`cc_escon=−ΣμQ` |
| TM d/onsite 磁矩 | **DeltaSpin / d-投影** 或 **MBIS 盆内 l** | `sc|` 掩码；盆内 l 用平均半径 |
| 极化/线性算符 | **DeltaP operator** | `deltap_corr 1`；`gamma` 力勿用 |
| 快速/定性分片 | **Becke / 周期 Hirshfeld** | 固定半径/自由原子参考 |
| 约束收敛 | **ALM** / **secant+trust** | ALM `κ≈1–10`；stiff `kappa_min≈0.01` |
| 并行生产 | MBIS + pool gather | 验证 `np` 一致 |

## 15.3 已知限制

1. **FeO charge+spin 受控验证需 DFT+U**：PBE 无 U 时金属/磁多解，SCF 换态（能量记账已修，
   但磁性不稳）。
2. **大 Q 能量**（`−ΣμQ`）需在 gapped 体系复核自旋翻转对称性（FM± 一致）。
3. **主循环默认 `constraint_mbis_images=0`** 只适合大盒；小胞应自动 ≥1（或按盒尺寸自适应）。
4. **盆内 l 精确版**（`∫w_A φφ` 加权重叠）待 C++ 落地；当前为径向平均近似。
5. 主循环 MBIS 在**所有 pool rank 冗余运行** `compute_mbis`（正确确定）；可优化为 rank0+Bcast。

## 15.4 TODO（优先级）

1. **DFT+U FeO**：完成受控 charge+spin 验证（小约束步只产生 meV 能量变化、无 BRANCH_FLIP，
   FM± 对称）。
2. **自动 images**：`build_mbis` 按盒子/原子距边界自动选 0/≥1（避免小胞误用 direct）。
3. **精确盆内 l**：`∫w_A φ_μ φ_ν` 加权重叠（复用 allocation 机制）→ `m_{A,d}` 精确。
4. **MPI 效率**：rank0 运行 + Bcast 权重，避免冗余。
5. **非共线/SOC**：DeltaSpin 的 `σ` 向量推广；空间盆（MBIS 权重由电荷建）本身不受 SOC 影响。
6. **应力/力主路径**：确认 MBIS 在 `module_constraint` 通用力核下的大体系表现。

## 15.5 适用边界（何时别用）

- 只需**定性分片**、且**几何不变**：用 Becke（别用 MBIS 迭代）。
- 只要**极化**：用 DeltaP（别用空间盆）。
- **强关联/磁性氧化物**（如 FeO）：**必须 DFT+U** 才有唯一/稳定基态；无 U 时任何约束都可能跨分支，
  `constraint_branch_tol` 会（正确地）拒绝验证。
# §16 文档映射与阅读指引

## 16.1 本文与既有文档的关系

| 文档 | 定位 |
|---|---|
| **本文 `constraint-algorithms-complete-review.md`** | **2026-05-17→今 完整演进综述**（公式/机制/有效性/台账） |
| `local-moment-constraint-algorithms-review.md` | 算法**细节子集**（Allocation→MBIS 的公式速查） |
| `mbis-development-history.md` | MBIS **权威索引 + 更正台账 + 当前规范** |
| `deltap-development-log.md` | 逐轮开发日志（S6.34–S6.75 覆盖 MBIS 全过程） |
| `docs/constraint_developer_guide.md` | 框架**开发者**文档（M0–M8、数据流、生命周期陷阱） |
| `docs/constraint_user_manual.md` | 约束**用户手册** |
| `docs/deltap_user_manual.md` | DeltaP **用户手册** |

## 16.2 各阶段代表文档

- **DeltaSpin（05-17）**：`2026-05-17-deltaspin-parameter-unification-design.md`
  （`docs/superpowers/plans/` 有对应 plan）。
- **DeltaP（07–09）**：`docs/deltap_user_manual.md`、`2026-08-13-deltap-capability-boundaries.md`、
  dev log 2026-07～09。
- **Allocation（10-21～29）**：`2026-10-21-unified-local-moment-allocation-design.md`、
  `2026-10-29-allocation-development-summary.md`、`2026-10-27..29-allocation-*` 系列、
  `2026-10-30-allocation-vs-mbis-formula-analysis.md`。
- **MBIS（10-30）**：`2026-10-30-mbis-*` 系列（约 45 篇），关键：
  `mbis-charge-spin-constraint-algorithm`、`mbis-grid-bridge`/`weightgrid-backend`/`constraint-loop-integration`、
  `mbis-alm-*`、`mbis-minimum-image-symmetry-fix`、`mbis-real-cube-stability`、
  `mbis-d-degeneracy-fallback`、`mbis-smooth-force-pro`、`mbis-energy-audit-deltaspin-comparison`、
  `mbis-mpi-gather`、`mbis-faster-mu-controller`、`mbis-feo-*`。
- **数值对比出处**：`2026-10-27-fe2-licl-comparison`、`mbis-prototype-licl-comparison`、
  `mbis-fe2-test`、`mbis-in-basin-l-projection-test`、`2026-10-21-local-moment-algorithms-comparison`、
  `mbis-alm-comparison`。

## 16.3 阅读顺序建议

1. **只想知道"现在怎么做"**：本文 §1（框架）→ §8（MBIS）→ §4（框架落地）→ §10（求解器）。
2. **想理解数值差异（2.04/2.48/3.08/2.29/2.99）**：§5–§9 + §11 + §12。
3. **想知道"哪些旧结论被推翻"**：§14 + `mbis-development-history.md` §2。
4. **要动手改代码**：`docs/constraint_developer_guide.md`（M0–M8、数据流、生命周期陷阱）+
   §4 + §8。
5. **要做磁性体系约束**：§8（MBIS）+ §15.3/TODO（DFT+U）。

## 16.4 仓库/分支

- 分支 `mbis`（原 `feat/unified-local-moment-allocation`），远程 `zdy`。
- PR: https://github.com/dyzheng/abacus-develop/pull/new/mbis
- 本综述随分支维护；早期 spec 若与本文冲突，**以本文 + `mbis-development-history.md` 为准**。
