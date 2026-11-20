# 2026-11-19 MBIS 约束 SCF：第一轮对角化后的电荷密度与目标态方向

> 范围：只讨论 MBIS / MBIS 衍生方案。不测试 allocation/projector 路径。
> 目的：分析“第一轮对角化后密度如何确定，才能尽快朝目标电子态迭代”，
> 并给出可实现的自动初猜/方向控制方案。

## 0. 结论摘要

1. **第一轮对角化后的密度 ρ1 不是“自动”指向目标态的**；它由初猜 ρ0、
   初始 μ、MBIS 权重 w[ρ0]、以及第一轮 H 的能级排列/占据共同决定。
2. 目标态方向的核心判据不是“能量最低”，而是 **电荷方向**：
   `ΔQ1 = Q(ρ1) - Q(ρ0)` 是否与 `Q* - Q(ρ0)` 同号。
   MBIS 的 Q 是 `∫ w ρ`，所以第一轮就能读出方向。
3. **MBIS 权重必须来自“目标 basin 的密度”**。若 w 来自离子态，用 w 去
   约束中性 Q*，物理上不是同一个分区；第一轮对角化即使 Q 接近，也可能
   落到错误的电子态。
4. 目标 Q=3/7 在 r=3 Å/20 Å 附近表现为**一级/共存型转变**：
   实测 Q 在 3.24 和 2.01 之间跳变；单点第一轮对角化很容易落到错误分支。
   因此需要 **小步 homotopy（μ 或 Q 连续变化）**，而不是一次大 μ。
5. 推荐的 MBIS 衍生流程：
   - 自由 SCF → 离子态 ρ0、Q0、能级/轨道 character；
   - 用能级对齐估 μ（不是盲猜），或者直接做 **目标 Q 的 homotopy**；
   - 每一步用上一步的 ρ/tau/μ 做 direct start，重建 MBIS 权重；
   - 第一轮后立即做 `ΔQ1` 方向检查：方向错就翻转/缩放 μ，方向对但太大就
     缩小 μ 步长；
   - 到达 Q* 后比较两态能量，检测跳变并细化交点。

---

## 1. 约束 SCF 与第一轮对角化的定义

MBIS 约束的 KS 方程：

```
H[ρ] ψ_i = ε_i ψ_i
H[ρ] = T + V_ext + V_H[ρ] + V_xc[ρ] + V_con
V_con(r) = Σ_A μ_A w_A(r)
w_A(r) = pro_A(r) / Σ_B pro_B(r)
N_A = ∫ w_A ρ,   Q = (N_A)
```

SCF 映射：

```
ρ_{n+1} = Mix( ρ_n, F[ρ_n; μ, w] )
F[ρ] = Σ_i f(ε_i - E_F) |ψ_i[ρ]|^2
```

固定点 `ρ* = F[ρ*; μ*, w[ρ*]]`，且 `Q(ρ*) = Q*`。

**第一轮对角化**：给定初猜 ρ0、初始 μ、初始权重 w0 = w[ρ0]，
对角化 `H[ρ0]` 得到 `{ψ_i, ε_i}`，按占据 `f` 得到 ρ1 = F[ρ0; μ,w0]。
`ρ1` 就是“第一轮对角化后的电荷密度”。

关键：`ρ1` 的**轨道 character 和占据**由 `H[ρ0]` 的能级排列决定；
`H[ρ0]` 的能级排列由 ρ0、μ、w0 决定。因此第一轮方向完全由初猜/μ/权重决定。

---

## 2. 第一轮对角化“朝哪走”的判据

### 2.1 电荷方向

MBIS 权重固定时：

```
Q(ρ1) - Q(ρ0) = ∫ w0 (ρ1 - ρ0)
```

目标方向：

```
dQ* = Q* - Q(ρ0)
```

**第一轮方向正确**当且仅当

```
sign(ΔQ1) = sign(dQ*)
```

且 `|ΔQ1|` 不应远大于 `|dQ*|`（否则可能越过目标/落入另一支）。

这是最便宜、最直接的判据：第一轮 SCF 后就能算，不需要等 SCF 收敛。
当前实现只在 SCF 收敛后由外循环用 Q；我们建议增加“第一轮 Q 检查/内层 μ
调整”，见 §6。

### 2.2 能级对齐

第一轮要得到目标占据，需要目标轨道在 H[ρ0] 中处于正确的能级位置。
近似地，约束势在原子附近 `w_A≈1`，所以目标轨道能级移动约 `μ_A`。
对 LiF 中性态（F 2p hole、Li 2s 电子）：

```
ε_Li,s + μ_Li  <  ε_F,p + μ_F
```

即 `μ_Li - μ_F < -(ε_Li,s - ε_F,p)`。
只由 **μ 差** 决定（共同平移是规范自由度）；具体分配可令 `μ_F=0` 或用
MBIS 权重在原子上的平均值做加权。

但注意：MBIS Q 是密度积分，不是轨道占据；能级对齐只是**第一轮方向的
必要条件**，不保证 Q* 精确。要 Q* 还需要外循环/内层调 μ。

### 2.3 第一轮密度的“目标方向”投影

若有一个 target-like 密度 ρ_t（来自 homotopy/固定 μ/中性初猜），可定义

```
d = ρ_t - ρ0
Δρ1 = ρ1 - ρ0
overlap = <Δρ1, d> / (||Δρ1|| ||d||)
```

`overlap>0` 说明第一轮朝目标走。没有 ρ_t 时，用 MBIS 的 `ΔQ1` 代替即可。

### 2.4 一级/共存转变

实测（20 Å，r=3.0）：

- ΔSCF 中性初猜 + μ=(-0.5,+0.5)：Q=(3.2408, 6.7572)，residual 0.243；
- 同一密度 + μ=(+0.5,-0.5)：Q=(2.012, 7.987)；
- μ 从 -0.5 到 +0.5，Q 从 3.24 跳到 2.01。

这说明 `Q(μ)` 在目标附近**不是连续小斜率**，而是接近一级跳变/共存。
第一轮对角化在 μ 接近转变点时，可能落到 3.24 支或 2.01 支；
单点大 μ 很容易选错支。**小步 homotopy** 是必要的。

---

## 3. MBIS 权重/Q 的密度依赖：为什么初猜必须“目标-like”

MBIS 的 w 依赖 ρ（通过 N_A, ζ_A）。约束 Q 是 `∫ w[ρ] ρ`。

- 若 w 由离子态 ρ_ion 构建，而目标态是中性，则 `Q*` 用的是**离子态的
  分区**；第一轮即使 `∫ w_ion ρ1` 接近 3/7，物理上也不是中性态的分区。
- 在 constrained 相，当前实现会把 w 冻结在 reference 密度；因此
  **reference/初猜密度必须在目标 basin**，否则 w 和 Q* 不一致。
- direct constrained start 的 `build_initial_mbis_weights` 从初始密度构建
  w；这正是为什么“原子初值 + 猜 μ”失败：w 是原子/离子-like，不是中性-like。
- ΔSCF 中性密度给出 Q=(3.24,6.76)，说明它的 w 已经接近中性分区；
  这是它比原子初值更接近目标的原因。

**结论**：初猜优化不仅是“让第一轮能级对齐”，还要**让 MBIS 权重来自
目标-like 密度**。这两者需要同时满足。

---

## 4. 已有测试的复盘（MBIS-only）

### 4.1 自由 SCF / 离子态

- 18 Å，1.2–3.0 Å：SCF 收敛，Q≈(2.038, 7.961)（Li⁺⁰·⁹⁶F⁻⁰·⁹⁶）。
- 20 Å，r=3.0：自由 SCF 收敛，E=−840.8045 eV，gap≈0.15 Ry（~2 eV）。
- 4.5/6.0 Å 自由 SCF 失败（charge sloshing）。

### 4.2 原子初值 + 猜 μ（20 Å/r=3.0）

- ionic/neutral 多组 μ 全部失败；Q 约 (2.02,7.98)（仍是离子-like），
  Drho 大，能量乱跳。
- nspin2 + 原子初值：181 步，Drho≈8–10，能量 +900~+1450 eV，
  总磁矩 2 μB，无 constraint audit。**nspin2 本身不解决初猜问题**。

### 4.3 ΔSCF 中性初猜（nspin1，固定占据）

- `ocp_set 2 2 2 2 1 1`（第 5 band 一个电子 promotion 到第 6 band）。
- SCF 150 步未完全收敛（Drho≈0.079），E≈−835.92 eV；
- 但密度已经是中性-like：MBIS Q=(3.2408, 6.7572)。

### 4.4 从 ΔSCF 中性密度做 direct neutral start

| initial μ (Li,F) | 结果 |
|---|---|
| (−0.5,+0.5) | SCF 帧在 192 步，outer step 1；Q=(3.2408,6.7572)，residual 0.243；之后 200 步未完成 |
| (+0.5,−0.5) | outer step 2/3：Q=(2.012,7.987)→(2.025,7.974)，μ 振荡 ±0.1；SCF 未收敛 |
| (−0.3,+0.3) | 无 outer step；SCF 250 步未收敛 |

解读：

- ΔSCF 初猜把 Q 从 2.02/7.98 拉到 3.24/6.76，**第一轮方向是对的**；
- 但 μ 稍大就跳到 2.01（离子支）；μ 介于中间时 SCF 不收敛；
- 这是典型的一级/共存 + 权重/外循环不一致问题，而不是“nspin2 不够”。

---

## 5. MBIS / MBIS 衍生的初猜与方向控制方案

> 不包含 allocation/projector 路径。

### A. 能级对齐的初始 μ（最便宜）

1. 自由 SCF 得到 F-p/Li-s 的能级差 `Δε`（通过 eigenvalues + 原子轨道
   projected character 判断，不需要 allocation）。
2. 用 `μ_Li - μ_F ≈ -Δε` 估计第一轮能级交叉的 μ；具体分配可用 MBIS
   权重在原子上的平均值 `w_A(R_A)` 做加权。
3. direct start 用这个 μ + 自由 SCF 密度。第一轮后检查 `ΔQ1` 方向；
   方向错就翻转 μ，太大就减小 μ。

优点：不需要额外 SCF；缺点：只解决能级，不解决 w 的目标-like 问题；
适合作为 homotopy 的起点。

### B. 固定 μ / 小步 μ homotopy

1. 从 μ=0（离子态）开始；
2. 以很小的 μ 步长（例如 0.05 Ry）递增/递减到目标方向；
3. 每一步用上一步的 density/tau/μ，direct start；MBIS 权重从当前密度重建；
4. 记录 Q、E、Drho；若 Q 跳变/SCF 失败，缩小 μ 步长或退回一步。

优点：自动、无需手动占据；第一轮总是从上一步的稳定密度出发；
能跟踪亚稳分支。缺点：步数多；如果转变是一级，可能在某一步跳支，
需要检测。

### C. 目标 Q homotopy（推荐）

1. 自由 SCF → Q0（离子态）；
2. 定义中间目标 `Q_k = Q0 + (k/n)(Q* - Q0)`，k=0..n；
3. 第 k 步：用 Q_k 作为 absolute target，初始 μ 用上一步的最终 μ
   （或用 `dQ/dμ` 线性预测），density 用上一步；
4. direct start 每步重建 MBIS 权重（从当前密度）；
5. 若某步 SCF 不收敛或 Q 跳变，减小 `Q*` 步长（例如 0.1 e）。

优点：目标 Q 连续变化，第一轮 ΔQ1 很小；MBIS 权重始终目标-like；
不需要手动占据/能带号；最适合找中性态。

### D. 混合密度初猜

若有离子密度 ρ_ion 和一个中性-like 密度 ρ_n（来自 ΔSCF/固定 μ/小步
homotopy），构造

```
ρ_λ = (1-λ) ρ_ion + λ ρ_n
```

选 λ 使 `Q(ρ_λ; w[ρ_λ]) ≈ Q*`（可用二分/插值）。
用 ρ_λ 做 `init_chg`，MBIS 权重从 ρ_λ 构建；再做 direct start。

优点：第一轮密度直接在目标 Q 附近，w 也目标-like；缺点：需要 ρ_n。

### E. 多起点/盆地选择

- 用若干初猜：ρ_ion、ρ_ΔSCF、ρ_mix、不同 μ 的固定 μ 密度；
- 每个跑一个短 SCF（40–60 步），记录 Q/Drho/E；
- 选 `ΔQ` 方向正确且 Drho 下降最快的作为正式 start。

优点：自动化、鲁棒；缺点：多次短 SCF 的成本。

### F. 自动 projected-DOS / 轨道 character（仅作 μ 估计）

- 用自由 SCF 的波函数和原子投影（onsite projector/atomic orbital）计算
  target-like 的 projected DOS；
- 找到使 `∫ PDOS_target = target occupation` 的 `ε_target`；
- 用 `ε_target - E_F` 估初始 μ（或作为能级对齐的 Δε）。

注意：这是**诊断/μ 估计**，不是 allocation 约束；不改变 MBIS 约束本身。
它可以把“盲猜 μ”变成“从电子结构读出的 μ”。

### G. 第一轮 Q 探针 / 内层 μ 控制（代码增强）

当前外循环只在 SCF 收敛后用 Q；第一轮的 Q 被浪费。建议：

- 在第一轮 SCF 后（或每轮），用固定的 w0 计算 Q1；
- 计算 `ΔQ1` 与 `dQ*` 的符号/大小；
- 若方向错/过大，立即调整 μ（翻转/缩放），再继续 SCF；
- 这可以接到现有 `constraint_mu_schedule inner` 的框架（但阈值要放到
  第一轮，而不是等 drho 很小）。

这是最贴近用户问题“第一轮对角化后的密度如何确定方向”的代码改动。

---

## 6. 推荐的详细流程（MBIS-only）

```
1. Free SCF (μ=0):
   - ρ_ion, Q0, eigenvalues, F-p/Li-s character
2. Estimate μ_start:
   - level alignment (A/F) or start from 0 (B/C)
3. Homotopy in Q (C):
   for k = 1..n:
     Q_k = Q0 + (k/n)(Q* - Q0)
     ρ_start = previous density (ρ_{k-1})
     w_start = MBIS weights built from ρ_start
     μ_start = previous μ (or linear prediction from dQ/dμ)
     direct constrained start:
       - first diagonalization with (ρ_start, w_start, μ_start)
       - Q1 = ∫ w_start ρ1
       - if sign(Q1-Q_{k-1}) != sign(Q_k-Q_{k-1}):
            flip/scale μ; retry first diagonalization
       - elif |Q1-Q_k| > trust: reduce μ step
     run SCF to convergence (or 60-step cap)
     record Q, E, μ, Drho, gap
     if fail: halve Q step, retry
4. At Q*=3/7:
   - check residual, gap, total charge
   - compare E_neutral with E_ionic (same box); detect jumps
5. If a jump occurs:
   - refine with smaller Q step around the jump
   - compare energies of the two branches (Maxwell/coexistence)
```

关键点：**每一步的“第一轮密度”就是上一步的收敛密度**，所以第一轮
不会从原子初值重新开始；MBIS 权重也始终来自目标-like 密度。
这比“原子初值 + 大 μ”稳定得多。

---

## 7. 判据与诊断

第一轮后（不需要等 SCF 收敛）：

- `ΔQ1 = Q1 - Q0`，`dQ* = Q* - Q0`；
  - `sign(ΔQ1) != sign(dQ*)` → μ 方向错；
  - `|ΔQ1| > 2|dQ*|` → μ 太大，减半；
  - `|ΔQ1| < 0.1|dQ*|` → μ 太小，加倍。
- Q1 与 Q* 的距离、Drho 趋势、gap、总电荷。
- 若 Q1 在 3.24/2.01 之间跳变，说明接近共存；减小 Q/μ 步长。

SCF 收敛后：

- residual < constraint_thr；
- ΣQ = nelec（或带约束的 total）；
- E 与离子态同盒比较；检测 BRANCH_FLIP/能量异常；
- 若 Q* 不可达（μ 到 cap 仍差很远），说明该价态在此几何/约束下不稳定。

---

## 8. 风险与开放问题

1. **一级/共存**：目标 Q=3/7 可能落在两支之间；homotopy 可能跳支；
   需要用能量比较/更小步长。
2. **MBIS 权重冻结**：constrained 相冻结 w；homotopy 每步重建 w
   （通过新的 direct start）是必要的；同一 SCF 内不要混用旧 w。
3. **nspin2**：中性态是开壳；nspin1 的 ΔSCF/混合密度只是空间初猜。
   nspin2 + homotopy 是下一步；但 nspin2 + 原子密度会发散（已实测）。
4. **20 Å 镜像**：r=10 时镜像间距 ~10 Å；定量交点需更大盒。
5. **计算成本**：每个 homotopy 点一次 SCF；0.1 e 步长 × ~10 步/态；
   用 GPU OMP=1，单点几分钟到十几分钟；需要早停/缓存。

---

## 9. 下一步建议

1. 实现 **目标 Q homotopy 驱动**（MBIS-only）：
   - 从 18 Å/3.0 的离子态 Q0 开始；
   - Q 从 Q0 到 3/7，步长 0.1 e；
   - 每步 direct start + MBIS 权重重建；
   - 第一轮 ΔQ1 方向检查 + μ 缩放。
2. 先在小盒（12–14 Å）/PBE 验证 homotopy 能否稳定到 Q=3/7；
   再回到 20 Å/r2SCAN。
3. 若 homotopy 在 3.24↔2.01 跳变，记录两支能量，做 Maxwell/细化；
   若中性支始终不稳定，再考虑 nspin2 的 homotopy。
4. 不使用 `ocp_set` 作为生产方案；ΔSCF 只作为诊断/混合密度的来源。
