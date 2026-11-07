# 2026-10-30 公式级分析：Allocation 为何偏离 MBIS，如何对齐

## 0. 结论

LiCl（LCAO，同一自由密度）上 Li 电子数：**Allocation 报告 n=3.0833（Becke 权重）**、
**Mulliken 2.4830**、**MBIS 2.0447**。差 ~1 e 的**主因不是"投影到某个壳层"，
而是空间权重 `w_i(r)` 的 pro-density 选取**：

- Allocation 的**空间总量** `n_tot,i=Tr[ρ W_i]=∫w_i^fixed(r)ρ(r)dr`，
  `w_i^fixed` 是**固定 Gaussian/Becke**（Li 用固定半径 → 偏宽）；
- MBIS 的 `q_A=∫w_A^MBIS ρ`，`w_A^MBIS` 用**迭代 Slater**（Li 优化到 ζ=4.60，很紧凑）。

两者是**同一泛函形式** `∫w ρ`，只是 `w` 不同；把 `w` 换成 MBIS 即得 MBIS 电荷。
轨道投影 `Q_i` 是**第二个独立问题**：它只改变**约束算符** `A_i`（使其 ≠ 纯权重、
且基组依赖），对**报告总量 `n_tot` 无影响**；对"电荷"它是多余的，应去掉。

## 1. 两个方案的精确定义

### MBIS（纯实空间 stockholder）
```
pro:   ρ_A^pro(r) = N_A (ζ_A³/8π) e^{−ζ_A|r−R_A|}
weight: w_A(r) = ρ_A^pro(r) / Σ_B ρ_B^pro(r)
atomic: ρ_A(r) = w_A(r) ρ(r)
charge: q_A = ∫ρ_A dr = ∫ w_A(r) ρ(r) dr
update: N_A = q_A,  ζ_A = 3 q_A / ∫ r w_A ρ dr = 3/⟨r⟩_A     (KL/均值半径)
```
收敛时 `Σ_A w_A=1 ⇒ Σ_A q_A=N`。**权重依赖 ρ 并迭代**（Li⁺ 会自动收缩）。

### Allocation（混合投影 + 空间互补）
```
A_i = Q_i + Q_c W_i Q_c,   P = Σ_i Q_i,  Q_c = I − P
q_i = Tr[ρ A_i] = Tr[ρ Q_i] + Tr[ρ Q_c W_i Q_c]      (约束坐标)
```
- `Q_i`：原子 i 上目标 NAO 轨道（角动量 l）的**正交投影算符**；
- `W_i`：实时空间权重（Becke 或固定 Gaussian 的周期 Hirshfeld）。

代码事实（`alloc_lcao_report.cpp:375-404`）：
```
n_tot,i = Tr[ρ W_i] = Σ_ir w_i(r)ρ(r) dV      ← 纯空间权重（与 target_l 无关）
n_T,i   = Tr[ρ Q_i]                            ← 目标壳层投影（诊断）
n_c,i   = n_tot,i − n_T,i                      ← 报告的"差"，不是算符互补
```
而约束算符的互补 `Tr[ρ Q_c W_i Q_c]` **不等于** `n_tot−n_T`（语义闭环 §3.3/§3.4
已定量证实：Fe2 的算符 `m_c=−0.20` vs 报告 `m_c=+0.78`）。且
`alloc_constraint_lcao.cpp:178-182` 规定**约束至少要有 1 个目标轨道**，
即当前无法运行"纯空间"约束。

## 2. 为什么差这么多（三项，按贡献排序）

**(A) pro-density 宽度（主因，~1 e）**
Allocation 的 `w_i` 用固定 Gaussian `σ=划分半径、q=zv`（或 Becke 多项式），
Li 半径大 → 分到 3.08 e。MBIS 把 Li 优化到 ζ=4.60（Li⁺ 很小）→ 2.04 e。
数学上
```
q_i^alloc − q_A^MBIS = ∫ [ w_i^fixed(r) − w_i^MBIS(r) ] ρ(r) dr
```
被积函数就是**权重差**；Li 附近 `w^fixed>w^MBIS`，积分差 1.04 e。

**(B) 权重是否依赖密度（结构性）**
MBIS 的 `ζ_A` 随 ρ 变（离子化后 Li 收缩）；Allocation 的 `W_i` 是
**几何纯函数**（`weight_grid.h` 明确"密度不进入权重"）。固定几何权重无法
跟随电荷态变化 → 系统性把过多密度分给 Li。

**(C) 轨道投影 `Q_i`（语义/基组依赖，影响约束而非报告总量）**
`Tr[ρ Q_i]` 把**整个目标轨道占据**判给其所属原子，与轨道在空间上是否
弥散无关（基组依赖、非局域）。Li 的 4s 弥散、伸入 Cl 区，`n_T=2.42` 明显
高估了 Li 的空间份额。对**电荷**这一项是多余的；它真正适合的是
**磁壳层矩（XMCD 口径）**。在混合算符里它还使约束坐标 `Tr[ρ A_i]` 偏离
纯空间电荷 `Tr[ρ W_i]`。

## 3. 如何优化才能与 MBIS 一致（公式级步骤）

目标：让约束坐标 `q_i = ∫ w_i^MBIS(r) ρ(r) dr = q_A^MBIS`。

**Step 1｜去掉轨道投影（纯空间极限）**
```
Q_i = 0, P = 0, Q_c = I  ⇒  A_i = W_i,  q_i = Tr[ρ W_i] = ∫ w_i ρ
```
需放开 `alloc_constraint_lcao.cpp` 的"≥1 目标"限制（报告路径已支持零目标）。
这一步把约束算符变成**纯权重**，与 MBIS 同形式。

**Step 2｜替换权重后端 `W_i → w^MBIS`**
```
ρ_A^pro = N_A ζ_A³/8π e^{−ζ_A r}       (Slater s，替代固定 Gaussian)
w_A = ρ_A^pro/Σ_B ρ_B^pro
N_A = ∫ w_A ρ,   ζ_A = 3 N_A / ∫ r w_A ρ
```
迭代到 `|ΔN|<1e-9`。此时 `q_i=∫w_A^MBIS ρ ≡ MBIS 电荷`（by construction；
原型已验证 `∫w^MBIS ρ=2.0447`）。

**Step 3｜允许权重依赖密度（外层迭代）**
MBIS 的 `ζ_A` 依赖 ρ，因此 `w^MBIS` 必须**在 ρ 更新后重建**（外层循环），
不能再把权重当几何常数。约束化的务实做法：**冻结分区近似**——每个外层/μ 步
用上一步收敛的 `w^MBIS(ρ_n)` 作固定算符，步间更新；力的显式部分用
`∂w/∂R`，隐式部分用包络定理/冻结近似修正。

**Step 4｜（可选）多极**
增加 `μ_A=−∫(r−R_A)ρ_A`、四极积分，与 ORCA 对齐全套输出。

## 4. 关键取舍：按观测量拆分，不要用一个混合算符

- **电荷**：用**纯空间 MBIS**（`A_i=w_i^MBIS`）。这是唯一与 MBIS 一致、
  且物理/实验可对照（偶极、ESP）的电荷口径。
- **自旋/壳层矩（XMCD）**：保留投影（DeltaSpin/`Q_i`）或用 **spin-MBIS**
  （同一 `w^MBIS` 作用在 `ρ↑−ρ↓`，属于空间 AIM，不等于投影矩）。
- **现有混合 allocation**（`Q_i+Q_cW_iQ_c`）：仅当研究意图就是"壳层+空间
  互补"时使用；它**不是**净电荷，也**不应**期望与 MBIS 数值一致。

## 5. 预期数值（LiCl，LCAO）

| 口径 | Li N (e) | 说明 |
|---|---:|---|
| Allocation 报告 `n_tot`（Becke 权重） | 3.0833 | 固定 Gaussian/Becke |
| Mulliken | 2.4830 | 轨道布居 |
| **MBIS（目标）** | **2.0447** | 迭代 Slater |
| **改后端后（Step1+2）** | **2.0447** | `∫w^MBIS ρ`，与 MBIS 逐位一致 |
| 分子偶极 | 6.48 D（MBIS，含原子偶极） | 实验 ~6.3 D |

即：**只要把权重后端换成 MBIS（并去投影），allocation 约束坐标就变成 MBIS
电荷**；差异的 ~1 e 全部来自 pro-density（固定宽 Li vs 优化紧凑 Li）。

## 6. 风险/注意

- 权重变为密度依赖 → 非线性：需要外层迭代；约束 Jacobian/力要含分区响应。
- 周期 MBIS 的镜像求和、Slater 长尾 cutoff、网格质量（偶极敏感）需处理。
- 不要用 MBIS 的 **spin** 去替代 DeltaSpin 投影矩（口径不同）。
- 与 Mulliken/投影族的数值不可混用；输出必须标注口径。
