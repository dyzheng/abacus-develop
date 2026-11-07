# 2026-10-30 Penalty + Lagrange 同时实现：增广拉格朗日（ALM）

## 0. 结论

**可以，而且是推荐做法。** 但"同时实现 penalty 和 constrain"不应写成两个互相
竞争的求解器；正确形态是把它们**合并为一个增广拉格朗日泛函**：

```
E_ALM[ρ; μ,k] = E_KS[ρ] + Σ_a [ μ_a (Q_a − t_a) + (k_a/2)(Q_a − t_a)² ]
v_con(r)     = Σ_a ( μ_a + k_a (Q_a − t_a) ) w_a(r) = Σ_a μ_eff,a w_a(r)
```
- **penalty 项** `(k/2)(Q−t)²`：给约束方向加曲率 → 加快/稳定外层，缓解近零模；
- **Lagrange 项** `μ(Q−t)` 与 `μ ← μ_eff = μ + k(Q−t)` 更新：消除 penalty 的
  偏置，收敛时精确满足约束；
- 代码 `source/source_estate/module_constraint/constraint_alm.{h,cpp}`
  （`AlmController`）**已实现该形式**：`arm(chi)` 按响应 `χ=dQ/dμ` 定
  `k=kappa_target/|χ|`、`damping α=1/(1+k|χ|)` 防 2-周期、`mu_eff/update/energy_correction`。

## 1. 为什么"纯 constraint 慢"、penalty 为什么帮上忙

约束问题是鞍点/固定点迭代（Uzawa）：
```
Q(μ) 的残差 r=Q−t；纯 Lagrange 更新 μ←μ−δ·r 的收敛率由 Schur 补
S = C A^{-1} B（约束响应）决定。若 |S| 小（近零模、软模），则慢/振荡
（此前 allocation 63 步、3.8 态近零模/2-周期即此类）。
```
penalty 把有效 Hessian 在约束方向抬高 `k`：
```
H_eff = A + k B Bᵀ   (A=KS Hessian, B=∂Q/∂ρ)
⇒ 约束方向曲率 A+k|χ|²，Schur 补从 S 变为 S/(1+k S) → 条件数改善
```
外层收敛率变为 `|1 − kχ|` 的收缩（代码注释：`|μ−μ*|` 每步缩 `1/|1−kχ|`）。
取 `k|χ|≈1`（`kappa_target≈1`）收缩最快。**这正是"constraint 慢"的解药。**

## 2. 三种方案对比

| 方案 | 约束精度 | 收敛速度 | 偏差 | 稳定性 |
|---|---|---|---|---|
| 纯 penalty（μ=0） | 近似，`C*=O(1/k)` | 快 | **有**：N/M、力、应力偏 `O(1/k)` | 取决于 k；过大→内层 SCF 变硬 |
| 纯 Lagrange（Uzawa） | 精确（收敛后） | 慢/可 2-周期 | 无 | 近零模差 |
| **ALM（二者兼有）** | **精确（μ→μ\*）** | **快（penalty 曲率）** | **无**（μ 更新消偏置） | **好**（+damping） |

- **纯 penalty 的偏差**必须强调：有限 `k` 下 `Q*≠t`，`ΔQ∝1/k`，直接污染
  力/应力/磁矩；所以**不能**只做 penalty。
- **ALM 的精确性**：外层 `μ←μ+k(Q−t)` 使 `Q→t`；收敛后
  `μ_eff=μ+k(Q−t)→μ*`（物理乘子），力/应力用 `μ*` 即精确。
- 代码里"ALM 默认 k → 200 步 RUNNING"的负结果，是**k 未标定**（固定默认值）；
  正解是**按响应自动 arm**（`k=kappa_target/|χ|`）+ damping，而不是放弃 ALM。

## 3. 电荷+自旋同时的 ALM（MBIS 权重）

每个原子两个约束分量 `a=(A,c)` 电荷、`(A,s)` 自旋：
```
Q_{A,c}=∫w_Aρ=N_A,  t_{A,c}=N̄_A
Q_{A,s}=∫w_Am=M_A,  t_{A,s}=M̄_A
v_con,σ(r) = −Σ_A [ (μ^c_A + k^c_A(N_A−N̄_A))
                   + σ (μ^s_A + k^s_A(M_A−M̄_A)) ] w_A(r)
E_ALM = E_KS + Σ_A[ μ^c_A(N_A−N̄_A)+(k^c_A/2)(N_A−N̄_A)²
                  + μ^s_A(M_A−M̄_A)+(k^s_A/2)(M_A−M̄_A)² ]
```
- `k^c_A`、`k^s_A` 分别按 `χ^c_{AB}=∂N_A/∂μ^c_B`、`χ^s=∂M_A/∂μ^s_B` arm；
- 非共线：`μ^s_A→μ⃗^s_A`、`σ_z→σ⃗`。
- **R8 冗余**：全原子电荷约束+固定 N 线性相关；ALM 对冗余的 `k` 也要处理
  （N−1+全局模式，或对 J 正则化）。

## 4. 力与 stress（ALM 版本）

ALM 下约束能含 penalty 项；在**收敛点**（`Q→t`）penalty 的贡献趋于 0，
物理乘子 `μ_eff→μ*`，力/应力用 §"MBIS 约束算法" 的公式即可：
```
F = F^DFT + Σ_A ∫ (∂_R w_A)[ μ*^c_A ρ + μ*^s_A m ] dr
σ = σ^DFT + (1/Ω)Σ_A[ μ*^c_A ∂_εN_A + μ*^s_A ∂_εM_A ]
（μ* 用 mu_eff；严格需含 KL 参数响应项）
```
- **若未完全收敛**：用 `μ_eff=μ+k(Q−t)` 计算力/应力（这是"当前约束势"对应的
  力），并报告 `Q−t` 残差；不要把有限 k 的 penalty 偏置当作物理。
- `AlmController::energy_correction` 已给 `E_ALM−E_KS`；`mu_eff` 给物理乘子。

## 5. 实施建议（针对"SCF 慢"的担心）

1. **启用 ALM（penalty+Lagrange）**，不要纯 Lagrange；代码已有，MBIS 只需
   把权重 `w_a` 换成 `w_A^MBIS`。
2. **k 按响应 arm**：先跑少量参考步测 `χ=dQ/dμ`，取 `k=kappa_target/|χ|`
   （`kappa_target≈1`，clamp 到 `[k_min,k_max]`）；**不要**用固定默认 k（会
   重现 200 步负结果）。
3. **damping**：对外层 `μ` 用 `α=1/(1+k|χ|)`（代码已实现）以避免 2-周期。
4. **自适应 k**：若残差下降慢→增 k；若内层 SCF 变慢/振荡→减 k。
   典型 `k|χ|∈[0.5,2]`。
5. **两级容差**：内层 SCF 收敛严于外层约束；ALM 允许内层"够用即可"
   （inexact ALM，通常更快）。
6. **电荷冗余**：N−1+全局模式；自旋均匀场同理。
7. **诊断**：打印 `Q−t`、`μ`、`k`、`|ΔE|`、`Δρ`；若 `Δρ` 卡但 `|ΔE|,Q−t`
   收敛→近零模/判据问题（见相关文档），用物理判据。

## 6. 结论

- **penalty 与 constraint 应合并为 ALM**，不是两个并行方案；
- ALM 用 penalty 提速/稳住、用 Lagrange 更新保证精确，正好解决"纯 constraint
  SCF 慢"的担忧；
- 代码已有 `AlmController`，关键工作是 **k 的响应标定 + damping + 双通道
  (c,s) 的 MBIS 权重**；
- 力/应力在收敛点用物理乘子 `μ*` 精确；未收敛时用 `μ_eff` 并报告残差。
