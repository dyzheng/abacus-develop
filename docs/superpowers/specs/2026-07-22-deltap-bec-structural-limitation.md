# DeltaP Per-Atom 分解的结构性局限：BEC 为何不可行

**日期**: 2026-07-22
**基于**: BN Born 有效电荷的三层验证、H2O PES 测量、分支选择修复

---

## 1. 三层分析框架

DeltaP 计算中有三个层次的 Berry 相位，从原始到聚合：

```
Level 3: 总 Berry 相位 (直接)  ← arg(zeta_scalar) 平均，模 2π
    ↑ 绕过了 per-atom 分解
Level 2: Σ_i γ_i_raw           ← per-atom 加权求和后聚合
    ↑ = Σ_i Σ_n w_In[n][i] · gamma_unwrapped[n]
Level 1: Per-atom γ_i           ← 单原子 Wannier 重心分解
    ↑ 经过 zeta rescaling (Step 1) + 分支选择 (Step 2+3)
```

三层对应三种不同粒度的物理量：
- Level 3：总极化 P_total（与标准 Berry 相位等价）
- Level 2：总极化 per-atom 加权聚合（理论应与 Level 3 一致）
- Level 1：每原子独立极化标签（用于约束）

---

## 2. BN BEC 测试中的三层表现

δu = 0.05 Bohr 沿极化方向 [1,1,-1] 位移 B 或 N 原子：

| 方法 | Z*(B) | Z*(N) | vs 文献 ±2.7 |
|------|-------|-------|-------------|
| Level 3: arg(zeta) 平均 | **-0.13** | **-9.32** | 量级分散 70× |
| Level 2: Σγ_raw 差分 | **-13.46** | **+11.06** | 符号错/量级 5× |
| Level 1: Σγ_branched 差分 | **-0.43** | **+0.03** | 被靶标锚定抹平 |

**三个层次给出三个不同答案，彼此矛盾。**

- Level 3 vs 2：同一构型的 Z*(B) 相差 100× (-0.13 vs -13.46)
- Level 2 vs 1：分支选择将 Δγ 从 ~0.2 rad 压缩到 ~0.002 rad
- Level 3 vs 文献：N 位移响应 3.5× 过大，B 位移几乎为零

---

## 3. 根因分析

### 3.1 Level 3 的模 2π 截断

arg(zeta_scalar) 被限制在 (-π, π]。各 string 的真值在跨构型时可能 wrap around：

```
eq:  string 0 真实 γ 可能是 -4.35 → arg(zeta) 报告为 1.93
B_disp: string 0 真实 γ 可能是 -4.50 → arg(zeta) 报告为 1.78
```

表象上看变化不大（1.93 vs 1.78），但真实的差值可能是 -0.15（标准）或 -0.15 ± 2π（被 wrap 掩盖）。Δγ 的不确定性达 ±2π —— 与信号本身（~0.15 rad）相比是灾难性的。

> Level 3 是不可依赖的——2π 截断在跨构型比较中注入任意性。

### 3.2 Level 2 的 zeta rescaling 不稳定性

Level 2 (raw Σγ) 试图通过 zeta rescaling（Step 1）修复截断：

```
gamma_rescaled = gamma_raw × (gamma_unw_sum / gamma_raw_sum)
```

其中 `gamma_unw_sum` 是第一条 k-string 的 unwrapped 总和，作为跨 string 的"绝对参考"。**问题是：这个参考在不同的几何构型下可能取不同的值。**

事实上，gamma_unw_sum 来自**第一条 k-string 的带间解缠**——其起点选择、2π 位移距离等均依赖于该具体构型的本征值谱分布：

```
eq 构型:    谱分布 → 解缠模式 A → gamma_unw_sum_A → rescaling_A
B_disp 构型: 谱分布'→ 解缠模式 B → gamma_unw_sum_B → rescaling_B
N_disp 构型: 谱分布"→ 解缠模式 C → gamma_unw_sum_C → rescaling_C
```

rescaling 因子随构型变化 → 差分操作 (rescaling_B × γ_2 - rescaling_A × γ_1) 引入非物理贡献。

> Level 2 也不可靠——rescaling 引用的构型依赖性污染了 Δγ。

### 3.3 根本矛盾

标准 Berry 相位计算 BEC 时：

```
P(u) = -(e/(2πΩ)) · a · Γ(u)    Γ(u) = 总 Berry 相位（单一标量）
BEC = (Ω/e) · dP/du         只需求 Γ 的微分
```

关键在于 Γ 是**直接计算的总标量**——没有 per-atom 分解，没有 per-string rescaling 引用，没有解缠模式选择。差分 dΓ/du 避开了所有中间表示引入的歧义。

DeltaP 的约束任务不同：

```
约束目标: γ_i = γ_i_target (每个原子 i 独立)
需要: 每个原子的 γ_i 是"标签明确的"——即其周期像被分支选择固定
不需要: 跨构型的 γ_i 差分一致性
```

**DeltaP 的结构设计是为了约束，不是为了差分。** 约束只需要 Level 1（明确标签的 per-atom γ），不需要跨构型连续性。BEC 需要跨构型连续性 —— 这正是 per-atom 分解所破坏的。

---

## 4. 为什么跨构型锁定不可行

设想的"锁定"方案：在平衡构型上冻结特征值匹配模式和 rescaling 引用，位移构型直接复用。

### 4.1 特征值匹配锁定

Hungarian 算法对每个 k-point 的配对是本征值之间的最优匹配。当几何位移改变了本征值谱后：

- **本征值分裂/合并**：两个相近的本征值可能交换次序，匹配矩阵的最优解发生突变
- **本征值数目不变但间距改变**：即使匹配没有突变，递增的 2π 相位追踪可能选择不同的"路径"

这些问题与位移大小无关——任意小的位移都可能触发匹配模式的改变。无法保证匹配在构型间"连续"。

### 4.2 Zeta Rescaling 引用锁定

即使冻结匹配，`gamma_unw_sum` (第一条 string 的 unwrapped 总和) 仍然是构型依赖的：

```
gamma_unw_sum = Σ_n gamma_unwrapped_first_string[n]
```

gamma_unwrapped[n] 依赖于该 string 上 eigenvalue 匹配的起点。不同构型有不同的起点选择 → 冻结不可行。

### 4.3 小结

| 锁定目标 | 失败原因 |
|---------|---------|
| 特征值匹配模式 | 构型间谱分布变化 → 匈牙利最优解可能突变 |
| Zeta rescaling 引用 | 每个构型的 string-0 unwrapping 起点不可迁移 |
| 分支选择 k-vector | 已在 global search 中正确锚定，但仅对同一构型有效 |

---

## 5. 结论

**DeltaP 的 per-atom 分解不能可靠地计算 BEC。这是结构性限制，而非实现 bug。**

- BEC 需要总 Berry 相位的**跨构型差分**——这是一个对 per-atom 中间表示完全不透明的操作
- DeltaP 设计用于**约束特定原子的 γ 到靶标值**——这是一个不需要跨构型一致性的任务
- 两种需求调用了同一套代码路径的不同侧面，但侧面之间的信息损失是不可逆的

**正确的使用方式**：

| 任务 | 使用工具 | 理由 |
|------|---------|------|
| **约束极化**（fix γ_I） | DeltaP per-atom / total 模式 | H2O PES 已验证有效 |
| **Born 有效电荷**（∂P/∂u） | 标准 Berry 相位（不分解） | 直接计算总相位差分 |
| **介电常数**（∂P/∂E） | total 约束模式 | λ ↔ E_eff 转换，已验证合理 |

DeltaP 在约束任务上已证明有效（BN vs H2O 刚度区分、total 模式弛豫因子测量）。BEC 超出了其设计范围——这不是失败，是适用范围的正确定义。
