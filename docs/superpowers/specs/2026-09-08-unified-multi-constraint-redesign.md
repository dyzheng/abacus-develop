# 统一多约束框架架构重设计（电荷 / 自旋 / 偶极任意组合 + 高效迭代）

> 基线：`module_constraint` 开发者文档（feat/deltap `7bd4fd3ee`，2026-09-08）。
> 目标：把「单模式约束」升级为「同时约束电荷、自旋、分子偶极（或任意组合）」的统一框架，并引入更高效的迭代方案。
> 结论先行：**骨架 90% 已就绪**（μ/Q 已向量化、注入/读数已 Σ_α 求和、仅 v_eff/rho 耦合），核心改动是「约束规格逐约束化」+「耦合求解器」+「松-紧 SCF 调度」。体相极化仍是唯一本质边界（Berry 相位，走 DeltaP）。

---

## 0. 一句话现状与结论

现有实现的四个结构性优点**全部保留**，它们正是多约束扩展的地基：

1. **状态集中于 `ConstraintLoop`**（单例），M2/M3a/M5/M6 全是静态无状态核——多约束只是让核多跑几个分量，不破坏无状态性；
2. **观测量 == 注入算符**由「注入与读数共享同一 `WeightGrid` 实例」构造性保证——多约束下逐条保持同一 w_α 即可；
3. **仅 v_eff / rho 两个耦合点**（ψ 和 H 从不进模块）——新约束类型只要回答"注入哪个势分量、读哪个密度分量"，无需碰 KS 求解器；
4. **μ/Q/力 三者都已是 Σ_α 求和**——多约束是"多几个分量"而非"换结构"。

**要改的只有一处根因**：`constraint_type`（charge|spin）和 `DensityChannel` 目前是 **run 级全局**（§6.2 已点名），混合类型无法表达。把它下放到**逐约束**，其余都是顺水推舟。

---

## 1. 核心数据模型：`ConstraintSpec`（约束规格逐约束化）

每条约束 = 一个四元组，把"类型"还原成它的本质——**权重场 × 密度通道**：

```cpp
enum class ConstraintKind { Charge, Spin, Dipole };   // 只是快捷方式

struct ChannelProfile {          // 通道剖面：决定注入符号 + 读哪个密度分量
    int  read_up;                // 读 ρ↑ 的系数（charge/dipole=+1, spin=+1）
    int  read_dn;                // 读 ρ↓ 的系数（charge/dipole=+1, spin=−1）
    double inj_up;               // 注入 v_eff↑ 的符号（charge/dipole=+μ, spin=+μ）
    double inj_dn;               // 注入 v_eff↓ 的符号（charge/dipole=+μ, spin=−μ）
};

struct ConstraintSpec {
    ConstraintKind kind;
    int   weight_id;             // 指向 WeightGrid 中某条权重（原子组/片段/z 权重）
    ChannelProfile chan;
    double target;               // delta 或 absolute
    TargetMode mode;
    double mu_max;               // 逐约束熔断上限（软/硬通道可不同）
};
```

统一后，三类约束只是不同 `(weight_id, chan)` 组合：

| kind | weight | 读密度分量 | 注入符号（↑/↓） |
|---|---|---|---|
| Charge | 原子/片段 Becke | ρ↑+ρ↓ | +μ / +μ |
| Spin | 原子/片段 Becke | ρ↑−ρ↓ | +μ / −μ |
| Dipole | w = z−z₀（网格函数） | ρ↑+ρ↓ | +μ / +μ |

**关键不变量不变**：注入与读数必须用**同一条 w_α + 同一 `ChannelProfile`**。这由工厂保证——每条约束一个 `(WeightGrid, ChannelProfile)` 对，只读引用同时交给 inject 与 observe。

---

## 2. 各模块改动清单（对照现有 M0–M8）

| 模块 | 现状 | 改动 | 难度 | 必过检查 |
|---|---|---|---|---|
| M0 `partition` | Becke 核 + 导数 | **不改**（偶极 z 权重不是原子权重，不落在此层） | — | — |
| M1 `WeightGrid` | `w_[iat][ir]`、`cw_[alpha][ir]`、`dw_` | 新增**权重来源枚举**：原子 Becke / 片段(求和) / **z 权重**（`cw_α = z_g − z₀`）；z 权重导数 ∂w/∂R=0 | 低 | 单位分解断言**对 z 权重不适用**（Σz≠1，需换成"位置矩一致性"断言）；导数恒 0 |
| M2 `observe` | `Q_α=Σw·d·ΔV`，channel 全局 | channel **逐约束**：按 `spec.chan` 选读 ρ↑/ρ↓ 组合 | 低 | 混合通道 mock 对拍（已知合成密度） |
| M3a `inject_pw` | `v_eff+=Σμw`，channel 全局 | channel 逐约束：按 `inj_up/inj_dn` 分别注入两个自旋势 | 低 | **ObservableEqualsInjection per-channel**（逐条恒等式） |
| M3b `inject_lcao` | W^α 审计仪器 | 无结构改（审计逐 α 已支持） | — | ΣW≡S 已过 |
| **M4 `MuSolver`** | 对角 secant（κ_i 逐分量） | **升级为块对角 + Broyden + 对角 fallback**（见 §4） | **高（核心）** | 病态 mock 映射 + 近共线对（47 步基准，§6.3 已点名） |
| M5 `accounting` | `E_con=Σμ(Q−t)` | 无结构改（已求和）；审计行加 `kind` 字段 | 低 | 审计行 kind 标签 |
| M6 `constraint_force` | `F=−Σμ Σd·∂w/∂R ΔV` | 无结构改（已求和）；偶极 ∂w/∂R=0 → 偶极无显式力项；补**混合力 FD + 力矩 FD** | 中 | 混合力 FD；spin+dipole 同开时力矩 FD |
| M7 `constraint_io` | `type=charge\|spin` | 改为**约束列表**：每条 `{kind, weight, channel, target, mode, mu_max}` | 低 | schema + 组合合法性守卫（如 spin 要求 nspin=2） |
| M8 `ConstraintLoop` | 两阶段状态机 + conv_esolver 门控 | 加**松-紧 SCF 阈值调度**（§4.2） | 中 | 反假收敛全链路 + 混合配置冒烟 |

---

## 3. 高效迭代方案（分四层，按成本递进）

### 3.1 零成本：热启动 + δ 扫描续算
每轮外环用上一轮收敛密度起步；δ 扫描沿 δ 递增续算（上一 δ 的 ρ、μ 作下一 δ 初猜）。**当前代码已隐含，多约束下要显式保证**：μ 向量更新后，密度不重置。

### 3.2 低成本（改 M8）：松-紧 SCF 阈值调度
外环早期 μ 还差得远，SCF 用**松阈值**收敛（少几步对角化）；外环后期才收紧。把"每轮完整 SCF"变成"早期松 SCF + 末尾 1–2 轮紧 SCF"。实现上在 `ConstraintLoop` 增加一个 `scf_thr(outer_step)` 调度表，配 `conv_esolver` 门控。

### 3.3 中成本（改 M4，核心）：块对角 + Broyden 耦合求解
约束 Jacobian 有干净的物理解释（对称、负定）：

$$
J_{\alpha\beta}=\frac{\partial Q_\alpha}{\partial\mu_\beta}
=-\langle w_\alpha|\chi|w_\beta\rangle,\qquad \chi=\text{屏蔽密度响应函数}
$$

实测（R3）：**同类型内耦合弱**（H₂O 约束 Hessian 非对角仅对角 ~0.1%），**跨类型耦合强**（电荷↔偶极是硬×软通道）。所以：

- **块内**（同 kind）：对角 secant 够用（κ_i 逐分量，弱耦合）；
- **块间**（跨 kind）：Broyden 维护约化逆 Jacobian（块级 n×n，n = kind 数），超线性收敛；
- **fallback**：任意 Broyden 步发散 → 回退对角 secant（保留现有护栏）。

这一步把"对角 secant 在耦合通道上的慢收敛/振荡"降为 Broyden 的超线性，**直接减少外环 SCF 轮数**。

### 3.4 高成本（可选，仅在 3.3 不足时）：自洽线性响应 Jacobian
用 DFPT/Sternheimer 一次算出 $J=-\langle w|\chi|w\rangle$（**自洽屏蔽响应**，不是 T-7″ 的冻密度响应），再 Newton 步 1–2 次收敛 μ。这是最激进的方案，但需引入响应方程机制，工程量最大。

> **重要区分**：3.3/3.4 用的都是**自洽屏蔽响应**（Broyden 从真实 SCF 的 Q(μ) 累积，线性响应含屏蔽），与 T-7″ 的**冻密度响应**（无屏蔽、符号反转）是两回事。任何"高效方案"都不得退化成冻密度——那是已封口的坑。

### 3.5 效率对照

| 方案 | 外环轮数 | 每轮成本 | 新增工程量 | 适用 |
|---|---|---|---|---|
| 现状（对角 secant） | 单约束 3–6；多/混合 → 慢或振荡 | 全 SCF | 0 | 单/同类型弱耦合 |
| +热启动+松紧 SCF | 不变 | **降**（早期松） | 低 | 所有场景 |
| +块对角 Broyden | **降**（超线性） | 同 | 中 | 混合类型 |
| +线性响应 Jacobian | 1–2 | 同 + 一次响应 | 高 | 强耦合硬骨头 |

---

## 4. 偶极通道（w = z − z₀）的三个要点

1. **delta 模式强制**：$D_z=\int z\rho$ 随原点平移变 $c\cdot N_{el}$，绝对偶极无物理意义 → 必须 relative-to-reference（现有 delta/参考态两阶段已支持）。
2. **无显式力项**：$\partial(z-z_0)/\partial R_J=0$（z₀ 取固定单元胞中心）→ 偶极约束对原子力无显式贡献，只经自洽密度间接作用；力核更简单。
3. **R5 数值高危**：偶极 vs 电场对偶有 ~250× 刚度失配 + "10 Bohr 盒子"淬火。走统一框架 ≠ 和电荷一样好做，验收协议要细化。

---

## 5. 生命周期陷阱纪律（§3.1 永久检查项，扩展到新通道）

现有两条已踩中的坑（PW `get_vnew()` 快照、LCAO `cal_pulay_fs` Pulay）必须对**每一个新通道**重新回答：

> "力求值 / 能量决算时刻，注入的 μw 还在势里吗？"

新通道（偶极的 z 权重、未来的多极子）统一走 `ConstraintLoop::add_back_constraint_potential()` 的**修正副本**模式，禁止改写共享势。每个新注入点在 spec 里加一条"势存活检查"验收。

---

## 6. 本质边界（不可越过的唯一红线）

- **分子偶极 / 表面垂直偶极**（z 方向有限、有真空）：位置算符良定义 → 走统一框架（w=z）。
- **体相极化 / 表面面内极化**（PBC 方向）：位置算符无定义 → 必须 Berry 相位 / Wilson loop，**保留 DeltaP 独立机制**，不可并入统一权重路径。

---

## 7. 迁移与验证计划（三阶段）

1. **阶段 A（零风险）**：M7 约束列表 schema + M1 z 权重 + M2/M3a 逐约束 channel。验收：混合 mock 对拍 + `ObservableEqualsInjection per-channel` + 组合守卫 sabotage。
2. **阶段 B（核心）**：M4 块对角 + Broyden + 对角 fallback；M8 松紧 SCF 调度。验收：近共线对（47 步基准）+ 病态映射 mock + 反假收敛 + 热启动步数对比。
3. **阶段 C（物理）**：电荷+自旋、电荷+偶极真实体系集成用例；混合力 FD + 力矩 FD；R5 细化偶极验收。体相极化明确不做（走 DeltaP）。

---

## 附：关键公式汇总

注入（逐约束 channel 求和）：
$$
v_{\rm eff}^\sigma(\mathbf r) \;+=\; \sum_\alpha \mu_\alpha\, w_\alpha(\mathbf r)\cdot \mathrm{inj}_\alpha^\sigma,\qquad \sigma\in\{\uparrow,\downarrow\}
$$
读数（逐约束通道）：
$$
Q_\alpha = \sum_g w_\alpha(g)\,\big[\mathrm{read}_\alpha^{\uparrow}\rho_\uparrow(g)+\mathrm{read}_\alpha^{\downarrow}\rho_\downarrow(g)\big]\,\Delta V
$$
力（已求和，偶极 ∂w/∂R=0）：
$$
F_J = -\sum_\alpha \mu_\alpha \sum_g d_\alpha(g)\,\frac{\partial w_\alpha}{\partial R_J}(g)\,\Delta V
$$
记账：$E_{\rm con}=\sum_\alpha\mu_\alpha(Q_\alpha-t_\alpha)$；驻点处 $\mu_\alpha=\partial E/\partial Q_\alpha$（逐条，靠"观测量=注入算符"保证）。
