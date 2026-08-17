# DeltaP 算法使用手册

DeltaP 是 ABACUS 中的约束 DFT 模块，用于**约束原子级 Berry 相位（电子极化）**并按每原子分解计算极化贡献。设计对标 DeltaSpin（磁矩约束模块），支持 per-atom 约束、总约束、任意线性组合约束。

两种约束变量（`deltap_observable`）：

- **operator（默认，Route A+）**：约束算符期望 Γ_I = Γ_I^HR + Γ_I^HK，记账
  escon = −Σλ_I·Γ_I 使报告能量 E' = E_KS(ψ\*)。这是当前推荐路径（力/relax
  均走此路径，工作窗见 §2.8）。
- **gamma（旧模式）**：约束 Wilson loop Berry 相位 γ_I。功能可用但力/relax
  已封口（弱耦合 + γ-hold 路径泄漏，见 `2026-08-13-deltap-capability-boundaries.md`）。

---

## 一、快速入门

### 1.1 仅诊断（计算 γ，不修改哈密顿量）

```
INPUT:
  deltap_switch  1
  deltap_corr    0
```

### 1.2 约束模式 — STRU 中指定靶标（推荐）

**INPUT**：
```
deltap_switch  1
deltap_corr    1
```

**STRU**（原子行尾加关键词）：
```
O
0.0
1
6.744  7.500  8.086  dp_target -6.40  dp_constrain 1
H
0.0
2
6.744  7.500  7.043  dp_target -3.17  dp_constrain 1
8.256  7.500  8.086  dp_target -3.17  dp_constrain 1
```

### 1.3 Route A+ 约束 / 场模式（`deltap_observable operator`，推荐）

**INPUT**：
```
deltap_switch   1
deltap_corr     1
```

**约束极化（Γ-path，靶点靠近自然极化）**：先用 `deltap_corr 0` 跑一次拿自然
Γ（`[DeltaP]` 打印），写入 t_Γ 文件后冻结：
```
deltap_proxy_target_file  t_gamma_star.dat
deltap_secant             off        ← 冻结 t_Γ* 跨离子步（relax 必须）
deltap_inner_nmax         0          ← 同步 λ 更新（内循环对 γ 失效）
```

**场模式（均匀 λ ≈ 长程电场）**：所有原子 λ 相同（`deltap_lambda_init` +
`deltap_lambda_step 0` 冻结），等效电场 E_eff = +πλ/(2a)（公式 (b)，实测
响应校准 ×1.6）。窗口 |λ| ≤ 0.007 Ry 内能量/极化响应与锯齿场对拍 <1 meV。
场模式力侧当前挂起（见能力边界文档 §1.2）。

---

## 二、全部参数

### 2.1 总开关

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_switch` | bool | false | 启用 DeltaP |
| `deltap_corr` | bool | false | 启用约束哈密顿修正。false=仅诊断 |

### 2.2 计算控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_method` | string | berry_connection | `"berry_connection"`（推荐）、`"wannier"` |
| `deltap_gdir` | int | 3 | 约束方向：1=x, 2=y, 3=z。γ 对三方向均输出 |
| `deltap_rm` | double | 3.0 | SMO 重叠截断半径（Bohr） |
| `deltap_gauge_mode` | string | none | 规范固定：`"none"`、`"smo_anchored"` |
| `deltap_anchor_thr` | double | 1e-8 | 锚定重选阈值 |
| `deltap_dk_fd` | double | 1e-6 | Berry 联络有限差分 δk（T0 验证用） |

### 2.3 λ 控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_lambda_init` | double | 0.0 | 初始 λ（Ry） |
| `deltap_lambda_init_file` | string | "" | 逐原子初始 λ 文件（每行一个，nat 行；覆盖 `deltap_lambda_init`） |
| `deltap_lambda_step` | double | 0.01 | 梯度下降步长（仅 `inner_nmax=0`） |
| `deltap_lambda_mixing` | double | 0.1 | 混合因子（仅 `inner_nmax=0`） |
| `deltap_inner_nmax` | int | 0 | **内层 BFGS 最大迭代数**。0=两阶段阈值模式；3-5=内层优化 |
| `deltap_inner_thr` | double | 1e-3 | drho 阈值：触发 λ 更新 |
| `deltap_conv_thr` | double | 1e-3 | 内层收敛阈值 |

### 2.4 两阶段阈值模式（`deltap_inner_nmax=0`，默认）

```
Phase 1: λ = 0, SCF 自然收敛
  ↓ drho < deltap_inner_thr
Phase 2: 梯度下降更新 λ, mix_reset()
  ↓
Phase 3: λ 冻结, SCF 带约束继续收敛
```

每次 SCF 步均需电荷密度混合——梯度步偏后需多次 SCF 修正。

### 2.5 内层 BFGS 优化模式（`deltap_inner_nmax=3~5`，推荐）

```
Phase 1: λ = 0, SCF 自然收敛
  ↓ drho < deltap_inner_thr
内层: BFGS 优化 λ（冻结密度，3-5 次对角化）
  ↓ 收敛，锁定
冻结层: λ 固定，SCF 继续收敛
```

**与两阶段对比**：

| | 两阶段 (nmax=0) | 内层 BFGS (nmax=3) |
|---|---------------|-------------------|
| λ 优化算法 | 固定步长梯度下降 | BFGS 共轭梯度（对齐 DeltaSpin） |
| 每试步成本 | 全 SCF（混密），~1 次 | 仅对角化（不混密），~0.1 次 SCF |
| 约束矩阵支持 | ✅ | ✅ |
| drho 门控 | ✅ | ✅ |
| 收敛后锁定 | ✅ | ✅ |

**推荐**：设 `deltap_inner_nmax 3`。BFGS 在 3 步内找到最优 λ，远快于两阶段模式的重收敛过程。

### 2.6 约束模式

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_constraint_mode` | string | per_atom | `"per_atom"` 每原子独立；`"total"` 总和 Σγ |
| `deltap_constraint_matrix` | string | "" | 约束矩阵文件。设置后覆盖 `constraint_mode` |

### 2.7 靶标指定

| 方式 | 用法 |
|------|------|
| **STRU 关键字**（推荐） | 原子行尾加 `dp_target γ_val dp_constrain 0/1` |
| **target.dat 文件** | INPUT 设 `deltap_target_file target.dat` |
| **约束矩阵文件** | `deltap_constraint_matrix constraint.mat` |

> ⚠️ **无 target 语义分歧（LCAO vs PW）**：LCAO 路径无任何靶标（无
> `deltap_target_file` 且 STRU 无 `dp_target`）时 = **自由跑**（λ≡0，
> 4.3 修复后）；**PW 路径无 target 时 = 约束 γ→0**（历史语义保留，
> `deltap_pw.cpp` 显式文档化）。两后端语义相反——跨基组对比或迁移
> INPUT/STRU 前务必确认。PW 对齐 LCAO 语义属破坏性变更，挂起另议。

### 2.8 Route A+（operator 模式）控制

`deltap_observable operator` 时的关键参数（旧 gamma 模式锁定其中部分，见各行说明）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_observable` | string | operator | 约束变量：`"operator"`（Γ，Route A+，推荐）、`"gamma"`（旧 Wilson 相位） |
| `deltap_drive` | string | proxy | operator 模式 λ 驱动信号：`"proxy"`（默认，λ 残差驱动 Γ 对 t_Γ；secant 校准翻译层）、`"gamma"`（λ 直接驱动报告 γ 对 t_γ，t_Γ/secant 层退役；记账恒等式不变）。gamma 模式锁定 `gamma` |
| `deltap_operator_mode` | string | proxy | operator 模式约束算符：`"proxy"`（τ_α·P̂ 几何代理，默认）、`"hk"`（仅 H_HK，场模式实验开关）、`"ow"`（Ô_w=θ_n·P̂ 精确权重算符，实验性，力未实现）。hk/ow 应力受限（见能力边界） |
| `deltap_secant` | string | on | 外层 t_Γ secant：`"on"`（默认，校准 t_Γ 使 γ→t_γ）、`"off"`（冻结 t_Γ，T3/relax 位移腿必须） |
| `deltap_proxy_target_file` | string | "" | 逐原子 t_Γ 文件（每行一个，nat 行）。覆盖 t_Γ=t_γ 首轮初值；用于跨几何冻结校准后的 t_Γ*（T3 协议） |
| `deltap_outer_nmax` | int | 0 | 单点（scf）固定几何外层 secant 步数：0=历史单发 secant；>0=先 λ=0 自由跑测自然 (Γ,γ)，每步 secant 更新 t_Γ 后重驱动 SCF 直至 \|γ−t_γ\|≤`deltap_outer_thr` |
| `deltap_outer_thr` | double | 1e-2 | 外层 \|γ−t_γ\|∞ 收敛阈值（rad） |
| `deltap_branch_anchor` | string | continuity | 报告 γ 的分支锚定：`"continuity"`（默认，锚到上次测量，分支连续）、`"target"`（靶点感知选择，旧 gamma 模式锁定）。**收敛判据/FD 必须用 continuity 锚**（target-aware 读数钉在靶点上是自证循环，T4a/T3' 教训） |
| `deltap_branch_write` | bool | true | 收敛时是否写 `deltap_branch.dat` 参考：`false` 保留现有参考（多几何/FD 运行必须，防静默覆盖破坏 A/B 可比性，T-9' 守卫） |
| `deltap_inner_scheme` | string | cg | 内循环 λ 更新：`"cg"`（标量 α FR-CG，历史默认）、`"jacobi"`（逐分量 secant，对角 Jacobian——反号分量互不污染，T-7'）。约束矩阵模式锁定 cg |

**Route A+ 工作窗（力/relax 可用性）**：力/relax 只在 **λ 小**（靶点靠近自然
极化）时可信——Stage 4.2 三体系驻点力判据 0.0129 eV/Å 以 1.95× 裕度通过；
任意"严格钉 γ 到远离自然值"的靶点（γ-hold 路径）泄漏 0.5 eV/Å（T3'），
不可用。生产 relax 用法见 §7.5。

---

## 三、STRU 靶标关键字

### 3.1 格式

```
B  0.0  1  0.00  0.00  0.00  dp_target 4.0 dp_constrain 1
N  0.0  1  0.25  0.25  0.25  dp_target 3.5 dp_constrain 1
```

| 关键字 | 值 | 默认 | 说明 |
|--------|-----|------|------|
| `dp_target` | double | 0.0 | 目标 Berry 相位（rad） |
| `dp_constrain` | int | 1 | 0=自由（λ 始终 0），1=约束 |

### 3.2 优先规则

`deltap_target_file` > STRU 关键字 > 默认 0.0

---

## 四、约束矩阵 (`deltap_constraint_matrix`)

### 4.1 文件格式

```
m  n                    ← 约束数, 原子数
C_11 ... C_1n  t_1      ← 约束1
...
C_m1 ... C_mn  t_m
```

### 4.2 示例：差分约束

```
1 3
1 -1 0 -3.23            ← γ_O - γ_H1 = -3.23
```

### 4.3 速查表

| 约束 | 矩阵 | 物理含义 |
|------|------|---------|
| γ_1 = t | `1 0 0 ... t` | 单原子 |
| Σγ = t | `1 1 1 ... t` | 总极化 |
| γ_1−γ_2 = t | `1 −1 0 ... t` | 电荷转移 |
| w·γ = t | `w_1 w_2 ... t` | 加权极化 |

---

## 五、规范固定 (`deltap_gauge_mode`)

| 模式 | 行为 |
|------|------|
| `"none"` | 不固定。适合大部分场景 |
| `"smo_anchored"` | 对第一条 k-string 的每带选最强投影的 SMO 锚定相位。仅 `berry_connection` 方法下有效 |

---

## 六、Berry Connection vs Wannier

| | berry_connection | wannier |
|---|-----------------|---------|
| 路径 | 有限差分 Berry 联络 | Wilson loop + Resta-Z |
| 测试状态 | ✅ 全部验证 | 未充分测试 |
| 推荐 | 默认 | 交叉验证 |

---

## 七、常用计算场景

### 7.1 约束极化 PES

```
deltap_switch  1
deltap_corr    1
deltap_inner_nmax  3        ← 内层 BFGS 优化
```

STRU 中设 `dp_target`，不同靶标各自运行 SCF。

### 7.2 固定 λ 测力 (dF/dλ BEC)

```
deltap_switch  1
deltap_corr    1
deltap_lambda_init     1e-5
deltap_lambda_step     0.0     ← λ 不更新
deltap_inner_nmax      0       ← 不用内层
cal_force              1
```

λ=0 和 λ=δλ 各运行一次，Z* = (a/π) × dF/dλ。

### 7.3 差分约束（电荷转移刚度）

```
deltap_constraint_matrix  constraint.mat
```

`constraint.mat` 中 C = [1, −1, 0]，扫描不同 t → dλ/d(Δγ)。

### 7.4 诊断模式（不约束）

```
deltap_switch  1
deltap_corr    0
```

### 7.5 Γ-path relax（生产用法，Stage 4.3 验证）

靶点=自然极化附近的微小偏离（界面反场补偿类）：

```
deltap_switch           1
deltap_corr             1
deltap_proxy_target_file t_gamma_star.dat   ← λ=0 自然 Γ
deltap_secant           off
deltap_inner_nmax       0                    ← 同步 λ 更新
deltap_lambda_step      0.01
deltap_lambda_mixing    0.1
deltap_inner_thr        1e-3
```

机制：每离子步 λ 在 SCF 内重收敛使 |Γ−t_Γ*|<1e-3，力 = 驻点力（4.2 验证），
λ 自动落在 1e-5–1e-4 Ry 合法区。4.3 实测与纯 DFT 基线逐点一致（能量差
≤1.4e-4 eV、力差 ≤1.2e-3 eV/Å）。**禁止**：无 target 的 relax（4.3 前隐式
Γ→0 已修；现 LCAO 无 target=自由跑 λ≡0，**PW 无 target=约束 γ→0**，语义分歧见
§2.7）；远离自然值的 γ 靶点。

---

## 八、输出说明

```
[DeltaP P3] iter=50  γ=(3.999, 3.495) λ=(8.09e-07, -2.02e-06) |γ-t|=5.31e-03
[E-field] E_eff=1.01e-5 V/Angstrom  (λ_avg=-6.04e-07 Ry)
```

- **P1/P2/P3**：Phase 阶段（仅两阶段模式）
- **λ=(...)**：每原子约束力（Ry/rad）
- **E_eff**：等效电场。gamma 模式 `E_eff = -λ_avg × π / a_alpha × 51.42 V/Å`
  （历史共轭式）；operator 模式（Route A+）用 **公式 (b) `E_eff = +λ_avg ×
  π / a_alpha × 51.42 V/Å`**（D2 锯齿场交叉裁决，2026-08-13）。打印行尾带
  响应校准注记：`[formula (b) πλ/(2a); D2 response-calibrated ×~1.6 (proxy)
  / ×~3.1 (ow O)]`——h2o1 实测真实响应是公式 (b) 的 ~1.6×（proxy）/~3.1×
  （ow），对拍 efield 计算时请乘该校准（详见
  `docs/superpowers/specs/2026-08-13-deltap-d1-d2-kappa-field.md`）。
- **\|γ-t\|**：最大靶标偏差（rad）

> **⚠️ 极化率/偶极换算注意（LCAO 基组）**：γ→μ 换算（F2：μ = 24.28 D/rad × G，
> G=unwrap(Σγ)/2）已对实验偶极验证（0.9%），但**由 λ/电场响应推极化率 α 时，
> 紧缩 LCAO 基会低估 ~3.3×**（h2o1 2s2p1d/2s1p 实测 α=3.02 vs 实验 9.8 Bohr³，
> D2 轮）。任何 α 判据须用同基组参考值或能量 FD 自洽值，勿直接套实验 α。

约束矩阵模式额外显示 `C·γ=(...)` 和约束空间 λ。

---

## 九、与 DeltaSpin 对照

| 功能 | DeltaP | DeltaSpin |
|------|--------|-----------|
| 主开关 | `deltap_switch` | `sc_mag_switch` |
| 约束开关 | `deltap_corr` | switch 即约束 |
| 靶标 | STRU `dp_target` 或文件 | STRU `mag`/`magmom` |
| λ 优化 | BFGS (nmax>0) / 梯度 (nmax=0) | BFGS (nsc>0) |
| 约束组合 | `constraint_matrix` | 无 |
| 约束跳过 | `dp_constrain 0` | `sc 0 0 0` |
| 能量修正 | `dp_escon` | `escon` |
| 内层加速 | 冻结密度对角化 | 子空间/一阶 加速 |
| 扫描 | 无 | `linear_scan` |
| E-field输出 | ✅ | 无 |
