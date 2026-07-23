# DeltaP 算法使用手册

DeltaP（Delta Polarization）是 ABACUS 中的约束 DFT 模块，用于**约束原子级 Berry 相位（电子极化）**并按每原子分解计算极化贡献。

---

## 一、快速入门

### 最简配置（仅计算 Berry 相位，不施加约束）

```
INPUT 中:
deltap_switch  1
```

### 约束模式（固定 γ 到指定值）

```
INPUT 中:
deltap_switch  1
deltap_corr    1
deltap_target_file  target.dat
```

`target.dat` 每行一个数，按 STRU 中的原子顺序给出每个原子的目标 γ（单位：rad）。

---

## 二、全部参数

### 2.1 总开关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_switch` | bool | false | 启用 DeltaP。为 true 时计算每原子的 Berry 相位 γ_I |
| `deltap_corr` | bool | false | 启用约束哈密顿修正。需要 `deltap_switch=1`。为 true 时施加 λ 约束力 |

### 2.2 计算控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_method` | string | berry_connection | 极化计算方法。`"berry_connection"`: 使用 Berry 联络算子（推荐）；`"wannier"`: 使用 Wannier 函数 |
| `deltap_gdir` | int | 3 | 极化方向。1=x, 2=y, 3=z |
| `deltap_rm` | double | 3.0 | SMO 调制半径（Bohr）。控制 Wannier 投影的局域范围。0 时使用 onsite_radius |
| `deltap_npk_string` | int | 0 | 覆盖 k-string 密度。0 时使用 KPT 网格。设为正值可增加 k-string 密度以提高精度 |

### 2.3 规范固定

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_gauge_mode` | string | none | 规范固定模式。`"none"`: 不固定；`"smo_anchored"`: 使用 SMO 锚定 |
| `deltap_anchor_thr` | double | 1e-8 | SMO 锚定重新选择的阈值 |

### 2.4 λ 控制（约束强度）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_lambda_init` | double | 0.0 | λ 的初始值（Ry）。所有原子统一设置。若设为非零值且 `deltap_lambda_step=0`，λ 在整个 SCF 中不变——适用于固定约束力测量（如 dF/dλ BEC 测试） |
| `deltap_lambda_step` | double | 0.5 | λ 更新步长。λ_new = λ + step × (γ - target)。值越大收敛越快但可能不稳定；值越小越稳定但收敛越慢 |
| `deltap_lambda_mixing` | double | 0.0 | λ 混合因子。λ = β·λ_opt + (1-β)·λ_old。0=不混合（直接用计算值），1=完全混合。在 λ 震荡时使用中间值（如 0.1-0.3） |
| `deltap_nscf` | int | 5 | 内层 λ 优化最大迭代次数。0 时使用**同步模式**（每 SCF 步更新一次 λ）。>0 时每 SCF 步内做 nscf 次固定 λ 的子迭代 |
| `deltap_inner_thr` | double | 1e-4 | 激活内层 λ 循环的 drho 阈值。当 drho 低于此值时，电荷密度被认为足够收敛，λ 可以被更新。**两阶段阈值模式**的核心参数 |
| `deltap_conv_thr` | double | 1e-3 | 内层循环收敛阈值。当 max\|γ-target\| < 此值时停止内层迭代 |

### 2.5 两阶段阈值模式

当 `deltap_nscf = 0`（同步模式）时，λ 更新采用两阶段策略：

```
Phase 1: λ = 0，SCF 自然收敛。γ 被测量但不施加约束力
Phase 2: 当 drho < deltap_inner_thr 时，一次性梯度下降更新 λ，然后 λ 冻结
Phase 3: λ 冻结，SCF 带约束继续收敛
```

- `deltap_inner_thr` 控制 Phase 1→2 的切换时机：值越大越早更新（但密度可能不够精确）；值越小更新越晚（密度更精确但耗时更长）
- 典型设置：`deltap_inner_thr = 1.0e-3`，`deltap_lambda_step = 0.01`，`deltap_lambda_mixing = 0.1`

### 2.6 约束模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_constraint_mode` | string | per_atom | 约束模式。`"per_atom"`: 每原子独立约束 γ_i = target_i；`"total"`: 约束总和 Σγ_i = target_total |
| `deltap_constraint_matrix` | string | "" | 约束矩阵文件路径。设置后优先级高于 `deltap_constraint_mode`。详见第 3 节 |

### 2.7 靶标

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_target_file` | string | "" | 靶标文件路径。每行一个数值，`per_atom` 模式需 nat 行，`total` 模式需 1 行。单位为 rad |

### 2.8 其他

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deltap_dk_fd` | double | 1e-6 | T0 校验的有限差分 δk。仅用于开发调试 |

---

## 三、约束矩阵功能

通过 `deltap_constraint_matrix` 可实现任意线性约束 C·γ = t。

### 3.1 文件格式 (constraint.mat)

```
m  n                    ← m=约束数, n=原子数
C_11  C_12  ...  C_1n  t_1      ← 约束1: Σ_j C_1j·γ_j = t_1
C_21  C_22  ...  C_2n  t_2
...
C_m1  C_m2  ...  C_mn  t_m
```

### 3.2 常用约束

| 约束矩阵 C | 文件内容 | 物理含义 |
|-----------|---------|---------|
| 单位矩阵 (per_atom) | `n n` + 逐行 | 与 `per_atom` 模式等价 |
| 全1行 (total) | `1 n` + 一行全1 | 与 `total` 模式等价 |
| 差分约束 | `1 3` + `1 -1 0 t` | γ_1 - γ_2 = t（原子间电荷转移） |
| 加权总和 | `1 n` + 一行权重 | 加权总极化 |
| 部分原子约束 | 只包含部分原子的行 | 其他原子不受约束 |

### 3.3 差分约束的物理意义

约束 γ_A - γ_B = t → λ 是 A↔B 电荷转移的"广义力"。测量 dλ/d(Δγ) 得到原子间**电荷转移刚度**——这是 DeltaP 独有的物理量。

---

## 四、常用计算场景

### 4.1 仅计算 Berry 相位（不约束）

```
deltap_switch  1
deltap_target_file  target.dat   ← 用于分支锚定
deltap_corr    0                 ← 不施加约束力
```

γ 值通过 Wilson Loop 每 SCF 步打印一次。

### 4.2 约束极化 PES

```
deltap_switch  1
deltap_corr    1
deltap_nscf    0
deltap_lambda_init     0.0
deltap_lambda_step     0.01
deltap_lambda_mixing   0.1
deltap_inner_thr       1.0e-3
deltap_target_file     target.dat
```

在不同 `target.dat` 下各自运行 SCF，收集 (γ_target, E, λ)。

### 4.3 固定 λ 测力 (dF/dλ BEC)

```
deltap_switch  1
deltap_corr    1
deltap_lambda_init     δλ      ← 固定非零 λ
deltap_lambda_step     0.0     ← 不更新 λ
deltap_nscf    0
cal_force      1                ← 输出原子力
```

运行两次（λ=0 和 λ=δλ），计算 dF/dλ → Z*。

### 4.4 总约束模式

```
deltap_switch          1
deltap_corr            1
deltap_constraint_mode total
deltap_target_file     target_total.dat   ← 仅一行：目标 Σγ
```

### 4.5 差分约束

```
deltap_switch            1
deltap_corr              1
deltap_constraint_matrix constraint.mat
```

`constraint.mat`：`1 3\n1 -1 0 target_diff`

---

## 五、输出说明

### 5.1 每 SCF 迭代输出

```
[DeltaP P1] iter=10  γ=(4.002, 3.498) λ=(8.09e-07, -2.02e-06) |γ-t|=5.31e-03
```

- **P1/P2/P3**: Phase 1/2/3 阶段标识
- **γ=(...)** : 每原子的 Berry 相位（rad）
- **λ=(...)** : 每原子的约束拉格朗日乘子（Ry）
- **\|γ-t\|**: γ 与靶标的最大偏差（rad）

### 5.2 约束矩阵模式输出

```
[DeltaP P3] iter=50  γ=(3.999, 3.495) C·γ=(7.494) λ=4.66e-07 |γ-t|=1.32e-03
```

- **C·γ=(...)** : 约束空间的实际值（C·γ）
- **λ=...**: 约束空间的 λ 值（每约束一个）

### 5.3 额外诊断

```
[rawG] Σγ_raw=-0.257  γ0=-0.117  γ1=-0.139     ← 原始 γ（分支选择前）
[totalBP] alpha=2 avg_arg(zeta)=1.193980         ← 总 Berry 相位（arg(zeta)平均）
```

- `rawG` 在 `deltap_corr=1` 且 `deltap_switch=1` 时每 SCF 步打印
- `totalBP` 在 Wilson loop 完成时打印

---

## 六、注意事项

1. **效率基组 vs 精度基组**：效率基组（如 `O_gga_6au_100Ry_2s2p1d.orb`）对 BEC 的精度有限。如需对比文献值，使用精度基组（如 10au DZP）

2. **BEC 不能通过 γ 差分计算**：per-atom 分解在跨构型比较 β 时存在分支选择非确定性。使用 dF/dλ 方法代替

3. **零刚度体系**：BN 等级共价体系的 λ 可能位于噪声地板（~1 μRy），Hessian 拟合需多点采样

4. **分支选择**：当靶标值离自然 γ 太远时，K=5 的穷举搜索可能选到错误周期像。建议靶标值在自然 γ 的 ±1 rad 范围内
