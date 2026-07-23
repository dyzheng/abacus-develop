# DeltaP 算法使用手册

DeltaP 是 ABACUS 中的约束 DFT 模块，用于**约束原子级 Berry 相位（电子极化）**并按每原子分解计算极化贡献。设计对标 DeltaSpin（磁矩约束模块），支持 per-atom 约束、总约束、任意线性组合约束。

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

### 2.3 λ 控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_lambda_init` | double | 0.0 | 初始 λ（Ry） |
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

---

## 八、输出说明

```
[DeltaP P3] iter=50  γ=(3.999, 3.495) λ=(8.09e-07, -2.02e-06) |γ-t|=5.31e-03
[E-field] E_eff=1.01e-5 V/Angstrom  (λ_avg=-6.04e-07 Ry)
```

- **P1/P2/P3**：Phase 阶段（仅两阶段模式）
- **λ=(...)**：每原子约束力（Ry/rad）
- **E_eff**：等效电场。`E_eff = -λ_avg × π / a_alpha × 51.42 V/Å`
- **\|γ-t\|**：最大靶标偏差（rad）

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
