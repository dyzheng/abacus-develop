# DeltaP 算法使用手册

DeltaP 是 ABACUS 中的约束 DFT 模块，用于**约束原子级 Berry 相位（电子极化）**并按每原子分解计算极化贡献。设计对标 DeltaSpin（磁矩约束模块），支持 per-atom 约束、总约束、任意线性组合约束。

---

## 一、快速入门

### 1.1 仅诊断（不计入 γ，不修改哈密顿量）

```
INPUT:
  deltap_switch  1
  deltap_corr    0
```

γ 值每 SCF 步打印一次，但不施加约束力、不影响 SCF 收敛。

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
| `deltap_switch` | bool | false | 启用 DeltaP。为 true 时计算每原子 Berry 相位 |
| `deltap_corr` | bool | false | 启用约束哈密顿修正。为 false 时仅诊断 γ 不改变 SCF |

> `deltap_corr` 独立存在的原因是：用户可能只想要 γ 的诊断输出（类似于 `berry_phase=1` 的旧功能），而不希望约束力影响 SCF 收敛。对于约束 DFT 应用（PES、BEC 测量），必须设为 true。

### 2.2 计算控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_method` | string | berry_connection | `"berry_connection"`：Berry 联络算子路径（推荐）；`"wannier"`：Wilson loop + Resta-Z 路径 |
| `deltap_gdir` | int | 3 | 约束施加方向：1=x, 2=y, 3=z。三方向的 γ 均计算输出 |
| `deltap_gauge_mode` | string | none | 规范固定模式。详见第三节 |
| `deltap_anchor_thr` | double | 1e-8 | SMO 锚定重选的投影阈值（仅 `smo_anchored` 模式有效） |

### 2.3 λ 控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_lambda_init` | double | 0.0 | 初始 λ（Ry），所有原子统一 |
| `deltap_lambda_step` | double | 0.01 | 梯度下降步长 |
| `deltap_lambda_mixing` | double | 0.1 | 混合因子：λ = β·λ_new + (1-β)·λ_old |
| `deltap_inner_nmax` | int | 0 | 内层优化最大迭代数。0=两阶段阈值模式 |

### 2.4 两阶段阈值模式（`deltap_inner_nmax=0`）

```
Phase 1: λ = 0, SCF 自然收敛
  ↓ drho < deltap_inner_thr
Phase 2: 一次性梯度下降更新 λ, mix_reset()
  ↓
Phase 3: λ 冻结, SCF 带约束继续收敛
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_inner_thr` | double | 1e-3 | Phase 1→2 切换的 drho 阈值 |
| `deltap_conv_thr` | double | 1e-3 | 内层收敛阈值（仅 `deltap_inner_nmax>0` 时有效） |

### 2.5 约束模式

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `deltap_constraint_mode` | string | per_atom | `"per_atom"` 每原子；`"total"` 总和 Σγ |
| `deltap_constraint_matrix` | string | "" | 约束矩阵文件。设置后覆盖 `constraint_mode` |

### 2.6 靶标指定

| 方式 | 说明 |
|------|------|
| **STRU 关键字**（推荐） | 原子行尾加 `dp_target γ_val dp_constrain 0/1` |
| **target.dat 文件** | INPUT 设 `deltap_target_file target.dat` |

---

## 三、规范固定模式 (`deltap_gauge_mode`)

Wannier 函数的相位在 k 空间存在规范自由度：每个 k 点、每个带的波函数可以独立乘一个相位因子 `e^{iθ}`，不改变物理结果。但 Berry 联络的计算涉及相邻 k 点间波函数的内积，**相位不连续会导致离散导数发散**。

### 3.1 模式对比

| 模式 | 行为 |
|------|------|
| `"none"` | 不固定规范。直接用原始 Kohn-Sham 波函数计算 Berry 联络。适合大部分场景（KS 波函数随 k 连续变化） |
| `"smo_anchored"` | 对 **第一条 k-string** 的每个带在每个 k 点选取投影最强的 SMO 轨道作为锚点，追踪其相位，强制波函数该分量的相位连续。详见 3.2 |

### 3.2 smo_anchored 算法

```
对每个带 n:
  Phase 1 — k_0 确定锚点:
    在所有原子/轨道中，找 |D_I[iat][lm][n]| 最大的那个（即该带投影最强的 SMO）
    将波函数该分量的相位归一化到实数正方向

  Phase 2 — k_1..k_{N-1} 追踪:
    在后续 k 点，用同样的锚点 (iat, lm) 追踪相位
    若锚点投影衰减到 anchor_thr 以下 → 重新搜索最强锚点

该规范固定仅作用于第一条 k-string，用于确定 Berry 联络所有计算的相位参考。
```

### 3.3 适用范围

- `berry_connection` 方法下有效（`wannier` 方法有自己的相位处理逻辑）
- 大多数体系用 `"none"` 即可。仅当遇到相位跳变导致极化异常振荡时才需开启

---

## 四、Berry Connection vs Wannier (`deltap_method`)

| | berry_connection | wannier |
|---|-----------------|---------|
| 路径 | 有限差分 Berry 联络矩阵元 | Wilson loop（S 乘积 + 对角化）|
| 额外 | — | + Resta-Z 电子中心位移 |
| 测试状态 | ✅ 全部验证通过 | 未充分测试 |
| 推荐 | **默认** | 用于交叉验证 |

两种方法在原理上等价。`berry_connection` 直接计算 `⟨u_nk|∂_k|u_mk⟩`，`wannier` 通过对角化 `∏_j S(k_j,k_{j+1})` 取本征值。推荐用 `berry_connection` 作为主方法，`wannier` 作为交叉验证。

---

## 五、STRU 靶标关键字

### 5.1 格式

在 `ATOMIC_POSITIONS` 的原子坐标后添加：

| 关键字 | 值类型 | 默认 | 说明 |
|--------|--------|------|------|
| `dp_target` | double | 0.0 | 该原子的目标 Berry 相位（rad） |
| `dp_constrain` | int | 1 | 0=不约束（λ 始终为 0），1=约束 |

### 5.2 示例

**全部约束**：
```
B  0.0  1  0.00  0.00  0.00  dp_target 4.0 dp_constrain 1
N  0.0  1  0.25  0.25  0.25  dp_target 3.5 dp_constrain 1
```

**部分约束**（H2O 中只约束 O 和 H1，H2 自由）：
```
O   0.0  1  6.744  7.500  8.086  dp_target -6.40 dp_constrain 1
H1  0.0  1  6.744  7.500  7.043  dp_target -3.17 dp_constrain 1
H2  0.0  1  8.256  7.500  8.086  dp_constrain 0
```

> `dp_constrain=0` 的原子 λ 始终为 0，不参与 λ 更新，但其 γ 依然被计算和输出。

### 5.3 优先规则

1. `deltap_target_file` 设了 → 从文件读（向后兼容）
2. 没设 `deltap_target_file` → 从 STRU 读
3. `deltap_constraint_matrix` 设了 → 覆盖上述两种方式的靶标

---

## 六、约束矩阵 (`deltap_constraint_matrix`)

约束矩阵将 DeltaP 从"每原子固定 γ_i"推广到"固定任意线性组合 Σ_j C_ij·γ_j = t_i"。

### 6.1 文件格式 (`constraint.mat`)

```
m  n                    ← m=约束数, n=原子数
C_11  C_12  ...  C_1n  t_1
C_21  C_22  ...  C_2n  t_2
...
```

### 6.2 示例：差分约束 γ_O − γ_H1 = −3.23（H2O）

**约束文件 `constraint.mat`**：
```
1  3
1  -1  0  -3.23
```

**约束效果**：系统被强制满足 `γ_O − γ_H1 = −3.23`。第三个原子（H2）系数为 0，不受约束。λ 只有 1 个分量（与约束数对齐），其物理意义是 O↔H1 电荷转移的"广义力"。

**实测结果**（H2O，5 个靶标值）：

| target_diff | γ_O | γ_H1 | γ_O−γ_H1 | λ (μRy) | 偏差 |
|------------|------|------|----------|---------|------|
| −3.430 | −6.600 | −3.171 | **−3.429** | +0.38 | 1.4e-4 |
| −3.330 | −6.495 | −3.165 | **−3.330** | +0.57 | 5.7e-4 |
| −3.230 | −6.396 | −3.165 | **−3.231** | −0.89 | 9.5e-4 |
| −3.130 | −6.299 | −3.171 | **−3.128** | +2.14 | 1.8e-3 |
| −3.030 | −6.197 | −3.165 | **−3.032** | −1.83 | 1.8e-3 |

约束被精确满足（偏差 < 2 mrad），λ 与靶标差线性相关。测量 dλ/d(Δγ) 得到原子间**电荷转移刚度**。

### 6.3 约束矩阵的行为

1. **分支选择**：顺序贪心算法在每个原子上选 k-vector 使 C·γ 逼近 t
2. **λ 更新**：在约束空间（m 维）做梯度下降，再转换回 per-atom 有效 λ
3. **哈密顿修正**：使用 per-atom 有效 λ（与旧代码兼容）：
   ```
   λ_eff[iat] = Σ_α λ[α]·C[α][iat]
   ```
4. **输出**：显示每原子 γ、约束空间 C·γ、约束空间 λ

### 6.4 常用约束一览

| 约束 | 行 | 物理含义 |
|------|-----|---------|
| γ_1 = t | `1 0 0 ... t` | 单原子约束 |
| Σγ = t | `1 1 1 ... t` | 总极化约束 |
| γ_1 − γ_2 = t | `1 −1 0 ... t` | 原子间电荷转移 |
| w_1γ_1 + w_2γ_2 = t | `w_1 w_2 0 ... t` | 加权总极化 |
| γ_1=t, γ_2=t' | 两行 | 多原子独立约束 |

---

## 七、常用计算场景

### 7.1 约束极化 PES

```
deltap_switch  1
deltap_corr    1
```

STRU 中设 `dp_target`，不同靶标各自运行 SCF，收集 (γ, E, λ)。

### 7.2 dF/dλ 测 BEC

```
deltap_switch  1
deltap_corr    1
deltap_lambda_init     1e-5     ← 固定 λ
deltap_lambda_step     0.0      ← 不更新 λ
cal_force              1
```

λ=0 和 λ=δλ 各运行一次，`dF/dλ → Z*`。

### 7.3 差分约束（电荷转移刚度）

```
deltap_constraint_matrix  constraint.mat
```

`constraint.mat` 中 `C = [1, −1, 0]`，扫描不同 t 值 → dλ/d(Δγ)。

---

## 八、输出说明

```
[DeltaP P3] iter=50  γ=(3.999, 3.495) λ=(8.09e-07, -2.02e-06) |γ-t|=5.31e-03
```
- **P1/P2/P3**：Phase 阶段
- **γ=(...)**：每原子 Berry 相位（rad）
- **λ=(...)**：每原子 λ（Ry）
- **|γ-t|**：最大靶标偏差（rad）

约束矩阵模式额外显示：
```
C·γ=(7.494) λ=4.66e-07
```

---

## 九、与 DeltaSpin 对照

| 功能 | DeltaP | DeltaSpin |
|------|--------|-----------|
| 总开关 | `deltap_switch` | `sc_mag_switch` |
| 约束开关 | `deltap_corr` | switch 即约束 |
| 靶标指定 | STRU `dp_target` 或文件 | STRU `mag`/`magmom` |
| λ 步长 | 固定梯度 `lambda_step` | BFGS 自适应 `alpha_trial` |
| λ 混合 | `lambda_mixing` | 无 |
| 内层 | `inner_nmax` | `nsc`+`nsc_min` |
| 约束组合 | `constraint_matrix` | 无 |
| 约束跳过 | `dp_constrain 0` | `sc 0 0 0` |
| 加速 | 无 | `sc_acceleration_mode` |
| 扫描 | 无 | `linear_scan` |
