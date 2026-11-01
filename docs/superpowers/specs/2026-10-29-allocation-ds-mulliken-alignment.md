# 2026-10-29 更正：DeltaSpin 与 Mulliken 的对齐（此前对比无效）

## 1. 此前的错误

在讨论"allocation q=3.80 vs Becke 2.81 vs Mulliken 2.40"时，我把
**LCAO 无约束（free）态**的 Mulliken（±2.404）当成"DeltaSpin 约束后的
Mulliken"，并据此推断"DeltaSpin 投影与 Mulliken 差 1.4 μB"。**该推断错误。**

## 2. 根因：DeltaSpin 输入缺 `sc`，约束静默不生效

- DeltaSpin 的**约束掩码**来自 STRU 的 **`sc i j k`** 关键字；没有 `sc` 时
  `constrain_ = (0,0,0)`。
- 单分量 `mag x` 会经 `read_atoms_helper.cpp:286-305` 的后处理写入
  `m_loc_.z = x`，所以**目标值本身是对的**；缺的只是掩码。
- 掩码为空时，主 λ 循环里 `constrain_` 全零 → λ 更新被全部 mask → **约束
  完全不生效**，运行退化为无约束 SCF（且**不报错**）。

我此前的 LCAO DS 输入只有 `mag 3.8`、没有 `sc` → 目标 3.8 但掩码空 →
**DeltaSpin 根本没运行**，Mulliken 自然停在 free 值 ±2.40。

## 3. 正确输入与更正后的数据

正确写法（collinear，目标取 z 分量）：

```
0.00 0.00 0.00  mag 0 0 3.8   sc 0 0 1
0.51 0.51 0.51  mag 0 0 -3.8  sc 0 0 1
```

| 测量（同一体系 Fe2） | Fe1 (μB) | 说明 |
|---|---:|---|
| LCAO **free** Mulliken | **2.404** | 无约束态 |
| LCAO **DeltaSpin 目标 ±3.8（含 `sc`, `scf_thr=1e-5`）** 的 Mulliken | **3.6323**（14 步收敛） | λ 约束收敛（Mi≈3.8）；该态有电荷残差地板，1e-8 阈值不可达，见 `2026-10-30-lcao-ds-3.8-scf-plateau.md` |
| PW DeltaSpin ±3.8（`sc 1 1 1`）的末态原子矩 | **3.8001** | 13 步收敛（`fe2_bcc_ds`，有效基准） |
| PW free onsite l=2 投影 | 3.697 | 投影口径 |
| Allocation `q`（约束态） | 3.800 | `q = m_T + m_c` |
| Becke 空间矩（约束态） | 2.808 | 空间划分 |

## 4. 更正后的判决

1. **DeltaSpin 投影与 Mulliken 是对齐的**：正确激活 DS 后，目标 ±3.8 对应
   Mulliken ≈3.63（且 PW 侧收敛到 3.8001）。所谓 1.4 μB 差是"DS 未生效 +
   拿了 free 态"造成的假象。
2. **projector/布居家族（DeltaSpin ≈ Mulliken ≈ allocation q）是一致的**
   （3.6–3.9）；**Becke 空间矩（2.8–2.9）才是离群口径**。
3. 因此"Becke vs allocation 差 1 μB"的根源是 **Becke 的空间划分定义**，
   不是 allocation 算错；**allocation 与 DeltaSpin/Mulliken 同属投影家族**。
4. 我此前"allocation q 不是原子矩、Mulliken 2.4 才正确"的结论**作废**。

## 5. 对效率结论的影响：无

`fe2_bcc_ds` 带有 `sc 1 1 1`，其"13 步收敛到 ±3.8"是**有效基准**。因此
allocation 63 步 vs DeltaSpin 13 步（4.85×）的效率对比**仍然成立**。

## 6. 建议的代码守卫（重要可用性缺陷）

`sc_mag_switch=1` 但 STRU 没有 `sc` → DeltaSpin **静默不约束**，只按初始
磁化跑一个普通 SCF。这是容易误判的陷阱（本次已实际发生）。建议在
`configure/init` 处加守卫：`sc_mag_switch` 为真且所有 `constrain_` 为 0 时
**WARNING_QUIT**（提示"add `sc i j k` to the STRU atom lines"），而不是
静默运行。

## 7. 下一步

1. 给 DeltaSpin 加"空掩码"守卫（小改动，防再次误判）。
2. 用正确的 `sc` 格式重跑 LCAO DS 到收敛（当前 600 步停在 Mulliken 3.63），
   得到与 3.8 完全对齐的收敛读数。
3. 语义表统一为"同一态、同一口径"三列：projector（DeltaSpin/allocation）、
   Mulliken（布居）、Becke（空间）；明确每种实验对哪一列。
