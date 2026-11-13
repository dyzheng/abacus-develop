# 2026-10-30 能量表达式审计：constraint vs DeltaSpin（约束项是否被正确扣除）

## 1. 两套能量修正

**DeltaSpin**（`source_lcao/module_deltaspin/spin_constrain.cpp:38-52`）：
```cpp
escon_ = -Σ_iat (lambda·M)_iat      // 用"计算得到的 M"，负号
```
`E_tot = eband + ... + escon`（`escon` 用于**扣除**约束场 `∫B·m` 的贡献）。

**Constraint 模块**（`module_constraint/constraint_loop.cpp`）：
```cpp
// legacy（本 FeO 用到的路径）
output_energy_correction() = constraint_energy_functional() = Σ_alpha mu_alpha (Q_alpha - t_alpha)
// allocation（注释：eband 已含 mu.Q）
output_energy_correction() = -Σ_alpha mu_alpha * t_alpha
```
`fp_energy.h:50` 亦注明：`cc_escon = "legacy Σμ(Q-t), allocation -Σμt"`，
并被 `calculate_etot()`：`etot = eband + deband + ... + cc_escon`。

**结论**：legacy 的 `Σμ(Q-t)` 与 DeltaSpin 的 `−ΣλM` **结构不同**（符号+多 `−t`），
二者相差 `ΣμQ`。注释虽称"follows DeltaSpin escon form"，**实现并不一致**。

## 2. 判据：自旋翻转对称性

无 SOC 时体系在**整体自旋翻转**下能量不变。据此比较：

| 计算 | 磁性（MBIS Fe0/Fe1 M） | FINAL_ETOT (eV) | 收敛 |
|---|---|---|---|
| **纯 SCF mag0（FM，Fe 皆 +2.4）** | +2.42 / +2.43 | **−7636.46** | 是 |
| **约束（FM，Fe 皆 −3.6）** | −3.61 / −3.38 | **−7643.93** | — |
| 差 | 同一 FM 序（翻转等价） | **−7.47 eV** | |

- **同一 FM 序相差 7.5 eV 物理上不可能**（磁矩差 ~1.2 μB 只值 ~0.1–0.5 eV）。
- **LiCl（nspin 1，纯电荷约束）**此前**正常收敛**（无假 flip）⇒ **标量电荷通道**的
  `eband+deband` 与 `cc_escon` 自洽（`deband` 抵消了标量 `V_con`）。
- ⇒ **问题定位在 nspin 2 的“自旋通道”能量记账**：注入的**自旋分裂势**
  (`V_up += μ_s w, V_dn -= μ_s w`) 的能量**未被 `deband`/`cc_escon` 正确扣除**，
  `etot` 残留 ~`Σμ_s·M` 量级（eV 级）→ **假 BRANCH_FLIP**。

## 3. 回答"参考非全局极小 vs 算法问题"

- **不是**参考态非全局极小（7.5 eV 无法用亚稳解释；纯 AFM/FM 仅差 0.6 eV）。
- **不是** MBIS 权重/观测量 bug（参考 Fe0 charge=14.27、spin=2.85 与独立 MBIS 14.29/2.85 一致）。
- **是能量表达式问题（限 nspin 2 自旋通道）**：legacy `cc_escon = Σμ(Q−t)` **没有**像
  DeltaSpin 那样**扣除约束势的贡献**；对**纯电荷 nspin1**（LiCl）无害，对**自旋/大 Q**（FeO）失效。

## 4. 修复方向（待确认后实施）
- 将**自旋通道**的修正改为 DeltaSpin 形式：`cc_escon(α) = − μ_s,α · M_α`（用计算 M），
  电荷通道相应为 `− μ_c,α · Q_α`；即**扣除 `∫V_con ρ`**（`−ΣμQ`），与 DeltaSpin 一致。
- 或修 `cal_delta_eband`/注入，使 `deband` 包含**自旋分裂** `V_con` 的贡献（与标量情形一致）。
- 验证：LiCl 改为 `nspin 2`+自旋约束；FeO 复测——约束能量应满足自旋翻转对称（plain FM± 一致）。

## 5. 文件
本日志；代码定位见 `fp_energy.h:50`、`constraint_loop.cpp:2177 output_energy_correction`、
`spin_constrain.cpp:38 cal_escon`。无代码改动（纯审计）。
