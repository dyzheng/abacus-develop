# 2026-10-30 FeO charge+spin 同时约束验证（无 DFT+U）

## 0. 体系
- FeO：2 Fe(AFM) + 2 O，rhombohedral `a=8.19`，PBE，**无 DFT+U**，LCAO，`nspin 2`。
- MBIS，`constraint_target.json`：**两条**约束，同一 Fe 原子（index 2）:
  `charge` 与 `spin`（delta 模式）。`constraint_mbis_images 1`。

## 1. 结果

### (a) 小胞必须 `constraint_mbis_images 1`
| images | O0 N | O1 N | Fe0 N | Fe1 N | Fe0 M | Fe1 M |
|---|---|---|---|---|---|---|
| **0 (default)** | 1.42 | 8.68 | 0.38 | 33.5 | — | — |
| **1** | 7.67 | 7.65 | 14.39 | 14.28 | +2.30 | −2.81 |

- `images=0`（direct）→ **corner 低估、极端不对称**（同 Fe2 问题）；`images=1` → **合理对称**：
  O≈7.65、Fe≈14.3（Fe²⁺ 净 +1.7，Fe→O 电荷转移正确），Fe 磁矩 **±2.3–2.8 AFM**。

### (b) charge+spin **同时**约束生效，且参考值与独立一致
```
nconstraint=2
c[0] kind=charge q=14.2727 (未约束参考；独立 MBIS 14.29)
c[1] kind=spin   q= 2.8458 (未约束参考；独立 MBIS  2.85)
```
- **两条约束同时激活**，参考 Q 与独立 MBIS cube 一致 ⇒ **机制正确**。

### (c) 无 DFT+U 的 PBE FeO 无法受控约束
- `step_max=1.5`：BRANCH_FLIP（ΔE 72 eV）。
- `step_max=0.1, probe 0.05`（mu≈0.045）：仍 **BRANCH_FLIP**（约束能量 −561.818 Ry
  **低于**参考 −561.312 Ry，Δ≈6.9 eV）。
- 说明：**无 U 的 PBE FeO 参考态本身亚稳/近简并**；约束势把体系推到**另一（更低）分支**，
  约束循环**正确检测** `BRANCH_FLIP` 并拒绝验证目标。

## 2. 结论
- **MBIS 可以同时约束电荷与自旋**（机制与参考值均已验证）。
- **纯金属**上约束电荷意义不大（用户判断正确）；**FeO** 有真实电荷转移+自旋，但
  **无 U 时 PBE FeO 金属/亚稳**，约束会跨分支 ⇒ 要**受控**的 charge+spin 约束需 **DFT+U**
  （gapping Fe 3d、稳定 Fe²⁺ 高自旋）。DFT+U 暂不在本轮范围。
- 小胞必须 `constraint_mbis_images 1`（默认 0 仅适合大盒）。

## 3. 文件
`weight_grid.cpp`（已有 `constraint_mbis_images`），本日志；无代码改动（纯验证）。
