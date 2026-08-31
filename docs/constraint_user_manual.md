# 实空间权重约束（constraint）用户使用说明

> 版本基线：feat/deltap 分支 `c984b708e`（2026-08-31）。功能状态：**SCF 级电荷/自旋约束已验证可用；约束力已接线但尚未通过 FD 判决（Task 2.6 前不可用于 relax/MD）；应力不支持**。

## 1. 功能概述

在实空间网格上定义原子/片段权重 w_I(r)（逐点 Σ_I w_I ≡ 1），对约束观测量
Q_α = ∫ w_α(r) d_α(r) dr 施加 Lagrange 约束，约束势 V_con = Σ_α μ_α w_α(r)
注入有效势。乘子 μ 由外环逐分量 secant 自动收敛（带 κ 限幅、翻号检测、
μ_max 熔断）。

- **观测量 = 注入算符**：读数与注入共用同一权重场，记账恒等式自洽；
- **基组无关口径**：PW 与 LCAO 在同一密度网格上读数，同密度下逐位一致；
- **每 SCF 打印审计行**：sum rule（total_charge vs nelec）与单位分解偏差
  maxdev（应恒为 ~1e-16）永久自检。

## 2. INPUT 参数

| 参数 | 缺省 | 说明 |
|---|---|---|
| `constraint` | false | 总开关 |
| `constraint_type` | charge | `charge`（通道 ρ）或 `spin`（通道 m=ρ↑−ρ↓，**要求 nspin=2**）。同一次计算只能选一种（混合约束未实现） |
| `constraint_weight_type` | becke | 权重类型，当前仅 `becke`（Becke 模糊 Voronoi 分区，3 阶迭代多项式 + 共价半径异核修正） |
| `constraint_target_file` | （必填） | 靶点 JSON 文件路径；**缺失或为空 → WARNING_QUIT**（不允许无靶点隐式约束） |
| `constraint_target_mode` | delta | `delta`：靶点 = 参考态读数 + 偏移（推荐）；`absolute`：绝对靶点（会打印口径 WARNING——Becke 电荷与 SZV/ Mulliken 量级不同） |
| `constraint_mu_max` | 5.0 | 乘子熔断上限（Ry）；顶限且残差平台 → 判 UNREACHABLE 并报告 Q(μ) 端点 |
| `constraint_thr` | 1e-4 | 约束收敛容差（每分量 \|Q−t\|，单位 e 或 μB） |

## 3. 靶点文件（JSON）

```json
{"targets": [0.1, -0.1], "atoms": [[0], [1, 2]]}
```

- `targets[i]`：第 i 个约束的目标值（delta 模式为偏移量，charge 单位 e、spin 单位 μB）；
- `atoms[i]`：第 i 个约束的原子下标组（0 起；片段 = 多原子求和）；
- 约束个数 = len(targets) = len(atoms)，可多约束（逐分量独立 secant）。

## 4. 输出解读

```
CONSTRAINT_AUDIT nconstraint=1 e_con=... max_residual=... total_charge=... nelec=8 maxdev=2.2e-16
CONSTRAINT_AUDIT c[0] q=6.3554 t=6.3554 mu=-0.1765 res=3.06e-05
[constraint] outer step 6 after SCF iteration 62 (phase=constrained)
[constraint] final status: CONVERGED (targets reached within 0.0001 e)
```

- `q/t/mu/res`：逐约束读数/靶点/乘子（Ry）/残差；
- `total_charge vs nelec`：sum rule 审计（恒等即口径正确）；
- 终态：`CONVERGED`（全部达标）或 `UNREACHABLE`（熔断，附 Q(μ) 端点——目标物理不可达，非数值故障）。

**符号约定**：两通道均为负响应——正 μ 排斥该区域电荷/自旋上。delta>0（增电荷/增磁矩）对应 μ*<0。与 DeltaSpin 的 λ 换算：**μ = −λ**。

## 5. 当前边界（务必阅读）

1. **力**：已接线（PW/LCAO 同一网格核，含驻点守卫），**但未过 stationary4 FD 判决——禁止用于 relax/MD/几何优化**。SCF 单点力打印（test_force=1 的 CONSTRAINT FORCE 块）仅供调试；
2. **应力**：不支持；
3. **混合约束**：同 run 仅单一 constraint_type（charge+spin 混合会 WARNING_QUIT）；
4. **平台**：仅 double/CPU/double_grid（其余 WARNING_QUIT 拒绝）；KPAR=1；
5. **多自旋约束**：近共线靶点（如同时对 O 和 H 约束磁矩）收敛显著变慢（实测 47 外步），属预期行为；
6. 金属/近简并体系未验证。

## 6. 示例用例（已注册测试套件）

- `tests/01_PW/211_PW_constraint_h2o/`：PW 电荷约束（delta=+0.1 e on O）
- `tests/01_PW/212_PW_constraint_h2o_spin/`：PW 自旋约束（delta=+0.1 μB on O）
- `tests/02_NAO_Gamma/212_NAO_constraint_h2o/`：LCAO 电荷约束

各目录 README 含期望值；FD 验证脚本（开发用）：`tests/constraint_fd_force/tools/`。

## 7. 设计/验证文档

架构：`plan-architecture.md`；进展与测试总览：
`docs/superpowers/specs/2026-08-31-constraint-framework-progress-summary.md`；
术语表：`docs/superpowers/specs/deltap-constraint-glossary.md`。
