# DeltaQ / DeltaQS Q 系列测试辅助说明

> 对应: `可靠性测试设计集/Q01–Q10`，分支: `feat/deltaqs-unified-framework`
> 本文是对 Q 系列测试设计文档的 DeltaQ 操作补充。注意 DeltaQ 当前**仅在 LCAO 路径下可用**（与 DeltaSpin 共用算子框架）。

---

## 一、DeltaQ 操作速查

### 1.1 开关与模式

DeltaQ 通过 DeltaSpin 框架间接控制，没有独立的模块开关：

```
sc_charge_switch  true      # 电荷约束主开关
sc_mag_switch     false     # 自旋约束（为 false = 纯 DeltaQ 模式）
```

`sc_qs_mode` 控制联合行为：

| sc_qs_mode | sc_charge_switch | sc_mag_switch | 含义 |
|-----------|-----------------|---------------|------|
| auto | true | false | 自动推导为 deltaq |
| deltaq | true | false | 纯电荷约束 |
| deltaspin | false | true | 纯自旋约束（退化为 DeltaSpin） |
| deltaqs | true | true | 电荷+自旋联合约束 |

### 1.2 电荷约束参数

```
sc_charge_mode    absolute     # absolute=投影电荷, delta=相对平衡值, valence=价电子数
sc_charge_thr     1e-4         # 电荷约束收敛阈值 (e, RMS)
sc_charge_alpha   0.01         # mu 试探步长 (eV/e²)
sc_charge_sccut   3.0          # mu 单步变化上限 (eV/e)
```

### 1.3 STRU 中定义约束

每个原子的 block 中可加三行关键字：

```
Fe
0.0
1
0.0 0.0 0.0
target_charge  15.8            # 目标电荷
mu             0.0             # 初始 Lagrange 乘子
constrain_charge 1             # 1=约束, 0=自由
```

- `target_charge`: 在 absolute 模式下为投影电荷目标值 (e); delta 模式下为相对值; valence 模式下为价电子数
- `mu`: 初始 Lagrange 乘子 (Ry/e)。通常设 0.0，代码会自动优化
- `constrain_charge`: 0 = 该原子不被约束，1 = 被约束

### 1.4 关键输出

运行后 stdout 中搜索：

```
Charge of each atom:
ATOM  0:  N = 15.9927   target = 16.0   λ = -0.0034   |N-t| = 7.25e-03
```

含义：当前投影电荷、目标值、Lagrange 乘子 μ、偏差。

能量修正：
```
E_charge = -Σ μ_i * N_i = −0.0543 eV       （电荷约束能）
E_spin   = -Σ λ_i * M_i = 0.0000 eV        （自旋约束能，deltaq 模式下为 0）
```

物理能量 = E_band − E_charge − E_spin（代码自动处理，`E_KohnSham` 已修正）。

### 1.5 地面态搜索（实验性）

```
sc_ground_state_search  true
sc_outer_max_iter       50
sc_outer_thr            1e-4
```

开启后 DFT 能量对 μ 做外循环优化——用于找无约束的 KS 基态（避免初始猜测导致的局部极小），或扫描 (N,M) 空间的多个起点。

---

## 二、常见问题

### DeltaQ 不生效

- 检查 `sc_charge_switch true`（不是 `charge_switch` 或 `deltaq_switch`）
- 检查 STRU 中每个需约束的原子有 `constrain_charge 1`
- 检查 `basis_type lcao`（DeltaQ 仅在 LCAO 路径下可用）

### 电荷不收敛

- 降低 `sc_charge_alpha` (默认 0.01 eV/e² → 0.001)
- 降低 `sc_charge_sccut` (默认 3.0 eV/e → 1.0)，限制 mu 单步跳变
- 降低 `sc_charge_thr` (默认 1e-4 e)，先从 1e-3 开始收敛再收紧
- 检查 SCF 本身是否收敛（`scf_thr`, `mixing_beta`）

### Lagrange 乘子震荡

- 降低 `sc_charge_alpha`（震荡常是步长过大）
- 增大 `mixing_beta`（SCF 混合不够稳定 → mu 对噪声敏感）

### 开壳层计算

DeltaQ 支持 nspin=2。此时电荷约束作用于总密度 (ρ↑ + ρ↓)，自旋约束作用于自旋密度 (ρ↑ − ρ↓)。两者可在 `deltaqs` 模式下同时工作。

---

## 三、逐测试 DeltaQ 操作要点

以下仅补充设计文档未涵盖的 DeltaQ 操作细节。

### Q01 — CP-0 等价性

**验证场景**: target_charge = 无约束计算的投影电荷值（或 mu=0 且 constrain_charge=0）。

**操作**:
1. 先跑无约束 SCF → 记录 E₀ 和 `N_I`（无 DeltaQ 时的投影电荷）
2. 在 STRU 中设 `target_charge N_I`, `constrain_charge 1`, `mu 0.0`
3. 跑 DeltaQ → 对比 E_DeltaQ 和 E₀

**判据**: |ΔE| ≤ 1e-5 Ha。

**陷阱**: 第 1 步和第 2 步必须在完全相同的其他设置下跑（同 scf_thr, 同 mixing 等）。若 STRU 中 `mu` 未明确设 0，代码可能沿用默认值导致非零偏移。

---

### Q02 — H₂O 充电曲线与 IP

**前置**: 非中性计算协议（MP 修正、电位对齐、盒尺寸缩放）。

H₂O 总电荷 Q = −1/0/+1 三个态。DeltaQ 在每个态内约束片上（per-atom）电荷扫描。

**操作**:
1. 设 `nelec` 不同值实现总电荷变化（或等电子假设下改变 target_charge）。
2. 对每个总电荷态，做一系列 target_charge 扫描。
3. 记录 E_phys(N) 曲线，取 ∂E/∂N ≈ −μ 与实验 IP/EA 对标。

**输出**: stdout 中 `Charge Force` 行给出 μ 当前值，等价于化学势。

---

### Q03 — 带电水团簇三态

(H₂O)₆ 团簇解离为两个三聚体片段 (A 和 B)，总电荷 Q=−1/0/+1。片间距离 d=1–10 Å 扫描。

**操作**:
1. STRU 中分别标记 A 片段和 B 片段的原子（`constrain_charge 1`）。
2. 对 A 设 `target_charge N_A`，对 B 设 `target_charge N_B`（N_A + N_B = 总电子数 + Q）。
3. 逐 d 跑 DeltaQ → 记录收敛后的 `N_A(d)`, `μ_A(d)`。

**关键**: 距离 >4 Å 后，`N_A` 应收敛到整数极限（−1/0 或 0/+1）。团簇几何需逐 d 生成。

---

### Q04 — 给受体 CT 约束扫描

对标传统 CDFT（constrained DFT）：给体-受体对，通过 target_charge 差值扫描电荷转移坐标。

**操作**:
1. STRU 中标记给体(D)和受体(A)原子（分别 constrain_charge 1）。
2. 设 `target_charge_D` 从局域极限（D⁺A⁻）到离域极限（D⁰A⁰）扫描 5–9 个点。
3. 记录每个靶标的 E_phys 和 μ_D, μ_A。

**判据**: μ_D − μ_A 应等于 E_phys 对 CT 坐标的导数。

---

### Q05 — 带电缺陷形成能（FNV）

**前置**: 带电超胞的 FNV（Freysoldt-Neugebauer-Van de Walle）修正协议。

在超胞中约束缺陷原子的电荷。需要做多个盒尺寸的重复计算以实施 FNV 外推。

---

### Q06 — 免费双梯度验证

ΔU = U(N+δ) − U(N) 对比 −μ。同样对自旋通道做 ΔM 对比 −λ。

**操作**:
1. 对给定 (N, M)，记录收敛的 μ。
2. 改变 target_charge 为 N+δ，重跑 → ΔE/δN。
3. 对比 −μ 和 ΔE/ΔN。

**判据**: |ΔE/ΔN + μ| ≤ 5×10⁻³ Ry/e。

---

### Q07 — 电荷划分稳健性

对比 DeltaQ CSZ 投影电荷、Hirshfeld 电荷、Bader 电荷对同一密度的划分结果。

**操作**: 同一 SCF 密度，分别用三种方法计算各原子电荷。DeltaQ 的输出 `N_I` 可直接取。Hirshfeld 和 Bader 需后处理工具。

---

### Q08 — DeltaQ+DeltaSpin 联合 (O₂)

**前置**: Q01, Q06。

设 `sc_charge_switch true, sc_mag_switch true, sc_qs_mode deltaqs`。在 (N, M) 二维网格上扫描。O₂ 为开壳层 (nspin=2)，是验证联合约束的标志体系。

**操作**: STRU 中对两个 O 原子分别设 `target_charge N_I` 和 `target_mag M_I`。对 (N,O1, N,O2; M,O1, M,O2) 空间做网格扫描。使用 `run_qs_grid_scan` 自动遍历。

---

### Q09 — 约束收敛稳健性图谱

对 H₂O 九宫格参数扫描 (mixing_beta, sc_charge_alpha, sc_charge_thr)，绘制收敛迭代数热力图。

**操作**: 写脚本遍历参数组合，每种组合跑一次 DeltaQ 并记录 SCF 迭代数。发散时标记。输出热力图。

---

### Q10 — 大体系可扩展性

选取 ≥100 原子的体系，测试 DeltaQ 的 wall time 与原子数的标度关系。主要用于性能评估而非正确性。

---

## 四、命令模板

### 单点 DeltaQ 运行

```bash
# INPUT 关键行:
# sc_charge_switch true
# sc_charge_mode absolute
# sc_charge_thr 1e-4
# sc_charge_alpha 0.01

abacus > run.log 2>&1
grep "Charge of each atom\|E_charge\|Charge Force" run.log
```

### 检查电荷约束收敛

```bash
grep "|N-t|" run.log          # 每个原子的电荷偏差
grep "E_charge" run.log       # 约束能
grep "SCF IS NOT CONVERGED" run.log
```

### target_charge 扫描

```bash
for tgt in 5.8 5.9 6.0 6.1 6.2; do
    sed "s/TARGET_CHARGE/$tgt/" STRU.tmpl > STRU
    abacus > run_${tgt}.log 2>&1
    grep "Charge Force" run_${tgt}.log | tail -1
done
```
