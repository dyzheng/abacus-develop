# Fe3O4 电荷-磁矩耦合研究测试计划

**目的**：利用 DeltaQS 方法系统研究磁铁矿 (Fe3O4) 中电荷有序与磁序的耦合关系
**体系**：Fe3O4 反尖晶石结构，空间群 Fd-3m (#227)

---

## 1. 物理背景

### 1.1 晶体结构

Fe3O4 是反尖晶石结构 AB₂O₄：
- **A 位** (8a, 四面体): Fe³⁺，与 4 个 O 配位
- **B 位** (16d, 八面体): Fe²⁺·⁵（高温相）或 Fe²⁺/Fe³⁺（低温相），与 6 个 O 配位
- **O 位** (32e): O²⁻

惯用胞含 56 原子（8 Fe_A + 16 Fe_B + 32 O），原胞含 14 原子。

### 1.2 Verwey 转变

- T_V ≈ 120 K：从立方 Fd-3m 转变为单斜 Cc
- 高温相：B 位 Fe 等价 (Fe²⁺·⁵)，t₂g↓ 电子离域
- 低温相：B 位电荷有序 (Fe²⁺/Fe³⁺)，t₂g↓ 电子局域化
- 带隙：高温 ~0 eV（半金属），低温 ~0.15 eV

### 1.3 科学问题

1. **电荷有序的驱动力**：Hubbard U 驱动还是电子-声子耦合？
2. **charge-spin 耦合强度**：约束电荷时磁矩如何响应？
3. **E(N,M) 能量面**：不同电荷/磁矩组合的能量代价？
4. **Verwey 转变的 DeltaQS 特征**：约束能否区分高温/低温相？

---

## 2. 计算设置

### 2.1 结构选择

| 方案 | 原子数 | k点 | 适用阶段 |
|------|--------|-----|---------|
| 原胞 (primitive) | 14 (2 Fe_A + 4 Fe_B + 8 O) | 4×4×4 | Phase 1-3 |
| 惯用胞 (conventional) | 56 (8 Fe_A + 16 Fe_B + 32 O) | 2×2×2 | Phase 4-5 |

> 建议从原胞开始，验证后再用惯用胞。

### 2.2 计算参数

```
INPUT_PARAMETERS
  calculation     scf
  basis_type      lcao
  ecutwfc         100
  nspin           2              # collinear
  smearing_method gaussian
  smearing_sigma  0.01
  mixing_type     broyden
  mixing_beta     0.3
  ks_solver       genelpa
  symmetry        0              # 关闭对称性以允许电荷有序

  # DFT+U（关键！Fe3O4 需要 U 来正确描述）
  dft_plus_u      1
  orbital_corr    2              # simplified LDA+U
  hubbard_u       4.0 0.0        # Fe 3d: 4.0 eV, O: 0

  # DeltaQS
  sc_mag_switch     1
  sc_charge_switch  1
  sc_thr            1e-4
  sc_charge_thr     0.02
  nsc               50
  alpha_trial       0.01
  sc_charge_alpha   0.1
  sccut             3.0
  sc_charge_sccut   5.0

  pseudo_dir    /path/to/PP
  orbital_dir   /path/to/ORB
```

### 2.3 赝势与轨道

| 元素 | 赝势 | 轨道 | Z_val |
|------|------|------|-------|
| Fe | Fe.upf (ONCV PBE) | Fe_gga_8au_100Ry_4s2p2d1f.orb | 16 |
| O | O.upf (ONCV PBE) | O_gga_8au_100Ry_2s2p1d.orb | 6 |

---

## 3. 分阶段测试方案

### Phase 0: 基准计算（无约束）

**目标**：建立无约束基态，标定自然投影电荷和磁矩

#### 0a: 无自旋极化 SCF
```
nspin = 1
sc_mag_switch = 0
sc_charge_switch = 0
```
- 记录 E_total, 带隙, DOS
- 验证金属态（高温相特征）

#### 0b: 亚铁磁 SCF
```
nspin = 2
sc_mag_switch = 0
sc_charge_switch = 0

STRU 初始磁矩:
  Fe_A: mag -4.0 (自旋向上，反平行)
  Fe_B: mag +4.0 (自旋向下，平行)
  O:    mag 0.0
```
- 记录 E_total, M_total, 各原子 Mi
- 验证亚铁磁序：M_total ≈ 4 μB/f.u.
- **标定自然投影电荷**：N_A⁰, N_B⁰
- 记录 N_total = Σ(电子数)

**预期结果**：

| 量 | 预期值 | 说明 |
|----|--------|------|
| M(Fe_A) | -3.8 ~ -4.2 μB | d⁵ 高自旋，S=5/2 |
| M(Fe_B) | +3.5 ~ +3.8 μB | d⁵·⁵ 高自旋，S≈9/4 |
| M_total | ~4.0 μB/f.u. | 亚铁磁补偿 |
| E_gap | ~0 eV | 半金属（GGA+U 可能不准） |
| N_A⁰ | ~8-10 e | 四面体位投影电荷 |
| N_B⁰ | ~8-10 e | 八面体位投影电荷 |

#### 0c: 投影完备性分析
- 计算 ΣN_i (所有原子的投影电荷之和)
- 计算 N_gap = N_total - ΣN_i
- 计算投影完备性 p = ΣN_i / N_total
- **预期**: p ≈ 0.6-0.85（first-zeta 不完备）

---

### Phase 1: 纯自旋约束

**目标**：验证磁性结构，标定自旋约束行为

#### 1a: 约束所有 Fe 位磁矩
```
sc_mag_switch = 1, sc_charge_switch = 0

Fe_A: mag -4.0 sc 1 1 1
Fe_B: mag +3.7 sc 1 1 1
```
- 验证 CG 收敛
- 记录 λ_A, λ_B
- 比较 E 与无约束基态

#### 1b: 选择性约束
只约束 Fe_B 位磁矩，Fe_A 和 O 自由：
```
Fe_A: mag -4.0 sc 0 0 0
Fe_B: mag +3.7 sc 1 1 1
```
- 观察 Fe_A 磁矩的自洽响应
- 分析 A-B 磁耦合

---

### Phase 2: 纯电荷约束

**目标**：研究 B 位电荷有序，标定电荷约束行为

#### 2a: 均匀 B 位电荷约束
所有 Fe_B 约束到相同的 N_B：
```
sc_mag_switch = 0, sc_charge_switch = 1

Fe_B (all 4 in primitive cell): tc N_B⁰ cq 1
```
- 扫描 δN = -0.5, -0.3, -0.1, 0.0, +0.1, +0.3, +0.5
- 记录 E(δN), μ(δN), M(δN)
- **提取电荷硬度**: k_N = d²E/dN²

#### 2b: 电荷有序态
将 B 位分成两组（模拟 Verwey 有序）：
```
Fe_B1 (2 atoms): tc N_B⁰ + δN  cq 1   # 富电子 (Fe²⁺)
Fe_B2 (2 atoms): tc N_B⁰ - δN  cq 1   # 缺电子 (Fe³⁺)
```
- 扫描 δN = 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
- 记录 E(δN), μ_B1(δN), μ_B2(δN)
- 观察磁矩响应：M_B1(δN), M_B2(δN)
- **提取电荷有序能**: ΔE_CO = E(δN=1) - E(δN=0)

**预期物理**：
- δN=0: 高温相（等价 B 位）
- δN=0.5: 部分有序
- δN=1.0: 完全有序 (Fe²⁺/Fe³⁺)
- E(δN) 应为开口向上的抛物线
- μ_B1 ≈ -μ_B2（反对称约束力）

#### 2c: N_total 守恒验证
每个 δN 点验证：
- N_total 不变
- ΣN_i 变化反映间隙电荷重分配
- α 补偿系数分析

---

### Phase 3: Q+S 联合约束

**目标**：研究电荷-磁矩耦合

#### 3a: 联合约束基本测试
```
sc_mag_switch = 1, sc_charge_switch = 1

Fe_A: mag -4.0 sc 1 1 1
Fe_B1: mag +3.5 sc 1 1 1  tc N_B⁰+0.5  cq 1
Fe_B2: mag +4.0 sc 1 1 1  tc N_B⁰-0.5  cq 1
```
- 验证统一 CG 收敛
- 记录 λ 和 μ 的耦合行为
- 分析收敛曲线中 RMS_spin 和 RMS_charge 的同步性

#### 3b: 交叉耦合响应
固定 Fe_B1 电荷，扫描 Fe_B1 磁矩：
```
Fe_B1: tc N_B⁰+0.5 cq 1, mag 扫描: +2.0, +3.0, +3.5, +4.0, +4.5
```
- 记录 μ(δM), E(δM)
- **提取** ∂μ/∂M (电荷约束力对磁矩的响应)

固定 Fe_B1 磁矩，扫描 Fe_B1 电荷：
```
Fe_B1: mag +3.7 sc 1 1 1, tc 扫描: N_B⁰-0.5 ~ N_B⁰+0.5
```
- 记录 λ(δN), E(δN)
- **提取** ∂λ/∂N (自旋约束力对电荷的响应)

**关键物理量**：交叉耦合系数
```
k_NM = ∂²E / ∂N ∂M
```
- 通过混合偏导数近似
- 验证 Maxwell 关系：∂μ/∂M = ∂λ/∂N

---

### Phase 4: E(N, M) 能量面扫描

**目标**：系统映射 B 位的能量面

#### 4a: 二维网格扫描
对单个 Fe_B 原子：
```
N ∈ {N_B⁰ - 0.5, N_B⁰ - 0.25, N_B⁰, N_B⁰ + 0.25, N_B⁰ + 0.5}
M ∈ {M_B⁰ - 1.0, M_B⁰ - 0.5, M_B⁰, M_B⁰ + 0.5, M_B⁰ + 1.0}
```
- 5×5 = 25 个计算点
- 记录 E(N,M), μ(N,M), λ(N,M)
- 拟合二次面：E(N,M) = E₀ + ½k_N(ΔN)² + ½k_M(ΔM)² + k_NM·ΔN·ΔM

#### 4b: 能量面可视化
- 绘制 E(N,M) 等值线图
- 绘制 μ(N,M) 和 λ(N,M) 梯度场
- 分析能量面各向异性
- **提取耦合强度**: γ = |k_NM| / √(k_N·k_M)

#### 4c: 金属-绝缘体相界
- 对每个 (N,M) 点计算带隙 E_gap
- 绘制 E_gap(N,M) 等值线
- 定位 E_gap = 0 的金属-绝缘体相界线

---

### Phase 5: Verwey 转变分析

**目标**：用 DeltaQS 特征量表征 Verwey 转变

#### 5a: U 依赖性
在不同 Hubbard U 下重复 Phase 2b：
```
U = 0, 1, 2, 3, 4, 5, 6 eV
```
- 记录 ΔE_CO(U) = E(δN=1) - E(δN=0)
- **定位 U_c**：ΔE_CO(U_c) = 0
- 预期 U_c ≈ 3-4 eV

#### 5b: 电荷有序态电子结构
对 δN = 1.0 (完全有序) 态：
- 计算 PDOS：Fe_B1(Fe²⁺) vs Fe_B2(Fe³⁺)
- 分析 t₂g↓ 带的分裂
- 计算带隙
- 绘制电荷密度差图：Δρ = ρ(ordered) - ρ(disordered)

#### 5c: 磁交换参数
从不同磁构型的能量提取 J：
```
H = -Σ J_ij S_i·S_j
```
- J_AB (A-B 超交换)
- J_BB (B-B 双交换)
- 分析 δN 对 J 的影响

---

## 4. 结果验证清单

### 物理合理性
- [ ] A 位磁矩 ~ -4 μB (Fe³⁺, d⁵)
- [ ] B 位磁矩 ~ +3.7 μB (Fe²⁺·⁵, d⁵·⁵)
- [ ] M_total ≈ 4 μB/f.u. (亚铁磁)
- [ ] 高温相 DOS(E_F) > 0 (半金属)
- [ ] k_N > 0, k_M > 0 (稳定性)
- [ ] k_NM 的符号和量级合理

### 方法验证
- [ ] E(N) 二次拟合 R² > 0.99
- [ ] μ = dE/dN (误差 < 10%)
- [ ] ∂μ/∂M ≈ ∂λ/∂N (Maxwell 关系)
- [ ] |k_NM| < √(k_N·k_M) (Cauchy-Schwarz)
- [ ] N_total 在所有约束计算中守恒

### 与文献对比
| 量 | 本工作预期 | 文献值 | 来源 |
|----|-----------|--------|------|
| M(Fe_A) | ~4 μB | 4.0-4.2 μB | 中子衍射 |
| M(Fe_B) | ~3.7 μB | 3.5-4.0 μB | 中子衍射 |
| E_gap (ordered) | ~0.1-0.3 eV | ~0.15 eV | 光学/输运 |
| U_c | ~3-4 eV | ~3.5 eV | PRB 76, 195121 |
| J_AB | 负 (反铁磁) | -15 ~ -25 meV | 自旋波 |
| J_BB | 正 (铁磁) | +5 ~ +15 meV | 自旋波 |

---

## 5. 计算资源估算

| Phase | 计算点 | 单点时间(4核, 原胞) | 总时间 |
|-------|--------|---------------------|--------|
| 0 (基准) | 3 | 10 min | 30 min |
| 1 (纯spin) | 2 | 15 min | 30 min |
| 2a (均匀charge) | 7 | 15 min | 1.75 h |
| 2b (电荷有序) | 7 | 15 min | 1.75 h |
| 3a (联合基本) | 1 | 20 min | 20 min |
| 3b (交叉耦合) | 10 | 20 min | 3.3 h |
| 4a (2D扫描) | 25 | 20 min | 8.3 h |
| 5a (U依赖) | 42 | 15 min | 10.5 h |
| 5b-c (分析) | 5 | 30 min | 2.5 h |
| **总计** | **~102** | | **~29 h** |

> 使用 16 核节点: ~7 h
> 使用 64 核节点: ~2 h

---

## 6. 文件组织

```
tests/Fe3O4_DeltaQS/
├── STRU_primitive      # 原胞 14 原子
├── STRU_conventional   # 惯用胞 56 原子
├── KPT_primitive       # 4×4×4
├── KPT_conventional    # 2×2×2
├── Phase0_baseline/
│   ├── 0a_nospin/
│   └── 0b_ferrimagnetic/
├── Phase1_spin/
│   ├── 1a_all_constrained/
│   └── 1b_selective/
├── Phase2_charge/
│   ├── 2a_uniform/
│   ├── 2b_ordered/
│   └── 2c_conservation/
├── Phase3_combined/
│   ├── 3a_basic/
│   └── 3b_cross_coupling/
├── Phase4_landscape/
│   └── 4a_grid_scan/
├── Phase5_verwey/
│   ├── 5a_U_dependence/
│   ├── 5b_electronic/
│   └── 5c_exchange/
└── analysis/
    ├── fit_ENM.py
    ├── plot_landscape.py
    └── compare_literature.py
```
