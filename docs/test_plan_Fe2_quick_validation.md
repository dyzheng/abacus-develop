# Fe2 快速正确性验证测试计划

**目标**：用最小体系 (BCC Fe, 2 原子) 快速验证 DeltaQS 代码正确性
**预计时间**：~30 分钟（4 核）

---

## 0. 基准信息

```
体系:     BCC Fe, a = 2.87 Å (5.42 bohr)
原子数:   2 (Fe@000, Fe@0.5,0.5,0.5)
赝势:     Fe.upf (Z_val = 16)
轨道:     Fe_gga_6au_100Ry_4s2p2d1f.orb
nspin:    2 (collinear)
k点:      2×2×2 MP
ecutwfc:  100 Ry
N_total:  32 electrons (2 × 16)
```

---

## T1: 纯自旋约束回归

**目的**：Phase 1-6 修改未破坏已有 DeltaSpin 功能

### 设置
```
INPUT:
  sc_mag_switch     1
  sc_thr            1e-5
  nsc               20
  alpha_trial       0.01
  sccut             3.0
  sc_strategy       normal
  sc_charge_switch  0

STRU:
  0.00 0.00 0.00  mag  2.0  sc 1 1 1  lambda 1 1 1
  0.51 0.51 0.51  mag -2.0  sc 1 1 1  lambda 1 1 1
```

### 判定标准
- [ ] CG 收敛：RMS < 1e-5 uB
- [ ] |E - E_ref(24)| < 1e-4 Ry
- [ ] Mi[0] = +2.0 ± 0.001 uB, Mi[1] = -2.0 ± 0.001 uB
- [ ] λ 非零且合理

---

## T2: 纯电荷约束收敛

**目的**：新增 Q 约束可收敛，N_total 守恒

### 设置
```
INPUT:
  sc_mag_switch      0
  sc_charge_switch   1
  sc_charge_thr      0.01
  sc_charge_alpha    0.05
  sc_charge_sccut    5.0
  nsc                30

STRU:
  0.00 0.00 0.00  tc 8.5  cq 1
  0.51 0.51 0.51  tc 7.5  cq 1
```

> tc 值需根据 T0 标定的自然投影电荷 N₀ 调整。8.5/7.5 为示例。

### 判定标准
- [ ] CG 收敛：RMS_charge < 0.01 e
- [ ] Ni[0] = 8.5 ± 0.01, Ni[1] = 7.5 ± 0.01
- [ ] N_total = 32.0000
- [ ] μ[0] ≈ -μ[1]（反对称）

---

## T3: Q+S 联合约束收敛

**目的**：统一 CG 同时收敛 spin 和 charge

### 设置
```
INPUT:
  sc_mag_switch      1
  sc_charge_switch   1
  sc_thr             1e-4
  sc_charge_thr      0.01
  nsc                40
  alpha_trial        0.01
  sc_charge_alpha    0.05
  sccut              3.0
  sc_charge_sccut    5.0

STRU:
  0.00 0.00 0.00  mag 2.0  sc 1 1 1  lambda 1 1 1  tc 8.5  cq 1
  0.51 0.51 0.51  mag -2.0 sc 1 1 1  lambda 1 1 1  tc 7.5  cq 1
```

### 判定标准
- [ ] 联合 CG 收敛：RMS_total < max(sc_thr, sc_charge_thr)
- [ ] Mi 和 Ni 同时满足目标值
- [ ] λ 和 μ 均非零
- [ ] E_scon 包含 spin 项 + charge 项
- [ ] N_total = 32.0000

### 收敛曲线分析
记录每步：RMS_total, RMS_spin, RMS_charge
- [ ] 两条曲线同步下降
- [ ] 无严重振荡（允许 2-3 步轻微波动）

---

## T4: 能量一致性

**目的**：E_DFT = E_KS + E_scon 恒等式

### 方法
1. 取 T3 收敛后的输出
2. 从 running_scf.log 提取 E_KS（OUT 报告的总能）
3. 计算 E_scon = -Σ(λ_i·M_i + μ_i·N_i)
4. E_DFT = E_KS + E_scon

### 判定标准
- [ ] E_scon 中 spin 项和 charge 项均非零
- [ ] E_DFT 与无约束能量差异在合理范围（< 5 Ry）
- [ ] cal_escon() 返回值与手动计算一致

---

## T5: 热力学一致性 μ = dE/dN

**目的**：Lagrange 乘子等于能量对约束量的导数

### 方法
扫描 5 个 δN 值，记录 μ 和 E，数值微分比较：

| δN | atom0 tc | atom1 tc | 记录 |
|----|----------|----------|------|
| -0.2 | N₀-0.2 | N₁+0.2 | μ, E |
| -0.1 | N₀-0.1 | N₁+0.1 | μ, E |
| 0.0 | N₀ | N₁ | μ, E |
| +0.1 | N₀+0.1 | N₁-0.1 | μ, E |
| +0.2 | N₀+0.2 | N₁-0.2 | μ, E |

### 判定标准
- [ ] μ(δN) 和 dE/dN(δN) 相对误差 < 10%
- [ ] E(δN) 二次拟合 R² > 0.99

---

## T6: N_total 守恒验证

**目的**：约束单原子时总电子数严格守恒

### 方法
对 T5 的每个 δN 点，从 SCF 输出提取 N_total。

### 判定标准
- [ ] 所有 δN 点：|N_total - 32.0000| < 1e-8
- [ ] N_total 不随 δN 变化

---

## T7: 单原子扰动响应

**目的**：约束一个原子时，观察未约束原子的响应

### 设置
只约束 atom0，atom1 自由：
```
STRU:
  0.00 0.00 0.00  tc N₀+0.5  cq 1
  0.51 0.51 0.51  mag 0.0    cq 0
```

### 记录
- Ni[0]（应 ≈ target）
- Ni[1]（自由响应）
- ΔNi[1] = Ni[1] - N₁_natural
- α = -ΔNi[1] / 0.5（补偿系数）

### 判定标准
- [ ] N_total = 32.0000（守恒）
- [ ] α ∈ (0.1, 0.9)（部分补偿）
- [ ] |ΔNi[1]| > 0.01（确实响应）

---

## T8: μ=0 回归

**目的**：target = 自然值时约束自动消失

### 设置
```
STRU:
  0.00 0.00 0.00  mag M_natural  sc 1 1 1  tc N₀  cq 1
  0.51 0.51 0.51  mag -M_natural sc 1 1 1  tc N₁  cq 1
```

### 判定标准
- [ ] |μ| < 0.01 eV/e
- [ ] |λ| < 0.01 eV/uB
- [ ] |E - E_T0| < 1e-4 Ry

---

## 已知限制与风险

| 限制 | 影响 | 建议 |
|------|------|------|
| 投影不完备 (p < 1) | 约束只控制部分电荷 | 记录 p 值，谨慎解读 |
| 不支持总电荷可变 | 无法模拟开放体系 | 当前 N_total 固定为 32 |
| CG 可能振荡 | 强约束下收敛困难 | 减小约束强度或使用重启 |

---

## 执行脚本

```bash
#!/bin/bash
set -e
ABACUS=/path/to/abacus
MPIRUN="mpirun -np 4"

for test in T1 T2 T3 T4 T5 T6 T7 T8; do
    echo "=== $test ==="
    mkdir -p $test
    cp INPUT_$test $test/INPUT
    cp STRU_$test $test/STRU
    cp KPT $test/
    cd $test
    $MPIRUN $ABACUS > stdout.log 2>&1
    echo "Exit: $?"
    cd ..
done

echo "=== Summary ==="
echo "检查各测试目录下的 stdout.log 和 OUT.*/running_scf.log"
```

---

## 结果汇总模板

```
============================================
 Fe2 DeltaQS Quick Validation
============================================

T1 (Pure Spin):      [PASS/FAIL]  E = ___ Ry
T2 (Pure Charge):    [PASS/FAIL]  μ = ___ eV/e
T3 (Q+S Combined):   [PASS/FAIL]  RMS = ___
T4 (Energy):         [PASS/FAIL]  E_scon = ___ Ry
T5 (Thermo):         [PASS/FAIL]  max error = ___%
T6 (N_total):        [PASS/FAIL]  max |ΔN| = ___
T7 (Single Atom):    [PASS/FAIL]  α = ___
T8 (μ=0):            [PASS/FAIL]  |μ| = ___

Overall: ___/8 PASSED
============================================
```
