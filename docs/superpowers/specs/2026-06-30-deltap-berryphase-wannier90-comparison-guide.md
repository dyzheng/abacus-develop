# DeltaP / Berry Phase / Wannier90 极化对比方法指南

> **版本**: 2026-06-30  
> **用途**: 三种极化计算方法的换算公式与对比方法论  
> **测试结果**: 见 `2026-06-30-deltap-three-system-test-results.md`

---

## 1. 三种方法的物理量

| 方法 | 计算内容 | 与 Berry Phase 的对应 |
|------|---------|---------------------|
| **Berry Phase** | 总极化 P = P_离子 + P_电子 | 参考值 |
| **DeltaP** | 电子极化 P_电子（逐原子） | = Berry Phase 电子部分 |
| **Wannier90** | 总极化 P = P_离子 + P_电子（从 WF 中心） | = Berry Phase 总极化 |

**对比原则**:
- DeltaP ↔ Berry Phase **电子**部分
- Wannier90 ↔ Berry Phase **总**极化
- 离子极化的逐原子分配因 mod 归约而不明确，只对比电子部分

---

## 2. Berry Phase 分解

ABACUS 输出：
```
The Ionic Phase:    <ionic_phase>
Electronic Phase:    <electronic_phase>
P =    <P_total>  (mod  <P_quantum>)   e/bohr^2
```

公式：
```
total_phase = ionic_phase + electronic_phase
R_over_V    = P_total / total_phase        (bohr⁻¹)
P_离子      = R_over_V × ionic_phase
P_电子      = R_over_V × electronic_phase
```

---

## 3. DeltaP 输出

文件 `deltap_results.dat`：
```
# Atom    Px  Py  Pz          (a.u.)
     <idx>  0   0   <P_z_elec>     ← 电子极化，单位 e/bohr²
#
# Total   0   0   <P_z_elec_total>
```

DeltaP 只输出电子极化。总极化 = DeltaP 电子 + Berry Phase 离子。

---

## 4. Wannier90 WF 中心公式

从 `*.wout` 提取最终 WF 中心（取最后一组）：
```
WF centre and spread    1  (  x1,  y1,  z1)  <spread>
```

**电子极化公式**（所有坐标使用 bohr）：
```
γ_n = -2π × <r_n> / R                    ← 单个 WF 的 Berry 相位
γ_total = Σ γ_n                           ← 先求和
P_电子 = (γ_total / π) × (R / V)         ← 再转换为极化
```

其中 R = 沿极化方向的晶格常数（bohr），V = 晶胞体积（bohr³）。

**总极化**：
```
P_总 = P_离子 + P_电子
P_离子 = (Σ Z_I × r_I) / V               ← 但需 mod 归约（见 §2）
```

---

## 5. 分支切割修正

**问题**: `Σ γ_n` 可能超出 (-π, π]，与 Berry Phase 的电子相位相差 2π 的整数倍。

**修正步骤**:
1. 计算所有 WF 的 γ_n（不做 mod）
2. 求总和 γ_raw = Σ γ_n
3. 对总和 mod 2π：γ_correct = γ_raw mod 2π（归约到 (-π, π]）
4. 偏移量 offset = γ_raw - γ_correct
5. 按比例分配偏移量到各原子：

```
γ^I_corrected = γ^I_raw - offset × (γ^I_raw / γ_raw)
```

6. 逐原子电子极化：`P^I_电子 = (γ^I_corrected / π) × (R / V)`

**注意**: 对**总和** mod 2π，不是对单个 WF。

---

## 6. 逐原子电子极化对比

| 方法 | 公式 | 权重类型 |
|------|------|---------|
| **DeltaP** | `P^I = Σ_n w^I_n × γ_n × prefactor` | SMO 投影权重（连续 0~1） |
| **Wannier90** | `P^I = (γ^I_corrected / π) × (R / V)` | 最近原子硬归属（0 或 1） |

- **总和**应一致（偏差 < 1%）
- **逐原子分配**可能不同，因为权重定义不同
- DeltaP 的连续权重在共价体系中更合理（如 BN: B 57%/N 43% vs Wannier90: B 100%/N 0%）

---

## 7. 单位约定

| 物理量 | 单位 | 说明 |
|-------|------|------|
| 位置 r | bohr | 1 Å = 1.8897259886 bohr |
| 体积 V | bohr³ | V = (L × 1.8897)³ |
| 晶格常数 R | bohr | 沿极化方向 |
| 极化 P | e/bohr² | SI: C/m² = P × 1.602×10⁻¹⁹ / (5.292×10⁻¹¹)² |
| Berry 相位 γ | rad | 无量纲 |

---

## 8. 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|---------|
| 用 Å 而非 bohr | 极化差 ~6.75 倍 | 所有位置转 bohr |
| 对单个 WF mod 2π | 逐原子值错误 | 对总和 mod 2π |
| DeltaP 电子 vs Berry 总 | 数值和符号都不对 | DeltaP 电子 vs Berry 电子 |
| 离子逐原子 = Z×r/V | 总和不等于总离子极化 | 离子逐原子不明确，只对比电子 |
| Wannier90 最近原子在共价体系 | BN 中 N=0% | 用 DeltaP SMO 权重替代 |
