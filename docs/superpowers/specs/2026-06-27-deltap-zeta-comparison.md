# DeltaP 算法拆解与对比测试：执行结果

> **日期**: 2026-06-27
> **测试**: BaTiO3 ref, 10×10×10, symmetry=-1, berry_phase + DeltaP 同时运行

---

## 1. 测试 1: zeta_string 对比 (环节 B vs B')

### 1.1 方法

在 berry_phase 的 `stringPhase` 函数中添加 debug 输出:
```cpp
ofs << index_str << " " << zeta.real() << " " << zeta.imag() << " " << log(zeta).imag();
```

在 DeltaP 的 k-string 循环中添加 debug 输出:
```cpp
ofs << istring << " " << zeta.real() << " " << zeta.imag() << " " << std::arg(zeta);
```

zeta = ∏_j det(O_j) (berry_phase) vs zeta = det(W) = det(∏_j O_j) (DeltaP)

### 1.2 结果

100 个 k-string 的 arg(zeta) 对比:

| string | berry arg | DeltaP arg | diff | |diff| |
|--------|-----------|------------|------|-------|
| 0 | -1.124 | -0.238 | -0.886 | 0.886 |
| 1 | -1.118 | -0.314 | -0.804 | 0.804 |
| 2 | -1.100 | +2.675 | +2.509 | 2.509 |
| 3 | -1.061 | -3.031 | +1.970 | 1.970 |
| 4 | -1.006 | -3.022 | +2.016 | 2.016 |

**统计**:
- mean |diff| = 1.804
- max |diff| = 3.070
- 100/100 strings 有 |diff| > 0.01
- 74/100 strings 有 |diff| > 1.0

**berry avg arg** = -1.039, **DeltaP avg arg** = +0.237

### 1.3 模长对比

| | berry |zeta| | DeltaP |zeta| |
|---|---|---|
| string 0 | 1.063 | 0.0017 |

berry_phase 的 |zeta| ≈ 1 (O_j 接近酉), DeltaP 的 |zeta| ≈ 0.002 (O_j 远非酉)。

### 1.4 结论

**zeta_string 完全不一致！** arg 和模长都不同。

**这意味着环节 A (重叠矩阵 O_j) 在两个实现中不同。**

---

## 2. 根因分析: 二中心积分表不一致

### 2.1 berry_phase 的重叠计算

berry_phase 通过 `unkOverlap_lcao` 类:
1. `cal_orb_overlap()`: 用 `center2_orb11` 计算 ⟨φ_μ|φ_ν(R)⟩
2. `cal_orb_r_overlap()`: 用 `center2_orb21_r` 计算 ⟨φ_μ|r|φ_ν(R)⟩
3. `prepare_midmatrix_pblas()`: 组装 M_{μν} = Σ_R phase × (overlap + i×position)
4. `det_berryphase()`: O = C† · M · C, return det(O)

### 2.2 DeltaP 的重叠计算

DeltaP 通过两个不同的途径:
1. `overlap_intor_->snap()`: 用 `TwoCenterIntegrator` 计算 ⟨φ_μ|φ_ν(R)⟩
2. `r_overlap_->get_psi_r_psi()`: 用 `cal_r_overlap_R` 计算 ⟨φ_μ|r|φ_ν(R)⟩
3. `compute_S_dk_link()`: 组装 S_dk_{μν} = Σ_R phase × (overlap + i×position)
4. pzgemm: O = C† · S_dk · C

### 2.3 差异

| | berry_phase | DeltaP |
|---|---|---|
| overlap ⟨φ\|φ(R)⟩ | center2_orb11 (unkOverlap_lcao) | snap (TwoCenterIntegrator) |
| position ⟨φ\|r\|φ(R)⟩ | center2_orb21_r (unkOverlap_lcao) | get_psi_r_psi (cal_r_overlap_R) |
| 初始化 | cal_orb_overlap (用 LCAO_Orbitals orb) | cal_r_overlap_R::init (用 LCAO_Orbitals orb) |
| 积分方法 | Center2_Orb::Orb11 (球贝塞尔展开) | TwoCenterIntegrator (可能不同方法) |

**关键**: `snap` (TwoCenterIntegrator) 和 `center2_orb11.cal_overlap` 可能使用不同的数值方法计算同一个积分。

### 2.4 验证方法

对同一对轨道 (T1, L1, N1, M1, T0, dtau), 比较:
- `snap` 返回的 ⟨φ_μ|φ_ν(R)⟩
- `center2_orb11.cal_overlap` 返回的 ⟨φ_μ|φ_ν(R)⟩

如果两者不同, 则找到了根因。

---

## 3. 修复方案

### 方案 1: 让 DeltaP 使用与 berry_phase 相同的积分表

在 `ctrl_scf_lcao.cpp` 中, 创建 `unkOverlap_lcao` 对象, 传给 DeltaP:
```cpp
unkOverlap_lcao berry_overlap;
berry_overlap.cal_orb_overlap(ucell, gd, orb, pv);
berry_overlap.cal_orb_r_overlap(ucell, gd, orb, pv);
dp.init(..., &berry_overlap);
```

DeltaP 的 `compute_S_dk_link` 改用 `berry_overlap.psi_psi` 和 `berry_overlap.psi_r_psi`。

### 方案 2: 直接调用 det_berryphase

在 DeltaP 的 k-string 循环中, 不自己构建 S_dk, 而是直接调用 `det_berryphase`:
```cpp
zeta = zeta * lcao_method.det_berryphase(ucell, ik_L, ik_R, dk, nocc, pv, psi, kv);
```

但这只返回 det(O_j), 不返回 O_j 矩阵。无法做特征值分解。

### 方案 3: 修改 det_berryphase 返回 O_j 矩阵

修改 `det_berryphase` 使其同时返回 O_j 矩阵 (nocc×nocc), 供 DeltaP 做特征值分解。

**推荐**: 方案 3, 最直接, 确保 O_j 完全一致。

---

## 4. 下一步

1. 验证 snap vs center2_orb11 的差异 (打印同一对轨道的值)
2. 实现方案 3 (修改 det_berryphase 返回 O_j)
3. 重新运行测试 1, 确认 zeta 一致
4. 如果 zeta 一致, 继续测试 2 (unwrap) 和测试 4 (Z*)

