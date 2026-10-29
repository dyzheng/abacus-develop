# 2026-10-29 Allocation 复 k 共轭约定修复（约定 C）与验证

## 1. 测试计划

按根因报告（`2026-10-29-allocation-charge-normalization-root-cause.md`）：
把目标正交化统一到约定 C——

```
phi_orth_a = sum_b conj(G)_ab phi_b
overlap_a  = sum_b G_ab <phi_b|psi>
d(phi_orth) = conj(G) dphi + conj(dG) phi
d(overlap)  = dG raw + G d_raw
```

覆盖 value / operator / force 三条路径，验收：

1. Fe2 两片段划分 `|dsum_n| < 1e-8`，SCF 正常收敛；
2. LiCl（Γ）读数/能量逐位不变（修复只应影响广义 k）；
3. H2O allocation 力 FD 仍通过（力路径被改动）；
4. `ctest -R MODULE_ESTATE_constraint` 全绿。

## 2. 测试设置

Fe2/LiCl/H2O 用例与前面各轮相同；4 MPI、单线程 BLAS/OMP；
`build_rel/abacus_basic_para`（性能）与 `build`（单测）。
力 FD：`tests/constraint_s5_parity/pw_alloc_force_{center,plus,minus}`，
O 原子沿 z 位移 ±0.002645886 Å。

## 3. 结果

### 3.1 统一修改点

| 位置 | 修改 |
|---|---|
| `alloc_pw.cpp` `orthogonalize_pw_overlaps` | `conj(G) raw` → `G raw` |
| `alloc_pw.cpp` `orthogonalize_pw_overlaps_derivative` | `conj(g) d_raw + conj(dg) raw` → `g d_raw + dg raw` |
| `alloc_pw.cpp` `build_pw_alloc_orthogonal_phi` | `G phi` → `conj(G) phi` |
| `alloc_pw.cpp` `vphi_orth`（V 变换目标） | `G vphi` → `conj(G) vphi` |
| `op_pw_alloc.cpp` `build_band_force_data` | 用 `phi_orth`（代替 raw φ）+ `G raw` 系数 |
| `alloc_pw_test.cpp` 解析导数 | `df G + f dG` → `conj(G) df + conj(dG) f` |

### 3.2 Fe2：分区精确 + 收敛

```text
#ELEC ITER# 63，final status: CONVERGED
FINAL_ETOT_IS = -6370.6039641351 eV
ALLOC-SPLIT audit sum_n=32.0000 nelec=32 dsum_n=8.57e-11
                 sum_n_T_all=11.6398 sum_n_c=20.3602 all_target_plus_c=32.0000
                 sum_band_norm=32.0000 sum_real_norm=20.3602
```

- `dsum_n` 从 **+0.376 → 8.6×10⁻¹¹**（机器精度级）；
- 目标 + 互补 = N；band norm = N；
- 迭代数 63（修复前 55；观测量改变后轨迹不同，仍正常收敛）；
- 末态 ETOT 移动 −6.2×10⁻³ eV：**这是坐标修正后的不同约束态**，不是数值误差。

### 3.3 LiCl（Γ）逐位不变

```text
q = 2.947887372  （修复前同值）
FINAL_ETOT_IS = -643.9756523800  （修复前同值）
```

确认修复只影响广义 k，Γ 点行为不变。

### 3.4 H2O allocation 力 FD

| 量 | 数值 |
|---|---:|
| `F_FD(z)`（δ=0.002645886 Å） | −5.668 eV/Å |
| 中心点解析 `F_analytic(O,z)` | −5.669 eV/Å |
| 差 | **0.001 eV/Å**（判据 0.0129，13× 富余） |

力路径（`build_band_force_data` 与导数）在约定 C 下自洽。

### 3.5 单元测试

`ctest -R MODULE_ESTATE_constraint`：**23/23 PASS**（87 s）。
其中 `AllocationOperatorDerivativeDecomposition` 在修复后先失败、
更新解析导数约定后通过——正好锁定了"value 与 derivative 必须同约定"。

## 4. 分析

1. 根因（复 k 共轭配对）已修复并全路径统一；Fe2 分区审计到 1e-11，
   LiCl Γ 不变，力 FD 通过，单测全绿。**charge allocation 的分区正确性
   障碍已解除。**
2. 被修正的观测量是一个**不同的坐标**：Fe2 同靶 ±3.8 μB 的末态 ETOT
   移动 6.2 meV、m_T 从 ±3.60 变为 ±3.70。语义闭环文档中的定量表应以
   本次修复后的数值为准（定性结论不变：`q = m_T + m_c`，且三者与
   DeltaSpin/Becke 不同口径）。
3. 性能：63 步 vs 55 步（同量级，外层步数仍由响应层缺失主导）；
   `fold_potential` 默认开启与 mixing-reset 修复不受影响。

## 5. 下一步

1. 以修复后的数值更新语义闭环表中的 `m_T/m_c/q` 与 ETOT（并重跑
   `target_l` 扫描，确认只有 l=2 可达 ±3.8 的结论不变）。
2. 电荷 allocation 恢复为"可生产候选"，但仍需：多片段/非共线/kpar>1
   的 `Σq=N` 集成回归（`dsum<1e-8`）与 `target_l` 敏感性扫描。
3. 性能（63 步 vs DeltaSpin 13 步）继续由响应层/continuation 方向决定；
   单次 hPsi 已达标（fold 后 3.93 ms）。
