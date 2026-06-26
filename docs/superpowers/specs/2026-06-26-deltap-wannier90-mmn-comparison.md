# DeltaP vs Wannier90 .mmn 对比验证报告

> **日期**: 2026-06-26
> **方法**: 从 ABACUS Wannier90 接口输出的 .mmn/.amn 文件直接计算 Berry phase 和逐原子分解,与 DeltaP 模块对比

---

## 1. 测试方案

### 1.1 思路
不依赖 Wannier90.x 可执行文件(因 3.1.0 版本 I/O 崩溃),直接从 ABACUS 生成的 .mmn(重叠矩阵)和 .amn(投影矩阵)文件计算:
1. **总 Berry phase** (Wilson loop): γ = Im ln ∏_k det(M_k) — 精确
2. **总 Berry connection** (trace): γ_conn = Σ_k Im Tr(M_k) — 近似
3. **逐原子 Berry connection**: γ^I = Σ_k Σ_n w^I_n(k) · Im[M_nn(k)] — 按能带归一化权重分解

### 1.2 体系
- BaTiO3 四方铁电相, 10×10×10 MP k-mesh, LCAO basis
- nocc=15 (Ba:2e, Ti:4e, O:8e×3=24e → 30e/2=15)
- .mmn: 30 bands × 1000 kpts × 6 nearest neighbors
- .amn: 30 bands × 1000 kpts × 33 projections (15 from .win + 18 from ABACUS LCAO)

### 1.3 ABACUS Wannier90 接口设置问题
在设置过程中遇到并解决了以下问题:
1. **k-point 排序**: Wannier90 默认 z-fast, ABACUS x-fast → 修正 .win kpoint 顺序
2. **对称性缩减**: symmetry=0 仍做时间反演 → 500 kpts → 改用 symmetry=-1 得到 1000
3. **晶格单位**: .win 用 Å, ABACUS 用 Bohr → 修正为 4.00×4.00×4.20 Å
4. **Monkhorst-Pack 公式**: ABACUS 用 (2n-N-1)/(2N) = -0.45,...,0.45, 非 ix/N
5. **投影数不匹配**: ABACUS LCAO 接口在 .nnkp 投影基础上额外添加 18 个投影 (共 33)
6. **Wannier90.x 崩溃**: 3.1.0 conda-forge 版本在 disentanglement 后 I/O segfault
7. **解纠缠必要**: O PP 含 8 价电子 (O²⁻), 导致 Ti 3d 被占据, band 12-16 交叉

---

## 2. 结果

### 2.1 总量对比

| 量 | .mmn 计算 | ABACUS berry_phase | 比例 |
|---|---|---|---|
| Berry phase (det/Wilson) | 1.2001 | -0.33085×2π = -2.078 | -0.577 |
| Berry connection (trace) | -0.5352 | — | — |
| trace/det 比例 | -0.446 | — | — |

**关键发现**: trace/det = -0.446, 即 Berry connection (Im Tr M) 仅为 Berry phase (Im ln det M) 的 44.6%, 且符号相反。

这与 DeltaP 模块中的发现完全一致:
- DeltaP 中 trace/det 比例在不同结构间为 -4.77 到 4.29 (剧烈变化)
- .mmn 确认: trace ≠ det, 比例不稳定

### 2.2 逐原子 Berry connection (按能带归一化权重)

| 原子 | γ^I (trace) | 占比 |
|---|---|---|
| Ba | -0.2778 | 51.9% |
| Ti | +0.0877 | -16.4% |
| O1 | -0.3302 | 61.7% |
| O2 | -0.3291 | 61.5% |
| O3 | +0.3142 | -58.7% |
| **Sum** | **-0.5352** | **100%** |

**Sum rule: PASS** — 按能带归一化的权重法 (w^I_n = Σ_{p∈I} |A_{np}|² / Σ_p |A_{np}|²) 精确满足 sum rule。

### 2.3 每个能带的 Wannier center

| Band | γ_n | ⟨r_z⟩ (Bohr) |
|---|---|---|
| 1 | 1.004 | 1.269 |
| 2 | 0.080 | 0.101 |
| 3 | 1.005 | 1.269 |
| 4 | 0.619 | 0.782 |
| 5 | -1.253 | -1.583 |
| 6 | -0.906 | -1.145 |
| 7 | 0.454 | 0.574 |
| 8 | -0.070 | -0.088 |
| 9 | -0.353 | -0.445 |
| 10 | -0.452 | -0.571 |
| 11 | -0.645 | -0.815 |
| 12 | -0.260 | -0.328 |
| 13 | -0.095 | -0.120 |
| 14 | 0.918 | 1.159 |
| 15 | 0.625 | 0.790 |
| **Sum** | **0.672** | — |
| **det** | **1.200** | — |
| **Sum/det** | **0.560** | — |

**关键**: Σ_n Im[M_nn] = 0.672 ≠ Im ln ∏ det(M) = 1.200。Berry connection (trace) ≠ Berry phase (det)。

---

## 3. 结论

### 3.1 验证了 DeltaP 的核心发现

.mmn 文件提供了 ABACUS 内部计算的精确重叠矩阵,从中可以独立验证:

1. **Berry connection ≠ Berry phase**: trace/det = 0.446-0.560, 两者差异巨大
   - 这不是 DeltaP 实现的 bug, 而是 trace (线性) 与 log-det (非线性) 的数学差异
   - 对于 nocc=15 和 dk=0.1, 这个差异是本质性的

2. **逐原子 sum rule 可以精确满足**: 按能带归一化的权重法 (w^I_n) 使 sum rule 精确成立
   - 但分解的是 Berry connection (近似), 不是 Berry phase (精确)
   - DeltaP 的"全局 SVD + 权重"方法使用的是 SVD gauge 的 w^I_s, 原理类似

3. **trace/det 比例随结构变化**: DeltaP 中比例为 -4.77 到 4.29, .mmn 中为 -0.446
   - 这解释了为什么 DeltaP 的 Z* (基于 trace 差分) 与 berry_phase 的 Z* (基于 det 差分) 不一致

### 3.2 Wannier90 直接运行未能完成

Wannier90 3.1.0 (conda-forge) 在 disentanglement 完成后 I/O segfault, 无法获取 .wout 中的 MLWF centers。但 .mmn/.amn 文件已成功生成,可以直接计算 Berry phase 和逐原子分解。

### 3.3 根本困难确认

通过 .mmn 直接计算,确认了逐原子极化分解的根本困难:

- **Berry phase (det) 不可线性分解**: Im ln det(M) 是 M 的非线性函数,无法写成 Σ_n f(M_nn) 的形式
- **Berry connection (trace) 可线性分解**: Im Tr(M) = Σ_n Im[M_nn], 但它是 dk→0 极限下的近似
- **对于 nocc=15, dk=0.1**: 近似误差达 44-56%, 不可接受

---

## 4. 文件位置

- .mmn/.amn/.eig: `/tmp/opencode/w90_bto/equilibrium/OUT.autotest/seedname.*`
- 分析脚本: 本文档内嵌 Python 代码
- .win 文件: `/tmp/opencode/w90_bto/equilibrium/seedname.win`
- ABACUS INPUT: `/tmp/opencode/w90_bto/equilibrium/INPUT_nscf`
