# MBIS + ALM 电荷/自旋约束：开发进展与测试结果总结

**日期**：2026-10-30
**分支**：`feat/unified-local-moment-allocation`
**范围**：从"allocation 与 ORCA MBIS 对齐"到"MBIS+ALM 独立模块、真实密度验证、
接入 ABACUS 主约束循环"的完整记录。

---

## 0. 摘要

- **动机**：现有 allocation 坐标 `q=m_T+m_c` 在 LiCl 上给出近中性 Li（≈3.08 e），
  与 ORCA MBIS（Li≈2.04 e）差约 1 e；用户判定 MBIS 更正确，要求以 MBIS 权重
  实现**电荷+自旋同时约束**。
- **算法**：MBIS 实空间权重 + 增广拉格朗日（penalty+Lagrange）；给出能量修正、
  约束哈密顿量、力、应力表达式。
- **实现**：新增**独立、仅依赖 STL** 的 C++ 模块 `module_mbis_alm`（MBIS 权重、
  ALM、观测量/势/力、薄 adapter、探针标定），并接入现有 `WeightGrid`/
  `constraint_loop`（`constraint_weight_type mbis`）。
- **验证**：独立测试 **11/11 PASS**；C++ 与 Python 在真实 LiCl/Fe2 cube 上逐位一致；
  LiCl PW 自由（delta=0）运行**收敛**，ETOT 与 Becke 自由态一致。
- **待办**：受约束运行仍慢（MBIS 重建太频）且 ALM 步长未按 MBIS 响应标定。

---

## 1. 关键物理/算法结论

### 1.1 三种"电荷"不可互换（LiCl，LCAO 自由密度）

| 方案 | Li 电子数 | Li 净电荷 | Cl 净电荷 | 分子偶极 |
|---|---:|---:|---:|---:|
| **MBIS** | 2.0447 | **+0.955** | −0.955 | **6.48 D**（含原子偶极） |
| Mulliken | 2.4830 | +0.517 | −0.517 | 5.02 D（点电荷） |
| Allocation（l=0/1/2） | 3.0833（与 l 无关） | −0.083 | +0.083 | — |
| 气相实验 LiCl | — | — | — | ~6.3 D |

- MBIS 给出物理上最合理的离子性并复现分子偶极；allocation `q` 本质是 Becke 空间
  电子数、非净电荷。差异主因是 **pro-density 宽度**（MBIS 把 Li 优化到 ζ=4.60 紧凑）。

### 1.2 Fe2（Γ, LCAO 自由态，同一密度）

| 口径 | 自旋 (μB) |
|---|---:|
| MBIS 空间自旋 | **±2.647**（ζ=1.07789，电荷 16/16） |
| Becke 空间矩 | ±2.285 |
| onsite d 投影 `m_T` | ±2.985 |
| Mulliken | ±3.082 |

盆内 l 投影（MBIS 平均半径 R=2.78）：**d 自旋 3.069**，与 Mulliken d=3.034、
投影 d=2.985 一致（d 主导）；R=6 截断会高估（3.241）。

### 1.3 约束能量/哈密顿量/力/应力

```
E_c = E_KS − Σ_A[ λ_A^c(N_A−N̄_A) + λ_A^s(M_A−M̄_A) ]
ALM: E + μ(Q−t) + (k/2)(Q−t)² ;  v_con,σ = −Σ_A(μ_eff^c ± σ μ_eff^s) w_A
μ_eff = μ + k(Q−t) ;  μ += k(Q−t) ;  k|χ|∈(0,2)，外层收缩率 1/(1+k|χ|)
F_J = Σ_A ∫ (∂_{R_J}w_A)(μ_c ρ + μ_s m) dr
σ_con = (1/Ω)Σ_A[ μ_c ∂N_A/∂ε + μ_s ∂M_A/∂ε ]
```
- `w_A = ρ_A^pro/Σρ^pro`，`ρ_A^pro=N_A ζ_A³/8π e^{−ζ_A r}`；
  KL：`N_A=∫w_Aρ`，`ζ_A=3N_A/∫r w_Aρ`。

---

## 2. 实现结构（解耦优先）

```
source/source_estate/module_mbis_alm/     # 独立、仅 STL
├── mbis_alm.{h,cpp}          # MBIS 权重/观测量/约束势/力 + Alm 控制器
├── mbis_alm_adapter.{h,cpp}  # SolverAdapter + run_alm_mbis + calibrate_alm
├── pw_grid_bridge.h          # PW 网格 -> Grid/atoms（a_i=latvec*lat0, dv=a/n）
└── test/                     # gtest：核心 9 + adapter 2 = 11
tools/
├── mbis_prototype.py         # Python MBIS（cube 后处理，交叉验证）
├── mbis_l_projection.py      # 盆内 l 投影（DMR + .orb）
├── mbis_alm_compare.cpp      # ALM vs Lagrange vs penalty（线性模型）
├── mbis_cube_io.h / mbis_alm_cube.cpp   # 真实 cube 交叉验证
└── mbis_alm_e2e.cpp          # 真实 cube 半真实 e2e + 自动 k 标定
source_estate/module_constraint/
├── weight_grid.{h,cpp}       # + WeightType::MBIS、build_mbis、init_zero、温启动
└── constraint_loop.{h,cpp}   # + constraint_weight_type mbis；observe() 重建权重
```

---

## 3. 测试结果

### 3.1 独立 gtest（11/11 PASS）

| 测试 | 内容 | 结果 |
|---|---|---|
| PartitionOfUnity | Σ_A w_A=1 (<1e-12) | ✅ |
| ChargeSumRule | ΣN=N、ΣM=M_tot | ✅ |
| MbisMeanRadiusFixedPoint | ζ=3N/⟨r⟩ | ✅ |
| ConstraintPotential | V=Σcoef·w | ✅ |
| ForceFiniteDifference | 解析力 vs FD | ✅ |
| AlmConvergesExactlyOneStep | k=1/χ → 1 步 | ✅ |
| PenaltyOnlyHasBias | 有限 k 偏差 | ✅ |
| SymmetrizeRestoresEquivalence | bcc 对称化（回归 bug） | ✅ |
| GridBridgeMatchesLattice | LiCl/Fe2 网格约定 | ✅ |
| Adapter E2E ×2 | 电荷+自旋端到端 ALM | ✅ |

### 3.2 C++ ↔ Python 真实 cube 一致

| 体系 | 量 | C++ | Python |
|---|---|---|---|
| LiCl | Li N/ζ | 2.044663 / 4.59730 | 2.04466 / 4.59730 |
| | Cl N/ζ | 7.955367 / 1.61196 | 7.95537 / 1.61196 |
| Fe2 Γ | Fe N | 16.00/16.00 | 16.00/16.00 |
| | ζ / 自旋 | 1.07789 / ±2.647 | 1.078 / ±2.647 |

（交叉验证抓到并修复了一个"对称化原地覆盖"bug，已加回归测试。）

### 3.3 算法对比（响应模型）

- **线性模型**（χ={10,2,0.5,0.01}）：Lagrange(全局) **23016 步**；
  ALM k=1/χ **34 步**、k=10/χ **10 步**；penalty 偏差 0.50/0.091。
- **半真实（真实权重重叠）**：
  - LiCl（χ: Li −0.389、Cl −6.21，**异质**）：ALM **25 步** vs Lagrange **295 步**；
    penalty 偏差 0.0995。
  - Fe2（χ≈−20.5，**同质**）：Lagrange **12 步** vs ALM 21 步；penalty 偏差 0.089。
- **结论**：ALM 的收益来自**按模态 k**，只在响应异质时显著；penalty 永远有偏。

### 3.4 接入 ABACUS 主约束循环（LiCl PW, serial）

- 网格约定 `a_i=latvec_i·lat0`、`dv_i=a_i/n_i` 已用真实 cube 精确核对。
- **delta=0（no-op）**：SCF **CONVERGED**，ETOT **−643.9756525**
  （Becke 自由态 −643.9756523）；`CONSTRAINT_AUDIT q=t=2.0323, res=0`。
  - 说明：delta 模式 target 取自**初始密度**首观测（2.0323）；收敛自由密度 MBIS=2.0447
    已由 cube 组件验证，差 0.012 为 delta 语义，非 bug。
- **绝对 target=2.0**：900 s 内未收敛（末次 q=2.032, μ=0.05, res=0.032）。

---

## 4. 当前状态与限制

**已完成**
- MBIS 权重、ALM、观测量、力、adapter、探针标定的独立实现与单元测试。
- 真实密度组件验证（C++==Python）与轨道分解对比。
- `WeightType::MBIS` 接入 `WeightGrid`/`constraint_loop`；自由运行收敛。

**限制**
1. **性能**：MBIS 每次 `observe` 重建成本高（192³ KL）；已加温启动，仍需节流。
2. **ALM 标定**：`constraint_loop` 的 `mu` 步长未按 MBIS 响应 `chi` 标定
   （末次 μ=0.05 → 收敛慢）。
3. **MPI**：`build_mbis` 串行守卫；并行需 gather 或 MPI-aware KL。
4. **力/应力**：`build_derivatives` 未对 MBIS 实现（模块有 `constraint_force`）。
5. **小/常规胞**：bcc 需 `symmetrize` 输入开关。
6. **口径**：MBIS 是空间 AIM，spin-MBIS ≠ DeltaSpin 投影矩；数值不可混用。

---

## 5. 下一步（按优先级）

1. **MBIS 重建节流**：按外层 μ 步 / 每 K 次 observe 重建（热启动已使后续 KL 变快）。
2. **ALM 响应标定**：把 `calibrate_alm` 的探针接进 `ConstraintLoop` 的 ALM arm，
   使 `k≈1/|χ|`、`kmin<1/|χ|`（前面 lesson）。
3. **MPI gather** 或 MPI-aware `compute_mbis`。
4. **力/应力核**（MBIS 应变导数）+ FD 测试。
5. bcc `symmetrize` 输入开关；Fe2 约束运行回归。

---

## 6. 复现命令

```bash
# 独立模块测试（无 ABACUS 依赖）
g++ -std=c++14 -O2 -I source \
  source/source_estate/module_mbis_alm/mbis_alm.cpp \
  source/source_estate/module_mbis_alm/test/mbis_alm_test.cpp \
  -lgtest -lgtest_main -pthread -o /tmp/mbis_alm_test && /tmp/mbis_alm_test

# 真实 cube 交叉验证 / 半真实 e2e
g++ -std=c++14 -O2 -I source tools/mbis_alm_cube.cpp \
  source/source_estate/module_mbis_alm/mbis_alm.cpp -o /tmp/mbis_alm_cube
/tmp/mbis_alm_cube OUT.*/chg.cube
/tmp/mbis_alm_e2e OUT.*/chgs1.cube OUT.*/chgs2.cube --images 2 --symmetrize

# 接入主循环（LiCl PW, serial）
# INPUT: constraint true / constraint_weight_type mbis / constraint_target_mode delta
mpirun -np 1 build_rel/abacus_basic_para
```
