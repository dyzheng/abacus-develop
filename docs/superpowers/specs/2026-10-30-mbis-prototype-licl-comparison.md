# 2026-10-30 MBIS 原型实现与 LiCl（LCAO）三方案对比

## 0. 结论先行

在 **LCAO** 的 LiCl 自由密度（同一电子态）上，三种"原子电荷"给出发散的结果：

| 方案 | Li 电子数 N | Li 净电荷 | Cl 净电荷 | 分子偶极 |
|---|---:|---:|---:|---:|
| **MBIS**（本次实现） | **2.0447** | **+0.955** | −0.955 | **6.48 D**（含原子偶极）× |
| **Mulliken**（ABACUS） | 2.4830 | +0.517 | −0.517 | 5.02 D（仅点电荷） |
| **Allocation**（LCAO，l=0/1/2） | 3.0833 | **−0.083** | +0.083 | 不适用（q 非电荷） |
| 气相实验 LiCl | — | — | — | ~6.3 D |

- 三者对 Li 的净电荷相差近 **1 e**；**不可互换**。
- **MBIS 给出物理上最合理的离子性（Li⁺≈+0.96）**，且**原子偶极**使其分子偶极
  （6.48 D）接近实验（~6.3 D）；Mulliken 明显低估离子性（+0.52）。
- **Allocation 的 `q`≈3.08 e/Li 本质是 Becke 空间电子数**（≈中性、甚至微阴离子），
  它是"目标壳层+互补"的建模坐标，**不是净电荷**（与语义闭环文档一致）。

## 1. 实现（只读原型）

- 脚本：`tools/mbis_prototype.py`（Python/numpy，读 ABACUS `chg.cube`）。
- 输入密度：`tests/constraint_s5_parity/licl_alloc` 改成 **LCAO**
  （`basis_type lcao`、`ks_solver genelpa`、`gamma_only 1`、`nspin 1`、
  `out_chg 1 6`、`out_mul 1`，charge 约束 delta=0 空操作）。
  15 Å 盒，LiCl 相距 ~2.02 Å；192³ 网格，间距 0.1476 Bohr，密度 Bohr⁻³。
- pro-density：Slater s，`ρ_A^pro = N_A ζ_A³/(8π) e^{−ζ_A r}`；
- 迭代：`w_A=ρ_A^pro/Σρ_B^pro`、`ρ_A=w_Aρ`，按 MBIS 的
  **布居+均值半径** 更新 `N_A=∫ρ_A`、`ζ_A=3N_A/∫rρ_A`（阻尼 0.5），
  收敛 `|ΔN|<1e-9`（~49 步；ΣN=10.00003，与 cube 的 10.00003 一致）。
- 输出：`q_A=Z_A−N_A`、`μ_A=−∫(r−R_A)ρ_A`、`μ_mol=Σ(q_A R_A+μ_A)`。

复现：
```bash
# 在 LCAO LiCl 运行目录
python3 tools/mbis_prototype.py OUT.licl_alloc/chg.cube
```

## 2. 结果

### 2.1 电荷（电子数 / 净电荷）

- MBIS：Li `N=2.04466`（+0.95534）、Cl `N=7.95537`（−0.95537）；
  `ζ_Li=4.5973`（很紧凑）、`ζ_Cl=1.6120`（较弥散）。
- Mulliken（`mulliken.txt`）：Li `2.4830`（+0.5170）、Cl `7.5170`（−0.5170）。
- Allocation（LCAO，`constraint_alloc_report`）：
  - **`n` 与 `target_l` 无关**：l=0/1/2 均为 Li `3.083315696`、Cl `6.916684304`；
  - 只有拆分变：l=0 `n_T=2.41498,n_c=0.66833`；l=1 `0.80325,2.28007`；
    l=2 `0.0,3.08332`；`dsum_n=9.6e-14`。

### 2.2 分子偶极（MBIS 的价值所在）

- MBIS 原子偶极：Li `0.015 D`、Cl `2.807 D`（以各自核为原点）；
- 重建 `μ_mol=Σ(q_A R_A+μ_A)=6.479 D`；
- 仅用 MBIS 点电荷 `Σq_A R_A` / Mulliken 点电荷 → ~5.0 D；
- 气相 LiCl 实验偶极 ~6.3 D ⇒ **MBIS（含原子偶极）明显更接近**。

## 3. 分析：差异从哪来

1. **pro-density 宽度是主因**。MBIS 把 Li 优化到 **紧凑**（ζ=4.60），把大部分
   密度留给 Cl → Li⁺；Becke/allocation 用**固定半径**（Li 半径较大）→ Li 分到
   3.08 e（近中性/微阴）。这正是 §"对齐 ORCA"计划要改的核心（固定 Gaussian →
   迭代 Slater）。
2. **三种量回答不同问题**：
   - Mulliken：轨道布居（基组依赖）；
   - Allocation `q`：目标壳层 + Becke 互补（建模坐标，求和精确，非净电荷）；
   - MBIS：迭代 stockholder 的**空间 AIM**（含多极）。
   同一电子态下数值差 ~1 e 是**语义差**，不是数值 bug。
3. **Allocation 的 `n` 对 target_l 不变**：说明其总量 = 完整空间权重
   `Tr[ρ W_i]`（Becke 电子数），target 只重分配 `n_T/n_c`。
   这与修复前 PW 报告的 l 依赖（2.948/3.085/3.028）不一致——**PW 侧应
   在复杂 k 修复后重查是否为旧归一化 bug 的残留**。

## 4. 注意事项

- 本实现是 **Python 只读原型**（非 C++ Phase A）：均匀 cube 网格积分、
  MBIS 偶极对网格敏感（建议更细网格或径向求积）；未与 ORCA 直接对照
  （本机无 ORCA），以分子偶极对实验作为间接校验。
- PBC：中性盒内偶极可定义；带电/强极性周期体系应改用 Berry 相位/sawtooth。
- **不要**把 allocation `q` 当净电荷；也**不要**把 MBIS spin 当 DeltaSpin 投影矩
  （MBIS 是空间 AIM，与投影族口径不同）。
- 未改变任何约束/SCF 路径，纯后处理。

## 5. 下一步

1. 把原型移植为 C++ 只读观察量（`WeightType::MBIS` + Slater + KL + 多极），
   复用现有周期镜像/C2 窗口，并与 ORCA 在 H2O/LiCl（分子）上逐项对照。
2. 用修复后代码重查 PW allocation 的 `target_l` 依赖（预期也应不变）。
3. 视需要再评估 Phase B（MBIS 作为约束，冻结分区 + 外循环）。
