# DeltaP T7-c Round: A2 实现验证 + 组②激活约束 FD 判决（2026-08-03）

## 0. 本轮目标（用户评审"修正后的顺序"）

1. **A2 实现**（∂τ/∂R Hellmann–Feynman 项，逐原子对角力）——已实现，本轮定量验证。
2. **组② 判决性 FD**：target=γ*(base)（消除分支交换噪声），A1+A2+B 残差应 ≈ 0——
   relax 可用性的判决，成本远低于 C。
3. C 暂停（dspin 定理的 DeltaP 版：约束激活时 λ·∂γ/∂R + 响应项恒等相消）。

## 1. 测试计划

| 项 | 内容 |
|---|---|
| A2 公式闭合 | h2o1 base（λ*=(−5.514e-3,−3.602e-3,−3.602e-3) Ry）A2 前后 TOTAL-FORCE 差 == −λ⟨P̂⟩(L⁻¹)zz/lat0（含 FORCE_STRESS 均值扣除） |
| 组② FD 矩阵 | h2o1 全 3 原子 × 3 方向 × ±δ(0.005 Bohr)，target=γ*(base)，deltap_lambda_step 0.01（λ 每几何重导出），E'=FINAL_ETOT_IS（含 dp_escon） |
| 纯 DFT 对照 | 同一几何无 deltap：FD vs 解析力（隔离 deltap 协议污染与标准 LCAO 力问题） |
| 组① 参考 | O1-z 冻结 λ*（init_file + step 0.0）：E(λ) 曲面 FD，残差应为 C + 响应 |
| 内循环 | deltap_inner_nmax>0（BFGS λ）——唯一可逼近"λ 收敛于约束驻点"的协议，检查可用性 |

**判据**：|F_FD − F_ana| < 5e-4 Ry/Bohr = 0.01286 eV/Å（dev-guide §4）。

## 2. 测试设置

- 系统：`tests/deltap_fd_force/h2o1`（O1(7.9365,7.9365,7.9365)、H1(7.1794,7.9365,8.5223)、H2(8.6936,7.9365,8.5223) Å，15.873×15.873×15.873 lat0，gdir=3）。
- 生产设置（D-D 要求）：ecutwfc=100、ecutrho=400（显式）、scf_thr=1e-8、OMP_NUM_THREADS=4。
- 二进制：`build/abacus_basic_para`（commit 1b2625fdd + A2 未提交改动 + hhrdbg 调试打印 #if 1）。
- E'=FINAL_ETOT_IS（`fp_energy.cpp:19` 已含 dp_escon）；位移 δ=0.005 Bohr=0.0026459 Å（Cartesian_angstrom 直接加）。
- λ 机制（同步两阶段）：drho<inner_thr 时单步 GD `λ = λ + mixing·step·(γ−t)`（mixing=0.1, step=0.01）后冻结。
- 位移 STRU 已校验（O1-z+δ → z=7.9391458861 Å）。

## 3. 结果

### 3.1 A2 验证（公式闭合）

A2 代码（`deltap_force_stress.hpp`）：`force(iat,β) −= λ_iat·⟨P̂_iat⟩·(L⁻¹)_{αβ}/lat0`，reduce_all 前加入。
h2o1 base（λ*≠0）A2 前后对比（OMP_NUM_THREADS=1 保证逐位可比）：

| 量 | 基线(无A2) | 含A2 | 差 |
|---|---|---|---|
| E' (eV) | −483.917570760897 | −483.917570760897 | 0（SCF 逐位不变） |
| O1 z (eV/Å) | −0.4562574183 | −0.4379843881 | +0.0182730302 |
| H1 z | +0.2281287518 | +0.2189922368 | −0.0091365150 |
| H2 z | +0.2281286665 | +0.2189921513 | −0.0091365152 |

x/y 分量逐位一致（A2 只作用于 gdir=z）。公式预测（λ、⟨P̂⟩ 用 6 位打印值，含均值扣除 sum/nat）：

```
raw_i = −λ_i·⟨P̂_i⟩·(L⁻¹)_zz/lat0  (Ry/Bohr) × 25.711 → eV/Å
O1: +0.0340176,  H1/H2: +0.0066081  → 均值扣除后 +0.0182730 / −0.0091365
观测差: +0.0182730 / −0.0091365     → 偏差 < 2e-8 eV/Å (4 ppm)
```

**A2 判定：PASS**（hhrdbg 打印 ⟨P̂⟩=(7.1975, 2.1403, 2.1403)；⟨P̂⟩ 语义 = SMO 值块×DM 对角收缩，与 E_H_HR=λτ⟨P̂⟩ 自洽）。

### 3.2 组② FD 矩阵（target=γ*(base)）

γ*(base)（当前二进制、target=0 测得）：O1=−5.5205672, H1=−3.5991442, H2=−3.5991442。
target=γ*(base) 的 base 运行：γ=(−5.520,−3.599,−3.599)，|γ−t|=7.0e-4，**λ=(6.9e-7, 2.2e-7, 2.2e-7) ≈ 0**，escon≈5e-6 Ry。

| iat axis | F_FD (eV/Å) | F_ana (eV/Å) | \|残差\| | 判据 0.0129 |
|---|---|---|---|---|
| O1 x | −9e-6 | −4.2e-5 | 3.3e-5 | PASS |
| O1 y | −1e-5 | −4.2e-5 | 3.2e-5 | PASS |
| O1 z | −0.6276 | −0.7473 | **0.1197** | FAIL |
| H1 x | +0.0035 | −0.3167 | **0.3203** | FAIL |
| H1 y | −2e-8 | +2.1e-5 | 2.1e-5 | PASS |
| H1 z | +0.3921 | +0.3736 | **0.0184** | FAIL |
| H2 x | −0.0035 | +0.3168 | **0.3203** | FAIL |
| H2 y | −0.0082 | +2.1e-5 | 0.0082 | PASS |
| H2 z | +0.3921 | +0.3736 | **0.0184** | FAIL |

所有位移点 λ ≈ 1e-7–1e-5（约束近似激活，无 2π 分支翻转——目标处方有效消除了分支噪声）。
残差 0.018–0.32 eV/Å，全部 FAIL 分量在 z（gdir）与 H1/H2 x（沿 O–H 键投影方向）。

### 3.3 纯 DFT 对照（隔离标准力）

同一几何、无 deltap（deltap_switch 关）：

| 方向 | F_FD (eV/Å) | F_ana (eV/Å) | 残差 |
|---|---|---|---|
| H1 x | −0.31701 | −0.31671 | 3.0e-4 |
| O1 z | −0.74756 | −0.74726 | 3.0e-4 |

**标准 LCAO 力与自身 E 曲面完全一致**——基础机制无误，残差全部来自 deltap 相关能量项。

### 3.4 组① 参考（冻结 λ*，O1-z）

λ*=(−5.514e-3,−3.602e-3,−3.602e-3) 冻结（init_file+step 0.0，两位移点 λ 逐位一致）：

```
E'(+) = −483.924110, E'(−) = −483.923063, E'(0) = −483.923767
F_FD(O1z) = +0.198 eV/Å
F_ana_raw(O1z) = 打印值 −0.4322 + ΣF/3 = −0.4322 + 0.0157 = −0.4165 eV/Å
残差 = +0.615 eV/Å
```

（ΣF/3 来自 A2：Σλ⟨P̂⟩=−0.0551 → ΣF_A2=+1.84e-3 Ry/Bohr，唯一非零和项；A1/B/f_hk 和均为 0。）

### 3.5 内循环（deltap_inner_nmax>0）可用性检查

`tests/deltap_bn_sampling/test_stru_target`（B/N dp_target 4.0/3.5，deltap_inner_nmax=3）：
- 锚点（4 MPI rank × 3 线程）曾得 γ=(4.001,3.500)、λ=(−5e-3,−4.39e-3)、|γ−t|=1.2e-3。
- 1 rank × 4 线程重跑 base：γ=(3.968,3.768)、|γ−t|=0.27，E'=−339.157 vs 锚点 −338.310——**并行配置改变结果**。
- SCF 轨迹：drho 在 6e-4–1e-3 极限环振荡（GE11 达 6.9e-6 后 λ 更新再次扰动密度），scf_nmax=30 未收敛。
- N 位移 ±0.0026 Å 触发分支翻转（γ=(−1.88,−2.05)，λ 符号反转）——内循环协议对位移点不稳定。

## 4. 分析

### 4.1 组② 残差完整归因：λ 响应项（不是 A1/A2/B 错误，也不是干净 O5 定量）

组② 位移点的 E' = E_plain + ΔE_deltap，其中 ΔE_deltap = E_HK + E_H_HR + escon ∝ λ(R)，
λ(R) = 0.001·(γ(R)−t) 随几何重导出。H1-x 三点分解（deltap−plain 能量偏移）：

| 几何 | escon (Ry) | E_HK (Ry) | E_H_HR (Ry) | Σ (eV) | 实测 E'偏移 |
|---|---|---|---|---|---|
| base | +5e-6 | +8.0e-6 | +3.0e-6 | +2.2e-4 | +2.1e-4 |
| H1 x +δ | −2.0e-5 | −2.88e-5 | −1.33e-5 | −8.4e-4 | −8.1e-4 |
| H1 x −δ | +2.5e-5 | +3.75e-5 | +7.4e-6 | +9.5e-4 | +8.9e-4 |

`−d(ΔE_deltap)/dx = +0.32 eV/Å` = 实测组②残差 (0.3203)。**残差 = O·dλ/dR**：
λ 在 ±δ 间变化 ~1.5e-5（单步 GD 重导出），E' 对 λ 极度敏感（dE'/dλ ≈ 405 eV/Ry，
含 ψ 重收敛响应；target=0 与 target=γ* 两个 base 差 2.23 eV / 5.5e-3 Ry）。

**为什么 dspin 定理不适用**：定理前提是 λ 处于约束驻点（γ≡t，dγ/dR=0，∂E'/∂λ=0）。
同步两阶段协议只做单步弱 GD（λ=0.001(γ−t)），|γ−t| 不为 0（3e-4–6e-3 rad），
λ(R) 随几何漂移 → O·dλ/dR 项不消。残差是"代理算符差距 × 协议 dλ/dR"，
**不是干净的 O5 定量**（其值依赖协议，dλ/dR 是单步 GD 的人为产物）。

### 4.2 组① 残差 = C + 响应（理论预期一致）

组①（λ 冻结）残差 +0.615 eV/Å = λ·dγ/dR（C 项）+ λ·(δγ/δψ)(dψ/dR)（响应）。
O1 自身 C 项估算：λ_O·∂γ_O/∂z = −5.5e-3×(−0.83/Bohr) = +0.12 eV/Å；
其余为跨原子 C + 响应。与用户"组① 还差响应项，C 的解析显式部分实现了组①也不严格闭合"一致。

### 4.3 判决结论

1. **A2 实现正确**（公式闭合 4 ppm，SCF 不变）。
2. **组②（同步两阶段 + target=γ*）不能作为 relax 可用性判决**：λ 未收敛于约束驻点，
   残差被 O·dλ/dR 主导（0.02–0.32 eV/Å），既不能证明也不能证伪 A1+A2+B。
3. **判决所需协议 = 内循环（deltap_inner_nmax>0，BFGS λ 收敛）**，但当前内循环 SCF 不稳定
   （bn 极限环 + 位移点分支翻转 + 并行配置依赖）——是下一个必须修的阻塞。
4. 纯 DFT 对照证明标准 LCAO 力与 E 曲面一致（3e-4 eV/Å）——残差无标准力成分。

## 5. Next steps

1. **修内循环 SCF 稳定性**（bn 极限环）：λ 更新与密度混合解耦（如 λ 更新后 reset 混合器 /
   阻尼 λ 步长 / inner_thr 收紧），使 deltap_inner_nmax>0 在 ±δ 位移点可复现收敛。
2. 内循环组② 判决 FD（bn 或 h2o1 + 非平凡 target 使 λ≠0）：残差应 ≈ 0（定理），
   若显著则按用户决策树定为 O5 定量 → 决定补 C 或回到算符形式。
3. 组① 残差（=C+响应）在修复内循环后重测，作为 C 项预算。
4. hhrdbg/#if 0 恢复，调试打印全部 flag 包裹后提交。
5. D-D 高精度轮（ecutwfc=100 已用）作为最终验收设置保留。

## 6. 数据位置

- `/tmp/b6_smoke/fd2/meas`（target=0 base，λ* 源）；`/tmp/b6_smoke/fd2/g2/*`（组② 19 点）；
- `/tmp/b6_smoke/fd2/plain/*`（纯 DFT 对照 5 点）；`/tmp/b6_smoke/fd2/g1/*`（组① 2 点）；
- `/tmp/b6_smoke/fd2/bn/*`（内循环可用性检查）。

## 7. 回归面（本轮提交前）

- **MPI 冒烟**（`tests/deltap_mpi_smoke/run.sh`）：PW 2-rank、LCAO 4-rank、
  test_stru_target 4-rank 内循环——**3/3 PASS**（A2 走既有 reduce_all，多 rank 安全）。
- **单测**：MODULE_LCAO_deltap_math（3/3）、deltap_gauge（4/4）、
  MODULE_ESOLVER_deltap_common（10/10）PASS。
- **MODULE_LCAO_deltap_smoothness：4/8 FAIL（预先存在，非本轮引入）**——
  在提交点 1b2625fdd（无 A2）复现同样 4 个失败：WilsonLoopIsSmoothUnderPerturbation /
  CompareSmoothnessAllMethods / WilsonLoopGaugeInvariant / BerryConnectionSmoothWithoutAnchorJump。
  根因指向 B-6 轮（commit 53f94042d）S_dk 相位 τ→分数坐标约定改变，该单测用合成数据
  按旧约定断言 Wilson 环平滑/线性/规范不变性——**测试参考约定需随新相位约定更新**
  （P1 TODO；不影响生产路径，锚点/FD 均按新约定验证通过）。

## 8. 提交内容

- `deltap_force_stress.hpp`：A2 实现（p_hat 转正 + 逐原子对角力）；hhrdbg 恢复 #if 0。
- `2026-08-03-deltap-force-stress-t7c-a2-group2-fd.md`（本文档）；
- `deltap-development-log.md`（续7）。
- 排除：`STRU.cif`（杂散）；`tests/deltap_fd_force/h2o1/target.dat` 已还原为 (0,0,0)
  （组② 用 γ*(base) 由运行器临时写入，见 §2）。
