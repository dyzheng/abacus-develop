# 2026-08-03 DeltaP B-6 修复轮：H_HR τ 单位 → 分数坐标

> 触发：用户评审（"B-6 → 锚点 → A2/C → D-D"顺序），B-6 已被相位链证明为
> 量纲 bug（非偏好选择）。本轮：实现 B-6 修复包（代码 + E-field 重推导 +
> 锚点 A/B 验证），为锚点重建铺路。

## 1. 测试计划

1. 实现 B-6：全部 λτ 项（H_HR 的 SCF 侧 / 力侧 / hhrdbg）τ_α 从
   `atoms[].tau`（lat0 单位）改为 `atoms[].taud`（Direct 分数坐标）。
2. 验证 A/B（同一二进制，修复前 vs 修复后，1-rank）：
   - h2o base（ecutwfc=50，KPT 1×1×2，gdir=3，cal_force=1，L=15.8753）：
     对比 λ、γ、escon、E_H_HR（hhrdbg）、E_HK、FINAL_ETOT_IS。
   - BN center（ecutwfc=100，KPT 2×2×2，L=3.615）：对比 λ、γ、escon、E'。
   - 预期（算符不变性）：E_H_HR 精确缩小 L 倍；E' 位移 = TrρH_HR 位移；
     λ/escon/γ 的行为取决于 λ 更新协议（同步模式单步冻结 → 不重收敛）。
3. 重推导 E-field 等效公式 E=−πλ/(2a) 在分数 τ 下的适用性，评估
   bn_sampling 标定的连带影响。

## 2. 测试设置

- 二进制：`build/abacus_basic_para`（B-6 前后各构建一次，OMP 1 线程，
  `mpirun -np 1`）。
- h2o base：`/tmp/b6_smoke/h2o_base_{pre,post}`（复制自
  `/tmp/fd_t7b/b7/base`，suffix 分别 h2o_b6pre/h2o_b6post；scf_thr 1e-7，
  deltap_lambda_step 0.01，target=0,0,0）。
- BN center：`/tmp/b6_smoke/center100_{pre,post}`（复制自
  `tests/deltap_bn_sampling/center`，ecutwfc=100 与提交版一致；
  target=4.00,3.50）。
- 注：BN 在 ecutwfc=50 下 γ 分支振荡（|γ−t| 不稳），不收敛；ecutwfc=100
  仅 24–44 s，故锚点直接用提交版设置，不用 50 冒烟。
- 环境：伪势/轨道 `/root/pporb/apns-*`（与既有测试一致）。

## 3. 结果

### 3.1 代码修改（B-6）

| 位置 | 修改 |
|---|---|
| `deltap_force_stress.hpp:63` | `atoms[T0].tau[I0][alpha_idx]` → `taud`（力侧 λτ） |
| `deltap_force_stress.hpp:226` | 同上（hhrdbg E_H_HR） |
| `deltap_lcao.cpp:125` | 同上（SCF 侧增量 H_HR 应用，coeff=dλ·τ） |
| 三处均加 B-6 注释 | τ 必须 Direct 分数坐标；勿混入 lat0 单位 H_HR τ |

PW 路径（`op_pw_proj.cpp`）无 τ 因子（设计差异 §2.2-c），不受影响。

### 3.2 h2o base A/B（L = a/lat0 = 30/1.889726 = 15.8753）

| 量 | 修复前 | 修复后 | 比值/位移 |
|---|---|---|---|
| λ (Ry) | (−5.514e-3, −3.602e-3, −3.602e-3) | 相同 | **不变** |
| γ (rad) | (−5.509, −3.605, −3.605) | (−5.521, −3.599, −3.599) | ~0.2% 漂移 |
| escon (Ry) | −0.056344 | −0.056368 | 不变（0.04% 漂移） |
| E_H_HR (Ry) | −0.4470868692 | −0.0281215151 | **缩小 15.90×**（≈L，0.15% 吻合） |
| E_HK (Ry) | −8.6306372e-2 | −8.6468643e-2 | +0.19%（ρ 微变） |
| E' = FINAL_ETOT_IS (Ry) | −35.985465 | −35.567286 | +0.418179 |

**E' 位移闭环**：ΔE' = +0.418179 vs ΔE_H_HR = +0.418965（0.4470869 −
0.0281215），闭合 99.8%——E' 位移完全由 H_HR 算符缩小 L 倍解释
（escon/E0/E_HK 基本不动）。

### 3.3 BN center A/B（L = 3.615）

| 量 | 修复前 | 修复后 |
|---|---|---|
| λ (Ry) | (2.5817e-3, −2.3242e-3) | 相同（**不变**） |
| γ (rad, iter 50) | (3.453, 4.030) | (6.572, 1.180)（分支翻转振荡） |
| escon (Ry) | +0.000453 | −0.014223（γ 分支态不同，非稳态量） |
| E' (Ry) | −24.9156 | −24.9129（+2.73e-3） |

λ 不变；E' 位移 ≈ TrρH_HR 缩小量（λ_N·τ_z(N)/L·P̂_N 量级）。γ 在
(4.000,3.500) 目标附近分支振荡（修复前 iter48/后 iter49 命中目标一次），
非收敛态，不作定量判据。

### 3.4 内循环参考锚点（B-6 后，D2 首点）

`tests/deltap_bn_sampling/test_stru_target`（ecutwfc=100，KPT 2×2×2，
inner_nmax=3，4-rank，`/tmp/b6_smoke/tst_inner/run.log`）：
inner loop done λ=(5.0000e-03, 4.3749e-03)，γ 收敛至目标
γ=(3.999,3.497)（|γ−t|~1e-3），escon=−0.0353 Ry，无跨 rank 发散。
**×L 重收敛仍需 pre/post A/B 实测**（D2 缺口，同步模式已证 λ 不重收敛，
内循环 BFGS 预期相反）。

## 4. 分析

### 4.1 机制：同步模式 λ 单步冻结 → λ 不重收敛（关键发现）

`deltap_scf.cpp update_lambda_gd`：两阶段协议在 drho < inner_thr 时
**只做一次** GD 步（`state_.lambda_set = true` 后冻结），λ = λ₀ +
mixing·step·r（r 为 λ=0 态残差）。因此：

- λ 只依赖 λ=0 的 γ（与 H_HR 无关，B-6 前后相同）→ **λ 不变**；
- H_HR = λ·τ_frac·P̂ 在冻结 λ 下**精确缩小 L 倍**（E_H_HR 实测 15.90×，
  0.15% 吻合）——即同步模式下约束驱动强度降为 1/L；
- escon = −λγ 不变（λ、γ 均不变）；E' 位移 = TrρH_HR 位移（99.8% 闭合）。

**内循环 BFGS 模式（inner_nmax>0）预期不同**：λ 被收敛到 γ→target，
γ 响应 dγ/dλ ∝ τ 缩小 L 倍 → 收敛 λ 预期重收敛 ×L，算符 λτP̂ 不变
（物理等价）。**未验证——正是 D2 缺口（内循环零 MPI 覆盖）的实测内容**，
留本轮后补 4-rank inner_nmax>0 冒烟。

### 4.2 E-field 等效公式重推导（B-6 连带项）

- 规格（dev-guide §2.1）本就规定 τ_α 为分数坐标；B-6 使代码与规格一致。
- 物理等效：H_HR = λ·τ_frac·P̂ 与电场焓 −E·r̂（r̂ = τ_frac·a·P̂，a 为
  Bohr 晶胞长）对偶 → λ ↔ E·a。任务书 F1 公式 **E = −πλ/(2a)**（λ 为
  Ry、a 为 Bohr）**只在分数 τ 下成立**；旧代码（lat0 τ）同一 λ 对应
  L 倍强的等效场（h2o 下 2L/π ≈ 10.1× 于 F1 标称，含 F1 未定的 π/2
  因子）。
- **结论**：F1 公式与 λ↔E 工作值（λ=±0.02 ↔ E=∓0.001047 Ha/Bohr，
  a=30 Bohr）**不变**，B-6 后代码与公式一致；π/2 因子之争仍属 F1
  备忘录待定项。bn_sampling 的"每单位 λ 的约束驱动"现为 1/L——
  若用户要恢复旧驱动强度，`deltap_lambda_step` 需 ×L（记入 F1 备忘录）。

### 4.3 锚点影响

- **test_C_I（λ≡0）不受影响**（H_HR=0·τ·P̂≡0）——SCF 锚点无需重建。
- λ≠0 用例（bn_sampling 9-label、deltap_bn_test、relax）需重建：λ 不变、
  escon 基本不变、E' 位移 ≈ TrρH_HR 位移（BN center 实测 +2.7e-3 Ry；
  h2o +0.418 Ry）、γ 轨迹可能分支翻转。
- FD 预算（§10.8 已同步回改）：冻结 λ 下 A2 均匀 = **+1.85e-3**、
  C 均匀 = **+3.70e-3** Ry/Bohr（B-6 前为 +2.93e-2 / +3.70e-3）。

## 5. 下一步

1. 重建剩余锚点：bn_sampling 9-label results.csv、deltap_bn_test、
   relax 用例（ecutwfc=100，每个 ≤1 min；test_C_I 跳过——λ≡0 不受影响）。
2. 补 D2 缺口：4-rank inner_nmax>0 冒烟，实测内循环 λ 是否 ×L 重收敛
   （兼验 B-6 在内循环模式的算符不变性）。
3. A2/C 成对实现 + 成对验证（冻结 λ FD，按 4.3 的新预算）。
4. D-D 高精度轮（ecutwfc=100 + ecutrho≥400 + scf_thr 1e-8）最终验收。
5. 清理 hkdbg/hkchk/fsdbg/hhrdbg；MPI/ASAN 回归。

## 6. 复现

- 修复前二进制基线：`/tmp/b6_smoke/h2o_base_pre/run_pre.log`、
  `center100_pre/run_pre.log`；修复后：`h2o_base_post/run_post.log`、
  `center100_post/run_post.log`。
