# 2026-11-18 MBIS reference 只构建一次 + tol/iter 可调

## 1. 测试计划

1. 在 reference 相（`mu=0`）不再每个 SCF 迭代重建 MBIS，只在 SCF 收敛帧
   构建一次；验证 `Q_ref`、`FINAL_ETOT` 与旧的“每步重建”一致。
2. 新增 `constraint_mbis_tol` / `constraint_mbis_max_iter`，验证放宽后结果在
   约束阈值内不变。
3. 复测 `examples/35_constraint` 的带电算例耗时。
4. 全量 constraint/MBIS 单测回归。

## 2. 实现

### 2.1 reference deferral

- `ConstraintLoop::observe(..., bool scf_converged=false)` 新增收敛标志；
  `esolver_ks_pw`/`esolver_ks_lcao` 传入 `conv_esolver`。
- `wg_->is_mbis() && phase_==REFERENCE` 时：
  - 若 `constraint_mbis_defer_reference=true`（默认）且本帧未收敛：
    跳过 MBIS 构建/观察，`Q_` 置零、`mu_eff_=mu_`（reference 相 `mu=0`）；
  - SCF 收敛帧才构建一次权重并计算 `Q_ref`；
  - constrained 相不变：首次构建后冻结。
- 新增输入 `constraint_mbis_defer_reference`（默认 `True`），可设 `False`
  恢复旧行为。

### 2.2 可调固定点参数

- `constraint_mbis_tol`（默认 `1e-8`）：Picard 布居收敛阈值。
- `constraint_mbis_max_iter`（默认 `200`）：固定点迭代上限。
- `WeightGrid::build_mbis` 使用这两个输入（带下限保护）。

## 3. 结果

### 3.1 reference deferral（LiCl 15 Å，OMP=1，np=1）

设置：PW，`ecut 40/160`，8 bands，`scf_nmax 40`，`scf_thr 1e-7`，
MBIS 电荷约束（target delta 0，`images=1`，cutoff=10）。

| 模式 | wall | `Q_ref` | `FINAL_ETOT` |
|---|---:|---:|---:|
| 旧行为：每个 reference 迭代重建（`defer=0`） | **2m39.4s** | 2.033754427 | −643.0900167502319 eV |
| 新默认：只收敛帧构建（`defer=1`） | **57.9s** | 2.03375443 | −643.0900167502319 eV |
| `defer=1` + `tol=1e-6`, `max_iter=50` | **56.1s** | 2.0337546 | −643.0900167502319 eV |

- 加速 **2.75×**（16 个 reference SCF 迭代从 16 次 MBIS 构建降到 1 次）。
- `Q_ref` 差 < 1e-7 e，`FINAL_ETOT` 完全一致（到 1e-13 eV）。
- `tol=1e-6` 仅再省 1.8 s；该小体系 MBIS 单次构建已只需数秒。

### 3.2 examples 复测（OMP=1，np=1）

`03_charged_anion_cutoff`：

| cutoff | wall（本轮） | 之前（np=2×OMP=10） | `total_charge` |
|---|---:|---:|---:|
| 10 | **1m22.8s** | 7m10s | 7.998424731（warning） |
| 20 | **1m23.6s** | 6m40s | 7.999999829（no warning） |

`02_charged_ionic`（LiCl⁺，CPU OMP=1，`scf_nmax=80`）：

- wall **5m37s**；reference 只构建一次 MBIS；外步 residual 到
  **1.39e-3**，略高于 `constraint_thr=1e-3`，在 `scf_nmax=80` 下
  budget freeze。增大 `scf_nmax`（100+）即可完成最后一次 settle。
- 剩余时间主要是外步各自的完整 SCF（每步约 3–4 s/迭代 × 10 步），
  不是 MBIS 重复构建。


### 3.3 磁矩（spin）约束复测

体系：bcc Fe2（0.50 体心），`nspin=2`，gamma-only，`ecutwfc=40`，
`ecutrho=160`，24 bands，OMP=1，np=1；两个 spin 约束（target delta 0，
即取 reference 磁矩），`constraint_mbis_images=1`。

| 模式 | wall | `Q_ref` (atom 0) | `FINAL_ETOT` |
|---|---:|---:|---:|
| `defer=0`（每步重建） | **21.9s** | −3.980092681 μB | −6753.407264934619 eV |
| `defer=1`（收敛帧一次） | **7.4s** | −3.980092681 μB | −6753.407264934619 eV |

- 该 reference SCF 用了 35 步，deferral 把 35 次 MBIS 构建降到 1 次，加速
  **2.96×**；`Q_ref` 与总能量完全一致。
- 低 cutoff 的小 Fe2（ecut 20/80，9 步 SCF）中 MBIS 单次构建很便宜，
  加速只有 2.0s→2.4s（~18%）；**加速比随 MBIS 单次成本和 reference
  SCF 步数增长**。
- spin 约束与 charge 约束走同一 `is_mbis() && phase_==REFERENCE` 分支，
  因此“是否重建”与约束类型无关；constrained 相仍是首次构建后冻结。

### 3.4 回归

- `MODULE_ESTATE_constraint_loop` 58/58 PASS（CPU 与 CUDA 构建）。
- `MODULE_ESTATE_constraint_deriv`、`MODULE_ESTATE_mbis_alm_gpu` PASS。
- 示例结果与旧数据一致（`Q_ref`/`total_charge`/`FINAL_ETOT`）。

## 4. 分析

- reference 相的 MBIS 权重只用于定义 `Q_ref`，中间迭代的 `Q` 不参与目标；
  因此“收敛帧构建一次”不改变物理定义。旧实现的每步重建是**纯冗余**。
- 大体系/多原子时，单次 MBIS 构建更贵，deferral 的收益更大
  （从 O(SCF 步) 降到 O(1)）。
- `tol=1e-6` 对最终 `Q` 影响 <1e-6，约束阈值 1e-3 下完全够用；默认仍保守
  保留 `1e-8`，用户可自行放宽。
- 小网格上 `OMP=1` 往往比 `np=2×OMP=10` 快很多（避免线程/MPI 过载），
  与 MBIS 无关。

## 5. 风险 / 选项

- 若用户依赖 reference 相中间迭代的 `Q` 用于监控，设
  `constraint_mbis_defer_reference false` 恢复旧行为。
- 收敛帧才构建意味着第一次 MBIS 构建没有中间 warm start；单次成本可能
  略高于后续 warm build，但总次数从 N 降到 1，净收益显著。
- `constraint_mbis_tol` 不建议大于 1e-5（可能影响 `Q_ref` 的末位）。
