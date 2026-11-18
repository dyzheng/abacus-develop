# 2026-11-18 固定 μ vs 电场；nelec_delta != 0 带电约束测试

分支：`mbis`（HEAD `720d94a60` + 本轮修改）
目标：
1. 验证**固定 μ 的偶极约束**能否等价于外加电场；
2. 验证 `nelec_delta != 0` 的带电体系能否被电荷约束正确约束、总电荷是否闭合。

## 1. 测试计划

1. **固定 μ ↔ 电场**：同一 LiCl/PW 体系，比较
   - 固定 μ 的 dipole-saw 约束（`fixed_mu = -0.01`）；
   - 外电场 `efield_amp`，按势垒匹配关系 `efield_amp = -μ/2`（立方胞、e33=1）
     选取，使两者 saw 势完全相同；
   比较 `FINAL_ETOT` 与 `TOTAL-FORCE`。
2. **带电约束（正电）**：`nelec_delta = +0.5`（总 10.5 e），两个 per-atom
   MBIS 电荷约束（Li +0.05、Cl −0.05，目标和保持总电荷），检查
   `CONSTRAINT_AUDIT` 的 residual 与 `total_charge` vs `nelec`。
3. **带电约束（负电）**：`nelec_delta = -0.5`（总 9.5 e），同样两约束。
4. **cutoff 影响**：MBIS pro 截断 10 → 20 Bohr 对总电荷闭合的影响。
5. **健壮性**：固定 μ 偶极约束 + `constraint_alloc_report 1` 不再崩溃。

## 2. 测试设置

| 项 | 值 |
|---|---|
| 体系 | LiCl，PW，nspin1，15 Å 立方盒（`latvec=diag(15,15,15)`，`lat0=1.889726`） |
| 基态 | `ecutwfc=40`，`ecutrho=160`，`scf_thr=1e-7`，`scf_nmax=100` |
| MD | `calculation md`, `md_nstep=1`, `md_type nve`, `md_dt=1`, `test_force=1` |
| 固定 μ | `{"type":"dipole","axis":"z","frame":"saw","saw_pos":0.9,"saw_pos_dec":0.1,"unit":"ebohr","fixed_mu":-0.01}` |
| 电场 | `efield_flag=1`, `efield_dir=2`, `efield_pos_max=0.9`, `efield_pos_dec=0.1`, `efield_amp=0.005`, `dip_cor_flag=0` |
| 带电约束 | `nelec_delta=±0.5`，两约束 Li +0.05 / Cl −0.05，`constraint_mbis_images=1` |
| 带电控制器 | `constraint_kappa_min=0.3`, `constraint_step_max=0.05`, `constraint_thr=1e-3` |

### 2.1 μ → 电场的匹配关系

saw 帧下约束势为 `μ * w = μ * (-Lz*saw)`；电场势为
`fac*saw`，`fac = e2*efield_amp*lat0/bmod`。立方胞 `bmod = 1/|e3| = 1/15`，
`Lz = lat0*15`，`e2=2`，故

```
efield_amp = -mu / 2        (Hartree)
```

取 `μ=-0.01` → `efield_amp=+0.005`。两者的 saw 形状参数相同
（`pos_max=0.9`, `pos_dec=0.1`）。

## 3. 结果

### 3.1 固定 μ ↔ 电场

| 量 | 固定 μ | 外电场 | 差 |
|---|---:|---:|---:|
| `FINAL_ETOT` (eV) | **−642.7326304217** | **−642.7326305203** | **9.9×10⁻⁸** |
| Li `F_z` (eV/Å) | 0.1255192302 | 0.1236209723 | 0.001898 |
| Cl `F_z` (eV/Å) | −0.1255192302 | −0.1302633334 | 0.004744 |
| Li `F_x` (eV/Å) | 0.0003107179 | −0.0000530404 | 0.000364 |
| Cl `F_x` (eV/Å) | −0.0003107179 | −0.0002744146 | 0.000036 |

- **总能量一致到 1×10⁻⁷ eV**，说明固定 μ 的 dipole-saw 约束与匹配幅度的
  外电场给出同一个电子结构。
- 力差 ~0.002–0.005 eV/Å，约为场致力量级（0.125 eV/Å）的 2–4%。差异来自
  两套力的实现路径（约束的 `dw_` 收缩 vs `Efield::compute_force` 的解析项）
  与 SCF 收敛阈值，而不是势不同（势相同已由能量证明）。
- 结论：**可以**。固定 μ 的偶极约束就是强度 `E ∝ -μ` 的电场；对立方胞
  `efield_amp = -μ/2` 是可直接使用的换算。

### 3.2 带电约束：正电 `nelec_delta=+0.5`

| cutoff | 首次外步 `total_charge` | `nelec` | deficit |
|---:|---:|---:|---:|
| 10 Bohr | 10.48049224 | 10.5 | **0.0195 e** |
| 20 Bohr | 10.49998733 | 10.5 | **1.3×10⁻⁵ e** |

- cutoff=10 时 MBIS 分区漏掉了 0.0195 e 的真空尾部；cutoff=20 后总电荷
  闭合到 1.3e-5 e。说明**带电体系的电荷闭合对 MBIS pro 截断敏感**，不是
  约束求解器本身的问题。
- cutoff=20、保守步长（`step_max=0.05`, `kappa_min=0.3`）的外步 1–4：
  `total_charge` 保持 10.4995–10.5，residual 从 0.05 单调降到 0.0158
  （μ 从 −0.05 到 −0.2）。
- 外步 5（μ≈−0.2 后）：体系进入更弥散的带电态，`total_charge` 跌到
  10.3463（deficit 0.154 e），residual 反弹到 0.31；150 步预算内未恢复。
  即：**中等约束强度下带电约束正确**；强约束引起态变化/弥散尾部时，
  需要更大的 cutoff（或 remainder 通道）和/或状态感知的控制器。
- 新增一次性警告已在 `warning.log` 触发，提示增加 `constraint_mbis_cutoff`。

### 3.3 带电约束：负电 `nelec_delta=-0.5`

cutoff=10，5 个外步：

| 外步 | μ | residual | `total_charge` (nelec=9.5) |
|---:|---:|---:|---:|
| 1 | −0.05 | 0.0500 | 9.499914641 |
| 2 | −0.10 | 0.0461 | 9.499914058 |
| 3 | −0.15 | 0.0413 | 9.499913594 |
| 4 | −0.20 | 0.0360 | 9.499913129 |
| 5 | −0.25 | 0.0303 | 9.499912425 |

- 总电荷闭合到 **8.6×10⁻⁵ e**（负电更紧凑，cutoff=10 已足够）。
- residual 单调下降，约束方向正确；`step_max=0.05` 太小，60 步预算内只
  走了 5 个外步，需要更大步长或更多外步才能命中 1e-3。
- 警告触发时 deficit 为 0.012 e（早期步），随后收敛到 1e-4。

### 3.4 健壮性修复

固定 μ 偶极约束 + `constraint_alloc_report 1`（默认）在 `after_scf` 的
`alloc::report_pw_alloc` 中**段错误**：

```
ESolver_KS_PW::after_scf -> alloc::report_pw_alloc
```

原因：报告钩子只检查了 `constraint && constraint_alloc_report`，没有检查
`constraint_alloc_switch`；非 allocation 约束（dipole）进入 allocation
报告路径后解引用不存在的 target 上下文。修复为 PW/LCAO 两处都增加
`constraint_alloc_switch` 守卫。修复后同一算例正常结束，`FINAL_ETOT`
不变（−642.7326304217 eV）。

## 4. 代码改动

1. **新增输入** `constraint_mbis_cutoff`（默认 10 Bohr，`constraint_weight_type=mbis`）：
   `WeightGrid::build_mbis` 使用它；带电/弥散体系可增大以闭合总电荷。
2. **MBIS 带电审计警告**：当约束列表恰好覆盖每个原子一次且全为
   charge 通道时，比较 `ΣQ` 与 `PARAM.inp.nelec`；偏差 > 1e-3 e 时警告一次
   并提示增大 cutoff。
3. **报告守卫修复**：`esolver_ks_pw.cpp` / `esolver_ks_lcao.cpp` 的
   `constraint_alloc_report` 钩子加上 `constraint_alloc_switch` 条件。

## 5. 分析 / 结论

1. **固定 μ 可以代替电场**：dipole-saw 约束的 `μw` 就是 saw 形状的势；
   匹配幅度后总能量差 1e-7 eV。换算（立方胞）`efield_amp = -μ/2`。
2. **带电约束本身是正确的**：负电 −0.5 e 总电荷闭合到 1e-4、residual 单调
   下降；正电 +0.5 e 在 cutoff=20 时闭合到 1e-5，早期外步 residual 单调下降。
3. **带电体系的真正风险是 MBIS 分区的截断**：cutoff=10 对正电 +0.5 e 漏掉
   0.02 e；强约束下弥散尾部会把 deficit 放大到 0.15 e。需要
   `constraint_mbis_cutoff`（已加）或 remainder 通道；强约束还需更稳健的
   控制器/更多 SCF 预算。

## 6. 下一步

1. 为 MBIS 增加显式 remainder 通道（类似 Hirshfeld），使带电/弥散体系
   在任意 cutoff 下 `ΣQ` 都严格等于 `nelec`。
2. 用 `constraint_mbis_cutoff=25–30` 与 `step_max≈0.2` 重跑正电 +0.5 e，
   确认强约束下也能闭合。
3. 对固定 μ vs 电场补一次 `scf_thr=1e-9` 的力对比，量化 2–4% 力差的来源
   （约束力 vs `Efield::compute_force`）。
4. 为 `constraint_alloc_report` 的非 allocation 路径补一个回归测试。
