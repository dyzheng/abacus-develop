# 2026-11-19 LiF 拉伸扫描：r2SCAN SCF 收敛性与 MBIS 电荷

## 1. 测试计划

1. 在 18 Å 大盒中放一个 LiF 分子，沿 z 方向逐步拉伸 Li–F 距离；
2. 每个距离用 **r2SCAN**（`MGGA_X_R2SCAN+MGGA_C_R2SCAN`）做 PW SCF，
   检查是否收敛；
3. 用 MBIS 电荷约束（两原子各一个 charge 约束、delta target 0、μ=0）
   读取收敛密度上的 Li/F MBIS 电荷；
4. 观察拉伸过程中 SCF 收敛性、总能量与 MBIS 电荷（离子性）变化。

## 2. 测试设置

- 盒子：18×18×18 Å，Li 固定在 (9,9,9)，F 在 (9,9,9+r) Å；
- 赝势：Li_APNS.upf（Li 3e）、F_ONCV_PBE-1.0.upf（F 7e），10 价电子；
- `nspin=1`, `gamma_only=1`, `nbands=6`, `ecutwfc/ecutrho=40/160`；
- `scf_thr=1e-6`, `scf_nmax=120`, `mixing_type=broyden`, `mixing_beta=0.3`,
  `smearing_method=gaussian`, `smearing_sigma=0.002`；
- `device=cpu`, OMP=20, np=1；
- 编译：`build_fix_cuda` 打开 `ENABLE_LIBXC=ON`，Libxc 5.1.7；
  `dft_functional MGGA_X_R2SCAN+MGGA_C_R2SCAN`；
- MBIS：`constraint_type=charge`, `constraint_weight_type=mbis`,
  `constraint_mbis_images=1`, `constraint_mbis_cutoff=10`,
  `constraint_mbis_defer_reference=1`, `constraint_branch_tol=0`；
  target file 两个 charge 约束（Li、F，`target=0`，delta 模式），
  即读取 reference 布居，不施加偏置。

## 3. 结果

### 3.1 SCF 收敛性与时间

| r (Å) | SCF | wall time (OMP=20) | FINAL_ETOT (eV) |
|---:|---|---:|---:|
| 1.20 | ✅ CONVERGED | 434 s | −841.4489306600 |
| 1.40 | ✅ CONVERGED | 356 s | −843.3282454438 |
| 1.56 | ✅ CONVERGED | 343 s | −843.6744872609 |
| 1.80 | ✅ CONVERGED | 379 s | −843.4241979066 |
| 2.20 | ✅ CONVERGED | 381 s | −842.4900404244 |
| 3.00 | ✅ CONVERGED | 443 s | −840.8471181745 |
| 4.50 | ❌ **NOT CONVERGED**（120 步） | 2196 s | −839.2809615531（不可信） |
| 6.00 | ❌ **NOT CONVERGED**（120 步） | 1820 s | −835.0004467939（不可信） |

- 1.2–3.0 Å 在 6–7.4 min 内收敛；SCF 在 1.56 Å 附近能量最低。
- 4.5 Å 和 6.0 Å 在 `scf_nmax=120` 内不收敛；能量只是最后一步的输出，
  **不能作为结果**。
- 4.5 Å 的失败诊断（warm start 尝试）：用 r=3.0 的
  `CHARGE-DENSITY.restart` 作初值、`mixing_beta=0.2`、
  `smearing_sigma=0.005`、`scf_nmax=150`，跑到 118 步仍未收敛
  （Drho≈0.133，`E_gap≈0.086 eV`，接近零 gap/真空态），1 h 后手动终止。
  说明不是单纯初值问题；拉伸后的近简并/电荷转移使 SCF 很难收敛。

### 3.2 MBIS 电荷（仅收敛点）

| r (Å) | Q_Li (e) | Q_F (e) | ΣQ | Li 净正电荷 (3−Q) | F 净负电荷 (7−Q) |
|---:|---:|---:|---:|---:|---:|
| 1.20 | 2.046955 | 7.952633 | 9.999588 | +0.9530 | −0.9526 |
| 1.40 | 2.038596 | 7.960998 | 9.999594 | +0.9614 | −0.9610 |
| 1.56 | 2.037847 | 7.961746 | 9.999594 | +0.9622 | −0.9617 |
| 1.80 | 2.038266 | 7.961334 | 9.999599 | +0.9617 | −0.9613 |
| 2.20 | 2.037780 | 7.961815 | 9.999595 | +0.9622 | −0.9618 |
| 3.00 | 2.038811 | 7.960758 | 9.999568 | +0.9612 | −0.9608 |

- 总 MBIS 电荷 `ΣQ≈9.9996`，相对 `nelec=10` 的 deficit 约 **4×10⁻⁴ e**
  （来自 10 Bohr pro cutoff；低于 1e-3 警告阈值，未触发警告）。
- Li/F 的净电荷在 1.4–3.0 Å 内几乎不变：**Li^+0.96 F^−0.96**，
  变化 <0.01 e；只有最短的 1.2 Å 离子性略低（0.953）。
- 因此，在 SCF 可收敛的拉伸区间内，**MBIS 电荷分布对键长不敏感**，
  分子基本保持接近满离子性；没有看到明显的共价化/电荷转移随拉伸的连续变化。
- 4.5/6.0 Å 没有 MBIS 电荷，因为 reference SCF 未收敛，MBIS 的
  reference deferral 不会构建权重；无法给出可信的 Q。

## 4. 分析

1. **不是所有拉伸距离都能收敛**：当前设置下 1.2–3.0 Å 收敛，
   4.5/6.0 Å 失败。失败与 r2SCAN 无直接关系；更可能是大盒子中拉伸 LiF 的
   近简并/真空态 + 电荷转移导致 SCF 电荷 sloshing，120 步不够。
2. **warm start 不足以解决**：从 r=3.0 的收敛密度出发，4.5 Å 仍在 118 步
   后 Drho≈0.133、gap≈0.086 eV；需要更强的混合/预条件（Kerker、
   更小 mixing_beta、更多步）、不同 smearing/occupation，或更小的盒子/
   不同基组设置。
3. **MBIS 电荷在可收敛区间内几乎不变**：LiF 在 1.4–3.0 Å 基本是
   Li^+0.96 F^−0.96，说明 MBIS 电荷对拉伸不敏感；这与 LiF 的强离子性
   一致。1.2 Å 压缩时离子性略降。
4. **总电荷 deficit**：4×10⁻⁴ e 来自 MBIS pro cutoff；若要更严格的
   电荷闭合，增大 `constraint_mbis_cutoff` 或加 remainder 通道。
5. **大盒子 + r2SCAN 的代价**：每个收敛点 ~6–7 min（OMP=20）；失败点
   因为跑满 120 步，耗时 30–36 min。SCF 是主要成本，MBIS 只构建一次
   （reference deferral）。

## 5. 下一步

1. 对 4.5/6.0 Å 尝试更强的 SCF 稳定化：
   - Kerker/更小 `mixing_beta`（0.05–0.1）、更大 `smearing_sigma`
     （0.01–0.02）、`mixing_type pulay/local_tf`；
   - 更小盒子（如 14–15 Å）或不同 `ks_solver`/`nbands`；
   - 从更近的 r 逐步 warm start（3.0→3.75→4.5），而不是直接 3.0→4.5。
2. 若目标是研究解离极限，考虑显式 fragment charge 约束（Li^+/F^−）
   或有限场/Berry-phase 场方案。
3. 若要更干净的 MBIS 电荷，增大 `constraint_mbis_cutoff`（如 15–20）
   并记录 `ΣQ−nelec`。
4. 需要的话，把 1.2–3.0 Å 的收敛点整理成 `examples/` 下的 LiF 拉伸
   示例（含 r2SCAN + MBIS 电荷读取）。
