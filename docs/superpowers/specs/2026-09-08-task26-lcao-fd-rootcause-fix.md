# Task 2.6.2 LCAO 力 FD——μw Pulay 生命周期缺失修复 + 可证伪验证通过

> 日期：2026-09-08
> 承接：评审定案（`2026-09-08-task26-lcao-fd-rootcause-review.md`，归档）——
> 根因=力求值时刻 μw 不在 v_eff 里（μw Pulay 缺失）；批复执行路径=定点修复
> （复用 `add_back_constraint_potential`）→ 轻量 O-z 复验，预言 `net_z≈0` 且
> `|d|≪0.0128555`。
> 状态：**修复已实现、构建完成、可证伪验证两预言均通过**（|d|=0.0004）。
> 本文件为"修改+测试"轮次的 dated spec（AGENTS.md）。

---

## 1. 测试方案（本轮 = 修复 + 可证伪验证）

1. 代码级核实根因（评审推断的机制）：μw 在 SCF 内由
   `ESolver_KS_LCAO::hamilt2rho_single` 于 HSolver 前注入 v_eff；收敛那一轮
   `ElecState::cal_converged() → Potential::get_vnew()` 内部调
   `update_from_charge()` **用输出密度重建 v_eff（μw 被抹掉）**，随后 SCF 循环
   在下次注入前退出——力评估时 `cal_pulay_fs`（fvl_dphi）读到的 v_eff 已无 μw。
   （与 PW SCC 缺陷同族：势在生命周期末端丢 μw；PW 丢在 vnew 快照、已修；
   LCAO 丢在 v_eff 重建。）
2. 定点修复：仅围绕 LCAO `fvl_dphi`（`cal_pulay_fs`）评估加回 μw——
   新增 RAII 守卫 `ConstraintPulayPotGuard`（FORCE_STRESS.cpp，匿名命名空间）：
   约束开启且 μ≠0 时先快照 v_eff，`add_back_constraint_potential(v_eff)` 回加
   μw，作用域退出恢复物理势。forcecon 仍照旧提供显式 ∂w/∂R 一半；两半合起
   平移不变。
3. 可证伪验证（轻量，固定-μ 协议 np4，3 腿 ~2 min/腿）：
   修复后重跑 `/tmp/cfd_lcao_fixedmu/{oz_minus2,oz_0,oz_plus2}`；
   预言：(a) R0 全轴净力≈0；(b) O-z `F_ana(R0)≈F_FD`（|d|≪0.0128555）。
4. 判据纪律：O-z 判据沿用 stationary4 的 0.0128555 eV/Å，不豁免。

## 2. 测试设置

- 体系/网格/约束同前：H₂O 15 Å 盒、LCAO 高网格（ecutwfc=100/ecutrho=400/
  scf_thr=1e-8）、charge-Becke-O、绝对 t*=6.505559879、δ=0.005 Bohr、
  γ-only np4、自 base 密度 restart。
- 固定-μ 环境开关：`ABA_CONSTRAINT_FIXED_MU=-0.2193905904`（冻结常数外势、
  无外环，每腿仅数次电子步）。
- 二进制：`/root/abacus-develop/build/abacus_basic_para`（2026-09-08 17:52
  重建，含本轮 FORCE_STRESS.cpp 修复；能量路径零改动）。

## 3. 结果

代码改动：`source/source_lcao/FORCE_STRESS.cpp`
（+`ConstraintPulayPotGuard`；两处 `cal_pulay_fs` 调用点包裹——nspin 1/2 与
nspin 4 分支）。构建：仅 FORCE_STRESS.cpp 重编，make 退出 0。

修复后固定-μ 三腿（能量与修复前逐位一致：−466.2137876462331 /
−466.2070054151704 / −466.1998835520661 eV → F_FD=−2.627493 eV/Å）：

| 腿 | O-z (eV/Å) | H1-z | H2-z | net_z |
|---|---|---|---|---|
| R−δ | −2.5056545 | +1.2528273 | +1.2528273 | 0.0000000 |
| R0 | **−2.6271286** | +1.3135643 | +1.3135643 | 0.0000000 |
| R+δ | −2.7486436 | +1.3743218 | +1.3743218 | ~0 |

- **预言 (a) 通过**：全轴净力≈0（x/y/z 三轴 ≤1e-8，平移不变性恢复；
  修复前净_z = +7.078 eV/Å）。
- **预言 (b) 通过**：`|F_FD − F_ana(R0)| = |−2.627493 − (−2.6271286)| =
  **0.000364 eV/Å** ≪ 判据 0.0128555`（富余 35×）。
- 腿自洽：R± 解析 O-z −2.5057/−2.7486 跨 R0 −2.6271（均值 −2.6271），
  与能量曲率方向一致。
- 修复量：R0 O-z −1.3602401 → −2.6271286（Δ=−1.26689 ≈ 缺失的 μw Pulay）。

## 4. 分析

- 修复前 net_z=7.078 eV/Å = Σ forcecon（7.081，0.04% 吻合）→ 分析力只有 μw
  显式一半；修复后 net≈0 → 基函数导数一半（经 v_eff 回加 μw 的 fvl_dphi）
  归位，两半闭合。与评审定案完全一致。
- x/y 轴在 R0 同步恢复平移不变（H1-x/H2-x 各改 ~±2.20 eV/Å、O-x≈0，
  与 Becke 胞 x 方向不对称性对应）；O-z 是唯一有 FD 腿的轴，判据在此裁决。
- 未做（评审未要求、且属重型）：R7 重优化-μ 协议 18 腿全量重跑、
  残余∝|Q−t| 档位检查。fixed-μ FD(−2.6275) 与 pinned FD(−2.629) 修复前
  已证同值，修复只动力路径 → 预期 pinned 全量同判 PASS。

## 5. 下一步

1. （待批准，中量）R7 pinned-μ 协议 LCAO 18 腿全量重跑（修复后二进制）：
   目标 9 轴全 PASS；约 3× 现有预算（base 8 min + 18 腿）。
2. 2.6.2 收尾：残余误差∝|Q−t| 档位检查（外环容差 1e-5 单腿重跑）。
3. 队列后续：2.6.3 力矩 FD → 2.6.4 → 2.6.1 → 2.7 判决门 + 文档收尾
   （含评审纪律项：净力/补偿前力入 FD 归因标准检查；F_ana 补偿前后双值入档）。
4. 单测复核（改动仅 LCAO 力路径，单测不受影响；闭合前跑一遍）。

## 6. 文件与数据

- spec：本文件 + `2026-09-08-task26-lcao-force-fd-attribution.md`（归因轮）+
  `2026-09-08-task26-lcao-fd-rootcause-review.md`（评审定案，归档）。
- 代码：`source/source_lcao/FORCE_STRESS.cpp`（ConstraintPulayPotGuard）。
- 数据：/tmp/cfd_lcao_fixedmu/{oz_minus2,oz_0,oz_plus2}/OUT.s（修复后）；
  修复前数值见归因 spec §3。
- 日志：`/tmp/cfd_lcao_fix_verify.log`、`/tmp/build_lcao_fix.log`。
