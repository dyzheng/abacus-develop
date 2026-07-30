# P16 PW 约束核整改验证（onsite_radius 扫描 × PW/LCAO 对照）

> 系列：DeltaP ｜ 前置：R6 记录、单位备忘录 ｜ 状态：无阻塞（判定有效）｜ 设计文档：`../P16-PW约束核整改验证.md`

## 1. 目的

R6 定位了 PW 响应赤字机制：约束核 λΣP_SMO 中 SMO 球仅覆盖约 19% 体积。本测试扫描 onsite_radius ∈ {6,10,14,20} Bohr，实测 PW 的 dγ/dλ(r) 响应曲线，验证大半径整改方案使 PW 响应恢复到与 LCAO 一致（≤10%）。

## 2. 目录结构

```
P16-PW约束核整改验证/
  README.md
  run.sh
  cases/
    KPT                Γ 点
    STRU_15BOHR        15 Bohr (7.9365 Å) 盒 H2O（复现 R6 设置）
    STRU_30BOHR        30 Bohr (15.873 Å) 盒 H2O（生产设置）
    INPUT.pw.tmpl      PW 模板（ecut 80, berry_phase 1, deltap_switch true, @RADIUS@ 占位）
    INPUT.lcao.tmpl    LCAO 对照模板（ecut 100, @RADIUS@=6.0 固定）
  runs/                run.sh 生成（box{15,30}/{pw_r{6,10,14,20},lcao}/lam_*）
```

## 3. 用法

```bash
bash run.sh
NPROC=4 bash run.sh
```

共 30 次 SCF：2 盒 ×（4 半径 × 3 λ）PW + 2 盒 × 3 λ LCAO。λ ∈ {−0.01, 0, +0.01} Ry，λ 冻结（lambda_step 0），total 模式。

## 4. 提取与换算

- PW：`[DeltaP-PW]` 行 `γ_total`（wrapped mod [−π,π]）；响应 `dγ/dλ(r) = unwrap(γ(+0.01)−γ(−0.01))/0.02`（脚本对跨分支差值做 ±2π 解绕）。
- LCAO 对照：`[rawG] Σγ_raw`（unwrapped），同样差分。比较用差值不用绝对值（两通道差 2πN 属正常，见 `../DeltaP测试说明.md` §三）。
- **覆盖率线索**：脚本依次尝试 (a) run.log 中 SMO/投影电荷相关行；(b) `OUT.*/deltap_results.dat` 的 `# SMO radius` / `# Total` 行。当前版本 PW 路径**无覆盖率直接输出键**（run.sh 中留有 TODO 注释）；判据只依赖 dγ/dλ 响应收敛行为，覆盖率为辅助观察量。

## 5. 判据表

| 编号 | 判据 | 阈值 | 状态 |
|---|---|---|---|
| J-box15 | 15 Bohr 盒：r=20 处 \|PW dγ/dλ\| vs \|LCAO dγ/dλ\| 相对差 | ≤10% | 有效 |
| J-box30 | 30 Bohr 盒：同上 | ≤10% | 有效 |
| 趋势 | dγ/dλ(r) 随 r 趋稳（r14→r20 变化打印，人工检查） | 参考 | 人工 |

每点检查 `SCF IS NOT CONVERGED`，未收敛点所在通道标 INVALID 并排除出判据（按 FAIL 计）。大半径 SMO 可能破坏正交化数值稳定性——若 r=20 点 INVALID，先检查 run.log。

## 6. 风险与注意

- LCAO 对照组 onsite_radius/deltap_rm 固定 6.0 Bohr（LCAO 基组本身局域，无 PW 式覆盖率赤字；R6 背景见设计文档）。
- 15 Bohr 盒中 r=20 Bohr 的 SMO 球大于盒子——这正是"覆盖率→1"的极限测试，SCF 可能不稳定，关注 INVALID 标记。
- PW 显式 `nbands 8`；负 λ 自动 mixing_beta=0.3。
- 若 r=20 处 PW 响应仍不收敛到 LCAO 值（>10%），按设计文档 §4 立项整改 B（全位置算符核）。
