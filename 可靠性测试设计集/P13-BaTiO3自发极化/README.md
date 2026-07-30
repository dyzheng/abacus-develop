# P13　BaTiO₃ 自发极化（可执行算例）

> 系列：DeltaP ｜ 前置：P09（分支确定性，见 §6 标注）｜ 设计文档：`../P13-BaTiO3自发极化.md`

## 1. 目的

铁电自发极化是 Berry 相位方法的标志性应用，涉及分支选择、极化量子、软模位移三重难点。
验证 DeltaP 在 BaTiO₃ 四方相上给出与 PW berry_phase 一致的 Ps 与能量面，
并展示约束扫描（固定 D 系综）绘制铁电双势阱 E(γ) 的能力。

## 2. 目录结构

```
P13-BaTiO3自发极化/
  README.md
  run.sh
  cases/
    STRU.tmpl               # 4.00×4.00×4.20 Å 四方晶胞（复用 tests/deltap_bto_sampling
                            # 的晶胞与 Ba/O 原子位），Ti 的 z 坐标为占位符 Z_TI
    KPT                     # 6×6×6
    INPUT_lcao.tmpl         # LCAO λ=0 测量模式
    INPUT_lcao_target.tmpl  # LCAO 约束模式（lambda_step 0.01 + target.dat）
    INPUT_pw.tmpl           # PW berry_phase + deltap_switch true
  runs/                     # 运行产物（含 results.txt）
```

## 3. 用法

```bash
bash run.sh                  # 或 NPROC=4 bash run.sh
```

共 17 个 ABACUS 算例：路径 9 + PW 2 + 约束 6。

## 4. 工作流

**① Ti 位移路径**：z_Ti = 0.48→0.56 步长 0.01 共 9 点（0.52 为平衡附近），
每点生成 STRU 变体，LCAO λ=0 读 `[rawG] Σγ_raw` 与 `E_KohnSham`。
分支连续性：相邻点 |Δγ| > 1.0 rad 判为假跳变 FAIL。

**② Ps 三通道**（参考点 z_Ti=0.50，铁电点 z_Ti=0.52）：
- 通道① DeltaP raw γ：Ps = P_CONV·Δγ，P=(e/Ω)·(c/π)·γ（常量区）；
- 通道② PW berry_phase gdir=3：γ_total 差值用**通道① Δγ 作预测值展开 2π 整数倍**
  （小步长预测-校正，防跨量子误展开），同一 P_CONV 换算；
- 通道③ wannier90：`command -v wannier90` 检测，无则 SKIP；有则生成 .win 模板
  （num_wann/窗口为占位值，首次实跑按实际能带确认后才启用数值判定）。

**③ 约束能量面**：对 z_Ti=0.50/0.52/0.54 三点做 target 模式约束
（`deltap_lambda_step 0.01`，target = 该点平衡 Σγ_raw ± 0.2 rad 两档），
连同 9 个平衡点输出 E(γ) 双势阱采样点表（按 γ 排序，写入 results.txt）。

## 5. 判据表

| 编号 | 判据 | 状态 |
|---|---|---|
| J1 | 全部算例 SCF 收敛且有 [rawG]/E_KohnSham 输出 | 有效（硬判） |
| J2 | 路径无假跳变：相邻 |Δγ| ≤ 1.0 rad | 有效（硬判） |
| J3 | Ps 两通道（DeltaP vs PW）互差 ≤10% | 有效（硬判） |
| J4 | E(γ) 光滑：每组三点相邻能量差无符号振荡（符号变化 ≤1） | 有效（硬判） |
| — | 约束达到精度（\|γ−target\| 残差） | INFO（打印不判） |
| — | 与实验 Ps≈0.26–0.27 C/m² 的偏差 | INFO（标注泛函误差，不判） |

末尾 `SUMMARY: n/n PASS`，退出码 0/1；结果存 `runs/results.txt`。

## 6. 前置与阻塞标注

- **前置 P09**：本测试的分支连续性判据（J2）以 P09 的"无缓存确定性"结论为前提——
  若同一结构重跑 γ 不可复现，J2 的跳变判定无法区分物理跳变与数值噪声。
  本测试本身不被 F1/F2 阻塞（Ps 用 Δγ 差值，换算常量 P_CONV 在头部单点修改）。
- **小步长预测-校正**：PW 的 γ_total 是 wrapped 量（mod 2π），直接用两点差
  可能跨极化量子；脚本用 LCAO unwrapped Δγ 作预测值对 PW 差值做 2π 整数倍校正，
  路径步长 0.01 保证单步 Δγ 远小于量子，该校正可靠。
- **w90 通道**：检测不到 wannier90 时 SKIP 不计入 SUMMARY；检测到时也仅生成模板，
  因 num_wann/能量窗口依赖实际能带结构，需首次实跑人工确认后启用数值判定。

## 7. 风险与注意（抄自设计文档 §7）

- Ps 对泛函/体积敏感，判据以"同泛函三通道互洽"为主（J3 只比 DeltaP 与 PW 两通道）；
- 跨极化量子时必须小步长预测-校正（见 §6）。

补充（工程层面）：

- 参考点 z_Ti=0.50 并非严格中心对称结构（Ba 在 z=0.9937），Ps 为两点**相对**极化；
  与实验绝对值对比需注意此约定（INFO 已标注）；
- 约束目标半宽 0.2 rad 对应 ΔP≈0.06 C/m² 量级，若 SCF 难收敛可在常量区调 `TARGET_HALF`；
- P_CONV 的自旋因子约定（任务书公式 P=(e/Ω)(c/π)γ）若与 PW 通道定义差因子 2，
  J3 会系统性失败——这是 F2 备忘录需要钉死的点，届时单点改 `P_CONV`。
