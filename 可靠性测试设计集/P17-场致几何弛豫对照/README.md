# P17 场致几何弛豫对照（约束 vs 外场）

> 系列：DeltaP ｜ 前置：P01、P08 ｜ **状态：阻塞（判定待 F1/F2 备忘录定稿）** ｜ 设计文档：`../P17-场致几何弛豫对照.md`

## 1. 目的

固定 λ 下的 Hellmann–Feynman 力应与等效外场下的力一致（极化梯度力）。对比两路径弛豫后的键长/键角增量，为"固定 D 系综弛豫"提供合法性验证。

## 2. 目录结构

```
P17-场致几何弛豫对照/
  README.md
  run.sh
  cases/
    KPT                       Γ 点
    STRU                      30 Bohr 盒 H2O（O 居盒心，C2 沿 z）
    INPUT.relax.deltap.tmpl   relax + DeltaP 固定 λ 模板
    INPUT.relax.efield.tmpl   relax + efield 固定 E 模板（dip_cor=1）
  runs/                       run.sh 生成（lam_*/ef_* 各 ±0.02、±0.04）
```

## 3. 用法

```bash
bash run.sh
NPROC=4 bash run.sh
```

8 次 relax：`calculation relax`、`relax_nmax 50`、`force_thr_ev 1.0e-3`（eV/Å）、`cal_force 1`、scf_thr 1.0e-8。λ=±0.02、±0.04 Ry；efield 侧 E=−πλ/(2a)（F1 工作值，a=30 Bohr，run.sh 头部常量区）。

## 4. 提取方法（几何）

弛豫末帧取自 `OUT.*/STRU_ION_D`——ABACUS relax 每个离子步覆写该文件（`relax_driver.cpp` 中 `print_stru_file`，Direct 坐标），末帧即最终几何。run.sh 内 awk 解析 LATTICE_CONSTANT/LATTICE_VECTORS 与 Direct 坐标，换算到 Å 后计算：

- `r_OH` = 两条 O–H 键长均值，`Δr = r_OH − r_OH(初始)`；
- `∠HOH` 由两个 O–H 矢量点积求得，`Δ∠ = ∠ − ∠(初始)`；
- 初始几何从 `cases/STRU`（Cartesian_angstrom）直接计算。

注意：`OUT.*/STRU.cif` 是运行**开始前**写出的初始结构（`esolver_fp.cpp before_all_runners`），不能用于本测试。

## 5. 判据表（全部阻塞）

| 编号 | 判据 | 阈值 | 状态 |
|---|---|---|---|
| J1 | 每 λ 点：两路径 Δr_OH 之差 | ≤0.005 Å | **阻塞**（P01/F1） |
| J2 | 每 λ 点：两路径 Δ∠ 之差 | ≤0.5° | **阻塞** |
| J3 | 符号一致（λ>0 与对应 E 同号） | sign 相同 | **阻塞** |

判定段打印 `WARNING: 判定待 F1/F2 备忘录定稿`，数值照常记录，`SUMMARY: 0/0 PASS`，退出码 0。

## 6. 风险与注意

- 场致几何变化本身是微小量：scf_thr 已收紧至 1e-8，force_thr_ev=1e-3 eV/Å；若 SCF 未收敛（INVALID），该点几何不可信。
- F1 工作值 E=−πλ/(2a) 使 λ=0.02 ↔ E≈−0.001047 Ha/Bohr、λ=0.04 ↔ E≈−0.002094；备忘录若改 F1 定义（如 E=−πλ/a），只需改 run.sh 头部 `e_of_lam`。
- 大 λ 点注意非线性（偶极自洽回灌）；Δr/Δ∠ 随 |λ| 应近似线性增长，可在 results.txt 数据表中人工核查。
- 若 relax 未在 relax_nmax 50 内收敛，STRU_ION_D 仍是最后一帧——判据照常记录，但请人工确认 warning.log。
