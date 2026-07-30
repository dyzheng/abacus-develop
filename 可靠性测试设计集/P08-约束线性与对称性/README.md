# P08　约束响应线性与 ±λ 对称性

> 系列：DeltaP ｜ 优先级：P0 ｜ **阻塞状态：无**

## 1. 目的

完整刻画 DeltaP total 约束的响应函数：定位线性区间与失稳边界（λ_crit）。响应测量的合法性建立在约束窗口的线性与 ±λ 反对称上。产出是 P01/P05 窗口选择的依据。

## 2. 目录结构

```
P08-约束线性与对称性/
  README.md
  run.sh
  cases/h2o/{STRU,KPT}     # H2O 实验几何，30 Bohr 盒，Γ 点
  runs/
    lam_*/                 # lambda = ±0.08/±0.04/±0.02/±0.01/0 九点
    lambda_scan.dat        # lambda, γ_raw, SCF 迭代数, O 原子 Fz, VALID/INVALID
    results.txt
```

## 3. 用法

```bash
./run.sh
ABACUS=/path/to/abacus NPROC=4 ./run.sh
```

环境变量同 P01：`ABACUS` / `PSEUDO_DIR` / `ORBITAL_DIR` / `NPROC`。

## 4. 工作流与判据

九点独立 SCF（固定 λ，lambda_step=0，total 模式；负 λ 用 mixing_beta=0.3）。每点 grep 到 `SCF IS NOT CONVERGED` 或提取不到 γ 时标记 INVALID 并记录为 λ_crit 候选（取最小 |λ| INVALID 点）。

| 项 | 判据 |
|---|---|
| 全线性 | VALID 点最小二乘 R² ≥ 0.98 |
| 反对称 | max \|γ(λ)+γ(−λ)\| / \|γ(λ)−γ(−λ)\| < 5%（λ=0.01/0.02/0.04/0.08 四对） |
| 子窗口斜率 | \|斜率(\|λ\|≤0.02) − 斜率(全窗口)\| / \|全窗口斜率\| ≤ 3% |

末尾 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

`lambda_scan.dat` 同时记录逐点 SCF 迭代数与 TOTAL-FORCE 中 O 原子 z 分量（eV/Å），**供 P03 Born 有效电荷复用**。

## 5. 已知风险（设计文档 §7）

- 逐点记录 SCF 迭代数——迭代数骤增是"近失稳"前兆；λ_crit 由 INVALID 点定位并记录。
- 若 ±0.08 已非线性，生产窗口应缩至 ±0.04（看子窗口斜率差判据）。
- 负 λ 不收敛是既有失败模式（R1）：本脚本负 λ 用 mixing_beta=0.3；仍发散时按《DeltaP测试说明.md》§三 用 λ=0 电荷密度热启动手工补点。
