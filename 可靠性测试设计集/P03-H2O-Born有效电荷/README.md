# P03　H₂O Born 有效电荷（dF/dλ → Z*）

> 系列：DeltaP 可靠性测试 ｜ 前置：**P01（阻塞）** ｜ 设计文档：`../P03-H2O-Born有效电荷.md`

## 1. 目的

Born 有效电荷 $Z^*_I$ 是极化对原子位移的导数。DeltaP 提供独特测法：Maxwell 关系
$Z^*_I=(a/\pi)\,dF_I/d\lambda$。本测试用两条独立通道互验：

- **通道①（DeltaP 约束力）**：固定几何，扫 λ ∈ {−0.01, −0.005, 0.005, 0.01} Ry，
  取每点 O/H 的 z 向 `TOTAL-FORCE`，线性拟合 dF_O/dλ、 dF_H/dλ（R²≥0.98）→
  Z*_O = (a/π)·dF_O/dλ，Z*_H 同理（对称性要求 Z*_H = −Z*_O/2）。
- **通道②（berry_phase 位移 FD）**：PW 路径，O 原子 z 向 ±0.005 Å 位移各跑一次，
  取 `[DeltaP-PW] γ_total`，Z*_O = (a/π)·Δγ/Δr。

## 2. 目录结构

```
cases/h2o/STRU, KPT          H2O 实验几何, 30 Bohr (15.8753 Å) 立方盒, 偶极沿 z
cases/INPUT_lcao_force.tmpl  LCAO DeltaP 力测量模板 (cal_force 1)
cases/INPUT_pw_berry.tmpl    PW berry_phase + DeltaP-PW 测量模板 (λ=0)
run.sh                       主工作流
runs/                        运行产物 (run.sh 生成, 结果汇总于 runs/results.txt)
```

## 3. 用法

```bash
bash run.sh                 # 默认 ABACUS=/root/abacus-develop/build/abacus_basic_para
NPROC=4 bash run.sh         # MPI 并行
ABACUS=/path/to/abacus bash run.sh
```

## 4. 判据

| 项 | 判据 | 说明 |
|---|---|---|
| 通道① 力-λ 线性 | R² ≥ 0.98（O、H 各自） | 窗口线性先决条件 |
| 两通道互差 | \|Z*_O(ΔF/Δλ) − Z*_O(位移FD)\| ≤ 0.05 | 双实现对标 |
| 电荷中和 | \|Z*_O + 2·Z*_H\| ≤ 0.05 | 声学求和规则 |
| 文献区间 | Z*_O ∈ [−2.0, −1.5] | 越界打印 WARNING（不计 FAIL） |

末尾打印 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

## 5. 阻塞状态

**阻塞于 P01**（F1 λ↔E、F2 γ↔μ 换算链可信度）。run.sh 完整可跑，判定段打印
`WARNING: 判定待 F1/F2 备忘录定稿`。换算系数集中于 run.sh 头部常量区
（`F1_PI_OVER_A`、`EVA_TO_RYBOHR`、`F2_SPIN_FACTOR`），备忘录定稿后单点修改。

## 6. 风险与注意（抄自设计文档 §7）

- F_I 对 λ 的窗口内线性必须先验证（R²≥0.98），否则拟合斜率无意义；
- 位移 FD 与约束 FD 的步长误差同阶，取双侧差分；
- 负 λ 的 SCF 更难收敛，脚本对负 λ 自动改用 mixing_beta 0.3；
- TOTAL-FORCE 单位为 eV/Å，换算为 Ry/Bohr 的系数 `EVA_TO_RYBOHR` 待备忘录确认。
