# P10 E–D 曲线：total 约束扫描 vs 外场扫描

> 系列：DeltaP ｜ 前置：P01 ｜ **状态：阻塞（判定待 F1/F2 备忘录定稿）** ｜ 设计文档：`../P10-ED曲线约束vs外场.md`

## 1. 目的

Legendre 对偶预言：扫 λ（固定 D 系综）与扫 E（电焓系综）给出同一条 E–D 曲线。本测试用两条独立路径的曲线重合度直接验证"DeltaP 就是有限电场"的论断。

## 2. 目录结构

```
P10-ED曲线约束vs外场/
  README.md
  run.sh
  cases/
    KPT                 Γ 点
    STRU                30 Bohr (15.873 Å) 盒 H2O，O 居盒心，C2 沿 z
    INPUT.deltap.tmpl   DeltaP total 模板（@LAMBDA@ 占位）
    INPUT.efield.tmpl   efield 模板（@EFIELD@ 占位，dip_cor_flag 1）
  runs/                 run.sh 生成（lam_* / ef_* 各 7 点）
```

## 3. 用法

```bash
bash run.sh
NPROC=4 bash run.sh
```

14 次 SCF：λ ∈ {−0.08,−0.04,−0.02,0,0.02,0.04,0.08} Ry（DeltaP total，λ 冻结）；E ∈ {−0.004,…,0.004} Ha/Bohr（efield，dip_cor=1）。每点同时记录 `Σγ_raw` 与 `E_KohnSham`。

## 4. 分析方法

- F1（工作值，run.sh 头部常量区）：λ 点换算 `E = −π·λ/(2a)`，a=30 Bohr。
- F2：`μ = (a/π)·γ_raw`。
- DeltaP 侧物理能量（Ry 记账）：`E_phys = E_KS − λ·Σγ_raw`（减去约束能，同 DeltaSpin escon 记账）。
- 两条曲线 γ(E) 各自线性最小二乘：斜率、截距；α 等价量 = −(a/π)·slope。
- 能量二阶导 κ 三方：
  - κ_dp：E_phys 对 γ 二次拟合（κ=2c2，Ry/rad²）→ α 等价 = −(2a²/π²)/κ_dp；
  - κ_ef：E_KS 对 E 二次拟合（Ry/Ha²）→ α 等价 = −κ_ef；
  - κ_fd：efield 能量在 δ=0.001 的中心差分 → α 等价 = −κ_fd/2。
  三方 κ 量纲不同，可比量为各自 α 等价（见 run.sh 报表）；备忘录定稿后改为统一单位直接比 κ。

## 5. 判据表（全部阻塞）

| 编号 | 判据 | 阈值 | 状态 |
|---|---|---|---|
| J1 | 两曲线斜率差（即 α 一致性） | ≤10% | **阻塞**（P01/F1-F2） |
| J2 | γ(E=0) 与 γ(λ=0) 自洽差 | <0.01 rad | **阻塞** |
| J3 | κ 三方互差 | ≤10% | **阻塞** |

判定段打印 `WARNING: 判定待 F1/F2 备忘录定稿`，全部数值照常记录于 `runs/results.txt`，`SUMMARY: 0/0 PASS`，退出码 0。备忘录定稿后仅需修改 run.sh 头部 F1/F2 常量与判据段启用。

## 6. 风险与注意

- sawtooth 不连续面远离分子（30 Bohr 盒自然满足）；每点 dip_cor=1。
- 负 λ 收敛困难：模板对 λ<0 自动用 mixing_beta=0.3（正 λ 用 0.4）；若仍发散，按 `../DeltaP测试说明.md` §三 热启动。
- 大 |λ|=0.08 Ry 点可能超出线性窗口（参 P08），离群点会拉偏线性拟合——分析时对照 data.tsv 检查 γ(λ) 线性度，必要时剔除端点后手动复核。
- λ 扫描的 E 轴覆盖范围（±π·0.08/60≈±0.00419 Ha/Bohr）与 efield 扫描（±0.004）近似匹配，这是本测试设计有意为之；F1 改动会改变匹配关系，需同步调整 E_POINTS。
