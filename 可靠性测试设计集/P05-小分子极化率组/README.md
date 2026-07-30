# P05　小分子极化率组三方对标（H₂O/NH₃/CH₄/CO/HF）

> 系列：DeltaP 可靠性测试 ｜ 前置：**P01（阻塞）** ｜ 设计文档：`../P05-小分子极化率组.md`

## 1. 目的

把 P01 的裁决从单点扩展到 5 分子组，覆盖 α 从 ~5.6（HF）到 ~17.5 a.u.（CH₄），
检验 DeltaP 响应测量在不同极化率量级上的可靠性；同时以 DZP/TZDP 双基组层级
标注基组误差。

## 2. 算例清单与通道

分子与盒：H₂O（30 Bohr 盒）、NH₃/CH₄/CO/HF（12 Å 盒），实验几何，主轴沿 z。
CH₄ 无极性方向，gdir=1/2/3 三方向独立测量后取平均。

- **通道①（efield 能量 FD，裁判）**：`efield_flag 1, dip_cor_flag 1, efield_dir=方向`，
  efield_amp ∈ {−0.001, −0.0005, 0, +0.0005, +0.001} Ha，取 `E_KohnSham`（Ry）
  对 δ(Ha) 做二次拟合：α_ref = −c₂（c₂ 为 E(Ry) 对 δ(Ha) 的二次系数，
  推导：α = −d²E_Ha/dδ²，E_Ha = E_Ry/2 ⇒ α = −c₂_Ry）。
- **通道②（DeltaP）**：λ ∈ {−0.08, −0.02, +0.02, +0.08} Ry，取 `[rawG] Σγ_raw`，
  线性拟合 dγ/dλ，α_DeltaP = (2a²/π²)·dγ/dλ
  （由 E = −πλ/(2a)、α = −(a/π)·dγ/dE 复合而得，系数见 run.sh 常量区 `alpha_coef`）。

基组层级：`tzdp/`（默认轨道 2s2p1d/H 2s1p）与 `dzp/`（占位层级，
轨道文件名如 `O_gga_6au_100Ry_2s1p.orb`）。run.sh 运行前检查 ORBITAL_DIR 中
层级轨道文件存在性，缺失则 **SKIP 该层级**并打印说明。

## 3. 目录结构与用法

```
cases/INPUT_efield.tmpl    efield FD 模板（@EFIELD_AMP@/@EFIELD_DIR@）
cases/INPUT_deltap.tmpl    DeltaP 模板（@LAMBDA_INIT@/@GDIR@/@MIXING_BETA@）
cases/tzdp/<mol>/{STRU,KPT}   TZDP 层级五分子
cases/dzp/<mol>/{STRU,KPT}    DZP 层级五分子（占位）
run.sh                     主工作流；结果汇总 runs/results.txt
```

```bash
bash run.sh                 # 或 NPROC=4 bash run.sh
```

## 4. 判据

| 项 | 判据 |
|---|---|
| 逐分子双通道一致 | \|α_DeltaP − α_ref\|/α_ref ≤ 10%（5/5） |
| 对 CCSD(T) 组 MAE（两通道均值，相对） | ≤ 15%（9.85/14.6/17.5/13.1/5.6 a.u.） |
| α 排序 | 与参考一致：HF < H₂O < CO < NH₃ < CH₄ |

末尾 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

## 5. 阻塞状态

**阻塞于 P01**（F1 λ↔E、F2 γ↔μ 换算链）。run.sh 完整可跑，判定段打印
`WARNING: 判定待 F1/F2 备忘录定稿`。换算系数集中在 run.sh 头部常量区
（`alpha_coef`、`F2_SPIN_FACTOR`）。PW 第三方通道待 P16 整改后补入。

## 6. 风险与注意（抄自设计文档 §7）

- CH₄ 无极性方向，三方向独立测（脚本对 gdir=1/2/3 各跑双通道并平均）；
- 低 α 分子（HF）是小信号，注意 λ=±0.08 处非线性侵入窗口——run.sh 输出每窗口
  dγ/dλ 的 R² 供巡检；
- 负 λ 更难收敛，脚本对负 λ 自动改用 mixing_beta 0.3；
- DZP 层级轨道当前未在 ORBITAL_DIR 提供，该层级默认 SKIP。
