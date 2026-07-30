# P15　极化率张量各向异性（分量级验证）

> 系列：DeltaP 可靠性测试 ｜ 前置：**P01（阻塞）** ｜ 设计文档：`../P15-极化率张量各向异性.md`

## 1. 目的

P01/P05 验证各向同性平均；本测试验证三分量独立测量。H₂O 的 α_∥ 与 α_⊥ 差异
（约 10–15%）与 CO 的 ∥/⊥ 大差异是小而真实的物理信号，检验 DeltaP 的分量
分辨率，为 Pol130/K3 张量测试奠基。

## 2. 算例清单与通道

分子：H₂O（30 Bohr 盒，C2 轴沿 z）、NH₃（12 Å 盒，C3 轴沿 z）、CO（12 Å 盒，
键轴沿 z）。每分子 gdir=1/2/3 三方向，每方向双通道：

- **通道①（efield FD）**：`efield_flag 1, dip_cor_flag 1, efield_dir=gdir`，
  efield_amp ∈ {−0.001, 0, +0.001} Ha，三点 FD：
  α_ii = −[E(+δ)+E(−δ)−2E(0)]/(2δ²)（E 取 `E_KohnSham`，Ry；δ 单位 Ha，除 2 化为 Ha 制）。
- **通道②（DeltaP）**：λ ∈ {−0.02, 0, +0.02} Ry（含 λ=0 点使 R² 有意义），
  取 `[rawG] Σγ_raw`，线性拟合 dγ/dλ（要求 R²≥0.99），
  α_ii = (2a²/π²)·dγ/dλ（系数见 run.sh 常量区 `alpha_coef`）。

## 3. 目录结构与用法

```
cases/<mol>/{STRU,KPT}   三分子实验几何（主轴精确沿坐标轴）
cases/INPUT_efield.tmpl  efield FD 模板
cases/INPUT_deltap.tmpl  DeltaP 模板
run.sh                   主工作流；结果汇总 runs/results.txt
```

```bash
bash run.sh                 # 或 NPROC=4 bash run.sh
```

## 4. 判据

| 项 | 判据 |
|---|---|
| 逐分量两通道一致 | \|α_DeltaP − α_ref\|/α_ref ≤ 10%（3 分子 × 3 分量 = 9 项） |
| 窗口斜率线性 | 每个 DeltaP 窗口 R² ≥ 0.99（9 项） |
| 各向异性比 α_∥/α_⊥（∥=z，⊥=(x+y)/2） | 两通道比值差 ≤ 5%（3 项） |
| CO 方向关系 | α_∥ > α_⊥（两通道各自，2 项） |

末尾 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

## 5. 阻塞状态

**阻塞于 P01**（F1/F2 换算链）。run.sh 完整可跑，判定段打印
`WARNING: 判定待 F1/F2 备忘录定稿`。

## 6. 风险与注意（抄自设计文档 §7）

- 分子必须精确对齐坐标轴（主轴旋转 1e-6 rad 内）——cases/ 中几何由解析坐标
  直接生成，不做任何弛豫；
- 小差异要求窗口斜率 R²≥0.99，run.sh 对每个 DeltaP 窗口单独判定；
- 负 λ 更难收敛，脚本对负 λ 自动改用 mixing_beta 0.3。
