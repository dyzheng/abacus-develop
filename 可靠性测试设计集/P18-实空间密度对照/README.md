# P18 实空间 Δρ(r) 对照（微扰的密度指纹）

> 系列：DeltaP ｜ 前置：P01 ｜ **状态：阻塞（判定待 F1/F2 备忘录定稿）** ｜ 设计文档：`../P18-实空间密度对照.md`

## 1. 目的

最底层等价性检验：等效扰动（λ=0.02 Ry vs E=−πλ/(2a)）产生的密度变化 Δρ(r) 应在实空间一致——比标量 α 敏感得多的约束核完备性成像。

## 2. 目录结构

```
P18-实空间密度对照/
  README.md
  run.sh
  cases/
    KPT                 Γ 点
    STRU                30 Bohr 盒 H2O
    INPUT.deltap.tmpl   LCAO DeltaP 模板（scf_thr 1e-9, out_chg 1）
    INPUT.efield.tmpl   LCAO efield 模板（同上 + dip_cor=1）
    INPUT.pw.tmpl       PW 模板（ecut 80, 可选 LCAO-vs-PW 通道）
  runs/                 run.sh 生成（lam_0, lam_p0p02, ef_0, ef_pert, pw_0, pw_p0p02）
```

## 3. 用法

```bash
bash run.sh
NPROC=4 bash run.sh
```

6 次 SCF：λ=0、λ=0.02、E=0、E=−π·0.02/60≈−0.00104720 Ha/Bohr（LCAO），外加 PW λ=0/λ=0.02（可选通道）。scf_thr 收紧到 1e-9。

## 4. 密度产物与分析

- `out_chg 1`（nspin=1, scf）产物为 `OUT.autotest/chg.cube`（源码 `ctrl_output_fp.cpp`：nspin=1 无自旋后缀、scf 无几何步后缀）。旧版/其他设置可能写成 `<suffix>-CHARGE-DENSITY.restart`（格点格式不同）——脚本只认 `chg.cube`，缺失则 SKIP 并在 results.txt 说明。
- 分析：每条 cube 投影到 z 轴的平面平均剖面 Δρ(z)（cube 数据 x 指标最快，`prof[iz]=Σ_{ix,iy}ρ/(nx·ny)`）。
- Δρ_λ(z) = prof(λ=0.02) − prof(λ=0)；Δρ_E(z) = prof(E) − prof(E=0)。
- **归一化按 Δμ 标定**（非名义强度）：SCALE = Δγ_λ/Δγ_E（μ=(a/π)γ，比值中 a/π 约去），Δρ_E 乘 SCALE 后与 Δρ_λ 比较。
- 指标：Pearson 相关系数 r 与 RMS 残差（RMS(Δρ_λ−s·Δρ_E)/RMS(Δρ_λ)）。有 python3 用 python3，否则用 awk 等价实现（同算法，z 剖面）。
- 密度对齐：同一 STRU/网格设置保证两 cube 同网格同原点。

## 5. 判据表（全部阻塞）

| 编号 | 判据 | 阈值 | 状态 |
|---|---|---|---|
| J1 | LCAO Δρ_λ vs Δρ_E：Pearson r / RMS 残差 | r≥0.99 / ≤5% | **阻塞**（P01/F1-F2） |
| J2 | LCAO vs PW 同 λ 密度对照（可选通道，PW cube 缺失则 SKIP） | 参考 | **阻塞** |

判定段打印 `WARNING: 判定待 F1/F2 备忘录定稿`，`SUMMARY: 0/0 PASS`，退出码 0。

## 6. 风险与注意

- Δμ 标定依赖 `[rawG] Σγ_raw` 提取成功；若某点 SCF 未收敛（run.sh 启动段打印 INVALID），该通道判据无效。
- PW 通道 γ_total 为 wrapped 值，SCALE_PW 计算中已做 ±2π 解绕。
- 若残差在键区/尾部系统性聚集 → 约束核完备性问题，联动 P16/T09（见设计文档 §5）。
- z 剖面是 3D 全相关的降维；如需逐点 3D Pearson，可在 analyze() 基础上去掉平面平均（网格一致时直接逐体素比较）。
