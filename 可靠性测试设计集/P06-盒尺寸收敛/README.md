# P06 盒尺寸收敛（偶极与极化率 vs L）

> 系列：DeltaP ｜ 前置：无（偶极）/ P01（α 部分）｜ 设计文档：`../P06-盒尺寸收敛.md`

## 1. 目的

系统测定 H₂O 偶极矩 μ 与极化率 α 对盒尺寸 L 的收敛曲线，确立各性质的"收敛盒尺寸"红线；排除 07-22 内部验证中发现的表观 α 的 a² 标度假象。

## 2. 目录结构

```
P06-盒尺寸收敛/
  README.md            本文
  run.sh               主工作流：生成 runs/ → 批量运行 → 提取 → 判据 → PASS/FAIL
  cases/
    KPT                Γ 点
    STRU_L12..STRU_L24 H2O 五个盒尺寸（L=12/15/18/21/24 Å，O 居盒心，C2 沿 z）
    INPUT.deltap.tmpl  LCAO DeltaP 模板（@LAMBDA@ @MIXB@ 占位）
    INPUT.efield.tmpl  LCAO efield 模板（@EFIELD@ 占位，dip_cor_flag 1，efield_dir 2=z）
  runs/                run.sh 生成的产物（L$L/lam_0、ef_*、lam_m0p02、lam_p0p02）
```

## 3. 用法

```bash
bash run.sh                     # 默认 ABACUS=/root/abacus-develop/build/abacus_basic_para
NPROC=4 bash run.sh             # MPI 并行
ABACUS=/path/to/abacus bash run.sh
```

每盒 7 次 SCF（λ=0、efield 五点、λ=±0.02），共 35 次。每次运行前 `rm -rf OUT.*`，输出 `run.log`。结果汇总在 `runs/results.txt`，原始提取在 `runs/data.tsv`。

## 4. 提取与换算

- `[rawG]` 行 `Σγ_raw=`（run.log）；`E_KohnSham`（`OUT.*/running_scf.log` 末行）。
- 偶极：`μ = (a/π)·Σγ_raw`，a = L×1.889726 Bohr（F2 工作值）。
- α_ref(L)：五点 efield（±0.0005/±0.001/0 Ha/Bohr）对 E_KohnSham(Ry) 做二次最小二乘，α = −c2（Ry/Ha² → Ha 制 a.u.，推导见 run.sh 注释）。
- α_DeltaP(L)：α = (2a²/π²)·[γ(+0.02)−γ(−0.02)]/0.04，由 F1（E=−πλ/(2a)）与 F2 组合推出。

## 5. 判据表

| 编号 | 判据 | 阈值 | 状态 |
|---|---|---|---|
| J1 | μ：15 Å 与 24 Å 相对差 | ≤3% | 有效 |
| J2 | α_ref：15 Å 与 24 Å 相对差 | ≤5% | **阻塞**（P01/F1-F2） |
| J3 | α_DeltaP：15 Å 与 24 Å 相对差 | ≤5% | **阻塞** |
| J4 | 两通道 α(L) 单调趋势一致（逐段符号比对） | 4/4 段 | **阻塞** |

阻塞段打印 `WARNING: 判定待 F1/F2 备忘录定稿`，数值照常记录，不计入 SUMMARY；换算常量集中在 run.sh 头部（PI、BOHR_PER_ANG、LAM_DP），备忘录定稿后单点修改。

## 6. 风险与注意

- 大盒 LCAO 需检查 SMO 半径（onsite_radius=6 Bohr）与盒的相容性；L=24 Å 时分子距边界仍 >9 Å，预期安全。
- efield 通道每点 `dip_cor_flag 1`（模板已内置）。
- α_ref 的二次拟合对 scf_thr 敏感，已统一 1.0e-8；若某点 SCF 未收敛，data.tsv 标记 INVALID，该盒对应量缺失并打印 WARNING。
- J4 只做符号一致性（4 段），若某通道在某段差值为 0 该段自动跳过。
