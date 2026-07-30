# P07 基组与截断双收敛（DZP/TZDP/QZDP × ecut 40–100 Ry）

> 系列：DeltaP ｜ 前置：无 ｜ 设计文档：`../P07-基组与截断双收敛.md`

## 1. 目的

分别测定 LCAO 基组层级与 PW 截断能对 H₂O（15 Å 盒）偶极 μ 与极化率 α 的收敛，确立生产基组；并为"PW≈LCAO"对标提供各自收敛后的公平比较基准。

## 2. 目录结构

```
P07-基组与截断双收敛/
  README.md
  run.sh
  cases/
    KPT                     Γ 点
    basis_dzp/STRU          15 Å H2O，轨道名 O/H_gga_6au_100Ry_DZP.orb（占位）
    basis_tzdp/STRU         同上，O_gga_6au_100Ry_2s2p1d.orb / H_gga_6au_100Ry_2s1p.orb（默认层级，已存在）
    basis_qzdp/STRU         同上，O/H_gga_6au_100Ry_QZDP.orb（占位）
    INPUT.deltap.tmpl       LCAO λ 测量模板（@ECUT@ 占位，LCAO 固定 100）
    INPUT.efield.tmpl       LCAO efield 模板
    INPUT.pw.tmpl           PW λ=0 测量模板（berry_phase 1, gdir 3, deltap_switch true）
    INPUT.pw_efield.tmpl    PW efield 模板
  runs/                   run.sh 生成
```

**层级轨道补充**：dzp/qzdp 层级的 STRU 使用占位轨道文件名（`*_DZP.orb`、`*_QZDP.orb`）。run.sh 运行前检测 `$ORBITAL_DIR` 下文件存在性，缺失则 **SKIP 该层级** 并打印缺失文件名；请把对应层级轨道文件放入 `$ORBITAL_DIR`（默认 `/root/pporb/apns-orbitals-efficiency-v1`）后重跑，或修改 STRU 中 `NUMERICAL_ORBITAL` 文件名指向实际文件。

## 3. 用法

```bash
bash run.sh
NPROC=4 ORBITAL_DIR=/path/to/orbitals bash run.sh
```

LCAO 每层级 6 次 SCF（λ=0 + efield 五点），PW 每截断 6 次，最多 42 次。结果在 `runs/results.txt`，原始提取在 `runs/data.tsv`。

## 4. 提取与换算

- LCAO 偶极：`[rawG] Σγ_raw`；PW 偶极：`[DeltaP-PW] γ_total`（wrapped，mod [−π,π]）。
- `μ = (a/π)·γ`，a = 15×1.889726 = 28.346 Bohr。
- α：efield 五点（±0.0005/±0.001/0 Ha/Bohr，dip_cor=1）对 E_KohnSham(Ry) 二次最小二乘，α = −c2。

## 5. 判据表（全部有效，无阻塞）

| 编号 | 判据 | 阈值 |
|---|---|---|
| J1 | LCAO 相邻层级（存在的连续对）μ 差 / α 差 | ≤2% / ≤3% |
| J2 | PW ecut80→100 μ 差 / α 差 | ≤1% / ≤2% |
| J3 | LCAO 最高可用层级 vs PW ecut100 的 μ（γ 域 mod 2π 对齐分支后）互差 | ≤5% |

J3 说明：LCAO Σγ_raw 为 unwrapped，PW γ_total wrapped，绝对值差 2πN 属正常；脚本在 γ 域取 mod 2π 最小差后换算回 μ 比较。若分子偶极在 γ 域接近分支边界（|γ|≈π），mod 对齐仍可能错档，此时请人工核对 `runs/data.tsv` 中两通道的 γ 值。

## 6. 风险与注意

- 极化率对弥散函数敏感，DZP 数据仅作误差标注，不作生产。
- PW 通道显式 `nbands 8`（H₂O 8 电子，nspin=1 占 4 带，留空带保 SCF 稳定）。
- PW + efield + dip_cor 的组合依赖与 LCAO 相同的外场模块；若某点 SCF 未收敛，data.tsv 标 INVALID，该点量缺失、判据 FAIL（数据缺失）。
- 若全部 LCAO 层级缺失（极端情况），J1/J3 退化，仅 PW 判据有效。
