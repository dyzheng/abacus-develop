# P02　H₂O 平衡偶极四方对标

> 系列：DeltaP ｜ 优先级：P0 ｜ **阻塞状态：无**

## 1. 目的

验证四条独立路径给出同一平衡偶极：LCAO Wilson-γ、PW 标准 Berry 相位、wannier90 Wannier 中心、实验值 1.855 D。平衡量不经过约束核，验证的是 γ 测量机器本身——P01 及一切响应测试的读数基础。

## 2. 目录结构

```
P02-H2O平衡偶极四方对标/
  README.md
  run.sh
  cases/h2o/{STRU,KPT}   # H2O 实验几何（偶极沿 z），30 Bohr 盒，Γ 点
  runs/
    lcao_gdir3/          # 通道1: LCAO DeltaP lambda=0 测量
    pw_gdir{1,2,3}/      # 通道2: PW berry_phase=1 三方向
    results.txt
```

## 3. 用法

```bash
./run.sh
ABACUS=/path/to/abacus NPROC=4 ./run.sh
```

环境变量同 P01：`ABACUS` / `PSEUDO_DIR` / `ORBITAL_DIR` / `NPROC`。

## 4. 换算

- `μ(e·Bohr) = (A_BOHR/π)·Σγ_raw`（LCAO）或 `(A_BOHR/π)·γ_total`（PW），`A_BOHR=30.0`
- Debye = e·Bohr × 2.541746（= ÷0.393430）
- **离子项假设**：假设代码输出的 γ 已含离子点电荷项（参照《DeltaP测试说明.md》§2），**此假设待 P02 实跑确认**；若确认不含，需手工加离子项后重判。

## 5. 判据

| 项 | 判据 |
|---|---|
| LCAO vs PW | 互差 ≤ 0.02 D（取 z 分量） |
| LCAO vs 实验 | ≤ 0.05 D |
| PW vs 实验 | ≤ 0.05 D |

wannier90 通道：`command -v wannier90` 检测，未安装打印 SKIP（安装时亦需手动后处理，见 run.sh 输出）。末尾 `SUMMARY: n/n PASS`，全过 exit 0。

## 6. 已知风险（设计文档 §7）

- wannier90 的 MLWF 不唯一（disentanglement 窗口），取收敛后中心。
- 分支/规范：PW 的 γ_total 是 wrapped（mod 2π），LCAO 的 Σγ_raw 是 unwrapped，绝对值可差 2πN——比较偶极前需确认落在同一物理分支；30 Bohr 盒中 2π 对应约 1.67 D，分支错位会立刻表现为 FAIL。
- PBE 系统性高估约 0.03 D 属已知，记录即可（判据已留余量）。
- 若通道互差超限，P01 的 F2（γ↔μ 因子）必须优先完成——本测试同时是 F2 的数据来源。
