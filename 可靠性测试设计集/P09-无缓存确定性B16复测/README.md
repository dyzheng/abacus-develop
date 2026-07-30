# P09　无缓存确定性（B16 复测）

> 系列：DeltaP ｜ 优先级：P0 ｜ **阻塞状态：无**

## 1. 目的

正式关闭历史风险 B16：无缓存全新运行三次 γ 值不同（+0.098/−0.062/−0.098）。根因已定位为 zgeev 特征值排序不定 + 贪心匹配 + 累加器 bug，07-21 全局 K=5 修复后未复测。本测试不通过则一切响应测试暂停（读数不可复现，无从对标）。

## 2. 目录结构

```
P09-无缓存确定性B16复测/
  README.md
  run.sh
  cases/h2o/{STRU,KPT}   # H2O 实验几何，30 Bohr 盒，Γ 点（与 P01/P02/P08 相同）
  cases/hbn/{STRU,KPT}   # h-BN 六方晶胞 a=2.512 c=6.692 Å，4x4x2 k 点
  runs/
    h2o/{omp1_a,omp1_b,omp1_c,omp4,omp8}/
    hbn/{omp1_a,omp1_b,omp1_c,omp4,omp8}/
    results.txt
```

两体系均为 LCAO + DeltaP λ=0 测量模式（gdir=3）。PW 双实现为后续扩展（见风险节）。

## 3. 用法

```bash
./run.sh
ABACUS=/path/to/abacus NPROC=4 ./run.sh
```

环境变量同 P01：`ABACUS` / `PSEUDO_DIR` / `ORBITAL_DIR` / `NPROC`。

## 4. 工作流与判据

每体系 5 次**无缓存全新运行**（每次 `rm -rf OUT.*`）：`OMP_NUM_THREADS=1` × 3 次，`OMP_NUM_THREADS=4`、`8` 各 1 次。对比 5 次的 `[rawG] Σγ_raw` 与 `E_KohnSham`（Ry→Ha 换算 ÷2）：

| 项 | 判据 |
|---|---|
| 逐位一致（γ） | max\|Δγ\|（对 5 次均值） ≤ 1e-6 rad |
| 逐位一致（E） | max\|ΔE\| ≤ 1e-8 Ha |

H₂O 与 h-BN 各判 2 项共 4 项。末尾 `SUMMARY: n/n PASS`，全过 exit 0，否则 exit 1。

**记录要求**：run.sh 开头自动打印 ABACUS 版本、链接的 BLAS/LAPACK（ldd）、编译器与 MPI 版本——zgeev 排序对 LAPACK 实现敏感，结果存档时必须随附此段输出。

## 5. 已知风险（设计文档 §7）

- 必须"无缓存全新运行"（含 wavefunction/charge 缓存）——脚本每次运行前 `rm -rf OUT.*` 保证。
- 记录编译器/BLAS 版本（zgeev 排序对 LAPACK 实现敏感）。
- 换机器/换 BLAS 后本测试应重跑；NPROC>1（MPI 并行归约）情形下确定性未在本脚本覆盖，如需请设 NPROC 后重跑并记录。
- 原设计要求 LCAO 与 PW 双实现；当前脚本仅 LCAO 通道（PW 通道的 γ_total 是 wrapped 量，比较口径需另行约定），作为已知偏差记录。
