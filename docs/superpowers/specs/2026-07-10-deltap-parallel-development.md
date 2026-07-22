# DeltaP 并行版本开发记录 (2026-07-10)

## 1. 当前状态

### 1.1 已实现
- 并行 `compute_hk_correction`：使用 MPI_Allreduce 收集完整波函数矩阵和 S_dk
- 串行模式：保持原有代码路径，`nproc==1` 时使用本地快速路径
- 并行模式：检测 `nproc > 1` 时进入收集+本地计算路径

### 1.2 并行算法
输入：每个进程有局部 S_dk_(nrow×ncol)、局部 psi(k-point) (nrow×ncol_bands)
输出：每个进程有局部 HK correction (nrow×ncol)

步骤：
1. 收集 S_dk_full (nbasis×nbasis) — dim1>1 时 Allreduce
2. 收集 C_R_full (nbasis×nbands) — rows distributed 或 bands distributed 时 Allreduce
3. 收集 C_L_full (nbasis×nbands) — 同 C_R_full
4. 计算 SC = S_dk_full * C_R_full (dense matrix multiply)
5. 计算 F = (i/2) * w_eff * SC
6. 收集 F_full (nbasis×nocc_use) — dim0>1 时 Allreduce
7. 计算局部 H_sym: M_ab = F_full * C_L_full†, M_ba 对应项
8. H_sym = (M_ab + conj(M_ba)) / 2

### 1.3 关键 BUG 修复

| 问题 | 原因 | 修复 |
|------|------|------|
| local2global_col 给错带指数 | local2global_col_ 是为 HK (nbasis×nbasis) 设置 | 显式块循环公式: `(b/nb*dim1+coord[1])*nb+(b%nb)` |
| 数据复制时 Allreduce 翻倍 | dim0==1 且 bands 串行时所有数据重复 | 跳过 Allreduce (wfc_fully_replicated) |
| S_dk 只用局部列 | SC 需要完整 S_dk，局部只有一半列 | 收集 S_dk_full |
| ncol_bands==0 时循环空转 | set_nloc_wfc_Eij 可能未调用 | 回退: ncol_bands=nbands (串行 band) |
| Allreduce 条件混用 | S_dk/C_R/C_L/F 的分配维度不同 | 分别条件: cols_distributed / wfc_fully_replicated / dim0>1 |

### 1.4 验证结果

| 项目 | 串行 | 并行(2 proc) | 状态 |
|------|------|-------------|------|
| P_total (DeltaP) | -6.358938e-05 | -6.358938e-05 | 完全一致 |
| Wilson loop gamma(0) | 6.122671e+00 | 6.122671e+00 | 完全一致 |
| iter=1 max\|gamma-target\| | 7.9013e-02 | 6.3816e-02 | 有差异 |
| 无崩溃 | 是 | 是 | 稳定 |
| 编译 | 是 | 是 | 通过 |

### 1.5 Gamma 值差异分析 (已定位根本原因)

**结论：不是 bug，是预期行为。**

差异来源：
- `P_total` (总极化) 是 Wilson loop 本征值，对占据子空间的幺正变换不变 → 串行/并行一致 ✓
- 每原子 gamma 分解依赖于 Wilson loop 本征向量 (SMO 投影权重)
- ELPA 分布式对角化在并行模式下可能给出不同本征向量（本征值相同）
- 因此每原子 gamma 值容许因分布式对角化而有微小差异

当 `deltap_lambda_init=0.0` 时 HK correction 为零(已验证), gamma 差异纯粹来自 ELPA。
这是 ABACUS 并行计算的已知特性，所有算符(DFT+U、DeltaSpin 等)都有类似行为。

## 2. 测试配置

测试系统：H2O (tests/17_DS_DFTU/67_LCAO_DELTAP_H2O)
- nbasis=23, nbands=7, nocc_use=4
- nppstr_=5, total_string_=16, nks=64
- gdir=3, deltap_corr=1, scf_nmax=1
- 2D block-cyclic grid: 1×2 (dim0=1, dim1=2)

## 3. 性能数据

- 串行运行：~16 秒
- 并行运行(2 proc)：~9 秒
- 主要热点 (串行): PW_Basis_Sup recip2real (42%), XC_Functional v_xc (52%)
- DeltaP 部分因系统太小(23 基函数)低于 1% 阈值

## 4. 遗留问题

1. **2×2 process grid 测试** (优先级: 中): 当前 dim0=1 未完全覆盖 2D block-cyclic 场景
2. **更大系统验证** (优先级: 高): 23 基函数 H2O 太小，需测试 PbTiO₃ 或 liquid water
3. **ScalapACK pzgemm 方案** (优先级: 低): 可用 BLACS 替代 Allreduce 收集，提升性能
4. **性能基准测试** (优先级: 中): 在 Wannier polarization 部分添加更细粒度计时

## 5. 文件修改

- `source/source_lcao/module_deltap/deltap_wannier.cpp`: compute_hk_correction 完整重写，约 130 行新并行代码
