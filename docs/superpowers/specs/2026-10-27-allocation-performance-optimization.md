# PW allocation 性能优化（S5.5）

- 日期：2026-10-27
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_pw/module_pwdft/op_pw_alloc.cpp`
- 关联文档：`docs/superpowers/specs/2026-10-27-allocation-critical-review.md`、
  `docs/superpowers/specs/2026-10-27-fe2-licl-comparison.md`

## Test plan

1. 修正观测路径的 MPI 归约错误，确认 Fe2 首步约束读数恢复，并最终收敛回
   `±3.800 μB`。
2. 验证零乘子快路径不改变 LiCl 的密度、总能和局域电荷，只移除无意义的
   allocation 成本。
3. 验证按 lambda 缓存的局域势不会改变非零约束轨迹，并量化其收益。
4. 重跑 allocation 单测、PW MPI 单测和完整 constraint 回归。

## Test setup

- 构建：
  `cmake --build build --target abacus_basic_para MODULE_ESTATE_constraint_alloc_observe MODULE_ESTATE_constraint_alloc_operator -j 8`
- 运行：4 个 MPI rank；`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`。
- Fe2-bcc：`tests/constraint_s5_parity/fe2_bcc_alloc` 的 `2x2x2` k 网格、
  `kpar=2`、`nspin=2`、`nbands=27`、目标 `-3.8/+3.8 μB`、trust-region。
- LiCl：`tests/constraint_s5_parity/licl_alloc` 的 Γ 点、`nspin=1`、
  `nelec=10`、目标电荷增量 `0.0 e`。Becke 对照来自 `licl_free`。
- 测试目录复制到 `/tmp`，只把 `pseudo_dir/orbital_dir` 改为绝对路径；仓库
  输入未改动。

## Implementation

### 1. 零乘子早退

`AllocConstraintPW::act()` 在构造实空间势场之前检查 `mu_eff()`。若全部为
精确零，直接返回；随后仍保留对 `lambda=mu*sign` 全零的二次检查。这样
LiCl 的每次 hPsi 不再分配、清零和填充 `192^3` 的势场数组。

### 2. lambda 势场缓存

把原有 `build_mu_potential()` 拆成“构造 lambda”和“由 lambda 构造势场”两步。
`AllocPwCache` 新增 `LocalPotential`，以 weight-grid 指针和完整 lambda 向量
为键。固定 mu 的 SCF 内层中，每个 Hamiltonian 调用复用同一势场；只有外层
mu 更新时才重建。

### 3. 观测路径批量化

原实现对每个约束 alpha 重复构造 `vphi/m_raw/orthogonal`，再逐 band 逐 alpha
计算 trace。新实现对每个 k 只构造一次 `phi_orth`，每个 band 只计算一次

```text
q_alpha = sum_{a in Q_alpha} |<phi_a|psi>|^2
        + Re <Q_c psi|W_alpha|Q_c psi>
```

并把所有 alpha 合并成一次池归约。

## Results

### Fe2-bcc

| 版本 | 电子迭代 | hPsi 次数 | hPsi 总时间/s | 单次 hPsi/ms | 墙钟/s |
|---|---:|---:|---:|---:|---:|
| 优化前 | 81 | 2723 | 56.68 | 20.82 | 84 |
| 首轮优化 | 84 | 2924 | 44.42 | 15.19 | 59 |
| 势场缓存版 | 84 | 2924 | 47.35 | 16.19 | 63 |

两次优化后运行的墙钟分别为 59 s 和 63 s，差异属于当前机器的运行噪声；
相对基线均下降约 25%–30%。hPsi 次数增加 7.4% 是因为电子迭代从 81 变为
84，但单次成本下降 22%–27%。

最终数值与基线一致到能量/约束误差量级：

```text
CONSTRAINT_AUDIT c[0] kind=spin q=-3.799952058 t=-3.8 mu=-0.01508849778 res=4.794166203e-05
CONSTRAINT_AUDIT c[1] kind=spin q= 3.799963386 t= 3.8 mu= 0.0144781842 res=-3.661355794e-05
FINAL_ETOT_IS -6.3706101511917359e+03 eV
```

与 DeltaSpin 的 `3.51 s` 相比仍约慢 `17–18x`；单次 hPsi 仍约为
DeltaSpin 的 `2.2–2.3x`。因此 Fe2 的 `<1.5x`、`<3x` 门槛尚未达到。

### LiCl

| 版本 | 电子迭代 | hPsi 次数 | hPsi 总时间/s | 单次 hPsi/ms | 墙钟/s |
|---|---:|---:|---:|---:|---:|
| Becke | 19 | 809 | 99.23 | 122.66 | 169 |
| allocation 优化前 | 19 | 809 | 291.78 | 360.66 | 385 |
| 零 lambda 快路径 | 19 | 809 | 111.93 | 138.35 | 196 |
| 提前 mu=0 返回 + 势场缓存 | 19 | 809 | 102.59 | 126.80 | 183 |

当前 allocation 的墙钟为 Becke 的 `1.083x`，hPsi 总时间为 `1.034x`，已满足
`<1.3x` 门槛。局域电荷和总能与优化前逐位一致：

```text
CONSTRAINT_AUDIT c[0] kind=charge q=2.947887372 t=2.947887372 mu=0 res=0
FINAL_ETOT_IS -6.4397565238003517e+02 eV
```

### 回归

- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS。
- allocation operator/observe 单测：2/2 PASS。
- PW MPI 单测在 1/2/4 rank 下通过。

## Analysis

### 数值 bug：全局量与 rank-local 量被重复归约

首轮观测改写后，Fe2 首步 `q` 约为 `-7.61/+7.58 μB`，200 步未收敛。
根因是 `build_raw_overlap()` 已经对 `<phi_a|psi>` 做池归约，因此
`|<phi_a|psi>|^2` 已是全局量；随后又把包含该项的整个 `q_alpha` 归约一次，
在大小为 2 的 k-pool 中把 `Q_i` 贡献重复计算了两次。

修复后把贡献拆成：

```text
q_alpha   <- 已归约的 Q_i 贡献
qc_alpha  <- rank-local 的 Q_c W_i Q_c 贡献，单独归约
q_alpha  += qc_alpha
```

Fe2 首步恢复到正常的约 `-3.9 μB`，最终回到 `±3.800 μB`。这说明该问题不是
物理分支，而是归约作用域错误。

### 收益来源

- LiCl 的主要收益来自“在构造势场之前识别 mu=0”。原实现每次 hPsi 都处理
  `192^3` 网格，即使所有冻结通道系数为零。
- Fe2 的收益主要来自观测/目标乘积复用和单次 hPsi 常数下降；势场缓存对
  约 `30^3` 网格的 Fe2 没有可测收益，但对大网格、非零 lambda 的体系可避免
  重复分配和填充。
- 目标空间、`S^{-1/2}`、`phi_orth`、`Vphi` 和 R×R 矩阵仍按几何/lambda
  缓存；不能缓存的是对每个新 trial wavefunction 的 `V|psi>` 与目标重叠。

## Next steps

1. 已在 S5.6 完成：raw overlap 和 `<Vphi|psi>` 按 8-band block 合并归约，
   并复用 FFT 临时 workspace；详见
   `docs/superpowers/specs/2026-10-27-allocation-band-batching.md`。
2. 按约束预存 `W_i phi` 与 `Phi^dagger W_i Phi`，换 lambda 时线性组装，
   避免重建 `Vphi` 和 R×R 矩阵。
3. 评估复用 DeltaSpin 的 `nbands x nbands` 子空间旋转；这是把 Fe2 的
   “84 次电子迭代”压下来的关键，单独优化 hPsi 常数无法达到 `<3x`。
4. 以本轮为新的性能基线做正式 go/no-go：若子空间/warm-start 后 Fe2 仍
   高于 DeltaSpin 的 `5x`，停止 allocation 生产约束路径，只保留只读诊断。
