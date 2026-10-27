# PW allocation FFT 缓冲裁剪与实空间观测（S5.7）

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_pw/module_pwdft/op_pw_alloc.cpp`
- 前置文档：
  `docs/superpowers/specs/2026-10-27-allocation-band-batching.md`

## Test plan

1. 删除局域势 FFT 每 band 对 `work/porter` 的整数组清零和输入拷贝，验证
   数值逐位不变、Fe2/LiCl 仍收敛。
2. 观测路径只对互补波函数做一次 `recip_to_real`，再在实空间对每个约束权重
   积分；用同一电子态上的旧 FFT 核交叉检查全局读数。
3. 在 Fe2-bcc 与 LiCl 上记录 hPsi、墙钟、约束残差和总能。
4. 跑 allocation 目标测试与完整 constraint 回归。

## Test setup

- 构建：
  `cmake --build build --target abacus_basic_para MODULE_ESTATE_constraint_alloc_operator MODULE_ESTATE_constraint_alloc_observe -j 8`
- MPI 4 rank；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- Fe2：`2x2x2` k 点、`kpar=2`、`nspin=2`、`nbands=27`、
  `constraint_alloc_target_l=2`、目标 `-3.8/+3.8 μB`。
- LiCl：Γ 点、`nspin=1`、`nbands=8`、192³ FFT 网格、电荷约束、`mu=0`。
- H2O 观测交叉检查：4 MPI 改为 2 MPI，`scf_nmax=1`，同一波函数上比较
  `qc_real` 与 `qc_recip` 的全局值。
- 运行目录：
  `/tmp/fe2_alloc_ffttrim.UFYwYg`（仅缓冲裁剪）、
  `/tmp/fe2_alloc_obsreal.kc6kit`（最终代码）、
  `/tmp/licl_alloc_fftobs.vxNyQ4`（最终代码）、
  `/tmp/h2o_obsdbg2.osyCEu`（观测交叉检查）。

## Implementation

### 1. 删除 FFT 前的冗余清零与拷贝

`PW_Basis_K::recip2real` 只读取前 `npw` 个倒空间系数，并完整覆盖 `nrxx`
个实空间输出；`PW_Basis_K::real2recip` 也完整写出前 `npw` 个系数。因此
`apply_local_potential_reuse()` 现在直接把 `input` 送入 FFT、把
`output` 直接作为逆变换目标，不再分配或清零第二块 `nmaxgr` 数组。

`build_target_products()` 也改为所有目标向量共用一个 `porter`，避免每次
lambda 变化重复分配 R 块大数组。

### 2. 观测路径的实空间积分

对每个 band 先形成倒空间互补态
\(Q_c|\psi\rangle=|\psi\rangle-\sum_a|\phi_a\rangle\langle\phi_a|\psi\rangle\)，
只做一次 `recip_to_real`，然后对每个约束计算

```text
q_c(alpha) = (1/nxyz) * sum_ir w_alpha(ir) * |Q_c psi(ir)|^2
```

这替代了原先“每个约束一次 recip_to_real + real_to_recip”的路径。全局交叉
检查得到 `real=0.842349`、`recip=0.842349`，差 `6.8e-14`（第二个 k 点为
`6.95e-14`），确认积分口径一致。

## Results

### Fe2-bcc

| 版本 | 电子迭代 | hPsi 次数 | hPsi 总时间/s | 单次 hPsi/ms | 墙钟/s |
|---|---:|---:|---:|---:|---:|
| S5.6 最终 | 85 | 3089 | 40.55–42.73 | 13.13–13.83 | 56–61 |
| 仅删除 FFT 冗余缓冲 | 85 | 3089 | 41.73 | 13.51 | 58.34 |
| 最终：缓冲裁剪 + 实空间观测 | 91 | 3342 | 42.47 | 12.71 | 56.45 |

最终约束与能量：

```text
CONSTRAINT_AUDIT c[0] kind=spin q=-3.799950168 t=-3.8 mu=-0.01508843596 res=4.983169514e-05
CONSTRAINT_AUDIT c[1] kind=spin q= 3.799956801 t= 3.8 mu= 0.01447817507 res=-4.319904167e-05
FINAL_ETOT_IS -6.3706101512221821e+03 eV
```

相对 S5.6，电子迭代 `85→91`，但单次 hPsi 从 `13.13–13.83 ms` 降到
`12.71 ms`，墙钟从 `56–61 s` 降到 `56.45 s`。迭代分支变化来自观测求和
次序的浮点差，总能差 `2.2e-8 eV`，约束差小于 `1e-5 μB`。

### LiCl

| 版本 | 电子迭代 | hPsi 次数 | hPsi 总时间/s | 墙钟/s | q/e |
|---|---:|---:|---:|---:|---:|
| 本次前运行 | 19 | 809 | 111.93 | 202.36 | 2.947887372 |
| 最终代码 | 19 | 809 | 92.21 | 160 | 2.947887372 |

最终总能 `-643.97565238003517 eV`、`dsum_n=-2.18847163e-12`，与修改前
逐位一致。LiCl 的收益来自观测路径减少的约束 FFT 和更低的缓冲带宽压力。

### 回归

- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS。
- allocation operator/observe：2/2 PASS。
- PW allocation MPI 1/2/4 rank：3/3 PASS。

## Analysis

- 删除 FFT 前的整数组清零/拷贝是低风险收益：不改变 FFT 调用次数，但显著
  降低大网格内存带宽；LiCl 的 192³ 网格对此最敏感。
- 实空间观测把每个 band 的约束 FFT 从 `nconstraint` 对降为 1 次正变换，
  并保持全局数值等价性；Fe2 的轨迹分支变化属于舍入误差，不是物理口径变化。
- Fe2 的墙钟仍由 91 次电子迭代主导；per-band 常数优化已接近边际。下一步
  应转向子空间旋转/warm start，而不是继续压缩单个 FFT 的实现常数。

## Next steps

1. 按约束预存 `W_i phi` 与 `Phi^dagger W_i phi`，供 inner/ALM 频繁换
   lambda 时线性组装。
2. 研究复用 DeltaSpin 的 `nbands x nbands` 子空间旋转和 warm start，
   压缩 Fe2 的电子迭代数。
3. 在 `nbands=100+` 体系复测分块大小、内存峰值与本轮的 FFT 缓冲裁剪收益。
