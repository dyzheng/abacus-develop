# PW allocation 局域势并入 Veff

- 日期：2026-10-28
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_pw/module_pwdft/{hamilt_pw.cpp,op_pw_alloc.h,op_pw_alloc.cpp,op_pw_veff.h,op_pw_veff.cpp}`、
  `source/source_estate/module_constraint/{alloc_pw.h,alloc_pw.cpp}`、
  `source/source_estate/module_constraint/test/alloc_pw_test.cpp`、
  `source/source_io/module_parameter/{input_parameter.h,read_input_item_other.cpp}`、
  `docs/superpowers/specs/deltap-development-log.md`
- 前置文档：
  `docs/superpowers/specs/2026-10-28-allocation-subspace-warm-start.md`、
  `docs/superpowers/specs/2026-10-27-allocation-critical-review.md`

## Test plan

1. 验证把 \(V_{\rm alloc}=\sum_i\lambda_i w_i\) 并入现有 `Veff` 后，allocation
   算符只补投影修正项，代数上仍等于原来的
   \(V-PV-VP+PVP+\sum_i\lambda_iQ_i\)。
2. 在 Fe2-bcc 上做严格 A/B：同一二进制、同一输入，只切换新增的
   `constraint_alloc_fold_potential`。比较迭代数、hPsi 次数/时间、FFT 调用数、
   墙钟、最终磁矩、约束残差和总能。
3. 确认关闭折叠时与上一轮 S5.8 结果逐位一致，排除共享 cache 重构造成的回归。
4. 用 LiCl 的 `mu=0` 路径确认新接口不误触发折叠，并跑完整 constraint 单测。

## Test setup

- 4 MPI rank；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。
- Fe2-bcc：`2x2x2` k、`kpar=2`、`nspin=2`、`nbands=27`、
  `ecutwfc=20 Ry`、`scf_thr=1e-8`、目标 `-3.8/+3.8 μB`、
  `constraint_alloc_subspace_rotation=1`、`constraint_alloc_subspace_iterations=1`。
- LiCl：Γ 点、`nspin=1`、`nbands=8`、`ecutwfc=100 Ry`、`ecutrho=400 Ry`、
  `constraint_thr=1e-4`；原目标使 `mu=0`。
- 新开关 `constraint_alloc_fold_potential` 默认 false；A/B 只把 on 组的输入
  追加为 1。

## Results

### Fe2 A/B（4 rank）

| 版本 | 迭代 | hPsi 次数 | hPsi/s | veff_pw/s | recip2real 次数 | real2recip 次数 | 墙钟/s | ETOT/eV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 折叠关闭 | 82 | 1983 | 28.83 | 11.23 | 72609 | 58343 | 38.61 | -6370.6101511972 |
| 折叠开启 | 82 | 1985 | 18.36–18.70 | 11.55–11.70 | 46730 | 32464 | 28.84–29.73 | -6370.6101511999 |

- 单次 hPsi 从 `14.54 ms` 降到 `9.25–9.42 ms`（约 -35%），总墙钟从
  `38.61 s` 降到 `28.84–29.73 s`（约 -23% 至 -25%）。
- 折叠关闭组与上一轮 S5.8 的 ETOT、磁矩、残差逐位一致。
- 折叠开启组：`q=-3.800014112/+3.799903104 μB`，最大残差
  `9.69e-5 μB`；相对关闭组磁矩最大差 `4.4e-7 μB`，ETOT 差
  `2.7e-9 eV`，来自不同的浮点求和次序和 Davidson 分支。

### 单测与无操作路径

- `alloc_pw` 新增 `include_vpsi=false` 等价性测试：折叠结果加回
  \(V|\psi\rangle\) 后与完整 reduced 结果的最大差 `<1e-13`。
- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS。
- LiCl 的 `mu=0` 路径未创建势、未进入折叠分支：19 次迭代、809 次 hPsi、
  hPsi `98.24 s`、墙钟 `163.01 s`，`q=2.947887372 e`、
  `ETOT=-643.97565238003517 eV`、`dsum_n=-2.188e-12`，与 S5.8 一致。
- 另启动了一个 LiCl `target delta=0.1 e` 的非零电荷压力用例；它在第一个
  outer step 得到 `mu=-0.05`，但后续 400 Ry 网格的 SCF 收敛过慢，本轮终止，
  因此不把它当作性能数据。

## Analysis

- **收益来源是消除重复 FFT，而不是减少 hPsi 次数。** `Veff` 本来就把每个
  band 变换到实空间一次；把 allocation 势同时加在同一个实空间乘法中，消除了
  allocation 算符自己的 `recip_to_real + real_to_recip`。因此 `veff_pw`
  自身略慢（多一次 `O(nrxx)` 数组相加），但总 hPsi 明显下降。
- **内存副作用可控。** `Veff` 只额外保存一块自旋相关 `veff_with_alloc`
  （`nrxx` 个 `Real`），每次 `act()` 重建一次，不修改共享的
  `Potential::veff_smooth`，因此不会污染力、应力或后续密度混合。
- **正确性边界清晰。** 只有 double/CPU、`npol=1`、allocation 已激活且权重网格
  与 Veff 网格一致时才折叠；其余情况保留原路径。`__DSP` 明确回退。折叠时
  allocation 的 reduced kernel 不再接收 `V|psi>`，避免重复计入。
- **默认仍关闭。** 单例结果很好，但这是一项全局 Hamiltonian 路径改动；在
  force/stress、USPP、不同 spin/k-point 布局和大 `nbands` 验证前，不应把
  A/B 开关直接改成默认。

## Next steps

1. 在 force/stress 和至少一个 `mu!=0` 的电荷约束上复测，确认临时势完全不泄漏。
2. 在 `nbands=100+` 上测 `veff_with_alloc` 的额外内存和缓存局部性。
3. 下一项结构优化仍是预存 \(W_i\phi\) 与
   \(\Phi^\dagger W_i\Phi\)，降低 lambda 变化时的 target-product 重建成本。
4. 真正压缩外层迭代需要构造 \(\langle\psi_m|A_i|\psi_n\rangle\) 并做受约束
   子空间求解；仅继续削减单次 hPsi 常数已接近收益上限。
