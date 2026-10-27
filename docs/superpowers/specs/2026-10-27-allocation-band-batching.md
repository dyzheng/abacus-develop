# PW allocation band 批处理与 FFT workspace 复用（S5.6）

- 日期：2026-10-27
- 分支：`feat/unified-local-moment-allocation`
- 修改文件：`source/source_pw/module_pwdft/op_pw_alloc.cpp`、
  `source/source_estate/module_constraint/alloc_pw.{h,cpp}`
- 前置文档：`docs/superpowers/specs/2026-10-27-allocation-performance-optimization.md`

## Test plan

1. 把 Hamiltonian 路径中每 band 的 `<phi|psi>` 与 `<Vphi|psi>` 两处池归约
   合并，验证 Fe2 多 k、`kpar=2` 下数值不漂移。
2. 复用局域势 FFT 的 work/porter 缓冲，避免每 band 重新分配。
3. 用 8-band 分块限制一次缓存的 `V|psi>` 数量，避免大 `nbands` 的内存放大。
4. 合并 `P V P` 与 owned-\(Q_i\) 的系数，删除每 band 的 4 个 `nbasis`
   临时向量。
5. 跑 allocation 单测、PW MPI 单测和完整 constraint 回归。

## Test setup

- 构建：
  `cmake --build build --target abacus_basic_para MODULE_ESTATE_constraint_alloc_operator MODULE_ESTATE_constraint_alloc_observe -j 8`
- Fe2：`2x2x2` k 网格、`kpar=2`、`nspin=2`、`nbands=27`、
  目标 `-3.8/+3.8 μB`、trust-region；4 MPI rank，OpenMP/OpenBLAS/MKL
  均固定单线程。
- 输入从 `tests/constraint_s5_parity/fe2_bcc_alloc` 复制到 `/tmp`，仓库输入
  未改动。
- 运行目录：`/tmp/fe2_alloc_breduce.mJZXY1`、
  `/tmp/fe2_alloc_combine.qomhzp`、`/tmp/fe2_alloc_chunk.hiNW0j`、
  `/tmp/fe2_alloc_ablocal.jIS3sK`。

## Implementation

### 1. 批量归约

新增 `apply_pw_alloc_delta_h_orthogonal_reduced()`。它接收已经池归约的
`raw_overlap` 与 `braw_overlap`，只做每 band 的 \(R\) 维旋转和低秩代数。
`act()` 先对所有 band 计算局域重叠，再按 8-band 分块打包成一个
`reduce_pool()`。

每 block 的归约次数：

```text
S5.5 及之前：2 * nbands = 54 次/act（Fe2 nbands=27）
S5.6：ceil(nbands / 8) = 4 次/act
```

### 2. FFT workspace 复用

新增 `apply_local_potential_reuse()`。`act()` 每次只分配一对
`nmaxgr` 缓冲，所有 band 和各 block 共用；每个 band 仍只做必要的复 FFT，
但不再重复分配和初始化大数组。

### 3. 临时向量合并

原每 band 分配 `pvpsi/v_ppsi/pvp/qlambda/delta` 五个 `nbasis` 向量。现在：

```text
delta = V|psi>
delta -= sum_a phi_orth[a] <phi_a|V|psi>
delta -= sum_a Vphi_orth[a] <phi_a|psi>
delta += sum_b phi_orth[b] (M_orth[b,:] o + owned lambda_b o_b)
```

`P V P` 与 owned-\(Q_i\) 共用同一个 `phi_orth[b]`，因此先合并系数再更新
向量，只保留 `delta` 一个 `nbasis` 缓冲。

## Results

### Fe2-bcc

| 版本 | 电子迭代 | hPsi 次数 | hPsi 总时间/s | 单次 hPsi/ms | 墙钟/s |
|---|---:|---:|---:|---:|---:|
| S5.5 | 84 | 2924 | 44.42–47.35 | 15.19–16.19 | 59–63 |
| 一次归约（未分块） | 84 | 2924 | 39.19 | 13.40 | 55 |
| 最终：8-band 分块 + 局部 scratch | 85 | 3089 | 40.55–42.73 | 13.13–13.83 | 56–61 |

最终相对 S5.5 的墙钟下降约 3%–11%，单次 hPsi 下降约 14%–19%；相对最初
基线 `84 s / 20.82 ms`，墙钟下降 33%，单次 hPsi 下降 37%，相对 DeltaSpin
的单次成本由 `2.95x` 降到 `1.86x`。

上述范围包含机器频率/负载波动造成的同幅漂移；不应把单个计时点当作稳定
性能数字。8-band 分块加局部 scratch 与 persistent scratch 的同机 A/B 中，
后者 hPsi 为 `46.21–46.82 s`，局部 scratch 为 `42.73 s`。因此最终实现选择
每次 `act()` 分配一次 work/porter、在全部 band 和 block 间复用，而不把
scratch 放进跨调用的持久缓存。

最终约束和能量：

```text
CONSTRAINT_AUDIT c[0] kind=spin q=-3.799950937 t=-3.8 mu=-0.01508851235 res=4.906256453e-05
CONSTRAINT_AUDIT c[1] kind=spin q= 3.799961892 t= 3.8 mu= 0.0144781873 res=-3.810814185e-05
FINAL_ETOT_IS -6.3706101512006135e+03 eV
```

不限内存的一次归约版本与 8-band 分块版本逐位一致。临时向量合并改变了浮点
求和次序，使 Fe2 轨迹由 84 步变为 85 步；最终能量差 `8.9e-9 eV`，约束差
约 `1e-6 μB`，属于可接受的浮点分支差异。

### LiCl

本轮未重跑。LiCl 的 `mu=0` 在构造势场前已返回，完全不会进入本轮修改的
`V|psi>`、目标重叠或低秩代数路径；S5.5 的 `183 s`、`q=2.947887372 e` 和
`FINAL_ETOT=-643.97565238003517 eV` 仍然有效。

### 回归

- `ctest -R MODULE_ESTATE_constraint`：23/23 PASS。
- allocation operator/observe：2/2 PASS。
- PW allocation MPI 单测在 1/2/4 rank：3/3 PASS。

## Analysis

- 归约批处理解决了小 \(R\)、多 band 场景中最明显的集合通信放大；Fe2 的
  hPsi 时间从 `44–47 s` 降到约 `40 s`。
- FFT workspace 复用降低了分配器压力，但没有改变每 band 仍要做一次额外
  \(V|\psi>\) 局域势的事实；FFT 仍是剩余成本主体。
- 分块上限为 8 个 band，避免 `nbands * npw` 的临时存储随 band 数线性膨胀；
  代价相对不限内存版本约 1.4%，可以接受。
- 合并 `P V P`/\(Q_i\) 后单次 hPsi 再降约 3%，但总墙钟被多出的一次电子
  迭代抵消；收益属于内存和常数级，不改变迭代数瓶颈。

## Next steps

1. 按约束预存 `W_i phi` 与 `Phi^dagger W_i Phi`，换 lambda 时线性组装。
2. 复用 DeltaSpin 的 `nbands x nbands` 子空间旋转和 warm start，压缩 Fe2 的
   85 次电子迭代；这是进入 DeltaSpin 同一量级的必要条件。
3. 在 `nbands=100+` 的体系上测量 8-band 分块的内存峰值和最佳 block 大小。
4. 若子空间/联合收敛后 Fe2 仍高于 DeltaSpin `5x`，按既有门槛停止生产路径。
