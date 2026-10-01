# PW 基组 DeltaSpin 单精度（float）支持 —— 进展总结

## 目标

在 `accel` 分支上完成 PW 基组 DeltaSpin 的单精度支持：模板化 DeltaSpin/OnsiteProjector/
Onsite_Proj_tools/DFT+U，使 `precision single` 下非共线（nspin=4）DeltaSpin 算例能收敛，
且与双精度结果误差在单精度容差内。

## 已完成的模板化工作（已提交）

- `SpinConstrain<TK>` + `SpinScalar<TK>` 模板化，逐 TU 显式实例化 double。
- `OnsiteProjector` / `Onsite_Proj_tools` 上游数据（vkb/tab/wg/ekb/gcar/sk/ylm/gemm 系数）float 化。
- `OnsiteProj`（`op_pw_proj.cpp`）算子模板化（`T` / `GetTypeReal<T>` / `SpinConstrain<T>`）。
- 能量/输出/混合/stress 的 `SpinConstrain` 调用点按精度分发。
- LCAO 专用函数 `if constexpr` 桩 + `SpinConstrain<std::complex<float>>` 实例化。

## 本轮定位的根因

单精度 DeltaSpin lambda 循环发散（`ETOT=-6660 eV` vs 双精度 `-6368.95 eV`，`Mi` 反号、`λ_z` 巨大）。
排查过程（按顺序排除）：

1. 在站投影 `vkb/ylm/vq/sk/pref` —— float 与 double 完全一致，正确。
2. 子空间旋转 `diag_subspace_psi`（LAPACK `chegvd`）—— 一致，正确。
3. 纯非共线 SCF（`sc_mag_switch=0`）单精度收敛且与双精度一致 → 基础哈密顿量 + Davidson 本身正确。
4. 三个求解器 `dav_subspace` / `cg` / `dav` 均发散 → 非方法特异的。
5. `hegvx_op`(chegvx) vs `hegvd_op`(chegvd)、`need_subspace`、`pw_solve=false` 均非根因。

**最终根因**：`OnsiteProj<OperatorPW<std::complex<float>>>` 对
`add_onsite_proj` / `update_becp` / `cal_ps_delta_spin` / `cal_ps_dftu` 提供了**空特化**（no-op），
导致单精度下 λ·σ（以及 DFT+U）哈密顿量修正**完全没有施加**。全空间 Davidson 对角化的因此是
无约束的 DFT 哈密顿量，把约束“抹掉”，spin 翻转 → 发散。双精度走的是通用模板，所以正常。

## 本轮修复（本次提交）

1. **移除 `OnsiteProj<complex<float>>` 的空特化**（CPU+GPU 共 8 个），使通用模板对 float 也生效。
   - 文件：`source/source_pw/module_pwdft/op_pw_proj.cpp`
   - 连带修复：DFT+U 的 `cal_ps_dftu` 对 float 也不再是 no-op（编译通过，但未单独验证正确性）。

2. **修复 Davidson 广播的 MPI 类型**（float 用 `MPI_C_FLOAT_COMPLEX`/`MPI_FLOAT`，
   原来硬编码 `MPI_DOUBLE_COMPLEX`/`MPI_DOUBLE`，会导致 float 缓冲区越界读写）。
   - 文件：`source/source_hsolver/diago_dav_subspace.cpp`、`source/source_hsolver/diago_david.cpp`

## 验证结果（非共线 90° BCC-Fe，nspin=4，dav_subspace，-np 4）

| | ETOT (eV) | TMAGX / TMAGZ | Mi（Fe1 / Fe2） |
|---|---|---|---|
| 双精度 | -6368.95441 | 2.41 / -2.41 | 达目标 |
| 单精度（修复前） | -6660.03（发散） | — | Fe1 z=+0.42（反号） |
| 单精度（修复后） | -6368.95475 | 2.41 / -2.41 | Fe1≈[0,0,-2.000]，Fe2≈[2.000,0,0] |

单精度与双精度总能量误差 ≈ 3.4e-4 eV，在单精度容差内。

## 待办

- 误差进一步缩小（见下：从算法与变量误差大小角度的分析）。
- DFT+U `cal_ps_dftu` 单精度正确性需单独验证（本轮仅启用，未测试）。
- 单精度下的 `scf_thr`/`sc_thr` 等收敛判据是否需要放宽尚未评估。
