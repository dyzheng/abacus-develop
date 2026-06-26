# nspin=4 LCAO Subspace 加速开发记录

## 一、背景与动机

DeltaSpin 模块的 LCAO 基组 subspace 加速此前仅支持 nspin=2（共线磁化），nspin=4（非共线/自旋轨道耦合）被三处硬编码 `nspin_ == 2` 拦截，永远退回全空间对角化路径。这导致 nspin=4 的 DeltaSpin 计算无法受益于子空间加速，在高吞吐场景下性能受限。

## 二、问题分解

要让 nspin=4 LCAO subspace 工作，需要解决三类问题：

### 2.1 拦截消除（3 处）

| 拦截点 | 文件 | 行号 | 原条件 | 修改后条件 |
|--------|------|------|--------|-----------|
| 加速开关 | `spin_constrain.h` | 371 | `nspin_ == 2 && mode != "off" && thr > 0` | `mode != "off" && thr > 0` |
| lambda 循环 | `lambda_loop.cpp` | 737 | `basis=="lcao" && nspin_==2 && mode!="off" && ...` | `basis=="lcao" && mode!="off" && ...` |
| 缓存构建 | `cal_mw_from_lambda.cpp` | 608 | `i_step==-2 && accel_enabled && nspin_==2` | `i_step==-2 && accel_enabled` |

### 2.2 `cal_PI_sub` 对 complex-type DeltaSpin 算子的支持

原有 `cal_PI_sub` 方法仅在 `DeltaSpin<OperatorLCAO<complex<double>, double>>` 上实现。nspin=4 使用 `DeltaSpin<OperatorLCAO<complex<double>, complex<double>>>`，缺少此方法。

**解决方案**：`cal_PI_sub` 的模板实现已在 `dspin_lcao.h/cpp` 中定义，complex-type 实例化自动获得此方法。关键改动在于所有调用点需根据 `nspin` 选择正确的 `dynamic_cast` 目标类型。

### 2.3 `calculate_PI_sub_from_hr` 对 complex HContainer 的支持

原有函数签名仅接受 `HContainer<double>*`，但 nspin=4 的 `pre_hr` 类型为 `HContainer<complex<double>>*`。

**解决方案**：新增重载函数，接受 `HContainer<complex<double>>*`，实现逻辑与 double 版本完全对称（folding_HR + ScaLAPACK gemm），包含完整的 MPI + 混合精度 (fp32) 支持。

### 2.4 `calculate_delta_hcc_lcao` 的 npol=2 Pauli 系数修正

原有 npol=2 分支仅使用 `coeff0(lambda_z)`，完全忽略 spin-flip 项（lambda_x ± i*lambda_y）。

**解决方案**：分布式版本（ParaV 重载）正确实现 2×2 Pauli 行变换：

- 对每个局部行，通过 `local2global_row` 和 `global2local_row` 识别自旋上/下对应行
- 自旋上行：`h += coeff0 * P_up + coeff2 * P_dn`
- 自旋下行：`h += coeff1 * P_up + coeff3 * P_dn`
- 当配对行不在同一进程时，仅施加对角项（安全退化，不引入错误）

聚集版本（nbands×nbands 全矩阵）添加注释说明其局限性：由于缺乏显式自旋块结构，无法直接应用 Pauli 变换，仅保留 `coeff0` 贡献作为安全回退。该版本仅由 `DiagonalizationEngine` 使用，nspin=4 场景不应通过此路径运行。

### 2.5 `diagonalization_engine.cpp` 的 `dynamic_cast` 更新

`SubspaceDiagonalizer::build_subspace` 和 `FirstOrderResponseEngine::build_subspace` 中的 `cal_PI_sub` 调用原硬编码为 double-type 算子。修改为根据 `sc_.get_nspin()` 分别 `dynamic_cast` 到正确的算子类型。

### 2.6 一阶响应模式对 nspin=4 的限制

一阶响应模式 `first_order` 使用 `spin_sign * delta_lambda_z * P_I_sub_diag` 公式，仅包含 z 分量偏移。对 nspin=4 此公式不完整（缺少 x/y 方向贡献）。因此将条件改为 `first_order && nspin_ == 2`，nspin=4 自动回落到 subspace 对角化模式。

## 三、修改文件清单

| # | 文件 | 修改类型 | 描述 |
|---|------|----------|------|
| 1 | `density_matrix.cpp` | Bug Fix | `func_xyz_to_updown` rho_y 符号修正 |
| 2 | `elpa_new.cpp` | Bug Fix | 添加缺失的 `blacs_context` 设置 |
| 3 | `gint_common.cpp` | Bug Fix | `clx_j` 符号修正 + `std::conj()` Hermiticity 修复 |
| 4 | `dftu_force_stress.hpp` | Bug Fix | 移除 nspin=4 下多余的 `VU/=2`；`force*=2` 仅在 nspin≠4 时生效 |
| 5 | `dftu_lcao.cpp` | Bug Fix | `transfer_vu` Pauli→spinor 符号修正 |
| 6 | `dspin_force_stress.hpp` | Bug Fix | `force*=2` 仅在 nspin≠4 时生效 |
| 7 | `dspin_lcao.cpp` | Feature + Bug Fix | `cal_coeff_lambda` sigma_y 符号修正 |
| 8 | `spin_constrain.h` | Feature | 新增 `calculate_PI_sub_from_hr` complex 重载声明；移除 `nspin_==2` 拦截 |
| 9 | `lcao_subspace.cpp` | Feature | 实现 complex `calculate_PI_sub_from_hr`；修复 `calculate_delta_hcc_lcao` npol=2 Pauli 系数 |
| 10 | `cal_mw_from_lambda.cpp` | Feature | 移除缓存构建 `nspin_==2` 拦截；添加 nspin=4 缓存构建分支；限制一阶模式为 nspin=2 |
| 11 | `lambda_loop.cpp` | Feature | 移除加速判定 `nspin_==2` 拦截 |
| 12 | `diagonalization_engine.cpp` | Feature | `cal_PI_sub` 调用根据 nspin 选择正确的算子类型 |
| 13 | `scf_angle_spin4/result.ref` | Test | 更新参考值 |
| 14 | `scf_u_spin4/result.ref` | Test | 更新参考值 |

## 四、支持矩阵

| 特性 | nspin=2 LCAO | nspin=4 LCAO | nspin=4 PW |
|------|-------------|-------------|-----------|
| 全空间对角化 | 支持 | 支持 | 支持 |
| subspace 对角化 (`sc_strategy=fast/normal`) | 支持 | **支持（本次新增）** | 支持 |
| 一阶响应 (`sc_acceleration_mode=first_order`) | 支持 | 自动回落到 subspace | N/A |
| Pauli 系数修正 | N/A（仅 z 分量） | **完整 2×2 Pauli 矩阵** | 已有 |
| `cal_PI_sub`（全矩阵汇聚版本） | 支持 | 支持（退化：仅 coeff0） | N/A |
| `calculate_PI_sub_from_hr`（分布式版本） | 支持 | **支持（本次新增）** | N/A |

## 五、已知限制

1. **一阶响应模式不支持 nspin=4**：`PI_sub_diag` 仅存储 z 分量的对角元，无法直接推广到含 x/y 方向的一阶偏移。nspin=4 将自动回落到 subspace 对角化模式。
2. **DiagonalizationEngine 聚集路径的退化**：`calculate_delta_hcc_lcao` 的全矩阵版本无法正确应用完整 Pauli 变换（因缺乏显式自旋块结构），仅保留 `coeff0` 贡献。生产环境应使用分布式路径。
3. **测试参考值待更新**：`tests/17_DS_DFTU/` 下所有 LCAO S4 测试目录的 `result.ref` 可能需要在修复后重新运行测试更新。
