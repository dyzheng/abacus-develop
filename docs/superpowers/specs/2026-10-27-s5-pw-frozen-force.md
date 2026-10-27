# S5.1 PW 冻结目标位置力与多进程正交化验证

日期：2026-10-27

计划：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`，Task S5.1。

本轮把 PW allocation 的冻结目标位置力接入 `ESolver_KS_PW::cal_force()`：
从最终能带形成 `Q_c|psi>`，累计互补密度 `Q_c D Q_c`，再复用共享的
`constraint_force()` 权重导数核。实现对应设计 §6.2 的冻结近似，即只保留
`Q_c dW Q_c`，暂不包含 onsite projector 的几何导数 `dQ`。

## 1. 测试计划

1. 验证 `phi_orth = phi G` 在平面波分片和 pool 归约后仍正交归一。
2. 在 `OMP_NUM_THREADS=1` 下构建 `abacus_basic_para`，只以 MPI 多进程运行 PW 测试。
3. 对 H2O/O-d 固定 `fixed_mu` 构型，比较 `np=2` 与 `np=4` 的中心构型能量、
   约束力和 O-z 总力。
4. 以 `np=2` 做 O-z 三点中心差分，比较总力解析值与能量 FD。
5. 单独提取 `e_con` 做三点差分，判断冻结约束力项与约束能量导数的一致性。

## 2. 测试设置

- 分支：`feat/unified-local-moment-allocation`
- 构建：`build/abacus_basic_para`，CPU，OpenMPI
- 体系：H2O，O-d 目标（`constraint_alloc_target_l=2`），Becke 权重
- `nspin=2`、`nbands=8`、`ecutwfc=20 Ry`、`ecutrho=80 Ry`
- `constraint_alloc_switch=1`、`fixed_mu=0.01 Ry`、`scf_thr=1e-8`
- `cal_force=1`、`test_force=1`、15 Angstrom 立方盒
- 位移：O-z 中心 ±`0.005 Bohr`（±`0.002645886055 Angstrom`）
- MPI/线程环境：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### 2.1 实现

1. `alloc_pw.{h,cpp}` 抽出 `build_pw_alloc_orthogonal_phi()`，供接触发与
   Hamiltonian 共享同一个 `phi_orth = phi G`。
2. `compute_pw_alloc_complement_density()` 对每个占据 band 计算
   `Q_c|psi> = |psi> - sum_a |phi_orth_a><phi_orth_a|psi>`，变换到实空间后
   按 `occupation / omega` 累计每个自旋通道的 `rho_c`。
3. `ESolver_KS_PW::cal_force()` 在调用 `Forces::cal_force()` 前构造 `rho_c`，
   通过 `set_allocation_force_density()` 交给共享约束力核。
4. 当前力路径显式拒绝 `KPAR * bndpar > 1`：现有约束力核只在 pool 内归约，
   多 pool 会静默漏掉其他 k 点。
5. MPI 单测增加分片 `phi_orth` 正交性检查；测试数据使用当前
   `PwTargetSpace` 所支持的实投影 overlap。

## 3. 结果

### 3.1 构建与 MPI 单测

```text
cmake --build build --target MODULE_ESTATE_constraint_alloc_pw_mpi abacus_basic_para -j 14  PASS
ctest -R '^MODULE_ESTATE_constraint_alloc_pw_mpi_(2np|4np)$'                              PASS (2/2)
```

正交性检查：

| MPI ranks | `max_dev` |
|---:|---:|
| 2 | 1.554e-15 |
| 4 | 1.554e-15 |

### 3.2 `np=2` 与 `np=4` 中心构型

两进程中心能量为 `-442.15878837395684 eV`，四进程为
`-442.15878837390800 eV`，差 `4.9e-11 eV`。

| 量 | `np=2` | `np=4` | 差 |
|---|---:|---:|---:|
| `e_con` (Ry) | -0.005100062102 | -0.005100062102 | 0 |
| constraint force O-z (Ry/Bohr) | -0.0000048249 | -0.0000048249 | 0 |
| total force O-z (eV/A) | -5.6632940742 | -5.6632940745 | 3e-10 |
| `q` | -0.01000621017 | -0.0100062102 | <1e-12 |

### 3.3 三点位置 FD

`np=2` 三点总能：

| 构型 | 总能 (eV) | total force O-z (eV/A) |
|---|---:|---:|
| plus | -442.1436610734193 | -5.7577593587 |
| center | -442.15878837395684 | -5.6632940742 |
| minus | -442.17365438467897 | -5.5682479501 |

```text
F_FD        = -5.667914385141006 eV/A
F_analytic  = -5.6632940742 eV/A
|difference| =  0.004620310941005 eV/A
criterion   =  0.0129 eV/A                              PASS
E+ + E- - 2E0 = 2.612898e-4 eV
```

### 3.4 `e_con` 直接差分

| 构型 | `e_con` (Ry) | constraint force O-z (Ry/Bohr) |
|---|---:|---:|
| plus | -0.005099502584 | -0.0000048388 |
| center | -0.005100062102 | -0.0000048249 |
| minus | -0.005100591061 | -0.0000048024 |

```text
F_con(e_con FD) = -(e+ - e-)/(2 * 0.005 Bohr)
                = -1.088477e-4 Ry/Bohr
                = -2.798587e-3 eV/A
F_con(analytic) = -4.8249e-6 Ry/Bohr
                = -1.240532e-4 eV/A
|difference|    =  2.674534e-3 eV/A
e_con curvature = e+ + e- - 2e0 = 3.0559e-8 Ry
```

## 4. 分析

1. 总力三点 FD 通过 S5.1 闸门，2/4 进程中心构型的约束力和总力一致到浮点
   归约噪声。新增密度构造没有引入 rank 依赖的数值偏差。
2. `e_con` 的三点二阶差分仅 `3.06e-8 Ry`，差分信号不是曲率或明显 SCF 噪声
   主导；但解析冻结约束力只有 `~1.24e-4 eV/A`，`e_con` 导数却是
   `~2.80e-3 eV/A`。两者相差 `2.67e-3 eV/A`，说明冻结的 `Q_c dW Q_c`
   只覆盖了约束能量几何响应的一小部分。
3. 该差值与本轮总力 FD 误差 `4.62e-3 eV/A` 同量级。由于总 KS 力约为
   `-5.66 eV/A`，总力 FD 对这个小约束项并不敏感；当前数据不能把它当作
   `Q_c dW Q_c` 项已精确验证的证据，只能证明接入后总力仍在闸门内。
4. 缺失项应来自设计 §6.1/§6.2 的 `dQ`：onsite projector 随原子移动、
   `S^{-1/2}` 随之变化，以及 delta 目标可能随参考态移动。该结论与设计
   R5 的风险描述一致，不是符号翻转或 MPI 归约错误。
5. 额外发现：`PwTargetSpace::ST` 目前只保存实对称投影 overlap，
   `build_pw_projector_overlap()` 丢弃虚部。当前 Gamma 点/实投影路径可用，
   但一般非 Gamma k 点需要复数 Hermitian 存储；本轮没有扩展该表示。

## 5. 下一步

1. 实现 PW 的完整 `dQ`/目标旋转导数，优先复用 onsite projector 非局域
   赝势的几何导数，并在同一点用 `e_con` FD 闭合到总力 FD 量级。
2. 完成后移除 `KPAR * bndpar > 1` 的力路径守卫：先做互补密度的
   all-pool 归约，再交给共享力核。
3. 将 `PwTargetSpace` 的 `ST`/`ST^{-1/2}` 扩展为复数 Hermitian，并补
   非 Gamma k 点测试。
4. 继续 PW allocation 应力与完整 `kpar=2` 收敛回归。
