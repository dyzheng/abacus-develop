# S5.1 PW 分配算符接线：阶段进度报告

日期：2026-10-26

对应执行方案：`docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`
Task S5.1。

本文件记录一次进行中的修改，不等同于验收报告。当前状态是：PW allocation
约束算符与分配观测量已经接入代码并通过编译，固定 `fixed_mu` 的完整 SCF
和 FD 力验证尚未完成。

## 1. 测试计划

本轮计划验证四件事：

1. `HamiltPW` 在 `constraint_alloc_switch=1` 时注册 PW 专用分配算符，
   并把设计 §5.2 的
   `V - P V - V P + P V P + sum_i lambda_i Q_i`
   直接加到 `H|psi>`，而不是继续向 `v_eff` 注入 `mu*w`。
2. `ConstraintLoop` 的 allocation 观测量来自同一低秩算符：
   `n_i/m_i = Tr(D K_i)`，避免外层读出与哈密顿量注入不一致。
3. H2O、O-d、Becke、固定 `fixed_mu=0.01 Ry` 的 PW SCF 能收敛，且
   `ALLOC` 读数、能量补正和 `sum(n)/sum(m)` 自洽。
4. 编译与既有 constraint 测试不回归；后续再进入位置 FD 力。

## 2. 测试设置

### 2.1 代码修改

- 新增 `source_pw/module_pwdft/op_pw_alloc.{h,cpp}`：
  - `AllocConstraintPW` 在 `act()` 中构造当前 k 点、当前自旋通道的
    实空间权重势 `V`；
  - 使用 `OnsiteProjector` 的 `tab_atomic` 作为目标轨道
    `|alpha_a>`，构造 `S_T=<alpha|alpha>` 和正交目标空间；
  - 通过 `PW_Basis_K::recip_to_real/real_to_recip` 计算
    `V|psi>`、`V|alpha>`、`m_raw=<alpha|V|alpha>`；
  - 调用现有 `apply_pw_alloc_delta_h()` 得到单 band 修正并累加到 `H|psi>`。
- `HamiltPW` 在 `constraint_alloc_switch` 打开时注册该算子，并支持
  Hamiltonian 复制构造路径。
- `ESolver_KS_PW::iter_finish()` 在 allocation 模式下使用
  `compute_pw_alloc_observables()`，从真实波函数计算每个约束的
  `Tr(D K_i)`；非 allocation 模式保持原网格 `observe()`。
- `ESolver_KS_PW::before_scf()` 移除“PW 尚不支持”的拒绝，改为设置
  `cfg.allocation_mode=true`。首个闭环显式限制为 `nspin=1/2`、
  `calculation=scf`，并拒绝 dipole 约束。
- `apply_pw_alloc_delta_h()` 的 `braw=<alpha|V|psi>` 增加 pool 归约，
  修正 MPI 下只在 rank-local 平面波切片上做点积的问题。

### 2.2 运行输入

- H2O，单 Γ，15 Angstrom 立方盒。
- `nspin=2`、`nbands=8`、`nelec=8`。
- `ecutwfc=20 Ry`、`ecutrho=80 Ry`。
- `constraint_type spin`，目标原子 O，`constraint_alloc_target_l=2`。
- Becke 权重，`onsite_radius=6.9 Bohr`。
- 目标文件：

```json
{"constraints": [{"type": "spin", "target": 0.5, "atoms": [0], "fixed_mu": 0.01}]}
```

- 新增临时运行目录 `tests/constraint_s5_parity/pw_alloc/`。该目录当前是
  验证副本，未纳入提交。
- 运行命令：

```text
OMP_NUM_THREADS=1 ../../../build/abacus_basic_para > run_alloc.log 2>&1
```

## 3. 结果

### 3.1 编译

```text
cmake --build build --target abacus_basic_para -j 14   PASS
```

新增文件与求解器接线均通过编译。编译器给出一个非致命警告：

```text
warning: 'if constexpr' only available with '-std=c++17' or '-std=gnu++17'
```

当前工程仍按 C++14 风格编译，GCC 接受了该扩展；后续应改成显式的模板分派
或 C++14 可移植写法。

### 3.2 固定 `fixed_mu` SCF

SCF 已启动并进入正常电子迭代，但按要求在 CG3 后人工中断，未完成收敛：

```text
CG1   9.27e-15   1.54e-01   -4.41428071e+02   0.00000000e+00   2.9946e-01   34.65 s
CG2   1.22e-14   5.08e-02   -4.42052396e+02  -6.24324613e-01   4.9908e-02   11.80 s
CG3  -2.31e-15   4.79e-02   -4.42145172e+02  -9.27760997e-02   4.2995e-03   12.31 s
```

作为对照，只读 `ALLOC-PW` 版本同一输入的 CG1--CG3 为：

```text
CG1   -4.41435257e+02   3.84 s
CG2   -4.42075786e+02   1.98 s
CG3   -4.42132078e+02   1.89 s
```

因此可以确认 allocation 算符确实进入了 `H`（总能轨迹已改变），但本轮没有
得到 `#SCF IS CONVERGED#`、最终 `ALLOC` 或 `FINAL_ETOT`。

### 3.3 观测量开销

加入逐 SCF 迭代的 `Tr(D K_i)` 计算后，CG1 从约 `3.8 s` 增至 `34.7 s`，
CG2/CG3 从约 `2 s` 增至 `12 s`。开销来自每个 band、每个约束都要重复执行
`V|psi>`、`V|alpha>` 的 FFT，以及目标空间重建。

## 4. 分析

1. **已确认的进展**：`constraint_alloc_switch=1` 不再被直接拒绝，PW
   Hamiltonian 中已存在分配算符；总能与只读版本明显不同，说明算子已实际
   参与 SCF，而不是空注册。
2. **尚不能验收**：没有完成 SCF，无法判断固定点是否收敛，也没有最终
   `ALLOC`、求和规则或 FD 力数据。当前不能宣称 S5.1 算符验收通过。
3. **主要风险是性能**：分配观测量在每个 SCF 迭代都重建目标空间并做多次
   FFT，导致约 4--9 倍耗时。首个闭环应先保证正确性，再缓存
   `phi/S_T/ST^{-1/2}` 和 `V|alpha>`，或只在收敛/审计帧计算完整观测量。
4. **MPI 语义已补但不能算验证完成**：`braw` 的 pool 归约已加入；尚未在
   `KPAR>1` 上跑出收敛数据。
5. **算子正确性仍缺独立测试**：目前只有代码路径与能量轨迹间接证据，缺少
   固定目标下 `H|psi>` 有限性、`<psi|DeltaH|psi>` 与 LCAO 量对照，以及
   关闭开关时逐位不变的回归证据。

## 5. 下一步

1. 优化 allocation 观测量：缓存几何固定的目标空间与目标乘积，避免每个
   SCF 迭代重复构造；固定 `fixed_mu` 时可只在收敛帧计算完整 `Tr(D K_i)`。
2. 重跑 H2O 固定 `fixed_mu=0.01 Ry` PW SCF 至收敛，检查：
   - `#SCF IS CONVERGED#`；
   - `ALLOC` 的 `sum_n/sum_m` 审计；
   - 关闭 `constraint_alloc_switch` 时只读路径逐位不变。
3. 增加 PW allocation 算子单测/集成测试，覆盖 `H|psi>` 有限性、
   目标空集拒绝、`npol=1/nspin=1/2` 守卫和 MPI pool 一致性。
4. SCF 通过后做固定 `lambda` 三点位置 FD 力，门槛
   `<0.0129 eV/A`；之后再处理 `KPAR>1` 与应力。

