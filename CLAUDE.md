# CLAUDE.md — f-electron-scf 移植项目

本仓库是 ABACUS 的 worktree，分支 `feat/dftu-pw-port`，用于将 DFT+U PW 和 DeltaSpin 功能从 `abacus-zdy-tmp` 移植到 `abacus-develop`。

继承上游 `/root/abacus-develop/CLAUDE.md` 的所有构建、测试、代码风格规范。以下为本项目的额外约束。

---

## 强制验收规则（4-Gate）

**任何代码变更必须按顺序通过以下 4 个 Gate，缺一不可。不得以任何理由跳过。**

### Gate 1: 编译 — MUST PASS

```bash
cd /root/abacus-dftu-pw-port/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON .. 2>&1 | tee /tmp/cmake.log
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
```

- `build.log` 中零 `error:` 行
- 生成可执行文件 `abacus`
- 必须保留编译日志

### Gate 2: 单元测试 — MUST PASS

```bash
ctest --test-dir /root/abacus-dftu-pw-port/build/<相关测试目录> -j4 --output-on-failure 2>&1 | tee /tmp/unit_test.log
```

- 所有现有单元测试 PASS
- 新增/修改的 public 方法必须有对应 gtest 用例
- 新增的代码分支（如 nspin=1 vs nspin=4）必须有对应 test case
- 不允许以"依赖链重"为由跳过单元测试 — 提取 static helper 来测试
- 必须保留测试日志

### Gate 3: 集成测试 — MUST PASS

```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a <abacus路径> -n 4 -r "<case正则>" 2>&1 | tee /tmp/integration_test.log
```

- 所有相关现有集成测试 PASS
- 新增功能需要新建集成测试 case（INPUT, STRU, KPT, result.ref, README）
- 必须保留测试日志

### Gate 4: 代码审查 — MUST PASS

- 修改与 zdy-tmp 参考代码逻辑一致
- API 适配正确（Plus_U 而非 DFTU，PARAM.inp 而非 GlobalV，参数传递而非 GlobalC）
- 无遗留 debug print（std::cout/printf/std::cerr 调试输出）
- 无 WIP 代码（TODO/FIXME/HACK 中的临时代码）
- 无未使用的 include
- 函数签名与 develop 风格一致

---

## 执行者行为约束

### 禁止行为

1. 禁止声称"编译通过"但不提供编译日志
2. 禁止以"依赖链重"、"mock 成本高"为由跳过单元测试
3. 禁止以"需要后续 PR"为由跳过回归测试
4. 禁止在验收证据不完整时声称任务完成
5. 禁止修改验收标准以降低要求

### 必须行为

1. 编译后必须检查 build.log 中的 error 数量
2. 新增代码必须有对应的单元测试
3. 提交前必须运行相关回归测试
4. 必须填写完整的验收证据（见下方模板）
5. 遇到测试失败必须修复，不得标记为"已知问题"跳过

---

## 与用户的交互规则

### 何时必须暂停并询问用户

1. **API 设计选择**：当 zdy-tmp 和 develop 的实现方式有本质差异时（如 mixing 策略），必须向用户说明差异并请求决策
2. **测试失败无法修复**：同一错误重试 2 次仍失败时，输出根因分析并请求指导
3. **发现 task instruction 中的矛盾或遗漏**：不要自行假设，向用户确认
4. **需要修改非 task instruction 指定的文件**：说明原因并请求批准
5. **回归测试出现非预期偏差**：报告偏差数值和可能原因，请求判断

### 何时可以自主执行

1. task instruction 中明确指定的代码修改
2. 编译错误的修复（在 2 次重试限制内）
3. 单元测试的编写和调试
4. 集成测试 case 的创建（按 task instruction 规格）
5. 代码风格调整（clang-format 等）

### 任务完成时的报告格式

任务完成后必须输出以下验收证据，缺失任何一项即视为未完成：

```
## 验收证据

### Gate 1: 编译
- 编译结果: PASS / FAIL
- 编译日志: /tmp/build.log
- error 数量: 0
- 生成二进制: <路径和大小>

### Gate 2: 单元测试
- 新增测试文件: <文件路径列表>
- 新增测试用例: <TestSuite.TestCase 列表>
- 测试结果: X tests passed, 0 failed
- 测试日志: /tmp/unit_test.log

### Gate 3: 集成测试
- 回归测试结果: PASS / FAIL
- 新增 case: <列表>
- 测试日志: /tmp/integration_test.log

### Gate 4: 代码审查
- [ ] 逻辑一致性
- [ ] API 适配
- [ ] 无 debug print
- [ ] 无 WIP 代码
- [ ] 无未使用 include
- [ ] 风格一致性
```

---

## 项目特定信息

### 目录映射（zdy-tmp → develop）

| zdy-tmp | develop |
|---|---|
| `module_hamilt_lcao/module_dftu/` | `source_lcao/module_dftu/` |
| `module_hamilt_lcao/module_deltaspin/` | `source_lcao/module_deltaspin/` |
| `module_hamilt_pw/hamilt_pwdft/` | `source_pw/module_pwdft/` |

### API 映射

| zdy-tmp | develop |
|---|---|
| `ModuleDFTU::DFTU` | `Plus_U` |
| `GlobalV::NSPIN` | `PARAM.inp.nspin` |
| `GlobalV::KPAR` | `PARAM.inp.kpar` |
| `GlobalV::NPROC_IN_POOL` | `PARAM.globalv.nproc_in_pool` |
| `GlobalC::ucell` | `const UnitCell& cell`（参数传递） |
| `psi_p->npol` | `psi_p->get_npol()` |
| `FS_Nonlocal_tools` | `Onsite_Proj_tools` |

### 参考仓库

- worktree: `/root/abacus-dftu-pw-port`（本仓库）
- 上游 develop: `/root/abacus-develop`
- zdy-tmp 参考: `/root/abacus-zdy-tmp`
- PM agent: `/root/pm-agent/projects/f-electron-scf/`

### 任务指令位置

`/root/pm-agent/projects/f-electron-scf/tasks/PR-<N>-task-instruction.md`

### 构建命令

```bash
cd /root/abacus-dftu-pw-port/build
cmake -DENABLE_LCAO=ON -DUSE_OPENMP=ON -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
```

---

## 模块详细信息

### module_dftu（DFT+U）

**位置**: `source/source_lcao/module_dftu/`

**核心类**: `Plus_U`（定义于 `dftu.h`）

**关键静态成员**:
- `U`, `U0`: Hubbard U 参数向量
- `orbital_corr`: 关联轨道角动量标记（-1 表示无关联）
- `energy_u`: DFT+U 能量修正
- `Yukawa`: 是否使用 Yukawa 势计算 U/J
- `mixing_dftu`, `omc`, `uramping`: 混合/控制参数

**关键数据结构**:
- `locale[iat][l][n][spin]`: 局域占据矩阵（ModuleBase::matrix）
- `locale_save`: 上一步保存的占据矩阵
- `eff_pot_pw[index]`: PW 基组有效势（std::complex<double> 向量）
- `eff_pot_pw_index[iat]`: 每个原子在 eff_pot_pw 中的起始索引
- `uom_array`, `uom_save`: 用于 mixing 的占据矩阵一维展开
- `iatlnmipol2iwt[iat][l][n][m][ipol]`: 索引变换

**PW 基组关键方法**（本次移植新增）:
- `cal_occ_pw(iter, psi_in, wg_in, cell, p_chgmix)`: 从 PW 波函数计算占据矩阵
  - 通过 OnsiteProjector 计算 becp = <alpha|psi>
  - nspin=1/2: index = ib*nkb + begin_ih + m_begin + m
  - nspin=4: index = ib*2*nkb + begin_ih + m_begin + m（含 spinor 分量）
  - 计算有效势和能量修正
- `cal_VU_pot_pw(spin)`: 计算 PW 有效势矩阵（当前为空实现）
- `set_locale(ucell)`: 从 uom_array 恢复 locale 矩阵
- `get_eff_pot_pw(iat)`: 获取指定原子的有效势指针

**LCAO 基组方法**:
- `cal_occup_m_k()`: k 点占据矩阵计算
- `cal_occup_m_gamma()`: Gamma 点占据矩阵计算
- `cal_energy_correction()`: 能量修正计算
- `force_stress()`: 力和应力计算
- `cal_eff_pot_mat_complex/real()`: LCAO 有效势矩阵
- `cal_slater_UJ()`: Yukawa 势计算 U/J

**源文件**:
| 文件 | 功能 |
|------|------|
| `dftu.cpp` | 类定义、init、能量修正 |
| `dftu_pw.cpp` | PW 基组占据矩阵和有效势 |
| `dftu_occup.cpp` | LCAO 占据矩阵计算 |
| `dftu_hamilt.cpp` | Hamiltonian 贡献 |
| `dftu_tools.cpp` | get_onebody_eff_pot 等工具函数 |
| `dftu_force.cpp` | 力和应力 |
| `dftu_io.cpp` | 读写占据矩阵 |
| `dftu_folding.cpp` | S/dS 矩阵折叠 |
| `dftu_yukawa.cpp` | Yukawa 势 |

**现有测试**: `test/dftu_pw_test.cpp`（能量权重、becp 索引、set_locale 逻辑）

---

### module_deltaspin（自旋约束 DFT / DeltaSpin）

**位置**: `source/source_lcao/module_deltaspin/`

**核心类**: `SpinConstrain<TK>`（模板单例，TK = double 或 std::complex<double>）

**命名空间**: `spinconstrain`

**关键数据成员**:
- `lambda_`: 拉格朗日乘子向量（Ry/uB）
- `target_mag_`: 目标磁矩（uB）
- `Mi_`: 当前原子磁矩（uB）
- `constrain_`: 约束标志 Vector3<int>
- `nspin_`, `npol_`: 自旋/极化参数
- `strategy_`: Lambda 更新策略实例

**Lambda 更新策略**（`lambda_update_strategies.h`）:
- `LambdaUpdateStrategy`（抽象基类）
- `LinearResponseUpdate`（Scheme B）: 线性响应一步更新
- `AugmentedLagrangianUpdate`（Scheme C）: 增广拉格朗日
- `HybridDelayedUpdate`（Scheme D）: 混合延迟更新
- 辅助函数: `compute_rms_error()`, `count_converged()`, `cap_lambda()`

**PW 基组关键方法**（本次移植新增）:
- `cal_Mi_pw()`: 从 PW 波函数计算原子磁矩
  - npol=1 (nspin=2): Mi.z += sign * weight * |becp|^2
  - npol=2 (nspin=4): 计算 occ[0..3] 得到 Mi.x/y/z
- `calculate_delta_hcc(h_tmp, becp_k, delta_lambda, nbands, nkb, nh_iat, sign)`:
  - 计算 lambda 变化对 Hamiltonian 的修正
  - npol=2: 使用 Pauli 矩阵系数
  - npol=1: 仅 z 分量
- `cal_mw_from_lambda(i_step, delta_lambda)`: 从 lambda 更新磁化
- `update_psi_charge(delta_lambda, pw_solve)`: 更新电荷密度和波函数

**LCAO 基组方法**:
- `cal_MW(step, print)`: 通过实空间投影计算磁矩
- `cal_mi_lcao(step, print)`: cal_MW 的包装
- `cal_MW_k(dm)`: 从密度矩阵计算 MW
- `cal_h_lambda(h_lambda, Sloc2, column_major, isk)`: 计算 H_lambda
- `run_lambda_loop(outer_step, rerun)`: Lambda 迭代主循环
- `cal_escon()`: 自旋约束能量

**辅助函数**（`basic_funcs.h`）:
- `maxval_abs_2d()`, `maxloc_abs_2d()`, `sum_2d()`
- `scalar_multiply_2d()`, `add_scalar_multiply_2d()`, `subtract_2d()`
- `fill_scalar_2d()`, `where_fill_scalar_2d()`, `where_fill_scalar_else_2d()`
- `print_2d()`

**源文件**:
| 文件 | 功能 |
|------|------|
| `spin_constrain.h/cpp` | 主类定义、getter/setter |
| `cal_mw.cpp` | cal_MW, cal_mi_lcao, cal_Mi_pw |
| `cal_mw_from_lambda.cpp` | calculate_delta_hcc, update_psi_charge |
| `cal_mw_helper.cpp` | MW 计算辅助 |
| `cal_h_lambda.cpp` | H_lambda 计算 |
| `lambda_loop.cpp` | Lambda 迭代主循环 |
| `lambda_loop_helper.cpp` | check_rms_stop 等辅助 |
| `lambda_update_strategies.h/cpp` | 策略模式实现 |
| `lambda_strategy_integration.cpp` | 策略集成 |
| `init_sc.cpp` | 初始化 |
| `basic_funcs.h/cpp` | 向量运算工具 |
| `template_helpers.cpp` | double 模板特化（空实现） |
| `sc_parse_json.cpp` | JSON 解析 |

**现有测试**:
- `test/basic_test.cpp`: basic_funcs 工具函数
- `test/spin_constrain_test.cpp`: SpinConstrain 基本接口
- `test/lambda_update_strategies_test.cpp`: 三种策略的收敛性测试
- `test/lambda_loop_helper_test.cpp`: check_rms_stop 等
- `test/template_helpers_test.cpp`: double 模板特化
