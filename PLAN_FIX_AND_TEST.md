# DFT+U + DeltaSpin PW Port — 问题清单、修复计划与测试方案

> 更新日期: 2026-04-28
> 分支: feat/dftu-pw-port
> 参考分支: zdy-tmp (`/root/abacus-zdy-tmp`)

---

## 一、已知问题清单

### P0 — 必须修复

| # | 问题描述 | 文件位置 | 影响范围 |
|---|---------|---------|---------|
| P0-1 | nspin=2 PW+DFT+U SCF 在 DS3 发散 | `dftu_pw.cpp` / `op_pw_proj.cpp` | 测试 222 |
| P0-2 | DeltaSpin GPU 代码 `delete[] becp_cpu` 但 becp_cpu 非 new[] 分配 | `cal_mw_from_lambda.cpp:126` | GPU nspin=2 DS |
| P0-3 | `sc_direction_only` 参数未从 zdy-tmp 同步 | `spin_constrain.h/cpp` + 多处 | noncolin+DS 方向约束 |

### P1 — 重要修复

| # | 问题描述 | 文件位置 | 状态 |
|---|---------|---------|------|
| P1-1 | `[DS-LAMBDA]` debug fprintf 未清理 | `deltaspin_pw.cpp:12,24,30,35,45` | ✅ 已清理 |
| P1-2 | cal_mw_from_lambda npol=2 Mi 计算缺少 nspin=2 k 点自旋分裂处理 | `cal_mw_from_lambda.cpp:452-481` | ✅ 已验证逻辑与 zdy-tmp 一致，无 bug |
| P1-3 | 测试未验证 DeltaSpin 内循环收敛 + SCF 收敛双重条件 | 测试框架 | ⏳ 待实现 |
| P1-4 | 测试未验证 lambda 值与 zdy-tmp 一致性 | 测试框架 | ⏳ 待实现 |
| P1-5 | 测试未覆盖 STRU sc/lambda 特殊用法 (sc_scf_thr < scf_thr) | 测试用例 | ⏳ 待实现 |

### P2 — 清理与优化

| # | 问题描述 | 文件位置 | 状态 |
|---|---------|---------|------|
| P2-1 | op_pw_proj.cpp 中注释掉的旧代码块 | `op_pw_proj.cpp:172-210, 292-339` | ✅ 已清理 (89 行) |
| P2-2 | lambda_loop.cpp 中 [DIAG-*]/[LAMBDA-LOOP] 调试输出 | `lambda_loop.cpp` | ✅ 已清理 (10+ 处) |
| P2-3 | 集成测试日志文件 (log, log-tmp) 不应 tracked | `tests/17_DS_DFTU/*/log*` | ⏳ 待处理 |

---

## 二、修复计划

### Phase 1: P0 核心修复

#### P0-1: nspin=2 PW+DFT+U SCF 发散
**状态**: 已修复 4 个 bug，但仍发散
**下一步**:
- [ ] 逐 k 点比对 vu_device 应用前后的 hpsi 值（对比 zdy-tmp）
- [ ] 检查 `cal_ps_dftu` 中 spin-down 通道的 vu_device sync 是否覆盖完整数据范围
- [ ] 验证 `onsite_ps_op` kernel 的 npol=1 分支是否正确处理了 DFT+U vu 的 stride
- [ ] 对比 zdy-tmp 和当前代码的 `cal_VU_pot_mat` 调用路径（LCAO vs PW 差异）

#### P0-2: GPU delete[] 内存错误
**状态**: ✅ 已修复 (commit c9e6d747f)
**修复**: 将 line 126 的 `delete[] becp_cpu;` 改为 `delete_memory_op<std::complex<double>, base_device::DEVICE_CPU>()(becp_cpu);`

#### P0-3: sc_direction_only 参数移植
**状态**: ✅ 已修复 (commit c9e6d747f)
**已修改文件**:
- `input_parameter.h` — 添加 `sc_direction_only` 成员
- `read_input_item_other.cpp` — 添加参数解析和文档
- `spin_constrain.h` — 添加 `direction_only_` 成员 + 更新 `init_sc` 签名
- `init_sc.cpp` — 更新签名并赋值 `direction_only_`
- `deltaspin_lcao.cpp` — 传递 `inp.sc_direction_only`
- `setup_pot.cpp` — 传递 `PARAM.inp.sc_direction_only`
- `lcao_others.cpp` — 传递 `PARAM.inp.sc_direction_only`
- `lambda_loop.cpp` — 添加 4 处 lambda 投影逻辑

### Phase 2: P1 功能验证修复

#### P1-1: 清理 debug 输出
**状态**: ✅ 已清理 (commit 37d7df178)
- 删除 `deltaspin_pw.cpp` 中所有 `[DS-LAMBDA]` fprintf (5 处)
- 删除 `lambda_loop.cpp` 中 `[DIAG-LOOP]`, `[LAMBDA-LOOP]`, `[DIAG-BEFORE-255]`, `[DIAG-LAMBDA]` 输出 (10+ 处)
- 保留有用输出: RMS error, convergence messages, timing

#### P1-2: cal_mw_from_lambda npol=2 Mi 计算
**状态**: ✅ 已验证逻辑正确
- 对比 zdy-tmp 代码，Mi 计算逻辑完全一致
- npol=2 (nspin=4): 使用 `ib * npol * nkb + begin_ih + ih` 索引 + 4 Pauli 分量
- npol=1 (nspin=2): 使用 `sign = isk[ik] == 0 ? 1 : -1` 区分自旋

#### P1-3~5: 测试框架增强 (详见第三部分)

### Phase 3: P2 代码清理

**已完成**:
- ✅ P2-1: 删除 op_pw_proj.cpp 注释代码块 (89 行)
- ✅ P2-2: 删除 lambda_loop.cpp 调试输出 (10+ 处)
- ⏳ P2-3: 集成测试日志文件 .gitignore (待处理)

---

## 三、测试方案

**规则**: 一个测试用例通过必须同时满足:
1. DeltaSpin 内循环达到设定阈值 (`sc_thr`)
2. SCF 外循环达到设定阈值 (`scf_thr`)

**实现方式**:
- 在 `result.ref` 中增加 `DS_INNER_CONVERGED` 标记行
- 修改 `Autotest.sh` 的 `check_out` 函数，检查:
  - `SCF converged` 或 `!FINAL_ETOT_IS` 存在
  - 输出中包含 `DeltaSpin inner loop converged` 或等效标记
- 如果任一未收敛，测试标记为 FAIL

**测试用例覆盖**:
| 用例 | 基础类型 | 需验证 |
|------|---------|--------|
| 12_PW_DS_S2_Z | PW DS nspin=2 | SCF + DS 内循环 |
| 13_PW_DS_S4_XY | PW DS nspin=4 XY | SCF + DS 内循环 |
| 14_PW_DS_S4_XYZ | PW DS nspin=4 XYZ | SCF + DS 内循环 |
| 18_PW_DFTU_DS_S2_Z | PW DFTU+DS nspin=2 | SCF + DS 内循环 |
| 19~23_PW_DFTU_DS_S4_* | PW DFTU+DS nspin=4 | SCF + DS 内循环 |
| 24_LCAO_DS_S2_Z | LCAO DS nspin=2 | SCF + DS 内循环 |
| 30~35_LCAO_DFTU_DS_* | LCAO DFTU+DS | SCF + DS 内循环 |

### 3.2 Lambda 值一致性验证 (对应需求 2)

**方法**: 对于每个开启 DeltaSpin 的测试用例:
1. 运行当前分支，提取 `OUT.autotest/running_scf.log` 中最终的 lambda 值
2. 运行 zdy-tmp 分支（相同 INPUT/STRU/KPT），提取 lambda 值
3. 逐原子逐分量比较，差异 < 1e-6 视为 PASS

**实现**:
- 在 `running_scf.log` 输出中增加 `[LAMBDA-FINAL]` 行（或在已有输出中解析）
- 新建脚本 `compare_lambda.py` 自动对比两个分支的输出
- 在 `result.ref` 中添加 lambda 参考值

**测试优先级**:
- 先验证纯 DeltaSpin (无 DFT+U): 用例 12, 13, 14
- 再验证 DeltaSpin + DFT+U: 用例 18, 19, 20

### 3.3 STRU sc/lambda 设置验证 (对应需求 3)

#### 3.3.1 基本约束验证
- [ ] 确保所有 DeltaSpin 测试用例的 STRU 中至少有一个原子设置了 `sc 1` (或其他非零约束标志)
- [ ] 验证约束原子的磁矩最终收敛到 STRU 中设定的目标值 (误差 < 1e-3 μB)

#### 3.3.2 初始 lambda 验证
- [ ] 在 STRU 中设置不同 `lambda` 初始值 (0.1, 1.0, 5.0)
- [ ] 验证 SCF 迭代次数和最终能量不受 lambda 初始值影响（应收敛到相同结果）

#### 3.3.3 外场模拟模式 (sc_scf_thr < scf_thr)
**原理**: 当 `sc_scf_thr` 设置为 1e-10 或小于 `scf_thr` 时，DeltaSpin 内循环在 SCF 未完全收敛时就触发，模拟外加磁场效果

**测试设计**:
| 用例 | scf_thr | sc_scf_thr | 预期行为 |
|------|---------|-----------|---------|
| DS_S2_Thr1e10_Z | 1e-6 | 1e-10 | 内循环提前触发，模拟外场 |
| DS_S4_Thr1e10_XY | 1e-6 | 1e-10 | 同上 |
| DS_S2_Thr10_Z | 1e-6 | 10 | 内循环延迟触发 |

- 已有用例 38, 39 (Thr1e10) 和 40, 41 (Thr10) 覆盖此场景
- 需要验证: 内循环触发时机是否符合预期

### 3.4 sc_direction_only 验证 (对应需求 4)

**移植完成后新增测试**:
| 用例 | 描述 | 预期 |
|------|------|------|
| DS_S4_DirOnly_XYZ | noncolin + DS + sc_direction_only=1 | 磁矩方向收敛到目标，大小自由变化 |
| DS_S4_DirOnly_XY | 同上，XY 平面约束 | 方向约束在 XY 平面 |

**验证方法**:
- 对比 `sc_direction_only=1` 和 `sc_direction_only=0` 的最终磁矩大小和方向
- `sc_direction_only=1`: 磁矩方向应接近目标，大小可变化
- `sc_direction_only=0`: 磁矩大小和方向都应接近目标

### 3.5 测试执行计划

```
Phase 1 (阻塞): 修复 P0-1, P0-2 → 运行用例 222 (nspin=2 PW+DFTU)
Phase 2: 修复 P0-3, P1-1, P1-2 → 运行全部 52 个用例
Phase 3: 收敛性 + Lambda 验证 → 更新 result.ref
Phase 4: STRU sc/lambda 特殊场景 → 新增/修改测试用例
Phase 5: sc_direction_only → 新增测试用例
```

---

## 四、用户修改意见确认

请在下方回复你的具体修改意见，我会纳入计划:

1. 关于 DeltaSpin 内循环收敛判断：是否需要修改输出格式以便测试脚本解析？
2. 关于 lambda 值比对：是否需要自动化脚本还是手动比对即可？
3. 关于 sc_direction_only：优先级别如何？是否需要立即移植？
4. 其他需要补充的测试场景或修复项？

---

## 五、参考信息

### zdy-tmp 收敛能量 (222_PW_DFTU_S2_Z)
- Final ETOT: -6792.3335741003975272 eV
- SCF 迭代: 45 次
- DeltaSpin: 未开启 (纯 DFT+U 测试)

### 当前分支状态 (更新: 2026-04-28)
- nspin=4 PW+DFTU: ✅ PASS
- nspin=2 LCAO+DFTU: ✅ PASS  
- nspin=2 PW (无 DFT+U): ✅ PASS
- nspin=2 PW+DFTU: ❌ FAIL (DS3 发散) — **当前阻塞项**
- sc_direction_only 参数: ✅ 已移植
- GPU 内存管理: ✅ 已修复
- 调试输出清理: ✅ 已完成
- cal_mw_from_lambda Mi 计算: ✅ 已验证逻辑正确

### 本轮已完成修复
| Commit | 修复内容 |
|--------|---------|
| 6f28a3b | nspin=2 DFT+U: npol 硬编码、vu_begin 计算、spin-down vu sync、locale 自旋索引 |
| c9e6d74 | DeltaSpin: GPU 内存修复、sc_direction_only 移植、调试输出清理 |
| 37d7df1 | 代码清理: 删除注释代码块 (89 行)、删除调试输出 (10+ 处) |
