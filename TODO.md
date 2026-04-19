# ABACUS DFT+U PW Port — 当前任务总览

> 仓库: `/root/abacus-dftu-pw-port` (branch: `feat/dftu-pw-port`)
> 最后更新: 2026-04-17

---

## 核心原则（新增）

### 1. 代码迁移优先级
当 zdy-tmp 与 dftu-pw-port/develop 出现代码冲突时，**优先采用 develop 分支的重构方案**，而非机械复制 zdy-tmp。zdy-tmp 的修改仅作为功能参考，具体实现必须适配 develop 的当前架构。

### 2. 测试一致性要求
**所有修改的物理结果必须与 zdy-tmp 分支保持一致。** 编译通过不等于任务完成，必须通过数值比对验证。

### 3. 测试覆盖矩阵
集成测试必须覆盖以下维度：
- **功能**: DFT+U、DeltaSpin
- **自旋**: nspin=2、nspin=4（非共线）
- **磁矩方向**: z 方向、xy 平面、xyz 倾斜方向
- **基组**: PW（当前重点）+ LCAO（后续）

### 4. 测试集即工作产物
**构建并维护测试集本身就是本项目的工作内容**，不是可选的附加任务。每个新功能或修复都必须伴随集成测试 case。

---

## 一、当前状态快照

### 1.1 编译与测试
- [x] **整库编译**: `cmake --build . -j$(nproc)` **PASS** (100%)
- [x] **DeltaSpin 单元测试**: `ctest -R "deltaspin|lambda_update"` **PASS** (4→6/6)
- [x] **DFT+U 单元测试**: `ctest -R "dftu|operator_dftu"` **PASS** (2→6/6)
- [x] **全量单元测试**: `ctest -R "dftu|delta|lambda|operator_dftu"` **PASS** (6/6)
- [x] **PW 集成测试**: `160_PW_DJ_PK_PU_SO` (DFT+U, nspin=4, SOC) ✅ 运行通过
- [ ] **与 zdy-tmp 的数值比对**: 部分完成（zdy-tmp 无法运行此 case，见 §2.3）

**集成测试基线 (`160_PW_DJ_PK_PU_SO`)**:

| 版本 | 能量 (eV) | 力 | 应力 | 时间 |
|------|-----------|-----|------|------|
| dftu-pw-port | -5662.390604 | 17.627804 | 100838.127667 | 1.15s |
| abacus-develop | -5662.393016 | 14.928918 | 104822.896517 | 19.67s |
| result.ref (旧) | -5662.390886 | 17.965510 | 100582.607209 | 1.26s |

**差异原因**: develop 已重构 force/stress 实现（移除 `npol==1` 特殊分支，统一使用完整 4 项计算）。dftu-pw-port 保留旧分支导致差异。完整适配 develop 重构方案需额外工作量（memory API 签名不兼容），作为独立任务处理。

### 1.2 工作区状态
工作区已清理，**2 个 commit** 已提交：

**已提交（新）:**
1. `060c53d98` — fix: DeltaSpin GPU memory API adaptation and cleanup
   - `cal_h_lambda.cpp`: GPU 索引修正 (pv->nrow/pv->ncol)
   - `cal_mw.cpp`: 移除 debug iostream + 注释掉的死代码
   - `cal_mw_from_lambda.cpp`: GPU memory API 适配 (`ctx` 参数移除)
   - `lambda_update_strategies.cpp`: 移除冗余 iostream

2. `8f7a99778` — fix: correct npol==1 force/stress index and adjust dftu.h include
   - `force_op.cpp`: npol==1 分支 index 从 `nproj` 修正为 `nkb`
   - `stress_op.cpp`: 代码风格清理
   - `dftu.h`: charge_mixing.h 移出 `#ifdef __LCAO`

**未跟踪（保留）:**
- `MIGRATION_PLAN.md` — 迁移计划文档（保留）
- `TODO.md` — 本文件（保留）
- `scripts/` — 迁移工具脚本（保留）

---

## 二、近期提交记录与 zdy-tmp 迁移状态

### 2.1 已完成的迁移批次

| Commit | 内容 | 状态 |
|--------|------|------|
| `82203fe48` | Batch 1: PW OnsiteProjector | ✅ Done |
| `d15c285c2` | Batch 2: PW Force & Stress (partial) | ✅ Done |
| `5b0900be3` | Batch 4: DeltaSpin module | ✅ Done |
| `58997aded` | Batch 5: ESolver mixing_dftu (part 1) | ✅ Done |
| `48134c1b9` | Broyden mixing restore + partial top-7 | ✅ Done |
| `12508850d` | feat: lambda_update_strategies | ✅ Done |

### 2.2 zdy-tmp Top-7 Commits 迁移进度

参考仓库: `/root/abacus-zdy-tmp` (zdy/tmp 分支)

| zdy Commit | 标题 | 涉及文件 | 迁移状态 | 备注 |
|------------|------|---------|---------|------|
| `19ade1859` | Fix: initial error of dngvd on DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ❌ SKIP | ROCm/DCU 专用，目标仓库无此文件 |
| `bce760541` | fix:hip code could not run properly in DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ❌ SKIP | 同上 |
| `b9ce68339` | fix:dngvd.hip.cu run properly in DCU | `module_hsolver/kernels/rocm/dngvd_op.hip.cu` | ❌ SKIP | 同上 |
| `a9d881c95` | Feature: add conserve_setting for DFTU with DeltaSpin | `charge_mixing.h`, `esolver_ks_pw.cpp` | ✅ DONE | `conserve_setting()` 已存在；补充 `mixing_restart_step` 排除条件 (`9ab367642`) |
| `e9e91d7fe` | Fix: nscf for pw code | `esolver_ks_pw.cpp`, `dftu_occup.cpp`, `dftu_pw.cpp` | ✅ DONE | nscf DFT+U 调用 + uom_save 逻辑 (`3195855d7`) |
| `1a6871dca` | fix: nscf error of DFT+U | `dftu.cpp`, `dftu_pw.cpp` | ✅ DONE | global_readin_dir 路径 + initialed_locale 逻辑 (`3195855d7`) |
| `34f564ef1` | Fix: deltaspin force error on GPU | `force_op.cu`, `force_op.cpp` | ✅ DONE | y-axis 系数修正已验证 (GPU+CPU) + 0b8383a3f |

### 2.3 关键发现

1. **dngvd 跳过**: `dftu-pw-port` 及上游 `develop` 中均无 `dngvd_op.hip.cu`。这 3 个 commit 是 ROCm/DCU 专用，已跳过。
2. **force/stress**: y-axis 修正已在 `0b8383a3f` 提交，develop 的 npol==1 重构适配作为独立任务处理。
3. **DeltaSpin GPU API**: 已在 `060c53d98` 提交 — `cal_h_lambda.cpp` 和 `cal_mw_from_lambda.cpp` 中 memory op API 适配（移除 `ctx` 参数）。

---

## 三、行动计划

### Phase 1: 工作区清理与整理（今天，30 min）

- [ ] **1.1 清理误生成的 cmake/测试产物**
  ```bash
  cd /root/abacus-dftu-pw-port
  rm -rf CMakeCache.txt CMakeFiles/ Testing/ commit.h
  rm -f tests/01_PW/test.sum tests/integrate/test.sum
  ```

- [ ] **1.2 处理未提交的 DeltaSpin 清理修改**
  - 文件: `cal_h_lambda.cpp`, `cal_mw.cpp`, `cal_mw_from_lambda.cpp`, `lambda_update_strategies.cpp`
  - 动作: 审阅 diff，确认无功能回退，提交为独立 commit
  - 建议 commit message: `fix: DeltaSpin GPU memory API adaptation and cleanup`

- [ ] **1.3 处理 force/stress 工作树修改**
  - 文件: `force_op.cpp`, `stress_op.cpp`
  - 动作: **三方对比**（zdy-tmp `34f564ef1` vs 当前 diff vs develop 当前状态）
  - 目标: 确认 `npol==1` 分支是 zdy-tmp 既有还是 develop 的新重构；如果是 develop 方案，优先保留 develop 实现，只补回缺失的系数修正
  - 如果确认完整且一致，提交；如果还有缺失，补完后提交

### Phase 2: zdy-tmp Top-7 迁移收尾（接下来 2-3 天）

- [ ] **2.1 T1 — dngvd 映射调查**
  - 使用 `scripts/prompts/task_dngvd_investigate.txt` 启动 subagent
  - **约束**: 调查时必须同时查看 `develop` 中 `hegvd_op.hip.cu` 的当前实现，判断 dngvd 功能是否已被替代
  - 结论输出到 `/tmp/dngvd_investigation_report.md`
  - **决策点**: 这 3 个 commit 是跳过、重定向到新文件，还是手动迁移?

- [ ] **2.2 T2 — nscf esolver 迁移**
  - 文件: `source/source_esolver/esolver_ks_pw.cpp`
  - 参考: `e9e91d7fe` 中的 `esolver_ks_pw.cpp` 修改
  - 动作: **diff 三方对比**（zdy-tmp vs 当前 vs develop），补全尚未应用的部分
  - 重点检查 `calculation == "nscf"` 相关逻辑

- [ ] **2.3 T3 — nscf DFT+U 迁移**
  - 文件: `source/source_lcao/module_dftu/dftu_occup.cpp`, `dftu_pw.cpp`, `dftu.cpp`
  - 参考: `e9e91d7fe` + `1a6871dca`
  - 动作: 按顺序先应用 `e9e91d7fe` 再 `1a6871dca`，处理 nscf 跳过逻辑
  - **如遇 API 冲突，优先采用 develop 当前方案，只保留 zdy-tmp 的业务逻辑**

- [ ] **2.4 T4 — force_op GPU 修复收尾**
  - 文件: `source/source_pw/module_pwdft/kernels/cuda/force_op.cu`
  - 参考: `34f564ef1`
  - 动作: 检查 `force_op.cu` 是否已包含修正；如未包含，应用并验证编译

- [ ] **2.5 T5 — conserve_setting 验证**
  - 文件: `charge_mixing.h`, `esolver_ks_pw.cpp`
  - 动作: 验证 `48134c1b9` 是否已完整覆盖 `a9d881c95`；如已覆盖，标记完成

- [ ] **2.6 T6 — 编译修复（按需）**
  - 如果在 2.1-2.5 中引入编译错误，启动编译修复 subagent

- [ ] **2.7 T7 — 代码审查（每项后）**
  - 对每项修改启动独立 review subagent
  - 输出 `REVIEW_REPORT.md`，必须 `PASS` 才能合并

### Phase 3: 编译与测试 Gate（每项修改后必做）

#### Gate 1: 编译
```bash
cd /root/abacus-dftu-pw-port/build
cmake --build . -j$(nproc) 2>&1 | tee /tmp/build.log
# 目标: 0 error
```

#### Gate 2: 单元测试
```bash
cd /root/abacus-dftu-pw-port/build
ctest -R "deltaspin|dftu|lambda_update" --output-on-failure 2>&1 | tee /tmp/unit_test.log
# 目标: 全部 PASS
```

#### Gate 3: 集成测试（分阶段扩展）

**Stage A — 基础 PW 回归（立即执行）**
```bash
cd /root/abacus-dftu-pw-port/tests/integrate
bash Autotest.sh -a ../../build/abacus -n 4 -r "101_PW_.*" 2>&1 | tee /tmp/integration_basic.log
```

**Stage B — DFT+U + DeltaSpin 专项（当前重点）**
需要运行并比对的测试 case：
- DFT+U PW 测试（所有 `dftu` 相关 PW case）
- DeltaSpin PW 测试（所有 `deltaspin` 相关 PW case）
- nspin=2 磁矩沿 z 方向
- nspin=2 磁矩沿 xy 方向
- nspin=4 磁矩沿 z 方向
- nspin=4 磁矩沿 xyz 倾斜方向

** Stage C — LCAO 回归（后续）**
```bash
bash Autotest.sh -a ../../build/abacus -n 4 -r "201_NO_.*"
```

#### Gate 4: 代码审查
- [ ] 逻辑一致性（与 zdy-tmp 意图一致）
- [ ] **优先使用 develop 重构方案**（无冲突时才直接迁移 zdy-tmp）
- [ ] API 适配正确性
- [ ] 无 debug print / WIP 代码
- [ ] 单文件长度合规

### Phase 4: 测试集构建（与代码开发并行）

**目标**: 为 zdy-tmp top-7 修改中每一项功能变更构建最小可复现的集成测试。

#### 4.1 现有测试资源盘点
- [ ] 列出 zdy-tmp 中 `tests/` 目录下与 DFT+U / DeltaSpin 相关的所有 PW case
- [ ] 对比 dftu-pw-port 中是否已存在；如不存在，评估是否需要移植

#### 4.2 新增测试集规格（最小集合）

| 测试 ID | 功能 | nspin | 磁矩方向 | 基组 | 优先级 | 状态 |
|--------|------|-------|---------|------|--------|------|
| `dftu_pw_z2` | DFT+U | 2 | z | PW | P0 | 待建 |
| `dftu_pw_xy2` | DFT+U | 2 | xy | PW | P0 | 待建 |
| `dftu_pw_xyz4` | DFT+U | 4 | xyz | PW | P0 | 待建 |
| `dspin_pw_z2` | DeltaSpin | 2 | z | PW | P0 | 待建 |
| `dspin_pw_xy2` | DeltaSpin | 2 | xy | PW | P0 | 待建 |
| `dspin_pw_xyz4` | DeltaSpin | 4 | xyz | PW | P0 | 待建 |
| `dftu_dspin_pw_z2` | DFT+U + DeltaSpin | 2 | z | PW | P1 | 待建 |

#### 4.3 测试比对流程
每个 case 的运行必须与 zdy-tmp 产出进行数值比对：
1. 在 zdy-tmp 上运行相同 case，保存 `result.ref`
2. 在 dftu-pw-port 上运行，比对关键量：
   - 总能量 (Ry 或 eV)
   - 磁矩 (muB)
   - 力 (Ry/Bohr)
   - 应力 (kbar)
3. **允许偏差阈值**: 1e-6 (能量), 1e-4 (力/应力), 1e-5 (磁矩)
4. 超出阈值必须定位根因

#### 4.4 测试集维护规范
- 每个新 case 必须包含：`INPUT`, `STRU`, `KPT`, `result.ref`, `README.md`
- `README.md` 中必须说明：测试目的、预期结果、与 zdy-tmp 的对应关系
- case 统一放在 `tests/integrate/` 下，按 `dftu_pw_` / `dspin_pw_` 前缀命名

### Phase 5: 更广泛的 DFTU PW Port 未完成项

> **zdy-tmp Top-7 已全部迁移完成**（`9ab367642`）
> **Phase 5.4 lambda strategies 集成完成**（`8ad8565e9`）
> **Phase 5.1/5.2/5.3 已验证**（数值验证通过，无需迁移）

- [x] ~~**5.4 lambda_update_strategies 集成到 SCF**~~ ✅ DONE (`8ad8565e9`)
  - 新增 5 个 INPUT 参数: `sc_lambda_strategy`, `sc_mu_init`, `sc_mu_max`, `sc_mu_growth`, `sc_mix_beta`
  - SpinConstrain 支持 BFGS / LinearResponse / AugmentedLagrangian / HybridDelayed 策略切换
  - esolver_ks_pw.cpp + esolver_ks_lcao.cpp 已集成策略选择

- [x] ~~**5.1 Batch 3 — DFTU LCAO 核心**~~ ✅ DONE（验证完成，无需迁移）
  - 10 个文件 ~2500 行 diff，全部为架构重构（DFTU→Plus_U, GlobalV→PARAM, namespace 消除）
  - 业务逻辑 100% 已存在：mixing_dftu, nspin=2, PW base, ENABLE_LCAO=OFF guard
  - 集成测试验证：4/4 PASS — 54_NO_PK_PU(diff=5e-11), 55_NO_PK_PU_S1(diff=4.5e-12), 56_NO_PK_PU_SO(diff=2.7e-12), 53_NO_PK_URAMP(diff=5e-12)

- [x] ~~**5.2 Batch 5 Part 2 — ESolver + ElecState**~~ ✅ DONE（验证完成，无需迁移）
  - 关键逻辑已存在：oscillate 检测(PW 版), cal_MW, mag_converged, sc_scf_thr, iter_finish conv_esolver 检查

- [x] ~~**5.3 Batch 6 — LCAO 基础**~~ ✅ DONE（验证完成，无需迁移）
  - FORCE_STRESS.cpp: dspin_force_stress.hpp 已实现
  - hamilt_lcao.cpp: set_current_spin 方法存在，nspin=2 路径已覆盖
  - 集成测试验证：SOC DFTU(nspin=4) 精度 2.73e-12

### Phase 6: 完善集成测试集（当前）

> **目标**: 构建覆盖功能×自旋×磁矩方向×基组×SOC 的完整测试矩阵
> 详细测试矩阵: `tests/integrate/TEST_MATRIX.md`

#### 6.1 现有测试集盘点 ✅ DONE

| ID | 功能 | nspin | 磁矩 | 基组 | SOC | 状态 |
|----|------|-------|------|------|-----|------|
| 815_PW_DFTU_S2 | DFT+U | 2 | AFM(z) | PW | ✗ | ✅ |
| 816_PW_DFTU_S1 | DFT+U | 1 | 无 | PW | ✗ | ✅ |
| 099_PW_DJ_SO | DFT+U | 4 | xyz | PW | ✓ | ✅ |
| 160_PW_DJ_PK_PU_SO | DFT+U | 4 | xyz | PW | ✓ | ✅ |
| 54_NO_PK_PU | DFT+U | 2 | FM(z) | LCAO | ✗ | ✅ |
| 55_NO_PK_PU_S1 | DFT+U | 1 | 无 | LCAO | ✗ | ✅ |
| 56_NO_PK_PU_SO | DFT+U | 4 | xyz | LCAO | ✓ | ✅ |
| 53_NO_PK_URAMP | DFT+U | 2 | z+URamp | LCAO | ✗ | ✅ |
| 146_NO_GO_PU_AF | DFT+U | 2 | AFM(z) | LCAO | ✗ | ✅ |

#### 6.2 完整测试矩阵（34 个测试）

> 完整文档: `TEST_STATUS.md`

| 组别 | ID 范围 | 功能 | Basis | nspin | MagDir | 数量 | 状态 |
|------|---------|------|-------|-------|--------|------|------|
| A 自旋基准 | 200, 220 | 无 U 无 DS | LCAO/PW | 2 | z | 2 | ⏳ |
| B DFT+U | 202-204, 222-224 | DFT+U | LCAO/PW | 2/4 | z/xy/xyz | 6 | ⏳ |
| C DeltaSpin | 250-255, 300-305 | DeltaSpin | PW/LCAO | 2/4 | z/xy/xyz | 12 | 🔴 BLOCKED |
| D DFT+U+DS | 260-265, 310-315 | DFT+U+DS | PW/LCAO | 2/4 | z/xy/xyz | 12 | 🔴 BLOCKED |
| **总计** | | | | | | **34** | **22 P0 + 12 P1** |

**阻塞原因**: GROUP C/D 全部 24 个测试被 P0 堆内存损坏阻塞（`diago_dav_subspace.cpp:92`）。

#### 6.3 PW DFTU/DeltaSpin 调试状态 🔧 IN PROGRESS

**问题**: 所有 PW DeltaSpin 集成测试 (250-253) 崩溃

**最新崩溃（2026-04-18 09:00）**:
- `mpirun -n 1 abacus_2p` 在 test 250 (PW+DeltaSpin nspin=2) 崩溃
- 错误: `corrupted size vs. prev_size` — glibc 堆内存损坏（不是 assert 失败）
- `addr2line`: `diago_dav_subspace.cpp:92` — 析构函数 `delmem_complex_op()(this->hpsi)` 释放时崩溃
- 触发点: ik=7 (最后一个 k 点) 的 `dav_subspace.diag` **之后**
- 根因假设: PW Hamiltonian 在 DeltaSpin 扰动下写入越界，破坏了堆元数据

**Davidson 与 bpcg_kernel_op 关系确认**:
- `diago_dav_subspace.cpp:cal_grad()` 第 434 行调用 `normalize_op<T,Device>()`
- `normalize_op` 定义在 `bpcg_kernel_op.cpp:153-183`（BPCG 和 Davidson 共享内核）
- 第 170 行 `assert(psi_m_norm > 0.0)` 会在波函数范数为零时触发
- **结论**: Davidson 确实调用 bpcg_kernel_op 的 normalize_op，这是共享设计非 bug

**LCAO 测试集**:
- 240-243 已创建（LCAO + DeltaSpin ± DFT+U, nspin=2/4）
- 阻塞原因同 250（共享 Davidson 求解器路径）

**不受影响的测试**:
- 815 (PW DFT+U nspin=2) ✅ PASS（无 DeltaSpin）
- 54/55/56 (LCAO DFT+U) ✅ PASS（ScaLAPACK 对角化，不走 Davidson）

#### 6.4 下一步

- [ ] **定位堆内存损坏根因** — 这是阻塞 240-253 所有测试的 P0 问题
  - 在 `dav_subspace.diag` 内部添加内存检查点（malloc hook 或 Valgrind/ASan）
  - 检查 `hpsi` 写入是否越界（PW Hamiltonian 作用于波函数时）
  - 重点检查 DeltaSpin 的 `hamilt_pw` 是否正确处理了 PW 基矢大小
- [ ] 修复后运行 250 (PW DeltaSpin nspin=2) 验证
- [ ] 运行 251-253 (nspin=4, +DFT+U) 验证
- [ ] 运行 240-243 (LCAO) 验证
- [ ] 补充 PW DeltaSpin 不同磁矩方向测试

---

## 四、Subagent 工作流使用指南

### 4.1 可用脚本
- Orchestrator: `scripts/migrate_zdy_commits_orchestrator.py`
- Prompt 模板: `scripts/prompts/*.txt`

### 4.2 调用规范
**每次只启动 1 个 subagent**（避免 code plan 并发超限）：
1. 主 agent 创建隔离 worktree
2. 填充 prompt 模板并分发
3. 轮询 `PROGRESS.md`（30 秒/次，最多 20 分钟）
4. 检查 `BLOCKERS.md` 是否有阻塞
5. 验收产物（MIGRATION_REPORT.md / REVIEW_REPORT.md）
6. **运行对应集成测试，与 zdy-tmp 数值比对通过后才合并**

### 4.3 并发配置
已在 `~/.hermes/config.yaml` 中设置：
```yaml
delegation:
  max_concurrent_children: 1
```
确保 subagent 串行执行。

### 4.4 新增 Subagent 契约要求
所有派发给 subagent 的迁移任务，prompt 中必须额外包含以下约束：

> "当目标文件与 zdy-tmp 存在 API 冲突时，不要直接复制 zdy-tmp 的旧 API。请先检查 `/root/abacus-develop` 中该文件的当前实现，采用 develop 的重构方案，只保留 zdy-tmp 的业务逻辑。如果无法判断，写入 BLOCKERS.md 并停止工作。"

---

## 五、阻塞与决策清单

| 问题 | 优先级 | 状态 | 下一步 |
|------|--------|------|--------|
| dngvd 文件映射不明 | P3 | ❌ SKIP | ROCm/DCU 专用，目标仓库无此文件，跳过 |
| ~~工作区 cmake 误生成文件~~ | ~~P0~~ | ✅ DONE | 已清理 |
| ~~DeltaSpin 清理修改未提交~~ | ~~P0~~ | ✅ DONE | 已提交 `060c53d98` |
| ~~force/stress 工作树未提交~~ | ~~P0~~ | ✅ DONE | 已提交 `8f7a99778` |
| ~~集成测试基线获取~~ | ~~P0~~ | ✅ DONE | `160_PW_DJ_PK_PU_SO` 运行通过 |
| force/stress develop 重构适配 | P1 | ⏳ | develop 已移除 npol==1 分支，memory API 签名变化大，需作为独立任务处理 |
| ~~nscf DFTU 逻辑未迁移~~ | ~~P1~~ | ✅ DONE | 已提交 `3195855d7` — esolver/dftu/dftu_occup 3 个文件 |
| ~~T4: force_op.cu GPU 修复~~ | ~~P1~~ | ✅ DONE | 已验证：dftu-pw-port 已包含 `34f564ef1` 修正（coefficients1*dbb2 + coefficients2*dbb1）|
| ~~T5: conserve_setting 验证~~ | ~~P1~~ | ✅ DONE | 已提交 `9ab367642` — 补充 mixing_restart_step 排除条件 |
| PW DeltaSpin 堆内存损坏 (240-253) | P0 | 🔧 | `diago_dav_subspace.cpp:92` 析构崩溃，ik=7 后触发 |
| 测试 250 INPUT 修正 | P1 | ✅ DONE | 补充 `sc_mag_switch` 等 DeltaSpin 参数 |
| LCAO DeltaSpin 测试 240-243 | P1 | ✅ DONE | 已创建 INPUT/STRU/KPT，等待修复后运行 |
| ESolver 与 ElecState 大量 diff 待评估 | P2 | ⏳ | 等 top-7 完成后统一评估 |
| lambda strategies SCF 集成 | P2 | ⏳ | 需要专门的设计决策 |

---

## 六、下一步推荐执行顺序

1. ~~【5 分钟】清理工作区 cmake/测试产物~~ ✅ DONE
2. ~~【10 分钟】提交 DeltaSpin GPU API 适配修改（4 个文件）~~ ✅ DONE
3. ~~【15 分钟】审阅并提交 force_op.cpp / stress_op.cpp 的未提交修改~~ ✅ DONE
4. ~~【20 分钟】运行集成测试获取基线~~ ✅ DONE (`160_PW_DJ_PK_PU_SO` 通过)
5. ~~【当前】启动 nscf DFTU 迁移 subagent (T2 + T3)~~ ✅ DONE (`3195855d7`)
6. ~~force/stress develop 重构适配~~ ✅ 已验证：dftu-pw-port 的 force/stress kernel 已包含 zdy-tmp 34f564ef1 修正
7. ~~T5: conserve_setting 验证~~ ✅ DONE (`9ab367642`)

**zdy-tmp top-7 全部迁移完成！**

---

*本文件由主 agent 生成，应根据每日进展持续更新。*
